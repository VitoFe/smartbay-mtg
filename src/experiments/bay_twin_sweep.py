"""Validation-only thermal-state discovery on raw, masked Gen-2 observations.

All learners predict increments from the last *observed* target anchor.
Site outages hide every temperature channel within 60 m after that anchor.
Selection labels are never training labels. 2026 is excluded from model data.
"""
from pathlib import Path
import argparse, json, hashlib, time, os, traceback
import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import lightgbm as lgb
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8'); tmp.replace(path)


def masked_lags(values, query, target, age, task, near):
    """Availability clock per channel. This function does not fill a missing anchor."""
    clock=np.full(values.shape[1],query-age if task=='forecast' else query,dtype=int)
    if task=='site_outage': clock[near[target]]=query-age
    return np.stack([values[clock-lag,np.arange(values.shape[1])] for lag in [0,6,24]],axis=-1)


def build_data(task,age,cfg,quick=False):
    source=ROOT/'reports/bay_twin'
    meta=pd.read_csv(source/'sensor_geometry.csv')
    all_data=pd.read_csv(source/'hourly_observations.csv.gz',index_col=0,parse_dates=True)
    # Never expose final-period values to the model or the hyperparameter search.
    all_data=all_data.loc[all_data.index < pd.Timestamp('2026-01-01',tz='UTC')]
    cols=[s+'::temperature' for s in meta.sensor]
    all_data=all_data[cols].asfreq('1h'); values=all_data.to_numpy(dtype=np.float32)
    geo=np.load(source/'geometry.npz')
    near=geo['near_site']; dist=geo['distance_m']
    names=['nominal_depth_m','metadata_bottom_m','bathy_low_m','bathy_high_m','land_distance_2018_m']
    names += [c for c in meta if '_fraction_' in c]
    static=meta[names].to_numpy(dtype=np.float32)
    static[:,:4]/=30; static[:,4]/=1500
    static=np.nan_to_num(static,nan=0.)
    features=[]; targets=[]; bases=[]; stations=[]; times=[]
    stride=24 if quick else cfg['query_stride_hours']
    for q in range(24+age,len(values),stride):
        stamp=all_data.index[q]
        calendar=np.array([np.sin(stamp.dayofyear*2*np.pi/365.25),np.cos(stamp.dayofyear*2*np.pi/365.25),
                           np.sin(stamp.hour*2*np.pi/24),np.cos(stamp.hour*2*np.pi/24)],dtype=np.float32)
        for target in range(len(cols)):
            base=values[q-age,target]; truth=values[q,target]
            if not np.isfinite(base+truth): continue
            lags=masked_lags(values,q,target,age,task,near)
            mask=np.isfinite(lags).astype(np.float32)
            normalized=np.nan_to_num((lags-20)/10,nan=0.)
            features.append(np.concatenate([normalized,mask,static,np.tile(calendar,(len(cols),1))],axis=1))
            targets.append(truth-base); bases.append(base); stations.append(target); times.append(stamp)
    X=np.asarray(features,dtype=np.float32); y=np.asarray(targets,dtype=np.float32)
    base=np.asarray(bases); ids=np.asarray(stations); times=pd.DatetimeIndex(times)
    adjacency=np.exp(-dist/1500).astype(np.float32)
    # Explicit geometry prior, not a conservation law; no edge through notebook breakwater.
    adjacency[geo['barrier_crossing']]=0; np.fill_diagonal(adjacency,0)
    adjacency/=np.maximum(adjacency.sum(axis=1,keepdims=True),1e-8)
    folds=[]
    for f in cfg['folds']:
        start=pd.Timestamp(f['train_end'],tz='UTC'); end=pd.Timestamp(f['selection_end'],tz='UTC')
        train=times < start-pd.Timedelta(hours=cfg['embargo_hours'])
        val=(times>=start)&(times<end)
        eligible=[]
        for s in range(len(cols)):
            if ((ids==s)&train).sum()>=cfg['min_training_pairs'] and ((ids==s)&val).sum()>=cfg['min_selection_pairs']:
                eligible.append(s)
        # All comparisons within a fold use exactly these sensor/outcome rows.
        train &= np.isin(ids,eligible); val &= np.isin(ids,eligible)
        if not train.any() or not val.any(): raise ValueError('Empty eligible fold')
        folds.append((np.flatnonzero(train),np.flatnonzero(val)))
    return X,y,base,ids,times,folds,adjacency,meta


def tabular(X,ids,family):
    own=X[np.arange(len(X)),ids]
    local=np.concatenate([own[:,:6],own[:,-4:]],axis=1)
    if family.startswith('local'): return local
    network=X[:,:,:6].reshape(len(X),-1)
    if family=='network_tree': return np.concatenate([local,network],axis=1)
    return np.concatenate([own,network],axis=1)


class SensorGraph(nn.Module):
    def __init__(self,nfeatures,hidden,layers,adjacency):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(nfeatures,hidden),nn.SiLU())
        self.messages=nn.ModuleList([nn.Linear(hidden*2,hidden) for _ in range(layers)])
        self.head=nn.Linear(hidden,1)
        self.register_buffer('adjacency',torch.as_tensor(adjacency))

    def forward(self,x,ids):
        z=self.encoder(x)
        for layer in self.messages:
            neighbors=torch.einsum('ij,bjh->bih',self.adjacency,z)
            z=z+torch.nn.functional.silu(layer(torch.cat([z,neighbors],dim=-1)))
        return self.head(z[torch.arange(len(z),device=z.device),ids]).squeeze(-1)


def neural_fit(X,y,ids,tr,va,family,params,adjacency,seed):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    isgraph=family=='sensor_graph'
    features=X if isgraph else tabular(X,ids,family)
    tx=torch.as_tensor(features,device=device); ty=torch.as_tensor(y,device=device)
    ti=torch.as_tensor(ids,device=device)
    if isgraph:
        model=SensorGraph(X.shape[2],params['hidden'],params['layers'],adjacency).to(device)
    else:
        layers=[nn.Linear(features.shape[1],params['hidden']),nn.SiLU()]
        for _ in range(params['layers']-1): layers += [nn.Linear(params['hidden'],params['hidden']),nn.SiLU()]
        layers.append(nn.Linear(params['hidden'],1)); model=nn.Sequential(*layers).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
    counts=np.bincount(ids[tr],minlength=int(ids.max())+1)
    weights=1/np.maximum(counts[ids[tr]],1); weights/=weights.sum()
    best=float('inf'); best_state=None; stale=0
    def predict(indices):
        out=[]
        with torch.no_grad():
            for b in np.array_split(indices,max(1,int(np.ceil(len(indices)/2048)))):
                p=model(tx[b],ti[b]) if isgraph else model(tx[b]).squeeze(-1)
                out.append(p.cpu().numpy())
        return np.concatenate(out)
    for epoch in range(params['epochs']):
        model.train()
        sampled=rng.choice(tr,size=len(tr),replace=True,p=weights)
        for b in np.array_split(sampled,max(1,int(np.ceil(len(sampled)/512)))):
            opt.zero_grad(set_to_none=True)
            p=model(tx[b],ti[b]) if isgraph else model(tx[b]).squeeze(-1)
            loss=torch.nn.functional.huber_loss(p,ty[b],delta=.5)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        model.eval(); pred=predict(va)
        score=macro_error(y[va],pred,ids[va])
        if score < best-1e-5:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=10: break
    model.load_state_dict(best_state); prediction=predict(va)
    return prediction,{'epochs_run':epoch+1,'device':device}


def macro_error(truth,pred,ids):
    return float(np.mean([np.abs(truth[ids==s]-pred[ids==s]).mean() for s in np.unique(ids)]))


def run(output,task,age,family,quick=False):
    cfg=json.loads((ROOT/'config/bay_twin_campaign.json').read_text())
    X,y,base,ids,times,folds,adjacency,meta=build_data(task,age,cfg,quick)
    out=output/f'{task}_{age}h_{family}'; out.mkdir(parents=True,exist_ok=True)
    contracts=[]
    for tr,va in folds:
        contracts.append({'train_pairs':len(tr),'selection_pairs':len(va),'train_last':str(times[tr].max()),
            'selection_first':str(times[va].min()),'selection_last':str(times[va].max()),
            'sensor_counts':{meta.iloc[s].sensor:{'train':int((ids[tr]==s).sum()),'selection':int((ids[va]==s).sum())} for s in np.unique(ids[va])}})
    save(out/'contract.json',{'task':task,'age_hours':age,'family':family,'smoke_only':quick,'folds':contracts,
        'target':'Observed end-of-hour water temperature, Celsius',
        'score':'Mean of sensor MAEs within fold, then mean of fold ratios to persistence; selection only.',
        'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(study_name=out.name,storage='sqlite:///'+str(out/'optuna.sqlite3'),
                             sampler=optuna.samplers.TPESampler(seed=42),direction='minimize',load_if_exists=True)
    def objective(trial):
        if family=='local_ridge': params={'alpha':trial.suggest_float('alpha',.01,10000,log=True)}
        elif family.endswith('tree'):
            params={'n_estimators':trial.suggest_int('n_estimators',100,500,step=100),
                    'num_leaves':trial.suggest_int('num_leaves',7,63),
                    'min_child_samples':trial.suggest_int('min_child_samples',30,200),
                    'learning_rate':trial.suggest_float('learning_rate',.015,.12,log=True),
                    'reg_lambda':trial.suggest_float('reg_lambda',.1,30,log=True)}
        else:
            params={'hidden':trial.suggest_categorical('hidden',[32,64,128]),'layers':trial.suggest_int('layers',1,3),
                    'learning_rate':trial.suggest_float('learning_rate',.0003,.005,log=True),
                    'weight_decay':trial.suggest_float('weight_decay',.00001,.01,log=True),'epochs':2 if quick else 80}
        rows=[]; scores=[]; details=[]
        for fold,(tr,va) in enumerate(folds):
            if family in ['network_mlp','sensor_graph']:
                pred,detail=neural_fit(X,y,ids,tr,va,family,params,adjacency,42)
            else:
                Z=tabular(X,ids,family)
                model=(make_pipeline(StandardScaler(),Ridge(**params)) if family=='local_ridge' else
                       lgb.LGBMRegressor(**params,objective='regression_l1',random_state=42,n_jobs=4,verbosity=-1))
                counts=np.bincount(ids[tr]); weights=1/counts[ids[tr]]; weights*=len(weights)/weights.sum()
                model.fit(Z[tr],y[tr],**({'ridge__sample_weight':weights} if family=='local_ridge' else {'sample_weight':weights}))
                pred=model.predict(Z[va]); detail={}
            error=macro_error(y[va],pred,ids[va]); persistence=macro_error(y[va],np.zeros(len(va)),ids[va])
            scores.append(error/max(persistence,1e-8))
            details.append({'fold':fold,'mae_c':error,'persistence_mae_c':persistence,**detail})
            rows.append(pd.DataFrame({'time':times[va],'sensor':meta.sensor.to_numpy()[ids[va]],'fold':fold,
                                     'observed':y[va]+base[va],'persistence':base[va],'prediction':pred+base[va]}))
        value=float(np.mean(scores)); trial.set_user_attr('fold_metrics',details)
        prediction_path=out/f'trial_{trial.number:03d}_predictions.csv.gz'
        temporary=prediction_path.with_suffix('.tmp')
        pd.concat(rows).to_csv(temporary,index=False,compression='gzip'); temporary.replace(prediction_path)
        save(out/f'trial_{trial.number:03d}.json',{'params':params,'selection_score':value,'folds':details,
            'prediction_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
        print(out.name,'trial',trial.number,'selection ratio',round(value,4),flush=True)
        return value
    study.optimize(objective,n_trials=1 if quick else cfg['trials_per_family'])
    save(out/'selection_summary.json',{'best_trial':study.best_trial.number,'score':study.best_value,
         'params':study.best_params,'fold_metrics':study.best_trial.user_attrs['fold_metrics'],
         'completed_trials':len(study.trials),'status':'DEVELOPMENT_SELECTION_ONLY'})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--family',required=True); p.add_argument('--quick',action='store_true')
    p.add_argument('--task',choices=['forecast','site_outage']); p.add_argument('--hours',type=int); args=p.parse_args()
    output=ROOT/'results'; output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    status=output/f'status_{args.family}.json'
    save(status,{'state':'RUNNING','pid':os.getpid(),'family':args.family})
    try:
        for task in ([args.task] if args.task else ['forecast','site_outage']):
            for age in ([args.hours] if args.hours else [1,6,24]): run(output,task,age,args.family,args.quick)
        outputs=[p for d in output.glob('*_'+args.family) for p in d.iterdir() if p.suffix in ['.json','.gz']]
        save(status,{'state':'COMPLETED','family':args.family,
            'result_sha256':{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}})
    except BaseException as exc:
        save(status,{'state':'FAILED','error':str(exc),'family':args.family}); traceback.print_exc(); raise

if __name__=='__main__': main()
