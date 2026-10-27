"""Joint neural profile field: causal remote context and paired stability ablations."""
from pathlib import Path
import argparse,hashlib,json,os,traceback
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.data.bay_profile_context import ROOT,TARGETS,load_profiles,load_observations,context_features
from src.models.bay_query_field import BayQueryField,stability_loss,density_zero_pressure
from src.experiments.bay_twin_sweep import save

SCALE=np.array([2.,1.,20.])
NN_FAMILIES=['direct_mlp','thermo_mlp','stability_01','stability_1']


def aggregate(data):
    metrics={}
    for target in TARGETS:
        error=(data[target]-data['pred_'+target]).abs()
        # Casts first, then stations and calendar folds: repeated Baia profiles
        # and deeper profiles must not dominate the field objective.
        z=data[['fold','station','cast_id']].assign(error=error)
        metrics[target+'_mae']=float(z.groupby(['fold','station','cast_id']).error.mean().groupby(['fold','station']).mean().groupby('fold').mean().mean())
    metrics['normalized_score']=float(np.mean([metrics[t+'_mae']/s for t,s in zip(TARGETS,SCALE)]))
    return metrics


def profile_pairs(d):
    upper=[]; lower=[]; dz=[]
    for _,g in d.groupby('cast_id'):
        ids=g.sort_values('depth_m').index.to_numpy()
        delta=np.diff(d.loc[ids,'depth_m'])
        valid=delta>0
        upper.extend(ids[:-1][valid]); lower.extend(ids[1:][valid]); dz.extend(delta[valid])
    return np.array(upper,dtype=int),np.array(lower,dtype=int),np.array(dz,dtype=float)


def prepare(task,cfg,quick):
    data,excluded=load_profiles(); meta,frames=load_observations()
    contexts={}; bundles=[]; contracts=[]
    stations=sorted(data.station.unique()) if task=='held_site' else [None]
    if quick and task=='held_site': stations=['PBoaT']
    for held in stations:
        contexts[held]=context_features(data,meta,frames,held)
    for fold_id,f in enumerate(cfg['folds']):
        if quick and fold_id>0: break
        start=pd.Timestamp(f['train_end'],tz='UTC'); end=pd.Timestamp(f['selection_end'],tz='UTC')
        selection_period=(data.t_utc>=start)&(data.t_utc<end)
        counts=data.loc[selection_period].groupby('station').cast_id.nunique()
        eligible=counts[counts>=2].index
        for held in stations:
            train=(data.t_utc<start-pd.Timedelta(hours=72))
            val=selection_period & data.station.isin(eligible)
            if held is not None: train &=data.station!=held; val &=data.station==held
            if data.loc[train,'cast_id'].nunique()<15 or data.loc[val,'cast_id'].nunique()<2: continue
            tr=np.flatnonzero(train); va=np.flatnonzero(val); X=contexts[held]
            fallback=np.median(data.loc[train,TARGETS].to_numpy(),axis=0)
            base=X[['temperature_context','salinity_context','oxygen_context']].to_numpy().copy()
            for j in range(3): base[~np.isfinite(base[:,j]),j]=fallback[j]
            imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(X.iloc[tr])
            raw=imputer.transform(X); scaler=StandardScaler().fit(raw[tr]); sx=scaler.transform(raw).astype('float32')
            training=data.iloc[tr].reset_index(drop=True)
            cast_n=training.groupby('cast_id').cast_id.transform('size').to_numpy()
            station_casts=training.groupby('station').cast_id.transform('nunique').to_numpy()
            weights=1/(cast_n*station_casts); weights/=weights.mean()
            pairs=profile_pairs(training)
            label=data[TARGETS].to_numpy()
            identity=f'fold{fold_id}_{held or "all"}'
            contract={'id':identity,'fold':fold_id,'held_station':held,
                'training_cast_ids':sorted(data.loc[train,'cast_id'].unique()),'selection_cast_ids':sorted(data.loc[val,'cast_id'].unique()),
                'training_stations':sorted(data.loc[train,'station'].unique()),'selection_stations':sorted(data.loc[val,'station'].unique()),
                'last_training':str(data.loc[train,'t_utc'].max()),'first_selection':str(data.loc[val,'t_utc'].min()),
                'last_selection':str(data.loc[val,'t_utc'].max()),'training_fallback':fallback.tolist(),
                'selection_bins':len(va),'n_features':sx.shape[1]}
            bundles.append({'identity':identity,'fold':fold_id,'tr':tr,'va':va,'sx':sx,'base':base.astype('float32'),
                'y':label.astype('float32'),'weights':weights.astype('float32'),'pairs':pairs,
                'selection':data.iloc[va].reset_index(drop=True),'raw_features':list(X),
                'imputation_statistics':imputer.statistics_.tolist(),'indicator_features':imputer.indicator_.features_.tolist(),
                'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(), 'contract':contract})
            contracts.append(contract)
    if not bundles: raise ValueError('No eligible profile folds')
    return bundles,contracts,excluded


def prediction_frame(bundle,pred):
    d=bundle['selection'][['cast_id','station','t_utc','depth_m']+TARGETS].copy()
    d['fold']=bundle['fold']
    for j,t in enumerate(TARGETS): d['pred_'+t]=pred[:,j]
    return d


def fit_neural(b,params,family,seed,device,quick=False):
    torch.manual_seed(seed)
    tr,va=b['tr'],b['va']
    tx=torch.as_tensor(b['sx'][tr],device=device); bx=torch.as_tensor(b['base'][tr],device=device)
    target=torch.as_tensor(b['y'][tr],device=device); weights=torch.as_tensor(b['weights'],device=device)
    scale=torch.tensor(SCALE,dtype=torch.float32,device=device)
    pairs=[torch.as_tensor(v,device=device,dtype=torch.float32 if j==2 else torch.long) for j,v in enumerate(b['pairs'])]
    model=BayQueryField(tx.shape[1],params['hidden'],family!='direct_mlp').to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
    guidance={'direct_mlp':0.,'thermo_mlp':0.,'stability_01':.1,'stability_1':1.}[family]
    for _ in range(8 if quick else params['epochs']):
        optimizer.zero_grad(); pred=model(tx,bx)
        fit=(torch.nn.functional.smooth_l1_loss(pred/scale,target/scale,reduction='none').mean(1)*weights).mean()
        stability=stability_loss(pred,*pairs)
        loss=fit+guidance*stability
        if not torch.isfinite(loss): raise ValueError('Nonfinite neural loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step()
    with torch.no_grad():
        pred=model(torch.as_tensor(b['sx'][va],device=device),torch.as_tensor(b['base'][va],device=device)).cpu().numpy()
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    return pred,{'data_loss':float(fit.detach().cpu()),'stability_loss':float(stability.detach().cpu()),'weight':guidance},state


def run(task,quick=False,output=None):
    output=output or ROOT/'results'; output.mkdir(parents=True,exist_ok=True)
    status=output/f'status_field_{task}.json'; save(status,{'state':'RUNNING','task':task,'pid':os.getpid()})
    try:
        torch.set_num_threads(4); device='cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg=json.loads((ROOT/'config/bay_profile_field.json').read_text())
        bundles,contracts,excluded=prepare(task,cfg,quick)
        root=output/f'field_{task}'; root.mkdir(exist_ok=True)
        save(root/'contract.json',{'task':task,'smoke_only':quick,'device':device,'config':cfg,'folds':contracts,'excluded_bottom_metadata_casts':excluded,
            'objective':'Equal cast, station, fold mean absolute errors divided by [2 C, 1 PSU, 20 umol/L], then equal target mean.',
            'selection':'Common Optuna proposals selected by unguided thermo_mlp; guidance arms have identical configurations and initialization. Per-arm minima are exploratory.',
            'features':bundles[0]['raw_features'],'physics':'UNESCO zero-pressure density stability prior; not TEOS-10 buoyancy or a transport PDE.'})
        print(task,'prepared',len(bundles),'fits per arm; device',device,flush=True)
        baseline_rows=[]; tree_rows=[]
        for b in bundles:
            baseline_rows.append(prediction_frame(b,b['base'][b['va']]))
            predictions=[]
            for j in range(3):
                model=lgb.LGBMRegressor(n_estimators=200,num_leaves=15,min_child_samples=30,learning_rate=.03,reg_lambda=10,
                    verbosity=-1,n_jobs=4,random_state=42,objective='regression_l1')
                model.fit(b['sx'][b['tr']],b['y'][b['tr'],j]-b['base'][b['tr'],j],sample_weight=b['weights'])
                predictions.append(b['base'][b['va'],j]+model.predict(b['sx'][b['va']]))
            tree_rows.append(prediction_frame(b,np.stack(predictions,axis=1)))
        for family,rows in [('remote_idw',baseline_rows),('residual_tree',tree_rows)]:
            d=pd.concat(rows,ignore_index=True); d.to_csv(root/(family+'_predictions.csv.gz'),index=False)
            save(root/(family+'_metrics.json'),aggregate(d))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=cfg['seed'],n_startup_trials=cfg['startup_trials']),
            storage='sqlite:///'+str(root/'optuna.sqlite3'),study_name=root.name)
        best={family:float('inf') for family in NN_FAMILIES}

        def evaluate(params,seed,stem):
            results={}
            for family in NN_FAMILIES:
                rows=[]; losses=[]; states={}
                for b in bundles:
                    pred,loss,state=fit_neural(b,params,family,seed,device,quick)
                    rows.append(prediction_frame(b,pred)); losses.append({'fit':b['identity'],**loss}); states[b['identity']]=state
                data=pd.concat(rows,ignore_index=True); metrics=aggregate(data)
                path=root/f'{stem}_{family}_predictions.csv.gz'; data.to_csv(path,index=False)
                results[family]={'metrics':metrics,'training_losses':losses,'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                if metrics['normalized_score']<best[family] and stem.startswith('trial_'):
                    best[family]=metrics['normalized_score']
                    torch.save({'family':family,'params':params,'seed':seed,'states':states,
                        'preprocessing':{b['identity']:{k:b[k] for k in ['raw_features','imputation_statistics','indicator_features','scaler_mean','scaler_scale','contract']} for b in bundles}},root/(family+'_best.pt'))
                print(task,stem,family,round(metrics['normalized_score'],4),flush=True)
            save(root/(stem+'.json'),{'params':params,'seed':seed,'families':results})
            return results

        def objective(trial):
            params={'hidden':trial.suggest_categorical('hidden',[32,64]),'epochs':trial.suggest_categorical('epochs',[120,240]),
                    'learning_rate':trial.suggest_float('learning_rate',.0005,.005,log=True),
                    'weight_decay':trial.suggest_float('weight_decay',.001,.1,log=True)}
            return evaluate(params,cfg['seed'],f'trial_{trial.number:03d}')['thermo_mlp']['metrics']['normalized_score']
        study.optimize(objective,n_trials=1 if quick else cfg['trials'])
        save(root/'selection_summary.json',{'trial':study.best_trial.number,'params':study.best_params,'score':study.best_value,
            'scope':'Shared configuration selected by unguided thermodynamic network; confirmation seeds do not retune.'})
        if not quick:
            for seed in cfg['confirmation_seeds']: evaluate(study.best_params,seed,f'confirmation_seed{seed}')
        files=[p for p in root.iterdir() if p.name.endswith(('.json','.csv.gz','.pt'))]
        save(status,{'state':'COMPLETED','task':task,'device':device,'result_sha256':{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    except BaseException as exc:
        save(status,{'state':'FAILED','task':task,'error':str(exc)}); traceback.print_exc(); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['temporal','held_site'],required=True)
    p.add_argument('--quick',action='store_true'); p.add_argument('--output',type=Path)
    a=p.parse_args(); run(a.task,a.quick,a.output)
