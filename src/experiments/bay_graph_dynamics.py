"""Six-hour sensor-state dynamics; unit capacities are not bay water volumes."""
import argparse,hashlib,json,os,traceback
import numpy as np
import pandas as pd
import torch
import optuna
from src.data.bay_profile_context import ROOT,load_observations
from src.models.bay_graph_dynamics import BayGraphDynamics
from src.experiments.bay_twin_sweep import save


def build(variable):
    meta,frames=load_observations()
    names=[n for n,f in frames.items() if variable in f and f[variable].notna().any()]
    meta=meta.loc[names]
    index=pd.date_range('2024-07-24','2025-12-31T23:00:00',freq='h',tz='UTC')
    raw=pd.DataFrame({n:frames[n][variable].reindex(index) for n in names},index=index)
    center,scale=(20.,10.) if variable=='temperature' else (230.,50.)
    # Reindex hourly before filling; never fill labels or observation-update data.
    latest=(raw.ffill(limit=6)-center)/scale
    z=(raw-center)/scale
    q=np.arange(24,len(index)-6,6)
    x=latest.to_numpy()[q];available=np.isfinite(x)
    truth=z.to_numpy()[q+6];valid=np.isfinite(z.to_numpy()[q])&np.isfinite(truth)
    trend=z.to_numpy()[q]-z.to_numpy()[q-6]
    xy=meta[['x_m','y_m']].to_numpy();distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    adjacency=np.exp(-distance/1500);np.fill_diagonal(adjacency,0)
    static=np.column_stack([(xy[:,0]-568000)/5000,(xy[:,1]-4880000)/5000,meta.nominal_depth_m/10])
    calendar=np.column_stack([np.sin(index[q].dayofyear*2*np.pi/365.25),np.cos(index[q].dayofyear*2*np.pi/365.25),np.sin(index[q].hour*2*np.pi/24),np.cos(index[q].hour*2*np.pi/24)])
    features=[np.nan_to_num(trend)[...,None],np.isfinite(trend)[...,None],available[...,None],np.broadcast_to(static,(len(q),*static.shape)),np.broadcast_to(calendar[:,None,:],(len(q),len(names),4))]
    for col,offset,divisor in [('temperature',20,10),('salinity',35,5)]:
        a=pd.DataFrame({n:frames[n][col].reindex(index) if col in frames[n] else np.nan for n in names},index=index).ffill(limit=6).to_numpy()[q]
        features.extend([np.nan_to_num((a-offset)/divisor)[...,None],np.isfinite(a)[...,None]])
    forcing=np.concatenate(features,axis=-1).astype('float32')
    future=np.stack([z.to_numpy()[q+h] for h in range(1,7)],axis=1)
    return dict(names=names,times=index[q],x=x,available=available,truth=truth,valid=valid,forcing=forcing,
                future=future,adjacency=adjacency,near=distance<=250,distance=distance,center=center,scale=scale)


def folds(data,quick):
    t=data['times'];out=[]
    for number,(start,end) in enumerate([('2025-07-01','2025-10-01'),('2025-10-01','2026-01-01')]):
        begin=pd.Timestamp(start,tz='UTC');finish=pd.Timestamp(end,tz='UTC')
        tr=np.flatnonzero(t+pd.Timedelta(hours=6+72)<begin)
        va=np.flatnonzero((t>=begin)&(t+pd.Timedelta(hours=6)<finish))
        eligible=(data['valid'][tr].sum(0)>=100)&(data['valid'][va].sum(0)>=30)
        if not eligible.any():continue
        if quick:tr=tr[-100:];va=va[:20]
        fallback=pd.DataFrame(data['x'][tr]).median().to_numpy();fallback=np.nan_to_num(fallback)
        out.append(dict(number=number,tr=tr,va=va,eligible=eligible,fallback=fallback,
            audit={'fold':number,'last_training_outcome':str(t[tr].max()+pd.Timedelta(hours=6)),
                   'first_selection':str(t[va].min()),'selection_end_exclusive':end,
                   'eligible_sensors':[n for n,e in zip(data['names'],eligible) if e],
                   'training_counts':data['valid'][tr].sum(0).tolist(),'selection_counts':data['valid'][va].sum(0).tolist()}))
    return out


def tensors(data,fold,device):
    arrays={k:data[k] for k in ['available','forcing','truth','future']}
    arrays['x']=np.where(np.isfinite(data['x']),data['x'],fold['fallback'])
    arrays['valid']=data['valid']&fold['eligible'][None,:]
    arrays['future_mask']=np.isfinite(data['future'])
    return {k:torch.as_tensor(np.nan_to_num(a),device=device) for k,a in arrays.items()}


def fit(data,fold,params,exchange,seed,device,quick,relaxation=False):
    torch.manual_seed(seed);a=tensors(data,fold,device)
    model=BayGraphDynamics(data['adjacency'],data['forcing'].shape[-1],params['hidden'],exchange,relaxation).to(device)
    model=model.float();a={k:v.float() if v.dtype!=torch.bool else v for k,v in a.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
    counts=a['valid'][fold['tr']].sum(0).clamp_min(1);weights=1/counts.float();weights/=weights.mean()
    rng=np.random.default_rng(seed)
    for _ in range(3 if quick else params['steps']):
        ids=rng.choice(fold['tr'],min(128,len(fold['tr'])),replace=False)
        pred,_=model.rollout(a['x'][ids],a['available'][ids],a['forcing'][ids])
        mask=a['valid'][ids];loss=(torch.nn.functional.smooth_l1_loss(pred,a['truth'][ids],reduction='none')*mask*weights).sum()/(mask*weights).sum().clamp_min(1)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
    rows=[];diagnostics=[]
    with torch.no_grad():
        ids=fold['va'];x=a['x'][ids];available=a['available'][ids];forcing=a['forcing'][ids]
        forecast,budget=model.rollout(x,available,forcing)
        diagnostics.append({k:float(v.cpu()) for k,v in budget.items()})
        for target,name in enumerate(data['names']):
            if not fold['eligible'][target]:continue
            denied=torch.as_tensor(data['near'][target][None,:],device=device).expand(len(ids),-1)
            updated,budget=model.rollout(x,available,forcing,a['future'][ids],a['future_mask'][ids],denied)
            diagnostics.append({k:float(v.cpu()) for k,v in budget.items()})
            # Remote endpoint change baseline has exactly the same outage observation budget.
            remote=np.isfinite(data['future'][ids,-1])&np.isfinite(data['x'][ids])&~data['near'][target]
            weight=remote/(data['distance'][target]+100)**2
            delta=np.nan_to_num(data['future'][ids,-1]-data['x'][ids])
            change=(weight*delta).sum(1)/np.maximum(weight.sum(1),1e-20)
            for mode,pred,baseline in [('forecast',forecast[:,target],x[:,target].cpu().numpy()),('remote_update',updated[:,target],x[:,target].cpu().numpy()+change)]:
                p=pred.cpu().numpy()
                for j,idx in enumerate(ids):
                    if not data['valid'][idx,target]:continue
                    rows.append({'fold':fold['number'],'sensor':name,'time':str(data['times'][idx]),'mode':mode,
                        'truth':float(data['truth'][idx,target]*data['scale']+data['center']),
                        'prediction':float(p[j]*data['scale']+data['center']),
                        'persistence':float(data['x'][idx,target]*data['scale']+data['center']),
                        'baseline':float(baseline[j]*data['scale']+data['center'])})
    diagnostic={k:max(d[k] for d in diagnostics) for k in diagnostics[0]}
    diagnostic['exchange_rate']=float((.25*torch.sigmoid(model.log_rate)).detach().cpu()) if exchange else 0.
    diagnostic['relaxation_rate_per_hour']=float((.1*torch.sigmoid(model.log_relaxation)).detach().cpu()) if relaxation else 0.
    checkpoint={'state_dict':model.cpu().state_dict(),'params':params,'exchange':exchange,'relaxation':relaxation,'fallback':fold['fallback'].tolist(),'fold':fold['audit'],'names':data['names'],'center':data['center'],'scale':data['scale']}
    return pd.DataFrame(rows),diagnostic,checkpoint


def metrics(d):
    result={}
    for mode,s in d.groupby('mode'):
        for col in ['prediction','persistence','baseline']:
            result[mode+'_'+col+'_mae']=float(s.assign(error=(s.truth-s[col]).abs()).groupby(['fold','sensor']).error.mean().groupby('fold').mean().mean())
    return result


def run(variable,quick=False,reactive=False):
    output=ROOT/'results';output.mkdir(exist_ok=True);status=output/f'status_dynamics_{variable}.json'
    save(status,{'state':'RUNNING','variable':variable,'pid':os.getpid()})
    try:
        torch.set_num_threads(4);device='cuda:0' if torch.cuda.is_available() else 'cpu'
        if reactive and variable!='oxygen':raise ValueError('Saturation relaxation is oxygen-specific')
        data=build(variable);splits=folds(data,quick);folder=output/f'dynamics_{variable}';folder.mkdir(exist_ok=True)
        save(folder/'contract.json',{'variable':variable,'smoke_only':quick,'device':device,'folds':[f['audit'] for f in splits],
            'horizon_hours':6,'mask_radius_m':250,'reactive':reactive,'seeds':[42] if quick else [42,7,123],
            'scope':'Raw Gen2 endpoints; no 2026; development only. Symmetric exchange over unit-capacity sensor nodes, not surveyed water volumes. Frozen issuance forcing. Forecast excludes future observations; remote updates deny whole target site.'})
        def evaluate(params,seed,prefix,keep=False):
            report={'params':params,'seed':seed,'families':{}}
            families=[('learned_source',False,False),('conservative_exchange',True,False)]
            if reactive:families += [('saturation_source',False,True),('saturation_exchange',True,True)]
            for family,exchange,relaxation in families:
                frames=[];budgets=[];states=[]
                for fold in splits:
                    d,b,s=fit(data,fold,params,exchange,seed,device,quick,relaxation);frames.append(d);budgets.append(b);states.append(s)
                d=pd.concat(frames,ignore_index=True);path=folder/f'{prefix}_{family}_predictions.csv.gz';d.to_csv(path,index=False)
                report['families'][family]={'metrics':metrics(d),'budgets':budgets,'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                if keep:torch.save(states,folder/f'{prefix}_{family}.pt')
            save(folder/(prefix+'.json'),report);return report
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=4))
        def objective(trial):
            params={'hidden':trial.suggest_categorical('hidden',[16,32,64]),'lr':trial.suggest_float('lr',1e-4,3e-3,log=True),
                    'weight_decay':trial.suggest_float('weight_decay',1e-5,.1,log=True),'steps':trial.suggest_categorical('steps',[150,300])}
            r=evaluate(params,42,f'trial_{trial.number:03d}')
            return float(np.mean([a['metrics']['forecast_prediction_mae']/a['metrics']['forecast_persistence_mae'] for a in r['families'].values()]))
        study.optimize(objective,n_trials=1 if quick else 8)
        save(folder/'selection_summary.json',{'trial':study.best_trial.number,'params':study.best_params,'objective':'Mean forecast/persistence MAE ratio across all paired families; same configuration for paired comparison.'})
        for seed in ([42] if quick else [42,7,123]):evaluate(study.best_params,seed,f'selected_seed{seed}',True)
        save(status,{'state':'COMPLETED','variable':variable,'device':device,'result_sha256':{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}})
    except BaseException as e:
        save(status,{'state':'FAILED','error':str(e)});traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variable',choices=['temperature','oxygen'],required=True);p.add_argument('--quick',action='store_true');p.add_argument('--reactive',action='store_true');a=p.parse_args();run(a.variable,a.quick,a.reactive)
