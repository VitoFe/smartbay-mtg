"""Matched-cohort tree benchmark for sensor dynamics and remote updates."""
import argparse,hashlib,traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from src.experiments.bay_graph_dynamics import ROOT,build,folds,metrics,save


def features(data,fold,mode):
    x=np.where(np.isfinite(data['x']),data['x'],fold['fallback'])
    k=data['adjacency'][None]*data['available'][:,:,None]*data['available'][:,None,:]
    neighbor=np.einsum('bij,bj->bi',k,x)/np.maximum(k.sum(-1),1e-6)
    neighbor=np.where(k.sum(-1)>0,neighbor,x)
    features=[x[...,None],neighbor[...,None],data['forcing']]
    remote_change=np.zeros_like(x)
    if mode=='remote_update':
        for h in range(6):
            change=np.zeros_like(x);count=np.zeros_like(x)
            for target in range(len(data['names'])):
                valid=np.isfinite(data['future'][:,h])&data['available']&~data['near'][target]
                w=valid/(data['distance'][target]+100)**2
                delta=np.nan_to_num(data['future'][:,h]-data['x'])
                change[:,target]=(w*delta).sum(1)/np.maximum(w.sum(1),1e-20);count[:,target]=valid.sum(1)
            features += [change[...,None],count[...,None]]
            if h==5:remote_change=change
    return np.concatenate(features,axis=-1),x+remote_change


def run(variable,quick=False):
    output=ROOT/'results';output.mkdir(exist_ok=True);status=output/f'status_dynamics_{variable}.json'
    save(status,{'state':'RUNNING','variable':variable})
    try:
        data=build(variable);splits=folds(data,quick);folder=output/f'dynamics_{variable}';folder.mkdir(exist_ok=True)
        save(folder/'contract.json',{'variable':variable,'smoke_only':quick,'tree':True,'device':'cpu','folds':[f['audit'] for f in splits],
            'horizon_hours':6,'mask_radius_m':250,'seeds':[42],'scope':'Deterministic matched endpoint tree baseline; 2026 excluded. Same causal issuance inputs plus six remote-change summaries for outage estimation.'})
        cache={(f['number'],mode):features(data,f,mode) for f in splits for mode in ['forecast','remote_update']}
        def evaluate(params,prefix):
            rows=[]
            for fold in splits:
                mask=data['valid']&fold['eligible'][None,:]
                weights=1/np.maximum(mask[fold['tr']].sum(0),1)
                for mode in ['forecast','remote_update']:
                    features_,baseline=cache[fold['number'],mode]
                    tr=fold['tr'];va=fold['va'];train_mask=mask[tr];selection_mask=mask[va]
                    trainx=features_[tr][train_mask];y=(data['truth'][tr]-data['x'][tr])[train_mask]
                    w=np.broadcast_to(weights,train_mask.shape)[train_mask];w/=w.mean()
                    model=lgb.LGBMRegressor(**params,objective='regression_l1',verbosity=-1,n_jobs=4,random_state=42,deterministic=True,force_col_wise=True)
                    model.fit(trainx,y,sample_weight=w)
                    p=model.predict(features_[va][selection_mask])+data['x'][va][selection_mask]
                    for (j,target),value in zip(np.argwhere(selection_mask),p):
                        idx=va[j];rows.append({'fold':fold['number'],'sensor':data['names'][target],'time':str(data['times'][idx]),'mode':mode,
                            'truth':float(data['truth'][idx,target]*data['scale']+data['center']),
                            'prediction':float(value*data['scale']+data['center']),
                            'persistence':float(data['x'][idx,target]*data['scale']+data['center']),
                            'baseline':float(baseline[idx,target]*data['scale']+data['center'])})
                    if prefix.startswith('selected'):save(folder/f'{prefix}_fold{fold["number"]}_{mode}_model.json',model.booster_.dump_model())
            d=pd.DataFrame(rows);path=folder/f'{prefix}_tree_predictions.csv.gz';d.to_csv(path,index=False)
            report={'params':params,'seed':42,'families':{'tree':{'metrics':metrics(d),'budgets':[], 'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            save(folder/(prefix+'.json'),report);return report
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=4))
        def objective(trial):
            params={'num_leaves':trial.suggest_categorical('num_leaves',[7,15,31]),'n_estimators':trial.suggest_categorical('n_estimators',[150,400]),
                    'learning_rate':trial.suggest_float('learning_rate',.01,.1,log=True),'reg_lambda':trial.suggest_float('reg_lambda',1.,100.,log=True),'min_child_samples':trial.suggest_categorical('min_child_samples',[20,50,100])}
            r=evaluate(params,f'trial_{trial.number:03d}')['families']['tree']['metrics']
            return r['forecast_prediction_mae']/r['forecast_persistence_mae']
        study.optimize(objective,n_trials=1 if quick else 8)
        save(folder/'selection_summary.json',{'trial':study.best_trial.number,'params':study.best_params,'objective':'Forecast macro MAE / persistence; same settings used for remote-update model.'})
        evaluate(study.best_params,'selected_seed42')
        save(status,{'state':'COMPLETED','variable':variable,'device':'cpu','result_sha256':{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}})
    except BaseException as e:
        save(status,{'state':'FAILED','error':str(e)});traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variable',choices=['temperature','oxygen'],required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();run(a.variable,a.quick)
