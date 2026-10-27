"""Four-quarter retrospective stress test; frozen tree settings, causal labels.

Earlier hyperparameter selection used 2025 development data. These additional
quarters broaden diagnostics, not independent validation of those parameters.
"""
import argparse,hashlib,json,traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.experiments.bay_graph_dynamics import ROOT,build,save
from src.experiments.bay_dynamics_tree import features
from src.data.bay_forcing import weather
from src.protocols.bay_calibration import finite_radius


def interval_rows(cal,d,policy):
    c=cal.copy();q=d.copy()
    c['time']=pd.to_datetime(c.time,utc=True);q['time']=pd.to_datetime(q.time,utc=True)
    history=pd.concat([c,q],ignore_index=True);available=history.time+pd.Timedelta(hours=7)
    rows=[]
    for r in q.itertuples():
        pool=c[c.time+pd.Timedelta(hours=7)<=r.time]
        state='fixed'
        if policy=='rolling30':
            recent=history[(available<=r.time)&(available>=r.time-pd.Timedelta(days=30))]
            if len(recent)>=30:pool=recent;state='rolling30'
            else:state='fixed_fallback'
        supported=len(pool)>=30
        radius=finite_radius(pool.truth-pool.prediction) if supported else np.nan
        rows.append({**r._asdict(),'time':str(r.time),'policy':policy,'lower':r.prediction-radius,'upper':r.prediction+radius,
            'interval_status':state if supported else 'insufficient_calibration','residual_count':len(pool),
            'latest_residual_available':str(pool.time.max()+pd.Timedelta(hours=7)) if len(pool) else None})
    return pd.DataFrame(rows).drop(columns='Index',errors='ignore')


def run(variable):
    root=ROOT/'results';out=root/f'seasonal_{variable}';out.mkdir(parents=True,exist_ok=True)
    status=root/f'status_seasonal_{variable}.json';save(status,{'state':'RUNNING','variable':variable})
    try:
        data=build(variable);t=data['times'];met=weather(t).to_numpy()
        params=json.loads((ROOT/f'reports/bay_twin/runs/20260910T120301Z_dynamics_tree_pilot/results/dynamics_{variable}/selection_summary.json').read_text())['params']
        rows=[];calrows=[];eligibility=[];contracts=[]
        boundaries=['2025-01-01','2025-04-01','2025-07-01','2025-10-01','2026-01-01']
        for fold,(begin,finish) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            start=pd.Timestamp(begin,tz='UTC');end=pd.Timestamp(finish,tz='UTC')
            cal_end=start-pd.Timedelta(hours=72);cal_start=cal_end-pd.Timedelta(days=45)
            tr=np.flatnonzero(t+pd.Timedelta(hours=78)<cal_start)
            ca=np.flatnonzero((t>=cal_start)&(t+pd.Timedelta(hours=6)<cal_end))
            va=np.flatnonzero((t>=start)&(t+pd.Timedelta(hours=6)<end))
            eligible=data['valid'][tr].sum(0)>=100
            for s,name in enumerate(data['names']):eligibility.append({'fold':fold,'sensor':name,'train_count':int(data['valid'][tr,s].sum()),'calibration_count':int(data['valid'][ca,s].sum()),'selection_count':int(data['valid'][va,s].sum()),'trained_support':bool(eligible[s])})
            if not eligible.any():continue
            fallback=np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
            base,_=features(data,{'fallback':fallback},'forecast')
            x_weather=np.concatenate([base,np.broadcast_to(met[:,None],(len(t),len(data['names']),met.shape[1]))],axis=-1)
            mask=data['valid']&eligible[None];tm=mask[tr]
            weight=1/np.maximum(tm.sum(0),1);w=np.broadcast_to(weight,tm.shape)[tm];w/=w.mean()
            contract={'fold':fold,'selection_start':str(start),'selection_end':str(end),'last_training_outcome':str(t[tr].max()+pd.Timedelta(hours=6)),
              'first_calibration':str(t[ca].min()),'last_calibration_outcome':str(t[ca].max()+pd.Timedelta(hours=6)),
              'supported_sensors':[n for n,e in zip(data['names'],eligible) if e]}
            contracts.append(contract)
            for family,x in [('tree',base),('weather_tree',x_weather)]:
                model=lgb.LGBMRegressor(**params,objective='regression_l1',verbosity=-1,n_jobs=4,random_state=42,deterministic=True,force_col_wise=True)
                model.fit(x[tr][tm],(data['truth'][tr]-data['x'][tr])[tm],sample_weight=w)
                save(out/f'{family}_fold{fold}_model.json',{'model_string':model.booster_.model_to_string(),'params':params,'fallback':fallback.tolist(),
                  'names':data['names'],'center':data['center'],'scale':data['scale'],'feature_count':x.shape[-1],'contract':contract})
                phase_rows={}
                for phase,ids in [('calibration',ca),('selection',va)]:
                    pm=mask[ids];pred=model.predict(x[ids][pm]);records=[]
                    for (j,s),value in zip(np.argwhere(pm),pred):
                        i=ids[j];anchor=float(data['x'][i,s]*data['scale']+data['center'])
                        records.append({'fold':fold,'sensor':data['names'][s],'time':str(t[i]),'family':family,
                            'truth':float(data['truth'][i,s]*data['scale']+data['center']),'prediction':float(anchor+value*data['scale']),'persistence':anchor})
                    phase_rows[phase]=pd.DataFrame(records)
                c=phase_rows['calibration'];d=phase_rows['selection'];calrows.append(c)
                for sensor,g in d.groupby('sensor'):
                    cs=c[c.sensor==sensor]
                    for policy in ['fixed','rolling30']:rows.append(interval_rows(cs,g,policy))
            print(variable,'completed quarter',fold+1,flush=True)
        pd.concat(rows,ignore_index=True).to_csv(out/'predictions.csv.gz',index=False)
        pd.concat(calrows,ignore_index=True).to_csv(out/'calibration.csv.gz',index=False)
        pd.DataFrame(eligibility).to_csv(out/'eligibility.csv.gz',index=False)
        save(out/'contract.json',{'variable':variable,'horizon_hours':6,'folds':contracts,'params':params,
            'scope':'Four 2025 quarters; frozen previously selected development parameters, not an untouched test. Training support >=100 true endpoints; interval support >=30 past residuals. All forecasts retain a known local anchor. Salinity availability does not filter targets.',
            'interval_policy':'Fixed earlier 45-day sample versus rolling30 with fixed fallback. Six-hour outcomes usable after an additional hour. Missing intervals are retained, not dropped from point-error evaluation.',
            'limitations':['Quarterly covariate shift; selection settings previously informed by 2025.','Known-sensor forecasts are not spatial oxygen estimates.','No animal outcomes, 2026 values or operational coverage guarantee.']})
        save(status,{'state':'COMPLETED','variable':variable,'result_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}})
    except BaseException as e:
        save(status,{'state':'FAILED','error':str(e)});traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variable',choices=['temperature','oxygen'],required=True);a=p.parse_args();run(a.variable)
