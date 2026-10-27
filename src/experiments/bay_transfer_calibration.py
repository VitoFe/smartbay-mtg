"""Frozen dense-site transfer and forward-block residual calibration experiments."""
import argparse,hashlib,json,os,traceback
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from src.data.bay_profile_context import ROOT,TARGETS,load_profiles,load_observations,context_features
from src.data.bay_thermal_supervision import thermal_rows
from src.experiments.bay_split_calibration import auxiliary_boundary,SOURCE,SCALE
from src.experiments.bay_field_export import EXTRAS,predict_checkpoint
from src.experiments.bay_twin_sweep import save

ALPHA=65648.17611753459


def groups(meta,radius=250):
    """Connected spatial groups; no close colocated probes cross the holdout."""
    xy=meta[['x_m','y_m']].to_numpy();pending=set(range(len(meta)));result=[]
    while pending:
        stack=[min(pending)];group=set(stack);pending-=group
        while stack:
            i=stack.pop();near={j for j in pending if np.linalg.norm(xy[i]-xy[j])<=radius}
            pending-=near;group|=near;stack.extend(near)
        result.append(meta.index[sorted(group)].tolist())
    return result


def fit_predict(raw,extra,y,base,tr,weights,alpha=ALPHA,tree=False):
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(raw.iloc[tr])
    a=imputer.transform(raw);s=StandardScaler().fit(a[tr]);a=s.transform(a).astype('float32')
    a=np.column_stack([a,extra]);s2=StandardScaler().fit(a[tr]);a=s2.transform(a).astype('float32')
    target=y-base
    if tree:
        model=LGBMRegressor(n_estimators=150,num_leaves=15,max_depth=5,min_child_samples=40,learning_rate=.03,reg_lambda=10,verbosity=-1,n_jobs=4,random_state=42)
        model.fit(a[tr],target[tr],sample_weight=weights)
    else:model=Ridge(alpha=alpha,fit_intercept=False).fit(a[tr],target[tr],sample_weight=weights)
    return base+model.predict(a)


def ctd_predict(data,raw,extra,tr,alpha=ALPHA):
    train=data.iloc[tr];fallback=np.median(train[TARGETS].to_numpy(),axis=0)
    base=raw[['temperature_context','salinity_context','oxygen_context']].to_numpy()
    base=np.where(np.isfinite(base),base,fallback).astype('float32');base=(base+extra[:,3:6]).astype('float32')
    base[:,0]=np.clip(base[:,0],-2,40);base[:,1]=np.clip(base[:,1],0,42)
    w=1/(train.groupby('cast_id').cast_id.transform('size').to_numpy()*train.groupby('station').cast_id.transform('nunique').to_numpy());w=(w/w.mean()).astype('float32')
    pred=fit_predict(raw,extra,data[TARGETS].to_numpy(dtype='float32').astype(float)/SCALE,base.astype(float)/SCALE,tr,w,alpha)*SCALE
    pred[:,0]=np.clip(pred[:,0],-2,40);pred[:,1]=np.clip(pred[:,1],0,42)
    return pred


def forward_blocks(data,ids,block_size=10):
    casts=data.iloc[ids].groupby('cast_id').t_utc.first().sort_values()
    for begin in range(15,len(casts),block_size):
        calids=casts.iloc[begin:begin+block_size].index
        cutoff=casts.loc[calids].min()-pd.Timedelta(hours=72)
        train=data.iloc[ids];train=train[train.t_utc<cutoff]
        if train.cast_id.nunique()<15:continue
        yield train.index.to_numpy(),data.index[data.cast_id.isin(calids)].to_numpy()


def blocked(root):
    data,_=load_profiles();meta,frames=load_observations();extra=data[['cast_id','depth_m']].merge(pd.read_csv(ROOT/'reports/bay_twin/profile_boundary_context.csv.gz'),on=['cast_id','depth_m'],validate='one_to_one')[EXTRAS].to_numpy(float)
    rows=[];cal=[];contracts=[];cache={}
    for task in ['temporal','held_site']:
        for path in sorted((ROOT/'reports/bay_twin/runs'/SOURCE/'results'/f'field_export_{task}').glob('*_model.json')):
            cp=json.loads(path.read_text());c=cp['contract'];held=c['held_station']
            if held not in cache:cache[held]=context_features(data.drop(columns=TARGETS),meta,frames,held)
            raw=cache[held];ids=data.index[data.cast_id.isin(c['training_cast_ids'])].to_numpy();va=data.index[data.cast_id.isin(c['selection_cast_ids'])].to_numpy()
            pred=predict_checkpoint(cp,raw,extra);pred[:,0]=np.clip(pred[:,0],-2,40);pred[:,1]=np.clip(pred[:,1],0,42)
            r=data.iloc[va][['cast_id','station','t_utc','depth_m']+TARGETS].copy();r['task']=task;r['checkpoint']=c['id'];r['fold']=c['fold']
            for j,t in enumerate(TARGETS):r['pred_'+t]=pred[va,j]
            rows.append(r);blocks=[]
            for block,(tr,ca) in enumerate(forward_blocks(data,ids)):
                p=ctd_predict(data,raw,extra,tr,cp['alpha'])
                r=data.iloc[ca][['cast_id','station','t_utc','depth_m']+TARGETS].copy();r['task']=task;r['checkpoint']=c['id'];r['block']=block
                for j,t in enumerate(TARGETS):r['pred_'+t]=p[ca,j]
                cal.append(r);blocks.append({'block':block,'training_cast_ids':sorted(data.iloc[tr].cast_id.unique()),'calibration_cast_ids':sorted(data.iloc[ca].cast_id.unique())})
            contracts.append(dict(task=task,checkpoint=c['id'],held_station=held,training_cast_ids=c['training_cast_ids'],selection_cast_ids=c['selection_cast_ids'],blocks=blocks))
            print('blocked',task,c['id'],len(blocks),flush=True)
    pd.concat(rows).to_csv(root/'predictions.csv.gz',index=False);pd.concat(cal).to_csv(root/'calibration.csv.gz',index=False)
    save(root/'contract.json',{'folds':contracts,'policy':'Forward blocks of 10 casts after first15; >=15 train casts;72h embargo. Final point model uses all original pre-evaluation training casts. Residuals from earlier fits, no exchangeability or conformal guarantee. Fixed absolute whole-profile scores; queried station excluded; minimum10 casts and2 stations.'})


def dense(root):
    meta,frames=load_observations();rows=[];contracts=[]
    for group_id,held in enumerate(groups(meta)):
        d=thermal_rows(meta,frames,excluded_source_sensors=held);d.t_utc=pd.to_datetime(d.t_utc,utc=True)
        extra=auxiliary_boundary(d,meta);raw=d.drop(columns=['sensor','t_utc','target_temperature'])
        for fold,(start,end) in enumerate([('2025-01-01','2025-07-01'),('2025-07-01','2026-01-01')]):
            start=pd.Timestamp(start,tz='UTC');end=pd.Timestamp(end,tz='UTC')
            tr=np.flatnonzero((d.t_utc<start-pd.Timedelta(hours=72))&~d.sensor.isin(held))
            va=np.flatnonzero((d.t_utc>=start)&(d.t_utc<end)&d.sensor.isin(held))
            if not len(va):continue
            assert len(tr)>=100
            w=1/d.iloc[tr].groupby('sensor').sensor.transform('size').to_numpy();w=w/w.mean()
            bias=np.average(d.target_temperature.to_numpy()[tr]-extra[tr,0],weights=w)
            physical=np.clip(extra[:,0]+bias,-2,40)
            contract={'group':group_id,'fold':fold,'held_sensors':held,'training_sensors':sorted(d.iloc[tr].sensor.unique()),'training_last':str(d.iloc[tr].t_utc.max()),'selection_first':str(d.iloc[va].t_utc.min()),'training_rows':len(tr),'evaluation_rows':len(va),'boundary_bias':float(bias)}
            contracts.append(contract)
            for scenario in ['remote_available','network_outage24','network_outage72']:
                qraw=raw.copy()
                if scenario!='remote_available':
                    for v in ['temperature','salinity','oxygen']:
                        for suffix in ['context','spread','age','distance']:qraw.loc[va,v+'_'+suffix]=np.nan
                        qraw.loc[va,v+'_count']=0
                b=np.where(np.isfinite(qraw.temperature_context),qraw.temperature_context+extra[:,3],physical)
                # Each scenario changes evaluation features only; training and fit are fixed.
                predictions={'physical_background':physical,'remote_idw':qraw.temperature_context.to_numpy(),
                    'ridge_residual':fit_predict(qraw,extra,d.target_temperature.to_numpy(),b,tr,w),
                    'tree_residual':fit_predict(qraw,extra,d.target_temperature.to_numpy(),b,tr,w,tree=True)}
                for family,p in predictions.items():
                    r=d.iloc[va][['sensor','t_utc','target_temperature']].copy();r['prediction']=np.clip(p[va],-2,40);r['family']=family;r['scenario']=scenario;r['fold']=fold;r['group']=group_id
                    r['source_count']=qraw.iloc[va].temperature_count.to_numpy();rows.append(r)
                # Prior target measurements are allowed ONLY for this outage baseline,
                # not for unseen-site model fitting or remote-context inputs.
                if scenario!='remote_available':
                    hours=int(scenario.removeprefix('network_outage'));values=[];times=[]
                    for q in d.iloc[va].itertuples():
                        cutoff=q.t_utc-pd.Timedelta(hours=hours);s=frames[q.sensor].temperature
                        s=s.loc[(s.index<cutoff)&(s.index>=cutoff-pd.Timedelta(hours=6))].dropna()
                        values.append(float(s.iloc[-1]) if len(s) else np.nan);times.append(str(s.index[-1]) if len(s) else None)
                    r=d.iloc[va][['sensor','t_utc','target_temperature']].copy();r['prediction']=values;r['source_time']=times;r['family']='local_preoutage_persistence';r['scenario']=scenario;r['fold']=fold;r['group']=group_id;rows.append(r)
            print('dense',group_id,fold,len(tr),len(va),flush=True)
    pd.concat(rows).to_csv(root/'predictions.csv.gz',index=False)
    save(root/'contract.json',{'folds':contracts,'policy':'Connected250m held groups absent from all training labels and context sources. Gen2 observed12h targets only; two2025 half-years;72h training embargo. Fixed ridge alpha and fixed150-tree residual learner. Network outages24/72 expire all6h remote context, hence identical model inputs. Local persistence uses held-site pre-outage history and is only an outage comparator; unavailable predictions retained. No oxygen targets, no2026, no claim of coordinate support beyond sampled sites.'})


def run(task):
    out=ROOT/'results';out.mkdir(exist_ok=True);status=out/f'status_transfer_calibration_{task}.json';save(status,{'state':'RUNNING','pid':os.getpid()})
    try:
        root=out/f'transfer_calibration_{task}';root.mkdir(exist_ok=True)
        (dense if task=='dense' else blocked)(root)
        save(status,{'state':'COMPLETED','task':task,'result_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}})
    except BaseException as exc:save(status,{'state':'FAILED','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['dense','blocked'],required=True);run(p.parse_args().task)
