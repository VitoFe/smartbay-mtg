"""Independent oxygen outage study with earlier-data model and policy fitting."""
import hashlib,json,os,traceback
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.data.bay_profile_context import ROOT,load_observations
from src.experiments.bay_split_calibration import auxiliary_boundary
from src.experiments.bay_transfer_calibration import groups
from src.experiments.bay_thermal_memory import QUARTERS
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.experiments.bay_twin_sweep import save

FAMILIES=['persistence','coarse025','coarse050','coarse100','solubility','ridge']
FEATURES=['initial_oxygen','initial_temperature','initial_salinity','coarse_delta','solubility_delta','temperature_delta','salinity_delta','depth_m','year_sin','year_cos']


def build_grid(meta,frames):
    mapping={s:i for i,g in enumerate(groups(meta)) for s in g};rows=[]
    for start in pd.date_range('2024-08-05','2025-12-22',freq='7D',tz='UTC'):
        for sensor in meta.index:
            f=frames[sensor];window=f.loc[(f.index<start)&(f.index>=start-pd.Timedelta(hours=6))];s=window.oxygen.dropna()
            stamp=s.index[-1] if len(s) else pd.NaT
            initial=float(s.iloc[-1]) if len(s) else np.nan
            ts={}
            for v in ['temperature','salinity']:
                observed=window[v].dropna() if v in window else pd.Series(dtype=float)
                ts['initial_'+v]=float(observed.iloc[-1]) if len(observed) else np.nan
            for h in [24,72]:
                q=start+pd.Timedelta(hours=h);rows.append(dict(sensor=sensor,group=mapping[sensor],episode=str(start),start=start,t_utc=q,horizon=h,depth_m=meta.loc[sensor,'nominal_depth_m'],initial_oxygen=initial,source_time=stamp,target_oxygen=f.oxygen.get(q,np.nan),**ts))
    return pd.DataFrame(rows)


def score(d,pred):
    return d.assign(error=abs(d.target_oxygen-pred)).groupby(['group','sensor']).error.mean().groupby('group').mean().mean()


def run():
    out=ROOT/'results';out.mkdir(exist_ok=True);status=out/'status_oxygen_memory.json';save(status,{'state':'RUNNING','pid':os.getpid()})
    try:
        root=out/'oxygen_memory';root.mkdir(exist_ok=True);meta,frames=load_observations();meta=meta.loc[[s for s,f in frames.items() if 'oxygen' in f and f.oxygen.notna().any()]]
        d=build_grid(meta,frames);now=auxiliary_boundary(d,meta);anchor=d.copy();anchor.t_utc=d.source_time.fillna(d.start);before=auxiliary_boundary(anchor,meta)
        d['coarse_delta']=now[:,2]-before[:,2];d['solubility_delta']=solubility(now[:,0],now[:,1])-solubility(before[:,0],before[:,1]);d['temperature_delta']=now[:,0]-before[:,0];d['salinity_delta']=now[:,1]-before[:,1]
        d['year_sin']=np.sin(d.t_utc.dt.dayofyear*2*np.pi/365.25);d['year_cos']=np.cos(d.t_utc.dt.dayofyear*2*np.pi/365.25)
        rows=[];contracts=[]
        for fold,(begin,end) in enumerate(zip(QUARTERS[:-1],QUARTERS[1:])):
            begin=pd.Timestamp(begin,tz='UTC');end=pd.Timestamp(end,tz='UTC');calstart=begin-pd.Timedelta(days=100)
            for h in [24,72]:
                tr=d[(d.horizon==h)&(d.start+pd.Timedelta(hours=72)<calstart-pd.Timedelta(hours=72))].dropna(subset=['initial_oxygen','target_oxygen'])
                ca=d[(d.horizon==h)&(d.start>=calstart)&(d.start+pd.Timedelta(hours=72)<begin-pd.Timedelta(hours=72))]
                ev=d[(d.horizon==h)&(d.start>=begin)&(d.start+pd.Timedelta(hours=72)<end)]
                predictions={'persistence':d.initial_oxygen.to_numpy()}
                for name,w in [('coarse025',.25),('coarse050',.5),('coarse100',1.)]:predictions[name]=d.initial_oxygen.to_numpy()+w*d.coarse_delta.to_numpy()
                predictions['solubility']=d.initial_oxygen.to_numpy()+d.solubility_delta.to_numpy()
                ml_supported=len(tr)>=30 and tr.episode.nunique()>=6 and tr.sensor.nunique()>=2
                predictions['ridge']=np.full(len(d),np.nan)
                if ml_supported:
                    w=1/tr.groupby('sensor').sensor.transform('size').to_numpy();w=w/w.mean()
                    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=100))
                    model.fit(tr[FEATURES],tr.target_oxygen-tr.initial_oxygen,ridge__sample_weight=w)
                    predictions['ridge']=d.initial_oxygen.to_numpy()+model.predict(d[FEATURES])
                predictions={name:np.clip(p,0,600) for name,p in predictions.items()}
                supported=ca.dropna(subset=['initial_oxygen','target_oxygen']);scores={}
                if supported.episode.nunique()>=10 and supported.group.nunique()>=2:
                    for family,p in predictions.items():
                        if np.isfinite(p[supported.index]).all():scores[family]=float(score(supported,p[supported.index]))
                chosen=min(scores,key=scores.get) if scores else 'persistence'
                contracts.append(dict(fold=fold,horizon=h,selected=chosen,scores=scores,ml_supported=ml_supported,training_rows=len(tr),training_episodes=sorted(tr.episode.unique()),calibration_episodes=sorted(ca.episode.unique()),evaluation_episodes=sorted(ev.episode.unique())))
                for family in FAMILIES+['selected']:
                    r=ev.copy();r['fold']=fold;r['family']=family;r['prediction']=predictions[chosen if family=='selected' else family][ev.index];rows.append(r)
                print(fold,h,chosen,'ridge_supported',ml_supported,flush=True)
        d.to_csv(root/'episode_inputs.csv.gz',index=False);pd.concat(rows).to_csv(root/'predictions.csv.gz',index=False)
        save(root/'contract.json',{'folds':contracts,'features':FEATURES,'families':FAMILIES,'policy':'Weekly non-overlapping72h episodes;24/72h oxygen state estimates. Initial oxygen/T/S strictly pre-outage within6h. External forcing48h proxy continues during outage. Ridge100 predicts oxygen increment;fit only before100day policy-calibration window and72h embargo;requires30 labels,6 weeks,2 sensors. Policy chosen on earlier100day calibration only,>=10 supported weeks and2 groups;else persistence. Solubility-change candidate preserves initial absolute oxygen disequilibrium with external T/S;not a biological oxygen budget. Clip0-600umol/L. Entire scheduled grid retained;no2026.'})
        save(status,{'state':'COMPLETED','result_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}})
    except BaseException as exc:save(status,{'state':'FAILED','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':run()
