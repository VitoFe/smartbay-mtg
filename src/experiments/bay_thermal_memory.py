"""Causal thermal state retention across non-overlapping sensor-network outages."""
import hashlib,json,os,traceback
import numpy as np
import pandas as pd
from src.data.bay_profile_context import ROOT,load_observations
from src.experiments.bay_split_calibration import auxiliary_boundary
from src.experiments.bay_transfer_calibration import groups
from src.experiments.bay_twin_sweep import save

WEIGHTS=[0.,.25,.5,1.]
QUARTERS=['2025-01-01','2025-04-01','2025-07-01','2025-10-01','2026-01-01']


def initial_state(frame,start):
    s=frame.temperature.loc[(frame.index<start)&(frame.index>=start-pd.Timedelta(hours=6))].dropna()
    return (float(s.iloc[-1]),s.index[-1]) if len(s) else (np.nan,pd.NaT)


def episode_grid(meta,frames):
    assignments={sensor:i for i,g in enumerate(groups(meta)) for sensor in g};rows=[]
    for start in pd.date_range('2024-08-05','2025-12-22',freq='7D',tz='UTC'):
        for sensor in meta.index:
            value,stamp=initial_state(frames[sensor],start)
            for horizon in [24,72]:
                q=start+pd.Timedelta(hours=horizon)
                truth=frames[sensor].temperature.get(q,np.nan)
                rows.append(dict(sensor=sensor,group=assignments[sensor],episode=str(start),start=start,t_utc=q,horizon=horizon,depth_m=meta.loc[sensor,'nominal_depth_m'],initial_temperature=value,source_time=stamp,target_temperature=truth))
    return pd.DataFrame(rows)


def macro(d,prediction):
    # Each scheduled week has one endpoint per sensor and horizon. Average
    # within sensor, then within held spatial groups, then across groups.
    return d.assign(error=abs(d.target_temperature-prediction)).groupby(['group','sensor']).error.mean().groupby('group').mean().mean()


def select_weight(cal):
    cal=cal[np.isfinite(cal.target_temperature)&np.isfinite(cal.initial_temperature)].copy()
    if cal.episode.nunique()<10 or cal.group.nunique()<2:
        return 0.,'INSUFFICIENT_CALIBRATION',{}
    scores={str(w):float(macro(cal,np.clip(cal.initial_temperature+w*cal.boundary_delta,-2,40))) for w in WEIGHTS}
    return min(WEIGHTS,key=lambda w:scores[str(w)]),'SELECTED',scores


def run():
    out=ROOT/'results';out.mkdir(exist_ok=True);status=out/'status_thermal_memory.json';save(status,{'state':'RUNNING','pid':os.getpid()})
    try:
        root=out/'thermal_memory';root.mkdir(exist_ok=True);meta,frames=load_observations();d=episode_grid(meta,frames)
        d['boundary_now']=auxiliary_boundary(d,meta)[:,0]
        anchor=d.copy();anchor.t_utc=d.source_time.fillna(d.start)
        d['boundary_initial']=auxiliary_boundary(anchor,meta)[:,0]
        d['boundary_delta']=d.boundary_now-d.boundary_initial
        outputs=[];contracts=[]
        for fold,(begin,end) in enumerate(zip(QUARTERS[:-1],QUARTERS[1:])):
            begin=pd.Timestamp(begin,tz='UTC');end=pd.Timestamp(end,tz='UTC')
            # Entire maximum-duration episodes precede the embargo; no split
            # calendar week contributes endpoints to calibration and evaluation.
            calibration=(d.start>=begin-pd.Timedelta(days=100))&(d.start+pd.Timedelta(hours=72)<begin-pd.Timedelta(hours=72))
            evaluation=(d.start>=begin)&(d.start+pd.Timedelta(hours=72)<end)
            for horizon in [24,72]:
                cal=d[calibration&(d.horizon==horizon)];ev=d[evaluation&(d.horizon==horizon)].copy()
                weight,reason,scores=select_weight(cal)
                contracts.append(dict(fold=fold,horizon=horizon,weight=weight,reason=reason,scores=scores,calibration_episodes=sorted(cal.episode.unique()),evaluation_episodes=sorted(ev.episode.unique()),calibration_last=str(cal.t_utc.max()),selection_first=str(ev.start.min())))
                for family,w in [('persistence',0.),('physics025',.25),('physics050',.5),('physics100',1.),('selected_physics',weight)]:
                    r=ev.copy();r['family']=family;r['weight']=w;r['fold']=fold;r['prediction']=np.clip(r.initial_temperature+w*r.boundary_delta,-2,40);outputs.append(r)
                print('fold',fold,'horizon',horizon,'weight',weight,reason,flush=True)
        pd.concat(outputs).to_csv(root/'predictions.csv.gz',index=False);d.to_csv(root/'episode_inputs.csv.gz',index=False)
        save(root/'contract.json',{'folds':contracts,'weights':WEIGHTS,'policy':'Weekly Monday00 UTC simulated sensor-network outages,24/72h endpoints,all19 sensors. Last real temperature within6h strictly before outage. Remote sensor observations unavailable; external Copernicus temperature remains available using48h-lag proxy at query time and last observation time. This is sequential outage state estimation, not a forecast using only outage-start forcing. Earlier100day calibration,whole72h episodes before72h embargo; >=10 supported episodes and2 spatial groups,else persistence. Equal-sensor then equal-spatial-group calibration MAE; global weight per horizon/quarter; ties favor persistence. Full scheduled grid and missing labels/state retained. No2026.'})
        save(status,{'state':'COMPLETED','result_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}})
    except BaseException as exc:save(status,{'state':'FAILED','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':run()
