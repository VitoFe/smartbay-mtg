"""Causal remote observations for CTD queries; never use CTD target values as inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.bay_observations import read_export

ROOT=Path(__file__).resolve().parents[2]
TARGETS=['temp_c','salinity_psu','do_umol_l']


def allowed_sources(source_xy,query_xy,denied_xy=None,radius_m=250):
    allow=np.linalg.norm(source_xy-np.asarray(query_xy),axis=1)>radius_m
    if denied_xy is not None:
        allow &= np.linalg.norm(source_xy-np.asarray(denied_xy),axis=1)>radius_m
    return allow


def latest_context(frame,stamp,max_age_hours=6):
    """Hourly interval ends already encode availability; missing ages stay missing."""
    window=frame.loc[(frame.index<=stamp)&(frame.index>=stamp-pd.Timedelta(hours=max_age_hours))]
    result=[]
    for col in frame:
        s=window[col].dropna()
        result.append((float(s.iloc[-1]),(stamp-s.index[-1]).total_seconds()/3600) if len(s) else (np.nan,np.nan))
    return np.array(result)


def load_profiles():
    d=pd.read_csv(ROOT/'data/processed/ctd_all_casts.csv.gz')
    d['t_utc']=pd.to_datetime(d.t_utc,utc=True)
    d=d[(d.t_utc>=pd.Timestamp('2024-07-24',tz='UTC'))&(d.t_utc<pd.Timestamp('2026-01-01',tz='UTC'))].copy()
    audit=pd.read_csv(ROOT/'reports/bay_twin/gen2_profile_inventory.csv')
    excluded=audit.loc[audit.depth_exceeds_metadata_by_gt_1m,'cast_id'].tolist()
    d=d[~d.cast_id.isin(excluded)]
    d=d[np.isfinite(d[TARGETS]).all(axis=1)&(d.qc_flag==0)]
    d=d[d.temp_c.between(-2,40)&d.salinity_psu.between(0,42)&d.do_umol_l.between(0,600)]
    return d.sort_values(['t_utc','cast_id','depth_m']).reset_index(drop=True),excluded


def load_observations():
    meta=pd.read_csv(ROOT/'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    frames={}
    for name in meta.index:
        _,frame,_=read_export(ROOT/'data/raw'/(name+'.txt'),include_auxiliary=True)
        frames[name]=frame[frame.index<pd.Timestamp('2026-01-01',tz='UTC')]
    return meta,frames


def context_features(d,meta,frames,denied_station=None,observation_delay_hours=0,outage_hours=0,reconnected_hours=0,outage_scope='network'):
    if not np.isfinite(observation_delay_hours) or observation_delay_hours<0:
        raise ValueError('Observation delay must be finite and nonnegative')
    if not np.isfinite([outage_hours,reconnected_hours]).all() or min(outage_hours,reconnected_hours)<0 or outage_scope not in ['network','nearest_site']:
        raise ValueError('Invalid outage scenario')
    denied=None
    if denied_station is not None:
        row=d[d.station==denied_station].iloc[0]; denied=np.array([row.UTM_X,row.UTM_Y])
    source_xy=meta[['x_m','y_m']].to_numpy()
    contexts={}
    for cast,g in d.groupby('cast_id',sort=False):
        q=g.iloc[0]; qxy=np.array([q.UTM_X,q.UTM_Y])
        allow=allowed_sources(source_xy,qxy,denied)
        distance=np.linalg.norm(source_xy-qxy,axis=1)
        failed=np.ones(len(source_xy),dtype=bool)
        if outage_scope=='nearest_site' and allow.any():
            nearest=np.flatnonzero(allow)[np.argmin(distance[allow])]
            failed=np.linalg.norm(source_xy-source_xy[nearest],axis=1)<=60
        values={v:[] for v in ['temperature','salinity','oxygen']}
        for j,(name,frame) in enumerate(frames.items()):
            if not allow[j]: continue
            # Delay experiments hold the last available state: six-hour search
            # behind the cutoff, while feature ages remain relative to query time.
            cutoff=q.t_utc-pd.Timedelta(hours=observation_delay_hours)
            observed=frame
            if outage_hours and failed[j]:
                end=q.t_utc-pd.Timedelta(hours=reconnected_hours);start=end-pd.Timedelta(hours=outage_hours)
                observed=frame.loc[(frame.index>=cutoff-pd.Timedelta(hours=6))&(frame.index<=cutoff)]
                blocked=(observed.index>=start)&((observed.index<end) if reconnected_hours else (observed.index<=end))
                observed=observed.loc[~blocked]
            latest=latest_context(observed,cutoff)
            latest[:,1]+=observation_delay_hours
            for col,(value,age) in zip(frame.columns,latest):
                if np.isfinite(value): values[col].append((value,age,distance[j]))
        result={}
        for v,records in values.items():
            z=np.array(records)
            if len(z):
                w=np.exp(-z[:,1]/6)/(z[:,2]+100)**2; w/=w.sum()
                result[v+'_context']=float(w@z[:,0])
                result[v+'_spread']=float(np.sqrt(w@((z[:,0]-result[v+'_context'])**2)))
                result[v+'_age']=float(w@z[:,1]); result[v+'_distance']=float(z[:,2].min()/1000)
            else:
                for suffix in ['context','spread','age','distance']: result[v+'_'+suffix]=np.nan
            result[v+'_count']=len(z)
        contexts[cast]=result
    x=pd.DataFrame([contexts[c] for c in d.cast_id])
    day=d.t_utc.dt.dayofyear.to_numpy()*2*np.pi/365.25
    hour=(d.t_utc.dt.hour.to_numpy()+d.t_utc.dt.minute.to_numpy()/60)*2*np.pi/24
    x['x_km']=(d.UTM_X.to_numpy()-568000)/1000; x['y_km']=(d.UTM_Y.to_numpy()-4880000)/1000
    x['depth_m']=d.depth_m.to_numpy(); x['metadata_bottom_m']=d.seabed_m.to_numpy()
    x['year_sin']=np.sin(day); x['year_cos']=np.cos(day); x['hour_sin']=np.sin(hour); x['hour_cos']=np.cos(hour)
    return x
