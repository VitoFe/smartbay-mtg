"""Explicit measured-probe outage fallback; never an arbitrary-coordinate field."""
import numpy as np
import pandas as pd

THERMAL_RUN='20260917T184229Z_thermal_memory_pilot'


def temperature_outage_state(frame,sensor,metadata,outage_start,time):
    start=pd.Timestamp(outage_start);q=pd.Timestamp(time)
    if pd.isna(start) or pd.isna(q) or start.tzinfo is None or q.tzinfo is None:raise ValueError('Timezone-aware timestamps required')
    start=start.tz_convert('UTC');q=q.tz_convert('UTC');hours=float((q-start)/pd.Timedelta(hours=1))
    result={'status':'UNSUPPORTED','sensor':sensor,'variable':'temperature','unit':'degC','query_time':str(q),'outage_start':str(start),'outage_hours':hours,'prediction':None,'source_time':None,'source_age_hours':None,'interval':None,'warning':None,
        'location':{k:float(metadata[k]) for k in ['latitude','longitude','nominal_depth_m']},'provenance':{'run_id':THERMAL_RUN,'policy':'last_real_preoutage_temperature','feed_mode':'supplied_frames_unverified'},'scope':'Experimental measured-probe recovery at tested 24/72 h endpoints; no spatial interpolation, oxygen estimate or calibrated interval.'}
    if not pd.Timestamp('2025-01-01',tz='UTC')<=start<=q<pd.Timestamp('2026-01-01',tz='UTC'):
        return result|{'reason':'Outside 2025 development window; 2026 reserved'}
    if hours not in (24.,72.):return result|{'reason':'Only 24/72 h outage endpoints were evaluated'}
    if not isinstance(frame.index,pd.DatetimeIndex) or frame.index.tz is None or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:raise ValueError('Sorted unique timezone-aware hourly feed required')
    if 'temperature' not in frame:return result|{'reason':'No temperature channel'}
    s=frame.temperature.loc[(frame.index<start)&(frame.index>=start-pd.Timedelta(hours=6))].dropna()
    s=s[np.isfinite(s)&s.between(-2,40)]
    if not len(s):return result|{'reason':'No valid real temperature in 6 h before outage'}
    stamp=s.index[-1]
    return result|{'status':'DEGRADED_PERSISTENCE','reason':'Sensor outage; retaining last measured state','prediction':float(s.iloc[-1]),'source_time':str(stamp),'source_age_hours':float((q-stamp)/pd.Timedelta(hours=1)),'initial_age_hours':float((start-stamp)/pd.Timedelta(hours=1))}
