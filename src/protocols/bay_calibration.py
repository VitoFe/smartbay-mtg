"""Predeclared causal residual calibration policies, not coverage guarantees."""
import numpy as np
import pandas as pd


def finite_radius(errors,coverage=.9):
    e=np.sort(np.abs(np.asarray(errors,float)))
    if len(e)<30 or not np.isfinite(e).all():raise ValueError('Need 30 finite residuals')
    return float(e[min(len(e)-1,int(np.ceil(coverage*(len(e)+1)))-1)])


def calibrate_stream(calibration,predictions,days=None,adaptive=True):
    """A residual becomes usable one hour after its six-hour endpoint.

    Sparse rolling windows fall back to the original fixed calibration sample;
    never to future labels. Predictions are not retrained or altered.
    """
    cal=calibration.copy();d=predictions.copy()
    cal['time']=pd.to_datetime(cal.time,utc=True);d['time']=pd.to_datetime(d.time,utc=True)
    cal['available']=cal.time+pd.Timedelta(hours=7)
    d['available']=d.time+pd.Timedelta(hours=7)
    cal['error']=cal.truth-cal.prediction;d['error']=d.truth-d.prediction
    history=pd.concat([cal[['time','available','error']],d[['time','available','error']]],ignore_index=True)
    rows=[]
    for r in d.sort_values('time').itertuples():
        pool=history[history.available<=r.time] if adaptive else cal[cal.available<=r.time]
        if days is not None:pool=pool[pool.available>=r.time-pd.Timedelta(days=days)]
        fallback=len(pool)<30
        if fallback:pool=cal[cal.available<=r.time]
        radius=finite_radius(pool.error)
        rows.append({'time':str(r.time),'truth':r.truth,'prediction':r.prediction,'lower':r.prediction-radius,'upper':r.prediction+radius,
                     'residual_count':len(pool),'latest_residual_available':str(pool.available.max()),'fixed_fallback':fallback})
    return pd.DataFrame(rows)
