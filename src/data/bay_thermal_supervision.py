"""Dense, target-site-masked thermal supervision in the CTD query feature schema."""
import numpy as np
import pandas as pd
from src.data.bay_profile_context import allowed_sources


def thermal_rows(meta,frames,denied_xy=None,excluded_source_sensors=()):
    times=pd.date_range('2024-07-24','2025-12-31 12:00',freq='12h',tz='UTC')
    variables=['temperature','salinity','oxygen']; xy=meta[['x_m','y_m']].to_numpy()
    values=np.full((len(times),len(meta),3),np.nan); ages=values.copy(); labels=np.full(values.shape[:2],np.nan)
    for j,(name,frame) in enumerate(frames.items()):
        labels[:,j]=frame.temperature.reindex(times).to_numpy()
        for k,var in enumerate(variables):
            if var not in frame: continue
            # Reindex a full hourly grid before filling; never fill target labels.
            grid=pd.date_range(min(times.min(),frame.index.min()),max(times.max(),frame.index.max()),freq='h',tz='UTC')
            raw=frame[var].reindex(grid)
            clock=pd.Series(grid,index=grid).where(raw.notna()).ffill(limit=6)
            val=raw.ffill(limit=6).reindex(times)
            age=(pd.Series(times,index=times)-clock.reindex(times)).dt.total_seconds()/3600
            values[:,j,k]=val.to_numpy(); ages[:,j,k]=age.to_numpy()
    rows=[]
    for target,(name,station) in enumerate(meta.iterrows()):
        # A held CTD station has neither nearby auxiliary labels nor context.
        if denied_xy is not None and np.linalg.norm(xy[target]-denied_xy)<=250: continue
        allow=allowed_sources(xy,xy[target],denied_xy)
        allow &= ~meta.index.isin(excluded_source_sensors)
        distances=np.linalg.norm(xy-xy[target],axis=1)
        for q in np.flatnonzero(np.isfinite(labels[:,target])):
            row={'sensor':name,'t_utc':str(times[q]),'target_temperature':float(labels[q,target])}
            for k,var in enumerate(variables):
                valid=allow & np.isfinite(values[q,:,k]) & (ages[q,:,k]<=6)
                if valid.any():
                    v=values[q,valid,k]; a=ages[q,valid,k]; distance=distances[valid]
                    w=np.exp(-a/6)/(distance+100)**2; w/=w.sum()
                    mean=float(w@v)
                    row.update({var+'_context':mean,var+'_spread':float(np.sqrt(w@((v-mean)**2))),
                                var+'_age':float(w@a),var+'_distance':float(distance.min()/1000)})
                else:
                    row.update({var+'_'+suffix:np.nan for suffix in ['context','spread','age','distance']})
                row[var+'_count']=int(valid.sum())
            t=times[q]; day=t.dayofyear*2*np.pi/365.25; hour=t.hour*2*np.pi/24
            row.update(x_km=(station.x_m-568000)/1000,y_km=(station.y_m-4880000)/1000,
                depth_m=station.nominal_depth_m,metadata_bottom_m=station.metadata_bottom_m,
                year_sin=np.sin(day),year_cos=np.cos(day),hour_sin=np.sin(hour),hour_cos=np.cos(hour))
            rows.append(row)
    return pd.DataFrame(rows)
