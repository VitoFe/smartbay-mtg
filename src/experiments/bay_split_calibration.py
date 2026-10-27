"""Dedicated pre-evaluation cast calibration and fixed dense-temperature ablations."""
import argparse,json,hashlib,os,traceback
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.data.bay_profile_context import ROOT,TARGETS,load_profiles,load_observations,context_features
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_twin_sweep import save

SOURCE='20260917T011757Z_field_export_pilot';SCALE=np.array([2.,1.,20.])
FAMILIES={'ctd_only':0.,'dense025':.25,'dense100':1.}


def auxiliary_boundary(d,meta):
    """Vectorized exact counterpart of nearest-cell, lag48h, depth interpolation."""
    absolute=[];deltas=[];locations=meta.loc[d.sensor];q=pd.DatetimeIndex(d.t_utc).tz_localize(None)-pd.Timedelta(hours=48)
    depth=d.depth_m.to_numpy();folder=ROOT/'data/external/Copernicus/T2023_2026'
    for token,var,unit in [('phy-tem','thetao','degrees_C'),('phy-sal','so','0.001'),('bgc-bio','o2','mmol m-3')]:
        with xr.open_dataset(next(folder.glob('*'+token+'*.nc'))) as ds:
            assert ds[var].attrs['units']==unit
            ti=ds.indexes['time'].get_indexer(q,method='pad');assert (ti>=0).all()
            xi=abs(ds.longitude.to_numpy()[None,:]-locations.longitude.to_numpy()[:,None]).argmin(axis=1)
            yi=abs(ds.latitude.to_numpy()[None,:]-locations.latitude.to_numpy()[:,None]).argmin(axis=1)
            arr=ds[var].isel(time=xr.DataArray(ti,dims='query'),longitude=xr.DataArray(xi,dims='query'),latitude=xr.DataArray(yi,dims='query')).transpose('query','depth').to_numpy()
            z=ds.depth.to_numpy();rows=np.arange(len(d))
            def interp(target):
                x=np.clip(np.broadcast_to(target,depth.shape),z.min(),z.max());hi=np.clip(np.searchsorted(z,x),1,len(z)-1);lo=hi-1
                w=(x-z[lo])/(z[hi]-z[lo]);return arr[rows,lo]*(1-w)+arr[rows,hi]*w
            a=interp(depth);absolute.append(a);deltas.append(a-interp(3.))
            if var=='thetao':
                dist=Geod(ellps='WGS84').inv(locations.longitude.to_numpy(),locations.latitude.to_numpy(),ds.longitude.to_numpy()[xi],ds.latitude.to_numpy()[yi])[2]
                clamped=(depth<z.min())|(depth>z.max())
    out=np.column_stack(absolute+deltas+[dist,clamped]);assert np.isfinite(out).all();return out


def run(task):
    out=ROOT/'results';out.mkdir(exist_ok=True);status=out/f'status_split_calibration_{task}.json';save(status,{'state':'RUNNING','pid':os.getpid()})
    try:
        data,_=load_profiles();meta,frames=load_observations()
        extra=data[['cast_id','depth_m']].merge(pd.read_csv(ROOT/'reports/bay_twin/profile_boundary_context.csv.gz'),on=['cast_id','depth_m'],validate='one_to_one')[EXTRAS].to_numpy(dtype=float)
        source=ROOT/'reports/bay_twin/runs'/SOURCE/'results'/f'field_export_{task}'
        root=out/f'split_calibration_{task}';root.mkdir(exist_ok=True);outputs=[];contracts=[];cal_outputs=[];cache={}
        manifest=json.loads((ROOT/'reports/bay_twin/thermal_supervision/manifest.json').read_text())
        for path in sorted(source.glob('*_model.json')):
            original=json.loads(path.read_text());c=original['contract'];held=c['held_station'];start=pd.Timestamp(original['selection_window']['train_end'],tz='UTC')
            pre=data[(data.t_utc<start-pd.Timedelta(hours=72))&((data.station!=held) if held else True)]
            casts=pre.groupby('cast_id').t_utc.first().sort_values()
            calids=casts.iloc[-20:].index;cal_start=casts.loc[calids].min()
            train=pre[pre.t_utc<cal_start-pd.Timedelta(hours=72)];tr=train.index.to_numpy();ca=pre[pre.cast_id.isin(calids)].index.to_numpy()
            va=data[data.cast_id.isin(c['selection_cast_ids'])].index.to_numpy()
            contract={'id':c['id'],'fold':c['fold'],'held_station':held,'training_cast_ids':sorted(train.cast_id.unique()),'calibration_cast_ids':sorted(calids),
                'selection_cast_ids':c['selection_cast_ids'],'training_last':str(train.t_utc.max()),'calibration_first':str(cal_start),'calibration_last':str(data.iloc[ca].t_utc.max()),'selection_first':str(data.iloc[va].t_utc.min())}
            if train.cast_id.nunique()<15 or len(calids)<20:
                contract['status']='INSUFFICIENT_TRAINING_CASTS_AFTER_CALIBRATION_RESERVATION';contracts.append(contract);continue
            if held not in cache:
                raw=context_features(data.drop(columns=TARGETS),meta,frames,held)
                ap=ROOT/'reports/bay_twin/thermal_supervision'/((held or 'all')+'.csv.gz')
                entry=next(r for r in manifest['datasets'] if r['denied_station']==held)
                assert hashlib.sha256(ap.read_bytes()).hexdigest()==entry['sha256']
                aux=pd.read_csv(ap);aux.t_utc=pd.to_datetime(aux.t_utc,utc=True)
                cache[held]=(raw,aux,auxiliary_boundary(aux,meta))
            raw,aux,ae=cache[held]
            ai=np.flatnonzero(aux.t_utc<cal_start-pd.Timedelta(hours=72));assert len(ai)>0
            assert set(train.cast_id).isdisjoint(calids) and set(train.cast_id).isdisjoint(c['selection_cast_ids'])
            imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(raw.iloc[tr])
            scaler=StandardScaler().fit(imputer.transform(raw.iloc[tr]))
            full=np.column_stack([scaler.transform(imputer.transform(raw)).astype('float32'),extra])
            auxfull=np.column_stack([scaler.transform(imputer.transform(aux[list(raw)])).astype('float32'),ae])
            second=StandardScaler().fit(full[tr]);x=second.transform(full).astype('float32');ax=second.transform(auxfull).astype('float32')
            fallback=np.median(data.iloc[tr][TARGETS].to_numpy(),axis=0)
            def base(raw,e):
                b=raw[['temperature_context','salinity_context','oxygen_context']].to_numpy();b=np.where(np.isfinite(b),b,fallback).astype('float32')
                b=(b+e[:,3:6]).astype('float32');b[:,0]=np.clip(b[:,0],-2,40);b[:,1]=np.clip(b[:,1],0,42);return b.astype(float)
            b=base(raw,extra);ab=base(aux,ae);truth=data[TARGETS].to_numpy(dtype='float32').astype(float)
            w=1/(train.groupby('cast_id').cast_id.transform('size').to_numpy()*train.groupby('station').cast_id.transform('nunique').to_numpy());w=w/w.mean()
            au=aux.iloc[ai];aw=1/au.groupby('sensor').sensor.transform('size').to_numpy();aw=aw/aw.sum()*w.sum()
            contract.update(status='FIT',auxiliary_rows=len(ai),auxiliary_last=str(au.t_utc.max()),alpha=original['alpha']);contracts.append(contract)
            for family,mass in FAMILIES.items():
                predictions=[];scales=[]
                for j in range(3):
                    tx=x[tr];ty=(truth[tr,j]-b[tr,j])/SCALE[j];tw=w
                    if j==0 and mass:
                        tx=np.concatenate([tx,ax[ai]]);ty=np.concatenate([ty,(aux.target_temperature.to_numpy()[ai]-ab[ai,j])/SCALE[j]]);tw=np.concatenate([w,aw*mass])
                    model=Ridge(alpha=original['alpha'],fit_intercept=False).fit(tx,ty,sample_weight=tw)
                    prediction=b[:,j]+model.predict(x)*SCALE[j]
                    if j<2:prediction=np.clip(prediction,[-2,0][j],[40,42][j])
                    # Fixed training-only scale estimator. In-sample training residuals
                    # are allowed here; independent calibration outcomes set the radius.
                    err=abs(truth[tr,j]-prediction[tr]);floor=max(float(np.median(err)),[.05,.05,1.][j])
                    scale_model=Ridge(alpha=10).fit(x[tr],np.log(np.maximum(err,floor*.1)),sample_weight=w)
                    scale=np.exp(np.clip(scale_model.predict(x),np.log(floor*.2),np.log(floor*5)))
                    predictions.append(prediction);scales.append(scale)
                for ids,kind in [(ca,'calibration'),(va,'evaluation')]:
                    rows=data.iloc[ids][['cast_id','station','t_utc','depth_m']+TARGETS].copy();rows['fold']=c['fold'];rows['checkpoint']=c['id'];rows['family']=family
                    for j,t in enumerate(TARGETS):rows['pred_'+t]=predictions[j][ids];rows['scale_'+t]=scales[j][ids]
                    (cal_outputs if kind=='calibration' else outputs).append(rows)
            print(task,c['id'],'fit',len(tr),'CTD bins;',len(ai),'thermal rows',flush=True)
        pd.concat(outputs,ignore_index=True).to_csv(root/'predictions.csv.gz',index=False);pd.concat(cal_outputs,ignore_index=True).to_csv(root/'calibration.csv.gz',index=False)
        save(root/'contract.json',{'task':task,'folds':contracts,'families':FAMILIES,'policy':'Reserve latest20 pre-evaluation casts, embargo72h before/after; require15 earlier training casts. Frozen original ridge alpha, CTD-fitted preprocessing. Dense temperature auxiliary loss mass0/.25/1 relative to CTD; other targets unchanged.',
            'scale':'Training-only log absolute residual ridge with alpha10; scales clipped0.2–5 times training median-error floor. Calibration labels never fit point/scale models.',
            'limits':'Development recipe; no2026. Unsupported fits remain in contract for cohort/availability accounting. Original hyperparameter alpha selected on2025 previously. Dense telemetry supports temperature only.'})
        save(status,{'state':'COMPLETED','task':task,'result_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}})
    except BaseException as exc:save(status,{'state':'FAILED','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['temporal','held_site'],required=True);run(p.parse_args().task)
