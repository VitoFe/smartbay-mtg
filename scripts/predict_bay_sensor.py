"""Saved-model inference from causal export observations, not prediction lookup.

No live feed is connected. The pure feature adapter also accepts hourly frames
from a future ingestion service. Hyperparameters remain development-selected.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.bay_profile_context import load_observations
from src.data.bay_forcing import weather
from src.models.bay_sensor_inference import causal_features
from src.models.bay_forecast_output import forecast_payload
from src.protocols.bay_calibration import finite_radius
from src.models.bay_temperature_memory import temperature_outage_state,THERMAL_RUN


def outage_state(variable,sensor,time,outage_start):
    if variable!='temperature':return {'status':'UNSUPPORTED','reason':'Oxygen outage recovery is not validated for this interface'}
    root=ROOT/'reports/bay_twin/runs'/THERMAL_RUN
    if json.loads((root/'verification.json').read_text())['state']!='VERIFIED':raise ValueError('Unverified thermal memory study')
    manifest=json.loads((root/'snapshot_manifest.json').read_text())
    for name in ['reports/bay_twin/sensor_geometry.csv','data/raw/'+sensor+'.txt']:
        if name not in manifest['files_sha256']:return {'status':'UNSUPPORTED','reason':'Sensor absent from verified snapshot'}
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=manifest['files_sha256'][name]:raise ValueError('Sensor input hash mismatch')
    meta,frames=load_observations()
    if sensor not in frames:return {'status':'UNSUPPORTED','reason':'Unknown sensor'}
    result=temperature_outage_state(frames[sensor],sensor,meta.loc[sensor],outage_start,time)
    result['provenance']['feed_mode']='verified_archive';result['provenance']['live_feed_connected']=False
    return result


def predict(run_id,variable,sensor,time,blend_run=None):
    if not re.fullmatch(r'\d{8}T\d{6}Z_seasonal_pilot',run_id):raise ValueError('Expected verified seasonal run')
    if variable not in ['temperature','oxygen']:raise ValueError('Unknown variable')
    q=pd.Timestamp(time)
    if q.tzinfo is None:raise ValueError('Timezone required')
    q=q.tz_convert('UTC')
    if q>=pd.Timestamp('2026-01-01',tz='UTC'):return {'status':'UNSUPPORTED','reason':'2026 archive reserved'}
    root=ROOT/'reports/bay_twin/runs'/run_id
    if json.loads((root/'verification.json').read_text())['state']!='VERIFIED':raise ValueError('Unverified model run')
    out=root/'results'/f'seasonal_{variable}';contract=json.loads((out/'contract.json').read_text())
    folds=[f for f in contract['folds'] if pd.Timestamp(f['selection_start'])<=q<pd.Timestamp(f['selection_end'])]
    if not folds:return {'status':'UNSUPPORTED','reason':'Outside validated checkpoint time window'}
    f=folds[0]
    if sensor not in f['supported_sensors']:return {'status':'UNSUPPORTED','reason':'Insufficient training support at requested sensor'}
    status=json.loads((root/f'results/status_seasonal_{variable}.json').read_text())
    path=out/f'weather_tree_fold{f["fold"]}_model.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=status['result_sha256'][path.relative_to(root/'results').as_posix()]:raise ValueError('Checkpoint hash mismatch')
    checkpoint=json.loads(path.read_text());checkpoint['variable']=variable
    meta,frames=load_observations()
    # Explicitly truncate the feed; no six-hour outcome is loaded into the feature builder.
    frames={n:a.loc[a.index<=q] for n,a in frames.items()}
    x,anchors,freshness=causal_features(frames,meta,checkpoint,q,weather(pd.DatetimeIndex([q])).iloc[0].to_numpy())
    s=checkpoint['names'].index(sensor)
    if not np.isfinite(anchors[s]):return {'status':'UNSUPPORTED','reason':'No actual target observation at issuance; stale anchors are not substituted'}
    model=lgb.Booster(model_str=checkpoint['model_string'])
    value=float(anchors[s]+model.predict(x[s:s+1])[0]*checkpoint['scale'])
    # Interval feedback is a separate input: only past outcomes available by q enter.
    for name in ['calibration.csv.gz','predictions.csv.gz']:
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=status['result_sha256'][f'seasonal_{variable}/{name}']:raise ValueError('Residual ledger hash mismatch')
    cal=pd.read_csv(out/'calibration.csv.gz');past=pd.read_csv(out/'predictions.csv.gz')
    cal=cal[(cal.family=='weather_tree')&(cal.fold==f['fold'])&(cal.sensor==sensor)].copy()
    past=past[(past.family=='weather_tree')&(past.policy=='fixed')&(past.fold==f['fold'])&(past.sensor==sensor)].copy()
    for a in [cal,past]:a['time']=pd.to_datetime(a.time,utc=True)
    history=pd.concat([cal,past],ignore_index=True)
    availability=history.time+pd.Timedelta(hours=7)
    recent=history[(availability<=q)&(availability>=q-pd.Timedelta(days=30))]
    pool=recent if len(recent)>=30 else cal[cal.time+pd.Timedelta(hours=7)<=q]
    supported=len(pool)>=30;radius=finite_radius(pool.truth-pool.prediction) if supported else np.nan
    row={'sensor':sensor,'time':str(q),'prediction':value,'persistence':float(anchors[s]),'lower':value-radius,'upper':value+radius,
         'policy':'rolling30','interval_status':('rolling30' if len(recent)>=30 else 'fixed_fallback') if supported else 'insufficient_calibration',
         'residual_count':len(pool),'latest_residual_available':str(pool.time.max()+pd.Timedelta(hours=7)) if len(pool) else None}
    result=forecast_payload(row,meta.loc[sensor].to_dict(),{'run_id':run_id,'variable':variable,'family':'weather_tree','fold':f['fold'],
        'last_training_outcome':f['last_training_outcome'],'selection_limit':'2025 development-selected hyperparameters; not prospective validation.'})
    result['status']='EXPERIMENTAL_CAUSAL_INFERENCE';result['context']=freshness
    result['support']='Saved-model known-sensor forecast from causal export observations. No live feed or continuous spatial field is connected.'
    result['uncertainty_source']='Causal past residuals from archived forecast ledger; not a live residual service.'
    if blend_run is not None:
        if not re.fullmatch(r'\d{8}T\d{6}Z_robust_pilot',blend_run):raise ValueError('Expected verified robust run')
        blend_root=ROOT/'reports/bay_twin/runs'/blend_run
        if json.loads((blend_root/'verification.json').read_text())['state']!='VERIFIED':raise ValueError('Unverified blend run')
        bc=json.loads((blend_root/'results/robust_forecast/contract.json').read_text())
        if bc['source_run']!=run_id:raise ValueError('Blend belongs to a different point-model run')
        decision_path=blend_root/'results/robust_forecast/blend_decisions.json'
        bs=json.loads((blend_root/'results/status_robust_forecast.json').read_text())
        if hashlib.sha256(decision_path.read_bytes()).hexdigest()!=bs['result_sha256']['robust_forecast/blend_decisions.json']:raise ValueError('Blend decision hash mismatch')
        decisions=json.loads(decision_path.read_text())
        decision=next(r for r in decisions if r['variable']==variable and r['fold']==f['fold'] and r['sensor']==sensor)
        assert pd.Timestamp(decision['latest_calibration_outcome'])<q
        alpha=decision['global_weight'];value=float(anchors[s]+alpha*(value-anchors[s]))
        result['prediction']=value;result['predicted_change']=value-float(anchors[s]);result['predicted_change_per_hour']=(value-float(anchors[s]))/6
        result['provenance']['blend']={'run_id':blend_run,'policy':'global_calibration_blend','ml_weight':alpha,'latest_calibration_outcome':decision['latest_calibration_outcome']}
        verification=json.loads((blend_root/'blend_interval_verification.json').read_text())
        ledger_path=blend_root/'results/robust_forecast/blend_predictions.csv.gz'
        if verification['state']!='VERIFIED' or hashlib.sha256(ledger_path.read_bytes()).hexdigest()!=verification['source_sha256']:
            raise ValueError('Unverified blend residual ledger')
        ledger=pd.read_csv(ledger_path)
        ledger=ledger[(ledger.variable==variable)&(ledger.family=='global_blend')&(ledger.fold==f['fold'])&(ledger.sensor==sensor)].copy()
        ledger['time']=pd.to_datetime(ledger.time,utc=True)
        available=ledger.time+pd.Timedelta(hours=7)
        pool=ledger[(available<=q)&(available>=q-pd.Timedelta(days=30))]
        supported=len(pool)>=30
        radius=finite_radius(pool.truth-pool.prediction) if supported else np.nan
        row.update(prediction=value,lower=value-radius,upper=value+radius,policy='blend_oos_rolling30',
            interval_status='rolling30' if supported else 'insufficient_blend_calibration',residual_count=len(pool),
            latest_residual_available=str(pool.time.max()+pd.Timedelta(hours=7)) if len(pool) else None)
        calibrated=forecast_payload(row,meta.loc[sensor].to_dict(),result['provenance'])
        result['interval']=calibrated['interval'];result['interval_status']=calibrated['interval_status']
        result['uncertainty_source']='Only later out-of-sample blend residuals, available endpoint+1h; reset each quarter, minimum30 in preceding30days. Weight-fitting residuals excluded.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--variable',required=True,choices=['temperature','oxygen']);p.add_argument('--sensor',required=True);p.add_argument('--time',required=True);p.add_argument('--output',type=Path);p.add_argument('--blend-run');p.add_argument('--outage-start')
    a=p.parse_args()
    if a.outage_start:
        if a.run_id or a.blend_run:p.error('--outage-start is a separate state-recovery mode; omit model run arguments')
        result=outage_state(a.variable,a.sensor,a.time,a.outage_start)
    else:
        if not a.run_id:p.error('--run-id is required for forecast mode')
        result=predict(a.run_id,a.variable,a.sensor,a.time,a.blend_run)
    text=json.dumps(result,indent=2,allow_nan=False)
    if a.output:a.output.write_text(text,encoding='utf-8')
    print(text)
