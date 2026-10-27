"""Export the frozen common linear+boundary recipe, with archive replay checks."""
import argparse, hashlib, json, os, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.experiments.bay_profile_field import ROOT,SCALE,prepare,prediction_frame,aggregate
from src.data.bay_profile_context import load_profiles,TARGETS
from src.experiments.bay_twin_sweep import save

SOURCE='20260910T065505Z_boundary_pilot'
EXTRAS=['boundary_'+v for v in ['temperature','salinity','oxygen']]
EXTRAS+= [c+'_vertical_delta' for c in EXTRAS.copy()]+['boundary_distance_m','outside_model_depth']


def predict_checkpoint(checkpoint, raw, extras):
    """Portable NumPy inference; no labels, fitted objects or prediction lookup."""
    x=raw[checkpoint['raw_features']].to_numpy(dtype=float)
    missing=np.isnan(x)
    imputed=np.where(missing,np.asarray(checkpoint['imputation_statistics']),x)
    imputed=np.concatenate([imputed,missing[:,checkpoint['indicator_features']]],axis=1)
    sx=((imputed-checkpoint['scaler_mean'])/checkpoint['scaler_scale']).astype('float32')
    design=np.concatenate([sx,np.asarray(extras,dtype=float)],axis=1)
    design=((design-checkpoint['boundary_scaler_mean'])/checkpoint['boundary_scaler_scale']).astype('float32')
    base=raw[['temperature_context','salinity_context','oxygen_context']].to_numpy().copy()
    base=np.where(np.isfinite(base),base,checkpoint['contract']['training_fallback']).astype('float32')
    base=(base+np.asarray(extras)[:,3:6]).astype('float32')
    base[:,0]=np.clip(base[:,0],-2,40);base[:,1]=np.clip(base[:,1],0,42)
    pred=base.astype(float)+design@np.asarray(checkpoint['coef']).T*SCALE
    pred[:,0]=np.clip(pred[:,0],-2,40);pred[:,1]=np.clip(pred[:,1],0,42)
    return pred


def run(task):
    output=ROOT/'results';output.mkdir(exist_ok=True)
    status=output/f'status_field_export_{task}.json'
    save(status,{'state':'RUNNING','task':task,'pid':os.getpid()})
    try:
        cfg=json.loads((ROOT/'config/bay_profile_field.json').read_text())
        bundles,contracts,excluded=prepare(task,cfg,False)
        source=ROOT/'reports/bay_twin/runs'/SOURCE/'results'/f'kernel_{task}'
        selection=json.loads((source/'selection_summary.json').read_text())
        # Common selected recipe, not the exploratory per-family minimum.
        alpha=selection['params']['alpha']
        reference=source/f'trial_{selection["trial"]:03d}_linear_residual_predictions.csv.gz'
        data,_=load_profiles()
        context=data[['cast_id','depth_m']].merge(pd.read_csv(ROOT/'reports/bay_twin/profile_boundary_context.csv.gz'),on=['cast_id','depth_m'],validate='one_to_one')
        extras=context[EXTRAS].to_numpy(dtype=float)
        root=output/f'field_export_{task}';root.mkdir(exist_ok=True)
        rows=[]
        for b in bundles:
            base=(b['base']+extras[:,3:6]).astype('float32')
            base[:,0]=np.clip(base[:,0],-2,40);base[:,1]=np.clip(base[:,1],0,42)
            features=np.concatenate([b['sx'],extras],axis=1)
            scaler=StandardScaler().fit(features[b['tr']])
            design=scaler.transform(features).astype('float32')
            target=(b['y'].astype(float)-base.astype(float))/SCALE
            model=Ridge(alpha=alpha,fit_intercept=False).fit(design[b['tr']],target[b['tr']],sample_weight=b['weights'])
            cp={k:b[k] for k in ['raw_features','imputation_statistics','indicator_features','scaler_mean','scaler_scale','contract']}
            cp.update(coef=model.coef_.tolist(),boundary_scaler_mean=scaler.mean_.tolist(),boundary_scaler_scale=scaler.scale_.tolist(),
                boundary_features=EXTRAS,alpha=alpha,source_run=SOURCE,source_trial=selection['trial'],task=task,
                selection_window=cfg['folds'][b['fold']],units=['degC','PSU','umol/L'],
                limits='2025 development-selected. Remote context excludes 250m around query and held station. Boundary lag48h, 3m anchor, depth clamping. No field uncertainty or validated local circulation.')
            save(root/(b['identity']+'_model.json'),cp)
            pred=base[b['va']].astype(float)+model.predict(design[b['va']])*SCALE
            pred[:,0]=np.clip(pred[:,0],-2,40);pred[:,1]=np.clip(pred[:,1],0,42)
            rows.append(prediction_frame(b,pred))
        d=pd.concat(rows,ignore_index=True)
        old=pd.read_csv(reference)
        keys=['cast_id','station','depth_m','fold']
        joined=d.merge(old,on=keys,validate='one_to_one',suffixes=('','_archive'))
        assert len(joined)==len(d)==len(old)
        errors={}
        for v in TARGETS:
            delta=joined['pred_'+v]-joined['pred_'+v+'_archive']
            errors[v]=float(abs(delta).max())
            np.testing.assert_allclose(delta,0,atol=1e-5,rtol=0)
        d.to_csv(root/'replay_predictions.csv.gz',index=False)
        save(root/'verification.json',{'state':'VERIFIED','rows':len(d),'checkpoints':len(bundles),'max_archive_difference':errors,
            'metrics':aggregate(d),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'scope':'Frozen common recipe refit on original training folds; archive prediction replay. Portable raw-feature inference verified separately.'})
        save(status,{'state':'COMPLETED','task':task,'result_sha256':{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}})
    except BaseException as exc:
        save(status,{'state':'FAILED','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['temporal','held_site'],required=True);a=p.parse_args();run(a.task)
