"""Calibration-only persistence blending and remote residual-transfer diagnostics."""
import hashlib,json,traceback
import numpy as np
import pandas as pd
from src.experiments.bay_twin_sweep import ROOT,save
from src.protocols.bay_calibration import finite_radius

SOURCE='20260916T194019Z_seasonal_pilot'


def choose_weight(cal,grid):
    """Equal site MAE, ties prefer the smaller learned increment."""
    eligible=cal.groupby('sensor').size();names=eligible[eligible>=30].index
    c=cal[cal.sensor.isin(names)]
    if c.empty:return 0.,[]
    scores=[]
    for alpha in grid:
        error=abs(c.truth-(c.persistence+alpha*(c.prediction-c.persistence)))
        scores.append({'alpha':float(alpha),'mae':float(error.groupby(c.sensor).mean().mean())})
    return min(scores,key=lambda r:(r['mae'],r['alpha']))['alpha'],scores


def run():
    output=ROOT/'results/robust_forecast';output.mkdir(parents=True,exist_ok=True)
    status=ROOT/'results/status_robust_forecast.json';save(status,{'state':'RUNNING'})
    try:
        rows=[];decisions=[];transfers=[]
        meta=pd.read_csv(ROOT/'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
        for variable in ['temperature','oxygen']:
            root=ROOT/'reports/bay_twin/runs'/SOURCE/'results'/f'seasonal_{variable}'
            d=pd.read_csv(root/'predictions.csv.gz');d=d[(d.family=='weather_tree')&(d.policy=='rolling30')].copy()
            c=pd.read_csv(root/'calibration.csv.gz');c=c[c.family=='weather_tree'].copy()
            for a in [d,c]:a['time']=pd.to_datetime(a.time,utc=True)
            for fold,g in d.groupby('fold'):
                cal=c[c.fold==fold];global_weight,global_scores=choose_weight(cal,[0,.25,.5,.75,1])
                for sensor,s in g.groupby('sensor'):
                    local=cal[cal.sensor==sensor];alpha,scores=choose_weight(local,[0,.25,.5,.75,1]);switch,switch_scores=choose_weight(local,[0,1])
                    if len(local)<30:alpha=global_weight;switch=0.
                    decisions.append({'variable':variable,'fold':int(fold),'sensor':sensor,'calibration_n':len(local),
                        'latest_calibration_outcome':str(cal.time.max()+pd.Timedelta(hours=6)),
                        'global_weight':global_weight,'site_weight':alpha,'site_switch':switch,'global_scores':global_scores,'site_scores':scores,'switch_scores':switch_scores,
                        'fallback':'global blend / persistence switch' if len(local)<30 else 'none'})
                    for family,w in [('ml',1.),('persistence',0.),('half',.5),('global_blend',global_weight),('site_blend',alpha),('site_switch',switch)]:
                        a=s[['fold','sensor','time','truth','persistence']].copy();a['prediction']=s.persistence+w*(s.prediction-s.persistence)
                        a['variable']=variable;a['family']=family;a['alpha']=w;rows.append(a)
                # Transfer uses frozen ML residuals; target-site residuals are wholly withheld in remote2.
                history=pd.concat([cal,g],ignore_index=True)
                ht=pd.DatetimeIndex(history.time).as_unit('ns').asi8+pd.Timedelta(hours=7).value
                err=(history.truth-history.prediction).to_numpy();names=history.sensor.to_numpy()
                for r in g.itertuples():
                    now=r.time.value;recent=(ht<=now)&(ht>=now-pd.Timedelta(days=30).value)&(names!=r.sensor)
                    source_names=np.unique(names[recent]);weights=[];values=[];source_count=0
                    for name in source_names:
                        use=recent&(names==name)
                        if use.sum()<10:continue
                        source_count+=1
                        distance=float(np.hypot(meta.loc[name,'x_m']-meta.loc[r.sensor,'x_m'],meta.loc[name,'y_m']-meta.loc[r.sensor,'y_m']))
                        values.extend(abs(err[use]));weights.extend([1/(250+distance)**2/use.sum()]*int(use.sum()))
                    radius=np.nan;effective_n=0.
                    if source_count>=2 and len(values)>=30:
                        order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order];w/=w.sum()
                        effective_n=float(1/(w*w).sum());radius=float(v[np.searchsorted(np.cumsum(w),.9)])
                    local_radius=(r.upper-r.lower)/2 if np.isfinite(r.lower) else np.nan
                    for policy,rad in [('local30',local_radius),('remote2',radius),('local_else_remote2',local_radius if np.isfinite(local_radius) else radius)]:
                        transfers.append({'variable':variable,'fold':fold,'sensor':r.sensor,'time':str(r.time),'truth':r.truth,'prediction':r.prediction,
                            'policy':policy,'lower':r.prediction-rad,'upper':r.prediction+rad,'remote_source_count':source_count,'remote_residual_count':len(values),'remote_effective_n':effective_n})
                print(variable,'robust checks completed quarter',fold+1,flush=True)
        pd.concat(rows,ignore_index=True).to_csv(output/'blend_predictions.csv.gz',index=False)
        pd.DataFrame(transfers).to_csv(output/'transfer_intervals.csv.gz',index=False)
        save(output/'blend_decisions.json',decisions)
        save(output/'contract.json',{'source_run':SOURCE,'scope':'Frozen quarterly weather-tree point predictions; no refits or 2026 labels. Blending weights use earlier 45-day calibration only.',
            'blend':'Grid 0/.25/.5/.75/1, ties favor persistence. Global equal-site calibration MAE, sites need >=30 calibration outcomes. Site blend falls back to global, site switch to persistence.',
            'uncertainty':'Remote2 excludes all target-site residuals, uses only preceding 30-day residuals available after endpoint+1h, >=2 remote sources with >=10 residuals each and >=30 total. Distance-weighted absolute-error 90th percentile; heuristic transfer, not a conformal guarantee. Local_else_remote2 fills only missing local intervals.',
            'limits':['Previously selected 2025 model hyperparameters.','No recalibrated intervals claimed for newly blended forecasts.','Remote calibration excludes target residuals, but base point model may have trained on target-site history; not whole-model leave-site-out.','Two nearby sensors may share errors; effective residual count does not measure independent oceanographic events.']})
        save(status,{'state':'COMPLETED','result_sha256':{p.relative_to(ROOT/'results').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}})
    except BaseException as e:save(status,{'state':'FAILED','error':str(e)});traceback.print_exc();raise


if __name__=='__main__':run()
