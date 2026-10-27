"""Independently refit oxygen candidates and check causal state/policy contracts."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.bay_profile_context import load_observations
from src.experiments.bay_split_calibration import auxiliary_boundary
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility


def macro(d,col='prediction'):
    return d.assign(error=abs(d.target_oxygen-d[col])).groupby(['fold','group','sensor']).error.mean().groupby(['fold','group']).mean().groupby('fold').mean().mean()


def sparse_sensitivity(root,p):
    rows=[]
    for horizon in [24,72]:
        a=p[(p.family=='ridge')&(p.horizon==horizon)].dropna(subset=['target_oxygen','prediction'])
        counts=a.groupby(['fold','sensor']).size();eligible=counts[counts>=5].reset_index()[['fold','sensor']]
        a=a.merge(eligible,on=['fold','sensor'])
        rows.append(dict(horizon=horizon,endpoints=len(a),ridge_mae=float(macro(a)),persistence_mae=float(macro(a,'initial_oxygen')),policy='Posthoc sensitivity only: at least5 observed endpoints per sensor/quarter. Not a selection rule.'))
    pd.DataFrame(rows).to_csv(root/'ridge_support_sensitivity.csv',index=False)


def main(root):
    manifest=json.loads((root/'snapshot_manifest.json').read_text())
    for name,h in manifest['files_sha256'].items():
        if name.startswith(('data/','reports/')):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    status=json.loads((root/'results/status_oxygen_memory.json').read_text());assert status['state']=='COMPLETED'
    for name,h in status['result_sha256'].items():assert hashlib.sha256((root/'results'/name).read_bytes()).hexdigest()==h
    folder=root/'results/oxygen_memory';d=pd.read_csv(folder/'episode_inputs.csv.gz');p=pd.read_csv(folder/'predictions.csv.gz');contract=json.loads((folder/'contract.json').read_text());meta,frames=load_observations()
    for f in [d,p]:
        for col in ['start','t_utc','source_time']:f[col]=pd.to_datetime(f[col],utc=True)
        assert (f.t_utc<pd.Timestamp('2026-01-01',tz='UTC')).all()
        assert (f.t_utc==f.start+pd.to_timedelta(f.horizon,unit='h')).all()
    key=['sensor','episode','horizon'];assert not d.duplicated(key).any()
    starts=pd.DatetimeIndex(d.start.unique()).sort_values();assert (np.diff(starts.as_unit('ns').asi8)>pd.Timedelta(hours=72).value).all()
    assert len(d)==len(starts)*d.sensor.nunique()*2
    for r in d.itertuples():
        f=frames[r.sensor];w=f.loc[(f.index<r.start)&(f.index>=r.start-pd.Timedelta(hours=6))]
        s=w.oxygen.dropna();assert np.isclose(r.initial_oxygen,s.iloc[-1] if len(s) else np.nan,equal_nan=True)
        assert (r.source_time==s.index[-1]) if len(s) else pd.isna(r.source_time)
        assert np.isclose(r.target_oxygen,f.oxygen.get(r.t_utc,np.nan),equal_nan=True)
        for v in ['temperature','salinity']:
            s=w[v].dropna() if v in w else []
            assert np.isclose(getattr(r,'initial_'+v),s.iloc[-1] if len(s) else np.nan,equal_nan=True)
    now=auxiliary_boundary(d,meta);anchor=d.copy();anchor.t_utc=d.source_time.fillna(d.start);before=auxiliary_boundary(anchor,meta)
    for col,values in [('coarse_delta',now[:,2]-before[:,2]),('temperature_delta',now[:,0]-before[:,0]),('salinity_delta',now[:,1]-before[:,1]),('solubility_delta',solubility(now[:,0],now[:,1])-solubility(before[:,0],before[:,1]))]:np.testing.assert_allclose(d[col],values,atol=1e-10)
    for c in contract['folds']:
        tr=d[(d.horizon==c['horizon'])&d.episode.isin(c['training_episodes'])].dropna(subset=['initial_oxygen','target_oxygen']);ca=d[(d.horizon==c['horizon'])&d.episode.isin(c['calibration_episodes'])];ev=d[(d.horizon==c['horizon'])&d.episode.isin(c['evaluation_episodes'])]
        assert not set(tr.episode)&set(ca.episode) and not set(ca.episode)&set(ev.episode)
        if len(tr):assert (tr.start+pd.Timedelta(hours=72)).max()<ca.start.min()-pd.Timedelta(hours=72)
        assert (ca.start+pd.Timedelta(hours=72)).max()<ev.start.min()-pd.Timedelta(hours=72)
        prediction={'persistence':d.initial_oxygen.to_numpy(),'solubility':d.initial_oxygen+d.solubility_delta,'ridge':np.full(len(d),np.nan)}
        for n,w in [('coarse025',.25),('coarse050',.5),('coarse100',1.)]:prediction[n]=d.initial_oxygen+w*d.coarse_delta
        assert c['ml_supported']==(len(tr)>=30 and tr.episode.nunique()>=6 and tr.sensor.nunique()>=2)
        if c['ml_supported']:
            weights=1/tr.groupby('sensor').sensor.transform('size').to_numpy();weights/=weights.mean()
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=100))
            model.fit(tr[contract['features']],tr.target_oxygen-tr.initial_oxygen,ridge__sample_weight=weights)
            prediction['ridge']=d.initial_oxygen+model.predict(d[contract['features']])
        prediction={n:np.clip(np.asarray(v),0,600) for n,v in prediction.items()};supported=ca.dropna(subset=['initial_oxygen','target_oxygen']);scores={}
        if supported.episode.nunique()>=10 and supported.group.nunique()>=2:
            for n in contract['families']:
                if np.isfinite(prediction[n][supported.index]).all():scores[n]=float(macro(supported.assign(fold=0,prediction=prediction[n][supported.index])))
        assert c['selected']==(min(scores,key=scores.get) if scores else 'persistence')
        for n,v in scores.items():np.testing.assert_allclose(v,c['scores'][n],atol=1e-5)
        for family,g in p[(p.fold==c['fold'])&(p.horizon==c['horizon'])].groupby('family'):
            joined=g.merge(ev[key].assign(index=ev.index),on=key,validate='one_to_one');assert len(joined)==len(ev)
            np.testing.assert_allclose(joined.prediction,prediction[c['selected'] if family=='selected' else family][joined['index']],atol=1e-5,equal_nan=True)
    summaries=[];sites=[];pairs=[]
    reference=p[p.family=='persistence']
    for (h,family),g in p.groupby(['horizon','family']):
        target=g.target_oxygen.notna();v=g.dropna(subset=['target_oxygen','prediction'])
        summaries.append(dict(horizon=h,family=family,scheduled=len(g),observed_targets=int(target.sum()),supported=len(v),availability_given_target=len(v)/target.sum(),mae=float(macro(v))))
        for (fold,group,sensor),a in v.groupby(['fold','group','sensor']):sites.append(dict(horizon=h,family=family,fold=fold,group=group,sensor=sensor,endpoints=len(a),mae=float(abs(a.target_oxygen-a.prediction).mean())))
        paired=v.merge(reference[key+['prediction']].rename(columns={'prediction':'baseline'}),on=key,validate='one_to_one').dropna(subset=['baseline'])
        pairs.append(dict(horizon=h,family=family,endpoints=len(paired),mae=float(macro(paired)),persistence_mae=float(macro(paired,'baseline'))))
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False);pd.DataFrame(sites).to_csv(root/'by_sensor.csv',index=False);pd.DataFrame(pairs).to_csv(root/'paired_summary.csv',index=False)
    sparse_sensitivity(root,p)
    rng=np.random.default_rng(20260917);boot=[]
    for h,family in [(24,'selected'),(72,'selected'),(24,'ridge'),(72,'ridge')]:
        a=p[(p.family==family)&(p.horizon==h)].dropna(subset=['target_oxygen','prediction']).copy();draws=[]
        for _ in range(500):
            pieces=[]
            for fold,g in a.groupby('fold'):
                weeks=g.episode.unique();pieces.extend([g[g.episode==week] for week in rng.choice(weeks,len(weeks),replace=True)])
            g=pd.concat(pieces,ignore_index=True);draws.append(macro(g)-macro(g,'initial_oxygen'))
        boot.append(dict(horizon=h,family=family,delta_mae=float(macro(a)-macro(a,'initial_oxygen')),lower=float(np.quantile(draws,.025)),upper=float(np.quantile(draws,.975)),draws=500))
    pd.DataFrame(boot).to_csv(root/'paired_bootstrap.csv',index=False)
    (root/'verification.json').write_text(json.dumps(dict(state='VERIFIED',input_rows=len(d),prediction_rows=len(p),oxygen_sensors=d.sensor.nunique(),checks='Input/output hashes; source measurements and timestamps; weekly nonoverlap; external-feature replay; chronological train/calibration/evaluation separation; all supported ridge fits independently rerun; policy selection and predictions replayed; matched baselines and conditional quarter-stratified week bootstrap.'),indent=2))
    print(pd.DataFrame(pairs).to_string(index=False));print(pd.DataFrame(boot).to_string(index=False));print('decisions',[(c['fold'],c['horizon'],c['selected'],c['ml_supported']) for c in contract['folds']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
