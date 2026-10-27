"""Tuning protocol v2 for the oxygen outage study (Table 5.7 Panel B).

Identical grid, features, folds, calibration windows and family-selection policy as
bay_oxygen_memory.py. The only change: the ridge challenger's penalty (v1: fixed alpha=100) is
tuned by a >=100-trial TPE study per (fold, horizon) on an inner split of that fold's training
episodes (latest 25 % of training episodes held out), so the calibration window remains untouched
for the unchanged family-selection step.
"""
import hashlib, json, os, traceback
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.data.bay_profile_context import ROOT, load_observations
from src.experiments.bay_split_calibration import auxiliary_boundary
from src.experiments.bay_thermal_memory import QUARTERS
from src.experiments.bay_oxygen_memory import build_grid, score, FEATURES, FAMILIES
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.experiments.bay_twin_sweep import save
from src.tuning.protocol_v2 import run_study, PROTOCOL


def ridge(alpha):
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha))


def fit(tr, alpha):
    w = 1 / tr.groupby('sensor').sensor.transform('size').to_numpy(); w = w / w.mean()
    return ridge(alpha).fit(tr[FEATURES], tr.target_oxygen - tr.initial_oxygen, ridge__sample_weight=w)


def run(quick=False):
    out = ROOT / 'results'; out.mkdir(exist_ok=True); root = out / 'v2_oxygen_memory'; root.mkdir(exist_ok=True)
    status = out / 'status_v2_oxygen_memory.json'; save(status, {'state': 'RUNNING', 'pid': os.getpid()})
    try:
        meta, frames = load_observations(); meta = meta.loc[[s for s, f in frames.items() if 'oxygen' in f and f.oxygen.notna().any()]]
        d = build_grid(meta, frames); now = auxiliary_boundary(d, meta); anchor = d.copy(); anchor.t_utc = d.source_time.fillna(d.start); before = auxiliary_boundary(anchor, meta)
        d['coarse_delta'] = now[:, 2] - before[:, 2]; d['solubility_delta'] = solubility(now[:, 0], now[:, 1]) - solubility(before[:, 0], before[:, 1])
        d['temperature_delta'] = now[:, 0] - before[:, 0]; d['salinity_delta'] = now[:, 1] - before[:, 1]
        d['year_sin'] = np.sin(d.t_utc.dt.dayofyear * 2 * np.pi / 365.25); d['year_cos'] = np.cos(d.t_utc.dt.dayofyear * 2 * np.pi / 365.25)
        rows = []; contracts = []
        for fold, (begin, end) in enumerate(zip(QUARTERS[:-1], QUARTERS[1:])):
            begin = pd.Timestamp(begin, tz='UTC'); end = pd.Timestamp(end, tz='UTC'); calstart = begin - pd.Timedelta(days=100)
            for h in [24, 72]:
                tr = d[(d.horizon == h) & (d.start + pd.Timedelta(hours=72) < calstart - pd.Timedelta(hours=72))].dropna(subset=['initial_oxygen', 'target_oxygen'])
                ca = d[(d.horizon == h) & (d.start >= calstart) & (d.start + pd.Timedelta(hours=72) < begin - pd.Timedelta(hours=72))]
                ev = d[(d.horizon == h) & (d.start >= begin) & (d.start + pd.Timedelta(hours=72) < end)]
                predictions = {'persistence': d.initial_oxygen.to_numpy()}
                for name, w in [('coarse025', .25), ('coarse050', .5), ('coarse100', 1.)]: predictions[name] = d.initial_oxygen.to_numpy() + w * d.coarse_delta.to_numpy()
                predictions['solubility'] = d.initial_oxygen.to_numpy() + d.solubility_delta.to_numpy()
                ml_supported = len(tr) >= 30 and tr.episode.nunique() >= 6 and tr.sensor.nunique() >= 2
                predictions['ridge'] = np.full(len(d), np.nan); sel = None
                if ml_supported:
                    episodes = sorted(tr.episode.unique()); held = set(episodes[int(len(episodes) * .75):])
                    itr = tr[~tr.episode.isin(held)]; iva = tr[tr.episode.isin(held)]
                    if len(itr) >= 20 and len(iva) >= 5 and itr.sensor.nunique() >= 2:
                        def objective(trial):
                            m = fit(itr, trial.suggest_float('alpha', 1e-3, 1e6, log=True))
                            return float(score(iva, np.clip(iva.initial_oxygen.to_numpy() + m.predict(iva[FEATURES]), 0, 600)))
                        sel = run_study(f'oxygen_memory_fold{fold}_h{h}', objective, root / 'studies', quick); alpha = sel['params']['alpha']
                    else:
                        alpha = 100.; sel = {'note': 'inner split too small; v1 alpha retained', 'completed_trials': 0}
                    predictions['ridge'] = d.initial_oxygen.to_numpy() + fit(tr, alpha).predict(d[FEATURES])
                predictions = {name: np.clip(p, 0, 600) for name, p in predictions.items()}
                supported = ca.dropna(subset=['initial_oxygen', 'target_oxygen']); scores = {}
                if supported.episode.nunique() >= 10 and supported.group.nunique() >= 2:
                    for family, p in predictions.items():
                        if np.isfinite(p[supported.index]).all(): scores[family] = float(score(supported, p[supported.index]))
                chosen = min(scores, key=scores.get) if scores else 'persistence'
                contracts.append(dict(fold=fold, horizon=h, selected=chosen, scores=scores, ml_supported=ml_supported, ridge_selection=sel, training_rows=len(tr)))
                for family in FAMILIES + ['selected']:
                    r = ev.copy(); r['fold'] = fold; r['family'] = family; r['prediction'] = predictions[chosen if family == 'selected' else family][ev.index]; rows.append(r)
                print(fold, h, chosen, 'ridge alpha', sel and sel.get('params'), flush=True)
        p = root / 'predictions.csv.gz'; pd.concat(rows).to_csv(p, index=False)
        save(root / 'contract.json', {'protocol': PROTOCOL['version'], 'folds': contracts, 'features': FEATURES, 'families': FAMILIES, 'v1_source': 'src/experiments/bay_oxygen_memory.py',
                                      'change_from_v1': 'ridge alpha tuned (>=100 TPE trials) on inner split of training episodes; everything else identical', 'smoke_only': quick})
        save(status, {'state': 'COMPLETED', 'predictions_sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    except BaseException as exc:
        save(status, {'state': 'FAILED', 'error': str(exc)}); traceback.print_exc(); raise


if __name__ == '__main__':
    import argparse; a = argparse.ArgumentParser(); a.add_argument('--quick', action='store_true'); run(a.parse_args().quick)
