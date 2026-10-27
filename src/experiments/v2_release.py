"""Release pipeline under tuning protocol v2: Forecast-2025 quarters, persistence blend, 2025-only refits.

The v1 modules (bay_seasonal_validation, bay_robust_forecast, bay_reserved_refit) are bound by the
v1 contract, untracked in git, and hard-code their hyperparameter sources, so they are not edited.
The functions below reproduce their logic with the source made an argument. `--params v1` must
reproduce the v1 artifacts; that check is run before any v2 number is trusted.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.experiments.bay_graph_dynamics import ROOT, build, save
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_seasonal_validation import interval_rows
from src.experiments.bay_robust_forecast import choose_weight
from src.data.bay_forcing import weather

V1_PARAMS = 'reports/bay_twin/runs/20260910T120301Z_dynamics_tree_pilot/results/dynamics_{v}/selection_summary.json'
V2_FORECAST_RUN = 'reports/bay_twin/runs/20260919T144822Z_v2_forecast/results/v2_forecast_{v}_weather_tree_release/{v}_weather_tree_release_selection.json'
V2_FIELD_RUN = 'reports/bay_twin/runs/20260919T145045Z_v2_field/results/v2_field_held_site_A1_both_direct/held_site_A1_both_direct_selection.json'
V1_ALPHA = 65648.17611753459


def tree_params(variable, which):
    if which == 'v1': return json.loads((ROOT / V1_PARAMS.format(v=variable)).read_text())['params']
    sel = json.loads((ROOT / V2_FORECAST_RUN.format(v=variable)).read_text())
    assert sel['completed_trials'] >= 100 and not sel['smoke_only']
    return {**sel['params'], 'subsample_freq': 1}


def field_alpha(which):
    if which == 'v1': return V1_ALPHA
    sel = json.loads((ROOT / V2_FIELD_RUN).read_text()); assert sel['completed_trials'] >= 100 and not sel['smoke_only']
    return sel['params']['alpha']


def fit_tree(params, x, y, w):
    return lgb.LGBMRegressor(**params, objective='regression_l1', verbosity=-1, n_jobs=4, random_state=42,
                             deterministic=True, force_col_wise=True).fit(x, y, sample_weight=w)


def seasonal(variable, params, out):
    """bay_seasonal_validation.run with explicit params."""
    out.mkdir(parents=True, exist_ok=True)
    data = build(variable); t = data['times']; met = weather(t).to_numpy()
    rows = []; calrows = []; contracts = []
    boundaries = ['2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01', '2026-01-01']
    for fold, (begin, finish) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        start = pd.Timestamp(begin, tz='UTC'); end = pd.Timestamp(finish, tz='UTC')
        cal_end = start - pd.Timedelta(hours=72); cal_start = cal_end - pd.Timedelta(days=45)
        tr = np.flatnonzero(t + pd.Timedelta(hours=78) < cal_start)
        ca = np.flatnonzero((t >= cal_start) & (t + pd.Timedelta(hours=6) < cal_end))
        va = np.flatnonzero((t >= start) & (t + pd.Timedelta(hours=6) < end))
        eligible = data['valid'][tr].sum(0) >= 100
        if not eligible.any(): continue
        fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
        base, _ = features(data, {'fallback': fallback}, 'forecast')
        x_weather = np.concatenate([base, np.broadcast_to(met[:, None], (len(t), len(data['names']), met.shape[1]))], axis=-1)
        mask = data['valid'] & eligible[None]; tm = mask[tr]
        weight = 1 / np.maximum(tm.sum(0), 1); w = np.broadcast_to(weight, tm.shape)[tm]; w /= w.mean()
        contracts.append({'fold': fold, 'selection_start': str(start), 'selection_end': str(end),
                          'supported_sensors': [n for n, e in zip(data['names'], eligible) if e]})
        for family, x in [('tree', base), ('weather_tree', x_weather)]:
            model = fit_tree(params, x[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], w)
            phase_rows = {}
            for phase, ids in [('calibration', ca), ('selection', va)]:
                pm = mask[ids]; pred = model.predict(x[ids][pm]); records = []
                for (j, s), value in zip(np.argwhere(pm), pred):
                    i = ids[j]; anchor = float(data['x'][i, s] * data['scale'] + data['center'])
                    records.append({'fold': fold, 'sensor': data['names'][s], 'time': str(t[i]), 'family': family,
                                    'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                                    'prediction': float(anchor + value * data['scale']), 'persistence': anchor})
                phase_rows[phase] = pd.DataFrame(records)
            c = phase_rows['calibration']; d = phase_rows['selection']; calrows.append(c)
            for sensor, g in d.groupby('sensor'):
                cs = c[c.sensor == sensor]
                for policy in ['fixed', 'rolling30']: rows.append(interval_rows(cs, g, policy))
        print(variable, 'completed quarter', fold + 1, flush=True)
    pd.concat(rows, ignore_index=True).to_csv(out / 'predictions.csv.gz', index=False)
    pd.concat(calrows, ignore_index=True).to_csv(out / 'calibration.csv.gz', index=False)
    save(out / 'contract.json', {'variable': variable, 'folds': contracts, 'params': params})


def robust(seasonal_roots, out):
    """bay_robust_forecast.run point-blend part (global/site blends) with explicit source."""
    out.mkdir(parents=True, exist_ok=True); rows = []; decisions = []
    for variable, root in seasonal_roots.items():
        d = pd.read_csv(root / 'predictions.csv.gz'); d = d[(d.family == 'weather_tree') & (d.policy == 'rolling30')].copy()
        c = pd.read_csv(root / 'calibration.csv.gz'); c = c[c.family == 'weather_tree'].copy()
        for a in [d, c]: a['time'] = pd.to_datetime(a.time, utc=True)
        for fold, g in d.groupby('fold'):
            cal = c[c.fold == fold]; global_weight, _ = choose_weight(cal, [0, .25, .5, .75, 1])
            for sensor, s in g.groupby('sensor'):
                local = cal[cal.sensor == sensor]; alpha, _ = choose_weight(local, [0, .25, .5, .75, 1]); switch, _ = choose_weight(local, [0, 1])
                if len(local) < 30: alpha = global_weight; switch = 0.
                decisions.append({'variable': variable, 'fold': int(fold), 'sensor': sensor, 'global_weight': global_weight, 'site_weight': alpha})
                for family, w in [('ml', 1.), ('persistence', 0.), ('half', .5), ('global_blend', global_weight), ('site_blend', alpha), ('site_switch', switch)]:
                    a = s[['fold', 'sensor', 'time', 'truth', 'persistence']].copy(); a['prediction'] = s.persistence + w * (s.prediction - s.persistence)
                    a['variable'] = variable; a['family'] = family; a['alpha'] = w; rows.append(a)
    pd.concat(rows, ignore_index=True).to_csv(out / 'blend_predictions.csv.gz', index=False)
    save(out / 'blend_decisions.json', decisions)


def refit_forecast(variable, params, out):
    """bay_reserved_refit.refit_forecast with explicit params (2025-only training, Q4-2025 calibration)."""
    out.mkdir(parents=True, exist_ok=True)
    data = build(variable); t = data['times']; met = weather(t).to_numpy()
    cal_end = pd.Timestamp('2025-12-28T18:00:00', tz='UTC'); cal_start = pd.Timestamp('2025-11-14T00:00:00', tz='UTC'); train_end = cal_start - pd.Timedelta(hours=78)
    tr = np.flatnonzero(t + pd.Timedelta(hours=6) <= train_end); ca = np.flatnonzero((t >= cal_start) & (t + pd.Timedelta(hours=6) <= cal_end))
    eligible = data['valid'][tr].sum(0) >= 100; fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
    base, _ = features(data, {'fallback': fallback}, 'forecast')
    x_weather = np.concatenate([base, np.broadcast_to(met[:, None], (len(t), len(data['names']), met.shape[1]))], axis=-1)
    mask = data['valid'] & eligible[None]; tm = mask[tr]
    weight = 1 / np.maximum(tm.sum(0), 1); w = np.broadcast_to(weight, tm.shape)[tm]; w /= w.mean()
    contract = {'model_family': 'weather_tree', 'variable': variable, 'train_start': str(t[tr].min()), 'last_training_outcome': str(t[tr].max() + pd.Timedelta(hours=6)),
                'first_calibration': str(t[ca].min()), 'last_calibration_outcome': str(t[ca].max() + pd.Timedelta(hours=6)),
                'supported_sensors': [n for n, e in zip(data['names'], eligible) if e],
                'training_counts': {n: int(data['valid'][tr, s].sum()) for s, n in enumerate(data['names']) if eligible[s]}}
    model = fit_tree(params, x_weather[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], w)
    cm = mask[ca]; pred_inc = model.predict(x_weather[ca][cm]); cal_records = []
    for (j, s), val in zip(np.argwhere(cm), pred_inc):
        i = ca[j]; anchor = float(data['x'][i, s] * data['scale'] + data['center'])
        cal_records.append({'sensor': data['names'][s], 'time': str(t[i]), 'variable': variable, 'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                            'prediction': float(anchor + val * data['scale']), 'persistence': anchor})
    cal_df = pd.DataFrame(cal_records); cal_df.to_csv(out / f'calibration_{variable}.csv.gz', index=False, compression='gzip')
    best_alpha, grid_scores = choose_weight(cal_df, [0.0, 0.25, 0.5, 0.75, 1.0])
    save(out / f'refit_{variable}_model.json', {'model_string': model.booster_.model_to_string(), 'params': params, 'fallback': fallback.tolist(), 'names': data['names'],
                                                'center': data['center'], 'scale': data['scale'], 'feature_count': x_weather.shape[-1], 'contract': contract,
                                                'global_blend_weight': best_alpha, 'calibration_grid_scores': grid_scores})
    print(f'[{variable}] refit blend weight {best_alpha}', flush=True)
    return best_alpha


def main(which):
    out = ROOT / 'reports/bay_twin/runs' / f'v2_release_{which}'; out.mkdir(parents=True, exist_ok=True)
    params = {v: tree_params(v, which) for v in ['temperature', 'oxygen']}
    roots = {}
    for v in ['temperature', 'oxygen']:
        roots[v] = out / f'seasonal_{v}'; seasonal(v, params[v], roots[v])
    robust(roots, out / 'robust_forecast')
    blend = {v: refit_forecast(v, params[v], out / 'refit_forecast') for v in ['temperature', 'oxygen']}
    save(out / 'release_manifest.json', {'which': which, 'tree_params': params, 'field_alpha': field_alpha(which), 'refit_blend_weights': blend,
                                         'files_sha256': {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'release_manifest.json'}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--params', choices=['v1', 'v2'], required=True); main(p.parse_args().params)
