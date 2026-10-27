"""Addendum-7 study 4: does a newly installed probe inherit forecast skill?

The spatial workflow is evaluated by holding out whole stations. The forecast workflow never has
been: every reported figure trains on the same probes it is scored on. An operator adding a probe
to the network wants to know whether it gets useful forecasts before it has a history of its own.

For each eligible sensor the released weather-tree family is refitted on every *other* sensor's
rows and scored on the held-out one, against the same family trained on all sensors and against
persistence. Same folds, same frozen macro rule.

No new hyperparameter selection: both arms use the configuration the existing 6 h nested study
chose, because the comparison is between two training protocols for one family and holding the
configuration fixed is what isolates the protocol change. Declared in
config/bay_reserved_protocol_v2_addendum7.json.
"""
import argparse, hashlib, json, os, zlib

import numpy as np
import pandas as pd

from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_horizon import ROOT, build, folds, climatology
from src.experiments.v2_forecast import make_tree
from src.experiments.v2_closed_loop import selected
from src.data.bay_forcing import weather
from src.experiments.v2_bathy import bathymetric_block

OUT = ROOT / 'reports/bay_twin/runs/v2_held_sensor'
DRAWS = 600


def fit_predict(data, design, fold, params, exclude=None):
    """Train on the fold's rows, optionally excluding one sensor's columns, and score the fold."""
    mask = data['valid'] & fold['eligible'][None, :]
    train_mask = mask.copy()
    if exclude is not None:
        train_mask[:, exclude] = False
    tr, va = fold['tr'], fold['va']
    tm = train_mask[tr]
    if tm.sum() < 200:
        return None
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]; w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})
    model.fit(design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    grid = np.full(data['x'][va].shape, np.nan)
    vm = mask[va]
    grid[vm] = model.predict(design[va][vm]) + data['x'][va][vm]
    return grid


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def paired(frame, x, y, label):
    rng = np.random.default_rng([20260920, zlib.crc32(label.encode())])
    keys = frame[['fold', 'sensor']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in frame.groupby(['fold', 'sensor'])}
    point = macro(frame, x) - macro(frame, y)
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(keys[i])] for i in rng.integers(0, len(keys), len(keys))])
        draws.append(macro(s, x) - macro(s, y))
    return [point, *np.percentile(draws, [2.5, 97.5]).tolist()]


def run(variable, horizon=6, bathy=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data = build(variable, horizon)
    outer = folds(data, horizon)
    met = weather(data['times']).to_numpy()
    params = selected(variable, 'weather_tree', horizon)

    rows = []
    for fold in outer:
        base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
        blocks = [base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))]
        if bathy:
            blocks.append(bathymetric_block(data))
        design = np.concatenate(blocks, axis=-1)
        clim, _ = climatology(data, fold)
        full = fit_predict(data, design, fold, params, exclude=None)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for s in np.flatnonzero(fold['eligible']):
            held = fit_predict(data, design, fold, params, exclude=s)
            if held is None:
                continue
            for j in np.flatnonzero(mask[:, s]):
                i = fold['va'][j]
                rows.append({
                    'fold': fold['number'], 'sensor': data['names'][s], 'time': str(data['times'][i]),
                    'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                    'all_sensor': float(full[j, s] * data['scale'] + data['center']),
                    'held_sensor': float(held[j, s] * data['scale'] + data['center']),
                    'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                    'climatology': float(clim[i, s] * data['scale'] + data['center'])})

    frame = pd.DataFrame(rows)
    suffix = '_bathy' if bathy else ''
    path = OUT / f'held_sensor_{variable}_h{horizon}{suffix}.csv.gz'
    frame.to_csv(path, index=False)

    arms = ['all_sensor', 'held_sensor', 'persistence', 'climatology']
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum7.json', 'variable': variable,
              'horizon_hours': horizon, 'params': params, 'no_new_selection': True, 'bathymetric_features': bathy,
              'sensors': int(frame.sensor.nunique()), 'endpoints': int(len(frame)),
              'mae': {a: macro(frame, a) for a in arms},
              'contrasts': {
                  'held_sensor - all_sensor': paired(frame, 'held_sensor', 'all_sensor', f'{variable}|h-a'),
                  'held_sensor - persistence': paired(frame, 'held_sensor', 'persistence', f'{variable}|h-p'),
                  'held_sensor - climatology': paired(frame, 'held_sensor', 'climatology', f'{variable}|h-c'),
                  'all_sensor - persistence': paired(frame, 'all_sensor', 'persistence', f'{variable}|a-p')},
              'per_sensor': {}}
    for sensor, group in frame.groupby('sensor'):
        e = {a: float((group[a] - group.truth).abs().mean()) for a in arms}
        report['per_sensor'][sensor] = {'endpoints': int(len(group)), 'mae': e,
                                        'penalty': e['held_sensor'] - e['all_sensor']}
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT / f'held_sensor_{variable}_h{horizon}{suffix}_summary.json', report)

    print(f"== {variable} h{horizon}  sensors={report['sensors']}  endpoints={report['endpoints']}")
    for a in arms:
        print(f"   {a:14s} {report['mae'][a]:.4f}")
    for k, (d, lo, hi) in report['contrasts'].items():
        print(f'   {k:32s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))
    worst = sorted(report['per_sensor'].items(), key=lambda kv: -kv[1]['penalty'])[:3]
    print('   largest per-sensor penalties:', [(k, round(v['penalty'], 3)) for k, v in worst])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--bathy', action='store_true')
    a = p.parse_args(); run(a.variable, a.horizon, a.bathy)
