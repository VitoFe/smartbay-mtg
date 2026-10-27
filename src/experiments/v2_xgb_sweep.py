"""Addendum 50: XGBoost against LightGBM at six hours over six issuance phases and three sampler seeds.

Chapter 5 reported the library comparison (xgb_tree - tree at 6 h) on the published issuance grid and
the protocol's sampler seed only, and marked it 'not swept'. This sweeps it the way the principal
contrasts were swept (v2_phase_shift --objective ratio), with every arm selected per outer fold as
addendum 47 does: one study per (variable, arm, offset, sampler seed, outer fold) whose objective is
the fold-macro forecast MAE over the persistence MAE on that fold's own inner window.

Cells: sampler seed 42 at offsets 1 to 5, and seeds 7 and 123 at offset 0. The offset-0, seed-42
cell is the per-fold selection of addendum 47 (LightGBM tree) and addendum 48 (XGBoost tree), reused
after gate G1 shows build_shifted(..., 0) is v2_horizon.build. Every XGBoost fit uses n_jobs = 2,
because XGBoost's row subsampling depends on the thread count (addendum 48 amendment 1).

Declared in config/bay_reserved_protocol_v2_addendum50.json and its amendment 1.
Usage: python -m src.experiments.v2_xgb_sweep gate
       python -m src.experiments.v2_xgb_sweep cell --variable oxygen --arm xgb_tree --offset 3 --seed 42
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_xgb_sweep'
DECLARED = 'config/bay_reserved_protocol_v2_addendum50.json'
HORIZON = 6
ARMS = ('tree', 'xgb_tree')
REPLICATES = [(o, 42) for o in range(1, 6)] + [(0, 7), (0, 123)]
KEY = ['fold', 'sensor', 'time']


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def cells():
    return [(v, a, o, s) for (o, s) in REPLICATES for v in ('temperature', 'oxygen') for a in ARMS]


def reused(variable, arm):
    """The offset-0, seed-42 per-fold predictions of addenda 47 and 48."""
    if arm == 'tree':
        return RUNS / 'v2_perfold/forecast/perfold' / f'{variable}_tree_h6'
    return RUNS / 'v2_perfold_rest/forecast/perfold' / f'xgb_{variable}_xgb_tree_h6'


def gate():
    """G1: build_shifted(variable, 6, 0) is v2_horizon.build(variable, 6) in every array the trees read."""
    from src.experiments.v2_horizon import build
    from src.experiments.v2_phase_shift import build_shifted
    out = {}
    for variable in ('temperature', 'oxygen'):
        a, b = build(variable, HORIZON), build_shifted(variable, HORIZON, 0)
        same = {'times': bool((a['times'] == b['times']).all()), 'names': a['names'] == b['names']}
        for k in ('x', 'truth', 'valid', 'forcing', 'available', 'adjacency'):
            same[k] = bool(a[k].shape == b[k].shape and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == 'f'))
        out[variable] = {'identical': same, 'pass': all(same.values())}
        for arm in ARMS:
            r = json.loads((reused(variable, arm) / 'report.json').read_text())
            out[variable][f'reused_{arm}'] = {'source': reused(variable, arm).relative_to(ROOT).as_posix(),
                                             'completed_trials': [s['completed_trials'] for s in r['selection'].values()],
                                             'threads': r.get('threads')}
    _json(OUT / 'gate_G1.json', out)
    print(json.dumps(out, indent=1))
    return out


def run_cell(variable, arm, offset, seed):
    from src.experiments.v2_forecast import THREADS, tree_space, xgb_space
    from src.experiments.v2_horizon import climatology, folds, inner_fold, ratio, tree_fit_score
    from src.experiments.v2_phase_shift import build_shifted, design_for
    from src.tuning import protocol_v2
    if arm == 'xgb_tree' and THREADS != 2:
        raise RuntimeError('amendment 1: every XGBoost fit uses n_jobs = 2 (V2_THREADS=2)')
    if (offset, seed) not in REPLICATES:
        raise ValueError('undeclared replicate')
    t0 = time.time()
    data = build_shifted(variable, HORIZON, offset)
    outer = folds(data, HORIZON)
    inner = [inner_fold(data, f, HORIZON) for f in outer]
    clim = {id(f): climatology(data, f) for f in outer + inner}
    cache = {id(f): design_for(data, f, False) for f in outer + inner}
    tag = f'{variable}_{arm}_h6_shift{offset}_seed{seed}'
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    space = xgb_space if arm == 'xgb_tree' else tree_space

    def fitted(raw):
        return dict(raw) if arm == 'xgb_tree' else {**raw, 'subsample_freq': 1}

    saved = protocol_v2.PROTOCOL['sampler_seed']
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    try:
        params, sels = {}, {}
        for f, fi in zip(outer, inner):
            def objective(trial, _f=fi):
                return ratio(tree_fit_score(data, cache[id(_f)], _f, space(trial), clim[id(_f)])[0])
            sel = protocol_v2.run_study(f'{tag}_perfold_fold{f["number"]}', objective, folder)
            sels[str(f['number'])] = sel
            params[f['number']] = fitted(sel['params'])
    finally:
        protocol_v2.PROTOCOL['sampler_seed'] = saved
    d = pd.concat([tree_fit_score(data, cache[id(f)], f, params[f['number']], clim[id(f)])[0] for f in outer],
                  ignore_index=True)
    again = pd.concat([tree_fit_score(data, cache[id(f)], f, params[f['number']], clim[id(f)])[0] for f in outer],
                      ignore_index=True)
    repeat = float(np.abs(again.prediction.to_numpy() - d.prediction.to_numpy()).max())
    path = folder / 'selected_predictions.csv.gz'
    d.to_csv(path, index=False)
    e = (d.truth - d.prediction).abs()
    mae = float(d.assign(e=e).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())
    report = {'declared_in': DECLARED, 'variable': variable, 'arm': arm, 'offset_hours': offset, 'sampler_seed': seed,
              'horizon_hours': HORIZON, 'threads': THREADS, 'selection': sels,
              'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'objective': "fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only",
              'endpoints': int(len(d)), 'mae': mae, 'gate_G3_repeat_max_abs_diff': repeat,
              'gate_G3_pass': bool(repeat <= 1e-9), 'seconds': round(time.time() - t0, 1),
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    _json(folder / 'report.json', report)
    print(f'{tag}: MAE {mae:.5f}  G3 {report["gate_G3_pass"]}  ({report["seconds"]} s)', flush=True)


def run_job(spec):
    """'gate' or 'cell:<variable>:<arm>:<offset>:<seed>'."""
    parts = spec.split(':')
    if parts[0] == 'gate':
        gate()
    elif parts[0] == 'cell':
        run_cell(parts[1], parts[2], int(parts[3]), int(parts[4]))
    else:
        raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('gate')
    sub.add_parser('cells')
    c = sub.add_parser('cell')
    c.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    c.add_argument('--arm', choices=ARMS, required=True)
    c.add_argument('--offset', type=int, required=True)
    c.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    if a.cmd == 'gate':
        gate()
    elif a.cmd == 'cells':
        for v, arm, o, s in cells():
            print(f'cell:{v}:{arm}:{o}:{s}')
    else:
        run_cell(a.variable, a.arm, a.offset, a.seed)


if __name__ == '__main__':
    main()
