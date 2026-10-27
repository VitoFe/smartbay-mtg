"""Addendum 82, part B: the exogenous feature families over the six issuance grids and three sampler seeds.

Table tab:bay_feature_families reads each exogenous family against the arm it augments. Eight of its
contrasts were point estimates on one issuance grid and one sampler seed. This module gives each the
treatment addenda 44, 45 and 78 gave the river level: for one (arm, variable, horizon, phase, seed) it
tunes the arm with its own 100-trial study on the reference objective (mean over the two inner folds of
fold-macro MAE over persistence MAE), refits it on both development folds and writes its predictions.
The plain and weather trees it is set against are reused from the phasefix run where they exist, and
are tuned here, by `base`, at 48 h on oxygen, where they do not. Pairing, gates and contrasts are made
by scripts/v2_phase_family_contrasts.py.

Every arm is built by its own published design function, imported and not copied:
    weather_river_tree  v2_phase_weather_river.weather_river_design (the Magra level beside the weather)
    sealevel_tree       v2_ispra.design_for (the ISPRA station's sea level)
    extboundary_tree    v2_ispra.design_for (the ISPRA station's water temperature)
    bathy_tree          v2_bathy.design_for (the seabed)

Declared in config/bay_reserved_protocol_v2_addendum82.json (part B). Development folds only: the grid
ends 2025-12-31T23:00 and no 2026 value is loaded.
Usage (from the repository root, V2_THREADS=2):
  python -m src.experiments.v2_phase_family arm --arm sealevel_tree --variable oxygen --horizon 6 --offset 0 [--seed 7]
  python -m src.experiments.v2_phase_family base --variable oxygen --horizon 48 --offset 0 [--seed 7]
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.v2_forecast import tree_space
from src.experiments.v2_horizon import ROOT, climatology, folds, inner_fold, ratio, tree_fit_score
from src.experiments.v2_phase_shift import build_shifted, design_for, fit_grid, macro
from src.tuning.protocol_v2 import run_study

DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
OUT = Path(os.environ.get('V2_ADD82_OUT', str(ROOT / 'reports/bay_twin/runs/v2_addendum82/B')))
ARMS = {'weather_river_tree': 'river', 'sealevel_tree': 'ispra', 'extboundary_tree': 'ispra',
        'bathy_tree': 'bathy'}
#: the eight swept cells: phases 0-5 at the reference seed, and seeds 7 and 123 on the reference grid
CELLS = [(k, None) for k in range(6)] + [(0, 7), (0, 123)]


def suffix(seed):
    return '' if seed is None else f'_seed{seed}'


def set_seed(seed):
    """run_study reads the sampler seed at call time, so the bound tuning module is left as it is."""
    from src.tuning import protocol_v2
    protocol_v2.PROTOCOL['sampler_seed'] = 42 if seed is None else int(seed)
    return protocol_v2.PROTOCOL['sampler_seed']


def arm_design(arm, data, fold):
    """The arm's published design function for this grid and fold."""
    if arm == 'weather_river_tree':
        from src.experiments.v2_phase_weather_river import weather_river_design
        return weather_river_design(data, fold)
    if arm in ('sealevel_tree', 'extboundary_tree'):
        from src.experiments.v2_ispra import design_for as ispra_design
        return ispra_design(data, fold, arm)
    if arm == 'bathy_tree':
        from src.experiments.v2_bathy import design_for as bathy_design
        return bathy_design(data, fold)
    raise ValueError(arm)


def environment():
    import lightgbm
    import optuna
    return {'host': socket.gethostname(), 'python': platform.python_version(), 'optuna': optuna.__version__,
            'lightgbm': lightgbm.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
            'V2_THREADS': os.environ.get('V2_THREADS'), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


def _quick(outer, inner):
    for f in outer + inner:
        f['tr'] = f['tr'][-300:]
        f['va'] = f['va'][:40]


def setup(variable, horizon, offset, quick):
    data = build_shifted(variable, horizon, offset)
    outer = folds(data, horizon)
    inner = [inner_fold(data, f, horizon) for f in outer]
    if quick:
        _quick(outer, inner)
    clim = {id(f): climatology(data, f) for f in outer + inner}
    return data, outer, inner, clim


def physical(data, grid, j, s):
    return float(grid[j, s] * data['scale'] + data['center'])


def run_arm(arm, variable, horizon, offset, seed=None, quick=False):
    if arm not in ARMS:
        raise ValueError(arm)
    t0 = time.time()
    out = OUT / 'smoke' if quick else OUT
    folder = out / 'arms'
    sampler_seed = set_seed(seed)
    data, outer, inner, clim = setup(variable, horizon, offset, quick)
    cache = {id(f): arm_design(arm, data, f) for f in outer + inner}

    def objective(trial):
        p = tree_space(trial)
        return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))

    name = f'{variable}_{arm}_h{horizon}_shift{offset}_ratio{suffix(seed)}'
    selection = run_study(name, objective, folder / 'studies', quick)
    params = {**selection['params'], 'subsample_freq': 1}

    rows = []
    for fold in outer:
        grid = fit_grid(data, fold, cache[id(fold)], params)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for j, i in enumerate(fold['va']):
            for s, sensor in enumerate(data['names']):
                if mask[j, s]:
                    rows.append({'fold': fold['number'], 'sensor': sensor, 'time': str(data['times'][i]),
                                 'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                                 arm: physical(data, grid, j, s)})
    frame = pd.DataFrame(rows)
    path = folder / f'phase_family_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    report = {'declared_in': DECLARED, 'part': 'B', 'mode': 'arm', 'family': ARMS[arm], 'arm': arm,
              'variable': variable, 'horizon_hours': horizon, 'offset_hours': offset, 'sampler_seed': sampler_seed,
              'objective': 'ratio: mean over the inner folds of fold-macro MAE / persistence MAE',
              'smoke_only': quick, 'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
              'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
              'selection': selection, 'mae': macro(frame, arm), 'environment': environment(),
              'seconds': round(time.time() - t0, 1), 'prediction_file': path.name,
              'prediction_sha256': _sha(path)}
    if ARMS[arm] == 'ispra':
        from src.data.bay_ispra import BINNINGS, binning
        report['hourly_binning'] = binning()
        report['hourly_binning_rule'] = BINNINGS[binning()]
    _save(folder / f'phase_family_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}_summary.json', report)
    print(f'== {name}: n={report["endpoints"]} MAE {report["mae"]:.6f} in {report["seconds"]} s', flush=True)


def run_base(variable, horizon, offset, seed=None, quick=False):
    """The plain and weather trees at one cell, exactly as the phasefix run tuned them (v2_phase_shift.run,
    objective 'ratio'), for one variable, written in that run's schema so the pairing code is shared."""
    t0 = time.time()
    out = OUT / 'smoke' if quick else OUT
    folder = out / f'base_h{horizon}'
    sampler_seed = set_seed(seed)
    data, outer, inner, clim = setup(variable, horizon, offset, quick)
    cache = {(id(f), w): design_for(data, f, w) for f in outer + inner for w in (True, False)}
    params, selections = {}, {}
    for with_weather, arm in [(False, 'tree'), (True, 'weather_tree')]:
        def objective(trial, w=with_weather):
            p = tree_space(trial)
            return float(np.mean([ratio(tree_fit_score(data, cache[(id(f), w)], f, p, clim[id(f)])[0])
                                  for f in inner]))
        sel = run_study(f'{variable}_{arm}_h{horizon}_shift{offset}_ratio{suffix(seed)}', objective,
                        folder / 'studies', quick)
        selections[arm] = sel
        params[arm] = {**sel['params'], 'subsample_freq': 1}

    rows = []
    for fold in outer:
        grids = {a: fit_grid(data, fold, cache[(id(fold), a == 'weather_tree')], params[a]) for a in params}
        cl, _ = clim[id(fold)]
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for j, i in enumerate(fold['va']):
            for s, sensor in enumerate(data['names']):
                if not mask[j, s]:
                    continue
                rows.append({'fold': fold['number'], 'sensor': sensor, 'time': str(data['times'][i]),
                             'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                             'tree': physical(data, grids['tree'], j, s),
                             'weather_tree': physical(data, grids['weather_tree'], j, s),
                             'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                             'climatology': float(cl[i, s] * data['scale'] + data['center'])})
    frame = pd.DataFrame(rows)
    path = folder / f'phase_shift_{variable}_offset{offset}_ratio{suffix(seed)}.csv.gz'
    frame.to_csv(path, index=False)
    entry = {'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
             'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()), 'params': params,
             'selections': selections,
             'mae': {a: macro(frame, a) for a in ['tree', 'weather_tree', 'persistence', 'climatology']},
             'prediction_sha256': _sha(path)}
    report = {'declared_in': DECLARED, 'part': 'B', 'mode': 'base', 'objective': 'ratio',
              'sampler_seed': sampler_seed, 'offset_hours': offset, 'horizon_hours': horizon,
              'smoke_only': quick, 'environment': environment(), 'seconds': round(time.time() - t0, 1),
              'variables': {variable: entry}}
    _save(folder / f'phase_shift_offset{offset}_ratio{suffix(seed)}_summary.json', report)
    print(f'== base {variable} h{horizon} offset {offset} seed {sampler_seed}: n={entry["endpoints"]} '
          f'tree {entry["mae"]["tree"]:.6f} weather {entry["mae"]["weather_tree"]:.6f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['arm', 'base'])
    p.add_argument('--arm', choices=sorted(ARMS))
    p.add_argument('--variable', required=True, choices=['temperature', 'oxygen'])
    p.add_argument('--horizon', type=int, required=True, choices=[6, 24, 48])
    p.add_argument('--offset', type=int, required=True, choices=range(6))
    p.add_argument('--seed', type=int, default=None, choices=[7, 123])
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.seed is not None and a.offset != 0:
        raise SystemExit('the further sampler seeds are run on the reference grid only')
    if a.mode == 'arm':
        if a.arm is None:
            raise SystemExit('--arm is required in arm mode')
        run_arm(a.arm, a.variable, a.horizon, a.offset, a.seed, a.quick)
    else:
        run_base(a.variable, a.horizon, a.offset, a.seed, a.quick)


if __name__ == '__main__':
    main()
