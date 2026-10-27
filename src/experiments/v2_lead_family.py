"""Addendum 83: every physical-feature family at every lead from 1 to 72 h, on both variables.

Table tab:bay_feature_families read each exogenous family at the one or two leads its arm was first run
at. This module runs the plain tree, the weather tree and the five feature arms at the eight leads 1, 3,
6, 12, 24, 36, 48 and 72 h under the two designs the thesis uses:

  base    the plain and weather trees at one (variable, lead, issuance phase, sampler seed) cell, tuned
          on the reference objective and written in the phasefix schema (v2_phase_shift.run, 'ratio'),
          so that the pairing code is shared with the existing sweeps
  arm     one feature arm at one cell, in the schema of addendum 82 part B (v2_phase_family.run_arm)
  perfold one arm selected per outer fold on that fold's own inner window (the per-fold design of
          addendum 48, v2_perfold_rest), on the reference grid at sampler seed 42

Every arm is built by its own published design function, imported and not copied:
    river_tree          v2_phase_river.river_design (the plain tree's features and the Magra level)
    weather_river_tree  v2_phase_weather_river.weather_river_design
    sealevel_tree       v2_ispra.design_for (the ISPRA station's sea level)
    extboundary_tree    v2_ispra.design_for (the ISPRA station's water temperature)
    bathy_tree          v2_bathy.design_for (the seabed)
and, per fold, through v2_perfold_rest.design_builder, which reads v2_river, v2_ispra and v2_bathy.
v2_horizon.run refuses 72 h; its build, folds, inner_fold and climatology take any lead and are used here.

Declared in config/bay_reserved_protocol_v2_addendum83.json. Development folds only: the grid ends
2025-12-31T23:00 and no 2026 value is loaded.
Usage (from the repository root, V2_THREADS=2):
  python -m src.experiments.v2_lead_family base --variable oxygen --horizon 72 --offset 0 [--seed 7]
  python -m src.experiments.v2_lead_family arm --arm river_tree --variable temperature --horizon 1 --offset 3
  python -m src.experiments.v2_lead_family perfold --arm sealevel_tree --variable oxygen --horizon 12
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

from src.experiments.v2_forecast import tree_space
from src.experiments.v2_horizon import ROOT, ratio, tree_fit_score
from src.experiments.v2_phase_family import environment, physical, set_seed, setup, suffix
from src.experiments.v2_phase_family import arm_design as family_design
from src.experiments.v2_phase_shift import design_for, fit_grid, macro
from src.tuning.protocol_v2 import run_study

DECLARED = 'config/bay_reserved_protocol_v2_addendum83.json'
OUT = Path(os.environ.get('V2_ADD83_OUT', str(ROOT / 'reports/bay_twin/runs/v2_addendum83')))
LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
ARMS = ('river_tree', 'weather_river_tree', 'sealevel_tree', 'extboundary_tree', 'bathy_tree')
#: the family each arm belongs to in v2_perfold_rest (design_builder, space, study names)
FAMILY = {'tree': 'horizon', 'weather_tree': 'horizon', 'river_tree': 'river', 'weather_river_tree': 'river',
          'sealevel_tree': 'ispra', 'extboundary_tree': 'ispra', 'bathy_tree': 'bathy'}
#: the eight swept cells: phases 0-5 at the reference seed, and seeds 7 and 123 on the reference grid
CELLS = [(k, None) for k in range(6)] + [(0, 7), (0, 123)]


def arm_design(arm, data, fold):
    """The arm's published design function for this grid and fold."""
    if arm == 'river_tree':
        from src.experiments.v2_phase_river import river_design
        return river_design(data, fold)
    return family_design(arm, data, fold)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


def _binning(report, arm):
    if arm in ('sealevel_tree', 'extboundary_tree'):
        from src.data.bay_ispra import BINNINGS, binning
        report['hourly_binning'] = binning()
        report['hourly_binning_rule'] = BINNINGS[binning()]


def _check(horizon, offset, seed):
    if horizon not in LEADS:
        raise SystemExit(f'undeclared lead {horizon}')
    if seed is not None and offset != 0:
        raise SystemExit('the further sampler seeds are run on the reference grid only')


def run_arm(arm, variable, horizon, offset, seed=None, quick=False):
    if arm not in ARMS:
        raise ValueError(arm)
    _check(horizon, offset, seed)
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
    stem = f'lead_family_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}'
    path = folder / f'{stem}.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    report = {'declared_in': DECLARED, 'mode': 'arm', 'family': FAMILY[arm], 'arm': arm,
              'variable': variable, 'horizon_hours': horizon, 'offset_hours': offset, 'sampler_seed': sampler_seed,
              'objective': 'ratio: mean over the inner folds of fold-macro MAE / persistence MAE',
              'smoke_only': quick, 'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
              'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
              'selection': selection, 'mae': macro(frame, arm), 'environment': environment(),
              'seconds': round(time.time() - t0, 1), 'prediction_file': path.name,
              'prediction_sha256': _sha(path)}
    _binning(report, arm)
    _save(folder / f'{stem}_summary.json', report)
    print(f'== {name}: n={report["endpoints"]} MAE {report["mae"]:.6f} in {report["seconds"]} s', flush=True)


def run_base(variable, horizon, offset, seed=None, quick=False):
    """The plain and weather trees at one cell, as the phasefix run tuned them (v2_phase_shift.run, objective
    'ratio'), for one variable, in that run's schema."""
    _check(horizon, offset, seed)
    t0 = time.time()
    out = OUT / 'smoke' if quick else OUT
    folder = out / 'base' / f'h{horizon}'
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
    report = {'declared_in': DECLARED, 'mode': 'base', 'objective': 'ratio',
              'sampler_seed': sampler_seed, 'offset_hours': offset, 'horizon_hours': horizon,
              'smoke_only': quick, 'outer_folds': [f['audit'] for f in outer],
              'inner_folds': [f['audit'] for f in inner], 'environment': environment(),
              'seconds': round(time.time() - t0, 1), 'variables': {variable: entry}}
    _save(folder / f'phase_shift_{variable}_offset{offset}_ratio{suffix(seed)}_summary.json', report)
    print(f'== base {variable} h{horizon} offset {offset} seed {sampler_seed}: n={entry["endpoints"]} '
          f'tree {entry["mae"]["tree"]:.6f} weather {entry["mae"]["weather_tree"]:.6f}', flush=True)


def run_perfold(arm, variable, horizon, quick=False, out=None):
    """One study per outer fold on that fold's own inner window, as v2_perfold_rest.run_forecast's 'perfold'
    design: same build, objective, space, refit and row schema (v2_horizon.tree_fit_score)."""
    from src.experiments.v2_horizon import build, climatology, folds, inner_fold
    from src.experiments.v2_perfold_rest import _summary, design_builder, space, to_params
    from src.tuning.protocol_v2 import PROTOCOL
    _check(horizon, 0, None)
    family = FAMILY[arm]
    started = time.time()
    set_seed(None)
    data = build(variable, horizon)
    outer = folds(data, horizon)
    inner = [inner_fold(data, f, horizon) for f in outer]
    if quick:
        for f in outer + inner:
            f['tr'] = f['tr'][-300:]
            f['va'] = f['va'][:40]
    clim = {id(f): climatology(data, f) for f in outer + inner}
    design = design_builder(family, arm, data)
    cache = {}

    def X(fold, lag):
        k = (id(fold), lag)
        if k not in cache:
            cache[k] = design(fold, lag)
        return cache[k]

    def score(fold, params, lag):
        return tree_fit_score(data, X(fold, lag), fold, params, clim[id(fold)])[0]

    tag = f'{family}_{variable}_{arm}_h{horizon}'
    base = Path(out) if out is not None else (OUT / 'smoke' if quick else OUT) / 'perfold'
    folder = base / tag
    folder.mkdir(parents=True, exist_ok=True)
    by_fold, sels = {}, {}
    for f, fi in zip(outer, inner):
        assert f['number'] == fi['number']

        def objective(trial, _f=fi):
            p, lag = space(family, trial)
            return ratio(score(_f, p, lag))
        sel = run_study(f'{tag}_perfold_fold{f["number"]}', objective, folder, quick)
        sels[str(f['number'])] = sel
        by_fold[f['number']] = to_params(family, sel['params'])
    d = pd.concat([score(f, *by_fold[f['number']]) for f in outer], ignore_index=True)
    path = folder / 'selected_predictions.csv.gz'
    d.to_csv(path, index=False)
    report = {'declared_in': DECLARED, 'design': 'perfold', 'family': family, 'variable': variable, 'arm': arm,
              'horizon_hours': horizon, 'protocol': PROTOCOL['version'], 'threads': int(os.environ['V2_THREADS']),
              'smoke_only': quick, 'outer_folds': [f['audit'] for f in outer],
              'inner_folds': [f['audit'] for f in inner], 'selection': sels,
              'objective': "fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only",
              'metrics': _summary(d), 'environment': environment(), 'seconds': round(time.time() - started, 1),
              'prediction_sha256': _sha(path)}
    _binning(report, arm)
    _save(folder / 'report.json', report)
    print(f'{tag} perfold: MAE {report["metrics"]["mae"]:.5f} in {report["seconds"]} s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['arm', 'base', 'perfold'])
    p.add_argument('--arm', choices=sorted(FAMILY))
    p.add_argument('--variable', required=True, choices=['temperature', 'oxygen'])
    p.add_argument('--horizon', type=int, required=True, choices=LEADS)
    p.add_argument('--offset', type=int, default=0, choices=range(6))
    p.add_argument('--seed', type=int, default=None, choices=[7, 123])
    p.add_argument('--out', default=None, help='perfold only: a folder other than the addendum folder (gate L-G7)')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.mode == 'arm':
        if a.arm not in ARMS:
            raise SystemExit('--arm must be one of ' + ', '.join(ARMS))
        run_arm(a.arm, a.variable, a.horizon, a.offset, a.seed, a.quick)
    elif a.mode == 'base':
        run_base(a.variable, a.horizon, a.offset, a.seed, a.quick)
    else:
        if a.arm is None or a.offset != 0 or a.seed is not None:
            raise SystemExit('perfold: --arm is required, on the reference grid at seed 42')
        run_perfold(a.arm, a.variable, a.horizon, a.quick, a.out)


if __name__ == '__main__':
    main()
