"""Addendum 82, parts D and H: forecasting at a held-out site, every arm with its own selection.

A site the network has never seen is forecast by a tree refitted without every probe at that site,
with the neighbour columns of the held probes rebuilt after removing every probe within 250 m. The
tree is selected by its own study per (fold, held site) on the inner window over the other sites
only, so no configuration sees the site it is scored on. The published held-site module
(`v2_held_station`) did this for the weather tree on temperature; this module does it for any of
three trees and on either variable:

    weather_tree  base features and the five Santa Teresa weather channels (part D, oxygen)
    tree          base features only, the plain tree (part H)
    bathy_tree    the weather tree plus the three seabed features of v2_bathy (part H)

The inner window is the published one (45 days ending 72 h before the outer fold). Where no other
site is eligible in it (at least 100 training and 10 scored issuances, the rule of
`v2_horizon.inner_fold`), the window moves back in 6 h steps until one is, with the same 72 h plus
lead gap before it for inner training. On the reference grid this moves only one cell, oxygen fold 1
with Baia Santa Teresa held out.

`allsite` selects the all-site tree per outer fold on the fold's own inner window, as the per-fold
design of addendum 47 does, for the grids and sampler seeds of the sweep, where no per-fold
reference exists yet. `refit` refits a cell at given configurations without a study (the
reproduction gates and the declared sensitivity).

Everything is written under reports/bay_twin/runs/v2_addendum82/<part>/; a smoke run writes under
smoke/ beside it and is never read. Nothing is written over a published file, and v2_held_station,
v2_horizon and v2_phase_shift are imported and left as they are.

Declared in config/bay_reserved_protocol_v2_addendum82.json (parts D and H).
Usage (from the repository root, V2_THREADS=2):
  python -m src.experiments.v2_held_site_nested windows
  python -m src.experiments.v2_held_site_nested cell --variable oxygen --arm weather_tree --site Baia_S.Teresa
  python -m src.experiments.v2_held_site_nested cell --variable temperature --arm weather_tree --site Mitili_Est --offset 3
  python -m src.experiments.v2_held_site_nested allsite --variable temperature --arm weather_tree --seed 7
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

from src.data.bay_forcing import weather
from src.experiments.bay_dynamics_tree import features
from src.experiments.v2_bathy import bathymetric_block
from src.experiments.v2_held_station import fit_predict, geometry, neighbour_masked
from src.experiments.v2_horizon import ROOT, build, climatology, folds, inner_fold, ratio, rows_from

DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
HORIZON = 6
ARMS = ('weather_tree', 'tree', 'bathy_tree')
PART = {'weather_tree': 'D', 'tree': 'H', 'bathy_tree': 'H'}
OUT = ROOT / 'reports/bay_twin/runs/v2_addendum82'
WINDOW_DAYS = 45
GAP_HOURS = 72
STEP_HOURS = 6
MIN_INNER_TRAIN = 100
MIN_INNER_SCORED = 10


def runtime():
    import lightgbm
    import optuna
    return {'host': platform.node(), 'python': sys.version.split()[0], 'numpy': np.__version__,
            'pandas': pd.__version__, 'optuna': optuna.__version__, 'lightgbm': lightgbm.__version__,
            'V2_THREADS': os.environ.get('V2_THREADS'), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))
    tmp.replace(path)


def build_data(variable, offset):
    """The reference grid is v2_horizon.build itself; other phases come from v2_phase_shift."""
    if offset == 0:
        return build(variable, HORIZON)
    from src.experiments.v2_phase_shift import build_shifted
    return build_shifted(variable, HORIZON, offset)


def design(data, fold, arm, held, distance, met):
    """The design array of `arm`, with the held probes' neighbour column masked within 250 m."""
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    if held is not None:
        base = base.copy()
        base[:, :, 1] = neighbour_masked(data, fold, held, distance)
    if arm == 'tree':
        return base
    n = len(data['names'])
    blocks = [base, np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))]
    if arm == 'bathy_tree':
        blocks.append(bathymetric_block(data))
    return np.concatenate(blocks, axis=-1)


def window(data, fold, end, horizon=HORIZON):
    """`v2_horizon.inner_fold` with the window's end as an argument (identical at the published end)."""
    t = data['times']
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    tr = np.flatnonzero(t + pd.Timedelta(hours=horizon + GAP_HOURS) < start)
    va = np.flatnonzero((t >= start) & (t + pd.Timedelta(hours=horizon) < end))
    eligible = (fold['eligible'] & (data['valid'][tr].sum(0) >= MIN_INNER_TRAIN)
                & (data['valid'][va].sum(0) >= MIN_INNER_SCORED))
    fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
    return dict(number=fold['number'], tr=tr, va=va, eligible=eligible, fallback=fallback,
                audit={'fold': fold['number'], 'horizon_hours': horizon,
                       'inner_selection_start': str(t[va].min()) if len(va) else None,
                       'inner_selection_end': str(end),
                       'last_inner_training_outcome': (str(t[tr].max() + pd.Timedelta(hours=horizon))
                                                       if len(tr) else None)})


def inner_window(data, fold, held, horizon=HORIZON):
    """The inner window for a held site: the published one, or moved back in 6 h steps until another
    site is eligible in it. Returns None if no window before the record's start qualifies."""
    t = data['times']
    first = t[fold['va']].min() - pd.Timedelta(hours=GAP_HOURS)
    end = first
    while end - pd.Timedelta(days=WINDOW_DAYS) > t[0]:
        inner = window(data, fold, end, horizon)
        others = inner['eligible'].copy()
        others[held] = False
        if others.any():
            inner['others'] = others
            inner['moved_hours'] = int((first - end) / pd.Timedelta(hours=1))
            inner['audit'].update(
                moved_hours=inner['moved_hours'],
                other_sites_probes=[data['names'][s] for s in np.flatnonzero(others)],
                other_scored_issuances={data['names'][s]: int(data['valid'][inner['va'], s].sum())
                                        for s in np.flatnonzero(others)},
                other_training_issuances={data['names'][s]: int(data['valid'][inner['tr'], s].sum())
                                          for s in np.flatnonzero(others)})
            return inner
        end -= pd.Timedelta(hours=STEP_HOURS)
    return None


def trim(fold):
    """The reduced folds of a smoke run (v2_horizon.run's convention, with a longer training tail so
    that a sparse sonde keeps the 200 training rows a fit needs)."""
    fold['tr'] = fold['tr'][-2000:]
    fold['va'] = fold['va'][:40]
    return fold


def part_dir(arm, quick):
    return OUT / PART[arm] / ('smoke' if quick else '')


def cell_dir(variable, arm, offset, seed, quick=False):
    return part_dir(arm, quick) / 'cells' / f'{variable}_{arm}_h{HORIZON}_o{offset}_s{seed}'


def allsite_dir(variable, arm, offset, seed, quick=False):
    return part_dir(arm, quick) / 'allsite' / f'{variable}_{arm}_h{HORIZON}_o{offset}_s{seed}'


def held_rows(data, fold, site, held, predicted, column='prediction'):
    mask = data['valid'][fold['va']] & fold['eligible'][None, :]
    rows = []
    for s in held:
        for j in np.flatnonzero(mask[:, s]):
            i = fold['va'][j]
            rows.append({'fold': fold['number'], 'site': site, 'sensor': data['names'][s],
                         'time': str(data['times'][i]),
                         'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                         'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                         column: float(predicted[j, s] * data['scale'] + data['center'])})
    return rows


def set_seed(seed):
    """A second sampler seed changes nothing but the TPE stream (the convention of v2_phase_shift)."""
    from src.tuning import protocol_v2
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    return protocol_v2


def run_cell(variable, arm, site, offset=0, seed=42, quick=False):
    """One held site: a study per outer fold in which the site is eligible, then the outer refit."""
    from src.experiments.v2_forecast import tree_space
    protocol = set_seed(seed)
    started = time.time()
    data = build_data(variable, offset)
    outer = folds(data, HORIZON)
    if quick:
        outer = [trim(f) for f in outer]
    met = weather(data['times']).to_numpy()
    sites, distance = geometry(data['names'])
    folder = cell_dir(variable, arm, offset, seed, quick)
    folder.mkdir(parents=True, exist_ok=True)
    rows, record = [], {}
    for fold in outer:
        held = np.flatnonzero((sites == site) & fold['eligible'])
        if not len(held):
            continue
        inner = inner_window(data, fold, held)
        if inner is None:
            record[str(fold['number'])] = {'not_selectable': 'no other site eligible in any inner window'}
            continue
        if quick:
            inner = trim(inner)
        design_inner = design(data, inner, arm, held, distance, met)
        clim_inner = climatology(data, inner)
        scored = dict(inner, eligible=inner['others'])

        def objective(trial):
            p = tree_space(trial)
            grid = fit_predict(data, design_inner, scored, p, exclude=held)
            if grid is None:
                raise ValueError('too few inner training rows')
            return ratio(rows_from(data, scored, grid, clim_inner))
        tag = f'held_site_{variable}_{arm}_h{HORIZON}_o{offset}_s{seed}_{site}_f{fold["number"]}'
        sel = protocol.run_study(tag, objective, folder, quick)
        params = {**sel['params'], 'subsample_freq': 1}
        predicted = fit_predict(data, design(data, fold, arm, held, distance, met), fold, params, exclude=held)
        if predicted is None:
            record[str(fold['number'])] = {'selection': sel, 'inner': inner['audit'],
                                           'not_scored': 'too few outer training rows'}
            continue
        rows += held_rows(data, fold, site, held, predicted)
        record[str(fold['number'])] = {'selection': sel, 'inner': inner['audit'],
                                       'held_probes': [data['names'][s] for s in held]}
    frame = pd.DataFrame(rows)
    path = folder / f'{site}.csv.gz'
    frame.to_csv(path, index=False)
    save(folder / f'{site}_record.json', {
        'declared_in': DECLARED, 'variable': variable, 'arm': arm, 'site': site, 'horizon_hours': HORIZON,
        'offset_hours': offset, 'sampler_seed': seed, 'smoke_only': quick,
        'objective': 'fold-macro forecast MAE / persistence MAE on the inner window over the other sites',
        'folds': record, 'endpoints': int(len(frame)),
        'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'runtime': runtime(), 'seconds': round(time.time() - started, 1)})
    print(f'{variable} {arm} {site} o{offset} s{seed}: {len(frame)} endpoints', flush=True)


def run_allsite(variable, arm, offset=0, seed=42, quick=False):
    """The all-site tree selected per outer fold on the fold's own inner window (addendum 47's design)."""
    from src.experiments.v2_forecast import tree_space
    from src.experiments.v2_horizon import tree_fit_score
    protocol = set_seed(seed)
    started = time.time()
    data = build_data(variable, offset)
    outer = folds(data, HORIZON)
    if quick:
        outer = [trim(f) for f in outer]
    inner = [inner_fold(data, f, HORIZON) for f in outer]
    if quick:
        inner = [trim(f) for f in inner]
    met = weather(data['times']).to_numpy()
    folder = allsite_dir(variable, arm, offset, seed, quick)
    folder.mkdir(parents=True, exist_ok=True)
    frames, record = [], {}
    for f, fi in zip(outer, inner):
        d_inner, c_inner = design(data, fi, arm, None, None, met), climatology(data, fi)

        def objective(trial):
            return ratio(tree_fit_score(data, d_inner, fi, tree_space(trial), c_inner)[0])
        sel = protocol.run_study(f'allsite_{variable}_{arm}_h{HORIZON}_o{offset}_s{seed}_fold{f["number"]}',
                                 objective, folder, quick)
        params = {**sel['params'], 'subsample_freq': 1}
        frames.append(tree_fit_score(data, design(data, f, arm, None, None, met), f, params,
                                     climatology(data, f))[0])
        record[str(f['number'])] = {'selection': sel, 'inner': fi['audit'], 'outer': f['audit']}
    frame = pd.concat(frames, ignore_index=True)
    path = folder / 'predictions.csv.gz'
    frame.to_csv(path, index=False)
    save(folder / 'record.json', {
        'declared_in': DECLARED, 'variable': variable, 'arm': arm, 'horizon_hours': HORIZON,
        'offset_hours': offset, 'sampler_seed': seed, 'smoke_only': quick, 'folds': record,
        'endpoints': int(len(frame)), 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'runtime': runtime(), 'seconds': round(time.time() - started, 1)})
    print(f'allsite {variable} {arm} o{offset} s{seed}: {len(frame)} endpoints', flush=True)


def refit(variable, arm, site, params_by_fold, offset=0, column='prediction', windows_only=False):
    """Refit one held site at given configurations ({fold number: params}), no study."""
    data = build_data(variable, offset)
    outer = folds(data, HORIZON)
    met = weather(data['times']).to_numpy()
    sites, distance = geometry(data['names'])
    rows = []
    for fold in outer:
        held = np.flatnonzero((sites == site) & fold['eligible'])
        if not len(held) or fold['number'] not in params_by_fold:
            continue
        params = {**params_by_fold[fold['number']], 'subsample_freq': 1}
        predicted = fit_predict(data, design(data, fold, arm, held, distance, met), fold, params, exclude=held)
        rows += held_rows(data, fold, site, held, predicted, column)
    return pd.DataFrame(rows)


def windows():
    """D-G3: the rule against inner_fold in every (variable, fold, held site) cell of the reference grid."""
    out = {}
    for variable in ('temperature', 'oxygen'):
        data = build_data(variable, 0)
        sites, distance = geometry(data['names'])
        for fold in folds(data, HORIZON):
            published = inner_fold(data, fold, HORIZON)
            for site in pd.unique(sites[fold['eligible']]):
                held = np.flatnonzero((sites == site) & fold['eligible'])
                inner = inner_window(data, fold, held)
                others = published['eligible'].copy()
                others[held] = False
                same = (inner is not None and inner['moved_hours'] == 0
                        and np.array_equal(inner['tr'], published['tr'])
                        and np.array_equal(inner['va'], published['va'])
                        and np.array_equal(inner['eligible'], published['eligible'])
                        and np.array_equal(inner['fallback'], published['fallback']))
                out[f'{variable}|{fold["number"]}|{site}'] = {
                    'published_window_selectable': bool(others.any()),
                    'identical_to_published_inner_fold': bool(same),
                    'moved_hours': None if inner is None else inner['moved_hours'],
                    'audit': None if inner is None else inner['audit']}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['cell', 'allsite', 'windows'])
    p.add_argument('--variable', choices=['temperature', 'oxygen'])
    p.add_argument('--arm', choices=ARMS, default='weather_tree')
    p.add_argument('--site')
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.command == 'cell':
        run_cell(a.variable, a.arm, a.site, a.offset, a.seed, a.quick)
    elif a.command == 'allsite':
        run_allsite(a.variable, a.arm, a.offset, a.seed, a.quick)
    else:
        print(json.dumps(windows(), indent=1))


if __name__ == '__main__':
    main()
