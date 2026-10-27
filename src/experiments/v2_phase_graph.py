"""Addendum 82, part E: the graph operator, constraint and saturation-prior arms on shifted issuance grids.

Only the oxygen operator at six hours had been moved across the six issuance grids (addendum 18); the
temperature operator, both constraint contrasts and the 24 h oxygen operator had been re-tuned under
further sampler seeds only, and the saturation prior under neither. For one (variable, arm, horizon,
phase, seed) this module tunes the graph arm with its own 100-trial study on the reference objective,
exactly as v2_horizon.run does for a neural arm (v2_horizon.neural_forecast trained at initialisation
seed 42 inside the objective, v2_horizon.neural_space), on the grid v2_phase_shift.build_shifted
builds, and scores the mean of three networks at initialisation seeds 42, 7 and 123 on both
development folds, as published. Pairing, gates and contrasts are made by
scripts/v2_phase_graph_contrasts.py.

Two gate modes run before any study:
  --refit-published  E-G1 and E-G2: the published selection refitted here at offset 0 reproduces the
                     published rows (1e-6; other columns 1e-9), and the objective at that selection
                     equals the published best value (1e-6)
  --reproduce        E-G3: the full study of the published arm at offset 0 and seed 42, compared trial
                     by trial with the published study's SQLite file (--published-db)

Declared in config/bay_reserved_protocol_v2_addendum82.json (part E). Development folds only: the
grid ends 2025-12-31T23:00 and no 2026 value is loaded.
Usage (from the repository root, V2_THREADS=2, CUDA_VISIBLE_DEVICES=1):
  python -m src.experiments.v2_phase_graph --variable temperature --arm learned_source --horizon 6 --offset 3
  python -m src.experiments.v2_phase_graph --variable oxygen --arm saturation_source --horizon 6 --offset 0 --seed 7
  python -m src.experiments.v2_phase_graph --variable oxygen --arm learned_source --horizon 24 --refit-published
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import platform
import socket
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.v2_horizon import ROOT, climatology, folds, inner_fold, neural_forecast, ratio
from src.experiments.v2_phase_shift import build_shifted, macro
from src.tuning.protocol_v2 import run_study

DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
OUT = Path(os.environ.get('V2_ADD82_OUT', str(ROOT / 'reports/bay_twin/runs/v2_addendum82/E')))
KEY = ['fold', 'sensor', 'time']
ARMS = ('learned_source', 'conservative_exchange', 'saturation_exchange', 'saturation_source')


def suffix(seed):
    return '' if seed is None else f'_seed{seed}'


def set_seed(seed):
    """run_study reads the sampler seed at call time, so the bound tuning module is left as it is."""
    from src.tuning import protocol_v2
    protocol_v2.PROTOCOL['sampler_seed'] = 42 if seed is None else int(seed)
    return protocol_v2.PROTOCOL['sampler_seed']


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


def environment(device):
    import optuna
    import torch
    return {'host': socket.gethostname(), 'python': platform.python_version(), 'optuna': optuna.__version__,
            'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'device': device,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'V2_THREADS': os.environ.get('V2_THREADS'), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}


def setup(variable, arm, horizon, offset, quick=False):
    """What v2_horizon.run builds for a neural arm, on the grid shifted by `offset` hours."""
    import torch
    from src.experiments.v2_forecast import NEURAL_ARMS, OXYGEN_ONLY, THREADS, graphs
    if arm in OXYGEN_ONLY and variable != 'oxygen':
        raise ValueError('saturation arms are oxygen-only')
    torch.set_num_threads(THREADS)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = build_shifted(variable, horizon, offset)
    graph, exchange, relaxation = NEURAL_ARMS[arm]
    data['adjacency'] = graphs(data)[graph]
    outer = folds(data, horizon)
    if quick:
        for f in outer:
            f['tr'] = f['tr'][-300:]
            f['va'] = f['va'][:40]
    inner = [inner_fold(data, f, horizon) for f in outer]
    if quick:
        for f in inner:
            f['tr'] = f['tr'][-300:]
            f['va'] = f['va'][:40]
    clim = {id(f): climatology(data, f) for f in outer + inner}
    return data, outer, inner, clim, exchange, relaxation, device


def score_networks(data, outer, clim, params, exchange, relaxation, device, seeds, quick):
    """Rows of every network (initialisation seed) on both outer folds, with the seed column."""
    frames = [neural_forecast(data, f, params, exchange, relaxation, s, device, clim[id(f)], quick).assign(seed=s)
              for s in seeds for f in outer]
    return pd.concat(frames, ignore_index=True)


def wide(d, arm):
    """One row per endpoint: each network's prediction and their mean (the published arm)."""
    base = d.groupby(KEY, as_index=False).agg(truth=('truth', 'first'), persistence=('persistence', 'first'),
                                              baseline=('baseline', 'first'), clim_hour=('clim_hour', 'first'),
                                              **{arm: ('prediction', 'mean')})
    for s, g in d.groupby('seed'):
        base = base.merge(g[KEY + ['prediction']].rename(columns={'prediction': f'{arm}_init{int(s)}'}),
                          on=KEY, how='left', validate='one_to_one')
    return base


def run_cell(variable, arm, horizon, offset, seed=None, quick=False):
    from src.experiments.v2_forecast import SEEDS, neural_space
    t0 = time.time()
    folder = (OUT / 'smoke' if quick else OUT) / 'arms'
    sampler_seed = set_seed(seed)
    data, outer, inner, clim, exchange, relaxation, device = setup(variable, arm, horizon, offset, quick)

    def objective(trial):
        p = neural_space(trial)
        return float(np.mean([ratio(neural_forecast(data, f, p, exchange, relaxation, 42, device, clim[id(f)], quick))
                              for f in inner]))

    name = f'{variable}_{arm}_h{horizon}_shift{offset}_ratio{suffix(seed)}'
    selection = run_study(name, objective, folder / 'studies', quick)
    seeds = [42] if quick else list(SEEDS)
    d = score_networks(data, outer, clim, selection['params'], exchange, relaxation, device, seeds, quick)
    frame = wide(d, arm)
    path = folder / f'phase_graph_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}.csv.gz'
    frame.to_csv(path, index=False)
    report = {'declared_in': DECLARED, 'part': 'E', 'mode': 'cell', 'variable': variable, 'arm': arm,
              'horizon_hours': horizon, 'offset_hours': offset, 'sampler_seed': sampler_seed,
              'objective': 'ratio: mean over the inner folds of fold-macro MAE / persistence MAE, network at '
                           'initialisation seed 42', 'initialisation_seeds': seeds, 'smoke_only': quick,
              'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
              'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
              'selection': selection, 'mae': macro(frame, arm),
              'mae_by_network': {str(s): macro(frame, f'{arm}_init{s}') for s in seeds},
              'environment': environment(device), 'seconds': round(time.time() - t0, 1),
              'prediction_file': path.name, 'prediction_sha256': _sha(path)}
    _save(folder / f'phase_graph_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}_summary.json', report)
    print(f'== {name}: n={report["endpoints"]} MAE {report["mae"]:.6f} in {report["seconds"]} s', flush=True)


def refit_published(variable, arm, horizon):
    """E-G1 and E-G2 at offset 0 for the published selection of this arm."""
    from src.experiments.v2_forecast import SEEDS
    from src.experiments.v2_perfold_neural import published_paths, published_rows
    t0 = time.time()
    set_seed(None)
    data, outer, inner, clim, exchange, relaxation, device = setup(variable, arm, horizon, 0)
    pub_folder, record = published_paths(variable, arm, horizon)
    sel = json.loads(record.read_text(encoding='utf-8'))
    params = sel['params']
    value = float(np.mean([ratio(neural_forecast(data, f, params, exchange, relaxation, 42, device, clim[id(f)]))
                           for f in inner]))
    d = score_networks(data, outer, clim, params, exchange, relaxation, device, list(SEEDS), False)
    # The selection's own run (v2_forecast at 6 h, whose 'baseline' column is defined differently, so only
    # truth and persistence are shared with this build), and the horizon run where it holds this arm too
    # (every column shared).
    sources = [(pub_folder, ('truth', 'persistence'))]
    horizon_folder = ROOT / 'reports/bay_twin/runs/20260920T083322Z_v2_horizon/results' / \
        f'v2_horizon_{variable}_{arm}_h{horizon}'
    if horizon_folder != pub_folder and (horizon_folder / 'selected_predictions.csv.gz').exists():
        sources.append((horizon_folder, ('truth', 'persistence', 'baseline', 'clim_hour')))
    checks = []
    for folder_pub, columns in sources:
        pub = published_rows(folder_pub)
        j = pub.merge(d, on=KEY + ['seed'], suffixes=('_pub', ''), how='outer', indicator=True)
        same_rows = bool((j['_merge'] == 'both').all()) and len(pub) == len(d)
        both = j[j['_merge'] == 'both']
        diff = float(np.abs(both.prediction - both.prediction_pub).max())
        other = {c: float(np.abs(both[c] - both[c + '_pub']).max()) for c in columns}
        checks.append({'published_predictions': (folder_pub / 'selected_predictions.csv.gz').relative_to(ROOT).as_posix(),
                       'published_predictions_sha256': _sha(folder_pub / 'selected_predictions.csv.gz'),
                       'rows_identical': same_rows, 'rows_here': int(len(d)), 'rows_published': int(len(pub)),
                       'max_abs_prediction_diff': diff, 'other_columns_max_abs_diff': other,
                       'pass': bool(same_rows and diff <= 1e-6 and max(other.values()) <= 1e-9)})
    folder = OUT / 'gates'
    path = folder / f'refit_{variable}_{arm}_h{horizon}.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False)
    diff = max(c['max_abs_prediction_diff'] for c in checks)
    report = {'declared_in': DECLARED, 'part': 'E', 'gate': 'E-G1 and E-G2', 'variable': variable, 'arm': arm,
              'horizon_hours': horizon, 'offset_hours': 0,
              'published_selection': record.relative_to(ROOT).as_posix(), 'published_selection_sha256': _sha(record),
              'params': params,
              'E-G1': {'sources': checks, 'max_abs_prediction_diff': diff, 'pass': all(c['pass'] for c in checks)},
              'E-G2': {'objective_here': value, 'published_best_value': sel['best_value'],
                       'abs_diff': abs(value - sel['best_value']),
                       'pass': bool(abs(value - sel['best_value']) <= 1e-6)},
              'environment': environment(device), 'seconds': round(time.time() - t0, 1),
              'prediction_sha256': _sha(path)}
    _save(folder / f'refit_{variable}_{arm}_h{horizon}.json', report)
    print(f'== refit {variable} {arm} h{horizon}: E-G1 {report["E-G1"]["pass"]} ({diff:.2e}), '
          f'E-G2 {report["E-G2"]["pass"]} ({report["E-G2"]["abs_diff"]:.2e})', flush=True)


def reproduce_study(variable, arm, horizon, published_db):
    """E-G3: the full study at offset 0 and seed 42, trial for trial against the published SQLite file."""
    from src.experiments.v2_forecast import neural_space
    t0 = time.time()
    set_seed(None)
    data, outer, inner, clim, exchange, relaxation, device = setup(variable, arm, horizon, 0)

    def objective(trial):
        p = neural_space(trial)
        return float(np.mean([ratio(neural_forecast(data, f, p, exchange, relaxation, 42, device, clim[id(f)]))
                              for f in inner]))

    folder = OUT / 'gates' / 'studies'
    name = f'{variable}_{arm}_h{horizon}_shift0_ratio_reproduce'
    selection = run_study(name, objective, folder)

    def trials(db):
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        try:
            rows = con.execute("select t.number, v.value from trials t join trial_values v on t.trial_id = v.trial_id "
                               "where t.state = 'COMPLETE' order by t.number").fetchall()
            params = con.execute("select t.number, p.param_name, p.param_value from trials t join trial_params p "
                                 "on t.trial_id = p.trial_id where t.state = 'COMPLETE'").fetchall()
            version = con.execute('select library_version from version_info').fetchone()[0]
        finally:
            con.close()
        by = {}
        for n, k, v in params:
            by.setdefault(n, {})[k] = v
        return {n: (v, by.get(n, {})) for n, v in rows}, version

    here, v_here = trials(folder / f'{name}.sqlite3')
    there, v_there = trials(Path(published_db))
    common = sorted(set(here) & set(there))
    value_diff = max(abs(here[n][0] - there[n][0]) for n in common) if common else float('inf')
    params_identical = all(here[n][1] == there[n][1] for n in common)
    best_here = min(here, key=lambda n: here[n][0])
    best_there = min(there, key=lambda n: there[n][0])
    report = {'declared_in': DECLARED, 'part': 'E', 'gate': 'E-G3', 'variable': variable, 'arm': arm,
              'horizon_hours': horizon, 'offset_hours': 0, 'sampler_seed': 42,
              'published_db': str(published_db), 'published_db_sha256': _sha(published_db),
              'optuna_here': v_here, 'optuna_published': v_there,
              'trials_here': len(here), 'trials_published': len(there), 'trials_compared': len(common),
              'max_abs_value_diff': value_diff, 'params_identical': params_identical,
              'best_trial_here': best_here, 'best_trial_published': best_there, 'selection': selection,
              'pass': bool(len(common) == len(here) == len(there) and value_diff <= 1e-6 and params_identical
                           and best_here == best_there),
              'environment': environment(device), 'seconds': round(time.time() - t0, 1)}
    _save(OUT / 'gates' / f'reproduce_{variable}_{arm}_h{horizon}.json', report)
    print(f'== E-G3 {name}: {report["pass"]} (max value diff {value_diff:.2e}, best {best_here} vs {best_there})',
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variable', required=True, choices=['temperature', 'oxygen'])
    p.add_argument('--arm', required=True, choices=ARMS)
    p.add_argument('--horizon', type=int, required=True, choices=[6, 24])
    p.add_argument('--offset', type=int, default=0, choices=range(6))
    p.add_argument('--seed', type=int, default=None, choices=[7, 123])
    p.add_argument('--quick', action='store_true')
    p.add_argument('--refit-published', action='store_true')
    p.add_argument('--reproduce', action='store_true')
    p.add_argument('--published-db')
    a = p.parse_args()
    if a.seed is not None and a.offset != 0:
        raise SystemExit('the further sampler seeds are run on the reference grid only')
    if a.refit_published:
        refit_published(a.variable, a.arm, a.horizon)
    elif a.reproduce:
        reproduce_study(a.variable, a.arm, a.horizon, a.published_db)
    else:
        run_cell(a.variable, a.arm, a.horizon, a.offset, a.seed, a.quick)


if __name__ == '__main__':
    main()
