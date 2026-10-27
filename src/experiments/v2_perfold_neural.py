"""Addendum 49: the graph-dynamics arms re-selected per outer fold, at every horizon the thesis quotes.

The published graph arms (`v2_forecast` at 6 h, `v2_horizon` at 1 to 48 h) choose one configuration
per arm on the mean over both inner windows and apply it to both outer folds, and fold 1's inner
window lies inside fold 0's scored quarter (addendum 47 counted the overlap). This module repeats
addendum 47's three designs for them, and for the plain tree that the operator contrasts are set
against, on the compute host where the published studies ran:

  published    the published selection refitted here (gate G1: it reproduces the published rows)
  shared       the published design re-run in this environment (the control)
  perfold      one study per (arm, outer fold) whose objective sees that fold's own inner window only

Every graph cell runs through `v2_horizon.neural_forecast` (at 6 h its rows equal the forecast-mode
rows of `v2_forecast.fit`, which gate G1 checks), trains its study objective at initialisation seed
42 and reports the mean of the networks at seeds 42, 7 and 123, as published.

Declared in config/bay_reserved_protocol_v2_addendum49.json before anything below was run.

Usage (from the repository root, V2_THREADS=2; the published runs are read from
V2PN_PUBLISHED_ROOT, default reports/bay_twin/runs, laid out <run>/results/<cell>/):
  python -m src.experiments.v2_perfold_neural graph --variable oxygen --arm learned_source --horizon 6
  python -m src.experiments.v2_perfold_neural tree --variable oxygen --horizon 12
A design whose report.json exists is skipped; a study on disk is resumed, not restarted.
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
OUT = RUNS / 'v2_perfold_neural'
PUBLISHED_ROOT = Path(os.environ.get('V2PN_PUBLISHED_ROOT', str(RUNS)))
FORECAST_RUN = '20260919T144822Z_v2_forecast'
HORIZON_RUN = {1: '20260920T133747Z_v2_horizon', 3: '20260920T133747Z_v2_horizon',
               12: '20260920T133747Z_v2_horizon', 36: '20260920T133747Z_v2_horizon',
               6: '20260920T083322Z_v2_horizon', 24: '20260920T083322Z_v2_horizon',
               48: '20260920T083322Z_v2_horizon'}
HORIZONS = (1, 3, 6, 12, 24, 36, 48)
GRAPH_SIX = {'temperature': ('learned_source', 'conservative_exchange', 'coast_exchange',
                             'coast_breakwater_exchange', 'coast_depth_exchange'),
             'oxygen': ('learned_source', 'conservative_exchange', 'coast_exchange',
                        'coast_breakwater_exchange', 'coast_depth_exchange',
                        'saturation_source', 'saturation_exchange')}
GRAPH_OTHER = {'temperature': ('learned_source', 'conservative_exchange'),
               'oxygen': ('learned_source', 'conservative_exchange', 'saturation_exchange')}
DESIGNS = ('published', 'shared', 'perfold')
KEY = ['fold', 'sensor', 'time']
DECLARED = 'config/bay_reserved_protocol_v2_addendum49.json'


def cells():
    """Every (variable, arm, horizon) graph cell of the addendum, in the order it is run."""
    out = [(v, a, 6) for v in ('temperature', 'oxygen') for a in GRAPH_SIX[v]]
    out += [(v, a, h) for h in (24, 1, 3, 12, 36, 48) for v in ('temperature', 'oxygen') for a in GRAPH_OTHER[v]]
    return out


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def published_paths(variable, arm, horizon):
    """(folder, selection record, whether the rows carry a 'mode' column) of the published cell."""
    if arm != 'tree' and horizon == 6:
        folder = PUBLISHED_ROOT / FORECAST_RUN / 'results' / f'v2_forecast_{variable}_{arm}_nested'
        return folder, folder / f'{variable}_{arm}_nested_selection.json'
    folder = PUBLISHED_ROOT / HORIZON_RUN[horizon] / 'results' / f'v2_horizon_{variable}_{arm}_h{horizon}'
    return folder, folder / f'{variable}_{arm}_h{horizon}_selection.json'


def published_rows(folder):
    d = pd.read_csv(folder / 'selected_predictions.csv.gz')
    if 'mode' in d:
        d = d[d['mode'] == 'forecast'].drop(columns='mode')
    return d


def run_study_complete(name, objective, folder, attempts=4):
    """run_study, resumed when failed trials left it short of the protocol's hundred."""
    from src.tuning.protocol_v2 import run_study
    for k in range(attempts):
        try:
            return run_study(name, objective, folder)
        except RuntimeError as e:
            if 'completed trials' not in str(e) or k == attempts - 1:
                raise
            print(f'  {name}: {e}; resuming', flush=True)


def seed_mean(d):
    return d.groupby(KEY, as_index=False).agg(
        truth=('truth', 'first'), persistence=('persistence', 'first'), baseline=('baseline', 'first'),
        clim_hour=('clim_hour', 'first'), prediction=('prediction', 'mean'))


def _macro(d, col='prediction'):
    e = (d.truth - d[col]).abs()
    return float(d.assign(e=e).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def _summary(d):
    out = {'endpoints': int(len(d)), 'mae': _macro(d), 'persistence_mae': _macro(d, 'persistence')}
    for k, g in d.groupby('fold'):
        out[f'fold{k}'] = {'endpoints': int(len(g)), 'mae': _macro(g), 'persistence_mae': _macro(g, 'persistence')}
    return out


# ---------------------------------------------------------------- graph arms

def graph_setup(variable, arm, horizon):
    """Exactly what v2_horizon.run builds for a neural arm, returned instead of run."""
    import torch
    from src.experiments.v2_horizon import build, folds, inner_fold, climatology
    from src.experiments.v2_forecast import NEURAL_ARMS, THREADS, graphs
    torch.set_num_threads(THREADS)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = build(variable, horizon)
    outer = folds(data, horizon)
    inner = [inner_fold(data, f, horizon) for f in outer]
    clim = {id(f): climatology(data, f) for f in outer + inner}
    graph, exchange, relaxation = NEURAL_ARMS[arm]
    data['adjacency'] = graphs(data)[graph]
    return data, outer, inner, clim, exchange, relaxation, device


def run_graph(variable, arm, horizon, designs, quick=False):
    from src.experiments.v2_forecast import SEEDS, neural_space
    from src.experiments.v2_horizon import neural_forecast, ratio
    from src.tuning.protocol_v2 import PROTOCOL
    started = time.time()
    data, outer, inner, clim, exchange, relaxation, device = graph_setup(variable, arm, horizon)
    tag = f'{variable}_{arm}_h{horizon}'
    seeds = [42] if quick else list(SEEDS)

    def nf(fold, params, seed):
        return neural_forecast(data, fold, params, exchange, relaxation, seed, device, clim[id(fold)], quick)

    def score(params_by_fold):
        return pd.concat([nf(f, params_by_fold[f['number']], s).assign(seed=s) for s in seeds for f in outer],
                         ignore_index=True)

    for design in designs:
        folder = OUT / ('graph_smoke' if quick else 'graph') / design / tag
        if (folder / 'report.json').exists():
            print(f'{tag} {design}: already done', flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        report = {'declared_in': DECLARED, 'design': design, 'variable': variable, 'arm': arm,
                  'horizon_hours': horizon, 'protocol': PROTOCOL['version'], 'device': device,
                  'threads': int(os.environ['V2_THREADS']), 'seeds': seeds, 'smoke_only': quick,
                  'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner]}
        if design == 'published':
            pub_folder, record = published_paths(variable, arm, horizon)
            sel = json.loads(record.read_text())
            params = sel['params']
            by_fold = {f['number']: params for f in outer}
            # G2: the shared objective at the published selection gives the published best value.
            value = float(np.mean([ratio(nf(f, params, 42)) for f in inner]))
            report.update(selection_source=str(record), selection_sha256=_sha(record), params=params,
                          published_best_value=sel['best_value'], published_completed_trials=sel['completed_trials'],
                          objective_here=value, gate_G2_abs_diff=abs(value - sel['best_value']),
                          gate_G2_pass=bool(abs(value - sel['best_value']) <= 1e-6))
        elif design == 'shared':
            def objective(trial):
                p = neural_space(trial)
                return float(np.mean([ratio(nf(f, p, 42)) for f in inner]))
            sel = (run_study_complete if not quick else _quick_study)(f'{tag}_shared', objective, folder)
            by_fold = {f['number']: sel['params'] for f in outer}
            report.update(selection=sel, objective='mean over both inner windows of fold-macro forecast MAE / persistence MAE')
        elif design == 'perfold':
            by_fold, sels = {}, {}
            for f, fi in zip(outer, inner):
                assert f['number'] == fi['number']

                def objective(trial, _f=fi):
                    return ratio(nf(_f, neural_space(trial), 42))
                sel = (run_study_complete if not quick else _quick_study)(f'{tag}_perfold_fold{f["number"]}', objective, folder)
                sels[str(f['number'])] = sel
                by_fold[f['number']] = sel['params']
            report.update(selection=sels, objective="fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only")
        else:
            raise ValueError(design)
        d = score(by_fold)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)
        if design == 'published':
            pub = published_rows(pub_folder)
            j = pub.merge(d, on=KEY + ['seed'], suffixes=('_pub', ''), how='outer', indicator=True)
            same_rows = bool((j['_merge'] == 'both').all()) and len(pub) == len(d)
            both = j[j['_merge'] == 'both']
            diff = float(np.abs(both.prediction - both.prediction_pub).max())
            other = {c: float(np.abs(both[c] - both[c + '_pub']).max()) for c in ('truth', 'persistence')}
            report.update(published_predictions=str(pub_folder / 'selected_predictions.csv.gz'),
                          published_predictions_sha256=_sha(pub_folder / 'selected_predictions.csv.gz'),
                          gate_G1_rows_identical=same_rows, gate_G1_max_abs_prediction_diff=diff,
                          gate_G1_other_columns_max_abs_diff=other,
                          gate_G1_pass=bool(same_rows and diff <= 1e-6 and max(other.values()) <= 1e-9))
        report.update(params_by_fold={str(k): v for k, v in by_fold.items()},
                      metrics=_summary(seed_mean(d)), seconds=round(time.time() - t0, 1),
                      prediction_sha256=_sha(path))
        _json(folder / 'report.json', report)
        print(f'{tag} {design}: MAE {report["metrics"]["mae"]:.5f} in {report["seconds"]} s'
              + (f'  G1 {report["gate_G1_pass"]} ({report["gate_G1_max_abs_prediction_diff"]:.2e})'
                 f'  G2 {report["gate_G2_pass"]} ({report["gate_G2_abs_diff"]:.2e})' if design == 'published' else ''),
              flush=True)
    print(f'{tag} done in {time.time() - started:.0f} s', flush=True)


def _quick_study(name, objective, folder):
    """Smoke only: two trials through the protocol's own function, never a reported selection."""
    from src.tuning.protocol_v2 import run_study
    return run_study(name, objective, folder, quick=True)


# ---------------------------------------------------------------- the plain tree reference

def run_tree(variable, horizon, designs, quick=False):
    """The plain residual tree under the same three designs, on the same host as the graph arms."""
    from src.experiments import v2_perfold as a47
    from src.experiments.v2_forecast import tree_space
    from src.experiments.v2_horizon import ratio, tree_fit_score
    from src.tuning.protocol_v2 import PROTOCOL
    data, outer, inner, clim, cache = a47.forecast_setup(variable, 'tree', horizon)
    tag = f'{variable}_tree_h{horizon}'
    for design in designs:
        folder = OUT / ('tree_smoke' if quick else 'tree') / design / tag
        if (folder / 'report.json').exists():
            print(f'{tag} {design}: already done', flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        report = {'declared_in': DECLARED, 'design': design, 'variable': variable, 'arm': 'tree',
                  'horizon_hours': horizon, 'protocol': PROTOCOL['version'], 'smoke_only': quick,
                  'threads': int(os.environ['V2_THREADS']),
                  'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner]}
        study = run_study_complete if not quick else _quick_study
        if design == 'published':
            pub_folder, record = published_paths(variable, 'tree', horizon)
            sel = json.loads(record.read_text())
            params = {**sel['params'], 'subsample_freq': 1}
            by_fold = {f['number']: params for f in outer}
            value = float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, params, clim[id(f)])[0]) for f in inner]))
            report.update(selection_source=str(record), selection_sha256=_sha(record), params=params,
                          published_best_value=sel['best_value'], objective_here=value,
                          gate_G2_abs_diff=abs(value - sel['best_value']),
                          gate_G2_pass=bool(abs(value - sel['best_value']) <= 1e-9))
        elif design == 'shared':
            def objective(trial):
                p = tree_space(trial)
                return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))
            sel = study(f'{tag}_shared', objective, folder)
            by_fold = {f['number']: {**sel['params'], 'subsample_freq': 1} for f in outer}
            report.update(selection=sel)
        elif design == 'perfold':
            by_fold, sels = {}, {}
            for f, fi in zip(outer, inner):
                def objective(trial, _f=fi):
                    return ratio(tree_fit_score(data, cache[id(_f)], _f, tree_space(trial), clim[id(_f)])[0])
                sel = study(f'{tag}_perfold_fold{f["number"]}', objective, folder)
                sels[str(f['number'])] = sel
                by_fold[f['number']] = {**sel['params'], 'subsample_freq': 1}
            report.update(selection=sels)
        else:
            raise ValueError(design)
        d = a47.score_folds(data, outer, by_fold, clim, cache)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)
        if design == 'published':
            pub = published_rows(pub_folder)
            j = pub.merge(d, on=KEY, suffixes=('_pub', ''), validate='one_to_one')
            same = len(j) == len(pub) == len(d)
            diff = float(np.abs(j.prediction - j.prediction_pub).max())
            report.update(published_predictions_sha256=_sha(pub_folder / 'selected_predictions.csv.gz'),
                          gate_G6_rows_identical=bool(same), gate_G6_max_abs_prediction_diff=diff,
                          gate_G6_pass=bool(same and diff <= 1e-9))
        report.update(params_by_fold={str(k): v for k, v in by_fold.items()}, metrics=_summary(d),
                      seconds=round(time.time() - t0, 1), prediction_sha256=_sha(path))
        _json(folder / 'report.json', report)
        print(f'{tag} {design}: MAE {report["metrics"]["mae"]:.5f} in {report["seconds"]} s'
              + (f'  G6 {report["gate_G6_pass"]}  G2 {report["gate_G2_pass"]}' if design == 'published' else ''),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('graph')
    g.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    g.add_argument('--arm', required=True)
    g.add_argument('--horizon', type=int, choices=HORIZONS, required=True)
    g.add_argument('--design', nargs='+', default=list(DESIGNS), choices=DESIGNS)
    g.add_argument('--quick', action='store_true')
    t = sub.add_parser('tree')
    t.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    t.add_argument('--horizon', type=int, choices=HORIZONS, required=True)
    t.add_argument('--design', nargs='+', default=list(DESIGNS), choices=DESIGNS)
    t.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'graph':
        if (a.variable, a.arm, a.horizon) not in cells():
            raise SystemExit(f'undeclared cell {a.variable} {a.arm} {a.horizon}')
        run_graph(a.variable, a.arm, a.horizon, a.design, a.quick)
    else:
        run_tree(a.variable, a.horizon, a.design, a.quick)


if __name__ == '__main__':
    main()
