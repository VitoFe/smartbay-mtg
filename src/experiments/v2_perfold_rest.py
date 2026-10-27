"""Addendum 48: selection per outer fold for the remaining tree and ridge arms.

Addendum 47 (`src/experiments/v2_perfold.py`) re-selected per outer fold the plain and weather trees at
6 and 24 h and the Field-57 arms A1-A4 and A7. This module applies the same three designs to every
other tuned tree or ridge arm the thesis quotes:

  published  the published selection refitted (gates G1, G2 and G4); the published rows themselves
             are what the contrasts report
  shared     the published design re-run in this environment: one study per arm on the mean over
             both inner windows (forecasting) or all eleven checkpoints (Field-57); the local
             Optuna (5.0.0) proposes differently from the published one (4.9.0) after the twenty
             random start-up trials, so this is the control for the design
  perfold    one study per (arm, outer fold) on that fold's own inner window or checkpoints only

Forecasting (Dev-2025): the plain and weather trees at 1, 3, 12, 36 and 48 h (v2_horizon), the river
arms at 6, 24 and 48 h (v2_river), the tuned-lag arms at 6 and 24 h (v2_river_lag), the ISPRA arms at
6, 24 and 48 h (v2_ispra), the seabed arm at 6 and 24 h (v2_bathy), the XGBoost trees at 6 h
(v2_forecast, built here with v2_horizon at 6 h, which gate G1 checks) and the apparent-utilisation
tree at 6 h (v2_aou_forecast, with its own objective). Field-57: A5 (v2_field_intercept), A6
(v2_field_ctdprior), P1 and P2 (v2_field_placebo) and A1_both_aou (v2_field).

Every fit and objective is the arm's own published code, imported, not copied. Declared in
config/bay_reserved_protocol_v2_addendum48.json before anything below was run. Amendment 1
(config/bay_reserved_protocol_v2_addendum48_amendment1.json): XGBoost's row subsampling depends on
the thread count and the platform, so the published XGBoost rows are checked through the fold models
stored with the run (G1x-a), every XGBoost fit here uses two threads, and `run_streams` refits the
published XGBoost and LightGBM configurations under ten random states of the subsampling.

Usage (from the repository root; a study already on disk is loaded, not re-run):
  python -m src.experiments.v2_perfold_rest cells
  python -m src.experiments.v2_perfold_rest forecast --family river --variable oxygen --arm river_tree --horizon 24
  python -m src.experiments.v2_perfold_rest aou
  python -m src.experiments.v2_perfold_rest field --arm A5_intercept_direct
  and, through run_job with V2_THREADS=2, 'streams:<variable>'

`field --arm A6_ctdprior_direct --design shared perfold --inner-prior` (review R3, 2026-10-05): the
inner station folds score against the cast prior rebuilt without each split's validation stations
(`v2_field_ctdprior.inner_split_priors`), where the published objective read the prior from all the
checkpoint's training stations; the refit is unchanged. Writes `runs/v2_perfold_rest/field_innerprior/`.
`field --arm P1_placebo_direct --design shared perfold --inner-prior` (addendum 71) does the same for the
fitted depth control, whose basis each inner split fits on its inner-training rows
(`v2_field_placebo.inner_split_mats`).
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
OUT = RUNS / 'v2_perfold_rest'
A47 = RUNS / 'v2_perfold'
DECLARED = 'config/bay_reserved_protocol_v2_addendum48.json'
KEY = ['fold', 'sensor', 'time']
DESIGNS = ('published', 'shared', 'perfold')

FAMILIES = {
    'horizon': {'arms': ('tree', 'weather_tree'), 'horizons': (1, 3, 12, 36, 48)},
    'river': {'arms': ('river_tree', 'weather_river_tree'), 'horizons': (6, 24, 48)},
    'riverlag': {'arms': ('ameglia_lag', 'bocca_lag', 'bocca_gated'), 'horizons': (6, 24)},
    'ispra': {'arms': ('sealevel_tree', 'extboundary_tree'), 'horizons': (6, 24, 48)},
    'bathy': {'arms': ('bathy_tree',), 'horizons': (6, 24)},
    'xgb': {'arms': ('xgb_tree', 'xgb_weather_tree'), 'horizons': (6,)},
}
PUBLISHED_RUN = {'river': '20260920T183451Z_v2_river', 'riverlag': '20260922T065137Z_v2_riverlag',
                 'ispra': '20261001T105500Z_v2_ispra', 'bathy': '20260920T194734Z_v2_bathy'}
FIELD_ARMS = ('A5_intercept_direct', 'A6_ctdprior_direct', 'P1_placebo_direct', 'P2_placebo_direct',
              'A1_both_aou')


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


def cells():
    """Every declared forecasting cell, as (family, variable, arm, horizon)."""
    return [(fam, v, a, h) for fam, spec in FAMILIES.items() for v in ('temperature', 'oxygen')
            for a in spec['arms'] for h in spec['horizons']]


# ---------------------------------------------------------------- forecasting families

def published_folder(family, variable, arm, horizon):
    """The published result folder of one cell and the stem of its selection record."""
    if family == 'horizon':
        run = '20260920T083322Z_v2_horizon' if horizon == 48 else '20260920T133747Z_v2_horizon'
        return RUNS / run / 'results' / f'v2_horizon_{variable}_{arm}_h{horizon}', f'{variable}_{arm}_h{horizon}'
    if family == 'xgb':
        return (RUNS / '20260919T183751Z_v2_xgb/results' / f'v2_forecast_{variable}_{arm}_nested',
                f'{variable}_{arm}_nested')
    return (RUNS / PUBLISHED_RUN[family] / 'results' / f'v2_{family}_{variable}_{arm}_h{horizon}',
            f'{variable}_{arm}_h{horizon}')


def design_builder(family, arm, data):
    """The arm's own design function, as its producer builds it, taking (fold, lag)."""
    if family in ('horizon', 'xgb'):
        from src.experiments.bay_dynamics_tree import features
        from src.data.bay_forcing import weather
        met = weather(data['times']).to_numpy() if arm.endswith('weather_tree') else None

        def design(fold, lag=None):
            base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
            if met is None:
                return base
            return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']),
                                                                        met.shape[1]))], axis=-1)
        return design
    if family == 'river':
        from src.experiments.v2_river import ARMS, design_for
        return lambda fold, lag=None: design_for(data, fold, ARMS[arm])
    if family == 'riverlag':
        from src.experiments.v2_river_lag import design_for
        return lambda fold, lag: design_for(data, fold, arm, lag)
    if family == 'ispra':
        from src.experiments.v2_ispra import design_for
        return lambda fold, lag=None: design_for(data, fold, arm)
    if family == 'bathy':
        from src.experiments.v2_bathy import design_for
        return lambda fold, lag=None: design_for(data, fold)
    raise ValueError(family)


def space(family, trial):
    """The arm's published search space, suggested in the published order (the lag first)."""
    from src.experiments.v2_forecast import tree_space, xgb_space
    lag = None
    if family == 'riverlag':
        from src.experiments.v2_river_lag import LAG_GRID
        lag = trial.suggest_categorical('river_lag_h', LAG_GRID)
    return (xgb_space(trial) if family == 'xgb' else tree_space(trial)), lag


def to_params(family, raw):
    """A selection record's parameters as the producer fits them, and the lag if any."""
    raw = dict(raw)
    lag = raw.pop('river_lag_h', None)
    return (raw if family == 'xgb' else {**raw, 'subsample_freq': 1}), lag


def _macro(d, col='prediction'):
    e = (d.truth - d[col]).abs()
    return float(d.assign(e=e).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def _summary(d):
    out = {'endpoints': int(len(d)), 'mae': _macro(d), 'persistence_mae': _macro(d, 'persistence')}
    for k, g in d.groupby('fold'):
        out[f'fold{k}'] = {'endpoints': int(len(g)), 'mae': _macro(g), 'persistence_mae': _macro(g, 'persistence')}
    return out


def xgb_stored_gate(data, outer, design, source):
    """Amendment 1, G1x-a: the fold models stored with the published run reproduce its rows here."""
    import xgboost as xgb
    published = pd.read_csv(source / 'selected_predictions.csv.gz')
    worst = 0.0
    for f in outer:
        stored = json.loads((source / f'selected_fold{f["number"]}_model.json').read_text())
        booster = xgb.Booster()
        booster.load_model(bytearray(stored['model_string'].encode()))
        mask = data['valid'] & f['eligible'][None, :]
        va = f['va']
        vm = mask[va]
        p = (booster.predict(xgb.DMatrix(design(f)[va][vm])) + data['x'][va][vm]) * data['scale'] + data['center']
        want = published[published.fold == f['number']].prediction.to_numpy()
        assert len(p) == len(want)
        worst = max(worst, float(np.abs(p - want).max()))
    return worst


def run_forecast(family, variable, arm, horizon, designs=DESIGNS):
    from src.experiments.v2_horizon import build, folds, inner_fold, climatology, ratio, tree_fit_score
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    if family == 'xgb' and os.environ['V2_THREADS'] != '2':
        raise RuntimeError('amendment 1: every XGBoost fit uses n_jobs = 2 (V2_THREADS=2)')
    started = time.time()
    data = build(variable, horizon)
    outer = folds(data, horizon)
    inner = [inner_fold(data, f, horizon) for f in outer]
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

    source, stem = published_folder(family, variable, arm, horizon)
    published = pd.read_csv(source / 'selected_predictions.csv.gz')
    sel_pub = json.loads((source / f'{stem}_selection.json').read_text())
    tag = f'{family}_{variable}_{arm}_h{horizon}'
    for name in designs:
        folder = OUT / 'forecast' / name / tag
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        report = {'declared_in': DECLARED, 'design': name, 'family': family, 'variable': variable, 'arm': arm,
                  'horizon_hours': horizon, 'protocol': PROTOCOL['version'], 'threads': int(os.environ['V2_THREADS']),
                  'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner]}
        if name == 'published':
            params, lag = to_params(family, sel_pub['params'])
            by_fold = {f['number']: (params, lag) for f in outer}
            value = float(np.mean([ratio(score(f, params, lag)) for f in inner]))
            report.update(selection_source=(source / f'{stem}_selection.json').relative_to(ROOT).as_posix(),
                          params=sel_pub['params'], published_best_value=sel_pub['best_value'], objective_here=value,
                          gate_G2_abs_diff=abs(value - sel_pub['best_value']))
        elif name == 'shared':
            def objective(trial):
                p, lag = space(family, trial)
                return float(np.mean([ratio(score(f, p, lag)) for f in inner]))
            sel = run_study(f'{tag}_shared', objective, folder)
            by_fold = {f['number']: to_params(family, sel['params']) for f in outer}
            report.update(selection=sel, objective="the arm's published objective, mean over both inner windows "
                                                   "of fold-macro forecast MAE / persistence MAE")
        elif name == 'perfold':
            by_fold, sels = {}, {}
            for f, fi in zip(outer, inner):
                assert f['number'] == fi['number']

                def objective(trial, _f=fi):
                    p, lag = space(family, trial)
                    return ratio(score(_f, p, lag))
                sel = run_study(f'{tag}_perfold_fold{f["number"]}', objective, folder)
                sels[str(f['number'])] = sel
                by_fold[f['number']] = to_params(family, sel['params'])
            report.update(selection=sels, objective="fold-macro forecast MAE / persistence MAE on that outer "
                                                    "fold's own inner window only")
        else:
            raise ValueError(name)
        if family == 'riverlag':
            report['selected_lag_hours'] = {str(k): v[1] for k, v in by_fold.items()}
        d = pd.concat([score(f, *by_fold[f['number']]) for f in outer], ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)
        if name == 'published':
            j = published.merge(d, on=KEY, suffixes=('_pub', ''), validate='one_to_one')
            same_rows = len(j) == len(published) == len(d)
            diff = float(np.abs(j.prediction - j.prediction_pub).max())
            rel = float((np.abs(j.prediction - j.prediction_pub) / np.maximum(np.abs(j.prediction_pub), 1e-12)).max())
            cols = [c for c in ('truth', 'persistence', 'baseline', 'clim_hour')
                    if c in published and not (family == 'xgb' and c == 'baseline')]
            other = {c: float(np.abs(j[c] - j[c + '_pub']).max()) for c in cols}
            exact = bool(same_rows and diff <= 1e-9 and max(other.values()) <= 1e-9)
            report.update(gate_G1_rows_identical=bool(same_rows), gate_G1_max_abs_prediction_diff=diff,
                          gate_G1_max_relative_prediction_diff=rel, gate_G1_other_columns_max_abs_diff=other,
                          gate_G1_exact=exact, gate_G1_pass=exact,
                          gate_G2_pass=bool(report['gate_G2_abs_diff'] <= 1e-9))
            if family == 'xgb':
                # Amendment 1: G1 and G2 are reported, not gated; G1x-a and G1x-b replace them.
                stored = xgb_stored_gate(data, outer, lambda f: X(f, None), source)
                again = pd.concat([score(f, *by_fold[f['number']]) for f in outer], ignore_index=True)
                repeat = float(np.abs(again.prediction.to_numpy() - d.prediction.to_numpy()).max())
                report.update(gate_G1_note='reported, not gated (amendment 1)',
                              gate_G1x_a_stored_models_max_abs_diff=stored, gate_G1x_a_pass=bool(stored <= 1e-9),
                              gate_G1x_b_repeat_max_abs_diff=repeat, gate_G1x_b_pass=bool(repeat <= 1e-9),
                              gate_G1_pass=bool(stored <= 1e-9 and repeat <= 1e-9 and same_rows),
                              gate_G2_pass=True, gate_G2_note='not applicable (amendment 1)')
        report.update(metrics=_summary(d), seconds=round(time.time() - t0, 1),
                      prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        _json(folder / 'report.json', report)
        print(f'{tag} {name}: MAE {report["metrics"]["mae"]:.5f} in {report["seconds"]} s'
              + (f'  G1 {report["gate_G1_pass"]} ({report["gate_G1_max_abs_prediction_diff"]:.2e})'
                 f'  G2 {report["gate_G2_pass"]} ({report["gate_G2_abs_diff"]:.2e})' if name == 'published' else ''),
              flush=True)
    print(f'{tag} done in {time.time() - started:.0f} s', flush=True)


def run_streams(variable, seeds=tuple(range(42, 52))):
    """Amendment 1: the published xgb_tree and LightGBM tree configurations under ten subsampling streams.

    Each arm is refitted exactly as v2_horizon.tree_fit_score fits it, with only the model's
    random_state changed; at 42 both reproduce the rows of the published design refitted here.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from src.experiments.v2_horizon import build, folds, climatology, rows_from
    from src.experiments.v2_forecast import THREADS
    if THREADS != 2:
        raise RuntimeError('amendment 1: every XGBoost fit uses n_jobs = 2 (V2_THREADS=2)')
    t0 = time.time()
    data = build(variable, 6)
    outer = folds(data, 6)
    clim = {id(f): climatology(data, f) for f in outer}
    design = design_builder('xgb', 'xgb_tree', data)
    cache = {id(f): design(f) for f in outer}
    xsource, xstem = published_folder('xgb', variable, 'xgb_tree', 6)
    xparams = json.loads((xsource / f'{xstem}_selection.json').read_text())['params']
    tsource = RUNS / '20260920T083322Z_v2_horizon/results' / f'v2_horizon_{variable}_tree_h6'
    tparams = {**json.loads((tsource / f'{variable}_tree_h6_selection.json').read_text())['params'], 'subsample_freq': 1}
    makers = {'xgb_tree': lambda s: xgb.XGBRegressor(**xparams, objective='reg:absoluteerror', tree_method='hist',
                                                     n_jobs=THREADS, random_state=s),
              'tree': lambda s: lgb.LGBMRegressor(**tparams, objective='regression_l1', verbosity=-1, n_jobs=THREADS,
                                                  random_state=s, deterministic=True, force_col_wise=True)}
    frame = None
    for seed in seeds:
        for arm, make in makers.items():
            parts = []
            for f in outer:
                x = cache[id(f)]
                mask = data['valid'] & f['eligible'][None, :]
                tr, va = f['tr'], f['va']
                tm = mask[tr]
                w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]
                w = w / w.mean()
                model = make(seed)
                model.fit(x[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
                grid = np.full(data['x'][va].shape, np.nan)
                vm = mask[va]
                grid[vm] = model.predict(x[va][vm]) + data['x'][va][vm]
                parts.append(rows_from(data, f, grid, clim[id(f)]))
            d = pd.concat(parts, ignore_index=True)
            if frame is None:
                frame = d[KEY + ['truth', 'persistence']].copy()
            assert (d[KEY].to_numpy() == frame[KEY].to_numpy()).all()
            frame[f'{arm}_s{seed}'] = d.prediction.to_numpy()
    folder = OUT / 'streams'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{variable}_h6.csv.gz'
    frame.to_csv(path, index=False)
    checks = {}
    ref = {'tree': pd.read_csv(tsource / 'selected_predictions.csv.gz'),
           'xgb_tree': pd.read_csv(OUT / 'forecast/published' / f'xgb_{variable}_xgb_tree_h6' / 'selected_predictions.csv.gz')}
    for arm, r in ref.items():
        j = frame.merge(r[KEY + ['prediction']], on=KEY, validate='one_to_one')
        checks[f'{arm}_seed42_vs_refit_rows_max_abs_diff'] = float(np.abs(j[f'{arm}_s42'] - j.prediction).max())
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum48_amendment1.json', 'variable': variable,
              'horizon_hours': 6, 'threads': THREADS, 'seeds': list(seeds), 'endpoints': int(len(frame)),
              'params': {'xgb_tree': xparams, 'tree': tparams}, 'checks': checks,
              'mae': {c: _macro(frame, c) for c in frame.columns if c.endswith(tuple(f'_s{s}' for s in seeds))},
              'seconds': round(time.time() - t0, 1), 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    _json(folder / f'{variable}_h6_report.json', report)
    print(f'streams {variable}: ' + json.dumps(checks), flush=True)


# ---------------------------------------------------------------- the apparent-utilisation tree

def _a47_params(design, variable, arm, horizon=6):
    """Addendum 47's selection of one tree under one design, per fold, as fitted."""
    if design == 'published':
        from src.experiments.v2_closed_loop import selected
        p = selected(variable, arm, horizon)
        return {0: p, 1: p}
    r = json.loads((A47 / 'forecast' / design / f'{variable}_{arm}_h{horizon}' / 'report.json').read_text())
    if design == 'shared':
        return {0: r['selection']['params'], 1: r['selection']['params']}
    return {int(k): s['params'] for k, s in r['selection'].items()}


def run_aou(designs=DESIGNS):
    """v2_aou_forecast.main under the three designs. Its objective is its own published one."""
    from src.experiments import v2_aou_forecast as A
    from src.experiments.v2_horizon import build, climatology, folds, inner_fold
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    H = A.HORIZON
    data = build('oxygen', H)
    outer = folds(data, H)
    inner = [inner_fold(data, f, H) for f in outer]
    tdata = build('temperature', H)
    tcol = {n: i for i, n in enumerate(tdata['names'])}
    take = [tcol[n] for n in data['names']]
    t_anchor = tdata['x'][:, take] * tdata['scale'] + tdata['center']
    t_observed = tdata['truth'][:, take] * tdata['scale'] + tdata['center']
    salt = np.full_like(t_anchor, A.SALINITY_PSU)
    do_anchor = data['x'] * data['scale'] + data['center']
    do_truth = data['truth'] * data['scale'] + data['center']
    aou_anchor = A.solubility(t_anchor, salt) - do_anchor
    aou_truth = A.solubility(t_observed, salt) - do_truth
    aou_data = dict(data)
    aou_data['x'] = (aou_anchor - data['center']) / data['scale']
    aou_truth_z = (aou_truth - data['center']) / data['scale']
    aou_data['valid'] = data['valid'] & np.isfinite(aou_anchor) & np.isfinite(aou_truth)
    wcache = {id(f): A.weather_design(data, f) for f in outer + inner}
    pcache = {id(f): A.plain_design(data, f) for f in outer + inner}
    clim = {id(f): climatology(data, f) for f in outer + inner}
    tfolds = folds(tdata, H)
    twdesign = {f['number']: A.weather_design(tdata, f) for f in tfolds}

    def inner_value(f, p):
        grid = A.fit_grid(aou_data, f, wcache[id(f)], p, aou_truth_z)
        m = aou_data['valid'][f['va']] & f['eligible'][None, :]
        return float(np.nanmean(np.abs(grid - aou_truth_z[f['va']])[m] * data['scale']))

    sel_pub = json.loads((RUNS / 'v2_aou_forecast/oxygen_aou_tree_h6_selection.json').read_text())
    published = pd.read_csv(RUNS / 'v2_aou_forecast/aou_forecast_h6.csv.gz')
    for name in designs:
        t0 = time.time()
        folder = OUT / 'aou' / name
        folder.mkdir(parents=True, exist_ok=True)
        report = {'declared_in': DECLARED, 'design': name, 'arm': 'aou_tree', 'horizon_hours': H,
                  'protocol': PROTOCOL['version'], 'threads': int(os.environ['V2_THREADS'])}
        if name == 'published':
            aou_params = {0: sel_pub['params'], 1: sel_pub['params']}
            value = float(np.mean([inner_value(f, sel_pub['params']) for f in inner]))
            report.update(published_best_value=sel_pub['best_value'], objective_here=value,
                          gate_G2_abs_diff=abs(value - sel_pub['best_value']),
                          gate_G2_pass=bool(abs(value - sel_pub['best_value']) <= 1e-9))
        elif name == 'shared':
            def objective(trial):
                from src.experiments.v2_forecast import tree_space
                p = tree_space(trial)
                return float(np.mean([inner_value(f, p) for f in inner]))
            sel = run_study('oxygen_aou_tree_h6_shared', objective, folder)
            aou_params = {0: sel['params'], 1: sel['params']}
            report.update(selection=sel)
        else:
            aou_params, sels = {}, {}
            for f in inner:
                def objective(trial, _f=f):
                    from src.experiments.v2_forecast import tree_space
                    return inner_value(_f, tree_space(trial))
                sel = run_study(f'oxygen_aou_tree_h6_perfold_fold{f["number"]}', objective, folder)
                sels[str(f['number'])] = sel
                aou_params[f['number']] = sel['params']
            report.update(selection=sels)
        # The comparison trees and the temperature forecast it inverts through, under the same design.
        t_params = _a47_params(name, 'temperature', 'weather_tree')
        w_params = _a47_params(name, 'oxygen', 'weather_tree')
        p_params = _a47_params(name, 'oxygen', 'tree')
        t_forecast = np.full_like(t_observed, np.nan)
        for f in tfolds:
            grid = A.fit_grid(tdata, f, twdesign[f['number']], t_params[f['number']], tdata['truth'])
            t_forecast[f['va']] = (grid * tdata['scale'] + tdata['center'])[:, take]
        rows = []
        for fold in outer:
            k = fold['number']
            aou_grid = A.fit_grid(aou_data, fold, wcache[id(fold)], aou_params[k], aou_truth_z)
            do_grid = A.fit_grid(data, fold, wcache[id(fold)], w_params[k], data['truth'])
            pl_grid = A.fit_grid(data, fold, pcache[id(fold)], p_params[k], data['truth'])
            cl, _ = clim[id(fold)]
            mask = data['valid'][fold['va']] & fold['eligible'][None, :]
            for j, i in enumerate(fold['va']):
                for s, sensor in enumerate(data['names']):
                    if not mask[j, s] or not np.isfinite(t_forecast[i, s]) or not np.isfinite(aou_anchor[i, s]):
                        continue
                    sol_hat = float(A.solubility(t_forecast[i, s], A.SALINITY_PSU))
                    rows.append({'fold': k, 'sensor': sensor, 'time': str(data['times'][i]),
                                 'truth': float(do_truth[i, s]),
                                 'aou_tree': sol_hat - float(aou_grid[j, s] * data['scale'] + data['center']),
                                 'weather_tree': float(do_grid[j, s] * data['scale'] + data['center']),
                                 'plain_tree': float(pl_grid[j, s] * data['scale'] + data['center']),
                                 'thermodynamic': sol_hat - float(aou_anchor[i, s]),
                                 'persistence': float(do_anchor[i, s]),
                                 'climatology': float(cl[i, s] * data['scale'] + data['center'])})
        frame = pd.DataFrame(rows)
        path = folder / 'aou_forecast_h6.csv.gz'
        frame.to_csv(path, index=False)
        arms = ['aou_tree', 'weather_tree', 'plain_tree', 'thermodynamic', 'persistence', 'climatology']
        report.update(aou_params_by_fold={str(k): v for k, v in aou_params.items()},
                      mae={a: A.macro(frame, a) for a in arms}, endpoints=int(len(frame)))
        if name == 'published':
            same = len(frame) == len(published) and list(frame.columns) == list(published.columns) and \
                (frame[KEY].astype(str).to_numpy() == published[KEY].astype(str).to_numpy()).all()
            diff = {c: float(np.abs(frame[c].to_numpy() - published[c].to_numpy()).max()) for c in arms + ['truth']} \
                if same else None
            report.update(gate_G1_rows_identical=bool(same), gate_G1_max_abs_diff=diff,
                          gate_G1_pass=bool(same and max(diff.values()) <= 1e-9))
        report.update(seconds=round(time.time() - t0, 1),
                      prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        _json(folder / 'report.json', report)
        print(f'aou {name}: ' + '  '.join(f'{a} {report["mae"][a]:.4f}' for a in arms)
              + (f'  G1 {report["gate_G1_pass"]} G2 {report["gate_G2_pass"]}' if name == 'published' else ''),
              flush=True)


# ---------------------------------------------------------------- Field-57 arms

def field_setup(arm, inner_prior=False):
    """Bundles, the arm's matrices per draw, and its own fit and inner score, as its producer builds them.

    With `inner_prior` (A6, review R3) the inner score of each checkpoint reads, in each inner
    split, the cast prior rebuilt without that split's validation stations
    (`v2_field_ctdprior.inner_split_priors`); the matrices of the refit are unchanged. The returned
    score then carries the per-split coverage as `score.coverage`. For P1 (addendum 71) each inner
    split reads the depth basis fitted on its inner-training rows (`v2_field_placebo.inner_split_mats`)
    and `score.coverage` is None."""
    if inner_prior and arm not in ('A6_ctdprior_direct', 'P1_placebo_direct'):
        raise ValueError('the inner-split fit applies to A6_ctdprior_direct and P1_placebo_direct only')
    from src.data.bay_profile_context import TARGETS, load_profiles
    from src.experiments.bay_field_export import EXTRAS
    from src.experiments.bay_profile_field import prepare
    from src.experiments import v2_field as F
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, _ = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                on=['cast_id', 'depth_m'], validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    stations, casts = data.station.to_numpy(), data.cast_id.to_numpy()
    truth = data[TARGETS].to_numpy(float)
    draws = [None]
    if arm == 'A1_both_aou':
        use_base, use_design, aou = F.ARMS[arm]
        draws = [[F.matrices(b, extras, use_base, use_design, aou) for b in bundles]]
        fit = lambda feats, base, y, tr, va, w, a: F.fit_predict(feats, base, y, tr, va, w, a, aou)[0]
        score = lambda b, m, a: F.inner_score(b, m, a, aou, stations, casts, truth)
        folder = RUNS / '20260919T145045Z_v2_field/results/v2_field_held_site_A1_both_aou'
    else:
        use_base, use_design, aou = F.ARMS['A1_both_direct']
        folder = ROOT / 'results' / f'v2_field_held_site_{arm}'
        if arm == 'A5_intercept_direct':
            from src.experiments import v2_field_intercept as I
            draws = [[F.matrices(b, extras, use_base, use_design, aou) for b in bundles]]
            fit = lambda feats, base, y, tr, va, w, a: I.fit_predict(feats, base, y, tr, va, w, a)[0]
            score = lambda b, m, a: I.inner_score(b, m, a, stations, casts, truth)
        else:
            fit = lambda feats, base, y, tr, va, w, a: F.fit_predict(feats, base, y, tr, va, w, a, aou)[0]
            score = lambda b, m, a: F.inner_score(b, m, a, aou, stations, casts, truth)
            if arm == 'A6_ctdprior_direct':
                from src.experiments import v2_field_ctdprior as C
                times = pd.to_datetime(data.t_utc, utc=True)
                query_bin = np.floor(data.depth_m.to_numpy() / C.BIN_M).astype(int)
                full = C.archive()
                mats = []
                for b, contract in zip(bundles, contracts):
                    cutoff = pd.Timestamp(contract['last_training'])
                    if cutoff.tzinfo is None:
                        cutoff = cutoff.tz_localize('UTC')
                    train = full[full.station.isin(set(contract['training_stations'])) & (full.t_utc <= cutoff)]
                    delta, _ = C.month_depth_prior(train, query_bin, times.dt.month.to_numpy())
                    e = extras.copy()
                    e[:, 3:6] = delta
                    mats.append(F.matrices(b, e, use_base, use_design, aou))
                draws = [mats]
                if inner_prior:
                    month = times.dt.month.to_numpy()
                    split, coverage = C.inner_split_mats(
                        bundles, contracts, full, stations, lambda t: C.month_depth_prior(t, query_bin, month),
                        extras, lambda b, e: F.matrices(b, e, use_base, use_design, aou))
                    by_bundle = {id(b): sm for b, sm in zip(bundles, split)}

                    def score(b, m, a):
                        return F.inner_score(b, m, a, aou, stations, casts, truth, split_mats=by_bundle[id(b)])
                    score.coverage = coverage
            else:   # P1, P2: the offsets in the published generator order (every draw, bundle by bundle)
                from src.experiments import v2_field_placebo as P
                kind = arm.split('_')[0]
                depth = data.depth_m.to_numpy(float)
                rng = np.random.default_rng(P.SEED)
                draws = []
                for _ in range(1 if kind == 'P1' else P.DRAWS):
                    mats = []
                    for b in bundles:
                        mask = np.zeros(len(depth), dtype=bool)
                        mask[b['tr']] = True
                        e = extras.copy()
                        e[:, 3:6] = P.placebo_offsets(kind, depth, mask, truth - b['base'].astype(float), rng)
                        mats.append(F.matrices(b, e, use_base, use_design, aou))
                    draws.append(mats)
                if inner_prior:
                    split = P.inner_split_mats(bundles, stations, depth, truth, extras,
                                               lambda b, e: F.matrices(b, e, use_base, use_design, aou))
                    by_bundle = {id(b): sm for b, sm in zip(bundles, split)}

                    def score(b, m, a):
                        return F.inner_score(b, m, a, aou, stations, casts, truth, split_mats=by_bundle[id(b)])
                    score.coverage = None
    return bundles, contracts, draws, fit, score, folder


def run_field(arm, designs=DESIGNS, inner_prior=False):
    """`inner_prior` (A6, designs shared and perfold; review R3): the objective reads the cast prior
    rebuilt per inner split (field_setup); outputs under OUT/'field_innerprior', study names with
    the suffix '_innerprior'."""
    from src.experiments.bay_profile_field import prediction_frame
    from src.experiments.v2_field import nested_mae
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    if inner_prior and set(designs) - {'shared', 'perfold'}:
        raise SystemExit('--inner-prior runs the shared and perfold designs only')
    bundles, contracts, draws, fit, score, source = field_setup(arm, inner_prior) if inner_prior else field_setup(arm)
    first = draws[0]
    sel_pub = json.loads((source / f'held_site_{arm}_selection.json').read_text())
    pub_report = json.loads((source / 'report.json').read_text())
    study = (lambda s: f'{s}_innerprior') if inner_prior else (lambda s: s)
    for name in designs:
        t0 = time.time()
        folder = OUT / ('field_innerprior' if inner_prior else 'field') / name / arm
        folder.mkdir(parents=True, exist_ok=True)
        report = {'declared_in': DECLARED, 'design': name, 'arm': arm, 'protocol': PROTOCOL['version'],
                  'draws': len(draws),
                  'checkpoints': [{'fold': c['fold'], 'held_station': c['held_station']} for c in contracts]}
        if inner_prior and arm == 'P1_placebo_direct':
            report.update(declared_in='config/bay_reserved_protocol_v2_addendum71.json', inner_prior=(
                'each inner split scored against the depth basis fitted on its inner-training rows; '
                'the refit uses the basis fitted on all training rows'))
        elif inner_prior:
            report.update(review='R3 (2026-10-05)', inner_prior=(
                'each inner split scored against the cast prior rebuilt without its validation stations; '
                'the refit uses the prior from all training stations'), inner_split_coverage=score.coverage)
        if name == 'published':
            a = sel_pub['params']['alpha']
            alpha = {0: a, 1: a}
            value = float(np.mean([score(b, m, a) for b, m in zip(bundles, first)]))
            report.update(selection_source=(source / f'held_site_{arm}_selection.json').relative_to(ROOT).as_posix(),
                          published_best_value=sel_pub['best_value'], objective_here=value,
                          objective_abs_diff=abs(value - sel_pub['best_value']))
        elif name == 'shared':
            def objective(trial):
                a = trial.suggest_float('alpha', .1, 1e8, log=True)
                return float(np.mean([score(b, m, a) for b, m in zip(bundles, first)]))
            sel = run_study(study(f'held_site_{arm}_shared'), objective, folder)
            alpha = {0: sel['params']['alpha'], 1: sel['params']['alpha']}
            report.update(selection=sel, objective="the arm's published objective over all eleven checkpoints")
        else:
            alpha, sels = {}, {}
            for k in (0, 1):
                pairs = [(b, m) for b, m in zip(bundles, first) if b['fold'] == k]

                def objective(trial, _pairs=pairs):
                    a = trial.suggest_float('alpha', .1, 1e8, log=True)
                    return float(np.mean([score(b, m, a) for b, m in _pairs]))
                sel = run_study(study(f'held_site_{arm}_perfold_fold{k}'), objective, folder)
                sels[str(k)] = sel
                alpha[k] = sel['params']['alpha']
            report.update(selection=sels, objective="the arm's published objective over that calendar fold's own "
                                                    "checkpoints only")
        per_draw = []
        for n, mats in enumerate(draws):
            rows = []
            for b, (base, y, feats) in zip(bundles, mats):
                pred = fit(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float), alpha[b['fold']])
                rows.append(prediction_frame(b, pred))
            d = pd.concat(rows, ignore_index=True)
            per_draw.append(nested_mae(d))
            if n == 0:
                path = folder / 'selected_predictions.csv.gz'
                d.to_csv(path, index=False)
                d0 = d
        report.update(alpha_by_fold={str(k): v for k, v in alpha.items()}, nested_mae=per_draw[0],
                      per_draw_nested_mae=per_draw if len(draws) > 1 else None,
                      nested_mae_mean_over_draws={t: float(np.mean([x[t] for x in per_draw])) for t in per_draw[0]},
                      rows=int(len(d0)), casts=int(d0.cast_id.nunique()))
        if name == 'published':
            pub = pd.read_csv(source / 'selected_predictions.csv.gz')
            cols = [c for c in pub.columns if c.startswith('pred_')]
            same = list(pub.columns) == list(d0.columns) and len(pub) == len(d0) and \
                (pub[['cast_id', 'depth_m', 'fold']].to_numpy() == d0[['cast_id', 'depth_m', 'fold']].to_numpy()).all()
            a, b = pub[cols].to_numpy(float), d0[cols].to_numpy(float)
            diff = float(np.nanmax(np.abs(a - b)))
            rel = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))
            want = (pub_report.get('nested_mae') or {t: v['mean'] for t, v in pub_report['summary'].items()})
            got = per_draw[0] if 'nested_mae' in pub_report else report['nested_mae_mean_over_draws']
            mae_diff = max(abs(got[t] - want[t]) for t in want)
            report.update(gate_G4_rows_identical=bool(same), gate_G4_max_abs_prediction_diff=diff,
                          gate_G4_max_relative_prediction_diff=rel, gate_G4_nested_mae_abs_diff=mae_diff,
                          gate_G4_pass=bool(same and rel <= 1e-6 and mae_diff <= 1e-6))
        report.update(seconds=round(time.time() - t0, 1),
                      prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        _json(folder / 'report.json', report)
        m = report['nested_mae_mean_over_draws']
        print(f'{arm} {name}: alpha {report["alpha_by_fold"]}  T {m["temp_c"]:.4f}  S {m["salinity_psu"]:.4f}'
              f'  O {m["do_umol_l"]:.3f}  ({report["seconds"]} s)'
              + (f'  G4 {report["gate_G4_pass"]} ({report["gate_G4_max_relative_prediction_diff"]:.1e})'
                 if name == 'published' else ''), flush=True)


def run_job(spec):
    """One queued job: 'forecast:<family>:<variable>:<arm>:<horizon>', 'aou' or 'field:<arm>'."""
    parts = spec.split(':')
    if parts[0] == 'forecast':
        run_forecast(parts[1], parts[2], parts[3], int(parts[4]))
    elif parts[0] == 'streams':
        run_streams(parts[1])
    elif parts[0] == 'aou':
        run_aou()
    elif parts[0] == 'field':
        run_field(parts[1])
    else:
        raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('cells')
    f = sub.add_parser('forecast')
    f.add_argument('--family', choices=list(FAMILIES), required=True)
    f.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    f.add_argument('--arm', required=True)
    f.add_argument('--horizon', type=int, required=True)
    f.add_argument('--design', nargs='+', default=list(DESIGNS), choices=DESIGNS)
    a_ = sub.add_parser('aou')
    a_.add_argument('--design', nargs='+', default=list(DESIGNS), choices=DESIGNS)
    s = sub.add_parser('field')
    s.add_argument('--arm', nargs='+', default=list(FIELD_ARMS), choices=FIELD_ARMS)
    s.add_argument('--design', nargs='+', default=list(DESIGNS), choices=DESIGNS)
    s.add_argument('--inner-prior', action='store_true',
                   help='A6: each inner split against the cast prior rebuilt without its validation stations')
    a = ap.parse_args()
    if a.cmd == 'cells':
        for c in cells():
            print(':'.join(['forecast', *map(str, c)]))
        print('aou')
        for arm in FIELD_ARMS:
            print(f'field:{arm}')
    elif a.cmd == 'forecast':
        if a.arm not in FAMILIES[a.family]['arms'] or a.horizon not in FAMILIES[a.family]['horizons']:
            raise SystemExit('undeclared cell')
        run_forecast(a.family, a.variable, a.arm, a.horizon, a.design)
    elif a.cmd == 'aou':
        run_aou(a.design)
    else:
        for arm in a.arm:
            if a.inner_prior:
                run_field(arm, a.design, inner_prior=True)
            else:
                run_field(arm, a.design)


if __name__ == '__main__':
    main()
