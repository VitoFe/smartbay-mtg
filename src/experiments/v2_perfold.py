"""Addendum 47: how much does sharing one configuration across the two outer folds matter?

The published forecasting and Field-57 studies choose one configuration per arm. For forecasting
(`v2_horizon.run`, `v2_forecast.run`) the objective is the mean over both inner windows, and fold
1's inner window (the 45 days ending 72 h before 2025-10-01) lies inside fold 0's scored window
(2025-07-01 to 2025-10-01). For Field-57 (`v2_field.run`, `v2_field_ctdprior.run`) one ridge penalty
is chosen on the mean over all eleven (fold, held-station) checkpoints, and the fold-1 checkpoints
train on, and therefore cross-validate over, the casts that fold 0 scores.

This module counts that overlap and measures its effect. Three designs per arm:

  published    the published selection, refitted here (gate: it reproduces the published rows)
  shared       the published design re-run in this environment, so that the design effect is not
               confused with the sampler: the local Optuna (5.0.0) proposes differently from the
               one the published studies ran (4.9.0) after the twenty random start-up trials
  perfold      one study per (arm, outer fold) whose objective sees that fold's own inner window
               (forecasting) or that fold's own checkpoints (Field-57) only

Declared in config/bay_reserved_protocol_v2_addendum47.json before anything below was run.

Usage (from the repository root, V2_THREADS=2; with no command it runs everything below in turn,
and a study already on disk is loaded, not re-run):
  python -m src.experiments.v2_perfold overlap
  python -m src.experiments.v2_perfold forecast --variable temperature --arm tree --horizon 6 --design perfold
  python -m src.experiments.v2_perfold field --arm A1_both_direct --design perfold

`field --inner-prior` (review R3, 2026-10-05; A7 only, designs shared and perfold): the inner
station folds of every checkpoint score against the cast prior rebuilt without that split's
validation stations (`v2_field_ctdprior.inner_split_priors`), where the published A7 objective read
the prior built from all the checkpoint's training stations; the refit at the selected penalty uses
that full prior, as before. Writes `runs/v2_perfold/field_innerprior/<design>/A7_ctdprior_direct/`.
  python -m src.experiments.v2_perfold field --arm A7_ctdprior_direct --design shared perfold --inner-prior
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
OUT = RUNS / 'v2_perfold'
HORIZON_RUN = RUNS / '20260920T083322Z_v2_horizon/results'
FIELD_RUN = RUNS / '20260919T145045Z_v2_field/results'
FIELD_LOCAL = ROOT / 'results'
FORECAST_ARMS = ('tree', 'weather_tree')
FORECAST_HORIZONS = (6, 24)
FIELD_ARMS = ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct', 'A4_base_only_direct',
              'A7_ctdprior_direct')
KEY = ['fold', 'sensor', 'time']


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def _rows(data, rows_idx, mask):
    """(time index, sensor) pairs of `mask` restricted to the time indices `rows_idx`."""
    j, s = np.nonzero(mask[rows_idx])
    return set(zip(rows_idx[j].tolist(), s.tolist()))


# ---------------------------------------------------------------- step 1: the overlap, counted

def forecast_overlap(variable, horizon):
    from src.experiments.v2_horizon import build, folds, inner_fold
    data = build(variable, horizon)
    outer = folds(data, horizon)
    inner = [inner_fold(data, f, horizon) for f in outer]
    valid = data['valid']
    scored = {f['number']: _rows(data, f['va'], valid & f['eligible'][None, :]) for f in outer}
    ival = {f['number']: _rows(data, f['va'], valid & f['eligible'][None, :]) for f in inner}
    itr = {f['number']: _rows(data, f['tr'], valid & f['eligible'][None, :]) for f in inner}
    t = data['times']
    out = {'variable': variable, 'horizon_hours': horizon,
           'endpoints': {str(k): len(v) for k, v in scored.items()},
           'endpoints_total': sum(len(v) for v in scored.values()),
           'inner_windows': [f['audit'] for f in inner], 'by_scored_fold': {}}
    for k, rows in scored.items():
        entry = {}
        for m, _ in scored.items():
            seen_val = rows & ival[m]
            seen_tr = rows & itr[m]
            entry[f'inner{m}'] = {
                'validation_rows': len(seen_val), 'training_rows': len(seen_tr),
                'either': len(seen_val | seen_tr),
                'validation_span': ([str(t[min(i for i, _ in seen_val)]), str(t[max(i for i, _ in seen_val)])]
                                    if seen_val else None),
                'validation_sensors': sorted({data['names'][s] for _, s in seen_val})}
        out['by_scored_fold'][str(k)] = entry
    f0 = out['by_scored_fold']['0']['inner1']
    out['fold0_scored_that_were_fold1_inner_validation'] = f0['validation_rows']
    out['share_of_fold0'] = f0['validation_rows'] / out['endpoints']['0']
    out['share_of_all'] = f0['validation_rows'] / out['endpoints_total']
    out['fold0_scored_that_were_fold1_inner_training'] = f0['training_rows']
    return out, data, outer, inner


def published_overlap_check(variable, horizon, inner):
    """The same count from the published rows alone (time stamps and sensors), as a cross-check."""
    d = pd.read_csv(HORIZON_RUN / f'v2_horizon_{variable}_tree_h{horizon}/selected_predictions.csv.gz')
    a = inner[1]['audit']
    start = pd.Timestamp(a['inner_selection_start']); end = pd.Timestamp(a['inner_selection_end'])
    time_ = pd.to_datetime(d.time, utc=True)
    hit = (d.fold == 0) & (time_ >= start) & (time_ + pd.Timedelta(hours=horizon) < end)
    return {'fold0_rows': int((d.fold == 0).sum()), 'rows': int(len(d)),
            'fold0_in_fold1_inner_window': int(hit.sum()),
            'note': 'counts every fold-0 row in the window; the mask count above also requires the '
                    'sensor to be eligible in the fold-1 inner split'}


def field_overlap():
    from src.data.bay_profile_context import load_profiles
    from src.experiments.bay_profile_field import prepare
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, _ = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    day = pd.to_datetime(data.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    casts = data.cast_id.to_numpy()
    out = {'checkpoints': [{'fold': b['fold'], 'held_station': c['held_station'],
                            'scored_casts': len(c['selection_cast_ids']),
                            'training_casts': len(c['training_cast_ids'])}
                           for b, c in zip(bundles, contracts)]}
    for k in (0, 1):
        scored = np.unique(np.concatenate([b['va'] for b in bundles if b['fold'] == k]))
        other = [b for b in bundles if b['fold'] != k]
        seen = np.unique(np.concatenate([np.intersect1d(scored, b['tr']) for b in other])) if other else np.array([], int)
        per_cast = pd.Series({c: sum(np.isin(np.flatnonzero(casts == c), b['tr']).any() for b in other)
                              for c in np.unique(casts[scored])})
        out[f'fold{k}'] = {
            'scored_rows': int(len(scored)), 'scored_casts': int(len(np.unique(casts[scored]))),
            'scored_dates': int(len(np.unique(day[scored]))),
            'rows_in_other_fold_training': int(len(seen)),
            'casts_in_other_fold_training': int(len(np.unique(casts[seen]))),
            'dates_in_other_fold_training': int(len(np.unique(day[seen]))),
            'other_fold_checkpoints': len(other),
            'checkpoints_training_on_each_scored_cast': {'min': int(per_cast.min()), 'max': int(per_cast.max())}}
    out['note'] = ('The inner GroupKFold of a checkpoint validates on every one of its training rows once, '
                   'so a row in another checkpoint\'s training set is an inner-validation row of the shared study.')
    return out


def overlap():
    report = {'forecast': {}, 'forecast_published_check': {}}
    for variable in ('temperature', 'oxygen'):
        for horizon in (1, 3, 6, 12, 24, 36, 48):
            r, _, _, inner = forecast_overlap(variable, horizon)
            report['forecast'][f'{variable}|{horizon}'] = r
            report['forecast_published_check'][f'{variable}|{horizon}'] = (
                published_overlap_check(variable, horizon, inner) if horizon in FORECAST_HORIZONS else None)
            print(f'{variable:12s} h{horizon:<3d} fold0 {r["endpoints"]["0"]:5d}  of which fold-1 inner validation '
                  f'{r["fold0_scored_that_were_fold1_inner_validation"]:5d} ({100 * r["share_of_fold0"]:.1f} % of fold 0, '
                  f'{100 * r["share_of_all"]:.1f} % of {r["endpoints_total"]}), inner training '
                  f'{r["fold0_scored_that_were_fold1_inner_training"]}', flush=True)
    report['field57'] = field_overlap()
    print(json.dumps(report['field57'], indent=1)[:1500])
    _json(OUT / 'overlap.json', report)
    print('wrote', OUT / 'overlap.json')


# ---------------------------------------------------------------- forecasting: three designs per cell

def published_forecast_selection(variable, arm, horizon):
    summary = json.loads((RUNS / 'v2_horizon_summary.json').read_text())
    params = dict(summary[variable][str(horizon)]['arms'][arm]['params'])
    record = HORIZON_RUN / f'v2_horizon_{variable}_{arm}_h{horizon}/{variable}_{arm}_h{horizon}_selection.json'
    sel = json.loads(record.read_text())
    assert sel['params'] == params, 'the summary and the selection record disagree'
    return {**params, 'subsample_freq': 1}, sel, record.relative_to(ROOT).as_posix()


def forecast_setup(variable, arm, horizon):
    """Exactly what v2_horizon.run builds for a tree arm, returned instead of run."""
    from src.experiments.v2_horizon import build, folds, inner_fold, climatology
    from src.experiments.bay_dynamics_tree import features
    from src.data.bay_forcing import weather
    data = build(variable, horizon)
    outer = folds(data, horizon)
    inner = [inner_fold(data, f, horizon) for f in outer]
    clim = {id(f): climatology(data, f) for f in outer + inner}
    met = weather(data['times']).to_numpy() if arm == 'weather_tree' else None

    def design(fold):
        base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
        if met is None:
            return base
        return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']),
                                                                    met.shape[1]))], axis=-1)
    cache = {id(f): design(f) for f in outer + inner}
    return data, outer, inner, clim, cache


def score_folds(data, folds_, params_by_fold, clim, cache):
    from src.experiments.v2_horizon import tree_fit_score
    return pd.concat([tree_fit_score(data, cache[id(f)], f, params_by_fold[f['number']], clim[id(f)])[0]
                      for f in folds_], ignore_index=True)


def _macro(d, col='prediction'):
    e = (d.truth - d[col]).abs()
    return float(d.assign(e=e).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def _summary(d):
    out = {'endpoints': int(len(d)), 'mae': _macro(d), 'persistence_mae': _macro(d, 'persistence')}
    for k, g in d.groupby('fold'):
        out[f'fold{k}'] = {'endpoints': int(len(g)), 'mae': _macro(g), 'persistence_mae': _macro(g, 'persistence')}
    return out


def run_forecast(variable, arm, horizon, designs):
    from src.experiments.v2_horizon import ratio, tree_fit_score
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    started = time.time()
    data, outer, inner, clim, cache = forecast_setup(variable, arm, horizon)
    tag = f'{variable}_{arm}_h{horizon}'
    published = pd.read_csv(HORIZON_RUN / f'v2_horizon_{tag}/selected_predictions.csv.gz')
    for design in designs:
        folder = OUT / 'forecast' / design / tag
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum47.json', 'design': design,
                  'variable': variable, 'arm': arm, 'horizon_hours': horizon, 'protocol': PROTOCOL['version'],
                  'threads': int(os.environ['V2_THREADS']),
                  'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner]}
        if design == 'published':
            params, sel, source = published_forecast_selection(variable, arm, horizon)
            by_fold = {f['number']: params for f in outer}
            # G2: the shared objective at the published selection gives the published best value.
            value = float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, params, clim[id(f)])[0]) for f in inner]))
            report.update(selection_source=source, params=params, published_best_value=sel['best_value'],
                          objective_here=value, gate_G2_abs_diff=abs(value - sel['best_value']),
                          gate_G2_pass=bool(abs(value - sel['best_value']) <= 1e-9))
        elif design == 'shared':
            def objective(trial):
                from src.experiments.v2_forecast import tree_space
                p = tree_space(trial)
                return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))
            sel = run_study(f'{tag}_shared', objective, folder)
            params = {**sel['params'], 'subsample_freq': 1}
            by_fold = {f['number']: params for f in outer}
            report.update(selection=sel,
                          objective='mean over both inner windows of fold-macro forecast MAE / persistence MAE')
        elif design == 'perfold':
            by_fold, sels = {}, {}
            for f, fi in zip(outer, inner):
                assert f['number'] == fi['number']

                def objective(trial, _f=fi):
                    from src.experiments.v2_forecast import tree_space
                    p = tree_space(trial)
                    return ratio(tree_fit_score(data, cache[id(_f)], _f, p, clim[id(_f)])[0])
                sel = run_study(f'{tag}_perfold_fold{f["number"]}', objective, folder)
                sels[str(f['number'])] = sel
                by_fold[f['number']] = {**sel['params'], 'subsample_freq': 1}
            report.update(selection=sels,
                          objective="fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only")
        else:
            raise ValueError(design)
        d = score_folds(data, outer, by_fold, clim, cache)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)
        if design == 'published':
            # G1: the refit reproduces the published rows.
            j = published.merge(d, on=KEY, suffixes=('_pub', ''), validate='one_to_one')
            same_rows = len(j) == len(published) == len(d)
            diff = float(np.abs(j.prediction - j.prediction_pub).max())
            other = {c: float(np.abs(j[c] - j[c + '_pub']).max()) for c in ('truth', 'persistence', 'baseline', 'clim_hour')}
            report.update(gate_G1_rows_identical=bool(same_rows), gate_G1_max_abs_prediction_diff=diff,
                          gate_G1_other_columns_max_abs_diff=other,
                          gate_G1_pass=bool(same_rows and diff <= 1e-9 and max(other.values()) <= 1e-9))
        report.update(metrics=_summary(d), seconds=round(time.time() - t0, 1),
                      prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        _json(folder / 'report.json', report)
        print(f'{tag} {design}: MAE {report["metrics"]["mae"]:.5f} in {report["seconds"]} s'
              + (f'  G1 {report["gate_G1_pass"]} ({report["gate_G1_max_abs_prediction_diff"]:.2e})'
                 f'  G2 {report["gate_G2_pass"]} ({report["gate_G2_abs_diff"]:.2e})' if design == 'published' else ''),
              flush=True)
    print(f'{tag} done in {time.time() - started:.0f} s', flush=True)


# ---------------------------------------------------------------- Field-57: three designs per arm

def field_setup(arms, inner_prior=False):
    """The bundles and each arm's matrices exactly as v2_field.run and v2_field_ctdprior.run build them.

    With `inner_prior` the result carries `common['split_mats'][arm]`: per checkpoint, one matrix
    triple per inner split with the cast prior rebuilt without that split's validation stations
    (A7 only), and `common['split_coverage'][arm]`."""
    from src.data.bay_profile_context import TARGETS, load_profiles
    from src.experiments.bay_field_export import EXTRAS
    from src.experiments.bay_profile_field import prepare
    from src.experiments.v2_field import ARMS, matrices
    from src.experiments import v2_field_ctdprior as ctd
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, excluded = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                on=['cast_id', 'depth_m'], validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    common = {'stations': data.station.to_numpy(), 'casts': data.cast_id.to_numpy(),
              'truth': data[TARGETS].to_numpy(float)}
    mats = {}
    for arm in arms:
        if arm == 'A7_ctdprior_direct':
            use_base, use_design, aou = ARMS[ctd.ARM_BASE]
            times = pd.to_datetime(data.t_utc, utc=True)
            query_bin = np.floor(data.depth_m.to_numpy() / ctd.BIN_M).astype(int)
            full = ctd.archive()
            mats[arm] = []
            for b, contract in zip(bundles, contracts):
                cutoff = pd.Timestamp(contract['last_training'])
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize('UTC')
                train = full[full.station.isin(set(contract['training_stations'])) & (full.t_utc <= cutoff)]
                delta, _ = ctd.harmonic_prior(train, query_bin, times.dt.dayofyear.to_numpy())
                e = extras.copy()
                e[:, 3:6] = delta
                mats[arm].append(matrices(b, e, use_base, use_design, aou))
            if inner_prior:
                doy = times.dt.dayofyear.to_numpy()
                split, coverage = ctd.inner_split_mats(
                    bundles, contracts, full, common['stations'], lambda t: ctd.harmonic_prior(t, query_bin, doy),
                    extras, lambda b, e: matrices(b, e, use_base, use_design, aou))
                common.setdefault('split_mats', {})[arm] = split
                common.setdefault('split_coverage', {})[arm] = coverage
        else:
            use_base, use_design, aou = ARMS[arm]
            mats[arm] = [matrices(b, extras, use_base, use_design, aou) for b in bundles]
    return bundles, contracts, mats, common


def published_field(arm):
    folder = (FIELD_LOCAL if arm == 'A7_ctdprior_direct' else FIELD_RUN) / f'v2_field_held_site_{arm}'
    sel = json.loads((folder / f'held_site_{arm}_selection.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    return sel, report, folder


def run_field(arms, designs, inner_prior=False):
    from src.experiments.bay_profile_field import prediction_frame
    from src.experiments.v2_field import fit_predict, inner_score, nested_mae
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    if inner_prior and (set(arms) - {'A7_ctdprior_direct'} or set(designs) - {'shared', 'perfold'}):
        raise SystemExit('--inner-prior applies to A7_ctdprior_direct under the shared and perfold designs')
    bundles, contracts, mats, c = field_setup(arms, inner_prior)
    aou = False  # every arm here is a direct-oxygen arm

    def score(b, m, alpha, sm=None):
        if sm is None:
            return inner_score(b, m, alpha, aou, c['stations'], c['casts'], c['truth'])
        return inner_score(b, m, alpha, aou, c['stations'], c['casts'], c['truth'], split_mats=sm)

    for arm in arms:
        # per checkpoint, the inner splits' own matrices (inner_prior), else nothing
        splits = c['split_mats'][arm] if inner_prior else [None] * len(bundles)
        study = (lambda name: f'{name}_innerprior') if inner_prior else (lambda name: name)
        for design in designs:
            t0 = time.time()
            folder = OUT / ('field_innerprior' if inner_prior else 'field') / design / arm
            folder.mkdir(parents=True, exist_ok=True)
            report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum47.json', 'design': design,
                      'arm': arm, 'protocol': PROTOCOL['version'],
                      'checkpoints': [{'fold': x['fold'], 'held_station': x['held_station']} for x in contracts]}
            if inner_prior:
                report.update(review='R3 (2026-10-05)', inner_prior=(
                    'each inner split scored against the cast prior rebuilt without its validation stations; '
                    'the refit uses the prior from all training stations'),
                    inner_split_coverage=c['split_coverage'][arm])
            if design == 'published':
                sel, pub_report, pub_folder = published_field(arm)
                alpha = {0: sel['params']['alpha'], 1: sel['params']['alpha']}
                value = float(np.mean([score(b, m, alpha[0]) for b, m in zip(bundles, mats[arm])]))
                report.update(selection_source=(pub_folder / f'held_site_{arm}_selection.json').relative_to(ROOT).as_posix(),
                              alpha=sel['params']['alpha'], published_best_value=sel['best_value'],
                              objective_here=value, objective_abs_diff=abs(value - sel['best_value']))
            elif design == 'shared':
                def objective(trial):
                    a = trial.suggest_float('alpha', .1, 1e8, log=True)
                    return float(np.mean([score(b, m, a, sm) for b, m, sm in zip(bundles, mats[arm], splits)]))
                sel = run_study(study(f'held_site_{arm}_shared'), objective, folder)
                alpha = {0: sel['params']['alpha'], 1: sel['params']['alpha']}
                report.update(selection=sel, objective='mean over all eleven checkpoints of the inner station-fold score')
            elif design == 'perfold':
                alpha, sels = {}, {}
                for k in (0, 1):
                    pairs = [(b, m, sm) for b, m, sm in zip(bundles, mats[arm], splits) if b['fold'] == k]

                    def objective(trial, _pairs=pairs):
                        a = trial.suggest_float('alpha', .1, 1e8, log=True)
                        return float(np.mean([score(b, m, a, sm) for b, m, sm in _pairs]))
                    sel = run_study(study(f'held_site_{arm}_perfold_fold{k}'), objective, folder)
                    sels[str(k)] = sel
                    alpha[k] = sel['params']['alpha']
                report.update(selection=sels, objective="mean over that calendar fold's own checkpoints of the inner station-fold score")
            else:
                raise ValueError(design)
            rows = []
            for b, (base, y, feats) in zip(bundles, mats[arm]):
                pred, _, _ = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                         alpha[b['fold']], aou)
                rows.append(prediction_frame(b, pred))
            d = pd.concat(rows, ignore_index=True)
            path = folder / 'selected_predictions.csv.gz'
            d.to_csv(path, index=False)
            report.update(alpha_by_fold={str(k): v for k, v in alpha.items()}, nested_mae=nested_mae(d),
                          rows=int(len(d)), casts=int(d.cast_id.nunique()))
            if design == 'published':
                pub = pd.read_csv(pub_folder / 'selected_predictions.csv.gz')
                cols = [col for col in pub.columns if col.startswith('pred_')]
                same = list(pub.columns) == list(d.columns) and len(pub) == len(d) and \
                    (pub[['cast_id', 'depth_m', 'fold']].to_numpy() == d[['cast_id', 'depth_m', 'fold']].to_numpy()).all()
                diff = float(np.nanmax(np.abs(pub[cols].to_numpy(float) - d[cols].to_numpy(float))))
                mae_diff = max(abs(report['nested_mae'][t] - pub_report['nested_mae'][t]) for t in report['nested_mae'])
                report.update(gate_G4_rows_identical=bool(same), gate_G4_max_abs_prediction_diff=diff,
                              gate_G4_nested_mae_abs_diff=mae_diff,
                              gate_G4_pass=bool(same and diff <= 1e-9 and mae_diff <= 1e-9))
            report.update(seconds=round(time.time() - t0, 1),
                          prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            _json(folder / 'report.json', report)
            print(f'{arm} {design}: alpha {report["alpha_by_fold"]}  T {report["nested_mae"]["temp_c"]:.4f}'
                  f'  S {report["nested_mae"]["salinity_psu"]:.4f}  O {report["nested_mae"]["do_umol_l"]:.3f}'
                  f'  ({report["seconds"]} s)' + (f'  G4 {report["gate_G4_pass"]} ({report["gate_G4_max_abs_prediction_diff"]:.1e})'
                                                  if design == 'published' else ''), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')   # no command: the whole addendum, in the order it was run
    sub.add_parser('overlap')
    f = sub.add_parser('forecast')
    f.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    f.add_argument('--arm', choices=FORECAST_ARMS, required=True)
    f.add_argument('--horizon', type=int, choices=FORECAST_HORIZONS, required=True)
    f.add_argument('--design', nargs='+', default=['published', 'shared', 'perfold'],
                   choices=['published', 'shared', 'perfold'])
    s = sub.add_parser('field')
    s.add_argument('--arm', nargs='+', default=list(FIELD_ARMS), choices=FIELD_ARMS)
    s.add_argument('--design', nargs='+', default=['published', 'shared', 'perfold'],
                   choices=['published', 'shared', 'perfold'])
    s.add_argument('--inner-prior', action='store_true',
                   help='A7: each inner split against the cast prior rebuilt without its validation stations')
    a = ap.parse_args()
    if a.cmd is None:
        overlap()
        for variable in ('temperature', 'oxygen'):
            for horizon in FORECAST_HORIZONS:
                for arm in FORECAST_ARMS:
                    run_forecast(variable, arm, horizon, ['published', 'shared', 'perfold'])
        run_field(list(FIELD_ARMS), ['published', 'shared', 'perfold'])
    elif a.cmd == 'overlap':
        overlap()
    elif a.cmd == 'forecast':
        run_forecast(a.variable, a.arm, a.horizon, a.design)
    elif a.inner_prior:
        run_field(a.arm, a.design, inner_prior=True)
    else:
        run_field(a.arm, a.design)


if __name__ == '__main__':
    main()
