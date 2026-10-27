"""Addendum-19: a matched non-physical placebo for the vertical prior.

Addendum 14 showed the 0.190 degC improvement does not depend on which physical source supplies
the vertical offset. It did not separate physical knowledge from the possibility that the
background was simply missing depth structure, and that any smooth depth-dependent function of
comparable complexity would supply it.

Two controls, both replacing only the three *_vertical_delta columns:

    P1  a piecewise-linear (hat-function) basis in depth, seven knots at training-depth quantiles,
        coefficients fitted by least squares on each fold's training rows. Time-invariant: one
        depth curve per checkpoint and nothing else, so it matches the prior's vertical
        interpolation but not its variation in time.
    P2  the same seven-dimensional basis with coefficients drawn at random, scaled so the realised
        offset matches the Copernicus offset's standard deviation of 1.20 degC. Twenty seeded
        draws. This is the true placebo and should recover nothing.

Development folds only. Declared in config/bay_reserved_protocol_v2_addendum19.json.

`--inner-prior` (P1 only; addendum 71, second external review F01): the published objective scored
every inner station split against offsets fitted on all of the checkpoint's training rows, the
inner-validation stations' labels included. With the flag each inner split is scored against
offsets whose knots and coefficients come from its inner-training rows only (inner_split_offsets);
the refit after selection uses the offsets from all training rows, as published. Writes
results/v2_field_held_site_P1_placebo_direct_innerprior/.

Usage: python -m src.experiments.v2_field_placebo --arm P1 [--inner-prior]
"""
import argparse
import hashlib
import json
import os
import traceback

import numpy as np
import pandas as pd

from src.data.bay_profile_context import TARGETS, load_profiles
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_profile_field import ROOT, SCALE, prepare, prediction_frame
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_field import ARMS, fit_predict, inner_score, matrices, nested_mae
from src.tuning.protocol_v2 import PROTOCOL, run_study

ARM_BASE = 'A1_both_direct'
KNOTS = 7                 # matches the seven Copernicus product depth levels
COPERNICUS_SD = 1.1987    # empirical sd of boundary_temperature_vertical_delta over cast rows
DRAWS = 20
SEED = 20260921


def basis(depth, knots):
    """Piecewise-linear interpolation basis on `knots`, centred to vanish at 3 m.

    This mirrors the physical construction exactly rather than approximating it. The Copernicus
    offset is np.interp of the model profile over its seven depth levels, minus the same
    interpolation evaluated at 3 m. A hat-function basis over seven knots is that operation
    written as a linear map, so a placebo built on it has the same vertical interpolation as the
    quantity it replaces, and is well conditioned. It is one curve per checkpoint, where the
    product's profile changes with time.
    """
    d = np.asarray(depth, dtype=float)
    k = np.asarray(knots, dtype=float)

    def hats(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        cols = []
        for j in range(len(k)):
            e = np.zeros(len(k))
            e[j] = 1.0
            cols.append(np.interp(x, k, e))
        return np.column_stack(cols)

    return hats(d) - hats(np.array([3.0]))


def placebo_offsets(arm, depth, fold_train_mask, residual, rng):
    """Return an (n, 3) offset block for the three targets."""
    knots = np.quantile(depth[fold_train_mask], np.linspace(0.02, 0.98, KNOTS))
    knots = np.unique(knots)
    if len(knots) < 4:
        knots = np.linspace(depth.min(), depth.max(), 4)
    B = basis(depth, knots)
    out = np.zeros((len(depth), 3))
    if arm == 'P1':
        for j in range(3):
            X = B[fold_train_mask]
            y = residual[fold_train_mask, j]
            keep = np.isfinite(y)
            if keep.sum() < B.shape[1] + 2:
                continue
            coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            out[:, j] = B @ coef
    else:
        for j in range(3):
            scale = COPERNICUS_SD * (SCALE[j] / SCALE[0])
            coef = rng.normal(size=B.shape[1])
            draw = B @ coef
            sd = draw[fold_train_mask].std()
            if sd > 0:
                draw = draw * (scale / sd)
            out[:, j] = draw
    return out


def inner_split_offsets(b, stations, depth, residual):
    """P1's offsets for each inner station split of one checkpoint, in inner_splits' order (addendum 71).

    Each split's knots and coefficients are fitted on its inner-training rows only, so no label of an
    inner-validation station reaches the offsets that split is scored against."""
    from src.experiments.v2_field import inner_splits
    out = []
    for itr, _ in inner_splits(b, stations):
        mask = np.zeros(len(depth), dtype=bool)
        mask[b['tr'][itr]] = True
        out.append(placebo_offsets('P1', depth, mask, residual, None))
    return out


def inner_split_mats(bundles, stations, depth, truth, base_extras, build):
    """Per checkpoint, the (background, targets, design) of each inner split with P1 fitted on that
    split's inner-training rows; `build(b, extras)` is the arm's own matrices call."""
    mats = []
    for b in bundles:
        per_split = []
        for offsets in inner_split_offsets(b, stations, depth, truth - b['base'].astype(float)):
            extras = base_extras.copy()
            extras[:, 3:6] = offsets
            per_split.append(build(b, extras))
        mats.append(per_split)
    return mats


INNER_PRIOR_OBJECTIVE = ('mean over checkpoints of inner cast-macro normalized MAE, each inner split scored '
                         'against P1 fitted on its inner-training rows only')


def run_inner_prior(quick=False):
    """Addendum 71: P1's shared study on the inner-split objective, then the published refit."""
    tag = 'P1_placebo_direct'
    output = ROOT / 'results'
    folder = output / f'v2_field_held_site_{tag}_innerprior'
    folder.mkdir(parents=True, exist_ok=True)
    use_base, use_design, aou = ARMS[ARM_BASE]
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, excluded = prepare('held_site', cfg, quick)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(
        pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
        on=['cast_id', 'depth_m'], validate='one_to_one')
    base_extras = context[EXTRAS].to_numpy(dtype=float)
    stations = data.station.to_numpy()
    casts = data.cast_id.to_numpy()
    truth = data[TARGETS].to_numpy(float)
    depth = data.depth_m.to_numpy(float)
    build = lambda b, e: matrices(b, e, use_base, use_design, aou)
    mats = []
    for b in bundles:
        mask = np.zeros(len(depth), dtype=bool)
        mask[b['tr']] = True
        extras = base_extras.copy()
        extras[:, 3:6] = placebo_offsets('P1', depth, mask, truth - b['base'].astype(float), None)
        mats.append(build(b, extras))
    split_mats = inner_split_mats(bundles, stations, depth, truth, base_extras, build)

    # gate GA1 (amendment 1): with every split given the outer offsets, the objective at the published
    # penalty is addendum 48's reproduction of it; the published best value (trial 1) is not reproducible
    # by its own code and is recorded beside it
    published = json.loads((output / f'v2_field_held_site_{tag}' / f'held_site_{tag}_selection.json').read_text())
    a48 = json.loads((ROOT / 'reports/bay_twin/runs/v2_perfold_rest/field/published' / tag / 'report.json')
                     .read_text())['objective_here']
    a_pub = published['params']['alpha']
    outer_only = float(np.mean([inner_score(b, m, a_pub, aou, stations, casts, truth,
                                            split_mats=[m] * len(sm))
                                for b, m, sm in zip(bundles, mats, split_mats)]))
    at_pub = float(np.mean([inner_score(b, m, a_pub, aou, stations, casts, truth, split_mats=sm)
                            for b, m, sm in zip(bundles, mats, split_mats)]))
    ga1 = {'reference': 'runs/v2_perfold_rest/field/published/P1_placebo_direct/report.json objective_here',
           'reference_value': a48, 'outer_offsets_in_every_split': outer_only,
           'abs_diff': abs(outer_only - a48), 'pass': bool(abs(outer_only - a48) <= 1e-12),
           'published_best_value': published['best_value'],
           'published_best_value_abs_diff': abs(outer_only - published['best_value']),
           'inner_split_objective_at_published_penalty': at_pub,
           'amended_by': 'config/bay_reserved_protocol_v2_addendum71_amendment1.json'}
    print('GA1', ga1, flush=True)
    if not quick and not ga1['pass']:
        raise SystemExit('gate GA1 failed: the outer offsets do not reproduce the objective at the published penalty')
    save(folder / 'contract.json', {
        'protocol': PROTOCOL['version'], 'arm': tag, 'smoke_only': quick,
        'declared_in': 'config/bay_reserved_protocol_v2_addendum71.json',
        'differs_from_published_by': 'the objective only: each inner split reads P1 fitted on its inner-training '
                                     'rows; the refit and the scored rows are as published',
        'knots': KNOTS, 'folds': contracts, 'excluded_casts': excluded,
        'search_space': 'ridge alpha 1e-1 to 1e8 (log)', 'objective': INNER_PRIOR_OBJECTIVE,
        'selection': 'shared: one penalty for both outer folds, as published', 'no_2026': True})

    def objective(trial):
        alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
        return float(np.mean([inner_score(b, m, alpha, aou, stations, casts, truth, split_mats=sm)
                              for b, m, sm in zip(bundles, mats, split_mats)]))

    sel = run_study(f'held_site_{tag}_innerprior', objective, folder, quick)
    rows = []
    for b, (bs, y, feats) in zip(bundles, mats):
        pred, _, _ = fit_predict(feats, bs, y, b['tr'], b['va'], np.asarray(b['weights'], float),
                                 sel['params']['alpha'], aou)
        rows.append(prediction_frame(b, pred))
    d = pd.concat(rows, ignore_index=True)
    path = folder / 'selected_predictions.csv.gz'
    d.to_csv(path, index=False, lineterminator='\n', compression={'method': 'gzip', 'mtime': 0})
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum71.json', 'arm': tag,
              'objective': INNER_PRIOR_OBJECTIVE, 'knots': KNOTS, 'selection': sel,
              'nested_mae': nested_mae(d), 'gate_GA1': ga1,
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(folder / 'report.json', report)
    print(f"{tag} inner-split: alpha {sel['params']['alpha']:.4g}  "
          + '  '.join(f'{t} {v:.4f}' for t, v in report['nested_mae'].items()), flush=True)
    return report


def run(arm, quick=False):
    tag = f'{arm}_placebo_direct'
    output = ROOT / 'results'
    folder = output / f'v2_field_held_site_{tag}'
    folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_v2_field_held_site_{tag}.json'
    save(status, {'state': 'RUNNING', 'arm': tag, 'pid': os.getpid()})
    try:
        use_base, use_design, aou = ARMS[ARM_BASE]
        cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
        bundles, contracts, excluded = prepare('held_site', cfg, quick)
        data, _ = load_profiles()
        context = data[['cast_id', 'depth_m']].merge(
            pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
            on=['cast_id', 'depth_m'], validate='one_to_one')
        base_extras = context[EXTRAS].to_numpy(dtype=float)
        stations = data.station.to_numpy()
        casts = data.cast_id.to_numpy()
        truth = data[TARGETS].to_numpy(float)
        depth = data.depth_m.to_numpy(float)

        n_draws = 1 if arm == 'P1' else (1 if quick else DRAWS)
        rng = np.random.default_rng(SEED)
        results, first_mats = [], None

        for draw in range(n_draws):
            mats = []
            for b in bundles:
                mask = np.zeros(len(depth), dtype=bool)
                mask[b['tr']] = True
                # residual the offset has to explain: observation minus boundary-free background
                residual = truth - b['base'].astype(float)
                extras = base_extras.copy()
                extras[:, 3:6] = placebo_offsets(arm, depth, mask, residual, rng)
                mats.append(matrices(b, extras, use_base, use_design, aou))
            if draw == 0:
                first_mats = mats

                def objective(trial):
                    alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
                    return float(np.mean([inner_score(b, m, alpha, aou, stations, casts, truth)
                                          for b, m in zip(bundles, first_mats)]))

                sel = run_study(f'held_site_{tag}', objective, folder, quick)
                alpha = sel['params']['alpha']

            rows = []
            for b, (bs, y, feats) in zip(bundles, mats):
                pred, _, _ = fit_predict(feats, bs, y, b['tr'], b['va'],
                                         np.asarray(b['weights'], float), alpha, aou)
                rows.append(prediction_frame(b, pred))
            d = pd.concat(rows, ignore_index=True)
            results.append(nested_mae(d))
            if draw == 0:
                path = folder / 'selected_predictions.csv.gz'
                d.to_csv(path, index=False)

        frame = pd.DataFrame(results)
        report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum19.json',
                  'arm': tag, 'knots': KNOTS, 'draws': n_draws, 'selection': sel,
                  'copernicus_offset_sd_degC': COPERNICUS_SD,
                  'per_draw_nested_mae': results,
                  'summary': {t: {'mean': float(frame[t].mean()), 'min': float(frame[t].min()),
                                  'max': float(frame[t].max()),
                                  'sd': float(frame[t].std(ddof=1)) if n_draws > 1 else 0.0}
                              for t in TARGETS},
                  'prediction_sha256': hashlib.sha256(
                      (folder / 'selected_predictions.csv.gz').read_bytes()).hexdigest()}
        save(folder / 'report.json', report)
        save(status, {'state': 'COMPLETED', 'arm': tag})

        print(f'{tag}: alpha {alpha:.4g}, {n_draws} draw(s)')
        for t in TARGETS:
            s = report['summary'][t]
            print(f"   {t:14s} mean {s['mean']:.4f}  min {s['min']:.4f}  max {s['max']:.4f}"
                  f"  sd {s['sd']:.4f}")
    except BaseException as e:
        save(status, {'state': 'FAILED', 'arm': tag, 'error': str(e)})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['P1', 'P2'], required=True)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--inner-prior', action='store_true',
                   help='P1: each inner split scored against P1 fitted on its inner-training rows (addendum 71)')
    a = p.parse_args()
    if a.inner_prior:
        if a.arm != 'P1':
            raise SystemExit('--inner-prior applies to P1 only: P2 reads no label')
        run_inner_prior(a.quick)
    else:
        run(a.arm, a.quick)
