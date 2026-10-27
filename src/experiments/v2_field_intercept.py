"""Addendum-12 part B: arm A5, the released field recipe with a fitted global intercept.

Equation 4.1 fits Ridge(fit_intercept=False), so a constant offset in (y - background) is not
representable and the penalty running away to near-total shrinkage cannot distinguish "the
features carry nothing" from "the background is biased in a way this model class cannot remove".
A5 is A1_direct in every respect except that the intercept is fitted, with its own >=100-trial
nested study over the penalty.

The intercept is global, not per-station. Field-57 holds out whole stations, so no per-station
intercept exists for the station being scored; fitting one would be leakage. Part A measures the
per-station biases as a diagnostic and decides whether a global term could absorb them.

Nothing here is a release candidate. Declared in
config/bay_reserved_protocol_v2_addendum12.json.

Usage: python -m src.experiments.v2_field_intercept
"""
import hashlib
import json
import os
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.bay_profile_context import TARGETS, load_profiles
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_profile_field import ROOT, SCALE, prepare, prediction_frame
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_field import ARMS, matrices, nested_mae
from src.tuning.protocol_v2 import PROTOCOL, run_study

ARM = 'A1_both_direct'
TAG = 'A5_intercept_direct'


def fit_predict(feats, base, y, tr, va, weights, alpha):
    """A1's fit_predict with fit_intercept=True. Direct oxygen space, so no AOU inversion."""
    scaler = StandardScaler().fit(feats[tr])
    design = scaler.transform(feats).astype('float32')
    model = Ridge(alpha=alpha, fit_intercept=True).fit(
        design[tr], ((y - base) / SCALE)[tr], sample_weight=weights)
    pred = base[va] + model.predict(design[va]) * SCALE
    pred[:, 0] = np.clip(pred[:, 0], -2, 40)
    pred[:, 1] = np.clip(pred[:, 1], 0, 42)
    return pred, model, scaler


def inner_score(b, mats, alpha, stations, casts, truth):
    base, y, feats = mats
    tr = b['tr']
    w = np.asarray(b['weights'], float)
    groups = stations[tr]
    k = min(4, len(np.unique(groups)))
    if k < 2:
        raise ValueError('fewer than two training stations')
    scores = []
    for itr, iva in GroupKFold(n_splits=k).split(tr, groups=groups):
        pred, _, _ = fit_predict(feats, base, y, tr[itr], tr[iva], w[itr], alpha)
        parts = []
        for j, s in enumerate(SCALE):
            e = pd.Series(np.abs(pred[:, j] - truth[tr[iva], j])).groupby(casts[tr[iva]]).mean()
            parts.append(e.mean() / s)
        scores.append(np.mean(parts))
    return float(np.mean(scores))


def main(quick=False):
    output = ROOT / 'results'
    folder = output / f'v2_field_held_site_{TAG}'
    folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_v2_field_held_site_{TAG}.json'
    save(status, {'state': 'RUNNING', 'arm': TAG, 'pid': os.getpid()})
    try:
        use_base, use_design, aou = ARMS[ARM]
        assert (use_base, use_design, aou) == (True, True, False), ARMS[ARM]
        cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
        bundles, contracts, excluded = prepare('held_site', cfg, quick)
        data, _ = load_profiles()
        context = data[['cast_id', 'depth_m']].merge(
            pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
            on=['cast_id', 'depth_m'], validate='one_to_one')
        extras = context[EXTRAS].to_numpy(dtype=float)
        stations = data.station.to_numpy()
        casts = data.cast_id.to_numpy()
        truth = data[TARGETS].to_numpy(float)
        mats = [matrices(b, extras, use_base, use_design, aou) for b in bundles]

        save(folder / 'contract.json', {
            'protocol': PROTOCOL['version'], 'arm': TAG, 'smoke_only': quick,
            'declared_in': 'config/bay_reserved_protocol_v2_addendum12.json',
            'differs_from_A1_by': 'Ridge fit_intercept=True (global, fitted on training stations)',
            'boundary_in_background': use_base, 'boundary_in_design': use_design,
            'oxygen_space': 'direct', 'folds': contracts, 'excluded_casts': excluded,
            'search_space': 'ridge alpha 1e-1 to 1e8 (log)',
            'selection': PROTOCOL['selection']['comparisons'] +
                         ' Inner GroupKFold (k<=4) over training stations; one alpha shared across outer folds.',
            'objective': 'mean over outer folds of inner cast-macro normalized MAE', 'no_2026': True,
            'release_candidate': False})

        def objective(trial):
            alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
            return float(np.mean([inner_score(b, m, alpha, stations, casts, truth)
                                  for b, m in zip(bundles, mats)]))

        sel = run_study(f'held_site_{TAG}', objective, folder, quick)
        rows, intercepts = [], []
        for b, (base, y, feats) in zip(bundles, mats):
            pred, model, _ = fit_predict(feats, base, y, b['tr'], b['va'],
                                         np.asarray(b['weights'], float), sel['params']['alpha'])
            intercepts.append([float(v) * s for v, s in zip(np.atleast_1d(model.intercept_), SCALE)])
            rows.append(prediction_frame(b, pred))
        d = pd.concat(rows, ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)
        report = {'selection': sel, 'arm': TAG, 'nested_mae': nested_mae(d),
                  'fitted_intercepts_physical_units': intercepts,
                  'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        save(folder / 'report.json', report)
        save(status, {'state': 'COMPLETED', 'arm': TAG})
        print('alpha', sel['params']['alpha'])
        print('nested MAE', {k: round(v, 6) for k, v in report['nested_mae'].items()})
        print('intercepts (physical units, per fold):')
        for k, v in enumerate(intercepts):
            print('   fold', k, [round(x, 4) for x in v])
    except BaseException as e:
        save(status, {'state': 'FAILED', 'arm': TAG, 'error': str(e)})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
