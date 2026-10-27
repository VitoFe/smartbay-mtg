"""Tuning protocol v2 for the held-station spatial field (Field-57) and its Copernicus decomposition.

Arms: where the coarse boundary product enters (background offset and/or design regressors) x
oxygen target space (direct DO or AOU). Each arm gets its own >=100-trial TPE study over the ridge
penalty. Selection is nested: the objective is an inner GroupKFold over the
*training* stations of each outer fold, never touching the held station or its selection window.
The A1 direct arm is the released recipe; its selected penalty is the v2 release penalty.
"""
import argparse, hashlib, json, os, traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from src.experiments.bay_profile_field import ROOT, SCALE, prepare, prediction_frame
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_twin_sweep import save
from src.data.bay_profile_context import load_profiles, TARGETS
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.tuning.protocol_v2 import run_study, PROTOCOL

# arm -> (boundary offset in background, boundary values as regressors, oxygen in AOU space)
ARMS = {f'{a}_{space}': (ub, ud, space == 'aou')
        for a, (ub, ud) in {'A1_both': (True, True), 'A2_none': (False, False),
                            'A3_design_only': (False, True), 'A4_base_only': (True, False)}.items()
        for space in ('direct', 'aou')}


def matrices(b, extras, use_base, use_design, aou):
    base = b['base'].astype(float).copy()
    if use_base: base = base + extras[:, 3:6]
    base[:, 0] = np.clip(base[:, 0], -2, 40); base[:, 1] = np.clip(base[:, 1], 0, 42)
    y = b['y'].astype(float).copy()
    if aou:
        y[:, 2] = solubility(y[:, 0], y[:, 1]) - y[:, 2]
        base[:, 2] = solubility(base[:, 0], base[:, 1]) - base[:, 2]
    return base, y, (np.concatenate([b['sx'], extras], axis=1) if use_design else b['sx'])


def fit_predict(feats, base, y, tr, va, weights, alpha, aou):
    scaler = StandardScaler().fit(feats[tr]); design = scaler.transform(feats).astype('float32')
    model = Ridge(alpha=alpha, fit_intercept=False).fit(design[tr], ((y - base) / SCALE)[tr], sample_weight=weights)
    pred = base[va] + model.predict(design[va]) * SCALE
    pred[:, 0] = np.clip(pred[:, 0], -2, 40); pred[:, 1] = np.clip(pred[:, 1], 0, 42)
    if aou: pred[:, 2] = solubility(pred[:, 0], pred[:, 1]) - pred[:, 2]
    return pred, model, scaler


def inner_splits(b, stations):
    """The inner GroupKFold splits of a checkpoint's training rows by station, in inner_score's
    order, as (inner training, inner validation) positions into b['tr']."""
    groups = stations[b['tr']]
    k = min(4, len(np.unique(groups)))
    if k < 2: raise ValueError('fewer than two training stations')
    return list(GroupKFold(n_splits=k).split(b['tr'], groups=groups))


def inner_score(b, mats, alpha, aou, stations, casts, truth, split_mats=None):
    """GroupKFold over the outer fold's training stations; cast-macro normalized error in DO space.

    `split_mats`, if given, holds one (background, targets, design) per inner split, in
    inner_splits' order, and is used in that split in place of `mats`: a cast prior rebuilt without
    the split's validation stations (review R3, v2_field_ctdprior.inner_split_priors). Without it
    every split reads `mats`, as always."""
    base, y, feats = mats; tr = b['tr']; w = np.asarray(b['weights'], float)
    splits = inner_splits(b, stations)
    if split_mats is not None and len(split_mats) != len(splits):
        raise ValueError(f'{len(split_mats)} split matrices for {len(splits)} inner splits')
    scores = []
    for n, (itr, iva) in enumerate(splits):
        if split_mats is not None:
            base, y, feats = split_mats[n]
        pred, _, _ = fit_predict(feats, base, y, tr[itr], tr[iva], w[itr], alpha, aou)
        parts = []
        for j, s in enumerate(SCALE):
            e = pd.Series(np.abs(pred[:, j] - truth[tr[iva], j])).groupby(casts[tr[iva]]).mean()
            parts.append(e.mean() / s)
        scores.append(np.mean(parts))
    return float(np.mean(scores))


def nested_mae(d):
    """Casts -> stations -> folds macro average."""
    out = {}
    for t in TARGETS:
        z = d[['fold', 'station', 'cast_id']].assign(error=(d[t] - d['pred_' + t]).abs())
        out[t] = float(z.groupby(['fold', 'station', 'cast_id']).error.mean().groupby(['fold', 'station']).mean().groupby('fold').mean().mean())
    return out


def run(task, arm, quick=False):
    output = ROOT / 'results'; folder = output / f'v2_field_{task}_{arm}'; folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_v2_field_{task}_{arm}.json'
    save(status, {'state': 'RUNNING', 'task': task, 'arm': arm, 'pid': os.getpid()})
    try:
        use_base, use_design, aou = ARMS[arm]
        cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
        bundles, contracts, excluded = prepare(task, cfg, quick)
        data, _ = load_profiles()
        context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                    on=['cast_id', 'depth_m'], validate='one_to_one')
        extras = context[EXTRAS].to_numpy(dtype=float)
        stations = data.station.to_numpy(); casts = data.cast_id.to_numpy(); truth = data[TARGETS].to_numpy(float)
        mats = [matrices(b, extras, use_base, use_design, aou) for b in bundles]
        save(folder / 'contract.json', {'protocol': PROTOCOL['version'], 'task': task, 'arm': arm, 'smoke_only': quick,
             'boundary_in_background': use_base, 'boundary_in_design': use_design, 'oxygen_space': 'aou' if aou else 'direct',
             'folds': contracts, 'excluded_casts': excluded, 'search_space': 'ridge alpha 1e-1 to 1e8 (log)',
             'selection': PROTOCOL['selection']['comparisons'] + ' Inner GroupKFold (k<=4) over training stations; one alpha shared across outer folds.',
             'objective': 'mean over outer folds of inner cast-macro normalized MAE (T/S/DO scaled), DO scored in DO space', 'no_2026': True})

        def objective(trial):
            alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
            return float(np.mean([inner_score(b, m, alpha, aou, stations, casts, truth) for b, m in zip(bundles, mats)]))
        sel = run_study(f'{task}_{arm}', objective, folder, quick)
        rows = []
        for b, (base, y, feats) in zip(bundles, mats):
            pred, model, scaler = fit_predict(feats, base, y, b['tr'], b['va'], np.asarray(b['weights'], float), sel['params']['alpha'], aou)
            rows.append(prediction_frame(b, pred))
        d = pd.concat(rows, ignore_index=True); path = folder / 'selected_predictions.csv.gz'; d.to_csv(path, index=False)
        save(folder / 'report.json', {'selection': sel, 'nested_mae': nested_mae(d), 'rows': len(d), 'casts': int(d.cast_id.nunique()),
                                      'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        save(status, {'state': 'COMPLETED', 'task': task, 'arm': arm,
                      'result_sha256': {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'task': task, 'arm': arm, 'error': str(e)}); traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', choices=['held_site', 'temporal'], required=True)
    p.add_argument('--arm', choices=list(ARMS), required=True); p.add_argument('--quick', action='store_true')
    a = p.parse_args(); run(a.task, a.arm, a.quick)
