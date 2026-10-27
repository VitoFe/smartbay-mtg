"""Addendum-14 (E1): is the 0.190 degC the Copernicus product, or vertical structure as such?

The released background is the nearest valid moored observation plus a depth-dependent offset,
defined in scripts/build_bay_boundary_context.py as the Copernicus profile value at the cast depth
minus its value at 3 m. This replaces that offset, and only that offset, with the same quantity
measured by this project's own CTD archive.

    A6  month x 1 m depth-bin mean of (value at depth - value at 3 m)
    A7  the same per depth bin as an annual harmonic, which fills months a fold never trains on

Both are built per outer fold from casts at that fold's TRAINING stations and strictly at or
before its last training timestamp, so neither the held station nor the scoring period contributes
to the prior used to score it. The climatology draws on the full 722-cast archive rather than the
121-cast Gen2 subset, which is the point: the archive reaches back to 2021 and forward past
2026-01-16, where the Copernicus extraction stops.

Declared in config/bay_reserved_protocol_v2_addendum14.json.
Usage: python -m src.experiments.v2_field_ctdprior --arm A6

`--inner-prior` (review R3, 2026-10-05). The prior above is built once per checkpoint from all its
training stations, and the penalty is chosen by an inner GroupKFold over those same stations, so each
inner-validation station is scored against a prior that contains its own casts. With the switch,
each inner split gets the prior rebuilt from the archive casts of the checkpoint's training stations
other than that split's validation stations, at or before the checkpoint's cutoff, with the
background and design rebuilt from it (`inner_split_priors`, computed once per checkpoint and split:
the prior does not depend on the penalty); the selected penalty is then refitted on the whole outer
training set with the prior from all its training stations, as before. The selection stays shared
(one penalty for both folds), as published. Writes `results/v2_field_held_site_<arm>_ctdprior_direct_innerprior/`;
the published folder is not touched.
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

REFERENCE_DEPTH_M = 3.0
BIN_M = 1.0
ARM_BASE = 'A1_both_direct'


def archive():
    """The full quality-controlled cast archive, with a 3 m reference delta per row."""
    d = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz')
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    d = d[(d.qc_flag == 0) & np.isfinite(d[TARGETS]).all(axis=1)]
    d = d[d.temp_c.between(-2, 40) & d.salinity_psu.between(0, 42) & d.do_umol_l.between(0, 600)]
    parts = []
    for cast, g in d.groupby('cast_id', sort=False):
        g = g.sort_values('depth_m')
        if len(g) < 2:
            continue
        row = g.copy()
        for t in TARGETS:
            reference = float(np.interp(REFERENCE_DEPTH_M, g.depth_m.to_numpy(), g[t].to_numpy()))
            row['delta_' + t] = g[t].to_numpy() - reference
        parts.append(row)
    out = pd.concat(parts, ignore_index=True)
    out['bin'] = np.floor(out.depth_m / BIN_M).astype(int)
    out['month'] = out.t_utc.dt.month
    out['doy'] = out.t_utc.dt.dayofyear
    return out


def month_depth_prior(train, query_bin, query_month):
    """A6: mean delta per (month, bin), with a bin-only fallback then zero."""
    cell = train.groupby(['month', 'bin'])[['delta_' + t for t in TARGETS]].mean()
    bin_only = train.groupby('bin')[['delta_' + t for t in TARGETS]].mean()
    out = np.zeros((len(query_bin), 3))
    fell_back = np.zeros(len(query_bin), dtype=int)
    for i, (b, m) in enumerate(zip(query_bin, query_month)):
        if (m, b) in cell.index:
            out[i] = cell.loc[(m, b)].to_numpy()
        elif b in bin_only.index:
            out[i] = bin_only.loc[b].to_numpy(); fell_back[i] = 1
        else:
            fell_back[i] = 2
    return out, fell_back


def harmonic_prior(train, query_bin, query_doy):
    """A7: per-bin least squares on [1, sin, cos] of day-of-year; bin-mean fallback."""
    def design(doy):
        a = 2 * np.pi * np.asarray(doy, dtype=float) / 365.25
        return np.column_stack([np.ones(len(a)), np.sin(a), np.cos(a)])

    fits, bin_only = {}, train.groupby('bin')[['delta_' + t for t in TARGETS]].mean()
    for b, g in train.groupby('bin'):
        if len(g) < 6:
            continue
        X = design(g.doy.to_numpy())
        Y = g[['delta_' + t for t in TARGETS]].to_numpy()
        fits[b] = np.linalg.lstsq(X, Y, rcond=None)[0]
    out = np.zeros((len(query_bin), 3))
    fell_back = np.zeros(len(query_bin), dtype=int)
    Q = design(query_doy)
    for i, b in enumerate(query_bin):
        if b in fits:
            out[i] = Q[i] @ fits[b]
        elif b in bin_only.index:
            out[i] = bin_only.loc[b].to_numpy(); fell_back[i] = 1
        else:
            fell_back[i] = 2
    return out, fell_back


def cutoff_of(contract):
    cutoff = pd.Timestamp(contract['last_training'])
    return cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff


def inner_split_priors(b, contract, full, stations, prior):
    """The prior of each inner split of one checkpoint, in v2_field.inner_splits' order (review R3).

    `prior(train)` maps archive rows to (delta per query row, fallback code per row). Each split's
    prior uses the archive casts of the checkpoint's training stations less that split's validation
    stations, at or before the checkpoint's last training time, so no inner-validation station
    contributes to the prior it is scored against. Returns [(delta, coverage)]."""
    from src.experiments.v2_field import inner_splits
    cutoff = cutoff_of(contract)
    training = set(contract['training_stations'])
    out = []
    for n, (itr, iva) in enumerate(inner_splits(b, stations)):
        held = set(stations[b['tr'][iva]])
        keep = training - held
        train = full[full.station.isin(keep) & (full.t_utc <= cutoff)]
        delta, fell = prior(train)
        out.append((delta, {'split': n, 'validation_stations': sorted(held), 'prior_stations': sorted(keep),
                            'prior_stations_are_inner_training_stations': keep == set(stations[b['tr'][itr]]),
                            'climatology_casts': int(train.cast_id.nunique()), 'climatology_rows': int(len(train)),
                            'fallback_bin_only': int((fell == 1).sum()), 'fallback_zero': int((fell == 2).sum())}))
    return out


def inner_split_mats(bundles, contracts, full, stations, prior, base_extras, build):
    """Per checkpoint, the (background, targets, design) of each inner split with the prior rebuilt
    without that split's validation stations (inner_split_priors), and each split's coverage.
    `base_extras` are the published extras, whose prior columns 3:6 are replaced; `build(b, extras)`
    is the arm's own matrices call."""
    mats, coverage = [], []
    for b, contract in zip(bundles, contracts):
        per_split, info = [], []
        for delta, c in inner_split_priors(b, contract, full, stations, prior):
            extras = base_extras.copy()
            extras[:, 3:6] = delta
            per_split.append(build(b, extras))
            info.append(c)
        mats.append(per_split)
        coverage.append(info)
    return mats, coverage


def save_lf(path, obj):
    """JSON with LF line endings whatever the host."""
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


GZIP = {'method': 'gzip', 'mtime': 0}
INNER_PRIOR_OBJECTIVE = ('mean over outer folds of inner cast-macro normalized MAE, each inner split scored '
                         'against a prior rebuilt without its validation stations')


def _run_inner_prior(tag, arm, folder, status, quick, bundles, contracts, excluded, mats, split_mats, coverage,
                     aou, stations, casts, truth):
    """run() with the inner-split prior: the shared study on the corrected objective, then the
    refit at the selected penalty with each checkpoint's full-training-station prior (`mats`)."""
    save_lf(folder / 'contract.json', {
        'protocol': PROTOCOL['version'], 'arm': tag, 'smoke_only': quick,
        'declared_in': 'config/bay_reserved_protocol_v2_addendum14.json',
        'review': 'R3 (2026-10-05): the prior rebuilt per inner split without its validation stations',
        'differs_from_published_by': 'the objective only: each inner split reads a prior built without the casts of '
                                     'its validation stations; the refit and the scored rows are as published',
        'reference_depth_m': REFERENCE_DEPTH_M, 'depth_bin_m': BIN_M,
        'climatology_coverage': coverage, 'folds': contracts, 'excluded_casts': excluded,
        'search_space': 'ridge alpha 1e-1 to 1e8 (log)', 'objective': INNER_PRIOR_OBJECTIVE,
        'selection': 'shared: one penalty for both outer folds, as published',
        'no_2026': True, 'release_candidate': False})

    def objective(trial):
        alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
        return float(np.mean([inner_score(b, m, alpha, aou, stations, casts, truth, split_mats=sm)
                              for b, m, sm in zip(bundles, mats, split_mats)]))

    sel = run_study(f'held_site_{tag}_innerprior', objective, folder, quick)
    rows = []
    for b, (bs, y, feats) in zip(bundles, mats):
        pred, _, _ = fit_predict(feats, bs, y, b['tr'], b['va'],
                                 np.asarray(b['weights'], float), sel['params']['alpha'], aou)
        rows.append(prediction_frame(b, pred))
    d = pd.concat(rows, ignore_index=True)
    path = folder / 'selected_predictions.csv.gz'
    d.to_csv(path, index=False, lineterminator='\n', compression=GZIP)
    report = {'selection': sel, 'arm': tag, 'objective': INNER_PRIOR_OBJECTIVE, 'nested_mae': nested_mae(d),
              'climatology_coverage': coverage,
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save_lf(folder / 'report.json', report)
    save_lf(status, {'state': 'COMPLETED', 'arm': tag})
    print('alpha', sel['params']['alpha'])
    print('nested MAE', {k: round(v, 6) for k, v in report['nested_mae'].items()})
    return report


def run(arm, quick=False, inner_prior=False):
    tag = f'{arm}_ctdprior_direct'
    output = ROOT / 'results'
    folder = output / f'v2_field_held_site_{tag}' if not inner_prior else \
        output / f'v2_field_held_site_{tag}_innerprior{"_smoke" if quick else ""}'
    folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_v2_field_held_site_{tag}.json' if not inner_prior else \
        folder / 'status.json'
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
        times = pd.to_datetime(data.t_utc, utc=True)
        query_bin = np.floor(data.depth_m.to_numpy() / BIN_M).astype(int)
        query_month = times.dt.month.to_numpy()
        query_doy = times.dt.dayofyear.to_numpy()

        full = archive()
        mats, coverage = [], []
        for b, contract in zip(bundles, contracts):
            train_stations = set(contract['training_stations'])
            cutoff = pd.Timestamp(contract['last_training'])
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize('UTC')
            train = full[full.station.isin(train_stations) & (full.t_utc <= cutoff)]
            if arm == 'A6':
                delta, fell = month_depth_prior(train, query_bin, query_month)
            else:
                delta, fell = harmonic_prior(train, query_bin, query_doy)
            extras = base_extras.copy()
            extras[:, 3:6] = delta
            mats.append(matrices(b, extras, use_base, use_design, aou))
            coverage.append({'fold': contract['fold'], 'held_station': contract['held_station'],
                             'climatology_casts': int(train.cast_id.nunique()),
                             'climatology_rows': int(len(train)),
                             'cutoff': str(cutoff),
                             'fallback_bin_only': int((fell == 1).sum()),
                             'fallback_zero': int((fell == 2).sum()),
                             'rows': int(len(fell))})

        if inner_prior:
            prior = ((lambda t: month_depth_prior(t, query_bin, query_month)) if arm == 'A6' else
                     (lambda t: harmonic_prior(t, query_bin, query_doy)))
            split_mats, split_coverage = inner_split_mats(bundles, contracts, full, stations, prior, base_extras,
                                                          lambda b, e: matrices(b, e, use_base, use_design, aou))
            for c, s in zip(coverage, split_coverage):
                c['inner_splits'] = s
            return _run_inner_prior(tag, arm, folder, status, quick, bundles, contracts, excluded, mats, split_mats,
                                    coverage, aou, stations, casts, truth)

        save(folder / 'contract.json', {
            'protocol': PROTOCOL['version'], 'arm': tag, 'smoke_only': quick,
            'declared_in': 'config/bay_reserved_protocol_v2_addendum14.json',
            'differs_from_A1_by': 'the three *_vertical_delta columns come from the CTD archive, '
                                  'not from Copernicus; everything else is identical',
            'reference_depth_m': REFERENCE_DEPTH_M, 'depth_bin_m': BIN_M,
            'climatology_coverage': coverage,
            'folds': contracts, 'excluded_casts': excluded,
            'search_space': 'ridge alpha 1e-1 to 1e8 (log)',
            'objective': 'mean over outer folds of inner cast-macro normalized MAE',
            'no_2026': True, 'release_candidate': False})

        def objective(trial):
            alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
            return float(np.mean([inner_score(b, m, alpha, aou, stations, casts, truth)
                                  for b, m in zip(bundles, mats)]))

        sel = run_study(f'held_site_{tag}', objective, folder, quick)
        rows = []
        for b, (bs, y, feats) in zip(bundles, mats):
            pred, _, _ = fit_predict(feats, bs, y, b['tr'], b['va'],
                                     np.asarray(b['weights'], float), sel['params']['alpha'], aou)
            rows.append(prediction_frame(b, pred))
        d = pd.concat(rows, ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)
        report = {'selection': sel, 'arm': tag, 'nested_mae': nested_mae(d),
                  'climatology_coverage': coverage,
                  'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        save(folder / 'report.json', report)
        save(status, {'state': 'COMPLETED', 'arm': tag})
        print('alpha', sel['params']['alpha'])
        print('nested MAE', {k: round(v, 6) for k, v in report['nested_mae'].items()})
        fb = sum(c['fallback_bin_only'] for c in coverage), sum(c['fallback_zero'] for c in coverage)
        print(f'fallback rows: bin-only {fb[0]}, zero {fb[1]}, of {sum(c["rows"] for c in coverage)}')
        print('climatology casts per fold:', [c['climatology_casts'] for c in coverage])
    except BaseException as e:
        save(status, {'state': 'FAILED', 'arm': tag, 'error': str(e)})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['A6', 'A7'], required=True)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--inner-prior', action='store_true',
                   help='rebuild the prior per inner split without its validation stations (new folder)')
    a = p.parse_args()
    run(a.arm, a.quick, a.inner_prior)
