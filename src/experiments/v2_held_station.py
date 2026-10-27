"""Addendum-10: what a newly instrumented SITE inherits, with the co-location leak closed.

Addendum 7 refitted on every other sensor and scored the held-out one. Six probe pairs sit at
0.00 m separation, so that protocol usually leaves the held probe's zero-separation twin in the
training set, and the neighbour-context feature is an adjacency-weighted mean that includes the
twin's current reading. Chapter 3 states the right principle for the spatial workflow, which holds
out whole stations and excludes a 250 m radius; the forecast workflow never applied it.

This applies it. The unit of holdout is the site, every probe at the held site is removed from
training, and for the held probes the neighbour feature is recomputed with every probe within
250 m zeroed out.

What is deliberately not withheld is the held probe's own causal anchor at issuance: this is a
forecasting task, and a probe being forecast always has its own history. That is what bounds the
penalty, and it is declared in the protocol rather than offered afterwards.

Declared in config/bay_reserved_protocol_v2_addendum10.json.
Usage: python -m src.experiments.v2_held_station --variable temperature
"""
import argparse
import hashlib
import json
import zlib

import numpy as np
import pandas as pd

from src.data.bay_forcing import weather
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_closed_loop import selected
from src.experiments.v2_forecast import make_tree
from src.experiments.v2_horizon import ROOT, build, folds, climatology

OUT = ROOT / 'reports/bay_twin/runs/v2_held_station'
DRAWS = 600
MASK_RADIUS_M = 250.0


def geometry(names):
    """Site label and pairwise distance for the probes, in the order the data uses."""
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    meta = meta.loc[list(names)]
    xy = meta[['x_m', 'y_m']].to_numpy(dtype=float)
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    return meta.site.to_numpy(), distance


def neighbour_masked(data, fold, held, distance):
    """The adjacency-weighted neighbour feature, with probes within 250 m of `held` removed.

    Mirrors bay_dynamics_tree.features exactly, then rebuilds only the held columns under the
    mask, so every other column is bit-identical to the unmasked design.
    """
    x = np.where(np.isfinite(data['x']), data['x'], fold['fallback'])
    adjacency = data['adjacency'].copy()
    # For each held probe, drop every source inside the radius (this removes its own twins).
    for t in held:
        adjacency[t, distance[t] <= MASK_RADIUS_M] = 0.0
    k = adjacency[None] * data['available'][:, :, None] * data['available'][:, None, :]
    neighbour = np.einsum('bij,bj->bi', k, x) / np.maximum(k.sum(-1), 1e-6)
    return np.where(k.sum(-1) > 0, neighbour, x)


def fit_predict(data, design, fold, params, exclude):
    mask = data['valid'] & fold['eligible'][None, :]
    train_mask = mask.copy()
    if exclude is not None:
        train_mask[:, exclude] = False
    tr, va = fold['tr'], fold['va']
    tm = train_mask[tr]
    if tm.sum() < 200:
        return None
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]
    w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})
    model.fit(design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    grid = np.full(data['x'][va].shape, np.nan)
    vm = mask[va]
    grid[vm] = model.predict(design[va][vm]) + data['x'][va][vm]
    return grid


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def paired(frame, x, y, label):
    rng = np.random.default_rng([20260921, zlib.crc32(label.encode())])
    keys = frame[['fold', 'sensor']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in frame.groupby(['fold', 'sensor'])}
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(keys[i])] for i in rng.integers(0, len(keys), len(keys))])
        draws.append(macro(s, x) - macro(s, y))
    return [macro(frame, x) - macro(frame, y), *np.percentile(draws, [2.5, 97.5]).tolist()]


def run(variable, horizon=6):
    OUT.mkdir(parents=True, exist_ok=True)
    data = build(variable, horizon)
    outer = folds(data, horizon)
    met = weather(data['times']).to_numpy()
    params = selected(variable, 'weather_tree', horizon)
    sites, distance = geometry(data['names'])
    n = len(data['names'])

    rows, coverage = [], {}
    for fold in outer:
        base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
        met_block = np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))
        design_full = np.concatenate([base, met_block], axis=-1)
        clim, _ = climatology(data, fold)
        full = fit_predict(data, design_full, fold, params, exclude=None)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]

        for site in pd.unique(sites):
            held = np.flatnonzero((sites == site) & fold['eligible'])
            if not len(held):
                continue
            # Rebuild the design with the held site's neighbour columns masked.
            masked = base.copy()
            masked[:, :, 1] = neighbour_masked(data, fold, held, distance)
            design = np.concatenate([masked, met_block], axis=-1)
            predicted = fit_predict(data, design, fold, params, exclude=held)
            if predicted is None:
                continue
            dropped = int((distance[held[0]] <= MASK_RADIUS_M).sum())
            coverage[site] = {'probes_at_site': int(len(held)),
                              'probes_masked_within_250m': dropped,
                              'nearest_surviving_m': float(
                                  distance[held[0]][distance[held[0]] > MASK_RADIUS_M].min())}
            for s in held:
                for j in np.flatnonzero(mask[:, s]):
                    i = fold['va'][j]
                    rows.append({
                        'fold': fold['number'], 'site': site, 'sensor': data['names'][s],
                        'time': str(data['times'][i]),
                        'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                        'all_sensor': float(full[j, s] * data['scale'] + data['center']),
                        'held_station': float(predicted[j, s] * data['scale'] + data['center']),
                        'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                        'climatology': float(clim[i, s] * data['scale'] + data['center'])})

    frame = pd.DataFrame(rows)
    path = OUT / f'held_station_{variable}_h{horizon}.csv.gz'
    frame.to_csv(path, index=False)

    arms = ['all_sensor', 'held_station', 'persistence', 'climatology']
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum10.json', 'variable': variable,
              'horizon_hours': horizon, 'params': params, 'no_new_selection': True,
              'mask_radius_m': MASK_RADIUS_M, 'sites': int(frame.site.nunique()),
              'sensors': int(frame.sensor.nunique()), 'endpoints': int(len(frame)),
              'site_geometry': coverage,
              'mae': {a: macro(frame, a) for a in arms},
              'contrasts': {
                  'held_station - all_sensor': paired(frame, 'held_station', 'all_sensor', f'{variable}|hs-a'),
                  'held_station - persistence': paired(frame, 'held_station', 'persistence', f'{variable}|hs-p'),
                  'held_station - climatology': paired(frame, 'held_station', 'climatology', f'{variable}|hs-c')},
              'per_site': {}}
    for site, group in frame.groupby('site'):
        e = {a: float((group[a] - group.truth).abs().mean()) for a in arms}
        report['per_site'][site] = {'endpoints': int(len(group)), 'mae': e,
                                    'penalty': e['held_station'] - e['all_sensor']}
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT / f'held_station_{variable}_h{horizon}_summary.json', report)

    print(f"== {variable} h{horizon}  sites={report['sites']}  sensors={report['sensors']}  "
          f"endpoints={report['endpoints']}")
    for a in arms:
        print(f"   {a:14s} {report['mae'][a]:.4f}")
    for k, (d, lo, hi) in report['contrasts'].items():
        print(f'   {k:30s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))
    worst = sorted(report['per_site'].items(), key=lambda kv: -kv[1]['penalty'])[:4]
    print('   largest per-site penalties:', [(k, round(v['penalty'], 4)) for k, v in worst])


def run_nested(variable, site, horizon=6, quick=False):
    """The held-site arm with its own selection, which never sees the held site.

    `run` reuses the configuration the all-sensor weather tree selected, which was chosen with the
    held site in its inner data. Here each (fold, held site) gets its own study on the published
    ratio objective, scored on the inner window over the other sites only, with the held site
    removed from inner training and its neighbour columns masked exactly as in the outer fit.
    One site per call, so the sites run as separate jobs; `combine_nested` joins them.
    """
    from src.experiments.v2_forecast import tree_space
    from src.experiments.v2_horizon import inner_fold, ratio, rows_from
    from src.tuning.protocol_v2 import run_study
    OUT.mkdir(parents=True, exist_ok=True)
    data = build(variable, horizon)
    outer = folds(data, horizon)
    met = weather(data['times']).to_numpy()
    sites, distance = geometry(data['names'])
    n = len(data['names'])
    met_block = np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))
    rows, selections = [], {}
    for fold in outer:
        held = np.flatnonzero((sites == site) & fold['eligible'])
        if not len(held):
            continue
        inner = inner_fold(data, fold, horizon)
        inner_clim = climatology(data, inner)
        base_inner, _ = features(data, {'fallback': inner['fallback']}, 'forecast')
        masked_inner = base_inner.copy()
        masked_inner[:, :, 1] = neighbour_masked(data, inner, held, distance)
        design_inner = np.concatenate([masked_inner, met_block], axis=-1)
        others = inner['eligible'].copy()
        others[held] = False
        if not others.any():
            # Oxygen, fold 1: the held site is the only one eligible in the inner window, so no
            # configuration can be chosen without it. Recorded rather than papered over.
            selections[str(fold['number'])] = {'not_selectable': 'no other site eligible in the inner window'}
            continue

        def objective(trial):
            p = tree_space(trial)
            scored = dict(inner, eligible=others)
            grid = fit_predict(data, design_inner, scored, p, exclude=held)
            if grid is None:
                raise ValueError('too few inner training rows')
            return ratio(rows_from(data, scored, grid, inner_clim))
        tag = f'held_station_{variable}_h{horizon}_nested_{site}_f{fold["number"]}'
        sel = run_study(tag, objective, OUT / ('nested_studies_smoke' if quick else 'nested_studies'), quick)
        params = {**sel['params'], 'subsample_freq': 1}
        selections[str(fold['number'])] = sel

        base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
        masked = base.copy()
        masked[:, :, 1] = neighbour_masked(data, fold, held, distance)
        predicted = fit_predict(data, np.concatenate([masked, met_block], axis=-1), fold, params,
                                exclude=held)
        if predicted is None:
            continue
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for s in held:
            for j in np.flatnonzero(mask[:, s]):
                i = fold['va'][j]
                rows.append({'fold': fold['number'], 'site': site, 'sensor': data['names'][s],
                             'time': str(data['times'][i]),
                             'held_station_nested': float(predicted[j, s] * data['scale']
                                                          + data['center'])})
    frame = pd.DataFrame(rows)
    # a smoke run writes where combine_nested's glob cannot pick it up
    dest = OUT / 'nested_smoke' if quick else OUT
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / f'held_station_{variable}_h{horizon}_nested_{site}.csv.gz', index=False)
    save(dest / f'held_station_{variable}_h{horizon}_nested_{site}_selection.json',
         {'variable': variable, 'site': site, 'objective': 'ratio, inner window, other sites only',
          'selections': selections, 'endpoints': int(len(frame))})
    print(f'{variable} {site}: {len(frame)} endpoints')


def combine_nested(variable, horizon=6):
    """Join the per-site nested predictions onto the published frame and contrast them."""
    published = pd.read_csv(OUT / f'held_station_{variable}_h{horizon}.csv.gz')
    parts = [pd.read_csv(p) for p in sorted(OUT.glob(f'held_station_{variable}_h{horizon}_nested_*.csv.gz'))]
    nested = pd.concat(parts, ignore_index=True)
    frame = published.merge(nested[['fold', 'sensor', 'time', 'held_station_nested']],
                            on=['fold', 'sensor', 'time'], how='left', validate='one_to_one')
    if frame.held_station_nested.isna().any():
        raise SystemExit(f'{int(frame.held_station_nested.isna().sum())} published rows have no nested prediction')
    path = OUT / f'held_station_{variable}_h{horizon}_nested.csv.gz'
    frame.to_csv(path, index=False)
    print(variable, {a: round(macro(frame, a), 4) for a in
                     ('all_sensor', 'held_station', 'held_station_nested', 'persistence')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--nested-site', default=None)
    p.add_argument('--combine', action='store_true')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.combine:
        combine_nested(a.variable, a.horizon)
    elif a.nested_site:
        run_nested(a.variable, a.nested_site, a.horizon, a.quick)
    else:
        run(a.variable, a.horizon)
