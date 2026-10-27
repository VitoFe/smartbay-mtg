"""Addendum 73 part B: spatial intervals calibrated on other stations' earlier casts.

A causal split-conformal interval around each held-station estimate. For a query cast at station s on
day d the calibration pool is the absolute errors of the same arm's held-station estimates at every depth
row of casts at stations other than s made on days strictly before d; the interval is the estimate plus
or minus the conformal quantile ceil((n + 1)(1 - a)) / n of the pool. A query whose pool holds fewer than
five casts or fewer than two stations abstains. Variants:

  pooled          one pool for every depth
  depth_band      the pool of the query row's depth band (0-2, 2-5, 5-10, over 10 m) when that band holds
                  at least 30 rows, the pooled one otherwise
  depth_band_aci  depth_band at a level adapted online (Gibbs and Candes): after each day, the level moves
                  by 0.05 times the target miss rate minus the mean miss share of that day's casts, clipped
                  to [0.01, 0.5]; a day's casts all use the level reached before it

Cohorts: Field-57 (the per-fold released recipe, pool from earlier Field-57 casts) and Field-2026 (the
inner-split cast-prior arm A7o, pool from its development estimates and earlier 2026 casts).

Usage: python -m src.experiments.v2_spatial_conformal
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_spatial_conformal'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum73.json'
FIELD57 = RUNS / 'v2_perfold/field/perfold/A1_both_direct/selected_predictions.csv.gz'
DEV_A7O = RUNS / 'v2_field_reserved2026/innerprior/dev_A7o_predictions.csv.gz'
F2026 = RUNS / 'v2_field_reserved2026/innerprior/reserved2026_held_station_predictions.csv.gz'
TARGETS = ('temp_c', 'salinity_psu', 'do_umol_l')
LEVELS = (0.8, 0.9)
VARIANTS = ('pooled', 'depth_band', 'depth_band_aci')
BANDS = (0., 2., 5., 10., np.inf)
MIN_CASTS, MIN_STATIONS, MIN_BAND_ROWS, GAMMA = 5, 2, 30, 0.05


def band(depth):
    return np.searchsorted(BANDS, np.asarray(depth, float), side='right') - 1


def quantile(errors, a):
    """The conformal quantile: the ceil((n + 1)(1 - a))-th smallest error, the largest when that exceeds n."""
    e = np.sort(np.asarray(errors, float))
    k = math.ceil((len(e) + 1) * (1 - a))
    return float(e[min(k, len(e)) - 1])


def intervals(query, pool, coverage, variant):
    """Half-widths for every query row (NaN where the cast abstains). query and pool carry station, day,
    cast_id, depth_m and err (the pool's absolute errors; the query's are read only after its day)."""
    target = 1 - coverage
    a = target
    half = np.full(len(query), np.nan)
    pool_band = band(pool.depth_m)
    for day, today in query.groupby('day', sort=True):
        misses = []
        for cast, rows in today.groupby('cast_id'):
            station = rows.station.iloc[0]
            usable = (pool.day.to_numpy() < day) & (pool.station.to_numpy() != station)
            p = pool[usable]
            if p.cast_id.nunique() < MIN_CASTS or p.station.nunique() < MIN_STATIONS:
                continue
            level = a if variant == 'depth_band_aci' else target
            idx = rows.index.to_numpy()
            if variant == 'pooled':
                half[idx] = quantile(p.err, level)
            else:
                qb, pb = band(rows.depth_m), pool_band[usable]
                for b in np.unique(qb):
                    sub = p.err.to_numpy()[pb == b]
                    half[idx[qb == b]] = quantile(sub if len(sub) >= MIN_BAND_ROWS else p.err, level)
            misses.append(float(np.mean(rows.err.to_numpy() > half[idx])))
        if variant == 'depth_band_aci' and misses:
            a = float(np.clip(a + GAMMA * (target - np.mean(misses)), 0.01, 0.5))
    return half


def nested(frame, column):
    """Rows within cast, casts within (fold, station), stations within fold, folds: the errors' rule."""
    return float(frame.groupby(['fold', 'station', 'cast_id'])[column].mean().groupby(['fold', 'station']).mean()
                 .groupby('fold').mean().mean())


def residuals(frame, t, column=None):
    f = frame[['fold', 'station', 'cast_id', 't_utc', 'depth_m']].copy()
    f['day'] = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d')
    f['err'] = (frame[t] - frame[column or f'pred_{t}']).abs().to_numpy()
    return f.reset_index(drop=True)


def score(query, pool, t):
    out = {}
    for coverage in LEVELS:
        for variant in VARIANTS:
            half = intervals(query, pool, coverage, variant)
            q = query.assign(half=half)
            answered = q[np.isfinite(q.half)].copy()
            answered['covered'] = (answered.err <= answered.half).astype(float)
            answered['width'] = 2 * answered.half
            by_station = {s: round(float(g.groupby('cast_id').covered.mean().mean()), 4)
                          for s, g in answered.groupby('station')}
            out[f'{variant}|{int(coverage * 100)}'] = {
                'coverage_nested': round(nested(answered, 'covered'), 4),
                'coverage_rows': round(float(answered.covered.mean()), 4),
                'width_nested': round(nested(answered, 'width'), 4),
                'casts': int(query.cast_id.nunique()), 'casts_answered': int(answered.cast_id.nunique()),
                'abstained_share': round(1 - answered.cast_id.nunique() / query.cast_id.nunique(), 4),
                'coverage_by_station': by_station,
                'station_range': round(max(by_station.values()) - min(by_station.values()), 4) if by_station else None}
    return out


def report():
    f57 = pd.read_csv(FIELD57)
    dev = pd.read_csv(DEV_A7O)
    f26 = pd.read_csv(F2026)
    rep = {'declared_in': DECLARATION, 'part': 'B', 'levels': LEVELS, 'variants': VARIANTS,
           'rules': {'min_casts': MIN_CASTS, 'min_stations': MIN_STATIONS, 'min_band_rows': MIN_BAND_ROWS,
                     'gamma': GAMMA, 'bands_m': [0, 2, 5, 10]},
           'units': 'record units: degC, PSU, umol/kg for oxygen (coverage is unit-free; oxygen widths not printed)',
           'cohorts': {}}
    for t in TARGETS:
        q57 = residuals(f57, t)
        rep['cohorts'].setdefault('Field-57', {})[t] = score(q57, q57, t)
        q26 = residuals(f26, t, f'A7o|{t}').assign(fold=0)
        pool26 = pd.concat([residuals(dev, t), q26], ignore_index=True)
        rep['cohorts'].setdefault('Field-2026', {})[t] = score(q26, pool26, t)
    c = rep['cohorts']
    rep['predictions'] = {
        'PB1': 0.70 <= c['Field-57']['temp_c']['pooled|80']['coverage_nested'] <= 0.90,
        'PB2': c['Field-57']['temp_c']['pooled|80']['station_range'] > 0.2,
        'PB3': 0.70 <= c['Field-2026']['temp_c']['depth_band|80']['coverage_nested'] <= 0.90}
    usable = {}
    for t in TARGETS:
        usable[t] = [v for v in VARIANTS if all(
            0.75 <= c[co][t][f'{v}|80']['coverage_nested'] <= 0.85 and c[co][t][f'{v}|80']['abstained_share'] <= 0.25
            for co in ('Field-57', 'Field-2026'))]
    rep['decision'] = {'usable_variants': usable}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_bytes(json.dumps(rep, indent=1).encode('utf-8'))
    for co, block in c.items():
        for t, s in block.items():
            for k, v in s.items():
                print(f'{co:10s} {t:12s} {k:20s} cover {v["coverage_nested"]:.3f} rows {v["coverage_rows"]:.3f} '
                      f'width {v["width_nested"]:.3f} abstain {v["abstained_share"]:.2f} station range {v["station_range"]}')
    print('predictions', rep['predictions'], 'usable', usable)


if __name__ == '__main__':
    report()
