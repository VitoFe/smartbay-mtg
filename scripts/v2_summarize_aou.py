"""Year-boundary comparison on the 38 casts of 2026: ridge, LightGBM and XGBoost, each fitted to
oxygen directly and to apparent utilisation (runs/v2_bottom_aou/aou and runs/v2_bottom_aou/aou_xgb predictions).

Metric: the nested macro of the spatial cohorts, depth bins within cast, casts within station,
stations (one chronological fold). Intervals: the paired bootstrap over cast *dates*, each drawn
date relabelled so a date drawn twice counts twice (2000 draws), and the delete-one-date
jackknife-t, as in src.experiments.v2_field_reserved2026.contrast; `contrast` below draws the same
dates from the same seed and returns the same numbers, by matrix algebra over (date, cell) sums
instead of concatenating frames, which is what makes the 240 dates of the near-bottom cohort
tractable. The 38 casts fall on 20 dates, so a contrast is called robust only where both exclude
zero (Chapter 4).

What the utilisation arms are (scripts/experiments/exp_spatial.py): the target is the
Garcia-Gordon solubility in umol/kg at the cast's own measured temperature and salinity at that
depth, minus the measured concentration in umol/L, and the prediction is inverted through the same
quantity. The direct arms see neither the measured temperature nor the measured salinity.
scripts/v2_aou_units.py reruns the ridge arms with the solubility in umol/L and with the direct arm
given the same two measurements.

Usage: python -m scripts.v2_summarize_aou
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.v2_field_reserved2026 import DRAWS, RNG_SEED, macro

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'reports/bay_twin/runs/v2_bottom_aou'
CASTS = ROOT / 'data/processed/ctd_all_casts.csv.gz'
COHORTS = ('gen2_164', 'full_722')
PAIRS = [(('M2_ridge_tel', 'aou'), ('M2_ridge_tel', 'direct_do')),
         (('M3_lightgbm', 'aou'), ('M3_lightgbm', 'direct_do')),
         (('M3b_xgboost', 'aou'), ('M3b_xgboost', 'direct_do')),
         (('M2_ridge_tel', 'aou'), ('M3_lightgbm', 'aou')),
         (('M2_ridge_tel', 'aou'), ('M3b_xgboost', 'aou')),
         (('M3b_xgboost', 'aou'), ('M3_lightgbm', 'aou'))]


def cast_meta():
    """Station and UTC calendar day of every cast, from the archive the predictions were made on."""
    c = pd.read_csv(CASTS, usecols=['cast_id', 'station', 't_utc'])
    g = c.groupby('cast_id').agg(station=('station', 'first'), t=('t_utc', 'first'))
    g['day'] = pd.to_datetime(g.t, utc=True).dt.floor('D')
    return g[['station', 'day']]


def cast_errors(frame, meta, fold):
    """Mean absolute error of each cast over its depth bins, keyed for the nested macro."""
    e = frame.assign(e=(frame.pred - frame.truth).abs()).groupby('cast_id', sort=False).e.mean()
    c = e.reset_index().join(meta, on='cast_id')
    if c.station.isna().any():
        raise SystemExit('a predicted cast is missing from the archive')
    return c.assign(fold=fold)[['fold', 'station', 'cast_id', 'e', 'day']]


def nested_rows(sums, counts, fold_of_cell):
    """Nested macro of every row of (draws x cells) sums and counts: cell means, their mean within
    each fold over the cells present, then the mean over the folds present."""
    with np.errstate(invalid='ignore', divide='ignore'):
        cell = sums / counts
    folds = []
    for f in np.unique(fold_of_cell):
        block = cell[:, fold_of_cell == f]
        present = np.isfinite(block)
        n = present.sum(axis=1)
        folds.append(np.where(n > 0, np.where(present, block, 0).sum(axis=1) / np.maximum(n, 1), np.nan))
    return np.nanmean(np.column_stack(folds), axis=1)


def contrast(ca, cb, draws=DRAWS):
    """a - b over cast-level errors: nested macro point, the bootstrap over dates with each drawn
    date relabelled, and the delete-one-date jackknife-t. Same draws and same output as
    src.experiments.v2_field_reserved2026.contrast."""
    j = ca.merge(cb[['fold', 'station', 'cast_id', 'e']], on=['fold', 'station', 'cast_id'],
                 suffixes=('_a', '_b'))
    point = macro(j.rename(columns={'e_a': 'e'})) - macro(j.rename(columns={'e_b': 'e'}))
    days = j.day.unique()
    n = len(days)
    day_id = pd.Index(days).get_indexer(j.day)
    cells = pd.MultiIndex.from_frame(j[['fold', 'station']])
    cell_id, uniq = pd.factorize(cells)
    fold_of_cell = pd.factorize(uniq.get_level_values(0))[0]
    shape = (n, len(uniq))
    sa, sb, cnt = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(sa, (day_id, cell_id), j.e_a.to_numpy(float))
    np.add.at(sb, (day_id, cell_id), j.e_b.to_numpy(float))
    np.add.at(cnt, (day_id, cell_id), 1.0)

    def diff(weights):
        c = weights @ cnt
        return nested_rows(weights @ sa, c, fold_of_cell) - nested_rows(weights @ sb, c, fold_of_cell)
    rng = np.random.default_rng(RNG_SEED)
    weights = np.zeros((draws, n))
    for i in range(draws):
        pick = rng.choice(np.arange(n), n, replace=True)
        weights[i] = np.bincount(pick, minlength=n)
    boot = diff(weights)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    jack = diff(1.0 - np.eye(n))
    se = np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))
    t = stats.t.ppf(.975, n - 1)
    return {'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0),
            'jackknife_t': [round(float(point - t * se), 4), round(float(point + t * se), 4)],
            'jackknife_excludes_zero': bool(point - t * se > 0 or point + t * se < 0), 'dates': int(n)}


def summarise(c):
    return {'diff': c['point'], 'ci': c['ci'], 'excludes_zero': c['excludes_zero'],
            'jackknife_t': c['jackknife_t'], 'jackknife_excludes_zero': c['jackknife_excludes_zero'],
            'robust': bool(c['excludes_zero'] and c['jackknife_excludes_zero']), 'dates': c['dates']}


def main():
    P = pd.concat([pd.read_csv(D / 'aou/predictions.csv.gz'), pd.read_csv(D / 'aou_xgb/predictions.csv.gz')],
                  ignore_index=True)
    meta = cast_meta()
    errors = {(co, m, sp): cast_errors(g, meta, '2026') for (co, m, sp), g in P.groupby(['cohort', 'model', 'space'])}
    some = next(iter(errors.values()))
    out = {'metric': 'nested macro MAE (umol/L): depth bins within cast, casts within station, stations',
           'unit': 'paired bootstrap over UTC cast dates, each drawn date relabelled, and the '
                   'delete-one-date jackknife-t; robust = both exclude zero',
           'draws': DRAWS, 'seed': RNG_SEED, 'casts': int(some.cast_id.nunique()),
           'dates': int(some.day.nunique()), 'rows_per_arm': int(len(P) // len(errors)),
           'mae': {}, 'contrasts': []}
    for co in COHORTS:
        for (c2, m, sp), e in errors.items():
            if c2 == co:
                out['mae'][f'{co}|{m}|{sp}'] = round(macro(e), 4)
        for (ma, sa), (mb, sb) in PAIRS:
            ca, cb = errors[(co, ma, sa)], errors[(co, mb, sb)]
            if not ca.cast_id.equals(cb.cast_id):
                raise SystemExit(f'{co}: {ma}|{sa} and {mb}|{sb} do not score the same casts')
            entry = {'cohort': co, 'a': f'{ma}|{sa}', 'b': f'{mb}|{sb}', **summarise(contrast(ca, cb))}
            out['contrasts'].append(entry)
            print(f"{co:9s} {entry['a']:22s} - {entry['b']:26s} {entry['diff']:+.3f} {entry['ci']} "
                  f"jk {entry['jackknife_t']}{'  robust' if entry['robust'] else ''}", flush=True)
    (D / 'aou_summary.json').write_text(json.dumps(out, indent=2))
    print(out['mae'])


if __name__ == '__main__':
    main()
