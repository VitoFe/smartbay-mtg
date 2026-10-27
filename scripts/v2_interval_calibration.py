"""How well the released six-hour bands are calibrated on Reserved-2026, beyond marginal coverage.

The headline coverage of the released forecaster on Reserved-2026 (89.2 % temperature, 86.9 % oxygen,
nominal 90 %) is marginal over the scored endpoints. This script replays the released ledger rule of
src/experiments/v2_reserved.py:evaluate_forecast from its two inputs, the calibration residuals of the
v2 refit (reports/bay_twin/runs/v2_release_v2/refit_forecast/calibration_*.csv.gz) and the scored 2026
rows of runs/v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz, and reports

* a gate: the replayed half-width equals the ledger's own on every endpoint with an interval;
* the band's width and the interval score of Gneiting and Raftery (alpha = 0.1), set against a band
  of the same construction around persistence (its own ledger of persistence residuals);
* coverage by quarter and by probe and quarter, with the cells' counts;
* the macro coverage (per-sensor mean within quarter, then over quarters, the headline rule) with a
  95 % interval over ISO weeks resampled with replacement and relabelled;
* which endpoints took their band from the last thirty days and which from the all-history fallback
  (fewer than thirty recent residuals), with each group's coverage.

The blend weight of the v2 refit is 1.0 for both variables, so the calibration residuals that seed
the ledger are residuals of the forecast being calibrated; the script records that weight.
Review R11. Usage: python -m scripts.v2_interval_calibration
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.protocols.bay_calibration import finite_radius
from src.units import factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
LEDGER = RUNS / 'v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'
REFIT = RUNS / 'v2_release_v2/refit_forecast'
OUT = RUNS / 'v2_interval_calibration.json'
ALPHA = 0.1
DRAWS = 2000
SEED = 20261005
MIN_CELL = 30
LAG = pd.Timedelta(hours=7)      # a residual is usable one hour after its six-hour endpoint
WINDOW = pd.Timedelta(days=30)


def replay(cal, rows, column):
    """Half-width and pool of every scored row under the released ledger rule, for `column`.

    The ledger starts from the calibration residuals, keyed by issuance time, and grows with each
    scored 2026 residual, keyed (as in v2_reserved.evaluate_forecast) by its target time. A residual
    is available at q when its key + 7 h <= q; the band reads the last thirty days when they hold at
    least thirty residuals of the sensor and every available residual otherwise.
    """
    radius = np.full(len(rows), np.nan)
    pool = np.full(len(rows), '', dtype=object)
    for sensor, idx in rows.groupby('sensor').indices.items():
        c = cal[cal.sensor == sensor]
        keys = list((c.time + LAG).dt.tz_convert(None).to_numpy('datetime64[ns]'))
        errs = list(c.truth - c[column])
        g = rows.iloc[idx].sort_values('time')
        for q, block in g.groupby('time', sort=True):
            k = np.array(keys, dtype='datetime64[ns]') if keys else np.array([], dtype='datetime64[ns]')
            e = np.asarray(errs, float)
            qn = np.datetime64(q.tz_convert(None).as_unit('ns'))
            avail = k <= qn
            recent = avail & (k >= qn - np.timedelta64(WINDOW))
            if recent.sum() >= 30:
                r, p = finite_radius(e[recent]), 'recent'
            elif avail.sum() >= 30:
                r, p = finite_radius(e[avail]), 'all_history'
            else:
                r, p = np.nan, 'none'
            for i in block.index:
                radius[rows.index.get_loc(i)] = r
                pool[rows.index.get_loc(i)] = p
            for _, row in block.iterrows():
                keys.append((row.target_time + LAG).tz_convert(None).as_unit('ns').to_datetime64())
                errs.append(row.truth - row[column])
    return radius, pool


def interval_score(y, centre, radius):
    lo, hi = centre - radius, centre + radius
    return (hi - lo) + 2 / ALPHA * np.clip(lo - y, 0, None) + 2 / ALPHA * np.clip(y - hi, 0, None)


def macro(frame, column):
    """Per-sensor mean within quarter, then the mean over quarters (the headline rule)."""
    return float(frame.groupby(['quarter', 'sensor'])[column].mean().groupby('quarter').mean().mean())


def week_interval(frame, column, rng):
    weeks = frame.week.unique()
    parts = {w: frame[frame.week == w] for w in weeks}
    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(weeks, size=len(weeks), replace=True)
        draws.append(macro(pd.concat([parts[w] for w in pick], ignore_index=True), column))
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def main():
    rng = np.random.default_rng(SEED)
    ledger = pd.read_csv(LEDGER)
    out = {'question': 'Released six-hour bands on Reserved-2026: width, interval score, conditional '
                       'coverage, coverage over weeks, and the all-history fallback (review R11)',
           'ledger': str(LEDGER.relative_to(ROOT)), 'calibration': str(REFIT.relative_to(ROOT)),
           'alpha': ALPHA, 'draws': DRAWS, 'seed': SEED, 'min_cell': MIN_CELL, 'variables': {}}
    for variable, rows in ledger.groupby('variable'):
        unit = factor() if variable == 'oxygen' else 1.0
        cp = json.loads((REFIT / f'refit_{variable}_model.json').read_text())
        cal = pd.read_csv(REFIT / f'calibration_{variable}.csv.gz')
        cal['time'] = pd.to_datetime(cal.time, utc=True)
        rows = rows[rows.has_anchor & rows.has_target].copy().reset_index(drop=True)
        rows['time'] = pd.to_datetime(rows.time, utc=True)
        rows['target_time'] = pd.to_datetime(rows.target_time, utc=True)
        radius, pool = replay(cal, rows, 'prediction')
        p_radius, _ = replay(cal, rows, 'persistence')
        gate = float(np.nanmax(np.abs(radius - rows.radius.to_numpy()))) if rows.has_interval.any() else None
        same_flags = bool(((~np.isnan(radius)) == rows.has_interval.to_numpy()).all())
        rows['pool'] = pool
        rows['covered'] = np.abs(rows.truth - rows.prediction) <= radius
        rows['p_covered'] = np.abs(rows.truth - rows.persistence) <= p_radius
        rows['width'] = 2 * radius * unit
        rows['p_width'] = 2 * p_radius * unit
        rows['score'] = interval_score(rows.truth, rows.prediction, radius) * unit
        rows['p_score'] = interval_score(rows.truth, rows.persistence, p_radius) * unit
        b = rows[~np.isnan(radius) & ~np.isnan(p_radius)]
        cells = b.groupby(['sensor', 'quarter']).agg(n=('covered', 'size'), coverage=('covered', 'mean'),
                                                     width=('width', 'mean'))
        big = cells[cells.n >= MIN_CELL]
        lo, hi = big.coverage.idxmin(), big.coverage.idxmax()
        out['variables'][variable] = {
            'blend_weight': cp['global_blend_weight'],
            'gate_max_abs_radius_difference': gate, 'gate_interval_flags_identical': same_flags,
            'scored': int(len(rows)), 'with_band': int(len(b)),
            'coverage_pooled': round(float(b.covered.mean()), 4),
            'coverage_macro': round(macro(b, 'covered'), 4),
            'coverage_macro_ci_weeks': week_interval(b, 'covered', rng),
            'weeks': int(b.week.nunique()),
            'mean_width': round(float(b.width.mean()), 4),
            'interval_score_macro': round(macro(b, 'score'), 4),
            'persistence_band': {'coverage_macro': round(macro(b, 'p_covered'), 4),
                                 'mean_width': round(float(b.p_width.mean()), 4),
                                 'interval_score_macro': round(macro(b, 'p_score'), 4)},
            'by_quarter': {q: {'n': int(len(g)), 'coverage': round(float(g.covered.mean()), 4),
                               'mean_width': round(float(g.width.mean()), 4),
                               'interval_score': round(float(g.score.mean()), 4)}
                           for q, g in b.groupby('quarter')},
            'by_sensor_quarter': [{'sensor': s, 'quarter': q, 'n': int(r.n), 'coverage': round(float(r.coverage), 4),
                                   'mean_width': round(float(r.width), 4)} for (s, q), r in cells.iterrows()],
            'cells_with_min_n': int(len(big)),
            'cell_coverage_range': [round(float(big.coverage.min()), 4), round(float(big.coverage.max()), 4)],
            'cell_lowest': list(lo), 'cell_highest': list(hi),
            'pool': {p: {'n': int(len(g)), 'coverage': round(float(g.covered.mean()), 4)}
                     for p, g in b.groupby('pool')},
            'units': 'degC' if variable == 'temperature' else 'mg/l (record unit x src.units.factor())',
        }
    OUT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({v: {k: x for k, x in d.items() if k != 'by_sensor_quarter'}
                      for v, d in out['variables'].items()}, indent=1))


if __name__ == '__main__':
    main()
