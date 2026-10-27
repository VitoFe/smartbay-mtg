"""The interval ledger behind the Reserved-2026 coverage, and the stated rule as a sensitivity.

Chapter 4 describes the forecast interval as a rolling thirty-day window of out-of-sample residuals,
reset quarterly, needing thirty residuals available by issuance, with the residuals that fitted the
blend weights kept out. `scripts/v2_seasonal_table.py` applies exactly that on Forecast-2025
(`interval_rows` with an empty calibration pool, per quarter and sensor). The Reserved-2026 reading
(`src/experiments/v2_reserved.evaluate_forecast`) does not:

  * its ledger starts from `calibration_<variable>.csv.gz`, the Q4-2025 residuals of the
    *unblended* tree on which the blend weight was chosen;
  * it is never reset, so Q2 intervals draw on Q1 residuals;
  * when fewer than thirty of the sensor's residuals fall in the last thirty days it falls back to
    the sensor's whole available history, calibration rows included;
  * a 2026 residual is stamped at its target hour and becomes usable seven hours after it, where a
    Forecast-2025 residual is usable one hour after its target.

The reading is frozen and is not re-run. This script works only from its stored endpoints
(prediction, truth, radius) and the stored calibration file:

  1. it rebuilds the published ledger and checks that every stored radius is reproduced (gate G1);
  2. it counts the intervals that drew on calibration residuals, on the fallback, or on the
     previous quarter;
  3. it rescores the same blended predictions under the stated rule, with the Forecast-2025 code
     itself (`bay_seasonal_validation.interval_rows`, empty calibration pool, per quarter and
     sensor), and once more with the reading's own seven-hour lag after the target, so the two
     departures can be told apart. The predictions and the scored endpoints do not change; only
     which endpoints receive an interval, and how wide it is.

Coverage, availability and width are aggregated with the Reserved-2026 table's rule
(`scripts/v2_reserved_table.py`: per quarter, the mean over sensors of each sensor's mean, then the
mean over quarters; availability is the pooled share of scored endpoints within each quarter), and
also with the Forecast-2025 table's rule (only sensor-quarters with at least thirty endpoints, and
coverage and width only where at least thirty intervals exist), and pooled over endpoints.

Usage: python -m scripts.v2_reserved_ledger
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.bay_seasonal_validation import interval_rows  # noqa: E402
from src.protocols.bay_calibration import finite_radius  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
READING = RUNS / 'v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'
CALIBRATION = RUNS / 'v2_release_v2/refit_forecast/calibration_{variable}.csv.gz'
OUT = RUNS / 'v2_reserved_ledger.json'
Q2 = pd.Timestamp('2026-04-01', tz='UTC')


def published(g, cal):
    """The ledger of v2_reserved.evaluate_forecast, rebuilt from stored residuals, with provenance."""
    cal = cal.assign(error=cal.truth - cal.prediction, source='calibration')
    own = pd.DataFrame({'time': g.target_time, 'sensor': g.sensor, 'error': g.truth - g.prediction,
                        'source': '2026'})
    ledger = pd.concat([cal[['time', 'sensor', 'error', 'source']], own], ignore_index=True)
    ledger['available'] = ledger.time + pd.Timedelta(hours=7)
    rows = []
    for r in g.itertuples():
        a = ledger[(ledger.sensor == r.sensor) & (ledger.available <= r.time)]
        recent = a[a.available >= r.time - pd.Timedelta(days=30)]
        fallback = len(recent) < 30
        pool = a if fallback else recent
        radius = finite_radius(pool.error.to_numpy()) if len(pool) >= 30 else np.nan
        rows.append({'radius_rebuilt': radius, 'fallback': fallback,
                     'uses_calibration': bool((pool.source == 'calibration').any()),
                     'uses_previous_quarter': bool(r.quarter == 'Q2' and ((pool.source == '2026')
                                                                         & (pool.time < Q2)).any())})
    return pd.DataFrame(rows, index=g.index)


def stated(g, lag_hours):
    """Empty ledger at each quarter start, thirty-day window, thirty residuals, no fallback.

    A residual of the blend becomes usable `lag_hours` after issuance: 7 is the Forecast-2025
    convention (one hour after the target), 13 the reading's own (seven hours after the target).
    """
    rows = []
    for (_, _), cell in g.groupby(['quarter', 'sensor']):
        cell = cell.sort_values('time')
        avail = cell.time + pd.Timedelta(hours=lag_hours)
        err = (cell.truth - cell.prediction).to_numpy()
        for r in cell.itertuples():
            pick = (avail <= r.time) & (avail >= r.time - pd.Timedelta(days=30))
            ok = int(pick.sum()) >= 30
            radius = finite_radius(err[pick.to_numpy()]) if ok else np.nan
            rows.append({'index': r.Index, 'radius': radius, 'has_interval': ok})
    out = pd.DataFrame(rows).set_index('index').reindex(g.index)
    out['covered'] = ((g.truth - g.prediction).abs() <= out.radius).where(out.has_interval)
    return out


def check_against_interval_rows(g, stated_frame):
    """Gate G2: `stated(g, 7)` is `interval_rows` with an empty calibration pool, per cell."""
    worst = 0.0
    for (_, _), cell in g.groupby(['quarter', 'sensor']):
        c = cell[['time', 'truth', 'prediction']].copy()
        iv = interval_rows(c.iloc[:0], c, 'rolling30')
        half = ((iv.upper - iv.lower) / 2).to_numpy()
        mine = stated_frame.loc[cell.index, 'radius'].to_numpy()
        both = np.isfinite(half) & np.isfinite(mine)
        if (np.isfinite(half) != np.isfinite(mine)).any():
            return float('inf')
        if both.any():
            worst = max(worst, float(np.abs(half[both] - mine[both]).max()))
    return worst


def aggregate(g, has_interval, radius, covered):
    """Coverage, availability and width under three rules."""
    f = pd.DataFrame({'quarter': g.quarter, 'sensor': g.sensor, 'has': has_interval.astype(bool),
                      'covered': covered.astype(float).where(has_interval),
                      'width': (2 * radius).where(has_interval)})
    table, f25, per_quarter, cells = {}, {}, {}, {}
    q_rows = []
    for q, h in f.groupby('quarter'):
        by_sensor = h.groupby('sensor')[['covered', 'width']].mean()
        per_quarter[q] = {'endpoints': int(len(h)), 'intervals': int(h.has.sum()),
                          'coverage': round(float(by_sensor.covered.mean()), 4),
                          'availability': round(float(h.has.mean()), 4),
                          'width': round(float(by_sensor.width.mean()), 4)}
        q_rows.append(per_quarter[q])
        for s, c in h.groupby('sensor'):
            cells[f'{q}|{s}'] = {'endpoints': int(len(c)), 'intervals': int(c.has.sum()),
                                 'coverage': None if not c.has.any() else round(float(c.covered.mean()), 4)}
    table = {k: round(float(np.mean([r[k] for r in q_rows])), 4)
             for k in ('coverage', 'availability', 'width')}
    s = f.groupby(['quarter', 'sensor']).agg(n=('has', 'size'), n_int=('has', 'sum'),
                                             availability=('has', 'mean'), coverage=('covered', 'mean'),
                                             width=('width', 'mean')).reset_index()
    s = s[s.n >= 30].copy()
    s.loc[s.n_int < 30, ['coverage', 'width']] = np.nan
    q = s.groupby('quarter')[['availability', 'coverage', 'width']].mean()
    f25 = {k: round(float(q[k].mean()), 4) for k in ('coverage', 'availability', 'width')}
    pooled = {'coverage': round(float(f.covered.mean()), 4),
              'availability': round(float(f.has.mean()), 4),
              'width': round(float(f.width.mean()), 4)}
    return {'reserved_table_rule': table, 'forecast2025_rule': f25, 'pooled': pooled,
            'by_quarter': per_quarter, 'by_quarter_sensor': cells}


def main():
    e = pd.read_csv(READING)
    for c in ('time', 'target_time'):
        e[c] = pd.to_datetime(e[c], utc=True)
    scored = e[e.error.notna()].copy()
    report = {'reading': 'runs/v2_reserved_reading_v2 (frozen; not re-run)',
              'nominal': 0.90, 'variables': {}}
    for variable in ('temperature', 'oxygen'):
        g = scored[scored.variable == variable].copy()
        cal = pd.read_csv(CALIBRATION.as_posix().format(variable=variable))
        cal['time'] = pd.to_datetime(cal.time, utc=True)
        pub = published(g, cal)
        g1 = float((pub.radius_rebuilt - g.radius).abs().max())
        assert g1 < 1e-9, (variable, g1)
        covered_pub = g.covered.astype('boolean').astype(float)
        seven = stated(g, 7)
        g2 = check_against_interval_rows(g, seven)
        assert g2 < 1e-12, (variable, g2)
        thirteen = stated(g, 13)
        flagged = pub.fallback | pub.uses_calibration | pub.uses_previous_quarter
        block = {
            'scored_endpoints': int(len(g)),
            'calibration_rows': int(len(cal)),
            'gates': {'G1_published_radius_rebuilt_max_abs_diff': g1,
                      'G2_stated_rule_equals_interval_rows_max_abs_diff': g2},
            'published_ledger_sources': {
                'intervals': int(g.has_interval.sum()),
                'fallback_to_whole_history': int(pub.fallback.sum()),
                'draw_on_calibration_residuals': int(pub.uses_calibration.sum()),
                'calibration_or_fallback': int((pub.fallback | pub.uses_calibration).sum()),
                'q2_draw_on_q1_residuals': int(pub.uses_previous_quarter.sum()),
                'any_of_the_three': int(flagged.sum()),
                'share_calibration_or_fallback': round(float((pub.fallback | pub.uses_calibration).mean()), 4)},
            'published': aggregate(g, g.has_interval.astype(bool), g.radius, covered_pub),
            'stated_rule_forecast2025_lag': aggregate(g, seven.has_interval.astype(bool), seven.radius,
                                                       seven.covered),
            'stated_rule_reading_lag': aggregate(g, thirteen.has_interval.astype(bool), thirteen.radius,
                                                 thirteen.covered)}
        report['variables'][variable] = block
        for name in ('published', 'stated_rule_forecast2025_lag', 'stated_rule_reading_lag'):
            t = block[name]['reserved_table_rule']
            print(f"{variable:11s} {name:30s} coverage {t['coverage']:.4f}  availability "
                  f"{t['availability']:.4f}  width {t['width']:.4f}  "
                  f"by quarter {[(q, v['coverage'], v['availability']) for q, v in block[name]['by_quarter'].items()]}",
                  flush=True)
        print(variable, block['published_ledger_sources'], flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
