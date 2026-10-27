"""Addendum 64's scoring and alert rules on synthetic data, each against a property it must have."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import v2_probabilistic as P  # noqa: E402
from src.experiments import v2_probabilistic_report as R  # noqa: E402


def test_quantile_score_of_a_single_value_is_its_absolute_error():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    m = rng.normal(size=200)
    Q = np.repeat(m[:, None], len(P.LEVELS), axis=1)
    assert np.allclose(P.quantile_score(y, Q), np.abs(y - m))
    assert np.allclose(R.quantile_score(y, Q), np.abs(y - m))


def test_quantile_score_rewards_a_calibrated_spread_over_a_point():
    rng = np.random.default_rng(2)
    y = rng.normal(size=20000)
    from scipy.stats import norm
    spread = np.repeat(norm.ppf(P.LEVELS)[None, :], len(y), axis=0)
    point = np.zeros_like(spread)
    assert P.quantile_score(y, spread).mean() < P.quantile_score(y, point).mean()


def test_tide_is_a_function_of_the_clock_only():
    t = pd.date_range('2026-01-01', periods=48, freq='h', tz='UTC')
    design = P._tide_design(t)
    assert design.shape == (48, 1 + 2 * len(P.CONSTITUENTS))
    # M2 repeats after its period: the cosine column of M2 at t and t + 12.4206 h agree
    period = 360 / P.CONSTITUENTS['M2']
    a = P._tide_design(pd.DatetimeIndex([t[0]]))[0, 1]
    b = P._tide_design(pd.DatetimeIndex([t[0] + pd.Timedelta(hours=period)]))[0, 1]
    assert abs(a - b) < 1e-9


def test_cooldown_keeps_the_first_alert_of_each_window():
    t0 = pd.Timestamp('2026-01-01', tz='UTC')
    times = [t0 + pd.Timedelta(hours=h) for h in (0, 1, 2, 5, 6, 13)]
    assert R.cooldown_filter(times, 6) == [times[0], times[4], times[5]]


def test_forecast_flag_fires_on_a_six_hour_drop_ending_within_six_hours():
    t = pd.date_range('2026-01-01', periods=24, freq='h', tz='UTC')
    hourly = pd.Series(200.0, index=t)
    issue = t[10]
    row = {'sensor': 's', 'time': issue, 'arm': 'a'}
    for a in P.WARN_LEVELS:
        for k in range(1, 7):
            row[f'k{k}_{P.qcol(a)}'] = 200.0
    # the 0.5 quantile three hours ahead falls 31 below the reading at issue - 3
    row[f'k3_{P.qcol(0.5)}'] = 169.0
    warn = pd.DataFrame([row])
    assert R.forecast_flags(warn, hourly, 's', 'a', 0.5, 30) == [issue]
    assert R.forecast_flags(warn, hourly, 's', 'a', 0.5, 31) == []
    # the reference reading missing: that lead is skipped
    hourly[issue - pd.Timedelta(hours=3)] = np.nan
    assert R.forecast_flags(warn, hourly, 's', 'a', 0.5, 30) == []


def test_matching_counts_only_alerts_that_precede_a_crossing():
    curve = R._curve()
    t = pd.date_range('2026-01-01', periods=60, freq='h', tz='UTC')
    hourly = pd.Series(200.0, index=t)
    hourly[t[30]:] = 160.0                      # a 40 drop: crossings at t[30] to t[35]
    early, late = t[26], t[40]
    res = R.score(hourly, len(t), [early, late], curve)
    assert res['crossings'] == 6
    assert res['matched_causal'] == 1 and res['matched_sym'] == 2
    assert res['caught_causal'] == 6


def test_joint_draws_are_reproducible_and_centre_on_the_macro():
    rng = np.random.default_rng(3)
    rows = []
    for fold in ('0', '1'):
        for sensor in ('a', 'b', 'c'):
            for week in range(1, 11):
                for _ in range(5):
                    rows.append({'fold': fold, 'sensor': sensor, 'week': week, 'x': rng.normal(1.0, 0.2),
                                 'y': rng.normal(1.1, 0.2)})
    f = pd.DataFrame(rows)
    a = R.joint_draws(f, ['x', 'y'], 'test', draws=200)
    b = R.joint_draws(f, ['x', 'y'], 'test', draws=200)
    assert np.array_equal(a, b)
    assert abs(a[:, 0].mean() - R.macro(f, 'x')) < 0.02
    c = R.contrast(f, a, ['x', 'y'], 'x', 'y')
    assert c['ci'][0] < c['point'] < c['ci'][1]
