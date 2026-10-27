"""Addendum 66 on synthetic data: the lags read only the past, the per-probe ridge recovers a known
autoregression, and the GARCH-type bands use no error before its target time and widen after large ones."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import v2_arima as A  # noqa: E402
from src.experiments import v2_arima_report as AR  # noqa: E402
from src.experiments import v2_probabilistic as P  # noqa: E402


def test_lag_changes_read_only_the_past():
    series = np.arange(100, dtype=float)[:, None]
    q = np.array([0, 10, 80])
    out = A.lag_changes(series, q, lags=(1, 6, 72))
    assert np.isnan(out[0, 0]).all()                       # nothing before the first hour
    assert out[1, 0, 0] == 1 and out[1, 0, 1] == 6 and np.isnan(out[1, 0, 2])
    assert np.allclose(out[2, 0], [1, 6, 72])


def test_design_columns_follow_the_configuration():
    n, s = 5, 2
    feats = {'chg': np.zeros((n, s, len(A.LAGS))), 'level': np.zeros((n, s)),
             'hour': np.arange(n), 'doy': np.arange(n), 't_chg': np.zeros((n, s, len(A.LAGS))), 't_level': np.zeros((n, s))}
    cfg = {'max_lag': 24, 'daily': 2, 'annual': 1, 'level': True}
    lags = sum(k <= 24 for k in A.LAGS)
    assert A.design(feats, cfg, joint=False).shape == (n, s, 2 * lags + 2 + 4 + 2)
    assert A.design(feats, cfg, joint=True).shape == (n, s, 2 * (2 * lags + 2) + 4 + 2)


def test_ridge_recovers_a_known_autoregression():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 1, 3))
    beta = np.array([0.5, -0.3, 0.2])
    change = x[:, 0] @ beta + 0.1 + rng.normal(scale=0.05, size=n)
    data = {'valid': np.ones((n, 1), bool), 'truth': change[:, None], 'x': np.zeros((n, 1)),
            'times': pd.date_range('2025-01-01', periods=n, freq='6h', tz='UTC')}
    cfg = {'log_alpha': -3.0, 'by_hour': False}
    models = A.fit_linear(data, x, np.arange(n), np.array([True]), cfg)
    got = A.predict_change(models, x, np.arange(n), 1)[:, 0]
    assert np.abs(got - (x[:, 0] @ beta + 0.1)).max() < 0.02
    assert abs(models[0]['own'][P.LEVELS.index(0.5)]) < 0.01


def stream(errors, start='2025-01-01'):
    t = pd.date_range(start, periods=len(errors), freq='6h', tz='UTC')
    return pd.DataFrame({'cohort': 'Dev-2025', 'sensor': 's', 'time': t, 'lead': 12,
                         'truth': np.asarray(errors, float), 'q50': 0.0})


def test_bands_use_no_error_before_its_target_time():
    out, info = AR.ewma_bands(stream(np.ones(50)), {'s': 1.0}, 0.94)
    assert info['violations'] == 0
    assert info['checked'] == 48                            # a 12 h lead on a 6 h grid: the last two unseen
    # the first two issuances precede every target, so they keep the starting sigma with normal quantiles
    width = out[P.qcol(0.9)] - out[P.qcol(0.1)]
    assert np.allclose(width.iloc[:3], width.iloc[0])


def test_bands_widen_after_large_errors():
    errors = np.r_[np.full(30, 0.1), np.full(10, 5.0), np.full(30, 0.1)]
    out, _ = AR.ewma_bands(stream(errors), {'s': 0.1}, 0.94)
    width = (out[P.qcol(0.9)] - out[P.qcol(0.1)]).to_numpy()
    assert width[45] > 3 * width[29]                        # after the large errors are observed
