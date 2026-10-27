"""Addendum 67 on synthetic data: N-BEATS windows end at their issuance and its training targets precede
the cut; the issuance grid and the multipliers. The state-space forecasts are checked against
statsmodels' own on every split of the campaign (gate G2), where statsmodels is installed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import v2_probabilistic as P  # noqa: E402
from src.experiments import v2_statespace as S  # noqa: E402


def test_windows_end_at_the_issuance():
    index = pd.date_range('2025-01-01', periods=600, freq='h', tz='UTC')
    y = np.arange(600, dtype=float)
    y[100:105] = np.nan
    X, Y, M, keys = S.windows({'a': y}, {'a': np.array([200, 400])}, 24, False, index)
    win = X[:, :24]
    assert np.allclose(win[0], np.arange(-23, 1))           # changes from the last reading, ending at zero
    assert np.allclose(Y[0, :3], [1, 2, 3])                  # the next hours, as changes
    assert keys == [('a', 200), ('a', 400)]


def test_training_targets_precede_the_cut():
    y = np.random.default_rng(0).normal(size=2000)
    pos = S.training_positions({'a': y}, cut=1500, L=168)['a']
    assert pos.max() + S.HMAX < 1500
    assert pos.min() >= 336


def test_grid_hours_and_horizon():
    idx = S.DEV_INDEX
    g = S.grid(idx, idx[100], idx[400], end=380)
    assert set(idx[g].hour) <= set(S.GRID_HOURS)
    assert (g + S.HMAX < 380).all()


def test_multipliers_fall_back_to_normal_with_few_errors():
    y = np.zeros(1000)
    mean, var = np.zeros((5, S.HMAX)), np.ones((5, S.HMAX))
    m = S.multipliers(y, mean, var, np.arange(5))
    assert np.allclose(m[6], np.sort(m[6]))
    from scipy import stats
    assert np.allclose(m[6], stats.norm.ppf(P.LEVELS))
