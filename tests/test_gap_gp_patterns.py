"""The Gaussian-process gap filler fits its linear mean per pattern of reporting neighbours.

Synthetic panels only; no data is read. Addendum 59 part A.
"""
import numpy as np
import pytest

pytest.importorskip('sklearn')

from src.experiments.v2_gap_recovery import (GP_MIN_PATTERN_HOURS, _gp_reduce,  # noqa: E402
                                              _gp_temporal, _gpr_by_pattern)


def panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 15 + 2 * np.sin(2 * np.pi * t / 24) + 0.002 * t
    values = np.column_stack([base + rng.normal(0, 0.05, n) + k * 0.3 for k in range(4)])
    return values


def test_reduce_drops_the_neighbour_with_fewest_margin_hours():
    finite = np.ones((144, 3), bool)
    finite[:130, 2] = False                      # neighbour 2 reports on 14 margin hours
    finite[:60, 1] = False                       # neighbour 1 on 84
    nb = np.where(finite, np.random.default_rng(1).normal(size=finite.shape), np.nan)
    hours_of = finite.sum(axis=0)
    # together they share only 14 hours: the one with fewest is dropped, then 0 and 1 share 84
    assert _gp_reduce((0, 1, 2), finite, hours_of, nb) == (0, 1)
    assert _gp_reduce((2,), finite, hours_of, nb) == ()           # 14 < 24: empty pattern
    assert GP_MIN_PATTERN_HOURS == 24


def test_a_silent_neighbour_never_enters_the_fit():
    values = panel()
    g0, length, margin = 200, 24, 72
    ctx = np.concatenate([np.arange(g0 - margin, g0), np.arange(g0 + length, g0 + length + margin)])
    gap = np.arange(g0, g0 + length)
    # neighbour 2 is silent through the second half of the gap
    values[g0 + 12:g0 + length, 2] = np.nan
    pred, used, hours, failures = _gpr_by_pattern(values, 0, [1, 2, 3], ctx, values[ctx, 0], gap, g0)
    assert not failures and np.isfinite(pred).all()
    assert (used[:12] == 3).all() and (used[12:] == 2).all()
    # the rows of the second pattern equal a fit that never saw neighbour 2
    clean = values.copy()
    clean[:, 2] = np.nan
    alone, used_alone, _, _ = _gpr_by_pattern(clean, 0, [1, 2, 3], ctx, values[ctx, 0], gap, g0)
    np.testing.assert_allclose(pred[12:], alone[12:], atol=1e-9)
    assert (used_alone == 2).all()


def test_the_empty_pattern_is_the_temporal_process_alone():
    values = panel()
    g0, length, margin = 200, 12, 72
    ctx = np.concatenate([np.arange(g0 - margin, g0), np.arange(g0 + length, g0 + length + margin)])
    gap = np.arange(g0, g0 + length)
    y = values[ctx, 0]
    pred, used, hours, _ = _gpr_by_pattern(values, 0, [], ctx, y, gap, g0)
    t_ctx, t_gap = ((ctx - g0) / 24.0)[:, None], ((gap - g0) / 24.0)[:, None]
    expected = _gp_temporal(t_ctx, y - y.mean(), t_gap) + y.mean()
    np.testing.assert_allclose(pred, expected, atol=1e-12)
    assert (used == 0).all() and (hours == len(ctx)).all()
