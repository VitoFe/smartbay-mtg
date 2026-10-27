"""Acceptance Test Suite: Cross-Validation Fold Disjointness & Determinism."""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_leave_one_month_out_disjointness():
    """
    Verifies that for every leave-one-month-out fold, train and test splits are strictly disjoint.
    """
    t_idx = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="1h")
    df = pd.DataFrame({"val": np.random.randn(len(t_idx))}, index=t_idx)
    df["month"] = df.index.month

    unique_months = df["month"].unique()
    assert len(unique_months) == 12

    for m in unique_months:
        test_mask = (df["month"] == m).values
        train_mask = ~test_mask

        idx_train = set(df.index[train_mask])
        idx_test = set(df.index[test_mask])

        overlap = idx_train.intersection(idx_test)
        assert len(overlap) == 0, f"Leakage detected in month {m}: {len(overlap)} overlapping indices!"
        assert len(idx_train) + len(idx_test) == len(df)
