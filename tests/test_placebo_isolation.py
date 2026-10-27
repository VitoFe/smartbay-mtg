"""Addendum 71 gate GA2: the fitted depth control of an inner split reads no label of that split's
validation stations, where the construction fitted on all training rows does.

A synthetic case runs everywhere; the case on the real Field-57 checkpoints needs the data folder."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import v2_field_placebo as P  # noqa: E402

DATA = (ROOT / 'data/raw').is_dir()


def perturbed_change(b, stations, depth, residual):
    """Largest change in each split's own offsets, and in the all-training-rows offsets at the same
    split's validation rows, when 1 degC is added to that split's validation labels."""
    from src.experiments.v2_field import inner_splits
    before = P.inner_split_offsets(b, stations, depth, residual)
    mask = np.zeros(len(depth), dtype=bool)
    mask[b['tr']] = True
    full = P.placebo_offsets('P1', depth, mask, residual, None)
    own, published = [], []
    for n, (_, iva) in enumerate(inner_splits(b, stations)):
        r = residual.copy()
        rows = b['tr'][iva]
        r[rows, 0] += 1.0
        after = P.inner_split_offsets(b, stations, depth, r)[n]
        own.append(float(np.nanmax(np.abs(after - before[n]))))
        moved = P.placebo_offsets('P1', depth, mask, r, None)
        published.append(float(np.nanmax(np.abs(moved[rows] - full[rows]))))
    return own, published


def test_synthetic_split_offsets_ignore_validation_labels():
    rng = np.random.default_rng(0)
    stations = np.repeat(np.array(['a', 'b', 'c', 'd', 'e']), 60)
    depth = rng.uniform(0.5, 14.0, len(stations))
    residual = np.column_stack([np.sin(depth / 3) + rng.normal(0, .1, len(depth)) for _ in range(3)])
    b = {'tr': np.arange(len(stations))}
    own, published = perturbed_change(b, stations, depth, residual)
    assert max(own) == 0.0
    assert min(published) > 0.0


@pytest.mark.skipif(not DATA, reason='the data folder is not linked (scripts/data_setup.py link)')
def test_field57_split_offsets_ignore_validation_labels():
    from src.data.bay_profile_context import TARGETS, load_profiles
    from src.experiments.bay_profile_field import prepare
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, _, _ = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    stations, depth = data.station.to_numpy(), data.depth_m.to_numpy(float)
    truth = data[TARGETS].to_numpy(float)
    for b in bundles[:3]:
        own, published = perturbed_change(b, stations, depth, truth - b['base'].astype(float))
        assert max(own) == 0.0
        assert max(published) > 0.0
