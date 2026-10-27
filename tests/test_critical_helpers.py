"""The helpers every headline number passes through, on synthetic inputs: the oxygen unit, the spatial
nested macro and its date bootstrap, the self-neighbour guard, the overlap clusters of the gap cohorts
and the tuning protocol's refusals. Each rule here has produced a wrong number once: a pooled mean in
place of the nested one, a bootstrap over overlapping windows, a probe read as its own neighbour, a
repeated unit merged with itself, and smoke trials counted in a full study."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import units  # noqa: E402

# the factor is computed from the data by scripts/v2_oxygen_units.py; without it these cannot run
needs_units = pytest.mark.skipif(not units.UNITS.is_file(),
                                 reason='the oxygen unit factor has not been computed '
                                        '(python -m scripts.v2_oxygen_units)')


@needs_units
def test_oxygen_factor_is_the_recorded_one():
    recorded = json.loads((ROOT / 'reports/bay_twin/runs/v2_oxygen_units.json').read_text())
    assert units.factor() == pytest.approx(recorded['c_mg_per_l_per_umol_kg'], abs=0)
    assert units.factor() == pytest.approx(0.032844, abs=5e-6)


@needs_units
def test_oxygen_mg_passes_none_and_nan_and_maps_lists():
    assert units.oxygen_mg(None) is None
    assert math.isnan(units.oxygen_mg(float('nan')))
    assert units.oxygen_mg([1.0, 2.0]) == pytest.approx([units.factor(), 2 * units.factor()])


@needs_units
def test_oxygen_decimals_keep_the_floor_readable():
    # the 1.8 umol/kg floor prints as 0.059 mg/l: two significant figures kept
    d = units.mg_decimals(1.8, 1)
    assert f'{units.oxygen_mg(1.8):.{d}f}' == '0.059'
    # a column takes the most any value needs and never more than two beyond its own decimals
    assert units.column_decimals([1.8, 25.0], 1) == d
    assert units.column_decimals([0.0, 0.0], 2) == 4


def test_spatial_nested_macro_is_not_the_pooled_mean():
    from scripts.v2_field_contrasts import nested
    keys = pd.DataFrame({'fold': [0, 0, 0, 0, 1, 1],
                         'station': ['A', 'A', 'A', 'B', 'C', 'C'],
                         'cast_id': ['a1', 'a1', 'a2', 'b1', 'c1', 'c2']})
    error = np.array([1.0, 3.0, 4.0, 10.0, 2.0, 6.0])
    # casts: a1 = 2, a2 = 4, b1 = 10, c1 = 2, c2 = 6; stations: A = 3, B = 10, C = 4; folds: 6.5, 4
    assert nested(keys, error) == pytest.approx(5.25)
    assert error.mean() != pytest.approx(5.25)


def test_date_bootstrap_counts_days_and_is_zero_for_identical_arms():
    from scripts.v2_field_contrasts import contrast
    rng = np.random.default_rng(0)
    rows = []
    for day in range(8):
        for cast in range(2):                    # two casts share each day
            for depth in range(3):
                rows.append({'fold': day % 2, 'station': f's{cast}', 'cast_id': f'd{day}c{cast}',
                             'depth_m': depth, 'day': f'2025-01-{day + 1:02d}'})
    keys = pd.DataFrame(rows)
    truth = rng.normal(size=(len(keys), 3))
    same = truth + 0.5
    got = contrast(keys, truth, {'x': same, 'y': same.copy()}, 'x', 'y', 'temp_c', 'day', 'test')
    assert got['units'] == 8
    assert got['point'] == 0 and got['ci'] == [0, 0] and not got['excludes_zero']


def test_a_probe_is_never_its_own_neighbour():
    from src.experiments.v2_gap_recovery import nearest_others
    # probes 0 and 1 share coordinates; argsort can place 1 before 0 on the tie
    distance = np.array([[0., 0., 5., 9.], [0., 0., 5., 9.], [5., 5., 0., 4.], [9., 9., 4., 0.]])
    assert nearest_others(distance, 1) == [0, 2, 3]
    assert nearest_others(distance, 0, 2) == [1, 2]
    assert all(i not in nearest_others(distance, i) for i in range(4))


def test_overlapping_gaps_on_one_probe_form_one_cluster():
    from scripts.v2_gap_intervals import clusters
    frame = pd.DataFrame({
        'episode': ['p|a', 'p|a', 'p|b', 'p|b', 'p|c', 'q|a'],
        'sensor': ['p', 'p', 'p', 'p', 'p', 'q'],
        'time': ['2025-01-01T00', '2025-01-01T05', '2025-01-01T04', '2025-01-01T09', '2025-01-03T00',
                 '2025-01-01T00']})
    c = clusters(frame)
    assert c['p|a'] == c['p|b'] != c['p|c']
    assert c['q|a'] not in (c['p|a'], c['p|c'])


def test_the_protocol_refuses_too_few_trials_and_a_smoke_database(tmp_path):
    from src.tuning.protocol_v2 import run_study

    def failing(trial):
        trial.suggest_float('x', 0, 1)
        raise ValueError('every trial fails')
    with pytest.raises(RuntimeError, match='completed trials'):
        run_study('fails', failing, tmp_path / 'a', n_trials=3)

    def fine(trial):
        return trial.suggest_float('x', 0, 1)
    run_study('mixed', fine, tmp_path / 'b', quick=True)
    with pytest.raises(RuntimeError, match='smoke study'):
        run_study('mixed', fine, tmp_path / 'b', n_trials=3)
