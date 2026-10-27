"""The released Copernicus configuration against the cast prior selected inside each inner split, on 2026.

Addendum 38 set the released configuration (A1_both) against the cast prior of addendum 36's declared
reading (A7o) on the 38 casts of 2026. The second reading of those casts (addendum 69,
runs/v2_field_reserved2026/innerprior/) selects A7o with its prior rebuilt inside each inner station
split, so the same comparison is repeated against it, with the bootstrap and jackknife of
src/experiments/v2_field_reserved2026.contrast. Temperature only: the record-unit Copernicus predictions
were not kept, and the mg/l rerun of addendum 60 (runs/v2_oxygen_mgl/C1/v2_field_reserved2026_copernicus/)
holds the same temperature predictions, which gate G1 checks by reproducing addendum 38's own
A1_both - A7o contrast from them.

It also stores the inner-split arm's own contrast with the arm without a prior to six decimals: the
report rounds it to four, and a bound stored as -0.2135 is -0.21349 before rounding.

Usage: python -m scripts.v2_field2026_innerprior_copernicus
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_field_reserved2026 as R
from src.experiments.v2_field_reserved2026 import cast_errors, contrast, macro


def unrounded(ca, cb):
    """v2_field_reserved2026.contrast's point and bootstrap interval, same stream, six decimals."""
    j = ca.merge(cb[['fold', 'station', 'cast_id', 'e']], on=['fold', 'station', 'cast_id'], suffixes=('_a', '_b'))
    point = macro(j.rename(columns={'e_a': 'e'})) - macro(j.rename(columns={'e_b': 'e'}))
    days = j.day.unique()
    groups = {d: g for d, g in j.groupby('day')}
    rng = np.random.default_rng(R.RNG_SEED)
    boot = []
    for _ in range(R.DRAWS):
        pick = rng.choice(days, len(days), replace=True)
        s = pd.concat([groups[d].assign(cast_id=groups[d].cast_id + f'#{k}') for k, d in enumerate(pick)])
        boot.append(macro(s.rename(columns={'e_a': 'e'})) - macro(s.rename(columns={'e_b': 'e'})))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'point': round(float(point), 6), 'ci': [round(float(lo), 6), round(float(hi), 6)]}

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
COPERNICUS = RUNS / 'v2_oxygen_mgl/C1/v2_field_reserved2026_copernicus/reserved2026_held_station_predictions.csv.gz'
INNER = RUNS / 'v2_field_reserved2026/innerprior/reserved2026_held_station_predictions.csv.gz'
PUBLISHED = RUNS / 'v2_field_reserved2026_copernicus/report.json'
OUT = RUNS / 'v2_field_reserved2026/innerprior/copernicus_contrast.json'


def main():
    cop = pd.read_csv(COPERNICUS, parse_dates=['t_utc'])
    inner = pd.read_csv(INNER, parse_dates=['t_utc'])
    t = 'temp_c'
    released = cast_errors(cop, 'A1_both', t)
    g1 = contrast(released, cast_errors(cop, 'A7o', t))
    want = json.loads(PUBLISHED.read_text())['held_station']['contrasts']['A1_both - A7o'][t]
    gate = all(abs(float(g1[k]) - want[k]) < 1e-12 for k in ('point',)) and \
        all(abs(float(a) - b) < 1e-12 for k in ('ci', 'jackknife_t') for a, b in zip(g1[k], want[k]))
    if not gate:
        raise SystemExit(f'gate G1: {g1} does not reproduce {want}')
    a7o = cast_errors(inner, 'A7o', t)
    c = contrast(released, a7o)
    c = {k: ([float(x) for x in v] if isinstance(v, list) else v) for k, v in c.items()}
    report = {'what': 'A1_both (released Copernicus configuration) - A7o selected with the cast prior rebuilt per '
                      'inner split, temperature, held station, the 38 casts of 2026',
              'declared': 'not separately: the addendum-38 contrast repeated on the addendum-69 second reading',
              'sources': {k: (p.relative_to(ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                          for k, p in (('copernicus', COPERNICUS), ('inner_prior', INNER))},
              'gate_G1_reproduces_addendum38': True,
              'nested_mae': {'A1_both': round(macro(released), 4), 'A7o_inner_prior': round(macro(a7o), 4)},
              'contrast': {'A1_both - A7o_inner_prior': {t: c}},
              'A7o_inner_prior - A2_none_six_decimals': {t: unrounded(a7o, cast_errors(inner, 'A2_none', t))}}
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(report['contrast'], indent=1))


if __name__ == '__main__':
    main()
