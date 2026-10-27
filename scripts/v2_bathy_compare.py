"""Addendum-8: do the bathymetric features help, and where?

The two held-sensor runs, with and without the seabed features, are scored on identical endpoints,
so they can be compared as a paired contrast rather than by reading two summary tables side by
side. Both evaluations are reported, because the study's point is that a feature can help transfer
to an unseen probe while doing nothing, or harm, in the pooled model that already knows every
probe's coordinates.

Usage: python scripts/v2_bathy_compare.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bay_profile_context import ROOT

HELD = ROOT / 'reports/bay_twin/runs/v2_held_sensor'
OUT = ROOT / 'reports/bay_twin/runs/v2_bathy_compare.json'
KEY = ['fold', 'sensor', 'time']


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def paired(frame, left, right, label):
    rng = np.random.default_rng([20260920, zlib.crc32(label.encode())])
    keys = frame[['fold', 'sensor']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in frame.groupby(['fold', 'sensor'])}
    draws = []
    for _ in range(600):
        s = pd.concat([groups[tuple(keys[i])] for i in rng.integers(0, len(keys), len(keys))])
        draws.append(macro(s, left) - macro(s, right))
    return [macro(frame, left) - macro(frame, right), *np.percentile(draws, [2.5, 97.5]).tolist()]


def main():
    out = {}
    for variable in ['temperature', 'oxygen']:
        plain = pd.read_csv(HELD / f'held_sensor_{variable}_h6.csv.gz')
        bathy = pd.read_csv(HELD / f'held_sensor_{variable}_h6_bathy.csv.gz')
        merged = plain.merge(bathy[KEY + ['all_sensor', 'held_sensor']], on=KEY,
                             suffixes=('', '_bathy'), validate='one_to_one')
        assert len(merged) == len(plain), (variable, len(merged), len(plain))
        entry = {
            'endpoints': int(len(merged)),
            'mae': {'all_sensor': macro(merged, 'all_sensor'),
                    'all_sensor_bathy': macro(merged, 'all_sensor_bathy'),
                    'held_sensor': macro(merged, 'held_sensor'),
                    'held_sensor_bathy': macro(merged, 'held_sensor_bathy')},
            'contrasts': {
                'pooled: bathy - plain': paired(merged, 'all_sensor_bathy', 'all_sensor',
                                                f'{variable}|pooled'),
                'held-sensor: bathy - plain': paired(merged, 'held_sensor_bathy', 'held_sensor',
                                                     f'{variable}|held'),
                'transfer penalty, plain': paired(merged, 'held_sensor', 'all_sensor',
                                                  f'{variable}|pen-plain'),
                'transfer penalty, bathy': paired(merged, 'held_sensor_bathy', 'all_sensor_bathy',
                                                  f'{variable}|pen-bathy')}}
        out[variable] = entry
        print(f'== {variable}  n={entry["endpoints"]}')
        for k, v in entry['mae'].items():
            print(f'   {k:20s} {v:.4f}')
        for k, (d, lo, hi) in entry['contrasts'].items():
            print(f'   {k:30s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))
    OUT.write_text(json.dumps(out, indent=2))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
