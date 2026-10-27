"""Addendum-11: what happens to the reserved battery when the sensor axis is resampled too.

Every forecast interval in this thesis resamples ISO week blocks within quarter and holds the
sensor set fixed. The reported statistic is a nested macro average whose outer axes are quarter and
sensor, so such an interval answers "how would this contrast vary over other weeks, for these exact
probes" rather than "how would it vary for this network".

Three schemes are compared on the already-frozen predictions. Nothing is refitted.

    published    week blocks within quarter, sensors fixed
    sensor_only  sensors resampled with replacement, all their endpoints carried
    two_way      sensors resampled, then week blocks within each drawn sensor

At two eligible oxygen sensors the sensor axis admits three multisets, so the macro statistic takes
at most three values and its percentiles degenerate to the per-sensor values. The per-sensor
contrast values are therefore printed beside every interval, so that a degenerate interval is
visibly degenerate rather than merely wide.

Declared in config/bay_reserved_protocol_v2_addendum11.json.
Usage: python scripts/v2_sensor_axis.py
"""
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'reports/bay_twin/runs/battery/results'
KEY = ['variable', 'sensor', 'time']
OUT = ROOT / 'reports/bay_twin/runs/v2_sensor_axis.json'
DRAWS = 600


def load():
    f = pd.read_csv(T / 'forecast/forecast_predictions.csv.gz')[
        KEY + ['quarter', 'week', 'truth', 'MODEL', 'B0_persistence', 'B1_clim_increment',
               'CHRONOS_bolt_small', 'CHRONOS_bolt_base']]
    c = pd.read_csv(T / 'chronos/chronos_predictions.csv.gz')
    zs = [k for k in c.columns
          if ('|ZS_raw' in k and not k.endswith('_C2')) or ('|ZS_aou' in k and not k.endswith('_C2'))]
    d = f.merge(c[KEY + zs], on=KEY, validate='one_to_one')
    e2 = pd.read_csv(ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz')
    e2 = e2[e2.has_anchor & e2.has_target]
    d = d.merge(e2[KEY + ['prediction']].rename(columns={'prediction': 'MODEL_v2'}), on=KEY,
                validate='one_to_one')
    for p in sorted((ROOT / 'reports/bay_twin/runs/battery/v2_chronos')
                    .glob('v2_chronos*_predictions*.csv.gz')):
        x = pd.read_csv(p)
        cols = [k for k in x.columns if k.startswith(('V2FT_', 'V2C2_', 'CH2_')) and k not in d.columns]
        if cols:
            d = d.merge(x[KEY + cols], on=KEY, validate='one_to_one')
    return d.rename(columns={'MODEL': 'MODEL_v1'})


def macro(s, col):
    s = s.dropna(subset=[col])
    return float(s.assign(e=(s[col] - s.truth).abs())
                 .groupby(['quarter', 'sensor']).e.mean().groupby('quarter').mean().mean())


def resample(frame, scheme, rng, sensors, week_groups, week_blocks):
    if scheme == 'published':
        return pd.concat([week_groups[tuple(week_blocks[i])]
                          for i in rng.integers(0, len(week_blocks), len(week_blocks))])
    drawn = [sensors[i] for i in rng.integers(0, len(sensors), len(sensors))]
    parts = []
    for rank, name in enumerate(drawn):
        g = frame[frame.sensor == name]
        if scheme == 'two_way':
            blocks = g[['quarter', 'week']].drop_duplicates().values.tolist()
            groups = {tuple(k): gg for k, gg in g.groupby(['quarter', 'week'])}
            g = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        # A sensor drawn twice must count twice in the macro average, so relabel the copies.
        parts.append(g.assign(sensor=f'{name}#{rank}'))
    return pd.concat(parts)


def interval(frame, x, y, scheme, label, sensors):
    rng = np.random.default_rng([20260921, zlib.crc32(f'{label}|{scheme}'.encode())])
    week_blocks = frame[['quarter', 'week']].drop_duplicates().values.tolist()
    week_groups = {tuple(k): g for k, g in frame.groupby(['quarter', 'week'])}
    draws = []
    for _ in range(DRAWS):
        s = resample(frame, scheme, rng, sensors, week_groups, week_blocks)
        draws.append(macro(s, x) - macro(s, y))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'ci': [float(lo), float(hi)], 'excludes_zero': bool(lo > 0 or hi < 0),
            'distinct_draw_values': int(len(np.unique(np.round(draws, 9))))}


def main():
    d = load()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum11.json', 'draws': DRAWS,
              'schemes': ['published', 'sensor_only', 'two_way'], 'variables': {}}

    for variable in ['temperature', 'oxygen']:
        s0 = d[d.variable == variable].copy()
        sensors = sorted(s0.sensor.unique())
        arms = [m for m in s0.columns
                if m not in KEY + ['quarter', 'week', 'truth', 'MODEL_v2'] and s0[m].notna().all()]
        entry = {'n': int(len(s0)), 'sensors': len(sensors),
                 'endpoints_per_sensor': s0.groupby('sensor').size().to_dict(),
                 'multisets_available': int(
                     math.comb(2 * len(sensors) - 1, len(sensors))) if len(sensors) < 12 else None,
                 'contrasts': {}}
        print(f'== {variable}: n={len(s0)}  sensors={len(sensors)}  '
              f'distinct sensor multisets={entry["multisets_available"]}')
        for arm in arms:
            label = f'{arm} - MODEL_v2'
            per_sensor = {}
            for name in sensors:
                g = s0[s0.sensor == name]
                per_sensor[name] = round(macro(g, arm) - macro(g, 'MODEL_v2'), 4)
            row = {'point': macro(s0, arm) - macro(s0, 'MODEL_v2'), 'per_sensor': per_sensor}
            for scheme in report['schemes']:
                row[scheme] = interval(s0, arm, 'MODEL_v2', scheme, label, sensors)
            entry['contrasts'][label] = row
            flags = ''.join('*' if row[s]['excludes_zero'] else '.' for s in report['schemes'])
            print(f'   {arm:26s} {row["point"]:+8.4f}  pub/sens/two-way {flags}  '
                  f'two-way [{row["two_way"]["ci"][0]:+.3f}, {row["two_way"]["ci"][1]:+.3f}]  '
                  f'per-sensor {list(per_sensor.values())}')
        report['variables'][variable] = entry

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
