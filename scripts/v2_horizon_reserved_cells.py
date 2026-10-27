"""The reserved 24 h reading (addendum 3, study 3), re-read for sparse cells and multiplicity.

The frozen macro rule averages sensors equally within each quarter and quarters equally, so a
(quarter, sensor) cell holding two endpoints weighs as much as one holding two hundred. On
temperature one such cell (Nodo_S04 in Q2, two endpoints) carries most of the graph arms' margin
over the tree. Nothing here re-reads 2026 or refits anything: the frozen predictions are
re-aggregated.

Per variable and declared contrast it reports:

  frozen      the published point estimate and week-block interval, reproduced
  cells_ge5   the same rule over (quarter, sensor) cells with at least five endpoints, the
              restriction the outage study already applies
  two_way     sensors and week blocks resampled together, over the frozen rule
  holm        the week-block bootstrap p-value of each contrast, Holm-adjusted within the
              declared family of thirteen

and, per contrast, what share of the frozen point estimate the smallest cell contributes.

Usage: python -m scripts.v2_horizon_reserved_cells
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SOURCE = RUNS / 'v2_horizon_reserved_h24'
OUT = RUNS / 'v2_horizon_reserved_cells.json'
DRAWS = 2000
MIN_ENDPOINTS = 5
ARMS = ('tree', 'weather_tree', 'learned_source', 'conservative_exchange')


def frames_for(variable):
    raw = pd.read_csv(SOURCE / f'reserved_h24_{variable}.csv.gz')
    raw = raw.groupby(['arm', 'sensor', 'time'], as_index=False).agg(
        quarter=('quarter', 'first'), week=('week', 'first'), truth=('truth', 'first'),
        persistence=('persistence', 'first'), baseline=('baseline', 'first'),
        prediction=('prediction', 'mean'))
    wide = raw.pivot_table(index=['sensor', 'time'], columns='arm', values='prediction')
    keys = raw.drop_duplicates(['sensor', 'time']).set_index(['sensor', 'time'])[
        ['quarter', 'week', 'truth', 'persistence', 'baseline']]
    frame = keys.join(wide, how='inner').reset_index()
    return frame.rename(columns={'baseline': 'climatology'})


def macro(d, col, cells=None):
    d = d.dropna(subset=[col])
    per = d.assign(e=(d.truth - d[col]).abs()).groupby(['quarter', 'sensor']).e.mean()
    if cells is not None:
        per = per[per.index.isin(cells)]
    return float(per.groupby('quarter').mean().mean())


def contrasts(summary):
    return [tuple(k.split(' - ')) for k in summary['contrasts']]


def week_draws(frame, x, y, label, cells=None):
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    blocks = frame[['quarter', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in frame.groupby(['quarter', 'week'])}
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        s = pd.concat([groups[tuple(blocks[j])] for j in rng.integers(0, len(blocks), len(blocks))])
        out[i] = macro(s, x, cells) - macro(s, y, cells)
    return out


def two_way_draws(frame, x, y, label):
    """Resample sensors (relabelled, so a sensor drawn twice counts twice) and week blocks."""
    rng = np.random.default_rng([20260924, zlib.crc32(label.encode())])
    sensors = sorted(frame.sensor.unique())
    by_sensor = {s: g for s, g in frame.groupby('sensor')}
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        picked = [by_sensor[sensors[j]].assign(sensor=f'{sensors[j]}#{n}')
                  for n, j in enumerate(rng.integers(0, len(sensors), len(sensors)))]
        s = pd.concat(picked)
        blocks = s[['quarter', 'week']].drop_duplicates().values.tolist()
        groups = {tuple(k): g for k, g in s.groupby(['quarter', 'week'])}
        s = pd.concat([groups[tuple(blocks[j])] for j in rng.integers(0, len(blocks), len(blocks))])
        out[i] = macro(s, x) - macro(s, y)
    return out


def interval(point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0)}


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[key]))
        out[key] = round(running, 4)
    return out


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum3.json study_3',
              'source': 'runs/v2_horizon_reserved_h24/reserved_h24_<variable>.csv.gz',
              'rule': 'frozen macro: mean over quarters of the per-sensor mean absolute error',
              'min_endpoints_per_cell': MIN_ENDPOINTS, 'draws': DRAWS, 'variables': {}}
    for variable in ('temperature', 'oxygen'):
        summary = json.loads((SOURCE / f'reserved_h24_{variable}_summary.json').read_text())
        frame = frames_for(variable)
        counts = frame.groupby(['quarter', 'sensor']).size()
        keep = list(counts[counts >= MIN_ENDPOINTS].index)
        block = {'endpoints': int(len(frame)),
                 'cells': {f'{q}|{s}': int(n) for (q, s), n in counts.items()},
                 'cells_below_min': {f'{q}|{s}': int(n) for (q, s), n in counts.items()
                                     if n < MIN_ENDPOINTS},
                 'contrasts': {}}
        pvalues = {}
        for x, y in contrasts(summary):
            label = f'{variable}|{x}|{y}'
            point = macro(frame, x) - macro(frame, y)
            draws = week_draws(frame, x, y, label)
            restricted = week_draws(frame, x, y, label + '|ge5', keep)
            per_cell = (frame.assign(ex=(frame.truth - frame[x]).abs(),
                                     ey=(frame.truth - frame[y]).abs())
                        .groupby(['quarter', 'sensor'])[['ex', 'ey']].mean())
            per_cell['d'] = per_cell.ex - per_cell.ey
            weights = 1 / per_cell.groupby('quarter').d.transform('size') / \
                per_cell.index.get_level_values('quarter').nunique()
            share = per_cell.d * weights
            smallest = counts.idxmin()
            p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
            pvalues[f'{x} - {y}'] = float(max(p, 1 / DRAWS))
            block['contrasts'][f'{x} - {y}'] = {
                'frozen': interval(point, draws),
                'published': [round(v, 4) for v in summary['contrasts'][f'{x} - {y}']],
                'cells_ge5': interval(macro(frame, x, keep) - macro(frame, y, keep), restricted),
                'two_way': interval(point, two_way_draws(frame, x, y, label)),
                'smallest_cell': {'cell': '|'.join(smallest), 'endpoints': int(counts.min()),
                                  'contribution': round(float(share.loc[smallest]), 4)},
                'bootstrap_p': round(pvalues[f'{x} - {y}'], 4)}
        adjusted = holm(pvalues)
        for key, value in adjusted.items():
            block['contrasts'][key]['holm_p'] = value
        report['variables'][variable] = block
        print(f'== {variable}: {len(frame)} endpoints; cells below {MIN_ENDPOINTS}: '
              f'{block["cells_below_min"]}')
        for key, e in block['contrasts'].items():
            print(f"   {key:40s} frozen {e['frozen']['point']:+.4f} {e['frozen']['ci']}  "
                  f"ge5 {e['cells_ge5']['point']:+.4f} {e['cells_ge5']['ci']}  "
                  f"two-way {e['two_way']['ci']}  holm p {e['holm_p']}")
    OUT.write_text(json.dumps(report, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
