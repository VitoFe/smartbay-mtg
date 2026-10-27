"""Every augmented arm against the PLAIN tree, which is the reference that matters.

Addenda 6 and 7 each compared an augmented arm against the arm it augments, the weather tree. At
six hours on oxygen that base is itself handicapped: the weather features harm the oxygen tree
there (addendum 2), so any extra feature that shifts the tuner's selection can appear to deliver a
large gain while the result is still worse than a plain tree with no exogenous input at all.

This recomputes every augmented arm against the plain tree, with the same conventions, so the
apparent gains can be separated from genuine ones.

With no arguments it reads the published runs named below; the station runs are those with its hourly
means labelled at the end of their hour (config/bay_reserved_protocol_v2_addendum59.json, part B).

Usage: python scripts/v2_augmented_vs_plain.py [<river-run> <ispra-run> <horizon-run> [<horizon-run>]]
"""
import json, sys, zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
KEY = ['fold', 'sensor', 'time']
HORIZONS = [6, 24, 48]
ARMS = [('v2_river', 'river_tree'), ('v2_river', 'weather_river_tree'),
        ('v2_ispra', 'sealevel_tree'), ('v2_ispra', 'extboundary_tree'),
        ('v2_horizon', 'weather_tree')]
RIVER_RUN = '20260920T183451Z_v2_river'
ISPRA_RUN = '20261001T105500Z_v2_ispra'
HORIZON_RUNS = ['20260920T083322Z_v2_horizon']


def macro(d, col='prediction'):
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def load(runs, prefix, variable, arm, horizon):
    for r in runs:
        hits = list((RUNS / r / 'results').glob(f'{prefix}_{variable}_{arm}_h{horizon}/selected_predictions.csv.gz'))
        if hits:
            d = pd.read_csv(hits[0])
            if 'seed' in d:
                d = d.groupby(KEY, as_index=False).agg(truth=('truth', 'first'),
                                                       prediction=('prediction', 'mean'))
            return d[KEY + ['truth', 'prediction']]
    return None


def paired(base, arm, stream):
    j = base.merge(arm[KEY + ['prediction']], on=KEY, suffixes=('_base', ''), validate='one_to_one')
    rng = np.random.default_rng([20260920, zlib.crc32(stream.encode())])
    j = j.assign(week=pd.to_datetime(j.time, utc=True).dt.isocalendar().week.astype(int))
    blocks = j[['fold', 'week']].drop_duplicates().values.tolist()
    groups = {k: g for k, g in j.groupby(['fold', 'week'])}
    draws = []
    for _ in range(600):
        s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(macro(s, 'prediction') - macro(s, 'prediction_base'))
    return [macro(j, 'prediction') - macro(j, 'prediction_base'),
            *np.percentile(draws, [2.5, 97.5]).tolist()]


def main():
    river_run, ispra_run, horizon_runs = ((sys.argv[1], sys.argv[2], sys.argv[3:]) if len(sys.argv) > 3
                                          else (RIVER_RUN, ISPRA_RUN, HORIZON_RUNS))
    sources = {'v2_river': [river_run], 'v2_ispra': [ispra_run], 'v2_horizon': horizon_runs}
    out = {}
    for variable in ['temperature', 'oxygen']:
        out[variable] = {}
        print(f'== {variable}   (negative beats the plain tree)')
        for horizon in HORIZONS:
            plain = load(horizon_runs, 'v2_horizon', variable, 'tree', horizon)
            if plain is None:
                continue
            entry = {'plain_tree_mae': macro(plain), 'vs_plain': {}}
            line = f'  h{horizon:<3d} plain {entry["plain_tree_mae"]:.4f} | '
            for prefix, arm in ARMS:
                d = load(sources[prefix], prefix, variable, arm, horizon)
                if d is None:
                    continue
                value = paired(plain, d, f'{variable}|{horizon}|{arm}|vsplain')
                entry['vs_plain'][arm] = {'mae': macro(d), 'minus_plain': value}
                star = '*' if value[1] > 0 or value[2] < 0 else ' '
                line += f'{arm.replace("_tree", "")} {value[0]:+.4f}{star} '
            out[variable][horizon] = entry
            print(line)
    path = RUNS / 'v2_augmented_vs_plain.json'
    path.write_bytes(json.dumps(out, indent=2).encode())
    print('\nwrote', path)


if __name__ == '__main__':
    main()
