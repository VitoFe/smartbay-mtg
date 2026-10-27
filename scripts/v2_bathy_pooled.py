"""Addendum-8 part B, pooled evaluation: seabed features against BOTH reference arms.

The addendum's decision rule is explicit that an arm is reported against the weather tree and
against the plain tree, because measuring only against the weather tree manufactures gains at six
hours on oxygen, where the weather features are themselves harmful.

Usage: python scripts/v2_bathy_pooled.py <bathy-run> <horizon-run> [<horizon-run> ...]
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
KEY = ['fold', 'sensor', 'time']
HORIZONS = [6, 24]
REFERENCES = [('tree', 'plain tree'), ('weather_tree', 'weather tree')]


def macro(d, col='prediction'):
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def load(runs, prefix, variable, arm, horizon):
    for r in runs:
        hits = list((RUNS / r / 'results').glob(
            f'{prefix}_{variable}_{arm}_h{horizon}/selected_predictions.csv.gz'))
        if hits:
            d = pd.read_csv(hits[0])
            if 'seed' in d:
                d = d.groupby(KEY, as_index=False).agg(truth=('truth', 'first'),
                                                       prediction=('prediction', 'mean'))
            return d[KEY + ['truth', 'prediction']]
    return None


def paired(base, arm, stream):
    j = base.merge(arm[KEY + ['prediction']], on=KEY, suffixes=('_base', ''), validate='one_to_one')
    rng = np.random.default_rng([20260921, zlib.crc32(stream.encode())])
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
    bathy_run, horizon_runs = sys.argv[1], sys.argv[2:]
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum8.json', 'results': {}}
    for variable in ['temperature', 'oxygen']:
        out['results'][variable] = {}
        print(f'== {variable}   (negative favours the bathymetric arm)')
        bathy_all = {h: load([bathy_run], 'v2_bathy', variable, 'bathy_tree', h) for h in HORIZONS}
        for horizon in HORIZONS:
            bathy = bathy_all[horizon]
            if bathy is None:
                continue
            entry = {'bathy_tree_mae': macro(bathy), 'contrasts': {}}
            print(f'  h{horizon}: bathy_tree {entry["bathy_tree_mae"]:.4f}')
            for arm, label in REFERENCES:
                ref = load(horizon_runs, 'v2_horizon', variable, arm, horizon)
                if ref is None:
                    print(f'      vs {label:12s} MISSING')
                    continue
                value = paired(ref, bathy, f'{variable}|{horizon}|{arm}')
                entry['contrasts'][arm] = {'reference_mae': macro(ref), 'bathy_minus_reference': value}
                star = '*' if value[1] > 0 or value[2] < 0 else ' '
                print(f'      vs {label:12s} {macro(ref):.4f}   '
                      f'{value[0]:+.4f} [{value[1]:+.4f}, {value[2]:+.4f}]{star}')
            out['results'][variable][horizon] = entry
    path = RUNS / 'v2_bathy_pooled.json'
    path.write_text(json.dumps(out, indent=2))
    print('\nwrote', path)


if __name__ == '__main__':
    main()
