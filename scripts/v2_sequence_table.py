"""Score the addendum-23 arms from the saved per-endpoint predictions, under both macro rules.

The declaration specified the reserved-period convention, a mean over quarters of the per-sensor
mean. Every other development-fold contrast in this thesis aggregates over *folds* instead
(v2_phase_shift, and the horizon sweep behind it), because the folds are the chronological blocks
the cohort is built from. Both are computed here: the fold rule is reported as primary so these
figures sit on the same footing as the rest of the development evidence, and the declared quarter
rule beside it, so the deviation is visible rather than silent.

Reads runs/v2_sequence_arms/perfold_tree/sequence_arms_<variable>.csv.gz, the rows with the tree
selected per outer fold (addendum 69; every other arm is the published rows), and writes
sequence_arms_table.json beside them. `--shared` reads and writes runs/v2_sequence_arms/ instead, the
tree at the configuration shared by both folds.
Usage: python -m scripts.v2_sequence_table [--shared]
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / 'reports/bay_twin/runs/v2_sequence_arms'
PERFOLD = SHARED / 'perfold_tree'
RUNS = PERFOLD
ARMS = ['tree', 'lstm', 'chronos_zs', 'timesfm_zs']
SCALE = {'temperature': 10.0, 'oxygen': 50.0}
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}
DRAWS = 600


def macro(frame, column, outer):
    f = frame.dropna(subset=[column])
    cell = f.assign(e=(f[column] - f.truth).abs()).groupby([outer, 'sensor']).e.mean()
    return float(cell.groupby(outer).mean().mean())


def paired(frame, a, b, outer, scale, label):
    rng = np.random.default_rng([20260921, zlib.crc32(label.encode())])
    f = frame.assign(week=pd.to_datetime(frame.time, utc=True).dt.isocalendar().week.astype(int))
    blocks = f[['fold', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in f.groupby(['fold', 'week'])}
    draws = []
    for _ in range(DRAWS):
        pick = rng.integers(0, len(blocks), len(blocks))
        s = pd.concat([groups[tuple(blocks[i])] for i in pick])
        draws.append((macro(s, a, outer) - macro(s, b, outer)) * scale)
    point = (macro(frame, a, outer) - macro(frame, b, outer)) * scale
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(point, 4), round(float(lo), 4), round(float(hi), 4)]


def score(variable):
    frame = pd.read_csv(RUNS / f'sequence_arms_{variable}.csv.gz')
    frame = frame.dropna(subset=ARMS)
    scale, floor = SCALE[variable], FLOOR[variable]
    out = {'variable': variable, 'endpoints': int(len(frame)),
           'floor': floor, 'rules': {}}
    for label, outer in [('fold (primary; matches the other development contrasts)', 'fold'),
                         ('quarter (as declared in addendum 23)', 'quarter')]:
        block = {'mae': {a: round(macro(frame, a, outer) * scale, 4) for a in ARMS},
                 'contrasts': {}}
        for i, a in enumerate(ARMS):
            for b in ARMS[i + 1:]:
                block['contrasts'][f'{a} - {b}'] = paired(frame, a, b, outer, scale,
                                                          f'{variable}|{outer}|{a}|{b}')
        out['rules'][label] = block
    return out


def main():
    global RUNS
    if '--shared' in sys.argv[1:]:
        RUNS = SHARED
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum23.json',
              'cohort': 'Forecast-2025 development folds; no reserved endpoint read',
              'variables': {}}
    if RUNS == PERFOLD:
        report['tree_selection'] = 'per outer fold (addendum 69); the other arms are the published rows'
    for variable in ('temperature', 'oxygen'):
        path = RUNS / f'sequence_arms_{variable}.csv.gz'
        if not path.exists():
            print(f'  {variable}: predictions not present, skipped')
            continue
        r = score(variable)
        report['variables'][variable] = r
        print(f"== {variable}  n={r['endpoints']}  floor {r['floor']}")
        for label, block in r['rules'].items():
            print(f"   -- {label}")
            print('      mae ' + '  '.join(f'{a} {v}' for a, v in block['mae'].items()))
            for k, v in block['contrasts'].items():
                star = '*' if v[1] * v[2] > 0 else ' '
                below = 'below floor' if abs(v[0]) < r['floor'] else 'ABOVE FLOOR'
                print(f'      {k:26s} {v[0]:+9.4f} [{v[1]:+.4f}, {v[2]:+.4f}] {star}  {below}')
    (RUNS / 'sequence_arms_table.json').write_bytes(json.dumps(report, indent=2).encode('utf-8'))
    print('\nwrote', (RUNS / 'sequence_arms_table.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
