"""The two diurnal baselines on the endpoints of the sequence-arm comparison (addendum 23).

`table_bay_sequence_arms` compares four architectures on 3,931 temperature and 719 oxygen endpoints
of Dev-2025 (the rows where every arm, the 512 h sequence arms included, has a prediction). Chapter 4
asks for persistence and the climatological increment beside any forecast comparison, and on
Dev-2025 the hour-only increment is the stronger of the two tables. Both are read from the published
six-hour horizon run, where they are fitted on each outer fold's training rows only
(`v2_horizon.climatology`), and joined to the sequence-arm endpoints on (fold, sensor, time); the
truth columns must agree. `tree_published` is the plain tree of that run on the same rows: the
sequence study refits the tree without per-probe weights, the published run weights each probe by the
inverse of its row count, so the two trees differ. The contrasts use the scheme of
`scripts/v2_sequence_table.py`: fold macro, (fold, ISO week) blocks, 600 draws, one seeded stream
per contrast.

Reads runs/v2_sequence_arms/perfold_tree/sequence_arms_<variable>.csv.gz (the tree selected per
outer fold, addendum 69) and
runs/20260920T083322Z_v2_horizon/results/v2_horizon_<variable>_tree_h6/selected_predictions.csv.gz,
and writes sequence_baselines.json beside the rows; there it also reports `tree_perfold_open_loop`,
addendum 47's per-fold plain tree (with its per-probe weights) on the same rows. `--shared` reads and
writes runs/v2_sequence_arms/, the tree shared by both folds.
Usage: python -m scripts.v2_sequence_baselines [--shared]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_sequence_table import ARMS, SCALE, macro, paired

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SHARED = RUNS / 'v2_sequence_arms'
SEQ = SHARED / 'perfold_tree'
PERFOLD_TREE = RUNS / 'v2_perfold/forecast/perfold/{variable}_tree_h6/selected_predictions.csv.gz'
HORIZON = RUNS / '20260920T083322Z_v2_horizon/results/v2_horizon_{variable}_tree_h6/selected_predictions.csv.gz'
CENTER = {'temperature': 20.0, 'oxygen': 230.0}
KEY = ['fold', 'sensor', 'time']


def main():
    global SEQ
    if '--shared' in sys.argv[1:]:
        SEQ = SHARED
    report = {'cohort': 'Dev-2025, the endpoints of table_bay_sequence_arms',
              'baselines_from': '20260920T083322Z_v2_horizon, six-hour plain tree run '
                                '(persistence and hour-only increment, fitted on each outer fold\'s '
                                'training rows)',
              'rule': 'fold macro; contrasts arm - baseline, (fold, ISO week) blocks, 600 draws',
              'variables': {}}
    for variable in ('temperature', 'oxygen'):
        scale, center = SCALE[variable], CENTER[variable]
        seq = pd.read_csv(SEQ / f'sequence_arms_{variable}.csv.gz').dropna(subset=ARMS)
        base = pd.read_csv(HORIZON.as_posix().format(variable=variable))
        base = base[KEY + ['truth', 'persistence', 'clim_hour', 'prediction']].rename(
            columns={'truth': 'truth_h', 'prediction': 'tree_published'})
        f = seq.merge(base, on=KEY, how='left', validate='one_to_one')
        assert f.truth_h.notna().all(), variable
        gap = float(np.abs(f.truth * scale + center - f.truth_h).max())
        assert gap < 1e-6, (variable, gap)
        extra = ['tree_published']
        if SEQ != SHARED:
            pf = pd.read_csv(PERFOLD_TREE.as_posix().format(variable=variable))[KEY + ['prediction']]
            f = f.merge(pf.rename(columns={'prediction': 'tree_perfold_open_loop'}), on=KEY, how='left',
                        validate='one_to_one')
            assert f.tree_perfold_open_loop.notna().all(), variable
            extra.append('tree_perfold_open_loop')
        for col in ('persistence', 'clim_hour', *extra):
            f[col] = (f[col] - center) / scale
        block = {'endpoints': int(len(f)), 'truth_max_abs_gap': gap,
                 'mae': {c: round(macro(f, c, 'fold') * scale, 4)
                         for c in ARMS + ['persistence', 'clim_hour', *extra]},
                 'contrasts': {}}
        for a in ARMS + ['persistence']:
            block['contrasts'][f'{a} - clim_hour'] = paired(f, a, 'clim_hour', 'fold', scale,
                                                            f'{variable}|fold|{a}|clim_hour')
        report['variables'][variable] = block
        print(variable, block['mae'])
        for k, v in block['contrasts'].items():
            print(f'   {k:26s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]' + (' *' if v[1] * v[2] > 0 else ''))
    (SEQ / 'sequence_baselines.json').write_bytes(json.dumps(report, indent=2).encode('utf-8'))
    print('wrote', (SEQ / 'sequence_baselines.json').relative_to(ROOT))


if __name__ == '__main__':
    main()
