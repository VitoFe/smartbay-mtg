"""Addendum 59 part A, the gate: with the fix switched off, the Gaussian process reproduces itself.

Before the corrected process replaces any published column, the code with the fix switched off
(`gpr_fill(..., by_pattern=False)`) must reproduce the published predictions of every run it will
replace, to 1e-9, on 200 episodes per published prediction file drawn with seed 20261001. Nothing is
rerun unless every file passes. The same draw, with the fix switched on, also reports how far the
corrected process moves those rows, which is a description and decides nothing.

Run on the machine that reruns the process (the compute host), from a snapshot holding the
published predictions, with the thread count of the published jobs (two). The process's kernel
optimiser follows a path that depends on the last bits of its linear algebra, which depend on the
number of BLAS threads: a first launch at one thread missed the tolerance on every file (up to
1.4e-4) and a launch at two threads, as published, reproduced every file to 6e-14. Both launches
are kept, and `--combine` writes the record of both.

Usage: python -m scripts.v2_gap_gp_gate [--episodes 200] [--out FILE] [--only RUN]
       python -m scripts.v2_gap_gp_gate --combine   (the two launches -> runs/v2_gap_gp_gate.json)
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_gp_gate.json'
SEED = 20261001
TOLERANCE = 1e-9

#: Every published prediction file that carries a Gaussian-process column the thesis reads, as
#: (run, results directory, arm, the record it was built from). The development short-gap process
#: is the corrected-neighbour rerun that replaced the addendum-30 column (v2_gap_climatology).
PUBLISHED_GP = (
    ('20260922T195802Z_v2_gaprecovery', 'v2_gap_recovery', 'gpr', 'development'),
    ('20260922T204658Z_v2_gaprecovery', 'v2_gap_recovery', 'gpr_k8', 'development'),
    ('20260922T204715Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026', 'gpr', 'reserved'),
    ('20260922T204715Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026', 'gpr_k8', 'reserved'),
    ('20260922T195727Z_v2_gaprecovery', 'v2_gap_recovery_long', 'gpr', 'development'),
    ('20260922T204732Z_v2_gaprecovery', 'v2_gap_recovery_long', 'gpr_k8', 'development'),
    ('20260922T195745Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026', 'gpr', 'reserved'),
    ('20260922T204749Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026', 'gpr_k8', 'reserved'),
)


#: the two gate launches on the compute host (v2_campaign.py --campaign gpgate), as fetched
LAUNCHES = (('20261001T105447Z_v2_gpgate', 'one BLAS thread per job'),
            ('20261001T110138Z_v2_gpgate', 'two BLAS threads per job, as the published jobs ran'))


def combine():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json',
              'rule': f'with the fix switched off, max |new - published| <= {TOLERANCE} and the same '
                      f'rows defined, on 200 episodes per file drawn with seed {SEED}',
              'launches': {}, 'passed': None}
    for run, threads in LAUNCHES:
        files = []
        for path in sorted((RUNS / run / 'results').glob('gate_*.json')):
            files += json.loads(path.read_text(encoding='utf-8'))['files']
        report['launches'][run] = {'threads': threads, 'files': files,
                                   'passed': bool(files) and all(e['passed'] for e in files),
                                   'max_abs_difference': max(e['max_abs_difference'] for e in files)}
    # the gate that decides is the launch run as the published jobs ran
    report['passed'] = report['launches'][LAUNCHES[-1][0]]['passed']
    report['deciding_launch'] = LAUNCHES[-1][0]
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print({r: (v['passed'], v['max_abs_difference']) for r, v in report['launches'].items()},
          '->', OUT)


def draw(published, n):
    keys = published.episode.unique()
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(keys, size=min(n, len(keys)), replace=False))
    return published[published.episode.isin(chosen)].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--out', default=str(OUT))
    p.add_argument('--only', default=None, help='restrict to one published run id')
    p.add_argument('--arm', default=None, choices=['gpr', 'gpr_k8'])
    p.add_argument('--variable', default=None, choices=['temperature', 'oxygen'])
    p.add_argument('--combine', action='store_true')
    a = p.parse_args()
    if a.combine:
        return combine()
    from src.experiments import v2_gap_recovery as G
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json',
              'rule': f'with the fix switched off, max |new - published| <= {TOLERANCE} and the same '
                      f'rows defined, on {a.episodes} episodes per file drawn with seed {SEED}',
              'files': [], 'passed': None}
    for run, results, arm, period in PUBLISHED_GP:
        if (a.only and run != a.only) or (a.arm and arm != a.arm):
            continue
        for variable in ('temperature', 'oxygen'):
            if a.variable and variable != a.variable:
                continue
            path = RUNS / run / results / f'{variable}_{arm}' / 'predictions.csv.gz'
            published = pd.read_csv(path, usecols=['episode', 'position', 'gap_length', 'truth', arm])
            rows = draw(published, a.episodes)
            frame = rows[['episode', 'position']].copy()
            G.PERIOD = period
            started = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                off = G.gpr_fill(frame, variable, neighbours=8 if arm == 'gpr_k8' else 4,
                                 by_pattern=False)
                t_off = time.time() - started
                started = time.time()
                diag = {}
                on = G.gpr_fill(frame, variable, neighbours=8 if arm == 'gpr_k8' else 4,
                                by_pattern=True, diagnostics=diag)
                t_on = time.time() - started
            pub = rows[arm].to_numpy(float)
            same_defined = bool((np.isfinite(off) == np.isfinite(pub)).all())
            both = np.isfinite(off) & np.isfinite(pub)
            worst = float(np.max(np.abs(off[both] - pub[both]))) if both.any() else 0.0
            err_pub = np.abs(rows.truth.to_numpy() - pub)
            err_on = np.abs(rows.truth.to_numpy() - on)
            ok = np.isfinite(err_pub) & np.isfinite(err_on)
            gap = np.abs(off - pub)
            over = both & (gap > TOLERANCE)
            entry = {'run': run, 'results': results, 'variable': variable, 'arm': arm,
                     'period': period, 'episodes': int(rows.episode.nunique()),
                     'rows': int(len(rows)), 'rows_defined_published': int(np.isfinite(pub).sum()),
                     'same_rows_defined': same_defined, 'max_abs_difference': worst,
                     'passed': bool(same_defined and worst <= TOLERANCE),
                     'rows_over_tolerance': int(over.sum()),
                     'episodes_over_tolerance': int(rows.episode[over].nunique()),
                     'rows_over_1e-6': int((both & (gap > 1e-6)).sum()),
                     'median_abs_difference': float(np.median(gap[both])) if both.any() else 0.0,
                     'mae_difference_off_minus_published': float(
                         np.abs(rows.truth.to_numpy()[both] - off[both]).mean()
                         - err_pub[both].mean()) if both.any() else 0.0,
                     'threads': os.environ.get('OMP_NUM_THREADS'),
                     'seconds_fix_off': round(t_off, 1), 'seconds_fix_on': round(t_on, 1),
                     'fix_on': {
                         'rows_defined': int(np.isfinite(on).sum()),
                         'max_abs_change': float(np.nanmax(np.abs(on - pub))),
                         'row_mae_published': round(float(err_pub[ok].mean()), 6),
                         'row_mae_corrected': round(float(err_on[ok].mean()), 6),
                         'share_rows_published_entered_a_silent_neighbour': round(float(
                             np.nanmean(diag['gp_published_substituted'])), 4),
                         'failures': len(diag['failures'])}}
            report['files'].append(entry)
            print(json.dumps(entry), flush=True)
    report['passed'] = bool(report['files']) and all(e['passed'] for e in report['files'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print('gate', 'PASSED' if report['passed'] else 'FAILED', '->', a.out)


if __name__ == '__main__':
    main()
