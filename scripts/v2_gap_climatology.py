"""The declared climatology baseline, attached to the published gap-recovery runs.

Addendum 30 declares the climatology comparand as a month-by-hour mean fitted outside the masked
stretch. The published runs of `v2_gap_recovery.build` fitted it over the whole record, masked
stretch included. On short gaps that costs a few per cent. On a month-long gap at a probe with a single
July on record, the cell is mostly the stretch being scored: at 720 h on oxygen the in-sample column
made climatology look like the best filler (8.2 umol/l) when, fitted as declared, it is the worst
(13.5).

No fitted arm ever saw the column, because `feature_columns` drops it, so no arm result changes.
The runs are left exactly as they were written. This rebuilds each run's episodes under the
environment recorded in its transport.json, checks they match the published predictions row for
row, and writes the declared column to `runs/v2_gap_climatology/`, which `v2_gap_intervals` and
`v2_gap_table` read in place of the published one.

Usage: python -m scripts.v2_gap_climatology [<run-id> ...]   (default: the published runs)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SIDECARS = RUNS / 'v2_gap_climatology'
SUMMARY = RUNS / 'v2_gap_climatology.json'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
#: gzip writes the wall-clock time into its header, which would change the digest on every rebuild
GZIP = {'method': 'gzip', 'mtime': 0}

#: The gap-recovery runs the thesis reports, as (run id, results directory, interval artifact).
#: The pipeline runs every producer with no arguments, so each gap script falls back to these, and
#: this is the one place they are written down.
PUBLISHED = (
    ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery', 'v2_gap_intervals.json'),
    ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026',
     'v2_gap_intervals_reserved2026.json'),
    ('20260922T180829Z_v2_gaprecovery', 'v2_gap_recovery_seed20260923',
     'v2_gap_intervals_seed20260923.json'),
)


#: Arms rerun after publication, keyed by the published run, mapped to the run that supersedes them.
#: The addendum-30 Gaussian process took the first four probes in panel order as its neighbours
#: instead of the four nearest; it was rerun on the identical episodes with that fixed. For oxygen
#: the two sets coincide, since only five oxygen sondes exist.
#: A key is an arm, replaced for both variables, or 'variable/arm', replaced for that variable only.
#: Every fitted temperature arm that read the neighbour features had been given the probe's own
#: reading on five probes; they were rerun on the identical episodes with the neighbour ordering
#: fixed. Oxygen had no twin and is not replaced.
#: Addendum 59: the Gaussian process entered a silent neighbour at its margin mean. Both of its
#: arms (four and eight neighbours) were rerun on the identical episodes of every run that carries
#: one, with the linear mean fitted per pattern of reporting neighbours; the four-neighbour
#: development column had already been replaced once (run 20260922T195802Z, neighbours by distance)
#: and is now replaced by the corrected rerun. Any run, not only those in PUBLISHED, may carry an
#: entry: the addendum-33 runs hold the 2026 and eight-neighbour processes that the variants read.
GP_RERUNS = {'development': '20261001T110649Z_v2_gaprecovery',
             'reserved': '20261001T110810Z_v2_gaprecovery',
             'long_development': '20261001T111025Z_v2_gaprecovery',
             'long_reserved': '20261001T111209Z_v2_gaprecovery'}
REPLACED_ARMS = {
    '20260922T171859Z_v2_gaprecovery': {
        'gpr': GP_RERUNS['development'],
        **{f'temperature/{arm}': '20260923T101604Z_v2_gaprecovery'
           for arm in ('regression_causal', 'regression_retro', 'rolling_forward', 'rolling_backward',
                       'hybrid', 'polynomial', 'tabpfn', 'tabicl')}},
    '20260922T204658Z_v2_gaprecovery': {'gpr_k8': GP_RERUNS['development']},
    '20260922T204715Z_v2_gaprecovery': {'gpr': GP_RERUNS['reserved'],
                                        'gpr_k8': GP_RERUNS['reserved']},
    '20260922T195727Z_v2_gaprecovery': {'gpr': GP_RERUNS['long_development']},
    '20260922T204732Z_v2_gaprecovery': {'gpr_k8': GP_RERUNS['long_development']},
    '20260922T195745Z_v2_gaprecovery': {'gpr': GP_RERUNS['long_reserved']},
    '20260922T204749Z_v2_gaprecovery': {'gpr_k8': GP_RERUNS['long_reserved']},
}


def sidecar(run, variable):
    return SIDECARS / f'{run}__{variable}.csv.gz'


def arm_dirs(run, results):
    """(variable, arm) -> the folder holding that arm's predictions, replaced arms resolved."""
    out = {}
    for path in sorted((RUNS / run / results).glob('*/predictions.csv.gz')):
        name = path.parent.name
        variable = 'temperature' if name.startswith('temperature') else 'oxygen'
        arm = name[len(variable) + 1:]
        table = REPLACED_ARMS.get(run, {})
        replacement = table.get(f'{variable}/{arm}', table.get(arm))
        out[(variable, arm)] = (RUNS / replacement / results / name) if replacement else path.parent
    return out


def fix_neighbours(frame, run, variable):
    """Replace the published `nearest_probe` with the one rebuilt without the self-neighbour leak.

    On five temperature probes the published column was the probe's own reading.
    `scripts/v2_gap_neighbours.py` writes the corrected column per run; a temperature frame from a
    run it has not rebuilt is refused rather than read with the leak in it.
    """
    if variable != 'temperature' or 'nearest_probe' not in frame:
        return frame
    path = RUNS / 'v2_gap_neighbours' / f'{run}__{variable}.csv.gz'
    if not path.exists():
        raise SystemExit(f'{run}: no corrected nearest_probe; run scripts/v2_gap_neighbours.py')
    side = pd.read_csv(path, usecols=['episode', 'position', 'nearest_probe'])
    out = frame.drop(columns='nearest_probe').merge(
        side, on=['episode', 'position'], how='left', validate='one_to_one', indicator=True)
    if (out._merge != 'both').any():
        raise SystemExit(f'{run}: corrected nearest_probe does not cover the published rows')
    # More rows are undefined than before, and legitimately: where the real twin is dark at the
    # gap's edge its offset cannot be formed, whereas the probe's own reading always could be.
    return out.drop(columns='_merge')


def read_arm(run, results, variable, arm, usecols=None):
    """One arm's published predictions as written, with a replaced arm read from the run that
    supersedes it, refused unless it scores exactly the rows of the arm it replaces. No column is
    substituted; `load_predictions` adds the declared climatology and the corrected nearest probe."""
    folder = arm_dirs(run, results)[(variable, arm)]
    frame = pd.read_csv(folder / 'predictions.csv.gz')
    if folder.parent.parent.name != run:
        keys = pd.read_csv(RUNS / run / results / folder.name / 'predictions.csv.gz',
                           usecols=['episode', 'position'])
        if not keys.equals(frame[['episode', 'position']]):
            raise SystemExit(f'{folder} does not score the rows of the arm it replaces')
    return frame if usecols is None else frame[list(usecols)]


def load_predictions(run, results, variable, arm):
    """One arm's predictions with the declared climatology and the corrected nearest probe.

    A run built with the declared climatology records its design in report.json and already carries the
    declared column. An earlier one gets it from its sidecar, or loses the column if none exists. A
    replaced arm must have scored exactly the rows of the arm it replaces, or it is refused.
    """
    folder = arm_dirs(run, results)[(variable, arm)]
    frame = pd.read_csv(folder / 'predictions.csv.gz')
    original = RUNS / run / results / folder.name / 'predictions.csv.gz'
    if folder.parent.parent.name != run:
        keys = pd.read_csv(original, usecols=['episode', 'position'])
        if not keys.equals(frame[['episode', 'position']]):
            raise SystemExit(f'{folder} does not score the rows of the arm it replaces')
    frame = fix_neighbours(frame, run, variable)
    if 'design' in json.loads((folder / 'report.json').read_text()):
        return frame
    declared = sidecar(run, variable)
    if declared.exists():
        side = pd.read_csv(declared, usecols=['episode', 'position', 'climatology'])
        return frame.drop(columns='climatology').merge(
            side, on=['episode', 'position'], how='left', validate='one_to_one')
    return frame.drop(columns='climatology')


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def worker(run, variable):
    """Runs under the run's own environment, so the module-level switches match what it used."""
    from src.experiments.v2_gap_recovery import OUT, SEED, build
    published = sorted((RUNS / run / OUT.name).glob(f'{variable}_*/predictions.csv.gz'))
    if not published:
        raise SystemExit(f'no {variable} predictions under {run}/{OUT.name}')
    pub = pd.read_csv(published[0])
    frame = build(variable, np.random.default_rng(SEED))
    new = frame[['episode', 'position', 'gap_length', 'sensor', 'truth', 'climatology']]
    joined = pub[['episode', 'position', 'truth', 'climatology']].merge(
        new, on=['episode', 'position'], how='outer', suffixes=('_published', ''), indicator=True)
    unmatched = int((joined._merge != 'both').sum())
    drift = float(np.nanmax(np.abs(joined.truth - joined.truth_published)))
    if unmatched or drift > 1e-9:
        raise SystemExit(f'{run} {variable}: rebuilt episodes do not match the published ones '
                         f'({unmatched} unmatched rows, truth drift {drift})')
    SIDECARS.mkdir(parents=True, exist_ok=True)
    out = joined[['episode', 'position', 'gap_length', 'sensor', 'truth', 'climatology']]
    out.sort_values(['episode', 'position']).to_csv(sidecar(run, variable), index=False,
                                                    compression=GZIP)
    report = {'results': OUT.name, 'rows': int(len(out)),
              'rows_without_outside_data': int(out.climatology.isna().sum()), 'by_gap_length': {}}
    both = joined.rename(columns={'climatology_published': 'in_sample', 'climatology': 'declared'})
    for scope, part in (('all probes', both), ('Baia EXO2', both[both.sensor == EXO2])):
        rows = {}
        for length, block in part.groupby('gap_length'):
            rows[str(int(length))] = {'episodes': int(block.episode.nunique()),
                                      'in_sample': macro(block, 'in_sample'),
                                      'declared': macro(block, 'declared')}
        if rows:
            report['by_gap_length'][scope] = rows
    print(json.dumps(report))


def main():
    if sys.argv[1:2] == ['--worker']:
        return worker(sys.argv[2], sys.argv[3])
    runs = sys.argv[1:] or [run for run, _, _ in PUBLISHED]
    summary = {'declared_in': 'config/bay_reserved_protocol_v2_addendum30.json',
               'definition': 'month-by-hour mean of the probe, fitted outside the masked stretch',
               'runs': {}}
    for run in runs:
        recorded = json.loads((RUNS / run / 'transport.json').read_text()).get('forwarded_env', {})
        env = {k: v for k, v in os.environ.items() if not k.startswith('V2_GAP_')}
        env.update(recorded)
        env['PYTHONHASHSEED'] = '0'
        summary['runs'][run] = {'forwarded_env': recorded, 'variables': {}}
        for variable in ('temperature', 'oxygen'):
            done = subprocess.run([sys.executable, '-m', 'scripts.v2_gap_climatology', '--worker',
                                   run, variable], cwd=ROOT, env=env, capture_output=True,
                                  text=True)
            if done.returncode:
                raise SystemExit(done.stderr[-2000:] or done.stdout[-2000:])
            report = json.loads(done.stdout.strip().splitlines()[-1])
            summary['runs'][run]['variables'][variable] = report
            print(f'{run} {variable}: {report["rows"]} rows matched; declared climatology written')
            for length, e in report['by_gap_length'].get('all probes', {}).items():
                print(f'   {length:>5} h  in-sample {e["in_sample"]}  declared {e["declared"]}')
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print('wrote', SUMMARY)


if __name__ == '__main__':
    main()
