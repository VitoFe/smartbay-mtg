"""The closed-loop rollout of addendum 4, re-read at every lead against both declared baselines.

`src/experiments/v2_closed_loop.py` resamples (fold, sensor) cells. Oxygen has three such cells, and
a macro average over a resample of three cells mostly redraws the same three, so its intervals are
intervals on a sample of three (no interval is reported on fewer than five units). The
published table also showed six of the eight scored leads and led with persistence, which at 24 and
48 h is phase-matched to the diurnal cycle and is strongest exactly there.

This reads the saved predictions and changes nothing in the bound module or its artifacts. Per lead
it reports the frozen (fold, sensor) macro MAE of every arm and the rollout's contrast with
persistence and with the climatological increment, under these resampling schemes:

  week blocks  ISO weeks of the onset drawn with replacement jointly within each fold, every
               cell's episodes of a drawn week entering together, cells fixed (a draw that
               leaves a cell empty is redrawn); this is the reported interval
  within cell  ISO weeks of the onset resampled within each (fold, sensor) cell on their own,
               the earlier scheme, kept beside it from the same seed
  cells        the published scheme, reported only where there are at least five cells

the sign of the contrast in each cell, so a reader can see whether two sondes agree, and the
contrast without every cell of fewer than five episodes, with each such cell's term in the macro.

The oxygen rollout read here is the one whose neighbour column is the rolled state at a step with
no other probe available (v2_closed_loop --rolled-neighbour); temperature has no such step.

Addendum 69: by default the rollouts are the per-fold ones (v2_closed_loop --perfold: every arm at
its outer fold's own selection, runs/v2_closed_loop/perfold/) and the report is
runs/v2_closed_loop_leads_a69.json; `--shared` reads the rollouts at the configuration shared by both
folds and writes runs/v2_closed_loop_leads.json.

Usage: python -m scripts.v2_closed_loop_leads [--shared]
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.v2_bridge import joint_week_draws, MIN_EPISODES

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SHARED = '--shared' in sys.argv[1:]
OUT = RUNS / ('v2_closed_loop_leads.json' if SHARED else 'v2_closed_loop_leads_a69.json')
DRAWS = 600
MIN_UNITS = 5
ARMS = ('recursive_tree', 'graph_denied', 'persistence', 'climatology')
CONTRASTS = (('recursive_tree', 'persistence'), ('recursive_tree', 'climatology'),
             ('graph_denied', 'recursive_tree'))
_FOLDER = 'v2_closed_loop' if SHARED else 'v2_closed_loop/perfold'
SOURCE = {'temperature': f'{_FOLDER}/closed_loop_temperature.csv.gz',
          'oxygen': f'{_FOLDER}/closed_loop_oxygen_rolled_neighbour.csv.gz'}


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def week_blocks(frame, x, y, label):
    """Onset weeks drawn jointly within each fold (reported), and resampled within each (fold, sensor)
    cell (`within_cell`), from the same seed; the cells stay fixed in both."""
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    cells = []
    for _, cell in frame.groupby(['fold', 'sensor']):
        blocks = [g for _, g in cell.groupby('week')]
        cells.append(blocks)
    point = macro(frame, x) - macro(frame, y)
    draws = []
    for _ in range(DRAWS):
        parts = []
        for blocks in cells:
            parts.extend(blocks[i] for i in rng.integers(0, len(blocks), len(blocks)))
        s = pd.concat(parts)
        draws.append(macro(s, x) - macro(s, y))
    lo_w, hi_w = np.percentile(draws, [2.5, 97.5])
    joint, redrawn = joint_week_draws(frame, x, y, np.random.default_rng([20260923, zlib.crc32(label.encode())]),
                                      DRAWS)
    lo, hi = np.percentile(joint, [2.5, 97.5])
    return {'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0),
            'within_cell': {'ci': [round(float(lo_w), 4), round(float(hi_w), 4)],
                            'excludes_zero': bool(lo_w > 0 or hi_w < 0)},
            'joint_redrawn': int(redrawn)}


def sparse(frame, x, y, label):
    """The contrast without every (fold, sensor) cell of fewer than MIN_EPISODES episodes, with its
    joint-week interval, and each such cell's own contrast and term in the published macro."""
    size = frame.groupby(['fold', 'sensor']).size()
    small = size[size < MIN_EPISODES]
    if small.empty:
        return None
    per_fold = size.groupby(level='fold').size()
    terms = {}
    for (fold, sensor), n in small.items():
        c = frame[(frame.fold == fold) & (frame.sensor == sensor)]
        d = float((c.truth - c[x]).abs().mean() - (c.truth - c[y]).abs().mean())
        terms[f'{fold}|{sensor}'] = {'episodes': int(n), 'cell_contrast': round(d, 4),
                                     'term_in_macro': round(d / len(per_fold) / int(per_fold[fold]), 4)}
    rest = frame[frame.set_index(['fold', 'sensor']).index.isin(size[size >= MIN_EPISODES].index)]
    point = macro(rest, x) - macro(rest, y)
    draws, _ = joint_week_draws(rest, x, y, np.random.default_rng([20260923, zlib.crc32(f'{label}|dense'.encode())]),
                                DRAWS)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'sparse_cells': terms, 'without_sparse_cells': {
        'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
        'excludes_zero': bool(lo > 0 or hi < 0), 'cells': int(rest.groupby(['fold', 'sensor']).ngroups)}}


def cell_scheme(frame, x, y, label):
    """The published scheme, refused below MIN_UNITS cells."""
    keys = frame[['fold', 'sensor']].drop_duplicates().values.tolist()
    if len(keys) < MIN_UNITS:
        return {'refused': f'{len(keys)} cells, fewer than {MIN_UNITS}'}
    rng = np.random.default_rng([20260920, zlib.crc32(label.encode())])
    groups = {tuple(k): g for k, g in frame.groupby(['fold', 'sensor'])}
    draws = []
    for _ in range(DRAWS):
        picked = [groups[tuple(keys[i])].assign(fold=n)
                  for n, i in enumerate(rng.integers(0, len(keys), len(keys)))]
        s = pd.concat(picked)
        draws.append(macro(s, x) - macro(s, y))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'ci': [round(float(lo), 4), round(float(hi), 4)], 'excludes_zero': bool(lo > 0 or hi < 0)}


def per_cell(frame, x, y):
    out = {}
    for (fold, sensor), cell in frame.groupby(['fold', 'sensor']):
        e = lambda c: float(cell.dropna(subset=[c]).pipe(lambda f: (f.truth - f[c]).abs().mean()))
        out[f'{fold}|{sensor}'] = round(e(x) - e(y), 4)
    return out


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum4.json',
              'source': {v: f'runs/{s}' for v, s in SOURCE.items()},
              'week_blocks': 'onset weeks drawn jointly within each fold, cells fixed; within_cell: '
                             'onset weeks resampled within each (fold, sensor) cell, the same seed',
              'sparse_cell_rule': f'cells of fewer than {MIN_EPISODES} episodes, reported with their term in the macro',
              'aggregation': 'frozen (fold, sensor) macro: mean over cells within fold, then folds',
              'draws': DRAWS, 'variables': {}}
    if not SHARED:
        report['selection'] = 'per outer fold (addendum 69): every arm at its fold\'s own configuration'
    for variable in ('temperature', 'oxygen'):
        frame = pd.read_csv(RUNS / SOURCE[variable], parse_dates=['time'])
        # v2_closed_loop writes the onset in `time` (the issuance it re-anchors at), not the target
        # hour, so the week block is the onset's own week. Subtracting the lead here would move
        # every block boundary back by the lead.
        onset = frame.time
        iso = onset.dt.isocalendar()
        frame['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
        cells = frame.groupby(['fold', 'sensor']).ngroups
        weeks = frame.groupby(['fold', 'sensor']).week.nunique()
        block = {'cells': int(cells), 'sensors': int(frame.sensor.nunique()),
                 'week_blocks': int(weeks.sum()), 'min_weeks_per_cell': int(weeks.min()),
                 # the unit the reported intervals draw: an onset week within a fold
                 'fold_weeks': int(frame.groupby('fold').week.nunique().sum()),
                 'onset_hours_utc': sorted(int(h) for h in onset.dt.hour.unique()),
                 'by_lead': {}}
        for lead, sub in frame.groupby('lead'):
            entry = {'mae': {a: round(macro(sub, a), 4) for a in ARMS}, 'contrasts': {}}
            for x, y in CONTRASTS:
                label = f'{variable}|{lead}|{x}|{y}'
                entry['contrasts'][f'{x} - {y}'] = {
                    'week_blocks': week_blocks(sub, x, y, label),
                    'cells': cell_scheme(sub, x, y, label),
                    'per_cell': per_cell(sub, x, y)}
                thin = sparse(sub, x, y, label)
                if thin is not None:
                    entry['contrasts'][f'{x} - {y}'].update(thin)
            block['by_lead'][str(int(lead))] = entry
        report['variables'][variable] = block
        for lead, e in block['by_lead'].items():
            c = e['contrasts']
            print(f"{variable:11s} {lead:>2} h  vs persistence {c['recursive_tree - persistence']['week_blocks']}  "
                  f"vs increment {c['recursive_tree - climatology']['week_blocks']}")
    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
