"""Addendum 51: the sequence comparison across six issuance phases and three sampler seeds.

Reads runs/v2_sequence_sweep/panel/ (tree per sampler seed, Chronos-Bolt, TimesFM 2.5, Chronos-2,
persistence and the hour-of-day increment on each grid's paired contexts) and
runs/v2_sequence_sweep/gru/{random_split,calendar}/ (the recurrent arm re-tuned in every cell), and
writes runs/v2_sequence_sweep.json: every contrast of Table bay_sequence_arms in every cell, with the
reading of Chapter 4's rule. Contrasts follow scripts/v2_sequence_table.paired (fold macro, (fold,
ISO week) blocks, 600 draws); the published grid and seed keep the published labels, every other
cell has its own stream 'a51|offset<k>|seed<s>|...'. Contrasts involving the recurrent arm are read
on the calendar design (addendum 49, D4); the random-split sweep is reported beside it.

Addendum 69: by default the panels are read from runs/v2_sequence_sweep/panel_perfold/, where the tree
column of every grid and seed is selected per outer fold (every other column is the saved panel's),
and the report is runs/v2_sequence_sweep_a69.json. `--shared` reads runs/v2_sequence_sweep/panel/ and
writes runs/v2_sequence_sweep.json, the tree at the configuration shared by both folds.

Usage: python -m scripts.v2_sequence_sweep_contrasts [--shared]
Declared in config/bay_reserved_protocol_v2_addendum51.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import zlib

from scripts import v2_block_bootstrap as bb
from scripts.v2_sequence_table import SCALE, macro
from src.experiments.sweep_verdict import verdict as sweep_verdict

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
SWEEP = RUNS / 'v2_sequence_sweep'
SHARED = '--shared' in sys.argv[1:]
#: the panel folder: the tree per outer fold (addendum 69) unless --shared
PANEL = 'panel' if SHARED else 'panel_perfold'
OUT = RUNS / ('v2_sequence_sweep.json' if SHARED else 'v2_sequence_sweep_a69.json')
KEY = ['fold', 'sensor', 'time']
CELLS = [(o, 42) for o in range(6)] + [(0, 7), (0, 123)]
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}
CONTRASTS = [('tree', 'lstm'), ('tree', 'chronos_zs'), ('tree', 'timesfm_zs'), ('lstm', 'chronos_zs'),
             ('lstm', 'timesfm_zs'), ('chronos_zs', 'timesfm_zs'), ('chronos2_zs', 'chronos_zs'),
             ('tree', 'clim_hour'), ('lstm', 'clim_hour'), ('chronos_zs', 'clim_hour'), ('timesfm_zs', 'clim_hour'),
             ('chronos2_zs', 'clim_hour')]
PUBLISHED_ARMS = {'tree', 'lstm', 'chronos_zs', 'timesfm_zs'}


DIVERGED = {}   # (variable, offset, seed, design) -> the networks whose output was not finite


def cell_frame(variable, offset, seed, design):
    """The panel of one grid and seed with the recurrent arm of one design.

    The arm is the mean of its three networks (initialisation seeds 42, 7, 123). A network whose
    output is not finite (training diverged; v2_sequence_arms.fit_recurrent returns it as it stands)
    leaves that mean undefined; such a cell is recorded in DIVERGED and scored on the mean of the
    finite networks, which the declaration did not foresee, so it is reported and flagged."""
    p = SWEEP / PANEL / f'panel_{variable}_offset{offset}.csv.gz'
    g = SWEEP / 'gru' / design / f'{variable}_offset{offset}_seed{seed}' / 'predictions.csv.gz'
    if not p.exists() or not g.exists():
        return None
    f = pd.read_csv(p).rename(columns={f'tree_seed{seed}': 'tree'})
    nets = pd.read_csv(g)
    inits = [c for c in nets.columns if c.startswith('lstm_init')]
    bad = {c.removeprefix('lstm_init'): sorted(int(k) for k in nets.loc[nets[c].isna(), 'fold'].unique())
           for c in inits if nets[c].isna().any()}
    if bad:
        DIVERGED[(variable, offset, seed, design)] = bad
        nets['lstm'] = nets[inits].mean(axis=1, skipna=True)
    f = f.merge(nets[KEY + ['lstm']], on=KEY, how='left', validate='one_to_one')
    assert f.lstm.notna().all(), (variable, offset, seed, design)
    return f


def label(variable, offset, seed, a, b):
    base = f'{variable}|fold|{a}|{b}'
    if (offset, seed) == (0, 42) and {a, b} <= PUBLISHED_ARMS | {'clim_hour'} and 'chronos2_zs' not in (a, b):
        return base
    return f'a51|offset{offset}|seed{seed}|{base}'


def paired(frame, a, b, scale, lab):
    """v2_sequence_table.paired (fold macro, the frame's (fold, ISO week) blocks, 600 draws, generator
    [20260921, crc32(label)]), computed from per-block sums: the same draws, rounded as it rounds."""
    rng = np.random.default_rng([20260921, zlib.crc32(lab.encode())])
    point, samples = bb.draws(frame, [a, b], bb.block_list(frame), rng)
    return [round(x * scale, 4) for x in bb.interval(point, samples, [(1, a), (-1, b)])]


def verdict(by_cell):
    """Chapter 4's rule over six phases (seed 42) and three seeds (offset 0)."""
    vals = {k: v for k, v in by_cell.items() if v is not None}
    if len(vals) < len(CELLS):
        return {'verdict': 'incomplete', 'cells': len(vals)}
    phase = [vals[f'offset{o}|seed42'][0] for o in range(6)]
    seeds = [vals[f'offset0|seed{s}'][0] for s in (42, 7, 123)]
    excl = [(v[0], v[1] > 0 or v[2] < 0) for v in vals.values()]
    mean = float(np.mean(phase))
    v = sweep_verdict([vals[f'offset0|seed{s}'] for s in (42, 7, 123)],
                      [vals[f'offset{o}|seed42'] for o in range(6)])
    return {'verdict': v, 'phase_mean': round(mean, 4), 'phase_range': round(max(phase) - min(phase), 4),
            'seed_values': [round(x, 4) for x in seeds], 'starred': sum(e for _, e in excl), 'cells': len(vals)}


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum51.json', 'cells': [f'offset{o}|seed{s}' for o, s in CELLS],
              'variables': {}}
    for variable in ('temperature', 'oxygen'):
        scale = SCALE[variable]
        v_out = {'designs': {}}
        for design in ('calendar', 'random_split'):
            d_out = {'mae': {}, 'endpoints': {}, 'contrasts': {}}
            frames = {f'offset{o}|seed{s}': cell_frame(variable, o, s, design) for o, s in CELLS}
            for cell, f in frames.items():
                if f is None:
                    continue
                d_out['endpoints'][cell] = int(len(f))
                d_out['mae'][cell] = {a: round(macro(f, a, 'fold') * scale, 4)
                                      for a in ('tree', 'lstm', 'chronos_zs', 'timesfm_zs', 'chronos2_zs', 'persistence', 'clim_hour')}
            for a, b in CONTRASTS:
                by = {}
                for (o, s) in CELLS:
                    f = frames[f'offset{o}|seed{s}']
                    by[f'offset{o}|seed{s}'] = None if f is None else paired(f, a, b, scale, label(variable, o, s, a, b))
                d_out['contrasts'][f'{a} - {b}'] = {'by_cell': by, **verdict(by)}
            d_out['diverged_networks'] = {f'offset{o}|seed{s_}': v for (var, o, s_, dz), v in DIVERGED.items()
                                          if var == variable and dz == design}
            v_out['designs'][design] = d_out
        cal = v_out['designs']['calendar']
        preds = {}
        if variable == 'oxygen':
            preds['S1'] = cal['contrasts']['tree - chronos_zs'].get('verdict') == 'direction'
            preds['S2'] = cal['contrasts']['tree - lstm'].get('verdict') not in ('result', 'incomplete')
            by = cal['contrasts']['chronos_zs - timesfm_zs']['by_cell']
            preds['S4'] = all(v[1] <= 0 <= v[2] for k, v in by.items() if v is not None and k.endswith('seed42'))
        else:
            big = [(c, k, v[0]) for c, e in cal['contrasts'].items() if 'clim_hour' not in c and 'chronos2' not in c
                   for k, v in e['by_cell'].items() if v is not None and abs(v[0]) >= FLOOR[variable]]
            preds['S3'] = {'held': not big, 'exceeding': big}
        by = cal['contrasts']['chronos2_zs - chronos_zs']['by_cell']
        preds['S5'] = all(abs(v[0]) < FLOOR[variable] for v in by.values() if v is not None)
        v_out['predictions'] = preds
        report['variables'][variable] = v_out
        print(f'== {variable}')
        for c, e in cal['contrasts'].items():
            vals = '  '.join(f"{v[0]:+.4f}{'*' if (v[1] > 0 or v[2] < 0) else ' '}" if v else '   n/a   ' for v in e['by_cell'].values())
            print(f'  {c:28s} {vals}   {e.get("verdict")}  mean {e.get("phase_mean")}')
        print('  predictions', preds)
    if not SHARED:
        report['tree_selection'] = 'per outer fold (addendum 69): runs/v2_sequence_sweep/panel_perfold/'
    OUT.write_bytes(json.dumps(report, indent=1, default=str).encode('utf-8'))
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
