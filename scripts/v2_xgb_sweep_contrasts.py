"""Addendum 50: XGBoost - LightGBM at six hours over six issuance phases and three sampler seeds.

Reads the per-fold selections of `src/experiments/v2_xgb_sweep.py` (offsets 1-5 at seed 42, seeds 7 and
123 at offset 0) and the offset-0, seed-42 per-fold selections of addenda 47 (LightGBM tree) and 48
(XGBoost tree), and reports for each variable the contrast xgb_tree - tree per replicate (week-block
bootstrap, 600 draws, generator [20260920, crc32('a50|<variable>|<offset>|<seed>|xgb_tree-tree')]), its
verdict by Chapter 4's rule, and beside it the ten-stream spread of addendum 48 amendment 1.
Secondary, descriptive and seed-only: xgb_tree - the conservative graph on oxygen at offset 0, the graph
as published under each seed (shared design, three-network mean).

Usage: python -m scripts.v2_xgb_sweep_contrasts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_perfold_contrasts import verdict
from src.experiments.sweep_verdict import verdict as sweep_verdict
from scripts.v2_perfold_rest_contrasts import Blocks, star

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
WORK = RUNS / 'v2_xgb_sweep'
OUT = RUNS / 'v2_xgb_sweep.json'
KEY = ['fold', 'sensor', 'time']
REPLICATES = [(0, 42)] + [(o, 42) for o in range(1, 6)] + [(0, 7), (0, 123)]
GRAPH = {42: RUNS / '20260920T083322Z_v2_horizon/results',
         7: RUNS / '20260923T110726Z_v2_seedrep/results',
         123: RUNS / '20260923T110752Z_v2_seedrep/results'}


def predictions(variable, arm, offset, seed):
    if (offset, seed) == (0, 42):
        folder = (RUNS / 'v2_perfold/forecast/perfold' / f'{variable}_tree_h6' if arm == 'tree'
                  else RUNS / 'v2_perfold_rest/forecast/perfold' / f'xgb_{variable}_xgb_tree_h6')
    else:
        folder = WORK / f'{variable}_{arm}_h6_shift{offset}_seed{seed}'
    d = pd.read_csv(folder / 'selected_predictions.csv.gz')
    d['time'] = d.time.astype(str)
    return d, json.loads((folder / 'report.json').read_text())


def seeds_only(seeds):
    """Chapter 4's rule over seeds alone (a 'seeds only' contrast), src.experiments.sweep_verdict."""
    return sweep_verdict(list(seeds.values()))


def main():
    gate = json.loads((WORK / 'gate_G1.json').read_text())
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum50.json',
           'amended_by': 'config/bay_reserved_protocol_v2_addendum50_amendment1.json',
           'gates': {'G1_build_offset0_identical': all(gate[v]['pass'] for v in ('temperature', 'oxygen'))},
           'variables': {}}
    trials, repeats = [], []
    rest = json.loads((RUNS / 'v2_perfold_rest.json').read_text())
    for variable in ('temperature', 'oxygen'):
        entry = {'replicates': {}}
        for offset, seed in REPLICATES:
            frames, reps = {}, {}
            for arm in ('tree', 'xgb_tree'):
                frames[arm], reps[arm] = predictions(variable, arm, offset, seed)
                trials += [s['completed_trials'] for s in reps[arm]['selection'].values()]
                if 'gate_G3_pass' in reps[arm]:
                    repeats.append(reps[arm]['gate_G3_pass'])
            j = frames['tree'][KEY + ['truth', 'persistence', 'prediction']].rename(columns={'prediction': 'tree'}).merge(
                frames['xgb_tree'][KEY + ['prediction']].rename(columns={'prediction': 'xgb_tree'}),
                on=KEY, how='inner', validate='one_to_one')
            assert len(j) == len(frames['tree']) == len(frames['xgb_tree']), (variable, offset, seed)
            b = Blocks(j)
            c = b.boot({'xgb_tree': 1.0, 'tree': -1.0}, 20260920, f'a50|{variable}|{offset}|{seed}|xgb_tree-tree')
            entry['replicates'][f'{offset}|{seed}'] = {
                'offset_hours': offset, 'sampler_seed': seed, 'endpoints': int(len(j)),
                'mae': {'tree': b.value({'tree': 1.0}), 'xgb_tree': b.value({'xgb_tree': 1.0}),
                        'persistence': b.value({'persistence': 1.0})},
                'xgb_tree - tree': c, 'excludes_zero': star(c[1:])}
            print(f'{variable:11s} offset {offset} seed {seed:3d}  xgb - tree {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]'
                  f'{"*" if star(c[1:]) else " "}', flush=True)
        r = entry['replicates']
        phases = {str(o): r[f'{o}|42']['xgb_tree - tree'] for o in range(6)}
        seeds = {str(s): r[f'0|{s}']['xgb_tree - tree'] for s in (42, 7, 123)}
        points = [x[0] for x in phases.values()]
        entry['phase_mean'] = float(np.mean(points))
        entry['phase_range'] = float(max(points) - min(points))
        entry['phases_starred'] = int(sum(star(x[1:]) for x in phases.values()))
        entry['seed_points'] = {k: x[0] for k, x in seeds.items()}
        entry['verdict'] = verdict(seeds, phases)
        s = rest['forecast']['streams'][variable]
        entry['published_grid_stream_spread'] = {'contrast_range': s['contrast_range'], 'spread': s['contrast_spread'],
                                                 'streams_excluding_zero': s['streams_excluding_zero']}
        out['variables'][variable] = entry
        print(f'{variable}: phase mean {entry["phase_mean"]:+.4f} range {entry["phase_range"]:.4f} '
              f'starred {entry["phases_starred"]}/6 seeds {entry["seed_points"]} -> {entry["verdict"]}', flush=True)
    # Secondary: oxygen, xgb_tree - conservative graph, seeds only, the graph as published.
    sec = {}
    for seed in (42, 7, 123):
        x, _ = predictions('oxygen', 'xgb_tree', 0, seed)
        g = pd.read_csv(GRAPH[seed] / 'v2_horizon_oxygen_conservative_exchange_h6/selected_predictions.csv.gz')
        g = g.groupby(KEY, as_index=False).prediction.mean()
        g['time'] = g.time.astype(str)
        j = x[KEY + ['truth', 'prediction']].rename(columns={'prediction': 'xgb_tree'}).merge(
            g.rename(columns={'prediction': 'conservative'}), on=KEY, how='inner', validate='one_to_one')
        assert len(j) == len(x), seed
        c = Blocks(j).boot({'xgb_tree': 1.0, 'conservative': -1.0}, 20260920, f'a50|oxygen|0|{seed}|xgb_tree-conservative')
        sec[str(seed)] = c
        print(f'oxygen seed {seed:3d}  xgb - conservative graph {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]'
              f'{"*" if star(c[1:]) else " "}', flush=True)
    out['secondary_oxygen_xgb_minus_conservative_graph'] = {
        'note': 'descriptive, seeds only: the graph arm as published under each sampler seed (shared design, '
                'three-network mean), the XGBoost tree selected per fold',
        'seeds': sec, 'verdict': seeds_only(sec)}
    out['gates']['G2_trials'] = bool(all(t >= 100 for t in trials))
    out['gates']['G3_refit_repeats'] = bool(all(repeats))
    t, o = out['variables']['temperature'], out['variables']['oxygen']
    out['registered_predictions'] = {
        'X1': {'held': bool(t['phases_starred'] == 0 and not any(star(x[1:]) for x in [t['replicates']['0|7']['xgb_tree - tree'],
                                                                                        t['replicates']['0|123']['xgb_tree - tree']])
                            and abs(t['phase_mean']) <= 0.002), 'verdict': t['verdict'], 'phase_mean': t['phase_mean']},
        'X2': {'held': o['verdict'] in ('not estimable', 'not distinguishable'),   # the declaration's 'not detected'
               'verdict': o['verdict']},
        'X3': {'held': len({np.sign(x[0]) for x in sec.values()}) > 1,
               'verdict': out['secondary_oxygen_xgb_minus_conservative_graph']['verdict']}}
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(out['gates']), {k: v['held'] for k, v in out['registered_predictions'].items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
