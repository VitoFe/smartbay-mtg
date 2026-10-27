"""Addendum 45, part A: the weather tree plus river features against the weather tree, swept.

Addendum 44 swept the river arm against the plain tree. Table tab:bay_river stars one more oxygen
contrast at 24 h, the weather tree augmented with the same three river features against the weather
tree (-0.086 umol/L on the published grid and seed). This gives it the same treatment: an arm with
its own study on the published objective at each of the six issuance phases and, on the published
grid, under two further sampler seeds, paired with the phasefix weather and plain trees.

Gates (declared): G1, at offset 0 and seed 42 the arm reproduces the published weather_river_tree
MAE and its contrast with the weather tree; G2, as addendum 44.

Declared in config/bay_reserved_protocol_v2_addendum45.json (part A). Development folds only.
Usage: python -m src.experiments.v2_phase_weather_river --offset 0 [--seed 7] [--quick]
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.bay_river import COLUMNS, river
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import tree_space
from src.experiments.v2_horizon import ROOT, climatology, folds, inner_fold, ratio, tree_fit_score
from src.experiments.v2_phase_river import KEY, TOLERANCE, reused_trees
from src.experiments.v2_phase_shift import build_shifted, design_for, fit_grid, macro, paired
from src.tuning.protocol_v2 import run_study

VARIABLE = 'oxygen'
HORIZON = 24
ARM = 'weather_river_tree'
OUT = ROOT / 'reports/bay_twin/runs/v2_phase_weather_river'
# The published weather-plus-river arm and its contrast with the weather tree, the reference of gate G1
PUBLISHED = ROOT / 'reports/bay_twin/runs/v2_river_summary.json'


def published():
    """(macro MAE, contrast with the weather tree) of the published arm at this horizon."""
    block = json.loads(PUBLISHED.read_text())[VARIABLE][str(HORIZON)]
    return block['mae'][ARM], block['contrasts'][f'{ARM} - weather_tree']['all'][0]


def weather_river_design(data, fold):
    """The weather tree's features plus the three river features, as v2_river.design_for builds them."""
    base = design_for(data, fold, True)
    flow = river(data['times'])[COLUMNS].to_numpy()
    n = len(data['names'])
    return np.concatenate([base, np.broadcast_to(flow[:, None], (len(data['times']), n, flow.shape[1]))],
                          axis=-1)


def run(offset, seed=None, quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '' if seed is None else f'_seed{seed}'
    if seed is not None:
        # run_study reads the sampler seed at call time, so the bound tuning module is left as it is
        from src.tuning import protocol_v2
        protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    data = build_shifted(VARIABLE, HORIZON, offset)
    outer = folds(data, HORIZON)
    inner = [inner_fold(data, f, HORIZON) for f in outer]
    cache = {id(f): weather_river_design(data, f) for f in outer + inner}
    clim = {id(f): climatology(data, f) for f in outer + inner}

    def objective(trial):
        p = tree_space(trial)
        return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))

    selection = run_study(f'{VARIABLE}_{ARM}_h{HORIZON}_shift{offset}_ratio{suffix}', objective, OUT, quick)
    params = {**selection['params'], 'subsample_freq': 1}

    rows = []
    for fold in outer:
        grid = fit_grid(data, fold, cache[id(fold)], params)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for j, i in enumerate(fold['va']):
            for s, name in enumerate(data['names']):
                if mask[j, s]:
                    rows.append({'fold': fold['number'], 'sensor': name, 'time': str(data['times'][i]),
                                 'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                                 ARM: float(grid[j, s] * data['scale'] + data['center'])})
    arm = pd.DataFrame(rows)

    trees, source = reused_trees(offset, suffix)
    joined = arm.merge(trees, on=KEY, how='outer', suffixes=('', '_tree'), indicator=True)
    same_endpoints = bool((joined['_merge'] == 'both').all())
    truth_gap = float((joined.truth - joined.truth_tree).abs().max()) if same_endpoints else float('nan')
    frame = joined.drop(columns=['_merge', 'truth_tree'])
    path = OUT / f'phase_weather_river_{VARIABLE}_offset{offset}{suffix}.csv.gz'
    frame.to_csv(path, index=False)

    mae = {a: macro(frame, a) for a in [ARM, 'weather_tree', 'tree', 'persistence', 'climatology']}
    label = f'{VARIABLE}|weather_river|{offset}|{42 if seed is None else seed}'
    contrasts = {f'{ARM} - weather_tree': paired(frame, ARM, 'weather_tree', label + '|weather'),
                 f'{ARM} - tree': paired(frame, ARM, 'tree', label + '|tree')}
    gates = {'G2': {'passed': source['hash_matches_summary'] and same_endpoints and truth_gap <= TOLERANCE,
                    'reused_file': source, 'same_endpoints': same_endpoints, 'max_truth_difference': truth_gap}}
    if offset == 0 and seed is None:
        reference_mae, reference_contrast = published()
        mae_gap = abs(mae[ARM] - reference_mae)
        contrast_gap = abs(contrasts[f'{ARM} - weather_tree'][0] - reference_contrast)
        gates['G1'] = {'passed': mae_gap <= TOLERANCE and contrast_gap <= TOLERANCE,
                       'mae_difference': mae_gap, 'contrast_difference': contrast_gap}
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum45.json', 'part': 'A',
              'variable': VARIABLE, 'horizon_hours': HORIZON, 'offset_hours': offset,
              'sampler_seed': 42 if seed is None else int(seed), 'objective': 'ratio', 'smoke_only': quick,
              'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
              'selection': selection, 'mae': mae, 'contrasts': contrasts, 'gates': gates,
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(OUT / f'phase_weather_river_offset{offset}{suffix}_summary.json', report)
    print(f'== offset {offset} seed {report["sampler_seed"]}  n={report["endpoints"]}  gates '
          + ', '.join(f'{k} {"ok" if v["passed"] else "FAILED"}' for k, v in gates.items()))
    for k, (d, lo, hi) in contrasts.items():
        print(f'   {k:34s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--offset', type=int, required=True, choices=range(6))
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run(a.offset, a.seed, a.quick)
