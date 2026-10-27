"""Addendum 78: the six-hour river contrast across the six issuance phases and three sampler seeds.

At six hours the river arm's advantage over the plain tree excludes zero only with every arm selected
per outer fold (0.0024 degC and 0.00484 mg/l), and on the reference grid and seed it does not. Every
other feature contrast the thesis gives a verdict has been replicated over the six issuance phases and
under two further sampler seeds, and on this record those two choices move a contrast by as much as
the contrast itself. This module gives the six-hour river contrast, on both variables, the treatment
v2_phase_river gives the 24 h oxygen one. The further seeds are replicates: every one is reported and
none is selected.

For one (variable, phase, seed) it tunes the river arm with its own study on the reference objective
(mean over inner folds of fold-macro MAE / persistence MAE), scores it on the development folds, and
pairs it with the plain tree and the weather tree that run 20260923T103430Z_v2_phasefix tuned at the
same horizon, phase and seed on the same objective. Those trees are reused rather than re-tuned, so
this contrast sits on exactly the predictions of the six-hour phase sweep.

Gates (declared): G1, at offset 0 and seed 42 the river arm reproduces the addendum-6 river_tree MAE
and its contrast with the plain tree; G2, the reused tree file hashes to what its summary records and
holds exactly this arm's endpoints with identical truth.

Declared in config/bay_reserved_protocol_v2_addendum78.json. Development folds only.
Usage: python -m src.experiments.v2_phase_river6 --variable temperature --offset 0 [--seed 7] [--quick]
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import tree_space
from src.experiments.v2_horizon import ROOT, climatology, folds, inner_fold, ratio, tree_fit_score
from src.experiments.v2_phase_river import river_design
from src.experiments.v2_phase_shift import build_shifted, fit_grid, macro, paired
from src.tuning.protocol_v2 import run_study

DECLARED = 'config/bay_reserved_protocol_v2_addendum78.json'
VARIABLES = ('temperature', 'oxygen')
HORIZON = 6
OUT = ROOT / 'reports/bay_twin/runs/v2_phase_river6'
TREES = ROOT / 'reports/bay_twin/runs/20260923T103430Z_v2_phasefix/reports/bay_twin/runs/v2_phase_shift'
KEY = ['fold', 'sensor', 'time']
# The addendum-6 river arm and its contrast with the plain tree, the reference of gate G1
PUBLISHED = ROOT / 'reports/bay_twin/runs/v2_augmented_vs_plain.json'
TOLERANCE = 1e-9


def published(variable):
    """(macro MAE, contrast with the plain tree) of the addendum-6 river arm at six hours."""
    arm = json.loads(PUBLISHED.read_text())[variable][str(HORIZON)]['vs_plain']['river_tree']
    return arm['mae'], arm['minus_plain'][0]


def reused_trees(variable, offset, suffix):
    """The phasefix trees at this phase and seed, after checking the file against its summary (G2)."""
    path = TREES / f'phase_shift_{variable}_offset{offset}_ratio{suffix}.csv.gz'
    summary = json.loads((TREES / f'phase_shift_offset{offset}_ratio{suffix}_summary.json').read_text())
    if summary['horizon_hours'] != HORIZON or summary['offset_hours'] != offset:
        raise SystemExit(f'{path.name}: its summary is for another horizon or phase')
    recorded = summary['variables'][variable]['prediction_sha256']
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path), {'file': path.relative_to(ROOT).as_posix(), 'sha256': actual,
                               'hash_matches_summary': actual == recorded}


def run(variable, offset, seed=None, quick=False):
    # a smoke run writes beside the declared outputs, never over them
    out = OUT / 'smoke' if quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    suffix = '' if seed is None else f'_seed{seed}'
    if seed is not None:
        # run_study reads the sampler seed at call time, so the bound tuning module is left as it is
        from src.tuning import protocol_v2
        protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    data = build_shifted(variable, HORIZON, offset)
    outer = folds(data, HORIZON)
    inner = [inner_fold(data, f, HORIZON) for f in outer]
    cache = {id(f): river_design(data, f) for f in outer + inner}
    clim = {id(f): climatology(data, f) for f in outer + inner}

    def objective(trial):
        p = tree_space(trial)
        return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))

    selection = run_study(f'{variable}_river_tree_h{HORIZON}_shift{offset}_ratio{suffix}', objective, out, quick)
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
                                 'river_tree': float(grid[j, s] * data['scale'] + data['center'])})
    arm = pd.DataFrame(rows)

    trees, source = reused_trees(variable, offset, suffix)
    joined = arm.merge(trees, on=KEY, how='outer', suffixes=('', '_tree'), indicator=True)
    same_endpoints = bool((joined['_merge'] == 'both').all())
    truth_gap = float((joined.truth - joined.truth_tree).abs().max()) if same_endpoints else float('nan')
    g2 = source['hash_matches_summary'] and same_endpoints and truth_gap <= TOLERANCE
    frame = joined.drop(columns=['_merge', 'truth_tree'])
    path = out / f'phase_river6_{variable}_offset{offset}{suffix}.csv.gz'
    frame.to_csv(path, index=False)

    mae = {a: macro(frame, a) for a in ['river_tree', 'tree', 'weather_tree', 'persistence', 'climatology']}
    label = f'{variable}|river|h{HORIZON}|{offset}|{42 if seed is None else seed}'
    contrasts = {'river_tree - tree': paired(frame, 'river_tree', 'tree', label + '|tree'),
                 'river_tree - weather_tree': paired(frame, 'river_tree', 'weather_tree', label + '|weather')}
    gates = {'G2': {'passed': g2, 'reused_file': source, 'same_endpoints': same_endpoints,
                    'max_truth_difference': truth_gap}}
    if offset == 0 and seed is None:
        reference_mae, reference_contrast = published(variable)
        mae_gap = abs(mae['river_tree'] - reference_mae)
        contrast_gap = abs(contrasts['river_tree - tree'][0] - reference_contrast)
        gates['G1'] = {'passed': mae_gap <= TOLERANCE and contrast_gap <= TOLERANCE,
                       'mae_difference': mae_gap, 'contrast_difference': contrast_gap}
    report = {'declared_in': DECLARED, 'variable': variable, 'horizon_hours': HORIZON, 'offset_hours': offset,
              'sampler_seed': 42 if seed is None else int(seed), 'objective': 'ratio', 'smoke_only': quick,
              'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
              'selection': selection, 'mae': mae, 'contrasts': contrasts, 'gates': gates,
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(out / f'phase_river6_{variable}_offset{offset}{suffix}_summary.json', report)
    print(f'== {variable} offset {offset} seed {report["sampler_seed"]}  n={report["endpoints"]}  gates '
          + ', '.join(f'{k} {"ok" if v["passed"] else "FAILED"}' for k, v in gates.items()))
    for k, (d, lo, hi) in contrasts.items():
        print(f'   {k:26s} {d:+.5f} [{lo:+.5f}, {hi:+.5f}]' + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', required=True, choices=VARIABLES)
    p.add_argument('--offset', type=int, required=True, choices=range(6))
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run(a.variable, a.offset, a.seed, a.quick)
