"""Addendum-18: does the graph operator's six-hour harm move with the issuance grid too?

Addendum 16 shifted the grid and the oxygen FEATURE harm vanished. It did not touch the OPERATOR
arm, and the abstract's six-hour negative result rests on the operator arms rather than on the
feature arm, so that claim has so far been untested against phase.

Addendum 17 excluded this arm as cluster work. That reason was wrong: the forecast build has two
outer and two inner folds, not the eleven of the spatial build, so a full 100-trial study costs
about half an hour per phase on CPU. It is run here at the full protocol.

Development folds only. Declared in config/bay_reserved_protocol_v2_addendum18.json.
Usage: python -m src.experiments.v2_phase_operator --offset 3
"""
import argparse
import hashlib
import zlib

import numpy as np
import pandas as pd
import torch

from src.experiments.bay_twin_sweep import save
from src.experiments.v2_horizon import (NEURAL_ARMS, SEEDS, THREADS, climatology, folds,
                                        graphs, inner_fold, neural_forecast, neural_space, ratio)
from src.experiments.v2_phase_shift import OUT, build_shifted, fit_grid, design_for, macro, paired
from src.tuning.protocol_v2 import run_study

ARM = 'learned_source'
HORIZON = 6


def run(offset, quick=False, tree_objective_kind='flat'):
    """`tree_objective_kind='ratio'` re-tunes only the matched tree, on the operator's own objective.

    The operator is tuned on the mean over inner folds of fold-macro MAE over persistence MAE, as
    every published arm is; addendum 18 tuned its matched tree on a flat pooled MAE, so the pair
    was selected on different objectives at every shifted phase. In 'ratio' mode
    the operator predictions already written for this offset are reused unchanged, the tree is
    re-tuned on the ratio objective, and everything is written beside the originals with a
    `_treeratio` suffix.
    """
    out = OUT / 'smoke' if quick else OUT          # a smoke run writes beside the published files
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = build_shifted('oxygen', HORIZON, offset)
    graph, exchange, relaxation = NEURAL_ARMS[ARM]
    data['adjacency'] = graphs(data)[graph]
    outer = folds(data, HORIZON)
    inner = [inner_fold(data, f, HORIZON) for f in outer]
    clim = {id(f): climatology(data, f) for f in outer + inner}
    key = ['fold', 'sensor', 'time']
    suffix = '' if tree_objective_kind == 'flat' else '_treeratio'

    if tree_objective_kind == 'flat':
        def objective(trial):
            p = neural_space(trial)
            return float(np.mean([ratio(neural_forecast(data, f, p, exchange, relaxation, 42,
                                                        device, clim[id(f)], quick))
                                  for f in inner]))

        sel = run_study(f'oxygen_{ARM}_shift{offset}', objective, out, quick)
        params = sel['params']
        seeds = [42] if quick else SEEDS

        frames = [neural_forecast(data, f, params, exchange, relaxation, seed, device,
                                  clim[id(f)], quick).assign(seed=seed)
                  for seed in seeds for f in outer]
        operator = pd.concat(frames, ignore_index=True)
        operator = operator.groupby(key, as_index=False).agg(truth=('truth', 'first'),
                                                             prediction=('prediction', 'mean'))
    else:
        published = pd.read_csv(OUT / f'phase_operator_oxygen_offset{offset}.csv.gz')
        operator = published[key + ['truth', 'learned_source']].rename(
            columns={'learned_source': 'prediction'})
        original = OUT / f'phase_operator_oxygen_offset{offset}_summary.json'
        import json
        prior = json.loads(original.read_text())
        sel, seeds = prior['selection'], prior['seeds']

    # The matched plain tree at this phase: as addendum 17 produced it ('flat'), or on the
    # operator's ratio objective ('ratio').
    tree_rows = []
    tree_params = None
    def tree_objective(trial):
        from src.experiments.v2_forecast import tree_space
        p = tree_space(trial)
        if tree_objective_kind == 'ratio':
            from src.experiments.v2_horizon import tree_fit_score
            return float(np.mean([ratio(tree_fit_score(data, design_for(data, f, False), f, p,
                                                       clim[id(f)])[0]) for f in inner]))
        scores = []
        for f in inner:
            grid = fit_grid(data, f, design_for(data, f, False), p)
            m = data['valid'][f['va']] & f['eligible'][None, :]
            scores.append(float(np.nanmean(np.abs(grid - data['truth'][f['va']])[m])))
        return float(np.mean(scores))
    tsel = run_study(f'oxygen_tree_operatorref_shift{offset}{suffix}', tree_objective, out, quick)
    tree_params = {**tsel['params'], 'subsample_freq': 1}
    for fold in outer:
        grid = fit_grid(data, fold, design_for(data, fold, False), tree_params)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for j, i in enumerate(fold['va']):
            for s, name in enumerate(data['names']):
                if mask[j, s]:
                    tree_rows.append({'fold': fold['number'], 'sensor': name,
                                      'time': str(data['times'][i]),
                                      'tree': float(grid[j, s] * data['scale'] + data['center'])})
    tree = pd.DataFrame(tree_rows)

    merged = operator.merge(tree, on=key, validate='one_to_one', suffixes=('', '_tree'))
    if 'truth_tree' in merged:
        merged = merged.drop(columns='truth_tree')
    merged = merged.rename(columns={'prediction': 'learned_source'})
    path = out / f'phase_operator_oxygen_offset{offset}{suffix}.csv.gz'
    merged.to_csv(path, index=False)

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum18.json', 'arm': ARM,
              'tree_objective': tree_objective_kind,
              'offset_hours': offset, 'horizon_hours': HORIZON, 'device': device,
              'issuance_hours_utc': sorted(set(pd.DatetimeIndex(data['times']).hour)),
              'selection': sel, 'tree_selection': tsel, 'seeds': seeds,
              'endpoints': int(len(merged)), 'sensors': int(merged.sensor.nunique()),
              'mae': {a: macro(merged, a) for a in ['learned_source', 'tree']},
              'contrasts': {'learned_source - tree':
                            paired(merged, 'learned_source', 'tree', f'op|{offset}')},
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(out / f'phase_operator_oxygen_offset{offset}{suffix}_summary.json', report)

    print(f'== offset {offset}  issuance {report["issuance_hours_utc"]}  n={report["endpoints"]}')
    for a, v in report['mae'].items():
        print(f'   {a:16s} {v:.4f}')
    d, lo, hi = report['contrasts']['learned_source - tree']
    print(f'   learned_source - tree   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
          + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--offset', type=int, required=True)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--tree-objective', choices=['flat', 'ratio'], default='flat')
    a = p.parse_args()
    run(a.offset, a.quick, a.tree_objective)
