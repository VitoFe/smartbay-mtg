"""Addendum-3 study 1: is a horizon-dependent tree/graph blend worth building?

The proposal was a unified forecaster yhat(h) = w(h)*tree + (1-w(h))*graph, with w(h) going from 1
at short lead to 0 at long lead. Fitting w(h) honestly needs predictions on the inner selection
windows, which the study artifacts do not retain, so a faithful nested version would cost another
campaign.

This computes the *oracle* weight instead: the w(h) that minimises error on the outer folds
themselves, which no honest procedure can beat because it is chosen on the rows it is scored on. If
the oracle blend does not improve materially on the better of its two components, then no nested
blend can, and the campaign is not worth running. The number is reported as an upper bound and
never as a deliverable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RS = [ROOT / 'reports/bay_twin/runs' / r / 'results' for r in sys.argv[1:]]
HORIZONS = [1, 3, 6, 12, 24, 36, 48]
KEY = ['fold', 'sensor', 'time']
GRID = np.linspace(0, 1, 101)


def macro(d, col):
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def load(variable, arm, horizon):
    hits = [p for r in RS for p in r.glob(f'v2_horizon_{variable}_{arm}_h{horizon}/selected_predictions.csv.gz')]
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    if 'seed' in d:
        d = d.groupby(KEY, as_index=False).agg(truth=('truth', 'first'), persistence=('persistence', 'first'),
                                               prediction=('prediction', 'mean'))
    return d[KEY + ['truth', 'persistence', 'prediction']]


def main():
    out = {}
    for variable in ['temperature', 'oxygen']:
        out[variable] = {}
        print(f'== {variable}')
        print(f'{"h":>4s} {"tree":>9s} {"graph":>9s} {"oracle w":>9s} {"blend":>9s} '
              f'{"gain vs best":>13s}  verdict')
        for horizon in HORIZONS:
            tree = load(variable, 'tree', horizon)
            graph = load(variable, 'learned_source', horizon)
            if tree is None or graph is None:
                continue
            j = tree.merge(graph[KEY + ['prediction']], on=KEY, suffixes=('_tree', '_graph'), validate='one_to_one')
            errors = []
            for w in GRID:
                j['blend'] = w * j.prediction_tree + (1 - w) * j.prediction_graph
                errors.append(macro(j, 'blend'))
            errors = np.array(errors)
            best_w = float(GRID[int(errors.argmin())])
            blend = float(errors.min())
            tree_mae, graph_mae = macro(j, 'prediction_tree'), macro(j, 'prediction_graph')
            better = min(tree_mae, graph_mae)
            gain = better - blend
            out[variable][horizon] = {'tree': tree_mae, 'graph': graph_mae, 'oracle_w': best_w,
                                      'oracle_blend': blend, 'oracle_gain_over_best_component': gain,
                                      'endpoints': int(len(j))}
            verdict = 'worth a nested study' if gain > 0.05 * better else 'not worth building'
            print(f'{horizon:4d} {tree_mae:9.4f} {graph_mae:9.4f} {best_w:9.2f} {blend:9.4f} '
                  f'{gain:+13.4f}  {verdict}')
    path = ROOT / 'reports/bay_twin/runs/v2_horizon_blend.json'
    path.write_text(json.dumps(out, indent=2))
    print('\nOracle weights are chosen on the rows they are scored on and are an upper bound only.')
    print('wrote', path)


if __name__ == '__main__':
    main()
