"""Addendum 61: the Arno lag searched up to seven days.

Addendum 26 tuned the lag applied to the river series over 0 to 96 h. This reruns the two Arno arms
(bocca_lag, bocca_gated) for both variables at 6 and 24 h with the lag grid widened to 0 to 168 h, in
the shared and per-fold designs of addendum 48, through that addendum's own runner
(src/experiments/v2_perfold_rest.run_forecast). Only the lag grid, the output folder and the
declaration change; the arms, folds, search space, objective and tuning protocol are the published ones.

Declared in config/bay_reserved_protocol_v2_addendum61.json before anything was run.

Usage (from the repository root, V2_THREADS=2):
  python -m src.experiments.v2_arno_lag7 --variable oxygen --arm bocca_gated --horizon 24
  python -m src.experiments.v2_arno_lag7 --all
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse

from src.experiments import v2_perfold_rest as rest
from src.experiments import v2_river_lag as lagmod

LAG_GRID = list(range(0, 169, 6))          # hours; a superset of addendum 26's 0-96 grid
OUT = rest.RUNS / 'v2_arno_lag7'
DECLARED = 'config/bay_reserved_protocol_v2_addendum61.json'
ARMS = ('bocca_lag', 'bocca_gated')
VARIABLES = ('temperature', 'oxygen')
HORIZONS = (6, 24)
DESIGNS = ('shared', 'perfold')


def cells():
    return [(v, a, h) for v in VARIABLES for a in ARMS for h in HORIZONS]


def run(variable, arm, horizon):
    if arm not in ARMS or variable not in VARIABLES or horizon not in HORIZONS:
        raise ValueError(f'undeclared cell {variable} {arm} {horizon}')
    # the published runner reads these at call time: the grid inside space(), the folder and the
    # declaration when it writes each report
    lagmod.LAG_GRID = LAG_GRID
    rest.OUT = OUT
    rest.DECLARED = DECLARED
    rest.run_forecast('riverlag', variable, arm, horizon, designs=DESIGNS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=VARIABLES)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--horizon', type=int, choices=HORIZONS)
    p.add_argument('--all', action='store_true')
    a = p.parse_args()
    todo = cells() if a.all else [(a.variable, a.arm, a.horizon)]
    for v, arm, h in todo:
        run(v, arm, h)


if __name__ == '__main__':
    main()
