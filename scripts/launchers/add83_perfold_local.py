"""Addendum 83's per-fold studies, run locally (Optuna 5.0.0, where every per-fold forecasting study ran).

Runs `python -m src.experiments.v2_lead_family perfold` for every (arm, variable, lead) without a published
per-fold value outside the ISPRA family (36 arm cells, two fold studies each; amendment 1 runs the 20 ISPRA cells
on the compute host), --jobs at a time, two threads each, after gate L-G7 has passed. A cell whose report.json exists is skipped; logs go to reports/bay_twin/runs/v2_addendum83/logs/perfold/.

usage: python scripts/launchers/add83_perfold_local.py --python <interpreter> [--jobs 5]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.v2_lead_family_contrasts import FAMILY, LEADS, PUBLISHED_REF, VARIABLES  # noqa: E402

HERE = REPO / 'reports/bay_twin/runs/v2_addendum83'


def cells():
    """Amendment 1: the ISPRA-based arms run on the compute host (add83_perfold_host.py); the other 36 cells here."""
    return [(arm, v, h) for v in VARIABLES for h in LEADS for arm in FAMILY
            if (arm, v, h) not in PUBLISHED_REF and FAMILY[arm] != 'ispra']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', required=True)
    p.add_argument('--jobs', type=int, default=5)
    a = p.parse_args()
    if not json.loads((HERE / 'gates/L-G7.json').read_text(encoding='utf-8'))['pass']:
        raise SystemExit('L-G7 has not passed')
    logs = HERE / 'logs' / 'perfold'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, V2_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('V2_ISPRA_BINNING', None)
    env.pop('V2_ADD83_OUT', None)
    todo = [c for c in cells() if not (HERE / 'perfold' / f'{FAMILY[c[0]]}_{c[1]}_{c[0]}_h{c[2]}' / 'report.json').exists()]
    print(len(cells()), 'cells,', len(todo), 'to run', flush=True)
    running, t0 = {}, time.time()
    while todo or running:
        for k, proc in list(running.items()):
            if proc.poll() is not None:
                print(f'{time.time() - t0:7.0f} s  done {k} exit {proc.returncode}', flush=True)
                del running[k]
        if todo and len(running) < a.jobs:
            arm, v, h = todo.pop(0)
            name = f'{FAMILY[arm]}_{v}_{arm}_h{h}'
            with open(logs / f'{name}.log', 'w') as log:
                running[name] = subprocess.Popen([a.python, '-u', '-m', 'src.experiments.v2_lead_family', 'perfold',
                                                  '--arm', arm, '--variable', v, '--horizon', str(h)],
                                                 cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            continue
        time.sleep(5)
    print(f'all done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
