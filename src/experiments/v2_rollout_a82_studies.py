"""Addendum 82, part A: the 2026 studies of the silent-probe rollout's step tree and rolled graphs.

On Bridge-Dev the rollout's step tree (the six-hour weather tree), the oxygen sonde's temperature
companion and the two rolled graph operators take the configuration selected on each outer fold's
own inner window (addenda 47 and 49). Bridge-2026 is its own outer fold (`v2_bridge.fold_2026` on the
2026-inclusive grid of `v2_horizon_reserved.build`), so each arm gets one study on that fold's inner
window, `v2_horizon.inner_fold(reserved grid, fold_2026, 6)`: validation issuances from 2025-11-14 to
before 2025-12-29, inner training targets before 2025-11-11. The objective is the per-fold objective of
addenda 47 (trees) and 49 (graphs): fold-macro six-hour forecast MAE over persistence MAE on that inner
window, graphs trained at initialisation seed 42. Nothing here reads a 2026 row: the inner window and
its training rows end before 2026 by construction, and the audit is checked (gate A-G6).

Gate A-G6 also evaluates this module's objective code at the stored fold-1 parameters of addendum 47
(trees, to 1e-9) and addendum 49 (graphs, to 1e-6) on Dev fold 1's inner window.

Declared in config/bay_reserved_protocol_v2_addendum82.json (part_A) before anything here was run.

Usage (V2_THREADS=2; trees with the GPUs hidden, graphs on one GPU):
  python -m src.experiments.v2_rollout_a82_studies --variable temperature --arm weather_tree [--quick]
  python -m src.experiments.v2_rollout_a82_studies --gate-only --variable oxygen --arm learned_source
`selection(variable, arm)` returns a finished study's configuration after checking it.
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_addendum82/A/studies'
DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json part_A'
HORIZON = 6
ARMS = ('weather_tree', 'conservative_exchange', 'learned_source')
MIN_TRIALS = 100
RESERVED_START = pd.Timestamp('2026-01-01', tz='UTC')
#: what the inner-window audit must show (gate A-G6)
INNER_VALIDATION_END = pd.Timestamp('2025-12-29', tz='UTC')
LAST_TRAINING_TARGET_BEFORE = pd.Timestamp('2025-11-11', tz='UTC')


def study_name(variable, arm):
    return f'a82A_{variable}_{arm}_h{HORIZON}_off0_seed42_fold2026'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, default=str).encode('utf-8'))


def environment():
    import lightgbm
    import optuna
    import torch
    return {'host': socket.gethostname(), 'platform': platform.platform(), 'python': sys.version.split()[0],
            'optuna': optuna.__version__, 'lightgbm': lightgbm.__version__, 'torch': torch.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__,
            'V2_THREADS': os.environ.get('V2_THREADS'), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_available': bool(torch.cuda.is_available())}


# --------------------------------------------------------------------------------------------------
# The reserved grid, its outer fold and inner window
# --------------------------------------------------------------------------------------------------
def reserved_setup(variable):
    from src.experiments import v2_horizon as vh
    from src.experiments import v2_horizon_reserved as hr
    from src.experiments.v2_bridge import fold_2026
    data = hr.build(variable, HORIZON)
    outer = fold_2026(data)
    inner = vh.inner_fold(data, outer, HORIZON)
    return data, outer, inner


def inner_audit(data, inner):
    t = data['times']
    last_target = t[inner['tr']].max() + pd.Timedelta(hours=HORIZON)
    first_val, last_val = t[inner['va']].min(), t[inner['va']].max()
    last_val_target = last_val + pd.Timedelta(hours=HORIZON)
    ok = bool(last_target < LAST_TRAINING_TARGET_BEFORE and last_val_target < INNER_VALIDATION_END
              and last_val_target < RESERVED_START)
    return {'first_validation_issuance': str(first_val), 'last_validation_issuance': str(last_val),
            'last_validation_target': str(last_val_target), 'validation_end_exclusive': str(INNER_VALIDATION_END),
            'last_training_target': str(last_target),
            'training_targets_before': str(LAST_TRAINING_TARGET_BEFORE),
            'validation_issuances': int(len(inner['va'])), 'training_issuances': int(len(inner['tr'])),
            'eligible_sensors': [n for n, e in zip(data['names'], inner['eligible']) if e],
            'no_2026_row': ok, **inner.get('audit', {})}


def weather_design(data, fold, met):
    from src.experiments.bay_dynamics_tree import features
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']),
                                                                met.shape[1]))], axis=-1)


def reserved_weather(times):
    from src.experiments import bay_reserved_evaluation as v1
    return v1.load_weather_2026(times).to_numpy()


# --------------------------------------------------------------------------------------------------
# The objectives of addenda 47 (trees) and 49 (graphs), on one inner window
# --------------------------------------------------------------------------------------------------
class TreeObjective:
    """`v2_perfold.run_forecast`'s per-fold objective: ratio(tree_fit_score(...)) on one inner window,
    the inner window's design built with the inner window's own fallback."""

    def __init__(self, data, inner, met):
        from src.experiments import v2_horizon as vh
        self.vh, self.data, self.inner = vh, data, inner
        self.design = weather_design(data, inner, met)
        self.clim = vh.climatology(data, inner)

    def value(self, params):
        return float(self.vh.ratio(self.vh.tree_fit_score(self.data, self.design, self.inner, params, self.clim)[0]))

    def __call__(self, trial):
        from src.experiments.v2_forecast import tree_space
        return self.value(tree_space(trial))


class GraphObjective:
    """`v2_perfold_neural.run_graph`'s per-fold objective: ratio(neural_forecast(...)) at seed 42."""

    def __init__(self, data, inner, arm, quick=False):
        import torch
        from src.experiments import v2_horizon as vh
        from src.experiments.v2_forecast import NEURAL_ARMS, THREADS, graphs
        torch.set_num_threads(THREADS)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        graph, self.exchange, self.relaxation = NEURAL_ARMS[arm]
        data['adjacency'] = graphs(data)[graph]
        self.vh, self.data, self.inner, self.quick = vh, data, inner, quick
        self.clim = vh.climatology(data, inner)

    def value(self, params):
        d = self.vh.neural_forecast(self.data, self.inner, params, self.exchange, self.relaxation, 42, self.device,
                                    self.clim, self.quick)
        return float(self.vh.ratio(d))

    def __call__(self, trial):
        from src.experiments.v2_forecast import neural_space
        return self.value(neural_space(trial))


# --------------------------------------------------------------------------------------------------
# Gate A-G6: the objective code reproduces the published per-fold study's best value
# --------------------------------------------------------------------------------------------------
def objective_gate(variable, arm):
    from src.experiments import v2_horizon as vh
    from src.experiments import v2_perfold_select as pfs
    from src.data.bay_forcing import weather
    data = vh.build(variable, HORIZON)
    outer = {f['number']: f for f in vh.folds(data, HORIZON)}
    inner = vh.inner_fold(data, outer[1], HORIZON)
    rec = pfs.forecast(variable, arm, HORIZON, 1)
    if arm == 'weather_tree':
        objective = TreeObjective(data, inner, weather(data['times']).to_numpy())
        value = objective.value({**rec['params'], 'subsample_freq': 1})
        tolerance = 1e-9
    else:
        objective = GraphObjective(data, inner, arm)
        value = objective.value(rec['params'])
        tolerance = 1e-6
    diff = abs(value - float(rec['best_value']))
    return {'published_record': rec['source'], 'published_sha256': rec['sha256'], 'published_study': rec['study'],
            'published_best_value': rec['best_value'], 'objective_here': value, 'abs_diff': diff,
            'tolerance': tolerance, 'passed': bool(diff <= tolerance)}


# --------------------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------------------
def folder_for(variable, arm, quick):
    return OUT / ('smoke' if quick else 'full') / study_name(variable, arm)


def run(variable, arm, quick=False, gate_only=False):
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    started = time.time()
    folder = folder_for(variable, arm, quick)
    report = {'declared_in': DECLARED, 'variable': variable, 'arm': arm, 'horizon_hours': HORIZON,
              'study': study_name(variable, arm), 'protocol': PROTOCOL['version'], 'smoke_only': bool(quick),
              'environment': environment()}
    gate = objective_gate(variable, arm)
    report['gate_A_G6_objective'] = gate
    print(f'{variable} {arm} A-G6 objective: {gate["objective_here"]:.12f} vs {gate["published_best_value"]:.12f} '
          f'diff {gate["abs_diff"]:.2e} passed {gate["passed"]}', flush=True)
    if gate_only:
        _json(folder.parent / f'{study_name(variable, arm)}_gate.json', report)
        return report
    if not gate['passed'] and not quick:
        report['stopped'] = 'gate A-G6 failed: the study is not run'
        _json(folder / 'report.json', report)
        raise SystemExit(report['stopped'])
    data, outer, inner = reserved_setup(variable)
    audit = inner_audit(data, inner)
    report['inner_window'] = audit
    report['outer_fold'] = {'number': int(outer['number']), 'training_issuances': int(len(outer['tr'])),
                            'last_training_target': str(data['times'][outer['tr']].max() + pd.Timedelta(hours=HORIZON)),
                            'eligible_sensors': [n for n, e in zip(data['names'], outer['eligible']) if e]}
    if not audit['no_2026_row']:
        report['stopped'] = 'gate A-G6 failed: the inner window reaches past its declared end'
        _json(folder / 'report.json', report)
        raise SystemExit(report['stopped'])
    if arm == 'weather_tree':
        objective = TreeObjective(data, inner, reserved_weather(data['times']))
    else:
        objective = GraphObjective(data, inner, arm, quick=quick)
    sel = run_study(study_name(variable, arm), objective, folder, quick=quick)
    report['selection'] = sel
    report['objective'] = "fold-macro six-hour forecast MAE / persistence MAE on fold_2026's inner window only"
    report['seconds'] = round(time.time() - started, 1)
    record = folder / f'{study_name(variable, arm)}_selection.json'
    report['selection_record_sha256'] = _sha(record)
    _json(folder / 'report.json', report)
    print(f'{study_name(variable, arm)}: best {sel["best_value"]:.6f} over {sel["completed_trials"]} trials '
          f'({sel["failed_trials"]} failed) in {report["seconds"]} s', flush=True)
    return report


def selection(variable, arm):
    """The finished 2026 study of one (variable, arm), checked as v2_perfold_select checks a record."""
    folder = folder_for(variable, arm, False)
    path = folder / f'{study_name(variable, arm)}_selection.json'
    if not path.exists():
        raise FileNotFoundError(f'no 2026 selection {path.relative_to(ROOT).as_posix()}')
    sel = json.loads(path.read_text(encoding='utf-8'))
    if sel.get('study') != study_name(variable, arm):
        raise ValueError(f'{path.name}: study {sel.get("study")!r}')
    if sel.get('smoke_only') or int(sel.get('completed_trials', 0)) < MIN_TRIALS:
        raise ValueError(f'{path.name}: not a protocol-v2 selection')
    rep = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    if not (rep['gate_A_G6_objective']['passed'] and rep['inner_window']['no_2026_row']):
        raise ValueError(f'{path.name}: gate A-G6 did not pass')
    return {'params': dict(sel['params']), 'source': path.relative_to(ROOT).as_posix(), 'sha256': _sha(path),
            'study': sel['study'], 'completed_trials': int(sel['completed_trials']),
            'failed_trials': int(sel.get('failed_trials', 0)), 'best_value': sel.get('best_value'),
            'environment': rep.get('environment')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--gate-only', action='store_true')
    a = p.parse_args()
    run(a.variable, a.arm, quick=a.quick, gate_only=a.gate_only)
