"""The per-fold selections of addenda 47-50, looked up for modules that otherwise refit one shared
configuration (external review R3, 2026-10-05).

The shared selection chooses one configuration per arm on the inner data of both outer development
folds together, and fold 1's inner window lies inside fold 0's scored quarter (addendum 47). The
per-fold studies already on disk chose one configuration per (arm, outer fold) on that fold's own
inner window (forecasting) or its own checkpoints (Field-57). The modules that read a shared
selection (the addendum-4 rollout, the addendum-5 gates, the addendum-23 sequence panel and its
addendum-51 sweep, the addendum-2 naive comparators) take theirs from here when run per fold.

Every record is checked before use: its study name is the one the folder implies, it is not a smoke
run and it completed at least the protocol's hundred trials. Nothing here writes a file.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
MIN_TRIALS = 100

#: addendum 47: plain and weather trees at 6 and 24 h (local, Optuna 5.0.0)
TREES = RUNS / 'v2_perfold/forecast/perfold'
#: addendum 49: the graph arms at every horizon (cluster, GPU 1)
GRAPHS = RUNS / 'v2_perfold_neural/graph/perfold'
#: addendum 50: the plain LightGBM tree at 6 h on the shifted grids and the other sampler seeds
SWEEP_TREES = RUNS / 'v2_xgb_sweep'
#: addenda 47 and 48: the Field-57 ridge arms
FIELD = (RUNS / 'v2_perfold/field/perfold', RUNS / 'v2_perfold_rest/field/perfold')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(path, study):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'no per-fold selection {path.relative_to(ROOT).as_posix()}')
    sel = json.loads(path.read_text(encoding='utf-8'))
    if sel.get('study') != study:
        raise ValueError(f'{path.name}: study {sel.get("study")!r}, expected {study!r}')
    if sel.get('smoke_only') or int(sel.get('completed_trials', 0)) < MIN_TRIALS:
        raise ValueError(f'{path.name}: not a protocol-v2 selection '
                         f'({sel.get("completed_trials")} completed trials, smoke {sel.get("smoke_only")})')
    return {'params': dict(sel['params']), 'source': path.relative_to(ROOT).as_posix(), 'sha256': _sha(path),
            'study': sel['study'], 'completed_trials': int(sel['completed_trials']),
            'best_value': sel.get('best_value')}


def forecast(variable, arm, horizon, fold):
    """The per-fold selection of a tree arm (addendum 47) or a graph arm (addendum 49) at one fold.

    The parameters are returned as the study stored them: a tree refit adds `subsample_freq` exactly
    where its shared counterpart does."""
    tag = f'{variable}_{arm}_h{horizon}'
    root = TREES if arm in ('tree', 'weather_tree') else GRAPHS
    return _record(root / tag / f'{tag}_perfold_fold{int(fold)}_selection.json', f'{tag}_perfold_fold{int(fold)}')


def sweep_tree(variable, offset, seed, fold):
    """The per-fold plain tree at six hours on the grid advanced by `offset` hours under sampler `seed`.

    The reference cell (offset 0, seed 42) is addendum 47's study; every other cell is addendum 50's
    LightGBM arm, whose studies use the same data (`build_shifted`, identical to `build` at offset 0,
    its gate G1), design, search space and objective (fold-macro MAE over persistence MAE on that
    fold's own inner window), under that sampler seed."""
    if (int(offset), int(seed)) == (0, 42):
        return forecast(variable, 'tree', 6, fold)
    tag = f'{variable}_tree_h6_shift{int(offset)}_seed{int(seed)}'
    return _record(SWEEP_TREES / tag / f'{tag}_perfold_fold{int(fold)}_selection.json', f'{tag}_perfold_fold{int(fold)}')


def field(arm, fold):
    """The per-fold ridge penalty of a Field-57 arm (addenda 47 and 48) for one calendar fold."""
    study = f'held_site_{arm}_perfold_fold{int(fold)}'
    for root in FIELD:
        path = root / arm / f'{study}_selection.json'
        if path.exists():
            return _record(path, study)
    raise FileNotFoundError(f'no per-fold selection for {arm} fold {fold}')


def predictions(kind, name):
    """The per-fold selected predictions of a forecasting tree ('forecast', '<variable>_<arm>_h<h>')
    or a Field-57 arm ('field', '<arm>'), with their digest."""
    if kind == 'forecast':
        path = TREES / name / 'selected_predictions.csv.gz'
    elif kind == 'field':
        path = next((r / name / 'selected_predictions.csv.gz' for r in FIELD
                     if (r / name / 'selected_predictions.csv.gz').exists()), FIELD[0] / name / 'selected_predictions.csv.gz')
    else:
        raise ValueError(kind)
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT).as_posix())
    return path, _sha(path)
