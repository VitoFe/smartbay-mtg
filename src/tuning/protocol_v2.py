"""Tuning protocol v2: one seeded TPE study per arm, >=100 completed trials, persisted to SQLite."""
import json
from pathlib import Path
import optuna

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / 'config/tuning_protocol_v2.json').read_text(encoding='utf-8'))


def run_study(name, objective, folder, quick=False, n_trials=None):
    """Run (or resume) a study and refuse to return fewer completed trials than the protocol minimum."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # A full run must not resume a study a smoke run left behind: its trials were scored on reduced folds,
    # would count towards the protocol minimum and could be returned as the best. The selection record
    # says what kind of run wrote the database last.
    previous = folder / f'{name}_selection.json'
    if not quick and previous.exists() and json.loads(previous.read_text(encoding='utf-8')).get('smoke_only'):
        raise RuntimeError(f'{name}: {folder} holds a smoke study; run the full study in a fresh folder')
    study = optuna.create_study(
        direction='minimize', study_name=name, load_if_exists=True,
        storage='sqlite:///' + str(folder / f'{name}.sqlite3'),
        sampler=optuna.samplers.TPESampler(seed=PROTOCOL['sampler_seed'], n_startup_trials=PROTOCOL['startup_trials']))
    target = 2 if quick else (n_trials or PROTOCOL['n_trials'])
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    def log(s, t):
        # best_value raises until one trial has completed, which would stop the study at a first failure
        completed = any(x.state == optuna.trial.TrialState.COMPLETE for x in s.trials)
        best = f'{s.best_value:.5f}' if completed else 'none yet'
        print(f'  [{name} {t.number + 1:03d}/{target}] {t.value if t.value is not None else "FAIL"} best {best}', flush=True)

    if done < target:
        # MaxTrialsCallback lets several workers share one study and all stop at the target.
        stop = optuna.study.MaxTrialsCallback(target, states=(optuna.trial.TrialState.COMPLETE,))
        study.optimize(objective, n_trials=target - done, callbacks=[log, stop], catch=(ValueError, RuntimeError))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not quick and len(complete) < PROTOCOL['min_completed_trials']:
        raise RuntimeError(f'{name}: only {len(complete)} completed trials, protocol requires {PROTOCOL["min_completed_trials"]}')
    summary = {'study': name, 'protocol': PROTOCOL['version'], 'completed_trials': len(complete),
               'failed_trials': len(study.trials) - len(complete), 'best_trial': study.best_trial.number,
               'best_value': study.best_value, 'params': study.best_params, 'smoke_only': quick}
    (folder / f'{name}_selection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
