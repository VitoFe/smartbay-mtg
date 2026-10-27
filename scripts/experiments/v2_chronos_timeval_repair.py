"""Addendum 42 repair after the disk filled: mark orphaned RUNNING trials FAIL, quarantine unreadable columns.

Run from the v2_timeval snapshot directory with no worker alive. Nothing is deleted: an unreadable
column file is moved to results/columns_quarantine/<study>/, and every change is printed.
"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import optuna

here = Path.cwd()
alive = subprocess.run(['pgrep', '-f', 'v2_chronos_timeval.py'], capture_output=True, text=True).stdout.split()
if alive:
    sys.exit(f'workers alive: {alive}; not touching the studies')
report = {}
for sqlite in sorted((here / 'results/studies').glob('timeval_*.sqlite3')):
    name = sqlite.stem
    storage = optuna.storages.RDBStorage('sqlite:///' + str(sqlite))
    study = optuna.load_study(study_name=name, storage=storage)
    orphans = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    for t in orphans:
        storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    quarantined = []
    coldir = here / 'results/columns' / name
    for f in sorted(coldir.glob('*.npy')):
        try:
            a = np.load(f)
            ok = a.ndim == 1 and len(a) > 0
        except Exception:
            ok = False
        if not ok:
            dest = here / 'results/columns_quarantine' / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest / f.name)); quarantined.append(f.name)
    states = {}
    for t in optuna.load_study(study_name=name, storage=storage).trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    report[name] = {'orphans_marked_failed': [t.number for t in orphans], 'quarantined_columns': quarantined, 'states': states}
print(json.dumps(report, indent=2))
(here / 'logs/repair_after_disk_full.json').write_text(json.dumps(report, indent=2))
