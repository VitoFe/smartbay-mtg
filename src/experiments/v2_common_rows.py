"""Addendum 85, part B: the plain and weather point trees on Reserved-2026 at every lead.

On Dev-2025 the horizon study's trees exist at 1-48 h selected per outer fold (addenda 47, 48) and at
72 h (addendum 83). On Reserved-2026 a point tree existed only at 6 h (the blended weather tree) and
24 h (the frozen shared configuration). This module fills that cell the way the 2026 silent-probe
arms were selected (addendum 82 part A): Reserved-2026 is its own outer fold, and each (variable, arm,
lead) gets one study on that fold's own inner window, `v2_horizon.inner_fold`: validation issuances
from 2025-11-14 with targets before 2025-12-29, inner training targets before 2025-11-11. The data are
the 2026-inclusive grid (`v2_horizon_reserved.build`) and the outer fold is the distribution study's
2026 split (`v2_horizon_reserved.split`: training targets before 2026 with a 72 h gap, 2026 issuances
whose target is on or before the reserved end), so the trees predict exactly the rows the quantile
trees and the six-hour battery scored. The objective is the horizon study's: fold-macro forecast MAE
over persistence MAE on the inner window (`v2_horizon.ratio(tree_fit_score(...))`), search space
`v2_forecast.tree_space`, at least 100 completed trials (`protocol_v2.run_study`). No 2026 row enters
a study (gate A85-G0, audited per study).

Gates, run before any study (`gates`), each host-independent:
  A85-G1  the objective here, at each stored per-fold Dev-2025 selection (fold 1, both arms, both
          variables, 1-48 h), reproduces that study's best value to 1e-9
  A85-G2  refitting the per-fold Dev-2025 trees at their stored configurations reproduces the stored
          predictions to 1e-9 (6 and 24 h, both arms, both variables, both folds)
  A85-G3  this module's 2026 path refitting the frozen shared 24 h configurations reproduces
          runs/v2_horizon_reserved_h24 (tree, weather_tree) to 1e-9
  A85-G4  at 6 h the weather tree's 2026 inner window equals addendum 82's; if so the objective here at
          addendum 82's selection reproduces its best value to 1e-9 and that selection is reused

Declared in config/bay_reserved_protocol_v2_addendum85.json (part_B) before anything here was run.

Usage (repository root; V2_THREADS=2, OMP_NUM_THREADS unset):
  python -m src.experiments.v2_common_rows gates
  python -m src.experiments.v2_common_rows study --variable temperature --arm tree --lead 12 [--quick]
  python -m src.experiments.v2_common_rows launch --workers 4
  python -m src.experiments.v2_common_rows plan
  python -m src.experiments.v2_common_rows          (gates unless passed, then every pending study)
`predictions(variable, arm, lead)` returns a finished study's 2026 rows after checking its record.
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import gzip
import hashlib
import io
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_addendum85'
STUDIES = OUT / 'B' / 'studies'
PRED = OUT / 'B' / 'trees_2026'
GATES = OUT / 'B' / 'gates_B.json'
LOGS = OUT / 'B' / 'logs'
DECLARED = 'config/bay_reserved_protocol_v2_addendum85.json part_B'
LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
DEV_LEADS = (1, 3, 6, 12, 24, 36, 48)
ARMS = ('tree', 'weather_tree')
VARIABLES = ('temperature', 'oxygen')
MIN_TRIALS = 100
RESERVED_START = pd.Timestamp('2026-01-01', tz='UTC')
INNER_VALIDATION_END = pd.Timestamp('2025-12-29', tz='UTC')
LAST_TRAINING_TARGET_BEFORE = pd.Timestamp('2025-11-11', tz='UTC')
A82_STUDIES = RUNS / 'v2_addendum82/A/studies/full'
PERFOLD_DIRS = {6: RUNS / 'v2_perfold/forecast/perfold', 24: RUNS / 'v2_perfold/forecast/perfold'}
REST = RUNS / 'v2_perfold_rest/forecast/perfold'
RESERVED_H24 = RUNS / 'v2_horizon_reserved_h24'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, default=str).encode('utf-8'))


def _csv_gz(path, frame):
    """A gzip CSV with a fixed header time and LF line ends, so a rebuild is byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as z:
        z.write(text)
    path.write_bytes(buf.getvalue())


def environment():
    import lightgbm
    import optuna
    return {'host': socket.gethostname(), 'platform': platform.platform(), 'python': sys.version.split()[0],
            'optuna': optuna.__version__, 'lightgbm': lightgbm.__version__, 'numpy': np.__version__,
            'pandas': pd.__version__, 'V2_THREADS': os.environ.get('V2_THREADS'),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')}


def study_name(variable, arm, lead):
    return f'a85B_{variable}_{arm}_h{lead}_fold2026'


# --------------------------------------------------------------------------------------------------
# Data, folds and designs
# --------------------------------------------------------------------------------------------------
def reserved_setup(variable, lead):
    """The 2026-inclusive grid at one lead, the 2026 outer fold (the distribution study's split) and
    its inner window."""
    from src.experiments import v2_horizon as vh
    from src.experiments import v2_horizon_reserved as hr
    data = hr.build(variable, lead)
    bundle = hr.split(data, lead)
    outer = dict(number=2, tr=bundle['tr'], va=bundle['va'], eligible=bundle['eligible'],
                 fallback=bundle['fallback'])
    inner = vh.inner_fold(data, outer, lead)
    return data, outer, inner


def reserved_weather(times):
    from src.experiments import bay_reserved_evaluation as v1
    return v1.load_weather_2026(times).to_numpy()


def dev_weather(times):
    from src.data.bay_forcing import weather
    return weather(times).to_numpy()


def design(data, fold, arm, met):
    """The horizon study's design (`v2_horizon.run`): the forecast features with the fold's own
    fallback, and for the weather tree the weather at issuance broadcast over the sensors."""
    from src.experiments.bay_dynamics_tree import features
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    if arm == 'tree':
        return base
    return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']),
                                                                met.shape[1]))], axis=-1)


def inner_audit(data, inner, lead):
    """Gate A85-G0: the inner window and its training rows end before 2026 by the declared margins."""
    t = data['times']
    last_target = t[inner['tr']].max() + pd.Timedelta(hours=lead)
    last_val = t[inner['va']].max()
    last_val_target = last_val + pd.Timedelta(hours=lead)
    ok = bool(last_target < LAST_TRAINING_TARGET_BEFORE and last_val_target < INNER_VALIDATION_END
              and last_val_target < RESERVED_START)
    return {'first_validation_issuance': str(t[inner['va']].min()), 'last_validation_issuance': str(last_val),
            'last_validation_target': str(last_val_target), 'validation_end_exclusive': str(INNER_VALIDATION_END),
            'last_training_target': str(last_target), 'training_targets_before': str(LAST_TRAINING_TARGET_BEFORE),
            'validation_issuances': int(len(inner['va'])), 'training_issuances': int(len(inner['tr'])),
            'eligible_sensors': [n for n, e in zip(data['names'], inner['eligible']) if e],
            'no_2026_row': ok}


class Objective:
    """The horizon study's per-fold objective on one inner window."""

    def __init__(self, data, inner, arm, met):
        from src.experiments import v2_horizon as vh
        self.vh, self.data, self.inner = vh, data, inner
        self.x = design(data, inner, arm, met)
        self.clim = vh.climatology(data, inner)

    def value(self, params):
        return float(self.vh.ratio(self.vh.tree_fit_score(self.data, self.x, self.inner, params, self.clim)[0]))

    def __call__(self, trial):
        from src.experiments.v2_forecast import tree_space
        return self.value(tree_space(trial))


# --------------------------------------------------------------------------------------------------
# Stored per-fold Dev-2025 selections
# --------------------------------------------------------------------------------------------------
def dev_record(variable, arm, lead, fold):
    if lead in PERFOLD_DIRS:
        tag = f'{variable}_{arm}_h{lead}'
        folder = PERFOLD_DIRS[lead] / tag
    else:
        tag = f'horizon_{variable}_{arm}_h{lead}'
        folder = REST / tag
    path = folder / f'{tag}_perfold_fold{fold}_selection.json'
    sel = json.loads(path.read_text(encoding='utf-8'))
    if sel.get('smoke_only') or int(sel.get('completed_trials', 0)) < MIN_TRIALS:
        raise ValueError(f'{path.name}: not a protocol-v2 selection')
    return {'params': dict(sel['params']), 'best_value': float(sel['best_value']), 'study': sel['study'],
            'source': path.relative_to(ROOT).as_posix(), 'sha256': _sha(path),
            'predictions': folder / 'selected_predictions.csv.gz'}


def gate_objective():
    """A85-G1."""
    from src.experiments import v2_horizon as vh
    out, ok = {}, True
    for variable in VARIABLES:
        for lead in DEV_LEADS:
            data = vh.build(variable, lead)
            outer = {f['number']: f for f in vh.folds(data, lead)}
            inner = vh.inner_fold(data, outer[1], lead)
            met = dev_weather(data['times'])
            for arm in ARMS:
                rec = dev_record(variable, arm, lead, 1)
                value = Objective(data, inner, arm, met if arm == 'weather_tree' else None).value(
                    {**rec['params'], 'subsample_freq': 1})
                diff = abs(value - rec['best_value'])
                passed = bool(diff <= 1e-9)
                ok &= passed
                out[f'{variable}|{arm}|{lead}'] = {'record': rec['source'], 'record_sha256': rec['sha256'],
                                                   'published_best_value': rec['best_value'],
                                                   'objective_here': value, 'abs_diff': diff, 'passed': passed}
                print(f'G1 {variable} {arm} h{lead}: {value:.12f} vs {rec["best_value"]:.12f} diff {diff:.1e}', flush=True)
    return {'tolerance': 1e-9, 'cells': out, 'passed': bool(ok)}


def gate_predictions():
    """A85-G2."""
    from src.experiments import v2_horizon as vh
    out, ok = {}, True
    for variable in VARIABLES:
        for lead in (6, 24):
            data = vh.build(variable, lead)
            folds = vh.folds(data, lead)
            met = dev_weather(data['times'])
            for arm in ARMS:
                stored = None
                for f in folds:
                    rec = dev_record(variable, arm, lead, f['number'])
                    if stored is None:
                        stored = pd.read_csv(rec['predictions'])
                    clim = vh.climatology(data, f)
                    x = design(data, f, arm, met if arm == 'weather_tree' else None)
                    d, _ = vh.tree_fit_score(data, x, f, {**rec['params'], 'subsample_freq': 1}, clim)
                    s = stored[stored.fold == f['number']].reset_index(drop=True)
                    same_keys = bool(len(d) == len(s) and (d.sensor.to_numpy() == s.sensor.to_numpy()).all()
                                     and (pd.to_datetime(d.time, utc=True).to_numpy()
                                          == pd.to_datetime(s.time, utc=True).to_numpy()).all())
                    diff = float(np.abs(d.prediction.to_numpy() - s.prediction.to_numpy()).max()) if same_keys else None
                    passed = bool(same_keys and diff <= 1e-9)
                    ok &= passed
                    out[f'{variable}|{arm}|{lead}|fold{f["number"]}'] = {
                        'rows': int(len(d)), 'stored_rows': int(len(s)), 'same_keys': same_keys,
                        'max_abs_prediction_difference': diff, 'passed': passed}
                    print(f'G2 {variable} {arm} h{lead} fold{f["number"]}: rows {len(d)} diff {diff}', flush=True)
    return {'tolerance': 1e-9, 'cells': out, 'passed': bool(ok)}


def gate_reserved24():
    """A85-G3."""
    from src.experiments import v2_horizon as vh
    out, ok = {}, True
    for variable in VARIABLES:
        data, outer, _ = reserved_setup(variable, 24)
        met = reserved_weather(data['times'])
        clim = vh.climatology(data, outer)
        published = pd.read_csv(RESERVED_H24 / f'reserved_h24_{variable}.csv.gz')
        # the configurations the 24 h reading refitted, as its summary records them (the horizon
        # study's own report.json files were not kept)
        provenance = json.loads((RESERVED_H24 / f'reserved_h24_{variable}_summary.json')
                                .read_text(encoding='utf-8'))['selection_provenance']
        for arm in ARMS:
            params, trials = provenance[arm]['params'], provenance[arm]['trials']
            record = (RUNS / '20260920T083322Z_v2_horizon/results' / f'v2_horizon_{variable}_{arm}_h24'
                      / f'{variable}_{arm}_h24_selection.json')
            same_params = (json.loads(record.read_text(encoding='utf-8'))['params'] == params
                           if record.exists() else None)
            x = design(data, outer, arm, met if arm == 'weather_tree' else None)
            d, _ = vh.tree_fit_score(data, x, outer, {**params, 'subsample_freq': 1}, clim)
            p = published[published.arm == arm][['sensor', 'time', 'truth', 'prediction']]
            d = d.assign(time=pd.to_datetime(d.time, utc=True))
            p = p.assign(time=pd.to_datetime(p.time, utc=True))
            j = d.merge(p, on=['sensor', 'time'], how='outer', suffixes=('', '_pub'), indicator=True)
            both = j[j._merge == 'both']
            diff = float((both.prediction - both.prediction_pub).abs().max()) if len(both) else None
            tdiff = float((both.truth - both.truth_pub).abs().max()) if len(both) else None
            passed = bool((j._merge != 'both').sum() == 0 and diff is not None and diff <= 1e-9 and tdiff <= 1e-9
                          and same_params is not False)
            ok &= passed
            out[f'{variable}|{arm}'] = {'selection_record': record.relative_to(ROOT).as_posix() if record.exists() else None,
                                        'record_params_equal_provenance': same_params,
                                        'rows_here': int(len(d)), 'rows_published': int(len(p)),
                                        'unmatched': int((j._merge != 'both').sum()),
                                        'max_abs_prediction_difference': diff, 'max_abs_truth_difference': tdiff,
                                        'trials': trials, 'passed': passed}
            print(f'G3 {variable} {arm}: rows {len(d)} vs {len(p)} diff {diff}', flush=True)
    return {'tolerance': 1e-9, 'cells': out, 'passed': bool(ok)}


def a82_record(variable):
    name = f'a82A_{variable}_weather_tree_h6_off0_seed42_fold2026'
    path = A82_STUDIES / name / f'{name}_selection.json'
    sel = json.loads(path.read_text(encoding='utf-8'))
    return sel, path


def gate_a82():
    """A85-G4: the 6 h weather tree's inner window against addendum 82's; reuse if identical."""
    from src.experiments import v2_horizon as vh
    from src.experiments.v2_bridge import fold_2026
    out = {}
    for variable in VARIABLES:
        data, outer, inner = reserved_setup(variable, 6)
        a82_outer = fold_2026(data)
        a82_inner = vh.inner_fold(data, a82_outer, 6)
        same = bool(np.array_equal(inner['tr'], a82_inner['tr']) and np.array_equal(inner['va'], a82_inner['va'])
                    and np.array_equal(inner['eligible'], a82_inner['eligible'])
                    and np.allclose(inner['fallback'], a82_inner['fallback'], atol=0, rtol=0))
        sel, path = a82_record(variable)
        e = {'inner_window_identical': same, 'a82_record': path.relative_to(ROOT).as_posix(), 'a82_sha256': _sha(path),
             'a82_best_value': float(sel['best_value']), 'a82_completed_trials': int(sel['completed_trials']),
             'eligible_here': [n for n, x in zip(data['names'], inner['eligible']) if x],
             'eligible_a82': [n for n, x in zip(data['names'], a82_inner['eligible']) if x]}
        if same:
            value = Objective(data, inner, 'weather_tree', reserved_weather(data['times'])).value(
                {**sel['params'], 'subsample_freq': 1})
            e.update(objective_here=value, abs_diff=abs(value - float(sel['best_value'])))
            e['reuse'] = bool(e['abs_diff'] <= 1e-9 and int(sel['completed_trials']) >= MIN_TRIALS)
        else:
            e['reuse'] = False
        print(f'G4 {variable}: identical {same} reuse {e["reuse"]} {e.get("abs_diff")}', flush=True)
        out[variable] = e
    return out


def gates():
    started = time.time()
    rep = {'declared_in': DECLARED, 'environment': environment()}
    rep['A85-G1_objective_reproduction'] = gate_objective()
    rep['A85-G2_prediction_reproduction'] = gate_predictions()
    rep['A85-G3_2026_path'] = gate_reserved24()
    rep['A85-G4_addendum82_study'] = gate_a82()
    rep['passed'] = bool(rep['A85-G1_objective_reproduction']['passed'] and rep['A85-G2_prediction_reproduction']['passed']
                         and rep['A85-G3_2026_path']['passed'])
    rep['seconds'] = round(time.time() - started, 1)
    _json(GATES, rep)
    print('gates passed' if rep['passed'] else 'GATES FAILED', rep['seconds'], 's', flush=True)
    return rep


# --------------------------------------------------------------------------------------------------
# One study and its 2026 predictions
# --------------------------------------------------------------------------------------------------
def folder_for(variable, arm, lead, quick):
    return STUDIES / ('smoke' if quick else 'full') / study_name(variable, arm, lead)


def prediction_path(variable, arm, lead, quick=False):
    return (PRED.parent / 'trees_2026_smoke' if quick else PRED) / f'{variable}_{arm}_h{lead}.csv.gz'


def gates_passed():
    if not GATES.exists():
        return False
    g = json.loads(GATES.read_text(encoding='utf-8'))
    return bool(g.get('passed'))


def run(variable, arm, lead, quick=False):
    from src.experiments import v2_horizon as vh
    from src.tuning.protocol_v2 import run_study, PROTOCOL
    if not quick and not gates_passed():
        raise SystemExit('gates A85-G1 to G3 have not passed: no study is run')
    started = time.time()
    folder = folder_for(variable, arm, lead, quick)
    report = {'declared_in': DECLARED, 'variable': variable, 'arm': arm, 'lead_hours': lead,
              'study': study_name(variable, arm, lead), 'protocol': PROTOCOL['version'], 'smoke_only': bool(quick),
              'environment': environment()}
    data, outer, inner = reserved_setup(variable, lead)
    if quick:
        inner['tr'], inner['va'] = inner['tr'][-300:], inner['va'][:40]
    audit = inner_audit(data, inner, lead)
    report['inner_window'] = audit
    report['outer_fold'] = {'training_issuances': int(len(outer['tr'])), 'reserved_issuances': int(len(outer['va'])),
                            'last_training_target': str(data['times'][outer['tr']].max() + pd.Timedelta(hours=lead)),
                            'eligible_sensors': [n for n, e in zip(data['names'], outer['eligible']) if e]}
    if not audit['no_2026_row']:
        report['stopped'] = 'gate A85-G0 failed: the inner window reaches past its declared end'
        _json(folder / 'report.json', report)
        raise SystemExit(report['stopped'])
    met = reserved_weather(data['times']) if arm == 'weather_tree' else None
    reuse = None
    if (arm, lead) == ('weather_tree', 6) and not quick and GATES.exists():
        g4 = json.loads(GATES.read_text(encoding='utf-8')).get('A85-G4_addendum82_study', {}).get(variable, {})
        if g4.get('reuse'):
            sel_a82, path = a82_record(variable)
            reuse = {'source': path.relative_to(ROOT).as_posix(), 'sha256': _sha(path)}
            sel = {'study': sel_a82['study'], 'protocol': sel_a82.get('protocol'),
                   'completed_trials': int(sel_a82['completed_trials']),
                   'failed_trials': int(sel_a82.get('failed_trials', 0)), 'best_trial': sel_a82.get('best_trial'),
                   'best_value': float(sel_a82['best_value']), 'params': dict(sel_a82['params']),
                   'smoke_only': False, 'reused_from': reuse}
    if reuse is None:
        objective = Objective(data, inner, arm, met)
        sel = run_study(study_name(variable, arm, lead), objective, folder, quick=quick)
    report['selection'] = sel
    report['reused_addendum82_study'] = reuse
    report['objective'] = "fold-macro forecast MAE / persistence MAE on the 2026 fold's inner window only"
    params = {**sel['params'], 'subsample_freq': 1}
    if quick:
        outer = {**outer, 'tr': outer['tr'][-300:]}
    x = design(data, outer, arm, met)
    clim = vh.climatology(data, outer)
    d, _ = vh.tree_fit_score(data, x, outer, params, clim)
    d = d.drop(columns=['mode']).assign(time=pd.to_datetime(d.time, utc=True), lead=lead, arm=arm)
    path = prediction_path(variable, arm, lead, quick)
    _csv_gz(path, d)
    report['predictions'] = {'file': path.relative_to(ROOT).as_posix(), 'sha256': _sha(path), 'rows': int(len(d)),
                             'sensors': int(d.sensor.nunique())}
    report['seconds'] = round(time.time() - started, 1)
    if reuse is None:
        record = folder / f'{study_name(variable, arm, lead)}_selection.json'
        report['selection_record_sha256'] = _sha(record)
    _json(folder / 'report.json', report)
    print(f'{study_name(variable, arm, lead)}: best {sel["best_value"]:.6f} over {sel["completed_trials"]} trials '
          f'({sel.get("failed_trials", 0)} failed), {len(d)} rows, {report["seconds"]} s', flush=True)
    return report


def predictions(variable, arm, lead):
    """The 2026 rows of one finished study, after checking its record and gate."""
    folder = folder_for(variable, arm, lead, False)
    rep = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    sel = rep['selection']
    if sel.get('smoke_only') or int(sel.get('completed_trials', 0)) < MIN_TRIALS:
        raise ValueError(f'{study_name(variable, arm, lead)}: not a protocol-v2 selection')
    if not rep['inner_window']['no_2026_row']:
        raise ValueError(f'{study_name(variable, arm, lead)}: gate A85-G0 did not pass')
    path = ROOT / rep['predictions']['file']
    if _sha(path) != rep['predictions']['sha256']:
        raise ValueError(f'{path.name}: digest differs from its study record')
    d = pd.read_csv(path)
    d['time'] = pd.to_datetime(d.time, utc=True)
    return d, {'study': sel['study'], 'completed_trials': int(sel['completed_trials']),
               'best_value': sel['best_value'], 'reused_from': rep.get('reused_addendum82_study'),
               'report': (folder / 'report.json').relative_to(ROOT).as_posix(), 'predictions': rep['predictions']}


def plan():
    out = []
    for variable in VARIABLES:
        for arm in ARMS:
            for lead in LEADS:
                done = (folder_for(variable, arm, lead, False) / 'report.json').exists()
                out.append((variable, arm, lead, done))
    return out


def launch(workers):
    """Every pending study, `workers` at a time, each in its own process at two threads."""
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = [(v, a, h) for v, a, h, done in plan() if not done]
    # the longest leads and the weather designs first
    pending.sort(key=lambda t: (-t[2], t[1] != 'weather_tree'))
    env = {k: v for k, v in os.environ.items() if k != 'OMP_NUM_THREADS'}
    env.update(V2_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    running = []
    while pending or running:
        while pending and len(running) < workers:
            v, a, h = pending.pop(0)
            log = open(LOGS / f'{study_name(v, a, h)}.log', 'wb')
            p = subprocess.Popen([sys.executable, '-m', 'src.experiments.v2_common_rows', 'study', '--variable', v,
                                  '--arm', a, '--lead', str(h)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, log, (v, a, h)))
            print('started', study_name(v, a, h), flush=True)
        time.sleep(5)
        for item in list(running):
            p, log, key = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print('finished', study_name(*key), 'exit', p.returncode, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # with no action (as the pipeline runs a producer): the gates unless they passed, then every pending study
    sub = parser.add_subparsers(dest='action')
    sub.add_parser('gates')
    s = sub.add_parser('study')
    s.add_argument('--variable', choices=VARIABLES, required=True)
    s.add_argument('--arm', choices=ARMS, required=True)
    s.add_argument('--lead', type=int, choices=LEADS, required=True)
    s.add_argument('--quick', action='store_true')
    lp = sub.add_parser('launch')
    lp.add_argument('--workers', type=int, default=4)
    sub.add_parser('plan')
    a = parser.parse_args()
    if a.action == 'gates':
        gates()
    elif a.action == 'study':
        run(a.variable, a.arm, a.lead, quick=a.quick)
    elif a.action == 'launch':
        launch(a.workers)
    elif a.action == 'plan':
        for row in plan():
            print(*row)
    else:
        if not gates_passed():
            gates()
        if gates_passed():
            launch(4)
