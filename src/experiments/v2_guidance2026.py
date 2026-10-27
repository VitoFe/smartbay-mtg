"""Addendum 86: physical guidance scored on the 2026 test period, the fits.

Declared in config/bay_reserved_protocol_v2_addendum86.json before anything here was written. This
module fits; scripts/v2_guidance2026_report.py scores. Reserved-2026 is its own outer fold, the 2026
split of v2_horizon_reserved.split, as part B of addendum 85 (v2_common_rows): every new study is
selected on that fold's own inner window, which ends with targets before 2025-12-29.

Parts (the declaration's):
  B  the exogenous feature arms (Magra level, sea level, the ISPRA water temperature, the seabed) on
     the 2026 fold, one 100-trial study per cell. Their designs are the development design functions'
     columns in the same order, built with loaders that reach 2026: bay_forcing.weather and the
     river and ISPRA loaders stop at 2026 by default, so v2_river/v2_ispra/v2_bathy.design_for run
     unchanged on 2026 would give the 2026 rows no feature (gate A86-G4 checks both sides).
  C  the six-hour graph operators refitted at addendum 82 part A's fold-2026 selections, three seeds;
     relaxation towards saturation needs its own study (timed first; see `saturation`).
  D  addendum 5's output gates on the rollout of the six-hour weather tree over the 2026 fold.
  E1 the utilisation ridge's development reading: the year-boundary split moved back one year, in mg/l.

Gates (the declaration's A86-G*): G1 part B's 2026 trees refitted bit for bit; G3 the objective at
the stored per-fold fold-1 selections; G4 the designs; G5 the graph refit path; G6 part D's step tree;
G7 part E1's mg/l context. G2 (part G's rows) is a join and lives in the report script.

Usage (repository root; V2_THREADS=2, OMP_NUM_THREADS unset):
  python -m src.experiments.v2_guidance2026 gates               G1, G3, G4
  python -m src.experiments.v2_guidance2026 study --variable oxygen --arm river_tree --lead 24 [--quick]
  python -m src.experiments.v2_guidance2026 launch --workers 2
  python -m src.experiments.v2_guidance2026 graph-gate          G5
  python -m src.experiments.v2_guidance2026 graphs              part C1
  python -m src.experiments.v2_guidance2026 saturation-time     part C2: the CPU estimate and its gate
  python -m src.experiments.v2_guidance2026 saturation-study    part C2: the study and refit, if the estimate allows
  python -m src.experiments.v2_guidance2026 rollouts            part D (G6 first)
  python -m src.experiments.v2_guidance2026 aou-dev --stage DIR part E1 (G7 first; inputs staged in DIR)
  python -m src.experiments.v2_guidance2026 field-p1            part E2 (the read reproduces addendum 38 first)
  python -m src.experiments.v2_guidance2026                     every step whose output is missing
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
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
OUT = RUNS / 'v2_addendum86'
GATES = OUT / 'gates'
DECLARED = 'config/bay_reserved_protocol_v2_addendum86.json'
MIN_TRIALS = 100
RESERVED_START = pd.Timestamp('2026-01-01', tz='UTC')
#: part B: (variable, arm, lead)
CELLS = (('oxygen', 'river_tree', 24), ('oxygen', 'river_tree', 36), ('oxygen', 'weather_river_tree', 6),
         ('temperature', 'sealevel_tree', 36), ('oxygen', 'sealevel_tree', 6), ('oxygen', 'sealevel_tree', 48),
         ('temperature', 'extboundary_tree', 6), ('oxygen', 'extboundary_tree', 6), ('oxygen', 'bathy_tree', 12))
FAMILY = {'river_tree': 'river', 'weather_river_tree': 'river', 'sealevel_tree': 'ispra',
          'extboundary_tree': 'ispra', 'bathy_tree': 'bathy'}
#: the arm each feature arm augments (each is also set against the plain tree)
BASE_OF = {'river_tree': 'tree', 'weather_river_tree': 'weather_tree', 'sealevel_tree': 'weather_tree',
           'extboundary_tree': 'weather_tree', 'bathy_tree': 'weather_tree'}
B_LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
GRAPH_ARMS = ('learned_source', 'conservative_exchange')
SEEDS = (42, 7, 123)
WEATHER_COLUMNS = 5


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, obj):
    """LF JSON on every host."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, default=_default, allow_nan=False).encode('utf-8'))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _lf(path):
    """Rewrite a text file another module wrote with the host's line ending (run_study's selection)."""
    path = Path(path)
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))


def _csv_gz(path, frame):
    from src.experiments.v2_common_rows import _csv_gz as write
    write(path, frame)


def _read_csv(path):
    """Floats read back exactly as written (pandas writes the shortest round-trip repr)."""
    return pd.read_csv(path, float_precision='round_trip')


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def environment():
    import lightgbm
    import optuna
    out = {'host': socket.gethostname(), 'platform': platform.platform(), 'python': sys.version.split()[0],
           'optuna': optuna.__version__, 'lightgbm': lightgbm.__version__, 'numpy': np.__version__,
           'pandas': pd.__version__, 'V2_THREADS': os.environ.get('V2_THREADS'),
           'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
           'V2_ISPRA_BINNING': os.environ.get('V2_ISPRA_BINNING')}
    try:
        import torch
        out.update(torch=torch.__version__, cuda_available=bool(torch.cuda.is_available()))
    except ImportError:
        pass
    return out


def nan_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
                and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


def nan_max_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return None
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


# --------------------------------------------------------------------------------------------------
# Part B: designs
# --------------------------------------------------------------------------------------------------
def broadcast(block, n):
    return np.broadcast_to(block[:, None], (block.shape[0], n, block.shape[1]))


def exogenous_2026(arm, data):
    """The columns the development design function appends after the forecast features, in its order,
    from loaders that reach 2026: load_weather_2026 (as part B's weather tree of addendum 85),
    river/sea_level/external_boundary with cutoff=None, the static seabed block."""
    from src.data.bay_ispra import BOUNDARY_COLUMNS, LEVEL_COLUMNS, external_boundary, sea_level
    from src.data.bay_river import COLUMNS as RIVER_COLUMNS, river
    from src.experiments import v2_common_rows as CR
    from src.experiments.v2_bathy import bathymetric_block
    times, n = data['times'], len(data['names'])
    blocks = []
    if arm != 'river_tree':
        blocks.append(broadcast(CR.reserved_weather(times), n))
    if arm in ('river_tree', 'weather_river_tree'):
        blocks.append(broadcast(river(times, cutoff=None)[RIVER_COLUMNS].to_numpy(), n))
    elif arm == 'sealevel_tree':
        blocks.append(broadcast(sea_level(times, cutoff=None)[LEVEL_COLUMNS].to_numpy(), n))
    elif arm == 'extboundary_tree':
        station = external_boundary(times, cutoff=None)[BOUNDARY_COLUMNS].to_numpy()
        blocks.append(broadcast(station, n))
        anchor = data['x'] * data['scale'] + data['center']
        blocks.append((station[:, 0:1] - anchor)[..., None])
    elif arm == 'bathy_tree':
        blocks.append(bathymetric_block(data))
    else:
        raise ValueError(arm)
    return blocks


class Designs:
    """The 2026 design of one arm on one grid, the exogenous columns built once."""

    def __init__(self, arm, data):
        self.arm, self.data = arm, data
        self.extra = exogenous_2026(arm, data)

    def __call__(self, fold):
        from src.experiments.bay_dynamics_tree import features
        base, _ = features(self.data, {'fallback': fold['fallback']}, 'forecast')
        return np.concatenate([base] + list(self.extra), axis=-1)


def dev_design(arm, data, fold):
    """The development design function itself (v2_perfold_rest.design_builder's choice)."""
    if arm in ('river_tree', 'weather_river_tree'):
        from src.experiments.v2_river import ARMS, design_for
        return design_for(data, fold, ARMS[arm])
    if arm in ('sealevel_tree', 'extboundary_tree'):
        from src.experiments.v2_ispra import design_for
        return design_for(data, fold, arm)
    from src.experiments.v2_bathy import design_for
    return design_for(data, fold)


def study_name(variable, arm, lead):
    return f'a86B_{variable}_{arm}_h{lead}_fold2026'


def dev_selection_path(variable, arm, lead):
    """The stored per-fold Dev-2025 fold-1 selection of the arm (addenda 48, 82 part B, 83)."""
    tag = f'{FAMILY[arm]}_{variable}_{arm}_h{lead}'
    if lead in (6, 24, 48):
        folder = RUNS / 'v2_perfold_rest/forecast/perfold' / tag
    else:
        folder = RUNS / 'v2_addendum83/perfold' / tag
    return folder / f'{tag}_perfold_fold1_selection.json'


class Objective:
    """The horizon study's per-fold objective on one inner window, any design."""

    def __init__(self, data, inner, x):
        from src.experiments import v2_horizon as vh
        self.vh, self.data, self.inner, self.x = vh, data, inner, x
        self.clim = vh.climatology(data, inner)

    def value(self, params):
        return float(self.vh.ratio(self.vh.tree_fit_score(self.data, self.x, self.inner, params, self.clim)[0]))

    def __call__(self, trial):
        from src.experiments.v2_forecast import tree_space
        return self.value(tree_space(trial))


# --------------------------------------------------------------------------------------------------
# Gates G1, G3, G4
# --------------------------------------------------------------------------------------------------
def b_tree_record(variable, arm, lead):
    from src.experiments import v2_common_rows as CR
    folder = CR.folder_for(variable, arm, lead, False)
    rep = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    return rep, folder


def gate_b_trees():
    """A86-G1: each of part B's 32 2026 trees refitted at its recorded selection on its 2026 outer fold
    reproduces the stored predictions bit for bit."""
    from src.experiments import v2_common_rows as CR
    from src.experiments import v2_horizon as vh
    out, ok = {}, True
    for variable in ('temperature', 'oxygen'):
        for lead in B_LEADS:
            data, outer, _ = CR.reserved_setup(variable, lead)
            met = CR.reserved_weather(data['times'])
            clim = vh.climatology(data, outer)
            for arm in ('tree', 'weather_tree'):
                rep, folder = b_tree_record(variable, arm, lead)
                sel = rep['selection']
                path = ROOT / rep['predictions']['file']
                digest_ok = _sha(path) == rep['predictions']['sha256']
                x = CR.design(data, outer, arm, met if arm == 'weather_tree' else None)
                d, _ = vh.tree_fit_score(data, x, outer, {**sel['params'], 'subsample_freq': 1}, clim)
                stored = _read_csv(path)
                same_keys = bool(len(d) == len(stored) and (d.sensor.to_numpy() == stored.sensor.to_numpy()).all()
                                 and (pd.to_datetime(d.time, utc=True).to_numpy()
                                      == pd.to_datetime(stored.time, utc=True).to_numpy()).all())
                diff = float(np.abs(d.prediction.to_numpy() - stored.prediction.to_numpy()).max()) if same_keys else None
                tdiff = float(np.abs(d.truth.to_numpy() - stored.truth.to_numpy()).max()) if same_keys else None
                exact = bool(same_keys and diff == 0.0 and tdiff == 0.0)
                flagged = bool(same_keys and not exact and diff <= 1e-9 and tdiff <= 1e-9)
                passed = bool(digest_ok and (exact or flagged))
                ok &= passed
                out[f'{variable}|{arm}|{lead}'] = {
                    'record': rel(folder / 'report.json'), 'predictions': rep['predictions']['file'],
                    'digest_matches_record': digest_ok, 'study': sel['study'],
                    'completed_trials': int(sel['completed_trials']), 'rows': int(len(d)),
                    'same_keys': same_keys, 'max_abs_prediction_difference': diff,
                    'max_abs_truth_difference': tdiff, 'bit_for_bit': exact, 'flagged_within_1e-9': flagged,
                    'passed': passed}
                print(f'G1 {variable} {arm} h{lead}: rows {len(d)} diff {diff} exact {exact}', flush=True)
    return {'rule': 'refit at the recorded selection on the 2026 outer fold; bit for bit, a difference up to '
                    '1e-9 flagged and read', 'cells': out, 'passed': bool(ok)}


def gate_objective(variable, arm, lead):
    """A86-G3 for one cell."""
    from src.experiments import v2_horizon as vh
    path = dev_selection_path(variable, arm, lead)
    sel = json.loads(path.read_text(encoding='utf-8'))
    data = vh.build(variable, lead)
    outer = {f['number']: f for f in vh.folds(data, lead)}
    inner = vh.inner_fold(data, outer[1], lead)
    value = Objective(data, inner, dev_design(arm, data, inner)).value({**sel['params'], 'subsample_freq': 1})
    diff = abs(value - float(sel['best_value']))
    return {'record': rel(path), 'record_sha256': _sha(path), 'study': sel['study'],
            'completed_trials': int(sel['completed_trials']), 'published_best_value': float(sel['best_value']),
            'objective_here': value, 'abs_diff': diff, 'passed': bool(diff <= 1e-9)}


def gate_design(variable, arm, lead):
    """A86-G4 for one cell."""
    from src.experiments import v2_common_rows as CR
    from src.experiments import v2_horizon as vh
    from src.experiments.bay_dynamics_tree import features
    dev = vh.build(variable, lead)
    res, outer, _ = CR.reserved_setup(variable, lead)
    n = len(dev['times'])
    prefix = bool(len(res['times']) >= n and (res['times'][:n] == dev['times']).all()
                  and list(res['names']) == list(dev['names']))
    fold = {'fallback': outer['fallback']}
    x26_full = Designs(arm, res)(fold)
    x26 = x26_full[:n]
    xdev = dev_design(arm, dev, fold)
    width = features(dev, fold, 'forecast')[0].shape[-1]
    cols = np.arange(x26.shape[-1])
    weather = np.arange(width, width + WEATHER_COLUMNS) if arm != 'river_tree' else np.array([], int)
    other = np.setdiff1d(cols, weather)
    out = {'prefix_rows': n, 'grid_rows': int(len(res['times'])), 'prefix_identical': prefix,
           'columns': int(x26.shape[-1]), 'development_columns': int(xdev.shape[-1]),
           'base_columns': int(width), 'weather_columns': weather.tolist()}
    out['same_width'] = bool(x26.shape == xdev.shape)
    out['non_weather_columns_identical'] = bool(out['same_width'] and nan_equal(x26[..., other], xdev[..., other]))
    if len(weather):
        part_b = CR.design(res, fold, 'weather_tree', CR.reserved_weather(res['times']))[..., weather]
        out['weather_equals_part_B_design'] = nan_equal(x26_full[..., weather], part_b)
        out['weather_max_abs_diff_to_development'] = nan_max_diff(x26[..., weather], xdev[..., weather])
        weather_ok = bool(out['weather_equals_part_B_design'] and out['weather_max_abs_diff_to_development'] is not None
                          and out['weather_max_abs_diff_to_development'] <= 1e-12)
    else:
        weather_ok = True
    in2026 = (res['times'] >= RESERVED_START)
    trap = {}
    if arm in ('river_tree', 'weather_river_tree', 'sealevel_tree', 'extboundary_tree'):
        first = width + (WEATHER_COLUMNS if arm != 'river_tree' else 0)
        k = 2 if arm == 'extboundary_tree' else 3
        block = x26_full[in2026, 0, first:first + k]
        trap = {'columns': list(range(first, first + k)),
                'finite_share_on_2026_rows': [round(float(v), 6) for v in np.isfinite(block).mean(axis=0)]}
        trap['passed'] = bool(min(trap['finite_share_on_2026_rows']) >= 0.9)
    out['trap_2026'] = trap
    out['passed'] = bool(prefix and out['non_weather_columns_identical'] and weather_ok
                         and (not trap or trap['passed']))
    return out


def gates():
    started = time.time()
    rep = {'declared_in': DECLARED, 'environment': environment()}
    rep['A86-G1_part_B_trees'] = gate_b_trees()
    g3, g4 = {}, {}
    for variable, arm, lead in CELLS:
        key = f'{variable}|{arm}|{lead}'
        g3[key] = gate_objective(variable, arm, lead)
        print(f'G3 {key}: {g3[key]["objective_here"]:.12f} vs {g3[key]["published_best_value"]:.12f} '
              f'passed {g3[key]["passed"]}', flush=True)
        g4[key] = gate_design(variable, arm, lead)
        print(f'G4 {key}: passed {g4[key]["passed"]} trap {g4[key]["trap_2026"]}', flush=True)
    rep['A86-G3_objective'] = {'tolerance': 1e-9, 'cells': g3}
    rep['A86-G4_design'] = {'cells': g4}
    rep['seconds'] = round(time.time() - started, 1)
    _json(GATES / 'gates_B.json', rep)
    print('G1', rep['A86-G1_part_B_trees']['passed'], 'G3', all(v['passed'] for v in g3.values()),
          'G4', all(v['passed'] for v in g4.values()), rep['seconds'], 's', flush=True)
    return rep


def cell_gates_passed(variable, arm, lead):
    path = GATES / 'gates_B.json'
    if not path.exists():
        return False
    g = json.loads(path.read_text(encoding='utf-8'))
    key = f'{variable}|{arm}|{lead}'
    return bool(g['A86-G3_objective']['cells'][key]['passed'] and g['A86-G4_design']['cells'][key]['passed'])


# --------------------------------------------------------------------------------------------------
# Part B: one study and its 2026 predictions
# --------------------------------------------------------------------------------------------------
def folder_for(variable, arm, lead, quick):
    return OUT / 'B' / 'studies' / ('smoke' if quick else 'full') / study_name(variable, arm, lead)


def prediction_path(variable, arm, lead, quick=False):
    return OUT / 'B' / ('arms_2026_smoke' if quick else 'arms_2026') / f'{variable}_{arm}_h{lead}.csv.gz'


def run(variable, arm, lead, quick=False):
    from src.experiments import v2_common_rows as CR
    from src.experiments import v2_horizon as vh
    from src.tuning.protocol_v2 import PROTOCOL, run_study
    if (variable, arm, lead) not in CELLS:
        raise SystemExit(f'undeclared cell {variable} {arm} {lead}')
    if not quick and not cell_gates_passed(variable, arm, lead):
        raise SystemExit('gates A86-G3 and G4 have not passed for this cell: no study is run')
    started = time.time()
    folder = folder_for(variable, arm, lead, quick)
    report = {'declared_in': DECLARED + ' part_B', 'variable': variable, 'arm': arm, 'lead_hours': lead,
              'family': FAMILY[arm], 'study': study_name(variable, arm, lead), 'protocol': PROTOCOL['version'],
              'smoke_only': bool(quick), 'environment': environment()}
    data, outer, inner = CR.reserved_setup(variable, lead)
    if quick:
        inner['tr'], inner['va'] = inner['tr'][-300:], inner['va'][:40]
    audit = CR.inner_audit(data, inner, lead)
    report['inner_window'] = audit
    report['outer_fold'] = {'training_issuances': int(len(outer['tr'])), 'reserved_issuances': int(len(outer['va'])),
                            'last_training_target': str(data['times'][outer['tr']].max() + pd.Timedelta(hours=lead)),
                            'eligible_sensors': [s for s, e in zip(data['names'], outer['eligible']) if e]}
    if not audit['no_2026_row']:
        report['stopped'] = 'gate A86-G0 failed: the inner window reaches past its declared end'
        _json(folder / 'report.json', report)
        raise SystemExit(report['stopped'])
    designs = Designs(arm, data)
    sel = run_study(study_name(variable, arm, lead), Objective(data, inner, designs(inner)), folder, quick=quick)
    record = folder / f'{study_name(variable, arm, lead)}_selection.json'
    _lf(record)
    report['selection'] = sel
    report['objective'] = "fold-macro forecast MAE / persistence MAE on the 2026 fold's inner window only"
    params = {**sel['params'], 'subsample_freq': 1}
    if quick:
        outer = {**outer, 'tr': outer['tr'][-300:]}
    clim = vh.climatology(data, outer)
    d, _ = vh.tree_fit_score(data, designs(outer), outer, params, clim)
    d = d.drop(columns=['mode']).assign(time=pd.to_datetime(d.time, utc=True), lead=lead, arm=arm)
    path = prediction_path(variable, arm, lead, quick)
    _csv_gz(path, d)
    report['predictions'] = {'file': rel(path), 'sha256': _sha(path), 'rows': int(len(d)),
                             'sensors': int(d.sensor.nunique())}
    report['selection_record_sha256'] = _sha(record)
    if FAMILY[arm] == 'ispra':
        from src.data.bay_ispra import BINNINGS, binning
        report['hourly_binning'] = binning()
        report['hourly_binning_rule'] = BINNINGS[binning()]
    report['seconds'] = round(time.time() - started, 1)
    _json(folder / 'report.json', report)
    print(f'{study_name(variable, arm, lead)}: best {sel["best_value"]:.6f} over {sel["completed_trials"]} trials '
          f'({sel.get("failed_trials", 0)} failed), {len(d)} rows, {report["seconds"]} s', flush=True)
    return report


def plan():
    return [(v, a, h, (folder_for(v, a, h, False) / 'report.json').exists()) for v, a, h in CELLS]


def launch(workers):
    """Every pending study, `workers` at a time, each in its own process at two threads."""
    logs = OUT / 'B' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    pending = [(v, a, h) for v, a, h, done in plan() if not done]
    env = {k: v for k, v in os.environ.items() if k != 'OMP_NUM_THREADS'}
    env.update(V2_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    running = []
    while pending or running:
        while pending and len(running) < workers:
            v, a, h = pending.pop(0)
            log = open(logs / f'{study_name(v, a, h)}.log', 'wb')
            p = subprocess.Popen([sys.executable, '-m', 'src.experiments.v2_guidance2026', 'study', '--variable', v,
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


def predictions(variable, arm, lead):
    """A finished part-B study's 2026 rows after checking its record."""
    folder = folder_for(variable, arm, lead, False)
    rep = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    sel = rep['selection']
    if sel.get('smoke_only') or int(sel.get('completed_trials', 0)) < MIN_TRIALS:
        raise ValueError(f'{study_name(variable, arm, lead)}: not a protocol-v2 selection')
    if not rep['inner_window']['no_2026_row']:
        raise ValueError(f'{study_name(variable, arm, lead)}: gate A86-G0 did not pass')
    path = ROOT / rep['predictions']['file']
    if _sha(path) != rep['predictions']['sha256']:
        raise ValueError(f'{path.name}: digest differs from its study record')
    d = _read_csv(path)
    d['time'] = pd.to_datetime(d.time, utc=True)
    return d, {'study': sel['study'], 'completed_trials': int(sel['completed_trials']),
               'best_value': sel['best_value'], 'report': rel(folder / 'report.json'),
               'predictions': rep['predictions']}


# --------------------------------------------------------------------------------------------------
# Part C: the graph operators
# --------------------------------------------------------------------------------------------------
def _torch_threads():
    import torch
    torch.set_num_threads(int(os.environ['V2_THREADS']))
    return 'cpu'


def graph_rows(variable, arm, params, lead, seeds, device):
    """v2_horizon.neural_forecast on the 2026 outer fold (part B's), one frame per seed."""
    from src.experiments import v2_common_rows as CR
    from src.experiments import v2_horizon as vh
    from src.experiments.v2_forecast import NEURAL_ARMS, graphs
    data, outer, _ = CR.reserved_setup(variable, lead)
    graph, exchange, relaxation = NEURAL_ARMS[arm]
    data['adjacency'] = graphs(data)[graph]
    clim = vh.climatology(data, outer)
    frames = []
    for seed in seeds:
        t0 = time.time()
        d = vh.neural_forecast(data, outer, params, exchange, relaxation, seed, device, clim)
        frames.append(d.drop(columns=['mode']).assign(time=pd.to_datetime(d.time, utc=True), seed=seed,
                                                      lead=lead, arm=arm))
        print(f'  {variable} {arm} h{lead} seed {seed}: {len(d)} rows, {time.time() - t0:.1f} s', flush=True)
    return frames, data, outer


def seed_mean(frames):
    stacked = pd.concat(frames, ignore_index=True)
    keys = ['fold', 'sensor', 'time']
    first = stacked.groupby(keys, as_index=False, sort=False).agg(
        truth=('truth', 'first'), persistence=('persistence', 'first'), baseline=('baseline', 'first'),
        clim_hour=('clim_hour', 'first'), prediction=('prediction', 'mean'), seeds=('seed', 'nunique'))
    return first


def graph_gate():
    """A86-G5: (i) this module's refit path against v2_horizon_reserved.neural_arm on the frozen shared 24 h
    learned_source configuration at seed 42; (ii) the three-seed mean at the frozen shared 24 h
    configurations against runs/v2_horizon_reserved_h24."""
    from src.experiments import v2_horizon as vh
    from src.experiments import v2_horizon_reserved as hr
    from src.experiments.v2_forecast import NEURAL_ARMS, graphs
    device = _torch_threads()
    rep = {'declared_in': DECLARED, 'environment': environment(), 'device': device, 'identity': {}, 'published': {}}
    started = time.time()
    for variable in ('temperature', 'oxygen'):
        summary = json.loads((hr.OUT / f'reserved_h24_{variable}_summary.json').read_text())
        prov = summary['selection_provenance']
        published = pd.read_csv(hr.OUT / f'reserved_h24_{variable}.csv.gz', float_precision='round_trip')
        # (i) identity
        params = prov['learned_source']['params']
        data = hr.build(variable, 24)
        bundle = hr.split(data, 24)
        graph, exchange, relaxation = NEURAL_ARMS['learned_source']
        data['adjacency'] = graphs(data)[graph]
        reference = hr.neural_arm(data, bundle, params, exchange, relaxation, 42, device,
                                  hr.climatology(data, bundle), 'learned_source')
        frames, _, _ = graph_rows(variable, 'learned_source', params, 24, (42,), device)
        mine = frames[0]
        a = reference[['sensor', 'time', 'prediction']].assign(time=lambda f: pd.to_datetime(f.time, utc=True))
        j = a.merge(mine[['sensor', 'time', 'prediction']], on=['sensor', 'time'], how='outer',
                    suffixes=('_ref', '_here'), indicator=True)
        both = j[j._merge == 'both']
        diff = float((both.prediction_ref - both.prediction_here).abs().max()) if len(both) else None
        rep['identity'][variable] = {'rows_reference': int(len(a)), 'rows_here': int(len(mine)),
                                     'unmatched': int((j._merge != 'both').sum()), 'max_abs_diff': diff,
                                     'passed': bool((j._merge != 'both').sum() == 0 and diff == 0.0)}
        print('G5(i)', variable, rep['identity'][variable], flush=True)
        # (ii) published reproduction, three-seed mean
        for arm in GRAPH_ARMS:
            params = prov[arm]['params']
            frames, _, _ = graph_rows(variable, arm, params, 24, SEEDS, device)
            m = seed_mean(frames)
            p = published[published.arm == arm][['sensor', 'time', 'truth', 'prediction']].assign(
                time=lambda f: pd.to_datetime(f.time, utc=True))
            j = p.merge(m[['sensor', 'time', 'prediction']], on=['sensor', 'time'], how='outer',
                        suffixes=('_pub', '_here'), indicator=True)
            both = j[j._merge == 'both']
            d = (both.prediction_pub - both.prediction_here).abs()
            q = both.assign(quarter=np.where(both.time.dt.month <= 3, 'Q1', 'Q2'))
            rep['published'][f'{variable}|{arm}'] = {
                'rows_published': int(len(p)), 'rows_here': int(len(m)), 'unmatched': int((j._merge != 'both').sum()),
                'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
                'published_mae': summary['mae'][arm],
                'mae_here': hr.macro(q.assign(prediction=q.prediction_here)),
                'passed': bool((j._merge != 'both').sum() == 0 and float(d.max()) <= 1e-6)}
            print('G5(ii)', variable, arm, {k: v for k, v in rep['published'][f'{variable}|{arm}'].items()
                                            if k != 'published_mae'}, flush=True)
    rep['identity_passed'] = bool(all(v['passed'] for v in rep['identity'].values()))
    rep['published_passed'] = bool(all(v['passed'] for v in rep['published'].values()))
    rep['host_dependent'] = not rep['published_passed']
    rep['seconds'] = round(time.time() - started, 1)
    _json(GATES / 'gate_G5_graph_path.json', rep)
    return rep


def graphs_2026():
    """Part C1: the two operators at 6 h at addendum 82 part A's fold-2026 selections, three seeds."""
    from src.experiments import v2_rollout_a82_studies as A
    g5 = json.loads((GATES / 'gate_G5_graph_path.json').read_text(encoding='utf-8'))
    if not g5['identity_passed']:
        raise SystemExit('gate A86-G5 (i) failed: part C is not run')
    device = _torch_threads()
    rep = {'declared_in': DECLARED + ' part_C', 'environment': environment(), 'device': device,
           'host_dependent': bool(g5['host_dependent']), 'arms': {}}
    started = time.time()
    for variable in ('temperature', 'oxygen'):
        for arm in GRAPH_ARMS:
            sel = A.selection(variable, arm)
            frames, _, _ = graph_rows(variable, arm, sel['params'], 6, SEEDS, device)
            per_seed = pd.concat(frames, ignore_index=True)
            path = OUT / 'C' / 'graphs_2026' / f'{variable}_{arm}_h6_per_seed.csv.gz'
            _csv_gz(path, per_seed)
            rep['arms'][f'{variable}|{arm}'] = {'selection': {k: sel[k] for k in ('source', 'sha256', 'study',
                                                                                 'completed_trials', 'params')},
                                               'predictions': rel(path), 'sha256': _sha(path),
                                               'rows_per_seed': int(len(frames[0])), 'seeds': list(SEEDS)}
    rep['seconds'] = round(time.time() - started, 1)
    _json(OUT / 'C' / 'graphs_2026.json', rep)
    return rep


def saturation_time():
    """Part C2: the A-G6 objective gate for saturation_exchange at its stored fold-1 per-fold selection on
    this host, and the CPU time of one objective evaluation on the development inner window at a
    configuration in the middle of the space, scaled to 100 trials."""
    from src.experiments import v2_horizon as vh
    from src.experiments import v2_perfold_select as pfs
    from src.experiments import v2_rollout_a82_studies as A
    _torch_threads()
    rec = pfs.forecast('oxygen', 'saturation_exchange', 6, 1)
    data = vh.build('oxygen', 6)
    outer = {f['number']: f for f in vh.folds(data, 6)}
    inner = vh.inner_fold(data, outer[1], 6)
    objective = A.GraphObjective(data, inner, 'saturation_exchange')
    t0 = time.time()
    value = objective.value(rec['params'])
    gate_seconds = time.time() - t0
    mid = {'hidden': 64, 'lr': 1e-3, 'weight_decay': 1e-3, 'steps': 300}
    t0 = time.time()
    objective.value(mid)
    mid_seconds = time.time() - t0
    # the space's steps are 150, 300 or 600 with equal prior weight: the mean is 350, scaled from 300
    estimate = 100 * mid_seconds * 350 / 300
    rep = {'declared_in': DECLARED + ' part_C C2', 'environment': environment(),
           'gate_A-G6_saturation_exchange': {'record': rec['source'], 'published_best_value': float(rec['best_value']),
                                             'objective_here': value, 'abs_diff': abs(value - float(rec['best_value'])),
                                             'tolerance': 1e-6,
                                             'passed': bool(abs(value - float(rec['best_value'])) <= 1e-6),
                                             'seconds': round(gate_seconds, 1)},
           'one_evaluation_seconds_at': {'params': mid, 'seconds': round(mid_seconds, 1)},
           'estimated_study_seconds': round(estimate, 0), 'limit_seconds': 7200}
    rep['run_here'] = bool(rep['gate_A-G6_saturation_exchange']['passed'] and estimate <= 7200)
    _json(OUT / 'C' / 'saturation_estimate.json', rep)
    print(json.dumps(rep, indent=1), flush=True)
    return rep


SATURATION_STUDY = 'a86C_oxygen_saturation_exchange_h6_fold2026'


def saturation_study():
    """Part C2, run here only when the estimate allows it: one 100-trial study of saturation_exchange on the
    2026 fold's inner window (the per-fold graph objective of addendum 49 at initialisation seed 42), then
    the three-seed refit on the 2026 outer fold."""
    from src.experiments import v2_common_rows as CR
    from src.experiments import v2_rollout_a82_studies as A
    from src.tuning.protocol_v2 import PROTOCOL, run_study
    est = json.loads((OUT / 'C' / 'saturation_estimate.json').read_text(encoding='utf-8'))
    if not est['run_here']:
        raise SystemExit('part C2 is left to the compute host: its estimate or its gate does not allow it here')
    device = _torch_threads()
    started = time.time()
    folder = OUT / 'C' / 'studies' / SATURATION_STUDY
    data, outer, inner = CR.reserved_setup('oxygen', 6)
    audit = CR.inner_audit(data, inner, 6)
    rep = {'declared_in': DECLARED + ' part_C C2', 'environment': environment(), 'device': device,
           'study': SATURATION_STUDY, 'protocol': PROTOCOL['version'], 'inner_window': audit,
           'estimate': rel(OUT / 'C' / 'saturation_estimate.json')}
    if not audit['no_2026_row']:
        rep['stopped'] = 'gate A86-G0 failed: the inner window reaches past its declared end'
        _json(OUT / 'C' / 'graphs_2026_saturation.json', rep)
        raise SystemExit(rep['stopped'])
    sel = run_study(SATURATION_STUDY, A.GraphObjective(data, inner, 'saturation_exchange'), folder)
    record = folder / f'{SATURATION_STUDY}_selection.json'
    _lf(record)
    rep['selection'] = sel
    rep['selection_record'] = rel(record)
    rep['selection_record_sha256'] = _sha(record)
    frames, _, _ = graph_rows('oxygen', 'saturation_exchange', sel['params'], 6, SEEDS, device)
    path = OUT / 'C' / 'graphs_2026' / 'oxygen_saturation_exchange_h6_per_seed.csv.gz'
    _csv_gz(path, pd.concat(frames, ignore_index=True))
    rep['predictions'] = rel(path)
    rep['sha256'] = _sha(path)
    rep['rows_per_seed'] = int(len(frames[0]))
    rep['seeds'] = list(SEEDS)
    rep['seconds'] = round(time.time() - started, 1)
    _json(OUT / 'C' / 'graphs_2026_saturation.json', rep)
    print(f'{SATURATION_STUDY}: best {sel["best_value"]:.6f} over {sel["completed_trials"]} trials, {rep["seconds"]} s',
          flush=True)
    return rep


# --------------------------------------------------------------------------------------------------
# Part D: output gates on the 2026 fold
# --------------------------------------------------------------------------------------------------
def rollouts():
    from src.experiments import v2_common_rows as CR
    from src.experiments import v2_physics_gate as PG
    from src.experiments.v2_closed_loop import episodes, fit_step_tree
    from src.experiments.v2_horizon import tree_fit_score, climatology
    started = time.time()
    rep = {'declared_in': DECLARED + ' part_D', 'environment': environment(), 'gate_G6': {}, 'runs': {}}
    for variable in ('temperature', 'oxygen'):
        data, outer, _ = CR.reserved_setup(variable, 6)
        sel, path = CR.a82_record(variable)
        params = {**sel['params'], 'subsample_freq': 1}
        met = CR.reserved_weather(data['times'])
        design = CR.design(data, outer, 'weather_tree', met)
        model = fit_step_tree(data, design, outer, sel['params'])
        # G6: the step tree's six-hour predictions against part B's stored weather tree
        mask = data['valid'] & outer['eligible'][None, :]
        va = outer['va']
        vm = mask[va]
        grid = np.full(data['x'][va].shape, np.nan)
        grid[vm] = model.predict(design[va][vm]) + data['x'][va][vm]
        rows = []
        for j, s in np.argwhere(vm):
            rows.append((data['names'][s], data['times'][va[j]], grid[j, s] * data['scale'] + data['center']))
        mine = pd.DataFrame(rows, columns=['sensor', 'time', 'prediction'])
        rep_b, _ = b_tree_record(variable, 'weather_tree', 6)
        stored = _read_csv(ROOT / rep_b['predictions']['file'])
        stored['time'] = pd.to_datetime(stored.time, utc=True)
        j = stored[['sensor', 'time', 'prediction']].merge(mine, on=['sensor', 'time'], how='outer',
                                                            suffixes=('_b', '_here'), indicator=True)
        both = j[j._merge == 'both']
        diff = float((both.prediction_b - both.prediction_here).abs().max())
        g6 = {'selection': rel(path), 'rows_stored': int(len(stored)), 'rows_here': int(len(mine)),
              'unmatched': int((j._merge != 'both').sum()), 'max_abs_diff': diff,
              'passed': bool((j._merge != 'both').sum() == 0 and diff <= 1e-9)}
        rep['gate_G6'][variable] = g6
        print('G6', variable, g6, flush=True)
        if not g6['passed']:
            continue
        limits = PG.bounds(data, outer)
        eps = episodes(data, outer, 120)
        eligible = np.flatnonzero(outer['eligible'])
        companion = True if variable == 'oxygen' else None
        steps = 120 // 6
        for regime in ('network', 'single'):
            out_rows, fire_log = [], []
            evaluable = {g: {'steps': 0, 'rate': 0, 'range': 0, 'saturation': 0} for g in PG.GATES}
            seen = set()
            for (i0, s) in eps:
                if regime == 'network':
                    if i0 in seen:
                        continue
                    seen.add(i0)
                    sensors = [int(k) for k in eligible]
                else:
                    sensors = [int(s)]
                for gate in PG.GATES:
                    paths, fires = PG.roll(data, design, model, i0, sensors, steps, gate, limits, regime,
                                           outer['fallback'], companion=companion, evaluable=evaluable[gate],
                                           issuance=True)
                    for f in fires:
                        fire_log.append({**f, 'gate_set': gate, 'issuance': str(data['times'][i0])})
                    for k in sensors:
                        for st in range(steps):
                            if not data['valid'][i0 + st, k]:
                                continue
                            out_rows.append({'fold': 2, 'sensor': data['names'][k], 'time': str(data['times'][i0]),
                                             'lead': (st + 1) * 6, 'gate': gate,
                                             'truth': float(data['truth'][i0 + st, k] * data['scale'] + data['center']),
                                             'prediction': float(paths[k][st] * data['scale'] + data['center'])})
            frame = pd.DataFrame(out_rows)
            frame['error'] = frame.prediction - frame.truth
            p = OUT / 'D' / f'gate_{variable}_{regime}_2026.csv.gz'
            _csv_gz(p, frame)
            fires = pd.DataFrame(fire_log)
            fire_counts = {} if fires.empty else {g: int(len(gg)) for g, gg in fires.groupby('gate_set')}
            fire_by_type = ({} if fires.empty else
                            {t: int(sum(t in f for f in fires[fires.gate_set == 'all'].gates))
                             for t in ['rate', 'range', 'saturation']})
            rep['runs'][f'{variable}|{regime}'] = {
                'predictions': rel(p), 'sha256': _sha(p), 'onsets': int(frame.time.nunique()),
                'episodes': int(frame.groupby(['sensor', 'time']).ngroups),
                'rows_per_gate': int(len(frame) / len(PG.GATES)), 'fire_counts': fire_counts,
                'fire_by_type': fire_by_type, 'evaluable_steps': evaluable}
            print(variable, regime, rep['runs'][f'{variable}|{regime}']['episodes'], 'episodes; fires', fire_counts,
                  flush=True)
    rep['seconds'] = round(time.time() - started, 1)
    _json(OUT / 'D' / 'rollouts_2026.json', rep)
    return rep


# --------------------------------------------------------------------------------------------------
# Part E1: the utilisation ridge, a development reading in mg/l
# --------------------------------------------------------------------------------------------------
E1_TRAIN_BEFORE = pd.Timestamp('2025-01-01', tz='UTC')
E1_TEST_END = pd.Timestamp('2026-01-01', tz='UTC')
C3_UNITS = RUNS / 'v2_oxygen_mgl/C3/v2_bottom_aou/aou_units/summary.json'


def aou_dev(stage, quick=False):
    """E1 inside addendum 60 part C3's mg/l context: gate A86-G7 first, then the year-earlier split."""
    from scripts import v2_aou_units as U
    from scripts.v2_oxygen_mgl_common import harmonised, sat_mgl
    from scripts.v2_summarize_aou import cast_errors, cast_meta, contrast, summarise
    from src.experiments import v2_bottom_aou as vr
    from src.experiments.v2_field_reserved2026 import macro
    started = time.time()
    folder = OUT / 'E' / ('E1_smoke' if quick else 'E1')
    eb, es = vr.import_experiments(Path(stage))
    saved = es.o2_solubility

    def sol(T, S=38.0):
        return sat_mgl(np.asarray(T, float), np.asarray(S, float))
    c3 = json.loads(C3_UNITS.read_text())
    rep = {'declared_in': DECLARED + ' part_E E1', 'environment': environment(), 'smoke_only': bool(quick),
           'units': 'mg/l (casts by their source unit, the per-litre Garcia-Gordon saturation x 0.0319988)'}
    preds, errors_all = [], {}
    with harmonised(True) as seen:
        es.o2_solubility = sol
        try:
            c = es.featurise(es.load())
            c['sol_l'] = sol(c.temp_c.to_numpy(float), c.salinity_psu.to_numpy(float))
            c['aou_l'] = c.sol_l - c.do_umol_l
            spec = U.arms(es)
            meta = cast_meta()
            cohorts = {'gen2_164': c[c.t_utc >= '2024-07-24'], 'full_722': c}
            # G7: the published 2026 split at the selections C3 recorded
            g7 = {}
            for cname in ('gen2_164',):
                tr, te = [f.reset_index(drop=True) for f in next(
                    (tr, te) for _, tr, te in es.folds(cohorts[cname], 'P2_chronological'))]
                for arm in ('direct_do', 'aou_l'):
                    cols, ycol, back = spec[arm]
                    alpha = c3['mae'][f'{cname}|{arm}']['alpha']
                    xtr, xte = vr.impute(tr, te, cols)
                    p = vr.make('ridge', {'alpha': alpha}).fit(xtr, tr[ycol].to_numpy(float)).predict(xte)
                    frame = pd.DataFrame(dict(cast_id=te.cast_id.values, truth=te.do_umol_l.values, pred=back(p, te)))
                    mae = macro(cast_errors(frame, meta, '2026'))
                    pub = c3['mae'][f'{cname}|{arm}']['nested']
                    g7[f'{cname}|{arm}'] = {'alpha': alpha, 'mae_here': mae, 'mae_c3': pub, 'abs_diff': abs(mae - pub),
                                            'passed': bool(abs(mae - pub) <= 1e-6)}
                    print('G7', cname, arm, g7[f'{cname}|{arm}'], flush=True)
            rep['A86-G7_mgl_context'] = {'cells': g7, 'passed': bool(all(v['passed'] for v in g7.values()))}
            if not rep['A86-G7_mgl_context']['passed'] and not quick:
                rep['stopped'] = 'gate A86-G7 failed: E1 is not read'
                _json(folder / 'report.json', rep)
                raise SystemExit(rep['stopped'])
            rep['split'] = {'train_before': str(E1_TRAIN_BEFORE), 'test_from': str(E1_TRAIN_BEFORE),
                            'test_before': str(E1_TEST_END)}
            rep['cohorts'], rep['mae'] = {}, {}
            for cname, cdf in cohorts.items():
                tr = cdf[cdf.t_utc < E1_TRAIN_BEFORE].reset_index(drop=True)
                te = cdf[(cdf.t_utc >= E1_TRAIN_BEFORE) & (cdf.t_utc < E1_TEST_END)].reset_index(drop=True)
                rep['cohorts'][cname] = {'training_rows': int(len(tr)), 'training_casts': int(tr.cast_id.nunique()),
                                         'test_rows': int(len(te)), 'test_casts': int(te.cast_id.nunique()),
                                         'first_training_cast': str(tr.t_utc.min()), 'last_training_cast': str(tr.t_utc.max()),
                                         'first_test_cast': str(te.t_utc.min()), 'last_test_cast': str(te.t_utc.max())}
                errors = {}
                for arm, (cols, ycol, back) in spec.items():
                    err = (lambda p, x, back=back: back(p, x) - x.do_umol_l.to_numpy(float))
                    p, sel = vr.tuned_predict('ridge', tr, te, cols, ycol, err, 'P2_chronological', folder / 'studies',
                                              f'a86E1_{cname}_ridge_{arm}', quick)
                    _lf(folder / 'studies' / f'a86E1_{cname}_ridge_{arm}_selection.json')
                    do = back(p, te)
                    frame = pd.DataFrame(dict(cohort=cname, space=arm, model='M2_ridge_tel', cast_id=te.cast_id.values,
                                              truth=te.do_umol_l.values, pred=do))
                    preds.append(frame)
                    errors[arm] = cast_errors(frame, meta, '2025')
                    rep['mae'][f'{cname}|{arm}'] = {'nested': round(macro(errors[arm]), 6), 'alpha': sel['params']['alpha'],
                                                    'completed_trials': sel['completed_trials'],
                                                    'best_value': sel['best_value']}
                    print(cname, arm, rep['mae'][f'{cname}|{arm}'], flush=True)
                errors_all[cname] = errors
        finally:
            es.o2_solubility = saved
    rep['conversions_seen'] = dict(seen)
    path = folder / 'predictions.csv.gz'
    _csv_gz(path, pd.concat(preds, ignore_index=True))
    rep['predictions'] = {'file': rel(path), 'sha256': _sha(path)}
    rep['contrasts'] = []
    import scripts.v2_summarize_aou as SA
    SA.round = six_decimals          # the module rounds to four; six are stored so nothing is rounded twice
    try:
        for cname, errors in errors_all.items():
            for a, b in (('aou_l', 'direct_do'), ('aou_l', 'direct_do_ts'), ('direct_do_ts', 'direct_do')):
                c = contrast(errors[a], errors[b])
                entry = {'cohort': cname, 'a': a, 'b': b, **summarise(c),
                         'mae_a': round(macro(errors[a]), 6), 'mae_b': round(macro(errors[b]), 6)}
                rep['contrasts'].append(entry)
                print(cname, a, '-', b, entry['diff'], entry['ci'], entry['jackknife_t'], flush=True)
    finally:
        del SA.round
    rep['seconds'] = round(time.time() - started, 1)
    _json(folder / 'report.json', rep)
    return rep


def six_decimals(x, n=None):
    import builtins
    return builtins.round(x, None if n is None else max(n, 6))


# --------------------------------------------------------------------------------------------------
# Part E2: the fixed depth curve P1 on Field-2026 (temperature)
# --------------------------------------------------------------------------------------------------
P1_SELECTION = ROOT / 'results/v2_field_held_site_P1_placebo_direct_innerprior/held_site_P1_placebo_direct_innerprior_selection.json'
A38_REPORT = RUNS / 'v2_field_reserved2026_copernicus/report.json'


def field2026_p1():
    """Addendum 38's held-station read on the 38 casts of 2026 (v2_field_reserved2026_copernicus.read), with
    one more arm: P1, A1_both with the three vertical-offset columns (in the background and among the
    regressors) replaced by the fixed depth curve of v2_field_placebo, fitted on the read's training rows,
    at the penalty of addendum 71's inner-split study. Check before P1 is read: the loop's A2_none and
    A1_both reproduce addendum 38's report (its per-row predictions were not kept)."""
    import src.experiments.v2_field_reserved2026_copernicus as C
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from src.data.bay_profile_context import TARGETS, context_features, load_profiles
    from src.experiments.bay_field_export import EXTRAS
    from src.experiments.v2_field import fit_predict
    from src.experiments.v2_field_placebo import placebo_offsets
    from src.experiments.v2_field_reserved2026 import END, GAP, START, observations_through, profiles_window
    started = time.time()
    pre, _ = load_profiles()
    new = profiles_window(START, END)
    data = pd.concat([pre, new], ignore_index=True)
    published = pre[['cast_id', 'depth_m']].merge(
        pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
        on=['cast_id', 'depth_m'], validate='one_to_one')[EXTRAS].to_numpy(float)
    fresh, stale = C.boundary(new.reset_index(drop=True), C.products(C.NEW))
    extras = np.vstack([published, fresh])
    meta, frames = observations_through(END)
    y = data[TARGETS].to_numpy(float)
    depth = data.depth_m.to_numpy(float)
    a36 = json.loads(C.ADDENDUM36.read_text())
    p1 = json.loads(P1_SELECTION.read_text(encoding='utf-8'))
    alphas = {'A2_none': a36['development']['A2_none']['selection']['params']['alpha'],
              'A1_both': C.PUBLISHED['A1_both']['alpha'], 'P1': p1['params']['alpha']}
    frames_out = []
    for s in sorted(new.station.unique()):
        X = context_features(data, meta, frames, s)
        tr = np.flatnonzero((data.t_utc < START - GAP).to_numpy() & (data.station != s).to_numpy())
        te = np.flatnonzero((data.t_utc >= START).to_numpy() & (data.station == s).to_numpy()
                            & np.isfinite(extras).all(axis=1))
        fallback = np.median(y[tr], axis=0)
        base = X[['temperature_context', 'salinity_context', 'oxygen_context']].to_numpy().copy()
        for j in range(3):
            base[~np.isfinite(base[:, j]), j] = fallback[j]
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True).fit(X.iloc[tr])
        sx = StandardScaler().fit(imputer.transform(X)[tr]).transform(imputer.transform(X))
        training = data.iloc[tr]
        w = 1 / (training.groupby('cast_id').cast_id.transform('size').to_numpy()
                 * training.groupby('station').cast_id.transform('nunique').to_numpy())
        w = w / w.mean()
        ex = np.nan_to_num(extras)
        mask = np.zeros(len(data), dtype=bool)
        mask[tr] = True
        offsets = placebo_offsets('P1', depth, mask, y - base, None)
        ex_p1 = ex.copy()
        ex_p1[:, 3:6] = offsets
        frame = data.iloc[te][['cast_id', 'station', 't_utc', 'depth_m'] + TARGETS].reset_index(drop=True)
        frame['fold'] = 0
        arms = {'A2_none': (0.0, sx), 'A1_both': (ex[:, 3:6], np.concatenate([sx, ex], axis=1)),
                'P1': (offsets, np.concatenate([sx, ex_p1], axis=1))}
        for arm, (offset, feats) in arms.items():
            bs = base + offset
            bs[:, 0] = np.clip(bs[:, 0], -2, 40)
            bs[:, 1] = np.clip(bs[:, 1], 0, 42)
            pred, _, _ = fit_predict(feats, bs, y, tr, te, w, alphas[arm], False)
            for j, t in enumerate(TARGETS):
                frame[f'{arm}|{t}'] = pred[:, j]
        frames_out.append(frame)
        print('held', s, 'test casts', frame.cast_id.nunique(), flush=True)
    d = pd.concat(frames_out, ignore_index=True)
    # addendum 38 kept its report, not its per-row predictions (trimmed from the repository): the read
    # must reproduce the report's nested errors of A2_none and A1_both at the four decimals it stores,
    # for every target, and its A1_both - A2_none contrasts (point, interval, jackknife) on temperature
    from src.experiments.v2_field_reserved2026 import cast_errors, contrast as a36_contrast, macro
    a38 = json.loads(A38_REPORT.read_text(encoding='utf-8'))['held_station']
    gate = {'source': rel(A38_REPORT), 'casts': int(d.cast_id.nunique()), 'casts_a38': a38['casts'],
            'dates': int(d.t_utc.dt.floor('D').nunique()), 'dates_a38': a38['dates'], 'stale_casts': stale,
            'nested_mae': {}}
    ok = gate['casts'] == gate['casts_a38'] and gate['dates'] == gate['dates_a38']
    for arm in ('A2_none', 'A1_both'):
        for t in TARGETS:
            here = round(macro(cast_errors(d, arm, t)), 4)
            gate['nested_mae'][f'{arm}|{t}'] = {'here': here, 'a38': a38['nested_mae'][arm][t]}
            ok &= abs(here - a38['nested_mae'][arm][t]) < 1e-9
    c = a36_contrast(cast_errors(d, 'A1_both', 'temp_c'), cast_errors(d, 'A2_none', 'temp_c'))
    pub = a38['contrasts']['A1_both - A2_none']['temp_c']
    gate['contrast_A1_both_minus_A2_none_temp'] = {'here': {k: c[k] for k in ('point', 'ci', 'jackknife_t')},
                                                   'a38': {k: pub[k] for k in ('point', 'ci', 'jackknife_t')}}
    ok &= c['point'] == pub['point'] and c['ci'] == pub['ci'] and c['jackknife_t'] == pub['jackknife_t']
    gate['passed'] = bool(ok)
    print('E2 gate', gate, flush=True)
    path = OUT / 'E' / 'E2' / 'field2026_p1_predictions.csv.gz'
    _csv_gz(path, d.assign(t_utc=d.t_utc.astype(str)))
    rep = {'declared_in': DECLARED + ' part_E E2', 'environment': environment(), 'gate_reproduces_addendum38': gate,
           'penalties': alphas, 'p1_selection': rel(P1_SELECTION), 'p1_selection_sha256': _sha(P1_SELECTION),
           'predictions': rel(path), 'sha256': _sha(path), 'casts': int(d.cast_id.nunique()),
           'dates': int(d.t_utc.dt.floor('D').nunique()), 'seconds': round(time.time() - started, 1),
           'not_run': 'A1_both_aou on Field-2026 is not run here: its oxygen needs the mg/l rerun of the 2026 '
                      'Copernicus read (addendum 60 part C1), which this module does not rebuild'}
    _json(OUT / 'E' / 'E2' / 'report.json', rep)
    return rep


# --------------------------------------------------------------------------------------------------
def everything():
    """What the pipeline runs (no action): every step whose output is missing, in the declared order.
    Part C2's study runs only if its estimate allows it here."""
    import shutil
    path = GATES / 'gates_B.json'
    if not path.exists():
        gates()
    if any(not done for *_, done in plan()):
        launch(2)
    if not (GATES / 'gate_G5_graph_path.json').exists():
        graph_gate()
    if not (OUT / 'C' / 'graphs_2026.json').exists():
        graphs_2026()
    if not (OUT / 'C' / 'saturation_estimate.json').exists():
        saturation_time()
    if (not (OUT / 'C' / 'graphs_2026_saturation.json').exists()
            and json.loads((OUT / 'C' / 'saturation_estimate.json').read_text(encoding='utf-8'))['run_here']):
        saturation_study()
    if not (OUT / 'D' / 'rollouts_2026.json').exists():
        rollouts()
    if not (OUT / 'E' / 'E1' / 'report.json').exists():
        stage = OUT / 'E'
        try:
            aou_dev(stage)
        finally:
            shutil.rmtree(stage / 'staged_inputs', ignore_errors=True)
    if not (OUT / 'E' / 'E2' / 'report.json').exists():
        field2026_p1()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action')
    sub.add_parser('gates')
    s = sub.add_parser('study')
    s.add_argument('--variable', choices=('temperature', 'oxygen'), required=True)
    s.add_argument('--arm', choices=sorted(FAMILY), required=True)
    s.add_argument('--lead', type=int, required=True)
    s.add_argument('--quick', action='store_true')
    lp = sub.add_parser('launch')
    lp.add_argument('--workers', type=int, default=2)
    sub.add_parser('plan')
    sub.add_parser('graph-gate')
    sub.add_parser('graphs')
    sub.add_parser('saturation-time')
    sub.add_parser('saturation-study')
    sub.add_parser('rollouts')
    sub.add_parser('field-p1')
    e = sub.add_parser('aou-dev')
    e.add_argument('--stage', required=True, help='a folder outside the repository for the staged inputs')
    e.add_argument('--quick', action='store_true')
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
    elif a.action == 'graph-gate':
        graph_gate()
    elif a.action == 'graphs':
        graphs_2026()
    elif a.action == 'saturation-time':
        saturation_time()
    elif a.action == 'saturation-study':
        saturation_study()
    elif a.action == 'rollouts':
        rollouts()
    elif a.action == 'field-p1':
        field2026_p1()
    elif a.action == 'aou-dev':
        aou_dev(a.stage, a.quick)
    else:
        everything()
