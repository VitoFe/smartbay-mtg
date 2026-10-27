"""Addendum 60 part C3: the apparent-utilisation forecasting tree and the thermodynamic baseline in mg/l.

The published utilisation arm (src/experiments/v2_aou_forecast.py, Dev-2025, 6 h) subtracts the sondes'
per-kilogram oxygen from the Garcia-Gordon per-litre saturation, so its target mixes the two. Under
addendum 60 amendment 4 (declared 2026-10-02T14:05Z, after the first part-C3 run) the analysis, whose
only oxygen source is the sondes, is rerun with the sonde oxygen converted by the constant c and the
saturation in mg/l from the per-litre form (x 0.0319988). Only the arms that use saturation are
refitted, with their published configuration (and, for addendum 48, the per-design selections): the
utilisation tree and the thermodynamic baseline. Every arm they are compared with (plain and weather
trees, persistence, the climatological increment) and the truth are their published values x c, so
those arms carry one value throughout the thesis; every contrast is then computed pairwise on the
same endpoints with the published bootstrap streams.

The forecasting z-score of oxygen (centre 230, scale 50 umol) is part of the configuration: it is
carried into mg/l by c, which leaves every sonde z-value as it was.

Each run goes first in the published unit, where it must give the published rows and summary, then
in mg/l:

  C3/v2_aou_forecast/            aou_forecast_h6.csv.gz, aou_forecast_h6_summary.json, the selection
  C3/v2_perfold_rest/aou/<design>/  the three designs of addendum 48
  C3/v2_perfold_rest.json        its utilisation section (the Field-57 section is C1/v2_perfold_rest.json)
  C3/aou_forecast_reproduction.json

With --hourly-factor it instead reruns the first part-C3 run (every sonde hour by its own factor,
every arm refitted) into C3/_hourly_factor/, the record amendment 4 replaced.

Usage (V2_THREADS=2): python -m scripts.v2_oxygen_mgl_aou_forecast [--hourly-factor]
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import contextlib
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

import scripts.v2_oxygen_mgl_common as common
from scripts.v2_oxygen_mgl_common import OUT, RUNS, constant_c, harmonised, write_csv, write_json
from scripts.v2_oxygen_mgl_field_contrasts import same

HOURLY = '--hourly-factor' in sys.argv
C3 = OUT / 'C3' / ('_hourly_factor' if HOURLY else '')
SCRATCH = OUT / '_reproduction'
CENTRE, SCALE = 230.0, 50.0
KEY = ['fold', 'sensor', 'time']
#: arms that read no saturation, and the truth: published values x c (amendment 4)
REFERENCES = ('truth', 'weather_tree', 'plain_tree', 'persistence', 'climatology')
CONTRASTS = [('aou_tree', 'weather_tree'), ('aou_tree', 'plain_tree'), ('aou_tree', 'persistence'),
             ('thermodynamic', 'persistence'), ('thermodynamic', 'weather_tree'), ('thermodynamic', 'climatology')]
ARMS = ['aou_tree', 'weather_tree', 'plain_tree', 'thermodynamic', 'persistence', 'climatology']


def oxygen_build(original):
    """v2_horizon.build with oxygen z-scored at (230, 50) x c; called while harmonised() converts the sondes."""
    def build(variable, horizon):
        d = original(variable, horizon)
        if variable != 'oxygen':
            return d
        c = constant_c()
        nc, ns = CENTRE * c, SCALE * c
        d = dict(d)
        for k in ('x', 'truth', 'future'):
            d[k] = (d[k] * SCALE + CENTRE - nc) / ns
        forcing = d['forcing'].copy()
        forcing[..., 0] = (forcing[..., 0].astype(np.float64) * (SCALE / ns)).astype(forcing.dtype)
        d['forcing'] = forcing
        d['center'], d['scale'] = nc, ns
        return d
    return build


@contextlib.contextmanager
def sonde_unit():
    """Amendment 4: every sonde hour by the constant c (the hourly factor only with --hourly-factor)."""
    if HOURLY:
        yield
        return
    saved = common.sonde_factor
    common.sonde_factor = lambda sensor, index: np.full(len(index), constant_c())
    try:
        yield
    finally:
        common.sonde_factor = saved


class Patched:
    """Install the mg/l build (when asked) and hand every study its published selection."""

    def __init__(self, mgl, selections):
        self.mgl, self.selections = mgl, selections

    def __enter__(self):
        import src.experiments.v2_horizon as H
        import src.tuning.protocol_v2 as T
        from src.experiments import v2_aou_forecast as A
        self.saved = (H.build, A.build, T.run_study, A.run_study)
        fake = lambda name, objective, folder, quick=False: self.selections[name]
        T.run_study = A.run_study = fake
        if self.mgl:
            H.build = A.build = oxygen_build(self.saved[0])
        self.unit = sonde_unit() if self.mgl else contextlib.nullcontext()
        self.unit.__enter__()
        self.h = harmonised(self.mgl)
        self.h.__enter__()
        return self

    def __exit__(self, *exc):
        self.h.__exit__(*exc)
        self.unit.__exit__(*exc)
        import src.experiments.v2_horizon as H
        import src.tuning.protocol_v2 as T
        from src.experiments import v2_aou_forecast as A
        H.build, A.build, T.run_study, A.run_study = self.saved
        return False


def selections():
    sel = json.loads((RUNS / 'v2_aou_forecast/oxygen_aou_tree_h6_selection.json').read_text())
    out = {'oxygen_aou_tree_h6': sel}
    shared = json.loads((RUNS / 'v2_perfold_rest/aou/shared/report.json').read_text())['selection']
    out['oxygen_aou_tree_h6_shared'] = shared
    for k, s in json.loads((RUNS / 'v2_perfold_rest/aou/perfold/report.json').read_text())['selection'].items():
        out[f'oxygen_aou_tree_h6_perfold_fold{k}'] = s
    return out


def references_times_c(frame, published):
    """Amendment 4: the truth and every arm that reads no saturation are the published values x c."""
    if len(frame) != len(published) or not (frame[KEY].astype(str).to_numpy() == published[KEY].astype(str).to_numpy()).all():
        raise SystemExit('the refitted rows are not the published rows')
    out = frame.copy()
    for col in REFERENCES:
        out[col] = published[col].to_numpy(float) * constant_c()
    return out


def run_aou_forecast(out, mgl):
    from src.experiments import v2_aou_forecast as A
    saved = (A.OUT, A.save)
    A.OUT = out
    A.save = lambda path, obj: write_json(path, obj)
    try:
        with Patched(mgl, selections()):
            A.main()
    finally:
        A.OUT, A.save = saved
    frame = pd.read_csv(out / 'aou_forecast_h6.csv.gz')
    summary = json.loads((out / 'aou_forecast_h6_summary.json').read_text())
    if mgl and not HOURLY:
        frame = references_times_c(frame, pd.read_csv(RUNS / 'v2_aou_forecast/aou_forecast_h6.csv.gz'))
        summary['mae'] = {a: A.macro(frame, a) for a in ARMS}
        summary['contrasts'] = {f'{a} - {b}': A.paired(frame, a, b, f'{a}|{b}') for a, b in CONTRASTS}
    write_csv(frame, out / 'aou_forecast_h6.csv.gz')
    summary['prediction_sha256'] = hashlib.sha256((out / 'aou_forecast_h6.csv.gz').read_bytes()).hexdigest()
    write_json(out / 'aou_forecast_h6_summary.json', summary)
    shutil.copyfile(RUNS / 'v2_aou_forecast/oxygen_aou_tree_h6_selection.json', out / 'oxygen_aou_tree_h6_selection.json')
    return summary, pd.read_csv(out / 'aou_forecast_h6.csv.gz')


def run_perfold(work, runs, mgl):
    from src.experiments import v2_aou_forecast as A
    from src.experiments import v2_perfold_rest as P
    saved = (P.OUT, P.RUNS)
    P.OUT, P.RUNS = work, runs
    try:
        with Patched(mgl, selections()):
            P.run_aou(('published', 'shared', 'perfold'))
    finally:
        P.OUT, P.RUNS = saved
    for d in ('published', 'shared', 'perfold'):
        f = work / 'aou' / d
        frame = pd.read_csv(f / 'aou_forecast_h6.csv.gz')
        r = json.loads((f / 'report.json').read_text())
        if mgl and not HOURLY:
            # each design's comparison trees are that design's published refits, x c
            frame = references_times_c(frame, pd.read_csv(RUNS / 'v2_perfold_rest/aou' / d / 'aou_forecast_h6.csv.gz'))
            r['mae'] = {a: A.macro(frame, a) for a in ARMS}
            if d == 'published':
                ref = pd.read_csv(runs / 'v2_aou_forecast/aou_forecast_h6.csv.gz')
                diff = {c: float(np.abs(frame[c].to_numpy() - ref[c].to_numpy()).max()) for c in ARMS + ['truth']}
                r.update(gate_G1_rows_identical=True, gate_G1_max_abs_diff=diff,
                         gate_G1_pass=bool(max(diff.values()) <= 1e-9))
        write_csv(frame, f / 'aou_forecast_h6.csv.gz')
        r['prediction_sha256'] = hashlib.sha256((f / 'aou_forecast_h6.csv.gz').read_bytes()).hexdigest()
        r['threads'] = int(os.environ['V2_THREADS'])
        write_json(f / 'report.json', r)


def contrasts48(runs, work):
    from scripts import v2_perfold_rest_contrasts as Q
    saved = (Q.RUNS, Q.WORK)
    Q.RUNS, Q.WORK = runs, work
    try:
        return Q.aou()
    finally:
        Q.RUNS, Q.WORK = saved


def frame_diff(a, b):
    if len(a) != len(b) or not (a[KEY].astype(str).to_numpy() == b[KEY].astype(str).to_numpy()).all():
        return None
    cols = [c for c in b.columns if c not in KEY]
    return float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float))))


def main():
    t0 = time.time()
    record = {'what': 'the utilisation forecasting tree and its comparison arms rerun in the published unit with '
                      'the published selections, against the published rows and summaries, before mg/l',
              'threads': int(os.environ['V2_THREADS']), 'steps': {}}
    pub = json.loads((RUNS / 'v2_aou_forecast/aou_forecast_h6_summary.json').read_text())
    got, frame = run_aou_forecast(SCRATCH / 'v2_aou_forecast', False)
    rows = frame_diff(frame, pd.read_csv(RUNS / 'v2_aou_forecast/aou_forecast_h6.csv.gz'))
    diff = same({k: got[k] for k in ('mae', 'contrasts', 'endpoints')}, {k: pub[k] for k in ('mae', 'contrasts', 'endpoints')},
                tol=1e-9)
    ok = rows is not None and rows <= 1e-9 and not diff
    record['steps']['v2_aou_forecast'] = {'pass': ok, 'max_abs_row_diff': rows, 'differences': diff[:20]}
    print('v2_aou_forecast reproduction', ok, rows, diff[:3], f'{time.time() - t0:.0f} s', flush=True)
    if not ok:
        write_json(C3 / 'aou_forecast_reproduction.json', record)
        raise SystemExit('the utilisation forecast does not reproduce')
    run_perfold(SCRATCH / 'v2_perfold_rest', RUNS, False)
    bad = {}
    for d in ('published', 'shared', 'perfold'):
        a = pd.read_csv(SCRATCH / 'v2_perfold_rest/aou' / d / 'aou_forecast_h6.csv.gz')
        b = pd.read_csv(RUNS / 'v2_perfold_rest/aou' / d / 'aou_forecast_h6.csv.gz')
        bad[d] = frame_diff(a, b)
    pubrest = json.loads((RUNS / 'v2_perfold_rest.json').read_text())['aou']
    got = json.loads(json.dumps(contrasts48(RUNS, SCRATCH / 'v2_perfold_rest')))
    diff = same({k: got[k] for k in ('mae', 'contrasts')}, {k: pubrest[k] for k in ('mae', 'contrasts')}, tol=1e-9)
    ok = all(v is not None and v <= 1e-9 for v in bad.values()) and not diff
    record['steps']['v2_perfold_rest.aou'] = {'pass': ok, 'max_abs_row_diff': bad, 'differences': diff[:20]}
    print('addendum 48 utilisation reproduction', ok, bad, diff[:3], f'{time.time() - t0:.0f} s', flush=True)
    if not ok:
        write_json(C3 / 'aou_forecast_reproduction.json', record)
        raise SystemExit('the addendum-48 utilisation designs do not reproduce')
    # amendment 4: the reference arms' published predictions x c give the MAEs every other table prints
    c = constant_c()
    record['references_times_c'] = {a: pub['mae'][a] * c for a in ('plain_tree', 'weather_tree', 'persistence', 'climatology')}

    # mg/l
    summary, _ = run_aou_forecast(C3 / 'v2_aou_forecast', True)
    summary['harmonised'] = {
        'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json'] +
                       ([] if HOURLY else ['config/bay_reserved_protocol_v2_addendum60_amendment4.json']),
        'unit': 'mg/l (columns keep their published names)',
        'sonde_oxygen': 'each hour by its own density factor (superseded by amendment 4)' if HOURLY else
                        'times the constant c (amendment 4)',
        'saturation': 'Garcia-Gordon per-litre form x 0.0319988 at salinity 38',
        'refitted': 'every arm' if HOURLY else 'aou_tree (published selection) and the thermodynamic baseline',
        'published_times_c': [] if HOURLY else list(REFERENCES),
        'oxygen_z_score': f'centre {CENTRE} and scale {SCALE} umol carried into mg/l by c',
        'selection': 'the published 100-trial selection, not re-selected'}
    write_json(C3 / 'v2_aou_forecast/aou_forecast_h6_summary.json', summary)
    if not HOURLY:
        for a, v in record['references_times_c'].items():
            if abs(summary['mae'][a] - v) > 1e-12:
                raise SystemExit(f'{a}: published x c does not give the reference MAE')
    run_perfold(C3 / 'v2_perfold_rest', C3, True)
    fa = contrasts48(C3, C3 / 'v2_perfold_rest')
    write_json(C3 / 'v2_perfold_rest.json', {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum48.json',
        'harmonised': 'addendum 60 part C3: the utilisation section only, saturation in mg/l, selections as '
                      'published (the Field-57 section is C1/v2_perfold_rest.json); ' +
                      ('every sonde hour by its own factor, every arm refitted' if HOURLY else
                       'amendment 4: sonde oxygen x c, the utilisation tree and the thermodynamic baseline '
                       'refitted, every other arm its published predictions x c'),
        'aou': fa})
    record['pass'] = True
    write_json(C3 / 'aou_forecast_reproduction.json', record)
    print('mg/l', {k: round(v, 5) for k, v in summary['mae'].items()}, f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
