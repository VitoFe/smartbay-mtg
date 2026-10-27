"""Addendum 60 parts C2 and C3: near-bottom oxygen and the utilisation ridge on the 2026 casts, in mg/l.

Both experiments read the 722-cast archive (219 per-kilogram casts, 502 per-litre) and the sondes'
per-kilogram oxygen, and their utilisation arms subtract the cast oxygen from a Garcia-Gordon
saturation in umol/kg (scripts/experiments/exp_bottom.py, exp_spatial.py) or umol/L
(scripts/v2_aou_units.py). Here every published arm is refitted with the configuration its own
study selected (src/experiments/v2_bottom_aou.py and scripts/v2_summarize_bottom.py read it back from
the published selection records instead of searching), first in the published unit, where it must
give the stored predictions, then with the casts and sondes in mg/l by their source unit and every
saturation the Garcia-Gordon per-litre form x 0.0319988. In mg/l the umol/kg and umol/L utilisation
arms of v2_aou_units are the same quantity; each keeps its own published penalty. The summaries are
recomputed with the published bootstraps (cast dates relabelled, 2000 draws, the delete-one-date
jackknife); they round to four decimals, which the mg/l outputs widen to six.

Outputs under reports/bay_twin/runs/v2_oxygen_mgl/:
  C2/v2_bottom_aou/bottom/        folds.csv, predictions.csv.gz, selections.json, contrasts.json
  C2/v2_bottom_aou/bottom_ref/    the reference arms and contrasts.json
  C3/v2_bottom_aou/aou/, aou_xgb/ the year-boundary arms, and C3/v2_bottom_aou/aou_summary.json
  C3/v2_bottom_aou/aou_units/     summary.json, predictions.csv.gz
  C2/bottom_reproduction.json

Usage (V2_THREADS=2): python -m scripts.v2_oxygen_mgl_bottom
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import builtins
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import OUT, RUNS, harmonised, lf_rewrite, sat_mgl, write_csv, write_json
from scripts.v2_oxygen_mgl_field_contrasts import same

PUB = RUNS / 'v2_bottom_aou'
C2, C3 = OUT / 'C2', OUT / 'C3'
SCRATCH = OUT / '_reproduction'
XGB_THREADS = 4


def finer(x, n=None):
    return builtins.round(x, None if n is None else n + 2)


def published_selections():
    """{(task folder, study name): selection}: the near-bottom control arms of bottom_ref were tuned
    again under the names of the bottom task, so a study is looked up in its own task's folder."""
    out = {}
    for p in PUB.glob('*/studies/*_selection.json'):
        out[(p.parent.parent.name, p.name[:-len('_selection.json')])] = json.loads(p.read_text())
    return out


class Units:
    """Install mg/l (or nothing), the published selections, and six-decimal rounding."""

    def __init__(self, mgl):
        self.mgl = mgl

    def __enter__(self):
        from src.experiments import v2_bottom_aou as vr
        from scripts import v2_summarize_aou as SA, v2_summarize_bottom as SB, v2_aou_units as U
        self.vr, self.mods = vr, (SA, SB, U)
        self.eb, self.es = vr.import_experiments(SCRATCH / 'bottom_stage')
        sels = published_selections()
        self.saved = (vr.run_study, self.eb.o2_solubility, self.es.o2_solubility, U.solubility_kg,
                      U.forecast_bound)
        vr.run_study = lambda name, objective, folder, quick=False: sels[(Path(folder).parent.name, name)]
        if self.mgl:
            def sol(T, S=38.0):
                return sat_mgl(np.asarray(T, float), np.asarray(S, float))
            self.eb.o2_solubility = sol
            self.es.o2_solubility = sol
            U.solubility_kg = sol
            U.forecast_bound = lambda: {'not_applicable': 'with oxygen and saturation both in mg/l there is no '
                                                          'umol/kg against umol/L saturation fault to bound'}
            for m in self.mods:
                m.round = finer
        self.h = harmonised(self.mgl)
        self.h.__enter__()
        return self

    def __exit__(self, *exc):
        self.h.__exit__(*exc)
        vr = self.vr
        SA, SB, U = self.mods
        vr.run_study, self.eb.o2_solubility, self.es.o2_solubility, U.solubility_kg, U.forecast_bound = self.saved
        for m in self.mods:
            if 'round' in m.__dict__:
                del m.round
        return False


def write_task(folder, res, preds, sels):
    folder.mkdir(parents=True, exist_ok=True)
    res.to_csv(folder / 'folds.csv', index=False, lineterminator='\n')
    write_csv(preds, folder / 'predictions.csv.gz')
    write_json(folder / 'selections.json', json.loads(json.dumps(sels, default=str)))


def pred_diff(a, b, keys):
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if len(a) != len(b) or not (a[keys].astype(str).to_numpy() == b[keys].astype(str).to_numpy()).all():
        return None
    return float(np.nanmax(np.abs(a[['truth', 'pred']].to_numpy(float) - b[['truth', 'pred']].to_numpy(float))))


def run(root_bottom, root_aou, mgl, record):
    """Every task in one unit. root_bottom/root_aou mirror runs/v2_bottom_aou for C2/C3 (or a scratch)."""
    with Units(mgl) as u:
        vr, eb, es = u.vr, u.eb, u.es
        SA, SB, U = u.mods
        t0 = time.time()
        # near-bottom arms
        res, preds, sels = vr.bottom(eb, root_bottom / 'bottom', False)
        write_task(root_bottom / 'bottom', res, preds, sels)
        print(f'bottom fitted ({time.time() - t0:.0f} s)', flush=True)
        # year-boundary arms
        res, preds, sels = vr.aou(es, root_aou / 'aou', False)
        write_task(root_aou / 'aou', res, preds, sels)
        # XGBoost's row subsampling depends on the thread count: the published arm ran with four
        # (src/experiments/v2_bottom_aou.py's default), and reproduces bit for bit only with four
        threads, vr.THREADS = vr.THREADS, XGB_THREADS
        try:
            res, preds, sels = vr.aou(es, root_aou / 'aou_xgb', False, (('M3b_xgboost', 'xgboost'),))
        finally:
            vr.THREADS = threads
        write_task(root_aou / 'aou_xgb', res, preds, sels)
        print(f'year-boundary arms fitted ({time.time() - t0:.0f} s)', flush=True)
        if not mgl:
            for task, keys in (('bottom', ['protocol', 'fold', 'model', 'cast_id']),
                               ('aou', ['cohort', 'space', 'model', 'cast_id']),
                               ('aou_xgb', ['cohort', 'space', 'model', 'cast_id'])):
                root = root_bottom if task == 'bottom' else root_aou
                a = pd.read_csv(root / task / 'predictions.csv.gz', dtype={'fold': str})
                b = pd.read_csv(PUB / task / 'predictions.csv.gz', dtype={'fold': str})
                d = pred_diff(a, b, keys)
                record['steps'][task] = {'max_abs_diff': d, 'pass': bool(d is not None and d <= 1e-6)}
                print('reproduce', task, d, flush=True)
        # reference arms and the near-bottom summaries
        saved = (SB.D, SB.REF)
        SB.D, SB.REF = root_bottom / 'bottom', root_bottom / 'bottom_ref'
        try:
            status = SB.rerun()
            argv = sys.argv
            sys.argv = [argv[0], '--summary-only']
            try:
                SB.main()
            finally:
                sys.argv = argv
        finally:
            SB.D, SB.REF = saved
        if not mgl:
            record['steps']['bottom_ref_control_gates'] = {'pass': bool(status['gates_passed'])}
        # year-boundary summary and the units rerun
        saved = SA.D
        SA.D = root_aou
        try:
            SA.main()
        finally:
            SA.D = saved
        saved = (U.OUT, U.PUBLISHED)
        U.OUT, U.PUBLISHED = root_aou / 'aou_units', root_aou / 'aou/predictions.csv.gz'
        try:
            U.main()
        finally:
            U.OUT, U.PUBLISHED = saved
        print(f'summaries written ({time.time() - t0:.0f} s)', flush=True)
    finish(root_bottom, root_aou)
    for p in [root_bottom / 'bottom/contrasts.json', root_bottom / 'bottom_ref/contrasts.json',
              root_bottom / 'bottom_ref/status.json', root_bottom / 'bottom_ref/selections.json',
              root_aou / 'aou_summary.json', root_aou / 'aou_units/summary.json']:
        write_json(p, json.loads(p.read_text()))
    for p in [root_bottom / 'bottom_ref/folds.csv']:
        pd.read_csv(p).to_csv(p, index=False, lineterminator='\n')
    for d in (root_bottom / 'staged_inputs', root_bottom / 'bottom_ref/staged_inputs', root_aou / 'aou_units/staged_inputs'):
        shutil.rmtree(d, ignore_errors=True)


def finish(root_bottom, root_aou):
    """The prediction files the published summarisers wrote with the host's line ending, rewritten with LF,
    and the digests that name them."""
    digest = lf_rewrite(root_bottom / 'bottom_ref/predictions.csv.gz')
    for p in (root_bottom / 'bottom_ref/status.json', root_bottom / 'bottom_ref/contrasts.json'):
        r = json.loads(p.read_text())
        (r['status'] if 'status' in r else r)['prediction_sha256'] = digest
        write_json(p, r)
    digest = lf_rewrite(root_aou / 'aou_units/predictions.csv.gz')
    r = json.loads((root_aou / 'aou_units/summary.json').read_text())
    r['prediction_sha256'] = digest
    write_json(root_aou / 'aou_units/summary.json', r)


def main():
    t0 = time.time()
    record = {'what': 'near-bottom and year-boundary arms refitted in the published unit with their published '
                      'selections, against the stored predictions and summaries, before mg/l',
              'threads': int(os.environ['V2_THREADS']), 'xgboost_threads': XGB_THREADS, 'steps': {}}
    scratch = SCRATCH / 'v2_bottom_aou'
    run(scratch, scratch, False, record)
    for name, path in (('bottom/contrasts.json', 'bottom/contrasts.json'),
                       ('bottom_ref/contrasts.json', 'bottom_ref/contrasts.json'),
                       ('aou_summary.json', 'aou_summary.json')):
        got = json.loads((scratch / path).read_text())
        pub = json.loads((PUB / path).read_text())
        if isinstance(got, dict):
            got.pop('status', None)
            pub.pop('status', None)
        diff = same(got, pub, tol=0)
        record['steps'][name] = {'pass': not diff, 'differences': diff[:20]}
        print('reproduce', name, 'ok' if not diff else diff[:4], flush=True)
    got = json.loads((scratch / 'aou_units/summary.json').read_text())
    pub = json.loads((PUB / 'aou_units/summary.json').read_text())
    keys = ('gates', 'mae', 'contrasts', 'solubility_offset_on_scored_rows', 'gates_passed')
    diff = same({k: got[k] for k in keys}, {k: pub[k] for k in keys}, tol=0)
    record['steps']['aou_units/summary.json'] = {'pass': not diff, 'differences': diff[:20]}
    print('reproduce aou_units', 'ok' if not diff else diff[:4], flush=True)
    record['pass'] = all(s['pass'] for s in record['steps'].values())
    write_json(C2 / 'bottom_reproduction.json', record)
    print(f'reproduction {"passed" if record["pass"] else "FAILED"} ({time.time() - t0:.0f} s)', flush=True)
    failed = [k for k, s in record['steps'].items() if not s['pass']]
    if failed:
        print('arms that do not reproduce:', failed, flush=True)
    # mg/l: every task is rerun; a step that failed reproduction is reported and not used
    run(C2 / 'v2_bottom_aou', C3 / 'v2_bottom_aou', True, {'steps': {}})
    record['mgl_not_to_be_used'] = failed
    write_json(C2 / 'bottom_reproduction.json', record)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
