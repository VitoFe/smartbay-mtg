"""Addendum 81, tasks A, B and F: the six-hour forecasts of every pretrained arm, gated and scored.

A  Dev-2025 at the six issuance phases (offsets 0-5) on addendum 51's panel rows
   (runs/v2_sequence_sweep/panel/). Fold-macro MAE (per-sensor mean within fold, then the mean over
   folds), in physical units (the panel and the contexts are standardised; errors x scale). The
   references are the tree of each phase, selected per outer fold (runs/v2_sequence_sweep/
   panel_perfold/, the primary estimate) with the shared selection beside it, and Chronos-Bolt
   base. Paired (fold, ISO week) block bootstrap, 600 draws, one stream per contrast seeded
   [20260921, crc32(label)] as scripts/v2_sequence_sweep_contrasts.py does. Each contrast gets the
   verdict of Table 4.1 over the six phases only (the models are deterministic, so there is no
   seed replicate); the tree reference is the one of sampler seed 42.
B  Reserved-2026, the battery's 3,451 temperature and 799 oxygen endpoints. The battery's macro
   (per-sensor MAE within quarter, then the mean over quarters); references the released model,
   persistence and the month-by-hour climatological increment. Intervals on the battery's own
   resamples (ISO weeks within quarter, 600 draws, one generator seeded 20260919, the temperature
   draws first) replayed from per-block sums, as scripts/v2_context_holes.py replays them; the
   battery's report is never written. Oxygen in the stored sonde unit and x c (mg/l).
F  Oxygen forecast as apparent utilisation on the battery's oxygen rows, in mg/l: each arm's
   utilisation forecast is turned back into oxygen at the same arm's six-hour temperature
   forecast of task B, o2_sol(T_hat, S) - AOU_hat (Garcia-Gordon per litre x 0.0319988, S = 38,
   T clipped to [-2, 40]); references the released model and the climatological increment x c
   and Chronos-Bolt base; the oxygen part of the battery's resamples. Persistence x c is reported
   beside them.

Contrasts K1-K4 of the declaration, each paired on the rows where both arms exist, with its
interval status ('excludes zero' or 'not distinguishable'), its row and unit counts, no interval
on fewer than five units, and its size read against the floor (0.029 degC paired; 0.059 mg/l).

Gates. G1: Chronos-Bolt base, Chronos-2 (uni and cov) and the clamped TimesFM 2.5 reproduce the
published columns on every row (A 1e-4 in physical units at every offset; B and F 1e-3). G2: every
arm is on exactly the published rows (keys and order), from the input the manifest records. H: the
bootstrap harness reproduces published intervals and errors of each task from published columns.
A failed or pending gate on a scored arm stops the reading of that task.

Usage (from the repository root):
  python -m scripts.v2_fm_zoo_score_6h                  # full scoring once preds/ is complete
  python -m scripts.v2_fm_zoo_score_6h --gates-only --out PATH
  python -m scripts.v2_fm_zoo_score_6h --arms chronos_bolt_small_uni,chronos_bolt_base_uni --out PATH
A partial run (gates only, a subset of arms, or preds/ not yet complete) writes only to --out.
Declared in config/bay_reserved_protocol_v2_addendum81.json and its amendments 1-3.
"""
import argparse
import json
import sys
import time
import warnings
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import v2_block_bootstrap as bb
from scripts import v2_fm_zoo_io as io
from scripts.v2_chronos_lora_contrasts import battery_frame
from scripts.v2_sequence_sweep_contrasts import label as sweep_label
from scripts.v2_sequence_table import SCALE
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l
from src.experiments.sweep_verdict import label as verdict_label
from src.experiments.sweep_verdict import verdict as sweep_verdict
from src.units import MG_PER_UMOL_L, factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
ZOO = RUNS / 'v2_fm_zoo'
INPUTS = ZOO / 'inputs'
PREDS = ZOO / 'preds'
CLAMPED_TFM25 = ZOO / 'preds_superseded_tfm25_clamped'
RERUN_CHECK = PREDS / 'rerun_check_a3.json'
OUT = ZOO / 'score_6h.json'
SWEEP = RUNS / 'v2_sequence_sweep'
MGL_COLUMNS = RUNS / 'v2_oxygen_mgl/C3/v2_aou_units_chronos/zeroshot_mgl_columns.csv.gz'
MGL_REPORT = RUNS / 'v2_oxygen_mgl/C3/v2_aou_units_chronos.json'
BATTERY_REPORT = RUNS / 'v2_reserved_reading_v2/battery_full_v2.json'
SWEEP_REPORT = {'shared': RUNS / 'v2_sequence_sweep.json', 'per_fold': RUNS / 'v2_sequence_sweep_a69.json'}
DECLARATION = ['config/bay_reserved_protocol_v2_addendum81.json'] + [
    f'config/bay_reserved_protocol_v2_addendum81_amendment{k}.json' for k in (1, 2, 3)]

EXPECTED_PREDICTIONS = 1128          # 141 input files x 8 arms
ARMS = io.ARMS
CHRONOS_ARMS = ARMS[:4]
TIMESFM_ARMS = ARMS[4:]
STEP = 5                             # the six-hour step of a six-step forecast
DRAWS = 600
OFFSETS = tuple(range(6))
VARIABLES = ('temperature', 'oxygen')
TOL = {'A': 1e-4, 'B': 1e-3, 'F': 1e-3}
HARDWARE_BOUND = 1e-3
FLOOR = {'temperature': 0.029, 'oxygen_mgl': 0.059}
MIN_UNITS = 5
DIGITS = 6
REFERENCES = {'A': ('tree', 'tree_shared', 'chronos_bolt_base_uni'),
              'B': ('MODEL_v2', 'B0_persistence', 'B1_clim_increment'),
              'F': ('MODEL_v2', 'B1_clim_increment', 'chronos_bolt_base_uni')}
REFERENCE_NAMES = {'tree': 'tree of the phase, selected per outer fold (primary)',
                   'tree_shared': 'tree of the phase, configuration shared by both folds',
                   'chronos_bolt_base_uni': 'Chronos-Bolt base, univariate (this run)',
                   'MODEL_v2': 'the released model', 'B0_persistence': 'persistence',
                   'B1_clim_increment': 'month-by-hour climatological increment'}


# ---------------------------------------------------------------------------------------------
# small helpers

def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), DIGITS)


def sig(x):
    """A gate difference with six significant figures."""
    return None if x is None or not np.isfinite(x) else float(f'{float(x):.6g}')


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def ns(series):
    t = pd.to_datetime(series, utc=True)
    return t.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]').to_numpy().astype('int64')


def input_keys(name):
    z = np.load(INPUTS / name, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    keys = pd.DataFrame({k[4:]: z[k] for k in z.files if k.startswith('key_')})
    return meta, keys


def clamp_setting(meta):
    a = meta.get('adapter', {})
    if 'forecast_config' in a:
        return a['forecast_config'].get('infer_is_positive')
    if 'flags' in a:
        return a['flags'].get('make_positive')
    return None


def batch_one(meta):
    a = meta.get('adapter', {})
    rows = a.get('rows_per_call') or []
    largest = rows.get('max') if isinstance(rows, dict) else (max(rows) if rows else None)
    return bool(a.get('per_core_batch_size') == 1 or largest == 1)


def load_arm(stem, arm, n, manifest_sha, folder=PREDS):
    """The six-hour median of one arm on one input file, and what G2 needs to know about it."""
    p = folder / f'{stem}__{arm}.npz'
    if not p.exists():
        return None, {'file': rel(p), 'present': False, 'passed': False}
    z = io.load_predictions(p)
    med, meta = z['median'], z['meta']
    a = meta.get('adapter', {})
    info = {'file': rel(p), 'present': True, 'shape': list(med.shape),
            'input_sha256_matches_manifest': meta.get('input_sha256') == manifest_sha,
            'amendments': [Path(x).stem.rsplit('_', 1)[-1] for x in meta.get('amendments', [])],
            'created_utc': meta.get('created_utc')}
    ok_shape = med.ndim == 2 and med.shape[0] == n and med.shape[1] > STEP
    values = med[:, STEP].astype(np.float64) if ok_shape else None
    info['missing_at_6h'] = int((~np.isfinite(values)).sum()) if ok_shape else None
    if arm.startswith('timesfm'):
        info['positivity_clamp_on'] = clamp_setting(meta)
        info['positivity_clamp_record'] = a.get('positivity_clamp')
        info['one_series_per_call'] = batch_one(meta)
    info['passed'] = bool(ok_shape and info['input_sha256_matches_manifest'])
    return values, info


def g1(new, published, tol, scale=1.0):
    new = np.asarray(new, float)
    published = np.asarray(published, float)
    pub_ok = np.isfinite(published)
    both = pub_ok & np.isfinite(new)
    worst = float(np.max(np.abs(new[both] - published[both])) * scale) if both.any() else float('nan')
    return {'published_rows': int(pub_ok.sum()), 'compared': int(both.sum()),
            'new_missing_where_published': int((pub_ok & ~np.isfinite(new)).sum()),
            'max_abs_diff': sig(worst), 'tolerance': tol,
            'passed': bool(pub_ok.any() and both.sum() == pub_ok.sum() and worst <= tol),
            'above_tolerance_below_1e-3': bool(np.isfinite(worst) and tol < worst <= HARDWARE_BOUND)}


def contrast_set(task, arms):
    """(a, b) -> the declared contrast families it belongs to, for the arms being scored."""
    out = {}

    def add(a, b, k):
        if a != b and a in arms and (b in arms or b not in ARMS):
            out.setdefault((a, b), []).append(k)
    for arm in arms:
        for r in REFERENCES[task]:
            add(arm, r, 'K1')
        add(arm, 'chronos_bolt_base_uni', 'K2')
    for m in ('chronos2', 'timesfm25', 'timesfm3'):
        add(f'{m}_cov', f'{m}_uni', 'K3')
    add('timesfm3_uni', 'timesfm25_uni', 'K4')
    add('chronos2_uni', 'chronos_bolt_base_uni', 'K4')
    return out


def status(lo, hi):
    if lo is None:
        return 'n.e. (fewer than five units)'
    return 'excludes zero' if (lo > 0 or hi < 0) else 'not distinguishable'


def floor_flags(p, lo, hi, floor):
    return {'floor': floor, 'abs_point_at_or_beyond_floor': bool(abs(p) >= floor),
            'interval_wholly_beyond_floor': bool(lo is not None and (lo >= floor or hi <= -floor))}


def entry(ks, p, lo, hi, n_rows, n_units, variable, unit, oxygen_factor=None):
    """One contrast as stored: [point, lo, hi] in its unit (and x c in mg/l for stored oxygen)."""
    if n_units < MIN_UNITS:
        lo = hi = None
    e = {'K': ks, 'rows': int(n_rows), 'units': int(n_units), 'unit': unit,
         'value': [r6(p), r6(lo), r6(hi)], 'status': status(lo, hi)}
    if variable == 'temperature':
        e['floor_check'] = floor_flags(p, lo, hi, FLOOR['temperature'])
    else:
        k = 1.0 if oxygen_factor is None else oxygen_factor
        pm, lm, hm = p * k, (None if lo is None else lo * k), (None if hi is None else hi * k)
        if oxygen_factor is not None:
            e['value_mgl'] = [r6(pm), r6(lm), r6(hm)]
        e['floor_check'] = floor_flags(pm, lm, hm, FLOOR['oxygen_mgl'])
    return e


# ---------------------------------------------------------------------------------------------
# the battery's resamples, replayed from per-block sums

class BatteryDraws:
    """scripts/v2_battery.py's 600 resamples: one generator seeded 20260919, the temperature draws
    first, each draw one rng.integers(0, B, B) over the variable's (quarter, week) blocks in their
    first-appearance order. A block drawn k times weighs k, which is what concatenating it k times
    and taking the battery macro does."""

    def __init__(self, d):
        rng = np.random.default_rng(20260919)
        self.blocks, self.weights = {}, {}
        for v in VARIABLES:
            s0 = d[d.variable == v]
            blocks = [(q, int(w)) for q, w in s0[['quarter', 'week']].drop_duplicates().values.tolist()]
            W = np.zeros((DRAWS, len(blocks)))
            for i in range(DRAWS):
                W[i] = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks))
            self.blocks[v], self.weights[v] = blocks, W

    def macro(self, frame, variable, cols):
        """{col: (point, draws)} of the battery macro over the rows where every col is finite, and
        the numbers of rows and of blocks those rows fall in."""
        f = frame.dropna(subset=list(cols))
        blocks = self.blocks[variable]
        bindex = {b: i for i, b in enumerate(blocks)}
        bi = np.array([bindex[(q, int(w))] for q, w in zip(f.quarter, f.week)])
        sensors = {s: i for i, s in enumerate(sorted(f.sensor.unique()))}
        si = np.array([sensors[s] for s in f.sensor])
        quarters = sorted({b[0] for b in blocks})
        qmask = np.array([[b[0] == q for b in blocks] for q in quarters], float)
        B, S = len(blocks), len(sensors)
        cnt = np.zeros((B, S))
        np.add.at(cnt, (bi, si), 1.0)
        W = np.vstack([np.ones(B), self.weights[variable]])
        n = np.einsum('db,qb,bs->dqs', W, qmask, cnt)
        present = (n > 0).any(axis=2)
        out = {}
        for c in cols:
            e = (f.truth - f[c]).abs().to_numpy(float)
            m = np.zeros((B, S))
            np.add.at(m, (bi, si), e)
            s = np.einsum('db,qb,bs->dqs', W, qmask, m)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN slices: absent cells
                per_q = np.where(present, np.nanmean(mean, axis=2), np.nan)
                v = np.nanmean(per_q, axis=1)
            out[c] = (float(v[0]), v[1:])
        return out, len(f), int(len(np.unique(bi)))

    def contrast(self, frame, variable, a, b):
        m, n_rows, n_units = self.macro(frame, variable, [a, b])
        p = m[a][0] - m[b][0]
        lo, hi = np.percentile(m[a][1] - m[b][1], [2.5, 97.5])
        return p, float(lo), float(hi), n_rows, n_units


def battery_point(frame, col):
    s = frame.dropna(subset=[col])
    return float(s.assign(e=(s[col] - s.truth).abs()).groupby(['quarter', 'sensor']).e.mean()
                 .groupby('quarter').mean().mean())


def fold_point(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f[col] - f.truth).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def a_contrast(frame, a, b, label):
    """v2_sequence_table.paired on the rows where both exist: fold macro, (fold, ISO week) blocks
    of the paired frame in first-appearance order, 600 draws, generator [20260921, crc32(label)]."""
    f = frame.dropna(subset=[a, b])
    blocks = bb.block_list(f)
    rng = np.random.default_rng([20260921, zlib.crc32(label.encode())])
    point, samples = bb.draws(f, [a, b], blocks, rng, DRAWS)
    p, lo, hi = bb.interval(point, samples, [(1, a), (-1, b)])
    return p, lo, hi, len(f), len(blocks)


def o2_sol_mgl(T, S):
    return o2sol_garcia_gordon_umol_per_l(np.clip(np.asarray(T, float), -2, 40), S) * MG_PER_UMOL_L


# ---------------------------------------------------------------------------------------------
# task A

def task_a(arms, manifest, c, score):
    out = {'gates': {'G1': {}, 'G2': {}, 'harness': {}}, 'timesfm25_clamp': {}}
    sweep = {k: json.loads(p.read_text()) for k, p in SWEEP_REPORT.items()}
    raw = {}                                   # variable -> (a, b) -> [six (p, lo, hi)] physical
    for variable in VARIABLES:
        scale = SCALE[variable]
        res_v = {'offsets': {}}
        raw[variable] = {}
        for o in OFFSETS:
            stem = f'A_{variable}_offset{o}'
            name = f'{stem}.npz'
            meta, keys = input_keys(name)
            mf = manifest['files'][name]
            pan = pd.read_csv(SWEEP / 'panel' / f'panel_{variable}_offset{o}.csv.gz')
            per = pd.read_csv(SWEEP / 'panel_perfold' / f'panel_{variable}_offset{o}.csv.gz')
            n = len(pan)
            st = meta.get('standardisation') or {}
            g2 = {'input_sha256_matches_manifest': io.digest(INPUTS / name) == mf['sha256'],
                  'rows': n, 'input_rows': int(len(keys)),
                  'keys_and_order_equal_panel': bool(
                      len(keys) == n and (keys.fold.to_numpy() == pan.fold.to_numpy()).all()
                      and (keys.sensor.to_numpy() == pan.sensor.to_numpy()).all()
                      and (keys.time.to_numpy() == ns(pan.time)).all()),
                  'per_fold_panel_rows_equal': bool(
                      len(per) == n and (per[['fold', 'sensor', 'time']].values == pan[['fold', 'sensor', 'time']].values).all()),
                  'scale_equals_published': st.get('scale') == SCALE[variable], 'arms': {}}
            frame = pan[['fold', 'sensor', 'time', 'quarter', 'truth', 'persistence', 'clim_hour',
                         'chronos_zs', 'timesfm_zs', 'chronos2_zs']].copy()
            frame['tree'] = per.tree_seed42.to_numpy()
            frame['tree_shared'] = pan.tree_seed42.to_numpy()
            for arm in ARMS:
                vals, info = load_arm(stem, arm, n, mf['sha256'])
                g2['arms'][arm] = info
                if vals is not None:
                    frame[arm] = vals
                elif arm in arms:
                    frame[arm] = np.nan
            g2['passed'] = bool(g2['input_sha256_matches_manifest'] and g2['keys_and_order_equal_panel']
                                and g2['per_fold_panel_rows_equal'] and g2['scale_equals_published']
                                and all(g2['arms'][a]['passed'] for a in arms))
            out['gates']['G2'][f'{variable}|offset{o}'] = g2
            # G1: Chronos-Bolt base and Chronos-2 (univariate) against the panel's columns
            g = {}
            for arm, col in (('chronos_bolt_base_uni', 'chronos_zs'), ('chronos2_uni', 'chronos2_zs')):
                if arm in frame:
                    g[arm] = {'published_column': col, **g1(frame[arm], frame[col], TOL['A'], scale)}
            # TimesFM 2.5: the published column ran with the library's positivity clamp on
            clamped, src = tfm25_clamped(stem, n, mf['sha256'])
            if clamped is None:
                g['timesfm25_uni_clamped'] = {'published_column': 'timesfm_zs', 'status': 'pending', **src}
            else:
                g['timesfm25_uni_clamped'] = {'published_column': 'timesfm_zs', **src,
                                              **g1(clamped, frame.timesfm_zs, TOL['A'], scale)}
                frame['timesfm25_uni_clamped'] = clamped
            if 'timesfm25_uni' in frame and g2['arms']['timesfm25_uni'].get('positivity_clamp_on') is False:
                d = g1(frame.timesfm25_uni, frame.timesfm_zs, TOL['A'], scale)
                g['timesfm25_uni_declared_unclamped_vs_published'] = {
                    'note': 'information only: the declared output has the clamp off',
                    'max_abs_diff': d['max_abs_diff'],
                    'rows_differing_from_published_by_more_than_1e-4': int(
                        (np.abs(frame.timesfm25_uni - frame.timesfm_zs) * scale > TOL['A']).sum())}
            out['gates']['G1'][f'{variable}|offset{o}'] = g
            # harness: the published intervals and errors of this offset from the published columns
            out['gates']['harness'][f'{variable}|offset{o}'] = a_harness(frame, variable, o, scale, sweep)
            # item: TimesFM 2.5 with and without the clamp
            out['timesfm25_clamp'][f'{variable}|offset{o}'] = clamp_effect(frame, g2, variable, scale, c)
            if not score:
                continue
            e_o = {'rows': int(n), 'mae': {}, 'mae_mgl': {} if variable == 'oxygen' else None, 'contrasts': {}}
            for col in list(arms) + ['tree', 'tree_shared', 'persistence', 'clim_hour',
                                     'chronos_zs', 'timesfm_zs', 'chronos2_zs']:
                m = fold_point(frame, col) * scale
                e_o['mae'][col] = r6(m)
                if variable == 'oxygen':
                    e_o['mae_mgl'][col] = r6(m * c)
            if variable == 'temperature':
                e_o.pop('mae_mgl')
            # amendment 2: the positivity-clamp record beside every TimesFM result of task A
            e_o['timesfm_positivity_clamp'] = {
                arm: {'clamp_on': g2['arms'][arm].get('positivity_clamp_on'),
                      'record': g2['arms'][arm].get('positivity_clamp_record')}
                for arm in arms if arm.startswith('timesfm')}
            for (a, b), ks in contrast_set('A', arms).items():
                lab = f'a81|A|{variable}|offset{o}|{a}|{b}'
                p, lo, hi, n_rows, n_units = a_contrast(frame, a, b, lab)
                p, lo, hi = p * scale, lo * scale, hi * scale
                raw[variable].setdefault((a, b), []).append((p, lo, hi) if n_units >= MIN_UNITS else (p, None, None))
                e_o['contrasts'][f'{a} - {b}'] = entry(
                    ks, p, lo, hi, n_rows, n_units, variable,
                    'degC' if variable == 'temperature' else 'umol/kg (stored sonde unit)',
                    None if variable == 'temperature' else c)
                e_o['contrasts'][f'{a} - {b}']['stream'] = lab
            res_v['offsets'][str(o)] = e_o
        if score:
            res_v['verdicts'] = {f'{a} - {b}': phase_verdict(cells, variable, c)
                                 for (a, b), cells in raw[variable].items()}
            out[variable] = res_v
    return out


def tfm25_clamped(stem, n, sha):
    """The clamped TimesFM 2.5 univariate output on a task-A file: the superseded file once it has
    been moved out of preds/, else the file still in preds/ while it carries the clamp."""
    for folder, where in ((CLAMPED_TFM25, 'superseded folder'), (PREDS, 'preds/ (not yet superseded)')):
        p = folder / f'{stem}__timesfm25_uni.npz'
        if not p.exists():
            continue
        z = io.load_predictions(p)
        if clamp_setting(z['meta']) is not True:
            continue
        vals, info = load_arm(stem, 'timesfm25_uni', n, sha, folder)
        if vals is None or not info['passed']:
            continue
        return vals, {'source': where, 'file': info['file'],
                      'positivity_clamp_record': info.get('positivity_clamp_record')}
    return None, {'source': None, 'file': rel(CLAMPED_TFM25 / f'{stem}__timesfm25_uni.npz')}


def clamp_effect(frame, g2, variable, scale, c):
    """The fold-macro MAE of TimesFM 2.5 univariate with the clamp (as published) and without it
    (as declared for task A by amendment 3)."""
    out = {'published_column_timesfm_zs': r6(fold_point(frame, 'timesfm_zs') * scale)}
    if 'timesfm25_uni_clamped' in frame:
        out['clamped'] = r6(fold_point(frame, 'timesfm25_uni_clamped') * scale)
    info = g2['arms'].get('timesfm25_uni', {})
    if info.get('present') and info.get('positivity_clamp_on') is False and 'timesfm25_uni' in frame:
        out['unclamped_declared'] = r6(fold_point(frame, 'timesfm25_uni') * scale)
        if 'timesfm25_uni_clamped' in frame:
            out['unclamped_minus_clamped'] = r6(
                (fold_point(frame, 'timesfm25_uni') - fold_point(frame, 'timesfm25_uni_clamped')) * scale)
            out['rows_where_outputs_differ'] = int(
                (np.abs(frame.timesfm25_uni - frame.timesfm25_uni_clamped) > 0).sum())
    else:
        out['unclamped_declared'] = 'pending: preds/ does not yet hold the declared (unclamped) output'
    if variable == 'oxygen':
        out['mgl'] = {k: (r6(v * c) if isinstance(v, float) else v) for k, v in out.items()
                      if k not in ('rows_where_outputs_differ',)}
    out['unit'] = 'degC' if variable == 'temperature' else 'umol/kg (stored sonde unit); mgl = x c'
    return out


def a_harness(frame, variable, o, scale, sweep):
    """Three published contrasts and the published errors of this offset (sampler seed 42), from
    the published columns, on the published streams; compared after the published rounding."""
    cell = f'offset{o}|seed42'
    diffs, checked = {}, 0
    for design, tree_col in (('shared', 'tree_shared'), ('per_fold', 'tree')):
        cal = sweep[design]['variables'][variable]['designs']['calendar']
        f = frame.assign(tree=frame[tree_col])
        for a, b in (('tree', 'chronos_zs'), ('chronos_zs', 'timesfm_zs'), ('chronos2_zs', 'chronos_zs')):
            if design == 'per_fold' and 'tree' not in (a, b):
                continue
            pub = cal['contrasts'][f'{a} - {b}']['by_cell'].get(cell)
            if pub is None:
                continue
            p, lo, hi, _, _ = a_contrast(f, a, b, sweep_label(variable, o, 42, a, b))
            mine = [round(x * scale, 4) for x in (p, lo, hi)]
            diffs[f'{design}: {a} - {b}'] = max(abs(x - y) for x, y in zip(mine, pub))
            checked += 1
        for col in ('tree', 'chronos_zs', 'timesfm_zs', 'chronos2_zs', 'persistence', 'clim_hour'):
            want = cal['mae'][cell].get(col)
            if want is not None:
                diffs[f'{design}: mae {col}'] = abs(round(fold_point(f, col) * scale, 4) - want)
                checked += 1
    worst = max(diffs.values()) if diffs else float('nan')
    return {'checked': checked, 'max_abs_diff_after_rounding_to_4': sig(worst),
            'passed': bool(diffs and worst <= 1e-9)}


def phase_verdict(cells, variable, c):
    if len(cells) < len(OFFSETS) or any(lo is None for _, lo, _ in cells):
        return {'verdict': 'incomplete', 'cells': len(cells)}
    code = sweep_verdict([cells[0]], cells)
    points = [p for p, _, _ in cells]
    mean = float(np.mean(points))
    out = {'verdict_code': code, 'verdict': verdict_label(code), 'phase_mean': r6(mean),
           'phase_range': r6(max(points) - min(points)),
           'phases_excluding_zero': int(sum(lo > 0 or hi < 0 for _, lo, hi in cells)),
           'by_phase': [[r6(p), r6(lo), r6(hi)] for p, lo, hi in cells]}
    if variable == 'oxygen':
        out['phase_mean_mgl'] = r6(mean * c)
        out['phase_range_mgl'] = r6((max(points) - min(points)) * c)
        out['floor_check_phase_mean'] = floor_flags(mean * c, None, None, FLOOR['oxygen_mgl'])
    else:
        out['floor_check_phase_mean'] = floor_flags(mean, None, None, FLOOR['temperature'])
    return out


# ---------------------------------------------------------------------------------------------
# tasks B and F

def task_bf(arms, manifest, c, score):
    d = battery_frame()
    d['tkey'] = ns(d.time)
    draws = BatteryDraws(d)
    out = {'B': {'gates': {'G1': {}, 'G2': {}, 'harness': {}}}, 'F': {'gates': {'G1': {}, 'G2': {}, 'harness': {}}}}
    published = json.loads(BATTERY_REPORT.read_text())
    temp_hat = {}                             # arm -> six-hour temperature forecast, task B rows
    for variable in VARIABLES:
        stem = f'B_{variable}_Reserved-2026'
        name = f'{stem}.npz'
        meta, keys = input_keys(name)
        mf = manifest['files'][name]
        s0 = d[d.variable == variable].reset_index(drop=True)
        n = len(s0)
        g2 = {'input_sha256_matches_manifest': io.digest(INPUTS / name) == mf['sha256'],
              'rows': n, 'input_rows': int(len(keys)),
              'keys_and_order_equal_battery': bool(
                  len(keys) == n and (keys.variable.to_numpy() == s0.variable.to_numpy()).all()
                  and (keys.sensor.to_numpy() == s0.sensor.to_numpy()).all()
                  and (keys.time.to_numpy() == s0.tkey.to_numpy()).all()), 'arms': {}}
        for arm in ARMS:
            vals, info = load_arm(stem, arm, n, mf['sha256'])
            g2['arms'][arm] = info
            if vals is not None:
                s0[arm] = vals
                if variable == 'temperature':
                    temp_hat[arm] = vals
            elif arm in arms:
                s0[arm] = np.nan
        g2['passed'] = bool(g2['input_sha256_matches_manifest'] and g2['keys_and_order_equal_battery']
                            and all(g2['arms'][a]['passed'] for a in arms))
        out['B']['gates']['G2'][variable] = g2
        g = {}
        for arm, col in (('chronos_bolt_base_uni', f'{variable}|ZS_raw'),
                         ('chronos2_uni', f'CH2_{variable}_raw_uni'), ('chronos2_cov', f'CH2_{variable}_raw_cov')):
            if arm in s0:
                g[arm] = {'published_column': col, **g1(s0[arm], s0[col], TOL['B'])}
        out['B']['gates']['G1'][variable] = g
        out['B']['gates']['harness'][variable] = b_harness(s0, variable, draws, published[variable])
        if score:
            out['B'][variable] = score_battery(s0, variable, draws, arms, 'B', c,
                                               'degC' if variable == 'temperature' else 'umol/kg (stored sonde unit)',
                                               record=('CHRONOS_bolt_small', 'CHRONOS_bolt_base', f'{variable}|ZS_raw',
                                                       f'CH2_{variable}_raw_uni', f'CH2_{variable}_raw_cov'))
    # F
    name = 'F_oxygen_aou_Reserved-2026.npz'
    stem = name[:-4]
    meta, keys = input_keys(name)
    mf = manifest['files'][name]
    rec = np.load(INPUTS / 'F_oxygen_aou_Reserved-2026_reconstruction.npz', allow_pickle=False)
    s0 = d[d.variable == 'oxygen'].reset_index(drop=True)
    for col in ('truth', 'MODEL_v2', 'B0_persistence', 'B1_clim_increment'):
        s0[col] = s0[col].to_numpy(float) * c
    n = len(s0)
    _, bkeys = input_keys('B_temperature_Reserved-2026.npz')
    trow = rec['temperature_row']
    g2 = {'input_sha256_matches_manifest': io.digest(INPUTS / name) == mf['sha256'],
          'reconstruction_sha256_matches_manifest': io.digest(INPUTS / 'F_oxygen_aou_Reserved-2026_reconstruction.npz')
          == mf['gate_references']['reconstruction_sha256'],
          'rows': n, 'input_rows': int(len(keys)),
          'keys_and_order_equal_battery_oxygen': bool(
              len(keys) == n and (keys.sensor.to_numpy() == s0.sensor.to_numpy()).all()
              and (keys.time.to_numpy() == s0.tkey.to_numpy()).all()),
          'reconstruction_keys_equal': bool((rec['key_sensor'] == keys.sensor.to_numpy()).all()
                                            and (rec['key_time'] == keys.time.to_numpy()).all()),
          'temperature_rows_same_sensor_and_issuance': bool(
              (bkeys.sensor.to_numpy()[trow] == keys.sensor.to_numpy()).all()
              and (bkeys.time.to_numpy()[trow] == keys.time.to_numpy()).all()),
          'salinity': sorted(set(np.round(rec['salinity'], 6).tolist())), 'arms': {}}
    for arm in ARMS:
        vals, info = load_arm(stem, arm, n, mf['sha256'])
        g2['arms'][arm] = info
        if vals is not None and arm in temp_hat:
            s0[arm] = o2_sol_mgl(temp_hat[arm][trow], rec['salinity']) - vals
            info['reconstructed_with'] = f'B_temperature_Reserved-2026__{arm}'
        elif arm in arms:
            s0[arm] = np.nan
            info['passed'] = False
            info['reconstructed_with'] = None
    g2['passed'] = bool(g2['input_sha256_matches_manifest'] and g2['reconstruction_sha256_matches_manifest']
                        and g2['keys_and_order_equal_battery_oxygen'] and g2['reconstruction_keys_equal']
                        and g2['temperature_rows_same_sensor_and_issuance']
                        and all(g2['arms'][a]['passed'] for a in arms))
    out['F']['gates']['G2']['oxygen'] = g2
    z = pd.read_csv(MGL_COLUMNS)
    z = z[z.variable == 'oxygen'].assign(tkey=lambda x: ns(x.time))
    cols = {'chronos_bolt_base_uni': 'oxygen|ZS_aou|mgl', 'chronos2_uni': 'CH2_oxygen_aou_uni|mgl',
            'chronos2_cov': 'CH2_oxygen_aou_cov|mgl'}
    s0 = s0.merge(z[['sensor', 'tkey'] + list(cols.values())], on=['sensor', 'tkey'], how='left', validate='one_to_one')
    g = {}
    for arm, col in cols.items():
        if arm in s0:
            g[arm] = {'published_column': col, **g1(s0[arm], s0[col], TOL['F'])}
    out['F']['gates']['G1']['oxygen'] = g
    out['F']['gates']['harness']['oxygen'] = f_harness(s0, draws, cols)
    if score:
        out['F']['oxygen'] = score_battery(s0, 'oxygen', draws, arms, 'F', None, 'mg/l',
                                           record=tuple(cols.values()))
    return out


def score_battery(s0, variable, draws, arms, task, c, unit, record=()):
    oxygen_stored = variable == 'oxygen' and c is not None
    e = {'rows': int(len(s0)), 'mae': {}, 'contrasts': {}}
    if oxygen_stored:
        e['mae_mgl'] = {}
    for col in list(arms) + ['MODEL_v2', 'B0_persistence', 'B1_clim_increment'] + list(record):
        if col in s0 and s0[col].notna().any():
            m = battery_point(s0, col)
            e['mae'][col] = r6(m)
            if oxygen_stored:
                e['mae_mgl'][col] = r6(m * c)
    for (a, b), ks in contrast_set(task, arms).items():
        p, lo, hi, n_rows, n_units = draws.contrast(s0, variable, a, b)
        e['contrasts'][f'{a} - {b}'] = entry(ks, p, lo, hi, n_rows, n_units, variable, unit,
                                             c if oxygen_stored else None)
    e['unit'] = unit + ('; mae_mgl and value_mgl = x c' if oxygen_stored else '')
    return e


def b_harness(s0, variable, draws, pub):
    """battery_full_v2.json's own numbers from the published columns on the replayed resamples."""
    diffs = {}
    for m, r in pub['methods'].items():
        if m in s0:
            p, lo, hi, _, _ = draws.contrast(s0, variable, m, 'MODEL_v2')
            diffs[f'{m} - MODEL_v2'] = max(abs(p - r['minus_v2']), abs(lo - r['ci'][0]), abs(hi - r['ci'][1]),
                                           abs(battery_point(s0, m) - r['mae']))
    for k, v in pub['pairs'].items():
        x, y = [t.strip() for t in k.split(' - ')]
        if x in s0 and y in s0:
            diffs[k] = max(abs(a - b) for a, b in zip(draws.contrast(s0, variable, x, y)[:3], v))
    worst = max(diffs.values())
    return {'checked': len(diffs), 'max_abs_diff': sig(worst), 'passed': bool(worst <= 1e-9)}


def f_harness(s0, draws, cols):
    """runs/v2_oxygen_mgl/C3/v2_aou_units_chronos.json from the published mg/l columns."""
    pub = json.loads(MGL_REPORT.read_text())
    name = {'bolt_aou': cols['chronos_bolt_base_uni'], 'ch2_aou_uni': cols['chronos2_uni'],
            'ch2_aou_cov': cols['chronos2_cov']}
    diffs = {}
    for k, v in pub['contrasts'].items():
        x, y = [t.strip() for t in k.split(' - ')]
        xa, ya = x.split('@')[0], y.split('@')[0]
        if not x.endswith('@mgl') or xa not in name:
            continue
        cy = name.get(ya) if y.endswith('@mgl') else (y if y in ('MODEL_v2', 'B1_clim_increment', 'B0_persistence') else None)
        if cy is None:
            continue
        diffs[k] = max(abs(a - b) for a, b in zip(draws.contrast(s0, 'oxygen', name[xa], cy)[:3], v))
    for r in ('MODEL_v2', 'B0_persistence', 'B1_clim_increment'):
        diffs[f'mae {r}'] = abs(battery_point(s0, r) - pub['references_mae'][r])
    worst = max(diffs.values())
    return {'checked': len(diffs), 'max_abs_diff': sig(worst), 'passed': bool(worst <= 1e-9)}


# ---------------------------------------------------------------------------------------------

def blocking(task_gates, arms, task):
    """The gates that stop a task's reading: G2 rows, the harness, and G1 of every scored arm (the
    clamped TimesFM 2.5 check stands for timesfm25_uni)."""
    bad = []
    for k, g in task_gates['G2'].items():
        if not g['passed']:
            bad.append(f'G2 {k}')
    for k, g in task_gates['harness'].items():
        if not g['passed']:
            bad.append(f'harness {k}')
    for k, per in task_gates['G1'].items():
        for arm, g in per.items():
            base = 'timesfm25_uni' if arm == 'timesfm25_uni_clamped' else arm
            if base in arms and 'passed' in g and not g['passed']:
                bad.append(f'G1 {k} {arm}')
            if base in arms and g.get('status') == 'pending':
                bad.append(f'G1 {k} {arm} pending')
    return bad


def declared_configuration(gates, arms):
    """Amendment 3: TimesFM arms of tasks A and F with both clamps off; TimesFM 3.0 one series per call."""
    issues = []
    for task in ('A', 'B', 'F'):
        for k, g in gates[task]['G2'].items():
            for arm, info in g['arms'].items():
                if arm not in arms or not arm.startswith('timesfm') or not info.get('present'):
                    continue
                if task in ('A', 'F') and info.get('positivity_clamp_on') is not False:
                    issues.append(f'{task} {k} {arm}: positivity clamp {info.get("positivity_clamp_on")}')
                if task == 'B' and info.get('positivity_clamp_on') is not True:
                    issues.append(f'{task} {k} {arm}: positivity clamp {info.get("positivity_clamp_on")}')
                if arm.startswith('timesfm3') and not info.get('one_series_per_call'):
                    issues.append(f'{task} {k} {arm}: not one series per call')
    return {'issues': issues, 'passed': not issues}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--gates-only', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    arms = tuple(a for a in ARMS if a in args.arms.split(','))
    unknown = set(args.arms.split(',')) - set(ARMS)
    if unknown:
        raise SystemExit(f'unknown arms: {sorted(unknown)}')
    t0 = time.time()
    n_preds = len(list(PREDS.glob('*.npz')))
    ready = n_preds == EXPECTED_PREDICTIONS and RERUN_CHECK.exists()
    full = ready and arms == ARMS and not args.gates_only
    score = not args.gates_only and (ready or arms != ARMS)
    if not ready and arms == ARMS and not args.gates_only:
        print(f'preds/ holds {n_preds} of {EXPECTED_PREDICTIONS} prediction files'
              f'{"" if RERUN_CHECK.exists() else " and rerun_check_a3.json is absent"}: gates only')
    manifest = json.loads((INPUTS / 'manifest_6h.json').read_text())
    c = factor()
    a = task_a(arms, manifest, c, score)
    bf = task_bf(arms, manifest, c, score)
    gates = {'A': a.pop('gates'), 'B': bf['B'].pop('gates'), 'F': bf['F'].pop('gates')}
    report = {
        'declared_in': {p: io.digest(ROOT / p) for p in DECLARATION},
        'script': 'scripts/v2_fm_zoo_score_6h.py', 'script_sha256': io.digest(Path(__file__)),
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'status': 'complete' if full else ('gates only' if not score else 'partial: arms ' + ', '.join(arms)),
        'preconditions': {'prediction_files': n_preds, 'expected': EXPECTED_PREDICTIONS,
                          'rerun_check_a3': RERUN_CHECK.exists()},
        'arms_scored': list(arms) if score else [],
        'c_mg_per_l_per_umol_kg': c,
        'floors': {'temperature_degC_paired': FLOOR['temperature'], 'oxygen_mg_per_l': FLOOR['oxygen_mgl'],
                   'reading': 'flags only: |point| at or beyond the floor, and the interval wholly beyond it'},
        'references': REFERENCE_NAMES,
        'resampling': {
            'A': '(fold, ISO week) blocks of the paired rows, 600 draws, generator [20260921, crc32(stream)], '
                 'fold macro; the verdict over the six phases by src/experiments/sweep_verdict.py (phases only)',
            'B': 'the battery resamples replayed: ISO weeks within quarter, 600 draws, one generator seeded '
                 '20260919, temperature first; quarter macro on the rows where both arms exist',
            'F': 'the oxygen part of the battery resamples (the same draws as task B oxygen)'},
        'contrast_families': {'K1': 'arm minus each reference of its task', 'K2': 'arm minus chronos_bolt_base_uni',
                              'K3': 'cov minus uni (chronos2, timesfm25, timesfm3)',
                              'K4': 'timesfm3_uni minus timesfm25_uni; chronos2_uni minus chronos_bolt_base_uni'},
        'gates': gates,
    }
    report['gates']['declared_configuration'] = declared_configuration(gates, arms) if (score and any(
        x in arms for x in TIMESFM_ARMS)) else 'not checked: no TimesFM arm scored'
    stopped = {t: blocking(gates[t], arms, t) for t in ('A', 'B', 'F')}
    if isinstance(report['gates']['declared_configuration'], dict) and not report['gates']['declared_configuration']['passed']:
        for t in stopped:
            stopped[t].append('declared configuration (amendment 3)')
    report['stopped'] = {t: v for t, v in stopped.items() if v}
    report['A_timesfm25_with_and_without_clamp'] = a.pop('timesfm25_clamp')
    if score:
        report['A'] = a if not stopped['A'] else {'stopped': stopped['A']}
        report['B'] = bf['B'] if not stopped['B'] else {'stopped': stopped['B']}
        report['F'] = bf['F'] if not stopped['F'] else {'stopped': stopped['F']}
    report['seconds'] = round(time.time() - t0, 1)
    summarise(report)
    target = Path(args.out) if args.out else (OUT if full else None)
    if target is None:
        print('partial run without --out: nothing written')
        return
    if target.resolve() == OUT.resolve() and not full:
        raise SystemExit('a partial run never writes score_6h.json')
    target.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))
    print('wrote', target)


def summarise(r):
    print(f"status: {r['status']}   {r['seconds']} s")
    for t in ('A', 'B', 'F'):
        g = r['gates'][t]
        g1s = [(k, arm, x.get('max_abs_diff'), x.get('passed', x.get('status'))) for k, per in g['G1'].items()
               for arm, x in per.items() if 'passed' in x or x.get('status') == 'pending']
        worst = {}
        for k, arm, m, ok in g1s:
            w = worst.setdefault(arm, [0.0, True])
            if m is not None:
                w[0] = max(w[0], m)
            w[1] = w[1] and ok is True
        print(f'{t} G1 ' + '  '.join(f'{arm} {m:.2e} {"ok" if ok else "FAIL/pending"}' for arm, (m, ok) in worst.items()))
        print(f'{t} G2 ' + ('ok' if all(x['passed'] for x in g['G2'].values()) else
                            'FAIL ' + str([k for k, x in g['G2'].items() if not x['passed']])))
        hw = [x.get('max_abs_diff', x.get('max_abs_diff_after_rounding_to_4')) for x in g['harness'].values()]
        print(f'{t} harness: {sum(x["checked"] for x in g["harness"].values())} checks, max diff {max(hw):.1e}, '
              + ('ok' if all(x['passed'] for x in g['harness'].values()) else
                 'FAIL ' + str([k for k, x in g['harness'].items() if not x['passed']])))
    if r.get('stopped'):
        print('stopped:', json.dumps(r['stopped']))
    if 'A' in r and 'stopped' not in r['A']:
        for v in VARIABLES:
            print(f'== A {v}: MAE by offset (physical units)')
            for arm in r['arms_scored'] + ['tree', 'tree_shared', 'clim_hour']:
                print(f'   {arm:24s} ' + ' '.join(f"{r['A'][v]['offsets'][str(o)]['mae'][arm]:.4f}" for o in OFFSETS))
            for k, x in r['A'][v]['verdicts'].items():
                if x.get('verdict_code'):
                    print(f"   {k:46s} mean {x['phase_mean']:+.4f}  {x['phases_excluding_zero']}/6  {x['verdict']}")
    for t in ('B', 'F'):
        if t in r and 'stopped' not in r[t]:
            for v, e in r[t].items():
                print(f'== {t} {v} ({e["unit"]})')
                for k, m in e['mae'].items():
                    print(f'   {k:28s} {m:.4f}' + (f"   {e['mae_mgl'][k]:.4f} mg/l" if 'mae_mgl' in e else ''))
                for k, x in e['contrasts'].items():
                    p, lo, hi = x['value']
                    print(f"   {k:46s} {p:+.4f} [{lo:+.4f}, {hi:+.4f}] {x['status']}")


if __name__ == '__main__':
    sys.exit(main())
