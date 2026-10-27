"""Addendum 82 part F: one causal recalibration rule for every forecaster of the distribution.

The rule is addendum 70's conformalized quantile regression on a causal ledger (src/experiments/
v2_ensemble.conformal, non-adaptive): per variable and lead, probes pooled, the conformity scores of rows
whose target hour is at or before the issuance only, Dev-2025 first and the 2026 ledger continued from it,
the raw interval kept below 30 scores. It is applied independently to each central pair of the nine
levels, (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), at nominal coverage 0.8, 0.6, 0.4 and 0.2, each
with its own ledger of s_k = max(q_tau - y, y - q_(1-tau)) and correction Q_k, the ceil((n + 1) nominal)-th
smallest of the n available scores. The median is not moved; each row's nine values are then sorted. At
(0.1, 0.9), before sorting, the interval is addendum 70's cqr by construction (gate F-G2).

Rows: addendum 81 task C's rows, assembled by scripts/v2_fm_zoo_score_dist_bridge.c_frame (whose task C
gates are rerun), read on the common rows of each (variable, cohort, lead): every recalibrated seed-42 arm
and both references present. Nothing is fitted or selected.

Reads stored quantiles only (published files and addendum 81's task C prediction files):
  python -m src.experiments.v2_recalibration [--skip-g1] [--out PATH]
Writes reports/bay_twin/runs/v2_addendum82_F.json and, under reports/bay_twin/runs/v2_addendum82/F/,
the per (sensor, fold or quarter) coverage cells and the per-cell arm table.

Declared in config/bay_reserved_protocol_v2_addendum82.json, part F.
"""
import argparse
import bisect
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v2_fdr as FDR  # noqa: E402
import scripts.v2_fm_zoo_score_dist_bridge as SDB  # noqa: E402
import scripts.v2_summary_table as S  # noqa: E402
from scripts.v2_fm_zoo_io import ARMS, digest  # noqa: E402
from src.experiments import v2_ensemble as E  # noqa: E402
from src.experiments import v2_probabilistic as P  # noqa: E402
from src.experiments import v2_probabilistic_report as R  # noqa: E402
from src.experiments.sweep_verdict import label as verdict_label  # noqa: E402
from src.experiments.sweep_verdict import verdict as sweep_verdict  # noqa: E402
from src.units import factor  # noqa: E402

DECLARATION = 'config/bay_reserved_protocol_v2_addendum82.json'
RUNS = ROOT / 'reports/bay_twin/runs'
OUT_JSON = RUNS / 'v2_addendum82_F.json'
OUT_DIR = RUNS / 'v2_addendum82' / 'F'
ENSEMBLE_REPORT = RUNS / 'v2_ensemble' / 'report.json'
SDB_REPORT = RUNS / 'v2_fm_zoo' / 'score_dist_bridge.json'

NINE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: (lower level, upper level, nominal coverage): the declared nominal values, written as literals
PAIRS = ((0.1, 0.9, 0.8), (0.2, 0.8, 0.6), (0.3, 0.7, 0.4), (0.4, 0.6, 0.2))
PAIR_INDEX = [(NINE.index(a), NINE.index(b)) for a, b, _ in PAIRS]
NOMINAL_TAG = {0: '80', 1: '60', 2: '40', 3: '20'}
MIN_SCORES = E.MIN_SCORES                         # 30, addendum 70
ALPHA80 = 0.2
MIN_CELL = 30                                     # scripts/v2_interval_calibration.MIN_CELL
MIN_UNITS = 5
SEEDS = (7, 123)
HOUR_NS = 3_600_000_000_000

NEW = tuple(ARMS)
PUB42 = ('clim_quantiles', 'quantile_tree', 'quantile_tree_weather', 'quantile_tree_tide', 'chronos2_net_weather',
         'linear_ar', 'linear_ar_tree', 'linear_ar_joint', 'arima_ss', 'auto_arima', 'nbeats', 'dlm', 'dlm_emp',
         'bsts', 'E3', 'E5')
EWMA = ('chronos2_net_ewma', 'linear_ar_ewma')
A64 = ('chronos2_a64', 'chronos2_net_a64', 'chronos2_net_tide_a64')
SEEDED_BASE = ('quantile_tree', 'quantile_tree_weather', 'quantile_tree_tide', 'linear_ar', 'linear_ar_tree',
               'linear_ar_joint', 'arima_ss', 'nbeats', 'dlm', 'dlm_emp', 'E3')
REFS = ('clim_quantiles', 'quantile_tree')
#: arms the common rows require (at the leads where each is stored)
REQUIRED = NEW + PUB42 + EWMA
EXTRA_CAPTURE = ('chronos2_net',)                 # chronos2_cov's published reproduction, reported beside it


def r6(x):
    return R.r(x)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# --------------------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------------------
def recalibrate(t_ns, lead, y, Q):
    """The nine-level rule for one (variable, lead): rows of every cohort that enter this arm's ledger.

    t_ns int64 issuance times (ns UTC), y the truth, Q [n, 9] at NINE. Returns (pre, post, recal, corr):
    the recalibrated quantiles before and after sorting, whether the row was recalibrated (30 or more
    scores available) and the correction Q_k of each pair (NaN where raw). The arithmetic of each pair is
    that of v2_ensemble.conformal: scores ordered by target time (stable), at each issuance the k scores
    with target at or before it, the ceil((k + 1) nominal)-th smallest."""
    t_ns = np.asarray(t_ns, np.int64)
    y = np.asarray(y, float)
    Q = np.asarray(Q, float)
    n = len(y)
    if not (np.isfinite(Q).all() and np.isfinite(y).all()):
        raise ValueError('non-finite quantiles or truth on the ledger rows')
    target = t_ns + np.int64(int(lead)) * HOUR_NS
    order = np.argsort(target, kind='stable')
    known = target[order]
    uniq, inv = np.unique(t_ns, return_inverse=True)
    by_time = np.split(np.argsort(inv, kind='stable'), np.cumsum(np.bincount(inv))[:-1])
    pre = Q.copy()
    recal = np.zeros(n, bool)
    corr = np.full((n, len(PAIRS)), np.nan)
    for k, ((il, ih), (_, _, level)) in enumerate(zip(PAIR_INDEX, PAIRS)):
        s = np.maximum(Q[:, il] - y, y - Q[:, ih])
        scores = s[order]
        ledger = []
        filled = 0
        for ti, rows in zip(uniq, by_time):
            kk = int(np.searchsorted(known, ti, side='right'))
            # no score enters a correction before its target hour (addendum 70 G3)
            assert kk == 0 or known[kk - 1] <= ti
            assert kk == len(known) or known[kk] > ti
            while filled < kk:
                bisect.insort(ledger, float(scores[filled]))
                filled += 1
            if kk < MIN_SCORES:
                continue
            q = ledger[min(kk - 1, max(0, math.ceil((kk + 1) * level) - 1))]
            pre[rows, il] = Q[rows, il] - q
            pre[rows, ih] = Q[rows, ih] + q
            corr[rows, k] = q
            if k == 0:
                recal[rows] = True
    post = np.sort(pre, axis=1)
    assert (np.diff(post, axis=1) >= 0).all()
    return pre, post, recal, corr


def covered(y, Q, k):
    il, ih = PAIR_INDEX[k]
    return ((Q[:, il] <= y) & (y <= Q[:, ih])).astype(float)


def qscore(y, Q):
    return SDB.qscore(y, Q, NINE)


def interval_score80(y, lo, hi):
    return (hi - lo) + (2 / ALPHA80) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))


# --------------------------------------------------------------------------------------------------
# Rows and arms: addendum 81's task C frame, with every arm's nine quantiles captured
# --------------------------------------------------------------------------------------------------
def wanted_names():
    names = set(REQUIRED) | set(A64) | set(EXTRA_CAPTURE)
    names |= {f'{a}@{s}' for a in SEEDED_BASE for s in SEEDS}
    return names


def task_c_frame(variable, man):
    """c_frame of addendum 81's task C scorer with put_metrics wrapped to keep each arm's nine quantiles."""
    captured = {}
    keep = wanted_names()
    original = SDB.put_metrics

    def capture(cols, name, y, Q9_, med, Q11=None):
        if name in keep:
            captured[name] = np.asarray(Q9_, float).copy()
        return original(cols, name, y, Q9_, med, Q11)

    SDB.put_metrics = capture
    try:
        frame, gates, info = SDB.c_frame(variable, NEW, man)
    finally:
        SDB.put_metrics = original
    return frame, gates, info, captured


def stored_leads(captured, frame):
    leads = frame.lead.to_numpy()
    out = {}
    for name, Q in captured.items():
        ok = np.isfinite(Q).all(1)
        out[name] = sorted(int(x) for x in np.unique(leads[ok]))
    return out


def common_mask(frame, captured, leads_of, variable):
    """Per row: True where every required arm stored at that row's lead is finite."""
    lead = frame.lead.to_numpy()
    mask = np.ones(len(frame), bool)
    used = {}
    for name in REQUIRED:
        if name not in captured:
            continue                                 # linear_ar_joint: oxygen only
        at = np.isin(lead, leads_of[name])
        ok = np.isfinite(captured[name]).all(1)
        mask &= ~at | ok
        used[name] = leads_of[name]
    return mask, used


# --------------------------------------------------------------------------------------------------
# Gates F-G0, F-G1, F-G2
# --------------------------------------------------------------------------------------------------
def ensemble_frame(variable):
    frame, _ = E.frame_for(variable)
    frame = frame.reset_index(drop=True)
    E.combine(frame, E.E3, 'E3')
    E.combine(frame, E.E5, 'E5')
    return frame


def gate_g0(variable, frame, mask, efr):
    pub = json.loads(ENSEMBLE_REPORT.read_text())['scores'][variable]
    out, ok = {}, True
    ek = efr[['cohort', 'sensor', 'time', 'lead']].copy()
    ek['t'] = pd.DatetimeIndex(ek.time).as_unit('ns').asi8
    mk = frame.loc[mask, ['cohort', 'sensor', 'time', 'lead']].copy()
    mk['t'] = SDB.ns(mk.time)
    for (cohort, lead), g in mk.groupby(['cohort', 'lead']):
        e = ek[(ek.cohort == cohort) & (ek.lead == lead)]
        a = set(zip(g.sensor, g.t))
        b = set(zip(e.sensor, e.t))
        n_pub = pub[cohort][str(lead)]['rows']
        same = a == b and len(a) == len(g) == n_pub
        out[f'{cohort}|{lead}'] = {'common_rows': int(len(g)), 'ensemble_rows': int(len(e)), 'published_rows': n_pub,
                                   'only_common': int(len(a - b)), 'only_ensemble': int(len(b - a)), 'equal': bool(same)}
        ok &= same
    return {'cells': out, 'passed': bool(ok)}


def gate_g1(scratch, skip=False):
    """python -m src.experiments.v2_ensemble --out <scratch>, JSON-equal to the published report."""
    target = Path(scratch)
    if not skip:
        env = dict(os.environ, V2_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, '-m', 'src.experiments.v2_ensemble', '--out', str(target)], cwd=ROOT,
                       env=env, check=True)
    if not target.exists():
        return {'passed': False, 'reason': f'{target} absent'}
    mine = json.loads(target.read_text(encoding='utf-8'))
    pub = json.loads(ENSEMBLE_REPORT.read_text(encoding='utf-8'))
    equal = mine == pub
    out = {'reproduction': str(target), 'json_equal': bool(equal), 'published_sha256': digest(ENSEMBLE_REPORT),
           'reproduction_sha256': digest(target)}
    if not equal:
        diffs = []

        def walk(a, b, path):
            if len(diffs) >= 20:
                return
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    walk(a.get(k), b.get(k), f'{path}/{k}')
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for i, (x, z) in enumerate(zip(a, b)):
                    walk(x, z, f'{path}[{i}]')
            elif a != b:
                diffs.append([path, a, b])
        walk(mine, pub, '')
        out['first_differences'] = diffs
    out['passed'] = bool(equal)
    return out


def gate_g2(variable, efr):
    """The new rule at (0.1, 0.9), before sorting, against v2_ensemble.conformal on addendum 70's rows."""
    pub = json.loads(ENSEMBLE_REPORT.read_text())['scores'][variable]
    out, ok = {}, True
    t_all = pd.DatetimeIndex(efr.time).as_unit('ns').asi8
    for arm in ('E3', 'E5', 'quantile_tree', 'chronos2_net', 'arima_ss'):
        lo_p, hi_p, rec_p = E.conformal(efr, arm)
        lo_n = np.full(len(efr), np.nan)
        hi_n = np.full(len(efr), np.nan)
        rec_n = np.zeros(len(efr), bool)
        Q9 = efr[[f'{arm}|{P.qcol(a)}' for a in NINE]].to_numpy(float)
        for lead, idx in efr.groupby('lead').indices.items():
            pre, _, rec, _ = recalibrate(t_all[idx], lead, efr.truth.to_numpy(float)[idx], Q9[idx])
            lo_n[idx], hi_n[idx], rec_n[idx] = pre[:, 0], pre[:, 8], rec
        d = float(max(np.abs(lo_n - lo_p).max(), np.abs(hi_n - hi_p).max()))
        flags = bool((rec_n == rec_p).all())
        cells_worst = 0.0
        y = efr.truth.to_numpy(float)
        tmp = efr[['cohort', 'lead', 'fold', 'sensor']].copy()
        tmp['c'] = ((lo_n <= y) & (y <= hi_n)).astype(float)
        tmp['w'] = hi_n - lo_n
        tmp['s'] = interval_score80(y, lo_n, hi_n)
        for (cohort, lead), cell in tmp.groupby(['cohort', 'lead']):
            b = pub[cohort][str(lead)]['bands'][f'cqr:{arm}']
            for col, key in (('c', 'coverage80'), ('w', 'width80'), ('s', 'interval_score80')):
                cells_worst = max(cells_worst, abs(r6(R.macro(cell, col)) - b[key]))
            if int((~rec_n[cell.index.to_numpy()]).sum()) != b['raw_rows']:
                flags = False
        passed = d == 0.0 and flags and cells_worst <= 1e-6
        out[arm] = {'max_abs_bound_difference': d, 'recalibrated_flags_identical': flags,
                    'max_abs_cell_difference': cells_worst, 'passed': bool(passed)}
        ok &= passed
    return {'arms': out, 'passed': bool(ok)}


# --------------------------------------------------------------------------------------------------
# Per lead: recalibrate every arm, per-row metrics
# --------------------------------------------------------------------------------------------------
def arm_rows(name, cohorts):
    if '@' in name:
        return cohorts == 'Dev-2025'
    return np.ones(len(cohorts), bool)


def lead_metrics(lead_frame, captured_lead, names):
    """Per-row metric columns of every arm on one lead's common rows (both cohorts)."""
    y = lead_frame.truth.to_numpy(float)
    t = SDB.ns(lead_frame.time)
    lead = int(lead_frame.lead.iloc[0])
    cohorts = lead_frame.cohort.to_numpy()
    cols, info = {}, {}
    for name in names:
        Q = captured_lead[name]
        sel = arm_rows(name, cohorts) & np.isfinite(Q).all(1)
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        pre, post, rec, corr = recalibrate(t[idx], lead, y[idx], Q[idx])
        yy, Qr = y[idx], Q[idx]

        def put(key, values):
            col = np.full(len(y), np.nan)
            col[idx] = values
            cols[f'{key}|{name}'] = col
        put('qr', qscore(yy, Qr))
        put('qc', qscore(yy, post))
        for k in range(len(PAIRS)):
            tag = NOMINAL_TAG[k]
            put(f'cr{tag}', covered(yy, Qr, k))
            put(f'cp{tag}', covered(yy, pre, k))
            put(f'cc{tag}', covered(yy, post, k))
        put('wr80', Qr[:, 8] - Qr[:, 0])
        put('wc80', post[:, 8] - post[:, 0])
        put('isr80', interval_score80(yy, Qr[:, 0], Qr[:, 8]))
        put('isc80', interval_score80(yy, post[:, 0], post[:, 8]))
        put('raw', (~rec).astype(float))
        moved = (post[:, 0] != pre[:, 0]) | (post[:, 8] != pre[:, 8])
        put('moved', moved.astype(float))
        put('anymoved', (post != pre).any(1).astype(float))
        info[name] = {'rows': int(len(idx)), 'missing_on_common_rows': int((arm_rows(name, cohorts) & ~sel).sum())}
    return pd.DataFrame(cols, index=lead_frame.index), info


# --------------------------------------------------------------------------------------------------
# Per cell: levels, contrasts, probe-by-season cells
# --------------------------------------------------------------------------------------------------
def floors():
    f = SDB.floors()
    return {'temperature': f['temperature_quantisation'], 'oxygen_mgl': f['oxygen_mgl']}


def mgl(variable, x, c):
    return None if variable != 'oxygen' or x is None else r6(x * c)


def contrast_entry(cell, draws, cols, x, y, variable, c, family):
    units = int(cell[['fold', 'week']].drop_duplicates().shape[0])
    rows = int(cell[[x, y]].dropna().shape[0])
    point = R.macro(cell, x) - R.macro(cell, y)
    if units < MIN_UNITS:
        e = {'point': r6(point), 'ci': None, 'excludes_zero': None, 'status': 'n.e. (fewer than five units)'}
    else:
        e = R.contrast(cell, draws, cols, x, y)
        e['status'] = 'excludes zero' if e['excludes_zero'] else 'not distinguishable'
    e.update(family=family, rows=rows, units=units)
    fl = floors()
    p, ci = e['point'], e.get('ci')
    if variable == 'oxygen':
        e['point_mgl'] = mgl(variable, p, c)
        e['ci_mgl'] = None if ci is None else [mgl(variable, v, c) for v in ci]
        p, ci, floor = e['point_mgl'], e['ci_mgl'], fl['oxygen_mgl']
    else:
        floor = fl['temperature']
    lo, hi = (None, None) if ci is None else ci
    e['floor'] = floor
    e['reading'] = S.against(p, lo, hi, floor)
    e['floor_flags'] = {'abs_point_at_or_beyond_floor': bool(abs(p) >= floor),
                        'interval_wholly_beyond_floor': bool(lo is not None and (lo >= floor or hi <= -floor))}
    if e.get('ci') is not None:
        ps = FDR.p_side(e['point'], *e['ci'])
        e['p_side_outside'] = ps is None
        e['p'] = float(f'{(ps if ps is not None else FDR.p_mid(*e["ci"])):.9g}')
    return e


def level_stats(cell, name, variable, c):
    s = cell[cell[f'qr|{name}'].notna()]
    if not len(s):
        return None
    e = {'rows': int(len(s)), 'sensors': int(s.sensor.nunique()),
         'quantile_score9_raw': r6(R.macro(s, f'qr|{name}')),
         'quantile_score9_recalibrated': r6(R.macro(s, f'qc|{name}'))}
    for k in range(len(PAIRS)):
        tag = NOMINAL_TAG[k]
        e[f'coverage{tag}_raw'] = r6(R.macro(s, f'cr{tag}|{name}'))
        e[f'coverage{tag}_recalibrated'] = r6(R.macro(s, f'cc{tag}|{name}'))
    e['coverage80_recalibrated_before_sorting'] = r6(R.macro(s, f'cp80|{name}'))
    e['width80_raw'] = r6(R.macro(s, f'wr80|{name}'))
    e['width80_recalibrated'] = r6(R.macro(s, f'wc80|{name}'))
    e['interval_score80_raw'] = r6(R.macro(s, f'isr80|{name}'))
    e['interval_score80_recalibrated'] = r6(R.macro(s, f'isc80|{name}'))
    e['rows_issued_raw'] = int(s[f'raw|{name}'].sum())
    e['rows_sorting_moved_q10_or_q90'] = int(s[f'moved|{name}'].sum())
    e['rows_sorting_moved_any_level'] = int(s[f'anymoved|{name}'].sum())
    if variable == 'oxygen':
        e['mgl'] = {k: mgl(variable, e[k], c) for k in ('quantile_score9_raw', 'quantile_score9_recalibrated',
                                                         'width80_raw', 'width80_recalibrated',
                                                         'interval_score80_raw', 'interval_score80_recalibrated')}
    return e


def probe_season(cell, name):
    """Coverage of the 80 % interval per (sensor, fold or quarter) cell with at least MIN_CELL rows."""
    s = cell[cell[f'qr|{name}'].notna()]
    g = s.groupby(['sensor', 'fold']).agg(n=(f'cr80|{name}', 'size'), raw=(f'cr80|{name}', 'mean'),
                                          rec=(f'cc80|{name}', 'mean'))
    return g[g.n >= MIN_CELL].reset_index()


def cell_names(cohort, names):
    return [n for n in names if cohort == 'Dev-2025' or '@' not in n]


def contrast_pairs(present):
    """(x column, y column, family, arm, reference) for every declared contrast present in the cell."""
    out = []
    for a in present:
        out.append((f'qc|{a}', f'qr|{a}', 'R', a, 'raw'))
    for a in present:
        if '@' in a:
            base, s = a.split('@')
            for ref in REFS:
                ref_s = ref if ref == 'clim_quantiles' else f'{ref}@{s}'
                if base != ref and ref_s in present:
                    out.append((f'qc|{a}', f'qc|{ref_s}', 'K1r', a, ref_s))
            continue
        for ref in REFS:
            if a != ref and ref in present:
                out.append((f'qc|{a}', f'qc|{ref}', 'K1r', a, ref))
    return out


# --------------------------------------------------------------------------------------------------
# Reproduction of the raw reading (F-G3), ledger parity (F-G5), divergence (F-G6)
# --------------------------------------------------------------------------------------------------
def gate_g3_cell(block_sdb, levels, variable):
    worst, compared, failed = 0.0, 0, []
    for name, e in levels.items():
        if e is None:
            continue
        ref = (block_sdb.get('arms', {}).get(name) or block_sdb.get('published_arms', {}).get(name))
        if not ref:
            continue
        for mine, theirs in ((e['quantile_score9_raw'], ref.get('quantile_score9')),
                             (e['coverage80_raw'], ref.get('coverage80'))):
            if theirs is None:
                continue
            d = abs(mine - theirs)
            compared += 1
            worst = max(worst, d)
            if d > 1e-6 + 1e-12:
                failed.append([name, mine, theirs, d / max(abs(theirs), 1e-300)])
    return worst, compared, failed


def divergence_cell(cell_rows, captured_cell, names):
    """Rows whose raw 80 % band is more than 100 times arima_ss's median band (v2_bayes_ts_report.divergence)."""
    ref = float(np.median(captured_cell['arima_ss'][:, 8] - captured_cell['arima_ss'][:, 0]))
    out = {}
    for name in names:
        Q = captured_cell.get(name)
        if Q is None:
            continue
        w = Q[:, 8] - Q[:, 0]
        big = np.isfinite(w) & (w > 100 * ref)
        if big.any():
            sens = pd.Series(cell_rows.sensor.to_numpy()[big]).value_counts()
            out[name] = {'rows': int(big.sum()), 'of': int(np.isfinite(w).sum()), 'max_width': float(np.nanmax(w)),
                         'arima_median_width': ref, 'sensors': {k: int(v) for k, v in sens.items()}}
    return out


# --------------------------------------------------------------------------------------------------
# One variable
# --------------------------------------------------------------------------------------------------
def run_variable(variable, man, c, gates):
    t0 = time.time()
    frame, cgates, info, captured = task_c_frame(variable, man)
    if frame is None:
        gates[f'F-G0|{variable}'] = {'passed': False, 'task_c_gates': cgates}
        return None
    reports = SDB.published_reports(variable)
    cgates['C-R'] = SDB.reproduce(variable, frame, reports)
    cgates['C-R']['passed'] = all(v['passed'] for v in cgates['C-R'].values() if isinstance(v, dict) and 'passed' in v)
    task_c_ok = all(cgates[k]['passed'] for k in ('C-G1', 'C-G2', 'C-T', 'C-R'))
    leads_of = stored_leads(captured, frame)
    mask, used = common_mask(frame, captured, leads_of, variable)
    efr = ensemble_frame(variable)
    g0 = gate_g0(variable, frame, mask, efr)
    g0['task_c_gates'] = {k: bool(cgates[k]['passed']) for k in ('C-G1', 'C-G2', 'C-T', 'C-R')}
    g0['required_arms_and_leads'] = used
    g0['passed'] = bool(g0['passed'] and task_c_ok)
    gates[f'F-G0|{variable}'] = g0
    gates[f'F-G2|{variable}'] = gate_g2(variable, efr)
    print(f'F {variable}: G0 {g0["passed"]}  G2 {gates[f"F-G2|{variable}"]["passed"]}  {time.time() - t0:.0f} s',
          flush=True)
    names = [n for n in list(NEW) + list(PUB42) + list(EWMA) + list(A64) +
             [f'{a}@{s}' for a in SEEDED_BASE for s in SEEDS] if n in captured]
    sdb = json.loads(SDB_REPORT.read_text())['C']['scores'][variable]
    pub_bands = json.loads(ENSEMBLE_REPORT.read_text())['scores'][variable]
    common = frame[mask].copy()
    block, cells_out, arm_table = {}, [], []
    g3 = {'worst': 0.0, 'compared': 0, 'failed': []}
    g5 = {}
    g6 = {}
    family = []
    for lead, lf in common.groupby('lead'):
        idx = lf.index.to_numpy()
        cap = {n: captured[n][idx] for n in names}
        here = [n for n in names if int(lead) in leads_of[n]]
        metrics, info_l = lead_metrics(lf, cap, here)
        lf = pd.concat([lf, metrics], axis=1)
        for cohort, cell in lf.groupby('cohort'):
            cn = [n for n in cell_names(cohort, here) if f'qr|{n}' in cell and cell[f'qr|{n}'].notna().any()]
            levels = {n: level_stats(cell, n, variable, c) for n in cn}
            # F-G3: the raw reading equals addendum 81's on the same rows
            w, k, f = gate_g3_cell(sdb[cohort][str(lead)], levels, variable)
            g3['worst'] = max(g3['worst'], w)
            g3['compared'] += k
            g3['failed'] += [[cohort, int(lead)] + x for x in f]
            # F-G5: rows issued raw, equal across the seed-42 arms and to addendum 70's count
            raws = {n: levels[n]['rows_issued_raw'] for n in cn if '@' not in n and n not in A64}
            pub_raw = pub_bands[cohort][str(lead)]['bands']['cqr:E3']['raw_rows']
            g5[f'{cohort}|{lead}'] = {'published': pub_raw, 'distinct_counts': sorted(set(raws.values())),
                                      'passed': bool(set(raws.values()) == {pub_raw})}
            # F-G6: divergence of the raw bands
            pos = lf.index.get_indexer(cell.index)
            div = divergence_cell(cell, {n: cap[n][pos] for n in cn if '@' not in n}, [n for n in cn if '@' not in n])
            if div:
                g6[f'{cohort}|{lead}'] = div
            # contrasts on one joint stream for the cell
            pairs = contrast_pairs(cn)
            cols = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
            draws = R.joint_draws(cell, cols, f'a82F|{variable}|{cohort}|{lead}')
            contrasts = {}
            for x, yname, fam, a, ref in pairs:
                key = f'{fam}|{a}|{ref}'
                contrasts[key] = contrast_entry(cell, draws, cols, x, yname, variable, c, fam)
                if contrasts[key].get('p') is not None:
                    family.append((variable, cohort, int(lead), key))
            for n in cn:
                ps = probe_season(cell, n)
                for row in ps.itertuples(index=False):
                    cells_out.append({'variable': variable, 'cohort': cohort, 'lead': int(lead), 'arm': n,
                                      'sensor': row.sensor, 'fold_or_quarter': row.fold, 'rows': int(row.n),
                                      'coverage80_raw': r6(row.raw), 'coverage80_recalibrated': r6(row.rec)})
                e = levels[n]
                if e is not None:
                    if len(ps):
                        e['probe_season'] = {
                            'cells': int(len(ps)), 'min_rows': MIN_CELL,
                            'coverage80_raw_range': [r6(ps.raw.min()), r6(ps.raw.max())],
                            'coverage80_recalibrated_range': [r6(ps.rec.min()), r6(ps.rec.max())],
                            'lowest_recalibrated': [ps.loc[ps.rec.idxmin(), 'sensor'], ps.loc[ps.rec.idxmin(), 'fold']],
                            'highest_recalibrated': [ps.loc[ps.rec.idxmax(), 'sensor'], ps.loc[ps.rec.idxmax(), 'fold']]}
                    arm_table.append({'variable': variable, 'cohort': cohort, 'lead': int(lead), 'arm': n,
                                      **{k: v for k, v in e.items() if not isinstance(v, dict)}})
            block.setdefault(cohort, {})[str(lead)] = {
                'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()),
                'units_fold_week': int(cell[['fold', 'week']].drop_duplicates().shape[0]),
                'arms': levels, 'contrasts': contrasts,
                'chronos2_cov_reproduction': {
                    'published_arm': 'chronos2_net (addendum 65); chronos2_cov reproduces it within 1e-3 (gate C-G1)',
                    'published_cqr_coverage80': pub_bands[cohort][str(lead)]['bands']['cqr:chronos2_net']['coverage80'],
                    'published_raw_coverage80': pub_bands[cohort][str(lead)]['bands']['raw:chronos2_net']['coverage80']}}
            print(f'  F {variable} {cohort} {lead} h: {len(cell)} rows, {len(cn)} arms, {len(contrasts)} contrasts',
                  flush=True)
    gates[f'F-G3|{variable}'] = {'compared': g3['compared'], 'max_abs_difference': g3['worst'],
                                 'failures': g3['failed'][:20], 'passed': bool(g3['compared'] > 0 and not g3['failed']),
                                 'failure_columns': ['cohort', 'lead', 'arm', 'here', 'addendum 81', 'abs diff / |value|'],
                                 'scope': 'a gate per arm and cell: a failure stops that arm in that cell only'}
    stopped_cells = []
    for cohort, lead, name, *_ in g3['failed']:
        cell = block[cohort][str(lead)]
        cell['arms'][name]['stopped'] = 'F-G3: the raw reading does not reproduce addendum 81 to 1e-6'
        for key, e in cell['contrasts'].items():
            _, a, ref = key.split('|')
            if name in (a, ref):
                e['stopped'] = 'F-G3 failed for ' + name + ' in this cell'
        stopped_cells.append([cohort, int(lead), name])
    family = [f for f in family if 'stopped' not in block[f[1]][str(f[2])]['contrasts'][f[3]]]
    gates[f'F-G3|{variable}']['stopped_arm_cells'] = stopped_cells
    gates[f'F-G5|{variable}'] = {'cells': g5, 'passed': all(v['passed'] for v in g5.values())}
    gates[f'F-G6|{variable}'] = {'divergence': g6, 'note': 'a reported check, not a stop'}
    block['_seed_verdicts'] = seed_verdicts(block)
    block['_dlm_without_diverged_probe'] = dlm_without(common, captured, g6, variable, c)
    block['_info'] = {'rows_task': int(len(frame)), 'rows_common': int(mask.sum()), 'arms': names,
                      'stored_leads': leads_of, 'seconds': round(time.time() - t0, 1)}
    return block, cells_out, arm_table, family


def seed_verdicts(block):
    """Dev-2025: R and K1r of each seeded arm over sampler seeds 42, 7 and 123 (sweep_verdict, seeds only)."""
    out = {}
    for lead, cell in block.get('Dev-2025', {}).items():
        c = cell['contrasts']
        for base in SEEDED_BASE:
            keys = {'R': [f'R|{base}|raw'] + [f'R|{base}@{s}|raw' for s in SEEDS]}
            keys['K1r vs clim_quantiles'] = [f'K1r|{base}|clim_quantiles'] + [f'K1r|{base}@{s}|clim_quantiles'
                                                                           for s in SEEDS]
            if base != 'quantile_tree':
                keys['K1r vs quantile_tree'] = [f'K1r|{base}|quantile_tree'] + [f'K1r|{base}@{s}|quantile_tree@{s}'
                                                                               for s in SEEDS]
            for what, ks in keys.items():
                if not all(k in c and c[k].get('ci') for k in ks):
                    continue
                if any('stopped' in c[k] for k in ks):
                    out[f'{lead}|{base}|{what}'] = {'verdict': 'stopped (F-G3 failed for a replicate in this cell)'}
                    continue
                cells = [(c[k]['point'], c[k]['ci'][0], c[k]['ci'][1]) for k in ks]
                code = sweep_verdict(cells)
                out[f'{lead}|{base}|{what}'] = {'points': [x[0] for x in cells], 'verdict_code': code,
                                                'verdict': verdict_label(code)}
    return out


def dlm_without(common, captured, g6, variable, c):
    """The DLM's recalibrated reading on Dev-2025 without the probe that holds most of its diverged rows."""
    counts = {}
    for key, div in g6.items():
        if key.startswith('Dev-2025|') and 'dlm' in div:
            for s, n in div['dlm']['sensors'].items():
                counts[s] = counts.get(s, 0) + n
    if not counts:
        return {'diverged_rows_on_dev': 0}
    probe = max(counts, key=counts.get)
    out = {'left_out': probe, 'diverged_rows_by_probe': counts, 'by_lead': {}}
    dev = common[(common.cohort == 'Dev-2025')]
    for lead, lf in dev.groupby('lead'):
        rows_all = common.index[(common.lead == lead).to_numpy()]
        # the ledger keeps every row of the lead (both cohorts); only the reading leaves the probe out
        sub = common.loc[rows_all]
        idx = sub.index.to_numpy()
        metrics, _ = lead_metrics(sub, {n: captured[n][idx] for n in ('dlm', 'arima_ss')}, ['dlm', 'arima_ss'])
        sub = pd.concat([sub, metrics], axis=1)
        cell = sub[(sub.cohort == 'Dev-2025') & (sub.sensor != probe)]
        e = {k: r6(R.macro(cell, f'{k}|dlm')) for k in ('qr', 'qc', 'cr80', 'cc80')}
        e['qc_minus_arima_ss_qc'] = r6(R.macro(cell, 'qc|dlm') - R.macro(cell, 'qc|arima_ss'))
        if variable == 'oxygen':
            e['mgl'] = {k: mgl(variable, e[k], c) for k in ('qr', 'qc', 'qc_minus_arima_ss_qc')}
        out['by_lead'][str(lead)] = e
    return out


def adjust_family(blocks, family):
    """Benjamini-Hochberg and Benjamini-Yekutieli at q = 0.05 over part F's R and K1r intervals."""
    ps = []
    for variable, cohort, lead, key in family:
        ps.append(blocks[variable][cohort][str(lead)]['contrasts'][key]['p'])
    bh = FDR.step_up(ps)
    by = FDR.step_up(ps, dependence_robust=True)
    for (variable, cohort, lead, key), a, b in zip(family, bh, by):
        e = blocks[variable][cohort][str(lead)]['contrasts'][key]
        e['bh'], e['by'] = bool(a), bool(b)
    excl = sum(1 for variable, cohort, lead, key in family
               if blocks[variable][cohort][str(lead)]['contrasts'][key]['excludes_zero'])
    by_fam = {}
    for (variable, cohort, lead, key), a, b in zip(family, bh, by):
        f = key.split('|')[0]
        d = by_fam.setdefault(f, [0, 0, 0, 0])
        d[0] += 1
        d[1] += int(blocks[variable][cohort][str(lead)]['contrasts'][key]['excludes_zero'])
        d[2] += int(a)
        d[3] += int(b)
    return {'q': FDR.Q, 'intervals': len(ps), 'excluding_zero': excl, 'bh_retained': int(sum(bh)),
            'by_retained': int(sum(by)),
            'by_contrast_family': {k: {'intervals': v[0], 'excluding_zero': v[1], 'bh': v[2], 'by': v[3]}
                                   for k, v in by_fam.items()},
            'p_rule': 'scripts/v2_fdr.p_side (estimate and the interval end towards zero); p_mid where the estimate '
                      'lies outside its own interval'}


def amend_dlm_cell(sonde='TMitiliE_05m-CTD(calibrato)', variable='oxygen', cohort='Dev-2025', lead=6):
    """Amendment 1 (F_G3_diverged_band): the arm-cell F-G3 stopped, dlm_emp@123 on Dev-2025 oxygen at 6 h, read
    without the sonde that holds its diverged band, as the thesis reads the DLM. The ledgers are those of the
    reading (every common row of the lead); only the reading leaves the sonde out. The seed verdicts of dlm_emp at
    that cell are read on the same rows for all three seeds. Part F's family is then recounted with the cell's three
    intervals read this way."""
    rep = json.loads(OUT_JSON.read_text())
    c = factor()
    man = SDB.manifest()
    frame, _, _, captured = task_c_frame(variable, man)
    leads_of = stored_leads(captured, frame)
    mask, _ = common_mask(frame, captured, leads_of, variable)
    lf = frame[mask & (frame.lead == lead).to_numpy()].copy()
    idx = lf.index.to_numpy()
    names = ['clim_quantiles', 'quantile_tree', 'quantile_tree@7', 'quantile_tree@123', 'dlm_emp', 'dlm_emp@7',
             'dlm_emp@123']
    metrics, _ = lead_metrics(lf, {n: captured[n][idx] for n in names}, names)
    lf = pd.concat([lf, metrics], axis=1)
    cell = lf[(lf.cohort == cohort).to_numpy() & (lf.sensor != sonde).to_numpy()]
    pairs = [('qc|dlm_emp@123', 'qr|dlm_emp@123', 'R', 'dlm_emp@123', 'raw'),
             ('qc|dlm_emp@123', 'qc|clim_quantiles', 'K1r', 'dlm_emp@123', 'clim_quantiles'),
             ('qc|dlm_emp@123', 'qc|quantile_tree@123', 'K1r', 'dlm_emp@123', 'quantile_tree@123'),
             ('qc|dlm_emp', 'qr|dlm_emp', 'R', 'dlm_emp', 'raw'),
             ('qc|dlm_emp', 'qc|clim_quantiles', 'K1r', 'dlm_emp', 'clim_quantiles'),
             ('qc|dlm_emp', 'qc|quantile_tree', 'K1r', 'dlm_emp', 'quantile_tree'),
             ('qc|dlm_emp@7', 'qr|dlm_emp@7', 'R', 'dlm_emp@7', 'raw'),
             ('qc|dlm_emp@7', 'qc|clim_quantiles', 'K1r', 'dlm_emp@7', 'clim_quantiles'),
             ('qc|dlm_emp@7', 'qc|quantile_tree@7', 'K1r', 'dlm_emp@7', 'quantile_tree@7')]
    cols = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    draws = R.joint_draws(cell, cols, f'a82F|{variable}|{cohort}|{lead}|without {sonde}')
    readings = {f'{f}|{a}|{ref}': contrast_entry(cell, draws, cols, x, y, variable, c, f) for x, y, f, a, ref in pairs}
    block = rep['scores'][variable][cohort][str(lead)]
    note = f'amendment 1: read without {sonde}, the sonde holding the diverged band'
    block['arms']['dlm_emp@123']['amendment1_reading'] = dict(level_stats(cell, 'dlm_emp@123', variable, c), note=note)
    for key in ('R|dlm_emp@123|raw', 'K1r|dlm_emp@123|clim_quantiles', 'K1r|dlm_emp@123|quantile_tree@123'):
        block['contrasts'][key]['amendment1_reading'] = dict(readings[key], note=note)
    sv = rep['scores'][variable]['_seed_verdicts']
    for what, ks in (('R', ['R|dlm_emp|raw', 'R|dlm_emp@7|raw', 'R|dlm_emp@123|raw']),
                     ('K1r vs clim_quantiles', ['K1r|dlm_emp|clim_quantiles', 'K1r|dlm_emp@7|clim_quantiles',
                                                'K1r|dlm_emp@123|clim_quantiles']),
                     ('K1r vs quantile_tree', ['K1r|dlm_emp|quantile_tree', 'K1r|dlm_emp@7|quantile_tree@7',
                                               'K1r|dlm_emp@123|quantile_tree@123'])):
        cells = [(readings[k]['point'], readings[k]['ci'][0], readings[k]['ci'][1]) for k in ks]
        code = sweep_verdict(cells)
        sv[f'{lead}|dlm_emp|{what}'] = {'points': [x[0] for x in cells], 'verdict_code': code,
                                        'verdict': verdict_label(code), 'note': note + ', for all three seeds',
                                        'replicates_read_without_sonde': {k: readings[k] for k in ks}}
    rep['gates'][f'F-G3|{variable}']['amendment1'] = (
        'config/bay_reserved_protocol_v2_addendum82_amendment1.json F_G3_diverged_band: the stopped arm-cell is read '
        f'without {sonde}; every other reading stands as declared')
    # part F's family, recounted with the cell's three intervals read without the sonde
    entries = []
    for v, blocks in rep['scores'].items():
        for coh, leads in blocks.items():
            if coh.startswith('_'):
                continue
            for ld, b in leads.items():
                for key, e in b['contrasts'].items():
                    use = e.get('amendment1_reading') if 'stopped' in e else e
                    if use is not None and use.get('p') is not None:
                        entries.append((key, use))
    ps = [u['p'] for _, u in entries]
    bh, by = FDR.step_up(ps), FDR.step_up(ps, dependence_robust=True)
    for (_, u), a, b in zip(entries, bh, by):
        u['bh'], u['by'] = bool(a), bool(b)
    by_fam = {}
    for (key, u), a, b in zip(entries, bh, by):
        d = by_fam.setdefault(key.split('|')[0], [0, 0, 0, 0])
        d[0] += 1
        d[1] += int(u['excludes_zero'])
        d[2] += int(a)
        d[3] += int(b)
    before = rep['multiplicity']
    rep['multiplicity'] = {'q': FDR.Q, 'intervals': len(ps), 'excluding_zero': int(sum(u['excludes_zero'] for _, u in entries)),
                           'bh_retained': int(sum(bh)), 'by_retained': int(sum(by)),
                           'by_contrast_family': {k: {'intervals': v[0], 'excluding_zero': v[1], 'bh': v[2], 'by': v[3]}
                                                  for k, v in by_fam.items()},
                           'p_rule': before['p_rule'],
                           'amendment1': 'recounted with the dlm_emp@123 Dev oxygen 6 h intervals read without the sonde',
                           'before_amendment1': {k: before[k] for k in ('intervals', 'excluding_zero', 'bh_retained',
                                                                       'by_retained')}}
    rep['amendments'] = ['config/bay_reserved_protocol_v2_addendum82_amendment1.json']
    OUT_JSON.write_bytes((json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                          + '\n').encode('utf-8'))
    print(json.dumps({k: readings[k] for k in list(readings)[:3]}, default=str)[:900])
    print('multiplicity', {k: rep['multiplicity'][k] for k in ('intervals', 'excluding_zero', 'bh_retained', 'by_retained')},
          'before', rep['multiplicity']['before_amendment1'])


def main():
    if '--amend-dlm-cell' in sys.argv:
        amend_dlm_cell()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT_JSON))
    ap.add_argument('--g1-scratch', default=str(OUT_DIR / 'ensemble_reproduction.json'))
    ap.add_argument('--skip-g1', action='store_true', help='read an existing F-G1 reproduction instead of rerunning')
    ap.add_argument('--variables', default='temperature,oxygen')
    a = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = factor()
    gates = {'F-G1': gate_g1(a.g1_scratch, skip=a.skip_g1)}
    print('F-G1', gates['F-G1']['passed'], flush=True)
    man = SDB.manifest()
    blocks, cells, table, family = {}, [], [], []
    stopped = []
    if not gates['F-G1']['passed']:
        stopped.append('F-G1')
    for variable in a.variables.split(','):
        res = run_variable(variable, man, c, gates)
        if res is None:
            stopped.append(f'F-G0|{variable}')
            continue
        blocks[variable], cv, tv, fv = res
        cells += cv
        table += tv
        family += fv
    for key, g in gates.items():
        if isinstance(g, dict) and 'passed' in g and not g['passed'] and key.split('|')[0] in (
                'F-G0', 'F-G1', 'F-G2', 'F-G5'):
            stopped.append(key)
    multiplicity = adjust_family(blocks, family) if family else None
    pd.DataFrame(cells).to_csv(OUT_DIR / 'coverage_probe_season.csv.gz', index=False)
    pd.DataFrame(table).to_csv(OUT_DIR / 'arms_by_cell.csv.gz', index=False)
    rep = {
        'declared_in': DECLARATION, 'declaration_sha256': digest(ROOT / DECLARATION), 'part': 'F',
        'producer': 'src/experiments/v2_recalibration.py', 'producer_sha256': digest(Path(__file__)),
        'created_utc': now(), 'python': sys.version.split()[0],
        'rule': ('nested conformalized quantile regression on addendum 70\'s causal ledger: pairs '
                 '(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6) at nominal 0.8, 0.6, 0.4, 0.2; per variable and '
                 'lead, probes pooled, scores of rows whose target hour is at or before the issuance, Dev-2025 '
                 'first and 2026 continued; raw below 30 scores; median unchanged; each row sorted'),
        'levels': list(NINE), 'min_scores': MIN_SCORES, 'draws': R.DRAWS,
        'stream': 'R.joint_draws, one stream per cell labelled a82F|<variable>|<cohort>|<lead> over every column',
        'aggregation': 'nested macro of addendum 64 (per-sensor mean within fold or quarter, then over folds or quarters)',
        'units': 'temperature degC; oxygen in the stored sonde column (umol/kg) with mgl = x c',
        'c_mg_per_l_per_umol_kg': c, 'floors': floors(),
        'arms': {'pretrained': list(NEW), 'published_seed42': list(PUB42), 'ewma_banded': list(EWMA),
                 'length_48_block': list(A64), 'seed_replicates_dev_only': [f'{x}@{s}' for x in SEEDED_BASE for s in SEEDS],
                 'note': 'chronos2_net is read once, as chronos2_cov; its published raw and cqr coverages are given '
                         'beside chronos2_cov in each cell'},
        'gates': gates, 'stopped': sorted(set(stopped)),
        'multiplicity': multiplicity,
        'scores': blocks if not stopped else {'withheld': 'a blocking gate failed', 'computed': blocks},
        'files': {'coverage_probe_season': rel(OUT_DIR / 'coverage_probe_season.csv.gz'),
                  'arms_by_cell': rel(OUT_DIR / 'arms_by_cell.csv.gz')},
        'seconds': round(time.time() - t0, 1)}
    Path(a.out).write_bytes((json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                             + '\n').encode('utf-8'))
    print('wrote', a.out, 'stopped:', rep['stopped'], f'{rep["seconds"]} s')


if __name__ == '__main__':
    main()
