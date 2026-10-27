"""Addendum 82, part A: score the silent-probe rollout and rolled graphs on Bridge-Dev and Bridge-2026.

Reads the rollouts of `src/experiments/v2_rollout_a82.py` (runs/v2_addendum82/A/closed_loop/ and
runs/v2_addendum82/A/closed_loop_learned_source/), joins them to the continuation files (runs/v2_bridge/bridge_*,
6-48 h; runs/v2_bridge72/bridge72_*, 54-72 h) on (fold, sensor, onset, lead), the continuation files
defining the scored rows and carrying the truth, the last value, both climatological increments, the
neighbour rule and the anchored continuation.

Order: (1) every gate the runs and studies recorded (A-G1 to A-G4, A-G6, A-G7) must have passed, or
nothing on Bridge-2026 is joined to its truth (A-G8); (2) gate A-G5: this scorer, given the published
rows and labels, reproduces runs/v2_closed_loop_leads_a69.json at four decimals, every contrast of
runs/v2_bridge/report_a69.json and runs/v2_bridge72/report.json at six, and the published fixed-band
coverage; (3) the contrasts C1-C6 by variable, cohort and lead, each under a label new to this addendum
(a contrast that already exists on Bridge-Dev at 6-48 h, on rows the fix left unchanged, is the
published one, computed under its published label, and is not counted as a new result); (4) the
rollout's causal fixed band, beside the published in-sample band on Bridge-Dev; (5) diagnostics; (6)
Benjamini-Hochberg and Benjamini-Yekutieli over the part's new intervals, p-values as addendum 79.

The within-cell scheme of `v2_bridge.week_blocks` is recomputed here from per-week sums with the
identical random stream (one draw per cell per replicate, in the same order), which is what makes
the scorer fast; gate A-G5 is the check that it reproduces the published intervals.

Declared in config/bay_reserved_protocol_v2_addendum82.json (part_A).
Usage: python -m scripts.v2_rollout_a82_report [--workers 8]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import v2_bridge as vb  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
A = RUNS / 'v2_addendum82/A'
REPORT = RUNS / 'v2_rollout_a82.json'
SUMMARY = RUNS / 'v2_addendum82_A.json'
CACHE = A / 'score_frames.pkl'
DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json part_A'
VARIABLES = ('temperature', 'oxygen')
COHORTS = ('Bridge-Dev', 'Bridge-2026')
COHORT_DIR = {'Bridge-Dev': 'dev', 'Bridge-2026': 'reserved'}
LEADS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
C_MGL = factor()
FLOOR = {'temperature': 0.20, 'oxygen_mgl': 0.059}
PUBLISHED_CL = {'temperature': 'closed_loop_temperature.csv.gz', 'oxygen': 'closed_loop_oxygen_rolled_neighbour.csv.gz'}
#: (id, x, y): the contrasts of part A
CONTRASTS = [('C1', 'recursive_tree', 'persistence'), ('C2', 'recursive_tree', 'clim_hour'),
             ('C3', 'anchored_direct', 'recursive_tree'), ('C4', 'graph_conservative', 'recursive_tree'),
             ('C5', 'graph_conservative', 'graph_unconstrained'), ('C6', 'recursive_tree', 'clim_month_hour')]
ARMS = ['recursive_tree', 'graph_conservative', 'graph_unconstrained', 'persistence', 'clim_hour',
        'clim_month_hour', 'anchored_direct', 'neighbour_change']


def r6(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 6)


def clean(obj):
    """NaN and infinities as null, so every artifact is strict JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ns(values):
    return pd.to_datetime(pd.Series(values), utc=True).dt.as_unit('ns').astype('int64').to_numpy()


def with_week(frame, col):
    t = pd.to_datetime(frame[col], utc=True)
    iso = t.dt.isocalendar()
    return frame.assign(week=iso.year.astype(str) + '-' + iso.week.astype(str))


# --------------------------------------------------------------------------------------------------
# The contrast: v2_bridge.week_blocks with a fast, stream-identical within-cell scheme
# --------------------------------------------------------------------------------------------------
def fast_week_blocks(frame, x, y, label, digits=6):
    frame = frame.dropna(subset=[x, y])
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    f = frame.assign(ex=(frame.truth - frame[x]).abs(), ey=(frame.truth - frame[y]).abs(), n=1.0)
    cells, folds = [], []
    for (fold, _), cell in f.groupby(['fold', 'sensor']):
        g = cell.groupby('week')[['ex', 'ey', 'n']].sum()
        cells.append((g.ex.to_numpy(), g.ey.to_numpy(), g.n.to_numpy()))
        folds.append(fold)
    folds = np.asarray(folds)
    fold_ids = sorted(set(folds.tolist()))
    point = vb.macro(frame, x) - vb.macro(frame, y)
    draws = np.empty(vb.DRAWS)
    for d in range(vb.DRAWS):
        mx, my = np.empty(len(cells)), np.empty(len(cells))
        for c, (sx, sy, n) in enumerate(cells):
            idx = rng.integers(0, len(n), len(n))
            tot = n[idx].sum()
            mx[c], my[c] = sx[idx].sum() / tot, sy[idx].sum() / tot
        draws[d] = (np.mean([mx[folds == k].mean() for k in fold_ids]) -
                    np.mean([my[folds == k].mean() for k in fold_ids]))
    lo_w, hi_w = np.percentile(draws, [2.5, 97.5])
    joint, redrawn = vb.joint_week_draws(frame, x, y, np.random.default_rng([20260923, zlib.crc32(label.encode())]))
    lo, hi = np.percentile(joint, [2.5, 97.5])
    signs = {f'{fo}|{sn}': round(float((c.truth - c[x]).abs().mean() - (c.truth - c[y]).abs().mean()), digits)
             for (fo, sn), c in frame.groupby(['fold', 'sensor'])}
    return {'point': round(point, digits), 'ci': [round(float(lo), digits), round(float(hi), digits)],
            'excludes_zero': bool(lo > 0 or hi < 0),
            'ci_within_cell': [round(float(lo_w), digits), round(float(hi_w), digits)],
            'excludes_zero_within_cell': bool(lo_w > 0 or hi_w < 0),
            'joint_redrawn': int(redrawn), 'paired_episodes': int(len(frame)), 'per_cell': signs}


FRAMES = {}


def _init(path):
    global FRAMES
    with open(path, 'rb') as stream:
        FRAMES = pickle.load(stream)


def task(args):
    """One contrast: (frame key, lead, x, y, label, per-sensor, digits)."""
    key, lead, x, y, label, with_sensor, digits = args
    sub = FRAMES[key]
    sub = sub[sub.lead == lead]
    c = fast_week_blocks(sub, x, y, label, digits)
    c.update(vb.sparse_cells(sub, x, y, label))
    if with_sensor:
        c['per_sensor'] = vb.per_sensor(sub, x, y, label)
    return args, c


def run_tasks(tasks, workers):
    if workers <= 1:
        return [task(t) for t in tasks]
    with Pool(workers, initializer=_init, initargs=(str(CACHE),)) as pool:
        return pool.map(task, tasks, chunksize=1)


# --------------------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------------------
def published_closed_loop(variable):
    pub = pd.read_csv(RUNS / 'v2_closed_loop/perfold' / PUBLISHED_CL[variable])
    return with_week(pub, 'time')


def published_bridge(variable):
    """The frames v2_bridge.report(perfold_rollout=True) scores."""
    frame = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz', parse_dates=['onset'])
    iso = frame.onset.dt.isocalendar()
    frame['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
    published = pd.read_csv(vb.ROLLOUT_A69[variable], parse_dates=['time'])
    published = published.rename(columns={'time': 'onset'})[['fold', 'sensor', 'onset', 'lead', 'truth',
                                                             'recursive_tree', 'persistence', 'climatology']]
    dev = frame[frame.cohort == 'Bridge-Dev'].merge(published, on=['fold', 'sensor', 'onset', 'lead'],
                                                    how='outer', suffixes=('', '_published'), indicator=True)
    dev = dev.drop(columns=['_merge', 'truth_published', 'persistence_published', 'climatology'])
    return {'Bridge-Dev': dev, 'Bridge-2026': frame[frame.cohort == 'Bridge-2026']}


def published_bridge72(variable):
    frame = pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz', parse_dates=['onset'])
    iso = frame.onset.dt.isocalendar()
    frame = frame.assign(week=iso.year.astype(str) + '-' + iso.week.astype(str))
    return {name: frame[frame.cohort == name] for name in COHORTS}


def a82_frame(variable, cohort):
    """The rollout joined to the continuation rows of one cohort (6-48 h and 54-72 h)."""
    d = COHORT_DIR[cohort]
    roll = pd.read_csv(A / 'closed_loop' / d / f'closed_loop_{variable}.csv.gz')
    ls = pd.read_csv(A / 'closed_loop_learned_source' / d / f'closed_loop_{variable}.csv.gz')
    key = ['fold', 'sensor', 'time', 'lead']
    roll = roll.merge(ls[key + ['graph_unconstrained', 'graph_unconstrained_off']], on=key, how='left', validate='one_to_one')
    roll['onset_ns'] = ns(roll.time)
    cont = pd.concat([pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz'),
                      pd.read_csv(RUNS / 'v2_bridge72' / f'bridge72_{variable}.csv.gz')], ignore_index=True)
    cont = cont[cont.cohort == cohort].reset_index(drop=True)
    cont['onset_ns'] = ns(cont.onset)
    keep = ['fold', 'sensor', 'onset_ns', 'lead', 'recursive_tree', 'graph_conservative', 'graph_unconstrained',
            'recursive_tree_off', 'graph_conservative_off', 'graph_unconstrained_off', 'climatology']
    keep += [c for c in ('rolled_temperature', 'salinity_at_onset') if c in roll]
    j = cont.merge(roll[keep], on=['fold', 'sensor', 'onset_ns', 'lead'], how='left', validate='one_to_one')
    missing = int(j.recursive_tree.isna().sum())
    j = with_week(j, 'onset')
    return j, missing


# --------------------------------------------------------------------------------------------------
# Gate records
# --------------------------------------------------------------------------------------------------
def gate_records():
    from src.experiments import v2_rollout_a82_studies as st
    out, ok = {'runs': {}, 'studies': {}}, True
    for cohort, d in COHORT_DIR.items():
        for v in VARIABLES:
            path = A / 'runs' / f'run_{d}_{v}.json'
            if not path.exists():
                out['runs'][f'{d}|{v}'] = {'missing': str(path.relative_to(ROOT))}
                ok = False
                continue
            r = json.loads(path.read_text())
            g = {k: r[k] for k in r if k.startswith('gate_')}
            out['runs'][f'{d}|{v}'] = {'gates': g, 'episodes': r['episodes'], 'rows': r['rows'],
                                       'selection': r['selection'], 'environment': r['environment'],
                                       'seconds': r['seconds'], 'predictions': r.get('predictions')}
            if r.get('smoke_only'):
                ok = False
            for name, gate in g.items():
                if name == 'gate_A_G7_open_loop':
                    ok = ok and bool(gate) and all(x['passed'] for x in gate.values())
                else:
                    ok = ok and gate['passed']
    for v in VARIABLES:
        for arm in st.ARMS:
            folder = st.folder_for(v, arm, False)
            path = folder / 'report.json'
            if not path.exists():
                out['studies'][f'{v}|{arm}'] = {'missing': True}
                ok = False
                continue
            r = json.loads(path.read_text())
            out['studies'][f'{v}|{arm}'] = {'study': r['study'], 'gate_A_G6_objective': r['gate_A_G6_objective'],
                                            'inner_window': r['inner_window'], 'environment': r['environment'],
                                            'completed_trials': r['selection']['completed_trials'],
                                            'failed_trials': r['selection']['failed_trials'],
                                            'best_value': r['selection']['best_value'], 'params': r['selection']['params'],
                                            'seconds': r['seconds'], 'selection_record_sha256': r['selection_record_sha256']}
            ok = ok and r['gate_A_G6_objective']['passed'] and r['inner_window']['no_2026_row'] \
                and r['selection']['completed_trials'] >= 100 and not r['selection']['smoke_only']
    out['gate_A_G8_no_2026_outcome_first'] = g8 = no_outcome_first()
    out['all_passed'] = bool(ok and g8['passed'])
    return out


def no_outcome_first():
    """A-G8: the Bridge-2026 prediction files carry no truth column, and the quick runs of Bridge-2026
    wrote no row; the truth is joined here only after every recorded gate has passed."""
    out, ok = {}, True
    for v in VARIABLES:
        for folder in ('v2_addendum82/A/closed_loop/reserved', 'v2_addendum82/A/closed_loop_learned_source/reserved'):
            cols = list(pd.read_csv(RUNS / folder / f'closed_loop_{v}.csv.gz', nrows=1).columns)
            out[f'{folder}/closed_loop_{v}.csv.gz'] = {'truth_column': 'truth' in cols}
            ok = ok and 'truth' not in cols
        smoke = A / 'runs/smoke' / f'run_reserved_{v}.json'
        rec = json.loads(smoke.read_text()) if smoke.exists() else {}
        out[f'quick_run_reserved_{v}'] = rec.get('written', 'no quick run on record')
        ok = ok and rec.get('written', '').startswith('nothing')
        ok = ok and not (A / 'closed_loop/smoke/reserved').exists()
    out['passed'] = bool(ok)
    return out


# --------------------------------------------------------------------------------------------------
# Gate A-G5
# --------------------------------------------------------------------------------------------------
def compare(mine, published, digits, path=''):
    """Every numeric leaf both dictionaries hold, rounded to `digits`; returns the mismatches."""
    bad = []
    if isinstance(published, dict):
        for k, v in published.items():
            if isinstance(mine, dict) and k in mine:
                bad += compare(mine[k], v, digits, f'{path}/{k}')
    elif isinstance(published, list):
        if not isinstance(mine, list) or len(mine) != len(published):
            bad.append((path, mine, published))
        else:
            for i, (a, b) in enumerate(zip(mine, published)):
                bad += compare(a, b, digits, f'{path}[{i}]')
    elif isinstance(published, bool) or published is None:
        if mine != published:
            bad.append((path, mine, published))
    elif isinstance(published, (int, float)):
        if mine is None or round(float(mine), digits) != round(float(published), digits):
            bad.append((path, mine, published))
    return bad


def g5_tasks():
    tasks, frames = [], {}
    for v in VARIABLES:
        frames[('pub_cl', v)] = published_closed_loop(v)
        for lead in LEADS[:8]:
            for x, y in (('recursive_tree', 'persistence'), ('recursive_tree', 'climatology'),
                         ('graph_denied', 'recursive_tree')):
                tasks.append((('pub_cl', v), lead, x, y, f'{v}|{lead}|{x}|{y}', False, 4))
        for name, f in published_bridge(v).items():
            frames[('pub_bridge', v, name)] = f
            arms = ['persistence', 'clim_month_hour', 'clim_hour', 'neighbour_change', 'anchored_direct', 'recursive_tree']
            present = [a for a in arms if a in f and f[a].notna().any()]
            for lead in sorted(f.lead.unique()):
                pairs = [('anchored_direct', o, f'{v}|{name}|{lead}|ad|{o}') for o in present if o != 'anchored_direct']
                pairs += [(x, y, f'{v}|{name}|{lead}|{x}|{y}') for x, y in
                          (('neighbour_change', 'persistence'), ('clim_hour', 'persistence'), ('clim_hour', 'clim_month_hour'))]
                for x, y, label in pairs:
                    tasks.append((('pub_bridge', v, name), int(lead), x, y, label, (name, v) in vb.PER_SENSOR, 6))
        for name, f in published_bridge72(v).items():
            frames[('pub_b72', v, name)] = f
            arms = ['persistence', 'clim_month_hour', 'clim_hour', 'neighbour_change', 'anchored_direct']
            for lead in sorted(f.lead.unique()):
                pairs = [('anchored_direct', o) for o in arms if o != 'anchored_direct']
                pairs += [('neighbour_change', 'persistence'), ('clim_hour', 'persistence'), ('clim_hour', 'clim_month_hour')]
                for x, y in pairs:
                    tasks.append((('pub_b72', v, name), int(lead), x, y, f'bridge72|{v}|{name}|{lead}|{x}|{y}',
                                  (name, v) in vb.PER_SENSOR, 6))
        frames[('pub_ls', v)] = constraint_frame(v)
        for lead in LEADS[:8]:
            for x, y in (('graph_conservative', 'graph_unconstrained'),):
                tasks.append((('pub_ls', v), lead, x, y, f'constraint|{v}|{lead}|{x}|{y}', False, 12))
    return tasks, frames


def constraint_frame(variable):
    """The frame addendum 76's scorer read: the published per-fold rollout with the unconstrained graph."""
    key = ['fold', 'sensor', 'time', 'lead']
    pub = pd.read_csv(RUNS / 'v2_closed_loop/perfold' / PUBLISHED_CL[variable])
    new = pd.read_csv(RUNS / 'v2_closed_loop_learned_source/perfold' / PUBLISHED_CL[variable])
    f = pub[key + ['truth', 'recursive_tree', 'graph_denied']].rename(columns={'graph_denied': 'graph_conservative'})
    f = f.merge(new[key + ['graph_denied']].rename(columns={'graph_denied': 'graph_unconstrained'}), on=key)
    return with_week(f, 'time')


def g5_check(results):
    leads69 = json.loads((RUNS / 'v2_closed_loop_leads_a69.json').read_text())
    rep69 = json.loads((RUNS / 'v2_bridge/report_a69.json').read_text())
    rep72 = json.loads((RUNS / 'v2_bridge72/report.json').read_text())
    constraint = json.loads((RUNS / 'v2_closed_loop_constraint.json').read_text())
    out = {'closed_loop_leads_a69': {'compared': 0, 'mismatches': []}, 'bridge_report_a69': {'compared': 0, 'mismatches': []},
           'bridge72_report': {'compared': 0, 'mismatches': []}, 'constraint_addendum76': {'compared': 0, 'mismatches': []}}
    for (key, lead, x, y, label, _, digits), c in results:
        kind, v = key[0], key[1]
        if kind == 'pub_cl':
            pub = leads69['variables'][v]['by_lead'][str(lead)]['contrasts'][f'{x} - {y}']['week_blocks']
            mine = {'point': c['point'], 'ci': c['ci'], 'excludes_zero': c['excludes_zero'],
                    'within_cell': {'ci': c['ci_within_cell'], 'excludes_zero': c['excludes_zero_within_cell']},
                    'joint_redrawn': c['joint_redrawn']}
            bad = compare(mine, pub, 4); slot = 'closed_loop_leads_a69'
        elif kind == 'pub_bridge':
            name = key[2]
            pub = rep69[v][name]['by_lead'][str(lead)]['contrasts'][f'{x} - {y}']
            bad = compare(c, pub, 6); slot = 'bridge_report_a69'
        elif kind == 'pub_b72':
            name = key[2]
            pub = rep72[v][name]['by_lead'][str(lead)]['contrasts'][f'{x} - {y}']
            bad = compare(c, pub, 6); slot = 'bridge72_report'
        else:
            # addendum 76 stored oxygen in mg/l (times c) at four decimals
            pub = constraint['variables'][v]['by_lead'][str(lead)]['contrasts'][f'{x} - {y}']
            k = C_MGL if v == 'oxygen' else 1.0
            mine = {'point': k * c['point'], 'ci': [k * z for z in c['ci']], 'excludes_zero': c['excludes_zero']}
            bad = compare(mine, pub, 4)
            slot = 'constraint_addendum76'
        out[slot]['compared'] += 1
        out[slot]['mismatches'] += [{'label': label, 'path': p, 'mine': m, 'published': q} for p, m, q in bad]
    for slot in out:
        out[slot]['passed'] = bool(out[slot]['compared'] > 0 and not out[slot]['mismatches'])
    return out


def g5_mae(frames):
    """MAE per arm and lead of report_a69 and the 72 h report, at six decimals."""
    rep69 = json.loads((RUNS / 'v2_bridge/report_a69.json').read_text())
    rep72 = json.loads((RUNS / 'v2_bridge72/report.json').read_text())
    bad, n = [], 0
    for (kind, *rest), f in frames.items():
        if kind not in ('pub_bridge', 'pub_b72'):
            continue
        v, name = rest
        rep = rep69 if kind == 'pub_bridge' else rep72
        for lead, sub in f.groupby('lead'):
            for arm, val in rep[v][name]['by_lead'][str(int(lead))]['mae'].items():
                n += 1
                if round(vb.macro(sub, arm), 6) != round(val, 6):
                    bad.append({'kind': kind, 'variable': v, 'cohort': name, 'lead': int(lead), 'arm': arm})
    return {'compared': n, 'mismatches': bad, 'passed': not bad}


def g5_band():
    """The published in-sample fixed band's coverage from the published per-fold rows."""
    pub = {'temperature': [0.900, 0.864, 0.843, 0.831, 0.794, 0.768, 0.739, 0.716],
           'oxygen': [0.899, 0.859, 0.887, 0.863, 0.820, 0.782, 0.777, 0.741]}
    out, ok = {}, True
    for v in VARIABLES:
        f = pd.read_csv(RUNS / 'v2_closed_loop/perfold' / PUBLISHED_CL[v])
        first = f[f.lead == 6]
        half = float(np.percentile((first.recursive_tree - first.truth).abs(), 90))
        cov = [float(((g.recursive_tree - g.truth).abs() <= half).mean()) for _, g in f.groupby('lead')]
        match = [round(c, 3) == p for c, p in zip(cov, pub[v])]
        out[v] = {'half_width': r6(half), 'coverage': [r6(c) for c in cov], 'published': pub[v], 'match': all(match)}
        ok = ok and all(match)
    out['passed'] = bool(ok)
    return out


# --------------------------------------------------------------------------------------------------
# Bands and diagnostics on the new rows
# --------------------------------------------------------------------------------------------------
MIN_BAND = vb.MIN_BAND


def causal_band(frames):
    """The rollout's 90 % band by the continuation's causal-ledger rule, held fixed through the gap: an
    episode's half-width is the 90th percentile of the rollout's absolute six-hour errors over every
    episode (Bridge-Dev, then Bridge-2026, sensors pooled) whose six-hour target hour is at or before
    its onset, at least MIN_BAND residuals or the band abstains."""
    first = pd.concat([f[f.lead == 6].assign(cohort=c) for c, f in frames.items()], ignore_index=True)
    t = pd.to_datetime(first.onset, utc=True).to_numpy('datetime64[ns]')
    known = t + np.timedelta64(6, 'h')
    order = np.argsort(known, kind='stable')
    known_sorted = known[order]
    r_sorted = (first.truth - first.recursive_tree).abs().to_numpy(float)[order]
    half = {}
    for ti in np.unique(t):
        k = int(np.searchsorted(known_sorted, ti, side='right'))
        half[ti] = float(np.percentile(r_sorted[:k], 90)) if k >= MIN_BAND else np.nan
    out = {'rule': f'expanding causal ledger of six-hour rollout residuals, sensors pooled, Bridge-Dev then Bridge-2026, '
                   f'at least {MIN_BAND} residuals, abstain otherwise; the six-hour half-width held at every lead'}
    for cohort, f in frames.items():
        q = np.array([half[x] for x in pd.to_datetime(f.onset, utc=True).to_numpy('datetime64[ns]')])
        e = (f.truth - f.recursive_tree).abs().to_numpy()
        entry = {}
        for lead in LEADS:
            m = (f.lead.to_numpy() == lead) & np.isfinite(q)
            a = (f.lead.to_numpy() == lead) & ~np.isfinite(q)
            entry[str(lead)] = {'coverage': r6(np.mean(e[m] <= q[m])) if m.any() else None, 'scored': int(m.sum()),
                                'abstained': int(a.sum()), 'mean_half_width': r6(np.mean(q[m])) if m.any() else None}
        out[cohort] = entry
    return out


def insample_band(f):
    first = f[f.lead == 6]
    half = float(np.percentile((first.recursive_tree - first.truth).abs(), 90))
    return {'half_width': r6(half), 'coverage': {str(int(L)): r6(((g.recursive_tree - g.truth).abs() <= half).mean())
                                                 for L, g in f.groupby('lead')}}


def diagnostics(f, variable):
    out = {}
    lo, hi = float(f.truth.min()), float(f.truth.max())
    out['observed_range'] = [r6(lo), r6(hi)]
    out['out_of_range'] = {a: int(((f[a] < lo) | (f[a] > hi)).sum()) for a in ('recursive_tree', 'graph_conservative',
                                                                                 'graph_unconstrained')}
    out['rows'] = int(len(f))
    if variable == 'oxygen':
        from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as sol
        ok = f.salinity_at_onset.notna() & f.rolled_temperature.notna()
        sub = f[ok]
        ratio = sub.recursive_tree / sol(sub.rolled_temperature.to_numpy(), sub.salinity_at_onset.to_numpy())
        out['supersaturation'] = {'rows_checked': int(len(sub)), 'above_140pct': int((ratio > 1.40).sum()),
                                  'max_ratio': r6(ratio.max()), 'median_ratio': r6(ratio.median())}
    for span, g in (('6-48', f[f.lead <= 48]), ('6-72', f)):
        by = g.groupby(['fold', 'sensor', 'onset'])
        sd = by.truth.std()
        amp = {}
        for arm in ('recursive_tree', 'graph_conservative', 'graph_unconstrained', 'clim_hour'):
            ratio = (by[arm].std() / sd)[sd > 0]
            amp[arm] = {'median_sd_ratio': r6(ratio.median()), 'quartiles': [r6(x) for x in ratio.quantile([.25, .75])],
                        'episodes': int(len(ratio))}
        out[f'amplitude_within_episode_{span}'] = amp
    return out


# --------------------------------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------------------------------
def p_value(point, lo, hi):
    from scripts.v2_fdr import p_side, p_mid
    p = p_side(point, lo, hi)
    return p if p is not None else p_mid(lo, hi)


def step_up(p, robust):
    from scripts.v2_fdr import step_up as su
    return su(p, dependence_robust=robust)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def entry(c, variable, floor_units):
    lo, hi = c['ci']
    e = {'point': c['point'], 'ci': c['ci'], 'excludes_zero': c['excludes_zero'],
         'status': 'excludes zero' if c['excludes_zero'] else 'not distinguishable',
         'ci_within_cell': c.get('ci_within_cell'), 'paired_episodes': c.get('paired_episodes')}
    if variable == 'oxygen':
        e['units'] = 'umol/kg (stored); mg/l = x c'
        e['point_mgl'] = r6(c['point'] * C_MGL)
        e['ci_mgl'] = [r6(lo * C_MGL), r6(hi * C_MGL)]
    else:
        e['units'] = 'degC'
    e['floor'] = floor_units
    e['point_beyond_floor'] = bool(abs(c['point']) >= floor_units)
    e['interval_beyond_floor'] = bool(lo >= floor_units or hi <= -floor_units)
    e['p'] = r6(p_value(c['point'], lo, hi))
    return e


def main(workers):
    started = time.time()
    report = {'declared_in': DECLARED, 'draws': vb.DRAWS, 'aggregation': 'frozen (fold, sensor) macro',
              'resampling': 'onset ISO weeks drawn jointly within each fold, cells fixed (v2_bridge.week_blocks)'}
    gates = gate_records()
    report['gates_recorded'] = gates
    if not gates['all_passed']:
        report['stopped'] = 'a recorded gate failed or a record is missing: nothing is scored'
        REPORT.write_bytes(json.dumps(clean(report), indent=2, default=str).encode('utf-8'))
        print(report['stopped']); print(json.dumps(gates, indent=1, default=str)[:4000])
        return

    # frames: published (for A-G5) and the new ones
    tasks5, frames = g5_tasks()
    a82, missing = {}, {}
    for v in VARIABLES:
        for cohort in COHORTS:
            f, m = a82_frame(v, cohort)
            frames[('a82', v, cohort)] = f
            a82[(v, cohort)] = f
            missing[f'{v}|{cohort}'] = m
    report['rows_without_rollout'] = missing
    if any(missing.values()):
        report['stopped'] = 'continuation rows without a rollout prediction'
        REPORT.write_bytes(json.dumps(clean(report), indent=2, default=str).encode('utf-8'))
        print(report['stopped'], missing)
        return
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, 'wb') as stream:
        pickle.dump(frames, stream)
    global FRAMES
    FRAMES = frames

    # A-G5 first
    t0 = time.time()
    res5 = run_tasks(tasks5, workers)
    g5 = g5_check(res5)
    g5['mae'] = g5_mae(frames)
    g5['fixed_band'] = g5_band()
    g5['passed'] = bool(g5['closed_loop_leads_a69']['passed'] and g5['bridge_report_a69']['passed']
                        and g5['bridge72_report']['passed'] and g5['mae']['passed'] and g5['fixed_band']['passed'])
    g5['seconds'] = round(time.time() - t0, 1)
    report['gate_A_G5_published_numbers'] = g5
    print('A-G5', {k: (v['passed'] if isinstance(v, dict) and 'passed' in v else v) for k, v in g5.items()}, flush=True)
    if not g5['passed']:
        report['stopped'] = 'gate A-G5 failed: no new contrast is scored'
        REPORT.write_bytes(json.dumps(clean(report), indent=2, default=str).encode('utf-8'))
        return

    # which Dev 6-48 h cells are the published contrasts (rows unchanged by the neighbour fix)
    unchanged = {}
    for v in VARIABLES:
        g2 = gates['runs'][f'dev|{v}']['gates']['gate_A_G2_neighbour_fix']['changes']
        unchanged[v] = all(x['rows_changed_6_48'] == 0 for x in g2.values())
    published_label = {'C1': lambda v, L: f'{v}|{L}|recursive_tree|persistence',
                       'C6': lambda v, L: f'{v}|{L}|recursive_tree|climatology',
                       'C4': lambda v, L: f'{v}|{L}|graph_denied|recursive_tree',
                       'C3': lambda v, L: f'{v}|Bridge-Dev|{L}|ad|recursive_tree',
                       'C5': lambda v, L: f'constraint|{v}|{L}|graph_conservative|graph_unconstrained'}
    tasks, meta = [], []
    for v in VARIABLES:
        for cohort in COHORTS:
            for lead in LEADS:
                for cid, x, y in CONTRASTS:
                    existing = bool(cohort == 'Bridge-Dev' and lead <= 48 and cid in published_label and unchanged[v])
                    label = published_label[cid](v, lead) if existing else f'a82A|{v}|{cohort}|{lead}|{cid}|{x}|{y}'
                    digits = 6
                    tasks.append((('a82', v, cohort), lead, x, y, label, (cohort, v) in vb.PER_SENSOR and not existing, digits))
                    meta.append((v, cohort, lead, cid, x, y, existing, label))
    t0 = time.time()
    res = run_tasks(tasks, workers)
    print(f'{len(tasks)} contrasts in {time.time() - t0:.0f} s', flush=True)

    contrasts, family = [], []
    for (v, cohort, lead, cid, x, y, existing, label), (_, c) in zip(meta, res):
        floor_units = FLOOR['temperature'] if v == 'temperature' else FLOOR['oxygen_mgl'] / C_MGL
        e = {'variable': v, 'cohort': cohort, 'lead': lead, 'contrast': cid, 'x': x, 'y': y, 'label': label,
             'existing_published': existing, 'artifact_only': cid == 'C6', **entry(c, v, floor_units),
             'sparse_cells': c.get('sparse_cells'), 'without_sparse_cells': c.get('without_sparse_cells'),
             'per_cell': c.get('per_cell')}
        if existing:
            e['note'] = 'the published contrast on unchanged rows, computed under its published label; not a new result'
        else:
            family.append({'id': f'{label}', 'kind': 'main', 'ref': e, 'point': c['point'], 'lo': c['ci'][0],
                           'hi': c['ci'][1], 'excludes_zero': c['excludes_zero']})
            ws = c.get('without_sparse_cells')
            if ws:
                family.append({'id': f'{label}|without_sparse_cells', 'kind': 'without_sparse_cells', 'ref': ws,
                               'point': ws['point'], 'lo': ws['ci'][0], 'hi': ws['ci'][1], 'excludes_zero': ws['excludes_zero']})
            for sk, sv in (c.get('per_sensor') or {}).items():
                if sv.get('ci'):
                    family.append({'id': f'{label}|sensor|{sk}', 'kind': 'per_sensor', 'ref': sv, 'point': sv['point'],
                                   'lo': sv['ci'][0], 'hi': sv['ci'][1], 'excludes_zero': sv['excludes_zero']})
            if c.get('per_sensor'):
                e['per_sensor'] = c['per_sensor']
        contrasts.append(e)

    # multiplicity over the part's own new intervals
    for fam in family:
        pv = p_value(fam['point'], fam['lo'], fam['hi'])
        fam['p'] = 1.0 if pv is None else pv
    def adjust(entries):
        p = [x['p'] for x in entries]
        bh, by = step_up(p, False), step_up(p, True)
        for x, a, b in zip(entries, bh, by):
            x['bh'] = bool(a and x['excludes_zero']); x['by'] = bool(b and x['excludes_zero'])
        return {'tests': len(entries), 'exclude_zero': sum(x['excludes_zero'] for x in entries),
                'bh_retained': sum(x['bh'] for x in entries), 'by_retained': sum(x['by'] for x in entries)}
    fdr = {'family': 'part A: every new interval on a difference (main contrast intervals, per-sensor intervals and '
                     'intervals without sparse cells); excluded: reproductions of published intervals',
           'all': adjust(family)}
    for x in family:
        x['ref']['p_family'] = r6(x['p']); x['ref']['bh'] = x['bh']; x['ref']['by'] = x['by']
    main_only = [dict(x) for x in family if x['kind'] == 'main']
    fdr['main_only'] = adjust(main_only)
    for x in main_only:
        x['ref']['bh_main_only'] = x['bh']; x['ref']['by_main_only'] = x['by']

    # MAE by arm
    mae = {}
    for (v, cohort), f in a82.items():
        mae[f'{v}|{cohort}'] = {str(int(L)): {'n': int(len(g)), **{a: r6(vb.macro(g, a)) for a in ARMS if a in g}}
                                for L, g in f.groupby('lead')}

    band = causal_band({c: a82[('temperature', c)] for c in COHORTS})
    band_o = causal_band({c: a82[('oxygen', c)] for c in COHORTS})
    bands = {'temperature': {'causal': band, 'insample_dev': insample_band(a82[('temperature', 'Bridge-Dev')])},
             'oxygen': {'causal': band_o, 'insample_dev': insample_band(a82[('oxygen', 'Bridge-Dev')])}}
    for v in VARIABLES:
        diff = {}
        for L in LEADS:
            cz = bands[v]['causal']['Bridge-Dev'][str(L)]['coverage']
            ins = bands[v]['insample_dev']['coverage'].get(str(L))
            diff[str(L)] = r6(cz - ins) if cz is not None and ins is not None else None
        bands[v]['causal_minus_insample_dev'] = diff

    diag = {f'{v}|{cohort}': diagnostics(f, v) for (v, cohort), f in a82.items()}

    report.update(contrasts=contrasts, fdr=fdr, mae=mae, bands=bands, diagnostics=diag,
                  floors={'temperature_degC': FLOOR['temperature'], 'oxygen_mgl': FLOOR['oxygen_mgl'],
                          'oxygen_umol_per_kg': r6(FLOOR['oxygen_mgl'] / C_MGL), 'c_mgl_per_umol_kg': C_MGL},
                  existing_dev_cells_unchanged=unchanged, seconds=round(time.time() - started, 1))
    REPORT.write_bytes(json.dumps(clean(report), indent=2, default=str, allow_nan=False).encode('utf-8'))
    write_summary(report)
    print('wrote', REPORT.relative_to(ROOT), 'and', SUMMARY.relative_to(ROOT), f'in {report["seconds"]} s')


def write_summary(report):
    keep = ['variable', 'cohort', 'lead', 'contrast', 'x', 'y', 'label', 'existing_published', 'artifact_only',
            'point', 'ci', 'status', 'paired_episodes', 'units', 'point_mgl', 'ci_mgl', 'floor', 'point_beyond_floor',
            'interval_beyond_floor', 'p', 'p_family', 'bh', 'by', 'bh_main_only', 'by_main_only', 'per_sensor',
            'without_sparse_cells', 'sparse_cells']
    summary = {'declared_in': DECLARED, 'report': REPORT.relative_to(ROOT).as_posix(),
               'gates': {'recorded': report['gates_recorded'], 'A_G5': report['gate_A_G5_published_numbers']},
               'contrasts': [{k: c.get(k) for k in keep if k in c} for c in report['contrasts']],
               'fdr': report['fdr'], 'bands': report['bands'], 'mae': report['mae'], 'diagnostics': report['diagnostics'],
               'floors': report['floors'], 'existing_dev_cells_unchanged': report['existing_dev_cells_unchanged']}
    SUMMARY.write_bytes(json.dumps(clean(summary), indent=2, default=str, allow_nan=False).encode('utf-8'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    main(a.workers)
