"""Addendum 57: the jump at a cleaning against the length of the interval it ends, and the CTD cast
as an outside reference (Vito, 2026-09-29).

Wiping the window removes whatever bias the film has built up, so the reading's jump at a cleaning
is the most direct measurement of accumulated fouling. Addendum 27 found its mean over every cleaning
ordinary. This asks the two questions it left open: does the jump grow with the length of the
interval it ends, and does it in summer, where fouling is expected.

Test 1. The jump is the mean of six settled hours after the cleaning minus the six hours before it,
on the residual of a reference model (addendum 56's inputs, fitted on every hour with months held
out, so it does not depend on the cleaning calendar) and on the raw oxygen. Its mean and its slope on
the interval length are compared with 1000 calendars shifted by whole days, which keep every fake
cleaning at its hour of day, and with the wiped sonde at the same site.

Test 2. At Baia a CTD cast is taken within minutes of most cleanings. Against that cast, a fouled
probe should disagree more before the cleaning than after it, and more after a longer interval.

Declared in config/bay_reserved_protocol_v2_addendum57.json.
Usage: python -m src.experiments.v2_fouling_jump [--quick] [--workers N]
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.bay_profile_context import ROOT
from src.experiments.v2_biofouling import read_export
from src.experiments.v2_fouling_reference import (CUTOFF, MAX_DOSE_DAYS, SITES, WARM, calendar, exogenous,
                                                  features, fit, space)
from src.experiments.v2_fouling_response import transplanted
from src.experiments.v2_logbook import deployed_spans, read_logbook
from src.experiments.v2_oxygen_floor import DEPTH_TOLERANCE_M, casts
from src.tuning.protocol_v2 import run_study

OUT = ROOT / 'reports/bay_twin/runs/v2_fouling_jump.json'
FOLDER = ROOT / 'reports/bay_twin/runs/v2_fouling_jump'
FOLDS, INNER = 5, 3
DRAWS, BOOT = 1000, 2000
SEED = 20260929
HOUR = pd.Timedelta(hours=1)
#: window offsets in hours from h0, the hour that contains the cleaning (hours are labelled by their start)
WINDOWS = {'R6': ((-6, -1), (2, 7), 4, 'mean'), 'O6': ((-6, -1), (2, 7), 4, 'mean'),
           'O24': ((-24, -1), (1, 24), 12, 'median')}
#: amendment 1: no statistic, and no decision, on fewer than five independent units
MIN_UNITS = 5
CAST_WINDOW = pd.Timedelta(hours=1)
PRE, POST = (-2, -1), (2, 3)
#: node -> (CTD station, the probe's nominal depth in m)
STATIONS = {'node1': ('PBaia', 1.0), 'node5': ('PMitiliE', 5.0)}
ROLES = ('hand-cleaned', 'wiped')


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def record(node, role, exo_all):
    """The hourly record, its reference inputs, the calendar, the deployed mask and spans."""
    rows = read_logbook()
    gen1, gen2, _ = SITES[node]
    cleanings = sorted(rows[(rows.node == node) & (rows.action == 'Pulizia')].time)
    if role == 'hand-cleaned':
        frame = read_export(gen1)
        spans = deployed_spans(rows, node)
        deployed = np.zeros(len(frame), bool)
        for a, b in spans:
            deployed |= (frame.index >= a) & (frame.index <= b)
        cal = cleanings
    else:
        frame = read_export(gen2)
        frame = frame[frame.index < CUTOFF]
        spans = [(frame.index[0], frame.index[-1])]
        deployed = np.ones(len(frame), bool)
        cal = [c for c in transplanted(cleanings, min(cleanings).floor('D'), frame.index[0].floor('D'))
               if c < CUTOFF]
    x = features(frame, exo_all.reindex(frame.index)).to_numpy(float)
    return frame.index, x, frame.oxygen.to_numpy(float), cal, deployed, spans


def months(index):
    return (index.year * 12 + index.month).to_numpy()


def outer_folds(groups, rows):
    """(train rows, test months) per outer fold, grouped by calendar month."""
    held = np.unique(groups[rows])
    return [(rows[np.isin(groups[rows], held[tr])], held[te])
            for tr, te in GroupKFold(n_splits=FOLDS).split(held, groups=held)]


def study_task(args):
    node, role, k, quick, exo_all = args
    os.environ['OMP_NUM_THREADS'] = '1'
    index, x, y, _, deployed, _ = record(node, role, exo_all)
    groups = months(index)
    train, _ = outer_folds(groups, np.flatnonzero(deployed & np.isfinite(y)))[k]

    def objective(trial):
        p = space(trial)
        errs = []
        for a, b in GroupKFold(n_splits=INNER).split(train, groups=groups[train]):
            m = fit(p, x[train[a]], y[train[a]])
            errs.append(float(np.mean(np.abs(m.predict(x[train[b]]) - y[train[b]]))))
        return float(np.mean(errs))
    t0 = time.time()
    folder = (FOLDER / 'smoke' if quick else FOLDER) / 'studies'
    sel = run_study(f'fouling_jump_{node}_{role}_f{k}', objective, folder, quick)
    log(f'{node} {role} fold {k}: study done in {time.time() - t0:.0f} s')
    return (node, role), k, {'params': dict(sel['params']), 'trials': sel['completed_trials']}


def reference(index, x, y, deployed, chosen):
    """Every hour predicted by the model of the fold that held its month out."""
    groups = months(index)
    rows = np.flatnonzero(deployed & np.isfinite(y))
    pred = np.full(len(y), np.nan)
    for k, (train, test_months) in enumerate(outer_folds(groups, rows)):
        model = fit(chosen[k]['params'], x[train], y[train])
        test = np.isin(groups, test_months)
        pred[test] = model.predict(x[test])
    return pred


def valid_pairs(cleanings, spans):
    """(previous, cleaning, L days) for every cleaning that ends a scorable interval."""
    cl = sorted(cleanings)
    out = []
    for p, c in zip(cl[:-1], cl[1:]):
        days = (c - p) / pd.Timedelta(days=1)
        if 0 < days <= MAX_DOSE_DAYS and any(a <= p and c <= b for a, b in spans):
            out.append((p, c, days))
    return out


def window(values, deployed, k, lo, hi):
    a, b = k + lo, k + hi + 1
    if a < 0 or b > len(values) or not deployed[a:b].all():
        return None
    return values[a:b]


def jumps(index, series, pairs, deployed, form):
    (b_lo, b_hi), (a_lo, a_hi), need, how = WINDOWS[form]
    agg = np.nanmean if how == 'mean' else np.nanmedian
    t0 = index[0]
    out = []
    for p, c, days in pairs:
        k = int((c.floor('h') - t0) / HOUR)
        before, after = window(series, deployed, k, b_lo, b_hi), window(series, deployed, k, a_lo, a_hi)
        if before is None or after is None:
            continue
        if np.isfinite(before).sum() < need or np.isfinite(after).sum() < need:
            continue
        out.append((c, days, c.month in WARM, float(agg(after) - agg(before))))
    return pd.DataFrame(out, columns=['cleaning', 'L', 'warm', 'J']).astype({'L': float, 'warm': bool,
                                                                             'J': float})


def seasons(table, cold=True):
    """Row subsets by season. The mask is a NumPy array: an empty object Series used as a key would
    select columns instead of rows."""
    w = table.warm.to_numpy(bool)
    out = [('warm', table[w])] + ([('cold', table[~w])] if cold else []) + [('all', table)]
    return [(name, part.reset_index(drop=True)) for name, part in out]


def slope(xv, yv):
    xv, yv = np.asarray(xv, float), np.asarray(yv, float)
    if len(xv) < MIN_UNITS or np.unique(xv).size < 2:
        return None
    xc = xv - xv.mean()
    return float((xc * (yv - yv.mean())).sum() / (xc * xc).sum())


def jump_stats(table):
    out = {}
    for season, part in seasons(table):
        out[season] = {'n': int(len(part)),
                       'J_mean': float(part.J.mean()) if len(part) >= MIN_UNITS else None,
                       'J_slope': slope(part.L, part.J),
                       'median_L': float(part.L.median()) if len(part) else None}
    return out


def day_shifted(cleanings, first, last, rng, draws):
    """Circular shifts by a whole number of days, in integer nanoseconds so no fake cleaning moves
    off its hour of day."""
    day0 = first.floor('D')
    span_days = int((last.floor('D') - day0) / pd.Timedelta(days=1))
    day_ns = 86_400 * 10**9
    off = np.array([(c - day0).value for c in sorted(cleanings)], dtype='int64')
    out = []
    for _ in range(draws):
        k = int(rng.integers(1, span_days))
        shifted = np.sort((off + k * day_ns) % (span_days * day_ns))
        out.append([day0 + pd.Timedelta(int(v), unit='ns') for v in shifted])
    return out


def placebo(real, draws):
    out = {}
    for season in ('warm', 'cold', 'all'):
        out[season] = {}
        for stat in ('J_mean', 'J_slope'):
            obs = real[season][stat]
            null = np.array([d[season][stat] for d in draws if d[season][stat] is not None], float)
            if obs is None or len(null) < 20:
                out[season][stat] = None
                continue
            mid = float(np.median(null))
            p = float(np.mean(np.abs(null - mid) >= abs(obs - mid)))
            out[season][stat] = {'observed': round(obs, 5), 'p': round(p, 4), 'draws': int(len(null)),
                                 'null_median': round(mid, 5),
                                 'null_p2_5': round(float(np.percentile(null, 2.5)), 5),
                                 'null_p97_5': round(float(np.percentile(null, 97.5)), 5)}
    return out


def cast_values(node):
    """One value per cast: the bin closest to the probe's nominal depth, within the tolerance."""
    station, depth = STATIONS[node]
    site = casts()
    site = site[(site.station == station) & (site.t_utc < CUTOFF)]
    rows = []
    for cast_id, g in site.groupby('cast_id'):
        near = g.loc[(g.depth_m - depth).abs() <= DEPTH_TOLERANCE_M]
        if near.empty:
            continue
        b = near.loc[(near.depth_m - depth).abs().idxmin()]
        if np.isfinite(b.do_umol_l):
            rows.append((cast_id, b.t_utc, float(b.do_umol_l)))
    return pd.DataFrame(rows, columns=['cast_id', 't', 'value'])


def mean_at(values, deployed, k, lo, hi):
    w = window(values, deployed, k, lo, hi)
    return float(np.nanmean(w)) if w is not None and np.isfinite(w).any() else np.nan


def visits(index, y, pred, pairs, deployed, cv):
    """Test 2a: each cleaning with a cast within an hour, the probe before and after against it."""
    t0, out = index[0], []
    for p, c, days in pairs:
        near = cv[(cv.t - c).abs() <= CAST_WINDOW]
        if near.empty:
            continue
        k = int((c.floor('h') - t0) / HOUR)
        kc = int((near.t.sort_values().iloc[len(near) // 2].floor('h') - t0) / HOUR)
        pre, post = mean_at(y, deployed, k, *PRE), mean_at(y, deployed, k, *POST)
        if not (np.isfinite(pre) and np.isfinite(post)):
            continue
        r = y - pred
        at_cast = pred[kc] if 0 <= kc < len(pred) else np.nan
        cast = float(near.value.median())
        out.append({'date': str(c.date()), 'L': days, 'warm': c.month in WARM, 'casts': int(len(near)),
                    'E_pre': cast - pre, 'E_post': cast - post,
                    'E_pre_adj': cast - (mean_at(r, deployed, k, *PRE) + at_cast),
                    'E_post_adj': cast - (mean_at(r, deployed, k, *POST) + at_cast)})
    return pd.DataFrame(out)


def boot(table, fn, rng, draws):
    """Percentile interval over resampled rows; each row is one visit or one cast date."""
    if len(table) < MIN_UNITS:
        return None
    vals = []
    for _ in range(draws):
        v = fn(table.iloc[rng.integers(0, len(table), len(table))].reset_index(drop=True))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < draws // 2:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def visit_stats(table, rng, draws):
    out = {}
    if table.empty:
        return out
    for season, part in seasons(table):
        entry = {'visits': int(len(part))}
        for form, (a, b) in (('raw', ('E_pre', 'E_post')), ('adjusted', ('E_pre_adj', 'E_post_adj'))):
            p = part.dropna(subset=[a, b]).reset_index(drop=True)
            q = {}
            for name, f in ((a, lambda t, a=a: t[a]), (b, lambda t, b=b: t[b]),
                            ('difference', lambda t, a=a, b=b: t[a] - t[b])):
                q[name] = {'slope': slope(p.L, f(p)),
                           'mean': float(f(p).mean()) if len(p) >= MIN_UNITS else None,
                           'slope_ci': boot(p, lambda t, f=f: slope(t.L, f(t)), rng, draws),
                           'mean_ci': boot(p, lambda t, f=f: float(f(t).mean()), rng, draws)}
            q['median_abs'] = {a: float(p[a].abs().median()) if len(p) else None,
                               b: float(p[b].abs().median()) if len(p) else None}
            entry[form] = q
        out[season] = entry
    return out


def cast_series(index, y, cleanings, deployed, cv):
    """Test 2b (descriptive): every cast against the probe hour that contains it."""
    interval, hours = calendar(index, cleanings, deployed)
    t0 = index[0]
    skip = set()
    for c in cleanings:
        k = int((c.floor('h') - t0) / HOUR)
        skip.update((k, k + 1))
    rows = []
    for _, r in cv.iterrows():
        k = int((r.t.floor('h') - t0) / HOUR)
        if k < 0 or k >= len(y) or k in skip or interval[k] < 0 or not np.isfinite(y[k]):
            continue
        rows.append({'date': str(r.t.date()), 'interval': int(interval[k]), 'dose': hours[k] / 24,
                     'warm': r.t.month in WARM, 'diff': r.value - y[k]})
    return pd.DataFrame(rows)


def within(table):
    t = table[(table.groupby('interval').dose.transform('size') >= 2).to_numpy(bool)]
    if len(t) < MIN_UNITS:
        return None
    xv = t.dose - t.groupby('interval').dose.transform('mean')
    yv = t['diff'] - t.groupby('interval')['diff'].transform('mean')
    den = float((xv * xv).sum())
    return float((xv * yv).sum() / den) if den > 0 else None


def series_stats(table, rng, draws):
    out = {}
    if table.empty:
        return out
    for season, part in seasons(table, cold=False):
        by_date = [g for _, g in part.groupby('date')]

        def resample(fn):
            if len(by_date) < MIN_UNITS:
                return None
            vals = []
            for _ in range(draws):
                pick = [by_date[i] for i in rng.integers(0, len(by_date), len(by_date))]
                v = fn(pd.concat(pick, ignore_index=True))
                if v is not None and np.isfinite(v):
                    vals.append(v)
            return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] \
                if len(vals) >= draws // 2 else None
        out[season] = {'pairs': int(len(part)), 'dates': int(len(by_date)),
                       'intervals': int(part.interval.nunique()),
                       'median_diff': float(part['diff'].median()) if len(part) else None,
                       'pooled_slope': slope(part.dose, part['diff']),
                       'pooled_slope_ci': resample(lambda t: slope(t.dose, t['diff'])),
                       'within_slope': within(part), 'within_slope_ci': resample(within)}
    return out


def job(args):
    """One (node, role): the reference, the jumps and their placebo, and for the hand-cleaned
    probes the casts."""
    node, role, quick, exo_all, chosen = args
    os.environ['OMP_NUM_THREADS'] = '1'
    t0 = time.time()
    index, x, y, cal, deployed, spans = record(node, role, exo_all)
    pred = reference(index, x, y, deployed, chosen)
    ok = deployed & np.isfinite(y) & np.isfinite(pred)
    series = {'R6': y - pred, 'O6': y, 'O24': y}
    pairs = valid_pairs(cal, spans)
    tables = {f: jumps(index, series[f], pairs, deployed, f) for f in WINDOWS}
    real = {f: jump_stats(t) for f, t in tables.items()}
    draws = 20 if quick else DRAWS
    rng = np.random.default_rng(SEED)
    null = {f: [] for f in WINDOWS}
    for fake in day_shifted(cal, index[0], index[-1], rng, draws):
        fp = valid_pairs(fake, spans)
        for f in WINDOWS:
            null[f].append(jump_stats(jumps(index, series[f], fp, deployed, f)))
    tag = f'{node}_{role}'
    base = FOLDER / 'smoke' if quick else FOLDER
    base.mkdir(parents=True, exist_ok=True)
    pd.concat([t.assign(form=f) for f, t in tables.items()]).to_csv(base / f'jumps_{tag}.csv.gz', index=False)
    entry = {'node': node, 'role': role, 'first': str(index[0].date()), 'last': str(index[-1].date()),
             'cleanings': len(cal), 'scorable_pairs': len(pairs), 'reference_selections': chosen,
             'reference_mae': round(float(np.mean(np.abs((y - pred)[ok]))), 3),
             'jumps': {f: {'statistics': real[f], 'placebo': placebo(real[f], null[f])} for f in WINDOWS}}
    size = {}
    for f in WINDOWS:
        s, L = real[f]['warm']['J_slope'], real[f]['warm']['median_L']
        size[f] = None if s is None or L is None else {'per_median_warm_interval': round(s * L, 3),
                                                        'per_7_days': round(s * 7, 3), 'per_21_days': round(s * 21, 3)}
    entry['size'] = size
    if role == 'hand-cleaned':
        cv = cast_values(node)
        boot_draws = 50 if quick else BOOT
        vt = visits(index, y, pred, pairs, deployed, cv)
        st = cast_series(index, y, cal, deployed, cv)
        vt.to_csv(base / f'visits_{tag}.csv.gz', index=False)
        st.to_csv(base / f'casts_{tag}.csv.gz', index=False)
        entry['casts_2a'] = visit_stats(vt, np.random.default_rng(SEED), boot_draws)
        entry['casts_2b_descriptive'] = series_stats(st, np.random.default_rng(SEED), boot_draws)
    log(f'{tag}: done in {time.time() - t0:.0f} s')
    return entry


def decisions(entries):
    """E1-E4 of the declaration, with amendment 1 (E1 needs five warm jumps; E3 reads the nodes where
    E1 can be computed)."""
    get = {(e['node'], e['role']): e for e in entries}
    out = {}
    for node in SITES:
        hand, wiped = get[(node, 'hand-cleaned')], get[(node, 'wiped')]
        if hand['jumps']['R6']['statistics']['warm']['n'] < MIN_UNITS:
            out[node] = {'E1': None, 'E1_note': 'not estimable: fewer than five warm jumps (amendment 1)',
                         'warm_jumps': hand['jumps']['R6']['statistics']['warm']['n']}
            continue
        passed, artefacts = [], []
        for stat in ('J_mean', 'J_slope'):
            h = hand['jumps']['R6']['placebo']['warm'][stat]
            w = wiped['jumps']['R6']['placebo']['warm'][stat]
            if h and h['p'] < 0.025:
                (artefacts if w and w['p'] < 0.05 else passed).append(stat)
        size = hand['size']['R6']
        out[node] = {'E1_signature': passed, 'E1_artefacts': artefacts, 'E1': bool(passed),
                     'E4_below_floor': bool(passed) and size is not None
                     and abs(size['per_median_warm_interval']) < 1.8}
    warm = get[('node1', 'hand-cleaned')].get('casts_2a', {}).get('warm', {}).get('raw', {})
    e2 = False
    if warm:
        pre, post = warm['E_pre'], warm['E_post']
        js = get[('node1', 'hand-cleaned')]['jumps']['R6']['statistics']['warm']['J_slope']
        excl = pre['slope_ci'] is not None and (pre['slope_ci'][0] > 0 or pre['slope_ci'][1] < 0)
        incl = post['slope_ci'] is not None and post['slope_ci'][0] <= 0 <= post['slope_ci'][1]
        e2 = bool(excl and incl and js is not None and np.sign(pre['slope']) == np.sign(js))
    out['E2'] = e2
    tested = [n for n in SITES if out[n]['E1'] is not None]
    out['E3_closed'] = bool(tested) and not any(out[n]['E1'] for n in tested) and not e2
    out['E3_tested_nodes'] = tested
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    exo_all = exogenous()
    keys = [(node, role) for node in SITES for role in ROLES]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        chosen = {key: {} for key in keys}
        for key, k, sel in pool.imap_unordered(study_task, [(*key, k, a.quick, exo_all)
                                                            for key in keys for k in range(FOLDS)]):
            chosen[key][k] = sel
        log('studies done; scoring')
        entries = pool.map(job, [(*key, a.quick, exo_all, chosen[key]) for key in keys])
    report = {'addendum': 57, 'declaration': 'config/bay_reserved_protocol_v2_addendum57.json',
              'amendment': 'config/bay_reserved_protocol_v2_addendum57_amendment1.json',
              'smoke_only': a.quick, 'entries': entries, 'decisions': decisions(entries)}
    out = OUT if not a.quick else OUT.with_name('v2_fouling_jump_smoke.json')
    out.write_bytes((json.dumps(report, indent=1, default=str) + '\n').encode('utf-8'))
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
