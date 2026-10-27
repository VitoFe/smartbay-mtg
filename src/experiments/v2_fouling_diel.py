"""Addendum 58: does the probe's response to sunlight and temperature drift away from the clean
probe's as the film builds up? (Vito, 2026-09-29)

A film on the optical window need not shift the reading's level. Algae on it make oxygen at the window
in sunlight and consume it at night, which exaggerates the day-night swing and leaves the daily mean
nearly unchanged; a diffusion barrier damps the swing instead. Addendum 56's clean-state model learned
how the freshly cleaned probe answers sunlight and temperature. Here, for every day, the residual's
within-day slope on sunlight and on temperature measures how far the probe's answer has moved from
the clean one; fouling should make that slope drift with days since cleaning and reset at a cleaning.

Part 3 splits addendum 57's six-hour jump into its night-time before window and its daytime after
window, to see which one carries its growth with interval length.

Declared in config/bay_reserved_protocol_v2_addendum58.json.
Usage: python -m src.experiments.v2_fouling_diel [--quick] [--workers N]
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.bay_profile_context import ROOT
from src.experiments.v2_fouling_jump import (HOUR, MIN_UNITS, WINDOWS, day_shifted, jumps, valid_pairs,
                                             window)
from src.experiments.v2_fouling_jump import OUT as JUMP_OUT
from src.experiments.v2_fouling_jump import record as jump_record
from src.experiments.v2_fouling_jump import reference as jump_reference
from src.experiments.v2_fouling_reference import (FOLDER as REF_FOLDER, FRESH_DAYS, MIN_HOURS, SITES,
                                                  STALE_DAYS, WARM, calendar, daily, exogenous, prepare,
                                                  residuals, within_slope)
from src.experiments.v2_fouling_reference import OUT as REF_OUT
from src.experiments.v2_logbook import deployed_spans, read_logbook

OUT = ROOT / 'reports/bay_twin/runs/v2_fouling_diel.json'
FOLDER = ROOT / 'reports/bay_twin/runs/v2_fouling_diel'
DRAWS = 1000
SEED = 20260929
ROLES = ('hand-cleaned', 'wiped')
COUPLINGS = ('b_S', 'b_T', 'g_M', 'DN', 'rho_S')
PRIMARY = ('b_S', 'b_T')
DAY_HOURS, NIGHT_HOURS = range(10, 16), range(0, 5)    # 10:00-16:00 and 00:00-05:00 UTC, start labels
MIN_SUN, MIN_T_RANGE, MIN_PRED_RANGE = 100.0, 0.2, 2.0
DAY_NS = 86_400 * 10**9


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def selections(path, node, role, arm=None):
    """A published study's configuration per outer fold, keys back to int."""
    report = json.loads(path.read_text(encoding='utf-8'))
    for e in report['entries']:
        if e['node'] == node and e['role'] == role and (arm is None or e.get('arm') == arm):
            chosen = e['selections'] if 'selections' in e else e['reference_selections']
            return {int(k): v for k, v in chosen.items()}
    raise KeyError((path.name, node, role, arm))


def setup(node, role, exo_all):
    index, x, y, cal, deployed, _ = prepare(node, 'A', role, exo_all)
    frame_exo = exo_all.reindex(index)
    spans = (deployed_spans(read_logbook(), node) if role == 'hand-cleaned'
             else [(index[0], index[-1])])
    temp = x[:, 0]                        # features(): the first column is the probe's own temperature
    return index, x, y, cal, deployed, spans, frame_exo.solar.to_numpy(float), temp


def _sums(g, a, b):
    """Grouped least-squares slope of b on a, and the two standard deviations, from sums."""
    n, sa, sb = g['n_' + a + b], g['s_' + a + b + '_a'], g['s_' + a + b + '_b']
    saa, sbb, sab = g['aa_' + a + b], g['bb_' + a + b], g['ab_' + a + b]
    va, vb, cab = saa - sa * sa / n, sbb - sb * sb / n, sab - sa * sb / n
    return cab / va, va, vb, cab


def couplings(index, res, y, sun, temp, interval, hours):
    """One row per (interval, day) with at least MIN_HOURS residual hours: the five couplings."""
    pred = y - res
    d = pd.DataFrame({'day': index.floor('D'), 'hour': index.hour, 'interval': interval,
                      'dose': hours / 24, 'r': res, 'obs': y, 'pred': pred, 'S': sun / 100.0, 'T': temp})
    d = d[(d.interval >= 0) & np.isfinite(d.r)]
    keys = ['interval', 'day']
    cols = {}
    for a, b in (('S', 'r'), ('T', 'r'), ('pred', 'r'), ('S', 'obs'), ('S', 'pred')):
        ok = np.isfinite(d[a]) & np.isfinite(d[b])
        va, vb = d[a].where(ok), d[b].where(ok)
        cols['n_' + a + b] = ok.astype(float)
        cols['s_' + a + b + '_a'], cols['s_' + a + b + '_b'] = va.fillna(0), vb.fillna(0)
        cols['aa_' + a + b], cols['bb_' + a + b] = (va * va).fillna(0), (vb * vb).fillna(0)
        cols['ab_' + a + b] = (va * vb).fillna(0)
    cols['day_r'] = d.r.where(d.hour.isin(DAY_HOURS))
    cols['night_r'] = d.r.where(d.hour.isin(NIGHT_HOURS))
    s = pd.concat([d[keys + ['dose', 'S', 'T', 'pred']], pd.DataFrame(cols, index=d.index)], axis=1)
    g = s.groupby(keys)
    sums = g[[c for c in cols if not c.endswith('_r')]].sum()
    out = pd.DataFrame({'n': g.size(), 'dose': g.dose.mean(), 'sun_max': g.S.max() * 100,
                        'T_range': g['T'].max() - g['T'].min(),
                        'pred_range': g.pred.max() - g.pred.min(),
                        'day_n': g.day_r.count(), 'night_n': g.night_r.count(),
                        'day_mean': g.day_r.mean(), 'night_mean': g.night_r.mean()})
    with np.errstate(divide='ignore', invalid='ignore'):
        b_s, *_ = _sums(sums, 'S', 'r')
        b_t, *_ = _sums(sums, 'T', 'r')
        g_m, *_ = _sums(sums, 'pred', 'r')
        _, v_s, v_o, c_so = _sums(sums, 'S', 'obs')
        _, v_s2, v_p, c_sp = _sums(sums, 'S', 'pred')
        rho = c_so / np.sqrt(v_s * v_o) - c_sp / np.sqrt(v_s2 * v_p)
    out['b_S'] = b_s.where(out.sun_max >= MIN_SUN)
    out['b_T'] = b_t.where(out.T_range >= MIN_T_RANGE)
    out['g_M'] = g_m.where(out.pred_range >= MIN_PRED_RANGE)
    out['DN'] = (out.day_mean - out.night_mean).where((out.day_n >= 4) & (out.night_n >= 4))
    out['rho_S'] = rho.where(out.sun_max >= MIN_SUN)
    out = out[out.n >= MIN_HOURS].reset_index()
    out['warm'] = out.day.dt.month.isin(WARM).to_numpy(bool)
    for c in COUPLINGS:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out[['interval', 'day', 'dose', 'warm'] + list(COUPLINGS)]


def resets(table, cleanings, pairs, column):
    """Per scorable cleaning: mean coupling on the two whole days before its day minus the two after."""
    marks = sorted(cleanings)
    pos = {c.value: i for i, c in enumerate(marks)}
    t = table.dropna(subset=[column])
    # days as integers (Timestamp.value is always in ns), so keys never depend on a column's unit
    lookup = {(int(i), d.value // DAY_NS): v for i, d, v in zip(t.interval, t.day, t[column])}
    out = []
    for p, c, days in pairs:
        k, day = pos[c.value], c.floor('D').value // DAY_NS
        before = [lookup.get((k - 1, day - j)) for j in (1, 2)]
        after = [lookup.get((k, day + j)) for j in (1, 2)]
        before, after = [v for v in before if v is not None], [v for v in after if v is not None]
        if before and after:
            out.append((c.month in WARM, float(np.mean(before) - np.mean(after))))
    return pd.DataFrame(out, columns=['warm', 'R']).astype({'warm': bool, 'R': float})


def auc(part, column):
    t = part.dropna(subset=[column])
    t = t[(t.dose < FRESH_DAYS) | (t.dose >= STALE_DAYS)]
    label = (t.dose >= STALE_DAYS).astype(int)
    if label.sum() < MIN_UNITS or (1 - label).sum() < MIN_UNITS:
        return None
    return float(roc_auc_score(label, t[column]))


def coupling_stats(table, cleanings, pairs):
    out = {}
    rs = {c: resets(table, cleanings, pairs, c) for c in COUPLINGS}
    for season in ('warm', 'cold', 'all'):
        part = table if season == 'all' else table[table.warm.to_numpy(bool) == (season == 'warm')]
        out[season] = {}
        for c in COUPLINGS:
            p = part.dropna(subset=[c])
            r = rs[c] if season == 'all' else rs[c][rs[c].warm.to_numpy(bool) == (season == 'warm')]
            out[season][c] = {'D': within_slope(p, c) if p.interval.nunique() >= MIN_UNITS else None,
                              'R': float(r.R.mean()) if len(r) >= MIN_UNITS else None,
                              'AUC': auc(p, c), 'days': int(len(p)), 'intervals': int(p.interval.nunique()),
                              'resets': int(len(r)), 'mean': float(p[c].mean()) if len(p) else None}
    return out


def placebo(real, draws):
    out = {}
    for season, block in real.items():
        out[season] = {}
        for c, stats in block.items():
            out[season][c] = {}
            for stat in ('D', 'R', 'AUC'):
                obs = stats[stat]
                null = np.array([d[season][c][stat] for d in draws if d[season][c][stat] is not None], float)
                if obs is None or len(null) < 20:
                    out[season][c][stat] = None
                    continue
                mid = float(np.median(null))
                out[season][c][stat] = {'observed': round(obs, 6),
                                        'p': round(float(np.mean(np.abs(null - mid) >= abs(obs - mid))), 4),
                                        'draws': int(len(null)), 'null_median': round(mid, 6),
                                        'null_p2_5': round(float(np.percentile(null, 2.5)), 6),
                                        'null_p97_5': round(float(np.percentile(null, 97.5)), 6)}
    return out


def real_task(args):
    """Gate G1, then the couplings and statistics on the real calendar."""
    node, role, quick, exo_all = args
    os.environ['OMP_NUM_THREADS'] = '1'
    t0 = time.time()
    index, x, y, cal, deployed, spans, sun, temp = setup(node, role, exo_all)
    chosen = selections(REF_OUT, node, role, arm='A')
    interval, hours = calendar(index, cal, deployed)
    res = residuals(x, y, interval, hours, chosen)
    mine = daily(index, res, interval, hours).sort_values(['interval', 'day']).reset_index(drop=True)
    pub = pd.read_csv(REF_FOLDER / f'daily_{node}_A_{role}.csv.gz', parse_dates=['day'])
    pub = pub.sort_values(['interval', 'day']).reset_index(drop=True)
    same = len(mine) == len(pub) and (mine.interval.to_numpy() == pub.interval.to_numpy()).all()
    gap = float(np.max(np.abs(mine.r.to_numpy() - pub.r.to_numpy()))) if same else float('inf')
    table = couplings(index, res, y, sun, temp, interval, hours)
    base = FOLDER / 'smoke' if quick else FOLDER
    base.mkdir(parents=True, exist_ok=True)
    table.to_csv(base / f'daily_couplings_{node}_{role}.csv.gz', index=False)
    stats = coupling_stats(table, cal, valid_pairs(cal, spans))
    log(f'{node} {role}: gate G1 max |diff| {gap:.2e}; real calendar in {time.time() - t0:.0f} s')
    return {'node': node, 'role': role, 'first': str(index[0].date()), 'last': str(index[-1].date()),
            'cleanings': len(cal), 'gate_G1_max_abs_diff': gap, 'gate_G1_rows': int(len(mine)),
            'days_with_couplings': int(len(table)), 'statistics': stats}


def placebo_chunk(args):
    node, role, exo_all, start, stop, total = args
    os.environ['OMP_NUM_THREADS'] = '1'
    index, x, y, cal, deployed, spans, sun, temp = setup(node, role, exo_all)
    chosen = selections(REF_OUT, node, role, arm='A')
    fakes = day_shifted(cal, index[0], index[-1], np.random.default_rng(SEED), total)
    out = []
    for fake in fakes[start:stop]:
        fi, fh = calendar(index, fake, deployed)
        res = residuals(x, y, fi, fh, chosen)
        out.append(coupling_stats(couplings(index, res, y, sun, temp, fi, fh), fake, valid_pairs(fake, spans)))
    return (node, role), out


def split(index, series, pairs, deployed):
    """Addendum 57's R6 windows, kept apart: the before and after means, with its inclusion rule."""
    (b_lo, b_hi), (a_lo, a_hi), need, _ = WINDOWS['R6']
    out = []
    for p, c, days in pairs:
        k = int((c.floor('h') - index[0]) / HOUR)
        before, after = window(series, deployed, k, b_lo, b_hi), window(series, deployed, k, a_lo, a_hi)
        if before is None or after is None or np.isfinite(before).sum() < need or np.isfinite(after).sum() < need:
            continue
        out.append((days, c.month in WARM, float(np.nanmean(before)), float(np.nanmean(after))))
    return pd.DataFrame(out, columns=['L', 'warm', 'before', 'after']).astype({'warm': bool})


def split_stats(table):
    w = table[table.warm.to_numpy(bool)]
    if len(w) < MIN_UNITS or w.L.nunique() < 2:
        return {'n': int(len(w)), 'before': None, 'after': None}
    xc = w.L - w.L.mean()
    return {'n': int(len(w)),
            'before': float((xc * (w.before - w.before.mean())).sum() / (xc * xc).sum()),
            'after': float((xc * (w.after - w.after.mean())).sum() / (xc * xc).sum())}


def jump_split(args):
    """Part 3 at node1, with gate G2 against addendum 57's published warm R6 slope."""
    quick, exo_all = args
    os.environ['OMP_NUM_THREADS'] = '1'
    node, role = 'node1', 'hand-cleaned'
    index, x, y, cal, deployed, spans = jump_record(node, role, exo_all)
    chosen = selections(JUMP_OUT, node, role)
    r = y - jump_reference(index, x, y, deployed, chosen)
    pairs = valid_pairs(cal, spans)
    published = next(e for e in json.loads(JUMP_OUT.read_text(encoding='utf-8'))['entries']
                     if e['node'] == node and e['role'] == role)['jumps']['R6']['statistics']['warm']['J_slope']
    table = jumps(index, r, pairs, deployed, 'R6')
    w = table[table.warm.to_numpy(bool)]
    xc = w.L - w.L.mean()
    mine = float((xc * (w.J - w.J.mean())).sum() / (xc * xc).sum())
    real = split_stats(split(index, r, pairs, deployed))
    draws = 20 if quick else DRAWS
    null = [split_stats(split(index, r, valid_pairs(f, spans), deployed))
            for f in day_shifted(cal, index[0], index[-1], np.random.default_rng(SEED), draws)]
    out = {'gate_G2_published': published, 'gate_G2_reproduced': mine, 'gate_G2_abs_diff': abs(mine - published),
           'warm': real, 'J_slope_from_split': (real['after'] - real['before'])
           if real['before'] is not None else None, 'placebo': {}}
    for side in ('before', 'after'):
        vals = np.array([d[side] for d in null if d[side] is not None], float)
        obs = real[side]
        if obs is None or len(vals) < 20:
            out['placebo'][side] = None
            continue
        mid = float(np.median(vals))
        out['placebo'][side] = {'observed': round(obs, 6), 'draws': int(len(vals)),
                                'p': round(float(np.mean(np.abs(vals - mid) >= abs(obs - mid))), 4),
                                'null_median': round(mid, 6), 'null_p2_5': round(float(np.percentile(vals, 2.5)), 6),
                                'null_p97_5': round(float(np.percentile(vals, 97.5)), 6)}
    log(f'part 3: gate G2 |diff| {out["gate_G2_abs_diff"]:.2e}')
    return out


def decisions(entries, part3):
    get = {(e['node'], e['role']): e for e in entries}
    out = {}
    for node in SITES:
        hand, wiped = get[(node, 'hand-cleaned')], get[(node, 'wiped')]
        verdict = {'passed': [], 'failed': [], 'not_estimable': [], 'detector': []}
        for c in PRIMARY:
            d, r = hand['placebo']['warm'][c]['D'], hand['placebo']['warm'][c]['R']
            w = wiped['placebo']['warm'][c]['D']
            if d is None or r is None or w is None:
                verdict['not_estimable'].append(c)
                continue
            ok = d['p'] < 0.025 and r['p'] < 0.05 and np.sign(r['observed']) == np.sign(d['observed']) \
                and w['p'] >= 0.05
            verdict['passed' if ok else 'failed'].append(c)
            a = hand['placebo']['warm'][c]['AUC']
            if ok and a is not None and a['observed'] > a['null_p97_5']:
                verdict['detector'].append(c)
        verdict['F1'] = None if not (verdict['passed'] or verdict['failed']) else bool(verdict['passed'])
        out[node] = verdict
    b, a = part3['placebo'].get('before'), part3['placebo'].get('after')
    js = part3['gate_G2_published']
    out['F2'] = bool(b and a and b['p'] < 0.05 and np.sign(b['observed']) == -np.sign(js) and a['p'] >= 0.05)
    tested = [n for n in SITES if out[n]['F1'] is not None]
    out['F4_closed'] = bool(tested) and not any(out[n]['F1'] for n in tested)
    out['F4_tested_nodes'] = tested
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    exo_all = exogenous()
    keys = [(node, role) for node in SITES for role in ROLES]
    total = 20 if a.quick else DRAWS
    chunk = 5 if a.quick else 20
    with mp.get_context('spawn').Pool(a.workers) as pool:
        part3 = pool.apply_async(jump_split, ((a.quick, exo_all),))
        entries = pool.map(real_task, [(*key, a.quick, exo_all) for key in keys])
        bad = [(e['node'], e['role'], e['gate_G1_max_abs_diff']) for e in entries if e['gate_G1_max_abs_diff'] > 1e-6]
        if bad:
            raise SystemExit(f'gate G1 failed: {bad}')
        log('gate G1 passed on every record; placebo next')
        tasks = [(*key, exo_all, s, min(s + chunk, total), total) for key in keys for s in range(0, total, chunk)]
        draws, done = {}, 0
        for key, part in pool.imap_unordered(placebo_chunk, tasks):
            draws.setdefault(key, []).extend(part)
            done += len(part)
            if done % 200 == 0:
                log(f'placebo draws done: {done} of {total * len(keys)}')
        part3 = part3.get()
    if part3['gate_G2_abs_diff'] > 1e-9:
        raise SystemExit(f'gate G2 failed: {part3["gate_G2_abs_diff"]}')
    for e in entries:
        e['placebo'] = placebo(e['statistics'], draws[(e['node'], e['role'])])
    report = {'addendum': 58, 'declaration': 'config/bay_reserved_protocol_v2_addendum58.json',
              'smoke_only': a.quick, 'entries': entries, 'part3_jump_split': part3,
              'decisions': decisions(entries, part3)}
    out = OUT if not a.quick else OUT.with_name('v2_fouling_diel_smoke.json')
    out.write_bytes((json.dumps(report, indent=1, default=str) + '\n').encode('utf-8'))
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
