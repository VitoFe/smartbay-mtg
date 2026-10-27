"""The rate rule's operating grid extended past its edges.

Declared in config/bay_reserved_protocol_v2_addendum87.json before this code was written.

The published sweep (scripts/experiments/exp_warning_curve.py) selects its operating point, the most
precise development point that catches at least half the development crossings, at the largest swept
rate (5.0 umol/kg/h) and the shortest swept cooldown (3 h), so it cannot show where development
precision peaks. This extends the grid to rates of 15 umol/kg/h and cooldowns from 1 h, applies the
same rule to the same development data, and scores the selected point on Warning-2026 exactly as the
published point is scored:

  crossing level  exp_warning_curve.evaluate / sweep, imported and called unchanged (the grid is set
                  on the imported module for the call and restored after it);
  episode level   src.experiments.v2_warning_episodes (series, crossings_of, cluster, one_to_one)
                  with the operating point passed to a copy of its alerts_of, checked against
                  alerts_of itself at the published point;
  leads           both definitions of the published tables: the crossing-level median lead of the
                  sweep, and the leads of the one-to-one matches at the 12 h clustering gap (with the
                  strictly positive matching of scripts/v2_observability_reanalysis.leads).

Nothing on 2026 enters the selection. On 2026 only the published curve and the published point's
episode scores are recomputed, as reproduction gates, and the one point the extended curve selects is
scored. Oxygen quantities stay in umol/kg; each rate is also given in mg/l/h by src.units.factor().

Gates (a failure stops the script before any extended point is read):
  A87-G0  the declaration's digest file matches the declaration;
  A87-G1  the published development and 2026 curves and the published selection are reproduced;
  A87-G2  the published episode scores and leads are reproduced at the published point, against the
          values runs/v2_warning_episodes.json and runs/v2_observability_reanalysis.json held for it
          (recorded below, PUBLISHED_EPISODES and PUBLISHED_LEADS: both files now read the selected
          point from this script's summary, so the gate reads neither);
  A87-G3  the extended curve's rows at the 90 published points equal the published rows.

Writes runs/v2_warning_grid_ext.json and runs/v2_warning_grid_ext/.
Usage: python -m scripts.v2_warning_grid_ext
"""
import hashlib
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import v2_warning_episodes as W  # noqa: E402
from src.units import factor  # noqa: E402

DECLARATION = 'config/bay_reserved_protocol_v2_addendum87.json'
DIGEST_FILE = ROOT / 'reports/bay_twin/protocol_v2_addendum87_declaration.sha256'
RUNS = ROOT / 'reports/bay_twin/runs'
PUBLISHED = RUNS / 'v2_warning_curve'
OUT = RUNS / 'v2_warning_grid_ext'
SUMMARY = RUNS / 'v2_warning_grid_ext.json'

RATES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
         8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
CONSEC = [1, 2, 3]
COOLDOWN = [1, 2, 3, 6, 12]
# the periods of exp_warning_curve's main block
DEVELOPMENT = (pd.Timestamp('2024-07-01', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC'))
TEST = (pd.Timestamp('2026-01-01', tz='UTC'), pd.Timestamp('2026-06-20', tz='UTC'))
PUBLISHED_POINT = (5.0, 2, 3)
FIXED_RULE = (1.5, 2, 6)
KEY = ['rate', 'consec', 'cooldown']
COUNTS = ['alerts', 'crossings']
#: The published point's Warning-2026 episode scores, as runs/v2_warning_episodes.json held them (by_gap,
#: ratios at three decimals), and its leads, as runs/v2_observability_reanalysis.json held them
#: (F3_warning_leads), before both moved to the selected point. A87-G2 reproduces them.
_EPISODES_PUBLISHED = {'episodes': 7, 'crossings': 12, 'alerts': 40, 'matched_one_to_one': 6,
                       'episode_recall': 0.857, 'episode_precision': 0.15}
PUBLISHED_EPISODES = {'6h': dict(_EPISODES_PUBLISHED),
                      '12h': dict(_EPISODES_PUBLISHED, episode_recall_exact_95=[0.421, 0.996],
                                  episode_precision_exact_95=[0.057, 0.298]),
                      '24h': dict(_EPISODES_PUBLISHED)}
PUBLISHED_LEADS = {
    'leads_h_sorted': [0.0, 1.0, 2.0, 2.0, 4.0, 4.0], 'median_lead_h': 2.0, 'range_h': [0.0, 4.0],
    'positive_lead_matched': 5, 'strictly_positive_matching_leads_h': [1.0, 2.0, 2.0, 4.0, 4.0],
    'matches': [
        {'sensor': 'TBaiaP', 'lead_h': 0.0, 'alert': '2026-06-04 02:00:00+00:00',
         'episode_onset': '2026-06-04 02:00:00+00:00'},
        {'sensor': 'TBaiaP', 'lead_h': 1.0, 'alert': '2026-04-21 05:00:00+00:00',
         'episode_onset': '2026-04-21 06:00:00+00:00'},
        {'sensor': 'TBaiaP', 'lead_h': 4.0, 'alert': '2026-02-10 21:00:00+00:00',
         'episode_onset': '2026-02-11 01:00:00+00:00'},
        {'sensor': 'TMitiliE', 'lead_h': 2.0, 'alert': '2026-02-13 01:00:00+00:00',
         'episode_onset': '2026-02-13 03:00:00+00:00'},
        {'sensor': 'TMitiliE', 'lead_h': 2.0, 'alert': '2026-03-17 22:00:00+00:00',
         'episode_onset': '2026-03-18 00:00:00+00:00'},
        {'sensor': 'TMitiliE', 'lead_h': 4.0, 'alert': '2026-02-17 21:00:00+00:00',
         'episode_onset': '2026-02-18 01:00:00+00:00'}]}


def curve_module():
    """scripts/experiments/exp_warning_curve.py, loaded as published (its main block does not run)."""
    spec = importlib.util.spec_from_file_location('exp_warning_curve',
                                                  ROOT / 'scripts/experiments/exp_warning_curve.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EWC = curve_module()


def sweep(data, period, label, rates, consec, cooldowns):
    """exp_warning_curve.sweep over the given grid, with the matched and caught counts added."""
    saved = (EWC.RATES, EWC.CONSEC, EWC.COOLDOWN)
    try:
        EWC.RATES, EWC.CONSEC, EWC.COOLDOWN = list(rates), list(consec), list(cooldowns)
        frame = EWC.sweep(data, period[0], period[1], label)
    finally:
        EWC.RATES, EWC.CONSEC, EWC.COOLDOWN = saved
    for tag in ('causal', 'sym'):
        matched = (frame[f'precision_{tag}'] * frame.alerts).fillna(0.0)
        caught = (frame[f'recall_{tag}'] * frame.crossings).fillna(0.0)
        for name, raw in ((f'matched_{tag}', matched), (f'caught_{tag}', caught)):
            if (raw - raw.round()).abs().max() > 1e-6:
                raise SystemExit(f'{name}: a count is not an integer')
            frame[name] = raw.round().astype(int)
    return frame


def published_selection(dev):
    """The published rule, as exp_warning_curve's main block applies it."""
    cand = dev[(dev.recall_causal >= 0.5) & (dev.alerts > 0)]
    return cand.sort_values('precision_causal', ascending=False).head(5).iloc[0]


def select(dev):
    """The published rule with the declared tie order: precision, recall, fewer alerts, then the
    lower rate, fewer consecutive hours and the shorter cooldown (values compared at nine decimals)."""
    cand = dev[(dev.recall_causal >= 0.5) & (dev.alerts > 0)].copy()
    cand['p9'] = cand.precision_causal.round(9)
    cand['r9'] = cand.recall_causal.round(9)
    order = cand.sort_values(['p9', 'r9', 'alerts', 'rate', 'consec', 'cooldown'],
                             ascending=[False, False, True, True, True, True], kind='mergesort')
    top = order.iloc[0]
    ties = order[(order.p9 == top.p9)]
    return top, ties, len(cand)


def point_of(row):
    return (float(row.rate), int(row.consec), int(row.cooldown))


def row_at(frame, point):
    rate, consec, cd = point
    hit = frame[(frame.rate == rate) & (frame.consec == consec) & (frame.cooldown == cd)]
    if len(hit) != 1:
        raise SystemExit(f'no unique row at {point}')
    return hit.iloc[0]


def same_frames(mine, published, what):
    """Every row and column of `published` reproduced (counts exactly, other values to 1e-9)."""
    a = mine.set_index(KEY).sort_index()
    b = published.set_index(KEY).sort_index()
    if list(a.index) != list(b.index):
        raise SystemExit(f'{what}: the grid differs')
    worst = 0.0
    for col in b.columns:
        if col == 'period':
            if not (a[col] == b[col]).all():
                raise SystemExit(f'{what}: period label differs')
            continue
        x, y = a[col].astype(float).to_numpy(), b[col].astype(float).to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(x - y))
        if np.isnan(diff).any():
            raise SystemExit(f'{what}: {col} missing on one side')
        tol = 0.0 if col in COUNTS else 1e-9
        if col == 'weeks':
            tol = 0.05 + 1e-9           # the published column is rounded to one decimal
        if diff.max() > tol:
            raise SystemExit(f'{what}: {col} differs by {diff.max()}')
        if col != 'weeks':
            worst = max(worst, float(diff.max()))
    return {'rows': int(len(b)), 'columns': [c for c in b.columns if c != 'period'],
            'max_abs_difference': worst}


# ------------------------------------------------------------------------------ episode level --
def alerts_at(oxygen, rate, consec, cooldown):
    """v2_warning_episodes.alerts_of with the operating point passed in."""
    d = -(oxygen - oxygen.shift(1))
    cond = d > rate
    for k in range(1, consec):
        cond = cond & (d.shift(k) > rate)
    out, last = [], None
    for t, flag in cond.dropna().items():
        if flag and (last is None or (t - last) >= pd.Timedelta(hours=cooldown)):
            out.append(t)
            last = t
    return out


def exact(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def episodes(series, point):
    """The episode scores of v2_warning_episodes.main and the leads of v2_observability_reanalysis."""
    rate, consec, cd = point
    cross = {n: W.crossings_of(f.oxygen) for n, f in series.items()}
    alerts = {n: alerts_at(f.oxygen, rate, consec, cd) for n, f in series.items()}
    total_cross = sum(len(v) for v in cross.values())
    total_alerts = sum(len(v) for v in alerts.values())
    by_gap, matches, strict = {}, [], []
    for gap in W.GAPS_H:
        n_ep, matched = 0, 0
        for name in series:
            eps = W.cluster(cross[name], gap)
            n_ep += len(eps)
            m = W.one_to_one(alerts[name], eps)
            matched += len(m)
            if gap == 12:
                matches.extend(dict(sensor=name.split('_')[0], **x) for x in m)
                cand = sorted(((ep[0] - a).total_seconds() / 3600.0, ai, ei)
                              for ei, ep in enumerate(eps) for ai, a in enumerate(alerts[name])
                              if 0 < (ep[0] - a).total_seconds() / 3600.0 <= W.MATCH_H)
                used_a, used_e = set(), set()
                for lead, ai, ei in cand:
                    if ai in used_a or ei in used_e:
                        continue
                    used_a.add(ai)
                    used_e.add(ei)
                    strict.append(dict(sensor=name.split('_')[0], lead_h=round(lead, 1),
                                       alert=str(alerts[name][ai]), episode_onset=str(eps[ei][0])))
        by_gap[f'{gap}h'] = {'episodes': n_ep, 'crossings': total_cross, 'alerts': total_alerts,
                             'matched_one_to_one': matched,
                             'episode_recall': matched / n_ep if n_ep else None,
                             'episode_precision': matched / total_alerts if total_alerts else None}
    g = by_gap['12h']
    g['episode_recall_exact_95'] = exact(g['matched_one_to_one'], g['episodes'])
    g['episode_precision_exact_95'] = exact(g['matched_one_to_one'], g['alerts'])
    lv = sorted(m['lead_h'] for m in matches)
    positive = sum(x > 0 for x in lv)
    leads = {'matches': matches, 'leads_h_sorted': lv,
             'median_lead_h': float(np.median(lv)) if lv else None,
             'range_h': [min(lv), max(lv)] if lv else None,
             'strictly_positive_lead': {'matched': positive, 'episodes': g['episodes'],
                                        'value': positive / g['episodes'] if g['episodes'] else None},
             'strictly_positive_matching': {'matched': len(strict),
                                            'leads_h': sorted(m['lead_h'] for m in strict),
                                            'recall': len(strict) / g['episodes'] if g['episodes'] else None}}
    return {'by_gap': by_gap, 'leads_12h_gap': leads,
            'alerts_by_sensor': {n.split('_')[0]: len(v) for n, v in alerts.items()},
            'crossings_by_sensor': {n.split('_')[0]: len(v) for n, v in cross.items()}}, alerts


def gate_episodes(series, ep):
    """A87-G2 at the published point, against the values recorded in PUBLISHED_EPISODES and PUBLISHED_LEADS."""
    for name, frame in series.items():
        if alerts_at(frame.oxygen, *PUBLISHED_POINT) != W.alerts_of(frame.oxygen, PUBLISHED_POINT):
            raise SystemExit(f'A87-G2 failed: alerts differ from v2_warning_episodes.alerts_of at {name}')
    pub = PUBLISHED_EPISODES
    for gap, block in pub.items():
        mine = ep['by_gap'][gap]
        for k in ('episodes', 'crossings', 'alerts', 'matched_one_to_one'):
            if mine[k] != block[k]:
                raise SystemExit(f'A87-G2 failed: {gap} {k} {mine[k]} != {block[k]}')
        for k in ('episode_recall', 'episode_precision'):
            if round(mine[k], 3) != block[k]:
                raise SystemExit(f'A87-G2 failed: {gap} {k}')
        for k in ('episode_recall_exact_95', 'episode_precision_exact_95'):
            if k in block and [round(x, 3) for x in mine[k]] != block[k]:
                raise SystemExit(f'A87-G2 failed: {gap} {k}')
    f3 = PUBLISHED_LEADS
    lv = ep['leads_12h_gap']
    checks = [(lv['leads_h_sorted'], f3['leads_h_sorted']), (lv['median_lead_h'], f3['median_lead_h']),
              (lv['range_h'], f3['range_h']),
              (lv['strictly_positive_lead']['matched'], f3['positive_lead_matched']),
              (lv['strictly_positive_matching']['leads_h'], f3['strictly_positive_matching_leads_h']),
              ([dict(m) for m in lv['matches']], f3['matches'])]
    for mine, theirs in checks:
        if mine != theirs:
            raise SystemExit(f'A87-G2 failed: leads {mine} != {theirs}')
    return {'alerts_equal_alerts_of_on_sensors': len(series),
            'by_gap_reproduced': sorted(pub), 'leads_reproduced': True,
            'reference': 'the published point\'s values recorded in the script (PUBLISHED_EPISODES, PUBLISHED_LEADS)'}


# ------------------------------------------------------------------------------------ output --
def r6(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    return x


def crossing_scores(row):
    keys = ['alerts', 'crossings', 'matched_causal', 'caught_causal', 'precision_causal', 'recall_causal',
            'unmatched_per_week_causal', 'precision_sym', 'recall_sym', 'unmatched_per_week_sym',
            'median_lead_h', 'weeks']
    return {k: r6(row[k]) for k in keys}


def describe_point(point):
    rate, consec, cd = point
    return {'rate_umol_per_kg_per_h': rate, 'rate_mg_per_l_per_h': r6(rate * factor()),
            'consecutive_hours': consec, 'cooldown_h': cd}


def write_lf(path, text):
    path.write_bytes(text.encode('utf-8'))


def write_csv(frame, path):
    buf = io.StringIO()
    frame.to_csv(buf, index=False, float_format='%.6f', lineterminator='\n')
    write_lf(path, buf.getvalue())


def main():
    t0 = time.time()
    decl = (ROOT / DECLARATION).read_bytes()
    digest = hashlib.sha256(decl).hexdigest()
    if DIGEST_FILE.read_text().strip() != digest:
        raise SystemExit('A87-G0 failed: the declaration differs from its recorded digest')
    print('A87-G0 passed: declaration digest', digest[:12])

    data = EWC.load()
    print(f'oxygen sensors with data: {len(data)}')

    # A87-G1: the published curves and selection
    pub_dev = pd.read_csv(PUBLISHED / 'warning_curve_development.csv')
    pub_test = pd.read_csv(PUBLISHED / 'warning_curve_reserved2026.csv')
    pub_sel = json.loads((PUBLISHED / 'warning_selection.json').read_text())
    pub_grid = (sorted(pub_dev.rate.unique()), sorted(pub_dev.consec.unique()), sorted(pub_dev.cooldown.unique()))
    dev90 = sweep(data, DEVELOPMENT, 'development', *pub_grid)
    test90 = sweep(data, TEST, 'reserved_2026', *pub_grid)
    g1_dev = same_frames(dev90[pub_dev.columns], pub_dev, 'A87-G1 development')
    g1_test = same_frames(test90[pub_test.columns], pub_test, 'A87-G1 2026')
    sel_pub = published_selection(dev90)
    if point_of(sel_pub) != (pub_sel['rate'], pub_sel['consec'], pub_sel['cooldown']) or \
            point_of(sel_pub) != PUBLISHED_POINT:
        raise SystemExit('A87-G1 failed: the published selection is not reproduced')
    if point_of(select(dev90)[0]) != PUBLISHED_POINT:
        raise SystemExit('A87-G1 failed: the declared tie order changes the published selection')
    pub_row26 = row_at(test90, PUBLISHED_POINT)
    for k, v in pub_sel['reserved_2026'].items():
        if v is not None and abs(float(pub_row26[k]) - v) > 1e-9:
            raise SystemExit(f'A87-G1 failed: warning_selection.json {k}')
    print(f'A87-G1 passed ({time.time() - t0:.0f} s)')

    # A87-G2: the published episode scores and leads at the published point
    series = W.series()
    ep_pub, _ = episodes(series, PUBLISHED_POINT)
    g2 = gate_episodes(series, ep_pub)
    print('A87-G2 passed')

    # the extended development curve
    dev = sweep(data, DEVELOPMENT, 'development', RATES, CONSEC, COOLDOWN)
    g3 = same_frames(dev[dev.rate.isin(pub_grid[0]) & dev.cooldown.isin(pub_grid[2])][pub_dev.columns],
                     pub_dev, 'A87-G3')
    print(f'A87-G3 passed; extended development curve: {len(dev)} points ({time.time() - t0:.0f} s)')

    top, ties, n_cand = select(dev)
    selected = point_of(top)
    moved = selected != PUBLISHED_POINT
    borderline = dev[(dev.recall_causal - 0.5).abs() < 1e-9]
    secondary = {}
    for name, mask in (('extended_rates_published_cooldowns', dev.cooldown.isin(pub_grid[2])),
                       ('published_rates_extended_cooldowns', dev.rate.isin(pub_grid[0]))):
        s_top, _, s_n = select(dev[mask])
        secondary[name] = {'point': describe_point(point_of(s_top)), 'candidates': s_n,
                           'development': crossing_scores(s_top)}

    # 2026: the selected point (and the published one beside it, from the gate)
    test_sel = sweep(data, TEST, 'reserved_2026', [selected[0]], [selected[1]], [selected[2]]).iloc[0]
    if moved:
        ep_sel, alerts_sel = episodes(series, selected)
    else:
        ep_sel = ep_pub
    print(f'selected {selected} (moved: {moved}) ({time.time() - t0:.0f} s)')

    # the development frontier: for each rate, the most precise point meeting the constraint
    frontier = []
    for rate in RATES:
        sub = dev[(dev.rate == rate)]
        best = select(sub)[0] if ((sub.recall_causal >= 0.5) & (sub.alerts > 0)).any() else None
        frontier.append({'rate_umol_per_kg_per_h': rate, 'rate_mg_per_l_per_h': r6(rate * factor()),
                         'best_meeting_constraint': None if best is None else
                         {'consec': int(best.consec), 'cooldown': int(best.cooldown),
                          'precision_causal': r6(best.precision_causal),
                          'recall_causal': r6(best.recall_causal), 'alerts': int(best.alerts)}})

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ['period', 'rate', 'consec', 'cooldown', 'alerts', 'crossings', 'matched_causal', 'caught_causal',
            'precision_causal', 'recall_causal', 'unmatched_per_week_causal', 'matched_sym', 'caught_sym',
            'precision_sym', 'recall_sym', 'unmatched_per_week_sym', 'median_lead_h', 'weeks']
    write_csv(dev[cols], OUT / 'warning_curve_development_ext.csv')
    rows26 = pd.DataFrame([pub_row26, test_sel])[cols]
    rows26.insert(1, 'point', ['published', 'selected_on_extended_grid'])
    write_csv(rows26, OUT / 'reserved2026_points.csv')
    write_lf(OUT / 'episodes_2026.json', json.dumps(r6({'published_point': ep_pub,
                                                        'selected_point': ep_sel}), indent=1) + '\n')

    summary = {
        'addendum': 87, 'declared_in': DECLARATION, 'declaration_sha256': digest,
        'what': 'the rate rule\'s operating grid extended past its edges, selected on development by the '
                'published rule, the selected point scored on Warning-2026 as the published point',
        'units': 'rates in umol/kg/h (the sondes\' column), each also in mg/l/h by c = '
                 f'{factor()}; alerts, crossings and episodes are counts',
        'grid': {'rates_umol_per_kg_per_h': RATES, 'consecutive_hours': CONSEC, 'cooldowns_h': COOLDOWN,
                 'points': int(len(dev))},
        'rule': 'highest development causal precision among points with development causal recall >= 0.5 '
                'and at least one alert; ties: higher recall, fewer alerts, lower rate, fewer consecutive '
                'hours, shorter cooldown (values at nine decimals)',
        'gates': {'A87-G0': 'passed', 'A87-G1': {'development': g1_dev, 'reserved_2026': g1_test,
                                                  'published_selection': describe_point(PUBLISHED_POINT)},
                  'A87-G2': g2, 'A87-G3': g3},
        'development': {
            'candidates_meeting_constraint': n_cand,
            'recall_within_1e-9_of_constraint': int(len(borderline)),
            'selected': describe_point(selected),
            'selected_scores': crossing_scores(top),
            'ties_at_selected_precision': [describe_point(point_of(r)) for _, r in ties.iterrows()],
            'published_point_scores': crossing_scores(row_at(dev, PUBLISHED_POINT)),
            'fixed_rule_scores': crossing_scores(row_at(dev, FIXED_RULE)),
            'moved': moved,
            'selected_rate_is_grid_maximum': selected[0] == max(RATES),
            'selected_cooldown_is_grid_minimum': selected[2] == min(COOLDOWN),
            'selected_rate_was_published_maximum': selected[0] == max(pub_grid[0]),
            'frontier_by_rate': frontier,
            'secondary_selections_development_only': secondary,
            'curve': [{k: r6(r[k]) for k in ['rate', 'consec', 'cooldown', 'alerts', 'matched_causal',
                                              'caught_causal', 'precision_causal', 'recall_causal',
                                              'unmatched_per_week_causal', 'median_lead_h']}
                      for _, r in dev.iterrows()]},
        'warning_2026': {
            'crossing_level': {'published_point': crossing_scores(pub_row26),
                               'selected_point': crossing_scores(test_sel)},
            'episode_level_12h_gap': {
                label: {'alerts': e['by_gap']['12h']['alerts'], 'episodes': e['by_gap']['12h']['episodes'],
                        'matched_one_to_one': e['by_gap']['12h']['matched_one_to_one'],
                        'episode_recall': r6(e['by_gap']['12h']['episode_recall']),
                        'episode_precision': r6(e['by_gap']['12h']['episode_precision']),
                        'episode_recall_exact_95': r6(e['by_gap']['12h']['episode_recall_exact_95']),
                        'episode_precision_exact_95': r6(e['by_gap']['12h']['episode_precision_exact_95']),
                        'by_gap_matched': {g: v['matched_one_to_one'] for g, v in e['by_gap'].items()},
                        'by_gap_episodes': {g: v['episodes'] for g, v in e['by_gap'].items()}}
                for label, e in (('published_point', ep_pub), ('selected_point', ep_sel))},
            'leads': {
                label: {'crossing_level_median_lead_h': r6(row['median_lead_h']),
                        'crossing_level_definition': 'exp_warning_curve: for each crossing the lead of the '
                                                     'earliest alert within 24 h before it; the median over '
                                                     'crossings at each sensor, then over sensors',
                        'episode_leads_h_sorted': r6(e['leads_12h_gap']['leads_h_sorted']),
                        'episode_median_lead_h': r6(e['leads_12h_gap']['median_lead_h']),
                        'episode_lead_definition': 'one-to-one matches at the 12 h clustering gap, smallest '
                                                   'non-negative lead within 24 h, pooled median',
                        'episodes_with_strictly_positive_lead': e['leads_12h_gap']['strictly_positive_lead']['matched'],
                        'strictly_positive_matching_leads_h': r6(e['leads_12h_gap']['strictly_positive_matching']['leads_h'])}
                for label, row, e in (('published_point', pub_row26, ep_pub),
                                      ('selected_point', test_sel, ep_sel))}},
        'outputs': ['runs/v2_warning_grid_ext/warning_curve_development_ext.csv',
                    'runs/v2_warning_grid_ext/reserved2026_points.csv',
                    'runs/v2_warning_grid_ext/episodes_2026.json'],
        'runtime_s': round(time.time() - t0, 1)}
    write_lf(SUMMARY, json.dumps(r6(summary), indent=1) + '\n')
    print(json.dumps(r6({k: summary['development'][k] for k in
                         ('selected', 'moved', 'selected_scores', 'published_point_scores',
                          'selected_rate_is_grid_maximum', 'selected_cooldown_is_grid_minimum')}), indent=1))
    print('wrote', SUMMARY, f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
