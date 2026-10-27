"""Addendum 59 part F: three re-analyses of the observability results, from saved predictions and
the inputs they were made from. Nothing is refitted and no configuration is selected.

  F1  near-bottom oxygen (runs/v2_bottom_aou): which held-station casts had the moored network's
      mean state, i.e. a network value joined within the 3 h tolerance of
      scripts/experiments/exp_bottom.build(); the network record starts on 2024-07-24 and every
      earlier cast reads the training mean in its place. The mean-state arms against the naive
      assumption (bottom = surface) on the covered casts, by the nested macro, with the paired
      bootstrap over cast dates and the delete-one-date jackknife (scripts.v2_summarize_aou.contrast,
      the helper of the published contrasts). The same on the dark casts, and on the 2026 casts of
      the chronological protocol, as description.
  F2  placement, vertical transfer (runs/v2_placement): exp_placement.vertical() replayed with the
      same seed, keeping each cast's error; every donor configuration against the naive baseline on
      the same held station (paired), the mean over a held station's donor sets, each single donor
      paired on the held stations it serves, and the mean by number of donors. Metric: the flat
      mean absolute error over a held station's casts, then the mean over held stations, as
      published. Intervals over cast dates where a contrast has at least five.
  F3  warning leads (src.experiments.v2_warning_episodes): the leads of the one-to-one matches on
      Warning-2026 at the 12 h clustering gap, their median and range, and recall counting only a
      strictly positive lead, also with the matching itself restricted to positive leads.

Also written, as description and not declared: the extended warning grid and its selection rule
read back from runs/v2_warning_grid_ext (the operating point F3 uses is read from there); both salinity and temperature windows of the 2026 episodes
(the six hours ending at the first crossing, runs/v2_decline_types.json, and the window running on
to the last crossing, runs/v2_warning_episodes.json); the consumption rate a decline with no
salinity change implies, beside the median summer night at 1 m (runs/v2_daily_cycle.json); and
every near-bottom arm on the held-station casts whose surface and bottom differ by at least
20 umol/L.

Gates: the published near-bottom contrasts and the placement MAE are reproduced before anything
else is computed.

Usage: python -m scripts.v2_observability_reanalysis
"""
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from scripts.v2_summarize_aou import cast_meta, contrast
from scripts.v2_summarize_bottom import cast_errors
from src.experiments.v2_field_reserved2026 import DRAWS, RNG_SEED, macro

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_observability_reanalysis.json'
BOTTOM = RUNS / 'v2_bottom_aou/bottom'
BOTTOM_REF = RUNS / 'v2_bottom_aou/bottom_ref'
PLACEMENT = RUNS / 'v2_placement'
MIN_DATES = 5
INTERVAL = f'paired bootstrap over cast dates, each drawn date relabelled, {DRAWS} draws, seed {RNG_SEED}; ' \
           'delete-one-date jackknife-t'


def entry(arm, base, **extra):
    c = contrast(arm, base)
    out = dict(mae=round(macro(arm), 4), naive=round(macro(base), 4), diff=c['point'], casts=int(len(arm)),
               dates=c['dates'], **extra)
    if c['dates'] >= MIN_DATES:
        out.update(ci=c['ci'], excludes_zero=c['excludes_zero'], jackknife_t=c['jackknife_t'],
                   jackknife_excludes_zero=c['jackknife_excludes_zero'],
                   robust=bool(c['excludes_zero'] and c['jackknife_excludes_zero']))
    else:
        out['interval'] = f'not reported: fewer than {MIN_DATES} dates'
    return out


# ------------------------------------------------------------------------------------------ F1 --
def near_bottom():
    from src.experiments import v2_bottom_aou as vr
    with tempfile.TemporaryDirectory() as tmp:
        eb, _ = vr.import_experiments(Path(tmp))
        d = eb.build()
    moored = list(eb.MOORED)
    design = d.set_index('cast_id')
    covered_any = design[moored].notna().any(axis=1)
    P = pd.read_csv(BOTTOM / 'predictions.csv.gz', dtype={'fold': str})
    R = pd.read_csv(BOTTOM_REF / 'predictions.csv.gz', dtype={'fold': str})
    meta = cast_meta()
    published = {(e['protocol'], e['model']): e for e in json.loads((BOTTOM / 'contrasts.json').read_text())}
    published.update({(e['protocol'], e['model']): e
                      for e in json.loads((BOTTOM_REF / 'contrasts.json').read_text())['against_naive']})
    out = {'what': 'near-bottom oxygen, mean-state arms (regime C2_moored_only) against the naive '
                   'assumption that the bottom equals the surface',
           'network_record': {'source': 'reports/bay_twin/hourly_observations.csv.gz',
                              'join': 'merge_asof on the cast start, nearest, tolerance 3 h (exp_bottom.build)',
                              'features': moored,
                              'dark_casts_read': 'the training mean of each feature (v2_bottom_aou.impute)'},
           'metric': 'nested macro: casts within (fold, station), stations within fold, folds; under the '
                     'held-station protocol the fold is the held station',
           'interval': INTERVAL, 'gates': {}, 'coverage': {}, 'contrasts': {}, 'version_contrasts': {}}
    arms = {}
    for pr in ('P1_held_station', 'P2_chronological'):
        sub, subr = P[P.protocol == pr], R[R.protocol == pr]
        base = cast_errors(sub[sub.model == 'B_surface'], meta)
        arms[pr] = {'B_surface': base}
        for m in sorted(set(sub.model) - {'B_surface'}):
            arms[pr][m] = cast_errors(sub[sub.model == m], meta)
        for m in sorted(m for m in set(subr.model) if m.rsplit('|', 1)[1] in ('aou_ref', 'aou_bed_ref')):
            arms[pr][m] = cast_errors(subr[subr.model == m], meta)
        # gate: the published contrasts of the mean-state arms, all casts
        for m, a in arms[pr].items():
            if not m.startswith('C2_moored_only'):
                continue
            c, p = contrast(a, base), published[(pr, m)]
            ok = c['point'] == p['diff'] and c['ci'] == p['ci'] and c['jackknife_t'] == p['jackknife_t']
            out['gates'][f'{pr}|{m}'] = bool(ok)
        ids = base.cast_id
        cov = covered_any.reindex(ids)
        if cov.isna().any():
            raise SystemExit(f'{pr}: a scored cast is missing from the design')
        frame = design.loc[ids]
        days = base.set_index('cast_id').day
        out['coverage'][pr] = {
            'casts': int(len(ids)), 'covered_any_feature': int(cov.sum()),
            'covered_share': round(float(cov.mean()), 4),
            'covered_network_temperature': int(frame.net_T.notna().sum()),
            'covered_network_oxygen': int(frame.net_DO.notna().sum()),
            'covered_all_features': int(frame[moored].notna().all(axis=1).sum()),
            'dates': int(days.nunique()), 'covered_dates': int(days[cov.values].nunique()),
            'first_covered_cast': str(frame.t_utc[cov.values].min()),
            'last_dark_cast': str(frame.t_utc[~cov.values].max()) if (~cov).any() else None,
            'by_station': {s: {'casts': int(len(g)), 'covered': int(g.sum())}
                           for s, g in cov.groupby(frame.station.values)}}
    if not all(out['gates'].values()):
        raise SystemExit(f'near-bottom gate failed: {out["gates"]}')
    covered_ids = set(covered_any[covered_any].index)
    for pr, block in arms.items():
        base = block['B_surface']
        for subset, keep in (('covered', True), ('dark', False)):
            b = base[base.cast_id.isin(covered_ids) == keep]
            if not len(b):
                continue
            rows = {}
            for m, a in block.items():
                if not m.startswith('C2_moored_only'):
                    continue
                a = a[a.cast_id.isin(covered_ids) == keep]
                rows[m] = entry(a, b, reads_bed_temperature_and_salinity=m.endswith('|aou'))
            out['contrasts'][f'{pr}|{subset}'] = rows
            # the second-version arms against the first version they replace, on the same casts
            ver = {}
            for mdl in ('ridge', 'lightgbm'):
                first = block[f'C2_moored_only|{mdl}|aou']
                first = first[first.cast_id.isin(covered_ids) == keep]
                for t in ('aou_ref', 'aou_bed_ref'):
                    a = block[f'C2_moored_only|{mdl}|{t}']
                    a = a[a.cast_id.isin(covered_ids) == keep]
                    c = contrast(a, first)
                    ver[f'C2_moored_only|{mdl}|{t} - C2_moored_only|{mdl}|aou'] = {
                        'diff': c['point'], 'ci': c['ci'], 'jackknife_t': c['jackknife_t'],
                        'robust': bool(c['excludes_zero'] and c['jackknife_excludes_zero']), 'dates': c['dates']}
            out['version_contrasts'][f'{pr}|{subset}'] = ver
    cov_rows = out['contrasts']['P1_held_station|covered']
    honest = {m: r for m, r in cov_rows.items() if not r['reads_bed_temperature_and_salinity']}
    out['PF1'] = {'prediction': 'On the covered casts no mean-state model beats the naive assumption with an '
                                'interval excluding zero.',
                  'better_with_interval_excluding_zero': sorted(m for m, r in cov_rows.items()
                                                                if r['diff'] < 0 and r['excludes_zero']),
                  'worse_with_interval_excluding_zero': sorted(m for m, r in cov_rows.items()
                                                               if r['diff'] > 0 and r['excludes_zero']),
                  'best_reading_nothing_at_the_bed': min(honest, key=lambda m: honest[m]['mae']),
                  'largest_gain_admitted_by_an_interval': round(-min(r['ci'][0] for r in cov_rows.values()), 4)}
    out['PF1']['held'] = not out['PF1']['better_with_interval_excluding_zero']
    naive_covered = next(iter(cov_rows.values()))['naive']
    out['PF1']['naive_covered'] = naive_covered
    out['PF1']['naive_less_largest_admitted_gain'] = round(naive_covered
                                                           - out['PF1']['largest_gain_admitted_by_an_interval'], 4)

    # stratified casts (description): every arm on the held-station casts with |surface - bottom| >= 20
    strat_ids = set(P[(P.protocol == 'P1_held_station') & (P.model == 'B_surface') & (P.abs_delta >= 20)].cast_id)
    base = arms['P1_held_station']['B_surface']
    b = base[base.cast_id.isin(strat_ids)]
    out['stratified'] = {'what': 'held-station casts whose surface and near-bottom oxygen differ by at least 20 '
                                 'umol/L; the naive error there is that difference, by construction',
                         'casts': int(len(b)), 'naive_flat_mean': round(float(b.e.mean()), 4),
                         'naive_nested': round(macro(b), 4), 'arms': {}}
    for m, a in arms['P1_held_station'].items():
        if m == 'B_surface':
            continue
        a = a[a.cast_id.isin(strat_ids)]
        out['stratified']['arms'][m] = entry(a, b, reads_bed_temperature_and_salinity=m.endswith('|aou'))
    return out


# ------------------------------------------------------------------------------------------ F2 --
def placement():
    from scripts.experiments import exp_placement as ep
    pairs = ep.build_pairs(ep.load())
    rng = np.random.default_rng(20260919)          # the seed of exp_placement.RNG, which only vertical() draws
    stations = sorted(pairs.station.unique())
    rows, errs = [], []
    for held in stations:                          # exp_placement.vertical(), keeping each cast's error
        te = pairs[pairs.station == held]
        if len(te) < 10:
            continue
        donors_all = [s for s in stations if s != held]
        for k in range(0, len(donors_all) + 1):
            combos = list(itertools.combinations(donors_all, k))
            if len(combos) > 12:
                combos = [tuple(donors_all[i] for i in rng.choice(len(donors_all), k, replace=False))
                          for _ in range(12)]
            for ci, combo in enumerate(combos):
                preds = []
                for _, r in te.iterrows():
                    if k == 0:
                        preds.append(r.surf_DO)
                        continue
                    don = pairs[pairs.station.isin(combo)]
                    near = don.iloc[(don.t_utc - r.t_utc).abs().to_numpy().argsort()[:3]]
                    if len(near) == 0:
                        preds.append(r.surf_DO)
                        continue
                    preds.append(r.surf_DO + float((near.bot_DO - near.surf_DO).mean()))
                e = np.abs(np.asarray(preds) - te.bot_DO.to_numpy(float))
                rows.append(dict(held_station=held, n_donor_stations=k, donors=','.join(combo), mae=float(e.mean())))
                errs.append(pd.DataFrame(dict(held=held, k=k, config=ci, donors=','.join(combo),
                                              cast_id=te.cast_id.to_numpy(), day=te.t_utc.dt.floor('D').to_numpy(),
                                              e=e)))
    pub = pd.read_csv(PLACEMENT / 'placement_vertical.csv').fillna({'donors': ''})
    mine = pd.DataFrame(rows)
    gap = float(np.max(np.abs(pub.mae.to_numpy() - mine.mae.to_numpy())))
    same = bool(len(pub) == len(mine) and (pub.donors.to_numpy() == mine.donors.to_numpy()).all()
                and (pub.held_station.to_numpy() == mine.held_station.to_numpy()).all())
    if not (same and gap < 1e-12):
        raise SystemExit(f'placement gate failed: rows match {same}, max |MAE difference| {gap}')
    E = pd.concat(errs, ignore_index=True)

    def frame(sub):
        return sub.assign(fold=sub.held, station=sub.held)[['fold', 'station', 'cast_id', 'e', 'day']] \
                  .reset_index(drop=True)

    def averaged(sub):
        """Each cast's error averaged over the donor sets in `sub` (balanced within a held station)."""
        return frame(sub.groupby(['held', 'cast_id'], sort=False)
                     .agg(e=('e', 'mean'), day=('day', 'first')).reset_index())
    naive = E[E.k == 0]
    out = {'what': 'near-bottom oxygen at a held station: its surface reading plus the mean surface-to-bottom '
                   'offset of the three donor casts nearest in time, against the naive assumption (bottom = '
                   'surface) on the same casts',
           'metric': 'mean absolute error over a held station\'s casts (flat), then the mean over held stations',
           'interval': INTERVAL + f'; reported where a contrast has at least {MIN_DATES} dates',
           'gate': {'rows_match': same, 'max_abs_mae_difference': gap},
           'pairs_by_station': {s: int(n) for s, n in pairs.station.value_counts().sort_index().items()},
           'held_stations': {}, 'single_donors': {}, 'by_number_of_donors': {}, 'configurations': []}
    for h in sorted(E.held.unique()):
        b = frame(naive[naive.held == h])
        donor_sets = E[(E.held == h) & (E.k > 0)]
        block = entry(averaged(donor_sets), b)
        diffs = []
        for (k, ci), g in donor_sets.groupby(['k', 'config'], sort=True):
            c = entry(frame(g), b)
            diffs.append(c['diff'])
            out['configurations'].append(dict(held_station=h, n_donor_stations=int(k), donors=g.donors.iloc[0],
                                              **{f: c[f] for f in ('mae', 'naive', 'diff', 'dates', 'ci',
                                                                   'jackknife_t', 'robust') if f in c}))
        mine_h = [c for c in out['configurations'] if c['held_station'] == h]
        best = min(mine_h, key=lambda c: c['diff'])
        # counts over the donor sets of this held station, each set's own bootstrap interval, unadjusted
        block.update(donor_sets=len(diffs), better_point=int(sum(x < 0 for x in diffs)),
                     better_with_interval=int(sum('ci' in c and c['ci'][1] < 0 for c in mine_h)),
                     worse_with_interval=int(sum('ci' in c and c['ci'][0] > 0 for c in mine_h)),
                     better_robust=int(sum(c.get('robust', False) and c['diff'] < 0 for c in mine_h)),
                     worse_robust=int(sum(c.get('robust', False) and c['diff'] > 0 for c in mine_h)),
                     best_set={f: best[f] for f in ('donors', 'mae', 'diff', 'ci', 'jackknife_t') if f in best},
                     sets_worse_by_point=[c['donors'] for c in mine_h if c['diff'] > 0],
                     casts=int(len(b)))
        out['held_stations'][h] = block
    out['all_held_stations_mean_over_donor_sets'] = entry(averaged(E[E.k > 0]), frame(naive))
    for dnr in sorted(pairs.station.unique()):
        sub = E[(E.k == 1) & (E.donors == dnr)]
        hs = sorted(sub.held.unique())
        out['single_donors'][dnr] = dict(held_stations=hs, **entry(frame(sub), frame(naive[naive.held.isin(hs)])))
    for k in sorted(E.k.unique()):
        if k == 0:
            continue
        out['by_number_of_donors'][int(k)] = entry(averaged(E[E.k == k]), frame(naive))
    hs = out['held_stations']
    out['PF2'] = {'prediction': 'Per held station, donors help at Mitili Est and hurt at Baia Santa Teresa.',
                  'PMitiliE': {f: hs['PMitiliE'][f] for f in ('diff', 'ci', 'jackknife_t', 'robust')},
                  'PBaia': {f: hs['PBaia'][f] for f in ('diff', 'ci', 'jackknife_t', 'robust')},
                  'held_by_point_estimate': bool(hs['PMitiliE']['diff'] < 0 < hs['PBaia']['diff']),
                  'held_with_intervals_excluding_zero': bool(hs['PMitiliE']['robust'] and hs['PBaia']['robust']
                                                             and hs['PMitiliE']['diff'] < 0 < hs['PBaia']['diff'])}
    return out


# ------------------------------------------------------------------------------------------ F3 --
def exact(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def leads():
    import src.experiments.v2_warning_episodes as W
    gap = 12
    matches, strict, episodes, alerts = [], [], 0, 0
    for name, frame in W.series().items():
        oxy = frame.oxygen
        eps = W.cluster(W.crossings_of(oxy), gap)
        al = W.alerts_of(oxy)
        episodes += len(eps)
        alerts += len(al)
        for m in W.one_to_one(al, eps):
            matches.append(dict(sensor=name.split('_')[0], **m))
        # the same greedy assignment with the lead required to be strictly positive
        cand = sorted(((ep[0] - a).total_seconds() / 3600.0, ai, ei) for ei, ep in enumerate(eps)
                      for ai, a in enumerate(al) if 0 < (ep[0] - a).total_seconds() / 3600.0 <= W.MATCH_H)
        used_a, used_e = set(), set()
        for lead, ai, ei in cand:
            if ai in used_a or ei in used_e:
                continue
            used_a.add(ai)
            used_e.add(ei)
            strict.append(dict(sensor=name.split('_')[0], lead_h=round(lead, 1), alert=str(al[ai]),
                               episode_onset=str(eps[ei][0])))
    rate, consec, cooldown = W.operating_point()
    published = json.loads((RUNS / 'v2_warning_episodes.json').read_text())
    point = published['operating_point']
    if (point['rate'], point['consecutive'], point['cooldown_h']) != (rate, consec, cooldown):
        raise SystemExit('runs/v2_warning_episodes.json is not at the selected operating point')
    published = published['by_gap'][f'{gap}h']
    if (len(matches), episodes, alerts) != (published['matched_one_to_one'], published['episodes'], published['alerts']):
        raise SystemExit('warning-episode gate failed')
    lv = sorted(m['lead_h'] for m in matches)
    positive = sum(x > 0 for x in lv)
    sel = json.loads(W.POINT_SOURCE.read_bytes())['warning_2026']['crossing_level']['selected_point']
    return {'what': f'Warning-2026, operating point selected on development over the extended grid '
                    f'(rate > {rate:g} umol/kg/h, {consec} consecutive hours, {cooldown} h cooldown; '
                    f'runs/v2_warning_grid_ext.json), episodes at the 12 h clustering gap, one-to-one matching by '
                    f'smallest non-negative lead within 24 h (v2_warning_episodes.one_to_one)',
            'gate': {'matched': len(matches), 'episodes': episodes, 'alerts': alerts},
            'matches': matches, 'leads_h_sorted': lv, 'median_lead_h': float(np.median(lv)),
            'range_h': [min(lv), max(lv)], 'zero_lead_matches': [m for m in matches if m['lead_h'] == 0],
            'recall': {'matched': len(matches), 'episodes': episodes, 'value': round(len(matches) / episodes, 3)},
            'recall_positive_lead': {'matched': positive, 'episodes': episodes,
                                     'value': round(positive / episodes, 3), 'exact_95': exact(positive, episodes)},
            'strictly_positive_matching': {'matched': len(strict), 'leads_h': sorted(m['lead_h'] for m in strict),
                                           'recall': round(len(strict) / episodes, 3)},
            'crossing_level_median_lead_h': sel['median_lead_h'],
            'crossing_level_definition': 'exp_warning_curve: for each crossing the lead of the earliest alert '
                                         'within 24 h before it; the median over crossings at each sensor, then '
                                         'the median over sensors'}


# ---------------------------------------------------------------------------- description --
def warning_grid():
    """The extended grid the operating point is selected on (runs/v2_warning_grid_ext), read back: the
    selection recomputed from its development curve by its rule, and where the point lies on the grid."""
    import src.experiments.v2_warning_episodes as W
    ext = json.loads(W.POINT_SOURCE.read_bytes())
    dev = pd.read_csv(RUNS / 'v2_warning_grid_ext/warning_curve_development_ext.csv')
    grid = ext['grid']
    cand = dev[(dev.recall_causal >= 0.5) & (dev.alerts > 0)].copy()
    cand['p9'], cand['r9'] = cand.precision_causal.round(9), cand.recall_causal.round(9)
    top = cand.sort_values(['p9', 'r9', 'alerts', 'rate', 'consec', 'cooldown'],
                           ascending=[False, False, True, True, True, True], kind='mergesort').iloc[0]
    point = (float(top.rate), int(top.consec), int(top.cooldown))
    rates, cooldowns = grid['rates_umol_per_kg_per_h'], grid['cooldowns_h']
    interior = min(rates) < point[0] < max(rates)
    note = ('the rate is interior to the swept rates' if interior else 'the rate is at an edge of the swept rates')
    if point[2] == min(cooldowns) == 1:
        note += ('; the cooldown is the shortest swept, 1 h, which on hourly readings is no cooldown (an alert may '
                 'fire at every hour the condition holds), so it cannot be shorter')
    return {'rule': ext['rule'],
            'rates_swept': rates, 'consecutive_swept': grid['consecutive_hours'], 'cooldowns_swept_h': cooldowns,
            'points': int(len(dev)),
            'selected': {'rate': point[0], 'consec': point[1], 'cooldown': point[2],
                         'development_precision_causal': round(float(top.precision_causal), 4),
                         'development_recall_causal': round(float(top.recall_causal), 4)},
            'matches_warning_grid_ext_json': bool(point == W.operating_point()),
            'selected_at_grid_edges': {
                'rate_is_grid_minimum': bool(point[0] == min(rates)),
                'rate_is_grid_maximum': bool(point[0] == max(rates)),
                'cooldown_is_grid_minimum': bool(point[2] == min(cooldowns)),
                'cooldown_is_grid_maximum': bool(point[2] == max(cooldowns)),
                'note': note},
            'reserved_2026_points_scored': 1,
            'reserved_2026_frontier': 'only the selected point is scored on 2026 '
                                      '(runs/v2_warning_grid_ext/reserved2026_points.csv, beside the point '
                                      'selected on the 90-point grid); selection reads development only'}


def episode_windows():
    dt = json.loads((RUNS / 'v2_decline_types.json').read_text())
    we = json.loads((RUNS / 'v2_warning_episodes.json').read_text())
    threshold = dt['threshold_psu']
    by_onset = {e['onset'][:16]: e for e in we['episodes']}
    rows = []
    for r in dt['reserved']['rows']:
        e = by_onset[r['onset']]
        typed = lambda ds: None if ds is None else ('water_mass' if abs(ds) > threshold else 'in_place')
        rows.append({'sensor': r['sensor'], 'first_crossing': r['onset'],
                     'dS_six_hours_ending_at_first_crossing': r['dS'],
                     'dS_first_crossing_minus_6h_to_last_crossing': e['salinity_change'],
                     'dT_six_hours_ending_at_first_crossing': r['dT'],
                     'dT_first_crossing_minus_6h_to_last_crossing': e['temperature_change'],
                     'type_declared_window': r['type'], 'type_extended_window': typed(e['salinity_change'])})
    return {'what': 'the 2026 episodes over the declared window (decline typing) and over the window of the '
                    'episode table, which runs on to the last crossing',
            'threshold': threshold, 'rows': rows,
            'typing_differs': [r['sensor'] + ' ' + r['first_crossing'] for r in rows
                               if r['type_extended_window'] is not None
                               and r['type_extended_window'] != r['type_declared_window']]}


def implied_rate():
    dt = json.loads((RUNS / 'v2_decline_types.json').read_text())
    cyc = json.loads((RUNS / 'v2_daily_cycle.json').read_text())
    ip = dt['development']['types']['in_place']
    night = cyc['oxygen']['JJA']['Baia S. Teresa, 1 m']
    hours = night['hour_min_utc'] + 24.0 - night['hour_max_utc']
    rate = ip['drop_median'] / 6.0
    return {'what': 'the net consumption a decline with no salinity change would need in place, against the '
                    'median summer (JJA) night at 1 m in Baia Santa Teresa',
            'drop_median_umol_l_per_6h': ip['drop_median'], 'rate_umol_l_per_h': round(rate, 4),
            'rate_umol_l_per_day_if_sustained': round(rate * 24.0, 2),
            'median_night': {'range_umol_l': night['range'], 'hour_max_utc': night['hour_max_utc'],
                             'hour_min_utc': night['hour_min_utc'], 'hours': hours,
                             'rate_umol_l_per_h': round(night['range'] / hours, 4)},
            'ratio': round(rate / (night['range'] / hours), 3),
            'sources': ['runs/v2_decline_types.json', 'runs/v2_daily_cycle.json']}


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json (part F)',
              'F1_near_bottom': near_bottom(),
              'F2_placement': placement(),
              'F3_warning_leads': leads(),
              'warning_grid': warning_grid(),
              'episode_windows_2026': episode_windows(),
              'decline_rate_in_place': implied_rate()}
    OUT.write_bytes((json.dumps(report, indent=2, default=str) + '\n').encode('utf-8'))
    f1, f2, f3 = report['F1_near_bottom'], report['F2_placement'], report['F3_warning_leads']
    print('F1 coverage', json.dumps(f1['coverage']['P1_held_station']))
    for m, r in f1['contrasts']['P1_held_station|covered'].items():
        print(f"  covered {m:38s} {r['mae']:8.4f} {r['diff']:+8.4f} {r['ci']} jk {r['jackknife_t']}")
    print('PF1', f1['PF1'])
    for h, r in f2['held_stations'].items():
        print(f"F2 {h:14s} naive {r['naive']:7.3f} change {r['diff']:+.4f} {r.get('ci')} jk {r.get('jackknife_t')} "
              f"better {r['better_point']}/{r['donor_sets']} (with interval {r['better_with_interval']}, worse with "
              f"interval {r['worse_with_interval']})")
    print('PF2', f2['PF2'])
    print('F3 leads', f3['leads_h_sorted'], 'median', f3['median_lead_h'], 'positive', f3['recall_positive_lead'])
    print('wrote', OUT)


if __name__ == '__main__':
    main()
