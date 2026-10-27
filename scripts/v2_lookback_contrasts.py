"""Addendum 46: score the look-back arms and apply the declaration's gates, rules and predictions.

Puts every arm on the published addendum-39 rows (runs/v2_bridge/bridge_<variable>.csv.gz):

  chronos_ctx<L>   the traces the cluster wrote (runs/v2_lookback/chronos/), the prediction at lead
                   h being step h of the trajectory from the onset, as addendum 41 scored it
  anchored_h<W>    the look-back continuations (runs/v2_lookback/h<W>/), on identical keys;
                   anchored_h6 is the published anchored_direct

The error at a lead is v2_bridge.macro (mean per fold and probe, over probes, over folds); every
contrast is v2_bridge.week_blocks, by lead and cohort, never pooled: its reported interval draws
onset weeks jointly within each fold (cells fixed, 600 draws), and the interval resampling onset
weeks within each fold and probe is kept beside it (`ci_within_cell`). The decisions and predictions
are read with the reported intervals and, beside them, with the within-cell ones. A gate that failed
stops the report of the arms it covers.

Writes reports/bay_twin/runs/v2_lookback.json.
Usage: python -m scripts.v2_lookback_contrasts [--workers 4]
Declared in config/bay_reserved_protocol_v2_addendum46.json.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_bridge

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
LB = RUNS / 'v2_lookback'
OUT = RUNS / 'v2_lookback.json'
LENGTHS = (48, 128, 256, 512, 1024)
WINDOWS = (24, 72)
SEEDS = (7, 123)
VARIABLES = ('temperature', 'oxygen')
COHORTS = ('Bridge-Dev', 'Bridge-2026')
KEY = ['cohort', 'fold', 'sensor', 'onset', 'lead']
FLOOR = {'temperature': 0.20, 'oxygen': 1.8}
REFERENCE = {'temperature': 'anchored_direct', 'oxygen': 'clim_hour'}


def rows(variable):
    frame = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz', parse_dates=['onset'])
    frame['onset'] = pd.to_datetime(frame.onset, utc=True)
    iso = frame.onset.dt.isocalendar()
    frame['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
    inputs = np.load(LB / 'inputs' / f'continuation_{variable}_512.npz', allow_pickle=True)
    key = pd.DataFrame({'cohort': inputs['cohort'], 'fold': inputs['fold'], 'sensor': inputs['sensor'],
                        'onset': pd.to_datetime(inputs['onset'], utc=True)})
    key['episode'] = np.arange(len(key))
    frame = frame.merge(key, on=['cohort', 'fold', 'sensor', 'onset'], how='left', validate='many_to_one')
    assert frame.episode.notna().all(), 'a published row has no context'
    step = frame.lead.to_numpy() - 1
    for length in LENGTHS:
        got = np.load(LB / 'chronos' / f'continuation_{variable}_{length}.npz')
        trace = np.full((len(key), got['traces'].shape[1]), np.nan)
        trace[got['rows']] = got['traces']
        frame[f'chronos_ctx{length}'] = trace[frame.episode.to_numpy().astype(int), step]
    frame = frame.rename(columns={'anchored_direct': 'anchored_h6'})
    frame['anchored_direct'] = frame.anchored_h6
    for window in WINDOWS:
        path = LB / f'h{window}' / f'bridge_{variable}.csv.gz'
        if not path.exists():
            continue
        arm = pd.read_csv(path, parse_dates=['onset'])
        arm['onset'] = pd.to_datetime(arm.onset, utc=True)
        col = f'anchored_h{window}'
        n0 = len(frame)
        frame = frame.merge(arm[KEY + [col]], on=KEY, how='left', validate='one_to_one')
        assert len(frame) == n0 and frame[col].notna().all(), f'{col} does not cover the published rows'
    return frame


def gates():
    g = {'G1': json.loads((LB / 'inputs' / 'gate_G1.json').read_text())}
    g['G1']['passed'] = all(v['passed'] for v in g['G1'].values())
    g2 = {}
    for variable in VARIABLES:
        ref = np.load(RUNS / 'v2_chronos_adapt' / f'continuation_{variable}_zs.npz')
        mine = np.load(LB / 'chronos' / f'continuation_{variable}_512.npz')
        full = np.full(mine['traces'].shape, np.nan)
        full[ref['rows']] = ref['traces']
        g2[variable] = float(np.nanmax(np.abs(full - mine['traces'])))
    g['G2'] = {'max_abs_diff': g2, 'passed': all(d <= 1e-4 for d in g2.values())}
    g34 = json.loads((LB / 'gate_G3_G4.json').read_text())
    g['G3'] = {v: g34[v]['G3'] for v in VARIABLES}
    g['G3']['passed'] = all(g34[v]['G3']['passed'] for v in VARIABLES)
    g['G4'] = {v: g34[v]['G4'] for v in VARIABLES}
    g['G4']['passed'] = all(g34[v]['G4']['passed'] for v in VARIABLES)
    trials = {}
    for folder in sorted(LB.glob('h*')):
        for variable in VARIABLES:
            path = folder / f'bridge_{variable}_studies.json'
            if path.exists():
                trials[f'{folder.name}_{variable}'] = {k: s['completed_trials'] for k, s in json.loads(path.read_text()).items()}
    g['G5'] = {'completed_trials': trials,
               'passed': bool(trials) and all(n >= 100 for t in trials.values() for n in t.values())}
    return g


def by_lead(frame, variable, cohort, pairs, pool):
    """Errors and contrasts by lead. Each contrast's bootstrap stream is keyed on its label, so
    running them in worker processes gives the same numbers as running them in turn."""
    f = frame[frame.cohort == cohort]
    out, jobs = {}, {}
    for lead in v2_bridge.LEADS:
        sub = f[f.lead == lead]
        arms = [c for c in sub.columns if (c.startswith('chronos_ctx') or c.startswith('anchored_h')
                                           or c in ('persistence', 'clim_hour', 'clim_month_hour', 'neighbour_change'))
                and sub[c].notna().all()]
        out[str(lead)] = {'mae': {a: round(v2_bridge.macro(sub, a), 6) for a in arms}, 'contrasts': {}}
        for x, y in pairs:
            if x in sub and y in sub:
                cols = list(dict.fromkeys(['fold', 'sensor', 'week', 'truth', x, y]))
                jobs[(str(lead), f'{x} - {y}')] = pool.submit(v2_bridge.week_blocks, sub[cols], x, y,
                                                              f'a46|{variable}|{cohort}|{lead}|{x}|{y}')
    for (lead, name), job in jobs.items():
        out[lead]['contrasts'][name] = job.result()
    return out


def seed_follow_up(frame, variable, window, pool):
    """The decision rule's follow-up: the arm re-tuned under sampler seeds 7 and 123 on Bridge-Dev,
    each against the published six-hour continuation on the same rows."""
    dev = frame[frame.cohort == 'Bridge-Dev']
    out = {}
    for seed in SEEDS:
        col = f'anchored_h{window}_seed{seed}'
        path = LB / f'h{window}_seed{seed}' / f'bridge_{variable}.csv.gz'
        if not path.exists():
            out[str(seed)] = {'missing': str(path.relative_to(ROOT))}
            continue
        arm = pd.read_csv(path, parse_dates=['onset'])
        arm['onset'] = pd.to_datetime(arm.onset, utc=True)
        arm = arm.rename(columns={f'anchored_h{window}': col})
        joined = dev.merge(arm[KEY + [col, 'truth']], on=KEY, how='left', suffixes=('', '_seed'), validate='one_to_one')
        assert joined[col].notna().all(), f'{col} does not cover the published Bridge-Dev rows'
        assert (joined.truth - joined.truth_seed).abs().max() == 0, f'{col} scored a different truth'
        block = by_lead(joined, variable, 'Bridge-Dev', [(col, 'anchored_h6')], pool)
        name = f'{col} - anchored_h6'
        out[str(seed)] = {'by_lead': {L: {'mae': e['mae'].get(col), **e['contrasts'][name]} for L, e in block.items()},
                          'leads_better': excluded_below(block, name),
                          'leads_better_within_cell': excluded_below(block, name, 'ci_within_cell'),
                          'mean_contrast_over_leads': float(np.mean([e['contrasts'][name]['point'] for e in block.values()]))}
    return out


def mean_over_leads(block, arm):
    return float(np.mean([e['mae'][arm] for e in block.values()]))


def excluded_below(block, name, interval='ci'):
    """Leads at which x - y has an interval wholly below zero (x better)."""
    return sum(1 for e in block.values() if name in e['contrasts'] and e['contrasts'][name][interval][1] < 0)


def excluded_above(block, name, leads=None, interval='ci'):
    return sum(1 for L, e in block.items() if (leads is None or L in leads)
               and name in e['contrasts'] and e['contrasts'][name][interval][0] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    workers = p.parse_args().workers
    rep = {'declared_in': 'config/bay_reserved_protocol_v2_addendum46.json', 'gates': gates(), 'variables': {}}
    g = rep['gates']
    chronos_ok = g['G1']['passed'] and g['G2']['passed']
    anchored_ok = g['G3']['passed'] and g['G4']['passed'] and g['G5']['passed']
    for variable in VARIABLES:
        frame = rows(variable)
        have = [w for w in WINDOWS if f'anchored_h{w}' in frame]
        pairs = []
        if chronos_ok:
            pairs += [(f'chronos_ctx{L}', 'chronos_ctx512') for L in LENGTHS if L != 512]
            pairs += [(f'chronos_ctx{L}', 'anchored_direct') for L in LENGTHS]
            pairs += [(f'chronos_ctx{L}', 'clim_hour') for L in LENGTHS]
        if anchored_ok:
            pairs += [(f'anchored_h{w}', 'anchored_h6') for w in have]
            if 72 in have and 24 in have:
                pairs.append(('anchored_h72', 'anchored_h24'))
        v = {}
        for cohort in COHORTS:
            with ProcessPoolExecutor(workers) as pool:
                block = by_lead(frame, variable, cohort, pairs, pool)
            v[cohort] = {'episodes': int(frame[frame.cohort == cohort].groupby(['fold', 'sensor', 'onset']).ngroups),
                         'by_lead': block,
                         'mean_over_leads': {a: round(mean_over_leads(block, a), 6) for a in block['6']['mae']}}
        rep['variables'][variable] = v
        print(variable, {c: v[c]['mean_over_leads'] for c in COHORTS}, flush=True)
        if anchored_ok:
            follow = {}
            for window in have:
                name = f'anchored_h{window} - anchored_h6'
                # the follow-up runs if either interval scheme calls for it, so both readings exist
                if max(excluded_below(v['Bridge-Dev']['by_lead'], name, k) for k in ('ci', 'ci_within_cell')) >= 4:
                    with ProcessPoolExecutor(workers) as pool:
                        follow[f'h{window}'] = seed_follow_up(frame, variable, window, pool)
            rep['variables'][variable]['seed_follow_up'] = follow
    rep['decisions'], rep['predictions'] = decide(rep, chronos_ok, anchored_ok)
    rep['decisions_within_cell'], rep['predictions_within_cell'] = decide(rep, chronos_ok, anchored_ok,
                                                                          'ci_within_cell')
    OUT.write_bytes(json.dumps(rep, indent=1).encode('utf-8'))
    print(json.dumps({'gates': {k: x['passed'] for k, x in g.items()}, 'decisions': rep['decisions'],
                      'predictions': rep['predictions']}, indent=1))


def decide(rep, chronos_ok, anchored_ok, interval='ci'):
    V = rep['variables']
    below = lambda block, name: excluded_below(block, name, interval)
    above = lambda block, name, leads=None: excluded_above(block, name, leads, interval)
    lead_count = 'leads_better' if interval == 'ci' else 'leads_better_within_cell'
    dec, pred = {}, {}
    if chronos_ok:
        for variable in VARIABLES:
            better = {L: [below(V[variable][c]['by_lead'], f'chronos_ctx{L} - chronos_ctx512') for c in COHORTS]
                      for L in LENGTHS if L != 512}
            beats = [L for L, n in better.items() if all(k >= 4 for k in n)]
            best, needs = {}, {}
            for c in COHORTS:
                means = {L: V[variable][c]['mean_over_leads'][f'chronos_ctx{L}'] for L in LENGTHS}
                best[c] = min(means, key=means.get)
                needs[c] = min(L for L in LENGTHS if means[L] - means[best[c]] < FLOOR[variable])
            dec[f'chronos_{variable}'] = {'lengths_beating_512_at_4_leads_both_cohorts': beats,
                                          'leads_beating_512': better, 'best_length': best,
                                          'shortest_length_within_floor_of_best': needs,
                                          'floor': FLOOR[variable],
                                          'verdict': '512 h was not the best length' if beats else '512 h stands'}
        t = {c: V['temperature'][c]['mean_over_leads'] for c in COHORTS}
        c1 = all(t[c]['chronos_ctx48'] >= t[c]['chronos_ctx128'] >= t[c]['chronos_ctx256'] > 0
                 and t[c]['chronos_ctx48'] > t[c]['chronos_ctx256']
                 and max(t[c][f'chronos_ctx{L}'] for L in (256, 512, 1024)) - min(t[c][f'chronos_ctx{L}'] for L in (256, 512, 1024)) < 0.01
                 for c in COHORTS)
        pred['C1'] = c1
        pred['C2'] = above(V['oxygen']['Bridge-Dev']['by_lead'], 'chronos_ctx48 - chronos_ctx512', {'6', '12', '18'}) >= 2
        pred['C3'] = all(below(V['temperature'][c]['by_lead'], f'chronos_ctx{L} - anchored_direct') == 0
                         for c in COHORTS for L in LENGTHS)
        pred['C4'] = all(below(V[v][c]['by_lead'], 'chronos_ctx1024 - chronos_ctx512') == 0
                         for v in VARIABLES for c in COHORTS)
    if anchored_ok:
        for variable in VARIABLES:
            dev = V[variable]['Bridge-Dev']['by_lead']
            res = V[variable]['Bridge-2026']['by_lead']
            leads = {w: below(dev, f'anchored_h{w} - anchored_h6') for w in WINDOWS}
            leads_2026 = {w: below(res, f'anchored_h{w} - anchored_h6') for w in WINDOWS}
            follow = [w for w, n in leads.items() if n >= 4]
            seeds = V[variable].get('seed_follow_up', {})
            seed_leads = {f'h{w}': {s: seeds.get(f'h{w}', {}).get(str(s), {}).get(lead_count) for s in SEEDS}
                          for w in follow}
            done = all(n is not None for d in seed_leads.values() for n in d.values())
            recommend = [w for w in follow if leads_2026[w] >= 4
                         and all((seed_leads[f'h{w}'][s] or 0) >= 4 for s in SEEDS)]
            if not follow:
                verdict = 'the six-hour continuation stays; a longer own history adds nothing reproducible'
            elif not done:
                verdict = 'follow-up incomplete: no size may be quoted'
            elif recommend:
                verdict = f'recommend the longer own history ({", ".join(f"h{w}" for w in recommend)}) for the release continuation'
            else:
                verdict = 'the six-hour continuation stays; a longer own history adds nothing reproducible'
            dec[f'anchored_{variable}'] = {'leads_beating_h6_dev': leads, 'leads_beating_h6_2026': leads_2026,
                                           'needs_seed_follow_up': follow, 'seed_follow_up_leads': seed_leads,
                                           'recommended': recommend, 'verdict': verdict}
        pred['L1'] = all(below(V['temperature']['Bridge-Dev']['by_lead'], f'anchored_h{w} - anchored_h6') < 4 for w in WINDOWS)
        diff = lambda v, c, a, b: float(np.mean([e['mae'][a] - e['mae'][b] for e in V[v][c]['by_lead'].values()]))
        pred['L2'] = all(-0.5 <= diff('oxygen', c, 'anchored_h24', 'anchored_h6') <= 0.5 for c in COHORTS)
        pred['L3'] = all(abs(diff(v, c, 'anchored_h72', 'anchored_h24')) < (0.01 if v == 'temperature' else 0.3)
                         for v in VARIABLES for c in COHORTS)
    return dec, pred


if __name__ == '__main__':
    main()
