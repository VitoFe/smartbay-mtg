"""Addendum 80, the reading: what the currents measured at Mitili Est add to addendum 65's trees.

Reads the new arms from the fetched run and addendum 65's weather arms from runs/v2_weather_oracle,
checks that every arm sits on the same rows (gate G2), and scores the declared contrasts with addendum
65's quantile score, nested macro and week-block bootstrap, on the endpoints whose window holds a 2 m
current value at three hours in four or more (the primary subgroup) and on all endpoints.

  python -m src.experiments.v2_currents_oracle_report
"""
import json

import numpy as np
import pandas as pd

from src.experiments import v2_probabilistic as P
from src.experiments import v2_probabilistic_report as R
from src.experiments import v2_currents_oracle as C
from src.experiments.sweep_verdict import verdict

RUN = '20261007T130838Z_v2_currents'
CELLS = P.RUNS / RUN / 'reports/bay_twin/runs/v2_currents_oracle'
WEATHER = P.RUNS / 'v2_weather_oracle'
OUT = P.RUNS / 'v2_currents_oracle.json'
KEY = ['cohort', 'sensor', 'time', 'lead']
NEW = tuple(C.ARMS)
OLD = ('quantile_tree_weather', 'quantile_tree_weather_future')
CONTRASTS = {'C1': ('quantile_tree_weather_future_currents_future', 'quantile_tree_weather_future'),
             'C2': ('quantile_tree_weather_currents', 'quantile_tree_weather'),
             'C3': ('quantile_tree_weather_future_currents_future', 'quantile_tree_weather')}


def tag(arm, seed):
    return arm if seed == 42 else f'{arm}@{seed}'


def read(path, seed, arm):
    """One arm's rows (addendum 65's weather file also holds the plain tree and the climatology at 72 h)."""
    f = pd.read_csv(path)
    f = f[f.arm == arm].copy()
    f['time'] = R.utc(f.time)
    return f.assign(arm=tag(arm, seed))


def wide(variable):
    parts = {}
    for seed in P.SEEDS:
        for arm in OLD:
            parts[tag(arm, seed)] = read(WEATHER / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz', seed, arm)
        for arm in NEW:
            parts[tag(arm, seed)] = read(CELLS / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz', seed, arm)
    # G2: each new arm on exactly its weather reference's rows, truth and last value
    d = P.DIGITS[variable]
    gate = {}
    for seed in P.SEEDS:
        for arm, ref in ((NEW[0], OLD[0]), (NEW[1], OLD[1])):
            a, b = parts[tag(arm, seed)], parts[tag(ref, seed)]
            j = a[KEY + ['truth', 'last']].merge(b[KEY + ['truth', 'last']], on=KEY, how='outer',
                                                 suffixes=('', '_ref'), indicator=True)
            both = j[j._merge == 'both']
            diff = float(max((both.truth - both.truth_ref).abs().max(), (both['last'] - both.last_ref).abs().max()))
            gate[f'{tag(arm, seed)}'] = {'rows': int(len(a)), 'rows_reference': int(len(b)),
                                        'unmatched': int((j._merge != 'both').sum()), 'max_abs_difference': diff,
                                        'passed': bool((j._merge == 'both').all() and diff <= 10 ** -d)}
    base = parts[OLD[0]][KEY + ['fold', 'truth', 'last']].copy()
    for arm, f in parts.items():
        g = f[KEY + P.QCOLS].rename(columns={c: f'{arm}|{c}' for c in P.QCOLS})
        if g.duplicated(KEY).any():
            raise ValueError(f'{variable} {arm}: duplicated rows')
        base = base.merge(g, on=KEY, how='left')
    base['fold'] = np.where(base.cohort == 'Dev-2025', base.fold.astype('float').astype('Int64').astype(str),
                            np.where(base.time.dt.month <= 3, 'Q1', 'Q2'))
    base['week'] = base.time.dt.isocalendar().week.astype(int).to_numpy()
    cov = np.full(len(base), np.nan)
    for lead, idx in base.groupby('lead').indices.items():
        cov[idx] = C.window_coverage(base.time.iloc[idx], int(lead))
    base['coverage'] = cov
    return base, gate


def score_cell(cell, arms, label):
    y = cell.truth.to_numpy(float)
    here = [a for a in arms if cell[f'{a}|q50'].notna().all()]
    for arm in here:
        cell[f'qs|{arm}'] = R.quantile_score(y, cell[[f'{arm}|{c}' for c in P.QCOLS]].to_numpy(float))
    cols = [f'qs|{a}' for a in here]
    draws = R.joint_draws(cell, cols, label)
    out = {'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()),
           'quantile_score': {a: R.r(R.macro(cell, f'qs|{a}')) for a in here}, 'contrasts': {}}
    for seed in P.SEEDS:
        for name, (x, yy) in CONTRASTS.items():
            x, yy = tag(x, seed), tag(yy, seed)
            if x in here and yy in here:
                out['contrasts'][f'{name}@{seed}'] = R.contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
    return out


def scores(variable, frame):
    arms = sorted({c.split('|')[0] for c in frame.columns if c.endswith('|q50')})
    out = {}
    for subgroup in ('covered', 'all'):
        part = frame[frame.coverage >= C.COVERED] if subgroup == 'covered' else frame
        for (cohort, lead), cell in part.groupby(['cohort', 'lead']):
            cell = cell.dropna(subset=[f'{a}|q50' for a in arms if '@' not in a or cohort == 'Dev-2025']).copy()
            if cell.empty:
                continue
            out.setdefault(subgroup, {}).setdefault(cohort, {})[str(lead)] = score_cell(
                cell, arms, f'currents|{variable}|{subgroup}|{cohort}|{lead}')
    return out


def seed_readings(block):
    """Chapter 4's rule over the three sampler seeds on Dev-2025 (seeds only), per contrast and lead."""
    out = {}
    for lead, cell in block.get('Dev-2025', {}).items():
        for name in ('C1', 'C2', 'C3'):
            cs = [cell['contrasts'].get(f'{name}@{s}') for s in P.SEEDS]
            if any(c is None or c['point'] is None for c in cs):
                continue
            out[f'{name}|{lead}'] = {'points': [c['point'] for c in cs],
                                     'verdict': verdict([(c['point'], *c['ci']) for c in cs])}
    return out


def decision(sc, floors):
    """On the covered endpoints: the leads at which C1's gain exceeds the floor with an interval excluding
    zero, and on Dev-2025 the seeds agree in sign."""
    found = []
    for variable, block in sc.items():
        cov = block['covered']
        seeds = block['seeds_covered']
        for cohort in R.COHORTS:
            for lead, cell in cov.get(cohort, {}).items():
                c = cell['contrasts'].get('C1@42')
                if c is None or not (c['point'] < -floors[variable] and c['excludes_zero']):
                    continue
                pts = seeds.get(f'C1|{lead}', {}).get('points', [])
                if cohort == 'Dev-2025' and not (pts and all(p < 0 for p in pts)):
                    continue
                found.append({'variable': variable, 'cohort': cohort, 'lead': int(lead), 'point': c['point'],
                              'ci': c['ci']})
    return {'leads_exceeding_floor': found,
            'reading': ('perfect currents at Mitili Est add more than the floor at the listed leads' if found else
                        'perfect currents at Mitili Est add less than the floors at every lead in both cohorts')}


def report():
    rep = {'declared_in': C.DECLARATION, 'run': RUN, 'leads': C.LEADS, 'covered_threshold': C.COVERED,
           'floors_per_kg_and_degC': R.FLOORS, 'draws': R.DRAWS}
    gates, sc = {}, {}
    for variable in P.VARIABLES:
        frame, g2 = wide(variable)
        trees = {s: json.loads((CELLS / f'seed{s}' / f'tree_{variable}_{a}.json').read_text())
                 for s in P.SEEDS for a in NEW}
        gates[variable] = {'G2_rows': g2,
                           'G1_design': all(t['gates']['G1_design']['passed'] for t in trees.values()),
                           'G3_currents': all(t['gates']['G3_currents']['passed'] for t in trees.values()
                                              if 'G3_currents' in t['gates']),
                           'G4_trials': min(s['completed_trials'] for t in trees.values() for s in t['studies'].values()),
                           'smoke': any(t['smoke_only'] for t in trees.values())}
        block = scores(variable, frame)
        block['seeds_covered'] = seed_readings(block['covered'])
        block['seeds_all'] = seed_readings(block['all'])
        block['covered_share'] = {f'{c}|{l}': R.r(float((g.coverage >= C.COVERED).mean()))
                                  for (c, l), g in frame.groupby(['cohort', 'lead'])}
        sc[variable] = block
        print(variable, 'done', flush=True)
    ok = all(all(x['passed'] for x in g['G2_rows'].values()) and g['G1_design'] and g['G3_currents']
             and g['G4_trials'] >= 100 and not g['smoke'] for g in gates.values())
    rep['gates'], rep['gates_passed'], rep['scores'] = gates, ok, sc
    rep['decision'] = decision(sc, R.FLOORS) if ok else {'reading': 'not read: a declared gate failed'}
    # "stays below both floors": the size of C1, a gain or a deficit, under the floor at every lead
    p1 = all(abs(cell['contrasts']['C1@42']['point']) < R.FLOORS[v]
             for v in sc for coh in sc[v]['covered'].values() for cell in coh.values() if 'C1@42' in cell['contrasts'])
    # read on the primary subgroup, as every contrast the declaration does not assign to one
    p2 = {v: sum(cell['contrasts']['C2@42']['excludes_zero'] for cell in sc[v]['covered']['Dev-2025'].values()) <= 2
          for v in sc}
    rep['predictions'] = {'P1': p1, 'P2': p2}
    OUT.write_bytes((json.dumps(rep, indent=1, default=str) + '\n').encode('utf-8'))
    print('wrote', OUT, 'gates', ok)
    print(json.dumps(rep['decision'], default=str), json.dumps(rep['predictions']))


if __name__ == '__main__':
    report()
