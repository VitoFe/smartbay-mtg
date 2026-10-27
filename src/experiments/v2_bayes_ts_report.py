"""Addendum 72, the reading: the Bayesian DLM and BSTS against the state-space ARIMA and the thesis's forecasters
of the distribution, on addendum 67's rows.

Run locally after the campaign is fetched into reports/bay_twin/runs/v2_bayes_ts/:
  python -m src.experiments.v2_bayes_ts_report [--levels 11] [--out PATH]

The quantile score reads the nine levels 0.1 to 0.9 (v2_probabilistic_report.SCORE_LEVELS); the eleven-level
score stays in scores_eleven_levels, on which the declared predictions and decision rule are read (their
nine-level readings are beside them). `--levels 11` writes the reading as it was before the scored levels moved.
"""
import json
from pathlib import Path

import numpy as np

from src.experiments import v2_probabilistic as P
from src.experiments import v2_probabilistic_report as R
from src.experiments import v2_statespace_report as SR
from src.experiments import v2_weather_oracle_report as W
from src.experiments.v2_bayes_ts import DECLARATION, LEADS, OUT

KEY = W.KEY
NEW = ('dlm', 'dlm_emp', 'bsts')
CONTRASTS = (('dlm', 'arima_ss'), ('dlm_emp', 'arima_ss'), ('bsts', 'arima_ss'), ('dlm', 'quantile_tree'),
             ('dlm', 'chronos2_net'), ('bsts', 'quantile_tree'), ('bsts', 'chronos2_net'), ('dlm', 'dlm_emp'))
SEED_CONTRAST = ('dlm', 'arima_ss')


def rows(variable, arm, seed):
    f = W.read(OUT / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz')
    return f[KEY + P.QCOLS]


def wide(variable):
    """addendum 67's frame (its references, linear_ar, auto_arima, arima_ss under three seeds) with this
    addendum's arms: dlm under three seeds, dlm_emp and bsts at seed 42."""
    base, counts, _ = SR.wide(variable, seeded=('arima_ss',))
    names = [('dlm', s) for s in P.SEEDS] + [('dlm_emp', 42), ('bsts', 42)]
    g1 = {}
    for arm, seed in names:
        name = W.tag(arm, seed)
        g = rows(variable, arm, seed)
        if g.duplicated(KEY).any():
            raise ValueError(f'{variable} {name}: duplicated rows')
        base = base.merge(g.rename(columns={c: f'{name}|{c}' for c in P.QCOLS}), on=KEY, how='left')
        need = base if seed == 42 else base[base.cohort == 'Dev-2025']
        missing = int(need[f'{name}|q50'].isna().sum())
        g1[name] = {'rows': int(len(need)), 'missing': missing, 'passed': missing == 0}
    return base, counts, g1


def scores(variable, frame, levels=R.SCORE_LEVELS):
    out = {}
    lv = P.LEVELS
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        cell = cell.copy()
        y = cell.truth.to_numpy(float)
        arms = sorted({c.split('|')[0] for c in cell.columns if c.endswith('|q50') and cell[c].notna().all()})
        stats_ = {}
        for arm in arms:
            Q = cell[[f'{arm}|{c}' for c in P.QCOLS]].to_numpy(float)
            cell[f'qs|{arm}'] = R.quantile_score(y, Q, levels)
            cell[f'ae|{arm}'] = np.abs(y - Q[:, lv.index(0.5)])
            cell[f'c80|{arm}'] = ((Q[:, lv.index(0.1)] <= y) & (y <= Q[:, lv.index(0.9)])).astype(float)
            stats_[arm] = {'quantile_score': R.r(R.macro(cell, f'qs|{arm}')), 'median_mae': R.r(R.macro(cell, f'ae|{arm}')),
                           'coverage80': R.r(R.macro(cell, f'c80|{arm}'))}
        cols = [f'qs|{a}' for a in arms]
        draws = R.joint_draws(cell, cols, f'bayests|{variable}|{cohort}|{lead}')
        pairs = list(CONTRASTS) + [(W.tag(SEED_CONTRAST[0], s), W.tag(SEED_CONTRAST[1], s)) for s in P.SEEDS[1:]]
        contrasts = {f'{x} - {yy}': R.contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
                     for x, yy in pairs if x in arms and yy in arms}
        out.setdefault(cohort, {})[str(lead)] = {'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()),
                                                'arms': stats_, 'contrasts': contrasts}
    return out


def divergence(frame):
    """Rows whose central 80 % interval is more than 100 times as wide as the state-space ARIMA's median one in
    the same cohort and lead, per arm: where a band diverged, which probes and when (widths in stored units)."""
    out = {}
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        ref = float((cell['arima_ss|q90'] - cell['arima_ss|q10']).median())
        for arm in NEW:
            w = cell[f'{arm}|q90'] - cell[f'{arm}|q10']
            big = cell[w > 100 * ref]
            if len(big):
                out[f'{cohort}|{lead}|{arm}'] = {
                    'rows': int(len(big)), 'of': int(len(cell)), 'max_width': float(w.max()), 'arima_median_width': ref,
                    'sensors': {k: int(v) for k, v in big.sensor.value_counts().items()},
                    'first': str(big.time.min()), 'last': str(big.time.max())}
    return out


def without_diverged(frame, div, levels=R.SCORE_LEVELS):
    """For each arm whose band diverged on Dev-2025, its quantile-score difference from the state-space ARIMA per
    lead with the probe that holds most of the diverged rows left out (sensor-macro, as the scores)."""
    out = {}
    for arm in NEW:
        counts = {}
        for key, x in div.items():
            if key.startswith('Dev-2025|') and key.endswith(f'|{arm}'):
                for s, n in x['sensors'].items():
                    counts[s] = counts.get(s, 0) + n
        if not counts:
            continue
        probe = max(counts, key=counts.get)
        by = {}
        for lead, cell in frame[(frame.cohort == 'Dev-2025') & (frame.sensor != probe)].groupby('lead'):
            cell = cell.copy()
            y = cell.truth.to_numpy(float)
            for a in (arm, 'arima_ss'):
                cell[f'qs|{a}'] = R.quantile_score(y, cell[[f'{a}|{c}' for c in P.QCOLS]].to_numpy(float), levels)
            by[str(lead)] = R.r(R.macro(cell, f'qs|{arm}') - R.macro(cell, 'qs|arima_ss'))
        out[arm] = {'left_out': probe, 'minus_arima_ss': by}
    return out


def seed_verdicts(block):
    out = {}
    x, y = SEED_CONTRAST
    for lead, cell in block.get('Dev-2025', {}).items():
        cs = [cell['contrasts'].get(f'{W.tag(x, s)} - {W.tag(y, s)}') for s in P.SEEDS]
        if any(c is None for c in cs):
            continue
        signs = {float(np.sign(c['point'])) for c in cs}
        excluding = sum(c['excludes_zero'] for c in cs)
        out[f'{lead}|{x} - {y}'] = {
            'points': [c['point'] for c in cs], 'excluding_zero': excluding,
            'verdict': ('not estimable' if len(signs) > 1 else 'result' if excluding == len(cs) else
                        'direction' if excluding else 'not distinguishable') + ' (seeds only)'}
    return out


def better(b, variable, arm):
    """The declared rule: in both cohorts lower with an interval excluding zero at one lead or more and higher
    with an interval excluding zero at none; for the DLM on Dev-2025, the sign at those leads holds under all
    three seeds."""
    detail = {}
    ok = True
    for cohort in R.COHORTS:
        wins, losses = [], []
        for lead, cell in b[variable][cohort].items():
            c = cell['contrasts'][f'{arm} - arima_ss']
            if c['excludes_zero']:
                (wins if c['point'] < 0 else losses).append(lead)
        if arm == 'dlm' and cohort == 'Dev-2025':
            seeds = b[variable]['seeds']
            wins = [L for L in wins if all(p < 0 for p in seeds.get(f'{L}|dlm - arima_ss', {}).get('points', [1]))]
        detail[cohort] = {'lower_excluding_zero': wins, 'higher_excluding_zero': losses}
        ok = ok and bool(wins) and not losses
    return {'better': ok, 'detail': detail}


def predictions(b, gates):
    out = {}
    fails = [[v, c, L, cell['arms']['dlm']['median_mae'], cell['arms']['arima_ss']['median_mae']]
             for v in P.VARIABLES for c in R.COHORTS for L, cell in b[v][c].items()
             if abs(cell['arms']['dlm']['median_mae'] - cell['arms']['arima_ss']['median_mae']) >= R.FLOORS[v]]
    out['P1'] = {'held': not fails, 'failures': fails}
    cs = {c: b['oxygen'][c]['1']['contrasts']['dlm - arima_ss'] for c in R.COHORTS}
    out['P2'] = {'held': all(not x['excludes_zero'] for x in cs.values()),
                 'contrasts': {c: [x['point'], x['ci']] for c, x in cs.items()}}
    closer, cells = 0, 0
    for v in P.VARIABLES:
        for c in R.COHORTS:
            for L, cell in b[v][c].items():
                cells += 1
                closer += abs(cell['arms']['dlm']['coverage80'] - 0.8) < abs(cell['arms']['arima_ss']['coverage80'] - 0.8)
    out['P3'] = {'held': closer > cells / 2, 'closer': closer, 'cells': cells}
    fails = [[v, c, L, cell['contrasts']['bsts - arima_ss']['point']]
             for v in P.VARIABLES for c in R.COHORTS for L, cell in b[v][c].items()
             if abs(cell['contrasts']['bsts - arima_ss']['point']) >= R.FLOORS[v]]
    out['P4'] = {'held': not fails, 'failures': fails}
    fails = []
    for v in P.VARIABLES:
        for c in R.COHORTS:
            for L, cell in b[v][c].items():
                s = cell['arms']
                best = min(s['quantile_tree']['quantile_score'], s['chronos2_net']['quantile_score'])
                for arm in ('dlm', 'bsts'):
                    if s[arm]['quantile_score'] <= best - R.FLOORS[v]:
                        fails.append([v, c, L, arm, s[arm]['quantile_score'], best])
    out['P5'] = {'held': not fails, 'failures': fails}
    out['P6'] = {'held': bool(gates['G4']['passed'])}
    return out


def report(levels=R.SCORE_LEVELS, target=None):
    levels = tuple(levels)
    target = OUT / 'report.json' if target is None else Path(target)
    rep = {'declared_in': DECLARATION, 'leads': LEADS, 'draws': R.DRAWS, 'floors': R.FLOORS}
    if R.secondary(levels):
        rep['levels'], rep['stored_levels'] = levels, P.LEVELS
        rep['eleven_levels'] = ('scores_eleven_levels: the same scores over the eleven stored levels 0.05-0.95; '
                                'the declared predictions and decision rule are read on it')
    gates = {'G1_rows': {}, 'G2': {}, 'G3': 'by construction: both filters advance hour by hour and the forecast is '
                                              'taken from the posterior after the issuance reading',
             'G4': {}, 'G5': {}}
    b, b11, studies, bsts = {}, {}, {}, {}
    for variable in P.VARIABLES:
        frame, counts, g1 = wide(variable)
        gates['G1_rows'][variable] = g1
        for seed in P.SEEDS:
            meta = json.loads((OUT / f'seed{seed}' / f'bayes_{variable}_dlm.json').read_text())
            gates['G2'][f'{variable}|{seed}'] = meta['gates']['G2']
            studies[f'{variable}|{seed}'] = {**meta['study'], 'config': meta['config']}
            gates['G5'][f'{variable}|{seed}'] = meta['study']['completed_trials'] >= 100
        meta = json.loads((OUT / 'seed42' / f'bayes_{variable}_bsts.json').read_text())
        gates['G4'][variable] = {**meta['gates']['G4'], 'above': sorted(
            [split, n, round(i['rhat_max'], 3)] for split, probes in meta['probes'].items() for n, i in probes.items()
            if i.get('rhat_max', 0) > 1.05)}
        bsts[variable] = {split: {n: {k: i.get(k) for k in ('readings_before_cut', 'rhat_max', 'acceptance',
                                                             'posterior_median', 'laplace', 'fallback')}
                                  for n, i in probes.items()} for split, probes in meta['probes'].items()}
        gibbs = json.loads((OUT / 'gibbs' / f'bayes_{variable}_bsts.json').read_text())
        gates['G4_gibbs_as_declared'] = {**gates.get('G4_gibbs_as_declared', {}), variable: gibbs['gates']['G4']}
        for block, lv in ((b, levels), (b11, P.LEVELS if R.secondary(levels) else None)):
            if lv is None:
                continue
            block[variable] = scores(variable, frame, lv)
            block[variable]['seeds'] = seed_verdicts(block[variable])
            block[variable]['row_counts'] = counts
            block[variable]['divergence'] = divergence(frame)
            block[variable]['without_diverged'] = without_diverged(frame, block[variable]['divergence'], lv)
        print(variable, 'done', flush=True)
    gates['G4']['passed'] = all(g['passed'] for k, g in gates['G4'].items() if isinstance(g, dict))
    ok = (all(x['passed'] for v in gates['G1_rows'].values() for x in v.values())
          and all(g['passed'] for g in gates['G2'].values()) and all(gates['G5'].values()))
    rep.update({'gates': gates, 'gates_passed': ok, 'scores': b, 'studies': studies, 'bsts': bsts})

    def decide(block):
        d = {f'{v}|{arm}': better(block, v, arm) for v in P.VARIABLES for arm in ('dlm', 'bsts')}
        d['integrate'] = any(x['better'] for x in d.values())
        return d
    rep['predictions'] = predictions(b11 or b, gates)
    rep['decision'] = decide(b11 or b)
    if b11:
        rep['predictions_nine_levels'] = predictions(b, gates)
        rep['decision_nine_levels'] = decide(b)
        rep['scores_eleven_levels'] = b11
    target.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    print('wrote', target, 'gates', ok, 'integrate', rep['decision']['integrate'])
    print(json.dumps(rep['predictions'], default=str)[:2500])


if __name__ == '__main__':
    report(*R.cli())
