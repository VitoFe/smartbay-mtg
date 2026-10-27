"""Addendum 67, the reading: state-space ARIMA and N-BEATS against the thesis's forecasters.

Run locally after the campaign is fetched into reports/bay_twin/runs/v2_statespace/:
  python -m src.experiments.v2_statespace_report [--levels 11] [--out PATH]

The quantile score reads the nine levels 0.1 to 0.9 (v2_probabilistic_report.SCORE_LEVELS); the eleven-level
score stays in scores_eleven_levels, on which the declared predictions are read. `--levels 11` writes the
reading as it was before the scored levels moved.
"""
import json
from pathlib import Path

import numpy as np

from src.experiments import v2_arima_report as AR
from src.experiments import v2_probabilistic as P
from src.experiments import v2_probabilistic_report as R
from src.experiments import v2_weather_oracle_report as W
from src.experiments.v2_arima import OUT as ARIMA
from src.experiments.v2_statespace import DECLARATION, LEADS, OUT

KEY = W.KEY
REFS = ('clim_quantiles', 'quantile_tree', 'chronos2_net')
SEEDED = ('arima_ss', 'nbeats')
CONTRASTS = (('arima_ss', 'clim_quantiles'), ('arima_ss', 'quantile_tree'), ('arima_ss', 'chronos2_net'),
             ('arima_ss', 'linear_ar'), ('auto_arima', 'arima_ss'), ('nbeats', 'quantile_tree'),
             ('nbeats', 'chronos2_net'), ('nbeats', 'clim_quantiles'))
SEED_CONTRASTS = (('arima_ss', 'clim_quantiles'), ('nbeats', 'clim_quantiles'))
SINGLES = ('arima_ss', 'auto_arima', 'nbeats', 'linear_ar', 'quantile_tree', 'chronos2_net')


def rows(variable, arm, seed):
    f = W.read(OUT / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz')
    return f[KEY + P.QCOLS]


def wide(variable, seeded=SEEDED):
    """addendum 65's rows and references, linear_ar (addendum 66, seed 42), and this addendum's arms."""
    full, _, counts, _ = W.wide(variable)
    keep = [c for c in full.columns if '|' not in c or c.split('|')[0] in REFS]
    base = full[keep].copy()
    lin = W.read(ARIMA / 'seed42' / f'quantiles_{variable}_linear_ar.csv.gz')[KEY + P.QCOLS]
    base = base.merge(lin.rename(columns={c: f'linear_ar|{c}' for c in P.QCOLS}), on=KEY, how='left')
    names = [('auto_arima', 42)] + [(a, s) for a in seeded for s in P.SEEDS]
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
        stats = {}
        for arm in arms:
            Q = cell[[f'{arm}|{c}' for c in P.QCOLS]].to_numpy(float)
            cell[f'qs|{arm}'] = R.quantile_score(y, Q, levels)
            cell[f'ae|{arm}'] = np.abs(y - Q[:, lv.index(0.5)])
            cell[f'c80|{arm}'] = ((Q[:, lv.index(0.1)] <= y) & (y <= Q[:, lv.index(0.9)])).astype(float)
            stats[arm] = {'quantile_score': R.r(R.macro(cell, f'qs|{arm}')), 'median_mae': R.r(R.macro(cell, f'ae|{arm}')),
                          'coverage80': R.r(R.macro(cell, f'c80|{arm}'))}
        cols = [f'qs|{a}' for a in arms]
        draws = R.joint_draws(cell, cols, f'statespace|{variable}|{cohort}|{lead}')
        contrasts = {}
        pairs = list(CONTRASTS) + [(f'{x}@{s}', W.tag(yy, s)) for s in P.SEEDS[1:] for x, yy in SEED_CONTRASTS]
        for x, yy in pairs:
            if x in arms and yy in arms:
                contrasts[f'{x} - {yy}'] = R.contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
        out.setdefault(cohort, {})[str(lead)] = {'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()),
                                                'arms': stats, 'contrasts': contrasts}
    return out


def seed_verdicts(block):
    out = {}
    for lead, cell in block.get('Dev-2025', {}).items():
        for x, y in SEED_CONTRASTS:
            cs = [cell['contrasts'].get(f'{x} - {y}' if s == 42 else f'{x}@{s} - {W.tag(y, s)}') for s in P.SEEDS]
            if any(c is None for c in cs):
                continue
            signs = {float(np.sign(c['point'])) for c in cs}
            excluding = sum(c['excludes_zero'] for c in cs)
            out[f'{lead}|{x} - {y}'] = {
                'points': [c['point'] for c in cs], 'excluding_zero': excluding,
                'verdict': ('not estimable' if len(signs) > 1 else 'result' if excluding == len(cs) else
                            'direction' if excluding else 'not distinguishable') + ' (seeds only)'}
    return out


def anticipation(variable, frame):
    old = AR.SINGLES
    AR.SINGLES = SINGLES
    try:
        return AR.anticipation(variable, frame)
    finally:
        AR.SINGLES = old


def predictions(b):
    out = {}
    with_nbeats = all('nbeats' in cell['arms'] for v in b for c in R.COHORTS for cell in b[v][c].values())
    for name, cohort in (('P1', 'Dev-2025'), ('P2', 'Reserved-2026')):
        c = b['oxygen'][cohort]['1']['contrasts']['arima_ss - chronos2_net']
        out[name] = {'held': bool(c['point'] < 0 and c['ci'][1] < 0), 'contrast': [c['point'], c['ci']]}
    fails = [[cohort, lead, b['temperature'][cohort][lead]['contrasts']['arima_ss - quantile_tree']['point']]
             for cohort in R.COHORTS for lead in ('24', '36', '48', '72')
             if b['temperature'][cohort][lead]['contrasts']['arima_ss - quantile_tree']['excludes_zero']]
    out['P3'] = {'held': not fails, 'failures': fails}
    fails = []
    for v in (b if with_nbeats else ()):
        for cohort in R.COHORTS:
            for lead, cell in b[v][cohort].items():
                s = cell['arms']
                best = min(s['quantile_tree']['quantile_score'], s['chronos2_net']['quantile_score'])
                if s['nbeats']['quantile_score'] <= best - R.FLOORS[v]:
                    fails.append([v, cohort, lead, s['nbeats']['quantile_score'], best])
    out['P4'] = {'held': not fails, 'failures': fails} if with_nbeats else {'held': None, 'note': 'nbeats not read'}
    fails = [[v, cohort, lead, cell['contrasts']['auto_arima - arima_ss']['point']]
             for v in b for cohort in R.COHORTS for lead, cell in b[v][cohort].items()
             if abs(cell['contrasts']['auto_arima - arima_ss']['point']) >= R.FLOORS[v]]
    out['P5'] = {'held': not fails, 'failures': fails}
    return out


def report(with_nbeats=True, levels=R.SCORE_LEVELS, target=None):
    """The reading; without N-BEATS it writes report_without_nbeats.json, a preliminary that nothing cites."""
    levels = tuple(levels)
    seeded = SEEDED if with_nbeats else ('arima_ss',)
    rep = {'declared_in': DECLARATION, 'leads': LEADS, 'draws': R.DRAWS, 'floors': R.FLOORS,
           'complete': with_nbeats}
    if R.secondary(levels):
        rep['levels'], rep['stored_levels'] = levels, P.LEVELS
        rep['eleven_levels'] = ('scores_eleven_levels: the same scores over the eleven stored levels 0.05-0.95; '
                                'the declared predictions are read on it')
    gates = {'G1_rows': {}, 'G2_forecast': {}, 'G3': 'asserted in the producers: the filter is causal and every '
                                                     'N-BEATS window ends at its issuance'}
    b, b11, a, studies, fallbacks = {}, {}, {}, {}, {}
    for variable in P.VARIABLES:
        frame, counts, g1 = wide(variable, seeded)
        gates['G1_rows'][variable] = g1
        for arm, seeds in (('arima_ss', P.SEEDS), ('auto_arima', (42,))) + ((('nbeats', P.SEEDS),) if with_nbeats else ()):
            for seed in seeds:
                meta = json.loads((OUT / f'seed{seed}' / f'statespace_{variable}_{arm}.json').read_text())
                if 'gates' in meta:
                    gates['G2_forecast'][f'{variable}|{arm}|{seed}'] = meta['gates']['G2']
                if 'study' in meta:
                    studies[f'{variable}|{arm}|{seed}'] = meta['study']
                if 'probes' in meta:
                    fallbacks[f'{variable}|{arm}|{seed}'] = {
                        split: {n: i.get('order') for n, i in probes.items() if i.get('fallback')}
                        for split, probes in meta['probes'].items()}
                    if arm == 'auto_arima':
                        studies[f'{variable}|auto_arima|orders'] = {
                            split: {n: i.get('order') for n, i in probes.items()} for split, probes in meta['probes'].items()}
        for block, lv in ((b, levels), (b11, P.LEVELS if R.secondary(levels) else None)):
            if lv is None:
                continue
            block[variable] = scores(variable, frame, lv)
            block[variable]['seeds'] = seed_verdicts(block[variable])
            block[variable]['row_counts'] = counts
        a[variable] = anticipation(variable, frame)
        print(variable, 'done', flush=True)
    ok = (all(x['passed'] for v in gates['G1_rows'].values() for x in v.values())
          and all(g['passed'] for g in gates['G2_forecast'].values()))
    rep.update({'gates': gates, 'gates_passed': ok, 'scores': b, 'anticipation': a, 'studies': studies,
                'fallbacks': fallbacks})
    rep['predictions'] = predictions(b11 or b)
    if b11:
        rep['predictions_nine_levels'] = predictions(b)
        rep['scores_eleven_levels'] = b11
    if target is None:
        target = OUT / ('report.json' if with_nbeats else 'report_without_nbeats.json')
    target = Path(target)
    target.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    print('wrote', target, 'gates', ok)
    print(json.dumps(rep['predictions'], default=str)[:2500])


if __name__ == '__main__':
    import sys
    without = '--without-nbeats' in sys.argv
    if without:
        sys.argv.remove('--without-nbeats')
    report(not without, *R.cli())
