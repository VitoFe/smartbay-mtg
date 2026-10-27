"""Addendum 65, the reading: what a perfect weather forecast adds, from what v2_weather_oracle wrote.

Run locally after the campaign is fetched into reports/bay_twin/runs/v2_weather_oracle/:
  python -m src.experiments.v2_weather_oracle_report [--levels 11] [--out PATH]

The quantile score reads the nine levels 0.1 to 0.9 (v2_probabilistic_report.SCORE_LEVELS); the eleven-level
score stays in scores_eleven_levels, on which the declared predictions are read. `--levels 11` writes the
reading as it was before the scored levels moved.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_probabilistic as P
from src.experiments import v2_probabilistic_report as R
from src.experiments.v2_weather_oracle import DECLARATION, LEADS, OUT, PROB, TREE_ARMS, WARNING_ARMS

KEY = ['cohort', 'sensor', 'time', 'lead']
ARMS = ('clim_quantiles', 'quantile_tree', 'quantile_tree_weather', 'quantile_tree_weather_future',
        'chronos2_net', 'chronos2_net_weather', 'chronos2_net_weather_future')
CONTRASTS = (('quantile_tree_weather_future', 'quantile_tree_weather'), ('quantile_tree_weather', 'quantile_tree'),
             ('chronos2_net_weather_future', 'chronos2_net_weather'), ('chronos2_net_weather', 'chronos2_net'),
             ('quantile_tree_weather_future', 'clim_quantiles'), ('chronos2_net_weather_future', 'clim_quantiles'),
             ('chronos2_net_weather_future', 'quantile_tree_weather_future'),
             ('quantile_tree', 'clim_quantiles'), ('chronos2_net', 'clim_quantiles'))
SEED_CONTRASTS = (('quantile_tree_weather_future', 'quantile_tree_weather'), ('quantile_tree_weather', 'quantile_tree'),
                  ('quantile_tree', 'clim_quantiles'))
SINGLES = ('quantile_tree', 'quantile_tree_weather', 'quantile_tree_weather_future', 'chronos2_net',
           'chronos2_net_weather', 'chronos2_net_weather_future')


def read(path):
    f = pd.read_csv(path)
    f['time'] = R.utc(f.time)
    return f


def tag(arm, seed):
    return arm if seed == 42 or arm == 'clim_quantiles' else f'{arm}@{seed}'


def long_frames(variable):
    """Every arm's quantile rows: addendum 64's plain tree and climatology at 1 to 48 h, this
    addendum's weather trees at every lead and the plain tree and climatology at 72 h, Chronos-2."""
    parts = []
    for seed in P.SEEDS:
        f = read(PROB / f'seed{seed}' / f'quantiles_{variable}_quantile_tree.csv.gz')
        parts.append(f.assign(arm=[tag(a, seed) for a in f.arm]))
        for arm in TREE_ARMS:
            f = read(OUT / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz')
            parts.append(f.assign(arm=[tag(a, seed) for a in f.arm]))
    parts.append(read(OUT / 'chronos' / f'chronos_{variable}.csv.gz'))
    return parts


def gate_rows(variable, points, parts):
    """G1: every tree row of this addendum at 1 to 48 h carries addendum 64's keys, truth and last value."""
    d = P.DIGITS[variable]
    ref = points[points.lead <= 48][KEY + ['truth', 'last']].copy()
    ref[['truth', 'last']] = ref[['truth', 'last']].round(d)
    out = {}
    for f in parts:
        for arm in f.arm.unique():
            if not arm.startswith('quantile_tree_weather'):
                continue
            g = f[(f.arm == arm) & (f.lead <= 48)]
            for cohort, h in g.groupby('cohort'):
                r = ref[ref.cohort == cohort]
                j = h[KEY + ['truth', 'last']].merge(r, on=KEY, how='outer', suffixes=('', '_64'), indicator=True)
                both = j[j._merge == 'both']
                diff = float(max((both.truth - both.truth_64).abs().max(), (both['last'] - both.last_64).abs().max()))
                out[f'{arm}|{cohort}'] = {'rows': int(len(h)), 'rows_64': int(len(r)),
                                          'unmatched': int((j._merge != 'both').sum()), 'max_abs_difference': diff,
                                          'passed': bool((j._merge == 'both').all() and diff <= 1e-9)}
    return out


def wide(variable):
    p64 = read(PROB / 'seed42' / f'points_{variable}.csv.gz')
    p72 = read(OUT / 'seed42' / f'points72_{variable}.csv.gz')
    points = pd.concat([p64, p72], ignore_index=True)
    parts = long_frames(variable)
    gates = gate_rows(variable, points, parts)
    base = points.copy()
    for f in parts:
        for arm, g in f.groupby('arm'):
            g = g[KEY + P.QCOLS].rename(columns={c: f'{arm}|{c}' for c in P.QCOLS})
            if g.duplicated(KEY).any():
                raise ValueError(f'{variable} {arm}: duplicated rows')
            base = base.merge(g, on=KEY, how='left')
            # an arm present in two files (the plain tree at 1-48 h and at 72 h) arrives as _x/_y columns
            for c in P.QCOLS:
                x, y = f'{arm}|{c}_x', f'{arm}|{c}_y'
                if x in base:
                    base[f'{arm}|{c}'] = base[x].combine_first(base[y])
                    base = base.drop(columns=[x, y])
    before = base.groupby(['cohort', 'lead']).size()
    full = base.dropna(subset=[f'{a}|q50' for a in ARMS]).copy()
    seeded = [c for c in full.columns if '@' in c and c.endswith('|q50')]
    dev = full.cohort == 'Dev-2025'
    if full.loc[dev, seeded].isna().any().any():
        raise ValueError(f'{variable}: a seed arm is missing on Dev-2025 rows')
    full['fold'] = np.where(full.cohort == 'Dev-2025', full.fold.astype('float').astype('Int64').astype(str),
                            np.where(full.time.dt.month <= 3, 'Q1', 'Q2'))
    full['week'] = full.time.dt.isocalendar().week.astype(int).to_numpy()
    after = full.groupby(['cohort', 'lead']).size()
    counts = {f'{c}|{l}': {'rows': int(before.get((c, l), 0)), 'kept': int(after.get((c, l), 0))} for c, l in before.index}
    arms = sorted({c.split('|')[0] for c in full.columns if c.endswith('|q50')})
    return full, arms, counts, gates


def scores(variable, frame, arms, levels=R.SCORE_LEVELS):
    out = {}
    lv = P.LEVELS
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        cell = cell.copy()
        y = cell.truth.to_numpy(float)
        here = [a for a in arms if cell[f'{a}|q50'].notna().all()]
        cell['qs|last_value'] = np.abs(y - cell['last'].to_numpy(float))   # a single value: its absolute error
        stats = {'last_value': {'quantile_score': R.r(R.macro(cell, 'qs|last_value'))}}
        for arm in here:
            Q = cell[[f'{arm}|{c}' for c in P.QCOLS]].to_numpy(float)
            cell[f'qs|{arm}'] = R.quantile_score(y, Q, levels)
            cell[f'c80|{arm}'] = ((Q[:, lv.index(0.1)] <= y) & (y <= Q[:, lv.index(0.9)])).astype(float)
            stats[arm] = {'quantile_score': R.r(R.macro(cell, f'qs|{arm}')),
                          'coverage80': R.r(R.macro(cell, f'c80|{arm}'))}
        cols = [f'qs|{a}' for a in here]
        draws = R.joint_draws(cell, cols, f'oracle|{variable}|{cohort}|{lead}')
        contrasts = {}
        pairs = list(CONTRASTS) + [(f'{x}@{s}', tag(y, s)) for s in P.SEEDS[1:] for x, y in SEED_CONTRASTS]
        for x, yy in pairs:
            if x in here and yy in here:
                contrasts[f'{x} - {yy}'] = R.contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
        out.setdefault(cohort, {})[str(lead)] = {'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()),
                                                'arms': stats, 'contrasts': contrasts}
    return out


def seed_verdicts(block):
    out = {}
    for lead, cell in block.get('Dev-2025', {}).items():
        for x, y in SEED_CONTRASTS:
            cs = [cell['contrasts'].get(f'{x} - {y}' if s == 42 else f'{x}@{s} - {tag(y, s)}') for s in P.SEEDS]
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
    out = {}
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        cell = cell.copy()
        y, y0 = cell.truth.to_numpy(float), cell['last'].to_numpy(float)
        d = y - y0
        fall, rise = np.zeros(len(cell), bool), np.zeros(len(cell), bool)
        for _, idx in cell.groupby('sensor').indices.items():
            fall[idx] = d[idx] <= np.quantile(d[idx], R.TAIL)
            rise[idx] = d[idx] >= np.quantile(d[idx], 1 - R.TAIL)
        cell['ae|last'] = np.abs(d)
        block = {}
        for arm in SINGLES + ('clim_hour',):
            f = (cell.clim_hour if arm == 'clim_hour' else cell[f'{arm}|q50']).to_numpy(float)
            cell['ae|f'] = np.abs(y - f)
            fs, fci = R.share_draws(cell, f, fall, f'oracle|antic|{variable}|{cohort}|{lead}|{arm}|fall')
            rs, rci = R.share_draws(cell, f, rise, f'oracle|antic|{variable}|{cohort}|{lead}|{arm}|rise')
            block[arm] = {'fall_share': fs, 'fall_ci': fci, 'rise_share': rs, 'rise_ci': rci,
                          'skill_vs_last': R.r(1 - R.macro(cell, 'ae|f') / R.macro(cell, 'ae|last'))}
        out.setdefault(cohort, {})[str(lead)] = block
    return out


def alert():
    """Addendum 64's alert rule, grid and selection, on the weather-informed hourly forecasts."""
    curve = R._curve()
    data = curve.load()
    warn = read(OUT / 'chronos' / 'chronos_warning.csv.gz')
    periods = {name: {s: p for s, ser in data.items() if (p := curve.period(ser, lo, hi)) is not None}
               for name, (lo, hi) in (('development', R.DEVELOPMENT), ('2026', R.RESERVED))}
    gate = {}
    pub = {'development': pd.read_csv(P.RUNS / 'v2_warning_curve/warning_curve_development.csv'),
           '2026': pd.read_csv(P.RUNS / 'v2_warning_curve/warning_curve_reserved2026.csv')}
    for name, sensors in periods.items():
        for rate, consec, cd in ((1.5, 2, 6), (5.0, 2, 3)):
            mine = R.aggregate({s: R.score(h, n, R.rate_alerts(h, rate, consec, cd), curve) for s, (h, n) in sensors.items()})
            row = pub[name][(pub[name].rate == rate) & (pub[name].consec == consec) & (pub[name].cooldown == cd)].iloc[0]
            gate[f'{name}|{rate}|{consec}|{cd}'] = bool(mine['alerts'] == row.alerts and mine['crossings'] == row.crossings
                                                        and abs(mine['recall_causal'] - row.recall_causal) < 1e-9)
    episodes = {}
    for s, (h, n) in periods['2026'].items():
        drop6 = -(h - h.shift(curve.EVENT_WIN))
        cr = list(drop6[drop6 > curve.EVENT_DROP].dropna().index)
        episodes[s] = R.v2_warning_episodes.cluster(cr, 12) if cr else []
    gate['passed'] = all(gate.values())
    out = {'gate_G3': gate}
    if not gate['passed']:
        return out

    def evaluate(arm, period, alpha, c, cd):
        per, alerts = {}, {}
        for s, (h, n) in periods[period].items():
            a = R.cooldown_filter(sorted(R.forecast_flags(warn, h, s, arm, alpha, c)), cd)
            alerts[s] = a
            per[s] = R.score(h, n, a, curve)
        return R.aggregate(per), alerts
    for arm in WARNING_ARMS:
        rows = [{'alpha': alpha, 'c': c, 'cooldown': cd, **evaluate(arm, 'development', alpha, c, cd)[0]}
                for alpha in R.GRID['alpha'] for c in R.GRID['c'] for cd in R.GRID['cooldown']]
        dev = pd.DataFrame(rows)
        cand = dev[(dev.recall_causal >= 0.5) & (dev.alerts > 0)]
        if not len(cand):
            out[arm] = {'selected': None}
            continue
        sel = cand.sort_values('precision_causal', ascending=False, kind='stable').iloc[0]
        agg, alerts = evaluate(arm, '2026', float(sel.alpha), int(sel.c), int(sel.cooldown))
        out[arm] = {'selected': {'alpha': float(sel.alpha), 'c': int(sel.c), 'cooldown': int(sel.cooldown),
                                 'development': {k: R.r(sel[k]) for k in ('alerts', 'crossings', 'precision_causal',
                                                                         'recall_causal', 'median_lead_h')}},
                    '2026': {k: R.r(v) for k, v in agg.items()},
                    '2026_episodes': R.episode_reading(alerts, episodes)}
    return out


def length_check(variable, frame):
    """chronos2_net at prediction length 48 (addendum 64) against 72 (here), medians on common rows."""
    a = read(PROB / 'chronos' / f'chronos_{variable}.csv.gz')
    a = a[a.arm == 'chronos2_net'][KEY + ['q50']]
    b = read(OUT / 'chronos' / f'chronos_{variable}.csv.gz')
    b = b[b.arm == 'chronos2_net'][KEY + ['q50']]
    j = a.merge(b, on=KEY, suffixes=('_48', '_72'))
    d = (j.q50_48 - j.q50_72).abs()
    return {'rows': int(len(j)), 'max_abs_difference': R.r(float(d.max())), 'mean_abs_difference': R.r(float(d.mean()))}


def predictions(b, a, c):
    out = {}
    fails = []
    for v in b:
        for cohort in R.COHORTS:
            for lead, cell in b[v].get(cohort, {}).items():
                for x in ('quantile_tree_weather_future - quantile_tree_weather',
                          'chronos2_net_weather_future - chronos2_net_weather'):
                    if cell['contrasts'][x]['point'] < -R.FLOORS[v]:
                        fails.append([v, cohort, lead, x, cell['contrasts'][x]['point']])
    out['P1'] = {'held': not fails, 'failures': fails}
    fails = []
    for cohort in R.COHORTS:
        cells = b['temperature'][cohort]
        for x in ('quantile_tree_weather_future - quantile_tree_weather',
                  'chronos2_net_weather_future - chronos2_net_weather'):
            best = min(cells, key=lambda L: cells[L]['contrasts'][x]['point'])
            if int(best) < 24:
                fails.append([cohort, x, 'largest gain at', best])
            if x.startswith('chronos') and not cells[best]['contrasts'][x]['ci'][1] < 0:
                fails.append([cohort, x, 'interval spans zero at', best])
    out['P2'] = {'held': not fails, 'failures': fails}
    fails = [['oxygen', cohort, lead, arm, x['fall_share']]
             for cohort in R.COHORTS for lead in ('24', '36', '48', '72')
             for arm, x in a['oxygen'][cohort][lead].items()
             if arm.endswith('_future') and x['fall_share'] is not None and x['fall_share'] >= 0.3]
    out['P3'] = {'held': not fails, 'failures': fails}
    e = c.get('chronos2_net_weather_future', {}).get('2026_episodes')
    out['P4'] = ({'held': e['alerted_6h_or_more_ahead'] == 0, 'leads_h': e['leads_h'], 'matched': e['matched']}
                 if e else {'held': None, 'note': 'no selected point'})
    return out


def report(levels=R.SCORE_LEVELS, target=None):
    levels = tuple(levels)
    target = OUT / 'report.json' if target is None else Path(target)
    rep = {'declared_in': DECLARATION, 'leads': LEADS, 'draws': R.DRAWS, 'floors': R.FLOORS}
    if R.secondary(levels):
        rep['levels'], rep['stored_levels'] = levels, P.LEVELS
        rep['eleven_levels'] = ('scores_eleven_levels: the same scores over the eleven stored levels 0.05-0.95; '
                                'the declared predictions are read on it')
    tree = {v: json.loads((OUT / 'seed42' / f'tree_{v}_quantile_tree_weather.json').read_text())['gates']
            for v in P.VARIABLES}
    rep['gates'] = {'G2_weather': tree, 'G1_rows': {}}
    b, b11, a, lengths = {}, {}, {}, {}
    for variable in P.VARIABLES:
        frame, arms, counts, g1 = wide(variable)
        rep['gates']['G1_rows'][variable] = g1
        for block, lv in ((b, levels), (b11, P.LEVELS if R.secondary(levels) else None)):
            if lv is None:
                continue
            block[variable] = scores(variable, frame, arms, lv)
            block[variable]['seeds'] = seed_verdicts(block[variable])
            block[variable]['row_counts'] = counts
        a[variable] = anticipation(variable, frame)
        lengths[variable] = length_check(variable, frame)
        print(variable, 'done', flush=True)
    ok = all(x['passed'] for v in rep['gates']['G1_rows'].values() for x in v.values()) and \
        all(t['G2_weather']['passed'] for t in tree.values() if 'G2_weather' in t)
    rep['gates_passed'] = ok
    rep['scores'], rep['anticipation'], rep['prediction_length_check'] = b, a, lengths
    rep['alert'] = alert()
    rep['predictions'] = predictions(b11 or b, a, rep['alert'])
    if b11:
        rep['predictions_nine_levels'] = predictions(b, a, rep['alert'])
        rep['scores_eleven_levels'] = b11
    target.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    print('wrote', target, 'gates', ok)
    print(json.dumps(rep['predictions'], default=str)[:2000])


if __name__ == '__main__':
    report(*R.cli())
