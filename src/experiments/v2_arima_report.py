"""Addendum 66, the reading: the ARIMA family against addendum 65's forecasters, and GARCH-type bands.

Run locally after the campaign is fetched into reports/bay_twin/runs/v2_arima/:
  python -m src.experiments.v2_arima_report [--levels 11] [--out PATH]

The quantile score reads the nine levels 0.1 to 0.9 (v2_probabilistic_report.SCORE_LEVELS); the eleven-level
score stays in scores_eleven_levels, on which the declared predictions are read. `--levels 11` writes the
reading as it was before the scored levels moved.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.experiments import v2_probabilistic as P
from src.experiments import v2_probabilistic_report as R
from src.experiments import v2_weather_oracle_report as W
from src.experiments.v2_arima import DECLARATION, LEADS, OUT
from src.experiments.v2_weather_oracle import OUT as ORACLE

KEY = W.KEY
REFS = ('clim_quantiles', 'quantile_tree', 'chronos2_net')
FITTED = {'temperature': ('linear_ar', 'linear_ar_tree'), 'oxygen': ('linear_ar', 'linear_ar_joint', 'linear_ar_tree')}
LAMBDA, SENSITIVITY, MIN_FHS = 0.94, (0.90, 0.97, 0.99), 200
BANDED = {'chronos2_net_ewma': 'chronos2_net', 'linear_ar_ewma': 'linear_ar'}
CONTRASTS = (('linear_ar', 'clim_quantiles'), ('linear_ar', 'quantile_tree'), ('linear_ar', 'chronos2_net'),
             ('linear_ar_tree', 'linear_ar'), ('linear_ar_tree', 'quantile_tree'), ('linear_ar_joint', 'linear_ar'),
             ('chronos2_net_ewma', 'chronos2_net'), ('linear_ar_ewma', 'linear_ar'))
SEED_CONTRASTS = (('linear_ar', 'clim_quantiles'), ('linear_ar_tree', 'linear_ar'), ('linear_ar_joint', 'linear_ar'))
MEDIAN_CONTRASTS = (('linear_ar', 'quantile_tree'), ('linear_ar', 'chronos2_net'), ('linear_ar_tree', 'quantile_tree'))
SINGLES = ('linear_ar', 'linear_ar_joint', 'linear_ar_tree', 'quantile_tree', 'chronos2_net')


def fitted_rows(variable, arm, seed):
    return W.read(OUT / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz')


def gate_rows(variable):
    """G1: linear_ar's rows (keys, truth, last value) equal the quantile tree's rows of addenda 64 and 65."""
    d = P.DIGITS[variable]
    ref = pd.concat([W.read(P.OUT / 'seed42' / f'points_{variable}.csv.gz'),
                     W.read(ORACLE / 'seed42' / f'points72_{variable}.csv.gz')], ignore_index=True)
    ref = ref[KEY + ['truth', 'last']].copy()
    ref[['truth', 'last']] = ref[['truth', 'last']].round(d)
    mine = fitted_rows(variable, 'linear_ar', 42)[KEY + ['truth', 'last']]
    out = {}
    for cohort, r in ref.groupby('cohort'):
        m = mine[mine.cohort == cohort]
        j = m.merge(r, on=KEY, how='outer', suffixes=('', '_tree'), indicator=True)
        both = j[j._merge == 'both']
        diff = float(max((both.truth - both.truth_tree).abs().max(), (both['last'] - both.last_tree).abs().max()))
        out[cohort] = {'rows': int(len(m)), 'rows_tree': int(len(r)), 'unmatched': int((j._merge != 'both').sum()),
                       'max_abs_difference': diff, 'passed': bool((j._merge == 'both').all() and diff <= 1e-9)}
    return out


# --------------------------------------------------------------------------------------------------
# GARCH-type bands
# --------------------------------------------------------------------------------------------------
def ewma_bands(rows, sigma0, lam):
    """One forecaster's rows at one lead (sensor, time, lead, truth, q50; both cohorts, any order): the band
    quantiles from an exponentially weighted error variance and filtered historical simulation. At issuance
    t only errors whose target time is at or before t enter (G3, counted)."""
    lead = int(rows.lead.iloc[0])
    f = rows.sort_values(['sensor', 'time']).reset_index(drop=True)
    e = (f.truth - f.q50).to_numpy(float)
    t = f.time.to_numpy('datetime64[ns]')
    tau = t + np.timedelta64(lead, 'h')
    sigma = np.empty(len(f))
    checked = violations = 0
    fallback = float(np.median(list(sigma0.values())))
    missing = set()
    for name, idx in f.groupby('sensor').indices.items():
        if name not in sigma0:
            missing.add(name)
        s2 = sigma0.get(name, fallback) ** 2
        j = 0
        for i in idx:
            while j < len(idx) and tau[idx[j]] <= t[i]:
                checked += 1
                violations += int(tau[idx[j]] > t[i])
                if np.isfinite(e[idx[j]]):
                    s2 = lam * s2 + (1 - lam) * e[idx[j]] ** 2
                j += 1
            sigma[i] = np.sqrt(s2)
    u = e / sigma
    order = np.argsort(tau, kind='stable')
    tau_sorted, u_sorted = tau[order], u[order]
    z_normal = norm.ppf(P.LEVELS)
    z = {}
    for ti in np.unique(t):
        k = int(np.searchsorted(tau_sorted, ti, side='right'))
        known = u_sorted[:k]
        known = known[np.isfinite(known)]
        z[ti] = np.quantile(known, P.LEVELS) if len(known) >= MIN_FHS else z_normal
    Z = np.stack([z[ti] for ti in t])
    Q = f.q50.to_numpy(float)[:, None] + sigma[:, None] * Z
    out = f[KEY].copy()
    for n, c in enumerate(P.QCOLS):
        out[c] = Q[:, n]
    return out, {'checked': checked, 'violations': violations, 'missing_sigma0': sorted(missing)}


def banded(variable, lam):
    init = json.loads((OUT / 'seed42' / f'ewma_init_{variable}.json').read_text())['sigma0']
    # both streams run over the quantile tree's rows (addenda 64 and 65), which carry the truth
    points = pd.concat([W.read(P.OUT / 'seed42' / f'points_{variable}.csv.gz'),
                        W.read(ORACLE / 'seed42' / f'points72_{variable}.csv.gz')], ignore_index=True)[KEY + ['truth']]
    chronos = W.read(ORACLE / 'chronos' / f'chronos_{variable}.csv.gz').query("arm == 'chronos2_net'")
    sources = {'chronos2_net': chronos[KEY + ['q50']].merge(points, on=KEY, how='inner'),
               'linear_ar': fitted_rows(variable, 'linear_ar', 42)}
    frames, gate = {}, {}
    for arm, base in BANDED.items():
        parts = []
        for lead, g in sources[base].groupby('lead'):
            sigma0 = {k: v['sigma0'] for k, v in init[str(lead)].items()}
            q, info = ewma_bands(g[KEY + ['truth', 'q50']], sigma0, lam)
            parts.append(q)
            gate[f'{arm}|{lead}'] = info
        frames[arm] = pd.concat(parts, ignore_index=True)
    return frames, gate


# --------------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------------
def wide(variable):
    full, _, counts, _ = W.wide(variable)
    keep = [c for c in full.columns if '|' not in c or c.split('|')[0] in REFS]
    base = full[keep].copy()
    added = []
    for arm in FITTED[variable]:
        for seed in P.SEEDS:
            g = fitted_rows(variable, arm, seed)
            name = W.tag(arm, seed)
            g = g[KEY + P.QCOLS].rename(columns={c: f'{name}|{c}' for c in P.QCOLS})
            base = base.merge(g, on=KEY, how='left')
            added.append(name)
    gates = {}
    for lam in (LAMBDA,) + SENSITIVITY:
        frames, gate = banded(variable, lam)
        for arm, g in frames.items():
            name = arm if lam == LAMBDA else f'{arm}@l{int(round(lam * 100))}'
            base = base.merge(g.rename(columns={c: f'{name}|{c}' for c in P.QCOLS}), on=KEY, how='left')
            added.append(name)
        if lam == LAMBDA:
            gates = gate
    dev = base.cohort == 'Dev-2025'
    for name in added:
        rows = dev if '@' in name and not name.split('@')[1].startswith('l') else slice(None)
        if base.loc[rows, f'{name}|q50'].isna().any():
            raise ValueError(f'{variable} {name}: missing on rows the references score')
    return base, counts, gates


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
        tested = [a for a in arms if '@l' not in a]
        cols = [f'qs|{a}' for a in tested] + [f'ae|{a}' for a in tested]
        draws = R.joint_draws(cell, cols, f'arima|{variable}|{cohort}|{lead}')
        contrasts = {}
        pairs = list(CONTRASTS) + [(f'{x}@{s}', W.tag(yy, s)) for s in P.SEEDS[1:] for x, yy in SEED_CONTRASTS]
        for x, yy in pairs:
            if x in tested and yy in tested:
                contrasts[f'{x} - {yy}'] = R.contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
        for x, yy in MEDIAN_CONTRASTS:
            if x in tested and yy in tested:
                contrasts[f'median_mae|{x} - {yy}'] = R.contrast(cell, draws, cols, f'ae|{x}', f'ae|{yy}')
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
    """Addendum 64's fall and rise shares, with addendum 65's draw labels so the references match."""
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
        for arm in [a for a in SINGLES if f'{a}|q50' in cell] + ['clim_hour']:
            f = (cell.clim_hour if arm == 'clim_hour' else cell[f'{arm}|q50']).to_numpy(float)
            cell['ae|f'] = np.abs(y - f)
            fs, fci = R.share_draws(cell, f, fall, f'oracle|antic|{variable}|{cohort}|{lead}|{arm}|fall')
            rs, rci = R.share_draws(cell, f, rise, f'oracle|antic|{variable}|{cohort}|{lead}|{arm}|rise')
            block[arm] = {'fall_share': fs, 'fall_ci': fci, 'rise_share': rs, 'rise_ci': rci,
                          'skill_vs_last': R.r(1 - R.macro(cell, 'ae|f') / R.macro(cell, 'ae|last'))}
        out.setdefault(cohort, {})[str(lead)] = block
    return out


def probe_errors(frame):
    """Each probe's mean absolute error of the median on Dev-2025, per fold, at 24, 48 and 72 h: linear_ar
    against the quantile tree, to show where a macro score comes from."""
    out = {}
    dev = frame[(frame.cohort == 'Dev-2025') & frame.lead.isin([24, 48, 72])]
    for (lead, fold, sensor), g in dev.groupby(['lead', 'fold', 'sensor']):
        y = g.truth.to_numpy(float)
        out[f'{lead}|{fold}|{sensor}'] = {arm: R.r(float(np.mean(np.abs(y - g[f'{arm}|q50'].to_numpy(float)))))
                                         for arm in ('linear_ar', 'quantile_tree')}
    return out


def selections(variable):
    """The selected linear configurations, and whether the hybrid's own linear study chose linear_ar's."""
    out, same = {}, {}
    for seed in P.SEEDS:
        lin = json.loads((OUT / f'seed{seed}' / f'arima_{variable}_linear_ar.json').read_text())['studies']
        hyb = json.loads((OUT / f'seed{seed}' / f'arima_{variable}_linear_ar_tree.json').read_text())['studies']
        out[str(seed)] = {k: v['params'] for k, v in lin.items()}
        for lead in LEADS:
            same[f'{seed}|{lead}'] = lin[f'linear_ar|{lead}']['params'] == hyb[f'linear_ar|{lead}']['params']
        if variable == 'oxygen':
            joint = json.loads((OUT / f'seed{seed}' / f'arima_{variable}_linear_ar_joint.json').read_text())['studies']
            out[f'{seed}|joint'] = {k: v['params'] for k, v in joint.items()}
    return out, same


def predictions(b, a):
    out = {}
    cells = b['temperature']['Dev-2025']
    fails = [[lead, ref] for lead in ('48', '72') for ref in ('quantile_tree', 'chronos2_net')
             if not cells[lead]['arms']['linear_ar']['quantile_score'] < cells[lead]['arms'][ref]['quantile_score']]
    out['P1'] = {'held': not fails, 'failures': fails}
    cells = b['temperature']['Reserved-2026']
    got = {lead: cells[lead]['contrasts']['linear_ar - chronos2_net'] for lead in ('48', '72')}
    out['P2'] = {'held': any(c['point'] < 0 and c['ci'][1] < 0 for c in got.values()),
                 'contrasts': {k: [v['point'], v['ci']] for k, v in got.items()}}
    fails = []
    for cohort in R.COHORTS:
        for lead, cell in b['oxygen'][cohort].items():
            for name, c in cell['contrasts'].items():
                if name.startswith('median_mae|') or '@' in name:
                    continue
                x, y = name.split(' - ')
                if ({x, y} & {'linear_ar', 'linear_ar_joint', 'linear_ar_tree'}) and abs(c['point']) >= R.FLOORS['oxygen']:
                    fails.append([cohort, lead, name, c['point']])
            j = cell['contrasts'].get('linear_ar_joint - linear_ar')
            if j is not None and abs(j['point']) > 0.15:
                fails.append([cohort, lead, 'linear_ar_joint - linear_ar beyond 0.15', j['point']])
    out['P3'] = {'held': not fails, 'failures': fails}
    fails = []
    for v in b:
        for cohort in R.COHORTS:
            for lead, cell in b[v][cohort].items():
                s = cell['arms']
                best = min(s['linear_ar']['quantile_score'], s['quantile_tree']['quantile_score'])
                if s['linear_ar_tree']['quantile_score'] <= best - R.FLOORS[v]:
                    fails.append([v, cohort, lead, s['linear_ar_tree']['quantile_score'], best])
    out['P4'] = {'held': not fails, 'failures': fails}
    fails = []
    for v in b:
        for cohort in R.COHORTS:
            for lead, cell in b[v][cohort].items():
                s = cell['arms']
                cov = s['chronos2_net_ewma']['coverage80']
                gap = s['chronos2_net_ewma']['quantile_score'] - s['chronos2_net']['quantile_score']
                if not 0.75 <= cov <= 0.85 or gap >= R.FLOORS[v]:
                    fails.append([v, cohort, lead, cov, R.r(gap)])
    out['P5'] = {'held': not fails, 'failures': fails}
    fails = [[v, cohort, lead, arm, a[v][cohort][lead][arm]['fall_share']]
             for v in a for cohort in R.COHORTS for lead in ('24', '48', '72') for arm in ('linear_ar', 'linear_ar_tree')
             if a[v][cohort][lead][arm]['fall_share'] is not None and a[v][cohort][lead][arm]['fall_share'] > 0.4]
    out['P6'] = {'held': not fails, 'failures': fails}
    return out


def report(levels=R.SCORE_LEVELS, target=None):
    levels = tuple(levels)
    target = OUT / 'report.json' if target is None else Path(target)
    rep = {'declared_in': DECLARATION, 'leads': LEADS, 'draws': R.DRAWS, 'floors': R.FLOORS,
           'bands': {'lambda': LAMBDA, 'sensitivity': SENSITIVITY, 'min_standardised_errors': MIN_FHS}}
    if R.secondary(levels):
        rep['levels'], rep['stored_levels'] = levels, P.LEVELS
        rep['eleven_levels'] = ('scores_eleven_levels: the same scores over the eleven stored levels 0.05-0.95; '
                                'the declared predictions are read on it')
    gates = {'G1_rows': {}, 'G2_hourly': {}, 'G3_bands': {}}
    b, b11, a, sel = {}, {}, {}, {}
    for variable in P.VARIABLES:
        gates['G1_rows'][variable] = gate_rows(variable)
        for arm in FITTED[variable]:
            for seed in P.SEEDS:
                g = json.loads((OUT / f'seed{seed}' / f'arima_{variable}_{arm}.json').read_text())['gates']
                gates['G2_hourly'][f'{variable}|{arm}|{seed}'] = all(x['passed'] for k, x in g.items()
                                                                     if k.startswith('G2'))
        frame, counts, g3 = wide(variable)
        gates['G3_bands'][variable] = {'checked': sum(x['checked'] for x in g3.values()),
                                       'violations': sum(x['violations'] for x in g3.values()),
                                       'missing_sigma0': sorted({m for x in g3.values() for m in x['missing_sigma0']})}
        for block, lv in ((b, levels), (b11, P.LEVELS if R.secondary(levels) else None)):
            if lv is None:
                continue
            block[variable] = scores(variable, frame, lv)
            block[variable]['seeds'] = seed_verdicts(block[variable])
            block[variable]['row_counts'] = counts
        a[variable] = anticipation(variable, frame)
        b[variable]['probe_errors'] = probe_errors(frame)
        sel[variable] = dict(zip(('selected', 'hybrid_linear_matches'), selections(variable)))
        print(variable, 'done', flush=True)
    ok = (all(x['passed'] for v in gates['G1_rows'].values() for x in v.values())
          and all(gates['G2_hourly'].values())
          and all(g['violations'] == 0 for g in gates['G3_bands'].values()))
    rep.update({'gates': gates, 'gates_passed': ok, 'scores': b, 'anticipation': a, 'selections': sel})
    rep['predictions'] = predictions(b11 or b, a)
    if b11:
        rep['predictions_nine_levels'] = predictions(b, a)
        rep['scores_eleven_levels'] = b11
    target.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    print('wrote', target, 'gates', ok)
    print(json.dumps(rep['predictions'], default=str)[:2500])


if __name__ == '__main__':
    report(*R.cli())
