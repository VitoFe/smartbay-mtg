"""Addendum 70: an equal-weight combination of the forecasters of the distribution, and conformal
recalibration of its central 80 % interval, on addendum 65's rows to 72 h.

E3 averages, level by level, the eleven quantiles of the quantile tree, Chronos-2 with the network
covariates and the state-space ARIMA (seed 42), and sorts each row (monotone rearrangement); E5 adds the
seasonal linear autoregression and N-BEATS. Nothing is fitted. cqr(X) widens or narrows X's central
80 % interval by the conformity scores of earlier rows (conformalized quantile regression with a causal
ledger: per variable and lead, probes pooled, only rows whose target hour is at or before the
issuance, Dev-2025 first and 2026 continued from it; raw interval below 30 scores). aci(E3) adapts the
level of that correction online (Gibbs and Candes 2021, gamma 0.005).

Reads the stored predictions only:
  python -m src.experiments.v2_ensemble [--levels 11] [--out PATH]

The quantile score reads the nine levels 0.1 to 0.9 (v2_probabilistic_report.SCORE_LEVELS); the combination
still averages and sorts all eleven stored levels. The eleven-level score stays in scores_eleven_levels, on
which the declared predictions are read. `--levels 11` writes the reading as it was before the scored levels
moved.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_probabilistic as P
from src.experiments import v2_probabilistic_report as R
from src.experiments import v2_statespace_report as SS
from src.experiments import v2_weather_oracle_report as W
from src.experiments.sweep_verdict import verdict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/bay_twin/runs/v2_ensemble'
STATESPACE = ROOT / 'reports/bay_twin/runs/v2_statespace/report.json'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum70.json'

E3 = ('quantile_tree', 'chronos2_net', 'arima_ss')
E5 = E3 + ('linear_ar', 'nbeats')
SEEDS = (7, 123)
NOMINAL = 0.8
ALPHA = 1 - NOMINAL
MIN_SCORES = 30
GAMMA = 0.005
LO, HI = P.LEVELS.index(0.1), P.LEVELS.index(0.9)
KEY = W.KEY


# --------------------------------------------------------------------------------------------------
# Rows and combinations
# --------------------------------------------------------------------------------------------------
def frame_for(variable):
    """addendum 67's frame (references, linear_ar, auto_arima, arima_ss and nbeats under every seed)
    with the quantile tree's seeds 7 and 123 from addendum 65's frame."""
    base, _, g1 = SS.wide(variable)
    full, _, _, _ = W.wide(variable)
    seeded = [c for c in full.columns if c.startswith('quantile_tree@')]
    base = base.merge(full[KEY + seeded], on=KEY, how='left')
    return base, g1


def quantile_block(frame, arm):
    return frame[[f'{arm}|{c}' for c in P.QCOLS]].to_numpy(float)


def combine(frame, members, name):
    """Equal-weight average of the members' quantiles, level by level, then sorted (G4)."""
    Q = np.mean([quantile_block(frame, m) for m in members], axis=0)
    Q = np.sort(Q, axis=1)
    assert (np.diff(Q, axis=1) >= 0).all()
    for i, c in enumerate(P.QCOLS):
        frame[f'{name}|{c}'] = Q[:, i]
    return frame


# --------------------------------------------------------------------------------------------------
# Conformal recalibration of the central 80 % interval
# --------------------------------------------------------------------------------------------------
def _ns(times):
    return pd.DatetimeIndex(times).tz_convert(None).as_unit('ns').to_numpy()


def conformal(frame, arm, adaptive=False):
    """Per lead, a causal ledger of conformity scores over every cohort in time order.

    Returns the issued interval (lo, hi) for every row of `frame` and whether it was recalibrated
    (False below MIN_SCORES available scores). With adaptive=True the level of the correction is
    alpha_t, updated by ALPHA - miss as each outcome becomes known.
    """
    lo = np.full(len(frame), np.nan)
    hi = np.full(len(frame), np.nan)
    recal = np.zeros(len(frame), bool)
    for lead, g in frame.groupby('lead'):
        idx = g.index.to_numpy()
        pos = frame.index.get_indexer(idx)
        t = _ns(g.time)
        target = t + np.timedelta64(int(lead), 'h')
        Q = quantile_block(g, arm)
        q10, q90 = Q[:, LO], Q[:, HI]
        y = g.truth.to_numpy(float)
        s = np.maximum(q10 - y, y - q90)
        order = np.argsort(target, kind='stable')
        known, scores = target[order], s[order]
        alpha = ALPHA
        pending = []                                  # (target time, miss) of recalibrated rows issued
        for ti in np.unique(t):
            rows = np.flatnonzero(t == ti)
            if adaptive:
                pending.sort(key=lambda p: p[0])
                while pending and pending[0][0] <= ti:
                    alpha += GAMMA * (ALPHA - pending.pop(0)[1])
            k = int(np.searchsorted(known, ti, side='right'))
            assert (known[:k] <= ti).all()                                      # G3
            if k < MIN_SCORES:
                lo[pos[rows]], hi[pos[rows]] = q10[rows], q90[rows]
                continue
            level = 1 - min(max(alpha, 0.001), 0.999) if adaptive else NOMINAL
            past = np.sort(scores[:k])
            Q = past[min(k - 1, max(0, math.ceil((k + 1) * level) - 1))]
            lo[pos[rows]], hi[pos[rows]] = q10[rows] - Q, q90[rows] + Q
            recal[pos[rows]] = True
            if adaptive:
                for r_ in rows:
                    miss = float(not (lo[pos[r_]] <= y[r_] <= hi[pos[r_]]))
                    pending.append((target[r_], miss))
    return lo, hi, recal


# --------------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------------
def interval_columns(cell, name, lo, hi):
    y = cell.truth.to_numpy(float)
    cell[f'c80|{name}'] = ((lo <= y) & (y <= hi)).astype(float)
    cell[f'w80|{name}'] = hi - lo
    cell[f'is80|{name}'] = (hi - lo) + (2 / ALPHA) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))


def score_cells(variable, frame, intervals, levels=R.SCORE_LEVELS):
    """Per (cohort, lead): quantile scores, interval coverage, width and interval score, and contrasts."""
    singles = list(E5) + ['E3', 'E5']
    out = {}
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        cell = cell.copy()
        y = cell.truth.to_numpy(float)
        arms = {}
        for arm in singles + [f'E3@{s}' for s in SEEDS] + [f'quantile_tree@{s}' for s in SEEDS]:
            if f'{arm}|q50' not in cell or cell[f'{arm}|q50'].isna().any():
                continue
            Q = quantile_block(cell, arm)
            cell[f'qs|{arm}'] = R.quantile_score(y, Q, levels)
            interval_columns(cell, f'raw:{arm}', Q[:, LO], Q[:, HI])
            arms[arm] = {'quantile_score': R.r(R.macro(cell, f'qs|{arm}'))}
        for name, (lo, hi, recal) in intervals.items():
            sel = frame.index.get_indexer(cell.index)
            interval_columns(cell, name, lo[sel], hi[sel])
        names = [f'raw:{a}' for a in arms] + list(intervals)
        bands = {n: {'coverage80': R.r(R.macro(cell, f'c80|{n}')), 'width80': R.r(R.macro(cell, f'w80|{n}')),
                     'interval_score80': R.r(R.macro(cell, f'is80|{n}'))} for n in names}
        for name, (lo, hi, recal) in intervals.items():
            bands[name]['raw_rows'] = int((~recal[frame.index.get_indexer(cell.index)]).sum())
        qs_cols = [f'qs|{a}' for a in arms]
        is_cols = [f'is80|{n}' for n in names]
        draws = R.joint_draws(cell, qs_cols + is_cols, f'ensemble|{variable}|{cohort}|{lead}')
        cols = qs_cols + is_cols
        contrasts = {}
        for x, yy in [('E3', m) for m in E3] + [('E5', 'E3')] + \
                     [(f'E3@{s}', f'quantile_tree@{s}') for s in SEEDS] + [(f'E3@{s}', 'chronos2_net') for s in SEEDS]:
            if x in arms and yy in arms:
                contrasts[f'qs|{x} - {yy}'] = R.contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
        for x, yy in [('cqr:E3', f'cqr:{m}') for m in E3] + [('cqr:E3', 'raw:E3'), ('aci:E3', 'cqr:E3'),
                                                            ('cqr:E5', 'cqr:E3')]:
            if x in bands and yy in bands:
                contrasts[f'is80|{x} - {yy}'] = R.contrast(cell, draws, cols, f'is80|{x}', f'is80|{yy}')
        members = {m: arms[m]['quantile_score'] for m in E3}
        out.setdefault(cohort, {})[str(lead)] = {
            'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()), 'arms': arms, 'bands': bands,
            'best_member': min(members, key=members.get), 'contrasts': contrasts}
    return out


def seed_verdicts(block):
    out = {}
    for lead, cell in block.get('Dev-2025', {}).items():
        c = cell['contrasts']
        for ref in ('quantile_tree', 'chronos2_net'):
            keys = [f'qs|E3 - {ref}'] + [f'qs|E3@{s} - {ref}@{s}' if ref == 'quantile_tree' else f'qs|E3@{s} - {ref}'
                                         for s in SEEDS]
            if not all(k in c for k in keys):
                continue
            cells = [(c[k]['point'], c[k]['ci'][0], c[k]['ci'][1]) for k in keys]
            out[f'{lead}|E3 - {ref}'] = {'points': [x[0] for x in cells], 'verdict': verdict(cells)}
    return out


def gate_g2(variable, block, levels=R.SCORE_LEVELS):
    """G2: every member's score and 80 % coverage equal the state-space report's, on the same levels."""
    stored = json.loads(STATESPACE.read_text())
    key = 'scores' if tuple(stored.get('levels', P.LEVELS)) == tuple(levels) else 'scores_eleven_levels'
    stored = stored[key][variable]
    worst = 0.0
    for cohort, leads in block.items():
        for lead, cell in leads.items():
            for arm in E5:
                ref = stored[cohort][lead]['arms'][arm]
                worst = max(worst, abs(cell['arms'][arm]['quantile_score'] - ref['quantile_score']),
                            abs(cell['bands'][f'raw:{arm}']['coverage80'] - ref['coverage80']))
    return {'max_abs_difference': worst, 'passed': worst <= 1e-6}


# --------------------------------------------------------------------------------------------------
# Predictions and decision
# --------------------------------------------------------------------------------------------------
def predictions(b):
    out = {}
    fails = [[c, l] for c in R.COHORTS for l, cell in b['oxygen'][c].items()
             if abs(cell['bands']['raw:E3']['coverage80'] - NOMINAL)
             >= abs(cell['bands']['raw:chronos2_net']['coverage80'] - NOMINAL)]
    out['P1'] = {'held': not fails, 'failures': fails}
    counts = {}
    for v in P.VARIABLES:
        for c in R.COHORTS:
            cells = b[v][c]
            wins = sum(cell['arms']['E3']['quantile_score'] < min(cell['arms'][m]['quantile_score'] for m in E3)
                       for cell in cells.values())
            counts[f'{v}|{c}'] = [wins, len(cells)]
    out['P2'] = {'held': all(w * 2 >= n for w, n in counts.values()), 'wins_of_cells': counts}
    fails = [[v, c, l, m, cell['contrasts'][f'qs|E3 - {m}']['point']] for v in P.VARIABLES for c in R.COHORTS
             for l, cell in b[v][c].items() for m in E3
             if abs(cell['contrasts'][f'qs|E3 - {m}']['point']) >= R.FLOORS[v]]
    out['P3'] = {'held': not fails, 'failures': fails}
    fails = [[v, c, l, cell['bands']['cqr:E3']['coverage80']] for v in P.VARIABLES for c in R.COHORTS
             for l, cell in b[v][c].items() if not 0.75 <= cell['bands']['cqr:E3']['coverage80'] <= 0.85]
    out['P4'] = {'held': not fails, 'failures': fails}
    cells = b['temperature']['Dev-2025']
    fails = [[l, cell['arms']['E5']['quantile_score'], cell['arms']['E3']['quantile_score']]
             for l, cell in cells.items() if l in ('24', '36', '48', '72')
             and not cell['arms']['E5']['quantile_score'] > cell['arms']['E3']['quantile_score']]
    out['P5'] = {'held': not fails, 'failures': fails}
    return out


def decision(b, p):
    worse = [[v, c, l, m, cell['contrasts'][f'is80|cqr:E3 - cqr:{m}']['point']]
             for v in P.VARIABLES for c in R.COHORTS for l, cell in b[v][c].items() for m in E3
             if cell['contrasts'][f'is80|cqr:E3 - cqr:{m}']['excludes_zero']
             and cell['contrasts'][f'is80|cqr:E3 - cqr:{m}']['point'] > 0]
    adopt = bool(p['P4']['held'] and not worse)
    return {'adopt_cqr_E3': adopt, 'coverage_within_0.75_0.85': p['P4']['held'],
            'cells_where_cqr_E3_is_worse_than_a_member': worse}


def main(levels=R.SCORE_LEVELS, target=None):
    levels = tuple(levels)
    eleven = R.secondary(levels)
    target = OUT / 'report.json' if target is None else Path(target)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {'declared_in': DECLARATION, 'members': {'E3': list(E3), 'E5': list(E5)}, 'nominal': NOMINAL,
           'min_scores': MIN_SCORES, 'gamma': GAMMA, 'draws': R.DRAWS, 'floors': R.FLOORS, 'gates': {},
           'scores': {}, 'seed_verdicts': {}}
    if eleven:
        rep.update({'levels': levels, 'stored_levels': P.LEVELS,
                    'eleven_levels': 'scores_eleven_levels: the same scores over the eleven stored levels 0.05-0.95; '
                                     'the declared predictions are read on it',
                    'scores_eleven_levels': {}, 'seed_verdicts_eleven_levels': {}})
    for variable in P.VARIABLES:
        frame, g1 = frame_for(variable)
        missing = {a: int(frame[[f'{a}|{c}' for c in P.QCOLS]].isna().any(axis=1).sum()) for a in E5}
        rep['gates'][f'G1|{variable}'] = {'missing_rows': missing, 'passed': not any(missing.values())}
        frame = frame.reset_index(drop=True)
        combine(frame, E3, 'E3')
        combine(frame, E5, 'E5')
        dev = frame.cohort == 'Dev-2025'
        for s in SEEDS:
            members = (f'quantile_tree@{s}', 'chronos2_net', f'arima_ss@{s}')
            Q = np.mean([quantile_block(frame, m) for m in members], axis=0)
            Q = np.sort(Q, axis=1)
            for i, c in enumerate(P.QCOLS):
                frame[f'E3@{s}|{c}'] = np.where(dev, Q[:, i], np.nan)
        intervals = {f'cqr:{a}': conformal(frame, a) for a in ('E3', 'E5') + E3}
        intervals['aci:E3'] = conformal(frame, 'E3', adaptive=True)
        block = score_cells(variable, frame, intervals, levels)
        rep['scores'][variable] = block
        rep['seed_verdicts'][variable] = seed_verdicts(block)
        rep['gates'][f'G2|{variable}'] = gate_g2(variable, block, levels)
        if eleven:
            b11 = score_cells(variable, frame, intervals, P.LEVELS)
            rep['scores_eleven_levels'][variable] = b11
            rep['seed_verdicts_eleven_levels'][variable] = seed_verdicts(b11)
            rep['gates'][f'G2_eleven_levels|{variable}'] = gate_g2(variable, b11, P.LEVELS)
        print(variable, 'done', flush=True)
    rep['gates']['G3'] = 'asserted: no conformity score enters a correction before its target hour'
    rep['gates']['G4'] = 'asserted: every combined row is non-decreasing in level'
    rep['gates_passed'] = all(g['passed'] for k, g in rep['gates'].items() if isinstance(g, dict))
    rep['predictions'] = predictions(rep['scores_eleven_levels'] if eleven else rep['scores'])
    rep['decision'] = decision(rep['scores'], rep['predictions'])
    if eleven:
        rep['predictions_nine_levels'] = predictions(rep['scores'])
    # the report has always been written in this host's text mode; keep the file's own line endings
    target.write_text(json.dumps(rep, indent=1) + '\n', encoding='utf-8')
    print('gates', rep['gates_passed'], json.dumps(rep['predictions'])[:900])
    print('decision', rep['decision'])


if __name__ == '__main__':
    main(*R.cli())
