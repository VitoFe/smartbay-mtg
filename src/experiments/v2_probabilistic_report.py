"""Addendum 64, the reading: parts A to D from what v2_probabilistic wrote.

Run locally after the campaign is fetched into reports/bay_twin/runs/v2_probabilistic/:
  python -m src.experiments.v2_probabilistic --report
  python -m src.experiments.v2_probabilistic_report [--levels 11] [--out PATH]

The quantile score reads the nine levels SCORE_LEVELS (0.1 to 0.9), which every forecaster of the
distribution gives, the pretrained ones included (addendum 81). The stored quantile sets keep the eleven
LEVELS; the eleven-level score stays in the report's secondary block (part_b_eleven_levels), the only
place the central 90 % interval (0.05 to 0.95) is read. `--levels 11` writes the reading as it was before
the scored levels moved, which reproduces the earlier report.json byte for byte.
"""
import argparse
import importlib.util
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import v2_warning_episodes
from src.experiments.v2_probabilistic import (DECLARATION, LEADS, LEVELS, OUT, QCOLS, ROOT, RUNS, SEEDS, TREE_ARMS,
                                              WARN_LEVELS, qcol)

#: the levels every forecast-distribution score reads; the stored quantile sets hold LEVELS
SCORE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DRAWS = 600
MAX_REDRAWS = 10_000
FLOORS = {'temperature': 0.029, 'oxygen': 1.8}
COHORTS = ('Dev-2025', 'Reserved-2026')
PROB_ARMS = ('clim_quantiles', 'quantile_tree', 'quantile_tree_tide', 'chronos2', 'chronos2_net', 'chronos2_net_tide')
CONTRASTS = [(a, 'clim_quantiles') for a in PROB_ARMS[1:]] + [
    ('quantile_tree_tide', 'quantile_tree'), ('chronos2_net_tide', 'chronos2_net'),
    ('chronos2_net', 'quantile_tree'), ('chronos2_net', 'chronos2')]
TAIL = 0.05
GRID = {'alpha': WARN_LEVELS, 'c': (10, 15, 20, 25, 30), 'cooldown': (3, 6, 12)}
DEVELOPMENT = (pd.Timestamp('2024-07-01', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC'))
RESERVED = (pd.Timestamp('2026-01-01', tz='UTC'), pd.Timestamp('2026-06-20', tz='UTC'))


def _curve():
    spec = importlib.util.spec_from_file_location('exp_warning_curve', ROOT / 'scripts/experiments/exp_warning_curve.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc(series):
    return pd.to_datetime(series, utc=True).astype('datetime64[ns, UTC]')


def r(x, d=6):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


# --------------------------------------------------------------------------------------------------
# Rows: every arm on the same rows
# --------------------------------------------------------------------------------------------------
def tree_file(seed, variable, arm):
    return OUT / f'seed{seed}' / f'quantiles_{variable}_{arm}.csv.gz'


def wide(variable, seed_arms=False):
    """One row per (cohort, sensor, time, lead): truth, last, the two increments and each arm's
    quantiles, as `<arm>|<q>`; restricted to the rows where every arm has a value."""
    base = pd.read_csv(OUT / 'seed42' / f'points_{variable}.csv.gz')
    base['time'] = utc(base.time)
    parts = [pd.read_csv(tree_file(42, variable, a)) for a in TREE_ARMS]
    if seed_arms:
        for seed in SEEDS[1:]:
            for a in TREE_ARMS:
                f = pd.read_csv(tree_file(seed, variable, a))
                parts.append(f.assign(arm=f.arm + f'@{seed}'))
    chronos = pd.read_csv(OUT / 'chronos' / f'chronos_{variable}.csv.gz')
    parts.append(chronos)
    key = ['cohort', 'sensor', 'time', 'lead']
    for f in parts:
        f['time'] = utc(f.time)
        for arm, g in f.groupby('arm'):
            g = g[key + QCOLS].rename(columns={c: f'{arm}|{c}' for c in QCOLS})
            if g.duplicated(key).any():
                raise ValueError(f'{variable} {arm}: duplicated rows')
            base = base.merge(g, on=key, how='left')
    arms = sorted({c.split('|')[0] for c in base.columns if '|' in c})
    main = [a for a in arms if '@' not in a]
    before = base.groupby(['cohort', 'lead']).size()
    full = base.dropna(subset=[f'{a}|q50' for a in main]).copy()
    seeded = [f'{a}|q50' for a in arms if '@' in a]
    if seeded:                                  # the other seeds exist on Dev-2025 only, and on all of it
        dev = full.cohort == 'Dev-2025'
        if full.loc[dev, seeded].isna().any().any():
            raise ValueError(f'{variable}: a seed arm is missing on Dev-2025 rows')
    after = full.groupby(['cohort', 'lead']).size()
    full['fold'] = np.where(full.cohort == 'Dev-2025', full.fold.astype('float').astype('Int64').astype(str),
                            np.where(full.time.dt.month <= 3, 'Q1', 'Q2'))
    full['week'] = full.time.dt.isocalendar().week.astype(int).to_numpy()
    counts = {f'{c}|{l}': {'rows': int(before.get((c, l), 0)), 'kept': int(after.get((c, l), 0))}
              for c, l in before.index}
    return full, arms, counts


def quantiles(frame, arm):
    return frame[[f'{arm}|{c}' for c in QCOLS]].to_numpy(float)


def quantile_score(y, Q, levels=LEVELS):
    """Per row, the mean over `levels` of twice the pinball loss; Q holds the stored levels LEVELS in order.
    The default reads all eleven; the reports pass SCORE_LEVELS."""
    levels = tuple(levels)
    if levels != LEVELS:
        Q = Q[:, [LEVELS.index(a) for a in levels]]
    a = np.asarray(levels)[None, :]
    diff = y[:, None] - Q
    return (2 * np.maximum(a * diff, (a - 1) * diff)).mean(1)


def secondary(levels):
    """True when `levels` leaves out 0.05 and 0.95, so a report adds the eleven-level block."""
    return tuple(levels) != LEVELS


# --------------------------------------------------------------------------------------------------
# Nested macro and the joint-week bootstrap
# --------------------------------------------------------------------------------------------------
def macro(frame, col):
    return float(frame.groupby(['fold', 'sensor'])[col].mean().groupby('fold').mean().mean())


def joint_draws(frame, cols, label, draws=DRAWS):
    """Macro of each column over `draws` replicates: in each fold the ISO weeks are drawn with
    replacement, every (fold, sensor) cell's rows of a drawn week entering together; a draw that
    empties a cell is redrawn (v2_bridge.joint_week_draws, for any per-row loss)."""
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    tables = []
    for _, fold in frame.groupby('fold', sort=True):
        sums = fold.groupby(['sensor', 'week'])[cols].sum()
        n = fold.groupby(['sensor', 'week']).size()
        weeks = sorted(fold.week.unique())
        sensors = sorted(fold.sensor.unique())
        grids = [sums[c].unstack('week', fill_value=0.0).reindex(index=sensors, columns=weeks, fill_value=0.0).to_numpy()
                 for c in cols]
        count = n.unstack('week', fill_value=0).reindex(index=sensors, columns=weeks, fill_value=0).to_numpy(float)
        tables.append((grids, count))
    out = np.empty((draws, len(cols)))
    for i in range(draws):
        per_fold = []
        for grids, count in tables:
            w = count.shape[1]
            for _ in range(MAX_REDRAWS):
                m = np.bincount(rng.integers(0, w, w), minlength=w).astype(float)
                n = count @ m
                if (n > 0).all():
                    break
            else:
                raise RuntimeError('a fold could not be redrawn')
            per_fold.append([float(np.mean((g @ m) / n)) for g in grids])
        out[i] = np.mean(per_fold, axis=0)
    return out


def contrast(frame, draws, cols, x, y):
    ix, iy = cols.index(x), cols.index(y)
    point = macro(frame, x) - macro(frame, y)
    lo, hi = np.percentile(draws[:, ix] - draws[:, iy], [2.5, 97.5])
    return {'point': r(point), 'ci': [r(lo), r(hi)], 'excludes_zero': bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------------------------------
# Part B
# --------------------------------------------------------------------------------------------------
def part_b(variable, frame, arms, levels=SCORE_LEVELS):
    out = {}
    every = arms
    wide90 = 0.05 in levels and 0.95 in levels        # the 90 % interval is read only where both ends are scored
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        cell = cell.copy()
        y = cell.truth.to_numpy(float)
        stats = {}
        arms = [a for a in every if cell[f'{a}|q50'].notna().all()]
        cell['qs|last_value'] = np.abs(y - cell['last'].to_numpy(float))
        for arm in arms:
            Q = quantiles(cell, arm)
            cell[f'qs|{arm}'] = quantile_score(y, Q, levels)
            cell[f'c80|{arm}'] = ((Q[:, LEVELS.index(0.1)] <= y) & (y <= Q[:, LEVELS.index(0.9)])).astype(float)
            if wide90:
                cell[f'c90|{arm}'] = ((Q[:, LEVELS.index(0.05)] <= y) & (y <= Q[:, LEVELS.index(0.95)])).astype(float)
            cell[f'w80|{arm}'] = Q[:, LEVELS.index(0.9)] - Q[:, LEVELS.index(0.1)]
            if wide90:
                cell[f'w90|{arm}'] = Q[:, LEVELS.index(0.95)] - Q[:, LEVELS.index(0.05)]
            cell[f'ae|{arm}'] = np.abs(y - Q[:, LEVELS.index(0.5)])
        read = (('coverage80', 'c80'), ('coverage90', 'c90'), ('width80', 'w80'), ('width90', 'w90'),
                ('median_abs_error', 'ae'))
        if not wide90:
            read = tuple(x for x in read if not x[0].endswith('90'))
        for arm in ('last_value', *arms):
            stats[arm] = {'quantile_score': r(macro(cell, f'qs|{arm}'))}
            if arm != 'last_value':
                stats[arm].update({k: r(macro(cell, f'{p}|{arm}')) for k, p in read})
        cols = [f'qs|{a}' for a in ('last_value', *arms)]
        draws = joint_draws(cell, cols, f'prob|{variable}|{cohort}|{lead}')
        contrasts = {}
        for x, yy in CONTRASTS + [(a, 'clim_quantiles') for a in arms if '@' in a] + [
                (f'quantile_tree_tide@{s}', f'quantile_tree@{s}') for s in SEEDS[1:] if f'quantile_tree@{s}' in arms]:
            if x in arms and yy in arms:
                contrasts[f'{x} - {yy}'] = contrast(cell, draws, cols, f'qs|{x}', f'qs|{yy}')
        out.setdefault(cohort, {})[str(lead)] = {
            'rows': int(len(cell)), 'sensors': int(cell.sensor.nunique()), 'arms': stats, 'contrasts': contrasts}
    return out


def seed_verdicts(block):
    """Addendum 63's rule over seeds 42, 7 and 123 for the two tree contrasts, Dev-2025."""
    out = {}
    for lead, cell in block['Dev-2025'].items():
        c = cell['contrasts']
        for name, pattern in (('quantile_tree - clim_quantiles', '{tq} - clim_quantiles'),
                              ('quantile_tree_tide - clim_quantiles', '{tt} - clim_quantiles'),
                              ('quantile_tree_tide - quantile_tree', '{tt} - {tq}')):
            cs = []
            for s in SEEDS:
                suffix = '' if s == 42 else f'@{s}'
                k = pattern.format(tq=f'quantile_tree{suffix}', tt=f'quantile_tree_tide{suffix}')
                if k in c:
                    cs.append(c[k])
            if len(cs) != len(SEEDS):
                continue
            signs = {float(np.sign(x['point'])) for x in cs}
            excluding = sum(x['excludes_zero'] for x in cs)
            out[f'{lead}|{name}'] = {
                'points': [x['point'] for x in cs], 'excluding_zero': excluding,
                'verdict': ('not estimable' if len(signs) > 1 else 'result' if excluding == len(cs) else
                            'direction' if excluding else 'not distinguishable') + ' (seeds only)'}
    return out


# --------------------------------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------------------------------
def published(variable, arm, lead):
    for run in ('20260920T083322Z_v2_horizon', '20260920T133747Z_v2_horizon'):
        f = RUNS / run / 'results' / f'v2_horizon_{variable}_{arm}_h{lead}' / 'selected_predictions.csv.gz'
        if f.exists():
            g = pd.read_csv(f)
            if 'seed' in g:
                g = g[g.seed == 42]
            g['time'] = utc(g.time)
            return g[['sensor', 'time', 'prediction']].rename(columns={'prediction': arm})
    return None


def released():
    f = pd.read_csv(RUNS / 'v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz')
    f = f[f.has_anchor & f.prediction.notna()].copy()
    f['time'] = utc(f.time)
    return f[['variable', 'sensor', 'time', 'prediction']]


def share_draws(cell, f, mask, label):
    """Sum of forecast change over sum of realised change on the masked rows, with joint-week draws."""
    sub = cell[mask].assign(fc=f[mask] - cell['last'].to_numpy(float)[mask],
                            rc=cell.truth.to_numpy(float)[mask] - cell['last'].to_numpy(float)[mask])
    point = sub.fc.sum() / sub.rc.sum()
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    tables = []
    for _, fold in sub.groupby('fold', sort=True):
        g = fold.groupby('week')[['fc', 'rc']].sum()
        tables.append((g.fc.to_numpy(), g.rc.to_numpy()))
    draws = []
    for _ in range(DRAWS):
        num = den = 0.0
        for fc, rc in tables:
            m = np.bincount(rng.integers(0, len(fc), len(fc)), minlength=len(fc))
            num += fc @ m
            den += rc @ m
        draws.append(num / den if den else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return r(point), [r(lo), r(hi)]


def part_a(variable, frame, arms):
    rel = released()
    rel = rel[rel.variable == variable].drop(columns='variable').rename(columns={'prediction': 'released'})
    out = {}
    for (cohort, lead), cell in frame.groupby(['cohort', 'lead']):
        cell = cell.copy()
        singles = {'clim_hour': cell.clim_hour, 'clim_month_hour': cell.clim_month_hour}
        if cohort == 'Dev-2025':
            for arm in ('tree', 'weather_tree'):
                g = published(variable, arm, lead)
                if g is not None:
                    cell = cell.merge(g, on=['sensor', 'time'], how='left')
                    singles[arm] = cell[arm]
        if cohort == 'Reserved-2026' and lead == 6:
            cell = cell.merge(rel, on=['sensor', 'time'], how='left')
            singles['released'] = cell['released']
        for arm in arms:
            singles[f'{arm} median'] = cell[f'{arm}|q50']
        y, y0 = cell.truth.to_numpy(float), cell['last'].to_numpy(float)
        d = y - y0
        fall = np.zeros(len(cell), bool)
        rise = np.zeros(len(cell), bool)
        for _, idx in cell.groupby('sensor').indices.items():
            fall[idx] = d[idx] <= np.quantile(d[idx], TAIL)
            rise[idx] = d[idx] >= np.quantile(d[idx], 1 - TAIL)
        cell['ae|last'] = np.abs(d)
        last = macro(cell, 'ae|last')
        block = {'rows': int(len(cell)), 'tail_rows': {'fall': int(fall.sum()), 'rise': int(rise.sum())},
                 'last_value_mae': r(last), 'arms': {}}
        for name, series in singles.items():
            f = pd.Series(series).to_numpy(float)
            ok = np.isfinite(f)
            if not ok.all():
                if ok.sum() < 0.99 * len(f):
                    block['arms'][name] = {'rows': int(ok.sum()), 'note': 'not on every row; not read'}
                    continue
            cell['ae|f'] = np.abs(y - f)
            fs, fci = share_draws(cell, f, fall & ok, f'antic|{variable}|{cohort}|{lead}|{name}|fall')
            rs, rci = share_draws(cell, f, rise & ok, f'antic|{variable}|{cohort}|{lead}|{name}|rise')
            block['arms'][name] = {
                'skill_vs_last': r(1 - macro(cell[ok], 'ae|f') / macro(cell[ok], 'ae|last')),
                'fall_share': fs, 'fall_ci': fci, 'rise_share': rs, 'rise_ci': rci,
                'correlation': r(np.corrcoef(f[ok] - y0[ok], d[ok])[0, 1])}
        out.setdefault(cohort, {})[str(lead)] = block
    return out


# --------------------------------------------------------------------------------------------------
# Part C: the forecast-based decline alert
# --------------------------------------------------------------------------------------------------
def score(hourly, n_obs, alerts, curve):
    """exp_warning_curve.evaluate's matching for a given list of alert times."""
    drop6 = -(hourly - hourly.shift(curve.EVENT_WIN))
    crossings = list(drop6[drop6 > curve.EVENT_DROP].dropna().index)
    H = curve.MATCH_H

    def within(dt, sym):
        return abs(dt) <= H if sym else 0 <= dt <= H
    res = {}
    for tag, sym in (('sym', True), ('causal', False)):
        m = sum(any(within((c - a).total_seconds() / 3600.0, sym) for c in crossings) for a in alerts)
        k = sum(any(within((c - a).total_seconds() / 3600.0, sym) for a in alerts) for c in crossings)
        res[f'matched_{tag}'], res[f'caught_{tag}'] = m, k
    leads = []
    for c in crossings:
        cand = [(c - a).total_seconds() / 3600.0 for a in alerts if 0 <= (c - a).total_seconds() / 3600.0 <= H]
        if cand:
            leads.append(max(cand))
    res.update(alerts=len(alerts), crossings=len(crossings), weeks=n_obs / 168.0,
               median_lead_h=float(np.median(leads)) if leads else np.nan, crossing_times=crossings)
    return res


def cooldown_filter(times, cooldown):
    out, last = [], None
    for t in times:
        if last is None or (t - last) >= pd.Timedelta(hours=cooldown):
            out.append(t)
            last = t
    return out


def rate_alerts(hourly, rate, consec, cooldown):
    d = -(hourly - hourly.shift(1))
    cond = d > rate
    for k in range(1, consec):
        cond = cond & (d.shift(k) > rate)
    return cooldown_filter([t for t, v in cond.dropna().items() if v], cooldown)


def aggregate(per_sensor):
    a = sum(p['alerts'] for p in per_sensor.values())
    c = sum(p['crossings'] for p in per_sensor.values())
    w = sum(p['weeks'] for p in per_sensor.values())
    m = sum(p['matched_causal'] for p in per_sensor.values())
    k = sum(p['caught_causal'] for p in per_sensor.values())
    leads = [p['median_lead_h'] for p in per_sensor.values() if np.isfinite(p['median_lead_h'])]
    return {'alerts': a, 'crossings': c, 'weeks': round(w, 1),
            'precision_causal': m / a if a else np.nan, 'recall_causal': k / c if c else np.nan,
            'unmatched_per_week_causal': (a - m) / w if w else np.nan,
            'median_lead_h': float(np.median(leads)) if leads else np.nan}


def forecast_flags(warn, hourly, sensor, arm, alpha, c):
    """Issuance times at which some lead k = 1..6 puts the alpha-quantile more than c below the
    reading at t + k - 6."""
    f = warn[(warn.sensor == sensor) & (warn.arm == arm)]
    f = f[(f.time >= hourly.index.min()) & (f.time <= hourly.index.max())]
    if not len(f):
        return []
    t = pd.DatetimeIndex(f.time)
    flag = np.zeros(len(f), bool)
    for k in range(1, 7):
        ref = hourly.reindex(t + pd.Timedelta(hours=k - 6)).to_numpy(float)
        q = f[f'k{k}_{qcol(alpha)}'].to_numpy(float)
        flag |= np.isfinite(ref) & (q < ref - c)
    return list(t[flag])


def clim_tables(series_by_sensor, curve):
    """Per sensor, per hour of day, the alpha-quantiles of the six-hour change ending at that hour,
    on the development period."""
    out = {}
    for s, ser in series_by_sensor.items():
        p = curve.period(ser, *DEVELOPMENT)
        if p is None:
            continue
        hourly = p[0]
        d = (hourly - hourly.shift(curve.EVENT_WIN)).dropna()
        out[s] = {a: d.groupby(d.index.hour).quantile(a).reindex(range(24)) for a in WARN_LEVELS}
    return out


def clim_flags(hourly, table, alpha, c):
    t = hourly.dropna().index
    flag = np.zeros(len(t), bool)
    q = table[alpha]
    for k in range(1, 7):
        h = (t + pd.Timedelta(hours=k)).hour
        flag |= (q.reindex(h).to_numpy(float) < -c)
    return list(t[flag])


def part_c():
    curve = _curve()
    data = curve.load()
    warn = pd.read_csv(OUT / 'chronos' / 'chronos_warning.csv.gz')
    warn['time'] = utc(warn.time)
    periods = {}
    for name, (lo, hi) in (('development', DEVELOPMENT), ('2026', RESERVED)):
        periods[name] = {s: p for s, ser in data.items() if (p := curve.period(ser, lo, hi)) is not None}
    # G5: the rate rule against the published curve (the fixed rule and the point selected on the 90-point
    # grid) and against the extended grid's curve and 2026 row (the operating point selected on it, read from
    # runs/v2_warning_grid_ext.json), and the 2026 episodes
    gate = {}
    selected = v2_warning_episodes.operating_point()
    published = {'development': pd.read_csv(RUNS / 'v2_warning_curve/warning_curve_development.csv'),
                 '2026': pd.read_csv(RUNS / 'v2_warning_curve/warning_curve_reserved2026.csv')}
    extended = {'development': pd.read_csv(RUNS / 'v2_warning_grid_ext/warning_curve_development_ext.csv'),
                '2026': pd.read_csv(RUNS / 'v2_warning_grid_ext/reserved2026_points.csv')}
    extended['2026'] = extended['2026'][extended['2026'].point == 'selected_on_extended_grid']
    for name, sensors in periods.items():
        for rate, consec, cd in ((1.5, 2, 6), (5.0, 2, 3), selected):
            mine = aggregate({s: score(h, n, rate_alerts(h, rate, consec, cd), curve) for s, (h, n) in sensors.items()})
            ref = curve.evaluate  # the published code path, per sensor, for the self-check
            check = {s: ref(h, n, rate, consec, cd) for s, (h, n) in sensors.items()}
            on_extended = (rate, consec, cd) == selected
            pub = extended[name] if on_extended else published[name]
            tol = 5e-7 + 1e-12 if on_extended else 1e-9      # the extended curve is stored at six decimals
            row = pub[(pub.rate == rate) & (pub.consec == consec) & (pub.cooldown == cd)].iloc[0]
            ok = (mine['alerts'] == row.alerts and mine['crossings'] == row.crossings
                  and abs(mine['precision_causal'] - row.precision_causal) < tol
                  and abs(mine['recall_causal'] - row.recall_causal) < tol
                  and sum(v['alerts'] for v in check.values()) == mine['alerts'])
            gate[f'{name}|{rate}|{consec}|{cd}'] = {'alerts': mine['alerts'], 'crossings': mine['crossings'],
                                                   'precision_causal': r(mine['precision_causal']),
                                                   'recall_causal': r(mine['recall_causal']), 'passed': bool(ok)}
    episodes = {}
    for s, (h, n) in periods['2026'].items():
        drop6 = -(h - h.shift(curve.EVENT_WIN))
        cr = list(drop6[drop6 > curve.EVENT_DROP].dropna().index)
        episodes[s] = v2_warning_episodes.cluster(cr, 12) if cr else []
    n_cross = sum(len(e) for eps in episodes.values() for e in eps)
    n_eps = sum(len(eps) for eps in episodes.values())
    gate['2026_episodes'] = {'crossings': n_cross, 'episodes': n_eps, 'passed': bool(n_cross == 12 and n_eps == 7)}
    gate['passed'] = all(v['passed'] for k, v in gate.items() if isinstance(v, dict))
    out = {'gate_G5': gate}
    if not gate['passed']:
        return out
    tables = clim_tables(data, curve)

    def alerts_for(arm, sensor, hourly, alpha, c, cd):
        if arm == 'clim_alarm':
            raw = clim_flags(hourly, tables[sensor], alpha, c) if sensor in tables else []
        else:
            raw = forecast_flags(warn, hourly, sensor, arm, alpha, c)
        return cooldown_filter(sorted(raw), cd)

    def evaluate(arm, period, alpha, c, cd):
        per = {}
        alerts = {}
        for s, (h, n) in periods[period].items():
            a = alerts_for(arm, s, h, alpha, c, cd)
            alerts[s] = a
            per[s] = score(h, n, a, curve)
        return aggregate(per), alerts

    out['arms'] = {}
    for arm in ('chronos2_net', 'chronos2_net_tide', 'clim_alarm'):
        rows = []
        for alpha in GRID['alpha']:
            for c in GRID['c']:
                for cd in GRID['cooldown']:
                    agg, _ = evaluate(arm, 'development', alpha, c, cd)
                    rows.append({'alpha': alpha, 'c': c, 'cooldown': cd, **agg})
        dev = pd.DataFrame(rows)
        cand = dev[(dev.recall_causal >= 0.5) & (dev.alerts > 0)]
        block = {'development_points': int(len(dev)), 'eligible_points': int(len(cand))}
        if not len(cand):
            block['selected'] = None
            out['arms'][arm] = block
            continue
        sel = cand.sort_values('precision_causal', ascending=False, kind='stable').iloc[0]
        block['selected'] = {'alpha': float(sel.alpha), 'c': int(sel.c), 'cooldown': int(sel.cooldown),
                             'development': {k: r(sel[k]) for k in ('alerts', 'crossings', 'precision_causal',
                                                                       'recall_causal', 'unmatched_per_week_causal',
                                                                       'median_lead_h')}}
        agg, alerts = evaluate(arm, '2026', float(sel.alpha), int(sel.c), int(sel.cooldown))
        block['2026'] = {k: r(v) for k, v in agg.items()}
        block['2026_episodes'] = episode_reading(alerts, episodes)
        out['arms'][arm] = block
    # the rate rule's two points on 2026 episodes, for the same reading
    for label, (rate, consec, cd) in (('rate_rule_released', (1.5, 2, 6)), ('rate_rule_selected', selected)):
        alerts = {s: rate_alerts(h, rate, consec, cd) for s, (h, n) in periods['2026'].items()}
        per = {s: score(h, n, alerts[s], curve) for s, (h, n) in periods['2026'].items()}
        out['arms'][label] = {'operating_point': {'rate': rate, 'consecutive': consec, 'cooldown_h': cd},
                              '2026': {k: r(v) for k, v in aggregate(per).items()},
                              '2026_episodes': episode_reading(alerts, episodes)}
    out['arms']['rate_rule_selected']['operating_point']['selected_in'] = 'runs/v2_warning_grid_ext.json'
    return out


def episode_reading(alerts, episodes):
    matched = []
    total_alerts = sum(len(a) for a in alerts.values())
    for s, eps in episodes.items():
        for m in v2_warning_episodes.one_to_one(alerts.get(s, []), eps):
            matched.append({'sensor': s, **m})
    leads = [m['lead_h'] for m in matched]
    n = sum(len(e) for e in episodes.values())
    return {'episodes': n, 'matched': len(matched), 'alerts': total_alerts,
            'recall': r(len(matched) / n) if n else None,
            'precision': r(len(matched) / total_alerts) if total_alerts else None,
            'leads_h': sorted(leads), 'median_lead_h': r(float(np.median(leads)), 2) if leads else None,
            'alerted_6h_or_more_ahead': int(sum(l >= 6 for l in leads)), 'matched_detail': matched}


# --------------------------------------------------------------------------------------------------
# The declared predictions
# --------------------------------------------------------------------------------------------------
def predictions(b, a, c):
    out = {}

    def cells(leads):
        for v in b:
            for cohort in COHORTS:
                for lead in leads:
                    cell = b[v].get(cohort, {}).get(str(lead))
                    if cell:
                        yield v, cohort, lead, cell
    p1 = [(v, co, l, x) for v, co, l, cell in cells((6, 12))
          for x in ('chronos2_net - clim_quantiles', 'quantile_tree - clim_quantiles')
          if not (cell['contrasts'][x]['ci'][1] < 0)]
    out['P1'] = {'held': not p1, 'failures': [list(f) for f in p1]}
    p2 = [(v, co, l, arm, cell['contrasts'][f'{arm} - clim_quantiles']['point'])
          for v, co, l, cell in cells((24, 48)) for arm in PROB_ARMS[1:]
          if cell['contrasts'][f'{arm} - clim_quantiles']['point'] < -FLOORS[v]]
    out['P2'] = {'held': not p2, 'failures': [list(f) for f in p2]}
    p3 = []
    for v, co, l, cell in cells(LEADS):
        for arm in ('chronos2', 'chronos2_net'):
            if not cell['arms'][arm]['coverage80'] < 0.80:
                p3.append((v, co, l, arm, cell['arms'][arm]['coverage80']))
        if co == 'Dev-2025' and not 0.75 <= cell['arms']['quantile_tree']['coverage80'] <= 0.85:
            p3.append((v, co, l, 'quantile_tree', cell['arms']['quantile_tree']['coverage80']))
    out['P3'] = {'held': not p3, 'failures': [list(f) for f in p3]}
    p4 = []
    for v in a:
        for cohort in COHORTS:
            for lead in LEADS:
                cell = a[v].get(cohort, {}).get(str(lead))
                if not cell:
                    continue
                arms = {k: x for k, x in cell['arms'].items() if 'fall_share' in x}
                if lead in (24, 48):
                    p4 += [(v, cohort, lead, k, x['fall_share']) for k, x in arms.items() if not x['fall_share'] < 0.5]
                if lead in (6, 12):
                    if not arms['chronos2_net median']['fall_share'] > arms['clim_hour']['fall_share']:
                        p4.append((v, cohort, lead, 'chronos2_net median vs clim_hour',
                                   arms['chronos2_net median']['fall_share'], arms['clim_hour']['fall_share']))
    out['P4'] = {'held': not p4, 'failures': [list(f) for f in p4]}
    sel = c.get('arms', {}).get('chronos2_net', {})
    ep = sel.get('2026_episodes')
    if ep and ep['matched']:
        longer = ep['median_lead_h'] > 2.0
        short = ep['alerted_6h_or_more_ahead'] < ep['matched'] / 2
        out['P5'] = {'held': bool(longer and short), 'median_lead_h': ep['median_lead_h'],
                     'alerted_6h_or_more_ahead': ep['alerted_6h_or_more_ahead'], 'matched': ep['matched']}
    else:
        out['P5'] = {'held': None, 'note': 'no matched 2026 episode or no selected point'}
    p6 = []
    for v, co, l, cell in cells(LEADS):
        for x in ('quantile_tree_tide - quantile_tree', 'chronos2_net_tide - chronos2_net'):
            if not abs(cell['contrasts'][x]['point']) < FLOORS[v]:
                p6.append((v, co, l, x, cell['contrasts'][x]['point']))
    for v in b:
        for k, x in b[v].get('seeds', {}).items():
            if k.endswith('quantile_tree_tide - quantile_tree') and x['verdict'].startswith('result'):
                p6.append((v, 'Dev-2025', k, x['verdict']))
    out['P6'] = {'held': not p6, 'failures': [list(f) for f in p6]}
    return out


def report(levels=SCORE_LEVELS, target=None):
    """The reading at `levels`; at SCORE_LEVELS it adds the eleven-level part B as a secondary block, and the
    declared predictions are read on that block, the score they were declared on (the nine-level reading of
    them is beside it)."""
    levels = tuple(levels)
    target = OUT / 'report.json' if target is None else Path(target)
    gates = {}
    for variable in ('temperature', 'oxygen'):
        tree = json.loads((OUT / 'seed42' / f'tree_{variable}_quantile_tree.json').read_text())
        gates[variable] = tree['gates']
    chronos = json.loads((OUT / 'chronos' / 'chronos.json').read_text())
    gates['chronos'] = chronos['gates']
    ok_rows = all(v['passed'] for var in ('temperature', 'oxygen') for k, v in gates[var].items()
                  if k.startswith(('G1', 'G2')))
    ok_tide = all(gates['temperature']['G3_tide'][p]['passed'] for p in ('development', '2026'))
    rep = {'declared_in': DECLARATION, 'levels': levels, 'draws': DRAWS, 'floors': FLOORS,
           'aggregation': 'nested macro: folds (Dev-2025) or quarters (2026), then sensors',
           'gates': gates, 'gates_passed': {'rows_G1_G2': ok_rows, 'tide_G3': ok_tide,
                                            'chronos_levels_G6': chronos['gates']['G6_levels']['passed']}}
    if secondary(levels):
        rep['stored_levels'] = LEVELS
        rep['eleven_levels'] = ('part_b_eleven_levels: the same reading over the eleven stored levels 0.05-0.95, '
                                'with the central 90 % interval; the declared predictions are read on it')
    b, b11, a = {}, {}, {}
    if ok_rows and rep['gates_passed']['chronos_levels_G6']:
        for variable in ('temperature', 'oxygen'):
            frame, arms, counts = wide(variable, seed_arms=True)
            main = [x for x in arms if '@' not in x]
            for block, lv in ((b, levels), (b11, LEVELS if secondary(levels) else None)):
                if lv is None:
                    continue
                block[variable] = part_b(variable, frame, arms, lv)
                block[variable]['seeds'] = seed_verdicts(block[variable])
                block[variable]['row_counts'] = counts
            a[variable] = part_a(variable, frame, main)
            print(variable, 'parts A and B done', flush=True)
    rep['part_b'], rep['part_a'] = b, a
    rep['part_c'] = part_c()
    if b:
        rep['predictions'] = predictions(b11 or b, a, rep['part_c'])
    if b11:
        rep['predictions_nine_levels'] = predictions(b, a, rep['part_c'])
        rep['part_b_eleven_levels'] = b11
    target.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    print('wrote', target)
    print(json.dumps(rep.get('predictions', {}), default=str)[:3000])


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--levels', type=int, choices=(9, 11), default=9,
                   help='11: the reading over the eleven stored levels only, as before the scored levels moved')
    p.add_argument('--out', default=None, help='where to write (default: the report.json of the run folder)')
    a = p.parse_args()
    return (SCORE_LEVELS if a.levels == 9 else LEVELS), a.out


if __name__ == '__main__':
    report(*cli())
