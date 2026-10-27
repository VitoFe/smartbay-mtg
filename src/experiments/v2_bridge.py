"""Addendum 39: bridging a silent probe with a continuation anchored to its last real reading.

While a probe is silent an operator can hold its last value (the delivered fallback), add the
change its nearest reporting neighbour has seen since onset, add the diurnal climatological
increment, or roll the six-hour tree forward on its own output (addendum 4). This fits the one
learned rule that can use all of those at once: a direct model of the change since onset,
y(onset + h) - y(onset), from what a live system has at the target hour. It is direct, so errors do
not compound, and anchored to the last real reading, which is what repaired the retrospective
regressions of addendum 34.

Cohorts (declared in config/bay_reserved_protocol_v2_addendum39.json):
  Bridge-Dev   the addendum-4 closed-loop episodes on the two Dev-2025 folds, joined by
               (sensor, onset, lead) so every contrast with the published rollout is paired
  Bridge-2026  the same episode rule on 1 January to 19 June 2026, read once

Nothing the silent probe records after onset enters a feature. Nothing is selected or calibrated on
2026. Every fit uses a nested Optuna study of at least 100 completed trials.

Usage:
  python -m src.experiments.v2_bridge --variable temperature [--quick]
  python -m src.experiments.v2_bridge --report
"""
import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as o2sol
from src.experiments import bay_reserved_evaluation as v1
from src.experiments import v2_horizon, v2_horizon_reserved
from src.experiments.v2_closed_loop import episodes, lead_climatology, climatological_value, STEP
from src.experiments.v2_forecast import tree_space, make_tree
from src.experiments.v2_gap_recovery import nearest_others
from src.tuning.protocol_v2 import run_study

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_bridge'
CLOSED_LOOP = RUNS / 'v2_closed_loop'
# The rollout each contrast is set against. The oxygen rollout is the one whose neighbour column is
# the rolled state at a step with no other probe available (v2_closed_loop --rolled-neighbour); the
# temperature rollout has no such step and is the published one.
ROLLOUT = {'temperature': CLOSED_LOOP / 'closed_loop_temperature.csv.gz',
           'oxygen': CLOSED_LOOP / 'closed_loop_oxygen_rolled_neighbour.csv.gz'}
# Addendum 69: the same rollouts with every arm at its outer fold's own selection
# (v2_closed_loop --perfold). `--report --perfold-rollout` sets the continuation against them and
# writes report_a69.json beside report.json, which it leaves untouched.
ROLLOUT_A69 = {'temperature': CLOSED_LOOP / 'perfold/closed_loop_temperature.csv.gz',
               'oxygen': CLOSED_LOOP / 'perfold/closed_loop_oxygen_rolled_neighbour.csv.gz'}
PER_SENSOR = (('Bridge-2026', 'oxygen'),)       # cohorts whose contrasts are also reported sonde by sonde
LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
MAX_LEAD = LEADS[-1]
GAP_H = 72
MAX_ROWS = 300_000
DRAWS = 600
SALINITY_FALLBACK = 38.0
RESERVED_START = pd.Timestamp('2026-01-01', tz='UTC')
FLOORS = {'temperature': {'quantisation': 0.029, 'colocated_low': 0.20}, 'oxygen': {'reproducibility': 1.8}}
WEATHER = ('air', 'solar', 'wind')


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def grid(variable, cohort):
    """The six-hour issuance grid of addendum 4 (dev) or its 2026-inclusive twin (reserved)."""
    if cohort == 'dev':
        return v2_horizon.build(variable, STEP)
    return v2_horizon_reserved.build(variable, STEP)


def hourly(names, variable, index, cohort):
    """Hourly observations on the same hourly index the grid was cut from, never forward-filled."""
    _, frames = (v2_horizon.load_observations() if cohort == 'dev' else v1.load_all_observations())
    frame = pd.DataFrame({n: frames[n][variable].reindex(index) if variable in frames[n] else np.nan
                          for n in names}, index=index)
    salinity = pd.DataFrame({n: frames[n]['salinity'].reindex(index) if 'salinity' in frames[n] else np.nan
                             for n in names}, index=index)
    return frame, salinity, frames


def hourly_index(cohort):
    end = '2025-12-31T23:00:00' if cohort == 'dev' else v2_horizon_reserved.RESERVED_END
    return pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), pd.Timestamp(end, tz='UTC') if isinstance(end, str) else end, freq='h')


def geometry():
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    return meta


def fold_2026(data):
    """Outer 'fold' for the reserved read: train on grid issuances whose last lead precedes 2026 by 72 h."""
    t = data['times']
    tr = np.flatnonzero(t + pd.Timedelta(hours=STEP + GAP_H) < RESERVED_START)
    va = np.flatnonzero((t >= RESERVED_START) & (t + pd.Timedelta(hours=MAX_LEAD) <= v2_horizon_reserved.RESERVED_END))
    eligible = (data['valid'][tr].sum(0) >= 100) & (data['valid'][va].sum(0) >= 30)
    fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
    return dict(number=2, tr=tr, va=va, eligible=eligible, fallback=fallback)


# --------------------------------------------------------------------------------------------------
# Features: what a live system has at the target hour, with the probe silent since onset
# --------------------------------------------------------------------------------------------------
class Context:
    """Everything the feature builder reads, prepared once per (variable, cohort)."""

    def __init__(self, variable, cohort):
        self.variable, self.cohort = variable, cohort
        self.data = grid(variable, cohort)
        self.names = list(self.data['names'])
        self.index = hourly_index(cohort)
        obs, sal, frames = hourly(self.names, variable, self.index, cohort)
        self.center, self.scale = self.data['center'], self.data['scale']
        self.obs = ((obs - self.center) / self.scale).to_numpy()   # the grid's scaled units
        self.sal = sal.to_numpy()                                   # PSU, for solubility
        meta = geometry()
        xy = meta.loc[self.names, ['x_m', 'y_m']].to_numpy()
        self.distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        self.order = [nearest_others(self.distance, i) for i in range(len(self.names))]
        self.depth = meta.loc[self.names, 'nominal_depth_m'].to_numpy()
        self.pos = self.index.get_indexer(self.data['times'])
        assert (self.pos >= 0).all(), 'grid times outside the hourly index'
        weather = v1.load_weather_2026(self.index)
        self.weather = weather[list(WEATHER)].to_numpy()
        self.temperature = None
        if variable == 'oxygen':
            tnames = [n for n in meta.index if n in frames and 'temperature' in frames[n]
                      and frames[n]['temperature'].notna().any()]
            self.tnames = tnames
            self.temperature = pd.DataFrame({n: frames[n]['temperature'].reindex(self.index) for n in tnames},
                                            index=self.index).to_numpy()
            txy = meta.loc[tnames, ['x_m', 'y_m']].to_numpy()
            d = np.linalg.norm(xy[:, None] - txy[None, :], axis=-1)
            # the silent sonde's own temperature channel belongs to the silent instrument: never a neighbour
            self.torder = [[j for j in np.argsort(d[i], kind='stable') if tnames[j] != self.names[i]]
                           for i in range(len(self.names))]
            self.tdistance = d

    def rows(self, onsets, sensors_mask, require_all_leads=False):
        """One row per (grid onset, sensor, lead) with a real reading at onset and at the target."""
        out = []
        n = len(self.names)
        for i0 in onsets:
            h0 = self.pos[i0]
            if h0 < 6:
                continue
            y0 = self.obs[h0]
            prev = self.obs[h0 - 6]
            w0 = self.weather[h0]
            for s in np.flatnonzero(sensors_mask):
                if not np.isfinite(y0[s]):
                    continue
                if require_all_leads and not all(h0 + h < len(self.index) and np.isfinite(self.obs[h0 + h, s]) for h in LEADS):
                    continue
                for h in LEADS:
                    hT = h0 + h
                    if hT >= len(self.index) or not np.isfinite(self.obs[hT, s]):
                        continue
                    out.append((i0, h0, s, h, hT))
        return out

    def features(self, rows, clim):
        """Feature matrix for rows (i0, h0, s, h, hT). Only obs[h0] and obs[h0-6] of probe s are read."""
        cols = {k: [] for k in FEATURES[self.variable]}
        target, anchor = [], []
        for i0, h0, s, h, hT in rows:
            # The silent probe is read at onset and six hours before it, never after (gate G2);
            # obs[hT, s] below is the label, not a feature.
            assert hT > h0
            y0 = self.obs[h0, s]
            d_all = self.obs[hT] - self.obs[h0]          # change since onset, every probe
            reporting = [j for j in self.order[s] if np.isfinite(d_all[j])]
            near = reporting[0] if reporting else None
            idw = reporting[:4]
            w = 1 / (self.distance[s, idw] + 50.0) if idw else np.array([])
            t_target = self.index[hT]
            cols['lead'].append(h)
            cols['hour_sin'].append(np.sin(2 * np.pi * t_target.hour / 24))
            cols['hour_cos'].append(np.cos(2 * np.pi * t_target.hour / 24))
            cols['doy_sin'].append(np.sin(2 * np.pi * t_target.dayofyear / 365.25))
            cols['doy_cos'].append(np.cos(2 * np.pi * t_target.dayofyear / 365.25))
            prev = self.obs[h0 - 6, s]
            cols['own_trend'].append(y0 - prev if np.isfinite(prev) else np.nan)
            cols['near_change'].append(d_all[near] if near is not None else np.nan)
            cols['near_distance'].append(np.log1p(self.distance[s, near]) if near is not None else np.nan)
            cols['idw_change'].append(float(np.sum(w * d_all[idw]) / np.sum(w)) if idw else np.nan)
            cols['median_change'].append(float(np.median(d_all[reporting])) if reporting else np.nan)
            cols['n_reporting'].append(len(reporting))
            cm, ch = clim(i0, s, h)
            cols['clim_month_hour'].append(cm)
            cols['clim_hour'].append(ch)
            for k, name in enumerate(WEATHER):
                cols[f'{name}_target'].append(self.weather[hT, k])
                cols[f'{name}_change'].append(self.weather[hT, k] - self.weather[h0, k])
            cols['depth'].append(self.depth[s])
            if self.variable == 'oxygen':
                tr_ = [j for j in self.torder[s] if np.isfinite(self.temperature[h0, j]) and np.isfinite(self.temperature[hT, j])]
                if tr_:
                    j = tr_[0]
                    t0, t1 = self.temperature[h0, j], self.temperature[hT, j]
                    sal = self.sal[h0, s] if np.isfinite(self.sal[h0, s]) else SALINITY_FALLBACK
                    cols['near_temperature_change'].append(t1 - t0)
                    cols['solubility_change'].append(float(o2sol(t1, sal) - o2sol(t0, sal)))
                else:
                    cols['near_temperature_change'].append(np.nan)
                    cols['solubility_change'].append(np.nan)
            target.append(self.obs[hT, s] - y0)
            anchor.append(y0)
        return pd.DataFrame(cols), np.asarray(target), np.asarray(anchor)


BASE = ['lead', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'own_trend', 'near_change', 'near_distance',
        'idw_change', 'median_change', 'n_reporting', 'clim_month_hour', 'clim_hour',
        *[f'{w}_{k}' for w in WEATHER for k in ('target', 'change')], 'depth']
FEATURES = {'temperature': BASE, 'oxygen': BASE + ['near_temperature_change', 'solubility_change']}


# --------------------------------------------------------------------------------------------------
# Climatologies (fitted on training issuances only)
# --------------------------------------------------------------------------------------------------
def hour_climatology(data, fold, steps):
    """Mean increment from the anchor to each lead by sensor and onset hour only (Chapter 4's table)."""
    t = data['times']; n = data['x'].shape[1]; tr = fold['tr']
    tr = tr[tr + steps < len(t)]
    tables = []
    for k in range(steps):
        delta = data['truth'][tr + k] - data['x'][tr]
        ok = data['valid'][tr + k] & np.isfinite(data['x'][tr])
        j, s = np.nonzero(ok)
        frame = pd.DataFrame({'sensor': s, 'hour': t.hour.to_numpy()[tr][j], 'd': delta[ok]})
        per_sensor = frame.groupby('sensor').d.mean().reindex(np.arange(n)).to_numpy()
        overall = float(frame.d.mean())
        fallback = np.where(np.isfinite(per_sensor), per_sensor, overall)
        tables.append((frame.groupby(['sensor', 'hour']).d.mean(), fallback))
    return tables


def hour_value(tables, data, i0, s, k):
    cell, fallback = tables[k]
    delta = cell.get((s, int(data['times'][i0].hour)), np.nan)
    return data['x'][i0, s] + (delta if np.isfinite(delta) else fallback[s])


def climatology_lookup(data, fold):
    """(i0, s, h) -> the two climatological increments at that lead, in scaled units."""
    steps = MAX_LEAD // STEP
    month_hour = lead_climatology(data, fold, steps)
    hour_only = hour_climatology(data, fold, steps)

    def lookup(i0, s, h):
        k = h // STEP - 1
        x0 = data['x'][i0, s]
        return (climatological_value(month_hour, data, i0, s, k) - x0, hour_value(hour_only, data, i0, s, k) - x0)
    return lookup


# --------------------------------------------------------------------------------------------------
# Fitting and scoring
# --------------------------------------------------------------------------------------------------
def subsample(rows, seed):
    if len(rows) <= MAX_ROWS:
        return rows
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(rows), MAX_ROWS, replace=False))
    return [rows[i] for i in keep]


def fit(ctx, rows, clim, params):
    X, y, _ = ctx.features(rows, clim)
    sensors = np.array([r[2] for r in rows])
    counts = np.bincount(sensors, minlength=len(ctx.names)).astype(float)
    w = 1 / np.maximum(counts[sensors], 1); w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})   # tree_space tunes with subsample_freq = 1
    model.fit(X, y, sample_weight=w)
    return model


def lead_macro_ratio(frame):
    """Mean over leads of (per-sensor macro MAE of the model) / (same for persistence)."""
    out = []
    for _, g in frame.groupby('lead'):
        m = g.assign(e=(g.truth - g.pred).abs(), p=(g.truth - g.anchor).abs()).groupby('sensor')[['e', 'p']].mean()
        out.append(m.e.mean() / max(m.p.mean(), 1e-9))
    return float(np.mean(out))


def training_rows(ctx, onsets, eligible, first_scored):
    """Grid onsets in the window whose target precedes the first scored onset by the 72 h gap."""
    cutoff = ctx.index.get_indexer([first_scored - pd.Timedelta(hours=GAP_H)])[0]
    return [r for r in ctx.rows(onsets, eligible) if r[4] < cutoff]


def episode_rows(ctx, eps):
    return [(i0, ctx.pos[i0], s, h, ctx.pos[i0] + h) for i0, s in eps for h in LEADS]


def select_and_fit(ctx, fold, label, quick):
    """Nested study on the inner window, then refit on the whole outer training window."""
    data = ctx.data
    inner = v2_horizon.inner_fold(data, fold, MAX_LEAD)
    lookup_inner = climatology_lookup(data, inner)
    va_eps = episodes(data, inner, MAX_LEAD)
    first_inner = data['times'][min(i for i, _ in va_eps)]
    tr_rows = subsample(training_rows(ctx, inner['tr'], inner['eligible'], first_inner),
                        zlib.crc32(f'{label}|inner'.encode()))
    va_rows = episode_rows(ctx, va_eps)
    if quick:
        tr_rows, va_rows = tr_rows[:4000], va_rows[:2000]
    Xv, yv, av = ctx.features(va_rows, lookup_inner)
    Xt, yt, _ = ctx.features(tr_rows, lookup_inner)
    sensors_t = np.array([r[2] for r in tr_rows])
    counts = np.bincount(sensors_t, minlength=len(ctx.names)).astype(float)
    wt = 1 / np.maximum(counts[sensors_t], 1); wt = wt / wt.mean()
    frame_v = pd.DataFrame({'sensor': [r[2] for r in va_rows], 'lead': [r[3] for r in va_rows],
                            'truth': av + yv, 'anchor': av})

    def objective(trial):
        params = tree_space(trial)
        model = make_tree(params)
        model.fit(Xt, yt, sample_weight=wt)
        return lead_macro_ratio(frame_v.assign(pred=frame_v.anchor + model.predict(Xv)))

    folder = OUT / ('smoke/studies' if quick else 'studies')      # smoke trials never seed a real study
    summary = run_study(f'bridge_{ctx.variable}_{label}', objective, folder, quick=quick)
    lookup_outer = climatology_lookup(data, fold)
    first_outer = data['times'][fold['va'].min()]
    rows = subsample(training_rows(ctx, fold['tr'], fold['eligible'], first_outer),
                     zlib.crc32(f'{label}|outer'.encode()))
    if quick:
        rows = rows[:4000]
    model = fit(ctx, rows, lookup_outer, summary['params'])
    return model, lookup_outer, summary, {'inner_training_rows': len(tr_rows), 'inner_validation_rows': len(va_rows),
                                          'outer_training_rows': len(rows)}


def neighbour_change(ctx, i0, s, h):
    """Last reading plus the change since onset of the nearest other probe reporting at both hours."""
    h0 = ctx.pos[i0]; hT = h0 + h
    for j in ctx.order[s]:
        if np.isfinite(ctx.obs[h0, j]) and np.isfinite(ctx.obs[hT, j]):
            return ctx.obs[h0, s] + ctx.obs[hT, j] - ctx.obs[h0, j]
    return np.nan


def score_rows(ctx, fold, eps, model, lookup):
    """Every arm on the episodes, in physical units."""
    rows = episode_rows(ctx, eps)
    X, y, anchor = ctx.features(rows, lookup)
    pred = anchor + model.predict(X)
    sc, ce = ctx.scale, ctx.center
    data = ctx.data
    out = []
    for n, (i0, h0, s, h, hT) in enumerate(rows):
        cm, ch = lookup(i0, s, h)
        x0 = data['x'][i0, s]
        truth_grid = data['truth'][i0 + h // STEP - 1, s]
        assert abs(truth_grid - ctx.obs[hT, s]) < 1e-9 and abs(x0 - ctx.obs[h0, s]) < 1e-9, 'grid and hourly disagree'
        out.append({'fold': fold['number'], 'sensor': ctx.names[s], 'onset': ctx.index[h0], 'lead': h,
                    'truth': ctx.obs[hT, s] * sc + ce, 'persistence': x0 * sc + ce,
                    'clim_month_hour': (x0 + cm) * sc + ce, 'clim_hour': (x0 + ch) * sc + ce,
                    'neighbour_change': neighbour_change(ctx, i0, s, h) * sc + ce,
                    'anchored_direct': pred[n] * sc + ce})
    return pd.DataFrame(out)


def run(variable, quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    frames, studies = [], {}
    # Bridge-Dev: the addendum-4 episodes, fold by fold
    ctx = Context(variable, 'dev')
    published = pd.read_csv(CLOSED_LOOP / f'closed_loop_{variable}.csv.gz', usecols=['sensor', 'time'],
                            parse_dates=['time']).drop_duplicates()
    published = set(zip(published.sensor, published.time))
    for fold in v2_horizon.folds(ctx.data, STEP):
        # exactly the published episodes (the oxygen rollout skipped sondes with no temperature twin)
        eps = [(i0, s) for i0, s in episodes(ctx.data, fold, MAX_LEAD)
               if (ctx.names[s], ctx.data['times'][i0]) in published]
        if quick:
            eps = eps[:40]
        model, lookup, summary, counts = select_and_fit(ctx, fold, f'fold{fold["number"]}', quick)
        studies[f'fold{fold["number"]}'] = {**summary, **counts}
        frames.append(score_rows(ctx, fold, eps, model, lookup).assign(cohort='Bridge-Dev'))
    # Bridge-2026: one fold, selected on the inner window before 2026
    rctx = Context(variable, 'reserved')
    rfold = fold_2026(rctx.data)
    reps = episodes(rctx.data, rfold, MAX_LEAD)
    if quick:
        reps = reps[:40]
    model, lookup, summary, counts = select_and_fit(rctx, rfold, 'reserved2026', quick)
    studies['reserved2026'] = {**summary, **counts}
    reserved = score_rows(rctx, rfold, reps, model, lookup).assign(cohort='Bridge-2026')
    if quick:
        # the pipeline is exercised end to end, but no 2026 outcome is written before the real read
        print(f'smoke: Bridge-2026 pipeline produced {len(reserved)} rows over {len(reps)} episodes (not written)')
    else:
        frames.append(reserved)
    frame = pd.concat(frames, ignore_index=True)
    target = OUT / ('smoke' if quick else '')
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / f'bridge_{variable}.csv.gz', index=False)
    (target / f'bridge_{variable}_studies.json').write_text(json.dumps(studies, indent=2, default=str))
    print(f'{variable}: {len(frame)} rows written to {target}')
    if quick:
        print('smoke only: no contrast computed')


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


MIN_EPISODES = 5        # a (fold, sensor) cell with fewer episodes is reported with its contribution
MAX_REDRAWS = 10_000


def joint_week_draws(frame, x, y, rng, draws=DRAWS):
    """Macro x - y over `draws` replicates of onset weeks drawn jointly within each fold.

    In each fold the onset weeks are drawn with replacement, as many as the fold has, and every
    (fold, sensor) cell's episodes of a drawn week enter together, so probes that share a week share
    its draw. A fold draw that leaves a cell with no episode is redrawn, so the cells are fixed.
    The frame must already hold only rows where both arms exist. Returns the draws and the number
    of fold draws that had to be redrawn.
    """
    f = frame.assign(ex=(frame.truth - frame[x]).abs(), ey=(frame.truth - frame[y]).abs(), n=1.0)
    tables = []
    for _, fold in f.groupby('fold', sort=True):
        t = fold.groupby(['sensor', 'week'])[['ex', 'ey', 'n']].sum()
        weeks = sorted(t.index.get_level_values('week').unique())
        sensors = sorted(t.index.get_level_values('sensor').unique())
        grid = {c: t[c].unstack('week', fill_value=0.0).reindex(index=sensors, columns=weeks, fill_value=0.0)
                .to_numpy() for c in ('ex', 'ey', 'n')}
        tables.append(grid)
    out = np.empty(draws)
    redrawn = 0
    for i in range(draws):
        fx, fy = [], []
        for grid in tables:
            w = grid['n'].shape[1]
            for _ in range(MAX_REDRAWS):
                m = np.bincount(rng.integers(0, w, w), minlength=w).astype(float)
                n = grid['n'] @ m
                if (n > 0).all():
                    break
                redrawn += 1
            else:
                raise RuntimeError('no joint week draw keeps every cell')
            fx.append(np.mean(grid['ex'] @ m / n))
            fy.append(np.mean(grid['ey'] @ m / n))
        out[i] = np.mean(fx) - np.mean(fy)
    return out, redrawn


def week_blocks(frame, x, y, label):
    """Paired contrast x - y by lead, with two resamplings of onset weeks from one seed.

    `ci` (reported): onset weeks drawn jointly within each fold, every cell's episodes of a drawn
    week entering together, cells fixed (`joint_week_draws`). Probes in one calendar week share its
    weather and often their onsets, so drawing each cell's weeks on its own understates the variance
    of the macro. `ci_within_cell`: onset weeks resampled within each (fold, sensor) cell on their
    own, the earlier scheme, kept beside it draw for draw.

    Only rows where both arms have a value enter the point estimate and the draws. `macro` drops
    missing values column by column, so without this an arm missing on some rows (the neighbour
    rule, where no other probe reports at both hours) was scored on other episodes than the arm it
    was set against. `paired_episodes` counts the rows kept, one per episode.
    """
    frame = frame.dropna(subset=[x, y])
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    cells = [[g for _, g in cell.groupby('week')] for _, cell in frame.groupby(['fold', 'sensor'])]
    point = macro(frame, x) - macro(frame, y)
    draws = []
    for _ in range(DRAWS):
        parts = []
        for blocks in cells:
            parts.extend(blocks[i] for i in rng.integers(0, len(blocks), len(blocks)))
        s = pd.concat(parts)
        draws.append(macro(s, x) - macro(s, y))
    lo_w, hi_w = np.percentile(draws, [2.5, 97.5])
    joint, redrawn = joint_week_draws(frame, x, y, np.random.default_rng([20260923, zlib.crc32(label.encode())]))
    lo, hi = np.percentile(joint, [2.5, 97.5])
    signs = {f'{f}|{sn}': round(float((c.truth - c[x]).abs().mean() - (c.truth - c[y]).abs().mean()), 6)
             for (f, sn), c in frame.groupby(['fold', 'sensor'])}
    return {'point': round(point, 6), 'ci': [round(float(lo), 6), round(float(hi), 6)],
            'excludes_zero': bool(lo > 0 or hi < 0),
            'ci_within_cell': [round(float(lo_w), 6), round(float(hi_w), 6)],
            'excludes_zero_within_cell': bool(lo_w > 0 or hi_w < 0),
            'joint_redrawn': int(redrawn), 'paired_episodes': int(len(frame)), 'per_cell': signs}


def sparse_cells(frame, x, y, label):
    """The macro x - y without every (fold, sensor) cell of fewer than MIN_EPISODES paired episodes,
    its joint-week interval, and each such cell's own contrast and its term in the published macro
    (the cell's contrast over the number of folds and the number of cells in its fold)."""
    frame = frame.dropna(subset=[x, y])
    size = frame.groupby(['fold', 'sensor']).size()
    sparse = size[size < MIN_EPISODES]
    if sparse.empty:
        return {'sparse_cells': {}}
    cells_in_fold = size.groupby(level='fold').size()
    n_folds = len(cells_in_fold)
    terms = {}
    for (fold, sensor), n in sparse.items():
        c = frame[(frame.fold == fold) & (frame.sensor == sensor)]
        d = float((c.truth - c[x]).abs().mean() - (c.truth - c[y]).abs().mean())
        terms[f'{fold}|{sensor}'] = {'episodes': int(n), 'weeks': int(c.week.nunique()), 'cell_contrast': round(d, 6),
                                     'term_in_macro': round(d / n_folds / int(cells_in_fold[fold]), 6)}
    keep = frame.set_index(['fold', 'sensor']).index.isin(size[size >= MIN_EPISODES].index)
    rest = frame[keep]
    point = macro(rest, x) - macro(rest, y)
    draws, redrawn = joint_week_draws(rest, x, y, np.random.default_rng([20260923, zlib.crc32(f'{label}|dense'.encode())]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'sparse_cells': terms,
            'without_sparse_cells': {'point': round(point, 6), 'ci': [round(float(lo), 6), round(float(hi), 6)],
                                     'excludes_zero': bool(lo > 0 or hi < 0), 'paired_episodes': int(len(rest)),
                                     'cells': int(rest.groupby(['fold', 'sensor']).ngroups), 'joint_redrawn': int(redrawn)}}


def per_sensor(frame, x, y, label):
    """x - y on each sensor's episodes alone, with onset weeks of that sensor resampled; no interval
    on fewer than five onset weeks."""
    frame = frame.dropna(subset=[x, y])
    out = {}
    for (fold, sensor), c in frame.groupby(['fold', 'sensor']):
        point = float((c.truth - c[x]).abs().mean() - (c.truth - c[y]).abs().mean())
        entry = {'episodes': int(len(c)), 'weeks': int(c.week.nunique()), 'point': round(point, 6),
                 'mae_x': round(float((c.truth - c[x]).abs().mean()), 6),
                 'mae_y': round(float((c.truth - c[y]).abs().mean()), 6)}
        if c.week.nunique() < MIN_EPISODES:
            entry.update(ci=None, excludes_zero=None,
                         not_estimable=f'{c.week.nunique()} onset weeks, fewer than {MIN_EPISODES}')
        else:
            draws, _ = joint_week_draws(c, x, y, np.random.default_rng([20260923, zlib.crc32(f'{label}|{sensor}'.encode())]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            entry.update(ci=[round(float(lo), 6), round(float(hi), 6)], excludes_zero=bool(lo > 0 or hi < 0))
        out[f'{fold}|{sensor}'] = entry
    return out


MIN_BAND = 30          # residuals a band needs before it is issued


def ledger_coverage(frame, leads):
    """Lead-calibrated 90 % band from an expanding ledger of earlier residuals (review R2).

    For an episode with onset t at lead h the half-width is the 90th percentile of |truth - anchored|
    over every episode, Bridge-Dev and then Bridge-2026 with the sensors pooled, whose target hour
    (its onset + h) is at or before t: only residuals a running system would hold at issuance. With
    fewer than MIN_BAND of them the band abstains. Bridge-Dev's first weeks therefore abstain, and the
    2026 ledger starts from the development residuals and grows with 2026's own. (The first
    version covered each development fold with the band of the other fold, so the summer fold borrowed
    the later autumn errors, and froze the 2026 band from both folds.)
    """
    f = frame.assign(r=(frame.truth - frame.anchored_direct).abs(),
                     onset=pd.to_datetime(frame.onset, utc=True))
    out = {'Bridge-Dev': {}, 'Bridge-2026': {}, 'rule': 'expanding causal ledger, sensors pooled, '
                                                       f'at least {MIN_BAND} residuals, abstain otherwise'}
    for lead in leads:
        d = f[f.lead == lead].reset_index(drop=True)
        t = d.onset.to_numpy('datetime64[ns]')
        known = t + np.timedelta64(int(lead), 'h')
        order = np.argsort(known, kind='stable')
        known_sorted, r_sorted = known[order], d.r.to_numpy(float)[order]
        q = np.full(len(d), np.nan)
        for ti in np.unique(t):
            k = int(np.searchsorted(known_sorted, ti, side='right'))
            if k >= MIN_BAND:
                q[t == ti] = np.percentile(r_sorted[:k], 90)
        d['q'] = q
        for cohort in ('Bridge-Dev', 'Bridge-2026'):
            c = d[d.cohort == cohort]
            ok = c.q.notna()
            entry = {'coverage': round(float((c.r[ok] <= c.q[ok]).mean()), 6), 'scored': int(ok.sum()),
                     'abstained': int((~ok).sum()), 'half_width': round(float(c.q[ok].mean()), 6)}
            if cohort == 'Bridge-Dev':
                entry['coverage_by_fold'] = {int(fo): round(float((g.r[g.q.notna()] <= g.q[g.q.notna()]).mean()), 6)
                                             for fo, g in c.groupby('fold')}
            out[cohort][lead] = entry
    return out


def coverage(frame):
    """The continuation's band at 6 to 48 h (ledger_coverage)."""
    return ledger_coverage(frame, LEADS)

def verdicts(rep, interval='ci'):
    """The registered predictions, read with the reported joint-week intervals (`ci`) or, for
    comparison, the within-cell ones (`ci_within_cell`)."""
    def c(var, cohort, lead, x, y):
        entry = rep[var][cohort]['by_lead'][str(lead)]['contrasts'].get(f'{x} - {y}')
        return None if entry is None else {**entry, 'ci': entry[interval]}
    v = {}
    t = 'temperature'; o = 'oxygen'
    v['P1'] = all(c(t, co, L, 'anchored_direct', 'persistence')['ci'][1] < 0 for co in ('Bridge-Dev', 'Bridge-2026') for L in LEADS)
    v['P2'] = all(c(t, 'Bridge-Dev', L, 'anchored_direct', 'recursive_tree')['ci'][1] < 0 for L in LEADS if L >= 24)
    v['P3'] = all(sum(c(t, co, L, 'anchored_direct', 'neighbour_change')['ci'][1] < 0 for L in LEADS) >= 4
                  for co in ('Bridge-Dev', 'Bridge-2026'))
    v['P4'] = all(c(o, co, L, 'anchored_direct', 'persistence')['point'] < -FLOORS[o]['reproducibility']
                  for co in ('Bridge-Dev', 'Bridge-2026') for L in (6, 12, 18))
    v['P5'] = all(c(o, co, L, 'anchored_direct', 'clim_hour')['point'] > -FLOORS[o]['reproducibility']
                  for co in ('Bridge-Dev', 'Bridge-2026') for L in LEADS)
    v['P6'] = all(0.85 <= rep[var]['intervals']['Bridge-2026'][L]['coverage'] <= 0.95 for var in (t, o) for L in LEADS)
    v['P7'] = sum(c(o, 'Bridge-Dev', L, 'anchored_direct', 'recursive_tree')['ci'][1] < 0 for L in LEADS if L >= 18) >= 2
    return v


def report(perfold_rollout=False):
    """`perfold_rollout` (addendum 69): the rollout arm is v2_closed_loop --perfold's, and the report is
    written to report_a69.json; nothing else changes."""
    rollouts = ROLLOUT_A69 if perfold_rollout else ROLLOUT
    target = OUT / ('report_a69.json' if perfold_rollout else 'report.json')
    rep = {'declared_in': 'config/bay_reserved_protocol_v2_addendum39.json', 'draws': DRAWS,
           'aggregation': 'frozen (fold, sensor) macro', 'floors': FLOORS}
    if perfold_rollout:
        rep['rollout_selection'] = 'per outer fold (addendum 69)'
    for variable in ('temperature', 'oxygen'):
        frame = pd.read_csv(OUT / f'bridge_{variable}.csv.gz', parse_dates=['onset'])
        iso = frame.onset.dt.isocalendar()
        frame['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
        published = pd.read_csv(rollouts[variable], parse_dates=['time'])
        published = published.rename(columns={'time': 'onset'})[['fold', 'sensor', 'onset', 'lead', 'truth',
                                                                 'recursive_tree', 'persistence', 'climatology']]
        dev = frame[frame.cohort == 'Bridge-Dev'].merge(published, on=['fold', 'sensor', 'onset', 'lead'],
                                                        how='outer', suffixes=('', '_published'), indicator=True)
        g1 = {'rows_bridge': int((frame.cohort == 'Bridge-Dev').sum()), 'rows_published': int(len(published)),
              'unmatched': int((dev._merge != 'both').sum()),
              'truth_max_abs_diff': float((dev.truth - dev.truth_published).abs().max()),
              'persistence_max_abs_diff': float((dev.persistence - dev.persistence_published).abs().max()),
              'clim_month_hour_max_abs_diff': float((dev.clim_month_hour - dev.climatology).abs().max())}
        g1['passed'] = bool(g1['unmatched'] == 0 and max(g1['truth_max_abs_diff'], g1['persistence_max_abs_diff'],
                                                          g1['clim_month_hour_max_abs_diff']) < 1e-6)
        block = {'G1': g1, 'rollout': str(rollouts[variable].relative_to(RUNS))}
        if not g1['passed']:
            rep[variable] = block
            print(variable, 'G1 FAILED', g1)
            continue
        dev = dev.drop(columns=['_merge', 'truth_published', 'persistence_published', 'climatology'])
        cohorts = {'Bridge-Dev': dev, 'Bridge-2026': frame[frame.cohort == 'Bridge-2026']}
        arms = ['persistence', 'clim_month_hour', 'clim_hour', 'neighbour_change', 'anchored_direct', 'recursive_tree']
        for name, f in cohorts.items():
            present = [a for a in arms if a in f and f[a].notna().any()]
            entry = {'episodes': int(f.groupby(['fold', 'sensor', 'onset']).ngroups),
                     'cells': int(f.groupby(['fold', 'sensor']).ngroups), 'sensors': sorted(f.sensor.unique()),
                     'weeks': int(f.week.nunique()), 'by_lead': {}}
            for lead, sub in f.groupby('lead'):
                e = {'n': int(len(sub)), 'mae': {a: round(macro(sub, a), 6) for a in present}, 'contrasts': {}}
                pairs = [('anchored_direct', other, f'{variable}|{name}|{lead}|ad|{other}')
                         for other in present if other != 'anchored_direct']
                pairs += [(x, y, f'{variable}|{name}|{lead}|{x}|{y}')
                          for x, y in (('neighbour_change', 'persistence'), ('clim_hour', 'persistence'),
                                       ('clim_hour', 'clim_month_hour'))]
                for x, y, label in pairs:
                    c = week_blocks(sub, x, y, label)
                    c.update(sparse_cells(sub, x, y, label))
                    if (name, variable) in PER_SENSOR:
                        c['per_sensor'] = per_sensor(sub, x, y, label)
                    e['contrasts'][f'{x} - {y}'] = c
                entry['by_lead'][str(int(lead))] = e
            block[name] = entry
        block['intervals'] = coverage(frame)
        rep[variable] = block
    rep['intervals_reported'] = ('joint: onset weeks drawn with replacement jointly within each fold, the '
                                 'episodes of every cell in a drawn week together, cells fixed; ci_within_cell: '
                                 'onset weeks resampled within each (fold, sensor) cell, the same seed')
    rep['sparse_cell_rule'] = f'cells of fewer than {MIN_EPISODES} paired episodes, reported with their term in the macro'
    if all(rep.get(v, {}).get('G1', {}).get('passed') for v in ('temperature', 'oxygen')):
        rep['predictions'] = verdicts(rep)
        rep['predictions_within_cell'] = verdicts(rep, 'ci_within_cell')
    target.write_bytes(json.dumps(rep, indent=2, default=str).encode('utf-8'))
    print(json.dumps(rep.get('predictions', {}), indent=1))
    print('wrote', target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'])
    p.add_argument('--quick', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--perfold-rollout', action='store_true',
                   help='with --report: the rollout selected per outer fold (addendum 69), into report_a69.json')
    a = p.parse_args()
    if a.report and a.perfold_rollout:
        report(perfold_rollout=True)
    elif a.report:
        report()
    else:
        run(a.variable, quick=a.quick)
