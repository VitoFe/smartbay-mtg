"""Addendum 64: forecast uncertainty and anticipated change.

A forecast at a fixed lead that its inputs cannot inform stays near the last value, and on a chart it
looks like the record shifted by the lead. This module measures how much of the realised change each
single-valued forecast anticipates (part A), forecasts the distribution instead (part B), asks whether
such a forecast anticipates sustained oxygen declines sooner than the rate rule (part C), and gives
the forecasters the one physical input known ahead without a weather service, the astronomical tide
(part D: the *_tide arms).

Declared in config/bay_reserved_protocol_v2_addendum64.json before anything was computed. The 2026
rows are read once, with every configuration selected before 2026 and seed 42; seeds 7 and 123
re-select the two quantile trees on Dev-2025 only.

Producers (Trinity) and the report (local, from what they wrote):
  python -m src.experiments.v2_probabilistic --part tree --variable temperature --arm quantile_tree [--seed 7] [--quick]
  python -m src.experiments.v2_probabilistic --part chronos [--quick]
  python -m src.experiments.v2_probabilistic --report
"""
import argparse
import functools
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

from src.data import bay_ispra
from src.experiments import bay_reserved_evaluation, v2_horizon, v2_horizon_reserved
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import THREADS, tree_space
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

ROOT = v2_horizon_reserved.ROOT
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_probabilistic'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum64.json'
VARIABLES = ('temperature', 'oxygen')
LEADS = list(v2_horizon.HORIZONS)                     # 1, 3, 6, 12, 24, 36, 48
LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
EDGE = (0.01, 0.99)                                   # Chronos-2's outputs also keep these
WARN_LEVELS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
SEEDS = (42, 7, 123)
MIN_CELL = 30
DEV_WINDOW = (pd.Timestamp('2025-07-01', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC'))
RESERVED_WINDOW = (v2_horizon_reserved.RESERVED_START, v2_horizon_reserved.RESERVED_END)
TIDE_FIT = (pd.Timestamp('2021-05-24', tz='UTC'), pd.Timestamp('2024-07-24', tz='UTC'))
#: Standard angular speeds, degrees per hour.
CONSTITUENTS = {'M2': 28.9841042, 'S2': 30.0, 'N2': 28.4397295, 'K2': 30.0821373,
                'K1': 15.0410686, 'O1': 13.9430356, 'P1': 14.9589314, 'Q1': 13.3986609}
TREE_ARMS = ('quantile_tree', 'quantile_tree_tide')
CHRONOS_ARMS = ('chronos2', 'chronos2_net', 'chronos2_net_tide')
WARNING_ARMS = ('chronos2_net', 'chronos2_net_tide')
CHRONOS2 = 'amazon/chronos-2'
CTX, MIN_FINITE, MAX_LEAD = 512, 48, 48               # v2_chronos2 / exp_chronos.predict_at
WARNING_SENSORS = ('TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)',
                   'TPortovenereP_02m-CTD(calibrato)', 'TMitiliO_05m-CTD(calibrato)',
                   'TPortovenereS_02m-CTD(calibrato)')   # exp_warning_curve.SENSORS
WARNING_SPAN = (pd.Timestamp('2024-07-01', tz='UTC'), pd.Timestamp('2026-06-20', tz='UTC'))
DIGITS = {'temperature': 4, 'oxygen': 3}


def qcol(a):
    return f'q{int(round(a * 100)):02d}'


QCOLS = [qcol(a) for a in LEVELS]


# --------------------------------------------------------------------------------------------------
# Loading once per process: the builders reload every export on each call
# --------------------------------------------------------------------------------------------------
v2_horizon.load_observations = functools.lru_cache(maxsize=None)(v2_horizon.load_observations)
bay_reserved_evaluation.load_all_observations = functools.lru_cache(maxsize=None)(
    bay_reserved_evaluation.load_all_observations)


# --------------------------------------------------------------------------------------------------
# The astronomical tide
# --------------------------------------------------------------------------------------------------
def _tide_design(times):
    hours = np.asarray((pd.DatetimeIndex(times) - pd.Timestamp('2000-01-01', tz='UTC')) / pd.Timedelta(hours=1), float)
    cols = [np.ones_like(hours)]
    for speed in CONSTITUENTS.values():
        w = np.deg2rad(speed) * hours
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


@functools.lru_cache(maxsize=None)
def _tide_fit():
    level = bay_ispra._hourly(cutoff=None)['level']
    fit = level[(level.index >= TIDE_FIT[0]) & (level.index < TIDE_FIT[1])].dropna()
    coef, *_ = np.linalg.lstsq(_tide_design(fit.index), fit.to_numpy(float), rcond=None)
    return coef, len(fit)


def tide(times):
    """The predicted level (m) at each time: a function of the clock alone."""
    return _tide_design(times) @ _tide_fit()[0]


def tide_features(times, lead):
    t0 = pd.DatetimeIndex(times)
    a, b = tide(t0), tide(t0 + pd.Timedelta(hours=lead))
    window = np.stack([tide(t0 + pd.Timedelta(hours=k)) for k in range(25)], axis=1)
    return np.column_stack([b, b - a, window.max(1) - window.min(1)]).astype('float32')


def tide_gate():
    """G3: the predicted six-hour change against the observed one, out of the fitting window."""
    level = bay_ispra._hourly(cutoff=None)['level']
    coef, rows = _tide_fit()
    out = {'fit_rows': rows, 'fit_window': [str(TIDE_FIT[0]), str(TIDE_FIT[1])],
           'amplitudes_m': {k: round(float(np.hypot(coef[1 + 2 * i], coef[2 + 2 * i])), 4)
                            for i, k in enumerate(CONSTITUENTS)}}
    for name, lo, hi in (('development', '2024-07-24', '2026-01-01'), ('2026', '2026-01-01', '2026-06-20')):
        s = level[(level.index >= pd.Timestamp(lo, tz='UTC')) & (level.index < pd.Timestamp(hi, tz='UTC'))]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
        p = pd.Series(tide(s.index), index=s.index)
        d, dp = s.diff(6), p.diff(6)
        ok = d.notna() & dp.notna()
        r = float(np.corrcoef(d[ok], dp[ok])[0, 1])
        out[name] = {'r_six_hour_change': round(r, 4), 'hours': int(ok.sum()), 'passed': bool(r >= 0.9)}
    return out


# --------------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------------
def quantile_score(y, Q, levels=LEVELS):
    """Per row, the mean over the levels of twice the pinball loss (equals |y - q| for one value)."""
    a = np.asarray(levels)[None, :]
    diff = np.asarray(y, float)[:, None] - np.asarray(Q, float)
    return (2 * np.maximum(a * diff, (a - 1) * diff)).mean(1)


def sensor_macro(values, sensors):
    return float(pd.Series(values).groupby(np.asarray(sensors)).mean().mean())


# --------------------------------------------------------------------------------------------------
# Part B, fitted arms: the climatological quantiles and the quantile tree
# --------------------------------------------------------------------------------------------------
def clim_table(data, tr, levels):
    """[sensor, issuance hour, level] quantiles of the lead-h increment (scaled units) on rows tr."""
    tm = data['valid'][tr]
    j, s = np.nonzero(tm)
    hours = data['times'][tr].hour.to_numpy()[j]
    d = (data['truth'][tr] - data['x'][tr])[tm]
    overall = np.quantile(d, levels)
    table = np.empty((len(data['names']), 24, len(levels)))
    for sensor in range(len(data['names'])):
        ds, hs = d[s == sensor], hours[s == sensor]
        own = np.quantile(ds, levels) if len(ds) >= MIN_CELL else overall
        for h in range(24):
            cell = ds[hs == h]
            table[sensor, h] = np.quantile(cell, levels) if len(cell) >= MIN_CELL else own
    return table


def clim_predict(data, table, va, mask):
    j, s = np.nonzero(mask)
    i = va[j]
    return table[s, data['times'][i].hour.to_numpy()] + data['x'][i, s][:, None]


def tree_design(data, fallback, lead, with_tide):
    base, _ = features(data, {'fallback': fallback}, 'forecast')
    if not with_tide:
        return base
    extra = tide_features(data['times'], lead)
    return np.concatenate([base, np.broadcast_to(extra[:, None, :], (*base.shape[:2], extra.shape[1]))], axis=-1)


def fit_predict(data, design, tr, va, eligible, params):
    """One LightGBM per level, shared configuration; sorted per row; scaled units."""
    import lightgbm as lgb
    mask = data['valid'] & eligible[None, :]
    tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]
    w = w / w.mean()
    X, y = design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm]
    vm = mask[va]
    Xv = design[va][vm]
    P = np.column_stack([lgb.LGBMRegressor(**params, objective='quantile', alpha=a, verbosity=-1, n_jobs=THREADS,
                                           random_state=42, deterministic=True, force_col_wise=True)
                         .fit(X, y, sample_weight=w).predict(Xv) for a in LEVELS])
    P.sort(axis=1)
    return P + data['x'][va][vm][:, None]


def scored(data, va, eligible):
    """The rows a fold scores, in v2_horizon.rows_from's order: (mask, row index, sensor index)."""
    mask = data['valid'][va] & eligible[None, :]
    j, s = np.nonzero(mask)
    return mask, va[j], s


def frame_of(data, va, eligible, cohort, fold, lead):
    mask, i, s = scored(data, va, eligible)
    c, k = data['center'], data['scale']
    return pd.DataFrame({'cohort': cohort, 'fold': fold, 'sensor': np.asarray(data['names'])[s],
                         'time': data['times'][i], 'lead': lead,
                         'truth': data['truth'][i, s] * k + c, 'last': data['x'][i, s] * k + c}), mask


def with_quantiles(frame, arm, Q, data):
    out = frame.copy()
    out['arm'] = arm
    for n, col in enumerate(QCOLS):
        out[col] = Q[:, n] * data['scale'] + data['center']
    return out


def hour_only(data, tr):
    """The hour-of-day increment (v2_horizon.climatology's second table) on rows tr, as a grid."""
    increment = data['truth'] - data['x']
    tm = data['valid'][tr]
    j, s = np.nonzero(tm)
    n = data['x'].shape[1]
    frame = pd.DataFrame({'sensor': s, 'hour': data['times'][tr].hour.to_numpy()[j], 'd': increment[tr][tm]})
    per_sensor = frame.groupby('sensor').d.mean().reindex(np.arange(n)).to_numpy()
    fallback = np.where(np.isfinite(per_sensor), per_sensor, float(frame.d.mean()))
    table = frame.groupby(['sensor', 'hour']).d.mean()
    grid = np.broadcast_to(np.arange(n), (len(data['times']), n)).ravel()
    hour = np.repeat(data['times'].hour.to_numpy(), n)
    d = table.reindex(pd.MultiIndex.from_arrays([grid, hour])).to_numpy().reshape(len(data['times']), n)
    return data['x'] + np.where(np.isfinite(d), d, fallback[None, :])


def published_tree_rows(variable, lead):
    for run in ('20260920T083322Z_v2_horizon', '20260920T133747Z_v2_horizon'):
        f = RUNS / run / 'results' / f'v2_horizon_{variable}_tree_h{lead}' / 'selected_predictions.csv.gz'
        if f.exists():
            g = pd.read_csv(f)
            g['time'] = pd.to_datetime(g.time, utc=True)
            return g
    raise FileNotFoundError(f'published tree rows {variable} h{lead}')


def compare_rows(mine, published, keys, columns):
    mine, published = mine.copy(), published.copy()
    for frame in (mine, published):
        frame['time'] = pd.to_datetime(frame['time'], utc=True).astype('datetime64[ns, UTC]')
    j = mine.merge(published, on=keys, how='outer', suffixes=('', '_pub'), indicator=True)
    out = {'rows_module': int(len(mine)), 'rows_published': int(len(published)),
           'only_module': int((j._merge == 'left_only').sum()), 'only_published': int((j._merge == 'right_only').sum())}
    both = j[j._merge == 'both']
    for a, b in columns:
        out[f'max_abs_{a}'] = float((both[a] - both[b]).abs().max()) if len(both) else None
    out['passed'] = bool(out['only_module'] == 0 and out['only_published'] == 0 and len(both) > 0
                         and all(out[f'max_abs_{a}'] <= 1e-9 for a, _ in columns))
    return out


def run_tree(variable, arm, seed=42, quick=False):
    """One quantile tree at every lead on Dev-2025 under one sampler seed, and at seed 42 its 2026
    reading. The seed-42 job of the plain tree also writes the climatological quantiles, the two
    increments and gates G1 to G3."""
    if arm not in TREE_ARMS:
        raise ValueError(arm)
    baselines = seed == 42 and arm == 'quantile_tree'
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    leads = [6, 24] if quick else LEADS
    quantiles, points, studies, gates = [], [], {}, {}
    if baselines:
        gates['G3_tide'] = tide_gate()
    for lead in leads:
        t0 = time.time()
        data = v2_horizon.build(variable, lead)
        outer = v2_horizon.folds(data, lead)
        if quick:
            for f in outer:
                f['tr'], f['va'] = f['tr'][-300:], f['va'][:40]
        inner = [v2_horizon.inner_fold(data, f, lead) for f in outer]
        if quick:
            for f in inner:
                f['tr'], f['va'] = f['tr'][-300:], f['va'][:40]
        clim = {id(f): clim_table(data, f['tr'], LEVELS) for f in outer + inner}
        if seed == 42:
            data26 = v2_horizon_reserved.build(variable, lead)
            bundle = v2_horizon_reserved.split(data26, lead)
            if quick:
                bundle['tr'], bundle['va'] = bundle['tr'][-300:], bundle['va'][:40]
        designs = {id(f): tree_design(data, f['fallback'], lead, arm.endswith('_tide')) for f in outer + inner}

        def objective(trial):
            p = tree_space(trial)
            ratios = []
            for f in inner:
                mask, i, s = scored(data, f['va'], f['eligible'])
                y = data['truth'][i, s]
                P = fit_predict(data, designs[id(f)], f['tr'], f['va'], f['eligible'], p)
                C = clim_predict(data, clim[id(f)], f['va'], mask)
                ratios.append(sensor_macro(quantile_score(y, P), s) / sensor_macro(quantile_score(y, C), s))
            return float(np.mean(ratios))
        sel = run_study(f'{variable}_{arm}_h{lead}', objective, folder / 'studies', quick)
        params = {**sel['params'], 'subsample_freq': 1}
        studies[str(lead)] = {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}
        for f in outer:
            frame, mask = frame_of(data, f['va'], f['eligible'], 'Dev-2025', f['number'], lead)
            P = fit_predict(data, designs[id(f)], f['tr'], f['va'], f['eligible'], params)
            quantiles.append(with_quantiles(frame, arm, P, data))
        if seed == 42:
            design26 = tree_design(data26, bundle['fallback'], lead, arm.endswith('_tide'))
            frame, mask = frame_of(data26, bundle['va'], bundle['eligible'], 'Reserved-2026', None, lead)
            P = fit_predict(data26, design26, bundle['tr'], bundle['va'], bundle['eligible'], params)
            quantiles.append(with_quantiles(frame, arm, P, data26))
        if baselines:
            # the climatological quantiles (score levels and the two edges) and the two increments
            for f in outer:
                frame, mask = frame_of(data, f['va'], f['eligible'], 'Dev-2025', f['number'], lead)
                C = clim_predict(data, clim[id(f)], f['va'], mask)
                quantiles.append(with_quantiles(frame, 'clim_quantiles', C, data))
                month_hour, hours = v2_horizon.climatology(data, f)
                _, i, s = scored(data, f['va'], f['eligible'])
                points.append(frame.assign(clim_month_hour=month_hour[i, s] * data['scale'] + data['center'],
                                           clim_hour=hours[i, s] * data['scale'] + data['center']))
            table26 = clim_table(data26, bundle['tr'], LEVELS)
            frame, mask = frame_of(data26, bundle['va'], bundle['eligible'], 'Reserved-2026', None, lead)
            quantiles.append(with_quantiles(frame, 'clim_quantiles',
                                            clim_predict(data26, table26, bundle['va'], mask), data26))
            _, i, s = scored(data26, bundle['va'], bundle['eligible'])
            month_hour = v2_horizon_reserved.climatology(data26, bundle)
            hours = hour_only(data26, bundle['tr'])
            points.append(frame.assign(clim_month_hour=month_hour[i, s] * data26['scale'] + data26['center'],
                                       clim_hour=hours[i, s] * data26['scale'] + data26['center']))
            if not quick:
                dev = pd.concat([p for p in points if p.cohort.iloc[0] == 'Dev-2025' and p.lead.iloc[0] == lead])
                pub = published_tree_rows(variable, lead)
                gates[f'G1_dev|{lead}'] = compare_rows(
                    dev[['fold', 'sensor', 'time', 'truth', 'last', 'clim_hour']],
                    pub[['fold', 'sensor', 'time', 'truth', 'persistence', 'clim_hour']]
                    .rename(columns={'truth': 'truth_pub', 'clim_hour': 'clim_hour_pub'}),
                    ['fold', 'sensor', 'time'],
                    [('truth', 'truth_pub'), ('last', 'persistence'), ('clim_hour', 'clim_hour_pub')])
                if lead == 24:
                    res = pd.read_csv(RUNS / 'v2_horizon_reserved_h24' / f'reserved_h24_{variable}.csv.gz')
                    res = res[res.arm == 'tree']
                    res['time'] = pd.to_datetime(res.time, utc=True)
                    mine = points[-1][['sensor', 'time', 'truth', 'last']]
                    gates['G2_2026|24'] = compare_rows(
                        mine, res[['sensor', 'time', 'truth', 'persistence']].rename(columns={'truth': 'truth_pub'}),
                        ['sensor', 'time'], [('truth', 'truth_pub'), ('last', 'persistence')])
        print(variable, arm, 'seed', seed, 'lead', lead, f'{time.time() - t0:.0f} s', flush=True)
    q = pd.concat(quantiles, ignore_index=True)
    num = ['truth', 'last'] + QCOLS
    q[num] = q[num].round(DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_{arm}.csv.gz', index=False)
    if points:
        p = pd.concat(points, ignore_index=True)
        p[['truth', 'last', 'clim_month_hour', 'clim_hour']] = p[['truth', 'last', 'clim_month_hour', 'clim_hour']].round(9)
        p.to_csv(folder / f'points_{variable}.csv.gz', index=False)
    save(folder / f'tree_{variable}_{arm}.json', {'declared_in': DECLARATION, 'variable': variable, 'arm': arm, 'seed': seed,
                                            'smoke_only': quick, 'leads': leads, 'studies': studies,
                                            'gates': gates, 'threads': THREADS})
    print(variable, 'seed', seed, 'gates', json.dumps({k: v.get('passed', v) for k, v in gates.items()
                                                      if not k.startswith('G3')}), flush=True)


# --------------------------------------------------------------------------------------------------
# Part B and C, zero-shot arms: Chronos-2 on the grids and hourly on the warning series
# --------------------------------------------------------------------------------------------------
def _filled(x):
    """v2_chronos2.covs, as the companion applies it: forward then backward fill, then the window's
    mean (or zero)."""
    s = pd.Series(x).ffill().bfill().to_numpy(float)
    fill = np.nanmean(x) if np.isfinite(x).any() else 0.0
    return np.nan_to_num(s, nan=fill).astype(np.float32)


class Panel:
    """Hourly series of one variable, the network's temperature mean and spread, each probe's own
    temperature, and the tide, on one hourly index that runs 48 h past the last issuance."""

    def __init__(self, series, temperatures, index):
        self.index = index
        self.series = {n: s.reindex(index).to_numpy(float) for n, s in series.items()}
        t = pd.DataFrame({n: s.reindex(index) for n, s in temperatures.items()})
        self.net = {'net_T': t.mean(axis=1).to_numpy(float), 'net_sd': t.std(axis=1).to_numpy(float)}
        self.own_t = {n: t[n].to_numpy(float) for n in t.columns}
        self.tide = tide(index).astype(np.float32)

    def items(self, name, positions, variable):
        """The three Chronos-2 inputs at each issuance position (context ends at the position)."""
        series = self.series[name]
        out = {arm: [] for arm in CHRONOS_ARMS}
        kept = []
        for p in positions:
            a = max(0, p - CTX + 1)
            w = series[a:p + 1]
            if np.isfinite(w).sum() < MIN_FINITE or not np.isfinite(series[p]) or p + MAX_LEAD >= len(self.index):
                continue
            assert a + len(w) - 1 == p                                      # G4: context ends at issuance
            target = w.astype(np.float32)
            past = {k: _filled(v[a:p + 1]) for k, v in self.net.items()}
            if variable == 'oxygen' and name in self.own_t:
                past['own_T'] = _filled(self.own_t[name][a:p + 1])
            out['chronos2'].append({'target': target})
            out['chronos2_net'].append({'target': target, 'past_covariates': past})
            out['chronos2_net_tide'].append({'target': target, 'past_covariates': {**past, 'tide': self.tide[a:p + 1]},
                                             'future_covariates': {'tide': self.tide[p + 1:p + 1 + MAX_LEAD]}})
            kept.append(p)
        return out, kept


def predict(pipe, items, keep):
    """Chronos-2 quantiles [item, level, lead], batched by covariate names (the pipeline requires one
    set per call); keep = indices of the levels to return."""
    import torch
    result = [None] * len(items)
    groups = {}
    for n, it in enumerate(items):
        groups.setdefault((tuple(sorted(it.get('past_covariates', {}))),
                           tuple(sorted(it.get('future_covariates', {})))), []).append(n)
    for members in groups.values():
        for k in range(0, len(members), 128):
            chunk = members[k:k + 128]
            got = pipe.predict([items[n] for n in chunk], prediction_length=MAX_LEAD)
            for n, o in zip(chunk, got):
                a = o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                result[n] = a.reshape(-1, len(pipe.quantiles), a.shape[-1])[0][keep]
    return np.stack(result).astype(np.float32)


def grid_panel(variable):
    meta, frames = bay_reserved_evaluation.load_all_observations()
    names = [n for n in meta.index if n in frames and variable in frames[n] and frames[n][variable].notna().any()]
    index = pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), RESERVED_WINDOW[1] + pd.Timedelta(hours=MAX_LEAD + 1), freq='h')
    temps = {n: f['temperature'] for n, f in frames.items() if 'temperature' in f and f['temperature'].notna().any()}
    return Panel({n: frames[n][variable] for n in names}, temps, index), names


def warning_panel():
    obs = pd.read_csv(ROOT / 'reports/bay_twin/hourly_observations.csv.gz', index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    index = pd.date_range(obs.index.min(), WARNING_SPAN[1] + pd.Timedelta(hours=MAX_LEAD + 1), freq='h')
    series = {}
    for s in WARNING_SENSORS:
        c = f'{s}::oxygen'
        if c in obs.columns:
            v = obs[c].dropna()
            series[s] = v[v.between(0, 600)]                               # exp_warning_curve.load
    temps = {c.split('::')[0]: obs[c] for c in obs.columns if c.endswith('::temperature')}
    return Panel(series, temps, index)


def run_chronos(quick=False):
    import torch
    from chronos import BaseChronosPipeline
    base = '/home/vito/smartbay_do/.hf'
    if os.path.isdir(base):
        os.environ.setdefault('HF_HOME', base)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    torch.set_num_threads(THREADS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS2, device_map=device, torch_dtype=torch.float32)
    levels = [float(a) for a in pipe.quantiles]
    want = (EDGE[0],) + LEVELS + (EDGE[1],)
    g6 = {'levels': levels, 'passed': all(any(abs(a - b) < 1e-9 for b in levels) for a in want + WARN_LEVELS)}
    if not g6['passed']:
        raise RuntimeError(f'G6 failed: {levels}')
    keep = [min(range(len(levels)), key=lambda n: abs(levels[n] - a)) for a in want]
    warn_keep = [want.index(a) for a in WARN_LEVELS]
    folder = OUT / ('smoke' if quick else '') / 'chronos'
    folder.mkdir(parents=True, exist_ok=True)
    cols = [qcol(a) for a in want]
    counts = {}
    for variable in VARIABLES:
        t0 = time.time()
        panel, names = grid_panel(variable)
        idx = panel.index
        grid = np.flatnonzero(np.isin(idx.hour, (0, 6, 12, 18))
                              & (((idx >= DEV_WINDOW[0]) & (idx < DEV_WINDOW[1]))
                                 | ((idx >= RESERVED_WINDOW[0]) & (idx <= RESERVED_WINDOW[1]))))
        rows = []
        for name in names:
            positions = grid[::max(1, len(grid) // 12)][:12] if quick else grid
            items, kept = panel.items(name, positions, variable)
            if not kept:
                continue
            kept = np.asarray(kept)
            target_seen = np.stack([np.isfinite(panel.series[name][kept + h]) for h in LEADS], axis=1)
            for arm in CHRONOS_ARMS:
                Q = predict(pipe, items[arm], keep)                      # [issuance, level, lead 1..48]
                for li, h in enumerate(LEADS):
                    ok = target_seen[:, li]
                    if not ok.any():
                        continue
                    t = idx[kept[ok]]
                    block = pd.DataFrame(Q[ok][:, :, h - 1], columns=cols)
                    block.insert(0, 'arm', arm)
                    block.insert(0, 'lead', h)
                    block.insert(0, 'time', t)
                    block.insert(0, 'sensor', name)
                    block.insert(0, 'cohort', np.where(t < RESERVED_WINDOW[0], 'Dev-2025', 'Reserved-2026'))
                    rows.append(block)
        frame = pd.concat(rows, ignore_index=True)
        frame[cols] = frame[cols].round(DIGITS[variable])
        frame.to_csv(folder / f'chronos_{variable}.csv.gz', index=False)
        counts[variable] = int(len(frame))
        print('chronos grid', variable, len(frame), f'{time.time() - t0:.0f} s', flush=True)
    # hourly, on the warning series: levels WARN_LEVELS at leads 1 to 6
    t0 = time.time()
    panel = warning_panel()
    idx = panel.index
    span = np.flatnonzero((idx >= WARNING_SPAN[0]) & (idx < WARNING_SPAN[1]))
    rows = []
    for name in panel.series:
        positions = span[::max(1, len(span) // 24)][:24] if quick else span
        items, kept = panel.items(name, positions, 'oxygen')
        if not kept:
            continue
        frame = pd.DataFrame({'sensor': name, 'time': idx[np.asarray(kept)]})
        for arm in WARNING_ARMS:
            Q = predict(pipe, items[arm], keep)[:, warn_keep, :6]
            block = frame.assign(arm=arm)
            for li, a in enumerate(WARN_LEVELS):
                for k in range(6):
                    block[f'k{k + 1}_{qcol(a)}'] = np.round(Q[:, li, k], 3)
            rows.append(block)
    warn = pd.concat(rows, ignore_index=True)
    warn.to_csv(folder / 'chronos_warning.csv.gz', index=False)
    counts['warning'] = int(len(warn))
    print('chronos warning', len(warn), f'{time.time() - t0:.0f} s', flush=True)
    save(folder / 'chronos.json', {'declared_in': DECLARATION, 'model': CHRONOS2, 'device': device,
                                   'smoke_only': quick, 'gates': {'G6_levels': g6,
                                                                  'G4': 'context ends at issuance (asserted per item); '
                                                                        'the only future covariate is the tide'},
                                   'rows': counts, 'context_hours': CTX, 'min_finite': MIN_FINITE})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['tree', 'chronos'])
    p.add_argument('--variable', choices=list(VARIABLES))
    p.add_argument('--arm', choices=list(TREE_ARMS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    if a.report:
        from src.experiments.v2_probabilistic_report import report
        report()
    elif a.part == 'tree':
        run_tree(a.variable, a.arm, a.seed, a.quick)
    else:
        run_chronos(a.quick)
