"""Addendum 65: a perfect weather forecast as an upper bound, to 72 hours.

Weather forecasts are generally available to about 72 h. This module gives addendum 64's forecasters
the Santa Teresa station's own record after issuance, standing in for a forecast that matches what
later happened, so whatever it gains is the most a real weather forecast could give. Each forecaster
is run three ways: without weather (addendum 64's arm), with the weather recorded up to issuance, and
with that plus the record after issuance.

Declared in config/bay_reserved_protocol_v2_addendum65.json before anything was computed. The 2026
rows are read once, at sampler seed 42; seeds 7 and 123 re-select the trees on Dev-2025 only.

Producers (Trinity) and the report (local):
  python -m src.experiments.v2_weather_oracle --part tree --variable temperature --arm quantile_tree_weather [--seed 7]
  python -m src.experiments.v2_weather_oracle --part chronos
  python -m src.experiments.v2_weather_oracle --report
"""
import argparse
import functools
import json
import os
import time

import numpy as np
import pandas as pd

from src.experiments import v2_horizon, v2_horizon_reserved
from src.experiments import v2_probabilistic as P
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import THREADS, tree_space
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

ROOT, RUNS = P.ROOT, P.RUNS
OUT = RUNS / 'v2_weather_oracle'
PROB = P.OUT                                          # addendum 64's outputs
DECLARATION = 'config/bay_reserved_protocol_v2_addendum65.json'
LEADS = list(P.LEADS) + [72]
MAX_LEAD = 72
WARNING_LEAD = 48                                     # addendum 64's prediction length on the warning series
CHANNELS = ('air', 'solar', 'humidity', 'pressure', 'wind', 'wind_east', 'wind_north', 'precip')
SHARED = ('air', 'solar', 'wind', 'humidity', 'pressure')   # the weather tree's five channels
TREE_ARMS = ('quantile_tree_weather', 'quantile_tree_weather_future')
CHRONOS_ARMS = ('chronos2_net', 'chronos2_net_weather', 'chronos2_net_weather_future')
WARNING_ARMS = ('chronos2_net_weather', 'chronos2_net_weather_future')


# --------------------------------------------------------------------------------------------------
# The station, hourly
# --------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def station():
    """Hourly channels labelled at the interval end (the hour q holds records in [q - 1 h, q))."""
    raw = pd.read_csv(ROOT / 'data/raw/T2-Meteo.txt', comment='#', sep=r'\s+', header=None, encoding='cp1252')
    t = pd.to_datetime(raw[0] + ' ' + raw[1], format='%d/%m/%Y %H:%M:%S', utc=True)
    cols = {'pressure': 3, 'air': 4, 'precip': 5, 'direction': 6, 'humidity': 7, 'solar': 8, 'wind': 12}
    a = pd.DataFrame({k: pd.to_numeric(raw[c], errors='coerce').replace(-9999, np.nan).to_numpy()
                      for k, c in cols.items()}, index=pd.DatetimeIndex(t))
    bounds = {'air': (-10, 50), 'solar': (0, 1500), 'wind': (0, 60), 'humidity': (0, 100), 'pressure': (850, 1100),
              'precip': (0, 100), 'direction': (0, 360)}
    for k, (lo, hi) in bounds.items():
        a.loc[~a[k].between(lo, hi), k] = np.nan
    toward = np.radians(a.direction + 180.0)          # the direction the wind blows towards
    a['wind_east'] = a.wind * np.sin(toward)
    a['wind_north'] = a.wind * np.cos(toward)
    means = a[[c for c in CHANNELS if c != 'precip']].resample('1h', closed='left', label='right').mean()
    means['precip'] = a['precip'].resample('1h', closed='left', label='right').sum(min_count=1)
    return means[list(CHANNELS)]


def positions(index, times):
    at = index.as_unit('ns').get_indexer(pd.DatetimeIndex(times).as_unit('ns'))
    if (at < 0).any():
        raise ValueError(f'{int((at < 0).sum())} issuance times are not on the station index')
    return at


def past_weather(times):
    """Mean (precipitation sum) over the six hourly labels ending at issuance: records before it."""
    h = station()
    rolled = h.rolling(6, min_periods=4).mean()
    rolled['precip'] = h['precip'].rolling(6, min_periods=4).sum()
    p = positions(h.index, times)
    return rolled.to_numpy(float)[p]


def future_weather(times, lead):
    """Over the labels issuance + 1 h to the target hour: the mean (precipitation sum), and the value at
    the target hour. Records from issuance onwards, the perfect forecast."""
    h = station()
    values = h.to_numpy(float)
    ok = np.isfinite(values)
    cs = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(np.where(ok, values, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(ok, axis=0)])
    p = positions(h.index, times)
    a, b = p + 1, p + lead + 1                        # labels p+1 .. p+lead
    assert (b <= len(values)).all(), 'the station record ends before a target hour'
    total, count = cs[b] - cs[a], cn[b] - cn[a]
    with np.errstate(invalid='ignore', divide='ignore'):
        window = np.where(count > 0, total / count, np.nan)
    precip = list(CHANNELS).index('precip')
    window[:, precip] = np.where(count[:, precip] > 0, total[:, precip], np.nan)
    return np.hstack([window, values[p + lead]])


def weather_design(data, fallback, lead, future):
    base = P.tree_design(data, fallback, lead, False)
    extra = past_weather(data['times'])
    if future:
        extra = np.hstack([extra, future_weather(data['times'], lead)])
    extra = extra.astype('float32')
    return np.concatenate([base, np.broadcast_to(extra[:, None, :], (*base.shape[:2], extra.shape[1]))], axis=-1)


def weather_gate():
    """G2: the past means of the five shared channels against bay_forcing.weather on development."""
    from src.data.bay_forcing import weather
    times = pd.date_range(pd.Timestamp('2024-07-25', tz='UTC'), pd.Timestamp('2025-12-31T18:00', tz='UTC'), freq='6h')
    mine = pd.DataFrame(past_weather(times), index=times, columns=CHANNELS)[list(SHARED)]
    ref = weather(times)[list(SHARED)]
    same_nan = bool((mine.isna().to_numpy() == ref.isna().to_numpy()).all())
    diff = float(np.nanmax(np.abs(mine.to_numpy() - ref.to_numpy())))
    return {'issuances': len(times), 'same_missing': same_nan, 'max_abs_difference': diff,
            'passed': bool(same_nan and diff <= 1e-9)}


# --------------------------------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------------------------------
def tree_lead(variable, arm, lead, seed, quick, folder, quantiles, points, studies, plain=False):
    """One arm at one lead on Dev-2025, and at seed 42 on 2026. `plain`: addendum 64's quantile tree
    (and, at seed 42, the climatological quantiles), here only at 72 h."""
    data = v2_horizon.build(variable, lead)
    outer = v2_horizon.folds(data, lead)
    if quick:
        for f in outer:
            f['tr'], f['va'] = f['tr'][-300:], f['va'][:40]
    inner = [v2_horizon.inner_fold(data, f, lead) for f in outer]
    if quick:
        for f in inner:
            f['tr'], f['va'] = f['tr'][-300:], f['va'][:40]
    clim = {id(f): P.clim_table(data, f['tr'], P.LEVELS) for f in outer + inner}

    def design(d, fallback):
        return P.tree_design(d, fallback, lead, False) if plain else weather_design(d, fallback, lead, arm.endswith('_future'))
    designs = {id(f): design(data, f['fallback']) for f in outer + inner}
    name = 'quantile_tree' if plain else arm

    def objective(trial):
        p = tree_space(trial)
        ratios = []
        for f in inner:
            mask, i, s = P.scored(data, f['va'], f['eligible'])
            y = data['truth'][i, s]
            Q = P.fit_predict(data, designs[id(f)], f['tr'], f['va'], f['eligible'], p)
            C = P.clim_predict(data, clim[id(f)], f['va'], mask)
            ratios.append(P.sensor_macro(P.quantile_score(y, Q), s) / P.sensor_macro(P.quantile_score(y, C), s))
        return float(np.mean(ratios))
    sel = run_study(f'{variable}_{name}_h{lead}', objective, folder / 'studies', quick)
    params = {**sel['params'], 'subsample_freq': 1}
    studies[f'{name}|{lead}'] = {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}
    for f in outer:
        frame, mask = P.frame_of(data, f['va'], f['eligible'], 'Dev-2025', f['number'], lead)
        quantiles.append(P.with_quantiles(frame, name, P.fit_predict(data, designs[id(f)], f['tr'], f['va'],
                                                                   f['eligible'], params), data))
        if plain and seed == 42:
            quantiles.append(P.with_quantiles(frame, 'clim_quantiles', P.clim_predict(data, clim[id(f)], f['va'], mask), data))
            month_hour, hours = v2_horizon.climatology(data, f)
            _, i, s = P.scored(data, f['va'], f['eligible'])
            points.append(frame.assign(clim_month_hour=month_hour[i, s] * data['scale'] + data['center'],
                                       clim_hour=hours[i, s] * data['scale'] + data['center']))
    if seed == 42:
        data26 = v2_horizon_reserved.build(variable, lead)
        bundle = v2_horizon_reserved.split(data26, lead)
        if quick:
            bundle['tr'], bundle['va'] = bundle['tr'][-300:], bundle['va'][:40]
        frame, mask = P.frame_of(data26, bundle['va'], bundle['eligible'], 'Reserved-2026', None, lead)
        quantiles.append(P.with_quantiles(frame, name, P.fit_predict(data26, design(data26, bundle['fallback']),
                                                                   bundle['tr'], bundle['va'], bundle['eligible'],
                                                                   params), data26))
        if plain:
            table = P.clim_table(data26, bundle['tr'], P.LEVELS)
            quantiles.append(P.with_quantiles(frame, 'clim_quantiles', P.clim_predict(data26, table, bundle['va'], mask),
                                              data26))
            _, i, s = P.scored(data26, bundle['va'], bundle['eligible'])
            month_hour = v2_horizon_reserved.climatology(data26, bundle)
            hours = P.hour_only(data26, bundle['tr'])
            points.append(frame.assign(clim_month_hour=month_hour[i, s] * data26['scale'] + data26['center'],
                                       clim_hour=hours[i, s] * data26['scale'] + data26['center']))


def run_tree(variable, arm, seed=42, quick=False):
    if arm not in TREE_ARMS:
        raise ValueError(arm)
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    leads = [6, 72] if quick else LEADS
    quantiles, points, studies, gates = [], [], {}, {}
    if seed == 42 and arm == 'quantile_tree_weather':
        gates['G2_weather'] = weather_gate()
    for lead in leads:
        t0 = time.time()
        tree_lead(variable, arm, lead, seed, quick, folder, quantiles, points, studies)
        if lead == 72 and arm == 'quantile_tree_weather':
            tree_lead(variable, arm, lead, seed, quick, folder, quantiles, points, studies, plain=True)
        print(variable, arm, 'seed', seed, 'lead', lead, f'{time.time() - t0:.0f} s', flush=True)
    q = pd.concat(quantiles, ignore_index=True)
    num = ['truth', 'last'] + P.QCOLS
    q[num] = q[num].round(P.DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_{arm}.csv.gz', index=False)
    if points:
        p = pd.concat(points, ignore_index=True)
        cols = ['truth', 'last', 'clim_month_hour', 'clim_hour']
        p[cols] = p[cols].round(9)
        p.to_csv(folder / f'points72_{variable}.csv.gz', index=False)
    save(folder / f'tree_{variable}_{arm}.json', {'declared_in': DECLARATION, 'variable': variable, 'arm': arm,
                                                  'seed': seed, 'smoke_only': quick, 'leads': leads,
                                                  'studies': studies, 'gates': gates, 'threads': THREADS})
    print(variable, arm, 'seed', seed, 'gates', json.dumps({k: v.get('passed') for k, v in gates.items()}), flush=True)


# --------------------------------------------------------------------------------------------------
# Chronos-2
# --------------------------------------------------------------------------------------------------
class WeatherPanel:
    """A variable's hourly series, the network covariates and the station, on one hourly index that runs
    past the last issuance by the prediction length."""

    def __init__(self, series, temperatures, index, length):
        self.index, self.length = index, length
        self.series = {n: s.reindex(index).to_numpy(float) for n, s in series.items()}
        t = pd.DataFrame({n: s.reindex(index) for n, s in temperatures.items()})
        self.net = {'net_T': t.mean(axis=1).to_numpy(float), 'net_sd': t.std(axis=1).to_numpy(float)}
        self.own_t = {n: t[n].to_numpy(float) for n in t.columns}
        w = station().reindex(index)
        self.weather = {f'wx_{c}': w[c].to_numpy(float) for c in CHANNELS}

    def items(self, name, positions_, variable, arms):
        series = self.series[name]
        out = {arm: [] for arm in arms}
        kept = []
        L = self.length
        for p in positions_:
            a = max(0, p - P.CTX + 1)
            w = series[a:p + 1]
            if np.isfinite(w).sum() < P.MIN_FINITE or not np.isfinite(series[p]) or p + L >= len(self.index):
                continue
            target = w.astype(np.float32)
            past = {k: P._filled(v[a:p + 1]) for k, v in self.net.items()}
            if variable == 'oxygen' and name in self.own_t:
                past['own_T'] = P._filled(self.own_t[name][a:p + 1])
            wx_past = {k: P._filled(v[a:p + 1]) for k, v in self.weather.items()}
            wx_future = {k: P._filled(v[p + 1:p + 1 + L]) for k, v in self.weather.items()}
            assert all(len(v) == L for v in wx_future.values())
            if 'chronos2_net' in out:
                out['chronos2_net'].append({'target': target, 'past_covariates': past})
            if 'chronos2_net_weather' in out:
                out['chronos2_net_weather'].append({'target': target, 'past_covariates': {**past, **wx_past}})
            if 'chronos2_net_weather_future' in out:
                out['chronos2_net_weather_future'].append({'target': target, 'past_covariates': {**past, **wx_past},
                                                           'future_covariates': wx_future})
            kept.append(p)
        return out, kept


def predict(pipe, items, keep, length):
    import torch
    result = [None] * len(items)
    groups = {}
    for n, it in enumerate(items):
        groups.setdefault((tuple(sorted(it.get('past_covariates', {}))),
                           tuple(sorted(it.get('future_covariates', {})))), []).append(n)
    for members in groups.values():
        for k in range(0, len(members), 128):
            chunk = members[k:k + 128]
            got = pipe.predict([items[n] for n in chunk], prediction_length=length)
            for n, o in zip(chunk, got):
                a = o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                result[n] = a.reshape(-1, len(pipe.quantiles), a.shape[-1])[0][keep]
    return np.stack(result).astype(np.float32)


def run_chronos(quick=False):
    base = '/home/vito/smartbay_do/.hf'
    if os.path.isdir(base):
        os.environ.setdefault('HF_HOME', base)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(THREADS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained(P.CHRONOS2, device_map=device, torch_dtype=torch.float32)
    levels = [float(a) for a in pipe.quantiles]
    want = (P.EDGE[0],) + P.LEVELS + (P.EDGE[1],)
    keep = [min(range(len(levels)), key=lambda n: abs(levels[n] - a)) for a in want]
    warn_keep = [want.index(a) for a in P.WARN_LEVELS]
    cols = [P.qcol(a) for a in want]
    folder = OUT / ('smoke' if quick else '') / 'chronos'
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    meta, frames = P.bay_reserved_evaluation.load_all_observations()
    temps = {n: f['temperature'] for n, f in frames.items() if 'temperature' in f and f['temperature'].notna().any()}
    for variable in P.VARIABLES:
        t0 = time.time()
        names = [n for n in meta.index if n in frames and variable in frames[n] and frames[n][variable].notna().any()]
        index = pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), P.RESERVED_WINDOW[1] + pd.Timedelta(hours=MAX_LEAD + 1),
                              freq='h')
        panel = WeatherPanel({n: frames[n][variable] for n in names}, temps, index, MAX_LEAD)
        grid = np.flatnonzero(np.isin(index.hour, (0, 6, 12, 18))
                              & (((index >= P.DEV_WINDOW[0]) & (index < P.DEV_WINDOW[1]))
                                 | ((index >= P.RESERVED_WINDOW[0]) & (index <= P.RESERVED_WINDOW[1]))))
        rows = []
        for name in names:
            pos = grid[::max(1, len(grid) // 12)][:12] if quick else grid
            items, kept = panel.items(name, pos, variable, CHRONOS_ARMS)
            if not kept:
                continue
            kept = np.asarray(kept)
            seen = np.stack([np.isfinite(panel.series[name][kept + h]) for h in LEADS], axis=1)
            for arm in CHRONOS_ARMS:
                Q = predict(pipe, items[arm], keep, MAX_LEAD)
                for li, h in enumerate(LEADS):
                    ok = seen[:, li]
                    if not ok.any():
                        continue
                    t = index[kept[ok]]
                    block = pd.DataFrame(Q[ok][:, :, h - 1], columns=cols)
                    block.insert(0, 'arm', arm)
                    block.insert(0, 'lead', h)
                    block.insert(0, 'time', t)
                    block.insert(0, 'sensor', name)
                    block.insert(0, 'cohort', np.where(t < P.RESERVED_WINDOW[0], 'Dev-2025', 'Reserved-2026'))
                    rows.append(block)
        frame = pd.concat(rows, ignore_index=True)
        frame[cols] = frame[cols].round(P.DIGITS[variable])
        frame.to_csv(folder / f'chronos_{variable}.csv.gz', index=False)
        counts[variable] = int(len(frame))
        print('chronos grid', variable, len(frame), f'{time.time() - t0:.0f} s', flush=True)
    # hourly on the warning series, as addendum 64
    t0 = time.time()
    obs = pd.read_csv(ROOT / 'reports/bay_twin/hourly_observations.csv.gz', index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    index = pd.date_range(obs.index.min(), P.WARNING_SPAN[1] + pd.Timedelta(hours=WARNING_LEAD + 1), freq='h')
    series = {}
    for s in P.WARNING_SENSORS:
        c = f'{s}::oxygen'
        if c in obs.columns:
            v = obs[c].dropna()
            series[s] = v[v.between(0, 600)]
    temps_w = {c.split('::')[0]: obs[c] for c in obs.columns if c.endswith('::temperature')}
    panel = WeatherPanel(series, temps_w, index, WARNING_LEAD)
    span = np.flatnonzero((index >= P.WARNING_SPAN[0]) & (index < P.WARNING_SPAN[1]))
    rows = []
    for name in panel.series:
        pos = span[::max(1, len(span) // 24)][:24] if quick else span
        items, kept = panel.items(name, pos, 'oxygen', WARNING_ARMS)
        if not kept:
            continue
        frame = pd.DataFrame({'sensor': name, 'time': index[np.asarray(kept)]})
        for arm in WARNING_ARMS:
            Q = predict(pipe, items[arm], keep, WARNING_LEAD)[:, warn_keep, :6]
            block = frame.assign(arm=arm)
            for li, a in enumerate(P.WARN_LEVELS):
                for k in range(6):
                    block[f'k{k + 1}_{P.qcol(a)}'] = np.round(Q[:, li, k], 3)
            rows.append(block)
    warn = pd.concat(rows, ignore_index=True)
    warn.to_csv(folder / 'chronos_warning.csv.gz', index=False)
    counts['warning'] = int(len(warn))
    print('chronos warning', len(warn), f'{time.time() - t0:.0f} s', flush=True)
    save(folder / 'chronos.json', {'declared_in': DECLARATION, 'model': P.CHRONOS2, 'device': device, 'smoke_only': quick,
                                   'prediction_length': {'grid': MAX_LEAD, 'warning': WARNING_LEAD},
                                   'hf_home': os.environ.get('HF_HOME'), 'rows': counts,
                                   'channels': list(CHANNELS)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['tree', 'chronos'])
    p.add_argument('--variable', choices=list(P.VARIABLES))
    p.add_argument('--arm', choices=list(TREE_ARMS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    if a.report:
        from src.experiments.v2_weather_oracle_report import report
        report()
    elif a.part == 'tree':
        run_tree(a.variable, a.arm, a.seed, a.quick)
    else:
        run_chronos(a.quick)
