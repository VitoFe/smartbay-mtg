"""Addendum 80: the currents measured at Mitili Est as a perfect input, to 72 hours.

Addendum 65 gave the quantile tree the station's weather record after issuance, standing in for a
perfect weather forecast. This module does the same for the water's motion: the north and east
velocity components measured at Mitili Est at 2 and 8 m, first only up to issuance (what the meters
could give an operational forecaster), then also after it (a perfect forecast of the currents at that
mooring), each on top of addendum 65's arm with the same weather. What the second gains estimates what
a perfect current forecast there adds to this model; it does not bound what a circulation model, which
supplies fields everywhere, could add.

Declared in config/bay_reserved_protocol_v2_addendum80.json before any current value entered a design.
The 2026 rows are read once, at sampler seed 42; seeds 7 and 123 re-select the trees on Dev-2025 only.

Producer (Trinity):
  python -m src.experiments.v2_currents_oracle --variable temperature --arm quantile_tree_weather_currents [--seed 7]
"""
import argparse
import functools
import json
import time

import numpy as np
import pandas as pd

from src.experiments import v2_horizon, v2_horizon_reserved
from src.experiments import v2_probabilistic as P
from src.experiments import v2_weather_oracle as W
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import THREADS, tree_space
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

ROOT, RUNS = P.ROOT, P.RUNS
OUT = RUNS / 'v2_currents_oracle'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum80.json'
LEADS = W.LEADS                                       # 1, 3, 6, 12, 24, 36, 48, 72
METERS = {'e2': 'data/raw/TMitiliE_02m-Corrente.txt', 'e8': 'data/raw/TMitiliE_08m-Corrente.txt'}
COMPONENTS = ('north', 'east')
ARMS = {'quantile_tree_weather_currents': False, 'quantile_tree_weather_future_currents_future': True}
SENTINEL = -999.0
SPEED_TOLERANCE = 0.002                               # m/s, gate G3
COVERED = 0.75                                        # share of window hours with a 2 m value, the primary subgroup


# --------------------------------------------------------------------------------------------------
# The meters, hourly
# --------------------------------------------------------------------------------------------------
def read_meter(path):
    """The raw records: time (UTC), speed and the two components, sentinel rows dropped."""
    rows = [line.split() for line in (ROOT / path).read_text(encoding='latin-1').splitlines()
            if line.strip() and not line.startswith('#')]
    raw = pd.DataFrame(rows)
    t = pd.to_datetime(raw[0] + ' ' + raw[1], format='%d/%m/%Y %H:%M:%S', utc=True)
    a = pd.DataFrame({'speed': pd.to_numeric(raw[4], errors='coerce'),
                      'north': pd.to_numeric(raw[6], errors='coerce'),
                      'east': pd.to_numeric(raw[7], errors='coerce')}).set_index(pd.DatetimeIndex(t))
    return a[(a > SENTINEL).all(axis=1)].sort_index()


@functools.lru_cache(maxsize=None)
def currents():
    """Hourly means labelled at the interval end, on the station's hourly index (W.station)."""
    index = W.station().index
    cols = {}
    for tag, path in METERS.items():
        hourly = read_meter(path)[list(COMPONENTS)].resample('1h', closed='left', label='right').mean()
        for c in COMPONENTS:
            cols[f'{tag}_{c}'] = hourly[c].reindex(index)
    return pd.DataFrame(cols, index=index)


def currents_gate():
    """G3: the speed recomputed from the two components against the file's speed."""
    out = {}
    for tag, path in METERS.items():
        a = read_meter(path)
        err = (np.hypot(a.north, a.east) - a.speed).abs()
        share = float((err <= SPEED_TOLERANCE).mean())
        out[tag] = {'records': int(len(a)), 'share_within_tolerance': share, 'passed': share >= 0.99}
    return {'meters': out, 'passed': all(v['passed'] for v in out.values())}


def past_currents(times):
    """Mean over the six hourly labels ending at issuance (at least four present)."""
    h = currents()
    p = W.positions(h.index, times)
    return h.rolling(6, min_periods=4).mean().to_numpy(float)[p]


def future_currents(times, lead):
    """Over the labels issuance + 1 h to the target hour: the mean, and the value at the target hour."""
    h = currents()
    values = h.to_numpy(float)
    ok = np.isfinite(values)
    cs = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(np.where(ok, values, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(ok, axis=0)])
    p = W.positions(h.index, times)
    a, b = p + 1, p + lead + 1
    assert (b <= len(values)).all(), 'the current record index ends before a target hour'
    total, count = cs[b] - cs[a], cn[b] - cn[a]
    with np.errstate(invalid='ignore', divide='ignore'):
        window = np.where(count > 0, total / count, np.nan)
    return np.hstack([window, values[p + lead]])


def window_coverage(times, lead):
    """The share of the labels issuance + 1 h to the target hour with a 2 m north component."""
    h = currents()
    ok = np.isfinite(h['e2_north'].to_numpy(float))
    cn = np.concatenate([[0], np.cumsum(ok)])
    p = W.positions(h.index, times)
    return (cn[p + lead + 1] - cn[p + 1]) / lead


def currents_design(data, fallback, lead, future):
    """Addendum 65's weather design (past weather; with `future`, past and future weather), then the
    current block: past means, and with `future` the window means and target-hour values."""
    base = W.weather_design(data, fallback, lead, future)
    extra = past_currents(data['times'])
    if future:
        extra = np.hstack([extra, future_currents(data['times'], lead)])
    extra = extra.astype('float32')
    return base, np.concatenate([base, np.broadcast_to(extra[:, None, :], (*base.shape[:2], extra.shape[1]))],
                                axis=-1)


# --------------------------------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------------------------------
def tree_lead(variable, arm, lead, seed, quick, folder, quantiles, studies, gates):
    """One arm at one lead on Dev-2025, and at seed 42 on 2026, as v2_weather_oracle.tree_lead."""
    future = ARMS[arm]
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
    designs, same = {}, True
    for f in outer + inner:
        base, full = currents_design(data, f['fallback'], lead, future)
        same &= bool(np.array_equal(base, W.weather_design(data, f['fallback'], lead, future), equal_nan=True))
        designs[id(f)] = full

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
    sel = run_study(f'{variable}_{arm}_h{lead}', objective, folder / 'studies', quick)
    params = {**sel['params'], 'subsample_freq': 1}
    studies[f'{arm}|{lead}'] = {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}
    for f in outer:
        frame, _ = P.frame_of(data, f['va'], f['eligible'], 'Dev-2025', f['number'], lead)
        quantiles.append(P.with_quantiles(frame, arm, P.fit_predict(data, designs[id(f)], f['tr'], f['va'],
                                                                  f['eligible'], params), data))
    if seed == 42:
        data26 = v2_horizon_reserved.build(variable, lead)
        bundle = v2_horizon_reserved.split(data26, lead)
        if quick:
            bundle['tr'], bundle['va'] = bundle['tr'][-300:], bundle['va'][:40]
        base, full = currents_design(data26, bundle['fallback'], lead, future)
        same &= bool(np.array_equal(base, W.weather_design(data26, bundle['fallback'], lead, future), equal_nan=True))
        frame, _ = P.frame_of(data26, bundle['va'], bundle['eligible'], 'Reserved-2026', None, lead)
        quantiles.append(P.with_quantiles(frame, arm, P.fit_predict(data26, full, bundle['tr'], bundle['va'],
                                                                  bundle['eligible'], params), data26))
    gates.setdefault('G1_design', {})[str(lead)] = same


def run_tree(variable, arm, seed=42, quick=False):
    if arm not in ARMS:
        raise ValueError(arm)
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    leads = [6, 72] if quick else LEADS
    quantiles, studies, gates = [], {}, {}
    if seed == 42:
        gates['G3_currents'] = currents_gate()
    for lead in leads:
        t0 = time.time()
        tree_lead(variable, arm, lead, seed, quick, folder, quantiles, studies, gates)
        print(variable, arm, 'seed', seed, 'lead', lead, f'{time.time() - t0:.0f} s', flush=True)
    gates['G1_design']['passed'] = all(v for k, v in gates['G1_design'].items() if k != 'passed')
    q = pd.concat(quantiles, ignore_index=True)
    num = ['truth', 'last'] + P.QCOLS
    q[num] = q[num].round(P.DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_{arm}.csv.gz', index=False)
    save(folder / f'tree_{variable}_{arm}.json', {'declared_in': DECLARATION, 'variable': variable, 'arm': arm,
                                                  'seed': seed, 'smoke_only': quick, 'leads': leads,
                                                  'meters': METERS, 'studies': studies, 'gates': gates,
                                                  'threads': THREADS})
    print(variable, arm, 'seed', seed, 'gates', json.dumps({k: v.get('passed') for k, v in gates.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=list(P.VARIABLES), required=True)
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run_tree(a.variable, a.arm, a.seed, a.quick)
