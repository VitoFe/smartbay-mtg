"""Addendum 53: build, locally, every context the compute host needs, with the published code paths.

For each task and variable this writes one npz under reports/bay_twin/runs/v2_context_holes/inputs/:

  raw          the context window with its holes as NaN, right-aligned in 512 columns (`length`
               gives how many are real); the mask treatment passes exactly this to Chronos-Bolt
  linear       the published context: A1 as v2_sequence_arms.gather builds it (interpolated, then
               left-padded with its first value to 512), A2 and B interpolated with
               limit_direction='both' at their own length (right-aligned, NaN to the left)
  fwd_*, bwd_* the Chronos-2 contexts of every hole to bridge: forward, the 512 h before the hole;
               backward, the record from the hole's end to the last hour the forecast may read,
               at most 512 h, reversed. Both keep their own holes as NaN
  bridge_*     one row per (hole, backward context): its forward item, backward item and length
  place_*      where each bridged hour goes: context row, bridge row, offset in the hole, column

The tasks (config/bay_reserved_protocol_v2_addendum53.json):

  A1  the addendum-23 six-hour endpoints (Forecast-2025 validation rows of both outer folds)
  A2  the reserved 2026 battery's rows (exp_chronos.predict_at's contexts)
  B   the forward-only gap filler's left windows, Gap-Dev and Gap-2026, gaps of 1 to 72 h

Gate G4 (causality) is asserted here: no bridging context reads the last hour the forecast may use
or anything after it, beyond what that forecast itself reads.

Usage: python -m scripts.v2_context_holes_inputs
"""
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT
from src.experiments import v2_gap_recovery as gap
from src.experiments import v2_sequence_arms as seq
from src.experiments.v2_horizon import build, folds

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_context_holes' / 'inputs'
CTX = 512
BATTERY = ROOT / 'reports/bay_twin/runs/battery/results/chronos/chronos_predictions.csv.gz'
BATTERY_OBS = ROOT / 'reports/bay_twin/hourly_observations.csv.gz'
GAP_RUNS = {'Gap-Dev': ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery', 'development'),
            'Gap-2026': ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026', 'reserved')}
MAX_GAP = 72


def runs_of_nan(series):
    """(start, stop) inclusive of every maximal run of missing values."""
    missing = ~np.isfinite(series)
    if not missing.any():
        return np.zeros(0, int), np.zeros(0, int)
    edges = np.diff(np.concatenate([[0], missing.astype(int), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1


class Plan:
    """Collects contexts, bridging items and placements for one task and variable."""

    def __init__(self):
        self.raw, self.length, self.linear = [], [], []
        self.fwd, self.bwd, self.bridges, self.place = {}, {}, {}, []
        self.holes_any, self.holes_long = [], []

    def add(self, series, runs, lo, end, linear):
        """One context: the window series[lo:end + 1], `end` the last hour the forecast reads."""
        n = len(self.raw)
        window = series[lo:end + 1]
        assert len(window) <= CTX
        row = np.full(CTX, np.nan, np.float32)
        row[CTX - len(window):] = window
        self.raw.append(row)
        self.length.append(len(window))
        lin = np.full(CTX, np.nan, np.float32)
        lin[CTX - len(linear):] = linear
        self.linear.append(lin)
        starts, stops = runs
        # runs that intersect the window and close before its last hour
        first = np.searchsorted(stops, lo)
        longest = 0
        for a, b in zip(starts[first:], stops[first:]):
            if a > end:
                break
            longest = max(longest, min(b, end) - max(a, lo) + 1)
            if b >= end:
                continue                      # no right edge inside what the forecast may read
            forward = series[max(0, a - CTX):a]
            if not np.isfinite(forward).any():
                continue                      # nothing before it: left to the linear treatment
            e = min(end, b + CTX)
            assert b + 1 <= e <= end and max(0, a - CTX) >= 0 and a - 1 < end   # G4
            backward = series[b + 1:e + 1][::-1]
            fk = (a, b)
            if fk not in self.fwd:
                self.fwd[fk] = (len(self.fwd), forward)
            bk = (a, b, e)
            if bk not in self.bwd:
                self.bwd[bk] = (len(self.bwd), backward)
            key = (fk, bk)
            if key not in self.bridges:
                self.bridges[key] = (len(self.bridges), self.fwd[fk][0], self.bwd[bk][0], b - a + 1)
            bridge = self.bridges[key][0]
            for p in range(max(a, lo), b + 1):
                self.place.append((n, bridge, p - a, CTX - 1 - (end - p)))
        self.holes_any.append(longest > 0)
        self.holes_long.append(longest >= 24)

    def arrays(self):
        def stack(items):
            ctx = np.full((len(items), CTX), np.nan, np.float32)
            lens = np.zeros(len(items), int)
            for i, (_, v) in enumerate(sorted(items.values(), key=lambda t: t[0])):
                ctx[i, CTX - len(v):] = v
                lens[i] = len(v)
            return ctx, lens
        fwd_ctx, fwd_len = stack(self.fwd)
        bwd_ctx, bwd_len = stack(self.bwd)
        bridges = sorted(self.bridges.values())
        place = np.array(self.place, int).reshape(-1, 4)
        return dict(raw=np.stack(self.raw), length=np.array(self.length), linear=np.stack(self.linear),
                    fwd_ctx=fwd_ctx, fwd_len=fwd_len, bwd_ctx=bwd_ctx, bwd_len=bwd_len,
                    bridge_f=np.array([b[1] for b in bridges], int),
                    bridge_b=np.array([b[2] for b in bridges], int),
                    bridge_len=np.array([b[3] for b in bridges], int),
                    place_n=place[:, 0], place_bridge=place[:, 1], place_k=place[:, 2],
                    place_pos=place[:, 3], hole_any=np.array(self.holes_any),
                    hole_24h=np.array(self.holes_long))


def a1(variable):
    """The addendum-23 validation rows, in v2_sequence_arms.run's order."""
    data = build(variable, seq.HORIZON)
    names, index, z, scale = seq.hourly_panel(variable)
    assert names == data['names']
    q = seq.assert_aligned(data, z, index)
    plan, keys = Plan(), []
    runs = {s: runs_of_nan(z[:, s]) for s in range(len(names))}
    for fi, fold in enumerate(folds(data, seq.HORIZON)):
        va = seq.eligible_rows(data, fold, 'va')
        tr = seq.eligible_rows(data, fold, 'tr')
        va_ctx, va_keep = seq.gather(data, z, q, va)
        _, tr_keep = seq.gather(data, z, q, tr)
        if not len(va_keep) or not len(tr_keep):
            continue
        for k, (i, s) in enumerate(va_keep):
            end = int(q[i])
            lo = max(0, end - seq.CTX + 1)
            linear = va_ctx[k]
            plan.add(z[:, s], runs[s], lo, end, linear)
            keys.append((fi, names[s], str(data['times'][i])))
    out = plan.arrays()
    out.update(pad_edge=True, horizon=seq.HORIZON, scale=scale,
               key_fold=np.array([k[0] for k in keys]), key_sensor=np.array([k[1] for k in keys]),
               key_time=np.array([k[2] for k in keys]))
    return out


def battery_series():
    """exp_chronos's `obs`, rebuilt from the same file (checked equal to the host's input)."""
    obs = pd.read_csv(BATTERY_OBS, index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    return obs.sort_index().reindex(pd.date_range(obs.index.min(), obs.index.max(), freq='h'))


def a2(variable, obs, endpoints):
    """exp_chronos.predict_at's contexts: 512 h ending at the issuance, admissible with >= 48 observed."""
    part = endpoints[endpoints.variable == variable]
    plan, keys = Plan(), []
    for sensor in sorted(part.sensor.unique()):
        column = f'{sensor}::{variable}'
        if column not in obs.columns:
            continue
        series = obs[column].to_numpy(float)
        runs = runs_of_nan(series)
        for t in pd.DatetimeIndex(sorted(part.loc[part.sensor == sensor, 'time'])):
            end = int(obs.index.get_indexer([t])[0])
            assert end >= 0
            lo = max(0, end - CTX + 1)
            hist = series[lo:end + 1]
            if np.isfinite(hist).sum() < 48:
                continue
            linear = pd.Series(hist).interpolate(limit_direction='both').to_numpy(np.float32)
            plan.add(series, runs, lo, end, linear)
            keys.append((variable, sensor, str(t)))
    out = plan.arrays()
    out.update(pad_edge=False, horizon=6,
               key_variable=np.array([k[0] for k in keys]), key_sensor=np.array([k[1] for k in keys]),
               key_time=np.array([k[2] for k in keys]))
    return out


def b(variable, cohort):
    """chronos_fill's forward windows for the published episodes of 1 to 72 h, in its order."""
    run, results, period = GAP_RUNS[cohort]
    frame = pd.read_csv(RUNS / run / results / f'{variable}_chronos_forward' / 'predictions.csv.gz')
    frame = frame[frame.gap_length <= MAX_GAP]
    meta, frames = gap.observations(period)
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    panel = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    position_of = {t: i for i, t in enumerate(index)}
    plan, keys, lengths = Plan(), [], []
    runs = {n: runs_of_nan(panel[n].to_numpy(float)) for n in names}
    for key in frame.episode.unique():
        sensor, start, length = key.rsplit('|', 2)
        g0 = position_of[pd.Timestamp(start)]
        series = panel[sensor].to_numpy(float)
        lo, end = max(0, g0 - CTX), g0 - 1
        window = series[lo:end + 1]
        linear = np.nan_to_num(pd.Series(window).interpolate(limit_direction='both').to_numpy()
                               ).astype(np.float32)
        plan.add(series, runs[sensor], lo, end, linear)
        keys.append(key)
        lengths.append(int(length))
    out = plan.arrays()
    out.update(pad_edge=False, key_episode=np.array(keys), gap_len=np.array(lengths))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    endpoints = pd.read_csv(BATTERY)[['variable', 'sensor', 'time']]
    endpoints['time'] = pd.to_datetime(endpoints.time, utc=True)
    obs = battery_series()
    manifest = {'declared_in': 'config/bay_reserved_protocol_v2_addendum53.json', 'files': {}}
    jobs = [(f'A1_{v}', lambda v=v: a1(v)) for v in ('temperature', 'oxygen')]
    jobs += [(f'A2_{v}', lambda v=v: a2(v, obs, endpoints)) for v in ('temperature', 'oxygen')]
    jobs += [(f'B_{c}_{v}', lambda v=v, c=c: b(v, c)) for c in GAP_RUNS for v in ('temperature', 'oxygen')]
    for name, make in jobs:
        arrays = make()
        path = OUT / f'{name}.npz'
        np.savez_compressed(path, **arrays)
        manifest['files'][path.name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'contexts': int(len(arrays['raw'])),
            'with_any_hole': int(arrays['hole_any'].sum()),
            'with_a_hole_of_24h': int(arrays['hole_24h'].sum()),
            'holes_to_bridge': int(len(arrays['bridge_len'])),
            'chronos2_forward_items': int(len(arrays['fwd_len'])),
            'chronos2_backward_items': int(len(arrays['bwd_len'])),
            'bridged_hours': int(len(arrays['place_n']))}
        print(name, manifest['files'][path.name], flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
