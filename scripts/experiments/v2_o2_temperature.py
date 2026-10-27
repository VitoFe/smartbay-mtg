"""Addendum 54 on the compute host: the sonde's own temperature as a covariate for oxygen.

Run from the uploaded snapshot's root with PYTHONPATH set to it, on GPU 1, SMARTBAY_FULL_CACHE
naming the companion precompute (its panels and zero-shot forecasts):

  python -u scripts/experiments/v2_o2_temperature.py [--stages gaps,forecasts] [--quick]

gaps       For every published oxygen episode (inputs/episodes.csv.gz), the sonde's temperature over
           the same hours filled by the thesis's temperature method (its members from the thesis's
           own functions, the stacked ensemble with the frozen weights), then Chronos-2 from both
           edges without and with the sonde's temperature (observed, filled), with and without the
           four nearest sondes. The no-temperature runs are gate G1; G2 re-fills 20 episodes with the
           sonde's temperature inside the gap hidden.
forecasts  For every issuance (inputs/issuances.json), Chronos-2 48 h ahead with the oxygen record,
           the sonde's past temperature, and the temperature ahead forecast (the precompute's
           temperature c2_net) or observed; with and without the network means. c2 and c2net are
           gate G3 against the precompute.

Writes results/gaps_<cohort>.csv.gz, results/g2.json and results/forecasts.npz.
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / 'reports/bay_twin/runs/v2_o2_temperature/inputs'
OUT = ROOT / 'results'
CACHE = Path(os.environ.get('SMARTBAY_FULL_CACHE', ROOT / 'cache'))
PLAN_GPR = (336, 1440)            # thesis PLAN for temperature: the Gaussian process there, the stack elsewhere
FALLBACK = ('gpr', 'chronos2_cov_bidir', 'chronos_bidir', 'linear')
LEADS = 48
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'companion'))


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


@functools.lru_cache(maxsize=None)
def gap_module():
    """v2_gap_recovery with its reader cached per period and able to hide the target sonde's
    temperature inside one gap (gate G2); nothing it computes is changed."""
    from src.experiments import v2_gap_recovery as G
    base = functools.lru_cache(maxsize=4)(G.observations)

    def observations(period=None):
        meta, frames = base(period or G.PERIOD)
        if G._hide is None:
            return meta, frames
        sensor, a, b = G._hide
        frames = dict(frames)
        f = frames[sensor].copy()
        f.loc[(f.index >= a) & (f.index < b), 'temperature'] = np.nan
        frames[sensor] = f
        return meta, frames
    G._hide = None
    G.observations = observations
    try:
        from smartbay_companion.thesis import cache_chronos_pipelines
        cache_chronos_pipelines()
    except ImportError:
        pass
    return G


def episode_frame(eps):
    parts = [pd.DataFrame({'episode': r.episode, 'position': np.arange(1, r.gap_length + 1),
                           'gap_length': r.gap_length, 'sensor': r.sensor}) for r in eps.itertuples()]
    return pd.concat(parts, ignore_index=True)


def _gpr_chunk(args):
    period, frame = args
    os.environ['OMP_NUM_THREADS'] = '1'
    G = gap_module()
    G.PERIOD = period
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return G.gpr_fill(frame.reset_index(drop=True), 'temperature')
        except RuntimeError:
            return np.full(len(frame), np.nan)


def temperature_members(G, frame, period, workers):
    """The thesis's temperature members on the oxygen episodes' hours (their gaps are never read)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = {'chronos_bidir': np.asarray(G.chronos_fill('temperature', frame, True), float),
               'chronos2_cov_bidir': np.asarray(G.chronos_fill('temperature', frame, True, repo=G.CHRONOS2_REPO,
                                                               covariates=True), float)}
    keys = frame.episode.to_numpy()
    cut = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    shares = np.array_split(np.arange(len(cut) - 1), max(1, min(workers, len(cut) - 1)))
    jobs = [(period, frame.iloc[cut[s[0]]:cut[s[-1] + 1]]) for s in shares if len(s)]
    if workers > 1 and len(jobs) > 1:
        with mp.get_context('spawn').Pool(len(jobs)) as pool:
            got = pool.map(_gpr_chunk, jobs)
    else:
        got = [_gpr_chunk(j) for j in jobs]
    out['gpr'] = np.concatenate(got)
    # linear interpolation between the sonde's temperature at the gap's edges
    _, frames = G.observations()
    lin, truth = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for key, rows in frame.groupby('episode', sort=False).groups.items():
        sensor, start, length = key.rsplit('|', 2)
        t0, L = pd.Timestamp(start), int(length)
        s = frames[sensor]['temperature']
        y0, y1 = s.get(t0 - pd.Timedelta(hours=1), np.nan), s.get(t0 + pd.Timedelta(hours=L), np.nan)
        k = np.arange(1, L + 1)
        lin[rows] = y0 + (y1 - y0) * k / (L + 1)
        truth[rows] = s.reindex(pd.date_range(t0, periods=L, freq='h')).to_numpy(float)
    out['linear'] = lin
    return out, truth


def temperature_fill(members, frame, eps, weights, period):
    """The thesis's temperature method for each gap's length, with the declared fallbacks."""
    info = eps.set_index('episode')
    stack = np.full(len(frame), np.nan)
    band = frame.gap_length.map(lambda L: '1-6' if L <= 6 else '12-72' if L <= 72 else '168-1440').to_numpy()
    fold = frame.episode.map(info.fold).to_numpy()
    M = np.column_stack([members[m] for m in weights['members']])
    for i in range(len(frame)):
        w = weights['by_band'][band[i]]['all' if period == 'reserved' else f'f{int(fold[i])}']
        stack[i] = float(np.dot(w, M[i]))
    members = {**members, 'ens_stack': stack}
    fill, pick = np.full(len(frame), np.nan), np.empty(len(frame), dtype=object)
    for key, rows in frame.groupby('episode', sort=False).groups.items():
        L = int(key.rsplit('|', 2)[2])
        order = (('gpr',) if L in PLAN_GPR else ('ens_stack',)) + FALLBACK
        for m in order:
            v = members[m][rows]
            if np.isfinite(v).all():
                fill[rows], pick[rows] = v, m
                break
        else:
            pick[rows] = 'none'
    return fill, pick, members


def c2_fill(G, variable, frame, neighbours, own):
    """chronos_fill's Chronos-2 branch (v2_gap_recovery.chronos_fill, repo=CHRONOS2_REPO), line for
    line, with one more covariate: the sonde's own temperature (own='obs' its readings over the gap,
    or {episode: fill} the filled ones; None leaves the published arm unchanged, gate G1)."""
    import torch
    from chronos import BaseChronosPipeline
    CHRONOS_CTX, NEIGHBOURS = G.CHRONOS_CTX, G.NEIGHBOURS
    meta, frames = G.observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    panel = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    temp = {n: frames[n]['temperature'].reindex(index).to_numpy(float) if 'temperature' in frames[n]
            else np.full(len(index), np.nan) for n in names}
    position_of = {t: i for i, t in enumerate(index)}
    xy = meta.loc[names, ['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    column = {n: i for i, n in enumerate(names)}

    def neighbour_covariates(sensor, g0, length, direction):
        i = column[sensor]
        near = [int(j) for j in np.argsort(distance[i]) if j != i][:NEIGHBOURS]
        past, future = {}, {}
        for c, j in enumerate(near):
            s = panel[names[j]].to_numpy(float)
            if direction == 'fwd':
                p, f = s[max(0, g0 - CHRONOS_CTX):g0], s[g0:g0 + length]
            else:
                p, f = s[g0 + length:g0 + length + CHRONOS_CTX][::-1], s[g0:g0 + length][::-1]
            if np.isfinite(p).any() and np.isfinite(f).any():
                past[f'nb{c}'] = p.astype(np.float32)
                future[f'nb{c}'] = f.astype(np.float32)
        return past, future

    def own_covariate(key, sensor, g0, length, direction):
        s = temp[sensor]
        gap = s[g0:g0 + length] if own == 'obs' else np.asarray(own[key], float)
        if direction == 'fwd':
            p, f = s[max(0, g0 - CHRONOS_CTX):g0], gap
        else:
            p, f = s[g0 + length:g0 + length + CHRONOS_CTX][::-1], gap[::-1]
        if np.isfinite(p).any() and np.isfinite(f).any():
            return {'own_T': p.astype(np.float32)}, {'own_T': f.astype(np.float32)}
        return {}, {}

    episodes = []
    for key in frame.episode.unique():
        sensor, start, length = key.rsplit('|', 2)
        episodes.append((key, sensor, position_of[pd.Timestamp(start)], int(length)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained(G.CHRONOS2_REPO, device_map=device, torch_dtype=torch.float32)
    median = int(np.argmin(np.abs(np.asarray(pipe.quantiles) - 0.5)))
    predictions = {}
    by_length = {}
    for key, sensor, g0, length in episodes:
        by_length.setdefault(length, []).append((key, sensor, g0))
    for length, group in sorted(by_length.items()):
        for direction in ('fwd', 'bwd'):
            items = []
            for key, sensor, g0 in group:
                series = panel[sensor].to_numpy(float)
                window = series[max(0, g0 - CHRONOS_CTX):g0] if direction == 'fwd' \
                    else series[g0 + length:g0 + length + CHRONOS_CTX][::-1]
                filled = pd.Series(window).interpolate(limit_direction='both').to_numpy()
                target = np.nan_to_num(filled).astype(np.float32)
                item = {'target': target}
                past, future = neighbour_covariates(sensor, g0, length, direction) if neighbours else ({}, {})
                if own is not None:
                    more_past, more_future = own_covariate(key, sensor, g0, length, direction)
                    past, future = {**past, **more_past}, {**future, **more_future}
                if past:
                    item['past_covariates'] = {k: v[-len(target):] for k, v in past.items()}
                    item['future_covariates'] = future
                items.append(item)
            traces = [None] * len(items)
            groups = {}
            for n, item in enumerate(items):
                groups.setdefault(tuple(sorted(item.get('past_covariates', {}))), []).append(n)
            for members in groups.values():
                for k in range(0, len(members), 64):
                    chunk = members[k:k + 64]
                    got = pipe.predict([items[n] for n in chunk], prediction_length=length)
                    for n, o in zip(chunk, got):
                        a = o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                        traces[n] = a.reshape(-1, len(pipe.quantiles), a.shape[-1])[0, median]
            stacked = np.stack(traces)
            for row, (key, _, _) in enumerate(group):
                trace = stacked[row]
                predictions.setdefault(key, {})[direction] = trace if direction == 'fwd' else trace[::-1]
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = np.full(len(frame), np.nan)
    keys, offsets, spans = frame.episode.to_numpy(), frame.position.to_numpy(int) - 1, frame.gap_length.to_numpy(int)
    for i in range(len(frame)):
        tr = predictions.get(keys[i])
        if tr:
            k, L = offsets[i], spans[i]
            w = (L - k) / (L + 1)
            out[i] = w * tr['fwd'][k] + (1 - w) * tr['bwd'][k]
    return out


def gaps_stage(quick, workers):
    G = gap_module()
    eps_all = pd.read_csv(IN / 'episodes.csv.gz')
    weights = json.loads((IN / 'weights.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    g2 = []
    for cohort, eps in eps_all.groupby('cohort', sort=False):
        if quick:
            eps = eps.groupby('band').head(6)
        period = eps.period.iloc[0]
        G.PERIOD = period
        frame = episode_frame(eps)
        t0 = time.time()
        members, t_truth = temperature_members(G, frame, period, workers)
        t_fill, t_pick, members = temperature_fill(members, frame, eps, weights, period)
        log(f'{cohort}: temperature filled on {eps.episode.nunique()} episodes ({time.time() - t0:.0f} s)')
        fills = {key: t_fill[rows] for key, rows in frame.groupby('episode', sort=False).groups.items()}
        result = frame.copy()
        result['T_truth'], result['T_fill'], result['T_pick'] = t_truth, t_fill, t_pick
        for m in ('linear', 'chronos_bidir', 'chronos2_cov_bidir', 'gpr', 'ens_stack'):
            result[f'T_{m}'] = members[m]
        for arm, nb, own in (('c2_ref', False, None), ('c2cov_ref', True, None), ('c2_T_obs', False, 'obs'),
                             ('c2_T_fill', False, fills), ('c2cov_T_obs', True, 'obs'), ('c2cov_T_fill', True, fills)):
            t1 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result[arm] = c2_fill(G, 'oxygen', frame, nb, own)
            log(f'{cohort}: {arm} ({time.time() - t1:.0f} s)')
        result.to_csv(OUT / f'gaps_{cohort}.csv.gz', index=False)
        # G2: the temperature members, one episode at a time, with and without the sonde's
        # temperature inside the gap hidden
        rng = np.random.default_rng(54)
        pick = rng.choice(eps.episode.unique(), size=min(5, eps.episode.nunique()), replace=False)
        for key in pick:
            one = episode_frame(eps[eps.episode == key])
            sensor, start, length = key.rsplit('|', 2)
            a = pd.Timestamp(start)
            plain, _ = temperature_members(G, one, period, 1)
            G._hide = (sensor, a, a + pd.Timedelta(hours=int(length)))
            try:
                hidden, _ = temperature_members(G, one, period, 1)
            finally:
                G._hide = None
            diff = {m: float(np.nanmax(np.abs(plain[m] - hidden[m]))) if np.isfinite(plain[m]).any() else None
                    for m in plain}
            g2.append({'cohort': cohort, 'episode': key, 'max_abs_difference': diff})
        log(f'{cohort}: done ({time.time() - t0:.0f} s)')
    (OUT / 'g2.json').write_text(json.dumps(g2, indent=1) + '\n', encoding='utf-8')


def ffill_bfill(x):
    """v2_chronos2.covs, as the companion precompute applies it."""
    s = pd.Series(x).ffill().bfill().to_numpy(float)
    fill = np.nanmean(x) if np.isfinite(x).any() else 0.0
    return np.nan_to_num(s, nan=fill).astype(np.float32)


def forecasts_stage(quick):
    import torch
    from chronos import BaseChronosPipeline
    P = np.load(CACHE / 'panels.npz')
    O, T = np.load(CACHE / 'forecasts_oxygen.npz'), np.load(CACHE / 'forecasts_temperature.npz')
    ctx = 512
    names_o, names_t = [str(n) for n in P['names_oxygen']], [str(n) for n in P['names_temperature']]
    iss = json.loads((IN / 'issuances.json').read_text(encoding='utf-8'))
    net = {'net_T': P['net_T'], 'net_sd': P['net_sd']}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map=device, torch_dtype=torch.float32)
    median = int(np.argmin(np.abs(np.asarray(pipe.quantiles) - 0.5)))
    arms = ('c2', 'c2_T', 'c2_T_fc', 'c2_T_obs', 'c2net', 'c2net_T_fc', 'c2net_T_obs')
    out = {}
    for s in names_o:
        if f'{s}|t' not in O.files or not iss.get(s) or s not in names_t:
            continue
        rows_o = {int(t): r for r, t in enumerate(O[f'{s}|t'])}
        rows_t = {int(t): r for r, t in enumerate(T[f'{s}|t'])} if f'{s}|t' in T.files else {}
        keep = [p for p in iss[s] if p in rows_o]
        if quick:
            keep = keep[:12]
        y = P['values_oxygen'][:, names_o.index(s)].astype(float)
        own = P['values_temperature'][:, names_t.index(s)].astype(float)
        items = {a: [] for a in arms}
        ok = {a: [] for a in arms}
        for p in keep:
            a0 = max(0, p - ctx + 1)
            target = y[a0:p + 1].astype(np.float32)
            n = len(target)
            past_own = {'own_T': ffill_bfill(own[a0:p + 1])[-n:]}
            past_net = {k: ffill_bfill(x[a0:p + 1])[-n:] for k, x in net.items()} | past_own
            fc = T[f'{s}|c2_net'][rows_t[p]].astype(np.float32) if p in rows_t else None
            ahead = own[p + 1:p + 1 + LEADS]
            ahead = np.r_[ahead, np.full(LEADS - len(ahead), np.nan)].astype(np.float32)
            obs_ok = bool(np.isfinite(ahead).any())
            for arm, past, future, good in (
                    ('c2', None, None, True), ('c2_T', past_own, None, True),
                    ('c2_T_fc', past_own, {'own_T': fc} if fc is not None else None, fc is not None),
                    ('c2_T_obs', past_own, {'own_T': ahead}, obs_ok),
                    ('c2net', past_net, None, True),
                    ('c2net_T_fc', past_net, {'own_T': fc} if fc is not None else None, fc is not None),
                    ('c2net_T_obs', past_net, {'own_T': ahead}, obs_ok)):
                item = {'target': target}
                if past:
                    item['past_covariates'] = past
                if future and good:
                    item['future_covariates'] = future
                items[arm].append(item)
                ok[arm].append(good)
        out[f'{s}|t'] = np.array(keep, np.int32)
        for arm in arms:
            traces = [None] * len(items[arm])
            groups = {}
            for n, it in enumerate(items[arm]):
                groups.setdefault((tuple(sorted(it.get('past_covariates', {}))),
                                   tuple(sorted(it.get('future_covariates', {})))), []).append(n)
            for members in groups.values():
                for k in range(0, len(members), 128):
                    chunk = members[k:k + 128]
                    got = pipe.predict([items[arm][n] for n in chunk], prediction_length=LEADS)
                    for n, o in zip(chunk, got):
                        a = o.detach().float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                        traces[n] = a.reshape(-1, len(pipe.quantiles), a.shape[-1])[0, median]
            arr = np.stack(traces).astype(np.float32) if traces else np.zeros((0, LEADS), np.float32)
            arr[~np.array(ok[arm], bool)] = np.nan
            out[f'{s}|{arm}'] = arr
        # G3: the two arms the precompute ran
        rows = [rows_o[p] for p in keep]
        out[f'{s}|g3_c2'] = np.array([np.nanmax(np.abs(out[f'{s}|c2'] - O[f'{s}|c2'][rows]))], np.float32)
        out[f'{s}|g3_c2net'] = np.array([np.nanmax(np.abs(out[f'{s}|c2net'] - O[f'{s}|c2_net'][rows]))], np.float32)
        log(f'forecasts {s}: {len(keep)} issuances; G3 c2 {float(out[f"{s}|g3_c2"][0]):.2e}, '
            f'c2net {float(out[f"{s}|g3_c2net"][0]):.2e}')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'forecasts.npz', **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='gaps,forecasts')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('V2_WORKERS', '8')))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('V2_THREADS', '3')))
    stages = a.stages.split(',')
    if 'forecasts' in stages:
        forecasts_stage(a.quick)
    if 'gaps' in stages:
        gaps_stage(a.quick, a.workers)
    log('done')


if __name__ == '__main__':
    main()
