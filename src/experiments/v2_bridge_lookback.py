"""Addendum 46: the anchored continuation with more of the silent probe's own history.

The continuation of addendum 39 reads the silent probe at two hours only: its reading at onset and
its change over the six hours before. This module gives it the probe's own record over the 24 h or
72 h ending at onset, and changes nothing else: the same episodes, climatologies, neighbours,
weather, row subsamples, nested studies and scoring, through v2_bridge unchanged. The added
features, all read at or before onset:

  24 h   the change over the last 12, 18 and 24 h (reading at onset minus reading k hours before),
         the departure of the onset reading from the probe's own 24 h mean, and the 24 h range
  72 h   the 24 h set, plus the change over 48 and 72 h, the departure from the 72 h mean and the
         72 h range

A lagged change is missing where the probe did not report at that hour; a mean or range is missing
unless at least half of its window was observed.

Usage:
  python -m src.experiments.v2_bridge_lookback --gate                    G3 (window 6 reproduces the
                                                                          published continuation) and G4
  python -m src.experiments.v2_bridge_lookback --window 24 --variable temperature [--quick]
  python -m src.experiments.v2_bridge_lookback --window 24 --variable oxygen --seed 7
                                          the decision rule's follow-up under a further sampler seed
Declared in config/bay_reserved_protocol_v2_addendum46.json.
"""
import argparse
import json
import zlib

import numpy as np
import pandas as pd

from src.experiments import v2_bridge, v2_horizon
from src.experiments.v2_closed_loop import STEP, episodes
from src.tuning import protocol_v2

PUBLISHED = v2_bridge.RUNS / 'v2_bridge'
OUT = v2_bridge.RUNS / 'v2_lookback'
WINDOWS = (24, 72)
SEEDS = (7, 123)
KEY = ['cohort', 'fold', 'sensor', 'onset', 'lead']
TOLERANCE = 1e-9


def lags(window):
    return [k for k in (12, 18, 24, 48, 72) if k <= window]


def spans(window):
    return [w for w in (24, 72) if w <= window]


def own_history(obs, rows, window):
    """The added columns for rows (i0, h0, s, h, hT); only obs[h0 - k, s] with k >= 0 is read."""
    cols = {f'own_change_{k}': [] for k in lags(window)}
    for w in spans(window):
        cols[f'own_departure_{w}'] = []
        cols[f'own_range_{w}'] = []
    for i0, h0, s, h, hT in rows:
        y0 = obs[h0, s]
        for k in lags(window):
            past = obs[h0 - k, s] if h0 - k >= 0 else np.nan
            cols[f'own_change_{k}'].append(y0 - past if np.isfinite(past) else np.nan)
        for w in spans(window):
            seg = obs[max(0, h0 - w + 1):h0 + 1, s]
            seg = seg[np.isfinite(seg)]
            ok = len(seg) >= w / 2
            cols[f'own_departure_{w}'].append(y0 - seg.mean() if ok else np.nan)
            cols[f'own_range_{w}'].append(seg.max() - seg.min() if ok else np.nan)
    return pd.DataFrame(cols)


class LookbackContext(v2_bridge.Context):
    """v2_bridge's context, reading the silent probe's own record over `window` hours before onset."""

    def __init__(self, variable, cohort, window):
        super().__init__(variable, cohort)
        self.window = int(window)

    def features(self, rows, clim):
        X, y, anchor = super().features(rows, clim)
        if self.window <= 6:
            return X, y, anchor
        extra = own_history(self.obs, rows, self.window)
        return pd.concat([X, extra.set_index(X.index)], axis=1), y, anchor


def dev_folds(ctx):
    """Bridge-Dev's folds and exactly the published episodes, as v2_bridge.run builds them."""
    closed = pd.read_csv(v2_bridge.CLOSED_LOOP / f'closed_loop_{ctx.variable}.csv.gz', usecols=['sensor', 'time'],
                         parse_dates=['time']).drop_duplicates()
    keep = set(zip(closed.sensor, closed.time))
    return [(fold, [(i0, s) for i0, s in episodes(ctx.data, fold, v2_bridge.MAX_LEAD)
                    if (ctx.names[s], ctx.data['times'][i0]) in keep])
            for fold in v2_horizon.folds(ctx.data, STEP)]


def refit(ctx, fold, eps, label, params):
    """Fit `params` on the outer training window exactly as select_and_fit does, and score."""
    lookup = v2_bridge.climatology_lookup(ctx.data, fold)
    first = ctx.data['times'][fold['va'].min()]
    rows = v2_bridge.subsample(v2_bridge.training_rows(ctx, fold['tr'], fold['eligible'], first),
                               zlib.crc32(f'{label}|outer'.encode()))
    model = v2_bridge.fit(ctx, rows, lookup, params)
    return v2_bridge.score_rows(ctx, fold, eps, model, lookup)


def published(variable):
    frame = pd.read_csv(PUBLISHED / f'bridge_{variable}.csv.gz', parse_dates=['onset'])
    frame['onset'] = pd.to_datetime(frame.onset, utc=True)
    return frame


def gate():
    """G3: window 6 builds the published feature matrix and reproduces the published predictions.
    G4: the added features do not change when every reading of the silent probe after onset is
    removed."""
    report = {}
    for variable in ('temperature', 'oxygen'):
        frames = []
        dctx = LookbackContext(variable, 'dev', 6)
        for fold, eps in dev_folds(dctx):
            label = f'fold{fold["number"]}'
            sel = json.loads((PUBLISHED / 'studies' / f'bridge_{variable}_{label}_selection.json').read_text())
            frames.append(refit(dctx, fold, eps, label, sel['params']).assign(cohort='Bridge-Dev'))
        rctx = LookbackContext(variable, 'reserved', 6)
        rfold = v2_bridge.fold_2026(rctx.data)
        sel = json.loads((PUBLISHED / 'studies' / f'bridge_{variable}_reserved2026_selection.json').read_text())
        frames.append(refit(rctx, rfold, episodes(rctx.data, rfold, v2_bridge.MAX_LEAD), 'reserved2026',
                            sel['params']).assign(cohort='Bridge-2026'))
        mine = pd.concat(frames, ignore_index=True)
        mine['onset'] = pd.to_datetime(mine.onset, utc=True)
        joined = mine.merge(published(variable), on=KEY, how='outer', suffixes=('', '_pub'), indicator=True)
        diff = float((joined.anchored_direct - joined.anchored_direct_pub).abs().max())
        g3 = {'unmatched': int((joined._merge != 'both').sum()), 'anchored_direct_max_abs_diff': diff}
        g3['passed'] = g3['unmatched'] == 0 and diff <= TOLERANCE
        # G4 on the 72 h set, which contains the 24 h one: remove every reading after each onset
        rows = v2_bridge.episode_rows(rctx, episodes(rctx.data, rfold, v2_bridge.MAX_LEAD))[:4000]
        full = own_history(rctx.obs, rows, 72)
        same = True
        for h0 in sorted({r[1] for r in rows})[:300]:
            masked = rctx.obs.copy()
            masked[h0 + 1:] = np.nan
            sub = [r for r in rows if r[1] == h0]
            got = own_history(masked, sub, 72)
            ref = full.iloc[[i for i, r in enumerate(rows) if r[1] == h0]].reset_index(drop=True)
            same &= bool(np.allclose(got.to_numpy(float), ref.to_numpy(float), equal_nan=True, atol=0))
        report[variable] = {'G3': g3, 'G4': {'onsets_checked': min(300, len({r[1] for r in rows})), 'passed': same}}
        print(variable, report[variable], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gate_G3_G4.json').write_text(json.dumps(report, indent=2))
    if not all(r['G3']['passed'] and r['G4']['passed'] for r in report.values()):
        raise SystemExit('G3 or G4 failed; no look-back arm may be reported')


def run(variable, window, quick=False, seed=None):
    """One look-back arm: a nested study per development fold and for the 2026 selection. With a
    seed, the decision rule's follow-up (addendum 45 part B's procedure): Bridge-Dev only, the
    sampler seed set, the studies written under their own folder."""
    name = f'h{window}' if seed is None else f'h{window}_seed{seed}'
    folder = OUT / ('smoke' if quick else '') / name
    v2_bridge.OUT = folder                  # select_and_fit writes its studies under v2_bridge.OUT
    if seed is not None:
        protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    frames, studies = [], {}
    dctx = LookbackContext(variable, 'dev', window)
    for fold, eps in dev_folds(dctx):
        if quick:
            eps = eps[:40]
        label = f'fold{fold["number"]}'
        model, lookup, summary, counts = v2_bridge.select_and_fit(dctx, fold, label, quick)
        studies[label] = {**summary, **counts}
        frames.append(v2_bridge.score_rows(dctx, fold, eps, model, lookup).assign(cohort='Bridge-Dev'))
    if seed is not None:
        write(frames, studies, folder, variable, window)
        return
    rctx = LookbackContext(variable, 'reserved', window)
    rfold = v2_bridge.fold_2026(rctx.data)
    reps = episodes(rctx.data, rfold, v2_bridge.MAX_LEAD)
    if quick:
        reps = reps[:40]
    model, lookup, summary, counts = v2_bridge.select_and_fit(rctx, rfold, 'reserved2026', quick)
    studies['reserved2026'] = {**summary, **counts}
    frames.append(v2_bridge.score_rows(rctx, rfold, reps, model, lookup).assign(cohort='Bridge-2026'))
    write(frames, studies, folder, variable, window)


def write(frames, studies, folder, variable, window):
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True).rename(columns={'anchored_direct': f'anchored_h{window}'})
    frame.to_csv(folder / f'bridge_{variable}.csv.gz', index=False)
    (folder / f'bridge_{variable}_studies.json').write_text(json.dumps(studies, indent=2, default=str))
    print(f'{variable} {folder.name}: {len(frame)} rows written to {folder}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gate', action='store_true')
    p.add_argument('--window', type=int, choices=WINDOWS)
    p.add_argument('--variable', choices=('temperature', 'oxygen'))
    p.add_argument('--seed', type=int, choices=SEEDS, help='the decision rule\'s follow-up, Bridge-Dev only')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.gate:
        gate()
    else:
        run(a.variable, a.window, a.quick, a.seed)
