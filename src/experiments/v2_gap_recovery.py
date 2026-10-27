"""Addendum-30: recovering a masked gap, by regression, by rolling forecast, and by both.

The feasibility probe established that 46 % of lost temperature hours and 11 % of lost oxygen hours
sit in gaps short enough for a parameter-free rule to beat the measurement floor, and that the rule
which wins changes with gap length. This fits the arms that could do better.

The organising distinction is what information the use case admits, not which model is used:

    causal          at hour t inside the gap: the target probe up to the gap's start, and any OTHER
                    probe up to t. This is what a live dashboard has, since the neighbours are
                    still reporting while this one is dark.
    retrospective   additionally the target probe after the gap ends. This is what repairing an
                    archive has, and it is the setting the 2024 filled datasets were produced in.

Arms, all scored on identical rows:

    persistence, linear, climatology, nearest_probe   parameter-free, recomputed here so the
                                                      comparison is on the same episodes
    regression_causal / regression_retro              direct per-hour prediction, tuned
    rolling_forward / rolling_backward                recursive one-step prediction from an edge
    hybrid                                            the two rollouts blended by a tuned profile
    tabpfn / tabicl                                   in-context tabular models, retrospective
    anchored_gbm[_rel][_exo]                          addendum 34: a tree predicting the
                                                      departure from linear interpolation, with
                                                      level or edge-relative features, with or
                                                      without the Santa Teresa weather and the Magra
    arx_bridge[_exo]                                  addendum 34: regression with AR(1) errors,
                                                      Kalman-smoothed between the two edges

Real gaps are never scored: they have no ground truth. Observed stretches are masked at the lengths
the record actually produces and recovered against what was really there.

Declared in config/bay_reserved_protocol_v2_addendum30.json.
Usage: python -m src.experiments.v2_gap_recovery --variable temperature --arm regression_retro
"""
import argparse
import hashlib
import json
import os
import traceback

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT, load_observations
from src.experiments.bay_graph_dynamics import save
from src.tuning.protocol_v2 import PROTOCOL, run_study

#: Addendum 31 reads the same machinery on a second period and a second seed. Both default to the
#: addendum-30 values, and at those defaults `build` is verified to reproduce the published episodes
#: exactly, so the development artifacts are not disturbed by the switch existing.
PERIOD = os.environ.get('V2_GAP_PERIOD', 'development')
if PERIOD not in ('development', 'reserved'):
    raise ValueError(f'V2_GAP_PERIOD must be development or reserved, not {PERIOD!r}')
_seed_override = os.environ.get('V2_GAP_SEED')
#: Addendum 32. `strict` is the addendum-30 design: a gap is masked only inside a fully observed
#: stretch, at randomly sampled starts. `relaxed` lets the masked interior contain real short gaps,
#: scores only the hours that were observed, and tiles windows deterministically so that gaps at one
#: length never share an hour. Past a week the strict rule leaves almost nothing to test; no filled
#: value is scored under either.
DESIGN = os.environ.get('V2_GAP_DESIGN', 'strict')
if DESIGN not in ('strict', 'relaxed'):
    raise ValueError(f'V2_GAP_DESIGN must be strict or relaxed, not {DESIGN!r}')
#: relaxed eligibility: observed edges, and at least this share observed in each margin and inside
RELAXED_MIN_OBSERVED = 0.9
_suffix = (('' if DESIGN == 'strict' else '_long')
           + ('' if PERIOD == 'development' else '_reserved2026')
           + ('' if _seed_override is None else f'_seed{_seed_override}'))
OUT = ROOT / f'reports/bay_twin/runs/v2_gap_recovery{_suffix}'


def tiled_starts(ok, length):
    """Addendum-32 placement: window starts, scanning forward, with gaps that never share an hour.

    A window is MARGIN + length + MARGIN hours. It is eligible when both edge hours are observed and
    each margin and the interior are at least RELAXED_MIN_OBSERVED observed. The first eligible
    window is taken and the next may not start its gap before this one's ends. Deterministic.
    """
    n, need = len(ok), MARGIN + length + MARGIN
    if n < need:
        return []
    cum = np.concatenate([[0], np.cumsum(ok)])
    starts, s = [], 0
    while s + need <= n:
        g0, g1 = s + MARGIN, s + MARGIN + length
        share = lambda a, b: (cum[b] - cum[a]) / (b - a)
        if (ok[g0 - 1] and ok[g1] and share(s, g0) >= RELAXED_MIN_OBSERVED
                and share(g1, g1 + MARGIN) >= RELAXED_MIN_OBSERVED
                and share(g0, g1) >= RELAXED_MIN_OBSERVED):
            starts.append(s)
            s = g1
        else:
            s += 1
    return starts


def observations(period=None):
    """The record this run is allowed to see.

    `load_observations` stops at 2026-01-01 by design and is not changed. The reserved period is
    read here instead, through the same export reader and the same screening, keeping only 2026.
    `period` defaults to the run's own; addendum 34's frozen 2026 read also needs the development
    record, to fit on.
    """
    if (period or PERIOD) == 'development':
        return load_observations()
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    frames = {}
    for name in meta.index:
        _, frame, _ = read_export(ROOT / 'data/raw' / (name + '.txt'), include_auxiliary=True)
        frames[name] = frame[frame.index >= pd.Timestamp('2026-01-01', tz='UTC')]
    return meta, frames
#: Out to sixty days, because that is where the lost hours are. 41 % of lost temperature hours and
#: 49 % of lost oxygen hours sit in gaps longer than a week, so a sweep that stops at 168 h leaves
#: the largest part of the problem untested. How far this can go is set by the record: masking a
#: gap of L hours needs L + 144 consecutive *observed* hours somewhere, and beyond 1440 h no probe
#: has that. The number of probes able to supply an episode falls with length (8 at a week, 3 at
#: sixty days for temperature; 2 at a week for oxygen, none past 720 h), so every long-gap row
#: describes a handful of instruments rather than the network, and the report carries that count.
LENGTHS = [int(x) for x in os.environ.get(
    'V2_GAP_LENGTHS', '1,2,3,6,12,18,24,48,72,168,336,504,720,1440').split(',')]
MARGIN = 72               # observed hours required either side of a masked stretch
PER_LENGTH_PER_PROBE = 60
#: Episodes are cheaper than rows: a 1440 h episode would contribute 1440 rows on its own. Long
#: gaps therefore get fewer episodes, and each is scored at a sample of positions rather than at
#: every hour, which preserves the error-against-position shape at a fraction of the size.
MAX_ROWS_PER_EPISODE = 48
NEIGHBOURS = 4            # nearest probes whose contemporaneous readings enter the features
FOLDS = 4                 # chronological blocks of episodes
SEED = int(_seed_override or 20260922)
TABPFN_TRAIN_CAP = 8000   # in-context models take the training set as a prompt
#: Trinity is shared. LightGBM sizes its own thread pool from `n_jobs`, which defaults to every
#: core and ignores OMP_NUM_THREADS, so ten jobs each took 61 threads and drove the load average
#: to 524 on a 112-core machine. Every estimator built here is given this explicitly.
THREADS = int(os.environ.get('V2_THREADS', '3'))

ARMS = ['regression_causal', 'regression_retro', 'rolling_forward', 'rolling_backward',
        'hybrid', 'tabpfn', 'tabicl', 'chronos_forward', 'chronos_bidir',
        'polynomial', 'gpr', 'compressed_sensing', 'lstm',
        # addendum 33: variations on the best performers
        'chronos2_bidir', 'chronos2_cov_bidir', 'gpr_k8',
        # addendum 34: regressions anchored to the edges, with and without exogenous series
        'anchored_gbm', 'anchored_gbm_exo', 'arx_bridge', 'arx_bridge_exo',
        # addendum 34 amendment 1: the same anchored target with edge-relative features only
        'anchored_gbm_rel', 'anchored_gbm_rel_exo',
        # addendum 35: Chronos with exogenous covariates, and Chronos on a covariate residual
        'chronos2_exo_bidir', 'chronos2_all_bidir', 'bolt_resid_bidir',
        'chronos2_cov_resid_bidir',
        # addendum 35 amendment 1: the same residual scheme with a linear covariate regressor
        'bolt_resid_lin_bidir', 'chronos2_cov_resid_lin_bidir']

#: Addendum 35. Chronos-2 reading the exogenous series in context, and Chronos filling what a
#: covariate regressor leaves, as AutoGluon-TimeSeries does with its covariate regressor.
A35_COVARIATE_ARMS = ('chronos2_exo_bidir', 'chronos2_all_bidir')
A35_LINEAR_ARMS = ('bolt_resid_lin_bidir', 'chronos2_cov_resid_lin_bidir')
A35_RESIDUAL_ARMS = ('bolt_resid_bidir', 'chronos2_cov_resid_bidir') + A35_LINEAR_ARMS
A35_ARMS = A35_COVARIATE_ARMS + A35_RESIDUAL_ARMS

#: Addendum 34. Every one of these is tuned, none reads the target sonde's other channels, and none
#: may train on a probe-hour within MARGIN of a gap it is scored on (see `window_mask`).
REL_ARMS = ('anchored_gbm_rel', 'anchored_gbm_rel_exo')
ANCHORED_ARMS = ('anchored_gbm', 'anchored_gbm_exo') + REL_ARMS
ARX_ARMS = ('arx_bridge', 'arx_bridge_exo')
A34_ARMS = ANCHORED_ARMS + ARX_ARMS
EXO_ARMS = ('anchored_gbm_exo', 'arx_bridge_exo', 'anchored_gbm_rel_exo')
#: the Santa Teresa meteorological station beside the Baia EXO2 sonde, and the Magra at Ameglia
EXO_CHANNELS = ('air', 'solar', 'wind', 'humidity', 'pressure', 'river')
#: A frozen 2026 read trains on the matching development cohort exactly as published: Gap-Dev at
#: its fourteen lengths, Gap-Long-Dev at its five.
TRAIN_LENGTHS = {'strict': [1, 2, 3, 6, 12, 18, 24, 48, 72, 168, 336, 504, 720, 1440],
                 'relaxed': [168, 336, 504, 720, 1440]}

#: Arms that work from the raw series of their own episode rather than from the feature table.
SERIES_ARMS = ('chronos_forward', 'chronos_bidir', 'compressed_sensing', 'gpr', 'lstm',
               'chronos2_bidir', 'chronos2_cov_bidir', 'gpr_k8')

CHRONOS_REPO = 'amazon/chronos-bolt-base'   # as addendum 23 used it
CHRONOS2_REPO = 'amazon/chronos-2'          # addendum 33, as addendum 1 used it
CHRONOS_CTX = 512                           # hours of history fed to the model
IN_CONTEXT_ARMS = ('tabpfn', 'tabicl', 'chronos_forward', 'chronos_bidir', 'chronos2_bidir',
                   'chronos2_cov_bidir')

#: Feature blocks. The causal arm is denied every column that looks past the gap's start at the
#: target probe; neighbours are contemporaneous and stay, because a live system still receives them.
#: It keeps `gap_length`, and with `position` that gives the distance to the far edge, which a live
#: system does not know; the "causal" arms are therefore one-edge arms, not causal ones, and the
#: thesis labels them so. They were not rerun: both lose to linear by a wide
#: margin and no conclusion rests on them.
RETROSPECTIVE_ONLY = ('after_edge', 'to_far_edge', 'linear', 'span_delta', 'back_roll',
                      'elapsed_bwd', 'anchor_bwd')

#: `anchor_fwd` and `anchor_bwd` hold the true value at the neighbouring scored position. They are
#: the recursion's state, supplied by the model's own prediction at inference and by the truth
#: during training, which is teacher forcing. Handing them to a direct regressor would be handing
#: it the answer at the adjacent hour, so they are withheld from every non-recursive arm.
RECURSION_STATE = ('anchor_fwd', 'anchor_bwd')


def nearest_others(distance, i, k=None):
    """Probes other than `i`, nearest first.

    `argsort(distance[i])[1:]` assumes the probe itself sorts first. Six probe pairs share
    coordinates, and with a tie at zero the sort can place the twin first and the probe second, so
    slicing off position 0 dropped the twin and kept the probe: its own unmasked reading became
    neighbour 0, and the nearest-probe fill returned the truth. Excluding `i` by index, with a
    stable sort, cannot do that.
    """
    others = [int(j) for j in np.argsort(distance[i], kind='stable') if j != i]
    return others if k is None else others[:k]


def build(variable, rng, quick=False, period=None, lengths=None):
    """One row per hour inside a masked stretch, with everything any arm might use.

    `period` and `lengths` default to the run's own; addendum 34 passes them to rebuild the
    matching development cohort as the training set of a frozen 2026 read.
    """
    meta, frames = observations(period)
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    meta = meta.loc[names]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    data = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    values = data.to_numpy(float)
    xy = meta[['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    order = {i: nearest_others(distance, i, NEIGHBOURS) for i in range(len(names))}
    twin = {i: bool((np.sort(distance[i])[1] if len(names) > 1 else 1e9) < 1.0)
            for i in range(len(names))}
    hour, month = index.hour.to_numpy(), index.month.to_numpy()
    doy = index.dayofyear.to_numpy()

    # Month-by-hour sums and counts per probe, so that each episode can take its own masked hours
    # out of the mean. Addendum 30 declares the climatology as fitted outside the masked stretch. A
    # mean over the whole record contains the stretch being scored, and on a month-long gap at a
    # probe with one July on record that is most of the cell: at 720 h on oxygen it scored 8.2
    # umol/l in-sample against 13.5 honestly.
    clim_sum = np.zeros((len(names), 13, 24))
    clim_n = np.zeros((len(names), 13, 24))
    for i in range(len(names)):
        ok = np.isfinite(values[:, i])
        np.add.at(clim_sum[i], (month[ok], hour[ok]), values[ok, i])
        np.add.at(clim_n[i], (month[ok], hour[ok]), 1)

    if lengths is None:
        lengths = LENGTHS[:4] if quick else LENGTHS
    rows = []
    contributing = {}
    for length in lengths:
        need = MARGIN + length + MARGIN
        # Long gaps are rarer and heavier, so take fewer of them.
        per = 6 if quick else (PER_LENGTH_PER_PROBE if length <= 24
                               else 20 if length <= 168 else 8)
        probes_here = 0
        for i, name in enumerate(names):
            ok = np.isfinite(values[:, i])
            cum = np.concatenate([[0], np.cumsum(ok)])
            if len(ok) < need:
                continue
            if DESIGN == 'strict':
                full = (cum[need:] - cum[:-need]) == need
                starts = np.flatnonzero(full)
                if not len(starts):
                    continue
                chosen = rng.choice(starts, size=min(per, len(starts)), replace=False)
            else:
                chosen = tiled_starts(ok, length)
                if not chosen:
                    continue
            probes_here += 1
            # Positions scored inside the gap. Short gaps get every hour; long ones get an evenly
            # spread sample, so the first and last hours and the middle are all represented.
            spread = (np.arange(length) if length <= MAX_ROWS_PER_EPISODE
                      else np.unique(np.linspace(0, length - 1, MAX_ROWS_PER_EPISODE).astype(int)))
            for s in chosen:
                g0 = int(s) + MARGIN
                # Under the relaxed design the interior may hold real short gaps; only hours that
                # were observed are scored, so no filled value is ever a target. Under the strict
                # design every hour is observed and this keeps them all.
                offsets = spread[np.isfinite(values[g0 + spread, i])]
                near, far = values[g0 - 1, i], values[g0 + length, i]
                history = values[g0 - MARGIN:g0, i]
                future = values[g0 + length:g0 + length + MARGIN, i]
                span = slice(g0, g0 + length)
                seen = np.isfinite(values[span, i])
                gap_sum, gap_n = np.zeros((13, 24)), np.zeros((13, 24))
                np.add.at(gap_sum, (month[span][seen], hour[span][seen]), values[span, i][seen])
                np.add.at(gap_n, (month[span][seen], hour[span][seen]), 1)
                outside = clim_n[i] - gap_n
                with np.errstate(invalid='ignore', divide='ignore'):
                    clim = np.where(outside > 0, (clim_sum[i] - gap_sum) / outside, np.nan)
                # The recursive arms step from one scored position to the next, and once positions
                # are sampled those are not one hour apart. Each row therefore carries how far it
                # is from the previous scored position and the true value there, so the recursion
                # predicts a change over the interval it actually spans rather than over an hour.
                previous = {int(k): (int(offsets[j - 1]) if j else -1)
                            for j, k in enumerate(offsets)}
                following = {int(k): (int(offsets[j + 1]) if j + 1 < len(offsets) else length)
                             for j, k in enumerate(offsets)}
                for k in offsets:
                    k = int(k)
                    t = g0 + k
                    prev_k, next_k = previous[k], following[k]
                    row = {
                        'sensor': name, 'sensor_index': i, 'has_twin': twin[i],
                        'gap_length': length, 'position': k + 1,
                        'to_near_edge': k + 1, 'to_far_edge': length - k,
                        'episode': f'{name}|{index[g0].isoformat()}|{length}',
                        'time': index[t], 'truth': values[t, i],
                        'near_edge': near, 'after_edge': far,
                        'span_delta': far - near,
                        'elapsed_fwd': k - prev_k,
                        'elapsed_bwd': next_k - k,
                        'anchor_fwd': near if prev_k < 0 else values[g0 + prev_k, i],
                        'anchor_bwd': far if next_k >= length else values[g0 + next_k, i],
                        'linear': near + (far - near) * (k + 1) / (length + 1),
                        'spline': np.nan,          # filled per episode below, needs all knots
                        'persistence': near,
                        'climatology': float(clim[month[t], hour[t]]),
                        'trend_before': history[-1] - history[-7] if len(history) >= 7 else np.nan,
                        'mean_before_24': np.nanmean(history[-24:]),
                        'sd_before_24': np.nanstd(history[-24:]),
                        'mean_after_24': np.nanmean(future[:24]),
                        'sin_h': np.sin(2 * np.pi * hour[t] / 24),
                        'cos_h': np.cos(2 * np.pi * hour[t] / 24),
                        'sin_d': np.sin(2 * np.pi * doy[t] / 365.25),
                        'cos_d': np.cos(2 * np.pi * doy[t] / 365.25),
                    }
                    for j, other in enumerate(order[i]):
                        now = values[t, other]
                        ref = values[g0 - 1, other]
                        # Offset-corrected: the neighbour's level differs from this probe's, so
                        # what transfers is its change since the gap opened, not its value.
                        row[f'nb{j}_offset'] = now - ref + near if np.isfinite(ref) else np.nan
                        row[f'nb{j}_raw'] = now
                        row[f'nb{j}_ok'] = float(np.isfinite(now))
                        row[f'nb{j}_dist'] = distance[i, other]
                    rows.append(row)
        contributing[length] = probes_here
    frame = pd.DataFrame(rows)
    frame['nearest_probe'] = frame['nb0_offset'].where(frame['nb0_ok'] > 0, frame['persistence'])
    frame['spline'] = spline_fill(frame, values, names, index)
    frame['pchip'] = spline_fill(frame, values, names, index, shape_preserving=True)
    frame['nearest_live_probe'] = nearest_live_fill(frame, values, names, index, distance)

    # Folds partition the CALENDAR, not the list of episodes. Ranking episodes by start time looks
    # chronological and is not: a 720 h episode spans a month, so episodes with different start
    # ranks cover the same hours, and the same (sensor, hour) truth then sits in a training fold
    # and a scored fold at once. Measured on the first run that was 38 % of cells at 504 h on
    # oxygen, where the in-context arms take the training pool as their prompt and can read the
    # answer off it. Partitioning time makes each hour belong to exactly one fold by construction;
    # episodes straddling a boundary belong to no fold and are dropped.
    edges = pd.date_range(frame.time.min(), frame.time.max(), periods=FOLDS + 1)
    span = frame.groupby('episode').time.agg(['min', 'max'])
    block = pd.Series(np.searchsorted(edges[1:-1], span['min'].to_numpy(), side='right'),
                      index=span.index)
    end_block = pd.Series(np.searchsorted(edges[1:-1], span['max'].to_numpy(), side='right'),
                          index=span.index)
    contained = block == end_block
    if DESIGN == 'relaxed':
        # Nothing in addendum 32 is fitted across episodes, so no hour can reach a training pool;
        # a straddling episode is kept and assigned to the block its gap starts in.
        contained = pd.Series(True, index=block.index)
    frame['fold'] = frame.episode.map(block.where(contained))
    dropped = int((~contained).sum())
    frame = frame[frame.fold.notna()].copy()
    frame['fold'] = frame['fold'].astype(int)
    frame.attrs['episodes_straddling_a_fold_boundary'] = dropped
    frame.attrs['probes_contributing'] = contributing
    return frame


def episode_parts(frame, variable):
    """(key, sensor index, gap start, length, whole series) for every episode in `frame`.

    The series arms need the record around their own gap rather than a row of summary features, so
    they rebuild the panel and slice it. The episode key carries sensor and start time, so nothing
    has to be threaded through the feature table to get back to the raw values.
    """
    meta, frames = observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    panel = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    where = {t: i for i, t in enumerate(index)}
    out = []
    for key in frame.episode.unique():
        sensor, start, length = key.rsplit('|', 2)
        out.append((key, sensor, where[pd.Timestamp(start)], int(length)))
    return out, panel, index


def spline_fill(frame, values, names, index, shape_preserving=False):
    """Cubic spline through the observed context either side, evaluated inside the gap.

    A smoother reading of the same two-sided information linear interpolation uses: it matches the
    slope at each edge instead of only the level, so it should win wherever the series is curved
    across the gap and lose wherever extrapolating curvature is a mistake. `shape_preserving` gives
    PCHIP, which cannot overshoot between knots; `scripts/v2_gap_splines.py` computed it separately
    for addendum 30, and it is a column of its own from addendum 32 on.
    """
    from scipy.interpolate import CubicSpline, PchipInterpolator
    maker = PchipInterpolator if shape_preserving else CubicSpline
    column = {n: i for i, n in enumerate(names)}
    out = np.full(len(frame), np.nan)
    positions = {t: i for i, t in enumerate(index)}
    for key, block in frame.groupby('episode', sort=False):
        sensor, start, length = key.rsplit('|', 2)
        length = int(length)
        g0 = positions[pd.Timestamp(start)]
        i = column[sensor]
        # Knots: the observed margins either side. Anything inside the gap is hidden by design.
        left = np.arange(max(0, g0 - MARGIN), g0)
        right = np.arange(g0 + length, min(len(index), g0 + length + MARGIN))
        knots = np.concatenate([left, right])
        y = values[knots, i]
        ok = np.isfinite(y)
        if ok.sum() < 8:
            continue
        try:
            spline = maker(knots[ok], y[ok], extrapolate=False)
        except ValueError:
            continue
        rows = frame.index.get_indexer(block.index)
        target = g0 + block.position.to_numpy(int) - 1
        out[rows] = spline(target)
    return out


def nearest_live_fill(frame, values, names, index, distance):
    """The nearest probe that is actually reporting, offset-corrected, hour by hour.

    `nearest_probe` reads the single geometrically nearest probe and falls back to persistence when
    that probe is dark, which on probes without a twin is most rows. This takes, at each scored hour,
    the nearest probe observed both at the hour before the gap and at that hour, so a poor result
    means an uninformative neighbour rather than an offline one. Parameter-free.
    """
    column = {n: i for i, n in enumerate(names)}
    positions = {t: i for i, t in enumerate(index)}
    out = np.full(len(frame), np.nan)
    for key, block in frame.groupby('episode', sort=False):
        sensor, start, _ = key.rsplit('|', 2)
        i, g0 = column[sensor], positions[pd.Timestamp(start)]
        order = [j for j in np.argsort(distance[i]) if j != i]
        rows = frame.index.get_indexer(block.index)
        for r, t in zip(rows, g0 + block.position.to_numpy(int) - 1):
            for j in order:
                if np.isfinite(values[g0 - 1, j]) and np.isfinite(values[t, j]):
                    out[r] = values[t, j] - values[g0 - 1, j] + values[g0 - 1, i]
                    break
    return out


def compressed_sensing_fill(frame, variable):
    """Sparse recovery in a cosine basis: the gap is whatever is consistent with a sparse spectrum.

    A moored temperature or oxygen record is dominated by a diurnal cycle and a seasonal trend, so
    it is compressible in frequency even when it is not smooth in time. Taking the window around a
    gap, writing it as a cosine expansion and asking for the fewest coefficients that reproduce the
    *observed* margins is the classic compressed-sensing reading of a missing stretch, and it uses
    the two edges jointly rather than drawing a line between them.

    L1 is imposed by Lasso on the observed rows of the basis; the penalty is fixed rather than
    tuned so the arm stays parameter-light and comparable with the other reconstructors.
    """
    from sklearn.linear_model import Lasso
    from scipy.fft import idct
    episodes, panel, index = episode_parts(frame, variable)
    out = np.full(len(frame), np.nan)
    by_key = {k: g for k, g in frame.groupby('episode', sort=False)}
    for key, sensor, g0, length in episodes:
        series = panel[sensor].to_numpy(float)
        lo, hi = max(0, g0 - MARGIN), min(len(series), g0 + length + MARGIN)
        window = series[lo:hi]
        n = len(window)
        if n < 32:
            continue
        gap_lo, gap_hi = g0 - lo, g0 - lo + length
        mask = np.ones(n, bool)
        mask[gap_lo:gap_hi] = False
        mask &= np.isfinite(window)
        if mask.sum() < 16:
            continue
        # Orthonormal DCT-III basis, columns are cosines; a sparse c reproduces a smooth signal.
        basis = idct(np.eye(n), norm='ortho', axis=0)
        centre = window[mask].mean()
        model = Lasso(alpha=0.02, max_iter=4000, fit_intercept=False)
        model.fit(basis[mask], window[mask] - centre)
        recovered = basis @ model.coef_ + centre
        block = by_key[key]
        rows = frame.index.get_indexer(block.index)
        out[rows] = recovered[gap_lo + block.position.to_numpy(int) - 1]
    return out


#: Addendum 59 part A. The published process kept a neighbour with as few as 12 finite margin hours
#: and entered every hour at which it was silent, in the margin fit and in the gap, at its margin
#: mean (`np.nan_to_num` of a standardised value), so the fitted mean jumped whenever a neighbour
#: switched between reporting and silent. With this switch on, each pattern of reporting neighbours
#: among an episode's scored hours gets its own linear mean, fitted only where the target and every
#: neighbour of the pattern report (see `_gpr_by_pattern`). V2_GAP_GP_PATTERNS=0 restores the
#: published process exactly, which is how the replacement was gated before anything was rerun.
GP_BY_PATTERN = os.environ.get('V2_GAP_GP_PATTERNS', '1') != '0'
#: a pattern needs this many margin hours at which the target and all its neighbours report
GP_MIN_PATTERN_HOURS = 24
#: Which four probes are "the four nearest". The published process sorted distances with numpy's
#: default (unstable) sort, and co-located probes tie: on the 2026 temperature panel that picks the
#: other probe of a co-located pair on six of twelve probes, compared with `nearest_others` (stable).
#: Addendum 59 names `nearest_others`, which the corrected process uses by default;
#: V2_GAP_GP_NEIGHBOURS=published keeps the published tie-break, as a sensitivity.
GP_NEIGHBOUR_ORDER = os.environ.get('V2_GAP_GP_NEIGHBOURS', 'nearest_others')
if GP_NEIGHBOUR_ORDER not in ('nearest_others', 'published'):
    raise ValueError(f'V2_GAP_GP_NEIGHBOURS must be nearest_others or published, '
                     f'not {GP_NEIGHBOUR_ORDER!r}')


def _gp_kernel():
    from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel
    return (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e3))
            + ConstantKernel(0.5, (1e-3, 1e3)) * ExpSineSquared(2.0, 1.0,
                                                                periodicity_bounds='fixed')
            + WhiteKernel(0.1, (1e-3, 1e1)))


def _gp_temporal(t_fit, target, t_pred):
    """The published temporal process: fitted to `target` at `t_fit` (days), predicted at `t_pred`."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    gp = GaussianProcessRegressor(kernel=_gp_kernel(), normalize_y=False,
                                  n_restarts_optimizer=0, alpha=1e-6)
    gp.fit(t_fit, target)
    return gp.predict(t_pred)


def _gp_reduce(pattern, finite_ctx, hours_of, nb_ctx):
    """The neighbours a pattern keeps: drop the one with the fewest margin hours (the farther on a
    tie) until the target and every kept neighbour report together on GP_MIN_PATTERN_HOURS margin
    hours. A neighbour constant over those hours is dropped too, as the published process dropped
    one constant over its margin. Columns index the episode's neighbour list, nearest first."""
    kept = list(pattern)
    while kept:
        hours = finite_ctx[:, kept].all(axis=1)
        if hours.sum() < GP_MIN_PATTERN_HOURS:
            kept.remove(max(kept, key=lambda c: (-hours_of[c], c)))
            continue
        flat = [c for c in kept if float(nb_ctx[hours, c].std()) < 1e-8]
        if flat:
            kept.remove(max(flat))
            continue
        break
    return tuple(kept)


def _gpr_by_pattern(values, i, others, ctx_rows, y, rows_gap, g0):
    """Universal kriging with one linear mean per pattern of reporting neighbours.

    For each pattern of the neighbours `others` that report among the scored hours `rows_gap`, the
    linear mean is fitted by least squares on the margin hours at which the target and every
    neighbour of the (reduced, `_gp_reduce`) pattern report, each neighbour standardised on those
    hours, and the temporal process is fitted to that mean's residual on the same hours. The empty
    pattern is the published temporal process alone on every finite margin hour. Returns the
    prediction per scored hour (NaN where a fit failed), the neighbours used per hour, the margin
    hours each fit used, and the failures.
    """
    t_ctx = ((ctx_rows - g0) / 24.0)[:, None]
    t_gap = ((rows_gap - g0) / 24.0)[:, None]
    nb_ctx = values[np.ix_(ctx_rows, others)] if others else np.zeros((len(ctx_rows), 0))
    nb_gap = values[np.ix_(rows_gap, others)] if others else np.zeros((len(rows_gap), 0))
    finite_ctx, finite_gap = np.isfinite(nb_ctx), np.isfinite(nb_gap)
    hours_of = finite_ctx.sum(axis=0)
    pred = np.full(len(rows_gap), np.nan)
    used = np.zeros(len(rows_gap), int)
    hours_used = np.zeros(len(rows_gap), int)
    failures = []
    groups = {}
    for r, pattern in enumerate(map(tuple, finite_gap)):
        groups.setdefault(tuple(c for c, on in enumerate(pattern) if on), []).append(r)
    fits = {}
    for pattern, rows in groups.items():
        kept = _gp_reduce(pattern, finite_ctx, hours_of, nb_ctx)
        rows = np.asarray(rows)
        used[rows] = len(kept)
        if kept not in fits:
            try:
                if not kept:
                    centre = float(y.mean())
                    fits[kept] = ('empty', centre,
                                  _gp_temporal(t_ctx, y - centre, t_gap), len(y))
                else:
                    hours = finite_ctx[:, list(kept)].all(axis=1)
                    z = nb_ctx[np.ix_(hours, list(kept))]
                    mu, sd = z.mean(axis=0), z.std(axis=0)
                    design = np.hstack([(z - mu) / sd, np.ones((int(hours.sum()), 1))])
                    coef, *_ = np.linalg.lstsq(design, y[hours], rcond=None)
                    residual = y[hours] - design @ coef
                    fits[kept] = ('mean', (mu, sd, coef),
                                  _gp_temporal(t_ctx[hours], residual, t_gap), int(hours.sum()))
            except Exception as exc:                   # noqa: BLE001 - counted, not swallowed
                fits[kept] = None
                failures.append(f'{type(exc).__name__} on {len(kept)} neighbours')
        fit = fits[kept]
        if fit is None:
            continue
        kind, params, temporal, n_hours = fit
        hours_used[rows] = n_hours
        if kind == 'empty':
            pred[rows] = temporal[rows] + params
        else:
            mu, sd, coef = params
            z = (nb_gap[np.ix_(rows, list(kept))] - mu) / sd
            pred[rows] = temporal[rows] + z @ coef[:-1] + coef[-1]
    return pred, used, hours_used, failures


def gpr_fill(frame, variable, neighbours=NEIGHBOURS, by_pattern=None, diagnostics=None):
    """Spatio-temporal Gaussian process: time plus the contemporaneous neighbours as inputs.

    Fitted per episode on the observed margins, which keeps every covariance matrix small enough to
    invert exactly and means no information crosses between episodes; the hyperparameters and the
    neighbour weights are therefore re-estimated for each gap's own season. The model is universal
    kriging: a linear mean in the standardised readings of the four nearest probes, and a GP in time
    on what that mean leaves, with a smooth RBF, an explicitly periodic term at twenty-four hours
    and white noise. The process can lean on the diurnal cycle where the gap is short and on the
    neighbours where it is long. (An earlier form put the neighbours inside the kernel; the comment
    below says why that fails.)

    `by_pattern` (default GP_BY_PATTERN) fits the linear mean per pattern of reporting neighbours
    (addendum 59); False is the published process. A dict passed as `diagnostics` receives, per row
    of `frame`, how many of the episode's neighbours report at that hour, how many the fit used, on
    how many margin hours, and whether the published process entered a silent neighbour there.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel
    by_pattern = GP_BY_PATTERN if by_pattern is None else by_pattern
    episodes, panel, index = episode_parts(frame, variable)
    names = list(panel.columns)
    values = panel.to_numpy(float)
    column = {n: i for i, n in enumerate(names)}
    # The neighbours are the four nearest probes by distance, as the feature table's are. Until
    # addendum 32 this took the first four probes in panel order, which for most probes are not
    # near at all.
    xy = observations()[0].loc[names, ['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    out = np.full(len(frame), np.nan)
    by_key = {k: g for k, g in frame.groupby('episode', sort=False)}
    fitted, failures = 0, []
    diag = None
    if diagnostics is not None:
        diag = {c: np.full(len(frame), np.nan) for c in (
            'gp_reporting', 'gp_used', 'gp_hours', 'gp_published_substituted',
            'gp_published_margin_substituted')}
    for key, sensor, g0, length in episodes:
        i = column[sensor]
        published_others = [int(j) for j in np.argsort(distance[i]) if j != i][:neighbours]
        others = (published_others if not by_pattern or GP_NEIGHBOUR_ORDER == 'published'
                  else nearest_others(distance, i, neighbours))
        lo, hi = max(0, g0 - MARGIN), min(len(values), g0 + length + MARGIN)
        rows_ctx = np.concatenate([np.arange(lo, g0), np.arange(g0 + length, hi)])
        block = by_key[key]
        rows_gap = g0 + block.position.to_numpy(int) - 1

        y_all = values[rows_ctx, i]
        ok = np.isfinite(y_all)
        if diag is not None:
            where = frame.index.get_indexer(block.index)
            diag['gp_reporting'][where] = np.isfinite(values[np.ix_(rows_gap, others)]).sum(axis=1)
            # what the published process did here: which neighbours it kept, and whether one of
            # them was silent at a scored hour or at a finite margin hour
            raw = values[np.ix_(rows_ctx[ok], published_others)]
            kept = [c for c in range(len(published_others))
                    if np.isfinite(raw[:, c]).sum() >= 12
                    and float(raw[np.isfinite(raw[:, c]), c].std()) >= 1e-8]
            pub = [published_others[c] for c in kept]
            diag['gp_published_substituted'][where] = (
                (~np.isfinite(values[np.ix_(rows_gap, pub)])).any(axis=1) if pub else 0)
            diag['gp_published_margin_substituted'][where] = float(
                bool(pub) and (~np.isfinite(raw[:, kept])).any())
        if ok.sum() < 24:
            continue
        if by_pattern:
            pred, used, hours_used, lost = _gpr_by_pattern(values, i, others, rows_ctx[ok],
                                                           y_all[ok], rows_gap, g0)
            out[frame.index.get_indexer(block.index)] = pred
            failures += [f'{key}: {e}' for e in lost]
            fitted += int(np.isfinite(pred).any())
            if diag is not None:
                diag['gp_used'][where] = used
                diag['gp_hours'][where] = hours_used
            continue
        del rows_gap  # rebuilt below once the usable neighbours are known
        rows_gap = g0 + block.position.to_numpy(int) - 1

        # Spatial part: a linear mean in the neighbours. Temporal part: a GP on what that leaves.
        # Putting the neighbours inside the kernel instead fails, because the periodic term is
        # isotropic and would impose a daily cycle across neighbour space as well as across time,
        # and a near-constant neighbour column then makes the covariance singular. A linear mean
        # with a temporal GP on the residual is universal kriging and is stable.
        ctx_rows = rows_ctx[ok]
        raw = values[np.ix_(ctx_rows, others)]
        usable, stats = [], []
        for c, j in enumerate(others):
            v = raw[:, c]
            finite = np.isfinite(v)
            if finite.sum() < 12:
                continue
            mu, sd = float(v[finite].mean()), float(v[finite].std())
            if sd < 1e-8:
                continue
            usable.append(j)
            stats.append((mu, sd))

        def spatial(idx):
            if not usable:
                return np.zeros((len(idx), 0))
            b = values[np.ix_(idx, usable)]
            return np.nan_to_num(np.column_stack(
                [(b[:, c] - mu) / sd for c, (mu, sd) in enumerate(stats)]))

        y = y_all[ok]
        centre = float(y.mean())
        target = y - centre
        design_ctx = spatial(ctx_rows)
        if design_ctx.shape[1]:
            coef, *_ = np.linalg.lstsq(
                np.hstack([design_ctx, np.ones((len(design_ctx), 1))]), target, rcond=None)
            mean_ctx = np.hstack([design_ctx, np.ones((len(design_ctx), 1))]) @ coef
            design_gap = spatial(rows_gap)
            mean_gap = np.hstack([design_gap, np.ones((len(design_gap), 1))]) @ coef
        else:
            mean_ctx = np.zeros(len(target))
            mean_gap = np.zeros(len(rows_gap))

        t_ctx = ((ctx_rows - g0) / 24.0)[:, None]
        t_gap = ((rows_gap - g0) / 24.0)[:, None]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e3))
                  + ConstantKernel(0.5, (1e-3, 1e3)) * ExpSineSquared(2.0, 1.0,
                                                                      periodicity_bounds='fixed')
                  + WhiteKernel(0.1, (1e-3, 1e1)))
        try:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                          n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(t_ctx, target - mean_ctx)
            out[frame.index.get_indexer(block.index)] = (
                gp.predict(t_gap) + mean_gap + centre)
            fitted += 1
        except Exception as exc:                       # noqa: BLE001 - counted, not swallowed
            failures.append(f'{key}: {type(exc).__name__}')
    if not fitted:
        raise RuntimeError(f'GPR fitted no episode of {len(episodes)}; '
                           f'first failures: {failures[:3]}')
    print(f'gpr: fitted {fitted} of {len(episodes)} episodes, {len(failures)} failed'
          + (' (mean per neighbour pattern)' if by_pattern else ''), flush=True)
    if diagnostics is not None:
        diagnostics.update(diag)
        diagnostics['failures'] = failures
    return out


def lstm_fill(frame, variable, folds_of_test, bidirectional=True):
    """A small recurrent network trained here, reading the gap from both edges.

    The counterpart to Chronos: the same bidirectional idea, but the weights are learned on this
    bay rather than pretrained on a corpus. If the two land together, the useful thing being
    brought is the architecture; if the pretrained one wins, it is the corpus.
    """
    import torch
    from torch import nn

    episodes, panel, index = episode_parts(frame, variable)
    values = panel.to_numpy(float)
    column = {n: i for i, n in enumerate(panel.columns)}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(2, 64, num_layers=1, batch_first=True,
                               bidirectional=bidirectional)
            self.head = nn.Linear(64 * (2 if bidirectional else 1) + 1, 1)

        def forward(self, context, position):
            _, (h, _) = self.rnn(context)
            h = torch.cat([h[0], h[1]], dim=-1) if bidirectional else h[0]
            return self.head(torch.cat([h, position], dim=-1)).squeeze(-1)

    def sample(key, sensor, g0, length, positions):
        i = column[sensor]
        left = values[max(0, g0 - MARGIN):g0, i]
        right = values[g0 + length:g0 + length + MARGIN, i]
        if len(left) < MARGIN or len(right) < MARGIN:
            return None
        ctx = np.stack([np.concatenate([left, right[::-1]]),
                        np.concatenate([np.zeros(len(left)), np.ones(len(right))])], axis=-1)
        if not np.isfinite(ctx[:, 0]).all():
            return None
        centre = float(np.nanmean(left))
        scale = float(np.nanstd(np.concatenate([left, right]))) or 1.0
        ctx = ctx.copy()
        ctx[:, 0] = (ctx[:, 0] - centre) / scale
        pos = (positions / max(length, 1)).astype('float32')[:, None]
        return ctx.astype('float32'), pos, centre, scale

    train_keys = [(k, s, g, L) for k, s, g, L in episodes
                  if k not in folds_of_test]
    test_keys = [(k, s, g, L) for k, s, g, L in episodes if k in folds_of_test]
    by_key = {k: g for k, g in frame.groupby('episode', sort=False)}

    batch_ctx, batch_pos, batch_y = [], [], []
    for key, sensor, g0, length in train_keys:
        block = by_key[key]
        got = sample(key, sensor, g0, length, block.position.to_numpy(int) - 1)
        if got is None:
            continue
        ctx, pos, centre, scale = got
        truth = (block.truth.to_numpy(float) - centre) / scale
        keep = np.isfinite(truth)
        if not keep.any():
            continue
        batch_ctx.append(np.repeat(ctx[None], keep.sum(), axis=0))
        batch_pos.append(pos[keep])
        batch_y.append(truth[keep].astype('float32'))
    if not batch_ctx:
        return np.full(len(frame), np.nan)

    x = torch.tensor(np.concatenate(batch_ctx), device=device)
    p = torch.tensor(np.concatenate(batch_pos), device=device)
    y = torch.tensor(np.concatenate(batch_y), device=device)
    net = Net().to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(y)
    for _ in range(12):
        order = torch.randperm(n, device=device)
        for k in range(0, n, 256):
            idx = order[k:k + 256]
            optimiser.zero_grad()
            loss = nn.functional.l1_loss(net(x[idx], p[idx]), y[idx])
            if torch.isfinite(loss):
                loss.backward()
                optimiser.step()

    out = np.full(len(frame), np.nan)
    net.eval()
    with torch.no_grad():
        for key, sensor, g0, length in test_keys:
            block = by_key[key]
            got = sample(key, sensor, g0, length, block.position.to_numpy(int) - 1)
            if got is None:
                continue
            ctx, pos, centre, scale = got
            xb = torch.tensor(np.repeat(ctx[None], len(pos), axis=0), device=device)
            pb = torch.tensor(pos, device=device)
            pred = net(xb, pb).cpu().numpy() * scale + centre
            out[frame.index.get_indexer(block.index)] = pred
    return out


def feature_columns(frame, causal, recursive=False):
    # The comparands are never features. `pchip` and `nearest_live_probe` joined them in addendum 32;
    # dropping them here keeps every addendum-30 arm's feature set exactly as published, and pchip
    # reads the far edge, so a causal arm must not see it.
    drop = {'sensor', 'sensor_index', 'episode', 'time', 'truth', 'fold', 'has_twin',
            'persistence', 'climatology', 'nearest_probe', 'linear', 'spline', 'pchip',
            'nearest_live_probe'}
    if not recursive:
        drop |= set(RECURSION_STATE)
    cols = [c for c in frame.columns if c not in drop]
    if causal:
        cols = [c for c in cols if not any(c.startswith(p) for p in RETROSPECTIVE_ONLY)
                and c != 'mean_after_24']
    return cols


def tree_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 20.0, log=True)}


def fit_predict(train, test, cols, params):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=SEED, n_jobs=THREADS)
    x = train[cols].to_numpy(float)
    model.fit(x, train.truth.to_numpy(float))
    return model.predict(test[cols].to_numpy(float))


def polynomial_predict(train, test, cols, params):
    """Multivariate polynomial regression, ridge-penalised.

    A middle term between the linear baselines and the trees: it can represent curvature and
    products of the neighbour readings, which a straight line cannot, without the piecewise
    structure a tree imposes. The feature set is trimmed to the strongest columns first, because
    an unrestricted quadratic over thirty features is nine hundred terms on a few thousand rows.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.impute import SimpleImputer
    degree = int(params.get('poly_degree', 2))
    alpha = float(params.get('poly_alpha', 1.0))
    keep = [c for c in cols if not c.startswith('nb') or c.endswith(('_offset', '_raw'))][:14]
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          PolynomialFeatures(degree=degree, include_bias=False),
                          Ridge(alpha=alpha))
    model.fit(train[keep].to_numpy(float), train.truth.to_numpy(float))
    return model.predict(test[keep].to_numpy(float))


def rolling(frame, train, test, cols, params, backward):
    """Recursive prediction from one edge, one hour at a time.

    The model predicts the next hour's change from the current value, its own prediction becomes
    the new current value, and it steps again. Errors compound, which is the point of comparing it
    with direct prediction.

    Stepping is done by *position across all episodes at once* rather than episode by episode. Row
    at a time this is roughly two hundred thousand single-row predictions per call, which at a
    hundred trials and four folds runs for hours; batched by position it is at most one call per
    distinct step.
    """
    import lightgbm as lgb
    anchor = 'anchor_fwd' if not backward else 'anchor_bwd'
    # The *other* anchor has to go, not just be ignored. Only the active one is overwritten by the
    # running prediction; leaving the opposite one in the matrix hands the model the true value at
    # the adjacent scored position and it stops recursing at all. Retrospective arms carry both
    # columns, so a backward rollout was reading `anchor_fwd` directly: oxygen scored 2.36 against
    # linear interpolation's 4.52, which is what reading the answer looks like.
    other = 'anchor_bwd' if not backward else 'anchor_fwd'
    use = [c for c in cols if c not in (anchor, other)] + [anchor]   # anchor last, to overwrite

    # Teacher forcing: train on the true value at the previous scored position. `elapsed_fwd` /
    # `elapsed_bwd` tell the model how many hours that step spans, which matters once positions are
    # sampled rather than consecutive.
    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=SEED, n_jobs=THREADS)
    model.fit(train[use].to_numpy(float), (train.truth - train[anchor]).to_numpy(float))

    features = test[use].to_numpy(float).copy()
    order = np.argsort(test.position.to_numpy() * (1 if not backward else -1), kind='stable')
    codes, _ = pd.factorize(test.episode.to_numpy())
    edge = (test.near_edge if not backward else test.after_edge).to_numpy(float)
    current = np.full(codes.max() + 1, np.nan)
    out = np.full(len(test), np.nan)
    # Step through the scored positions in order, all episodes at once at each rank, so the number
    # of model calls is the number of distinct positions rather than the number of rows.
    ranks = pd.Series(order).groupby(codes[order]).cumcount().to_numpy()
    rank_of = np.empty(len(test), int)
    rank_of[order] = ranks
    for r in range(int(rank_of.max()) + 1):
        rows = np.flatnonzero(rank_of == r)
        if not len(rows):
            continue
        state = np.where(np.isfinite(current[codes[rows]]), current[codes[rows]], edge[rows])
        block = features[rows]
        block[:, -1] = state
        value = state + model.predict(block)
        current[codes[rows]] = value
        out[rows] = value
    return out


def in_context(train, test, cols, kind):
    """TabPFN or TabICL as the estimator, with the training set as its prompt.

    Both ship a first-class regressor, so neither needs the target binned. TabPFN 9.0.0 additionally
    requires a one-time licence acceptance and an API token before it will download its checkpoint;
    without `TABPFN_TOKEN` in the environment it raises rather than running, and the caller records
    that as the reason the arm is absent instead of substituting something else.

    Missing values are set to zero before the prompt is built, in physical units, so a dark
    neighbour reads as 0 degC or 0 umol/l, and the prompt is at most TABPFN_TRAIN_CAP training rows
    drawn at random from the fold's pool; the tree arms receive the missing values as such. The
    thesis states both; the arms were not rerun.
    """
    rng = np.random.default_rng(SEED)
    if len(train) > TABPFN_TRAIN_CAP:
        train = train.iloc[rng.choice(len(train), TABPFN_TRAIN_CAP, replace=False)]
    x = np.nan_to_num(train[cols].to_numpy(float))
    y = train.truth.to_numpy(float)
    xt = np.nan_to_num(test[cols].to_numpy(float))
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    if kind == 'tabpfn':
        # The licence token lives in the operator's home directory at mode 600, never in this
        # repository, never in a run snapshot and never on a command line. Read it here so the
        # value stays out of the environment of every other process too.
        if not os.environ.get('TABPFN_TOKEN'):
            token = os.path.expanduser('~/.tabpfn_token')
            if os.path.exists(token):
                with open(token) as handle:
                    os.environ['TABPFN_TOKEN'] = handle.read().strip()
        from tabpfn import TabPFNRegressor
        model = TabPFNRegressor(device=device)
    else:
        from tabicl import TabICLRegressor
        model = TabICLRegressor(device=device, random_state=SEED)
    model.fit(x, y)
    return model.predict(xt)


def chronos_fill(variable, frame, bidirectional, repo=None, covariates=False, exogenous=None,
                 categorical_hour=False, baseline=None):
    """Chronos-Bolt run across the gap, forward and optionally backward from the far edge.

    This is the arm the tabular models cannot be: a time-series foundation model sees the shape of
    the history rather than a row of summary features, and gap filling is the task it was built
    for, run twice. Backward is the same model on the reversed series, which costs nothing and is
    what makes a *bidirectional* pretrained filler.

    It is also the sharpest test of the propagation hypothesis. Chronos has no access to the
    neighbouring probes at all, so if long gaps are lost to compounding rather than to a lack of
    information, Chronos should fail on them exactly as the recursive arms do while TabPFN, which
    regresses on contemporaneous neighbours, does not.

    Addendum 33 runs the same blend with Chronos-2 (`repo`), and with `covariates` the four nearest
    probes of the same variable enter as past covariates over the context and as future covariates
    over the gap, in both directions. Their readings inside the gap are known after the fact, which
    is what retrospective recovery means; the target probe's other channels are never used, since a
    real outage takes the whole sonde.

    Addendum 35 adds two inputs, both off by default so every earlier arm is unchanged. With
    `exogenous` (a frame of hourly channels on the panel index) each channel enters Chronos-2 as a
    past and a future covariate like a neighbour, and `categorical_hour` adds the hour of day as a
    categorical one. With `baseline` (a frame of probe-hour predictions on the panel index) Chronos
    fills the target minus the baseline and the baseline is added back over the gap: the
    covariate-regressor scheme of AutoGluon-TimeSeries.

    Horizons. amazon/chronos-bolt-base forecasts 64 steps natively from a context of up to 2048;
    asked for more, the pipeline warns that quality may degrade and continues autoregressively,
    appending its own forecast to the context in 64-step blocks (2 blocks at 72 h, 12 at 720 h,
    23 at 1440 h). amazon/chronos-2 forecasts up to 1024 steps natively (64 output patches of 16)
    from up to 8192, and is continued the same way only at 1440 h. Both are given CHRONOS_CTX
    hours of context here. Past 64 h the Chronos-Bolt fills are therefore its own continuations,
    which the thesis states.
    """
    import torch
    from chronos import BaseChronosPipeline

    meta, frames = observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    panel = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    position_of = {t: i for i, t in enumerate(index)}
    xy = meta.loc[names, ['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    column = {n: i for i, n in enumerate(names)}
    repo = repo or CHRONOS_REPO
    second_generation = repo == CHRONOS2_REPO

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
            # A neighbour dark for the whole context or the whole gap carries nothing; NaN inside
            # an otherwise observed series is passed through, which Chronos-2 accepts.
            if np.isfinite(p).any() and np.isfinite(f).any():
                past[f'nb{c}'] = p.astype(np.float32)
                future[f'nb{c}'] = f.astype(np.float32)
        return past, future

    exo_values = (exogenous.reindex(index).to_numpy(float) if exogenous is not None else None)
    exo_names = list(exogenous.columns) if exogenous is not None else []
    hour_labels = np.array([f'h{h:02d}' for h in index.hour]) if categorical_hour else None
    base = (baseline.reindex(index).reindex(columns=names).fillna(0.0).to_numpy(float)
            if baseline is not None else None)

    def window_pair(s, g0, length, direction):
        if direction == 'fwd':
            return s[max(0, g0 - CHRONOS_CTX):g0], s[g0:g0 + length]
        return s[g0 + length:g0 + length + CHRONOS_CTX][::-1], s[g0:g0 + length][::-1]

    def exogenous_covariates(g0, length, direction):
        past, future = {}, {}
        for c, name in enumerate(exo_names):
            p, f = window_pair(exo_values[:, c], g0, length, direction)
            if np.isfinite(p).any() and np.isfinite(f).any():
                past[f'ex_{name}'] = p.astype(np.float32)
                future[f'ex_{name}'] = f.astype(np.float32)
        if hour_labels is not None:
            past['hour'], future['hour'] = window_pair(hour_labels, g0, length, direction)
        return past, future

    episodes = []
    for key in frame.episode.unique():
        sensor, start, length = key.rsplit('|', 2)
        g0 = position_of[pd.Timestamp(start)]
        episodes.append((key, sensor, g0, int(length)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device,
                                               torch_dtype=torch.float32)
    median = (int(np.argmin(np.abs(np.asarray(pipe.quantiles) - 0.5)))
              if second_generation else None)
    predictions = {}
    try:
        # Grouped by length so one call covers every episode of that length.
        by_length = {}
        for key, sensor, g0, length in episodes:
            by_length.setdefault(length, []).append((key, sensor, g0))
        for length, group in sorted(by_length.items()):
            for direction in (('fwd', 'bwd') if bidirectional else ('fwd',)):
                contexts, items = [], []
                for key, sensor, g0 in group:
                    series = panel[sensor].to_numpy(float)
                    if base is not None:
                        series = series - base[:, column[sensor]]
                    if direction == 'fwd':
                        window = series[max(0, g0 - CHRONOS_CTX):g0]
                    else:
                        tail = series[g0 + length:g0 + length + CHRONOS_CTX]
                        window = tail[::-1]
                    # Chronos cannot take NaN; interior holes are bridged, and the mask guarantees
                    # 72 clean hours nearest the gap, which is the part that matters most.
                    filled = pd.Series(window).interpolate(limit_direction='both').to_numpy()
                    target = np.nan_to_num(filled).astype(np.float32)
                    if second_generation:
                        item = {'target': target}
                        past, future = {}, {}
                        if covariates:
                            past, future = neighbour_covariates(sensor, g0, length, direction)
                        if exo_names or hour_labels is not None:
                            more_past, more_future = exogenous_covariates(g0, length, direction)
                            past, future = {**past, **more_past}, {**future, **more_future}
                        if past:
                            # Past covariates must cover the target's own context exactly.
                            item['past_covariates'] = {k: v[-len(target):]
                                                       for k, v in past.items()}
                            item['future_covariates'] = future
                        items.append(item)
                    else:
                        contexts.append(torch.tensor(target, dtype=torch.float32))
                out = []
                if second_generation:
                    # Chronos-2 requires every item in a call to carry the same covariate names,
                    # and a neighbour dark over the window is dropped, so items are batched by the
                    # set of neighbours they carry and put back in order afterwards.
                    traces = [None] * len(items)
                    groups = {}
                    for n, item in enumerate(items):
                        groups.setdefault(tuple(sorted(item.get('past_covariates', {}))),
                                          []).append(n)
                    # Addendum 35's covariates multiply Chronos-2's memory (target and covariates
                    # attend as one group), and a 1440 h horizon asked for 25.65 GiB in one batch of
                    # 64. Its calls therefore take smaller batches as the gap lengthens; each series
                    # is predicted on its own, so this changes memory, not the forecast. Every
                    # earlier arm keeps its batch of 64.
                    size = 64
                    if exo_names or hour_labels is not None or base is not None:
                        size = 64 if length <= 72 else 16 if length <= 336 else 4
                    for members in groups.values():
                        for k in range(0, len(members), size):
                            chunk = members[k:k + size]
                            got = pipe.predict([items[n] for n in chunk],
                                               prediction_length=length)
                            for n, o in zip(chunk, got):
                                a = o.detach().float().cpu().numpy() if torch.is_tensor(o) \
                                    else np.asarray(o)
                                traces[n] = a.reshape(-1, len(pipe.quantiles),
                                                      a.shape[-1])[0, median]
                    out = traces
                    stacked = np.stack(out) if out else np.zeros((0, length), 'float32')
                else:
                    size = 64 if base is None else 64 if length <= 72 else 16
                    for k in range(0, len(contexts), size):
                        q, _ = pipe.predict_quantiles(inputs=contexts[k:k + size],
                                                      prediction_length=length,
                                                      quantile_levels=[0.1, 0.5, 0.9])
                        out.append(q[:, :, 1].float().cpu().numpy())
                    stacked = np.concatenate(out) if out else np.zeros((0, length), 'float32')
                for row, (key, _, _) in enumerate(group):
                    trace = stacked[row]
                    predictions.setdefault(key, {})[direction] = (
                        trace if direction == 'fwd' else trace[::-1])
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = np.full(len(frame), np.nan)
    keys = frame.episode.to_numpy()
    offsets = frame.position.to_numpy(int) - 1
    spans = frame.gap_length.to_numpy(int)
    starts = {key: g0 for key, _, g0, _ in episodes}
    sensors = frame.sensor.to_numpy()
    for i in range(len(frame)):
        traces = predictions.get(keys[i])
        if not traces:
            continue
        k, length = offsets[i], spans[i]
        forward = traces['fwd'][k]
        if not bidirectional:
            out[i] = forward
        else:
            backward = traces['bwd'][k]
            # Distance-weighted, the same shape as the tuned hybrid blend but with no free
            # parameter, because a zero-shot arm that needed tuning would not be zero-shot.
            w = (length - k) / (length + 1)
            out[i] = w * forward + (1 - w) * backward
        if base is not None:
            out[i] += base[starts[keys[i]] + k, column[sensors[i]]]
    return out


def exogenous_panel(index, period=None):
    """Hourly exogenous series on `index` (addendum 34): the Santa Teresa station and the Magra.

    The same five meteorological channels and the same screening bounds as `bay_forcing.weather`,
    but hourly and uncausal, because a retrospective fill may use what the station recorded during
    the gap. Labelled at the interval end, as `read_export` labels the probes. The development
    period never sees 2026, the rule every loader here follows; the reserved read does.
    """
    from src.data.bay_river import _read as river_series
    raw = pd.read_csv(ROOT / 'data/raw/T2-Meteo.txt', comment='#', sep=r'\s+', header=None,
                      encoding='cp1252')
    t = pd.to_datetime(raw[0] + ' ' + raw[1], format='%d/%m/%Y %H:%M:%S', utc=True)
    fields = {'air': 4, 'solar': 8, 'wind': 12, 'humidity': 7, 'pressure': 3}
    bounds = {'air': (-10, 50), 'solar': (0, 1500), 'wind': (0, 60), 'humidity': (0, 100),
              'pressure': (850, 1100)}
    met = pd.DataFrame({k: pd.to_numeric(raw[c], errors='coerce').replace(-9999, np.nan).to_numpy()
                        for k, c in fields.items()}, index=pd.DatetimeIndex(t))
    for k, (lo, hi) in bounds.items():
        met.loc[~met[k].between(lo, hi), k] = np.nan
    hourly = met.resample('1h', closed='left', label='right').mean().reindex(index)
    hourly['river'] = river_series().resample('1h').mean().reindex(index)
    if (period or PERIOD) == 'development':
        hourly.loc[hourly.index >= pd.Timestamp('2026-01-01', tz='UTC')] = np.nan
    return hourly[list(EXO_CHANNELS)]


def _positions(index, times):
    """Integer positions of `times` in `index`, refusing any that are absent.

    Both sides are brought to nanoseconds first: pandas 3 may carry microseconds on one side, and a
    lookup across units can silently return -1.
    """
    at = index.as_unit('ns').get_indexer(pd.DatetimeIndex(times).as_unit('ns'))
    if (at < 0).any():
        raise ValueError(f'{int((at < 0).sum())} timestamps are not on the panel index')
    return at


def _window_mean(values, a, b):
    """Mean of the finite entries of `values[a:b]`, row by row, from cumulative sums."""
    ok = np.isfinite(values)
    s = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    a, b = np.clip(a, 0, len(values)), np.clip(b, 0, len(values))
    count = n[b] - n[a]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, (s[b] - s[a]) / count, np.nan)


def add_exogenous(frame, period=None):
    """The fourteen addendum-34 exogenous features, in place.

    For each channel, the value at the hour and its departure from the channel's own linear
    interpolation between its 24 h means either side of the gap: the part of the forcing that the
    two edges do not already imply. Plus the centred 24 h means of solar radiation and air
    temperature, which carry cloudiness and the seasonal level rather than the hour.
    """
    pad = pd.Timedelta(hours=MARGIN + 48)
    index = pd.date_range(frame.time.min().floor('h') - pad, frame.time.max().ceil('h') + pad,
                          freq='h')
    exo = exogenous_panel(index, period)
    parts = frame.episode.str.rsplit('|', n=2)
    start = _positions(index, pd.to_datetime(parts.str[1], utc=True))
    length = parts.str[2].astype(int).to_numpy()
    at = _positions(index, frame.time)
    share = frame.position.to_numpy(float) / (length + 1)
    for channel in EXO_CHANNELS:
        v = exo[channel].to_numpy(float)
        before = _window_mean(v, start - 24, start)
        after = _window_mean(v, start + length, start + length + 24)
        frame[f'ex_{channel}'] = v[at]
        frame[f'ex_{channel}_dep'] = v[at] - (before + (after - before) * share)
    for channel in ('solar', 'air'):
        frame[f'ex_{channel}_24'] = exo[channel].rolling(24, center=True, min_periods=12) \
            .mean().to_numpy(float)[at]
    return frame


def add_relative(frame, variable, period=None):
    """Edge-relative features for the amendment-1 arms, in place.

    A tree cannot form the difference of two level features without many splits, so an anchored
    target fed levels is predicted from levels, which shift between calendar blocks. These give it
    the differences directly: the margin means relative to their edges, and each neighbour's
    departure from its own straight line between the two edge hours, which is the neighbour's
    analogue of the departure being predicted. The neighbours are chosen exactly as `build` chose
    them, from the same probes and distances.
    """
    meta, frames = observations(period)
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    meta = meta.loc[names]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    values = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names},
                          index=index).to_numpy(float)
    xy = meta[['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    column = {n: i for i, n in enumerate(names)}
    frame['before_minus_edge'] = frame.mean_before_24 - frame.near_edge
    frame['after_minus_edge'] = frame.mean_after_24 - frame.after_edge
    parts = frame.episode.str.rsplit('|', n=2)
    g0 = _positions(index, pd.to_datetime(parts.str[1], utc=True))
    length = parts.str[2].astype(int).to_numpy()
    position = frame.position.to_numpy(int)
    share = position / (length + 1)
    sensor = frame.sensor.to_numpy()
    deps = np.full((len(frame), NEIGHBOURS), np.nan)
    for name in np.unique(sensor):
        rows = np.flatnonzero(sensor == name)
        i = column[name]
        for j, other in enumerate(nearest_others(distance, i, NEIGHBOURS)):
            ref = values[g0[rows] - 1, other]
            far = values[g0[rows] + length[rows], other]
            now = values[g0[rows] + position[rows] - 1, other]
            deps[rows, j] = now - (ref + (far - ref) * share[rows])
    for j in range(NEIGHBOURS):
        frame[f'nb{j}_dep'] = deps[:, j]
    return frame


def pad_neighbours(frame):
    """Give a frame every neighbour column `build` can write, absent ones marked unavailable.

    2026 has fewer oxygen sondes than the development record, so its frame has fewer neighbour
    columns, and a model frozen on development would otherwise be handed a different feature set.
    A neighbour that does not exist is recorded exactly as one that is dark: not available, no value.
    """
    for j in range(NEIGHBOURS):
        if f'nb{j}_ok' not in frame:
            frame[f'nb{j}_ok'] = 0.0
            for suffix in ('offset', 'raw', 'dist'):
                frame[f'nb{j}_{suffix}'] = np.nan
    return frame


def relative_columns(frame, exogenous):
    """The amendment-1 whitelist: nothing that is an absolute level."""
    cols = ['gap_length', 'position', 'to_near_edge', 'to_far_edge', 'elapsed_fwd', 'elapsed_bwd',
            'span_delta', 'trend_before', 'sd_before_24', 'before_minus_edge', 'after_minus_edge',
            'sin_h', 'cos_h', 'sin_d', 'cos_d']
    for j in range(NEIGHBOURS):
        cols += [f'nb{j}_dep', f'nb{j}_ok', f'nb{j}_dist']
    if exogenous:
        cols += [f'ex_{c}_dep' for c in EXO_CHANNELS]
    missing = [c for c in cols if c not in frame]
    if missing:
        raise KeyError(f'relative features absent: {missing}')
    return cols


def window_mask(pool, episodes):
    """Rows of `pool` on a probe-hour within MARGIN of the gap of any episode in `episodes`.

    Under the tiled design an episode may straddle a fold boundary, and episodes at different
    lengths overlap, so "another fold" does not by itself keep a training target off a scored
    probe-hour. This does, and it also keeps the hours next to a scored gap out, since at hourly
    resolution a neighbouring hour is nearly the answer.
    """
    keys = pd.Series(episodes.episode.unique())
    parts = keys.str.rsplit('|', n=2)
    sensor = parts.str[0].to_numpy()
    start = pd.DatetimeIndex(pd.to_datetime(parts.str[1], utc=True)).as_unit('ns').asi8
    hour = np.int64(3_600_000_000_000)
    lo = start - MARGIN * hour
    hi = start + (parts.str[2].astype(int).to_numpy() + MARGIN) * hour
    t = pd.DatetimeIndex(pool.time).as_unit('ns').asi8
    mask = np.zeros(len(pool), bool)
    for s in np.unique(sensor):
        a, b = lo[sensor == s], hi[sensor == s]
        order = np.argsort(a)
        a, b = a[order], np.maximum.accumulate(b[order])
        rows = np.flatnonzero(pool.sensor.to_numpy() == s)
        if not len(rows):
            continue
        j = np.searchsorted(a, t[rows], side='right') - 1
        inside = (j >= 0) & (t[rows] < b[np.clip(j, 0, None)])
        mask[rows[inside]] = True
    return mask


def fit_predict_anchored(train, test, cols, params):
    """Addendum 34: the tree learns the departure from linear interpolation, not the level.

    The published direct regressions predict the value outright, so nothing ties them to the two
    readings that bound the gap and at 1 h they are several times worse than a straight line. Here
    the straight line is the starting point and the tree only corrects it.
    """
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=SEED, n_jobs=THREADS)
    model.fit(train[cols].to_numpy(float), (train.truth - train.linear).to_numpy(float))
    return test.linear.to_numpy(float) + model.predict(test[cols].to_numpy(float))


def series_panel(variable, period=None):
    """Hourly panel of one variable, every probe that reports it in `period`."""
    meta, frames = observations(period)
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    return pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)


def arx_design(index, harmonics, exo=None, window=1, stats=None):
    """Regressors of the state-space arm: intercept, diurnal and annual harmonics, and exogenous.

    Exogenous channels are smoothed by a centred window and standardised with `stats`, the training
    mean and spread; a missing value becomes the training mean, i.e. zero after standardising.
    """
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()
    cols = [np.ones(len(index))]
    for j in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * j * hour / 24), np.cos(2 * np.pi * j * hour / 24)]
    for j in (1, 2):
        cols += [np.sin(2 * np.pi * j * doy / 365.25), np.cos(2 * np.pi * j * doy / 365.25)]
    X = np.column_stack(cols)
    if exo is None:
        return X, None
    z = exo.rolling(window, center=True, min_periods=1).mean() if window > 1 else exo
    z = z.to_numpy(float)
    if stats is None:
        stats = (np.nanmean(z, axis=0), np.nanstd(z, axis=0) + 1e-9)
    z = (z - stats[0]) / stats[1]
    return np.column_stack([X, np.where(np.isfinite(z), z, 0.0)]), stats


def fold_span(episodes):
    """The calendar interval covered by a set of episodes' windows, margins included."""
    parts = episodes.episode.drop_duplicates().str.rsplit('|', n=2)
    start = pd.to_datetime(parts.str[1], utc=True)
    end = start + pd.to_timedelta(parts.str[2].astype(int), unit='h')
    pad = pd.Timedelta(hours=MARGIN)
    return start.min() - pad, end.max() + pad


def arx_fill(test, params, fit_panel, fit_allowed, test_panel, fit_exo=None, test_exo=None):
    """Regression with AR(1) errors, Kalman-smoothed across the gap (addendum 34).

    Per probe, the series is regressed on `arx_design` by ridge over the hours `fit_allowed` marks.
    With AR(1) errors of coefficient phi the smoother, given the residuals u_a and u_b at the two
    edge hours n = L + 1 apart, has a closed form at distance d from the near edge:

        E[u_d] = (phi^d (1 - phi^(2(n-d))) u_a + phi^(n-d) (1 - phi^(2d)) u_b) / (1 - phi^(2n))

    which meets both edges exactly and relaxes toward the regression as the gap lengthens. The fill
    is the regression plus that bridge. `fit_panel` and `test_panel` are the same panel for a
    development fold and differ only for the frozen 2026 read. A probe with no training hours falls
    back to linear interpolation; the number of such rows is returned beside the fill.
    """
    phi = 1.0 - 10.0 ** (-params['arx_s'])
    window = params.get('arx_window', 1)
    fit_X, stats = arx_design(fit_panel.index, params['arx_harmonics'], fit_exo, window)
    test_X, _ = arx_design(test_panel.index, params['arx_harmonics'], test_exo, window, stats) \
        if fit_exo is not None else arx_design(test_panel.index, params['arx_harmonics'])
    penalty = params['arx_alpha'] * np.eye(fit_X.shape[1])
    penalty[0, 0] = 0.0                                   # never shrink the intercept
    parts = test.episode.str.rsplit('|', n=2)
    g0 = _positions(test_panel.index, pd.to_datetime(parts.str[1], utc=True))
    length = parts.str[2].astype(int).to_numpy()
    d = test.position.to_numpy(int)
    out = test.linear.to_numpy(float).copy()
    fallback = 0
    log_phi = np.log(phi)
    for sensor in test.sensor.unique():
        rows = np.flatnonzero(test.sensor.to_numpy() == sensor)
        y_fit = fit_panel[sensor].to_numpy(float) if sensor in fit_panel else None
        ok = (np.isfinite(y_fit) & fit_allowed[sensor].to_numpy()) if y_fit is not None else None
        if ok is None or ok.sum() < 10 * fit_X.shape[1]:
            fallback += len(rows)
            continue
        X = fit_X[ok]
        beta = np.linalg.solve(X.T @ X + penalty, X.T @ y_fit[ok])
        y = test_panel[sensor].to_numpy(float)
        a, b, t = g0[rows] - 1, g0[rows] + length[rows], g0[rows] + d[rows] - 1
        u_a = y[a] - test_X[a] @ beta
        u_b = y[b] - test_X[b] @ beta
        n = length[rows] + 1
        dd = d[rows]
        p = lambda m: np.exp(m * log_phi)
        bridge = (p(dd) * (1 - p(2 * (n - dd))) * u_a + p(n - dd) * (1 - p(2 * dd)) * u_b) \
            / (1 - p(2 * n))
        out[rows] = test_X[t] @ beta + bridge
    return out, fallback


def addendum34(variable, arm, frame, cols, folder, quick):
    """Addendum-34 arms: anchored gradient boosting and the AR(1)-bridge regression.

    Development: per outer fold, a study nested in the other folds and a fit on them, neither ever
    training on a probe-hour within MARGIN of a scored gap. Reserved: one study and one fit on the
    matching development cohort, applied frozen to 2026, so nothing is fitted or tuned on 2026.
    """
    predictions = np.full(len(frame), np.nan)
    selections = []
    tag = f'{variable}_{arm}'
    exo = arm in EXO_ARMS
    if PERIOD == 'reserved':
        lengths = TRAIN_LENGTHS[DESIGN][:4] if quick else TRAIN_LENGTHS[DESIGN]
        dev = build(variable, np.random.default_rng(SEED), quick, period='development',
                    lengths=lengths)
        pad_neighbours(dev)
        if exo:
            add_exogenous(dev, 'development')
        if arm in REL_ARMS:
            add_relative(dev, variable, 'development')
        splits = [('frozen', dev, frame)]
        fit_period = 'development'
    else:
        splits = [(k, frame[frame.fold != k], frame[frame.fold == k]) for k in range(FOLDS)]
        fit_period = PERIOD
    if arm in ARX_ARMS:
        fit_panel = series_panel(variable, fit_period)
        test_panel = fit_panel if fit_period == PERIOD else series_panel(variable, PERIOD)
        fit_exo = exogenous_panel(fit_panel.index, fit_period) if exo else None
        test_exo = (fit_exo if test_panel is fit_panel else
                    exogenous_panel(test_panel.index, PERIOD)) if exo else None
        observed = fit_panel.notna()

        def allowed(*excluded):
            keep = np.ones(len(fit_panel), bool)
            for episodes in excluded:
                lo, hi = fold_span(episodes)
                keep &= ~((fit_panel.index >= lo) & (fit_panel.index <= hi))
            return pd.DataFrame(observed.to_numpy() & keep[:, None], index=fit_panel.index,
                                columns=fit_panel.columns)

    for label, pool, test in splits:
        if not len(test) or not len(pool):
            continue
        if label != 'frozen':
            pool = pool[~window_mask(pool, test)]
        hold = pool.fold.max()
        inner_test = pool[pool.fold == hold]
        inner_train = pool[pool.fold != hold]
        inner_train = inner_train[~window_mask(inner_train, inner_test)]
        outer = [test] if label != 'frozen' else []

        def objective(trial):
            if arm in ANCHORED_ARMS:
                pred = fit_predict_anchored(inner_train, inner_test, cols, tree_space(trial))
            else:
                q = {'arx_alpha': trial.suggest_float('arx_alpha', 1e-3, 1e3, log=True),
                     'arx_harmonics': trial.suggest_int('arx_harmonics', 1, 3),
                     'arx_s': trial.suggest_float('arx_s', 0.5, 3.5)}
                if exo:
                    q['arx_window'] = trial.suggest_categorical('arx_window', [1, 3, 6, 12, 24])
                pred, _ = arx_fill(inner_test, q, fit_panel, allowed(*outer, inner_test),
                                   fit_panel, fit_exo, fit_exo)
            return score(inner_test.assign(pred=pred), 'pred')

        selection = run_study(f'{tag}_f{label}', objective, folder, quick)
        params = dict(selection['params'])
        if arm in ANCHORED_ARMS:
            params['subsample_freq'] = 1
            pred = fit_predict_anchored(pool, test, cols, params)
        else:
            pred, fallback = arx_fill(test, params, fit_panel, allowed(*outer), test_panel,
                                      fit_exo, test_exo)
            selection['linear_fallback_rows'] = int(fallback)
        predictions[frame.index.get_indexer(test.index)] = pred
        selections.append({'fold': label, 'selection': selection,
                           'training_rows': int(len(pool)) if arm in ANCHORED_ARMS else None})
    return predictions, selections


def covariate_table(panel, exo, codes):
    """Long table of every probe-hour for the addendum-35 covariate regressor.

    Features are the exogenous channels at the hour, the centred 24 h means of solar radiation and
    air temperature, the clock, and the probe as a category (`codes` maps a probe to its category,
    fixed on the fitting panel so a frozen model reads 2026 probes with the development codes).
    """
    hours = len(panel)
    solar_24 = exo['solar'].rolling(24, center=True, min_periods=12).mean().to_numpy(float)
    air_24 = exo['air'].rolling(24, center=True, min_periods=12).mean().to_numpy(float)
    hour, doy = panel.index.hour.to_numpy(), panel.index.dayofyear.to_numpy()
    block = np.column_stack([exo.to_numpy(float), solar_24, air_24,
                             np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                             np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    names = list(panel.columns)
    probe = np.repeat([codes.get(n, -1) for n in names], hours).astype(float)
    X = np.column_stack([np.tile(block, (len(names), 1)), probe])
    y = panel.to_numpy(float).T.reshape(-1)
    t = np.tile(np.arange(hours), len(names))
    return X, y, t


class _Ridge:
    """Amendment 1's linear covariate regressor: ridge on standardised features, probe intercepts.

    The probe enters as one-hot intercepts rather than as a category, missing features take the
    training mean, and the intercepts are not shrunk. Its prediction is a linear function of smooth
    forcing and the clock, so it cannot step from one hour to the next as a tree does.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def _design(self, X):
        z = (X[:, :-1] - self.mean) / self.sd
        z = np.where(np.isfinite(z), z, 0.0)
        onehot = (X[:, -1:] == self.probes[None, :]).astype(float)
        return np.column_stack([z, onehot])

    def fit(self, X, y, categorical_feature=None):
        self.mean = np.nanmean(X[:, :-1], axis=0)
        self.sd = np.nanstd(X[:, :-1], axis=0) + 1e-9
        self.probes = np.unique(X[:, -1])
        D = self._design(X)
        penalty = self.alpha * np.eye(D.shape[1])
        penalty[-len(self.probes):, -len(self.probes):] = 0.0      # never shrink an intercept
        self.beta = np.linalg.solve(D.T @ D + penalty, D.T @ y)
        return self

    def predict(self, X):
        return self._design(X) @ self.beta


def covariate_regressor(fit_panel, fit_exo, test_panel, test_exo, excluded, inner, folder, name,
                        quick, linear=False):
    """Addendum 35: tune and fit the covariate regressor, return its prediction on `test_panel`.

    Trained on observed probe-hours outside every span in `excluded`; tuned by run_study on its own
    absolute error over the observed hours in `inner`'s span, trained on the hours outside it as
    well. A probe the fitting panel never saw gets a baseline of zero, which leaves Chronos to fill
    it exactly as the plain arm would; the count is returned. `linear` swaps the tree for the
    amendment-1 ridge, whose only parameter is its penalty.
    """
    import lightgbm as lgb
    codes = {n: i for i, n in enumerate(fit_panel.columns)}
    X, y, t = covariate_table(fit_panel, fit_exo, codes)
    index = fit_panel.index

    def outside(*spans):
        keep = np.ones(len(index), bool)
        for lo, hi in spans:
            keep &= ~((index >= lo) & (index <= hi))
        return keep[t] & np.isfinite(y)

    spans = [fold_span(e) for e in excluded]
    inner_span = fold_span(inner)
    train = outside(*spans, inner_span)
    in_inner = (index >= inner_span[0]) & (index <= inner_span[1])
    score_rows = in_inner[t] & outside(*spans)
    last = X.shape[1] - 1

    def fit(rows, params):
        if linear:
            return _Ridge(params['ridge_alpha']).fit(X[rows], y[rows])
        model = lgb.LGBMRegressor(**params, verbose=-1, random_state=SEED, n_jobs=THREADS)
        model.fit(X[rows], y[rows], categorical_feature=[last])
        return model

    def objective(trial):
        params = ({'ridge_alpha': trial.suggest_float('ridge_alpha', 1e-3, 1e4, log=True)}
                  if linear else tree_space(trial))
        model = fit(train, params)
        return float(np.mean(np.abs(y[score_rows] - model.predict(X[score_rows]))))

    selection = run_study(name, objective, folder, quick)
    params = dict(selection['params']) if linear else dict(selection['params'], subsample_freq=1)
    model = fit(outside(*spans), params)
    Xt, _, tt = covariate_table(test_panel, test_exo, codes)
    pred = model.predict(Xt).reshape(len(test_panel.columns), len(test_panel)).T
    unseen = [n for n in test_panel.columns if n not in codes]
    baseline = pd.DataFrame(pred, index=test_panel.index, columns=test_panel.columns)
    baseline[unseen] = 0.0
    selection['unseen_probes'] = unseen
    return baseline, selection


def addendum35(variable, arm, frame, folder, quick):
    """Addendum-35 arms: Chronos with exogenous covariates, and Chronos on a covariate residual.

    The covariate arms are zero-shot and read the whole frame at once. The residual arms fit the
    covariate regressor per outer fold on hours outside that fold's span, or for 2026 once on the
    development record, applied frozen; Chronos is zero-shot on what the regressor leaves.
    """
    tag = f'{variable}_{arm}'
    test_panel = series_panel(variable, PERIOD)
    test_exo = exogenous_panel(test_panel.index, PERIOD)
    if arm in A35_COVARIATE_ARMS:
        pred = chronos_fill(variable, frame, True, repo=CHRONOS2_REPO,
                            covariates=arm == 'chronos2_all_bidir', exogenous=test_exo,
                            categorical_hour=True)
        return pred, [{'fold': 'all', 'selection': {'tuned': False,
                                                    'reason': 'zero-shot, no search'}}]
    repo = CHRONOS_REPO if arm.startswith('bolt_') else CHRONOS2_REPO
    neighbours = arm.startswith('chronos2_cov_resid')
    predictions = np.full(len(frame), np.nan)
    selections = []
    if PERIOD == 'reserved':
        lengths = TRAIN_LENGTHS[DESIGN][:4] if quick else TRAIN_LENGTHS[DESIGN]
        dev = build(variable, np.random.default_rng(SEED), quick, period='development',
                    lengths=lengths)
        fit_panel = series_panel(variable, 'development')
        fit_exo = exogenous_panel(fit_panel.index, 'development')
        splits = [('frozen', [], dev[dev.fold == dev.fold.max()], frame)]
    else:
        fit_panel, fit_exo = test_panel, test_exo
        splits = []
        for k in range(FOLDS):
            test = frame[frame.fold == k]
            rest = frame[frame.fold != k]
            if len(test) and len(rest):
                splits.append((k, [test], rest[rest.fold == rest.fold.max()], test))
    for label, excluded, inner, test in splits:
        baseline, selection = covariate_regressor(fit_panel, fit_exo, test_panel, test_exo,
                                                  excluded, inner, folder,
                                                  f'{tag}_covreg_f{label}', quick,
                                                  linear=arm in A35_LINEAR_ARMS)
        pred = chronos_fill(variable, test, True, repo=repo, covariates=neighbours,
                            baseline=baseline)
        predictions[frame.index.get_indexer(test.index)] = pred
        selections.append({'fold': label, 'selection': selection})
    return predictions, selections


def score(frame, column):
    """Macro MAE: mean over episodes of the episode's mean absolute error."""
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return float('nan')
    return float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean())


def run(variable, arm, quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f'{variable}_{arm}'
    status = OUT / f'status_{tag}.json'
    save(status, {'state': 'RUNNING', 'variable': variable, 'arm': arm, 'pid': os.getpid()})
    try:
        rng = np.random.default_rng(SEED)
        frame = build(variable, rng, quick)
        if arm in A34_ARMS:
            pad_neighbours(frame)
        if arm in EXO_ARMS:
            add_exogenous(frame)
        causal = arm in ('regression_causal', 'rolling_forward', 'chronos_forward')
        recursive = arm in ('rolling_forward', 'rolling_backward', 'hybrid')
        cols = feature_columns(frame, causal, recursive)
        if arm in REL_ARMS:
            add_relative(frame, variable)
            cols = relative_columns(frame, arm in EXO_ARMS)
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)

        predictions = np.full(len(frame), np.nan)
        selections = []
        gp_diagnostics = []
        # Addendum 34 has its own fold loop: a leak guard the published arms predate, and a frozen
        # 2026 read fitted on the development cohort. The loop below is then skipped.
        if arm in A34_ARMS:
            predictions, selections = addendum34(variable, arm, frame, cols, folder, quick)
        elif arm in A35_ARMS:
            predictions, selections = addendum35(variable, arm, frame, folder, quick)
        for fold in range(0 if arm in A34_ARMS + A35_ARMS else FOLDS):
            test = frame[frame.fold == fold]
            pool = frame[frame.fold != fold]
            # Zero-shot and per-episode arms never read the pool, so an empty one is no reason to
            # leave a fold unscored.
            per_episode = arm.startswith('chronos') or arm in ('gpr', 'gpr_k8', 'compressed_sensing')
            if not len(test) or (not len(pool) and not per_episode):
                continue
            inner_hold = pool.fold.max()
            inner_train = pool[pool.fold != inner_hold]
            inner_test = pool[pool.fold == inner_hold]

            if arm in IN_CONTEXT_ARMS or arm in ('gpr', 'gpr_k8', 'compressed_sensing'):
                params, selection = {}, {'tuned': False,
                                         'reason': 'pretrained or per-episode fit, no search'}
            else:
                def objective(trial):
                    if arm == 'polynomial':
                        q = {'poly_degree': trial.suggest_int('poly_degree', 2, 3),
                             'poly_alpha': trial.suggest_float('poly_alpha', 1e-2, 1e3,
                                                               log=True)}
                        return score(inner_test.assign(
                            pred=polynomial_predict(inner_train, inner_test, cols, q)),
                            'pred')
                    p = tree_space(trial)
                    if arm == 'hybrid':
                        w = trial.suggest_float('blend_power', 0.25, 4.0, log=True)
                        fwd = rolling(frame, inner_train, inner_test, cols, p, backward=False)
                        bwd = rolling(frame, inner_train, inner_test, cols, p, backward=True)
                        pred = blend(inner_test, fwd, bwd, w)
                    elif arm == 'rolling_forward':
                        pred = rolling(frame, inner_train, inner_test, cols, p, backward=False)
                    elif arm == 'rolling_backward':
                        pred = rolling(frame, inner_train, inner_test, cols, p, backward=True)
                    else:
                        pred = fit_predict(inner_train, inner_test, cols, p)
                    return score(inner_test.assign(pred=pred), 'pred')

                selection = run_study(f'{tag}_f{fold}', objective, folder, quick)
                params = {k: v for k, v in selection['params'].items()
                          if k != 'blend_power'}
                if arm != 'polynomial':
                    params['subsample_freq'] = 1

            if arm == 'hybrid':
                w = selection['params']['blend_power']
                fwd = rolling(frame, pool, test, cols, params, backward=False)
                bwd = rolling(frame, pool, test, cols, params, backward=True)
                pred = blend(test, fwd, bwd, w)
            elif arm == 'rolling_forward':
                pred = rolling(frame, pool, test, cols, params, backward=False)
            elif arm == 'rolling_backward':
                pred = rolling(frame, pool, test, cols, params, backward=True)
            elif arm in ('tabpfn', 'tabicl'):
                pred = in_context(pool, test, cols, arm)
            elif arm.startswith('chronos2'):
                # Addendum 33: Chronos-2, univariate or with the nearest probes as covariates.
                pred = chronos_fill(variable, test, True, repo=CHRONOS2_REPO,
                                    covariates=arm == 'chronos2_cov_bidir')
            elif arm.startswith('chronos'):
                # Zero-shot and per-episode, so the fold's training pool is unused.
                pred = chronos_fill(variable, test, arm == 'chronos_bidir')
            elif arm == 'compressed_sensing':
                pred = compressed_sensing_fill(test, variable)
            elif arm in ('gpr', 'gpr_k8'):
                diag = {}
                pred = gpr_fill(test, variable, neighbours=8 if arm == 'gpr_k8' else NEIGHBOURS,
                                diagnostics=diag)
                gp_diagnostics.append(pd.DataFrame(
                    {'episode': test.episode.to_numpy(), 'position': test.position.to_numpy(),
                     **{c: v for c, v in diag.items() if c != 'failures'}}))
            elif arm == 'lstm':
                pred = lstm_fill(frame, variable, set(test.episode.unique()))[
                    frame.index.get_indexer(test.index)]
            elif arm == 'polynomial':
                pred = polynomial_predict(pool, test, cols, params)
            else:
                pred = fit_predict(pool, test, cols, params)
            predictions[frame.index.get_indexer(test.index)] = pred
            selections.append({'fold': fold, 'selection': selection})

        frame[arm] = predictions
        baselines = ['persistence', 'linear', 'spline', 'pchip', 'climatology', 'nearest_probe',
                     'nearest_live_probe']
        keep = ['episode', 'sensor', 'has_twin', 'gap_length', 'position', 'fold', 'time',
                'truth'] + baselines + [arm]
        path = folder / 'predictions.csv.gz'
        frame[keep].to_csv(path, index=False)
        gp_record = {}
        if gp_diagnostics:
            # Per scored hour: how many of the episode's neighbours report, how many the fit used and
            # on how many margin hours, and whether the published process entered a silent one there.
            diag = pd.concat(gp_diagnostics, ignore_index=True)
            diag.to_csv(folder / 'gp_diagnostics.csv.gz', index=False,
                        compression={'method': 'gzip', 'mtime': 0})
            gp_record = {
                'gp_mean': ('linear mean per pattern of reporting neighbours, fitted where the '
                            'target and every neighbour of the pattern report (addendum 59)'
                            if GP_BY_PATTERN else
                            'published: one linear mean, a silent neighbour entered at its margin '
                            'mean'),
                'gp_by_pattern': GP_BY_PATTERN,
                'gp_mean_declared_in': ('config/bay_reserved_protocol_v2_addendum59.json'
                                        if GP_BY_PATTERN else None),
                'gp_neighbour_order': GP_NEIGHBOUR_ORDER if GP_BY_PATTERN else 'published',
                'gp_min_pattern_hours': GP_MIN_PATTERN_HOURS,
                'gp_rows_published_process_entered_a_silent_neighbour': round(float(
                    diag.gp_published_substituted.mean()), 4)}

        by_length = {}
        contributing = frame.attrs.get('probes_contributing', {})
        for length, block in frame.groupby('gap_length'):
            by_length[str(int(length))] = {
                m: round(score(block, m), 4) for m in baselines + [arm]}
            # A sixty-day result rests on three probes. Carrying the count beside the number is the
            # difference between a statement about the network and one about a few instruments.
            by_length[str(int(length))]['probes'] = int(contributing.get(int(length), 0))
            by_length[str(int(length))]['episodes'] = int(block.episode.nunique())
        by_twin = {str(bool(t)): {m: round(score(b, m), 4)
                                  for m in ['linear', 'nearest_probe', arm]}
                   for t, b in frame.groupby('has_twin')}
        report = {'declared_in': ('config/bay_reserved_protocol_v2_addendum35_amendment1.json'
                                  if arm in A35_LINEAR_ARMS
                                  else 'config/bay_reserved_protocol_v2_addendum35.json'
                                  if arm in A35_ARMS
                                  else 'config/bay_reserved_protocol_v2_addendum34_amendment1.json'
                                  if arm in REL_ARMS
                                  else 'config/bay_reserved_protocol_v2_addendum34.json'
                                  if arm in A34_ARMS
                                  else 'config/bay_reserved_protocol_v2_addendum33.json'
                                  if arm in ('chronos2_bidir', 'chronos2_cov_bidir', 'gpr_k8')
                                  else 'config/bay_reserved_protocol_v2_addendum30.json'
                                  if DESIGN == 'strict'
                                  else 'config/bay_reserved_protocol_v2_addendum32.json'),
                  'design': DESIGN, 'period': PERIOD,
                  'protocol': PROTOCOL['version'], 'variable': variable, 'arm': arm,
                  'causal': causal, 'smoke_only': quick,
                  'features': len(cols), 'feature_names': cols,
                  'rows': int(len(frame)), 'episodes': int(frame.episode.nunique()),
                  'overall': {m: round(score(frame, m), 4) for m in baselines + [arm]},
                  'by_gap_length': by_length, 'by_twin': by_twin, 'selections': selections,
                  **gp_record,
                  'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        save(folder / 'report.json', report)
        save(status, {'state': 'COMPLETED', 'variable': variable, 'arm': arm,
                      'overall': report['overall']})
        print(json.dumps(report['overall'], indent=2))
    except BaseException as exc:
        save(status, {'state': 'FAILED', 'variable': variable, 'arm': arm, 'error': str(exc)})
        traceback.print_exc()
        raise


def blend(test, forward, backward, power):
    """Weight each rollout by how close its own edge is, with a tuned falloff."""
    a = test.to_near_edge.to_numpy(float) ** power
    b = test.to_far_edge.to_numpy(float) ** power
    w = b / (a + b)          # near the start, b dominates, so the forward rollout is trusted
    return w * forward + (1 - w) * backward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run(a.variable, a.arm, a.quick)


if __name__ == '__main__':
    main()
