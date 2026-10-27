"""A block bootstrap of nested macro errors, computed from per-block sums instead of concatenated rows.

The development-fold scripts (v2_horizon_table, v2_summarize_forecast, v2_physics_contrasts,
v2_seed_contrasts, v2_sequence_table) draw (fold, ISO week) blocks with replacement, concatenate the
drawn rows and take the nested macro: mean absolute error per (fold, sensor), then the mean over
sensors within each fold, then over folds. A block drawn k times contributes k times its rows, so
the same statistic follows from each block's per-sensor error sums and counts weighted by how often
the block was drawn. This module does that with the scripts' own random calls (one
`rng.integers(0, B, B)` per draw over the scripts' own block order), so it reproduces their draws; the
sums differ from the concatenated ones only in floating-point order (checked by the callers against
the published values).

Used by scripts/v2_perfold_neural_contrasts.py and scripts/v2_sequence_sweep_contrasts.py
(addenda 49 and 51).
"""
import numpy as np
import pandas as pd


def week(d):
    return pd.to_datetime(d.time, utc=True).dt.isocalendar().week.astype(int)


def block_list(frame, order='first'):
    """(fold, week) blocks in first-appearance order of `frame`, or sorted."""
    f = frame[['fold']].assign(week=week(frame))
    keys = [tuple(int(x) for x in k) for k in f.drop_duplicates().values.tolist()]
    return sorted(keys) if order == 'sorted' else keys


def draws(frame, columns, blocks, rng, n_draws=600):
    """Macro errors of `columns` on the full frame and on each of n_draws block resamples.

    Returns (point: dict col -> float, samples: dict col -> array of n_draws)."""
    f = frame.assign(week=week(frame))
    sensors = {s: i for i, s in enumerate(sorted(f.sensor.unique()))}
    bindex = {b: i for i, b in enumerate(blocks)}
    B, S = len(blocks), len(sensors)
    fold_of_block = np.array([b[0] for b in blocks])
    folds = np.unique(fold_of_block)
    bi = np.array([bindex[(int(a), int(b))] for a, b in zip(f.fold, f.week)])
    si = np.array([sensors[s] for s in f.sensor])
    cnt = np.zeros((B, S))
    np.add.at(cnt, (bi, si), 1.0)
    sums = {}
    for c in columns:
        e = (f.truth - f[c]).abs().to_numpy(float)
        if not np.isfinite(e).all():
            raise ValueError(f'{c}: missing values; pair the rows first')
        m = np.zeros((B, S))
        np.add.at(m, (bi, si), e)
        sums[c] = m
    fold_mask = np.stack([fold_of_block == k for k in folds])          # F x B

    def macro(weights):                                                 # weights: B (or D x B)
        w = np.atleast_2d(weights)
        out = {}
        n = np.einsum('db,fb,bs->dfs', w, fold_mask, cnt)                # D x F x S
        for c in columns:
            s = np.einsum('db,fb,bs->dfs', w, fold_mask, sums[c])
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
            per_fold = np.nanmean(mean, axis=2)                          # D x F (sensors present)
            present = (n > 0).any(axis=2)
            per_fold = np.where(present, per_fold, np.nan)
            out[c] = np.nanmean(per_fold, axis=1)
        return out

    point = {c: float(v[0]) for c, v in macro(np.ones(B)).items()}
    W = np.zeros((n_draws, B))
    for d in range(n_draws):
        W[d] = np.bincount(rng.integers(0, B, B), minlength=B)
    samples = macro(W)
    return point, samples


def interval(point, samples, combo):
    """[point, 2.5th, 97.5th percentile] of a linear combination [(weight, column), ...]."""
    p = sum(w * point[c] for w, c in combo)
    s = sum(w * samples[c] for w, c in combo)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return [float(p), float(lo), float(hi)]
