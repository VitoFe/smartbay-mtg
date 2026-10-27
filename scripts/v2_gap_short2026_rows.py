"""Addendum 82, part C: an arm's Gap-2026 rows at 1-72 h, the published 6-72 h rows with the 1-3 h
rows of this part's run appended.

The published Gap-2026 episodes stay exactly as published: a joint 1-72 h rebuild would draw the
1-3 h episodes first and change every later one. So the 6-72 h rows are always read from the
published runs, through the published readers (`v2_gap_climatology.load_predictions` and
`read_arm`, which apply the declared climatology, the corrected nearest probe and every rerun arm),
and the 1-3 h rows of the same arm are appended from runs/v2_addendum82/C/remote/<SHORT_RUN>. The
reader refuses an arm the short run does not hold, a published column the short rows lack, and any
(episode, position) present in both parts.

The short run was built by the current `build`, so its climatology is the declared one (fitted
outside the mask) and its nearest probe excludes the probe itself by index; the gate
(scripts/v2_gap_short2026_gate.py) shows that the same code reproduces both corrected published columns
on the 6-72 h rows.

Usage: imported by the part-C scorers.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
PART = RUNS / 'v2_addendum82' / 'C'
#: the full run of src/experiments/v2_gap_short2026.py (the smoke run is never read)
SHORT_RUN = '20261012T075809Z_v2_add82C'
RESULTS = 'v2_gap_recovery_reserved2026'
#: the gate run's results, 6-72 h, read only by the reproduction gate
GATE = PART / 'remote' / SHORT_RUN / 'results' / 'gate' / RESULTS
SHORT = PART / 'remote' / SHORT_RUN / 'results' / 'short' / RESULTS
SHORT_LENGTHS = (1, 2, 3)
KEYS = ['episode', 'position']


def short_rows(variable, arm):
    path = SHORT / f'{variable}_{arm}' / 'predictions.csv.gz'
    if not path.exists():
        raise SystemExit(f'the 1-3 h run holds no {variable} {arm}')
    frame = pd.read_csv(path)
    if not set(frame.gap_length.unique()) <= set(SHORT_LENGTHS):
        raise SystemExit(f'{path}: lengths other than 1-3 h')
    return frame


def append(published, variable, arm):
    """`published` (6-72 h rows of one arm, as a published reader returns them) + its 1-3 h rows."""
    if arm not in published:
        raise SystemExit(f'{variable} {arm}: the published frame does not carry the arm')
    if (published.gap_length < 6).any():
        raise SystemExit(f'{variable} {arm}: the published frame already holds rows below 6 h')
    short = short_rows(variable, arm)
    missing = [c for c in published.columns if c not in short.columns]
    if missing:
        raise SystemExit(f'{variable} {arm}: the 1-3 h rows lack published columns {missing}')
    both = pd.MultiIndex.from_frame(published[KEYS]).intersection(
        pd.MultiIndex.from_frame(short[KEYS]))
    if len(both):
        raise SystemExit(f'{variable} {arm}: {len(both)} rows in both parts')
    return pd.concat([published, short[list(published.columns)]], ignore_index=True)


def r6(x):
    return None if x is None else round(float(x), 6)


def paired6(frame, arm, reference, label, cluster_of, check=True):
    """`v2_gap_intervals.paired` kept to six decimals, rounded once by the table, with the published function's
    four-decimal result beside it as a check (`published_function`, `agrees_at_four_decimals`).

    Same rows (both arms and the truth present), same per-episode means, same clusters, the same
    600 draws from the stream seeded [20260922, crc32(label)], the same percentile interval and the
    same delete-one-cluster jackknife with t(k-1), centred on the unrounded estimate. Adds the paired
    row count, the interval status, the robust reading (twenty clusters or the jackknife agreeing)
    and the two-sided p of addendum 79 (scripts/v2_fdr.p_side).
    """
    import zlib

    import numpy as np
    from scipy.stats import t as student

    from scripts.v2_fdr import p_mid, p_side
    from scripts.v2_gap_intervals import DRAWS, MIN_CLUSTERS, paired

    d = frame.dropna(subset=[arm, reference, 'truth'])
    if not len(d):
        return None
    per = (d.assign(ea=(d.truth - d[arm]).abs(), eb=(d.truth - d[reference]).abs())
             .groupby('episode')[['ea', 'eb']].mean())
    per['cluster'] = per.index.map(cluster_of)
    if per.cluster.isna().any():
        raise SystemExit(f'{label}: episodes without a cluster')
    n = len(per)
    groups = [g[['ea', 'eb']].to_numpy() for _, g in per.groupby('cluster')]
    k = len(groups)
    point = float(per.ea.mean() - per.eb.mean())
    out = {'delta': r6(point), 'mae_arm': r6(per.ea.mean()), 'mae_reference': r6(per.eb.mean()),
           'episodes': int(n), 'clusters': int(k), 'paired_rows': int(len(d))}
    if k < MIN_CLUSTERS:
        out.update({'ci_low': None, 'ci_high': None, 'excludes_zero': False, 'estimable': False,
                    'status': 'n.e.', 'jackknife_t': None, 'jackknife_excludes_zero': None,
                    'robust': False, 'p': None})
    else:
        rng = np.random.default_rng([20260922, zlib.crc32(label.encode())])
        draws = np.empty(DRAWS)
        for i in range(DRAWS):
            pick = np.concatenate([groups[j] for j in rng.integers(0, k, k)])
            draws[i] = pick[:, 0].mean() - pick[:, 1].mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        sums = np.array([g.sum(axis=0) for g in groups])
        sizes = np.array([len(g) for g in groups])
        leave = (sums.sum(axis=0) - sums) / (n - sizes)[:, None]
        theta = leave[:, 0] - leave[:, 1]
        se = float(np.sqrt((k - 1) / k * ((theta - theta.mean()) ** 2).sum()))
        half = float(student.ppf(0.975, k - 1)) * se
        excludes = bool(lo > 0 or hi < 0)
        jack = bool(point - half > 0 or point + half < 0)
        p = p_side(point, float(lo), float(hi))
        out.update({'ci_low': r6(lo), 'ci_high': r6(hi), 'excludes_zero': excludes, 'estimable': True,
                    'status': 'excludes zero' if excludes else 'not distinguishable',
                    'jackknife_t': [r6(point - half), r6(point + half)],
                    'jackknife_excludes_zero': jack,
                    'robust': bool(excludes and (k >= 20 or jack)),
                    'p': r6(p if p is not None else p_mid(float(lo), float(hi)))})
    if check:
        published = paired(frame, arm, reference, label, cluster_of)
        out['published_function'] = {f: published.get(f) for f in (
            'delta', 'ci_low', 'ci_high', 'clusters', 'episodes', 'jackknife_t')}
        four = lambda x: None if x is None else round(x, 4)
        out['agrees_at_four_decimals'] = bool(
            published['delta'] == four(point) and published['clusters'] == k
            and published['episodes'] == n
            and (k < MIN_CLUSTERS or (published['ci_low'] == four(float(lo))
                                      and published['ci_high'] == four(float(hi)))))
    return out


def same_rows(frames):
    """Refuse unless every frame scores identical (episode, position) rows in the same order."""
    first = None
    for name, frame in frames.items():
        keys = frame[KEYS].reset_index(drop=True)
        if first is None:
            first = (name, keys)
        elif not keys.equals(first[1]):
            raise SystemExit(f'{name} scores other rows than {first[0]}')
