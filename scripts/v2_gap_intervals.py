"""Paired bootstrap intervals for every gap-recovery contrast.

The arm tables report point estimates, and the rest of this thesis does not accept those on their
own: a difference is reported with an interval or it is not reported. This supplies the missing
half, by the convention Chapter 5 uses everywhere else.

The resampling unit is an **overlap cluster**, not a row and not an episode. Rows inside one masked
stretch are one instrument over consecutive hours, so resampling rows is plainly wrong. Resampling
episodes is wrong too, less plainly: episodes are drawn from the few long clean stretches each
probe has, so episodes on one probe overlap each other heavily. Measured, 5 781 temperature episodes
form only 183 overlap-connected groups and 983 oxygen episodes only 73; at 720 h the eight oxygen
episodes are all windows onto one stretch of one probe. Treating those as independent made the
intervals far too narrow, and on the long oxygen gaps it produced intervals on what is a sample of
one. Episodes on the same probe whose gaps share any hour are merged into one cluster, and clusters
are resampled.

An interval needs several clusters to mean anything, so a contrast resting on fewer than
MIN_CLUSTERS is reported as not estimable rather than given a number.

Every contrast is paired: the same resampled clusters score both arms, so the difference is not
inflated by which stretches happened to be easy.

Usage: python -m scripts.v2_gap_intervals [<run-id> ...] [--results DIR] [--out FILE]

With no run id it rebuilds the three published interval sets listed in
`v2_gap_climatology.PUBLISHED`, which is how the pipeline calls it.

`--results` names the results directory inside each run (the addendum-31 confirmatory and
replication runs write to suffixed ones), and `--out` keeps their intervals from
overwriting the development set.
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import PUBLISHED, arm_dirs, load_predictions

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_intervals.json'
DRAWS = 600
REFERENCE = 'linear'
FLOOR = {'temperature': 0.20, 'oxygen': 1.8}
MIN_CLUSTERS = 5


def clusters(frame):
    """episode -> cluster id, merging episodes on one probe whose gaps share any hour."""
    times = pd.to_datetime(frame.time, utc=True)
    ep = (frame.assign(t=times).groupby('episode')
          .agg(sensor=('sensor', 'first'), t0=('t', 'min'), t1=('t', 'max')).reset_index())
    out, current = {}, 0
    for _, block in ep.groupby('sensor'):
        end = None
        for _, row in block.sort_values('t0').iterrows():
            if end is None or row.t0 > end:
                current += 1
                end = row.t1
            else:
                end = max(end, row.t1)
            out[row.episode] = current
    return out


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return float('nan')
    return float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean())


def paired(frame, arm, reference, label, cluster_of):
    """Bootstrap the macro-MAE difference (arm minus reference) by resampling overlap clusters.

    The point estimate is still the mean over episodes, so it matches the arm tables; only the
    uncertainty is computed at the level where the data are independent.
    """
    d = frame.dropna(subset=[arm, reference, 'truth']).copy()
    if not len(d):
        return None
    per = (d.assign(ea=(d.truth - d[arm]).abs(), eb=(d.truth - d[reference]).abs())
             .groupby('episode')[['ea', 'eb']].mean())
    per['cluster'] = per.index.map(cluster_of)
    n = len(per)
    groups = [g[['ea', 'eb']].to_numpy() for _, g in per.groupby('cluster')]
    k = len(groups)
    base = {'delta': round(float(per.ea.mean() - per.eb.mean()), 4),
            'episodes': int(n), 'clusters': int(k)}
    if k < MIN_CLUSTERS:
        return {**base, 'ci_low': None, 'ci_high': None, 'excludes_zero': False,
                'estimable': False}
    rng = np.random.default_rng([20260922, zlib.crc32(label.encode())])
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = np.concatenate([groups[j] for j in rng.integers(0, k, k)])
        draws[i] = pick[:, 0].mean() - pick[:, 1].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # A percentile bootstrap over a handful of clusters is too narrow. The delete-one-cluster
    # jackknife with a t(k-1) critical value is the small-sample check: where the two disagree on
    # whether zero is excluded, the result rests on the resampling scheme and is reported as such.
    from scipy.stats import t as student
    sums = np.array([g.sum(axis=0) for g in groups])
    sizes = np.array([len(g) for g in groups])
    leave = (sums.sum(axis=0) - sums) / (n - sizes)[:, None]
    theta = leave[:, 0] - leave[:, 1]
    se = float(np.sqrt((k - 1) / k * ((theta - theta.mean()) ** 2).sum()))
    half = float(student.ppf(0.975, k - 1)) * se
    point = base['delta']
    return {**base, 'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4),
            'excludes_zero': bool(lo > 0 or hi < 0), 'estimable': True,
            'jackknife_t': [round(point - half, 4), round(point + half, 4)],
            'jackknife_excludes_zero': bool(point - half > 0 or point + half < 0)}


def by_probe(frame, arm, reference):
    """Arm minus reference per probe, macro over that probe's episodes: who carries a pooled gain."""
    d = frame.dropna(subset=[arm, reference, 'truth'])
    per = (d.assign(ea=(d.truth - d[arm]).abs(), eb=(d.truth - d[reference]).abs())
             .groupby(['sensor', 'episode'])[['ea', 'eb']].mean())
    out = {}
    for sensor, g in per.groupby(level='sensor'):
        out[sensor] = {'delta': round(float(g.ea.mean() - g.eb.mean()), 4), 'episodes': int(len(g))}
    return out


def floor_verdicts(got, floor):
    """Beyond the floor by the point estimate, and by the whole interval."""
    got['exceeds_floor'] = bool(-got['delta'] > floor)
    got['interval_beyond_floor'] = bool(got.get('estimable') and got['ci_high'] is not None
                                        and got['ci_high'] < -floor)
    return got


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--results', default='v2_gap_recovery')
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    if not args.runs:
        # The pipeline calls every producer bare; that rebuilds all three published interval sets.
        for run, results, artifact in PUBLISHED:
            intervals([run], results, RUNS / artifact)
        return
    intervals(args.runs, args.results, Path(args.out))


def intervals(runs, results, out_path):
    frames, sources = {}, {}
    for run in runs:
        for (variable, name), folder in arm_dirs(run, results).items():
            # The published climatology column was fitted over the masked stretch as well; the
            # loader substitutes the declared one or drops it, and resolves arms rerun since.
            frames[(variable, name)] = load_predictions(run, results, variable, name)
            sources[f'{variable}/{name}'] = folder.relative_to(RUNS).as_posix()
    if not frames:
        raise SystemExit(f'no predictions under {runs}')

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum30.json',
              'runs': runs, 'results': results, 'arm_sources': sources,
              'draws': DRAWS, 'min_clusters': MIN_CLUSTERS,
              'unit': 'overlap cluster (episodes on one probe whose gaps share any hour)',
              'climatology': 'declared, fitted outside the masked stretch',
              'reference': REFERENCE, 'variables': {}}
    for variable in ['temperature', 'oxygen']:
        arms = {a: f for (v, a), f in frames.items() if v == variable}
        if not arms:
            continue
        block = {'overall': {}, 'by_gap_length': {}}
        # The parameter-free baselines are comparands too. The temperature long-gap result rests on
        # the nearest probe, which is a baseline, so leaving them out would leave that claim with
        # no interval at all.
        any_frame = next(iter(arms.values()))
        for base in ('spline', 'nearest_probe', 'persistence', 'climatology'):
            if base in any_frame.columns:
                arms.setdefault(base, any_frame)
        # Every arm scored the same episodes, so one clustering serves them all.
        cluster_of = clusters(any_frame)
        block['clusters'] = int(len(set(cluster_of.values())))
        for arm, frame in sorted(arms.items()):
            got = paired(frame, arm, REFERENCE, f'{variable}|{arm}|all', cluster_of)
            if got:
                got['beats_reference'] = bool(got['delta'] < 0)
                floor_verdicts(got, FLOOR[variable])
                got['by_probe'] = by_probe(frame, arm, REFERENCE)
                block['overall'][arm] = got
        for arm, frame in sorted(arms.items()):
            for length, part in frame.groupby('gap_length'):
                got = paired(part, arm, REFERENCE, f'{variable}|{arm}|{length}', cluster_of)
                if got:
                    floor_verdicts(got, FLOOR[variable])
                    got['by_probe'] = by_probe(part, arm, REFERENCE)
                    block['by_gap_length'].setdefault(str(int(length)), {})[arm] = got
        report['variables'][variable] = block

        print(f'\n== {variable}: arm minus {REFERENCE}, negative favours the arm '
              f'(floor {FLOOR[variable]})')
        for arm, e in sorted(block['overall'].items(), key=lambda kv: kv[1]['delta']):
            star = '*' if e['excludes_zero'] else ' '
            floor = ' FLOOR' if e['exceeds_floor'] else ''
            ci = (f"[{e['ci_low']:+8.4f}, {e['ci_high']:+8.4f}]" if e['estimable']
                  else '[  not estimable   ]')
            print(f"   {arm:<22} {e['delta']:+8.4f} {ci}{star}  "
                  f"n={e['episodes']} in {e['clusters']} clusters{floor}")
    out_path.write_bytes(json.dumps(report, indent=2).encode('utf-8'))  # LF on every host
    print('\nwrote', out_path)


if __name__ == '__main__':
    main()
