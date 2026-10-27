"""Addendum 75: how near a neighbour must be to help the Gaussian-process gap filler.

The published process (v2_gap_recovery.gpr_fill, fitted per pattern of reporting neighbours) reads the
four nearest other probes. Here it reads the four nearest probes at least `radius` metres from the
target: 250 m drops the co-located twin, 1,000 and 2,000 m drop the nearer sites. Temperature only.

  --gate      on 200 episodes per cohort (seed 20261007), radius 0 must reproduce the published column
              the thesis reads to 1e-9 with the same rows defined (two BLAS threads, as published)
  --fill      one (cohort, radius) over every episode of that cohort, written to results/
  --combine   locally, from the fetched fills: each radius against linear and against radius 0

Usage (compute host): python -m scripts.v2_gap_neighbour_radius --gate --cohort Gap-Dev
                      python -m scripts.v2_gap_neighbour_radius --fill --cohort Gap-Dev --radius 1000
      (local):        python -m scripts.v2_gap_neighbour_radius --combine --run <run id>
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_neighbour_radius.json'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum75.json'
#: Gap-Dev's original run (amendment 1): read_arm resolves its Gaussian process to addendum 59's rerun, the
#: column the thesis reads; the run first declared, 20260922T195802Z, held an earlier replacement of it
COHORTS = {'Gap-Dev': ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery', 'development', 6),
           'Gap-2026': ('20260922T204715Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026', 'reserved', 6),
           'Gap-Long-Dev': ('20260922T195727Z_v2_gaprecovery', 'v2_gap_recovery_long', 'development', 168),
           'Gap-Long-2026': ('20260922T195745Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026', 'reserved', 168)}
RADII = (0, 250, 1000, 2000)
SEED, GATE_EPISODES, TOLERANCE = 20261007, 200, 1e-9
BAND = (6, 72)


INPUTS = RUNS / 'v2_gap_neighbour_radius' / 'inputs'
COLUMNS = ['episode', 'position', 'gap_length', 'sensor', 'time', 'truth', 'linear', 'gpr']


def published(cohort):
    """The rows and the published column the thesis reads (replaced arms resolved), from the compact
    copy `--prepare` writes, which is what the compute host receives."""
    path = INPUTS / f'{cohort}.csv.gz'
    if path.exists():
        return pd.read_csv(path)
    from scripts.v2_gap_climatology import read_arm
    run, results, _, shortest = COHORTS[cohort]
    frame = read_arm(run, results, 'temperature', 'gpr')
    return frame[frame.gap_length >= shortest][COLUMNS].reset_index(drop=True)


def prepare():
    INPUTS.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        path = INPUTS / f'{cohort}.csv.gz'
        if path.exists():
            path.unlink()
        frame = published(cohort)
        frame.to_csv(path, index=False, lineterminator='\n', compression={'method': 'gzip', 'mtime': 0})
        print(cohort, frame.episode.nunique(), 'episodes', len(frame), 'rows')


def fill(frame, cohort, radius):
    """gpr_fill with its neighbours restricted to probes at least `radius` metres from the target."""
    from src.experiments import v2_gap_recovery as G
    G.PERIOD = COHORTS[cohort][2]
    original = G.nearest_others

    def beyond(distance, i, k=None):
        others = [j for j in original(distance, i) if distance[i, j] >= radius]
        return others if k is None else others[:k]
    G.nearest_others = beyond
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            diag = {}
            out = G.gpr_fill(frame[['episode', 'position']].copy(), 'temperature', neighbours=4, by_pattern=True,
                             diagnostics=diag)
    finally:
        G.nearest_others = original
    return out, diag


def gate(cohort, path):
    pub = published(cohort)
    keys = pub.episode.unique()
    chosen = set(np.random.default_rng(SEED).choice(keys, size=min(GATE_EPISODES, len(keys)), replace=False))
    rows = pub[pub.episode.isin(chosen)].reset_index(drop=True)
    started = time.time()
    got, _ = fill(rows, cohort, 0)
    ref = rows.gpr.to_numpy(float)
    same = bool((np.isfinite(got) == np.isfinite(ref)).all())
    both = np.isfinite(got) & np.isfinite(ref)
    worst = float(np.max(np.abs(got[both] - ref[both]))) if both.any() else 0.0
    entry = {'cohort': cohort, 'episodes': len(chosen), 'rows': int(len(rows)), 'same_rows_defined': same,
             'max_abs_difference': worst, 'passed': bool(same and worst <= TOLERANCE),
             'seconds': round(time.time() - started, 1)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((json.dumps(entry, indent=1) + '\n').encode('utf-8'))
    print(json.dumps(entry), flush=True)


def run_fill(cohort, radius, path):
    pub = published(cohort)
    started = time.time()
    got, diag = fill(pub, cohort, radius)
    out = pub[['episode', 'position', 'gap_length', 'sensor', 'time', 'truth', 'linear']].copy()
    out[f'gpr_r{radius}'] = got
    out['neighbours_used'] = diag.get('gp_used')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator='\n', compression={'method': 'gzip', 'mtime': 0})
    print(cohort, radius, 'episodes', pub.episode.nunique(), 'defined', int(np.isfinite(got).sum()),
          'of', len(got), 'seconds', round(time.time() - started), flush=True)


def nearest_distance(frame, radius, period):
    """Median over episodes of the distance from the target to the nearest probe the restricted process
    may read."""
    from src.experiments.v2_gap_recovery import nearest_others
    from scripts.v2_gap_neighbours import _panel
    names, _, _, distance = _panel('temperature', period)
    column = {n: i for i, n in enumerate(names)}
    d = []
    for sensor in frame.drop_duplicates('episode').sensor:
        i = column[sensor]
        others = [j for j in nearest_others(distance, i) if distance[i, j] >= radius]
        d.append(distance[i, others[0]] if others else np.nan)
    return round(float(np.nanmedian(d)), 0)


def combine(run):
    from scripts import v2_gap_long as L
    from scripts.v2_gap_intervals import clusters, paired
    from scripts.v2_gap_climatology import read_arm
    base = RUNS / run / 'results'
    gates = {c: json.loads((base / f'gate_{c}.json').read_text()) for c in COHORTS}
    # amendment 1: every row of each radius-0 fill against the column the thesis reads
    full = {}
    for cohort, (src, results, _, _) in COHORTS.items():
        ref = read_arm(src, results, 'temperature', 'gpr')[['episode', 'position', 'gpr']]
        fill0 = pd.read_csv(base / f'fill_{cohort}_r0.csv.gz')
        m = fill0.merge(ref, on=['episode', 'position'], how='left', validate='one_to_one')
        both = np.isfinite(m.gpr_r0) & np.isfinite(m.gpr)
        worst = float(np.abs(m.gpr_r0 - m.gpr)[both].max())
        full[cohort] = {'rows': int(len(m)), 'same_rows_defined': bool((np.isfinite(m.gpr_r0) == np.isfinite(m.gpr)).all()),
                        'max_abs_difference': worst}
        full[cohort]['passed'] = full[cohort]['same_rows_defined'] and worst <= TOLERANCE
    rep = {'declared_in': DECLARATION, 'amended_by': 'config/bay_reserved_protocol_v2_addendum75_amendment1.json',
           'run': run, 'gate_GD_200_episodes_as_launched': gates, 'gate_GD_every_row': full,
           'gate_passed': all(g['passed'] for g in full.values()), 'cohorts': {}}
    if not rep['gate_passed']:
        OUT.write_bytes(json.dumps(rep, indent=1).encode('utf-8'))
        raise SystemExit('gate GD failed: no radius is read')
    for cohort in COHORTS:
        frame = None
        for r in RADII:
            f = pd.read_csv(base / f'fill_{cohort}_r{r}.csv.gz')
            frame = f if frame is None else frame.merge(f[['episode', 'position', f'gpr_r{r}']],
                                                        on=['episode', 'position'], validate='one_to_one')
        long = cohort.startswith('Gap-Long')
        parts = (frame.groupby('gap_length') if long else
                 [('6-72', frame[frame.gap_length.between(*BAND)])])
        block = {}
        for length, part in parts:
            if long:
                primary, conservative = L.units(part)
                maps = {'cluster': primary, 'probe_month': conservative}
            else:
                maps = {'cluster': clusters(part)}
            row = {'episodes': int(part.episode.nunique()),
                   'mae': {c: round(float(part.dropna(subset=[c]).assign(e=lambda d, c=c: (d.truth - d[c]).abs())
                                          .groupby('episode').e.mean().mean()), 4)
                           for c in ['linear'] + [f'gpr_r{r}' for r in RADII]},
                   'nearest_distance_m': {str(r): nearest_distance(part, r, COHORTS[cohort][2]) for r in RADII},
                   'contrasts': {}}
            for r in RADII:
                for ref in ('linear', 'gpr_r0'):
                    if ref == f'gpr_r{r}':
                        continue
                    for unit, mapping in maps.items():
                        got = paired(part, f'gpr_r{r}', ref, f'radius|{cohort}|{length}|{r}|{ref}|{unit}', mapping)
                        row['contrasts'].setdefault(f'gpr_r{r} - {ref}', {})[unit] = got
            block[str(length) if not long else str(int(length))] = row
        rep['cohorts'][cohort] = block
    OUT.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    print('wrote', OUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', action='store_true'); p.add_argument('--fill', action='store_true')
    p.add_argument('--combine', action='store_true'); p.add_argument('--prepare', action='store_true')
    p.add_argument('--cohort', choices=list(COHORTS)); p.add_argument('--radius', type=int, choices=RADII)
    p.add_argument('--run')
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.gate:
        gate(a.cohort, f'results/gate_{a.cohort}.json')
    elif a.fill:
        run_fill(a.cohort, a.radius, f'results/fill_{a.cohort}_r{a.radius}.csv.gz')
    elif a.combine:
        combine(a.run)


if __name__ == '__main__':
    main()
