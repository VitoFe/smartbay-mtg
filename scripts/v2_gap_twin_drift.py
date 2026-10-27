"""Addendum 74: does drift explain why a co-located twin trails the Gaussian process once a gap has closed?

The published twin is the twin's reading shifted by the offset between the two probes at the hour before
the gap. Once the gap has closed the offset at the hour after it is known as well; `twin_two_edge`
interpolates the offset linearly in time between the two edges. If slow drift between the probes is why
the twin trails the Gaussian process, whose linear mean in the neighbours is fitted on both margins, the
two-edge twin should close most of that deficit; if short-term scatter is the reason, it should not.

Temperature only, parameter-free. Cohorts: the 6-72 h band of Gap-Dev and Gap-2026 (the episodes of
v2_gap_twin_rule, overlap clusters as there) and the long cohorts (v2_gap_long: episodes as the
declared unit, probe x month and site x month beside them).

Usage: python -m scripts.v2_gap_twin_drift
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import v2_gap_long as L
from scripts import v2_gap_twin_rule as T
from scripts.v2_gap_intervals import clusters, paired
from scripts.v2_gap_neighbours import _panel
from src.experiments.v2_gap_recovery import nearest_others

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_gap_twin_drift.json'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum74.json'
CONTRASTS = (('twin_two_edge', 'twin'), ('twin_two_edge', 'gpr'), ('twin', 'gpr'), ('twin_two_edge', 'linear'))
MIN_UNITS = 5


def two_edge(frame, period):
    """The twin with its offset interpolated in time between the hour before and the hour after the gap;
    NaN where the published twin is undefined or the twin is dark at the hour after the gap."""
    names, index, values, distance = _panel('temperature', period)
    column = {n: i for i, n in enumerate(names)}
    where = {t: k for k, t in enumerate(index)}
    out = np.full(len(frame), np.nan)
    for key, block in frame.groupby('episode', sort=False):
        sensor, start, length = key.rsplit('|', 2)
        i, g0, n = column[sensor], where[pd.Timestamp(start)], int(float(length))
        j = nearest_others(distance, i, 1)[0]
        pre, post = g0 - 1, g0 + n
        if post >= len(values) or not np.isfinite(values[[pre, post]][:, [i, j]]).all():
            continue
        off_pre, off_post = values[pre, i] - values[pre, j], values[post, i] - values[post, j]
        t = g0 + block.position.to_numpy(int) - 1
        offset = off_pre + (off_post - off_pre) * (t - pre) / (post - pre)
        rows = frame.index.get_indexer(block.index)
        live = block.twin_reports.to_numpy(bool)
        out[rows[live]] = values[t[live], j] + offset[live]
    return out


def mae(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4) if len(d) else None


def compare(frame, label, mappings):
    """Every declared contrast on the rows where both twin variants and the Gaussian process exist."""
    rows = frame.dropna(subset=['twin', 'twin_two_edge', 'gpr', 'truth'])
    out = {'episodes': int(rows.episode.nunique()), 'probes': sorted(rows.sensor.unique().tolist()),
           'mae': {c: mae(rows, c) for c in ('linear', 'twin', 'twin_two_edge', 'gpr')}, 'contrasts': {}}
    for x, y in CONTRASTS:
        for unit, mapping in mappings.items():
            n_units = len({mapping[e] for e in rows.episode.unique()})
            if n_units < MIN_UNITS:
                out['contrasts'].setdefault(f'{x} - {y}', {})[unit] = {'units': n_units, 'not_reported': 'fewer than five'}
                continue
            got = paired(rows, x, y, f'twin_drift|{label}|{x}|{y}|{unit}', mapping)
            out['contrasts'].setdefault(f'{x} - {y}', {})[unit] = got
    return out


def main():
    rep = {'declared_in': DECLARATION, 'cohorts': {}}
    for year in ('2025', '2026'):
        frame = T.load(year)
        frame = frame[frame.gap_length.between(*T.BAND)].copy()
        frame['twin_two_edge'] = two_edge(frame, T.PERIODS[year])
        label = f'Gap-{"Dev" if year == "2025" else "2026"} 6-72 h'
        rep['cohorts'][label] = compare(frame, label, {'cluster': clusters(frame)})
    for cohort, (run, results, period) in L.COHORTS.items():
        frame = L.load(run, results, 'temperature', period)
        frame['twin_two_edge'] = two_edge(frame, period)
        primary, conservative = L.units(frame)
        mappings = {'episode': primary, 'probe_month': conservative, 'site_month': L.site_units(frame)}
        rep['cohorts'][cohort] = {str(int(length)): compare(part, f'{cohort}|{int(length)}', mappings)
                                  for length, part in frame.groupby('gap_length')}
    OUT.write_bytes(json.dumps(rep, indent=1, default=str).encode('utf-8'))
    for cohort, block in rep['cohorts'].items():
        items = block.items() if cohort.startswith('Gap-Long') else [('6-72', block)]
        for length, b in items:
            c = b['contrasts']
            unit = 'cluster' if 'cluster' in c.get('twin - gpr', {}) else 'episode'
            show = {k: (v[unit]['delta'], v[unit]['ci_low'], v[unit]['ci_high'], v[unit]['excludes_zero'])
                    if v.get(unit) and 'delta' in v[unit] else v.get(unit) for k, v in c.items()}
            print(cohort, length, 'episodes', b['episodes'], b['mae'], show)


if __name__ == '__main__':
    main()
