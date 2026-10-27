"""The twin rule of addendum 31, scored against linear interpolation and against both its parts.

Temperature only. Where the dark probe has a co-located twin and that twin reports, the rule
returns the twin's offset-corrected reading; otherwise it returns bidirectional Chronos. It came out
of the addendum-30 stratification by twin, and addendum 31 declared it as an arm before the 2026
read, so 2026 is its first test on data it had not shaped.

The rule first read `nearest_probe`, which on a twin probe is the twin's reading where the twin
reports and persistence where it is dark, so the rule fell back to persistence rather than to
Chronos on those rows, and the twin itself was scored partly as persistence. The twin is now scored
only where it reports (`twin`, with every other method's error given on the same rows), and the
rule falls back to Chronos wherever there is no twin reading. The 2026 twin is
also set against the Gaussian process of addendum 33, which the confirmatory run did not include.

Both years are scored on 6-72 h, the band addendum 31 declared for the confirmatory read, and
by the cluster bootstrap of `v2_gap_intervals`, with clusters formed within the band. That is why
the 2025 cluster count (256) exceeds the whole-sweep figure (183): the long episodes that bridged
short ones into a single cluster are outside the band.

The first version of this artifact was computed inline and its seed labels were not recorded. This
producer reproduces its point estimates, episode counts and cluster counts exactly; its interval
endpoints differ in the fourth decimal, which is the bootstrap's own resolution at 600 draws.

Usage: python -m scripts.v2_gap_twin_rule
"""
import json
from pathlib import Path

import pandas as pd

from scripts.v2_gap_climatology import fix_neighbours
from scripts.v2_gap_intervals import clusters, paired
from scripts.v2_gap_neighbours import with_twin

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_twin_rule.json'
SOURCES = {'2025': '20260922T171859Z_v2_gaprecovery/v2_gap_recovery',
           '2026': '20260922T180814Z_v2_gaprecovery/v2_gap_recovery_reserved2026'}
PERIODS = {'2025': 'development', '2026': 'reserved'}
#: the addendum-33 read of the corrected Gaussian process on the same 2026 episodes
GPR_2026 = '20260922T204715Z_v2_gaprecovery/v2_gap_recovery_reserved2026'
BAND = (6, 72)
CONTRASTS = [('twin_rule', 'linear'), ('twin_rule', 'chronos_bidir'),
             ('twin_rule', 'nearest_probe'), ('chronos_bidir', 'linear'),
             ('chronos_bidir', 'nearest_probe'), ('nearest_probe', 'linear'),
             ('nearest_probe', 'chronos_bidir'),
             ('twin', 'linear'), ('twin', 'gpr'), ('twin', 'chronos_bidir')]


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def with_gpr(frame, source, variable):
    """Attach the corrected Gaussian process where the source run has one (2025 only).

    Its neighbours are the four nearest probes; the 2026 run did not include
    it at these lengths, so there it is simply absent.
    """
    from scripts.v2_gap_climatology import arm_dirs, load_predictions, read_arm
    run, results = source.split('/')
    if (variable, 'gpr') in arm_dirs(run, results):
        gpr = load_predictions(run, results, variable, 'gpr')[['episode', 'position', 'gpr']]
    elif source == SOURCES['2026']:
        # Addendum 33 ran the Gaussian process on the identical 2026 episodes; addendum 59 reran it
        # with the mean fitted per pattern of reporting neighbours (read through the replacement).
        gpr = read_arm(*GPR_2026.split('/'), variable, 'gpr', usecols=['episode', 'position', 'gpr'])
    else:
        return frame
    out = frame.merge(gpr, on=['episode', 'position'], how='left', validate='one_to_one')
    if out.gpr.isna().all():
        raise SystemExit(f'{source}: the Gaussian process covers none of these rows')
    return out


def by_twin_and_length(frame):
    """The addendum-30 stratification by length: where a twin wins and where it stops mattering.

    `nearest_probe` on a twin probe mixes the twin's reading with persistence; the twin itself is
    scored where it reports, with the other methods on those rows beside it.
    """
    columns = ('linear', 'nearest_probe', 'chronos_bidir') + (('gpr',) if 'gpr' in frame else ())
    out = {}
    for (twin, length), part in frame.groupby(['has_twin', 'gap_length']):
        row = {'episodes': int(part.episode.nunique()), 'probes': int(part.sensor.nunique()),
               **{c: macro(part, c) for c in columns}}
        if twin:
            live = part[part.twin_reports]
            row['share_of_rows_where_twin_reports'] = round(float(part.twin_reports.mean()), 3)
            row['where_twin_reports'] = {
                'episodes': int(live.episode.nunique()), 'probes': int(live.sensor.nunique()),
                **{c: macro(live, c) for c in ('twin',) + columns if c != 'nearest_probe'}}
        out.setdefault('twin' if twin else 'no twin', {})[str(int(length))] = row
    return out


def twin_by_length(frame, year):
    """The twin, where it reports, against linear and the Gaussian process at each length."""
    cluster_of = clusters(frame)
    out = {}
    for length, part in frame.groupby('gap_length'):
        row = {}
        for reference in ('linear', 'gpr', 'chronos_bidir'):
            if reference in part and part[reference].notna().any():
                row[f'twin - {reference}'] = paired(
                    part, 'twin', reference, f'twin|{year}|twin|{reference}|{int(length)}',
                    cluster_of)
        out[str(int(length))] = row
    return out


def load(year):
    source = SOURCES[year]
    frame = fix_neighbours(pd.read_csv(RUNS / source / 'temperature_chronos_bidir/predictions.csv.gz'),
                           source.split('/')[0], 'temperature')
    frame = with_gpr(frame, source, 'temperature')
    return with_twin(frame, 'temperature', PERIODS[year])


def nearest_live_probe(variable, source):
    """Is the neighbour comparand weak only because its neighbour is often dark?

    `nearest_probe` reads the single geometrically nearest probe and falls back to persistence when
    that probe is not reporting. Without a twin that fallback fires on most rows, so a poor result
    could mean an offline neighbour rather than an uninformative one. This takes, at each scored
    hour, the nearest probe that is reporting both at the gap start and at that hour, and scores it
    on the declared 6-72 h band. Parameter-free; nothing is fitted.
    """
    import numpy as np
    from src.data.bay_profile_context import load_observations
    _, frames = load_observations()
    frame = fix_neighbours(pd.read_csv(RUNS / source / f'{variable}_chronos_bidir/predictions.csv.gz'),
                           source.split('/')[0], variable)
    frame = with_gpr(frame[frame.gap_length.between(*BAND)].copy(), source, variable)
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    values = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names},
                          index=index).to_numpy(float)
    geometry = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    xy = geometry.loc[names, ['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    column, where = {n: i for i, n in enumerate(names)}, {t: k for k, t in enumerate(index)}
    live, used = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for episode, block in frame.groupby('episode', sort=False):
        sensor, start, _ = episode.rsplit('|', 2)
        i, g0 = column[sensor], where[pd.Timestamp(start)]
        order = [j for j in np.argsort(distance[i], kind='stable') if j != i]
        rows = frame.index.get_indexer(block.index)
        for r, t in zip(rows, g0 + block.position.to_numpy(int) - 1):
            for j in order:
                if np.isfinite(values[g0 - 1, j]) and np.isfinite(values[t, j]):
                    live[r] = values[t, j] - values[g0 - 1, j] + values[g0 - 1, i]
                    used[r] = distance[i, j]
                    break
    frame['nearest_live_probe'] = live
    fallback = np.isclose(frame.nearest_probe, frame.persistence)
    out = {}
    for twin, part in frame.assign(fallback=fallback, used=used).groupby('has_twin'):
        scored = part.dropna(subset=['nearest_live_probe'])
        out['twin' if twin else 'no twin'] = {
            'episodes': int(scored.episode.nunique()),
            'rows_where_nearest_probe_is_persistence': round(float(part.fallback.mean()), 3),
            'rows_with_a_live_neighbour': round(float(part.nearest_live_probe.notna().mean()), 3),
            'median_distance_to_live_neighbour_m': round(float(np.nanmedian(part.used)), 0),
            **{c: macro(scored, c) for c in ('linear', 'nearest_probe', 'nearest_live_probe',
                                             'chronos_bidir') + (('gpr',) if 'gpr' in scored
                                                                 else ())}}
    return out


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum31.json',
              'rule': 'the twin reading where the probe has a co-located twin and it reports, '
                      'chronos_bidir otherwise',
              'twin': 'the twin reading, scored only where the twin reports at the gap edge and '
                      'at that hour',
              'variable': 'temperature', 'band_hours': list(BAND), 'sources': SOURCES,
              'gpr_2026_source': GPR_2026}
    frames = {year: load(year) for year in SOURCES}
    report['by_twin_and_length'] = {year: by_twin_and_length(f) for year, f in frames.items()}
    report['twin_by_length'] = {year: twin_by_length(f, year) for year, f in frames.items()}
    report['nearest_live_probe_check'] = {
        'period': 'development', 'band_hours': list(BAND),
        **{variable: nearest_live_probe(variable, SOURCES['2025'])
           for variable in ('temperature', 'oxygen')}}
    for year in SOURCES:
        frame = frames[year]
        frame = frame[frame.gap_length.between(*BAND)].copy()
        frame['twin_rule'] = frame.twin.where(frame.twin_reports, frame.chronos_bidir)
        cluster_of = clusters(frame)
        block = {}
        for arm, reference in CONTRASTS:
            if reference in frame and frame[reference].notna().any():
                block[f'{arm} - {reference}'] = paired(frame, arm, reference,
                                                       f'twin|{year}|{arm}|{reference}', cluster_of)
        report[year] = block
        print(f'== {year}')
        for key, e in block.items():
            star = '*' if e['excludes_zero'] else ' '
            print(f"   {key:<32} {e['delta']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]{star}"
                  f"  {e['episodes']} episodes in {e['clusters']} clusters")
    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))  # LF on every host
    print('wrote', OUT)


if __name__ == '__main__':
    main()
