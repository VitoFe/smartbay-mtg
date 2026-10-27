"""The held-site, held-sensor and seabed-feature transfer contrasts, with intervals that count draws.

`v2_held_station`, `v2_held_sensor` and `v2_bathy_compare` resample (fold, sensor) cells and then
take a macro average grouped by those same cells, so a cell drawn twice is merged back into one and
each draw is a subset of the cells rather than a bootstrap sample. This re-reads
their saved predictions and changes nothing in the modules or their artifacts.

For every contrast it reports these schemes:

  week blocks  ISO weeks of the issuance drawn with replacement jointly within each fold, every
               (fold, sensor) cell's endpoints of a drawn week entering together, cells fixed (a
               draw that leaves a cell empty is redrawn); the probe set is held fixed. Probes in one
               week share its weather, so drawing each cell's weeks on its own understates the
               variance of the macro. `week_blocks.within_cell` keeps that earlier scheme, ISO weeks
               resampled within each cell on their own, from the same seed, draw for draw
  sites        whole sites resampled with relabelling, so a site drawn twice counts twice; the
               question an operator adding a new site asks

For the held-site study it also reports the penalty with each site dropped in turn, and the margin
against the hour-only climatological increment: on these folds the month-by-hour table is fitted
on a single year and is weaker than persistence, so beating it says little.

Three additions (2026-09-26):

  * Chapter 4 refuses an interval on fewer than five independent units and, below about twenty,
    calls a result robust only where the bootstrap and a delete-one-unit jackknife-t both exclude
    zero. Every site-resampled contrast now carries that jackknife (`sites_jackknife`, t with k - 1
    degrees of freedom, centred on the point estimate, the convention of
    `scripts/v2_gap_intervals.py`), and a study with fewer than five sites gets no site interval
    at all (`sites` is marked not estimable).
  * The per-site penalty is computed with the same nested macro as the headline figure (mean over
    the site's (fold, sensor) cells, then over folds), not as an endpoint-pooled mean.
  * The hour-only climatological increment the held site is compared with is fitted on the held
    probe's own training rows, which a newly installed probe would not have. `clim_hour_other_sites`
    is the same lookup built only from the probes the held-site model trains on (the eligible
    probes at every other site, each probe's hourly mean increment, averaged over probes).

Usage: python -m scripts.v2_transfer_intervals
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_transfer_intervals.json'
DRAWS = 2000
MIN_SITES = 5
KEY = ['fold', 'sensor', 'time']


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def weeks(frame):
    return pd.to_datetime(frame.time, utc=True).dt.isocalendar().week.astype(int).to_numpy()


def joint_weeks(frame, x, y, label):
    """ISO weeks drawn with replacement jointly within each fold, from the same seed as
    `week_blocks`: every (fold, sensor) cell's endpoints of a drawn week enter together, and a fold
    draw that leaves a cell with no endpoint of an arm it is scored on is redrawn, so the cells are
    fixed. The macro of each draw is computed from per-(cell, week) sums of absolute error and
    counts, each arm on its own rows, as `macro` scores it. Returns the draws and the number of fold
    draws that were redrawn."""
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    ex, ey = (frame.truth - frame[x]).abs(), (frame.truth - frame[y]).abs()
    f = frame.assign(week=weeks(frame), ex=ex.fillna(0.0), ey=ey.fillna(0.0),
                     nx=ex.notna().astype(float), ny=ey.notna().astype(float))
    tables = []
    for _, fold in f.groupby('fold', sort=True):
        t = fold.groupby(['sensor', 'week'])[['ex', 'nx', 'ey', 'ny']].sum()
        weeks_ = sorted(t.index.get_level_values('week').unique())
        sensors = sorted(t.index.get_level_values('sensor').unique())
        grid = {c: t[c].unstack('week', fill_value=0.0).reindex(index=sensors, columns=weeks_, fill_value=0.0)
                .to_numpy() for c in ('ex', 'nx', 'ey', 'ny')}
        # a cell scored by one arm only enters that arm's macro, as in `macro`
        grid['has_x'] = grid['nx'].sum(axis=1) > 0
        grid['has_y'] = grid['ny'].sum(axis=1) > 0
        tables.append(grid)
    out = np.empty(DRAWS)
    redrawn = 0
    for i in range(DRAWS):
        fx, fy = [], []
        for grid in tables:
            w = grid['nx'].shape[1]
            for _ in range(10_000):
                m = np.bincount(rng.integers(0, w, w), minlength=w).astype(float)
                nx, ny = grid['nx'] @ m, grid['ny'] @ m
                if (nx[grid['has_x']] > 0).all() and (ny[grid['has_y']] > 0).all():
                    break
                redrawn += 1
            else:
                raise RuntimeError('no joint week draw keeps every cell')
            if grid['has_x'].any():
                fx.append(np.mean((grid['ex'] @ m)[grid['has_x']] / nx[grid['has_x']]))
            if grid['has_y'].any():
                fy.append(np.mean((grid['ey'] @ m)[grid['has_y']] / ny[grid['has_y']]))
        out[i] = np.mean(fx) - np.mean(fy)
    return out, redrawn


def weekly(point, frame, x, y, label):
    """The reported joint-week interval, with the within-cell one beside it."""
    joint, redrawn = joint_weeks(frame, x, y, label)
    out = summary(point, joint)
    out['joint_redrawn'] = int(redrawn)
    within = summary(point, week_blocks(frame, x, y, label))
    out['within_cell'] = {'ci': within['ci'], 'excludes_zero': within['excludes_zero']}
    return out


def week_blocks(frame, x, y, label):
    """ISO weeks resampled within each (fold, sensor) cell on their own (the earlier scheme).

    The draws are those of the original concat-and-macro loop (same generator, same call order);
    the macro of each draw is computed from per-week sums of absolute error and counts, which is
    the same arithmetic without rebuilding a frame per draw.
    """
    rng = np.random.default_rng([20260923, zlib.crc32(label.encode())])
    f = frame.assign(week=weeks(frame))
    cells = []
    for (fold, _), cell in f.groupby(['fold', 'sensor']):
        per_week = []
        for _, g in cell.groupby('week'):
            ex, ey = (g.truth - g[x]).abs(), (g.truth - g[y]).abs()
            per_week.append((ex.sum(), ex.notna().sum(), ey.sum(), ey.notna().sum()))
        cells.append((fold, np.array(per_week, dtype=float)))
    folds_in = sorted({c[0] for c in cells})
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        cell_x = {k: [] for k in folds_in}
        cell_y = {k: [] for k in folds_in}
        for fold, table in cells:
            s = table[rng.integers(0, len(table), len(table))].sum(axis=0)
            if s[1] > 0:
                cell_x[fold].append(s[0] / s[1])
            if s[3] > 0:
                cell_y[fold].append(s[2] / s[3])
        mx = np.mean([np.mean(v) for v in cell_x.values() if v])
        my = np.mean([np.mean(v) for v in cell_y.values() if v])
        out[i] = mx - my
    return out


def by_site(frame, x, y, label, sites):
    """Whole sites resampled with relabelling; None below MIN_SITES, where no interval is reported."""
    names = sorted(frame[sites].unique())
    if len(names) < MIN_SITES:
        return None
    rng = np.random.default_rng([20260924, zlib.crc32(label.encode())])
    groups = {s: g for s, g in frame.groupby(sites)}
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        picked = [groups[names[j]].assign(sensor=groups[names[j]].sensor + f'#{n}')
                  for n, j in enumerate(rng.integers(0, len(names), len(names)))]
        s = pd.concat(picked)
        out[i] = macro(s, x) - macro(s, y)
    return out


def jackknife_sites(frame, x, y, point, sites):
    """Delete-one-site jackknife-t, centred on the full-sample point; None below MIN_SITES."""
    names = sorted(frame[sites].unique())
    k = len(names)
    if k < MIN_SITES:
        return None
    theta = np.array([macro(frame[frame[sites] != s], x) - macro(frame[frame[sites] != s], y)
                      for s in names])
    se = float(np.sqrt((k - 1) / k * ((theta - theta.mean()) ** 2).sum()))
    half = float(student.ppf(0.975, k - 1)) * se
    return {'ci': [round(point - half, 4), round(point + half, 4)],
            'excludes_zero': bool(point - half > 0 or point + half < 0),
            'leave_one_out': {s: round(float(v), 4) for s, v in zip(names, theta)}}


def summary(point, draws, units=None):
    if draws is None:
        return {'point': round(point, 4), 'ci': None, 'excludes_zero': None,
                'not_estimable': f'{units} sites, fewer than {MIN_SITES}'}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'point': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0)}


def site_schemes(frame, x, y, label, point):
    """The site bootstrap and the site jackknife, and whether both exclude zero."""
    units = int(frame.site.nunique())
    boot = summary(point, by_site(frame, x, y, label, 'site'), units)
    jack = jackknife_sites(frame, x, y, point, 'site')
    out = {'sites': boot}
    if jack is None:
        out['sites_jackknife'] = {'ci': None, 'excludes_zero': None,
                                  'not_estimable': f'{units} sites, fewer than {MIN_SITES}'}
        out['robust_over_sites'] = None
    else:
        out['sites_jackknife'] = jack
        out['robust_over_sites'] = bool(boot['excludes_zero'] and jack['excludes_zero'])
    return out


def hour_only_climatology(variable, sites=None, geometry=None):
    """The hour-only climatological increment at every (fold, sensor, time) of the forecast folds.

    With `sites` (sensor -> site) it also returns `clim_hour_other_sites`: for each held site, the
    hourly mean increment of every eligible probe at the other sites, from the fold's training rows,
    averaged over those probes. That is the lookup available to a probe with no record of its own,
    built from exactly the probes the held-site model is trained on. A `geometry` dict is filled
    with, for each site, the distance from each held probe to the nearest probe outside its 250 m
    mask, and to the nearest such probe that reports at least once in the scored fold.
    """
    from src.experiments.v2_horizon import build, climatology, folds
    data = build(variable, 6)
    t = data['times']
    increment = data['truth'] - data['x']
    site_of_probe = None if sites is None else np.array([sites[n] for n in data['names']])
    rows = []
    for fold in folds(data, 6):
        _, hour_only = climatology(data, fold)
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        others = {}
        if site_of_probe is not None and geometry is not None:
            reporting = data['available'][fold['va']].any(axis=0)
            for s in np.flatnonzero(fold['eligible']):
                d = data['distance'][s]
                outside = d > 250.0
                g = geometry.setdefault(str(site_of_probe[s]), {'nearest_surviving_m': set(),
                                                                'nearest_reporting_m': set()})
                g['nearest_surviving_m'].add(round(float(d[outside].min()), 1))
                g['nearest_reporting_m'].add(round(float(d[outside & reporting].min()), 1))
        if site_of_probe is not None:
            tr = fold['tr']
            hours = t.hour.to_numpy()[tr]
            for site in pd.unique(site_of_probe[fold['eligible']]):
                keep = fold['eligible'] & (site_of_probe != site)
                per_probe = []
                for s in np.flatnonzero(keep):
                    ok = data['valid'][tr, s]
                    per_probe.append(pd.Series(increment[tr, s][ok]).groupby(hours[ok]).mean())
                table = pd.concat(per_probe, axis=1).mean(axis=1)
                others[site] = table
        for j, s in np.argwhere(mask):
            i = fold['va'][j]
            row = {'fold': fold['number'], 'sensor': data['names'][s], 'time': str(t[i]),
                   'clim_hour': float(hour_only[i, s] * data['scale'] + data['center'])}
            if site_of_probe is not None:
                step = others[site_of_probe[s]].get(t[i].hour, np.nan)
                row['clim_hour_other_sites'] = float((data['x'][i, s] + step) * data['scale']
                                                     + data['center'])
            rows.append(row)
    return pd.DataFrame(rows)


def site_of():
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    return meta.site.to_dict()


def main():
    report = {'draws': DRAWS,
              'min_sites': MIN_SITES,
              'site_rule': 'no site interval below five sites; robust_over_sites = the site '
                           'bootstrap and the delete-one-site jackknife-t both exclude zero',
              'studies': {}}
    sites = site_of()
    for variable in ('temperature', 'oxygen'):
        nested = RUNS / f'v2_held_station/held_station_{variable}_h6_nested.csv.gz'
        # Temperature has a nested selection per (fold, held site); oxygen cannot (one fold's inner
        # window holds only the held sonde), so its arm keeps the all-site configuration.
        station = pd.read_csv(nested if nested.exists() else
                              RUNS / f'v2_held_station/held_station_{variable}_h6.csv.gz')
        geometry = {}
        station = station.merge(hour_only_climatology(variable, sites, geometry), on=KEY,
                                how='left', validate='one_to_one')
        block = {'endpoints': int(len(station)), 'sites': int(station.site.nunique()),
                 'sensors': int(station.sensor.nunique()), 'nested_selection': nested.exists(),
                 'site_geometry': {site: {k: sorted(v) for k, v in g.items()}
                                   for site, g in sorted(geometry.items())
                                   if site in set(station.site)},
                 'contrasts': {}}
        held_arm = 'held_station_nested' if nested.exists() else 'held_station'
        pairs = [('held_station', 'all_sensor'), ('held_station', 'persistence'),
                 ('held_station', 'climatology'), ('held_station', 'clim_hour'),
                 ('persistence', 'clim_hour')]
        if nested.exists():
            pairs += [('held_station_nested', 'all_sensor'), ('held_station_nested', 'persistence'),
                      ('held_station_nested', 'clim_hour'), ('held_station_nested', 'held_station')]
        # The lookup a new probe could actually use, built from the other sites only.
        pairs += [(held_arm, 'clim_hour_other_sites'), ('clim_hour_other_sites', 'clim_hour'),
                  ('persistence', 'clim_hour_other_sites')]
        for x, y in pairs:
            label = f'station|{variable}|{x}|{y}'
            point = macro(station, x) - macro(station, y)
            block['contrasts'][f'{x} - {y}'] = {
                'week_blocks': weekly(point, station, x, y, label),
                **site_schemes(station, x, y, label, point)}
        arms = ['all_sensor', 'held_station', 'persistence', 'climatology', 'clim_hour',
                'clim_hour_other_sites'] + (['held_station_nested'] if nested.exists() else [])
        block['mae'] = {a: round(macro(station, a), 4) for a in arms}
        block['penalty_dropping_each_site'] = {
            site: round(macro(station[station.site != site], held_arm)
                        - macro(station[station.site != site], 'all_sensor'), 4)
            for site in sorted(station.site.unique())}
        # Same nested macro as the headline penalty: (fold, sensor) cells, then folds.
        block['penalty_per_site'] = {
            site: round(macro(g, held_arm) - macro(g, 'all_sensor'), 4)
            for site, g in station.groupby('site')}
        block['penalty_per_site_rule'] = 'nested macro within the site: (fold, sensor), then fold'
        report['studies'][f'held_station_{variable}'] = block

        held = pd.read_csv(RUNS / f'v2_held_sensor/held_sensor_{variable}_h6.csv.gz')
        bathy = pd.read_csv(RUNS / f'v2_held_sensor/held_sensor_{variable}_h6_bathy.csv.gz')
        merged = held.merge(bathy[KEY + ['all_sensor', 'held_sensor']], on=KEY,
                            suffixes=('', '_bathy'), validate='one_to_one')
        merged['site'] = merged.sensor.map(sites)
        block = {'endpoints': int(len(merged)), 'sites': int(merged.site.nunique()),
                 'contrasts': {}}
        for x, y in (('held_sensor', 'all_sensor'), ('all_sensor_bathy', 'all_sensor'),
                     ('held_sensor_bathy', 'held_sensor'), ('held_sensor_bathy', 'all_sensor_bathy')):
            label = f'sensor|{variable}|{x}|{y}'
            point = macro(merged, x) - macro(merged, y)
            block['contrasts'][f'{x} - {y}'] = {
                'week_blocks': weekly(point, merged, x, y, label),
                **site_schemes(merged, x, y, label, point)}
        report['studies'][f'held_sensor_{variable}'] = block
        for name in (f'held_station_{variable}', f'held_sensor_{variable}'):
            for key, e in report['studies'][name]['contrasts'].items():
                print(f"{name:26s} {key:44s} {e['week_blocks']['point']:+.4f}  weeks "
                      f"{e['week_blocks']['ci']}  sites {e['sites']['ci']}  "
                      f"jackknife {e['sites_jackknife']['ci']}", flush=True)
    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
