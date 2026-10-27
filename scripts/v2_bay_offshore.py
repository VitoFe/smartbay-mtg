"""The inner bay's surface against the offshore column at PBoaT (Tinetto buoy), from the casts.

Raised by the supervisors on 2026-10-02: in winter the bay looks colder than the open water outside
the breakwater. Descriptive: no model is fitted and nothing is scored.

Rule. The surface layer is the mean of the quality-controlled bins from 0 to 1.25 m of a cast (the
layer of Table tab:bay_ctd_stratification). Casts of one station on one UTC day are averaged into one
cast date, the independent unit (casts minutes apart share the water). PBoaT is cast about monthly
and the inner stations weekly, so each PBoaT cast date is paired with the nearest cast date of the
inner station within 3 days, and differences are inner minus PBoaT. Seasons are DJF, MAM, JJA, SON.
Per season: the number of pairs, the median difference with a 95 % percentile bootstrap interval
over pairs (2,000 draws, seed 20261002), the interquartile range, and the share of pairs in which
the bay is colder (fresher). Sensitivity: a 1-day window, and PBoaS and PMitiliE in place of PBaia.

  python -m scripts.v2_bay_offshore   ->  reports/bay_twin/runs/v2_bay_offshore.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASTS = ROOT / 'data/processed/ctd_all_casts.csv.gz'
OUT = ROOT / 'reports/bay_twin/runs/v2_bay_offshore.json'
SURFACE_M = 1.25
OFFSHORE = 'PBoaT'
INNER = ('PBaia', 'PBoaS', 'PMitiliE')
WINDOWS_DAYS = (3, 1)
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
          6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
DRAWS, SEED = 2000, 20261002


def surface_by_date(casts, station):
    """Surface temperature and salinity per cast date of one station."""
    s = casts[(casts.station == station) & (casts.depth_m <= SURFACE_M)]
    per_cast = s.groupby('cast_id').agg(t=('t_utc', 'first'), T=('temp_c', 'mean'), S=('salinity_psu', 'mean'))
    per_cast['date'] = per_cast.t.dt.floor('D')
    return per_cast.groupby('date')[['T', 'S']].mean().sort_index()


def pairs(inner, offshore, days):
    """Each offshore cast date with the nearest inner cast date within `days` days."""
    o = offshore.reset_index().rename(columns={'date': 'date_offshore'})
    i = inner.reset_index().rename(columns={'date': 'date_inner'})
    m = pd.merge_asof(o.sort_values('date_offshore'), i.sort_values('date_inner'),
                      left_on='date_offshore', right_on='date_inner', direction='nearest',
                      tolerance=pd.Timedelta(days=days), suffixes=('_offshore', '_inner'))
    m = m.dropna(subset=['T_inner', 'T_offshore'])
    m['dT'] = m.T_inner - m.T_offshore
    m['dS'] = m.S_inner - m.S_offshore
    m['season'] = m.date_offshore.dt.month.map(SEASON)
    return m


def haversine_m(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a.lat, a.lon, b.lat, b.lon))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6_371_000 * np.arcsin(np.sqrt(h)))


def summary(values, rng):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    boot = np.median(rng.choice(v, size=(DRAWS, len(v)), replace=True), axis=1)
    return {'n': int(len(v)), 'median': round(float(np.median(v)), 4),
            'ci95': [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
            'iqr': [round(float(np.quantile(v, .25)), 4), round(float(np.quantile(v, .75)), 4)],
            'share_negative': round(float((v < 0).mean()), 4),
            'n_negative': int((v < 0).sum())}


def main():
    casts = pd.read_csv(CASTS, parse_dates=['t_utc'])
    casts = casts[casts.qc_flag == 0]
    offshore = surface_by_date(casts, OFFSHORE)
    rng = np.random.default_rng(SEED)
    out = {'question': 'Is the inner bay\'s surface colder or fresher than the offshore column at PBoaT '
                       '(Tinetto buoy), by season? (supervisors, 2026-10-02)',
           'descriptive': 'No model fitted, nothing scored.',
           'rule': {'surface_layer_m': [0, SURFACE_M], 'unit': 'cast date (casts of one station on one '
                    'UTC day averaged)', 'pairing': 'each PBoaT cast date with the nearest inner cast date '
                    'within the window', 'difference': 'inner minus PBoaT', 'bootstrap': {'draws': DRAWS,
                    'seed': SEED, 'over': 'pairs'}, 'qc': 'qc_flag == 0 bins only'},
           'offshore_cast_dates': int(len(offshore)), 'results': {}}
    # where the stations are: median cast position and recorded seabed, and the distance between the
    # two inner stations that differ in winter
    pos = casts.groupby('station')[['lat', 'lon', 'seabed_m']].median()
    out['stations'] = {s: {'lat': round(float(pos.lat[s]), 5), 'lon': round(float(pos.lon[s]), 5),
                           'seabed_m': float(pos.seabed_m[s])} for s in (OFFSHORE,) + INNER}
    out['distance_m'] = {'PBoaS-PBaia': round(haversine_m(pos.loc['PBoaS'], pos.loc['PBaia'])),
                         'PBaia-PBoaT': round(haversine_m(pos.loc['PBaia'], pos.loc[OFFSHORE]))}
    for station in INNER:
        inner = surface_by_date(casts, station)
        for days in WINDOWS_DAYS:
            m = pairs(inner, offshore, days)
            key = f'{station}|{days}d'
            res = {'pairs': int(len(m)), 'inner_cast_dates': int(len(inner)), 'by_season': {}}
            for season in ('DJF', 'MAM', 'JJA', 'SON'):
                g = m[m.season == season]
                res['by_season'][season] = {'temperature_c': summary(g.dT, rng),
                                            'salinity': summary(g.dS, rng)}
            if station == 'PBaia' and days == 3:
                res['pairs_list'] = [{'offshore_date': str(r.date_offshore.date()),
                                      'inner_date': str(r.date_inner.date()),
                                      'dT': round(float(r.dT), 4), 'dS': round(float(r.dS), 4)}
                                     for r in m.itertuples()]
            out['results'][key] = res
    OUT.write_bytes((json.dumps(out, indent=1) + '\n').encode('utf-8'))
    for key, res in out['results'].items():
        line = ', '.join(f"{s} n={v['temperature_c']['n']} dT {v['temperature_c']['median']:+.2f} "
                         f"({v['temperature_c']['n_negative']} colder)"
                         for s, v in res['by_season'].items() if v['temperature_c'])
        print(f'{key:14s} {line}')
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
