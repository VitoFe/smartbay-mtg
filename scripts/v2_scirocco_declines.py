"""Addendum 62: do the water-mass declines follow Scirocco (south-easterly) wind?

The episodes are addendum 37's (scripts/v2_decline_types.episodes), unchanged. The wind is the Santa
Teresa station's direction (degrees, from) and vector speed, projected per 10-minute record on the
south-east (135 degrees) and averaged to the hour. For each window W the mean of that component over
the W hours ending at an onset is taken as an anomaly from its month x hour median over the
development period.

    S1  water-mass anomaly positive in median and larger than in-place (one-sided Mann-Whitney)
    S2  water-mass mean anomaly against a circular shift of the onset calendar (2,000 shifts)

Declared in config/bay_reserved_protocol_v2_addendum62.json; the decision on the forecasting
expansion is S1 or S2 at W = 24 h.

  python -m scripts.v2_scirocco_declines   ->  reports/bay_twin/runs/v2_scirocco_declines.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from scripts.v2_decline_types import DEV, RESERVED, episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_scirocco_declines.json'
METEO = ROOT / 'data/raw/T2-Meteo.txt'
DECLARED = 'config/bay_reserved_protocol_v2_addendum62.json'
SCIROCCO_DEG = 135.0
WINDOWS = (24, 12, 48)
PRIMARY = 24
SHIFTS, SEED = 2000, 20261003


def scirocco_hourly():
    """Hourly mean of speed x cos(direction - 135 deg) from the 10-minute records (m/s, + from SE)."""
    with open(METEO, encoding='cp1252') as stream:
        head = stream.read(40000)
    names = next(l.lstrip('#').split() for l in head.splitlines() if l.startswith('#') and 'Data' in l and 'Ora' in l)
    f = pd.read_csv(METEO, comment='#', sep=r'\s+', header=None, encoding='cp1252', low_memory=False)
    f.columns = names[:f.shape[1]]
    t = pd.to_datetime(f.iloc[:, 0].astype(str) + ' ' + f.iloc[:, 1].astype(str),
                       format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    direction = pd.to_numeric(f['Direzione_del_Vento(°)'], errors='coerce')
    speed = pd.to_numeric(f["Velocita'_del_Vento(m/s)"], errors='coerce')
    ok = direction.between(0, 360) & speed.between(0, 60)
    s = pd.Series(np.where(ok, speed * np.cos(np.radians(direction - SCIROCCO_DEG)), np.nan), index=t)
    s = s[s.index.notna()].sort_index().groupby(level=0).mean()
    # which way the direction points: in summer the wind should come from the land (north) at night and
    # from the sea (south) in the afternoon, the sea breeze of a gulf that opens to the south
    summer = ok & (speed >= 0.5) & t.dt.month.isin([6, 7, 8]).to_numpy()
    hour = t.dt.hour.to_numpy()
    check = {}
    for label, hours in (('night_00_06', range(0, 7)), ('afternoon_12_16', range(12, 17))):
        x = np.radians(direction[summer & np.isin(hour, list(hours))].to_numpy())
        check[label] = round(float(np.degrees(np.arctan2(np.sin(x).mean(), np.cos(x).mean())) % 360), 1)
    return s.resample('1h', closed='left', label='right').mean(), int(ok.sum()), int(len(f)), check


def anomaly_series(hourly, window):
    """Windowed mean ending at each hour, minus its month x hour median over the development period."""
    w = hourly.rolling(window, min_periods=int(0.75 * window)).mean()
    dev = w.loc[DEV[0]:'2025-12-31 23:00']
    ref = dev.groupby([dev.index.month, dev.index.hour]).median()
    keys = list(zip(w.index.month, w.index.hour))
    return w - np.array([ref.get(k, np.nan) for k in keys])


def at(series, times):
    return np.array([series.get(t, np.nan) for t in times], float)


def circular_null(series, onsets, rng):
    """Mean anomaly over the onsets with every onset moved by one common offset inside the period."""
    period = series.loc[DEV[0]:'2025-12-31 23:00']
    index = period.index
    n = len(index)
    pos = np.array([index.get_loc(t) for t in onsets if t in index])
    values = period.to_numpy()
    out = np.empty(SHIFTS)
    for i in range(SHIFTS):
        k = rng.integers(1, n)
        out[i] = np.nanmean(values[(pos + k) % n])
    return out


def r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def main():
    hourly, good, records, convention = scirocco_hourly()
    dev, res = episodes(*DEV), episodes(*RESERVED)
    rng = np.random.default_rng(SEED)
    report = {'declared_in': DECLARED, 'scirocco_direction_deg': SCIROCCO_DEG,
              # the direction is where the wind blows from: the June-August mean direction at night and in
              # the afternoon, below, points off the land and off the sea respectively
              'direction_convention': {'reading': 'from', 'summer_mean_direction_deg': convention},
              'records': {'total': records, 'valid_direction_and_speed': good},
              'episodes': {'development': {k: int(v) for k, v in dev.type.value_counts().items()},
                           'reserved': {k: int(v) for k, v in res.type.value_counts().items()}},
              'windows': {}}
    for window in WINDOWS:
        a = anomaly_series(hourly, window)
        wm = dev[dev.type == 'water_mass'].onset
        ip = dev[dev.type == 'in_place'].onset
        awm, aip = at(a, wm), at(a, ip)
        p1 = mannwhitneyu(awm[np.isfinite(awm)], aip[np.isfinite(aip)], alternative='greater').pvalue
        s1 = {'water_mass_median': r(np.nanmedian(awm)), 'in_place_median': r(np.nanmedian(aip)),
              'n': [int(np.isfinite(awm).sum()), int(np.isfinite(aip).sum())], 'p_one_sided': r(p1)}
        s1['held'] = bool(s1['water_mass_median'] > 0 and p1 < .05)
        null = circular_null(a, list(wm), rng)
        observed = float(np.nanmean(awm))
        p2 = (1 + int((null >= observed).sum())) / (1 + SHIFTS)
        s2 = {'observed_mean': r(observed), 'null_mean': r(null.mean()), 'null_95th': r(np.quantile(null, .95)),
              'p_one_sided': r(p2), 'held': bool(p2 < .05)}
        # Not declared: the same statistics read in the opposite direction (less south-easterly wind than
        # usual), reported beside the declared tests because the observed values fall on that side.
        opposite = {'not_declared': True,
                    'S1_p_less': r(mannwhitneyu(awm[np.isfinite(awm)], aip[np.isfinite(aip)], alternative='less').pvalue),
                    'S2_p_lower_tail': r((1 + int((null <= observed).sum())) / (1 + SHIFTS)),
                    'null_5th': r(np.quantile(null, .05))}
        reserved = [{'sensor': row.sensor.split('_')[0], 'onset': str(row.onset)[:16], 'type': row.type,
                     'anomaly': r(a.get(row.onset, np.nan), 3)} for row in res.itertuples()]
        report['windows'][str(window)] = {'S1': s1, 'S2': s2, 'opposite_direction': opposite,
                                          'reserved_2026': reserved,
                                          'water_mass_values': [r(x, 3) for x in awm],
                                          'in_place_values': [r(x, 3) for x in aip]}
    primary = report['windows'][str(PRIMARY)]
    report['decision'] = {'primary_window_h': PRIMARY,
                          'pursue_forecasting_expansion': bool(primary['S1']['held'] or primary['S2']['held'])}
    OUT.write_bytes((json.dumps(report, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    for w, v in report['windows'].items():
        print(f"W={w:>2} h  S1: water-mass {v['S1']['water_mass_median']:+.3f} vs in-place {v['S1']['in_place_median']:+.3f} "
              f"m/s, p={v['S1']['p_one_sided']}  |  S2: observed {v['S2']['observed_mean']:+.3f}, null 95th "
              f"{v['S2']['null_95th']:+.3f}, p={v['S2']['p_one_sided']}")
    print('decision:', report['decision'])


if __name__ == '__main__':
    main()
