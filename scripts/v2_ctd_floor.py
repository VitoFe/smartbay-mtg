"""A floor for targets that are CTD casts: how far two casts at one station on one day disagree.

Written 2026-09-26. Chapter 3 sets its measurement floors on the moored probes: the 0.029 degC
quantisation of the WNode loggers, the 0.20 to 0.53 degC disagreement between co-located probes and
the 1.8 umol/L cast-against-sonde figure for oxygen. The spatial estimate and the near-bottom
analysis are scored against CTD cast bins instead, and nothing had said how precisely a cast bin is
known as a target.

The archive answers it without a model. On most sampling days the operator lowered the profiler two
or three times at a station within a few minutes (the casts a, b, c of one day). A perfect estimate
of the water at that station and hour would still miss any one cast by about as much as two such
casts disagree, so the mean absolute difference between them over their shared 0.5 m bins is to a
cast-scored error what the co-located pair disagreement is to a probe-scored one. It includes the
profiler's own repeatability, the bin-averaging of two descents and whatever the water did in the
minutes between them; it cannot include the profiler's calibration error, which is common to the
casts of one day and to both arms of any comparison scored against them.

The bins are the ones the delivered loader keeps (qc_flag 0, the three targets finite and inside
the range gates). Each station-day's cast pairs are averaged first, so a day with three casts
counts once; intervals resample cast dates, the unit Chapter 4 uses for the spatial cohorts.
Cohorts: the whole archive (CTD-722), the Gen. 2 years before 2026 (from 24 July 2024, the period of
Field-57) and January to June 2026 (Field-2026).

The standard uncertainties the operator states for the profiler's point readings, in the headers of
the *-CTD+pH files (SBE 19plus readings at the sonde depths), are recorded beside it.

Descriptive and not declared in advance: no model is fitted and nothing is scored.
Writes reports/bay_twin/runs/v2_ctd_floor.json.
Usage: python -m scripts.v2_ctd_floor
"""
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASTS = ROOT / 'data/processed/ctd_all_casts.csv.gz'
RAW = ROOT / 'data/raw'
OUT = ROOT / 'reports/bay_twin/runs/v2_ctd_floor.json'
TARGETS = {'temperature': 'temp_c', 'salinity': 'salinity_psu', 'oxygen': 'do_umol_l'}
GEN2_START = pd.Timestamp('2024-07-24', tz='UTC')
YEAR_2026 = pd.Timestamp('2026-01-01', tz='UTC')
MIN_SHARED_BINS = 4          # the archive's own minimum number of strata per cast
REPLICATE_MIN = 30.0         # casts this close in time are treated as replicates
DRAWS, SEED = 5000, 20260926


def load():
    d = pd.read_csv(CASTS)
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    keep = ((d.qc_flag == 0) & np.isfinite(d[list(TARGETS.values())]).all(axis=1)
            & d.temp_c.between(-2, 40) & d.salinity_psu.between(0, 42) & d.do_umol_l.between(0, 600))
    d = d[keep].copy()
    d['start'] = d.groupby('cast_id').t_utc.transform('min')
    d['date'] = d.start.dt.tz_localize(None).dt.normalize()
    return d


def cast_pairs(d):
    """One row per pair of casts at one station on one day: mean |difference| over shared bins."""
    rows = []
    for (station, date), day in d.groupby(['station', 'date']):
        casts = {c: g.set_index('depth_m') for c, g in day.groupby('cast_id')}
        if len(casts) < 2:
            continue
        for a, b in combinations(sorted(casts), 2):
            shared = casts[a].index.intersection(casts[b].index)
            if len(shared) < MIN_SHARED_BINS:
                continue
            row = {'station': station, 'date': str(date.date()), 'a': a, 'b': b,
                   'start': casts[a].start.iloc[0],
                   'minutes_apart': abs((casts[b].start.iloc[0] - casts[a].start.iloc[0]).total_seconds()) / 60.0,
                   'shared_bins': int(len(shared))}
            for name, column in TARGETS.items():
                diff = casts[a].loc[shared, column].to_numpy() - casts[b].loc[shared, column].to_numpy()
                row[f'{name}_mean_abs'] = float(np.abs(diff).mean())
                row[f'{name}_bin_abs'] = np.abs(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(pairs, draws=DRAWS, seed=SEED):
    """Station-day means first, then over station-days; intervals resample cast dates."""
    if pairs.empty:
        return None
    days = pairs.groupby(['station', 'date'])[[f'{n}_mean_abs' for n in TARGETS]].mean().reset_index()
    out = {'cast_pairs': int(len(pairs)), 'station_days': int(len(days)),
           'cast_dates': int(days.date.nunique()),
           'stations': {k: int(v) for k, v in days.station.value_counts().sort_index().items()},
           'minutes_apart': {'median': round(float(pairs.minutes_apart.median()), 1),
                             'p90': round(float(pairs.minutes_apart.quantile(.9)), 1),
                             'max': round(float(pairs.minutes_apart.max()), 1)},
           'shared_bins_median': float(pairs.shared_bins.median())}
    by_date = {date: g for date, g in days.groupby('date')}
    keys = list(by_date)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(keys), len(keys)) for _ in range(draws)]
    for name in TARGETS:
        col = f'{name}_mean_abs'
        bins = np.concatenate(pairs[f'{name}_bin_abs'].to_list())
        entry = {'mean_over_station_days': float(days[col].mean()),
                 'median_over_station_days': float(days[col].median()),
                 'p90_over_station_days': float(days[col].quantile(.9)),
                 'median_abs_over_bins': float(np.median(bins))}
        if len(keys) >= 5:
            means, medians = [], []
            for draw in idx:
                x = np.concatenate([by_date[keys[i]][col].to_numpy() for i in draw])
                means.append(x.mean())
                medians.append(np.median(x))
            entry['mean_95_over_cast_dates'] = [float(np.quantile(means, .025)), float(np.quantile(means, .975))]
            entry['median_95_over_cast_dates'] = [float(np.quantile(medians, .025)), float(np.quantile(medians, .975))]
        out[name] = entry
    return out


def stated_uncertainties():
    """Standard uncertainties the operator writes into the headers of the profiler point series."""
    out = {}
    for path in sorted(RAW.glob('*-CTD+pH.txt')):
        text = path.read_text(encoding='cp1252')
        meta = {k.strip(): v.strip() for k, v in re.findall(r'^#\s*([^=\n]+)=(.*)$', text, re.M)}
        stated = {k: v for k, v in meta.items() if k.lower().startswith('incertezza')}
        if stated:
            out[path.name] = {'instrument': meta.get('Sonda'), **stated}
    return out


def main():
    d = load()
    pairs = cast_pairs(d)
    cohorts = {
        'CTD-722': pairs,
        'gen2_before_2026': pairs[(pairs.start >= GEN2_START) & (pairs.start < YEAR_2026)],
        'year_2026': pairs[pairs.start >= YEAR_2026],
    }
    report = {
        'question': 'A measurement floor for targets that are CTD cast bins.',
        'descriptive': 'Not declared in advance; no model fitted, nothing scored.',
        'source': 'data/processed/ctd_all_casts.csv.gz',
        'construction': 'casts at one station on one day, compared over shared 0.5 m bins that pass the '
                        'delivered loader filters; station-day means of cast-pair mean |difference|',
        'what_it_contains': 'profiler repeatability, bin averaging of two descents and the change of the water '
                            'in the minutes between them; not the calibration error common to both casts',
        'min_shared_bins': MIN_SHARED_BINS,
        'casts_used': int(d.cast_id.nunique()), 'bins_used': int(len(d)),
        'station_days_with_two_or_more_casts': int((d.groupby(['station', 'date']).cast_id.nunique() >= 2).sum()),
        'cohorts': {k: summarise(v) for k, v in cohorts.items()},
        'replicates_within_30_min': {k: summarise(v[v.minutes_apart <= REPLICATE_MIN]) for k, v in cohorts.items()},
        'operator_stated_standard_uncertainty': stated_uncertainties(),
    }
    OUT.write_text(json.dumps(report, indent=2))
    for label, block in (('all pairs', report['cohorts']), ('within 30 min', report['replicates_within_30_min'])):
        for cohort, s in block.items():
            if s is None:
                continue
            print(f"{label:14s} {cohort:18s} pairs={s['cast_pairs']:4d} days={s['station_days']:4d} "
                  f"dates={s['cast_dates']:4d} apart={s['minutes_apart']['median']:6.1f} min  "
                  + '  '.join(f"{n} {s[n]['mean_over_station_days']:.3f}/{s[n]['median_over_station_days']:.3f}"
                              f" {[round(x, 3) for x in s[n].get('mean_95_over_cast_dates', [])]}"
                              for n in TARGETS))
    print('stated:', report['operator_stated_standard_uncertainty'])
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
