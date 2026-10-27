"""Addendum-9: how reproducible is an oxygen value in this network?

Chapter 3 gives temperature two independent floors, a quantisation step and an empirical
co-located reproducibility over eleven probe pairs, and Chapter 5 applies them. Oxygen has
neither, because no two oxygen probes share a site, so every oxygen contrast in the thesis has
been reported without an interpretive floor beside it.

There is a second co-location in this dataset that has not been used: each of the five EXO2
oxygen sondes sits at a CTD station. Pairing a cast's depth bin against the sonde's own reading
at the same hour measures how far two instruments disagree about the same water, which is the
quantity the co-located probe construction supplies for temperature.

The pair compounds two instruments, a depth-bin mismatch and a time offset, and none of those can
be separated here, so as a figure for one sonde it bounds reproducibility from above. Temperature
is run through the identical construction at the same sites and the same pairs, so the
temperature figure says how much of the disagreement belongs to the pairing method rather than to
the oxygen channel. The declaration fixed the median as the statistic and read a pooled median
below 2 umol/L as a figure that "can be read as a floor directly".

Three checks added on 2026-09-26 leave the declared figures untouched:
  * independence: the pairs fall on fewer cast dates than there are pairs, and some casts share
    one sonde hour, so the dates are counted and the pooled statistics are bootstrapped over them;
  * timing: the sonde series is matched through its hourly values, labelled at the END of the
    hour, so the matched reading is taken before the cast; the offsets are recorded, and the
    construction is repeated against the nearest raw reading in true time;
  * calibration: every sonde export is an operator-processed file named "CTD(calibrato)" whose
    header names the operator and the processing but no calibration reference or date, so whether
    the sondes were adjusted to the casts is not known from the archive.

Declared in config/bay_reserved_protocol_v2_addendum9.json.
Usage: python -m src.experiments.v2_oxygen_floor
"""
import json
import re

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT

OUT = ROOT / 'reports/bay_twin/runs/v2_oxygen_floor.json'
DEPTH_TOLERANCE_M = 1.0
WINDOWS_MIN = [30, 60]
BOOTSTRAP_DRAWS = 5000
SEED = 20260926
#: addendum 9's falsifiers: a pooled oxygen median below this, or a temperature control below the
#: lower end of the co-located band
PREDICTED_OXYGEN_ABOVE = 2.0
PREDICTED_TEMPERATURE_ABOVE = 0.20

# Each oxygen sonde and the CTD station occupying the same site.
SITES = {
    'TBaiaP_01m-CTD(calibrato)': 'PBaia',
    'TMitiliE_05m-CTD(calibrato)': 'PMitiliE',
    'TMitiliO_05m-CTD(calibrato)': 'PMitiliO',
    'TPortovenereP_02m-CTD(calibrato)': 'PPortovenereP',
    'TPortovenereS_02m-CTD(calibrato)': 'PPortovenereS',
}
VARIABLES = {'oxygen': 'do_umol_l', 'temperature': 'temp_c'}


def casts():
    d = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz')
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    return d[d.qc_flag == 0]


def pair_site(profiles, sonde, station, window_min, drop_2026):
    """One row per (cast, variable): the cast bin at the sonde's depth against the sonde reading."""
    meta, frame, _ = read_export(ROOT / 'data/raw' / f'{sonde}.txt')
    nominal = float(meta["Profondita'"])
    site = profiles[profiles.station == station]
    if drop_2026:
        site = site[site.t_utc < pd.Timestamp('2026-01-01', tz='UTC')]
    rows = []
    for cast_id, group in site.groupby('cast_id'):
        near = group.loc[(group.depth_m - nominal).abs() <= DEPTH_TOLERANCE_M]
        if near.empty:
            continue
        near = near.loc[(near.depth_m - nominal).abs().idxmin()]
        # The sonde series is labelled by the END of its observation hour.
        delta = (frame.index - near.t_utc).total_seconds() / 60.0
        within = np.flatnonzero(np.abs(delta) <= window_min)
        if not len(within):
            continue
        j = within[np.argmin(np.abs(delta[within]))]
        row = {'sonde': sonde, 'station': station, 'cast_id': cast_id,
               'cast_depth_m': float(near.depth_m), 'sonde_depth_m': nominal,
               'offset_min': float(delta[j]), 't_utc': str(near.t_utc)}
        for name, column in VARIABLES.items():
            if name not in frame.columns:
                continue
            a, b = float(near[column]), float(frame[name].iloc[j])
            if np.isfinite(a) and np.isfinite(b):
                row[f'{name}_cast'], row[f'{name}_sonde'] = a, b
                row[f'{name}_diff'] = a - b
        rows.append(row)
    return pd.DataFrame(rows)


def describe(frame, name):
    column = f'{name}_diff'
    if column not in frame or frame[column].notna().sum() == 0:
        return None
    d = frame[column].dropna()
    return {'n': int(len(d)), 'median_abs': float(d.abs().median()),
            'mean_abs': float(d.abs().mean()), 'mean_signed': float(d.mean()),
            'iqr_signed': [float(d.quantile(.25)), float(d.quantile(.75))],
            'p90_abs': float(d.abs().quantile(.9))}


def raw_readings(sonde):
    """The sonde's readings at their own timestamps, screened as read_export screens them."""
    path = ROOT / 'data/raw' / f'{sonde}.txt'
    frame = pd.read_csv(path, comment='#', sep=r'\s+', header=None, encoding='cp1252')
    t = pd.to_datetime(frame[0] + ' ' + frame[1], format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    out = pd.DataFrame({'temperature': pd.to_numeric(frame[4], errors='coerce').to_numpy(),
                        'oxygen': pd.to_numeric(frame[13], errors='coerce').to_numpy()},
                       index=pd.DatetimeIndex(t))
    out = out.loc[out.index.notna()].sort_index()
    out['temperature'] = out.temperature.where(out.temperature.between(-2, 40))
    out['oxygen'] = out.oxygen.where(out.oxygen.between(0, 600))
    return out.groupby(level=0).mean()


def header(sonde):
    text = (ROOT / 'data/raw' / f'{sonde}.txt').read_text(encoding='cp1252')
    meta = {k.strip(): v.strip() for k, v in re.findall(r'^#\s*([^=\n]+)=(.*)$', text, re.M)}
    return {'file': f'data/raw/{sonde}.txt', 'processing': meta.get('Sonda'),
            'operator_recorded': 'Operatore' in meta,
            'calibration_fields': sorted(k for k in meta if 'calib' in k.lower() or 'incertezza' in k.lower())}


def timing(pairs, raw):
    """Where the matched sonde reading sits in time, and the construction in true time.

    The hourly value labelled L averages the readings in [L - 1 h, L); the recorded offset_min is
    L minus the cast time. Matching instead on the nearest raw reading within a window gives the
    construction the declaration describes without the hourly labelling in between."""
    offsets = []
    oxygen_pairs = pairs.dropna(subset=['oxygen_diff'])
    for _, r in oxygen_pairs.iterrows():
        cast_t = pd.Timestamp(r.t_utc)
        label = cast_t + pd.Timedelta(minutes=r.offset_min)
        inside = raw[r.sonde].loc[(raw[r.sonde].index >= label - pd.Timedelta(hours=1))
                                  & (raw[r.sonde].index < label)]
        inside = inside[inside.oxygen.notna()]
        if len(inside):
            offsets.append(float(((inside.index - cast_t).total_seconds() / 60.0).to_numpy().mean()))
    offsets = np.array(offsets)
    out = {'pairs': 'oxygen pairs of the declared figure',
           'matched_hour_label_minus_cast_min': [round(float(oxygen_pairs.offset_min.min()), 1),
                                                 round(float(oxygen_pairs.offset_min.max()), 1)],
           'matched_reading_minus_cast_min': {
               'min': round(float(offsets.min()), 1), 'median': round(float(np.median(offsets)), 1),
               'max': round(float(offsets.max()), 1),
               'share_within_30_min': round(float((np.abs(offsets) <= 30).mean()), 3)},
           'true_time': {}}
    for window in WINDOWS_MIN:
        rows = []
        for _, r in pairs.iterrows():
            series = raw[r.sonde]
            cast_t = pd.Timestamp(r.t_utc)
            row = {'cast_id': r.cast_id, 'date': str(cast_t.date())}
            for name in ('oxygen', 'temperature'):
                s = series[name].dropna()
                if f'{name}_cast' not in r or not np.isfinite(r[f'{name}_cast']) or s.empty:
                    continue
                dt = np.abs((s.index - cast_t).total_seconds() / 60.0)
                k = int(np.argmin(dt))
                if dt[k] <= window:
                    row[f'{name}_diff'] = float(r[f'{name}_cast'] - s.iloc[k])
            rows.append(row)
        frame = pd.DataFrame(rows)
        out['true_time'][f'{window}min'] = {
            n: describe(frame, n) for n in ('oxygen', 'temperature')}
        if 'oxygen_diff' in frame:
            out['true_time'][f'{window}min']['oxygen_cast_dates'] = int(
                frame.dropna(subset=['oxygen_diff']).date.nunique())
    return out


def independence(pairs, name):
    d = pairs.dropna(subset=[f'{name}_diff']).copy()
    d['date'] = pd.to_datetime(d.t_utc, utc=True).dt.date
    d['label'] = pd.to_datetime(d.t_utc, utc=True) + pd.to_timedelta(d.offset_min, unit='min')
    hours = d.groupby(['sonde', 'label']).size()
    per_date = d.groupby('date')[f'{name}_diff'].apply(lambda s: s.abs().mean())
    return {'pairs': int(len(d)), 'cast_dates': int(d.date.nunique()),
            'station_dates': int(d.groupby(['station', 'date']).ngroups),
            'cast_dates_by_site': {k: int(v) for k, v in d.groupby('station').date.nunique().items()},
            'sonde_hours': int(len(hours)), 'sonde_hours_shared_by_casts': int((hours > 1).sum()),
            'over_cast_dates': {'median_of_date_means': float(per_date.median()),
                                'mean_of_date_means': float(per_date.mean())}}


def bootstrap_over_dates(pairs, name, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Percentile intervals of the pooled statistics, resampling cast dates with replacement.

    A date drawn twice contributes its pairs twice; the statistic is taken over the pooled rows."""
    d = pairs.dropna(subset=[f'{name}_diff']).copy()
    d['date'] = pd.to_datetime(d.t_utc, utc=True).dt.date
    groups = [g[f'{name}_diff'].abs().to_numpy() for _, g in d.groupby('date')]
    rng = np.random.default_rng(seed)
    med, mean, p90 = [], [], []
    for _ in range(draws):
        x = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        med.append(np.median(x))
        mean.append(x.mean())
        p90.append(np.quantile(x, .9))
    ci = lambda v: [round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)]
    return {'unit': 'cast date', 'units': len(groups), 'draws': draws, 'seed': seed,
            'median_abs_95': ci(med), 'mean_abs_95': ci(mean), 'p90_abs_95': ci(p90)}


def checks(profiles):
    """The 2026-09-26 checks on the declared cohort and window (pre-2026, 30 min)."""
    frames = [pair_site(profiles, s, st, 30, True) for s, st in SITES.items()]
    pairs = pd.concat([f for f in frames if len(f)], ignore_index=True)
    raw = {s: raw_readings(s) for s in SITES}
    pooled = {n: describe(pairs, n) for n in VARIABLES}
    return {
        'cohort': 'pre_2026, 30 min: the declared figures',
        'statistic_declared': 'median absolute difference (addendum 9, design.statistics)',
        'registered_prediction': {
            'oxygen_median_above_2': bool(pooled['oxygen']['median_abs'] > PREDICTED_OXYGEN_ABOVE),
            'temperature_median_at_or_above_0.20': bool(
                pooled['temperature']['median_abs'] >= PREDICTED_TEMPERATURE_ABOVE),
            'outcome': 'both halves contradicted' if (
                pooled['oxygen']['median_abs'] <= PREDICTED_OXYGEN_ABOVE
                and pooled['temperature']['median_abs'] < PREDICTED_TEMPERATURE_ABOVE) else 'see fields'},
        'independence': {n: independence(pairs, n) for n in VARIABLES},
        'bootstrap_over_cast_dates': {n: bootstrap_over_dates(pairs, n) for n in VARIABLES},
        'timing': timing(pairs, raw),
        'sources': {s: header(s) for s in SITES},
        'calibration_reference': 'not recorded in the exports: each file is named CTD(calibrato) and '
                                 'its header gives the operator and the processing, no reference, '
                                 'method or date; whether the sondes were adjusted to the casts is unknown',
    }


def main():
    profiles = casts()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum9.json',
              'depth_tolerance_m': DEPTH_TOLERANCE_M, 'construction': 'cast bin vs co-located sonde',
              'caveat': 'upper bound on reproducibility; two instruments, a depth mismatch and a '
                        'time offset are compounded and cannot be separated',
              'by_cohort': {}}

    for drop_2026 in (True, False):
        cohort = 'pre_2026' if drop_2026 else 'all_years'
        report['by_cohort'][cohort] = {}
        for window in WINDOWS_MIN:
            frames = [pair_site(profiles, s, st, window, drop_2026) for s, st in SITES.items()]
            frames = [f for f in frames if len(f)]
            pooled = pd.concat(frames, ignore_index=True)
            entry = {'pairs': int(len(pooled)),
                     'pooled': {n: describe(pooled, n) for n in VARIABLES},
                     'per_site': {}}
            for station, group in pooled.groupby('station'):
                entry['per_site'][station] = {n: describe(group, n) for n in VARIABLES}
            report['by_cohort'][cohort][f'{window}min'] = entry

    report['checks_2026_09_26'] = checks(profiles)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    for cohort, windows in report['by_cohort'].items():
        for tag, entry in windows.items():
            print(f"== {cohort} {tag}  pairs={entry['pairs']}")
            for name in VARIABLES:
                s = entry['pooled'][name]
                if s:
                    print(f"   {name:12s} n={s['n']:4d}  median|d|={s['median_abs']:.3f}  "
                          f"mean|d|={s['mean_abs']:.3f}  bias={s['mean_signed']:+.3f}  "
                          f"p90={s['p90_abs']:.3f}")
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
