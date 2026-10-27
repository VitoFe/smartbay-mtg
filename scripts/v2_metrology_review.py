"""Addendum 59, part G: three metrology measurements read from the exports themselves.

G1, measured depth. Every moored export carries a pressure-derived depth channel beside the
temperature. Each probe's depth is read hourly as read_export reads the temperature (mean of the
readings inside an hour, labelled at the end of the hour) and summarised by calendar quarter
(median and range). The eleven co-located pairs are then regrouped by their measured separation:
the median over the pair's overlap hours (the hours the published disagreement is computed on) of
|depth A - depth B|, split under 1 m, 1 to 2 m and over 2 m, each pair carrying its published
sustained disagreement (runs/v2_colocated_depth.json).

G2, sonde adjustment. Each EXO2 export carries the sonde's native oxygen reading in mg/l beside the
oxygen column the thesis reads. Converted on the export's own basis (x 31.2512 x 1000 / the export's
density column), the two should agree; per sonde and calendar month the median ratio of the read
column to the converted native column is reported, and months departing from 1 by more than 0.5 %
are flagged. The cast-against-sonde oxygen disagreement of addendum 9 (pre-2026 casts, 30 min,
cast bin within 1 m of the sonde's nominal depth, the construction of v2_oxygen_floor) is then
recomputed without the pairs whose sonde hour falls in a flagged month, beside the declared 1.8.

G3, temperature offsets against the casts. Each moored probe's hourly temperature against every
co-located cast (cast position within 60 m of the probe, the co-location radius of the pair
tables): the cast bin within 0.5 m of the probe's nominal depth, the probe's hourly value whose
label lies within 90 min of the cast. Offsets are probe minus cast; the median per probe with at
least five pairs, and per instrument family.

Registered predictions (declared before any of this was computed):
  PG1  measured separation does not explain the co-located disagreement: the under-1 m group spans
       most of the published range. Operationalised here as: the under-1 m group has at least two
       pairs and the range of their published disagreements covers at least half of the published
       range over all eleven pairs.
  PG2  the oxygen disagreement without flagged months is larger than 1.8 (the declared median).
  PG3  the WNode loggers' offsets against the casts are larger in magnitude than the sondes'.
       Operationalised here as: the median over probes of |per-probe median offset| is larger for the
       WNode family than for the EXO2 family (probes with at least five pairs); the pooled median
       |offset| over pairs is reported beside it.

Reported beside the declared figures and labelled as such (not declared): G3 repeated with the cast
bin matched to the probe's measured depth at the matched hour; for the oxygen pairs, the cast against
the converted native reading as well as against the read column; the reporting step of both oxygen
columns per sonde and of every temperature channel, with the readings per hour; the cast start hours and the morning part of the daily cycle at Baia that the
chapter quotes; the drift pooling of the variance budget with and without its two zero entries.

Descriptive: no model is fitted and nothing is scored.
Declared in config/bay_reserved_protocol_v2_addendum59.json (part_G_metrology).
Writes reports/bay_twin/runs/v2_metrology_review.json and the probe key table
thesis_template/content/tables/table_bay_probe_key.tex.
Usage: python -m scripts.v2_metrology_review
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export
from src.experiments import v2_oxygen_floor as floor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
BAY = ROOT / 'reports/bay_twin'
OUT = BAY / 'runs/v2_metrology_review.json'
TABLE = ROOT / 'thesis_template/content/tables/table_bay_probe_key.tex'
GEOMETRY = BAY / 'sensor_geometry.csv'

SENTINEL = -999.0
SEPARATION_BANDS = ((0.0, 1.0, 'under_1m'), (1.0, 2.0, '1_to_2m'), (2.0, np.inf, 'over_2m'))
O2_FACTOR = 31.2512          # umol per mg of O2, the conversion the exports use
ADJUSTMENT_TOLERANCE = 0.005  # a month departing from 1 by more than 0.5 % is flagged
COLOCATION_RADIUS_M = 60.0
CAST_DEPTH_TOLERANCE_M = 0.5
CAST_TIME_WINDOW_MIN = 90.0
MIN_PAIRS = 5
DECLARED_OXYGEN_FLOOR = 1.803161607142897  # runs/v2_oxygen_floor.json pre_2026 30min median
# column positions in the two export layouts (the '#' header line names them)
DEPTH_COL, TEMP_COL = 2, 4
MG_COL, DENSITY_COL, READ_O2_COL = 6, 11, 13


def raw_frame(name):
    """The export's rows at their own timestamps, sorted, sentinels set to NaN."""
    path = RAW / f'{name}.txt'
    f = pd.read_csv(path, comment='#', sep=r'\s+', header=None, encoding='cp1252')
    t = pd.to_datetime(f[0] + ' ' + f[1], format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    values = f.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    values.index = pd.DatetimeIndex(t)
    values = values.loc[values.index.notna()].sort_index()
    return values.where(values > SENTINEL)


def hourly(series):
    """As read_export: mean of duplicate timestamps, then of the readings in [L - 1 h, L), labelled L."""
    s = series.groupby(level=0).mean()
    return s.resample('1h', closed='left', label='right').mean()


def quarter_of(index):
    """The calendar quarter of an hourly label, taken at the centre of its hour."""
    return (index - pd.Timedelta(minutes=30)).tz_localize(None).to_period('Q').astype(str)


def stats(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return None
    return {'n': int(len(x)), 'median': float(x.median()), 'min': float(x.min()), 'max': float(x.max()),
            'p05': float(x.quantile(.05)), 'p95': float(x.quantile(.95))}


# ---------------------------------------------------------------------------------------------- G1

def measured_depth(geo):
    depth, temperature, per_probe = {}, {}, {}
    for name, row in geo.iterrows():
        z = hourly(raw_frame(name)[DEPTH_COL])
        depth[name] = z
        temperature[name] = read_export(RAW / f'{name}.txt')[1].temperature
        valid = z.dropna()
        quarters = {q: stats(g) for q, g in valid.groupby(quarter_of(valid.index))}
        medians = [v['median'] for v in quarters.values()]
        t = temperature[name].dropna()
        per_probe[name] = {
            'site': row.site, 'instrument': 'WNode' if name.startswith('Nodo_') else 'EXO2',
            'nominal_depth_m': float(row.nominal_depth_m), 'metadata_bottom_m': float(row.metadata_bottom_m),
            'hours_with_depth': int(len(valid)),
            'overall': stats(valid),
            'median_minus_nominal_m': float(valid.median() - row.nominal_depth_m),
            'quarterly_median_range_m': [float(min(medians)), float(max(medians))],
            'by_quarter': quarters,
            'temperature_first': str(t.index.min()), 'temperature_last': str(t.index.max()),
        }
    return depth, temperature, per_probe


def pair_separation(depth, temperature):
    published = json.loads((BAY / 'runs/v2_colocated_depth.json').read_text())
    all_mad = [p['mad'] for p in published['by_pair']]
    rows = []
    for p in published['by_pair']:
        a, b = p['a'], p['b']
        joined = pd.concat([temperature[a].rename('ta'), temperature[b].rename('tb'),
                            depth[a].rename('za'), depth[b].rename('zb')], axis=1)
        overlap = joined.dropna(subset=['ta', 'tb'])
        both = overlap.dropna(subset=['za', 'zb'])
        dz = (both.za - both.zb).abs()
        sep = float(dz.median())
        band = next(label for lo, hi, label in SEPARATION_BANDS if lo <= sep < hi)
        rows.append({
            'a': a, 'b': b, 'classes': p['classes'], 'horizontal_m': p['separation_m'],
            'nominal_separation_m': p['depth_separation_m'],
            'overlap_h': int(len(overlap)), 'overlap_h_published': p['overlap_h'],
            'mad_recomputed': round(float((overlap.ta - overlap.tb).abs().mean()), 4),
            'mad_published': p['mad'],
            'overlap_h_with_both_depths': int(len(both)),
            'measured_median_abs_separation_m': sep,
            'measured_separation_p05_p95_m': [float(dz.quantile(.05)), float(dz.quantile(.95))],
            'median_depth_a_m': float(both.za.median()), 'median_depth_b_m': float(both.zb.median()),
            'difference_of_medians_m': float(abs(both.za.median() - both.zb.median())),
            'share_of_hours_over_2m': float((dz > 2.0).mean()),
            'band': band,
        })
    frame = pd.DataFrame(rows)
    groups = {}
    for _lo, _hi, label in SEPARATION_BANDS:
        sub = frame[frame.band == label]
        groups[label] = None if sub.empty else {
            'pairs': [f"{r.a} / {r.b}" for r in sub.itertuples()], 'n': int(len(sub)),
            'mad_min': float(sub.mad_published.min()), 'mad_max': float(sub.mad_published.max()),
            'mad_mean': float(sub.mad_published.mean()), 'overlap_h_total': int(sub.overlap_h.sum())}
    lo, hi = min(all_mad), max(all_mad)
    under = groups['under_1m']
    share = None if under is None or under['n'] < 2 else (under['mad_max'] - under['mad_min']) / (hi - lo)
    pg1 = {'prediction': 'measured separation does not explain the co-located disagreement: the under-1 m '
                         'group spans most of the published range',
           'operationalised': 'under-1 m group has >= 2 pairs and its published range covers >= 50 % of '
                              'the published range over all pairs',
           'published_range': [lo, hi], 'under_1m_share_of_range': share,
           'held': bool(share is not None and share >= 0.5)}
    nominal_vs_measured = {
        'nominally_identical_depth_pairs_measured_over_1m': [
            f"{r.a} / {r.b}" for r in frame.itertuples()
            if r.nominal_separation_m == 0 and r.measured_median_abs_separation_m >= 1.0],
        'strictly_colocated_nominal': [f"{r.a} / {r.b}" for r in frame.itertuples()
                                       if r.horizontal_m == 0 and r.nominal_separation_m == 0],
        'zero_horizontal_and_measured_under_1m': [
            f"{r.a} / {r.b}" for r in frame.itertuples()
            if r.horizontal_m == 0 and r.measured_median_abs_separation_m < 1.0]}
    return rows, groups, pg1, nominal_vs_measured


# ---------------------------------------------------------------------------------------------- G2

def oxygen_columns(sonde):
    f = raw_frame(sonde)
    read = f[READ_O2_COL].where(f[READ_O2_COL].between(0, 600))   # read_export's screen
    mg = f[MG_COL].where(f[MG_COL] > 0)
    rho = f[DENSITY_COL].where(f[DENSITY_COL] > 900)
    native = mg * O2_FACTOR * 1000.0 / rho
    return f, read, mg, native


def sonde_adjustment(sonde):
    f, read, mg, native = oxygen_columns(sonde)
    ok = read.notna() & native.notna()
    ratio = (read / native)[ok]
    month = ratio.index.tz_localize(None).to_period('M').astype(str)
    months = {}
    for m, r in ratio.groupby(month):
        med = float(r.median())
        months[m] = {'rows': int(len(r)), 'median_ratio': med, 'mean_ratio': float(r.mean()),
                     'p05_ratio': float(r.quantile(.05)), 'p95_ratio': float(r.quantile(.95)),
                     'flagged': bool(abs(med - 1.0) > ADJUSTMENT_TOLERANCE)}
    flagged = [m for m, v in months.items() if v['flagged']]
    meds = [v['median_ratio'] for v in months.values()]
    return {
        'rows_with_both': int(ok.sum()),
        'rows_native_only': int((mg.notna() & read.isna()).sum()),
        'rows_read_only': int((read.notna() & mg.isna()).sum()),
        'median_ratio_all_rows': float(ratio.median()),
        'months': len(months), 'months_flagged': flagged,
        'monthly_median_ratio_range': [float(min(meds)), float(max(meds))],
        'largest_departure': max(months.items(), key=lambda kv: abs(kv[1]['median_ratio'] - 1))[0],
        'by_month': months,
    }, ratio


def floor_without_flagged(adjustment, ratios):
    """The addendum-9 construction (pre-2026, 30 min), with and without pairs in flagged months."""
    profiles = floor.casts()
    frames = [floor.pair_site(profiles, s, st, 30, True) for s, st in floor.SITES.items()]
    pairs = pd.concat([f for f in frames if len(f)], ignore_index=True)
    pairs = pairs.dropna(subset=['oxygen_diff']).copy()
    label = pd.to_datetime(pairs.t_utc, utc=True) + pd.to_timedelta(pairs.offset_min, unit='min')
    pairs['label'] = label
    pairs['month'] = (label - pd.Timedelta(minutes=1)).dt.tz_localize(None).dt.to_period('M').astype(str)
    pairs['month_flagged'] = [m in adjustment[s]['months_flagged'] for s, m in zip(pairs.sonde, pairs.month)]
    # the matched hour's own ratio and the converted native reading over the same hour (not declared)
    natives = {s: oxygen_columns(s)[3] for s in pairs.sonde.unique()}
    hour_ratio, native_hourly = [], []
    for s, lab in zip(pairs.sonde, pairs.label):
        r = ratios[s]
        inside = r[(r.index >= lab - pd.Timedelta(hours=1)) & (r.index < lab)]
        hour_ratio.append(float(inside.mean()) if len(inside) else np.nan)
        native = natives[s]
        nat = native[(native.index >= lab - pd.Timedelta(hours=1)) & (native.index < lab)].dropna()
        native_hourly.append(float(nat.mean()) if len(nat) else np.nan)
    pairs['hour_ratio'] = hour_ratio
    pairs['native_sonde'] = native_hourly
    pairs['native_diff'] = pairs.oxygen_cast - pairs.native_sonde

    declared = floor.describe(pairs, 'oxygen')
    kept = pairs[~pairs.month_flagged]
    kept_hour = pairs[(pairs.hour_ratio - 1).abs() <= ADJUSTMENT_TOLERANCE]
    out = {
        'construction': 'v2_oxygen_floor.pair_site, pre-2026 casts, 30 min, cast bin within 1 m',
        'declared_reproduced': declared,
        'pairs_in_flagged_months': int(pairs.month_flagged.sum()),
        'pairs_in_flagged_months_by_station': {k: int(v) for k, v in
                                               pairs[pairs.month_flagged].station.value_counts().items()},
        'without_flagged_months': floor.describe(kept, 'oxygen'),
        'without_flagged_months_cast_dates': int(pd.to_datetime(kept.t_utc, utc=True).dt.date.nunique()),
        'without_flagged_months_by_station': {st: floor.describe(g, 'oxygen') for st, g in kept.groupby('station')},
        'without_flagged_months_bootstrap': floor.bootstrap_over_dates(kept, 'oxygen'),
        'not_declared_without_hours_departing': {
            'rule': 'drop pairs whose matched hour has a mean ratio departing from 1 by more than 0.5 %',
            'summary': floor.describe(kept_hour, 'oxygen')},
        'not_declared_cast_against_read_and_native': {},
    }
    for label_, sub in (('flagged_months', pairs[pairs.month_flagged]), ('other_months', kept)):
        sub = sub.dropna(subset=['native_diff'])
        if sub.empty:
            continue
        out['not_declared_cast_against_read_and_native'][label_] = {
            'pairs': int(len(sub)),
            'median_abs_cast_minus_read': float(sub.oxygen_diff.abs().median()),
            'median_abs_cast_minus_native': float(sub.native_diff.abs().median()),
            'mean_abs_cast_minus_read': float(sub.oxygen_diff.abs().mean()),
            'mean_abs_cast_minus_native': float(sub.native_diff.abs().mean()),
            'read_closer_to_cast': int((sub.oxygen_diff.abs() < sub.native_diff.abs()).sum()),
            'by_station': {st: {'pairs': int(len(g)),
                                'median_abs_cast_minus_read': float(g.oxygen_diff.abs().median()),
                                'median_abs_cast_minus_native': float(g.native_diff.abs().median())}
                           for st, g in sub.groupby('station')}}
    med = out['without_flagged_months']['median_abs'] if out['without_flagged_months'] else None
    out['PG2'] = {'prediction': 'the oxygen disagreement without flagged months is larger than 1.8',
                  'statistic': 'median |cast - sonde| (the declared statistic of addendum 9)',
                  'declared': DECLARED_OXYGEN_FLOOR, 'without_flagged_months': med,
                  'held': bool(med is not None and med > DECLARED_OXYGEN_FLOOR)}
    return out


def reporting_step(sonde, columns=None):
    """Smallest non-zero difference between consecutive raw readings, and the decimals written."""
    text = (RAW / f'{sonde}.txt').read_text(encoding='cp1252').splitlines()
    rows = [line.split() for line in text if line.strip() and not line.startswith('#')]
    if columns is None:
        _f, read, mg, _native = oxygen_columns(sonde)
        columns = (('mg_per_l_native', mg, MG_COL), ('read_column', read, READ_O2_COL))
    out = {}
    for label, series, col in columns:
        x = series.dropna().to_numpy()
        d = np.round(np.abs(np.diff(x)), 9)
        d = d[d > 0]
        tokens = [r[col] for r in rows if len(r) > col and not r[col].startswith('-9999')]
        decimals = pd.Series([len(t.split('.')[1]) if '.' in t else 0 for t in tokens]).value_counts()
        out[label] = {'readings': int(len(x)), 'smallest_nonzero_step': float(d.min()),
                      'p01_nonzero_step': float(np.quantile(d, .01)),
                      'decimals_written': {int(k): int(v) for k, v in decimals.items()}}
    return out


def temperature_reporting(name):
    """The raw temperature channel: its reporting step, the decimals written, readings per hour."""
    t = raw_frame(name)[TEMP_COL]
    t = t.where(t.between(-2, 40)).dropna()
    out = reporting_step(name, (('temperature', t, TEMP_COL),))['temperature']
    t = t.groupby(level=0).mean()   # a timestamp written twice is one reading
    per_hour = t.resample('1h', closed='left', label='right').count()
    per_hour = per_hour[per_hour > 0]
    out['median_readings_per_reporting_hour'] = float(per_hour.median())
    out['share_of_reporting_hours_with_one_reading'] = float((per_hour == 1).mean())
    return out


# ---------------------------------------------------------------------------------------------- G3

def cast_bins():
    d = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz')
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    return d[(d.qc_flag == 0) & np.isfinite(d.temp_c) & d.temp_c.between(-2, 40)]


def offsets_for(name, row, temperature, depth, casts, measured=False):
    t = temperature.dropna()
    z = depth
    near = casts[np.hypot(casts.UTM_X - row.x_m, casts.UTM_Y - row.y_m) <= COLOCATION_RADIUS_M]
    out = []
    for cast_id, g in near.groupby('cast_id', sort=True):
        when = g.t_utc.iloc[0]
        delta = (t.index - when).total_seconds() / 60.0
        within = np.flatnonzero(np.abs(delta) <= CAST_TIME_WINDOW_MIN)
        if not len(within):
            continue
        j = within[np.argmin(np.abs(delta[within]))]
        if measured:
            target = z.get(t.index[j], np.nan)
            if not np.isfinite(target):
                continue
        else:
            target = float(row.nominal_depth_m)
        cand = g.loc[(g.depth_m - target).abs() <= CAST_DEPTH_TOLERANCE_M]
        if cand.empty:
            continue
        b = cand.loc[(cand.depth_m - target).abs().idxmin()]   # the rule of v2_oxygen_floor
        out.append({'probe': name, 'station': g.station.iloc[0], 'cast_id': cast_id,
                    'date': str(when.date()), 'cast_depth_m': float(b.depth_m), 'target_depth_m': float(target),
                    'minutes_label_minus_cast': float(delta[j]), 'probe_hour': str(t.index[j]),
                    'offset': float(t.iloc[j] - b.temp_c)})
    return out


def summarise_offsets(pairs, geo):
    per_probe = {}
    for name, g in pairs.groupby('probe'):
        per_probe[name] = {'family': 'WNode' if name.startswith('Nodo_') else 'EXO2',
                           'stations': sorted(g.station.unique()), 'pairs': int(len(g)),
                           'cast_dates': int(g.date.nunique()), 'probe_hours': int(g.probe_hour.nunique()),
                           'median_offset': float(g.offset.median()), 'mean_offset': float(g.offset.mean()),
                           'median_abs_offset': float(g.offset.abs().median()),
                           'iqr_offset': [float(g.offset.quantile(.25)), float(g.offset.quantile(.75))],
                           'qualifies': bool(len(g) >= MIN_PAIRS)}
    families = {}
    for fam in ('WNode', 'EXO2'):
        q = {k: v for k, v in per_probe.items() if v['family'] == fam and v['qualifies']}
        if not q:
            families[fam] = None
            continue
        meds = np.array([v['median_offset'] for v in q.values()])
        pooled = pairs[pairs.probe.isin(q)]
        families[fam] = {'probes': sorted(q), 'n_probes': len(q), 'pairs': int(len(pooled)),
                         'cast_dates': int(pooled.date.nunique()),
                         'per_probe_median_offsets': {k: v['median_offset'] for k, v in q.items()},
                         'median_of_abs_probe_medians': float(np.median(np.abs(meds))),
                         'mean_of_abs_probe_medians': float(np.mean(np.abs(meds))),
                         'range_of_probe_medians': [float(meds.min()), float(meds.max())],
                         'pooled_median_abs_offset': float(pooled.offset.abs().median()),
                         'pooled_median_offset': float(pooled.offset.median())}
    return per_probe, families


def temperature_offsets(geo, temperature, depth):
    casts = cast_bins()
    result = {}
    for variant, measured in (('nominal_depth', False), ('not_declared_measured_depth', True)):
        rows = []
        for name, row in geo.iterrows():
            rows += offsets_for(name, row, temperature[name], depth[name], casts, measured)
        pairs = pd.DataFrame(rows)
        per_probe, families = summarise_offsets(pairs, geo)
        result[variant] = {'pairs': int(len(pairs)), 'per_probe': per_probe, 'families': families}
    fam = result['nominal_depth']['families']
    w, e = fam['WNode'], fam['EXO2']
    result['PG3'] = {
        'prediction': "the WNode loggers' offsets against the casts are larger in magnitude than the sondes'",
        'operationalised': 'median over qualifying probes of |per-probe median offset|, WNode against EXO2',
        'wnode': w['median_of_abs_probe_medians'] if w else None,
        'exo2': e['median_of_abs_probe_medians'] if e else None,
        'pooled_median_abs_wnode': w['pooled_median_abs_offset'] if w else None,
        'pooled_median_abs_exo2': e['pooled_median_abs_offset'] if e else None,
        'held': bool(w and e and w['median_of_abs_probe_medians'] > e['median_of_abs_probe_medians'])}
    result['rule'] = {'colocation_radius_m': COLOCATION_RADIUS_M, 'cast_depth_tolerance_m': CAST_DEPTH_TOLERANCE_M,
                      'probe_hour_within_min': CAST_TIME_WINDOW_MIN, 'min_pairs': MIN_PAIRS,
                      'sign': 'probe minus cast', 'casts': 'ctd_all_casts.csv.gz, qc_flag 0, finite temperature',
                      'bin_choice': 'nearest bin within tolerance; a tie between the bins 0.25 m above and below '
                                    'goes to the first in file order (the shallower), as v2_oxygen_floor'}
    return result


# ----------------------------------------------------------------------------------- not declared

def cast_timing():
    d = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz')
    start = pd.to_datetime(d.groupby('cast_id').t_utc.first(), utc=True)
    out = {}
    for label, s in (('all_casts', start), ('from_2024_07_24', start[start >= pd.Timestamp('2024-07-24', tz='UTC')])):
        minutes = s.dt.hour * 60 + s.dt.minute
        out[label] = {'casts': int(len(s)),
                      'by_start_hour_utc': {int(k): int(v) for k, v in s.dt.hour.value_counts().sort_index().items()},
                      'earliest_start_hhmm': f'{int(minutes.min()) // 60:02d}:{int(minutes.min()) % 60:02d}',
                      'latest_start_hhmm': f'{int(minutes.max()) // 60:02d}:{int(minutes.max()) % 60:02d}',
                      'started_before_0730': int((minutes < 7 * 60 + 30).sum()),
                      'started_at_or_after_1200': int((minutes >= 12 * 60).sum())}
    return out


def morning_layering():
    """The daily part of Baia 1 m minus 5 m at the 08 UTC hour, from the curves in v2_daily_cycle.json."""
    dc = json.loads((BAY / 'runs/v2_daily_cycle.json').read_text())
    out = {}
    for season in ('MAM', 'JJA'):
        up = np.array(dc['temperature'][season]['Baia S. Teresa, 1 m']['curve'])
        lo = np.array(dc['temperature'][season]['Baia Pontone, 5 m']['curve'])
        diff = up - lo
        hours = np.arange(0.5, 24, 1.0)
        out[season] = {'peak': float(diff.max()), 'hour_of_peak_centre_utc': float(hours[diff.argmax()]),
                       'window_07_12_mean_below_peak': float(diff.max() - diff[(hours > 7) & (hours < 12)].mean()),
                       'below_peak_at_hour_centre': {f'{h:.1f}': float(diff.max() - diff[int(h)])
                                                     for h in (7.5, 8.5, 9.5, 10.5)}}
    oxygen_minimum = {season: {g: v['hour_min_utc'] for g, v in groups.items()}
                      for season, groups in dc['oxygen'].items()}
    return {'source': 'runs/v2_daily_cycle.json temperature curves (median by interval-centre hour)',
            'baia_1m_minus_5m': out, 'oxygen_minimum_hour_centre_utc': oxygen_minimum}


def drift_pooling():
    vc = json.loads((BAY / 'runs/v2_variance_components.json').read_text())
    pairs = pd.DataFrame(vc['pairs'])
    root2 = np.sqrt(2.0)
    zero = pairs[pairs.drift_sd == 0.0]
    rest = pairs[pairs.drift_sd > 0.0]
    u_b, u_e = vc['budget']['u_offset'], vc['budget']['u_residual']

    def shares(u_d):
        total = u_b ** 2 + u_d ** 2 + u_e ** 2
        return {'offset': u_b ** 2 / total, 'drift': u_d ** 2 / total, 'short_term': u_e ** 2 / total,
                'quadrature_sum': float(np.sqrt(total))}
    u_all = float(np.sqrt((pairs.drift_sd ** 2).mean()) / root2)
    u_nine = float(np.sqrt((rest.drift_sd ** 2).mean()) / root2)
    return {'zero_drift_pairs': [{'a': r.a, 'b': r.b, 'overlap_h': int(r.overlap_h), 'span_days': r.span_days,
                                  'common_support_h': int(r.common_support_h),
                                  'drift_sd_mean_filter': r.drift_sd_mean_filter} for r in zero.itertuples()],
            'u_drift_all_pairs': u_all, 'shares_all_pairs': shares(u_all),
            'u_drift_pairs_with_estimate': u_nine, 'pairs_with_estimate': int(len(rest)),
            'shares_pairs_with_estimate': shares(u_nine)}


# ------------------------------------------------------------------------------------ probe key

SITE_NAMES = {'Baia_S.Teresa': 'Baia Santa Teresa', 'Boa_S.Teresa': 'Boa Santa Teresa',
              'Baia_Pontone': 'Baia Pontone'}


def site_name(raw):
    return SITE_NAMES.get(raw, raw.replace('_', ' '))


def tex_id(name):
    return name.replace('-CTD(calibrato)', '').replace('_', '\\_')


def probe_key_table(per_probe):
    order = sorted(per_probe, key=lambda k: (site_name(per_probe[k]['site']), k))
    lines = []
    for k in order:
        p = per_probe[k]
        lo, hi = p['quarterly_median_range_m']
        span = f'${lo:.1f}$' if round(lo, 1) == round(hi, 1) else f'${lo:.1f}$--${hi:.1f}$'
        first, last = p['temperature_first'][:7], p['temperature_last'][:7]
        # The instrument is not a column: the identifier names it (Nodo_*: WNode; T*: EXO2), as the
        # note says, which lets the table print at 8 pt without scaling.
        lines.append(f"{tex_id(k)} & {site_name(p['site'])} & "
                     f"${p['nominal_depth_m']:.0f}$ & ${p['overall']['median']:.1f}$ & {span} & "
                     f"${p['metadata_bottom_m']:.0f}$ & {first} & {last} \\\\")
    body = '\n'.join(lines)
    return (
        '% Generated by scripts/v2_metrology_review.py; do not edit by hand.\n'
        '\\begin{table}[htbp]\n\\centering\n'
        '\\caption[The moored probes]{The moored probes, by site: the identifier used in the figures and '
        'tables, the nominal deployment depth recorded in each export\'s header, the depth '
        'its own pressure channel measured (median over the record, and the range of its quarterly medians), '
        'the seabed depth recorded with the deployment, and the first and last month with a temperature '
        'report.}\n'
        '\\label{tab:bay_probe_key}\n'
        '\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
        '\\begin{tabular}{@{}llrrrrll@{}}\n\\toprule\n'
        '\\textbf{ID} & \\textbf{Site} & '
        '\\makecell[r]{\\textbf{Nominal}\\\\\\textbf{depth}\\\\\\textbf{(m)}} & '
        '\\makecell[r]{\\textbf{Measured}\\\\\\textbf{median}\\\\\\textbf{(m)}} & '
        '\\makecell[r]{\\textbf{Quarterly}\\\\\\textbf{medians}\\\\\\textbf{(m)}} & '
        '\\makecell[r]{\\textbf{Seabed}\\\\\\textbf{(m)}} & '
        '\\textbf{First} & \\textbf{Last} \\\\\n\\midrule\n'
        f'{body}\n'
        '\\bottomrule\n\\end{tabular}\n'
        '\n\\vspace{2pt}\n'
        '\\scriptsize\\justifying\\noindent An identifier beginning Nodo is a WNode probe and the others are '
        'EXO2 sondes, whose identifiers omit the \\texttt{-CTD(calibrato)} suffix of their files. Measured '
        'depth is the pressure-derived depth channel of each export, averaged by hour as the temperature is '
        'read.\\artifact{runs/v2_metrology_review.json}\n'
        '\\end{table}\n')


# ------------------------------------------------------------------------------------------- main

def main():
    geo = pd.read_csv(GEOMETRY).set_index('sensor')
    depth, temperature, per_probe = measured_depth(geo)
    pairs, groups, pg1, nominal_vs_measured = pair_separation(depth, temperature)

    sondes = list(floor.SITES)
    adjustment, ratios = {}, {}
    for s in sondes:
        adjustment[s], ratios[s] = sonde_adjustment(s)
    floor_check = floor_without_flagged(adjustment, ratios)

    report = {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json (part_G_metrology)',
        'descriptive': 'No model fitted, nothing scored.',
        'G1_measured_depth': {
            'method': 'export depth channel (column 3 of the data rows), hourly mean labelled at the end of the '
                      'hour as read_export reads temperature; quarters of the interval centre',
            'per_probe': per_probe,
            'pairs': pairs, 'groups_by_measured_separation': groups,
            'separation_rule': 'median over the pair\'s overlap hours (both temperatures and both depths '
                               'present) of |depth A - depth B|; bands [0, 1), [1, 2), [2, inf) m',
            'nominal_against_measured': nominal_vs_measured,
            'PG1': pg1},
        'G2_sonde_adjustment': {
            'method': 'ratio of the read oxygen column to mg/l x 31.2512 x 1000 / the export density column, '
                      'row by row; per calendar month the median ratio, flagged if it departs from 1 by '
                      'more than 0.5 %',
            'per_sonde': adjustment,
            'oxygen_floor': floor_check,
            'PG2': floor_check['PG2']},
        'G3_temperature_offsets': temperature_offsets(geo, temperature, depth),
        'not_declared': {
            'oxygen_reporting_step': {s: reporting_step(s) for s in sondes},
            'temperature_reporting_step': {name: temperature_reporting(name) for name in geo.index},
            'cast_start_times': cast_timing(),
            'morning_daily_cycle': morning_layering(),
            'variance_budget_drift_pooling': drift_pooling()},
    }
    # the counts the chapter quotes, read off the per-probe and per-month entries above
    medians = {k: v['overall']['median'] for k, v in per_probe.items()}
    off = {k: abs(v['median_minus_nominal_m']) for k, v in per_probe.items()}
    beyond = {k: d for k, d in off.items() if d > 0.5}
    sondes = [k for k in per_probe if not k.startswith('Nodo_')]
    report['G1_measured_depth']['summary'] = {
        'probes': len(per_probe), 'within_0p5m_of_nominal': len(off) - len(beyond),
        'more_than_0p5m_from_nominal': sorted(beyond), 'more_than_1m_from_nominal': sorted(k for k, d in off.items() if d > 1.0),
        'departure_range_of_those_beyond_0p5m': [min(beyond.values()), max(beyond.values())],
        'median_depth_range_m': [min(medians.values()), max(medians.values())],
        'mean_of_probe_medians_m': float(np.mean(list(medians.values()))),
        'sonde_median_depth_range_m': [min(medians[k] for k in sondes), max(medians[k] for k in sondes)]}
    months = [(s, m, v) for s, a in adjustment.items() for m, v in a['by_month'].items()]
    report['G2_sonde_adjustment']['summary'] = {
        'sonde_months': len(months), 'sonde_months_flagged': sum(v['flagged'] for _, _, v in months),
        'largest_departure': max((abs(v['median_ratio'] - 1), s, m) for s, m, v in months),
        'flagged_months_in_2026': sorted(f'{s} {m}' for s, m, v in months if v['flagged'] and m.startswith('2026'))}
    report['PG1'], report['PG2'] = pg1, floor_check['PG2']
    report['PG3'] = report['G3_temperature_offsets']['PG3']
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    TABLE.write_bytes(probe_key_table(per_probe).encode('utf-8'))

    print('G1 measured depth (median, quarterly medians):')
    for k, p in per_probe.items():
        print(f"  {k:34s} nominal {p['nominal_depth_m']:.0f}  median {p['overall']['median']:.2f}  "
              f"quarters {p['quarterly_median_range_m'][0]:.2f}-{p['quarterly_median_range_m'][1]:.2f}")
    for r in pairs:
        print(f"  {r['a']:10s} {r['b']:34s} nominal dz {r['nominal_separation_m']:.0f}  measured "
              f"{r['measured_median_abs_separation_m']:.2f}  mad {r['mad_published']:.4f}  "
              f"(recomputed {r['mad_recomputed']:.4f}, {r['overlap_h']} h)  {r['band']}")
    print('  groups:', {k: (v and (v['n'], v['mad_min'], v['mad_max'])) for k, v in groups.items()})
    print('  PG1', pg1['held'], pg1['under_1m_share_of_range'])
    print('G2 flagged months:', {s: a['months_flagged'] for s, a in adjustment.items()})
    print('  floor without flagged months:', floor_check['without_flagged_months'])
    print('  PG2', floor_check['PG2']['held'])
    g3 = report['G3_temperature_offsets']
    for name, v in g3['nominal_depth']['per_probe'].items():
        print(f"  {name:34s} pairs {v['pairs']:3d}  dates {v['cast_dates']:3d}  median {v['median_offset']:+.3f}")
    print('  families:', {k: v and (v['median_of_abs_probe_medians'], v['pooled_median_abs_offset'])
                          for k, v in g3['nominal_depth']['families'].items()})
    print('  PG3', g3['PG3']['held'])
    print('wrote', OUT.relative_to(ROOT), TABLE.relative_to(ROOT))


if __name__ == '__main__':
    main()
