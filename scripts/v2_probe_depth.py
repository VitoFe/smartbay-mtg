"""Hourly measured depth of every moored probe, from the depth channel of its own export (review R4).

The spatial background (src/data/bay_profile_context.py: context_features) treats every remote
probe's reading as the value at 3 m, but the probes hang at 0.8 to 6.6 m (median over their records,
runs/v2_metrology_review.json G1). This builds the depth each probe actually reported, hour by hour,
for the opt-in depth-aware background of src/experiments/v2_field_depth.py. Nothing here changes an
existing artifact.

Source: column 2 of data/raw/<sensor>.txt ("Profondita'(m)", pressure-derived), read with
scripts/v2_metrology_review.raw_frame (sentinels set missing) for the 19 probes of
reports/bay_twin/sensor_geometry.csv.

QC, in this order:
  1. range    a raw reading outside [0, 15] m is set missing (the deepest mooring is nominally 5 m on
              an 11 m seabed, so 15 m is beyond any physical position; none is outside on 2026-10-05).
  2. hourly   mean of the readings in [L - 1 h, L), labelled L (v2_metrology_review.hourly, which is
              how read_export reads the temperature), so a depth hour describes the temperature hour
              that carries the same label.
  3. spike    an hourly value is dropped when it departs by more than 0.5 m from the median of the
              probe's own hourly depths in the trailing window [L - 4 h, L], provided that window holds
              at least three hours. The rule is causal: it reads nothing after L. An isolated hour is
              dropped; after a step between deployments the first two hours are dropped too, until the
              trailing median has followed the step, and they take the fallback.
Fallback when an hour has no QC'd depth: (a) the probe's last QC'd depth before the hour; (b) otherwise
its nominal depth. Both read nothing after the hour.

Writes reports/bay_twin/runs/v2_probe_depth/hourly_depth.csv.gz (one row per probe-hour with any raw
depth reading: the hourly mean after the range check, the QC'd value and the flag) and summary.json.
Descriptive input; not declared. Usage: python -m scripts.v2_probe_depth
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_metrology_review import DEPTH_COL, RAW, hourly, quarter_of, raw_frame
from src.data.bay_observations import read_export

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / 'reports/bay_twin/sensor_geometry.csv'
OUT = ROOT / 'reports/bay_twin/runs/v2_probe_depth'
HOURLY = OUT / 'hourly_depth.csv.gz'
SUMMARY = OUT / 'summary.json'
METROLOGY = ROOT / 'reports/bay_twin/runs/v2_metrology_review.json'

RANGE_M = (0.0, 15.0)
SPIKE_M = 0.5
SPIKE_WINDOW = '5h'          # trailing: [L - 4 h, L] on an hourly index
SPIKE_MIN_HOURS = 3
FALLBACK_RULE = ('the hour\'s own QC\'d depth; else the probe\'s last QC\'d depth before the hour; else its '
                 'nominal depth (causal: nothing after the hour is read)')


def qc_probe(name):
    """(hourly frame, counts) for one probe: depth_raw_m after the range check, depth_m after QC."""
    raw = raw_frame(name)[DEPTH_COL]
    present = raw.notna()
    in_range = raw.between(*RANGE_M)
    counts = {'raw_readings': int(len(raw)), 'raw_readings_with_depth': int(present.sum()),
              'raw_readings_out_of_range': int((present & ~in_range).sum())}
    any_reading = hourly(raw).notna()                  # hours with any depth reading before the range check
    z = hourly(raw.where(in_range))
    z = z[any_reading.reindex(z.index, fill_value=False)]
    centre = z.dropna().rolling(SPIKE_WINDOW, min_periods=SPIKE_MIN_HOURS).median()
    deviation = (z.dropna() - centre).abs()
    spike = deviation[deviation > SPIKE_M].index
    flag = pd.Series('ok', index=z.index)
    flag[z.isna()] = 'range'
    flag[spike] = 'spike'
    qc = z.where(flag == 'ok')
    counts.update(hours_with_raw_depth=int(len(z)), hours_dropped_range=int((flag == 'range').sum()),
                  hours_dropped_spike=int((flag == 'spike').sum()), hours_qc=int(qc.notna().sum()))
    frame = pd.DataFrame({'sensor': name, 'time': z.index, 'depth_raw_m': z.to_numpy(),
                          'depth_m': qc.to_numpy(), 'flag': flag.to_numpy()})
    return frame, counts


def stats(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return None
    return {'n': int(len(x)), 'median': float(x.median()), 'min': float(x.min()), 'max': float(x.max()),
            'p05': float(x.quantile(.05)), 'p95': float(x.quantile(.95))}


def build():
    geo = pd.read_csv(GEOMETRY).set_index('sensor')
    g1 = {}
    if METROLOGY.exists():
        g1 = json.loads(METROLOGY.read_text(encoding='utf-8'))['G1_measured_depth']['per_probe']
    frames, probes = [], {}
    for name, row in geo.iterrows():
        frame, counts = qc_probe(name)
        frames.append(frame)
        qc = frame.set_index('time').depth_m.dropna()
        temperature = read_export(RAW / f'{name}.txt')[1].temperature.dropna()
        with_depth = temperature.index.isin(qc.index)
        quarters = {q: {'n': int(len(v)), 'median': float(v.median())} for q, v in qc.groupby(quarter_of(qc.index))}
        overall = stats(qc)
        entry = {
            'site': row.site, 'instrument': 'WNode' if name.startswith('Nodo_') else 'EXO2',
            'nominal_depth_m': float(row.nominal_depth_m), **counts,
            'temperature_hours': int(len(temperature)),
            'temperature_hours_with_own_depth': int(with_depth.sum()),
            'coverage_of_temperature_hours': round(float(with_depth.mean()), 6) if len(temperature) else None,
            'first': str(qc.index.min()) if len(qc) else None, 'last': str(qc.index.max()) if len(qc) else None,
            'overall': overall,
            'median_minus_nominal_m': round(overall['median'] - float(row.nominal_depth_m), 6) if overall else None,
            'by_quarter': quarters,
            'fallback': {'nominal_depth_m': float(row.nominal_depth_m)},
        }
        if name in g1:
            # The metrology summary is computed before any QC: the QC'd median should sit on it.
            entry['metrology_G1_median_before_qc'] = g1[name]['overall']['median']
            entry['qc_median_minus_G1_median'] = round(overall['median'] - g1[name]['overall']['median'], 6)
        probes[name] = entry
        print(f'{name:34s} hours {counts["hours_with_raw_depth"]:6d}  range -{counts["hours_dropped_range"]}'
              f'  spike -{counts["hours_dropped_spike"]}  QC {counts["hours_qc"]:6d}'
              f'  median {overall["median"]:.3f} m (nominal {row.nominal_depth_m:g})'
              f'  temperature hours with own depth {entry["temperature_hours_with_own_depth"]}/{entry["temperature_hours"]}',
              flush=True)
    hourly_frame = pd.concat(frames, ignore_index=True)
    hourly_frame['time'] = pd.DatetimeIndex(hourly_frame.time).strftime('%Y-%m-%dT%H:%M:%SZ')
    OUT.mkdir(parents=True, exist_ok=True)
    # Six decimals, as every stored report; LF on every host so the digest does not depend on the OS.
    hourly_frame.to_csv(HOURLY, index=False, float_format='%.6f', lineterminator='\n')
    totals = {k: int(sum(p[k] for p in probes.values())) for k in
              ('raw_readings_with_depth', 'raw_readings_out_of_range', 'hours_with_raw_depth',
               'hours_dropped_range', 'hours_dropped_spike', 'hours_qc')}
    summary = {
        'purpose': 'measured depth of each remote probe for the depth-aware background (review R4); '
                   'descriptive input, not declared',
        'source': 'data/raw/<sensor>.txt column 2 (depth in m, pressure-derived), via scripts/v2_metrology_review.raw_frame',
        'qc': {'range_m': list(RANGE_M), 'range_applies_to': 'raw readings, before the hourly mean',
               'hourly': 'mean of the readings in [L - 1 h, L), labelled L (as read_export)',
               'spike_threshold_m': SPIKE_M, 'spike_window': 'trailing, [L - 4 h, L]',
               'spike_min_hours_in_window': SPIKE_MIN_HOURS,
               'note': 'causal: the first two hours after a step between deployments are dropped as well'},
        'fallback_rule': FALLBACK_RULE,
        'totals': totals, 'probes': probes,
        'files': {HOURLY.name: hashlib.sha256(HOURLY.read_bytes()).hexdigest()},
    }
    SUMMARY.write_bytes(json.dumps(summary, indent=2, allow_nan=False).encode('utf-8'))
    print('totals', totals)
    print('wrote', HOURLY.relative_to(ROOT).as_posix(), 'and', SUMMARY.relative_to(ROOT).as_posix())
    return summary


if __name__ == '__main__':
    build()
