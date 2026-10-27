"""What the nested aggregation rules change, computed on the released predictions.

Both headline metrics are nested macro averages (Chapter 4). This computes, for the released
models, the nested value and the flat alternative a reader would otherwise reach for, so the
thesis can show the difference on current numbers:

  spatial   A1 (released family) held-station temperature on Field-57: casts within (fold,
            station), then stations within fold, then folds; against a pooled mean over depth bins
  forecast  Reserved-2026 six-hour temperature: the mean over quarters of the per-sensor mean;
            against a flat mean over (quarter, sensor) cells

Usage: python -m scripts.v2_aggregation_rule
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_aggregation_rule.json'
SPATIAL = RUNS / ('20260919T145045Z_v2_field/results/v2_field_held_site_A1_both_direct/'
                  'selected_predictions.csv.gz')
FORECAST = RUNS / 'v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'


def spatial():
    d = pd.read_csv(SPATIAL)
    error = (d.temp_c - d.pred_temp_c).abs()
    z = d[['fold', 'station', 'cast_id']].assign(error=error)
    nested = (z.groupby(['fold', 'station', 'cast_id']).error.mean()
              .groupby(['fold', 'station']).mean().groupby('fold').mean().mean())
    return {'source': str(SPATIAL.relative_to(ROOT)).replace('\\', '/'),
            'nested': round(float(nested), 4), 'pooled_over_bins': round(float(error.mean()), 4),
            'bins': int(len(d)), 'casts': int(d.cast_id.nunique())}


def forecast():
    d = pd.read_csv(FORECAST)
    d = d[(d.variable == 'temperature') & d.has_anchor & d.has_target & d.error.notna()]
    cells = d.groupby(['quarter', 'sensor']).error.mean()
    nested = cells.groupby('quarter').mean().mean()
    return {'source': str(FORECAST.relative_to(ROOT)).replace('\\', '/'),
            'nested': round(float(nested), 4), 'flat_over_cells': round(float(cells.mean()), 4),
            'pooled_over_endpoints': round(float(d.error.mean()), 4),
            'endpoints': int(len(d)), 'cells': int(len(cells))}


def main():
    report = {'spatial_temperature_field57': spatial(),
              'forecast_temperature_reserved2026': forecast()}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
