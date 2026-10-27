"""The constant that turns the pipeline's oxygen into mg/l, from each hour's seawater density.

The Gen-2 sonde exports carry dissolved oxygen as the instrument's native mg/l reading (column 6)
and as a micromolar column (column 13) equal to mg/l x 31.2512 x 1000/rho, with rho the export's own
density column (column 11): a quantity per kilogram of seawater. The pipeline reads the micromolar
column, which in some sonde-months also carries the operator's drift correction towards the casts,
so it is converted with the density alone (0.0319988 x rho/1000 mg/l per umol/kg), never through the
native column. This script averages the density to the hour exactly as
src/data/bay_observations.read_export averages oxygen, falls back to TEOS-10 from the hour's
temperature, salinity and pressure where the density column is missing, and reports:

  c          the mean factor over every valid hourly oxygen observation of the five sondes
  hourly     the factor per sonde and hour (for per-reading conversions)
  native     how the native mg/l column compares, which shows where the correction applies

  python -m scripts.v2_oxygen_units   ->  reports/bay_twin/runs/v2_oxygen_units.json
                                          reports/bay_twin/runs/v2_oxygen_units/hourly_factor.csv.gz
"""
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.thermodynamics import seawater_density_kg_per_m3  # noqa: E402

RAW = ROOT / 'data/raw'
OUT = ROOT / 'reports/bay_twin/runs/v2_oxygen_units.json'
FOLDER = ROOT / 'reports/bay_twin/runs/v2_oxygen_units'
HOURLY = FOLDER / 'hourly_factor.csv.gz'
SONDES = ('TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)', 'TMitiliO_05m-CTD(calibrato)',
          'TPortovenereP_02m-CTD(calibrato)', 'TPortovenereS_02m-CTD(calibrato)')
MG_PER_UMOL = 31.9988e-3            # mg of O2 per umol: per-litre sources convert by this alone
COLS = {'pressure': 3, 'temperature': 4, 'mg': 6, 'salinity': 10, 'rho': 11, 'umol': 13}


def hourly(path):
    """Oxygen, density and the TEOS-10 inputs of one export, averaged to the hour as read_export does."""
    frame = pd.read_csv(path, comment='#', sep=r'\s+', header=None, encoding='cp1252')
    if frame.shape[1] != 18:
        raise SystemExit(f'{path.name}: expected the 18-column export, found {frame.shape[1]}')
    index = pd.to_datetime(frame[0] + ' ' + frame[1], format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    out = pd.DataFrame({k: pd.to_numeric(frame[c], errors='coerce').to_numpy() for k, c in COLS.items()},
                       index=pd.DatetimeIndex(index))
    out = out.loc[out.index.notna()].sort_index()
    # the loader's screening of the oxygen it reads; the other columns screened to physical ranges
    out['umol'] = out.umol.where(out.umol.between(0, 600))
    out['mg'] = out.mg.where(out.mg > 0)
    out['rho'] = out.rho.where(out.rho.between(1000, 1040))
    out['temperature'] = out.temperature.where(out.temperature.between(-2, 40))
    out['salinity'] = out.salinity.where(out.salinity.between(0, 42))
    out['pressure'] = out.pressure.where(out.pressure.between(0, 100))
    out = out.groupby(level=0).mean()
    h = out.resample('1h', closed='left', label='right').mean()
    h = h[h.umol.notna()]
    teos = seawater_density_kg_per_m3(h.temperature.to_numpy(), h.salinity.to_numpy(),
                                      h.pressure.fillna(0).to_numpy())
    h['rho_used'] = h.rho.where(h.rho.notna(), pd.Series(teos, index=h.index))
    h['source'] = np.where(h.rho.notna(), 'export', np.where(np.isfinite(teos), 'teos10', 'none'))
    h['factor'] = MG_PER_UMOL * h.rho_used / 1000
    return h


def main():
    per_sonde, rows = {}, []
    for sonde in SONDES:
        h = hourly(RAW / f'{sonde}.txt')
        valid = h[h.factor.notna()]
        native = (valid.mg / (valid.umol * valid.factor)).dropna()   # 1 where the export is uncorrected
        per_sonde[sonde] = {
            'oxygen_hours': int(len(h)), 'hours_with_density': int(len(valid)),
            'density_from_teos10': int((valid.source == 'teos10').sum()),
            'factor_mean': float(valid.factor.mean()), 'factor_min': float(valid.factor.min()),
            'factor_max': float(valid.factor.max()),
            'native_over_converted_median': float(native.median()),
            'hours_native_within_0p5pct': float((native.sub(1).abs() <= 0.005).mean()),
        }
        rows.append(pd.DataFrame({'sensor': sonde, 'time': valid.index.astype(str),
                                  'factor': valid.factor.to_numpy(), 'source': valid.source.to_numpy()}))
    allh = pd.concat(rows, ignore_index=True)
    f = allh.factor.to_numpy()
    c = float(f.mean())
    report = {
        'what': 'mg/l per (umol/kg) for the pipeline oxygen: 0.0319988 x rho/1000 per hour, rho from the '
                'export density column or TEOS-10',
        'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json',
                        'config/bay_reserved_protocol_v2_addendum60_amendment1.json'],
        'c_mg_per_l_per_umol_kg': c,
        'mg_per_l_per_umol_l': MG_PER_UMOL,
        'implied_mean_density_kg_m3': c / MG_PER_UMOL * 1000,
        'hours': int(len(f)),
        'factor_spread': {'min': float(f.min()), 'p01': float(np.percentile(f, 1)),
                          'p99': float(np.percentile(f, 99)), 'max': float(f.max()),
                          'p01_relative': float(np.percentile(f, 1) / c - 1),
                          'p99_relative': float(np.percentile(f, 99) / c - 1),
                          'max_abs_relative': float(np.max(np.abs(f / c - 1)))},
        'per_sonde': per_sonde,
        'hourly_factor': 'runs/v2_oxygen_units/hourly_factor.csv.gz',
    }
    HOURLY.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as gz:
        gz.write(allh.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    HOURLY.write_bytes(buf.getvalue())
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    s = report['factor_spread']
    print(f'c = {c:.6f} mg/l per umol/kg (density {report["implied_mean_density_kg_m3"]:.2f} kg/m3) over '
          f'{len(f)} hours; p01/p99 {s["p01_relative"]:+.3%} {s["p99_relative"]:+.3%}, max {s["max_abs_relative"]:.3%}')
    for k, v in per_sonde.items():
        print(f'  {k:34s} {v["hours_with_density"]:6d}/{v["oxygen_hours"]:6d} h, teos10 {v["density_from_teos10"]:5d}, '
              f'native/converted median {v["native_over_converted_median"]:.4f}, within 0.5 % {v["hours_native_within_0p5pct"]:.1%}')


if __name__ == '__main__':
    main()
