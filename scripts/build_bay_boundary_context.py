"""The Copernicus boundary context of every pre-2026 cast bin: lagged model priors for the field.

For each bin of the casts `load_profiles` returns, the Copernicus Marine daily analysis at or
before the cast time minus 48 hours, at the nearest horizontal cell, interpolated to the bin depth
(`boundary_<variable>`), and the same profile's value at that depth minus its value at 3 m
(`boundary_<variable>_vertical_delta`, the offset the released background adds). Temperature,
salinity and oxygen, from the three files in data/external/Copernicus/T2023_2026.

Writes reports/bay_twin/profile_boundary_context.csv.gz and profile_boundary_provenance.json. The
released spatial recipe and every Copernicus arm read the CSV (src/experiments/v2_field*.py,
bay_field_export.py, scripts/v2_background_bias.py).

The computation is the retired producer of the same name (inrim2026 commit 182a8a2^), unchanged.
`src/experiments/v2_field_reserved2026_copernicus.extract` is a second implementation of the same
rule; its gate G1 reproduces this file's numeric columns on all 2,208 rows. Because a gzip header
carries a timestamp, a rerun whose table is identical to the published one leaves the published
file (and the provenance JSON that records its digest) untouched instead of rewriting the same
numbers under a new digest; `--out DIR` writes elsewhere, for checking before anything is replaced.

Usage: python -m scripts.build_bay_boundary_context [--out DIR]
"""
import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bay_profile_context import ROOT, load_profiles  # noqa: E402

FOLDER = ROOT / 'data/external/Copernicus/T2023_2026'
PRODUCTS = [('phy-tem', 'thetao', 'temperature', 'degrees_C'), ('phy-sal', 'so', 'salinity', '0.001'),
            ('bgc-bio', 'o2', 'oxygen', 'mmol m-3')]
LIMITS = ['Daily analysis/forecast archive lacks historical issue/version timestamps; lag does not prove historical availability.',
          'One offshore latitude row is coarse model context, not local bay truth. Nearest horizontal cell used.',
          'Potential temperature used as shallow-water temperature proxy; salinity product uses its archived convention.',
          'Outside model depth range, nearest endpoint retained and flagged; no claim of resolved deep structure.',
          'Vertical-difference baseline references 3 m; continuous sensors have varying depths, so this is approximate.']


def build():
    import xarray as xr
    from pyproj import Geod
    d, _ = load_profiles()
    output = d[['cast_id', 'depth_m']].copy()
    files = {}
    geod = Geod(ellps='WGS84')
    for token, var, label, units in PRODUCTS:
        path = next(FOLDER.glob('*' + token + '*.nc'))
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset = xr.open_dataset(path)
        assert dataset[var].attrs['units'] == units
        depths = dataset.depth.to_numpy().astype(float)
        values, relative, stamps, distances, clamped = [], [], [], [], []
        for row in d.itertuples():
            # Daily model time plus 48 hours <= query time. Analyses may have been revised later;
            # this explicit lag is a proxy, not issue-time evidence.
            cutoff = (row.t_utc - pd.Timedelta(hours=48)).tz_localize(None).to_datetime64()
            profile = dataset[var].sel(time=cutoff, method='pad').sel(latitude=row.lat, longitude=row.lon,
                                                                      method='nearest')
            array = profile.to_numpy().astype(float)
            values.append(float(np.interp(row.depth_m, depths, array)))
            relative.append(float(np.interp(row.depth_m, depths, array) - np.interp(3., depths, array)))
            stamps.append(str(pd.Timestamp(profile.time.item()).tz_localize('UTC')))
            _, _, distance = geod.inv(row.lon, row.lat, float(profile.longitude), float(profile.latitude))
            distances.append(distance)
            clamped.append(not (depths.min() <= row.depth_m <= depths.max()))
        output['boundary_' + label] = values
        output['boundary_' + label + '_vertical_delta'] = relative
        if label == 'temperature':
            output['model_time'] = stamps
            output['boundary_distance_m'] = distances
            output['outside_model_depth'] = clamped
        dataset.close()
    if not np.isfinite(output.filter(regex='^boundary_').to_numpy()).all():
        raise ValueError('Missing boundary model context')
    assert (pd.to_datetime(output.model_time, utc=True) + pd.Timedelta(hours=48) <= d.t_utc).all()
    return output, files, depths


def _table_text(path):
    with gzip.open(path, 'rt', encoding='utf8', newline='') as f:
        return f.read()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, default=ROOT / 'reports/bay_twin')
    args = p.parse_args(argv)
    output, files, depths = build()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'profile_boundary_context.csv.gz'
    text = output.to_csv(index=False)
    if path.exists() and _table_text(path) == text:
        print('table identical to', path, '- left untouched')
    else:
        buffer = io.BytesIO()
        with gzip.GzipFile(filename='', mode='wb', fileobj=buffer, mtime=0) as gz:
            gz.write(text.encode('utf8'))
        path.write_bytes(buffer.getvalue())
        print('wrote', path)
    report = {'input_sha256': files, 'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'rows': len(output),
              'distance_min_m': float(output.boundary_distance_m.min()),
              'distance_max_m': float(output.boundary_distance_m.max()),
              'depth_clamped_rows': int(output.outside_model_depth.sum()),
              'model_depth_range_m': depths[[0, -1]].tolist(),
              'lag_hours': 48, 'oxygen_units': 'mmol/m3 is numerically equal to umol/L', 'limits': LIMITS}
    target = args.out / 'profile_boundary_provenance.json'
    new = json.dumps(report, indent=2)
    if target.exists() and json.loads(target.read_text()) == report:
        print('provenance identical to', target, '- left untouched')
    else:
        target.write_text(new)
        print('wrote', target)


if __name__ == '__main__':
    main()
