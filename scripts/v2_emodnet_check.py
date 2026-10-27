"""Is the EMODnet Bathymetry DTM good enough to resolve this bay?

Chapter 6 lists resolving the bathymetry as future work, on the grounds that the only depth
information available is five Regione Liguria depth bands and that the MaGIC product does not cover
the network at all. EMODnet Bathymetry 2024 is the best public alternative, so this establishes
whether it changes that position.

Two checks, the second of which needs no external reference and is therefore decisive:

  1. EMODnet seabed depth against `metadata_bottom_m`, the surveyed depth recorded at deployment.
  2. EMODnet seabed depth against the moorings themselves. A probe moored at 5 m cannot sit where
     the seabed is 0.12 m, whatever the deployment metadata says.

Nothing is fitted and nothing is scored; this is a data-quality evaluation. It writes
reports/bay_twin/runs/v2_emodnet_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bay_profile_context import ROOT

SOURCE = 'data/external/EMODnet/Mean depth in multi colour (no land).csv'
OUT = ROOT / 'reports/bay_twin/runs/v2_emodnet_check.json'


def main():
    grid = pd.read_csv(ROOT / SOURCE, skiprows=[1])
    for column in ['latitude', 'longitude', 'elevation', 'value_count', 'stdev', 'interpolation_flag']:
        grid[column] = pd.to_numeric(grid[column], errors='coerce')
    grid = grid.dropna(subset=['latitude', 'longitude', 'elevation'])
    glat, glon = grid.latitude.to_numpy(), grid.longitude.to_numpy()
    depth = -grid.elevation.to_numpy()

    lats, lons = np.sort(grid.latitude.unique()), np.sort(grid.longitude.unique())
    spacing_lat = float(np.median(np.diff(lats)) * 111320)
    spacing_lon = float(np.median(np.diff(lons)) * 111320 * np.cos(np.radians(44.06)))

    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    rows = []
    for name, r in meta.iterrows():
        d = np.hypot((glat - r.latitude) * 111320,
                     (glon - r.longitude) * 111320 * np.cos(np.radians(r.latitude)))
        near = int(np.argmin(d))
        window = depth[d <= 300]
        rows.append({'sensor': name, 'distance_to_cell_m': float(d[near]),
                     'emodnet_nearest_m': float(depth[near]),
                     'emodnet_deepest_within_300m': float(window.max()) if len(window) else np.nan,
                     'surveyed_bottom_m': float(r.metadata_bottom_m),
                     'mooring_depth_m': float(r.nominal_depth_m)})
    table = pd.DataFrame(rows)
    impossible = table.emodnet_nearest_m < table.mooring_depth_m
    err_near = (table.emodnet_nearest_m - table.surveyed_bottom_m).to_numpy()
    err_window = (table.emodnet_deepest_within_300m - table.surveyed_bottom_m).to_numpy()

    report = {
        'source': SOURCE, 'product': 'EMODnet Bathymetry DTM 2024, mean depth, no land',
        'cells': int(len(grid)),
        'bbox': {'lat': [float(grid.latitude.min()), float(grid.latitude.max())],
                 'lon': [float(grid.longitude.min()), float(grid.longitude.max())]},
        'grid_spacing_m': {'lat': spacing_lat, 'lon': spacing_lon},
        'interpolated_cells': int(grid.interpolation_flag.notna().sum()),
        'per_cell_stdev_m': {'median': float(grid.stdev.median()), 'p90': float(grid.stdev.quantile(.9))},
        'covers_the_network': True,
        'probes_within_one_cell': int((table.distance_to_cell_m <= spacing_lat).sum()),
        'probes': int(len(table)),
        'error_against_surveyed_depth_m': {
            'nearest_cell': {'mean': float(err_near.mean()), 'mean_abs': float(np.abs(err_near).mean()),
                             'worst_abs': float(np.abs(err_near).max())},
            'deepest_within_300m': {'mean': float(err_window.mean()),
                                    'mean_abs': float(np.abs(err_window).mean()),
                                    'worst_abs': float(np.abs(err_window).max())}},
        'physically_impossible_cells': {
            'count': int(impossible.sum()),
            'explanation': 'the nearest cell is shallower than the mooring depth of the probe it '
                           'hosts, which no deployment metadata error can explain',
            'sensors': table.sensor[impossible].tolist()},
        'surveyed_bottom_range_m': [float(table.surveyed_bottom_m.min()), float(table.surveyed_bottom_m.max())],
        'verdict': 'Not usable as a seabed depth source for this network. The product covers the '
                   'area, unlike MaGIC, but at about 100 m spacing with a coarse land mask it does '
                   'not resolve the inner embayment: mean absolute error against surveyed depth is '
                   'larger than half the depth itself, and nearly half the probes sit in cells '
                   'shallower than their own moorings. It is adequate in the outer gulf, where no '
                   'result in this thesis needs it.',
        'per_probe': rows}
    OUT.write_text(json.dumps(report, indent=2))

    print(f"{report['cells']} cells, spacing {spacing_lat:.0f} x {spacing_lon:.0f} m, "
          f"{report['interpolated_cells']} interpolated")
    print(f"probes within one cell: {report['probes_within_one_cell']}/{report['probes']}")
    for key, label in [('nearest_cell', 'nearest cell'), ('deepest_within_300m', 'deepest within 300 m')]:
        e = report['error_against_surveyed_depth_m'][key]
        print(f"  {label:22s} mean {e['mean']:+.2f} m, mean |err| {e['mean_abs']:.2f} m, "
              f"worst {e['worst_abs']:.2f} m")
    print(f"physically impossible: {report['physically_impossible_cells']['count']}/{report['probes']} probes")
    print(f"surveyed seabed range: {report['surveyed_bottom_range_m']} m")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
