"""Sensor geometry from the regional depth bands and the CLMS 2018 land cover, and the Chapter 3 map.

Reads reports/bay_twin/sensor_inventory.csv (written by `python -m src.data.bay_observations`),
data/external/CoastalZones2018/bay_landcover.geojson, the Regione Liguria depth bands and the CTD
archive, and writes

    reports/bay_twin/sensor_geometry.csv         the inventory plus land distance and land-cover
                                                 fractions within 500 m and 1500 m of each probe
    reports/bay_twin/geometry.npz                pairwise distances, co-location (< 60 m) and the
                                                 breakwater-crossing matrix
    reports/bay_twin/geometry_provenance.json    the land-cover digest and what each field means
    thesis_template/images/figures/fig_bay_sensor_geometry.png   Figure 3.1

The three data files are computed exactly as the retired producer of the same name computed them
(inrim2026 commit 182a8a2^), so a rerun reproduces the published CSV and JSON byte for byte and
the NPZ array for array; `--out DIR` writes them elsewhere so that can be checked before anything
is overwritten. The map is drawn for the thesis: the eleven sites under the names Chapter 3 uses,
the EXO2 sondes, the seven CTD stations by code, and a legend for the depth bands.

Usage: python -m scripts.build_bay_geometry [--out DIR] [--figure PATH] [--no-figure]
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURE = ROOT / 'thesis_template/images/figures/fig_bay_sensor_geometry.png'
LANDCOVER = ROOT / 'data/external/CoastalZones2018/bay_landcover.geojson'
BANDS = ROOT / 'data/external/Batimetrie_RegioneLiguria/Batimetrie.kml'
CASTS = ROOT / 'data/processed/ctd_all_casts.csv.gz'

#: The straight breakwater segment of the prior project's notebook (cell 8). It is kept for the
#: crossing matrix in geometry.npz, which is how it was published; the map does not draw it,
#: because it is an approximation the thesis never uses.
BARRIER_LONLAT = ((9.855842669002794, 44.06881022379191), (9.878986832216222, 44.07973866886565))

#: Site names as Chapter 3 writes them (sensor_inventory.csv carries the export's own spelling).
SITE_NAMES = {'Baia_S.Teresa': 'Baia Santa Teresa', 'Boa_S.Teresa': 'Boa Santa Teresa'}

#: Depth bands of the Regione Liguria layer, shallow to deep, on one sequential blue ramp.
BAND_COLOURS = {(0.0, 10.0): '#deebf7', (10.0, 20.0): '#c6dbef', (20.0, 30.0): '#9ecae1',
                (30.0, 50.0): '#6baed6', (50.0, 100.0): '#4292c6'}
LAND = '#c9c9bf'
SITE = '#d95f02'      # orange and purple: the validated pair of scripts/v2_figures.py
STATION = '#7b3294'
INK = '#1a1a1a'

#: Label offsets in points, chosen by eye so that no two labels collide at the printed size. The
#: Mitili sites are labelled on the inner (north-west) side of the breakwater, the breakwater on
#: its outer side.
SITE_OFFSETS = {
    'Portovenere Secco': (14, -2), 'Portovenere Benzinaio': (12, -4),
    'Portovenere Pontile': (10, 8), 'Mitili Ovest': (-12, 2), 'Mitili Centro Ovest': (-6, 12),
    'Mitili Centro': (-6, 12), 'Mitili Centro Est': (-6, 12), 'Mitili Est': (-8, 14),
    'Boa Santa Teresa': (6, -16), 'Baia Pontone': (-30, 30), 'Baia Santa Teresa': (8, 26),
}
#: Labels centred on their offset point rather than starting or ending there (the default follows the sign of
#: the horizontal offset): Baia Santa Teresa sits too close to the frame for a label to its right.
SITE_ALIGN = {'Baia Santa Teresa': 'center'}
#: Two-line labels where one line would run off the frame.
SITE_LABELS = {'Boa Santa Teresa': 'Boa Santa\nTeresa', 'Baia Santa Teresa': 'Baia Santa\nTeresa (3)'}
STATION_OFFSETS = {
    'PPortovenereS': (10, -18), 'PPortovenereP': (10, 22), 'PMitiliO': (-4, -16),
    'PMitiliE': (10, -4), 'PBoaS': (10, 2), 'PBaia': (10, -12),
}
#: Where to name the breakwater, in UTM km, and the angle of its long axis.
BREAKWATER_LABEL = ((569.42, 4880.22), 34)
#: The map is drawn at the size it prints at (\textwidth, 416.7 pt, by the earlier drawing's height there) and
#: saved without cropping, so no text prints below 7 pt and the page around it does not move. The map takes the
#: box (left, bottom, width, height, inches) and keeps its true aspect inside it; the legends sit to its right.
SIZE_IN = (416.7 / 72, 4.2526)
MAP_BOX_IN = (0.52, 0.40, 3.62, 3.80)


def compute(out):
    """The three data files, computed as the published ones were."""
    from shapely import make_valid
    from shapely.geometry import LineString, Point, shape
    from shapely.ops import transform, unary_union
    from pyproj import Transformer

    out.mkdir(parents=True, exist_ok=True)
    sensors = pd.read_csv(ROOT / 'reports/bay_twin/sensor_inventory.csv')
    features = json.loads(LANDCOVER.read_text())['features']
    project = Transformer.from_crs(4326, 32632, always_xy=True).transform
    classes = {}
    for f in features:
        code = int(f['properties']['CODE_5_18'])
        if f['properties'].get('NODATA_18'):
            continue
        classes.setdefault(code, []).append(transform(project, make_valid(shape(f['geometry']))))
    land = unary_union([g for c, gs in classes.items() if c < 80000 for g in gs])
    categories = {'built': lambda c: c < 20000, 'port': lambda c: 12300 <= c < 12400,
                  'agriculture': lambda c: 20000 <= c < 30000,
                  'vegetation': lambda c: 30000 <= c < 60000}
    unions = {name: unary_union([g for c, gs in classes.items() if pred(c) for g in gs])
              for name, pred in categories.items()}
    for i, row in sensors.iterrows():
        pt = Point(row.x_m, row.y_m)
        sensors.loc[i, 'land_distance_2018_m'] = pt.distance(land)
        sensors.loc[i, 'inside_mapped_land_2018'] = land.covers(pt)
        for radius in [500, 1500]:
            buffer = pt.buffer(radius)
            for name, geom in unions.items():
                sensors.loc[i, f'{name}_fraction_{radius}m'] = buffer.intersection(geom).area / buffer.area
    xy = sensors[['x_m', 'y_m']].to_numpy()
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    barrier = LineString([project(*BARRIER_LONLAT[0]), project(*BARRIER_LONLAT[1])])
    crossing = np.zeros(distances.shape, dtype=bool)
    for i in range(len(xy)):
        for j in range(i):
            crossing[i, j] = crossing[j, i] = LineString([xy[i], xy[j]]).crosses(barrier)
    # Each file is rewritten only when its content changes. The CSV and JSON would be rewritten
    # byte for byte anyway, but a zip archive stamps its members with the time of writing, so an
    # unconditional save would give geometry.npz a new digest with the same arrays in it.
    _write_if_changed(out / 'sensor_geometry.csv', sensors.to_csv(index=False).encode('utf8'))
    arrays = {'distance_m': distances, 'near_site': distances < 60, 'barrier_crossing': crossing}
    npz = out / 'geometry.npz'
    if npz.exists():
        with np.load(npz) as old:
            same = sorted(old.files) == sorted(arrays) and all(
                old[k].dtype == v.dtype and np.array_equal(old[k], v) for k, v in arrays.items())
    else:
        same = False
    if not same:
        np.savez_compressed(npz, **arrays)
    _write_if_changed(out / 'geometry_provenance.json', json.dumps({
        'landcover_sha256': hashlib.sha256(LANDCOVER.read_bytes()).hexdigest(),
        'crs': 'EPSG:32632',
        'land_distance': 'Distance to CLMS 2018 non-water polygons, a coastline proxy; not a current surveyed shoreline.',
        'fractions': 'Mapped category area divided by full circular buffer area, including water.',
        'bathymetry': 'Regional interval bounds retained; do not differentiate band midpoints into a measured slope.',
        'barrier': 'Straight segment from supplied notebook cell 8; approximation, not surveyed mesh boundary.',
        'land_flagged_sensors': sensors.loc[sensors.inside_mapped_land_2018 == True, 'sensor'].tolist()},
        indent=2).encode('utf8'))
    return sensors, land, project


def _write_if_changed(path, data):
    """Write text bytes with the platform's line ending, as `DataFrame.to_csv(path)` and
    `Path.write_text` do, unless the file already holds exactly those bytes. `to_csv()` without a
    path already ends its lines with os.linesep, so endings are normalised to LF first; replacing
    LF alone turned CRLF into CR CR LF."""
    import os
    data = data.replace(b'\r\n', b'\n').replace(b'\n', os.linesep.encode())
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def stations():
    """One position per CTD station: the most frequent cast position (two stations have a
    handful of casts logged at a neighbouring point)."""
    casts = pd.read_csv(CASTS, usecols=['cast_id', 'station', 'UTM_X', 'UTM_Y']).drop_duplicates('cast_id')
    rows = []
    for name, g in casts.groupby('station'):
        x, y = g.groupby(['UTM_X', 'UTM_Y']).size().idxmax()
        rows.append({'station': name, 'x_m': float(x), 'y_m': float(y), 'casts': int(len(g))})
    return pd.DataFrame(rows)


def draw(sensors, land, project, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from shapely.ops import transform
    from src.data.bay_observations import load_bands

    sensors = sensors.copy()
    sensors['site_name'] = [SITE_NAMES.get(s, s.replace('_', ' ')) for s in sensors.site]
    sensors['sonde'] = sensors.instrument.str.contains('EXO2')
    sites = sensors.groupby('site_name').agg(x=('x_m', 'mean'), y=('y_m', 'mean'),
                                             probes=('sensor', 'size'), sonde=('sonde', 'any'))
    ctd = stations()
    x0, x1 = sensors.x_m.min() / 1000 - .8, sensors.x_m.max() / 1000 + .8
    y0, y1 = sensors.y_m.min() / 1000 - .8, sensors.y_m.max() / 1000 + .8

    fig = plt.figure(figsize=SIZE_IN)
    left, bottom, width, height = MAP_BOX_IN
    ax = fig.add_axes((left / SIZE_IN[0], bottom / SIZE_IN[1], width / SIZE_IN[0], height / SIZE_IN[1]))
    ax.set_anchor('SW')
    bands_seen = set()
    for lo, hi, geom, _ in load_bands(BANDS):
        geom = transform(project, geom)
        colour = BAND_COLOURS.get((lo, hi), '#2171b5')
        bands_seen.add((lo, hi))
        for g in ([geom] if geom.geom_type == 'Polygon' else list(geom.geoms)):
            if g.geom_type != 'Polygon':
                continue
            x, y = g.exterior.xy
            ax.fill(np.array(x) / 1000, np.array(y) / 1000, color=colour, linewidth=0, zorder=1)
            for ring in g.interiors:
                x, y = ring.xy
                ax.fill(np.array(x) / 1000, np.array(y) / 1000, color='white', linewidth=0, zorder=1)
    for g in getattr(land, 'geoms', [land]):
        if g.geom_type == 'Polygon':
            x, y = g.exterior.xy
            ax.fill(np.array(x) / 1000, np.array(y) / 1000, color=LAND, linewidth=0, zorder=2)

    ax.scatter(sites.x / 1000, sites.y / 1000, s=22 + 14 * sites.probes, c=SITE, edgecolor='white',
               linewidth=0.8, zorder=5)
    sondes = sites[sites.sonde]
    ax.scatter(sondes.x / 1000, sondes.y / 1000, s=150, marker='s', facecolor='none', edgecolor=INK,
               linewidth=0.9, zorder=4)
    for name, r in sites.iterrows():
        label = SITE_LABELS.get(name, name + (f' ({r.probes})' if r.probes > 1 else ''))
        ax.annotate(label, (r.x / 1000, r.y / 1000), xytext=SITE_OFFSETS.get(name, (6, 6)),
                    textcoords='offset points', fontsize=7.2, color=INK, zorder=7,
                    ha=SITE_ALIGN.get(name, 'right' if SITE_OFFSETS.get(name, (6, 6))[0] < 0 else 'left'),
                    va='center', linespacing=1.0,
                    arrowprops=dict(arrowstyle='-', color='#666666', linewidth=0.5,
                                    shrinkA=0, shrinkB=3))
    (bx, by), angle = BREAKWATER_LABEL
    ax.text(bx, by, 'Diga Foranea (breakwater)', rotation=angle, rotation_mode='anchor',
            ha='center', va='center', fontsize=7, color='#555555', style='italic', zorder=7)

    inside = ctd[(ctd.x_m / 1000).between(x0, x1) & (ctd.y_m / 1000).between(y0, y1)]
    ax.scatter(inside.x_m / 1000, inside.y_m / 1000, s=34, marker='^', c=STATION, edgecolor='white',
               linewidth=0.6, zorder=6)
    for r in inside.itertuples():
        dx, dy = STATION_OFFSETS.get(r.station, (6, -10))
        ax.annotate(r.station, (r.x_m / 1000, r.y_m / 1000), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7, color=STATION, style='italic',
                    ha='right' if dx < 0 else 'left', zorder=7,
                    arrowprops=dict(arrowstyle='-', color=STATION, linewidth=0.4, shrinkA=0, shrinkB=3))
    for r in ctd[~ctd.station.isin(inside.station)].itertuples():
        # A station beyond the frame is pointed at from the edge, with its distance from the
        # nearest station inside it, so the frame can stay on the bay.
        near = inside.loc[np.hypot(inside.x_m - r.x_m, inside.y_m - r.y_m).idxmin()]
        km = math.hypot(r.x_m - near.x_m, r.y_m - near.y_m) / 1000
        bearing = math.degrees(math.atan2(r.x_m - near.x_m, r.y_m - near.y_m)) % 360
        compass = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW',
                   'W', 'WNW', 'NW', 'NNW'][int((bearing + 11.25) // 22.5) % 16]
        ex = float(np.clip(r.x_m / 1000, x0 + .15, x1 - .15))
        ey = y0 + .06 if r.y_m / 1000 < y0 else y1 - .06
        ax.annotate('', xy=(ex, ey), xytext=(ex, ey + (.45 if ey < y1 - 1 else -.45)),
                    arrowprops=dict(arrowstyle='-|>', color=STATION, linewidth=0.9), zorder=6)
        ax.annotate(f'{r.station}: {km:.1f} km {compass} of {near.station}', (ex, ey + .2),
                    xytext=(6, 0), textcoords='offset points', fontsize=7, color=STATION,
                    style='italic', va='center', zorder=7)

    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel='UTM 32N easting (km)', ylabel='UTM 32N northing (km)')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    bands = [Patch(facecolor=BAND_COLOURS[b], edgecolor='#9a9a9a', linewidth=0.4,
                   label=f'{b[0]:.0f} to {b[1]:.0f} m') for b in sorted(bands_seen & set(BAND_COLOURS))]
    marks = [Patch(facecolor=LAND, label='Land (CLMS 2018)'),
             Line2D([], [], marker='o', linestyle='', markersize=6, markerfacecolor=SITE,
                    markeredgecolor='white', label='Moored site (probe\ncount if more than one)'),
             Line2D([], [], marker='s', linestyle='', markersize=8, markerfacecolor='none',
                    markeredgecolor=INK, label='Site with an\nEXO2 sonde'),
             Line2D([], [], marker='^', linestyle='', markersize=6, markerfacecolor=STATION,
                    markeredgecolor='white', label='CTD station')]
    first = ax.legend(handles=bands, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
                      fontsize=7, title='Depth band\n(Regione Liguria)', title_fontsize=7.5,
                      alignment='left', handlelength=1.6, borderaxespad=0)
    ax.add_artist(first)
    ax.legend(handles=marks, loc='upper left', bbox_to_anchor=(1.02, 0.60), frameon=False,
              fontsize=7, handlelength=1.6, borderaxespad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return sites, ctd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, default=ROOT / 'reports/bay_twin',
                   help='where the CSV, NPZ and provenance JSON go (default: reports/bay_twin)')
    p.add_argument('--figure', type=Path, default=FIGURE, help='where the map goes')
    p.add_argument('--no-figure', action='store_true')
    args = p.parse_args(argv)
    sensors, land, project = compute(args.out)
    print('geometry written to', args.out, '; land-flagged sensors:',
          sensors.loc[sensors.inside_mapped_land_2018 == True, 'sensor'].tolist())
    if not args.no_figure:
        sites, ctd = draw(sensors, land, project, args.figure)
        print(f'map: {len(sites)} sites, {int(sites.sonde.sum())} with a sonde, '
              f'{len(ctd)} CTD stations -> {args.figure}')


if __name__ == '__main__':
    main()
