"""Raw Gen-2 observation audit. Never consumes filled HDF5 or model predictions."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def read_export(path,include_auxiliary=False):
    text = path.read_text(encoding='cp1252')
    meta = dict(re.findall(r'^#\s*([^=\n]+)=(.*)$', text, re.M))
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    frame = pd.read_csv(path, comment='#', sep=r'\s+', header=None, encoding='cp1252')
    index = pd.to_datetime(frame[0] + ' ' + frame[1], format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    if frame.shape[1] not in (6, 18):
        raise ValueError(f'Unsupported explicit export layout: {path.name}: {frame.shape[1]}')
    result = pd.DataFrame(index=pd.DatetimeIndex(index))
    result['temperature'] = pd.to_numeric(frame[4], errors='coerce').to_numpy()
    if frame.shape[1] == 18:
        result['oxygen'] = pd.to_numeric(frame[13], errors='coerce').to_numpy()
        if include_auxiliary:
            result['salinity']=pd.to_numeric(frame[10],errors='coerce').to_numpy()
    result = result.loc[result.index.notna()].sort_index()
    raw_count = len(result)
    duplicates = int(result.index.duplicated().sum())
    sentinels = (result <= -999).sum().to_dict()
    # Broad screening, not a biofouling correction. Preserve missing values.
    result['temperature'] = result.temperature.where(result.temperature.between(-2, 40))
    if 'oxygen' in result:
        result['oxygen'] = result.oxygen.where(result.oxygen.between(0, 600))
    if 'salinity' in result:
        result['salinity']=result.salinity.where(result.salinity.between(0,42))
    result = result.groupby(level=0).mean()
    # Label by END of observation hour: no future-within-hour availability leakage.
    result = result.resample('1h', closed='left', label='right').mean()
    return meta, result, {'raw_rows': raw_count, 'duplicate_timestamps': duplicates,
                          'sentinel_counts': sentinels}


def load_bands(path):
    from shapely.geometry import Polygon
    from shapely import make_valid
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    root = ET.parse(path)
    bands = []
    for mark in root.findall('.//k:Placemark', ns):
        fields = {x.attrib['name']: x.text for x in mark.findall('.//k:SimpleData', ns)}
        low, high = map(float, re.findall(r'\d+', fields['descrizione']))
        for poly in mark.findall('.//k:Polygon', ns):
            def ring(node):
                return [tuple(map(float, v.split(',')[:2])) for v in node.text.split()]
            outer = ring(poly.find('k:outerBoundaryIs/k:LinearRing/k:coordinates', ns))
            holes = [ring(x) for x in poly.findall('k:innerBoundaryIs/k:LinearRing/k:coordinates', ns)]
            geom = Polygon(outer, holes)
            bands.append((low, high, make_valid(geom), geom.is_valid))
    return bands


def audit(output=None):
    from shapely.geometry import Point, mapping
    from shapely.ops import transform
    from pyproj import Transformer
    output = output or ROOT / 'reports/bay_twin'
    output.mkdir(parents=True, exist_ok=True)
    kml = ROOT / 'data/external/Batimetrie_RegioneLiguria/Batimetrie.kml'
    bands = load_bands(kml)
    project = Transformer.from_crs(4326, 32632, always_xy=True).transform
    files = sorted((ROOT/'data/raw').glob('Nodo_*.txt')) + sorted((ROOT/'data/raw').glob('*-CTD(calibrato).txt'))
    records, series, sources = [], {}, {}
    for path in files:
        meta, frame, stats = read_export(path)
        frame = frame.loc['2024-07-24':]
        name = path.stem
        lon, lat = float(meta['Longitudine']), float(meta['Latitudine'])
        point = Point(lon, lat)
        matches = [(lo, hi) for lo, hi, geom, _ in bands if geom.covers(point)]
        x, y = project(lon, lat)
        record = dict(sensor=name, site=meta.get('Nome'), instrument=meta.get('Sonda'),
                      latitude=lat, longitude=lon, x_m=x, y_m=y,
                      nominal_depth_m=float(meta["Profondita'"]), metadata_bottom_m=float(meta['Fondale']),
                      bathy_low_m=min((v[0] for v in matches), default=None),
                      bathy_high_m=max((v[1] for v in matches), default=None),
                      bathy_matches=len(matches), source=path.relative_to(ROOT).as_posix(), **stats)
        for variable in frame:
            s=frame[variable]; valid=s.dropna()
            record[variable+'_count']=len(valid)
            record[variable+'_first']=str(valid.index.min()) if len(valid) else None
            record[variable+'_last']=str(valid.index.max()) if len(valid) else None
            record[variable+'_coverage']=float(s.notna().mean())
            series[name+'::'+variable]=s
        records.append(record)
        sources[record['source']]=hashlib.sha256(path.read_bytes()).hexdigest()
    table=pd.DataFrame(records)
    table.to_csv(output/'sensor_inventory.csv',index=False)
    pd.DataFrame(series).to_csv(output/'hourly_observations.csv.gz',compression='gzip')
    sources[kml.relative_to(ROOT).as_posix()]=hashlib.sha256(kml.read_bytes()).hexdigest()
    notes=[]
    for path in (ROOT/'data/notebooks_from_previous_projects').glob('*.ipynb'):
        obj=json.loads(path.read_text(encoding='utf8'))
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        notes.append({'file':path.name,'cells':len(obj['cells']), 'markdown':[ ''.join(c['source']) for c in obj['cells'] if c['cell_type']=='markdown']})
    report={'sources_sha256':sources,'sensors':len(records),'kml_polygons':len(bands),
            'kml_invalid_before_repair':sum(not b[3] for b in bands),
            'depth_bands':sorted(set((b[0],b[1]) for b in bands)), 'notebooks':notes,
            'constraints':['Depth bands are intervals, not a depth DEM.',
                          'No coastline supplied: bathymetric polygon edges include offshore boundaries, not just coast.',
                          'Hourly labels are interval ends in UTC. No interpolation or backward fill.',
                          'Gen 1 excluded by explicit source list. Gen 2 still requires QC.',
                          'Prior notebooks depend on absent watermodels configs; do not execute their training/fill cells.']}
    (output/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    features=[{'type':'Feature','properties':{'low_m':lo,'high_m':hi},'geometry':mapping(geom)} for lo,hi,geom,_ in bands]
    (output/'bathymetry_bands.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    print(table[['sensor','temperature_count','bathy_low_m','bathy_high_m','metadata_bottom_m']].to_string(index=False))


if __name__ == '__main__':
    audit()
