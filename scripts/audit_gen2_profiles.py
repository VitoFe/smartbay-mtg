"""Audit the existing profile archive before using it to supervise coordinate/depth fields."""
from pathlib import Path
import json,re,hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'data/processed/ctd_all_casts.csv.gz'; data=pd.read_csv(path)
    data['t_utc']=pd.to_datetime(data.t_utc,utc=True)
    data=data.loc[data.t_utc>=pd.Timestamp('2024-07-24',tz='UTC')]
    rows=[]
    for cast,group in data.groupby('cast_id'):
        raw=ROOT/'data/raw'/(cast+'.txt'); meta={}
        if raw.exists():
            with raw.open(encoding='cp1252') as stream:
                header=''.join(next(stream,'') for _ in range(25))
            meta=dict(re.findall(r'^#\s*([^=\n]+)=(.*)$',header,re.M))
        first=group.iloc[0]
        lat=float(meta.get('Latitudine','nan')); lon=float(meta.get('Longitudine','nan'))
        rows.append({'cast_id':cast,'station':first.station,'time':str(first.t_utc),'bins':len(group),
            'min_depth_m':float(group.depth_m.min()),'max_depth_m':float(group.depth_m.max()),
            'metadata_bottom_m':float(first.seabed_m),'raw_found':raw.exists(),
            'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None,
            'raw_lat':lat,'raw_lon':lon,'processed_lat':float(first.lat),'processed_lon':float(first.lon),
            'coordinates_match_raw':bool(np.isclose(lat,first.lat,atol=1e-6,rtol=0) and np.isclose(lon,first.lon,atol=1e-6,rtol=0)),
            'instrument':meta.get('Sonda'),'depth_exceeds_metadata_by_gt_1m':bool(group.depth_m.max()>first.seabed_m+1)})
    table=pd.DataFrame(rows); out=ROOT/'reports/bay_twin'
    table.to_csv(out/'gen2_profile_inventory.csv',index=False)
    summary={'processed_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'casts':len(table),'bins':len(data),
        'stations':int(table.station.nunique()),'raw_missing':int((~table.raw_found).sum()),
        'raw_coordinate_mismatches':table.loc[~table.coordinates_match_raw,'cast_id'].tolist(),
        'casts_beyond_bottom_metadata':table.loc[table.depth_exceeds_metadata_by_gt_1m,'cast_id'].tolist(),
        'by_station':table.groupby('station').agg(casts=('cast_id','size'),min_depth_m=('min_depth_m','min'),max_depth_m=('max_depth_m','max')).reset_index().to_dict('records'),
        'scope':'Date-era subset of existing screened CTD archive. Profile instruments differ from EXO2 telemetry. These are sparse supervision, not hourly deep-field truth.'}
    (out/'gen2_profile_audit.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
