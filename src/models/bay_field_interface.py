"""Supported coordinate/depth hindcasts from verified linear field checkpoints."""
import hashlib,json
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer,Geod
from shapely.geometry import Point,MultiPoint,shape
from shapely.strtree import STRtree
from src.data.bay_profile_context import ROOT,load_profiles,load_observations,context_features
from src.experiments.bay_field_export import predict_checkpoint
from src.models.bay_localized_update import supported_field_state
from src.data.bay_observations import load_bands

FIELD_RUN='20260917T011757Z_field_export_pilot'
UPDATE_RUN='20260917T080417Z_state_update_pilot'
VARIABLES=['temperature','salinity','oxygen']


def checked_json(path,digest=None):
    if digest and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Artifact hash mismatch: '+str(path))
    return json.loads(path.read_text())


def utc_time(value):
    t=pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:raise ValueError('A valid timezone-aware timestamp is required')
    return t.tz_convert('UTC')


def matched_bottom_metadata(records,x,y,stamp):
    distance=np.hypot(records.UTM_X-x,records.UTM_Y-y)
    matches=records[(distance<=10)&(records.t_utc<=stamp)&records.seabed_m.notna()]
    if not len(matches):return None
    low=float(matches.seabed_m.min());high=float(matches.seabed_m.max())
    return dict(status='coordinate_matched_bottom_metadata' if low==high else 'conflicting_bottom_metadata',
        bottom_low_m=low,bottom_high_m=high,metadata_cast_ids=sorted(matches.cast_id.unique()))


class BayField:
    def __init__(self):
        self.field_root=ROOT/'reports/bay_twin/runs'/FIELD_RUN
        self.update_root=ROOT/'reports/bay_twin/runs'/UPDATE_RUN
        for root in [self.field_root,self.update_root]:
            if checked_json(root/'verification.json')['state']!='VERIFIED':raise ValueError('Unverified run')
        hashes=checked_json(self.field_root/'results/status_field_export_temporal.json')['result_sha256']
        self.models={p.name:checked_json(p,hashes[p.relative_to(self.field_root/'results').as_posix()])
            for p in (self.field_root/'results/field_export_temporal').glob('*_model.json')}
        status=checked_json(self.update_root/'results/status_state_update_temporal.json')
        cal_path=self.update_root/'results/state_update_temporal/calibration.json'
        self.calibrations={c['checkpoint']:c for c in checked_json(cal_path,status['result_sha256']['state_update_temporal/calibration.json'])}
        # Ensure current observations and geometry match the immutable experiment inputs.
        snapshot=checked_json(self.update_root/'snapshot_manifest.json')
        for name,digest in snapshot['files_sha256'].items():
            if name.startswith('data/') or name in ['reports/bay_twin/sensor_geometry.csv','reports/bay_twin/gen2_profile_inventory.csv']:
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
        self.meta,self.frames=load_observations()
        profiles,_=load_profiles()
        self.geometry=profiles[['cast_id','station','t_utc','UTM_X','UTM_Y','seabed_m','depth_m']].copy()
        self.transformer=Transformer.from_crs(4326,32632,always_xy=True)
        land=checked_json(ROOT/'data/external/CoastalZones2018/bay_landcover.geojson')
        self.land_tree=STRtree([shape(f['geometry']) for f in land['features']
            if not f['properties'].get('NODATA_18') and int(f['properties']['CODE_5_18'])<80000])
        del land
        self.bands=load_bands(ROOT/'data/external/Batimetrie_RegioneLiguria/Batimetrie.kml')
        self.geometry_lookup=lru_cache(maxsize=128)(self.geometry_at)
        provenance=checked_json(ROOT/'reports/bay_twin/profile_boundary_provenance.json')
        self.products=[]
        for token,var,unit in [('phy-tem','thetao','degrees_C'),('phy-sal','so','0.001'),('bgc-bio','o2','mmol m-3')]:
            names=[n for n in provenance['input_sha256'] if token in n]
            if len(names)!=1:raise ValueError('Ambiguous boundary product')
            path=ROOT/names[0]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance['input_sha256'][names[0]]:raise ValueError('Boundary hash mismatch')
            with xr.open_dataset(path) as ds:
                if ds[var].attrs['units']!=unit:raise ValueError('Unexpected boundary units')
            self.products.append((path,var))

    def geometry_at(self,lon,lat):
        point=Point(lon,lat)
        if len(self.land_tree.query(point,predicate='intersects')):
            return {'status':'mapped_land_2018'}
        matches=[(lo,hi) for lo,hi,g,_ in self.bands if g.covers(point)]
        if not matches:return {'status':'unknown_bathymetry'}
        return {'status':'regional_depth_interval','bottom_low_m':min(a for a,b in matches),'bottom_high_m':max(b for a,b in matches)}

    def boundary(self,lon,lat,depths,stamp):
        cutoff=(stamp-pd.Timedelta(hours=48)).tz_localize(None).to_datetime64()
        absolute=[];delta=[];details=[]
        for path,var in self.products:
            with xr.open_dataset(path) as ds:
                p=ds[var].sel(time=cutoff,method='pad').sel(latitude=lat,longitude=lon,method='nearest')
                t=pd.Timestamp(p.time.values).tz_localize('UTC')
                if not pd.Timedelta(hours=48)<=stamp-t<=pd.Timedelta(hours=72):raise ValueError('Boundary unavailable or too stale')
                z=ds.depth.to_numpy().astype(float);values=p.to_numpy().astype(float)
                a=np.interp(depths,z,values);absolute.append(a);delta.append(a-np.interp(3.,z,values))
                distance=Geod(ellps='WGS84').inv(lon,lat,float(p.longitude),float(p.latitude))[2]
                details.append({'variable':var,'model_time':str(t),'distance_m':distance,'minimum_depth_m':float(z.min()),'maximum_depth_m':float(z.max()),
                    'depth_clamped':((np.asarray(depths)<z.min())|(np.asarray(depths)>z.max())).tolist()})
        first=details[0]
        for info in details[1:]:
            if (info['model_time'],info['minimum_depth_m'],info['maximum_depth_m'])!=(first['model_time'],first['minimum_depth_m'],first['maximum_depth_m']) or abs(info['distance_m']-first['distance_m'])>1e-6:
                raise ValueError('Boundary product grids differ from validated recipe')
        extras=np.column_stack(absolute+delta+[np.full(len(depths),first['distance_m']),first['depth_clamped']])
        if not np.isfinite(extras).all():raise ValueError('Nonfinite boundary context')
        return extras,details

    def select(self,stamp):
        for name,cp in self.models.items():
            window=cp['selection_window']
            if pd.Timestamp(window['train_end'],tz='UTC')<=stamp<pd.Timestamp(window['selection_end'],tz='UTC'):
                if pd.Timestamp(cp['contract']['last_training'])>=stamp:raise ValueError('Checkpoint is not causal')
                c=self.calibrations[name]
                if c['contract']!=cp['contract']:raise ValueError('Calibration/model contracts differ')
                return name,cp,c
        raise ValueError('Outside checkpoint development windows')

    def query(self,lon,lat,depths,time,observations_through=None,missing_variables=(),exclude_sensors=(),frames=None):
        stamp=utc_time(time)
        if not np.isfinite([lon,lat]).all() or not -180<=lon<=180 or not -90<=lat<=90:raise ValueError('Invalid coordinates')
        depths=np.asarray(depths,dtype=float)
        if depths.ndim!=1 or len(depths)==0 or not np.isfinite(depths).all() or (depths<0).any():raise ValueError('Finite nonnegative depths required')
        cutoff=stamp if observations_through is None else utc_time(observations_through)
        if cutoff>stamp:raise ValueError('Observation cutoff cannot exceed query time')
        if set(missing_variables)-set(VARIABLES):raise ValueError('Unknown missing variable')
        if set(exclude_sensors)-set(self.frames):raise ValueError('Unknown excluded sensor')
        result={'status':'UNSUPPORTED','time':str(stamp),'longitude':float(lon),'latitude':float(lat),'predictions':[],
            'provenance':{'field_run':FIELD_RUN,'background_run':UPDATE_RUN,'observation_cutoff':str(cutoff),
                'feed_mode':'verified_archive' if frames is None else 'supplied_frames_unverified',
                'excluded_sensors':list(exclude_sensors),'missing_variables':list(missing_variables)},
            'uncertainty':None,'uncertainty_status':'spatial_intervals_not_calibrated',
            'warning':{'status':'not_issued','reason':'Field estimates are not a validated decline-warning or biological-stress decision.'},
            'limits':['Retrospective development hindcast; no live ingestion or future field forecasting.',
                'Query-local observations within250m excluded; observations older than6h are unavailable.',
                'Copernicus48h lag does not establish historical product issue availability.',
                'Observation support is not a guarantee of geographic accuracy. Geometry scenarios are not confidence intervals.']}
        if stamp>=pd.Timestamp('2026-01-01',tz='UTC'):result['reason']='2026 archive reserved';return result
        try:name,cp,calibration=self.select(stamp)
        except ValueError as exc:result['reason']=str(exc);return result
        result['provenance'].update(checkpoint=name,last_training=cp['contract']['last_training'])
        geometry=dict(self.geometry_lookup(lon,lat));result['geometry']=geometry
        if geometry['status']!='regional_depth_interval':result['reason']=geometry['status'];return result
        x,y=self.transformer.transform(lon,lat)
        # Deployment coordinates and bottom metadata can change within one name.
        # Use all coordinate-matched records available by this query, not first station row.
        matched=matched_bottom_metadata(self.geometry,x,y,stamp)
        if matched is not None:geometry.update(matched)
        train=self.geometry[self.geometry.cast_id.isin(cp['contract']['training_cast_ids'])]
        hull=MultiPoint(train[['UTM_X','UTM_Y']].drop_duplicates().to_numpy()).convex_hull
        if not hull.buffer(10).covers(Point(x,y)):result['reason']='outside_training_station_hull';return result
        supplied=self.frames if frames is None else frames
        if list(supplied)!=list(self.meta.index):raise ValueError('Feed must retain verified sensor identities and order')
        feed={n:f.loc[f.index<=cutoff].copy() for n,f in supplied.items()}
        for n,f in feed.items():
            for v in f:
                if n in exclude_sensors or v in missing_variables:f[v]=np.nan
        for depth in depths:
            row={'depth_m':float(depth),'status':'UNSUPPORTED','scenarios':[]};result['predictions'].append(row)
            if depth>geometry['bottom_high_m']:row['reason']='below_seabed_interval';continue
            if depth>geometry['bottom_low_m']:row['reason']='uncertain_water_column';continue
            if not train.depth_m.min()<=depth<=train.depth_m.max():row['reason']='outside_training_depth_range';continue
            extras,info=self.boundary(lon,lat,[depth],stamp)
            row['boundary']=info;row['status']='EXPERIMENTAL_HINDCAST'
            for bottom in sorted(set([geometry['bottom_low_m'],geometry['bottom_high_m']])):
                query=pd.DataFrame([{'cast_id':'query','station':'query','t_utc':stamp,'UTM_X':x,'UTM_Y':y,'depth_m':depth,'seabed_m':bottom}])
                context=context_features(query,self.meta,feed)
                learned=predict_checkpoint(cp,context,extras)
                state,flags=supported_field_state(learned,context,extras,calibration)
                values={}
                for j,v in enumerate(VARIABLES):
                    age=context[v+'_age'].iloc[0]
                    values[v]={'value':float(state[0,j]),'units':['degC','PSU','umol/L'][j],'support':str(flags[0,j]),
                        'source_count':int(context[v+'_count'].iloc[0]),'weighted_observation_age_hours':float(age) if np.isfinite(age) else None}
                    if v=='oxygen' and state[0,j]<0:values[v].update(value=None,support='invalid_negative_prediction')
                row['scenarios'].append({'bottom_m':bottom,'values':values,'boundary_depth_clamped':bool(extras[0,-1])})
        if any(r['scenarios'] for r in result['predictions']):result['status']='EXPERIMENTAL_HINDCAST'
        else:result['reason']='no_supported_depths'
        return result
