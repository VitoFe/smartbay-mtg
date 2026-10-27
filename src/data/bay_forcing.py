"""Causal meteorological context and geometric aggregation for thermal pilots."""
import numpy as np
import pandas as pd
import xarray as xr
from src.data.bay_profile_context import ROOT


def weather(times):
    raw = pd.read_csv(ROOT/'data/raw/T2-Meteo.txt', comment='#', sep=r'\s+', header=None, encoding='cp1252')
    t = pd.to_datetime(raw[0]+' '+raw[1], format='%d/%m/%Y %H:%M:%S', utc=True)
    # Exclude the reserved archive before any aggregation.
    keep = t < pd.Timestamp('2026-01-01', tz='UTC')
    fields = {'air':4, 'solar':8, 'wind':12, 'humidity':7, 'pressure':3}
    a = pd.DataFrame({k:pd.to_numeric(raw.loc[keep, c], errors='coerce').replace(-9999, np.nan).to_numpy()
                      for k,c in fields.items()}, index=pd.DatetimeIndex(t[keep]))
    bounds = {'air':(-10,50), 'solar':(0,1500), 'wind':(0,60), 'humidity':(0,100), 'pressure':(850,1100)}
    for k,(lo,hi) in bounds.items(): a.loc[~a[k].between(lo,hi),k] = np.nan
    # A label at q contains observations strictly earlier than q, including exact-bin-start values.
    hourly = a.resample('1h', closed='left', label='right').mean()
    means = hourly.rolling(6, min_periods=4).mean().reindex(times)
    return means


def reduced_geometry(names):
    root = ROOT/'reports/bay_twin/layered_mesh'
    c = pd.read_csv(root/'columns.csv')
    with np.load(root/'mesh_central.npz') as d: m = {k:d[k] for k in d.files}
    # Geographic partition is fixed without looking at labels. Volumes/face conductances sum exactly.
    region = (c.x_m.to_numpy() >= c.x_m.median()).astype(int) + 2*(c.y_m.to_numpy() >= c.y_m.median()).astype(int)
    group = region[m['column']]*3 + np.tile(np.arange(3), len(c))
    # Check the existing mesh really uses column-major layer order.
    assert np.array_equal(m['column'], np.repeat(np.arange(len(c)),3))
    n = 12; volume = np.bincount(group, weights=m['volume_m3'], minlength=n)
    horizontal = np.zeros((n,n)); vertical = horizontal.copy()
    for i,j,a,d,v in zip(m['edge_i'],m['edge_j'],m['face_area_m2'],m['distance_m'],m['vertical']):
        u,w = group[i],group[j]
        if u == w: continue
        target = vertical if v else horizontal; target[u,w] += a/d; target[w,u] += a/d
    surface = np.zeros(n); surface[np.arange(4)*3] = np.bincount(region,weights=c.area_m2,minlength=4)
    boundary_v = np.bincount(group, weights=m['volume_m3']*(c.open_boundary_length_m.to_numpy()[m['column']]>1e-5), minlength=n)
    xy = c.iloc[m['column']][['x_m','y_m']].to_numpy()
    meta = pd.read_csv(ROOT/'reports/bay_twin/sensor_geometry.csv').set_index('sensor').loc[names]
    audit = pd.read_csv(root/'sensor_bathymetry_audit.csv').set_index('sensor')
    dh2 = ((xy[:,None]-meta[['x_m','y_m']].to_numpy()[None])**2).sum(-1)
    dz = m['depth_m'][:,None]-meta.nominal_depth_m.to_numpy()[None]
    w = 1/(100**2+dh2+(100*dz)**2); w /= w.sum(1,keepdims=True)
    weights = np.zeros((n,len(names)))
    for g in range(n): weights[g] = (w[group==g]*m['volume_m3'][group==g,None]).sum(0)/volume[g]
    query=[]
    for name,row in meta.iterrows():
        nodes=np.flatnonzero(m['column']==int(audit.loc[name,'mesh_column']))
        query.append(group[nodes[np.abs(m['depth_m'][nodes]-row.nominal_depth_m).argmin()]])
    mesh = dict(volume=volume,horizontal=horizontal,vertical=vertical,surface_area=surface,
                open_fraction=boundary_v/volume)
    return mesh, weights, np.array(query), group, m


def boundary_context(times, group, mesh):
    with xr.open_dataset(ROOT/'reports/bay_twin/layered_mesh/boundary_context.nc') as d:
        ti = np.searchsorted(d.time.values, times.tz_localize(None).to_numpy(), side='right')-1
        assert (ti>=0).all()
        sources=pd.DatetimeIndex(d.source_time.values[ti],tz='UTC')
        assert (sources+pd.Timedelta(hours=48)<=times).all()
        b=d.thetao.values[ti];nodes=d.node.values
    result=np.full((len(times),12),np.nan)
    for g in range(12):
        selected=group[nodes]==g
        if not selected.any(): continue
        ww=mesh['volume_m3'][nodes[selected]][None]*np.isfinite(b[:,selected])
        total=ww.sum(1)
        result[:,g]=np.divide((np.nan_to_num(b[:,selected])*ww).sum(1),total,
                              out=np.full(len(times),np.nan),where=total>0)
    return result,sources
