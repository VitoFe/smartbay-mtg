"""Reproducible descriptive checks; flags are not automatic exclusions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
#: the thesis prints the heatmap at \textwidth (416.7 pt) and at most 224 pt high; drawn at that size,
#: every label prints at its own size, 7 pt (the caption carries the title)
WIDTH_IN, HEIGHT_IN, FONT_PT = 416.7/72, 3.1, 7

def availability_figure(data, dest):
    """Share of each month's hours with a temperature reading, probe by probe (Figure 3.2)."""
    availability=data[[c for c in data if c.endswith('::temperature')]].notna().resample('MS').mean().T
    names=[s.replace('::temperature','').replace('-CTD(calibrato)',' EXO2') for s in availability.index]
    months=list(availability.columns)
    fig,ax=plt.subplots(figsize=(WIDTH_IN,HEIGHT_IN),layout='constrained')
    im=ax.imshow(availability.values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(names)),names,fontsize=FONT_PT)
    # a label every third month, January, April, July and October, each month a minor tick
    major=[i for i,d in enumerate(months) if d.month in (1,4,7,10)]
    ax.set_xticks(major,[months[i].strftime('%b %Y') for i in major],fontsize=FONT_PT)
    ax.set_xticks(range(len(months)),minor=True)
    ax.tick_params(axis='both',length=2,pad=2)
    ax.set_xlabel('Month (UTC)',fontsize=FONT_PT)
    bar=fig.colorbar(im,ax=ax,pad=0.015,fraction=0.04)
    bar.set_label('Observed fraction of hours\n(first and last months partial)',fontsize=FONT_PT)
    bar.ax.tick_params(labelsize=FONT_PT,length=2)
    fig.savefig(dest/'temperature_availability.png',dpi=300); plt.close(fig)

def main():
    dest=ROOT/'reports/bay_twin'
    data=pd.read_csv(dest/'hourly_observations.csv.gz',index_col=0,parse_dates=True)
    meta=pd.read_csv(dest/'sensor_geometry.csv')
    quality=[]
    for name in data:
        valid=data[name].dropna(); threshold=3 if name.endswith('temperature') else 80
        quality.append({'channel':name,'n':len(valid),'min':float(valid.min()),'median':float(valid.median()),
                        'max':float(valid.max()),'adjacent_jump_threshold':threshold,
                        'adjacent_hour_jump_count':int((data[name].diff().abs()>threshold).sum())})
    (dest/'quality_diagnostics.json').write_text(json.dumps(quality,indent=2))
    pairs=[]
    for i,a in meta.iterrows():
        for j,b in meta.iloc[i+1:].iterrows():
            distance=float(np.hypot(a.x_m-b.x_m,a.y_m-b.y_m))
            if distance<60:
                pair=data[[a.sensor+'::temperature',b.sensor+'::temperature']].dropna()
                pairs.append({'sensor_a':a.sensor,'sensor_b':b.sensor,'distance_m':distance,
                              'overlap_hours':len(pair),'mae_between_probes_c':float((pair.iloc[:,0]-pair.iloc[:,1]).abs().mean()) if len(pair) else None})
    (dest/'colocated_probe_diagnostics.json').write_text(json.dumps(pairs,indent=2))
    external=[]
    for path in (ROOT/'data/external/Copernicus').rglob('*thetao*.nc'):
        with xr.open_dataset(path) as ds:
            external.append({'path':path.relative_to(ROOT).as_posix(),'sizes':dict(ds.sizes),'variables':list(ds.data_vars),
                'coordinates':{k:{'min':str(ds[k].values.min()),'max':str(ds[k].values.max())} for k in ds.coords}})
    (dest/'copernicus_local_inventory.json').write_text(json.dumps(external,indent=2))
    availability_figure(data, dest)
    print('Saved descriptive QC, colocation, CMEMS and availability diagnostics.')

if __name__=='__main__': main()
