"""Check numerical MaGIC grid coverage without extrapolating offshore depths inland."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import CRS,Transformer

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'data/external/MaGIC_LaSpezia'; output=ROOT/'reports/bay_twin'
    crs=CRS.from_wkt((source/'04-isobate-10m.prj').read_text())
    project=Transformer.from_crs(4326,crs,always_xy=True)
    sensors=pd.read_csv(output/'sensor_geometry.csv')
    sx,sy=project.transform(sensors.longitude.to_numpy(),sensors.latitude.to_numpy()); xy=np.column_stack([sx,sy])
    report={'projection_file':str((source/'04-isobate-10m.prj').relative_to(ROOT)),
            'projection_interpretation':'WGS84 UTM zone 32N from accompanying contours .prj; numerical grid x/y magnitudes and spacing agree. RNDT CRS text alone is ambiguous.',
            'grids':[]}; rows=[]
    for file in sorted(source.glob('04-*m.txt')):
        grid=np.loadtxt(file,delimiter=','); spacing=int(file.stem.split('-')[1].rstrip('m'))
        distance,index=cKDTree(grid[:,:2]).query(xy)
        # Distance gate avoids falsely treating a convex hull across missing coast as coverage.
        covered=distance<=spacing*np.sqrt(2)
        report['grids'].append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'points':len(grid),'nominal_spacing_m':spacing,'x_range_m':[float(grid[:,0].min()),float(grid[:,0].max())],
            'y_range_m':[float(grid[:,1].min()),float(grid[:,1].max())],
            'z_range_m':[float(grid[:,2].min()),float(grid[:,2].max())],
            'sensors_with_local_grid_support':int(covered.sum()),
            'nearest_sensor_distance_range_m':[float(distance.min()),float(distance.max())]})
        for i,s in sensors.iterrows():
            rows.append({'sensor':s.sensor,'grid':file.name,'nearest_point_distance_m':distance[i],
                         'locally_supported':bool(covered[i]),'nearest_offshore_z_m':grid[index[i],2],
                         'assigned_sensor_depth_m':float(-grid[index[i],2]) if covered[i] else None})
    report['conclusion']='Check local support before interpolation. Unsupported sensors retain regional bands and deployment metadata; nearest offshore depths are not assigned.'
    (output/'magic_coverage.json').write_text(json.dumps(report,indent=2)); pd.DataFrame(rows).to_csv(output/'magic_sensor_coverage.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
