"""Query the verified linear spatial field with explicit background fallback."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.bay_field_interface import BayField

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--longitude',required=True,type=float);p.add_argument('--latitude',required=True,type=float)
    p.add_argument('--depths',required=True);p.add_argument('--time',required=True);p.add_argument('--observations-through')
    p.add_argument('--missing-variable',action='append',default=[]);p.add_argument('--exclude-sensor',action='append',default=[])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    r=BayField().query(a.longitude,a.latitude,[float(v) for v in a.depths.split(',')],a.time,a.observations_through,a.missing_variable,a.exclude_sensor)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(r,indent=2,allow_nan=False))
