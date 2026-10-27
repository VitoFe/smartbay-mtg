"""E. The unused rain gauge at the Santa Teresa station, against the Magra level and bay salinity."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, ROOT, load, meteo

sys.path.insert(0, str(ROOT))
m = meteo()
rain = m['Precipitazione_Atmosferica(mm)']
out = {'rain': {'span': [str(rain.dropna().index.min())[:10], str(rain.dropna().index.max())[:10]],
                'coverage': round(float(rain.notna().mean()), 3),
                'annual_mm': {int(y.year): round(float(v), 0) for y, v in rain.resample('YS').sum().items()},
                'wet_hours_share': round(float((rain > 0).mean()), 3)}}
from src.data.bay_river import _read as river_read  # noqa: E402
try:
    lvl = river_read()
    if isinstance(lvl, pd.DataFrame):
        lvl = lvl.iloc[:, 0]
    lvl.index = pd.to_datetime(lvl.index, utc=True)
    lvl = lvl.resample('1h').mean()
    out['magra_span'] = [str(lvl.dropna().index.min())[:10], str(lvl.dropna().index.max())[:10]]
except Exception as e:  # the loader's signature is not essential here
    lvl = None
    out['river_error'] = repr(e)
daily_rain = rain.resample('1D').sum()
if lvl is not None:
    dl = lvl.resample('1D').mean()
    out['rain_vs_magra'] = {f'lag_{k}d': round(float(daily_rain.corr(dl.shift(-k))), 3) for k in range(0, 4)}
# salinity at the sondes: freshening after rain, and after a Magra rise, by lag
for name in ('TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)', 'TPortovenereS_02m-CTD(calibrato)'):
    f, _ = load(name + '.txt')
    s = f["Salinita'(PSU)"].resample('1D').mean()
    ds = s.diff()
    rec = {f'rain_lag_{k}d': round(float(daily_rain.shift(k).corr(ds)), 3) for k in (0, 1, 2, 3)}
    if lvl is not None:
        dlv = lvl.resample('1D').mean()
        rec.update({f'magra_lag_{k}d': round(float(dlv.shift(k).corr(s)), 3) for k in (0, 1, 3, 7)})
    # partial: does rain add to the Magra level in explaining the daily salinity change?
    if lvl is not None:
        X = pd.DataFrame({'ds': ds, 'r0': daily_rain, 'r1': daily_rain.shift(1),
                          'l1': lvl.resample('1D').mean().diff().shift(1)}).dropna()
        if len(X) > 60:
            from numpy.linalg import lstsq
            def r2(cols):
                A = np.column_stack([np.ones(len(X))] + [X[c] for c in cols])
                resid = X.ds - A @ lstsq(A, X.ds, rcond=None)[0]
                return 1 - resid.var() / X.ds.var()
            rec['R2_level_only'] = round(float(r2(['l1'])), 3)
            rec['R2_level_plus_rain'] = round(float(r2(['l1', 'r0', 'r1'])), 3)
            rec['days'] = int(len(X))
    out[name.split('-')[0]] = rec
print(json.dumps(out, indent=1, default=str))
(OUT / 'eda_e_rain.json').write_text(json.dumps(out, indent=1, default=str))
