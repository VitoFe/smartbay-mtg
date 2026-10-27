"""B. The SeapHOx at 16 m on Boa Tinetto, and what it says about the Copernicus cells the prior uses.

Tinetto (44.0198 N, 9.8502 E) lies on the single 44.021 N row of the Copernicus extraction, so
its 16 m record is an in-situ check of the product at the 16.27 m level, outside the bay.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, MG_TO_UMOL, ROOT, load

out = {}
s, meta = load('TBoaT_16m-SeapHOx.txt')
s['O2'] = s['Ossigeno(mg/l)'] * MG_TO_UMOL
s = s.rename(columns={'Temperatura(°C)': 'T', "Salinita'(PSU)": 'S', 'Ossigeno(%)': 'O2pct',
                      'pH_T_Esterno': 'pH_T_ext', 'pH_T_Interno': 'pH_T_int'})
o = s.O2.dropna()
out['seaphox'] = {
    'span': [str(o.index.min()), str(o.index.max())], 'hours_with_O2': int(len(o)),
    'hours_in_span': int((o.index.max() - o.index.min()) / pd.Timedelta('1h')),
    'O2_min': round(float(o.min()), 1), 'O2_p01_p50_p99': [round(float(x), 1) for x in o.quantile([.01, .5, .99])],
    'below_186': round(float((o < 186).mean()), 4), 'below_125': int((o < 125).sum()),
    'O2pct_p01_p50': [round(float(x), 1) for x in s.O2pct.dropna().quantile([.01, .5])],
    'months_with_data': int(o.resample('MS').count().gt(24 * 5).sum()),
    'monthly_min_O2': {str(k)[:7]: round(float(v), 1) for k, v in o.resample('MS').min().dropna().items()},
    'pH_ext_coverage_in_span': round(float(s.pH_T_ext.loc[o.index.min():o.index.max()].notna().mean()), 3)}

# Copernicus at the Tinetto cell, 16.27 m, daily, against the SeapHOx daily mean
bio = xr.open_dataset(glob.glob(str(ROOT / 'data/external/Copernicus/T2023_2026/*bgc-bio*'))[0])
tem = xr.open_dataset(glob.glob(str(ROOT / 'data/external/Copernicus/T2023_2026/*phy-tem*'))[0])
sal = xr.open_dataset(glob.glob(str(ROOT / 'data/external/Copernicus/T2023_2026/*phy-sal*'))[0])
lon = 9.8502
cell = dict(latitude=44.0208, longitude=lon, method='nearest')
out['copernicus_cell_lon'] = float(bio.longitude.sel(longitude=lon, method='nearest'))
cm = pd.DataFrame({
    'O2': bio.o2.sel(**cell).sel(depth=16.27, method='nearest').to_series(),
    'T': tem.thetao.sel(**cell).sel(depth=16.27, method='nearest').to_series(),
    'S': sal.so.sel(**cell).sel(depth=16.27, method='nearest').to_series()})
cm.index = pd.to_datetime(cm.index).tz_localize('UTC').floor('D')
obs = s[['O2', 'T', 'S']].resample('1D').mean()
obs = obs[s.O2.resample('1D').count() >= 18]
j = obs.join(cm, rsuffix='_cm', how='inner').dropna()
out['copernicus_vs_seaphox_16m'] = {'days': int(len(j)), 'period': [str(j.index.min())[:10], str(j.index.max())[:10]]}
for v in ('O2', 'T', 'S'):
    d = j[v + '_cm'] - j[v]
    out['copernicus_vs_seaphox_16m'][v] = {
        'bias_cm_minus_obs': round(float(d.mean()), 3), 'mae': round(float(d.abs().mean()), 3),
        'corr_daily': round(float(j[v].corr(j[v + '_cm'])), 3),
        'corr_anomaly_30d': round(float((j[v] - j[v].rolling(30, center=True, min_periods=10).mean()).corr(
            j[v + '_cm'] - j[v + '_cm'].rolling(30, center=True, min_periods=10).mean())), 3)}

# Surface (1 m) at Tinetto from the buoy casts, and the 1-16 m difference, vs Copernicus
c1, _ = load('TBoaT_01m-CTD+pH.txt', hourly=False)
c16, _ = load('TBoaT_16m-CTD+pH.txt', hourly=False)
pair = []
for t, r in c1.iterrows():
    near = c16.index[(abs(c16.index - t) < pd.Timedelta('2h'))]
    if len(near):
        r16 = c16.loc[near[0]]
        day = t.floor('D')
        pair.append({'t': t, 'T1': r['Temperatura(°C)'], 'T16': r16['Temperatura(°C)'],
                     'O1': r['Ossigeno(umol/l)'], 'O16': r16['Ossigeno(umol/l)'], 'S1': r["Salinita'(PSU)"],
                     'S16': r16["Salinita'(PSU)"], 'day': day})
pair = pd.DataFrame(pair)
pair['dT_obs'] = pair.T1 - pair.T16
pair['dO_obs'] = pair.O1 - pair.O16
prof_T = tem.thetao.sel(**cell)
prof_O = bio.o2.sel(**cell)
dT_cm, dO_cm = [], []
for day in pair.day:
    dd = np.datetime64(day.tz_localize(None))
    if prof_T.time.min() <= dd <= prof_T.time.max():
        pt = prof_T.sel(time=dd, method='nearest')
        po = prof_O.sel(time=dd, method='nearest')
        dT_cm.append(float(pt.sel(depth=1.02, method='nearest') - pt.sel(depth=16.27, method='nearest')))
        dO_cm.append(float(po.sel(depth=1.02, method='nearest') - po.sel(depth=16.27, method='nearest')))
    else:
        dT_cm.append(np.nan)
        dO_cm.append(np.nan)
pair['dT_cm'], pair['dO_cm'] = dT_cm, dO_cm
pp = pair.dropna(subset=['dT_cm', 'dT_obs'])
out['tinetto_casts_1m_minus_16m'] = {
    'pairs_total': int(len(pair)), 'pairs_in_copernicus_period': int(len(pp)),
    'obs_dT_median': round(float(pair.dT_obs.median()), 3), 'obs_dO_median': round(float(pair.dO_obs.median()), 2),
    'obs_min_O2_16m_cast': round(float(pair.O16.min()), 1),
    'cm_vs_obs_dT': {'corr': round(float(pp.dT_obs.corr(pp.dT_cm)), 3),
                     'mae': round(float((pp.dT_cm - pp.dT_obs).abs().mean()), 3),
                     'bias': round(float((pp.dT_cm - pp.dT_obs).mean()), 3)},
    'cm_vs_obs_dO': {'corr': round(float(pp.dO_obs.corr(pp.dO_cm)), 3) if pp.dO_obs.notna().sum() > 3 else None,
                     'mae': round(float((pp.dO_cm - pp.dO_obs).abs().mean()), 2)}}

# How the offshore 16 m water compares with the bay's surface sondes (same hours, 2024-07 to 2025-03)
bay, _ = load('TBaiaP_01m-CTD(calibrato).txt')
mit, _ = load('TMitiliE_05m-CTD(calibrato).txt')
for name, f in (('Baia_1m', bay), ('MitiliE_5m', mit)):
    jj = pd.DataFrame({'off': s.O2, 'bay': f['Ossigeno(umol/l)'], 'offS': s.S, 'bayS': f["Salinita'(PSU)"]}).dropna()
    if len(jj) > 100:
        out[f'offshore16m_vs_{name}'] = {'hours': int(len(jj)), 'O2_bay_minus_off_median': round(float((jj.bay - jj.off).median()), 1),
                                        'S_bay_minus_off_median': round(float((jj.bayS - jj.offS).median()), 3),
                                        'corr_daily_O2': round(float(jj.resample('1D').mean().dropna().pipe(lambda x: x.off.corr(x.bay))), 3)}
print(json.dumps(out, indent=1))
(OUT / 'eda_b_seaphox.json').write_text(json.dumps(out, indent=1))
