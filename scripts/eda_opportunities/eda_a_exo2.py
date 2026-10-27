"""A. The EXO2 channels the pipeline never reads: chlorophyll-a, turbidity, pH (NBS and total scale).

Coverage, plausibility, diurnal cycle, coupling with oxygen, and what they did during the seven
rapid-decline episodes of Warning-2026.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from eda_lib import OUT, ROOT, load, meteo

EXO2 = ['TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)', 'TMitiliO_05m-CTD(calibrato)',
        'TPortovenereS_02m-CTD(calibrato)', 'TPortovenereP_02m-CTD(calibrato)']
CH = {'O2': 'Ossigeno(umol/l)', 'chl': 'Clorofilla-a(ug/l)', 'turb': "Torbidita'(FNU)",
      'pH_NBS': 'pH_NBS', 'pH_T': 'pH_T', 'T': 'Temperatura(°C)', 'S': "Salinita'(PSU)",
      'O2pct': 'Ossigeno(%)'}
out = {}
frames = {}
for name in EXO2:
    f, meta = load(name + '.txt')
    f = f.rename(columns={v: k for k, v in CH.items()})
    frames[name] = f
    span = f.O2.dropna()
    f = f.loc[span.index.min():span.index.max()]
    rec = {'span': [str(span.index.min()), str(span.index.max())], 'hours': len(f)}
    for c in ('O2', 'chl', 'turb', 'pH_NBS', 'pH_T'):
        v = f[c]
        rec[c] = {'coverage': round(float(v.notna().mean()), 3),
                  'p1_p50_p99': [round(float(x), 3) for x in v.quantile([.01, .5, .99])] if v.notna().any() else None}
    # diurnal cycle: mean anomaly from the daily mean, by hour of day, as peak-to-trough amplitude
    for c in ('O2', 'pH_T', 'chl'):
        v = f[c].dropna()
        if len(v) > 500:
            anom = v - v.groupby(v.index.floor('D')).transform('mean')
            prof = anom.groupby(anom.index.hour).mean()
            rec[c]['diurnal_amp'] = round(float(prof.max() - prof.min()), 4)
            rec[c]['diurnal_peak_hour'] = int(prof.idxmax())
            rec[c]['diurnal_trough_hour'] = int(prof.idxmin())
    # coupling of 6 h changes: dO2 vs dpH_T, dO2 vs dchl
    d = f[['O2', 'pH_T', 'pH_NBS', 'chl', 'T', 'S']].diff(6)
    for c in ('pH_T', 'pH_NBS', 'chl', 'T', 'S'):
        ok = d[['O2', c]].dropna()
        if len(ok) > 200:
            rec[f'corr_dO2_d{c}_6h'] = round(float(ok.O2.corr(ok[c])), 3)
            if c == 'pH_T':
                slope = np.polyfit(ok.O2, ok[c], 1)[0]
                rec['dpH_per_10umol'] = round(float(slope * 10), 4)
    # level coupling: O2 % saturation vs pH (hourly)
    ok = f[['O2pct', 'pH_T']].dropna()
    if len(ok) > 200:
        rec['corr_O2pct_pH_T'] = round(float(ok.O2pct.corr(ok.pH_T)), 3)
    out[name] = rec

# The seven Warning-2026 episodes: what else moved in the 6 h that defined each one
wep = json.loads((ROOT / 'reports/bay_twin/runs/v2_warning_episodes.json').read_text())
m = meteo()
rain = m['Precipitazione_Atmosferica(mm)']
wind = m["Velocita'_Scalare_del_Vento(m/s)"]
episodes = []
for e in wep['episodes']:
    f = frames[e['sensor']]
    t1 = pd.Timestamp(e['onset'])
    t0 = t1 - pd.Timedelta(hours=6)
    row = {'sensor': e['sensor'].split('_')[0], 'onset': e['onset'][:16], 'json_drop': e['drop_umol_l']}
    for c in ('O2', 'pH_T', 'pH_NBS', 'chl', 'turb', 'T', 'S'):
        a, b = f[c].get(t0, np.nan), f[c].get(t1, np.nan)
        row['d' + c] = None if pd.isna(a) or pd.isna(b) else round(float(b - a), 3)
    row['rain_24h_mm'] = round(float(rain.loc[t1 - pd.Timedelta(hours=24):t1].sum()), 1)
    row['wind_6h_ms'] = round(float(wind.loc[t0:t1].mean()), 2)
    episodes.append(row)
out['episodes'] = episodes

# Reference for the episodes: the pH change expected from the O2 change, from the sonde's own
# 6 h relation, and the pH changes seen on nights with no decline.
for name in ('TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)'):
    f = frames[name].loc['2026-01-01':'2026-06-30']
    d = f[['O2', 'pH_T', 'S']].diff(6).dropna()
    night = d[d.index.hour.isin(range(0, 7))]
    out[name + '|reserved_night_6h'] = {
        'n': len(night), 'dO2_p05_p50': [round(float(x), 2) for x in night.O2.quantile([.05, .5])],
        'dpH_T_p05_p50': [round(float(x), 4) for x in night.pH_T.quantile([.05, .5])],
        'dS_p05_p50': [round(float(x), 3) for x in night.S.quantile([.05, .5])]}

print(json.dumps(out, indent=1, default=str))
(OUT / 'eda_a_exo2.json').write_text(json.dumps(out, indent=1, default=str))
