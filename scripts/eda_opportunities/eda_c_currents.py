"""C. The ZPulse current meters: what drives the flow, and does flow explain the oxygen drops at 5 m?"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, load, meteo

out = {}
cur = {}
for name in ('TMitiliE_02m-Corrente', 'TMitiliE_08m-Corrente', 'TMitiliO_02m-Corrente',
             'TMitiliO_08m-Corrente', 'TPortovenereP_01m-Corrente'):
    f, meta = load(name + '.txt')
    f = f.rename(columns={"Velocita'(m/s)": 'spd', "Velocita'_Nord(m/s)": 'v', "Velocita'_Est(m/s)": 'u',
                          'Direzione(°)': 'dir', 'Temperatura(°C)': 'T', 'Ampiezza_Eco(dB)': 'echo'})
    span = f.spd.dropna()
    f = f.loc[span.index.min():span.index.max()]
    cur[name] = f
    uv = f[['u', 'v']].dropna()
    # principal axis of the hourly velocity
    cov = np.cov(uv.u, uv.v)
    w, vec = np.linalg.eigh(cov)
    major = vec[:, 1]
    axis_deg = (np.degrees(np.arctan2(major[0], major[1])) + 360) % 180  # compass bearing of the axis
    # variance by band, from the along-axis component
    a = (uv.u * major[0] + uv.v * major[1]).resample('1h').mean().interpolate(limit=3).dropna()
    a = a - a.mean()
    spec = np.abs(np.fft.rfft(a.to_numpy())) ** 2
    freq = np.fft.rfftfreq(len(a), d=1.0)  # cycles per hour
    per = np.divide(1, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    tot = spec[1:].sum()
    band = lambda lo, hi: float(spec[(per >= lo) & (per < hi)].sum() / tot)
    out[name] = {'span': [str(span.index.min())[:10], str(span.index.max())[:10]],
                 'coverage': round(float(f.spd.notna().mean()), 3),
                 'speed_p50_p90_p99_cm_s': [round(100 * float(x), 2) for x in f.spd.quantile([.5, .9, .99])],
                 'mean_flow_cm_s': [round(100 * float(uv.u.mean()), 2), round(100 * float(uv.v.mean()), 2)],
                 'principal_axis_bearing': round(float(axis_deg), 1),
                 'major_axis_variance_share': round(float(w[1] / w.sum()), 3),
                 'variance_share': {'semidiurnal_11-13.5h': round(band(11, 13.5), 3),
                                    'inertial_16-18.5h': round(band(16, 18.5), 3),
                                    'diurnal_22-26h': round(band(22, 26), 3),
                                    'synoptic_2-10d': round(band(48, 240), 3),
                                    'longer_10d+': round(band(240, np.inf), 3)}}

# Wind coupling: 6 h mean wind vector against the current, at the lag of best correlation
m = meteo()
ws = m["Velocita'_Scalare_del_Vento(m/s)"]
wd = m['Direzione_del_Vento(°)']
# meteorological convention: direction the wind blows FROM
wu, wv = -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))
wind = pd.DataFrame({'wu': wu, 'wv': wv, 'ws': ws}).rolling(6, min_periods=4).mean()
for name, f in cur.items():
    best = None
    for lag in (0, 3, 6, 12, 24):
        j = pd.concat([f[['u', 'v', 'spd']], wind.shift(lag)], axis=1).dropna()
        if len(j) < 500:
            continue
        r = max(abs(j.u.corr(j.wu)), abs(j.v.corr(j.wv)), abs(j.spd.corr(j.ws)))
        if best is None or r > best[1]:
            best = (lag, r, round(float(j.spd.corr(j.ws)), 3), round(float(j.u.corr(j.wu)), 3), round(float(j.v.corr(j.wv)), 3))
    out[name]['wind_best'] = {'lag_h': best[0], 'max_abs_r': round(best[1], 3), 'r_speed': best[2],
                              'r_u': best[3], 'r_v': best[4]} if best else None

# Shear 2 m vs 8 m at Mitili Est, and the relation to the 5 m sonde's oxygen and salinity changes
top, bot = cur['TMitiliE_02m-Corrente'], cur['TMitiliE_08m-Corrente']
exo, _ = load('TMitiliE_05m-CTD(calibrato).txt')
exo = exo.rename(columns={'Ossigeno(umol/l)': 'O2', "Salinita'(PSU)": 'S', 'Temperatura(°C)': 'T'})
j = pd.DataFrame({'u2': top.u, 'v2': top.v, 'u8': bot.u, 'v8': bot.v, 'T2': top['T'], 'T8': bot['T'],
                  'O2': exo.O2, 'S': exo.S, 'T5': exo['T']})
j['shear'] = np.hypot(j.u2 - j.u8, j.v2 - j.v8)
j['dT_2_8'] = j.T2 - j.T8
j['dO2_6h'] = j.O2.diff(6)
j['dS_6h'] = j.S.diff(6)
jj = j.dropna(subset=['shear', 'dO2_6h', 'dS_6h'])
out['mitiliE_shear_vs_5m'] = {
    'hours': int(len(jj)),
    'corr_shear_dO2_6h': round(float(jj.shear.corr(jj.dO2_6h)), 3),
    'corr_dT2_8_dO2_6h': round(float(jj.dT_2_8.corr(jj.dO2_6h)), 3),
    'corr_v8_dS_6h': round(float(jj.v8.corr(jj.dS_6h)), 3), 'corr_u8_dS_6h': round(float(jj.u8.corr(jj.dS_6h)), 3),
    'corr_v8_dO2_6h': round(float(jj.v8.corr(jj.dO2_6h)), 3), 'corr_u8_dO2_6h': round(float(jj.u8.corr(jj.dO2_6h)), 3)}
# Salinity-rise events at 5 m (dS over 6 h above +0.8): what do O2, the 2-8 m thermal difference and flow do?
ev = jj[jj.dS_6h > 0.8]
out['mitiliE_salinity_rise_events'] = {
    'hours_with_dS6h_gt_0.8': int(len(ev)), 'share_of_hours': round(len(ev) / len(jj), 4),
    'dO2_6h_median_in_events': round(float(ev.dO2_6h.median()), 1) if len(ev) else None,
    'dO2_6h_median_otherwise': round(float(jj[jj.dS_6h <= 0.8].dO2_6h.median()), 1),
    'dT2_8_median_in_events': round(float(ev.dT_2_8.median()), 3) if len(ev) else None,
    'dT2_8_median_otherwise': round(float(jj[jj.dS_6h <= 0.8].dT_2_8.median()), 3),
    'v8_median_in_events_cm_s': round(100 * float(ev.v8.median()), 2) if len(ev) else None,
    'v8_median_otherwise_cm_s': round(100 * float(jj[jj.dS_6h <= 0.8].v8.median()), 2),
    'hour_of_day_share_00_06': round(float(ev.index.hour.isin(range(0, 7)).mean()), 3) if len(ev) else None,
    'months': ev.index.month.value_counts().sort_index().to_dict() if len(ev) else None}
print(json.dumps(out, indent=1, default=str))
(OUT / 'eda_c_currents.json').write_text(json.dumps(out, indent=1, default=str))
