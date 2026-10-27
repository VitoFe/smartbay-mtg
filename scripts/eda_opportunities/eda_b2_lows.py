"""Are the 2023 SeapHOx lows at 16 m real? Duration, coherence with pH, T, S, and saturation."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, MG_TO_UMOL, load

s, _ = load('TBoaT_16m-SeapHOx.txt')
s['O2'] = s['Ossigeno(mg/l)'] * MG_TO_UMOL
s = s.rename(columns={'Temperatura(°C)': 'T', "Salinita'(PSU)": 'S', 'Ossigeno(%)': 'sat', 'pH_T_Esterno': 'pH'})
low = s.O2 < 160
runs = (low != low.shift()).cumsum()
episodes = []
for _, g in s[low].groupby(runs[low]):
    episodes.append({'start': g.index.min(), 'hours': len(g), 'min_O2': round(g.O2.min(), 1),
                     'min_sat': round(g.sat.min(), 1), 'T': round(g['T'].mean(), 2), 'S': round(g.S.mean(), 2),
                     'pH': round(g.pH.mean(), 3) if g.pH.notna().any() else None})
e = pd.DataFrame(episodes)
print('hours below 160:', int(low.sum()), ' episodes:', len(e), ' of which >= 6 h:', int((e.hours >= 6).sum()))
print(e.sort_values('min_O2').head(15).to_string(index=False))
# context: a week around the absolute minimum
t = s.O2.idxmin()
w = s.loc[t - pd.Timedelta('3D'): t + pd.Timedelta('3D'), ['O2', 'sat', 'T', 'S', 'pH']]
print('\naround the minimum', t)
print(w.resample('6h').mean().round(2).to_string())
# hourly jumps: is the record spiky?
d = s.O2.diff().abs()
print('\nabs hourly O2 change p50/p99/max:', round(d.median(), 2), round(d.quantile(.99), 2), round(d.max(), 1))
# seasonal: monthly median O2 and T, S at 16 m, and the autumn 2023 compared with autumn 2022
m = s[['O2', 'T', 'S', 'sat']].resample('MS').median()
print('\nmonthly medians'); print(m.loc['2022-06':'2023-12'].round(2).to_string())
import json  # noqa: E402
lowest = e.sort_values('min_O2').iloc[0]
baseline = s.pH.loc[t - pd.Timedelta('3D'):t - pd.Timedelta('1D')].median()
(OUT / 'eda_b2_lows.json').write_text(json.dumps({
    'hours_below_160': int(low.sum()), 'runs_below_160': int(len(e)), 'runs_6h_or_longer': int((e.hours >= 6).sum()),
    'minimum': {'time': str(t), 'O2': round(float(s.O2.min()), 1), 'saturation_pct': float(lowest.min_sat),
                'pH_T_ext_during': float(lowest.pH), 'pH_T_ext_median_previous_2_days': round(float(baseline), 3)},
    'runs': [{k: (str(v) if k == 'start' else v) for k, v in r.items()} for r in e.to_dict('records')]}, indent=1))
