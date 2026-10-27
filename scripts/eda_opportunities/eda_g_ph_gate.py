"""G. Could the co-located pH confirm an oxygen alert? Development years only (2024-07 to 2025-12).

Alerts and episodes come from src.experiments.v2_warning_episodes unchanged (the operating point
selected on development, read from runs/v2_warning_grid_ext.json: hourly drops above the selected rate
in umol/kg for the selected number of consecutive hours, then the selected cooldown; episodes: 6 h
drop > 30, crossings within 12 h merged). An alert is a hit if an episode's first crossing follows it within
24 h. For each alert, the pH_T change over the preceding 2 h and 6 h is read from the same sonde.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import src.experiments.v2_warning_episodes as W
from eda_lib import OUT, load

W.START, W.END = pd.Timestamp('2024-07-24', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC')
frames = W.series()
rows = []
for name, frame in frames.items():
    try:
        ph = load(name + '.txt')[0]['pH_T'].reindex(frame.index)
    except Exception:
        continue
    if ph.notna().mean() < .2:
        continue
    onsets = [ep[0] for ep in W.cluster(W.crossings_of(frame.oxygen), 12)]
    for a in W.alerts_of(frame.oxygen):
        hit = any(pd.Timedelta(0) <= (o - a) <= pd.Timedelta(hours=24) for o in onsets)
        rows.append({'sensor': name.split('_')[0], 'alert': a, 'hit': hit,
                     'dpH_2h': ph.get(a, np.nan) - ph.get(a - pd.Timedelta(hours=2), np.nan),
                     'dpH_6h': ph.get(a, np.nan) - ph.get(a - pd.Timedelta(hours=6), np.nan),
                     'dO2_2h': frame.oxygen.get(a, np.nan) - frame.oxygen.get(a - pd.Timedelta(hours=2), np.nan)})
r = pd.DataFrame(rows)
rate, consec, cooldown = W.operating_point()
out = {'operating_point': {'rate': rate, 'consecutive': consec, 'cooldown_h': cooldown,
                           'selected_in': 'runs/v2_warning_grid_ext.json'},
       'alerts': int(len(r)), 'hits': int(r.hit.sum()), 'alerts_with_pH': int(r.dpH_2h.notna().sum()),
       'by_sensor': r.groupby('sensor').hit.agg(['size', 'sum']).to_dict()}
k = r.dropna(subset=['dpH_2h'])
out['dpH_2h_median_hits'] = round(float(k[k.hit].dpH_2h.median()), 4) if k.hit.any() else None
out['dpH_2h_median_misses'] = round(float(k[~k.hit].dpH_2h.median()), 4)
out['dpH_2h_share_nonnegative_misses'] = round(float((k[~k.hit].dpH_2h >= 0).mean()), 3)
out['dpH_2h_share_nonnegative_hits'] = round(float((k[k.hit].dpH_2h >= 0).mean()), 3) if k.hit.any() else None
gate = []
for thr in (0.0, -0.005, -0.01, -0.02):
    kept = k[k.dpH_2h < thr]
    gate.append({'require_dpH_2h_below': thr, 'alerts_kept': int(len(kept)), 'hits_kept': int(kept.hit.sum()),
                 'precision': round(float(kept.hit.mean()), 3) if len(kept) else None})
out['gate'] = gate
out['ungated_precision_on_alerts_with_pH'] = round(float(k.hit.mean()), 3)
print(json.dumps(out, indent=1, default=str))
(OUT / 'eda_g_ph_gate.json').write_text(json.dumps(out, indent=1, default=str))
