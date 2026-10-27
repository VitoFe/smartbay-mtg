"""Is the 7-of-7 nocturnal onset a property of rapid declines, or of January-June 2026?

Uses src.experiments.v2_warning_episodes unchanged (series, crossings_of, cluster, EVENT_WIN) and
moves only START/END: first the reserved window (must reproduce 7 episodes, 7 inside 00-06 UTC),
then each half-year of the Gen-2 record and the whole of it. The reference is the module's own:
the share of days whose largest six-hour drop ends in 00-06 UTC on the same sensors and period.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.experiments.v2_warning_episodes as W
sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT

WINDOW = range(0, 7)


def read(start, end):
    W.START, W.END = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    frames = W.series()
    onsets, ends, rows = [], [], []
    for name, frame in frames.items():
        eps = W.cluster(W.crossings_of(frame.oxygen), 12)
        for ep in eps:
            t = ep[0]
            win = frame.loc[t - pd.Timedelta(hours=W.EVENT_WIN):t]
            rows.append({'sensor': name.split('_')[0], 'onset': str(t)[:16], 'hour': t.hour,
                         'dS_6h': round(float(win.salinity.iloc[-1] - win.salinity.iloc[0]), 2)
                         if 'salinity' in win and win.salinity.notna().sum() >= 2 else None,
                         'dT_6h': round(float(win.temperature.iloc[-1] - win.temperature.iloc[0]), 2)
                         if win.temperature.notna().sum() >= 2 else None})
            onsets.append(t.hour)
        drop = -(frame.oxygen - frame.oxygen.shift(W.EVENT_WIN)).dropna()
        daily = drop.groupby(drop.index.floor('D')).idxmax()
        ends += [t.hour for t in daily]
    share = float(np.isin(ends, WINDOW).mean()) if ends else float('nan')
    inside = int(np.isin(onsets, WINDOW).sum())
    n = len(onsets)
    return {'period': [start, end], 'sensors': sorted(frames), 'episodes': n, 'onsets_00_06': inside,
            'reference_share': round(share, 3),
            'p_greater': round(float(binomtest(inside, n, share, alternative='greater').pvalue), 4) if n else None,
            'rows': rows}


out = {'reserved_check': read('2026-01-01', '2026-06-20')}
for a, b in (('2024-07-24', '2025-01-01'), ('2025-01-01', '2025-07-01'), ('2025-07-01', '2026-01-01'),
             ('2024-07-24', '2026-01-01'), ('2024-07-24', '2026-06-20')):
    out[f'{a}..{b}'] = read(a, b)
for k, v in out.items():
    print(k, {x: v[x] for x in ('episodes', 'onsets_00_06', 'reference_share', 'p_greater')})
print(json.dumps(out['2024-07-24..2026-06-20']['rows'], indent=0)[:4000])
(OUT / 'eda_d2_nocturnal.json').write_text(json.dumps(out, indent=1, default=str))
