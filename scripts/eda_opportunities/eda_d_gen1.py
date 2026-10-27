"""D. The Gen-1 oxygen record (2021-2023), used only for what a calibration offset cannot fake:
the timing of rapid declines, and whether levels ever approached a biological threshold.

Event rule as in Warning-2026: a drop of more than 30 umol/l over 6 h; crossings within 12 h form
one episode; the onset is the first crossing hour. Reference: the share of days whose largest 6 h
drop ends in 00-06 UTC, on the same record.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, ROOT, load

out = {}


def events(o2, drop=30.0, gap_h=12):
    d6 = o2 - o2.shift(6)
    cross = d6[d6 < -drop].index
    eps, last = [], None
    for t in cross:
        if last is None or (t - last) > pd.Timedelta(hours=gap_h):
            eps.append(t)
        last = t
    return eps, d6


def daily_largest_share(d6):
    d = d6.dropna()
    day_min = d.groupby(d.index.floor('D')).idxmin()
    hours = pd.DatetimeIndex(day_min.values).hour
    return float(np.isin(hours, range(0, 7)).mean()), len(day_min)


sources = {'Gen1_Baia_1m': 'TBaia_01m-Ossigeno(calibrato).txt',
           'Gen1_Mitili_5.5m': 'TMitiliE_05m-Ossigeno(calibrato).txt',
           'Gen2_Baia_1m': 'TBaiaP_01m-CTD(calibrato).txt',
           'Gen2_MitiliE_5m': 'TMitiliE_05m-CTD(calibrato).txt'}
sal_gen1, _ = load("TBaia_01m-Conducibilita'(calibrato).txt")
sal_gen1 = sal_gen1["Salinita'(PSU)"]
for key, name in sources.items():
    f, meta = load(name)
    o2 = f['Ossigeno(umol/l)']
    # the reserved warning cohort is 2026; the Gen-2 rows here cover the whole record for scale
    eps, d6 = events(o2)
    share, days = daily_largest_share(d6)
    onset_hours = [t.hour for t in eps]
    inside = int(sum(h in range(0, 7) for h in onset_hours))
    rec = {'span': [str(o2.dropna().index.min())[:10], str(o2.dropna().index.max())[:10]],
           'hours': int(o2.notna().sum()), 'O2_min': round(float(o2.min()), 1),
           'O2_p01': round(float(o2.quantile(.01)), 1), 'hours_below_186': int((o2 < 186).sum()),
           'hours_below_125': int((o2 < 125).sum()), 'episodes': len(eps), 'onsets_00_06': inside,
           'daily_largest_drop_share_00_06': round(share, 3), 'days': days,
           'binomial_p_vs_daily_largest': round(float(binomtest(inside, len(eps), share, alternative='greater').pvalue), 4) if eps else None,
           'episodes_by_month': pd.Series([t.month for t in eps]).value_counts().sort_index().to_dict() if eps else {}}
    if key == 'Gen1_Baia_1m' and eps:
        ds = [float(sal_gen1.get(t, np.nan) - sal_gen1.get(t - pd.Timedelta(hours=6), np.nan)) for t in eps]
        rec['episode_dS_6h_median'] = round(float(np.nanmedian(ds)), 3)
        rec['episode_dS_6h_values'] = [round(x, 2) for x in ds if not np.isnan(x)][:40]
    out[key] = rec
print(json.dumps(out, indent=1, default=str))
(OUT / 'eda_d_gen1.json').write_text(json.dumps(out, indent=1, default=str))
