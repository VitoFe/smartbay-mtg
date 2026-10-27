"""Could the moored network see stratification? The ZPulse thermistors at 2 and 8 m against casts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eda_lib import OUT, ROOT, load

c = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz', parse_dates=['t_utc'])
c = c[c.qc_flag == 0].dropna(subset=['temp_c', 'depth_m']).copy()
c['t_utc'] = pd.to_datetime(c.t_utc, utc=True)
out = {'casts_by_station_gen2_era': c[c.t_utc >= '2024-07-24'].groupby('station').cast_id.nunique().to_dict(),
       'casts_by_station_2026': c[c.t_utc >= '2026-01-01'].groupby('station').cast_id.nunique().to_dict()}

# per cast: temperature near 2 m and near 8 m, surface-bottom T and DO difference
rec = []
for cid, g in c.groupby('cast_id'):
    g = g.sort_values('depth_m')
    near = lambda z: g.loc[(g.depth_m - z).abs().idxmin()] if (g.depth_m - z).abs().min() <= 0.75 else None
    r2, r8 = near(2.0), near(8.0)
    surf, bot = g[g.depth_m <= 2.0], g.tail(3)
    rec.append({'cast_id': cid, 'station': g.station.iloc[0], 't': g.t_utc.iloc[0],
                'T2': r2.temp_c if r2 is not None else np.nan, 'T8': r8.temp_c if r8 is not None else np.nan,
                'strat_T': surf.temp_c.mean() - bot.temp_c.mean() if len(surf) else np.nan,
                'delta_DO': bot.do_umol_l.mean() - surf.do_umol_l.mean() if len(surf) else np.nan,
                'bot_DO': bot.do_umol_l.mean(), 'bot_depth': bot.depth_m.mean()})
k = pd.DataFrame(rec)
k['dT_2_8_cast'] = k.T2 - k.T8
ok = k.dropna(subset=['strat_T', 'delta_DO'])
out['cast_strat_vs_bottom_deficit'] = {
    'casts': int(len(ok)), 'corr_strat_T_delta_DO': round(float(ok.strat_T.corr(ok.delta_DO)), 3),
    'corr_strat_T_bot_DO': round(float(ok.strat_T.corr(ok.bot_DO)), 3),
    'delta_DO_when_strat_gt_1C_median': round(float(ok[ok.strat_T > 1].delta_DO.median()), 2),
    'delta_DO_when_strat_le_0.2C_median': round(float(ok[ok.strat_T <= .2].delta_DO.median()), 2),
    'share_strat_gt_1C': round(float((ok.strat_T > 1).mean()), 3)}
# The ZPulse thermistors at the Mitili stations against the casts taken there
for site, stn, names in (('MitiliE', 'PMitiliE', ('TMitiliE_02m-Corrente', 'TMitiliE_08m-Corrente')),
                         ('MitiliO', 'PMitiliO', ('TMitiliO_02m-Corrente', 'TMitiliO_08m-Corrente'))):
    t2 = load(names[0] + '.txt')[0]['Temperatura(°C)']
    t8 = load(names[1] + '.txt')[0]['Temperatura(°C)']
    kk = k[k.station == stn].copy()
    kk['T2_moored'] = [t2.asof(t) if t >= t2.index.min() else np.nan for t in kk.t]
    kk['T8_moored'] = [t8.asof(t) if t >= t8.index.min() else np.nan for t in kk.t]
    # asof takes the last value even if old: keep only matches within 2 h
    for col, ser in (('T2_moored', t2), ('T8_moored', t8)):
        idx = [ser.dropna().index.asof(t) if t >= ser.dropna().index.min() else pd.NaT for t in kk.t]
        age = [(t - i) if i is not pd.NaT else pd.Timedelta('999D') for t, i in zip(kk.t, idx)]
        kk.loc[[a > pd.Timedelta('2h') for a in age], col] = np.nan
    m = kk.dropna(subset=['T8_moored', 'T8'])
    both = kk.dropna(subset=['T2_moored', 'T8_moored', 'strat_T'])
    out[f'zpulse_{site}'] = {
        'casts_at_station_with_T8_match': int(len(m)),
        'T8_moored_minus_cast_median': round(float((m.T8_moored - m.T8).median()), 3) if len(m) else None,
        'T8_abs_diff_median': round(float((m.T8_moored - m.T8).abs().median()), 3) if len(m) else None,
        'casts_with_both_thermistors': int(len(both)),
        'corr_moored_dT_vs_cast_strat': round(float((both.T2_moored - both.T8_moored).corr(both.strat_T)), 3) if len(both) > 4 else None,
        'corr_moored_dT_vs_cast_delta_DO': round(float((both.T2_moored - both.T8_moored).corr(both.delta_DO)), 3) if len(both) > 4 else None}
    # how often the moored column is stratified (2 m warmer than 8 m by > 0.5 C), by month
    d = (t2 - t8).dropna()
    out[f'zpulse_{site}']['hours_with_both'] = int(len(d))
    out[f'zpulse_{site}']['share_dT_gt_0.5_by_month'] = {int(mo): round(float((g > .5).mean()), 2) for mo, g in d.groupby(d.index.month)}
print(json.dumps(out, indent=1, default=str))
(OUT / 'eda_c2_strat.json').write_text(json.dumps(out, indent=1, default=str))
