"""Second 2026 reading: v2 vs v1 vs baselines, with the week-block bootstrap (600, seed 20260919)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
RNG = np.random.default_rng(20260919)
b = pd.read_csv(ROOT / 'reports/bay_twin/runs/battery/baseline_endpoints_2026.csv.gz')
e1 = pd.read_csv(ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v1/forecast_endpoints_2026.csv.gz')
e2 = pd.read_csv(ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz')
e1 = e1[e1.has_anchor & e1.has_target]; e2 = e2[e2.has_anchor & e2.has_target]
k = e1[['variable', 'sensor', 'time', 'prediction', 'truth']].merge(e2[['variable', 'sensor', 'time', 'prediction']].rename(columns={'prediction': 'MODEL_v2'}),
                                                                   on=['variable', 'sensor', 'time'], validate='one_to_one')
assert len(k) == len(e1) == len(e2) == len(b)
k['time'] = pd.to_datetime(k.time, utc=True); k['quarter'] = np.where(k.time.dt.month <= 3, 'Q1', 'Q2'); k['week'] = k.time.dt.isocalendar().week.astype(int)
b['_key'] = b.MODEL.round(9); k['_key'] = k.prediction.round(9); b['_t'] = b.truth.round(6); k['_t'] = k.truth.round(6)
keys = ['variable', 'sensor', 'quarter', 'week', '_t', '_key']  # rows tied on all keys are interchangeable for every metric used
b['_n'] = b.groupby(keys).cumcount(); k = k.sort_values('time'); k['_n'] = k.groupby(keys).cumcount()
m = b.reset_index().merge(k.drop(columns=['quarter']).assign(quarter=k.quarter), on=keys + ['_n'], validate='one_to_one', suffixes=('', '_ep'))
assert len(m) == len(b) and np.allclose(m.truth, m.truth_ep), 'alignment failed'
b = m.sort_values('index').drop(columns=['index', '_key', '_t', '_n', 'truth_ep', 'prediction', 'time']).rename(columns={'MODEL': 'MODEL_v1'}).reset_index(drop=True)

def macro(d, col):
    dd = d.dropna(subset=[col, 'truth'])
    return float(dd.assign(e=(dd[col] - dd.truth).abs()).groupby(['quarter', 'sensor']).e.mean().groupby('quarter').mean().mean())

out = {}
for var in ['temperature', 'oxygen']:
    d = b[b.variable == var]; blocks = d[['quarter', 'week']].drop_duplicates().values.tolist()
    pairs = [('MODEL_v2', 'B1_clim_increment'), ('MODEL_v1', 'B1_clim_increment'), ('MODEL_v2', 'MODEL_v1'), ('MODEL_v2', 'B0_persistence')]
    draws = {p: [] for p in pairs}
    for _ in range(600):
        pick = [blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]
        s = pd.concat([d[(d.quarter == q) & (d.week == w)] for q, w in pick])
        for a, c in pairs: draws[(a, c)].append(macro(s, a) - macro(s, c))
    out[var] = {'n': len(d), 'mae': {m: macro(d, m) for m in ['B0_persistence', 'B1_clim_increment', 'MODEL_v1', 'MODEL_v2']},
                'contrasts': {f'{a} - {c}': [macro(d, a) - macro(d, c), *np.percentile(draws[(a, c)], [2.5, 97.5]).tolist()] for a, c in pairs}}
    print(var, json.dumps(out[var], indent=1))
(ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/battery_v2.json').write_text(json.dumps(out, indent=2))
