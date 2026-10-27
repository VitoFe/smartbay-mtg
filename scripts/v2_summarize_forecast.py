"""v2 Table 5.8: nested-selected arms on identical endpoints; seed-mean MAE; paired week-block bootstrap vs the residual tree."""
import json, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RS = [ROOT / 'reports/bay_twin/runs' / r / 'results' for r in sys.argv[1:]]; R = RS[0]
RNG = np.random.default_rng(20260919)

def macro(d, col='prediction'):
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())

out = {}
for v in ['temperature', 'oxygen']:
    arms = {}
    for f in sorted(p for r in RS for p in r.glob(f'v2_forecast_{v}_*_nested/selected_predictions.csv.gz')):
        arm = f.parent.name.removeprefix(f'v2_forecast_{v}_').removesuffix('_nested')
        d = pd.read_csv(f); d = d[d['mode'] == 'forecast'] if 'mode' in d else d
        if 'seed' in d: d = d.groupby(['fold', 'sensor', 'time'], as_index=False).agg(truth=('truth', 'first'), persistence=('persistence', 'first'), prediction=('prediction', 'mean'))
        arms[arm] = d[['fold', 'sensor', 'time', 'truth', 'persistence', 'prediction']]
    ref = arms['tree']; key = ['fold', 'sensor', 'time']; res = {}
    ref = ref.assign(week=pd.to_datetime(ref.time, utc=True).dt.isocalendar().week.astype(int))
    blocks = ref[['fold', 'week']].drop_duplicates().values.tolist()
    for arm, d in arms.items():
        j = ref.merge(d[key + ['prediction']], on=key, suffixes=('_tree', ''), validate='one_to_one'); assert len(j) == len(ref) == len(d), arm
        obs = macro(j) - macro(j, 'prediction_tree'); draws = []
        if arm != 'tree':
            RNG = np.random.default_rng([20260919, zlib.crc32(f'{v}|{arm}'.encode())])  # per-arm stream: adding arms never shifts another arm's interval
            g = {k: s for k, s in j.groupby(['fold', 'week'])}
            for _ in range(600):
                s = pd.concat([g[tuple(blocks[i])] for i in RNG.integers(0, len(blocks), len(blocks))])
                draws.append(macro(s) - macro(s, 'prediction_tree'))
        res[arm] = {'mae': macro(j), 'persistence': macro(j, 'persistence'), 'endpoints': len(j),
                    'minus_tree': [obs, *(np.percentile(draws, [2.5, 97.5]).tolist() if draws else [0., 0.])],
                    'trials': json.loads(next(r / f'v2_forecast_{v}_{arm}_nested/report.json' for r in RS if (r / f'v2_forecast_{v}_{arm}_nested').exists()).read_text())['selection']['completed_trials']}
        print(f"{v:12s} {arm:28s} {res[arm]['mae']:.4f}  vs tree {obs:+.4f} [{res[arm]['minus_tree'][1]:+.4f}, {res[arm]['minus_tree'][2]:+.4f}]  pers {res[arm]['persistence']:.4f} n={len(j)} trials={res[arm]['trials']}")
    out[v] = res
(R.parent / 'v2_forecast_summary.json').write_text(json.dumps(out, indent=2))
