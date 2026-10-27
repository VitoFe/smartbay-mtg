"""Within-family physics contrasts for Table 5.8 (paired week-block bootstrap, 600; per-contrast seeded stream)."""
import json, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'reports/bay_twin/runs' / sys.argv[1] / 'results'
PAIRS = {'temperature': [('conservative_exchange', 'learned_source'), ('coast_exchange', 'conservative_exchange'),
                         ('coast_breakwater_exchange', 'coast_exchange'), ('coast_depth_exchange', 'coast_exchange')],
         'oxygen': [('conservative_exchange', 'learned_source'), ('saturation_source', 'learned_source'), ('saturation_exchange', 'conservative_exchange'),
                    ('coast_exchange', 'conservative_exchange'), ('coast_breakwater_exchange', 'coast_exchange'), ('coast_depth_exchange', 'coast_exchange')]}


def load(v, arm):
    d = pd.read_csv(R / f'v2_forecast_{v}_{arm}_nested/selected_predictions.csv.gz'); d = d[d['mode'] == 'forecast']
    return d.groupby(['fold', 'sensor', 'time'], as_index=False).agg(truth=('truth', 'first'), prediction=('prediction', 'mean'))


def macro(d, c):
    return d.assign(e=(d.truth - d[c]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean()


out = {}
for v, pairs in PAIRS.items():
    for a, b in pairs:
        j = load(v, a).merge(load(v, b), on=['fold', 'sensor', 'time'], suffixes=('_a', '_b'), validate='one_to_one').drop(columns='truth_b').rename(columns={'truth_a': 'truth'})
        j['week'] = pd.to_datetime(j.time, utc=True).dt.isocalendar().week.astype(int); g = {k: s for k, s in j.groupby(['fold', 'week'])}; keys = list(g)
        rng = np.random.default_rng([20260919, zlib.crc32(f'{v}|{a}|{b}'.encode())]); ds = []
        for _ in range(600):
            s = pd.concat([g[keys[i]] for i in rng.integers(0, len(keys), len(keys))]); ds.append(macro(s, 'prediction_a') - macro(s, 'prediction_b'))
        obs = macro(j, 'prediction_a') - macro(j, 'prediction_b'); lo, hi = np.percentile(ds, [2.5, 97.5]); out[f'{v}: {a} - {b}'] = [float(obs), float(lo), float(hi)]
        print(f'{v:12s} {a:26s} - {b:22s} {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))
(R.parent / 'v2_physics_contrasts.json').write_text(json.dumps(out, indent=2))
