"""Reserved-2026 comparison battery against the v2 supported model (Tables 5.4 and 5.5).

Parameter-free comparators (persistence, climatological increment, zero-shot Chronos) come from the
saved predictions in runs/battery/results/; the supported model from the second reading; tuned Chronos arms
(fine-tuned, residual corrector) from the v2 Chronos study when available. All joined on
(variable, sensor, issuance time); week-block bootstrap within quarter, 600 resamples, seed 20260919.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; T = ROOT / 'reports/bay_twin/runs/battery/results'
RNG = np.random.default_rng(20260919); KEY = ['variable', 'sensor', 'time']

f = pd.read_csv(T / 'forecast/forecast_predictions.csv.gz')[KEY + ['quarter', 'week', 'truth', 'MODEL', 'B0_persistence', 'B1_clim_increment', 'CHRONOS_bolt_small', 'CHRONOS_bolt_base']]
c = pd.read_csv(T / 'chronos/chronos_predictions.csv.gz')
zs = [k for k in c.columns if '|ZS_raw' in k and not k.endswith('_C2') or '|ZS_aou' in k and not k.endswith('_C2')]
d = f.merge(c[KEY + zs], on=KEY, validate='one_to_one')
e2 = pd.read_csv(ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'); e2 = e2[e2.has_anchor & e2.has_target]
d = d.merge(e2[KEY + ['prediction']].rename(columns={'prediction': 'MODEL_v2'}), on=KEY, validate='one_to_one')
for v2c in sorted((ROOT / 'reports/bay_twin/runs/battery/v2_chronos').glob('v2_chronos*_predictions*.csv.gz')):
    x = pd.read_csv(v2c); cols = [k for k in x.columns if k.startswith(('V2FT_', 'V2C2_', 'CH2_')) and k not in d.columns]
    if cols: d = d.merge(x[KEY + cols], on=KEY, validate='one_to_one')
assert len(d) == 4250, len(d)
d = d.rename(columns={'MODEL': 'MODEL_v1'})


def macro(s, col):
    s = s.dropna(subset=[col]); return float(s.assign(e=(s[col] - s.truth).abs()).groupby(['quarter', 'sensor']).e.mean().groupby('quarter').mean().mean())


out = {}
for var in ['temperature', 'oxygen']:
    s0 = d[d.variable == var].copy(); methods = [m for m in s0.columns if m not in KEY + ['quarter', 'week', 'truth', 'MODEL_v2'] and s0[m].notna().all()]
    blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist(); groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}
    draws = {m: [] for m in methods}
    pairs = [(x, y) for x, y in [('CH2_temperature_raw_cov', 'CH2_temperature_raw_uni'), ('CH2_oxygen_raw_cov', 'CH2_oxygen_raw_uni'),
             ('CH2_oxygen_aou_cov', 'CH2_oxygen_aou_uni'), ('CH2_oxygen_aou_uni', 'CH2_oxygen_raw_uni'), ('CH2_temperature_raw_uni', 'CHRONOS_bolt_base'),
             ('CH2_oxygen_raw_uni', 'CHRONOS_bolt_base'), ('oxygen|ZS_aou', 'oxygen|ZS_raw'), ('V2FT_temperature_raw', 'temperature|ZS_raw'),
             ('V2FT_oxygen_aou', 'oxygen|ZS_aou'), ('V2FT_temperature_raw', 'CH2_temperature_raw_cov'),
             # addendum-2 study 3: every arm was only ever contrasted with the supported model, so the
             # best arms were never tested against the fair baseline. Re-analysis of already-scored
             # predictions; the resample is drawn once per draw and shared, so published intervals do not move.
             ('oxygen|ZS_aou', 'B1_clim_increment'), ('oxygen|ZS_raw', 'B1_clim_increment'),
             ('CH2_oxygen_aou_cov', 'B1_clim_increment'), ('temperature|ZS_raw', 'B1_clim_increment'),
             ('CH2_temperature_raw_cov', 'B1_clim_increment'), ('MODEL_v1', 'B1_clim_increment'),
             ('B1_clim_increment', 'B0_persistence')] if x in methods and y in methods]
    pdraws = {pr: [] for pr in pairs}
    for _ in range(600):
        s = pd.concat([groups[tuple(blocks[i])] for i in RNG.integers(0, len(blocks), len(blocks))]); base = macro(s, 'MODEL_v2')
        for m in methods: draws[m].append(macro(s, m) - base)
        for x, y in pairs: pdraws[(x, y)].append(macro(s, x) - macro(s, y))
    out[var] = {'n': len(s0), 'MODEL_v2': macro(s0, 'MODEL_v2'),
                'methods': {m: {'mae': macro(s0, m), 'minus_v2': macro(s0, m) - macro(s0, 'MODEL_v2'), 'ci': np.percentile(draws[m], [2.5, 97.5]).tolist()} for m in methods}}
    out[var]['pairs'] = {f'{x} - {y}': [macro(s0, x) - macro(s0, y), *np.percentile(v, [2.5, 97.5]).tolist()] for (x, y), v in pdraws.items()}
    print(f"== {var}: n={len(s0)}  MODEL_v2 {out[var]['MODEL_v2']:.4f}")
    for k, (dd, lo, hi) in out[var]['pairs'].items(): print(f'   PAIR {k:50s} {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))
    for m, r in out[var]['methods'].items():
        print(f"   {m:24s} {r['mae']:.4f}  minus v2 {r['minus_v2']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" + ('  *' if r['ci'][0] > 0 or r['ci'][1] < 0 else ''))
(ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/battery_full_v2.json').write_text(json.dumps(out, indent=2))
