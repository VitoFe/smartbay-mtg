"""Addendum 42: score the time-held-out selections against zero-shot and the random-split selections.

Joins the comparison battery as scripts/v2_battery.py does (through v2_chronos_lora_contrasts), adds
the adapter selection of addendum 40 and the two selections of this addendum, and draws the same 600
week-block resamples within quarter (seed 20260919). Gate G3: the full-fine-tuning and zero-shot
pairs and intervals reproduce battery_full_v2.json exactly. Writes runs/v2_chronos_timeval/contrasts.json.

Usage: python -m scripts.v2_chronos_timeval_contrasts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_chronos_lora_contrasts import battery_frame, macro

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
RUN = RUNS / 'v2_chronos_timeval'
BATTERY = RUNS / 'v2_reserved_reading_v2/battery_full_v2.json'
SPACE = {'temperature': 'raw', 'oxygen': 'aou'}
ZS = {'temperature': 'temperature|ZS_raw', 'oxygen': 'oxygen|ZS_aou'}
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}


def load(path, columns):
    frame = pd.read_csv(path)
    frame['time_utc'] = pd.to_datetime(frame.time, utc=True)
    return frame[['variable', 'sensor', 'time_utc'] + columns]


def main():
    d = battery_frame()
    d['time_utc'] = pd.to_datetime(d.time, utc=True)
    lora_cols = [f'V2LORA_{v}_{s}' for v, s in SPACE.items()]
    tv_cols = [f'V2TV_{m}_{v}_{s}' for m in ('full', 'lora') for v, s in SPACE.items()]
    d = d.merge(load(RUNS / 'v2_chronos_lora/v2_chronos_lora_predictions.csv.gz', lora_cols),
                on=['variable', 'sensor', 'time_utc'], how='left', validate='one_to_one')
    d = d.merge(load(RUN / 'v2_chronos_timeval_predictions.csv.gz', tv_cols),
                on=['variable', 'sensor', 'time_utc'], how='left', validate='one_to_one').drop(columns='time_utc')
    assert len(d) == 4250
    published = json.loads(BATTERY.read_text())
    report = json.loads((RUN / 'v2_chronos_timeval_report.json').read_text())
    rng = np.random.default_rng(20260919)
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum42.json', 'draws': 600,
           'resampling': 'week blocks within quarter, the resamples of battery_full_v2.json', 'variables': {}}
    for var, space in SPACE.items():
        s0 = d[d.variable == var].copy()
        zs, ft, lora = ZS[var], f'V2FT_{var}_{space}', f'V2LORA_{var}_{space}'
        tv = {m: f'V2TV_{m}_{var}_{space}' for m in ('full', 'lora')}
        for c in tv.values():
            assert s0[c].notna().all(), f'{c} has missing endpoints'
        blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist(); groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}
        pairs = [(ft, zs)] + [(tv[m], ref) for m in tv for ref in (zs, ft if m == 'full' else lora, 'B1_clim_increment')]
        minus_v2 = [ft, zs] + list(tv.values())
        pd_draws = {p: [] for p in pairs}; v2_draws = {m: [] for m in minus_v2}
        for _ in range(600):
            s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))]); base = macro(s, 'MODEL_v2')
            for m in minus_v2: v2_draws[m].append(macro(s, m) - base)
            for x, y in pairs: pd_draws[(x, y)].append(macro(s, x) - macro(s, y))
        res = {f'{x} - {y}': [macro(s0, x) - macro(s0, y), *np.percentile(v, [2.5, 97.5]).tolist()] for (x, y), v in pd_draws.items()}
        ci_v2 = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in v2_draws.items()}
        g3 = (res[f'{ft} - {zs}'] == published[var]['pairs'][f'{ft} - {zs}']
              and all(ci_v2[m] == published[var]['methods'][m]['ci'] for m in (ft, zs)))
        out['variables'][var] = {
            'n': int(len(s0)), 'floor': FLOOR[var],
            'mae': {m: macro(s0, m) for m in [zs, ft, lora, *tv.values(), 'B1_clim_increment', 'MODEL_v2']},
            'pairs': {k: {'point': v[0], 'ci': v[1:], 'excludes_zero': bool(v[1] > 0 or v[2] < 0)} for k, v in res.items()},
            'minus_v2_ci': {m: ci_v2[m] for m in tv.values()},
            'G3': bool(g3)}
    out['diagnostics'] = {k: {x: v[x] for x in ('selection', 'windows', 'selected_mae_2026', 'zero_shot_mae_2026',
                                                'spearman_val_vs_2026', 'n_better_than_zero_shot', 'n_zero_shot_restored',
                                                'best_posthoc', 'mae_2026_range')}
                          for k, v in report.items() if k.startswith('timeval_')}
    V, D = out['variables'], out['diagnostics']
    pair = lambda var, m, ref: V[var]['pairs'][f'V2TV_{m}_{var}_{SPACE[var]} - {ref}']
    out['predictions'] = {
        'P1': all(d_['spearman_val_vs_2026'] > 0 for d_ in D.values()),
        'P2': all(not pair(v, m, ZS[v])['ci'][1] < 0 for v in SPACE for m in ('full', 'lora')),
        'P3': all(pair(v, m, f'V2FT_{v}_{SPACE[v]}' if m == 'full' else f'V2LORA_{v}_{SPACE[v]}')['point'] <= 0
                  for v in SPACE for m in ('full', 'lora')),
        'P4': all(abs(pair(v, m, ZS[v])['point']) < FLOOR[v] for v in SPACE for m in ('full', 'lora'))}
    out['gates_G3_passed'] = all(V[v]['G3'] for v in V)
    (RUN / 'contrasts.json').write_text(json.dumps(out, indent=2))
    for var in V:
        print(var, 'G3', V[var]['G3'])
        for k, p in V[var]['pairs'].items():
            print(f"  {k:62s} {p['point']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]" + ('  *' if p['excludes_zero'] else ''))
    for k, v in D.items():
        print(k, 'rho', round(v['spearman_val_vs_2026'], 3), 'selected', round(v['selected_mae_2026'], 4),
              'zs restored in', v['n_zero_shot_restored'], 'selected is zs:', v['selection']['zero_shot_restored'])
    print(json.dumps(out['predictions']))


if __name__ == '__main__':
    main()
