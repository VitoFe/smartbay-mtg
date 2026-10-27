"""Addendum 43: score the refits against zero-shot and the addendum-42 selections they came from.

Same join and the same 600 week-block resamples within quarter (seed 20260919) as
scripts/v2_battery.py; gate G3 reproduces its full-fine-tuning and zero-shot pairs and intervals
exactly. Writes runs/v2_chronos_refit/contrasts.json.

Usage: python -m scripts.v2_chronos_refit_contrasts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_chronos_lora_contrasts import battery_frame, macro

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
RUN = RUNS / 'v2_chronos_refit'
BATTERY = RUNS / 'v2_reserved_reading_v2/battery_full_v2.json'
SPACE = {'temperature': 'raw', 'oxygen': 'aou'}
ZS = {'temperature': 'temperature|ZS_raw', 'oxygen': 'oxygen|ZS_aou'}


def load(path, columns):
    frame = pd.read_csv(path)
    frame['time_utc'] = pd.to_datetime(frame.time, utc=True)
    return frame[['variable', 'sensor', 'time_utc'] + columns]


def main():
    d = battery_frame()
    d['time_utc'] = pd.to_datetime(d.time, utc=True)
    tv_cols = [f'V2TV_{m}_{v}_{s}' for m in ('full', 'lora') for v, s in SPACE.items()]
    rf_cols = [f'V2RF_{m}_{v}_{s}' for m in ('full', 'lora') for v, s in SPACE.items()]
    d = d.merge(load(RUNS / 'v2_chronos_timeval/v2_chronos_timeval_predictions.csv.gz', tv_cols),
                on=['variable', 'sensor', 'time_utc'], how='left', validate='one_to_one')
    d = d.merge(load(RUN / 'v2_chronos_refit_predictions.csv.gz', rf_cols),
                on=['variable', 'sensor', 'time_utc'], how='left', validate='one_to_one').drop(columns='time_utc')
    assert len(d) == 4250
    published = json.loads(BATTERY.read_text())
    report = json.loads((RUN / 'v2_chronos_refit_report.json').read_text())
    rng = np.random.default_rng(20260919)
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum43.json', 'draws': 600,
           'resampling': 'week blocks within quarter, the resamples of battery_full_v2.json', 'variables': {}}
    for var, space in SPACE.items():
        s0 = d[d.variable == var].copy()
        zs, ft = ZS[var], f'V2FT_{var}_{space}'
        rf = {m: f'V2RF_{m}_{var}_{space}' for m in ('full', 'lora')}
        tv = {m: f'V2TV_{m}_{var}_{space}' for m in ('full', 'lora')}
        for c in rf.values():
            assert s0[c].notna().all(), f'{c} has missing endpoints'
        blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist(); groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}
        pairs = [(ft, zs)] + [(rf[m], ref) for m in rf for ref in (zs, tv[m], 'B1_clim_increment')]
        minus_v2 = [ft, zs] + list(rf.values())
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
            'n': int(len(s0)),
            'mae': {m: macro(s0, m) for m in [zs, *tv.values(), *rf.values(), 'B1_clim_increment', 'MODEL_v2']},
            'pairs': {k: {'point': v[0], 'ci': v[1:], 'excludes_zero': bool(v[1] > 0 or v[2] < 0)} for k, v in res.items()},
            'minus_v2_ci': {m: ci_v2[m] for m in rf.values()},
            'G3': bool(g3)}
    out['refits'] = {k: v for k, v in report.items() if k.startswith('timeval_')}
    V = out['variables']
    pair = lambda var, m, ref: V[var]['pairs'][f'V2RF_{m}_{var}_{SPACE[var]} - {ref}']
    out['predictions'] = {
        'P1': all(pair(v, m, f'V2TV_{m}_{v}_{SPACE[v]}')['point'] <= 0 for v in SPACE for m in ('full', 'lora')),
        'P2': all(not pair(v, m, ZS[v])['ci'][1] < 0 for v in SPACE for m in ('full', 'lora')),
        'P3': not pair('temperature', 'lora', ZS['temperature'])['excludes_zero']}
    out['gates'] = {'G1': all(r['G1']['passed'] for r in out['refits'].values()),
                    'G2': all(r['G2']['passed'] for r in out['refits'].values()),
                    'G3': all(V[v]['G3'] for v in V)}
    (RUN / 'contrasts.json').write_text(json.dumps(out, indent=2))
    for var in V:
        print(var, 'G3', V[var]['G3'])
        for k, p in V[var]['pairs'].items():
            print(f"  {k:62s} {p['point']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]" + ('  *' if p['excludes_zero'] else ''))
    for k, r in out['refits'].items():
        print(k, 'restored step', r['restored_step'], 'G1', r['G1']['passed'], round(r['G1']['val_loss_rerun'] - r['G1']['val_loss_study'], 6),
              round(r['G1']['mae_rerun'] - r['G1']['mae_study'], 6), 'refit windows', r['refit_windows'], 'mae', round(r['mae_refit_2026'], 4))
    print(json.dumps({'gates': out['gates'], 'predictions': out['predictions']}))


if __name__ == '__main__':
    main()
