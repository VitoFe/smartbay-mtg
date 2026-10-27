"""Addendum 40: score the low-rank-adapter arm against zero-shot and full fine-tuning on Reserved-2026.

Joins the comparison battery exactly as scripts/v2_battery.py does, adds the adapter column from
runs/v2_chronos_lora/v2_chronos_lora_predictions.csv.gz, and draws the same 600 week-block resamples
within quarter (seed 20260919). Gate G3: the full-fine-tuning and zero-shot pairs, and their
intervals against the supported model, reproduce battery_full_v2.json exactly before any adapter
contrast is read. Writes runs/v2_chronos_lora/contrasts.json with the registered verdicts.

Usage: python -m scripts.v2_chronos_lora_contrasts
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'reports/bay_twin/runs/battery/results'
RUN = ROOT / 'reports/bay_twin/runs/v2_chronos_lora'
BATTERY = ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/battery_full_v2.json'
KEY = ['variable', 'sensor', 'time']
ARMS = {'temperature': ('V2LORA_temperature_raw', 'temperature|ZS_raw', 'V2FT_temperature_raw'),
        'oxygen': ('V2LORA_oxygen_aou', 'oxygen|ZS_aou', 'V2FT_oxygen_aou')}
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}


def battery_frame():
    """The join of scripts/v2_battery.py, line for line."""
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
    return d.rename(columns={'MODEL': 'MODEL_v1'})


def macro(s, col):
    s = s.dropna(subset=[col]); return float(s.assign(e=(s[col] - s.truth).abs()).groupby(['quarter', 'sensor']).e.mean().groupby('quarter').mean().mean())


def main():
    d = battery_frame()
    lora = pd.read_csv(RUN / 'v2_chronos_lora_predictions.csv.gz', parse_dates=['time'])
    lora['time'] = pd.to_datetime(lora.time, utc=True)
    d['time_utc'] = pd.to_datetime(d.time, utc=True)
    d = d.merge(lora[['variable', 'sensor', 'time'] + [a[0] for a in ARMS.values()]].rename(columns={'time': 'time_utc'}),
                on=['variable', 'sensor', 'time_utc'], how='left', validate='one_to_one').drop(columns='time_utc')
    assert len(d) == 4250
    published = json.loads(BATTERY.read_text())
    rng = np.random.default_rng(20260919)
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum40.json', 'draws': 600,
           'resampling': 'week blocks within quarter, the resamples of battery_full_v2.json', 'variables': {}}
    for var in ['temperature', 'oxygen']:
        s0 = d[d.variable == var].copy()
        lora_col, zs_col, ft_col = ARMS[var]
        assert s0[lora_col].notna().all(), f'{var}: the adapter column has missing endpoints'
        blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist(); groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}
        pairs = [(ft_col, zs_col), (lora_col, zs_col), (lora_col, ft_col), (lora_col, 'B1_clim_increment'), (lora_col, 'MODEL_v2')]
        minus_v2 = [ft_col, zs_col, lora_col]
        pd_draws = {p: [] for p in pairs}; v2_draws = {m: [] for m in minus_v2}
        for _ in range(600):
            s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))]); base = macro(s, 'MODEL_v2')
            for m in minus_v2: v2_draws[m].append(macro(s, m) - base)
            for x, y in pairs: pd_draws[(x, y)].append(macro(s, x) - macro(s, y))
        res = {f'{x} - {y}': [macro(s0, x) - macro(s0, y), *np.percentile(v, [2.5, 97.5]).tolist()] for (x, y), v in pd_draws.items()}
        ci_v2 = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in v2_draws.items()}
        g3_pair = res[f'{ft_col} - {zs_col}'] == published[var]['pairs'][f'{ft_col} - {zs_col}']
        g3_ci = all(ci_v2[m] == published[var]['methods'][m]['ci'] for m in (ft_col, zs_col))
        out['variables'][var] = {
            'n': int(len(s0)), 'floor': FLOOR[var],
            'mae': {m: macro(s0, m) for m in (lora_col, zs_col, ft_col, 'B1_clim_increment', 'MODEL_v2', 'B0_persistence')},
            'pairs': {k: {'point': v[0], 'ci': v[1:], 'excludes_zero': bool(v[1] > 0 or v[2] < 0)} for k, v in res.items()},
            'lora_minus_v2_ci': ci_v2[lora_col],
            'G3': {'passed': bool(g3_pair and g3_ci), 'pair_reproduced': bool(g3_pair), 'intervals_reproduced': bool(g3_ci)}}
    report = json.loads((RUN / 'v2_chronos_lora_report.json').read_text())
    out['diagnostics'] = {name: {k: report[name][k] for k in ('selected_mae_2026', 'zero_shot_mae_2026', 'spearman_val_vs_2026',
                                                              'spearman_params_vs_2026', 'n_better_than_zero_shot', 'best_posthoc',
                                                              'mae_2026_range', 'trainable_params_range')}
                          | {'completed_trials': report[name]['selection']['completed_trials'], 'params': report[name]['selection']['params'],
                             'selected_trainable_params': int(pd.read_csv(RUN / f'{name}_curve.csv').set_index('trial')
                                                              .loc[report[name]['selection']['best_trial'], 'trainable_params'])}
                          for name in ('lora_temperature_raw', 'lora_oxygen_aou')}
    V, D = out['variables'], out['diagnostics']
    pair = lambda var, y: V[var]['pairs'][f'{ARMS[var][0]} - {y}']
    out['predictions'] = {
        'P1': pair('temperature', ARMS['temperature'][2])['ci'][1] < 0,
        'P2': pair('oxygen', ARMS['oxygen'][2])['point'] < 0,
        'P3': all(not pair(v, ARMS[v][1])['ci'][1] < 0 for v in ARMS),
        'P4': all(abs(pair(v, ARMS[v][1])['point']) < FLOOR[v] for v in ARMS),
        'P5': all(D[n]['spearman_val_vs_2026'] < 0 for n in D),
        'P6': D['lora_temperature_raw']['spearman_params_vs_2026'] > 0,
        'P7': all(D[n]['n_better_than_zero_shot'] >= 1 for n in D)}
    out['gates_G3_passed'] = all(V[v]['G3']['passed'] for v in V)
    (RUN / 'contrasts.json').write_text(json.dumps(out, indent=2))
    for var in V:
        print(var, 'G3', V[var]['G3'])
        for k, p in V[var]['pairs'].items(): print(f"  {k:55s} {p['point']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]" + ('  *' if p['excludes_zero'] else ''))
    print(json.dumps(out['predictions']))


if __name__ == '__main__':
    main()
