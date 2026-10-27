"""Addendum 52: the pretrained utilisation arms in umol/L against their published umol/kg values.

Joins the comparison battery exactly as scripts/v2_battery.py does (through
scripts.v2_chronos_lora_contrasts.battery_frame), adds the published adapted columns of addenda 40,
42 and 43 and the columns written by scripts/experiments/v2_aou_units_chronos.py on the
compute host (runs/v2_aou_units_chronos/), and draws the battery's own 600 week-block resamples within
quarter (seed 20260919; the temperature draws are consumed first, as every battery script does), so
every published oxygen interval is reproduced (gate G4) and every umol/L value is read on the same
resamples. Writes runs/v2_aou_units_chronos.json.

Usage: python -m scripts.v2_aou_units_chronos_contrasts
Declared in config/bay_reserved_protocol_v2_addendum52.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_chronos_lora_contrasts import battery_frame, macro

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
HOST = RUNS / 'v2_aou_units_chronos'
BATTERY = RUNS / 'v2_reserved_reading_v2/battery_full_v2.json'
OUT = RUNS / 'v2_aou_units_chronos.json'
KEY = ['variable', 'sensor', 'time_utc']
FLOOR = 1.8
# arm -> published column (umol/kg, as the thesis quotes it), host rerun columns in each unit
ARMS = {
    'bolt_aou': ('oxygen|ZS_aou', 'oxygen|ZS_aou|kg', 'oxygen|ZS_aou|l'),
    'bolt_aou_c2': ('V2C2_oxygen_aou', 'V2C2_oxygen_aou|kg', 'V2C2_oxygen_aou|l'),
    'ch2_aou_uni': ('CH2_oxygen_aou_uni', 'CH2_oxygen_aou_uni|kg', 'CH2_oxygen_aou_uni|l'),
    'ch2_aou_cov': ('CH2_oxygen_aou_cov', 'CH2_oxygen_aou_cov|kg', 'CH2_oxygen_aou_cov|l'),
    'ft_random': ('V2FT_oxygen_aou', 'ft_random|kg', 'ft_random|l'),
    'ft_posthoc': (None, 'ft_posthoc|kg', 'ft_posthoc|l'),
    'lora_random': ('V2LORA_oxygen_aou', 'lora_random|kg', 'lora_random|l'),
    'lora_posthoc': ('LORA_posthoc_published', 'lora_posthoc|kg', 'lora_posthoc|l'),
    'tv_full': ('V2TV_full_oxygen_aou', 'tv_full|kg', 'tv_full|l'),
    'tv_lora': ('V2TV_lora_oxygen_aou', 'tv_lora|kg', 'tv_lora|l'),
    'refit_full': ('V2RF_full_oxygen_aou', 'refit_full|kg', 'refit_full|l'),
    'refit_lora': ('V2RF_lora_oxygen_aou', 'refit_lora|kg', 'refit_lora|l')}
SELECTED = ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_uni', 'ch2_aou_cov', 'ft_random', 'lora_random', 'tv_full', 'tv_lora')
PAIRS = [('bolt_aou', 'oxygen|ZS_raw'), ('bolt_aou', 'B1_clim_increment'),
         ('ch2_aou_uni', 'CH2_oxygen_raw_uni'), ('ch2_aou_cov', 'ch2_aou_uni'), ('ch2_aou_cov', 'B1_clim_increment'),
         ('ft_random', 'bolt_aou'),
         ('lora_random', 'bolt_aou'), ('lora_random', 'ft_random'), ('lora_random', 'B1_clim_increment'),
         ('tv_full', 'bolt_aou'), ('tv_full', 'ft_random'), ('tv_full', 'B1_clim_increment'),
         ('tv_lora', 'bolt_aou'), ('tv_lora', 'lora_random'), ('tv_lora', 'B1_clim_increment'),
         ('refit_full', 'bolt_aou'), ('refit_full', 'tv_full'), ('refit_full', 'B1_clim_increment'),
         ('refit_lora', 'bolt_aou'), ('refit_lora', 'tv_lora'), ('refit_lora', 'B1_clim_increment')]
UNITS = {'published': 0, 'kg': 1, 'l': 2}
REFERENCE = ('oxygen|ZS_raw', 'CH2_oxygen_raw_uni', 'B1_clim_increment', 'MODEL_v2', 'B0_persistence')


def column_of(name, unit):
    return ARMS[name][UNITS[unit]] if name in ARMS else name


def load(path, columns, rename=None):
    f = pd.read_csv(path)
    f['time_utc'] = pd.to_datetime(f.time, utc=True)
    f = f[KEY + columns]
    return f.rename(columns=rename or {})


def frame():
    d = battery_frame()
    d['time_utc'] = pd.to_datetime(d.time, utc=True)
    d = d.merge(load(RUNS / 'v2_chronos_lora/v2_chronos_lora_predictions.csv.gz', ['V2LORA_oxygen_aou']),
                on=KEY, how='left', validate='one_to_one')
    lora = pd.read_csv(RUNS / 'v2_chronos_lora/v2_chronos_lora_predictions.csv.gz')
    lora['time_utc'] = pd.to_datetime(lora.time, utc=True)
    trial = json.loads((RUNS / 'v2_chronos_lora/v2_chronos_lora_report.json').read_text())['lora_oxygen_aou']['best_posthoc']['trial']
    lora['LORA_posthoc_published'] = np.load(RUNS / 'v2_chronos_lora/lora_oxygen_aou_2026_columns.npz')[str(trial)]
    d = d.merge(lora[KEY + ['LORA_posthoc_published']], on=KEY, how='left', validate='one_to_one')
    d = d.merge(load(RUNS / 'v2_chronos_timeval/v2_chronos_timeval_predictions.csv.gz',
                     ['V2TV_full_oxygen_aou', 'V2TV_lora_oxygen_aou']), on=KEY, how='left', validate='one_to_one')
    d = d.merge(load(RUNS / 'v2_chronos_refit/v2_chronos_refit_predictions.csv.gz',
                     ['V2RF_full_oxygen_aou', 'V2RF_lora_oxygen_aou']), on=KEY, how='left', validate='one_to_one')
    host = {}
    zs = HOST / 'zeroshot_columns.csv.gz'
    if zs.exists():
        z = pd.read_csv(zs)
        cols = [c for c in z.columns if '|' in c and c not in ('variable', 'sensor', 'time')]
        z['time_utc'] = pd.to_datetime(z.time, utc=True)
        z = z[KEY + cols].rename(columns={c: f'host:{c}' for c in cols})
        d = d.merge(z, on=KEY, how='left', validate='one_to_one')
        host.update({c: f'host:{c}' for c in cols})
    for p in sorted(HOST.glob('adapted_*_columns.csv.gz')):
        a = pd.read_csv(p)
        cols = [c for c in a.columns if '|' in c]
        a['time_utc'] = pd.to_datetime(a.time, utc=True)
        a = a[KEY + cols].rename(columns={c: f'host:{c}' for c in cols})
        d = d.merge(a, on=KEY, how='left', validate='one_to_one')
        host.update({c: f'host:{c}' for c in cols})
    assert len(d) == 4250
    return d, host


def main():
    d, host = frame()
    published = json.loads(BATTERY.read_text())
    rng = np.random.default_rng(20260919)
    t0 = d[d.variable == 'temperature']
    n_t = len(t0[['quarter', 'week']].drop_duplicates())
    for _ in range(600):                         # the battery draws temperature first
        rng.integers(0, n_t, n_t)
    s0 = d[d.variable == 'oxygen'].copy()
    blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}

    def col(name, unit):
        c = column_of(name, unit)
        if c is None:
            return None
        if unit != 'published' and name in ARMS:
            c = host.get(c)
        return c if c is not None and c in s0 and s0[c].notna().all() else None

    series = {}
    for name in ARMS:
        for unit in UNITS:
            c = col(name, unit)
            if c is not None:
                series[(name, unit)] = c
    for r in REFERENCE:
        series[(r, 'published')] = r
    # the raw columns rerun on the host (a check of the environment, gate G1)
    for c in ('oxygen|ZS_raw', 'CH2_oxygen_raw_uni'):
        if f'host:{c}' in s0:
            series[(c, 'kg')] = f'host:{c}'
    combos = {}
    for (name, unit), c in series.items():
        if name in ARMS:
            combos[f'{name}@{unit} - MODEL_v2'] = ((c, 1), ('MODEL_v2', -1))
    for unit in UNITS:
        for x, y in PAIRS:
            cx = series.get((x, unit)) if x in ARMS else x
            cy = series.get((y, unit)) if y in ARMS else (series.get((y, 'kg')) if unit == 'kg' and (y, 'kg') in series else y)
            if cx and cy:
                combos[f'{x}@{unit} - {y}' + (f'@{unit}' if y in ARMS else '')] = ((cx, 1), (cy, -1))
    for name in ARMS:
        for a, b in (('l', 'published'), ('l', 'kg'), ('kg', 'published')):
            if (name, a) in series and (name, b) in series:
                combos[f'{name}: {a} - {b}'] = ((series[(name, a)], 1), (series[(name, b)], -1))
    cols = sorted({c for pair in combos.values() for c, _ in pair})
    draws = {k: [] for k in combos}
    for _ in range(600):
        s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        m = {c: macro(s, c) for c in cols}
        for k, pair in combos.items():
            draws[k].append(sum(m[c] * w for c, w in pair))
    point = {c: macro(s0, c) for c in cols}
    res = {k: [float(sum(point[c] * w for c, w in pair)), *np.percentile(draws[k], [2.5, 97.5]).tolist()]
           for k, pair in combos.items()}

    arms = {}
    for name in ARMS:
        e = {}
        for unit in UNITS:
            c = series.get((name, unit))
            if c is None:
                continue
            e[unit] = {'mae': point[c] if c in point else macro(s0, c),
                       'minus_v2': res.get(f'{name}@{unit} - MODEL_v2')}
        for key in ('l - published', 'l - kg', 'kg - published'):
            if f'{name}: {key}' in res:
                e[key] = res[f'{name}: {key}']
        arms[name] = e
    # G4: the published columns reproduce battery_full_v2.json and the adaptation contrasts.
    po = published['oxygen']
    g4 = {}
    for name, pub_col in (('bolt_aou', 'oxygen|ZS_aou'), ('bolt_aou_c2', 'V2C2_oxygen_aou'), ('ch2_aou_uni', 'CH2_oxygen_aou_uni'),
                          ('ch2_aou_cov', 'CH2_oxygen_aou_cov'), ('ft_random', 'V2FT_oxygen_aou')):
        if (name, 'published') in series:
            mine = res[f'{name}@published - MODEL_v2']
            g4[name] = max(abs(point[pub_col] - po['methods'][pub_col]['mae']),
                           *(abs(a - b) for a, b in zip(mine[1:], po['methods'][pub_col]['ci'])))
    for k, v in po['pairs'].items():
        x, y = [t.strip() for t in k.split(' - ')]
        rev = {v_[0]: n for n, v_ in ARMS.items() if v_[0]}
        if x in rev and (y in rev or y in REFERENCE):
            key = f'{rev[x]}@published - ' + (f'{rev[y]}@published' if y in rev else y)
            if key in res:
                g4[f'pair {k}'] = max(abs(a - b) for a, b in zip(res[key], v))
    for script, fname in (('lora', 'v2_chronos_lora/contrasts.json'), ('timeval', 'v2_chronos_timeval/contrasts.json'),
                          ('refit', 'v2_chronos_refit/contrasts.json')):
        pub = json.loads((RUNS / fname).read_text())['variables']['oxygen']['pairs']
        rev = {v_[0]: n for n, v_ in ARMS.items() if v_[0]}
        for k, v in pub.items():
            x, y = [t.strip() for t in k.split(' - ')]
            if x in rev:
                key = f'{rev[x]}@published - ' + (f'{rev[y]}@published' if y in rev else y)
                if key in res:
                    g4[f'{script}: {k}'] = max(abs(a - b) for a, b in zip(res[key], [v['point'], *v['ci']]))
    gate_g4 = {'max_abs_diff': max(g4.values()) if g4 else None, 'checked': len(g4), 'by_entry': g4,
               'passed': bool(g4 and max(g4.values()) <= 1e-12)}
    # host reports and gates
    reports = {}
    if (HOST / 'zeroshot_report.json').exists():
        z = json.loads((HOST / 'zeroshot_report.json').read_text())
        reports['zeroshot'] = {k: z[k] for k in ('G5', 'gates', 'c2_kg', 'c2_l', 'seconds') if k in z}
    for p in sorted(HOST.glob('adapted_*_report.json')):
        r = json.loads(p.read_text())
        reports[r['arm']] = {k: r[k] for k in ('trial', 'params', 'published_mae_2026', 'G3', 'G3_refit', 'refit_published', 'seconds')
                             if k in r}
        reports[r['arm']]['units'] = {u: {k: v for k, v in x.items() if k != 'curve'} for u, x in r['units'].items()}
    # per-sensor contrasts against the released model (Table bay_sensor_axis), descriptive
    per_sensor = {}
    for name in ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_cov'):
        for unit in ('published', 'l'):
            c = series.get((name, unit))
            if c:
                per_sensor[f'{name}@{unit}'] = {s: round(macro(g, c) - macro(g, 'MODEL_v2'), 4) for s, g in s0.groupby('sensor')}
    # Table bay_sensor_axis in umol/L (descriptive): its three schemes on the same streams as the
    # published rows (label '<published column> - MODEL_v2'), after reproducing the published rows.
    from scripts import v2_sensor_axis as sx
    sx_pub = json.loads((RUNS / 'v2_sensor_axis.json').read_text())['variables']['oxygen']['contrasts']
    sensors = sorted(s0.sensor.unique())
    sensor_axis = {}
    for name in ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_uni'):
        pub_col = ARMS[name][0]
        label = f'{pub_col} - MODEL_v2'
        entry = {}
        for unit in ('published', 'l'):
            c = series.get((name, unit))
            if c is None:
                continue
            row = {'point': macro(s0, c) - macro(s0, 'MODEL_v2'),
                   'per_sensor': {s: round(macro(g, c) - macro(g, 'MODEL_v2'), 4) for s, g in s0.groupby('sensor')}}
            for scheme in ('published', 'sensor_only', 'two_way'):
                row[scheme] = sx.interval(s0, c, 'MODEL_v2', scheme, label, sensors)
            entry[unit] = row
        if 'published' in entry and label in sx_pub:
            p = sx_pub[label]
            entry['reproduces_published'] = bool(
                abs(entry['published']['point'] - p['point']) <= 1e-12 and
                all(entry['published'][s]['ci'] == p[s]['ci'] for s in ('published', 'sensor_only', 'two_way')))
        sensor_axis[name] = entry
    # D2: the lowest selected oxygen arm in each unit
    table_rows = {'oxygen|ZS_raw': 'Chronos-Bolt zs', 'MODEL_v2': 'released model', 'B0_persistence': 'persistence'}
    ranking = {}
    for unit in ('published', 'l'):
        maes = {f'{n}@{unit}': point.get(series[(n, unit)], macro(s0, series[(n, unit)])) for n in SELECTED if (n, unit) in series}
        maes.update({table_rows[r]: macro(s0, r) for r in table_rows})
        ranking[unit] = sorted(maes.items(), key=lambda kv: kv[1])
    lowest_l = ranking['l'][0][0] if ranking.get('l') else None
    preds = {}
    if 'l - published' in arms.get('bolt_aou', {}):
        preds['U1'] = {'held': bool(abs(arms['bolt_aou']['l - published'][0]) < 0.1 and lowest_l == 'bolt_aou@l'),
                       'movement': arms['bolt_aou']['l - published'], 'lowest_in_l': lowest_l}
    u2 = [res.get('bolt_aou@l - oxygen|ZS_raw'), res.get('ch2_aou_uni@l - CH2_oxygen_raw_uni')]
    if all(u2):
        preds['U2'] = {'held': all(v[1] <= 0 <= v[2] for v in u2), 'values': u2}
    moves = {n: e['l - published'] for n, e in arms.items() if 'l - published' in e}
    moves_kg = {n: e['l - kg'] for n, e in arms.items() if 'l - kg' in e and n in ('ft_random', 'ft_posthoc')}
    changed = []
    for n, e in arms.items():
        if 'published' in e and 'l' in e and e['published']['minus_v2'] and e['l']['minus_v2']:
            a, b = e['published']['minus_v2'], e['l']['minus_v2']
            if (a[1] > 0 or a[2] < 0) != (b[1] > 0 or b[2] < 0):
                changed.append(n)
    if moves:
        preds['U3'] = {'held': bool(all(abs(v[0]) <= 0.2 for v in {**moves, **moves_kg}.values()) and not changed),
                       'largest_movement': max(abs(v[0]) for v in {**moves, **moves_kg}.values()), 'interval_changed': changed}
    ad = [n for n in ('ft_random', 'lora_random', 'tv_full', 'tv_lora', 'refit_full', 'refit_lora') if f'{n}@l - bolt_aou@l' in res]
    if ad:
        preds['U4'] = {'held': all(res[f'{n}@l - bolt_aou@l'][0] > 0 for n in ad),
                       'values': {n: res[f'{n}@l - bolt_aou@l'] for n in ad}}
    out = {'declared_in': 'config/bay_reserved_protocol_v2_addendum52.json', 'cohort': 'Reserved-2026 oxygen',
           'n': int(len(s0)), 'draws': 600, 'resampling': 'the battery resamples (week blocks within quarter, seed 20260919)',
           'floor': FLOOR, 'arms': arms, 'contrasts': res, 'gate_G4': gate_g4, 'host_reports': reports,
           'per_sensor_minus_v2': per_sensor, 'sensor_axis': sensor_axis, 'ranking': ranking, 'predictions': preds}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print('G4', gate_g4['passed'], gate_g4['max_abs_diff'], gate_g4['checked'])
    for n, e in arms.items():
        line = '  '.join(f"{u} {x['mae']:.4f}" for u, x in e.items() if isinstance(x, dict))
        mv = e.get('l - published') or e.get('l - kg')
        print(f'{n:14s} {line}' + (f"   L-pub {mv[0]:+.4f} [{mv[1]:+.4f}, {mv[2]:+.4f}]" if mv else ''))
    print(json.dumps(preds, indent=1, default=str))


if __name__ == '__main__':
    main()
