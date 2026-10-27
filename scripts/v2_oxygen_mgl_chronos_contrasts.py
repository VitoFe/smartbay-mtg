"""Addendum 60 part C3: the pretrained utilisation arms on Reserved-2026, in mg/l.

Reads the battery frame exactly as scripts/v2_aou_units_chronos_contrasts.py does, and the columns
that scripts/experiments/v2_oxygen_mgl_chronos.py and v2_oxygen_mgl_chronos_adapted.py wrote on the
compute host. Under addendum 60 amendment 4 (declared 2026-10-02T14:05Z, after the first part-C3 run)
the host reruns read the sonde oxygen times the constant c and the saturation in mg/l (the per-litre
Garcia-Gordon form x 0.0319988); here the truth and every arm that reads the sondes alone (the
released model, persistence, the climatological increment, the zero-shot raw forecasts) are their
published values x c, so they carry the values every other table prints, and every contrast is
paired on the same endpoints. Intervals are the battery's own 600 week-block resamples within
quarter (seed 20260919, the temperature draws consumed first); the sensor-axis schemes use their
published streams.

Units of the arms in the output:
  mgl           the zero-shot Chronos-Bolt and Chronos-2 utilisation arms rerun on the host, and the six
                adapted arms (fine-tuned and low-rank adapted, with the two refits) retrained there at
                their published configurations and seeds (the harmonised value)
  kg_x_c        the published umol/kg-saturation columns (addendum 52, 'kg') times c, shown beside them
  l_x_c         the published 'l' columns (the values the thesis prints) times c, for comparison

With --hourly-factor it recomputes the first part-C3 record instead (every oxygen row by its target
hour's factor, unit 'kg_x_factor'), from and into C3/_hourly_factor/.

Gates: the reimplemented draws reproduce the published 'l' values of runs/v2_aou_units_chronos.json; the
published-mode host reruns reproduce the published '|l' columns (zero-shot in each run, the adapted
arms in the first run, C3/_hourly_factor/); the references x c give the MAEs the other tables print.

Usage: python -m scripts.v2_oxygen_mgl_chronos_contrasts [--hourly-factor]
"""
import json
import sys

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import OUT, RUNS, constant_c, sonde_factor, write_json

HOURLY = '--hourly-factor' in sys.argv
ROOT_C3 = OUT / 'C3' / ('_hourly_factor' if HOURLY else '')
HOST = ROOT_C3 / 'v2_aou_units_chronos'
DST = ROOT_C3 / 'v2_aou_units_chronos.json'
#: the published-unit retrains of the adapted arms (first part-C3 run, bit-identical to the published columns)
ADAPTED_REPRODUCTION = OUT / 'C3/_hourly_factor/v2_aou_units_chronos'
KG = 'kg_x_factor' if HOURLY else 'kg_x_c'
KEY = ['variable', 'sensor', 'time_utc']
ZS = ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_uni', 'ch2_aou_cov')
ADAPTED = ('ft_random', 'ft_posthoc', 'lora_random', 'lora_posthoc', 'tv_full', 'tv_lora')
#: the adapted host columns: (arm file, column) -> arm name of the published table
ADAPTED_COLUMNS = {**{(a, a): a for a in ADAPTED}, ('tv_full', 'refit_full'): 'refit_full',
                   ('tv_lora', 'refit_lora'): 'refit_lora'}
HOSTCOL = {'bolt_aou': 'oxygen|ZS_aou', 'bolt_aou_c2': 'V2C2_oxygen_aou', 'ch2_aou_uni': 'CH2_oxygen_aou_uni',
           'ch2_aou_cov': 'CH2_oxygen_aou_cov'}
PER_KG = ('truth', 'MODEL_v2', 'MODEL_v1', 'B0_persistence', 'B1_clim_increment', 'oxygen|ZS_raw', 'CH2_oxygen_raw_uni',
          'CH2_oxygen_raw_cov', 'V2C2_oxygen_raw', 'CHRONOS_bolt_small', 'CHRONOS_bolt_base')


def compute(d, series, combos):
    """point and 600 battery draws of every combo, over the oxygen rows of d."""
    from scripts.v2_chronos_lora_contrasts import macro
    rng = np.random.default_rng(20260919)
    t0 = d[d.variable == 'temperature']
    n_t = len(t0[['quarter', 'week']].drop_duplicates())
    for _ in range(600):
        rng.integers(0, n_t, n_t)
    s0 = d[d.variable == 'oxygen'].copy()
    blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}
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
    return s0, point, res


def main():
    from scripts import v2_aou_units_chronos_contrasts as U
    from scripts import v2_sensor_axis as sx
    d, host = U.frame()
    published = json.loads((RUNS / 'v2_aou_units_chronos.json').read_text())
    c = constant_c()
    oxy = (d.variable == 'oxygen').to_numpy()

    # gate 1: the draws reimplemented here give the published 'l' values
    series_l = {n: host[U.ARMS[n][2]] for n in U.ARMS if U.ARMS[n][2] in host}
    combos = {f'{n}@l - MODEL_v2': ((col, 1), ('MODEL_v2', -1)) for n, col in series_l.items()}
    _, point, res = compute(d, series_l, combos)
    gate = {}
    for n, col in series_l.items():
        want = published['arms'][n]['l']
        gate[n] = max(abs(point[col] - want['mae']), *(abs(a - b) for a, b in zip(res[f'{n}@l - MODEL_v2'], want['minus_v2'])))
    gate1 = {'max_abs_diff': max(gate.values()), 'by_arm': gate, 'passed': bool(max(gate.values()) <= 1e-12)}
    print('gate 1 (published l values)', gate1['passed'], gate1['max_abs_diff'], flush=True)
    # gate 2: the published-mode host rerun reproduces the published '|l' columns
    hr = json.loads((HOST / 'zeroshot_published_report.json').read_text())
    gate2 = {'passed': bool(hr.get('passed')), 'host': hr.get('gate'), 'adapted': {}}
    for arm in ADAPTED:
        f = ADAPTED_REPRODUCTION / f'adapted_{arm}_published_report.json'
        if f.exists():
            r = json.loads(f.read_text())
            gate2['adapted'][arm] = {'passed': bool(r.get('passed')), 'gate': r.get('gate')}
    adapted_ok = [a for a, g in gate2['adapted'].items() if g['passed']]
    print('gate 2 (host reproduction)', gate2['passed'], flush=True)
    if not (gate1['passed'] and gate2['passed']):
        write_json(DST, {'gates': {'G1_draws': gate1, 'G2_host': gate2}, 'passed': False})
        raise SystemExit('a gate failed; nothing converted')

    # mg/l: every oxygen row by c (amendment 4), or by its target hour's factor (the first run)
    factor = np.full(len(d), c)
    if HOURLY:
        target = pd.to_datetime(d.time, utc=True) + pd.Timedelta(hours=6)
        for s in d.loc[oxy, 'sensor'].unique():
            rows = oxy & (d.sensor == s).to_numpy()
            factor[rows] = sonde_factor(s, target[rows])
    m = d.copy()
    for col in PER_KG:
        if col in m:
            m.loc[oxy, col] = m.loc[oxy, col].to_numpy(float) * factor[oxy]
    kg_cols = {n: host[U.ARMS[n][1]] for n in U.ARMS if U.ARMS[n][1] in host}
    for n, col in kg_cols.items():
        m[f'kgx:{n}'] = m[col]
        m.loc[oxy, f'kgx:{n}'] = m.loc[oxy, col].to_numpy(float) * factor[oxy]
    for n, col in series_l.items():
        m[f'lxc:{n}'] = m[col]
        m.loc[oxy, f'lxc:{n}'] = m.loc[oxy, col].to_numpy(float) * c
    z = pd.read_csv(HOST / 'zeroshot_mgl_columns.csv.gz')
    z['time_utc'] = pd.to_datetime(z.time, utc=True)
    z = z[KEY + [f'{HOSTCOL[n]}|mgl' for n in ZS]].rename(columns={f'{HOSTCOL[n]}|mgl': f'mgl:{n}' for n in ZS})
    m = m.merge(z, on=KEY, how='left', validate='one_to_one')
    rerun = list(ZS)
    for (arm, colname), name in ADAPTED_COLUMNS.items():
        f = HOST / f'adapted_{arm}_mgl_columns.csv.gz'
        if arm not in adapted_ok or not f.exists():
            continue
        a = pd.read_csv(f)
        a['time_utc'] = pd.to_datetime(a.time, utc=True)
        m = m.merge(a[KEY + [f'{colname}|mgl']].rename(columns={f'{colname}|mgl': f'mgl:{name}'}),
                    on=KEY, how='left', validate='one_to_one')
        rerun.append(name)
    assert len(m) == len(d)

    series = {}
    for n in rerun:
        series[(n, 'mgl')] = f'mgl:{n}'
    for n in kg_cols:
        series[(n, KG)] = f'kgx:{n}'
    for n in series_l:
        series[(n, 'l_x_c')] = f'lxc:{n}'
    combos = {}
    for (n, unit), col in series.items():
        combos[f'{n}@{unit} - MODEL_v2'] = ((col, 1), ('MODEL_v2', -1))
    for unit in ('mgl', KG, 'l_x_c'):
        for x, y in U.PAIRS:
            cx = series.get((x, unit)) if x in U.ARMS else x
            cy = series.get((y, unit)) if y in U.ARMS else y
            if unit == 'mgl' and y in U.ARMS and (y, 'mgl') not in series:
                cy = None
            if cx and cy:
                combos[f'{x}@{unit} - {y}' + (f'@{unit}' if y in U.ARMS else '')] = ((cx, 1), (cy, -1))
    for n in rerun:
        if n in kg_cols:
            combos[f'{n}: mgl - {KG}'] = ((f'mgl:{n}', 1), (f'kgx:{n}', -1))
        if n in series_l:
            combos[f'{n}: mgl - l_x_c'] = ((f'mgl:{n}', 1), (f'lxc:{n}', -1))
    s0, point, res = compute(m, series, combos)
    arms = {}
    for (n, unit), col in series.items():
        arms.setdefault(n, {})[unit] = {'mae': point[col], 'minus_v2': res[f'{n}@{unit} - MODEL_v2']}
    for n in rerun:
        for k in (f'mgl - {KG}', 'mgl - l_x_c'):
            if f'{n}: {k}' in res:
                arms[n][k] = res[f'{n}: {k}']
    from scripts.v2_chronos_lora_contrasts import macro
    references = {r: macro(s0, r) for r in ('MODEL_v2', 'B0_persistence', 'B1_clim_increment', 'oxygen|ZS_raw',
                                            'CH2_oxygen_raw_uni', 'CH2_oxygen_raw_cov', 'V2C2_oxygen_raw')}
    d0 = d[d.variable == 'oxygen']
    g3 = {r: {'published_x_c': macro(d0, r) * c, 'here': v} for r, v in references.items()}
    for r in g3:
        g3[r]['abs_diff'] = abs(g3[r]['here'] - g3[r]['published_x_c'])
    gate3 = {'what': 'the reference arms here against their published MAE x c (amendment 4)',
             'passed': bool(HOURLY or all(g['abs_diff'] <= 1e-12 for g in g3.values())), 'by_arm': g3}
    print('gate 3 (references = published x c)', gate3['passed'], flush=True)
    per_sensor = {f'{n}@mgl': {s: round(macro(g, f'mgl:{n}') - macro(g, 'MODEL_v2'), 6) for s, g in s0.groupby('sensor')}
                  for n in ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_cov')}
    sensors = sorted(s0.sensor.unique())
    sensor_axis = {}
    for n in ('bolt_aou', 'bolt_aou_c2', 'ch2_aou_uni'):
        label = f'{U.ARMS[n][0]} - MODEL_v2'
        col = f'mgl:{n}'
        row = {'point': macro(s0, col) - macro(s0, 'MODEL_v2'),
               'per_sensor': {s: round(macro(g, col) - macro(g, 'MODEL_v2'), 6) for s, g in s0.groupby('sensor')}}
        for scheme in ('published', 'sensor_only', 'two_way'):
            row[scheme] = sx.interval(s0, col, 'MODEL_v2', scheme, label, sensors)
        sensor_axis[n] = {'mgl': row}
    ranking = sorted([(f'{n}@mgl', point[f'mgl:{n}']) for n in rerun]
                     + [(f'{n}@{KG}', point[f'kgx:{n}']) for n in kg_cols if n not in rerun]
                     + [(k, v) for k, v in references.items()], key=lambda kv: kv[1])
    out = {'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json',
                           'config/bay_reserved_protocol_v2_addendum52.json'] +
                          ([] if HOURLY else ['config/bay_reserved_protocol_v2_addendum60_amendment4.json']),
           'cohort': 'Reserved-2026 oxygen', 'n': int(len(s0)), 'draws': 600, 'unit': 'mg/l',
           'resampling': 'the battery resamples (week blocks within quarter, seed 20260919)',
           'floor': 1.8 * c,
           'sonde_oxygen': 'x the target-hour factor (first run)' if HOURLY else 'x the constant c (amendment 4)',
           'units_of_arms': {'mgl': 'utilisation arms rerun (zero-shot) or retrained (adapted) in mg/l on the host',
                             KG: 'published umol/kg-saturation column x ' + ('target-hour factor' if HOURLY else 'c'),
                             'l_x_c': 'published umol/L-saturation column (printed) x c'},
           'arms': arms, 'references_mae': references, 'contrasts': res,
           'gates': {'G1_draws': gate1, 'G2_host': gate2, 'G3_references': gate3},
           'host_reports': {'published': hr, 'mgl': json.loads((HOST / 'zeroshot_mgl_report.json').read_text()),
                            'adapted': {p.name: json.loads(p.read_text()) for p in sorted(HOST.glob('adapted_*_report.json'))}},
           'reran_in_mgl': rerun,
           'per_sensor_minus_v2': per_sensor, 'sensor_axis': sensor_axis, 'ranking': ranking}
    write_json(DST, out)
    for n, e in arms.items():
        print(f'{n:14s} ' + '  '.join(f"{u} {x['mae']:.5f} {[round(v, 4) for v in x['minus_v2']]}"
                                       for u, x in e.items() if isinstance(x, dict)))


if __name__ == '__main__':
    main()
