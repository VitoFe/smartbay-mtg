"""Addendum 86: physical guidance scored on the 2026 test period, the reading.

Declared in config/bay_reserved_protocol_v2_addendum86.json before this script was written. It scores
what src/experiments/v2_guidance2026.py fitted and what earlier addenda stored, and refits nothing:

  A1  the weather tree minus the plain tree on Reserved-2026 at every lead (addendum 85 part B's trees)
  A2  the frozen utilisation tree and the thermodynamic baseline (addendum 82 part G) against part B's
      six-hour oxygen trees (gate A86-G2 first)
  B   each exogenous feature arm against the arm it augments and against the plain tree
  C   the six-hour graph operators (three-seed mean; each seed alone beside it)
  D   the range bound against no gate on the rollouts of the 2026 fold, and every gate's firings
  E1  the utilisation ridge's development reading, beside its 2026 result

Every contrast is paired on the rows where both arms and the truth exist, with the
development value it pairs with read from its artifact (path and key recorded). Reserved-2026 rows are
aggregated by the mean over sensors within each quarter of issuance, then over the quarters, and
resampled by ISO weeks of issuance drawn jointly within each quarter (v2_probabilistic_report), 600
draws; the delete-one-week jackknife-t is given beside every interval. Oxygen is computed in the
sondes' stored unit and reported in mg/l by c (src/units.factor()).

Writes reports/bay_twin/runs/v2_addendum86.json and runs/v2_addendum86/{A,B,C,D,E}.json.
Usage: python -m scripts.v2_guidance2026_report
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v2_fdr as FDR  # noqa: E402
from src.experiments import v2_common_rows as CR  # noqa: E402
from src.experiments import v2_guidance2026 as G  # noqa: E402
from src.experiments import v2_probabilistic_report as R  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_addendum86'
SUMMARY = RUNS / 'v2_addendum86.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum86.json'
C_MG = factor()
FLOORS = {'temperature': 0.029, 'temperature_colocated': 0.20, 'oxygen_mgl': 0.059}
MIN_UNITS = 5
TOL = 1e-9
A83 = 'runs/v2_addendum83.json'
NEURAL = 'runs/v2_perfold_neural.json'
AOU_REST = 'runs/v2_oxygen_mgl/C3/v2_perfold_rest.json'
RANGE_DEV = 'runs/v2_physics_gate/perfold/range_bound_effect.json'
G_FROZEN = RUNS / 'v2_addendum82/G/frozen2026/predictions_2026.npz'
G_REPORT = 'runs/v2_addendum82_G.json'
C3_UNITS = 'runs/v2_oxygen_mgl/C3/v2_bottom_aou/aou_units/summary.json'
#: part B's codes in runs/v2_addendum83.json: against the arm augmented, against the plain tree
A83_CODES = {'river_tree': ('P2', 'P2'), 'weather_river_tree': ('P3', 'S3'), 'sealevel_tree': ('P4', 'S4'),
             'extboundary_tree': ('P5', 'S5'), 'bathy_tree': ('P6', 'S6')}


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def load(path):
    return json.loads((ROOT / 'reports/bay_twin' / path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=1, allow_nan=False, default=G._default).encode('utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------
# One paired contrast on Reserved-2026 rows
# --------------------------------------------------------------------------------------------------
def utc(values):
    return pd.to_datetime(values, utc=True).astype('datetime64[ns, UTC]')


def frame_of(d, name):
    """(sensor, time) keyed truth and prediction of one arm."""
    f = d[['sensor', 'time', 'truth', 'prediction']].copy()
    f['time'] = utc(f.time)
    if f.duplicated(['sensor', 'time']).any():
        raise SystemExit(f'{name}: duplicated rows')
    return f.rename(columns={'truth': f'truth|{name}', 'prediction': name})


def jackknife(frame, cols):
    units = frame[['fold', 'week']].drop_duplicates().itertuples(index=False)
    est = []
    for f, w in units:
        s = frame[~((frame.fold == f) & (frame.week == w))]
        est.append(R.macro(s, cols[0]) - R.macro(s, cols[1]))
    est = np.asarray(est)
    n = len(est)
    se = float(np.sqrt((n - 1) / n * np.sum((est - est.mean()) ** 2)))
    return n, se


def status(lo, hi):
    return 'excludes zero' if (lo > 0 or hi < 0) else 'not distinguishable'


def floor_reading(point, lo, hi, floor):
    """v2_summary_table.against: an interval spanning zero is not read against a floor."""
    if not (lo > 0 or hi < 0):
        return 'interval spans zero'
    if lo > floor or hi < -floor:
        return 'clears floor'
    return 'exceeds floor' if abs(point) > floor else 'below floor'


def contrast(x, y, fx, fy, variable, label, scale=None):
    """x - y on the rows where both arms and the truth exist. `scale` converts the stored unit into the
    printed one (c for oxygen stored in umol/kg, 1 for degC or values already in mg/l)."""
    scale = (C_MG if variable == 'oxygen' else 1.0) if scale is None else scale
    j = fx.merge(fy, on=['sensor', 'time'], how='inner', validate='one_to_one')
    tdiff = float(np.nanmax(np.abs(j[f'truth|{x}'] - j[f'truth|{y}']))) if len(j) else None
    if tdiff is not None and tdiff > TOL:
        raise SystemExit(f'{label}: truths differ by {tdiff}')
    j = j.dropna(subset=[x, y, f'truth|{x}'])
    truth = j[f'truth|{x}'].to_numpy(float)
    frame = pd.DataFrame({'fold': np.where(j.time.dt.month <= 3, 'Q1', 'Q2'),
                          'week': j.time.dt.isocalendar().week.astype(int).to_numpy(),
                          'sensor': j.sensor.to_numpy(),
                          'ae|x': np.abs(truth - j[x].to_numpy(float)) * scale,
                          'ae|y': np.abs(truth - j[y].to_numpy(float)) * scale})
    units = int(frame[['fold', 'week']].drop_duplicates().shape[0])
    mx, my = R.macro(frame, 'ae|x'), R.macro(frame, 'ae|y')
    point = mx - my
    out = {'x': x, 'y': y, 'variable': variable, 'cohort': 'Reserved-2026',
           'unit': 'degC' if variable == 'temperature' else 'mg/l',
           'paired_rows': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
           'units_quarter_week': units, 'mae_x': r6(mx), 'mae_y': r6(my), 'difference': r6(point),
           'change_percent_of_y': r6(100 * point / my) if my else None, 'stream': label}
    if units < MIN_UNITS:
        out.update(ci=None, interval_status='n.e.', jackknife_t=None, p=None)
        return out
    draws = R.joint_draws(frame, ['ae|x', 'ae|y'], label)
    lo, hi = np.percentile(draws[:, 0] - draws[:, 1], [2.5, 97.5])
    n, se = jackknife(frame, ['ae|x', 'ae|y'])
    t = float(stats.t.ppf(0.975, n - 1))
    jk = [point - t * se, point + t * se]
    floor = FLOORS['temperature'] if variable == 'temperature' else FLOORS['oxygen_mgl']
    out.update(ci=[r6(lo), r6(hi)], interval_status=status(lo, hi), excludes_zero=bool(lo > 0 or hi < 0),
               jackknife_t=[r6(v) for v in jk], jackknife_excludes_zero=bool(jk[0] > 0 or jk[1] < 0),
               jackknife_units=n, floor=floor, floor_reading=floor_reading(point, lo, hi, floor))
    if variable == 'temperature':
        out['floor_reading_colocated'] = floor_reading(point, lo, hi, FLOORS['temperature_colocated'])
    p = FDR.p_side(point, lo, hi)
    out['p'] = None if p is None else float(f'{p:.9g}')
    return out


def stream(part, variable, lead, x, y):
    return f'a86|{part}|{variable}|{lead}|{x} - {y}'


# --------------------------------------------------------------------------------------------------
# Development counterparts
# --------------------------------------------------------------------------------------------------
def dev_a83(code, variable, lead):
    key = f'{code}|{variable}|{lead}'
    e = load(A83)['contrast_cells'][key]
    pf = e.get('per_fold')
    scale = C_MG if variable == 'oxygen' else 1.0
    return {'artifact': A83, 'key': f"contrast_cells['{key}'].per_fold", 'contrast': e['contrast'],
            'design': 'per fold, Dev-2025', 'stored_unit_value': pf,
            'value': None if pf is None else [r6(v * scale) for v in pf],
            'unit': 'degC' if variable == 'temperature' else 'mg/l',
            'status': None if pf is None else status(pf[1], pf[2]),
            'verdict_printed': e.get('printed'), 'per_fold_source': e.get('per_fold_source', e.get('source'))}


def dev_neural(variable, name, prefix='seed'):
    key = f'{prefix}|{variable}|6|{name}'
    v = load(NEURAL)['contrasts'][key]['values']['perfold']
    scale = C_MG if variable == 'oxygen' else 1.0
    return {'artifact': NEURAL, 'key': f"contrasts['{key}'].values.perfold", 'design': 'per fold, Dev-2025',
            'stored_unit_value': v, 'value': [r6(x * scale) for x in v],
            'unit': 'degC' if variable == 'temperature' else 'mg/l', 'status': status(v[1], v[2])}


def dev_aou_rest(name):
    e = next(c for c in load(AOU_REST)['aou']['contrasts'] if c['contrast'] == name)
    v = e['perfold']['value']
    return {'artifact': AOU_REST, 'key': f"aou.contrasts[contrast == '{name}'].perfold.value",
            'design': 'per fold, Dev-2025', 'value': [r6(x) for x in v], 'unit': 'mg/l', 'status': status(v[1], v[2])}


# --------------------------------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------------------------------
def gate_g1():
    g = json.loads((G.GATES / 'gates_B.json').read_text(encoding='utf-8'))
    return g, {k: v['passed'] for k, v in g['A86-G1_part_B_trees']['cells'].items()}


def b_tree(variable, arm, lead, g1):
    if not g1.get(f'{variable}|{arm}|{lead}'):
        raise SystemExit(f'gate A86-G1 did not pass for {variable} {arm} {lead}')
    d, _ = CR.predictions(variable, arm, lead)
    return frame_of(d, arm)


def part_a1(g1):
    out = {}
    for variable in ('temperature', 'oxygen'):
        for lead in G.B_LEADS:
            c = contrast('weather_tree', 'tree', b_tree(variable, 'weather_tree', lead, g1),
                         b_tree(variable, 'tree', lead, g1), variable,
                         stream('A1', variable, lead, 'weather_tree', 'tree'))
            c['lead'] = lead
            c['development'] = dev_a83('P1', variable, lead)
            out[f'{variable}|{lead}|weather_tree - tree'] = c
            print('A1', variable, lead, c['difference'], c.get('ci'), c['interval_status'], flush=True)
    return out


def part_a2(g1):
    """Gate A86-G2, then the part-G arms against part B's six-hour oxygen trees, in mg/l."""
    z = np.load(G_FROZEN, allow_pickle=True)
    keys = pd.DataFrame({'sensor': z['sensor'].astype(str),
                         'time': utc(z['time'].astype(np.int64))})
    wt = b_tree('oxygen', 'weather_tree', 6, g1)
    pt = b_tree('oxygen', 'tree', 6, g1)
    truth = wt[['sensor', 'time', 'truth|weather_tree']].rename(columns={'truth|weather_tree': 'truth'})
    arms = ('aou_tree_2026', 'aou_tree_2026_w1', 'aou_tree_2026_fold1', 'aou_tree_2026_fold1_w1', 'thermodynamic_2026')
    gframe = keys.assign(**{a: z[a].astype(float) for a in arms})
    j = gframe.merge(truth, on=['sensor', 'time'], how='left', validate='one_to_one')
    j['truth_mgl'] = j.truth * C_MG
    published = load(G_REPORT)['reserved_2026']['mae_mgl']
    gate = {'rows_part_G': int(len(gframe)), 'rows_joined': int(j.truth.notna().sum()), 'cells': {}}
    for a in ('thermodynamic_2026', 'aou_tree_2026', 'aou_tree_2026_w1', 'aou_tree_2026_fold1'):
        s = j.dropna(subset=['truth', a])
        f = pd.DataFrame({'fold': np.where(s.time.dt.month <= 3, 'Q1', 'Q2'), 'sensor': s.sensor,
                          'e': np.abs(s.truth_mgl - s[a])})
        mae = R.macro(f, 'e')
        gate['cells'][a] = {'mae_here': mae, 'published': published[a], 'abs_diff': abs(mae - published[a]),
                            'passed': bool(abs(mae - published[a]) <= 1e-6)}
    gate['passed'] = bool(gate['rows_joined'] == gate['rows_part_G']
                          and all(c['passed'] for c in gate['cells'].values()))
    print('G2', gate, flush=True)
    out = {'A86-G2_part_G_rows': gate, 'contrasts': {}}
    if not gate['passed']:
        out['stopped'] = 'gate A86-G2 failed: part A2 is not read'
        return out
    # everything in mg/l: part G is stored in mg/l; part B's trees and the truth are scaled by c
    def mgl(f, name):
        g = f.copy()
        g[name] = g[name] * C_MG
        g[f'truth|{name}'] = g[f'truth|{name}'] * C_MG
        return g
    wt_m, pt_m = mgl(wt, 'weather_tree'), mgl(pt, 'tree')
    for a in ('aou_tree_2026_fold1_w1', 'aou_tree_2026_w1', 'thermodynamic_2026'):
        fa = j[['sensor', 'time', 'truth_mgl', a]].rename(columns={'truth_mgl': f'truth|{a}'})
        for ref, fr in (('weather_tree', wt_m), ('tree', pt_m)):
            c = contrast(a, ref, fa, fr, 'oxygen', stream('A2', 'oxygen', 6, a, ref), scale=1.0)
            c['lead'] = 6
            c['role'] = 'primary' if a != 'aou_tree_2026_w1' else 'secondary (shared configuration)'
            dev_name = ('aou_tree' if a.startswith('aou') else 'thermodynamic') + \
                       (' - weather_tree' if ref == 'weather_tree' else ' - plain_tree')
            if ref == 'tree' and a == 'thermodynamic_2026':
                c['development'] = None
            else:
                c['development'] = dev_aou_rest(dev_name)
            out['contrasts'][f'oxygen|6|{a} - {ref}'] = c
            print('A2', a, ref, c['difference'], c.get('ci'), c['interval_status'], flush=True)
    return out


def part_b(g1):
    out = {}
    for variable, arm, lead in G.CELLS:
        d, rec = G.predictions(variable, arm, lead)
        fa = frame_of(d, arm)
        base = G.BASE_OF[arm]
        refs = [base] + ([] if base == 'tree' else ['tree'])
        code_base, code_plain = A83_CODES[arm]
        for ref in refs:
            c = contrast(arm, ref, fa, b_tree(variable, ref, lead, g1), variable,
                         stream('B', variable, lead, arm, ref))
            c['lead'] = lead
            c['role'] = 'against the arm it augments' if ref == base else 'against the plain tree'
            c['study'] = rec
            c['development'] = dev_a83(code_base if ref == base else code_plain, variable, lead)
            out[f'{variable}|{lead}|{arm} - {ref}'] = c
            print('B', variable, lead, arm, ref, c['difference'], c.get('ci'), c['interval_status'], flush=True)
    return out


def part_c(g1):
    path = OUT / 'C' / 'graphs_2026.json'
    if not path.exists():
        return {'not_run': 'part C1 has no refits'}
    meta = json.loads(path.read_text(encoding='utf-8'))
    out = {'host_dependent': meta['host_dependent'], 'device': meta['device'], 'contrasts': {}, 'per_seed': {}}
    for variable in ('temperature', 'oxygen'):
        tree = b_tree(variable, 'tree', 6, g1)
        per_seed, mean = {}, {}
        for arm in G.GRAPH_ARMS:
            e = meta['arms'][f'{variable}|{arm}']
            if sha(ROOT / e['predictions']) != e['sha256']:
                raise SystemExit(f'{e["predictions"]}: digest differs from its record')
            d = pd.read_csv(ROOT / e['predictions'], float_precision='round_trip')
            d['time'] = pd.to_datetime(d.time, utc=True)
            per_seed[arm] = {s: frame_of(g, arm) for s, g in d.groupby('seed')}
            m = d.groupby(['sensor', 'time'], as_index=False).agg(truth=('truth', 'first'),
                                                                  prediction=('prediction', 'mean'),
                                                                  seeds=('seed', 'nunique'))
            if (m.seeds != len(G.SEEDS)).any():
                raise SystemExit(f'{variable} {arm}: a row lacks a seed')
            mean[arm] = frame_of(m, arm)
        pairs = (('learned_source', 'tree'), ('conservative_exchange', 'learned_source'),
                 ('conservative_exchange', 'tree'))
        for x, y in pairs:
            fx = mean[x]
            fy = tree if y == 'tree' else mean[y]
            c = contrast(x, y, fx, fy, variable, stream('C', variable, 6, x, y))
            c['lead'] = 6
            c['predictions'] = 'mean over seeds 42, 7 and 123'
            c['host_dependent'] = meta['host_dependent']
            c['development'] = dev_neural(variable, f'{x} - {y}')
            out['contrasts'][f'{variable}|6|{x} - {y}'] = c
            print('C', variable, x, y, c['difference'], c.get('ci'), c['interval_status'], flush=True)
            for s in G.SEEDS:
                fx_s = per_seed[x][s]
                fy_s = tree if y == 'tree' else per_seed[y][s]
                cs = contrast(x, y, fx_s, fy_s, variable, stream('C', variable, 6, x, y) + f'|seed{s}')
                out['per_seed'][f'{variable}|6|{x} - {y}|seed{s}'] = {
                    k: cs.get(k) for k in ('difference', 'ci', 'interval_status', 'mae_x', 'mae_y', 'paired_rows')}
    sat = OUT / 'C' / 'saturation_estimate.json'
    if sat.exists():
        out['saturation_estimate'] = json.loads(sat.read_text(encoding='utf-8'))
    sat_path = OUT / 'C' / 'graphs_2026_saturation.json'
    if not sat_path.exists():
        out['saturation'] = {'not_run': 'part C2 has no study here'}
        return out
    srep = json.loads(sat_path.read_text(encoding='utf-8'))
    if sha(ROOT / srep['predictions']) != srep['sha256']:
        raise SystemExit(f'{srep["predictions"]}: digest differs from its record')
    sel = srep['selection']
    if sel.get('smoke_only') or int(sel['completed_trials']) < G.MIN_TRIALS or not srep['inner_window']['no_2026_row']:
        raise SystemExit('part C2: not a protocol-v2 selection on the 2026 inner window')
    d = pd.read_csv(ROOT / srep['predictions'], float_precision='round_trip')
    d['time'] = pd.to_datetime(d.time, utc=True)
    m = d.groupby(['sensor', 'time'], as_index=False).agg(truth=('truth', 'first'), prediction=('prediction', 'mean'),
                                                          seeds=('seed', 'nunique'))
    if (m.seeds != len(G.SEEDS)).any():
        raise SystemExit('oxygen saturation_exchange: a row lacks a seed')
    sat_mean = frame_of(m, 'saturation_exchange')
    cons = pd.read_csv(ROOT / meta['arms']['oxygen|conservative_exchange']['predictions'], float_precision='round_trip')
    cons['time'] = pd.to_datetime(cons.time, utc=True)
    cons_mean = frame_of(cons.groupby(['sensor', 'time'], as_index=False).agg(truth=('truth', 'first'),
                                                                                prediction=('prediction', 'mean')),
                         'conservative_exchange')
    tree = b_tree('oxygen', 'tree', 6, g1)
    for x, y, fy in (('saturation_exchange', 'conservative_exchange', cons_mean), ('saturation_exchange', 'tree', tree)):
        c = contrast(x, y, sat_mean, fy, 'oxygen', stream('C', 'oxygen', 6, x, y))
        c['lead'] = 6
        c['predictions'] = 'mean over seeds 42, 7 and 123'
        c['host_dependent'] = meta['host_dependent']
        c['study'] = {k: sel[k] for k in ('study', 'completed_trials', 'failed_trials', 'best_value', 'params')}
        try:
            c['development'] = dev_neural('oxygen', f'{x} - {y}', prefix='horizon')
        except KeyError:
            c['development'] = None
        out['contrasts'][f'oxygen|6|{x} - {y}'] = c
        print('C2', x, y, c['difference'], c.get('ci'), c['interval_status'], flush=True)
    out['saturation'] = {'study': srep['selection_record'], 'seconds': srep['seconds']}
    return out


def part_d():
    path = OUT / 'D' / 'rollouts_2026.json'
    if not path.exists():
        return {'not_run': 'part D has no rollouts'}
    meta = json.loads(path.read_text(encoding='utf-8'))
    dev = load(RANGE_DEV)
    out = {'gate_G6': meta['gate_G6'], 'statistic': 'mean absolute error over the scored rows of a lead, range '
                                                    'bound minus no gate (addendum 5)',
           'unit': 'onset ISO week drawn jointly within each quarter of onset', 'draws': 600, 'results': {}}
    for key, run in meta['runs'].items():
        variable, regime = key.split('|')
        if sha(ROOT / run['predictions']) != run['sha256']:
            raise SystemExit(f'{run["predictions"]}: digest differs from its record')
        frame = pd.read_csv(ROOT / run['predictions'], float_precision='round_trip')
        wide = (frame[frame.gate.isin(['none', 'range'])]
                .pivot_table(index=['sensor', 'time', 'lead'], columns='gate', values='error').reset_index())
        if not wide[['none', 'range']].notna().all(axis=None):
            raise SystemExit(f'{key}: unpaired rows')
        t = pd.to_datetime(wide.time, utc=True)
        iso = t.dt.isocalendar()
        wide['fold'] = np.where(t.dt.month <= 3, 'Q1', 'Q2')
        wide['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
        wide['d'] = wide.range.abs() - wide.none.abs()
        scale = C_MG if variable == 'oxygen' else 1.0
        entry = {'firings_range': int(run['fire_counts'].get('range', 0)),
                 'evaluable_steps_range': int(run['evaluable_steps']['range']['range']),
                 'fire_counts': run['fire_counts'], 'fire_by_type_all': run['fire_by_type'],
                 'evaluable_steps': run['evaluable_steps'], 'by_lead': {}}
        for lead, sub in wide.groupby('lead'):
            rng = np.random.default_rng([20261001, zlib.crc32(f'a86|D|{variable}|{regime}|{lead}'.encode())])
            tables = []
            for _, f in sub.groupby('fold'):
                g = f.groupby('week').d.agg(['sum', 'size'])
                tables.append(g[['sum', 'size']].to_numpy(float))
            draws = np.empty(600)
            for i in range(600):
                tot = np.zeros(2)
                for table in tables:
                    tot += table[rng.integers(0, len(table), len(table))].sum(axis=0)
                draws[i] = tot[0] / tot[1]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            none_mae = float(sub.none.abs().mean())
            range_mae = float(sub.range.abs().mean())
            point = range_mae - none_mae
            weeks = sub[['fold', 'week']].drop_duplicates()
            jack = np.array([sub[~((sub.fold == f) & (sub.week == w))].d.mean() for f, w in weeks.itertuples(index=False)])
            n = len(jack)
            se = float(np.sqrt((n - 1) / n * np.sum((jack - jack.mean()) ** 2))) if n > 1 else float('nan')
            tq = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float('nan')
            e = {'rows': int(len(sub)), 'weeks': int(n), 'rows_changed': int((sub.range != sub.none).sum()),
                 'mae_none': r6(none_mae * scale), 'mae_range': r6(range_mae * scale),
                 'difference': r6(point * scale),
                 'change_percent_of_none': r6(100 * point / none_mae) if none_mae else None,
                 'unit': 'degC' if variable == 'temperature' else 'mg/l'}
            if n >= MIN_UNITS:
                e.update(ci=[r6(lo * scale), r6(hi * scale)], interval_status=status(lo, hi),
                         excludes_zero=bool(lo > 0 or hi < 0),
                         jackknife_t=[r6((point - tq * se) * scale), r6((point + tq * se) * scale)])
                p = FDR.p_side(point, lo, hi) if (lo, hi) != (0.0, 0.0) else None
                e['p'] = None if p is None else float(f'{p:.9g}')
                floor = FLOORS['temperature'] if variable == 'temperature' else FLOORS['oxygen_mgl']
                e['floor'] = floor
                e['floor_reading'] = floor_reading(point * scale, lo * scale, hi * scale, floor)
            else:
                e.update(ci=None, interval_status='n.e.', p=None)
            dv = dev['results'][f'{variable}|{regime}']['by_lead'].get(str(int(lead)))
            e['development'] = None if dv is None else {
                'artifact': RANGE_DEV, 'key': f"results['{variable}|{regime}'].by_lead['{int(lead)}']",
                'design': 'per fold, Dev-2025', 'value': [r6(dv['point'] * scale), r6(dv['ci'][0] * scale),
                                                          r6(dv['ci'][1] * scale)],
                'status': status(*dv['ci']) if dv['ci'] != [0.0, 0.0] else 'not distinguishable',
                'unit': e['unit'], 'firings': dev['results'][f'{variable}|{regime}']['firings'],
                'evaluable_steps': dev['results'][f'{variable}|{regime}']['evaluable_steps']}
            entry['by_lead'][str(int(lead))] = e
        out['results'][key] = entry
        six = entry['by_lead']['6']
        print('D', key, 'fires', entry['firings_range'], '/', entry['evaluable_steps_range'], '6 h', six['difference'],
              six.get('ci'), flush=True)
    return out


def part_e1():
    path = OUT / 'E' / 'E1' / 'report.json'
    if not path.exists():
        return {'not_run': 'part E1 has no fits'}
    import scripts.v2_summary_table as S
    rep = json.loads(path.read_text(encoding='utf-8'))
    c3 = load(C3_UNITS)
    floor = float(S.floors()['cast_O_all'])
    out = {'A86-G7_mgl_context': rep['A86-G7_mgl_context'], 'split': rep['split'], 'cohorts': rep['cohorts'],
           'mae': rep['mae'], 'contrasts': {}}
    for e in rep['contrasts']:
        twin = next((c for c in c3['contrasts'] if c['cohort'] == e['cohort'] and c['a'] == e['a'] and c['b'] == e['b']),
                    None)
        key = f"{e['cohort']}|{e['a']} - {e['b']}"
        lo, hi = e['ci']
        p = FDR.p_side(e['diff'], lo, hi)
        mae_b = e['mae_b']
        out['contrasts'][key] = {
            'cohort': 'casts of 2025 (trained on casts before 2025)', 'cast_cohort': e['cohort'], 'x': e['a'],
            'y': e['b'], 'unit': 'mg/l', 'mae_x': e['mae_a'], 'mae_y': mae_b, 'difference': e['diff'],
            'change_percent_of_y': r6(100 * e['diff'] / mae_b), 'ci': e['ci'], 'interval_status': status(lo, hi),
            'excludes_zero': e['excludes_zero'], 'jackknife_t': e['jackknife_t'],
            'jackknife_excludes_zero': e['jackknife_excludes_zero'], 'units_cast_dates': e['dates'],
            'floor': floor, 'floor_reading': floor_reading(e['diff'], lo, hi, floor),
            'p': None if p is None else float(f'{p:.9g}'),
            'test_counterpart': None if twin is None else {
                'artifact': C3_UNITS, 'key': f"contrasts[cohort == '{e['cohort']}', a == '{e['a']}', b == '{e['b']}']",
                'cohort': 'Field-2026 casts (trained on casts before 2026)', 'value': [twin['diff'], *twin['ci']],
                'jackknife_t': twin['jackknife_t'], 'status': status(*twin['ci']), 'unit': 'mg/l'}}
        print('E1', key, e['diff'], e['ci'], flush=True)
    return out


P1_DEV = 'runs/v2_perfold_rest_a69.json'


def part_e2():
    """P1 minus A1_both and minus A2_none on the temperature of the 38 casts of 2026 (held station), by
    cast dates (scripts.v2_summarize_aou.contrast, six decimals), each with its per-fold Field-57 value."""
    path = OUT / 'E' / 'E2' / 'report.json'
    if not path.exists():
        return {'not_run': 'part E2 has no read'}
    import scripts.v2_summarize_aou as SA
    import scripts.v2_summary_table as S
    from src.experiments.v2_field_reserved2026 import cast_errors, macro
    rep = json.loads(path.read_text(encoding='utf-8'))
    out = {'gate_reproduces_addendum38': rep['gate_reproduces_addendum38'], 'penalties': rep['penalties'],
           'not_run': rep['not_run'], 'contrasts': {}}
    if not rep['gate_reproduces_addendum38']['passed']:
        out['stopped'] = 'the read does not reproduce addendum 38: P1 is not read'
        return out
    if sha(ROOT / rep['predictions']) != rep['sha256']:
        raise SystemExit('E2 predictions: digest differs from its record')
    d = pd.read_csv(ROOT / rep['predictions'], float_precision='round_trip')
    d['t_utc'] = pd.to_datetime(d.t_utc, utc=True)
    floor = float(S.floors()['cast_T_gen2'])
    dev = {c['reference']: c for c in load(P1_DEV)['field57']['contrasts']
           if c.get('arm') == 'P1_placebo_direct' and c.get('target') == 'temp_c'}
    SA.round = lambda x, n=None: round(x, None if n is None else max(n, 6))
    try:
        for ref, dev_ref in (('A1_both', 'A1_both_direct'), ('A2_none', 'A2_none_direct')):
            ca, cb = cast_errors(d, 'P1', 'temp_c'), cast_errors(d, ref, 'temp_c')
            c = SA.contrast(ca, cb)
            lo, hi = c['ci']
            mx, my = macro(ca), macro(cb)
            p = FDR.p_side(c['point'], lo, hi)
            v = dev[dev_ref]['perfold']['value']
            out['contrasts'][f'temperature|P1 - {ref}'] = {
                'x': 'P1', 'y': ref, 'variable': 'temperature', 'cohort': 'Field-2026 (held station)', 'unit': 'degC',
                'mae_x': r6(mx), 'mae_y': r6(my), 'difference': r6(c['point']),
                'change_percent_of_y': r6(100 * c['point'] / my), 'ci': [r6(lo), r6(hi)],
                'interval_status': status(lo, hi), 'excludes_zero': c['excludes_zero'],
                'jackknife_t': c['jackknife_t'], 'jackknife_excludes_zero': c['jackknife_excludes_zero'],
                'units_cast_dates': c['dates'], 'floor': floor, 'floor_reading': floor_reading(c['point'], lo, hi, floor),
                'p': None if p is None else float(f'{p:.9g}'),
                'development': {'artifact': P1_DEV,
                                'key': f"field57.contrasts[arm == 'P1_placebo_direct', reference == '{dev_ref}', "
                                       "target == 'temp_c'].perfold.value",
                                'design': 'per fold, Field-57', 'value': [r6(x) for x in v], 'unit': 'degC',
                                'status': status(v[1], v[2])}}
            print('E2 P1 -', ref, out['contrasts'][f'temperature|P1 - {ref}']['difference'], [r6(lo), r6(hi)], flush=True)
    finally:
        del SA.round
    return out


def family(parts):
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            if 'p' in o and ('ci' in o) and o.get('ci') is not None:
                out.append((path, o))
            for k, v in o.items():
                if k not in ('development', 'test_counterpart', 'per_seed'):
                    walk(v, path + (k,))
    walk(parts, ())
    return out


def multiplicity(parts):
    entries = [(p, e) for p, e in family(parts) if e.get('p') is not None]
    ps = [e['p'] for _, e in entries]
    bh = FDR.step_up(ps)
    by = FDR.step_up(ps, dependence_robust=True)
    for (_, e), a, b in zip(entries, bh, by):
        e['bh'], e['by'] = bool(a), bool(b)
    return {'family': 'every new interval of parts A-D, E1 and E2 (per-seed and reproduction gates excluded)',
            'intervals': len(entries), 'exclude_zero': int(sum(bool(e.get('excludes_zero')) for _, e in entries)),
            'bh_rejections': int(sum(bh)), 'by_rejections': int(sum(by)), 'q': FDR.Q}


def main():
    decl = (ROOT / DECLARED).read_bytes()
    g, g1 = gate_g1()
    parts = {'A': {'A1': part_a1(g1), 'A2': part_a2(g1)}, 'B': part_b(g1), 'C': part_c(g1), 'D': part_d(),
             'E': {'E1': part_e1(), 'E2': part_e2()}}
    mult = multiplicity(parts)
    for k, v in parts.items():
        write(OUT / f'{k}.json', {'declared_in': DECLARED, 'part': k, **v})
    summary = {'addendum': 86, 'declared_in': DECLARED, 'declaration_sha256': hashlib.sha256(decl).hexdigest(),
               'producer': 'scripts/v2_guidance2026_report.py', 'fits': 'src/experiments/v2_guidance2026.py',
               'c_mg_per_l_per_umol_kg': C_MG,
               'units': 'temperature degC; oxygen mg/l (sonde values x c; part G and casts already in mg/l)',
               'floors': FLOORS, 'aggregation': 'quarter x sensor macro (Reserved-2026); nested cast macro (casts)',
               'resampling': 'ISO weeks within quarter, 600 joint draws; cast dates, 2000 draws; jackknife beside',
               'gates': {'A86-G1': g['A86-G1_part_B_trees']['passed'],
                         'A86-G2': parts['A']['A2']['A86-G2_part_G_rows']['passed'],
                         'A86-G3': all(v['passed'] for v in g['A86-G3_objective']['cells'].values()),
                         'A86-G4': all(v['passed'] for v in g['A86-G4_design']['cells'].values())},
               'parts': {k: f'runs/v2_addendum86/{k}.json' for k in parts}, 'multiplicity': mult,
               'contrasts': compact(parts)}
    g5 = G.GATES / 'gate_G5_graph_path.json'
    if g5.exists():
        g5 = json.loads(g5.read_text(encoding='utf-8'))
        summary['gates']['A86-G5'] = {'identity': g5['identity_passed'], 'published': g5['published_passed']}
    if 'gate_G6' in parts['D']:
        summary['gates']['A86-G6'] = all(v['passed'] for v in parts['D']['gate_G6'].values())
    if 'A86-G7_mgl_context' in parts['E']['E1']:
        summary['gates']['A86-G7'] = parts['E']['E1']['A86-G7_mgl_context']['passed']
    write(SUMMARY, summary)
    print(json.dumps(mult, indent=1))


def compact(parts):
    """One line per contrast: the 2026 (or development, for E1) value beside its counterpart."""
    rows = []

    def add(part, key, c, counterpart):
        rows.append({'part': part, 'key': key, 'variable': c.get('variable', 'oxygen'), 'lead': c.get('lead'),
                     'x': c.get('x'), 'y': c.get('y'), 'cohort': c.get('cohort'), 'paired_rows': c.get('paired_rows'),
                     'units': c.get('units_quarter_week', c.get('units_cast_dates')), 'mae_x': c.get('mae_x'),
                     'mae_y': c.get('mae_y'), 'difference': c.get('difference'), 'ci': c.get('ci'),
                     'change_percent_of_y': c.get('change_percent_of_y'), 'interval_status': c.get('interval_status'),
                     'jackknife_t': c.get('jackknife_t'), 'bh': c.get('bh'), 'by': c.get('by'),
                     'counterpart': counterpart})
    for k, c in parts['A']['A1'].items():
        add('A1', k, c, c['development'])
    for k, c in parts['A']['A2'].get('contrasts', {}).items():
        add('A2', k, c, c['development'])
    for k, c in parts['B'].items():
        add('B', k, c, c['development'])
    for k, c in parts['C'].get('contrasts', {}).items():
        add('C', k, c, c['development'])
    for k, entry in parts['D'].get('results', {}).items():
        for lead, e in entry['by_lead'].items():
            variable, regime = k.split('|')
            rows.append({'part': 'D', 'key': f'{k}|{lead}|range - none', 'variable': variable, 'lead': int(lead),
                         'x': 'range', 'y': 'none', 'regime': regime, 'cohort': 'Reserved-2026',
                         'paired_rows': e['rows'], 'units': e['weeks'], 'mae_x': e['mae_range'], 'mae_y': e['mae_none'],
                         'difference': e['difference'], 'ci': e.get('ci'),
                         'change_percent_of_y': e['change_percent_of_none'], 'interval_status': e['interval_status'],
                         'jackknife_t': e.get('jackknife_t'), 'bh': e.get('bh'), 'by': e.get('by'),
                         'counterpart': e['development']})
    for k, c in parts['E']['E1'].get('contrasts', {}).items():
        add('E1', k, c, c['test_counterpart'])
    for k, c in parts['E']['E2'].get('contrasts', {}).items():
        add('E2', k, c, c['development'])
    return rows


if __name__ == '__main__':
    main()
