"""Addendum 49: contrasts, gates and decision rules for the graph arms and the recurrent arm.

Reads the three designs written by src/experiments/v2_perfold_neural.py and
src/experiments/v2_sequence_sweep.py (runs/v2_perfold_neural/, runs/v2_sequence_sweep/gru/) and the
published rows, and writes runs/v2_perfold_neural.json.

Every contrast a table quotes is recomputed under each design with that table's own random stream
and block order, so the published value is reproduced from the published rows (gate G3) and the
per-fold value is read on the same resamples; a movement (perfold - published, perfold - shared) is
the same linear combination of macro errors on the same draws. Arm-level design effects have their
own streams prefixed 'a49|'. Neural arms are the mean of their three networks, as published.

Published rows are read from this repository or, where only another checkout holds them, from the
runs folder named by the SMARTBAY_FROZEN_RUNS setting; the path used is recorded. With the setting
unset, a row file this repository lacks stops the run with an error naming the setting.

Usage: python -m scripts.v2_perfold_neural_contrasts [--workers 2]
Declared in config/bay_reserved_protocol_v2_addendum49.json.
"""
import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
#: A second runs folder (another checkout's reports/bay_twin/runs) for published rows this one lacks.
FROZEN = Path(os.environ['SMARTBAY_FROZEN_RUNS']) if os.environ.get('SMARTBAY_FROZEN_RUNS') else None
OUT = RUNS / 'v2_perfold_neural'
REPORT = RUNS / 'v2_perfold_neural.json'
KEY = ['fold', 'sensor', 'time']
DRAWS = 600
FORECAST_RUN = '20260919T144822Z_v2_forecast'
HORIZON_RUN = {1: '20260920T133747Z_v2_horizon', 3: '20260920T133747Z_v2_horizon', 12: '20260920T133747Z_v2_horizon',
               36: '20260920T133747Z_v2_horizon', 6: '20260920T083322Z_v2_horizon', 24: '20260920T083322Z_v2_horizon',
               48: '20260920T083322Z_v2_horizon'}
HORIZONS = (1, 3, 6, 12, 24, 36, 48)
GRAPH_SIX = {'temperature': ('learned_source', 'conservative_exchange', 'coast_exchange',
                             'coast_breakwater_exchange', 'coast_depth_exchange'),
             'oxygen': ('learned_source', 'conservative_exchange', 'coast_exchange',
                        'coast_breakwater_exchange', 'coast_depth_exchange', 'saturation_source', 'saturation_exchange')}
GRAPH_OTHER = {'temperature': ('learned_source', 'conservative_exchange'),
               'oxygen': ('learned_source', 'conservative_exchange', 'saturation_exchange')}
WITHIN = {'temperature': [('conservative_exchange', 'learned_source'), ('coast_exchange', 'conservative_exchange'),
                          ('coast_breakwater_exchange', 'coast_exchange'), ('coast_depth_exchange', 'coast_exchange')],
          'oxygen': [('conservative_exchange', 'learned_source'), ('saturation_source', 'learned_source'),
                     ('saturation_exchange', 'conservative_exchange'), ('coast_exchange', 'conservative_exchange'),
                     ('coast_breakwater_exchange', 'coast_exchange'), ('coast_depth_exchange', 'coast_exchange')]}
HORIZON_CONTRASTS = [('dynamical operator', 'learned_source', 'tree'),
                     ('hard constraint', 'conservative_exchange', 'learned_source'),
                     ('physical prior', 'saturation_exchange', 'conservative_exchange')]
SEED_SPREAD = [('temperature', 6, 'learned_source', 'tree', 0.0038), ('oxygen', 6, 'learned_source', 'tree', 0.194),
               ('temperature', 6, 'conservative_exchange', 'learned_source', 0.0008),
               ('oxygen', 6, 'conservative_exchange', 'learned_source', 0.211),
               ('temperature', 6, 'conservative_exchange', 'tree', 0.0046),
               ('oxygen', 6, 'conservative_exchange', 'tree', 0.018), ('oxygen', 24, 'learned_source', 'tree', 0.026)]
DESIGNS = ('published', 'refit', 'shared', 'perfold')
SCALE = {'temperature': 10.0, 'oxygen': 50.0}
FLOOR = {'temperature': 0.029, 'oxygen': 1.8}


def find(rel):
    for base in (RUNS, FROZEN):
        if base is not None and (base / rel).exists():
            return base / rel
    hint = (f'nor under SMARTBAY_FROZEN_RUNS ({FROZEN})' if FROZEN else
            'and SMARTBAY_FROZEN_RUNS, the runs folder of a checkout that holds it, is not set')
    raise FileNotFoundError(f'{rel} is not under {RUNS}, {hint}')


def week(d):
    return pd.to_datetime(d.time, utc=True).dt.isocalendar().week.astype(int)


def macro(d, col):
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean())


def seed_mean(d):
    if 'mode' in d:
        d = d[d['mode'] == 'forecast']
    if 'seed' in d:
        d = d.groupby(KEY, as_index=False).agg(truth=('truth', 'first'), persistence=('persistence', 'first'),
                                               prediction=('prediction', 'mean'))
    return d[KEY + ['truth', 'persistence', 'prediction']]


def published_path(variable, arm, h, table):
    """The published rows a table reads: Table physics_comparisons reads v2_forecast at 6 h (the tree
    too), every other table the horizon run."""
    if h == 6 and table == 'forecast':
        return f'{FORECAST_RUN}/results/v2_forecast_{variable}_{arm}_nested/selected_predictions.csv.gz'
    return f'{HORIZON_RUN[h]}/results/v2_horizon_{variable}_{arm}_h{h}/selected_predictions.csv.gz'


def design_path(kind, design, variable, arm, h):
    tag = f'{variable}_{arm}_h{h}' if kind == 'graph' else f'{variable}_tree_h{h}'
    return OUT / kind / design / tag / 'selected_predictions.csv.gz'


def arm_frame(variable, arm, h, table, used):
    """KEY, truth and one column per design for one arm, in the row order of the published file."""
    rel = published_path(variable, arm, h, table)
    path = find(rel)
    used[rel] = str(path)
    pub = seed_mean(pd.read_csv(path)).rename(columns={'prediction': f'{arm}@published'})
    kind = 'tree' if arm == 'tree' else 'graph'
    for design in ('refit', 'shared', 'perfold'):
        p = design_path(kind, 'published' if design == 'refit' else design, variable, arm, h)
        if design == 'shared' and not p.exists() and kind == 'graph' and \
                design_path(kind, 'perfold', variable, arm, h).exists():
            # Addendum 49 amendment 1: on the compute host every shared re-run reproduced its
            # published study trial for trial (runs/v2_perfold_neural/shared_vs_published.json), so a
            # cell started after the amendment takes the published rows as its shared column.
            pub[f'{arm}@shared'] = pub[f'{arm}@published']
            used['shared'] = 'published rows (addendum 49 amendment 1)'
            continue
        if not p.exists():
            return None
        d = seed_mean(pd.read_csv(p))[KEY + ['truth', 'prediction']].rename(columns={'prediction': f'{arm}@{design}',
                                                                                   'truth': 'truth_d'})
        pub = pub.merge(d, on=KEY, how='left', validate='one_to_one')
        gap = float(np.abs(pub.truth - pub.truth_d).max())
        assert pub[f'{arm}@{design}'].notna().all() and gap < 1e-9, (variable, arm, h, design, gap)
        pub = pub.drop(columns='truth_d')
    return pub


def boot(frame, block_order, combos, seed, draws=DRAWS):
    """Percentile intervals of linear combinations of macro errors on shared (fold, ISO week) draws.

    block_order is a frame whose first-appearance order of (fold, week) is the published script's
    block list, or 'sorted' for a script that iterated a groupby's sorted keys. The draws are the
    published scripts' own (scripts/v2_block_bootstrap.py), from per-block sums."""
    from scripts import v2_block_bootstrap as bb
    blocks = bb.block_list(frame, 'sorted') if isinstance(block_order, str) else bb.block_list(block_order)
    cols = sorted({c for combo in combos.values() for _, c in combo})
    point, samples = bb.draws(frame, cols, blocks, np.random.default_rng(seed), draws)
    return {n: bb.interval(point, samples, combo) for n, combo in combos.items()}


def contrast_combos(a, b, designs=('published', 'refit', 'shared', 'perfold')):
    c = {d: [(1, f'{a}@{d}'), (-1, f'{b}@{d}')] for d in designs}
    c['perfold - published'] = c['perfold'] + [(-k, x) for k, x in c['published']]
    c['perfold - shared'] = c['perfold'] + [(-k, x) for k, x in c['shared']]
    c['shared - published'] = c['shared'] + [(-k, x) for k, x in c['published']]
    return c


def excludes(v):
    return bool(v[1] > 0 or v[2] < 0)


# ---------------------------------------------------------------- jobs (each returns a dict)

def job_contrast(spec):
    """One quoted contrast under every design, with its table's stream and block order."""
    table, variable, h, a, b, label, seed, published = spec
    used = {}
    fa = arm_frame(variable, a, h, table, used)
    fb = arm_frame(variable, b, h, table, used)
    if fa is None or fb is None:
        return {'label': label, 'missing': True}
    if table == 'forecast':               # v2_summarize_forecast (b is the tree) or v2_physics_contrasts
        base, other = (fb, fa) if b == 'tree' else (fa, fb)
        order = fb if b == 'tree' else 'sorted'
    elif table == 'horizon':             # v2_horizon_table: joined in the order of the reference arm
        base, other, order = fb, fa, fb
    else:                                # v2_seed_contrasts: the published tree file's order
        base, other = fb, fa
        order = seed_mean(pd.read_csv(find(published_path(variable, 'tree', h, 'horizon'))))
        used['blocks'] = published_path(variable, 'tree', h, 'horizon')
    j = base.merge(other.drop(columns=['truth', 'persistence']), on=KEY, validate='one_to_one')
    res = boot(j, order, contrast_combos(a, b), seed)
    out = {'label': label, 'table': table, 'variable': variable, 'horizon': h, 'contrast': f'{a} - {b}',
           'values': res, 'published_value': published, 'sources': used}
    if published is not None:
        pv = res['published']
        rounded = table == 'seed'
        out['gate_G3_abs_diff'] = max(abs((round(x, 4) if rounded else x) - y) for x, y in zip(pv, published))
        out['gate_G3_pass'] = bool(out['gate_G3_abs_diff'] <= (0 if rounded else 1e-12))
    return out


def job_arm(spec):
    """Arm-level design effects for one cell, on its own stream."""
    kind, variable, arm, h = spec
    used = {}
    # At 6 h the published graph rows are v2_forecast's (the horizon run holds only three of the seven arms).
    f = arm_frame(variable, arm, h, 'forecast' if (kind == 'graph' and h == 6) else 'horizon', used)
    if f is None:
        return {'cell': f'{variable}|{arm}|h{h}', 'missing': True}
    combos = {d: [(1, f'{arm}@{d}')] for d in DESIGNS}
    combos['persistence'] = [(1, 'persistence')]
    combos.update({f'{x} - {y}': [(1, f'{arm}@{x}'), (-1, f'{arm}@{y}')]
                   for x, y in (('perfold', 'shared'), ('perfold', 'published'), ('shared', 'published'),
                                ('refit', 'published'))})
    label = f'a49|{variable}|{h}|{arm}'
    res = boot(f, f, combos, [20260920, zlib.crc32(label.encode())])
    per_fold = {}
    for k, g in f.groupby('fold'):
        per_fold[str(k)] = {d: macro(g, f'{arm}@{d}') for d in DESIGNS}
        per_fold[str(k)]['persistence'] = macro(g, 'persistence')
    reports = {}
    for d in ('published', 'shared', 'perfold'):
        rp = design_path(kind, d, variable, arm, h).parent / 'report.json'
        if not rp.exists():
            reports[d] = {'source': used.get(d, 'missing')}
            continue
        r = json.loads(rp.read_text())
        reports[d] = {k: r.get(k) for k in ('gate_G1_pass', 'gate_G1_max_abs_prediction_diff', 'gate_G2_pass',
                                              'gate_G2_abs_diff', 'gate_G6_pass', 'gate_G6_max_abs_prediction_diff',
                                              'params_by_fold', 'seconds')}
        sel = r.get('selection')
        if isinstance(sel, dict) and 'completed_trials' in sel:
            reports[d]['completed_trials'] = [sel['completed_trials']]
        elif isinstance(sel, dict):
            reports[d]['completed_trials'] = [v['completed_trials'] for v in sel.values()]
    return {'cell': f'{variable}|{arm}|h{h}', 'kind': kind, 'variable': variable, 'arm': arm, 'horizon': h,
            'values': res, 'by_fold': per_fold, 'endpoints': int(len(f)), 'reports': reports, 'sources': used}


def specs():
    fs = json.loads((RUNS / FORECAST_RUN / 'v2_forecast_summary.json').read_text())
    pc = json.loads((RUNS / FORECAST_RUN / 'v2_physics_contrasts.json').read_text())
    hs = json.loads((RUNS / 'v2_horizon_summary.json').read_text())
    sc = json.loads((RUNS / 'v2_seed_contrasts.json').read_text())['contrasts']
    out = []
    for v in ('temperature', 'oxygen'):
        for a in GRAPH_SIX[v]:
            out.append(('forecast', v, 6, a, 'tree', f'forecast|{v}|{a} - tree', [20260919, zlib.crc32(f'{v}|{a}'.encode())],
                        fs[v][a]['minus_tree']))
        for a, b in WITHIN[v]:
            out.append(('forecast', v, 6, a, b, f'forecast|{v}|{a} - {b}', [20260919, zlib.crc32(f'{v}|{a}|{b}'.encode())],
                        pc[f'{v}: {a} - {b}']))
        for h in HORIZONS:
            for _, a, b in HORIZON_CONTRASTS:
                if a == 'saturation_exchange' and v != 'oxygen':
                    continue
                out.append(('horizon', v, h, a, b, f'horizon|{v}|{h}|{a} - {b}',
                            [20260920, zlib.crc32(f'{v}|{h}|{a}-{b}'.encode())],
                            hs[v][str(h)]['contrasts'][f'{a} - {b}']['value']))
    for v, h, a, b, _ in SEED_SPREAD:
        key = f'{v}|{h}|{a} - {b}'
        e = sc[key]['by_seed']['42']
        out.append(('seed', v, h, a, b, f'seed|{key}', [20260923, zlib.crc32(f'{key}|42'.encode())],
                    [e['point'], *e['ci']]))
    return out


def arm_specs():
    out = [('graph', v, a, 6) for v in ('temperature', 'oxygen') for a in GRAPH_SIX[v]]
    out += [('graph', v, a, h) for h in HORIZONS if h != 6 for v in ('temperature', 'oxygen') for a in GRAPH_OTHER[v]]
    out += [('tree', v, 'tree', h) for h in HORIZONS for v in ('temperature', 'oxygen')]
    return out


# ---------------------------------------------------------------- the recurrent arm

#: the sequence-arm rows the recurrent block reads; --recurrent-only switches to the rows with the
#: tree selected per outer fold (addendum 69)
SEQ_DIR = RUNS / 'v2_sequence_arms'
#: addendum 69: this report with the recurrent block re-read on those rows, every other block as stored
REPORT_A69 = RUNS / 'v2_perfold_neural_a69.json'


def recurrent():
    from scripts.v2_sequence_table import macro as smacro
    seq = SEQ_DIR
    horizon = RUNS / '20260920T083322Z_v2_horizon/results/v2_horizon_{variable}_tree_h6/selected_predictions.csv.gz'
    center = {'temperature': 20.0, 'oxygen': 230.0}
    table = json.loads((seq / 'sequence_arms_table.json').read_text())
    base_pub = json.loads((seq / 'sequence_baselines.json').read_text())
    out = {}
    for variable in ('temperature', 'oxygen'):
        scale = SCALE[variable]
        f = pd.read_csv(seq / f'sequence_arms_{variable}.csv.gz').dropna(subset=['tree', 'lstm', 'chronos_zs', 'timesfm_zs'])
        base = pd.read_csv(horizon.as_posix().format(variable=variable))[KEY + ['persistence', 'clim_hour']]
        f = f.merge(base, on=KEY, how='left', validate='one_to_one')
        for c in ('persistence', 'clim_hour'):
            f[c] = (f[c] - center[variable]) / scale
        designs = {}
        for design in ('published', 'random_split', 'calendar'):
            folder = RUNS / 'v2_sequence_sweep/gru' / design / f'{variable}_offset0_seed42'
            if not (folder / 'report.json').exists():
                continue
            d = pd.read_csv(folder / 'predictions.csv.gz')[KEY + ['lstm']].rename(columns={'lstm': f'lstm@{design}'})
            f = f.merge(d, on=KEY, how='left', validate='one_to_one')
            assert f[f'lstm@{design}'].notna().all(), (variable, design)
            r = json.loads((folder / 'report.json').read_text())
            designs[design] = {'epochs': r['epochs'], 'params': {k: v['selection']['params'] for k, v in r['folds'].items()},
                               'completed_trials': {k: v['selection']['completed_trials'] for k, v in r['folds'].items()},
                               'gate_G4_pass': r.get('gate_G4_pass'), 'gate_G4_max_abs_diff': r.get('gate_G4_max_abs_diff'),
                               'seconds': r['seconds']}
        f = f.rename(columns={'lstm': 'lstm@published_rows'})
        cols = ['lstm@published_rows'] + [f'lstm@{d}' for d in designs]
        entry = {'endpoints': int(len(f)), 'designs': designs,
                 'mae': {c: round(smacro(f, c, 'fold') * scale, 4) for c in cols + ['tree', 'chronos_zs', 'timesfm_zs']},
                 'contrasts': {}}
        pub_c = table['variables'][variable]['rules']['fold (primary; matches the other development contrasts)']['contrasts']
        for other, name in (('tree', 'tree - lstm'), ('chronos_zs', 'lstm - chronos_zs'), ('timesfm_zs', 'lstm - timesfm_zs'),
                            ('clim_hour', 'lstm - clim_hour')):
            label = f'{variable}|fold|{"tree|lstm" if other == "tree" else "lstm|" + other}'
            combos = {}
            for c in cols:
                a_, b_ = ('tree', c) if other == 'tree' else (c, other)
                combos[c] = [(1, a_), (-1, b_)]
            ref = 'lstm@published_rows'
            for c in cols[1:]:
                sign = -1 if other == 'tree' else 1
                combos[f'{c} - published'] = [(sign, c), (-sign, ref)]
            # v2_sequence_table.paired's draws: the frame's own block order, one stream per label
            res = boot(f, f, combos, [20260921, zlib.crc32(label.encode())])
            by = {c: [round(x * scale, 4) for x in res[c]] for c in cols}
            mv = {k: v for k, v in res.items() if k.endswith(' - published')}
            published = (pub_c.get(name) if other != 'clim_hour' else
                         base_pub['variables'][variable]['contrasts'].get('lstm - clim_hour'))
            entry['contrasts'][name] = {'by_design': by, 'movement_scaled': {k: [x * scale for x in v] for k, v in mv.items()},
                                        'published': published,
                                        'gate_G3_pass': bool(published is None or by['lstm@published_rows'] == published)}
        out[variable] = entry
    return out


# ---------------------------------------------------------------- decisions

def decide(contrasts, arms, rec):
    by_label = {c['label']: c for c in contrasts if not c.get('missing')}
    rules = {'D3': {}, 'D2_unswept_changes': [], 'D4': 'the calendar-selected recurrent arm goes into Table bay_sequence_arms'}
    for v, h, a, b, rng_ in SEED_SPREAD:
        c = by_label.get(f'seed|{v}|{h}|{a} - {b}')
        if c is None:
            continue
        eff = abs(c['values']['perfold - shared'][0])
        rules['D3'][f'{v}|{h}|{a} - {b}'] = {'design_effect': eff, 'seed_range': rng_, 'triggered': bool(eff > rng_),
                                             'perfold': c['values']['perfold'], 'published': c['values']['published']}
    for c in by_label.values():
        if c['table'] == 'seed':
            continue
        p, q = c['values']['published'], c['values']['perfold']
        if np.sign(p[0]) != np.sign(q[0]) or excludes(p) != excludes(q):
            rules['D2_unswept_changes'].append({'label': c['label'], 'published': p, 'perfold': q})
    preds = {}
    t6 = [by_label.get(f'forecast|temperature|{a} - tree') for a in GRAPH_SIX['temperature']]
    if all(t6):
        preds['P1'] = all(c['values']['perfold'][0] > 0 and excludes(c['values']['perfold']) for c in t6)
    moves = [(a['variable'], abs(a['values']['perfold - published'][0])) for a in arms if not a.get('missing') and a['kind'] == 'graph']
    if moves:
        preds['P2'] = all(m < FLOOR[v] for v, m in moves)
    if rules['D3']:
        preds['P3'] = not any(r['triggered'] for r in rules['D3'].values())
    o24 = by_label.get('horizon|oxygen|24|learned_source - tree')
    if o24:
        preds['P4'] = bool(o24['values']['perfold'][0] < 0 and excludes(o24['values']['perfold']))
    g = [a for a in arms if not a.get('missing') and a['kind'] == 'graph']
    if g:
        opt = [a['by_fold']['0']['perfold'] / a['by_fold']['0']['persistence'] - a['by_fold']['0']['shared'] / a['by_fold']['0']['persistence']
               for a in g]
        preds['P5'] = {'held': bool(np.mean(opt) > 0), 'mean_fold0_ratio_perfold_minus_shared': float(np.mean(opt)), 'cells': len(g)}
    if rec:
        ep = []
        for v, e in rec.items():
            if 'calendar' in e['designs'] and 'random_split' in e['designs']:
                for k in e['designs']['calendar']['epochs']:
                    ep.append(e['designs']['calendar']['epochs'][k] < e['designs']['random_split']['epochs'][k])
        if ep:
            preds['P6'] = {'held': sum(ep) >= 3, 'fewer_epochs_in': int(sum(ep)), 'studies': len(ep)}
        ok = []
        for v, e in rec.items():
            if 'lstm@calendar' in e['mae']:
                tol = {'temperature': 0.005, 'oxygen': 0.3}[v]
                d = abs(e['mae']['lstm@calendar'] - e['mae']['lstm@published_rows'])
                s_old = np.sign(e['contrasts']['tree - lstm']['by_design']['lstm@published_rows'][0])
                s_new = np.sign(e['contrasts']['tree - lstm']['by_design']['lstm@calendar'][0])
                ok.append(bool(d <= tol and s_old == s_new))
        if ok:
            preds['P7'] = all(ok)
    return rules, preds


def recurrent_only():
    """Addendum 69: the recurrent block re-read on the sequence-arm rows whose tree is selected per
    outer fold (runs/v2_sequence_arms/perfold_tree/), the decision rules recomputed from the stored
    graph contrasts and arms, which do not read those rows; written to REPORT_A69, REPORT untouched."""
    global SEQ_DIR
    SEQ_DIR = RUNS / 'v2_sequence_arms' / 'perfold_tree'
    stored = json.loads(REPORT.read_text())
    contrasts, arms = list(stored['contrasts'].values()), list(stored['arms'].values())
    rec = recurrent()
    rules, preds = decide(contrasts, arms, rec)
    report = {**stored, 'recurrent': rec, 'decision_rules': rules, 'predictions': preds,
              'addendum69': {'recurrent_rows': (SEQ_DIR / 'sequence_arms_{variable}.csv.gz').relative_to(ROOT).as_posix(),
                             'other_blocks_from': REPORT.relative_to(ROOT).as_posix()}}
    REPORT_A69.write_bytes(json.dumps(report, indent=1, allow_nan=False).encode('utf-8'))
    print('wrote', REPORT_A69.relative_to(ROOT))
    for v, e in rec.items():
        print(v, e['mae'], {k: c['by_design'] for k, c in e['contrasts'].items()},
              {k: c['gate_G3_pass'] for k, c in e['contrasts'].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--recurrent-only', action='store_true',
                    help='addendum 69: re-read the recurrent block on the per-fold tree rows, into ' + REPORT_A69.name)
    a = ap.parse_args()
    if a.recurrent_only:
        recurrent_only()
        return
    with ProcessPoolExecutor(a.workers) as ex:
        contrasts = list(ex.map(job_contrast, specs()))
        arms = list(ex.map(job_arm, arm_specs()))
    rec = recurrent()
    rules, preds = decide(contrasts, arms, rec)
    missing = [c['label'] for c in contrasts if c.get('missing')] + [x['cell'] for x in arms if x.get('missing')]
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum49.json', 'draws': DRAWS,
              'missing': missing, 'contrasts': {c['label']: c for c in contrasts if not c.get('missing')},
              'arms': {x['cell']: x for x in arms if not x.get('missing')}, 'recurrent': rec,
              'decision_rules': rules, 'predictions': preds}
    REPORT.write_text(json.dumps(report, indent=1, allow_nan=False))
    print('wrote', REPORT.relative_to(ROOT), '| missing:', len(missing))
    for c in contrasts:
        if c.get('missing'):
            continue
        v = c['values']
        print(f"{c['label']:64s} pub {v['published'][0]:+.4f}{'*' if excludes(v['published']) else ' '}"
              f" shared {v['shared'][0]:+.4f}{'*' if excludes(v['shared']) else ' '}"
              f" perfold {v['perfold'][0]:+.4f}{'*' if excludes(v['perfold']) else ' '}"
              + (f"  G3 {c.get('gate_G3_pass')}" if 'gate_G3_pass' in c else ''))
    print(json.dumps(preds, indent=1))


if __name__ == '__main__':
    main()
