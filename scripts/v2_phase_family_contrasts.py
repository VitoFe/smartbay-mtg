"""Addendum 82, part B: gates, pairing, contrasts and verdicts of the feature-family sweep.

`g0`    B-G0 (and E-G0 with --part E): v2_phase_shift.build_shifted(v, h, 0) equals v2_horizon.build(v, h)
        elementwise, missing values included, for every array an arm reads, and every arm's design matrix
        built on the two grids is identical on every outer and inner fold. Run before any study.
`score` B-G1, B-G1b, B-G2, B-G3; for each of the eight swept contrasts and each of its eight cells (phases
        0-5 at sampler seed 42, seeds 7 and 123 on the reference grid) the contrast with the arm's base (the
        weather tree) and with the plain tree, on identical endpoints, each from its own seeded stream;
        Chapter 4's verdict without and with the per-fold replicate; the relabelled river rows; BH and BY
        over the part's own intervals. Writes reports/bay_twin/runs/v2_addendum82_B.json (six decimals).

Declared in config/bay_reserved_protocol_v2_addendum82.json (part B).
Usage: python -m scripts.v2_phase_family_contrasts g0 [--part E]
       python -m scripts.v2_phase_family_contrasts score
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.sweep_verdict import excludes, label, verdict  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
HERE = RUNS / 'v2_addendum82/B'
OUT = RUNS / 'v2_addendum82_B.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
PHASEFIX = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs'
KEY = ['fold', 'sensor', 'time']
C = factor()                      # mg/l per umol/kg
FLOORS = {'temperature': 0.029, 'oxygen': 0.059 / C}   # in the stored unit
TOLERANCE = 1e-9
CELLS = [(k, None) for k in range(6)] + [(0, 7), (0, 123)]
CONTRASTS = {
    'B1': ('oxygen', 6, 'weather_river_tree'),
    'B2': ('oxygen', 6, 'sealevel_tree'),
    'B3': ('oxygen', 48, 'sealevel_tree'),
    'B4': ('oxygen', 6, 'extboundary_tree'),
    'B5': ('oxygen', 24, 'bathy_tree'),
    'B6': ('temperature', 6, 'sealevel_tree'),
    'B7': ('temperature', 6, 'extboundary_tree'),
    'B8': ('temperature', 24, 'bathy_tree'),
}
#: B-G1: the published arm's macro MAE (declared values, runs/v2_augmented_vs_plain.json and v2_bathy_pooled.json)
PUBLISHED_MAE = {('oxygen', 6, 'weather_river_tree'): 6.363812935450921,
                 ('oxygen', 6, 'sealevel_tree'): 6.354320410084652,
                 ('oxygen', 6, 'extboundary_tree'): 6.430609989779052,
                 ('oxygen', 48, 'sealevel_tree'): 8.68542078852882,
                 ('temperature', 6, 'sealevel_tree'): 0.19427022495827323,
                 ('temperature', 6, 'extboundary_tree'): 0.19392021066066947,
                 ('oxygen', 24, 'bathy_tree'): 6.30383698710326,
                 ('temperature', 24, 'bathy_tree'): 0.2709220817356547}
#: B-G1b: the published 48 h oxygen plain and weather trees (run 20260920T083322Z_v2_horizon)
PUBLISHED_BASE_48 = {'tree': 9.124423571059427, 'weather_tree': 8.70553422684273}


def r6(x):
    return None if x is None else round(float(x), 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suffix(seed):
    return '' if seed is None else f'_seed{seed}'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode('utf-8'))


# ------------------------------------------------------------------------------------- G0

def _equal(x, y):
    if isinstance(x, pd.DatetimeIndex) or isinstance(y, pd.DatetimeIndex):
        return len(x) == len(y) and bool((pd.DatetimeIndex(x) == pd.DatetimeIndex(y)).all())
    if isinstance(x, (list, tuple)):
        return list(x) == list(y)
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if x.dtype.kind in 'fc':
            return bool(np.array_equal(x, y, equal_nan=True))
        return bool(np.array_equal(x, y))
    return x == y


def grid_identity(variable, horizon, designs):
    """build_shifted(v, h, 0) against v2_horizon.build(v, h); `designs` maps a name to f(data, fold)."""
    from src.experiments.v2_horizon import build, folds, inner_fold
    from src.experiments.v2_phase_shift import build_shifted
    a, b = build(variable, horizon), build_shifted(variable, horizon, 0)
    arrays = {k: _equal(a[k], b[k]) for k in sorted(set(a) | set(b)) if k in a and k in b}
    missing = sorted(set(a) ^ set(b))
    fa, fb = folds(a, horizon), folds(b, horizon)
    ia, ib = [inner_fold(a, f, horizon) for f in fa], [inner_fold(b, f, horizon) for f in fb]
    fold_ok = all(_equal(x[k], y[k]) for x, y in zip(fa + ia, fb + ib) for k in ('tr', 'va', 'eligible', 'fallback'))
    design_ok = {}
    for name, fn in designs.items():
        design_ok[name] = all(_equal(fn(a, x), fn(b, y)) for x, y in zip(fa + ia, fb + ib))
    ok = all(arrays.values()) and not missing and fold_ok and all(design_ok.values())
    return {'variable': variable, 'horizon_hours': horizon, 'arrays_equal': arrays, 'keys_only_in_one': missing,
            'folds_equal': fold_ok, 'designs_equal': design_ok, 'pass': ok}


def g0(part):
    from src.experiments.v2_phase_shift import design_for
    base = {'tree': lambda d, f: design_for(d, f, False), 'weather_tree': lambda d, f: design_for(d, f, True)}
    if part == 'B':
        from src.experiments.v2_phase_family import arm_design
        plan = [('oxygen', 6, ['weather_river_tree', 'sealevel_tree', 'extboundary_tree']),
                ('temperature', 6, ['sealevel_tree', 'extboundary_tree']),
                ('oxygen', 24, ['bathy_tree']), ('temperature', 24, ['bathy_tree']),
                ('oxygen', 48, ['sealevel_tree'])]
        checks = []
        for v, h, arms in plan:
            designs = dict(base)
            designs.update({arm: (lambda d, f, _a=arm: arm_design(_a, d, f)) for arm in arms})
            checks.append(grid_identity(v, h, designs))
            print(f"B-G0 {v} {h} h: {'pass' if checks[-1]['pass'] else 'FAIL'}", flush=True)
        path = HERE / 'gates' / 'B-G0.json'
    else:
        from src.experiments.v2_forecast import NEURAL_ARMS, graphs
        from src.experiments.v2_horizon import build
        checks = []
        for v, h in [('temperature', 6), ('oxygen', 6), ('oxygen', 24)]:
            res = grid_identity(v, h, base)
            # the graphs the arms read, built from each grid
            from src.experiments.v2_phase_shift import build_shifted
            a, b = build(v, h), build_shifted(v, h, 0)
            ga, gb = graphs(a), graphs(b)
            res['graphs_equal'] = {k: _equal(ga[k], gb[k]) for k in sorted({g for g, _, _ in NEURAL_ARMS.values()})}
            res['pass'] = res['pass'] and all(res['graphs_equal'].values())
            checks.append(res)
            print(f"E-G0 {v} {h} h: {'pass' if res['pass'] else 'FAIL'}", flush=True)
        path = RUNS / 'v2_addendum82/E' / 'gates' / 'E-G0.json'
    write_json(path, {'declared_in': DECLARED, 'gate': f'{part}-G0', 'checks': checks,
                      'pass': all(c['pass'] for c in checks)})
    print('wrote', path.relative_to(ROOT).as_posix())


# ------------------------------------------------------------------------------- scoring

def paired_cell(frame, a, b, label_):
    """v2_phase_shift.paired on the rows where both arms exist, with the paired count."""
    from src.experiments.v2_phase_shift import paired
    f = frame.dropna(subset=[a, b])
    point, lo, hi = paired(f, a, b, label_)
    return {'point': r6(point), 'lo': r6(lo), 'hi': r6(hi), 'excludes_zero': bool(excludes(lo, hi)),
            'paired_endpoints': int(len(f)), 'label': label_}


def base_file(variable, horizon, offset, seed):
    """The reused or newly tuned plain and weather trees at one cell, with its summary."""
    name = f'phase_shift_{variable}_offset{offset}_ratio{suffix(seed)}.csv.gz'
    summary = f'phase_shift_offset{offset}_ratio{suffix(seed)}_summary.json'
    if horizon == 6:
        folder = PHASEFIX / 'v2_phase_shift'
        return folder / name, PHASEFIX / 'v2_phase_shift' / summary
    if horizon == 24 and variable == 'oxygen':
        folder = PHASEFIX / 'v2_phase_shift_h24'
        return folder / name, folder / summary
    if horizon == 24:
        # copied byte for byte from the frozen repository's copy of the phasefix run; the digest is
        # checked against the local summary of that run (B-G2)
        return HERE / 'reused/phasefix_h24' / name, PHASEFIX / 'v2_phase_shift_h24' / summary
    if horizon == 48:
        folder = HERE / 'base_h48'
        return folder / name, folder / summary
    raise ValueError(horizon)


def read_perfold_rest():
    d = json.loads((RUNS / 'v2_perfold_rest.json').read_text(encoding='utf-8'))
    out = {}
    for e in d['forecast']['contrasts']:
        if e.get('subgroup', 'all') == 'all':
            out[(e['variable'], e['horizon'], e['contrast'])] = e
    return out


def triple(c):
    return (c['point'], c['lo'], c['hi'])


def p_value(point, lo, hi):
    from scripts.v2_fdr import p_mid, p_side
    p = p_side(point, lo, hi)
    return (p, 'estimate and end towards zero') if p is not None else (p_mid(lo, hi), 'midpoint')


def adjust(entries):
    from scripts.v2_fdr import step_up
    p = [e['p'] for e in entries]
    bh, by = step_up(p), step_up(p, dependence_robust=True)
    for e, r1, r2 in zip(entries, bh, by):
        e['bh'] = bool(r1 and e['excludes_zero'])
        e['by'] = bool(r2 and e['excludes_zero'])
    return {'intervals': len(entries), 'exclude_zero': sum(e['excludes_zero'] for e in entries),
            'bh_retained': sum(e['bh'] for e in entries), 'by_retained': sum(e['by'] for e in entries)}


def reading(cells, perfold, floor):
    """Both verdicts, the phase summary and the floor flags of one contrast."""
    seeds = [triple(cells['0|42']), triple(cells['0|7']), triple(cells['0|123'])]
    phases = [triple(cells[f'{k}|42']) for k in range(6)]
    without = verdict(seeds, phases)
    with_pf = verdict(seeds + [perfold], phases) if perfold is not None else None
    printed = with_pf if with_pf is not None else without
    points = [p for p, _, _ in phases]
    out = {'verdict_without_perfold': without, 'verdict_with_perfold': with_pf, 'verdict': printed,
           'printed': label(printed), 'phase_mean': r6(np.mean(points)), 'phase_range': r6(max(points) - min(points)),
           'negative_phases': int(sum(p < 0 for p in points)),
           'cells_excluding_zero': int(sum(c['excludes_zero'] for c in cells.values())),
           'floor': r6(floor)}
    if printed == 'result':
        allp = [c['point'] for c in cells.values()] + ([perfold[0]] if perfold is not None else [])
        out['size'] = {'per_fold_exceeds_floor': None if perfold is None else bool(abs(perfold[0]) > floor),
                       'phase_mean_exceeds_floor': bool(abs(np.mean(points)) > floor),
                       'cells_exceeding_floor': int(sum(abs(p) > floor for p in allp))}
    else:
        out['size'] = 'not read: the verdict is not stable (the verdict, phase mean and range are reported)'
    return out


def score():
    perfold_rest = read_perfold_rest()
    report = {'declared_in': DECLARED, 'part': 'B',
              'unit': 'oxygen umol/kg as stored (print x 0.032844 mg/l), temperature degC',
              'cells': 'phases 0-5 at sampler seed 42 and seeds 7 and 123 at offset 0, keyed offset|seed',
              'stream': 'v2_phase_shift.paired, default_rng([20260921, crc32(label)]), 600 (fold, ISO week) block draws',
              'gates': {}, 'contrasts': {}, 'relabelled': {}}
    g0_path = HERE / 'gates' / 'B-G0.json'
    g0r = json.loads(g0_path.read_text(encoding='utf-8')) if g0_path.exists() else None
    report['gates']['B-G0'] = {'pass': bool(g0r and g0r['pass']), 'file': g0_path.relative_to(ROOT).as_posix()}

    # B-G1b and B-G2 on every base cell, B-G3 on every ISPRA arm
    g1b = {}
    for arm_name, want in PUBLISHED_BASE_48.items():
        path, summary = base_file('oxygen', 48, 0, None)
        s = json.loads(summary.read_text(encoding='utf-8'))
        got = s['variables']['oxygen']['mae'][arm_name]
        g1b[arm_name] = {'mae_here': got, 'published': want, 'abs_diff': abs(got - want),
                         'pass': bool(abs(got - want) <= TOLERANCE)}
    report['gates']['B-G1b'] = {'cells': g1b, 'pass': all(v['pass'] for v in g1b.values())}
    report['gates']['B-G1'], report['gates']['B-G2'], report['gates']['B-G3'] = {}, {}, {}
    entries = []
    for cid, (variable, horizon, arm) in CONTRASTS.items():
        cells, plain, sparse, g2 = {}, {}, {}, {}
        g3_cells = {}
        for offset, seed in CELLS:
            key = f'{offset}|{42 if seed is None else seed}'
            stem = f'phase_family_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}'
            arm_path = HERE / 'arms' / f'{stem}.csv.gz'
            arm_summary = json.loads((HERE / 'arms' / f'{stem}_summary.json').read_text(encoding='utf-8'))
            if arm_summary['prediction_sha256'] != sha(arm_path):
                raise SystemExit(f'{arm_path.name}: digest differs from its summary')
            if arm_summary['selection']['completed_trials'] < 100 or arm_summary.get('smoke_only'):
                raise SystemExit(f'{arm_path.name}: fewer than 100 trials or a smoke run')
            if arm in ('sealevel_tree', 'extboundary_tree'):
                g3_cells[key] = arm_summary.get('hourly_binning')
            tree_path, tree_summary = base_file(variable, horizon, offset, seed)
            ts = json.loads(tree_summary.read_text(encoding='utf-8'))
            recorded = ts['variables'][variable]['prediction_sha256']
            actual = sha(tree_path)
            a = pd.read_csv(arm_path)
            t = pd.read_csv(tree_path)
            j = a.merge(t, on=KEY, how='outer', suffixes=('', '_tree'), indicator=True)
            same = bool((j['_merge'] == 'both').all())
            truth_gap = float((j.truth - j.truth_tree).abs().max()) if same else float('nan')
            ok = (recorded == actual) and same and truth_gap <= TOLERANCE and \
                ts['horizon_hours'] == horizon and ts['offset_hours'] == offset
            g2[key] = {'file': tree_path.relative_to(ROOT).as_posix(), 'sha256': actual,
                       'hash_matches_summary': recorded == actual, 'same_endpoints': same,
                       'max_truth_difference': truth_gap, 'pass': bool(ok)}
            frame = j.drop(columns=['_merge', 'truth_tree'])
            lab = f'{variable}|{arm}|h{horizon}|{offset}|{42 if seed is None else seed}'
            cells[key] = paired_cell(frame, arm, 'weather_tree', lab + '|weather')
            plain[key] = paired_cell(frame, arm, 'tree', lab + '|tree')
            n = frame.groupby(['fold', 'sensor']).size()
            sparse[key] = {f'{f}|{s}': int(c) for (f, s), c in n.items() if c < 5}
            cells[key]['mae_arm'] = r6(arm_summary['mae'])
            cells[key]['selection'] = {'best_trial': arm_summary['selection']['best_trial'],
                                       'completed_trials': arm_summary['selection']['completed_trials'],
                                       'failed_trials': arm_summary['selection']['failed_trials']}
            if offset == 0 and seed is None:
                from src.experiments.v2_phase_shift import macro, paired
                f = frame.dropna(subset=[arm, 'weather_tree'])
                point = macro(f, arm) - macro(f, 'weather_tree')
                pub = perfold_rest[(variable, horizon, f'{arm} - weather_tree')]['published']['value']
                mae = macro(frame.dropna(subset=[arm]), arm)
                report['gates']['B-G1'][cid] = {
                    'mae_here': mae, 'mae_published': PUBLISHED_MAE[(variable, horizon, arm)],
                    'mae_abs_diff': abs(mae - PUBLISHED_MAE[(variable, horizon, arm)]),
                    'contrast_here': point, 'contrast_published': pub[0], 'contrast_abs_diff': abs(point - pub[0]),
                    'pass': bool(abs(mae - PUBLISHED_MAE[(variable, horizon, arm)]) <= TOLERANCE and
                                 abs(point - pub[0]) <= TOLERANCE)}
        report['gates']['B-G2'][cid] = {'cells': g2, 'pass': all(c['pass'] for c in g2.values())}
        if g3_cells:
            report['gates']['B-G3'][cid] = {'binning': g3_cells, 'pass': all(v == 'end' for v in g3_cells.values())}
        pf_w = perfold_rest.get((variable, horizon, f'{arm} - weather_tree'))
        pf_t = perfold_rest.get((variable, horizon, f'{arm} - tree'))
        pfw = tuple(pf_w['perfold']['value']) if pf_w else None
        pft = tuple(pf_t['perfold']['value']) if pf_t else None
        floor = FLOORS[variable]
        entry = {'variable': variable, 'horizon_hours': horizon, 'arm': arm, 'base': 'weather_tree',
                 'cells': cells, 'per_fold': None if pfw is None else [r6(x) for x in pfw],
                 'reference_cell_published_stream': None if pf_w is None else [r6(x) for x in pf_w['published']['value']],
                 'sparse_cells': {k: v for k, v in sparse.items() if v}}
        entry.update(reading(cells, pfw, floor))
        # the reference cell in the published stream, if its interval status differs from this sweep's
        if pf_w is not None:
            pub = pf_w['published']['value']
            if excludes(pub[1], pub[2]) != cells['0|42']['excludes_zero']:
                alt = dict(cells)
                alt['0|42'] = {'point': pub[0], 'lo': pub[1], 'hi': pub[2], 'excludes_zero': excludes(pub[1], pub[2])}
                entry['reading_with_published_stream_reference'] = reading(alt, pfw, floor)
        entry['against_plain_tree'] = {'cells': plain, 'per_fold': None if pft is None else [r6(x) for x in pft]}
        entry['against_plain_tree'].update(reading(plain, pft, floor))
        report['contrasts'][cid] = entry
        for ref, cs in (('weather_tree', cells), ('tree', plain)):
            for key, c in cs.items():
                p, how = p_value(c['point'], c['lo'], c['hi'])
                entries.append({'contrast': cid, 'reference': ref, 'cell': key, 'point': c['point'], 'lo': c['lo'],
                                'hi': c['hi'], 'excludes_zero': c['excludes_zero'], 'p': p, 'p_from': how})
        print(f"{cid} {variable} {horizon} h {arm} - weather_tree: "
              + ' '.join(f"{k}:{c['point']:+.5f}{'*' if c['excludes_zero'] else ''}" for k, c in cells.items())
              + f"  per fold {pfw[0] if pfw else None:+.5f}  phase mean {entry['phase_mean']:+.6f}  -> {entry['printed']}",
              flush=True)

    # R1 and R2: the six-hour river rows addendum 78 swept
    river6 = json.loads((RUNS / 'v2_phase_river6.json').read_text(encoding='utf-8'))
    for rid, v in (('R1', 'oxygen'), ('R2', 'temperature')):
        e = river6[v]
        report['relabelled'][rid] = {'variable': v, 'horizon_hours': 6, 'contrast': 'river_tree - tree',
                                     'verdict_without_perfold': e['verdict_without_perfold'], 'verdict': e['verdict'],
                                     'printed': label(e['verdict']), 'phase_mean': e['summary']['phase_mean'],
                                     'phase_range': e['summary']['phase_range'],
                                     'source': 'runs/v2_phase_river6.json (addendum 78, run ' + river6['run'] + ')'}
    report['multiplicity'] = {'family': 'every new interval of part B: 8 contrasts x 8 cells x 2 references; '
                                        'per-fold values read from published artifacts excluded',
                              'q': 0.05, **adjust(entries), 'entries': entries}
    gates = report['gates']
    report['gates_pass'] = {'B-G0': gates['B-G0']['pass'], 'B-G1b': gates['B-G1b']['pass'],
                            'B-G1': {k: v['pass'] for k, v in gates['B-G1'].items()},
                            'B-G2': {k: v['pass'] for k, v in gates['B-G2'].items()},
                            'B-G3': {k: v['pass'] for k, v in gates['B-G3'].items()}}
    readable = {cid: bool(gates['B-G0']['pass'] and gates['B-G1'][cid]['pass'] and gates['B-G2'][cid]['pass']
                          and gates['B-G3'].get(cid, {'pass': True})['pass']
                          and (gates['B-G1b']['pass'] if CONTRASTS[cid][1] == 48 else True))
                for cid in CONTRASTS}
    report['readable'] = readable
    for cid, ok in readable.items():
        if not ok:
            report['contrasts'][cid]['status'] = 'NOT READ: a gate failed'
    write_json(OUT, report)
    print('wrote', OUT.relative_to(ROOT).as_posix(), '| gates:', json.dumps(report['gates_pass']))
    print('multiplicity:', {k: v for k, v in report['multiplicity'].items() if k != 'entries'})


def main():
    p = argparse.ArgumentParser()
    # without an action it scores, as the pipeline runs it (g0 is run by hand before any study)
    p.add_argument('action', nargs='?', default='score', choices=['g0', 'score'])
    p.add_argument('--part', choices=['B', 'E'], default='B')
    a = p.parse_args()
    if a.action == 'g0':
        g0(a.part)
    else:
        score()


if __name__ == '__main__':
    main()
