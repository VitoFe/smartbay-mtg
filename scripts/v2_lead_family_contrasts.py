"""Addendum 83: gates, pairing, contrasts and verdicts of the feature families at every lead from 1 to 72 h.

`g0`     L-G0 and L-G0b, before any study: v2_phase_shift.build_shifted(v, h, 0) equals v2_horizon.build(v, h)
         elementwise for every array an arm reads at all 16 (variable, lead) pairs, every arm's design is identical
         on the two grids on every outer and inner fold, and the sweep's Magra designs equal v2_river.design_for.
`g7`     L-G7: the per-fold mode of v2_lead_family, run with --out into gates/L-G7/, reproduces two published
         per-fold cells (best trial, best value and parameters per fold; predictions to 1e-9).
`score`  L-G1 to L-G6, L-G8 and L-G9; every contrast cell of the declaration at its eight swept cells and its
         per-fold value, both verdicts, the floor flags, BH and BY over the addendum's family.
         Writes reports/bay_twin/runs/v2_addendum83.json (six decimals).

Declared in config/bay_reserved_protocol_v2_addendum83.json.
Usage: python -m scripts.v2_lead_family_contrasts g0
       python -m scripts.v2_lead_family_contrasts g7
       python -m scripts.v2_lead_family_contrasts score [--workers N]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.sweep_verdict import excludes, label, verdict  # noqa: E402
from src.units import factor  # noqa: E402
from scripts.v2_phase_family_contrasts import (_equal, adjust, grid_identity, p_value, r6, reading, sha,  # noqa: E402
                                               triple, write_json)

RUNS = ROOT / 'reports/bay_twin/runs'
HERE = RUNS / 'v2_addendum83'
OUT = RUNS / 'v2_addendum83.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum83.json'
PHASEFIX = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs'
A82B = RUNS / 'v2_addendum82/B'
KEY = ['fold', 'sensor', 'time']
C = factor()                      # mg/l per umol/kg
FLOORS = {'temperature': 0.029, 'oxygen': 0.059 / C}   # in the stored unit
TOLERANCE = 1e-9
LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
VARIABLES = ('temperature', 'oxygen')
CELLS = [(k, None) for k in range(6)] + [(0, 7), (0, 123)]
ARMS = ('river_tree', 'weather_river_tree', 'sealevel_tree', 'extboundary_tree', 'bathy_tree')
FAMILY = {'tree': 'horizon', 'weather_tree': 'horizon', 'river_tree': 'river', 'weather_river_tree': 'river',
          'sealevel_tree': 'ispra', 'extboundary_tree': 'ispra', 'bathy_tree': 'bathy'}
#: contrast id -> (arm, reference); P against the arm's base, S the second reference (the plain tree)
CONTRASTS = {'P1': ('weather_tree', 'tree'), 'P2': ('river_tree', 'tree'),
             'P3': ('weather_river_tree', 'weather_tree'), 'P4': ('sealevel_tree', 'weather_tree'),
             'P5': ('extboundary_tree', 'weather_tree'), 'P6': ('bathy_tree', 'weather_tree'),
             'S3': ('weather_river_tree', 'tree'), 'S4': ('sealevel_tree', 'tree'),
             'S5': ('extboundary_tree', 'tree'), 'S6': ('bathy_tree', 'tree')}
T, O = 'temperature', 'oxygen'
#: (arm, variable, lead) swept under the shared selection before this addendum, at all eight cells
SWEPT_BEFORE = ({(a, v, h) for a in ('tree', 'weather_tree') for v, h in [(T, 6), (O, 6), (T, 24), (O, 24), (O, 48)]}
                | {('river_tree', v, h) for v, h in [(T, 6), (O, 6), (O, 24)]}
                | {('weather_river_tree', O, 6), ('weather_river_tree', O, 24)}
                | {('sealevel_tree', v, h) for v, h in [(O, 6), (T, 6), (O, 48)]}
                | {('extboundary_tree', O, 6), ('extboundary_tree', T, 6)}
                | {('bathy_tree', O, 24), ('bathy_tree', T, 24)})


def _published_ref():
    ref = {}
    for v in VARIABLES:
        for h in (1, 3, 6, 12, 24, 36, 48):
            ref[('tree', v, h)] = ref[('weather_tree', v, h)] = True
        for h in (6, 24, 48):
            for a in ('river_tree', 'weather_river_tree', 'sealevel_tree', 'extboundary_tree'):
                ref[(a, v, h)] = True
        for h in (6, 24):
            ref[('bathy_tree', v, h)] = True
    return set(ref)


PUBLISHED_REF = _published_ref()        # arms with a published reference-grid value and a per-fold value
#: L-G1: the declared reference-grid macro MAE of each arm whose offset-0 seed-42 study is re-run here
PUBLISHED_MAE = {
    ('tree', T, 1): 0.08692312240596264, ('tree', T, 3): 0.1535573492491659, ('tree', T, 12): 0.2482185855244337,
    ('tree', T, 36): 0.347699935909771, ('tree', T, 48): 0.3827562043742813,
    ('tree', O, 1): 2.739873911723282, ('tree', O, 3): 4.519536857322882, ('tree', O, 12): 7.172531320322227,
    ('tree', O, 36): 8.925855781704112,
    ('weather_tree', T, 1): 0.08632528046125765, ('weather_tree', T, 3): 0.15195133336925168,
    ('weather_tree', T, 12): 0.24059457688203578, ('weather_tree', T, 36): 0.3786016716552799,
    ('weather_tree', T, 48): 0.3747869242623212,
    ('weather_tree', O, 1): 2.7412722772752076, ('weather_tree', O, 3): 4.5132024451618555,
    ('weather_tree', O, 12): 6.946491147970559, ('weather_tree', O, 36): 8.496370269844995,
    ('river_tree', T, 24): 0.26988057529233583, ('river_tree', T, 48): 0.3785702358452975,
    ('river_tree', O, 48): 9.38803603298356,
    ('weather_river_tree', T, 6): 0.1923031945543808, ('weather_river_tree', T, 24): 0.26548491088997517,
    ('weather_river_tree', T, 48): 0.373178389048675, ('weather_river_tree', O, 48): 8.728038370423182,
    ('sealevel_tree', T, 24): 0.26353664806327987, ('sealevel_tree', T, 48): 0.37516632011899587,
    ('sealevel_tree', O, 24): 6.272575907825611,
    ('extboundary_tree', T, 24): 0.2647925765902506, ('extboundary_tree', T, 48): 0.3712965330735355,
    ('extboundary_tree', O, 24): 6.3330198411592455, ('extboundary_tree', O, 48): 8.762050986553785,
    ('bathy_tree', T, 6): 0.1931391432050763, ('bathy_tree', O, 6): 6.481337125858618,
}
#: L-G7: the two published per-fold cells the new per-fold mode must reproduce
G7_CELLS = [('sealevel_tree', 'oxygen', 48), ('weather_tree', 'temperature', 36)]


def suffix(seed):
    return '' if seed is None else f'_seed{seed}'


def ckey(offset, seed):
    return f'{offset}|{42 if seed is None else seed}'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


# ------------------------------------------------------------------------------------- L-G0

def _cache_loader():
    """build and build_shifted read the same loader; it is read once and shared by both (the check compares
    what each builds from it, not two reads of the files)."""
    import src.experiments.v2_horizon as hz
    import src.experiments.v2_phase_shift as ps
    original = hz.load_observations
    store = {}

    def cached():
        if 'o' not in store:
            store['o'] = original()
        return store['o']
    hz.load_observations = cached
    ps.load_observations = cached


def g0_pair(v, h):
    """L-G0 and L-G0b at one (variable, lead), written to gates/L-G0/<variable>_h<lead>.json."""
    from src.experiments.v2_horizon import build, folds, inner_fold
    from src.experiments.v2_lead_family import arm_design
    from src.experiments.v2_phase_shift import design_for
    from src.experiments.v2_river import design_for as river_design_for
    _cache_loader()
    designs = {'tree': lambda d, f: design_for(d, f, False), 'weather_tree': lambda d, f: design_for(d, f, True)}
    designs.update({arm: (lambda d, f, _a=arm: arm_design(_a, d, f)) for arm in ARMS})
    check = grid_identity(v, h, designs)
    data = build(v, h)
    outer = folds(data, h)
    fs = outer + [inner_fold(data, f, h) for f in outer]
    same = {'river_tree': all(_equal(arm_design('river_tree', data, f), river_design_for(data, f, False)) for f in fs),
            'weather_river_tree': all(_equal(arm_design('weather_river_tree', data, f),
                                             river_design_for(data, f, True)) for f in fs)}
    g0b = {'variable': v, 'horizon_hours': h, 'designs_equal_v2_river': same, 'pass': all(same.values())}
    write_json(HERE / 'gates' / 'L-G0' / f'{v}_h{h}.json', {'L-G0': check, 'L-G0b': g0b})
    print(f"L-G0 {v} {h} h: {'pass' if check['pass'] else 'FAIL'}; L-G0b {'pass' if g0b['pass'] else 'FAIL'}",
          flush=True)


def g0(pairs):
    """Each pair in its own process (each reads the loader once), then the merged record gates/L-G0.json."""
    with ProcessPoolExecutor(max_workers=len(pairs)) as ex:
        list(ex.map(_g0_job, pairs))
    checks, g0b = [], []
    for v in VARIABLES:
        for h in LEADS:
            f = HERE / 'gates' / 'L-G0' / f'{v}_h{h}.json'
            if f.exists():
                r = load_json(f)
                checks.append(r['L-G0'])
                g0b.append(r['L-G0b'])
    path = HERE / 'gates' / 'L-G0.json'
    complete = len(checks) == len(VARIABLES) * len(LEADS)
    write_json(path, {'declared_in': DECLARED, 'gate': 'L-G0 and L-G0b', 'L-G0': checks, 'L-G0b': g0b,
                      'complete': complete,
                      'pass_L-G0': complete and all(c['pass'] for c in checks),
                      'pass_L-G0b': complete and all(c['pass'] for c in g0b)})
    print('wrote', rel(path), 'complete' if complete else f'{len(checks)} of 16 pairs')


def _g0_job(pair):
    g0_pair(*pair)


# ------------------------------------------------------------------------------------- L-G7

#: amendment 1: L-G7 is read per environment; the ISPRA cell is read from its run on the compute host
#: (Optuna 4.9.0, where the published ISPRA per-fold studies ran), the horizon cell from its local run
G7_WHERE = {'ispra': ('L-G7_host_diagnostic', 'compute host, Optuna 4.9.0'),
            'horizon': ('L-G7', 'local, Optuna 5.0.0')}


def g7():
    out = {}
    for arm, v, h in G7_CELLS:
        tag = f'{FAMILY[arm]}_{v}_{arm}_h{h}'
        pub = RUNS / 'v2_perfold_rest/forecast/perfold' / tag
        folder, where = G7_WHERE[FAMILY[arm]]
        new = HERE / 'gates' / folder / tag
        rp, rn = load_json(pub / 'report.json'), load_json(new / 'report.json')
        sel = {}
        for k in ('0', '1'):
            a, b = rp['selection'][k], rn['selection'][k]
            sel[k] = {'best_trial': [a['best_trial'], b['best_trial']], 'best_value': [a['best_value'], b['best_value']],
                      'params_equal': a['params'] == b['params'], 'completed_trials': b['completed_trials'],
                      'pass': a['best_trial'] == b['best_trial'] and a['best_value'] == b['best_value']
                      and a['params'] == b['params']}
        dp = pd.read_csv(pub / 'selected_predictions.csv.gz')
        dn = pd.read_csv(new / 'selected_predictions.csv.gz')
        j = dp.merge(dn, on=KEY, how='outer', suffixes=('_pub', ''), indicator=True)
        same = bool((j['_merge'] == 'both').all()) and len(dp) == len(dn)
        diff = float((j.prediction - j.prediction_pub).abs().max()) if same else float('nan')
        out[tag] = {'environment': where, 'optuna_here': rn.get('environment', {}).get('optuna'),
                    'published_threads': rp.get('threads'), 'threads_here': rn.get('threads'), 'folds': sel,
                    'rows': [len(dp), len(dn)], 'same_rows': same, 'max_abs_prediction_diff': diff,
                    'pass': bool(same and diff <= TOLERANCE and all(s['pass'] for s in sel.values()))}
        print(f"L-G7 {tag}: {'pass' if out[tag]['pass'] else 'FAIL'} (max diff {diff:.2e})", flush=True)
    path = HERE / 'gates' / 'L-G7.json'
    write_json(path, {'declared_in': DECLARED, 'amended_by': 'config/bay_reserved_protocol_v2_addendum83_amendment1.json',
                      'gate': 'L-G7, read per environment (amendment 1)', 'cells': out,
                      'first_reading_local_only': 'the ISPRA cell run locally under Optuna 5.0.0 did not reproduce; the '
                                                  'record of that run is in gates/L-G7/ and logs/g7_sealevel.log',
                      'pass': all(c['pass'] for c in out.values())})
    print('wrote', rel(path))


# ------------------------------------------------------------------------------------- files

def base_file(variable, horizon, offset, seed):
    """(prediction file, summary, reused?) of the plain and weather trees at one cell."""
    name = f'phase_shift_{variable}_offset{offset}_ratio{suffix(seed)}.csv.gz'
    shared = f'phase_shift_offset{offset}_ratio{suffix(seed)}_summary.json'
    if horizon == 6:
        return PHASEFIX / 'v2_phase_shift' / name, PHASEFIX / 'v2_phase_shift' / shared, True
    if horizon == 24 and variable == 'oxygen':
        return PHASEFIX / 'v2_phase_shift_h24' / name, PHASEFIX / 'v2_phase_shift_h24' / shared, True
    if horizon == 24:
        return A82B / 'reused/phasefix_h24' / name, PHASEFIX / 'v2_phase_shift_h24' / shared, True
    if horizon == 48 and variable == 'oxygen':
        return A82B / 'base_h48' / name, A82B / 'base_h48' / shared, True
    folder = HERE / 'base' / f'h{horizon}'
    return (folder / name, folder / f'phase_shift_{variable}_offset{offset}_ratio{suffix(seed)}_summary.json',
            False)


def arm_file(variable, arm, horizon, offset, seed):
    stem = f'lead_family_{variable}_{arm}_h{horizon}_offset{offset}{suffix(seed)}'
    return HERE / 'arms' / f'{stem}.csv.gz', HERE / 'arms' / f'{stem}_summary.json'


def perfold_folder(variable, arm, horizon):
    """(folder, reused?) of the per-fold predictions of one arm."""
    if (arm, variable, horizon) in PUBLISHED_REF:
        if arm in ('tree', 'weather_tree') and horizon in (6, 24):
            return RUNS / 'v2_perfold/forecast/perfold' / f'{variable}_{arm}_h{horizon}', True
        return RUNS / 'v2_perfold_rest/forecast/perfold' / f'{FAMILY[arm]}_{variable}_{arm}_h{horizon}', True
    return HERE / 'perfold' / f'{FAMILY[arm]}_{variable}_{arm}_h{horizon}', False


def published_horizon_rows(variable, horizon):
    run = '20260920T133747Z_v2_horizon' if horizon in (1, 3, 12, 36) else '20260920T083322Z_v2_horizon'
    return RUNS / run / 'results' / f'v2_horizon_{variable}_tree_h{horizon}' / 'selected_predictions.csv.gz'


# ------------------------------------------------------------------------------------- existing readings

def existing_readings():
    """The 24 cells read as they stand and S3 on oxygen at 24 h (verdict from stored cells), keyed (cid, v, h)."""
    perfold = load_json(RUNS / 'v2_perfold.json')['forecast']['cells']
    rest = load_json(RUNS / 'v2_perfold_rest.json')['forecast']
    rest_c = {(e['variable'], e['horizon'], e['contrast']): e for e in rest['contrasts'] if e['subgroup'] == 'all'}
    swept = rest['swept']
    river6 = load_json(RUNS / 'v2_phase_river6.json')
    b = load_json(RUNS / 'v2_addendum82_B.json')
    out = {}

    def pf(v, h, arm, ref):
        e = rest_c.get((v, h, f'{arm} - {ref}'))
        return None if e is None else [r6(x) for x in e['perfold']['value']]

    for v in VARIABLES:
        for h in (6, 24):
            c = perfold[f'{v}|{h}']
            out[('P1', v, h)] = {'source': 'runs/v2_perfold.json (runs/v2_phase_matched.json, runs/v2_seed_contrasts.json)',
                                 'verdict': c['verdict']['with_perfold'],
                                 'verdict_without_perfold': c['verdict']['without_perfold'],
                                 'phase_mean': r6(c['verdict']['phase_mean']),
                                 'per_fold': [r6(x) for x in c['contrast']['perfold']['value']],
                                 'shared': [r6(x) for x in c['contrast']['published']['value']]}
        out[('P2', v, 6)] = {'source': 'runs/v2_phase_river6.json', 'verdict': river6[v]['verdict'],
                             'verdict_without_perfold': river6[v]['verdict_without_perfold'],
                             'phase_mean': r6(river6[v]['summary']['phase_mean']),
                             'phase_range': r6(river6[v]['summary']['phase_range']),
                             'per_fold': pf(v, 6, 'river_tree', 'tree')}
    out[('P2', O, 24)] = {'source': 'runs/v2_perfold_rest.json swept (runs/v2_phase_river.json)',
                          'verdict': swept['river_tree - tree']['with_perfold'],
                          'verdict_without_perfold': swept['river_tree - tree']['rule_without_perfold'],
                          'per_fold': pf(O, 24, 'river_tree', 'tree')}
    out[('P3', O, 24)] = {'source': 'runs/v2_perfold_rest.json swept (runs/v2_phase_weather_river.json)',
                          'verdict': swept['weather_river_tree - weather_tree']['with_perfold'],
                          'verdict_without_perfold': swept['weather_river_tree - weather_tree']['rule_without_perfold'],
                          'per_fold': pf(O, 24, 'weather_river_tree', 'weather_tree')}
    bmap = {'B1': ('3', O, 6), 'B2': ('4', O, 6), 'B3': ('4', O, 48), 'B4': ('5', O, 6), 'B5': ('6', O, 24),
            'B6': ('4', T, 6), 'B7': ('5', T, 6), 'B8': ('6', T, 24)}
    for bid, (n, v, h) in bmap.items():
        e = b['contrasts'][bid]
        for cid, block in ((f'P{n}', e), (f'S{n}', e['against_plain_tree'])):
            out[(cid, v, h)] = {'source': f'runs/v2_addendum82_B.json {bid}' + ('' if cid.startswith('P') else
                                                                                ' against_plain_tree'),
                                'verdict': block['verdict'] if b['readable'][bid] else None,
                                'verdict_without_perfold': block['verdict_without_perfold'],
                                'phase_mean': r6(block['phase_mean']), 'phase_range': r6(block['phase_range']),
                                'per_fold': [r6(x) for x in block['per_fold']] if block.get('per_fold') else None}
    # S3 on oxygen at 24 h: the stored cells of addendum 45, verdict computed here
    w = load_json(RUNS / 'v2_phase_weather_river.json')
    name = 'weather_river_tree - tree'
    phases = [tuple(w['phases'][str(k)]['contrasts'][name]) for k in range(6)]
    seeds = [phases[0]] + [tuple(w['seeds'][s]['contrasts'][name]) for s in ('7', '123')]
    pfv = pf(O, 24, 'weather_river_tree', 'tree')
    points = [p for p, _, _ in phases]
    out[('S3', O, 24)] = {'source': 'runs/v2_phase_weather_river.json (stored cells), verdict computed here',
                          'verdict': verdict(seeds + [tuple(pfv)], phases),
                          'verdict_without_perfold': verdict(seeds, phases),
                          'phase_mean': r6(np.mean(points)), 'phase_range': r6(max(points) - min(points)),
                          'per_fold': pfv,
                          'replicates': {'phases': [[r6(x) for x in c] for c in phases],
                                         'seeds_7_123': [[r6(x) for x in c] for c in seeds[1:]]}}
    for k, e in out.items():
        e['printed'] = label(e['verdict']) if e['verdict'] else 'not read'
        e['status'] = 'read from its artifact' if k != ('S3', O, 24) else 'verdict computed from stored cells'
    return out, rest_c, perfold


# ------------------------------------------------------------------------------------- scoring workers

def _paired_job(args):
    """One sweep-cell interval: v2_phase_shift.paired on the rows where both arms exist."""
    frame, a, b, lab = args
    from scripts.v2_phase_family_contrasts import paired_cell
    return paired_cell(frame, a, b, lab)


def _perfold_job(args):
    frame, lab = args
    sys.path.insert(0, str(ROOT))
    from scripts.v2_perfold_rest_contrasts import Blocks
    f = frame.dropna(subset=['arm', 'ref']).reset_index(drop=True)
    point, lo, hi = Blocks(f).boot({'arm': 1.0, 'ref': -1.0}, 20260920, lab)
    return {'point': r6(point), 'lo': r6(lo), 'hi': r6(hi), 'excludes_zero': bool(excludes(lo, hi)),
            'paired_endpoints': int(len(f)), 'label': lab}


def _macro(f, col):
    from src.experiments.v2_phase_shift import macro
    return macro(f, col)


# ------------------------------------------------------------------------------------- score

def score(workers):
    existing, rest_c, perfold_cells = existing_readings()
    gates = {'L-G1': {}, 'L-G2': {}, 'L-G3': {}, 'L-G4': {}, 'L-G5': {}, 'L-G6': {}, 'L-G8': {}, 'L-G9': {}}
    g0 = load_json(HERE / 'gates/L-G0.json')
    g7 = load_json(HERE / 'gates/L-G7.json')
    g0_by = {(c['variable'], c['horizon_hours']): c['pass'] for c in g0['L-G0']}
    g0b_by = {(c['variable'], c['horizon_hours']): c['pass'] for c in g0['L-G0b']}
    gates['L-G0'] = {'pass': g0['pass_L-G0'], 'file': rel(HERE / 'gates/L-G0.json')}
    gates['L-G0b'] = {'pass': g0['pass_L-G0b']}
    gates['L-G7'] = {'pass': g7['pass'], 'cells': {k: {'pass': v['pass'], 'max_abs_prediction_diff': v['max_abs_prediction_diff']}
                                                  for k, v in g7['cells'].items()}}
    hosts, studies_seen, problems = {}, {}, []

    def note_study(summary_sel, where):
        name = summary_sel['study']
        if name in studies_seen and studies_seen[name] != where:
            problems.append(f'study name used twice: {name}')
        studies_seen[name] = where
        return summary_sel['completed_trials'] >= 100 and not summary_sel.get('smoke_only')

    def env(e):
        k = json.dumps({x: e.get(x) for x in ('host', 'python', 'optuna', 'lightgbm', 'V2_THREADS', 'OMP_NUM_THREADS')},
                       sort_keys=True)
        hosts[k] = hosts.get(k, 0) + 1

    # ---------------------------------------------------------------- bases
    bases, levels = {}, {}
    for v in VARIABLES:
        for h in LEADS:
            for o, s in CELLS:
                path, summary, reused = base_file(v, h, o, s)
                sm = load_json(summary)
                entry = sm['variables'][v]
                frame = pd.read_csv(path)
                key = (v, h, ckey(o, s))
                ok = entry['prediction_sha256'] == sha(path) and sm['horizon_hours'] == h and sm['offset_hours'] == o
                if reused:
                    gates['L-G2'][f'base|{v}|{h}|{ckey(o, s)}'] = {'file': rel(path), 'hash_matches_summary': bool(ok),
                                                                   'pass': bool(ok)}
                else:
                    t6 = all(note_study(sel, rel(summary)) for sel in entry['selections'].values())
                    gates['L-G6'][f'base|{v}|{h}|{ckey(o, s)}'] = {'completed_trials': {a: x['completed_trials'] for a, x in
                                                                                        entry['selections'].items()},
                                                                   'smoke_only': sm.get('smoke_only'),
                                                                   'hash_matches_summary': bool(ok), 'pass': bool(t6 and ok)}
                    env(sm['environment'])
                bases[key] = frame
                levels[f'{v}|{h}|{ckey(o, s)}'] = {a: r6(entry['mae'][a]) for a in
                                                    ('tree', 'weather_tree', 'persistence', 'climatology')}
            # L-G1 on the re-run reference cells of the trees
            for arm in ('tree', 'weather_tree'):
                if (arm, v, h) in PUBLISHED_MAE:
                    f = bases[(v, h, '0|42')]
                    mae = _macro(f, arm)
                    want = PUBLISHED_MAE[(arm, v, h)]
                    g = {'mae_here': mae, 'mae_published': want, 'mae_abs_diff': abs(mae - want)}
                    if arm == 'weather_tree':
                        point = _macro(f.dropna(subset=['tree', 'weather_tree']), 'weather_tree') - \
                            _macro(f.dropna(subset=['tree', 'weather_tree']), 'tree')
                        pub = rest_c[(v, h, 'weather_tree - tree')]['published']['value'][0]
                        g.update(contrast='weather_tree - tree', contrast_here=point, contrast_published=pub,
                                 contrast_abs_diff=abs(point - pub))
                    g['pass'] = bool(g['mae_abs_diff'] <= TOLERANCE and g.get('contrast_abs_diff', 0.0) <= TOLERANCE)
                    gates['L-G1'][f'{arm}|{v}|{h}'] = g
            # L-G4 (1-48 h) and L-G5 (72 h) on new reference-grid bases
            path, summary, reused = base_file(v, h, 0, None)
            if not reused:
                f = bases[(v, h, '0|42')]
                if h == 72:
                    gates['L-G5'][f'base|{v}'] = rows_vs_points72(f, v, {'persistence': 'last',
                                                                         'climatology': 'clim_month_hour'})
                else:
                    gates['L-G4'][f'base|{v}|{h}'] = rows_vs_published(f, v, h, {'persistence': 'persistence',
                                                                               'climatology': 'baseline'})

    # ---------------------------------------------------------------- arms
    arm_frames = {}
    for v in VARIABLES:
        for h in LEADS:
            for arm in ARMS:
                if (arm, v, h) in SWEPT_BEFORE:
                    continue
                for o, s in CELLS:
                    path, summary = arm_file(v, arm, h, o, s)
                    sm = load_json(summary)
                    a = pd.read_csv(path)
                    b = bases[(v, h, ckey(o, s))]
                    j = a.merge(b, on=KEY, how='outer', suffixes=('', '_base'), indicator=True)
                    same = bool((j['_merge'] == 'both').all())
                    gap = float((j.truth - j.truth_base).abs().max()) if same else float('nan')
                    ok_hash = sm['prediction_sha256'] == sha(path)
                    ok6 = note_study(sm['selection'], rel(summary))
                    gates['L-G6'][f'{arm}|{v}|{h}|{ckey(o, s)}'] = {
                        'completed_trials': sm['selection']['completed_trials'],
                        'failed_trials': sm['selection']['failed_trials'], 'smoke_only': sm.get('smoke_only'),
                        'hash_matches_summary': bool(ok_hash), 'same_endpoints_as_base': same,
                        'max_truth_difference': gap,
                        'pass': bool(ok6 and ok_hash and same and gap <= TOLERANCE)}
                    if arm in ('sealevel_tree', 'extboundary_tree'):
                        gates['L-G3'][f'{arm}|{v}|{h}|{ckey(o, s)}'] = {'binning': sm.get('hourly_binning'),
                                                                        'pass': sm.get('hourly_binning') == 'end'}
                    env(sm['environment'])
                    arm_frames[(v, h, arm, ckey(o, s))] = j.drop(columns=['_merge', 'truth_base'])
                if (arm, v, h) in PUBLISHED_MAE:
                    f = arm_frames[(v, h, arm, '0|42')]
                    mae = _macro(f.dropna(subset=[arm]), arm)
                    want = PUBLISHED_MAE[(arm, v, h)]
                    g = {'mae_here': mae, 'mae_published': want, 'mae_abs_diff': abs(mae - want), 'contrasts': {}}
                    for ref in (('tree',) if arm == 'river_tree' else ('weather_tree', 'tree')):
                        ff = f.dropna(subset=[arm, ref])
                        point = _macro(ff, arm) - _macro(ff, ref)
                        pub = rest_c[(v, h, f'{arm} - {ref}')]['published']['value'][0]
                        g['contrasts'][ref] = {'here': point, 'published': pub, 'abs_diff': abs(point - pub)}
                    g['pass'] = bool(g['mae_abs_diff'] <= TOLERANCE and
                                     all(c['abs_diff'] <= TOLERANCE for c in g['contrasts'].values()))
                    gates['L-G1'][f'{arm}|{v}|{h}'] = g

    # ---------------------------------------------------------------- per-fold files
    pf_frames, weak = {}, {}
    for v in VARIABLES:
        for h in LEADS:
            for arm in ('tree', 'weather_tree') + ARMS:
                folder, reused = perfold_folder(v, arm, h)
                rep = load_json(folder / 'report.json')
                path = folder / 'selected_predictions.csv.gz'
                ok = rep['prediction_sha256'] == sha(path)
                if reused:
                    gates['L-G2'][f'perfold|{arm}|{v}|{h}'] = {'file': rel(path), 'hash_matches_report': bool(ok),
                                                               'pass': bool(ok)}
                else:
                    t6 = all(note_study(sel, rel(folder / 'report.json')) for sel in rep['selection'].values())
                    gates['L-G6'][f'perfold|{arm}|{v}|{h}'] = {
                        'completed_trials': {k: x['completed_trials'] for k, x in rep['selection'].items()},
                        'smoke_only': rep.get('smoke_only'), 'hash_matches_report': bool(ok), 'pass': bool(t6 and ok)}
                    if arm in ('sealevel_tree', 'extboundary_tree'):
                        gates['L-G3'][f'perfold|{arm}|{v}|{h}'] = {'binning': rep.get('hourly_binning'),
                                                                   'pass': rep.get('hourly_binning') == 'end'}
                    env(rep['environment'])
                    cols = {'persistence': 'persistence', 'baseline': 'baseline', 'clim_hour': 'clim_hour'}
                    d = pd.read_csv(path)
                    if h == 72:
                        gates['L-G5'][f'perfold|{arm}|{v}'] = rows_vs_points72(
                            d, v, {'persistence': 'last', 'baseline': 'clim_month_hour', 'clim_hour': 'clim_hour'})
                    else:
                        gates['L-G4'][f'perfold|{arm}|{v}|{h}'] = rows_vs_published(d, v, h, cols)
                d = pd.read_csv(path)
                pf_frames[(v, h, arm)] = d
                if arm in ('tree', 'weather_tree'):
                    m = rep['metrics']
                    weak[f'{v}|{h}|{arm}'] = {
                        f'fold{k}': {'ratio_to_persistence': r6(m[f'fold{k}']['mae'] / m[f'fold{k}']['persistence_mae']),
                                     'trails_persistence': bool(m[f'fold{k}']['mae'] > m[f'fold{k}']['persistence_mae'])}
                        for k in (0, 1) if f'fold{k}' in m}

    # ---------------------------------------------------------------- the cells
    jobs, where = [], []
    for v in VARIABLES:
        for h in LEADS:
            for cid, (arm, ref) in CONTRASTS.items():
                if (cid, v, h) in existing:
                    continue
                for o, s in CELLS:
                    k = ckey(o, s)
                    if arm == 'weather_tree':
                        frame = bases[(v, h, k)]
                    else:
                        frame = arm_frames[(v, h, arm, k)]
                    tag = 'weather' if ref == 'weather_tree' else 'tree'
                    lab = f'a83|{v}|{arm}|h{h}|{o}|{42 if s is None else s}|{tag}'
                    jobs.append((frame[KEY + ['truth', arm, ref]].copy(), arm, ref, lab))
                    where.append(('cell', cid, v, h, k))
                pa, pr = perfold_folder(v, arm, h)[1], perfold_folder(v, ref, h)[1]
                if not (pa and pr):
                    fa = pf_frames[(v, h, arm)][KEY + ['truth', 'prediction']].rename(columns={'prediction': 'arm'})
                    fr = pf_frames[(v, h, ref)][KEY + ['truth', 'prediction']].rename(columns={'prediction': 'ref'})
                    j = fa.merge(fr, on=KEY, how='outer', suffixes=('', '_ref'))
                    if float((j.truth - j.truth_ref).abs().max()) > TOLERANCE:
                        problems.append(f'per-fold truth differs: {cid} {v} {h}')
                    jobs.append(('PF', j.drop(columns=['truth_ref']), f'a83|{v}|{h}|perfold|{arm} - {ref}'))
                    where.append(('pf', cid, v, h, None))
    print(f'{len(jobs)} intervals to draw with {workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_dispatch, jobs, chunksize=4))

    cells = {}
    for (kind, cid, v, h, k), res in zip(where, results):
        e = cells.setdefault((cid, v, h), {'cells': {}, 'per_fold': None, 'per_fold_new': None})
        if kind == 'cell':
            e['cells'][k] = res
        else:
            e['per_fold_new'] = res

    # ---------------------------------------------------------------- readings
    report_cells, entries = {}, []
    for v in VARIABLES:
        for h in LEADS:
            for cid, (arm, ref) in CONTRASTS.items():
                ck = f'{cid}|{v}|{h}'
                if (cid, v, h) in existing:
                    report_cells[ck] = {'contrast': f'{arm} - {ref}', 'variable': v, 'horizon_hours': h,
                                        **existing[(cid, v, h)], 'floor': r6(FLOORS[v])}
                    continue
                e = cells[(cid, v, h)]
                if e['per_fold_new'] is not None:
                    pf = e['per_fold_new']
                    pfv = (pf['point'], pf['lo'], pf['hi'])
                    pf_source = 'computed here'
                elif arm == 'weather_tree' and h in (6, 24):
                    pfv = tuple(perfold_cells[f'{v}|{h}']['contrast']['perfold']['value'])
                    pf_source = 'runs/v2_perfold.json'
                else:
                    pfv = tuple(rest_c[(v, h, f'{arm} - {ref}')]['perfold']['value'])
                    pf_source = 'runs/v2_perfold_rest.json'
                out = {'contrast': f'{arm} - {ref}', 'variable': v, 'horizon_hours': h, 'cells': e['cells'],
                       'per_fold': [r6(x) for x in pfv], 'per_fold_source': pf_source,
                       'per_fold_detail': e['per_fold_new']}
                out.update(reading(e['cells'], pfv, FLOORS[v]))
                if out['verdict'] == 'result':
                    out['size']['floor_flag_per_fold'] = bool(abs(pfv[0]) > FLOORS[v])
                pub = rest_c.get((v, h, f'{arm} - {ref}'))
                if arm == 'weather_tree' and h in (6, 24):
                    pub = None
                if pub is not None:
                    pv = pub['published']['value']
                    out['reference_cell_published_stream'] = [r6(x) for x in pv]
                    if excludes(pv[1], pv[2]) != e['cells']['0|42']['excludes_zero']:
                        alt = dict(e['cells'])
                        alt['0|42'] = {'point': pv[0], 'lo': pv[1], 'hi': pv[2], 'excludes_zero': excludes(pv[1], pv[2])}
                        out['reading_with_published_stream_reference'] = reading(alt, pfv, FLOORS[v])
                sparse = {}
                for k, c in e['cells'].items():
                    frame = (bases[(v, h, k)] if arm == 'weather_tree' else arm_frames[(v, h, arm, k)])
                    n = frame.dropna(subset=[arm, ref]).groupby(['fold', 'sensor']).size()
                    small = {f'{a}|{b}': int(m) for (a, b), m in n.items() if m < 5}
                    if small:
                        sparse[k] = small
                out['sparse_cells'] = sparse
                gates['L-G8'][ck] = sparse
                bases_used = [b for b in (arm, ref) if b in ('tree', 'weather_tree')]
                out['weak_per_fold_bases'] = {b: weak.get(f'{v}|{h}|{b}') for b in bases_used}
                report_cells[ck] = out
                for k, c in e['cells'].items():
                    p, how = p_value(c['point'], c['lo'], c['hi'])
                    entries.append({'cell_id': ck, 'replicate': k, 'point': c['point'], 'lo': c['lo'], 'hi': c['hi'],
                                    'excludes_zero': c['excludes_zero'], 'p': p, 'p_from': how})
                if e['per_fold_new'] is not None:
                    c = e['per_fold_new']
                    p, how = p_value(c['point'], c['lo'], c['hi'])
                    entries.append({'cell_id': ck, 'replicate': 'per fold', 'point': c['point'], 'lo': c['lo'],
                                    'hi': c['hi'], 'excludes_zero': c['excludes_zero'], 'p': p, 'p_from': how})

    # ---------------------------------------------------------------- readability
    gates['L-G9'] = {'environments': hosts}
    readable = {}
    for ck, out in report_cells.items():
        cid, v, h = ck.split('|')
        h = int(h)
        arm, ref = CONTRASTS[cid]
        if 'cells' not in out:
            readable[ck] = out['verdict'] is not None
            continue
        ok = g0_by[(v, h)] and g0b_by[(v, h)]
        for a in (arm, ref):
            g = gates['L-G1'].get(f'{a}|{v}|{h}')
            if g is not None and not g['pass']:
                ok = False

        def relevant(gk, _v=v, _h=h, _a=arm, _r=ref):
            p = gk.split('|')
            if p[0] == 'base':
                return p[1] == _v and (len(p) == 2 and _h == 72 or len(p) > 2 and p[2] == str(_h))
            if p[0] == 'perfold':
                return p[1] in (_a, _r) and p[2] == _v and (len(p) == 3 and _h == 72 or len(p) > 3 and p[3] == str(_h))
            return p[0] in (_a, _r) and p[1] == _v and p[2] == str(_h)
        for gname in ('L-G2', 'L-G3', 'L-G4', 'L-G5', 'L-G6'):
            if any(relevant(gk) and not g['pass'] for gk, g in gates[gname].items()):
                ok = False
        if out.get('per_fold_source') == 'computed here' and not g7['pass']:
            ok = False
        readable[ck] = bool(ok)
        if not ok:
            out['status'] = 'NOT READ: a gate failed'
    family = adjust(entries)
    by_vl = {}
    for e in entries:
        _, v, h = e['cell_id'].split('|')[0], e['cell_id'].split('|')[1], e['cell_id'].split('|')[2]
        k = f'{v}|{h}'
        x = by_vl.setdefault(k, {'intervals': 0, 'exclude_zero': 0, 'bh_retained': 0, 'by_retained': 0})
        x['intervals'] += 1
        x['exclude_zero'] += int(e['excludes_zero'])
        x['bh_retained'] += int(e['bh'])
        x['by_retained'] += int(e['by'])
    counts = {}
    for ck, out in report_cells.items():
        _, v, h = ck.split('|')
        counts.setdefault(f'{v}|{h}', {})
        p = out.get('printed') if readable[ck] else 'not read'
        counts[f'{v}|{h}'][p] = counts[f'{v}|{h}'].get(p, 0) + 1
    report = {
        'declared_in': DECLARED,
        'unit': 'oxygen umol/kg as stored (print x 0.032844 mg/l), temperature degC',
        'cells': 'phases 0-5 at sampler seed 42 and seeds 7 and 123 at offset 0, keyed offset|seed',
        'streams': {'sweep': "v2_phase_shift.paired, default_rng([20260921, crc32(label)]), label "
                             "'a83|{variable}|{arm}|h{lead}|{offset}|{seed}|{weather|tree}', 600 (fold, ISO week) draws",
                    'per_fold': "scripts/v2_perfold_rest_contrasts.Blocks.boot, default_rng([20260920, crc32(label)]), "
                                "label 'a83|{variable}|{lead}|perfold|{arm} - {reference}', 600 draws"},
        'floors_stored_unit': {k: r6(x) for k, x in FLOORS.items()},
        'gates': gates,
        'gates_summary': {g: (gates[g]['pass'] if 'pass' in gates[g] else
                              all(x.get('pass', True) for x in gates[g].values() if isinstance(x, dict)))
                          for g in gates if g not in ('L-G8', 'L-G9')},
        'problems': problems,
        'levels': levels,
        'contrast_cells': report_cells,
        'readable': readable,
        'verdict_counts_by_variable_and_lead': counts,
        'multiplicity': {'family': 'addendum 83: every interval computed here on a difference (eight sweep cells of '
                                   'each contrast cell computed here, and every new per-fold value); intervals read '
                                   'from published artifacts and gate reproductions excluded',
                         'q': 0.05, **family, 'by_variable_and_lead': by_vl, 'entries': entries},
    }
    write_json(OUT, report)
    print('wrote', rel(OUT))
    print('gates:', json.dumps(report['gates_summary']))
    print('multiplicity:', json.dumps({k: v for k, v in family.items()}))
    if problems:
        print('PROBLEMS:', problems[:10])


def _dispatch(job):
    if isinstance(job[0], str) and job[0] == 'PF':
        return _perfold_job((job[1], job[2]))
    return _paired_job(job)


def rows_vs_published(frame, variable, horizon, cols):
    """L-G4: keys and truth (and the named columns) against the published horizon rows of this lead."""
    pub = pd.read_csv(published_horizon_rows(variable, horizon))
    j = frame.merge(pub, on=KEY, how='outer', suffixes=('', '_pub'), indicator=True)
    same = bool((j['_merge'] == 'both').all()) and len(frame) == len(pub)
    diffs = {'truth': float((j.truth - j.truth_pub).abs().max()) if same else float('nan')}
    for mine, theirs in cols.items():
        col = theirs + '_pub' if theirs in frame.columns else theirs
        diffs[mine] = float((j[mine] - j[col]).abs().max()) if same else float('nan')
    return {'rows': [int(len(frame)), int(len(pub))], 'same_keys': same, 'max_abs_diff': diffs,
            'pass': bool(same and all(x <= TOLERANCE for x in diffs.values()))}


def rows_vs_points72(frame, variable, cols):
    """L-G5: keys, truth and the named columns against addendum 65's Dev-2025 72 h rows, at its nine decimals."""
    pts = pd.read_csv(RUNS / 'v2_weather_oracle/seed42' / f'points72_{variable}.csv.gz')
    pts = pts[pts.cohort == 'Dev-2025'].drop(columns=['cohort', 'lead'])
    pts['fold'] = pts['fold'].astype(int)
    f = frame.copy()
    f['time'] = pd.to_datetime(f['time'], utc=True)
    pts['time'] = pd.to_datetime(pts['time'], utc=True)
    j = f.merge(pts, on=KEY, how='outer', suffixes=('', '_a65'), indicator=True)
    same = bool((j['_merge'] == 'both').all()) and len(f) == len(pts)
    diffs = {}
    for mine, theirs in {'truth': 'truth', **cols}.items():
        col = theirs + '_a65' if theirs in f.columns else theirs
        diffs[mine] = float((j[mine].round(9) - j[col]).abs().max()) if same else float('nan')
    by_fold = {str(k): int(n) for k, n in f.groupby('fold').size().items()}
    return {'rows': [int(len(f)), int(len(pts))], 'rows_by_fold': by_fold, 'sensors': int(f.sensor.nunique()),
            'same_keys': same, 'max_abs_diff_after_rounding_to_9': diffs,
            'pass': bool(same and all(x <= 1e-9 for x in diffs.values()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', nargs='?', default='score', choices=['g0', 'g7', 'score'])
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--pairs', nargs='*', default=None, help="g0 only: 'variable:lead' pairs (default all 16)")
    a = p.parse_args()
    if a.action == 'g0':
        pairs = ([(x.split(':')[0], int(x.split(':')[1])) for x in a.pairs] if a.pairs
                 else [(v, h) for v in VARIABLES for h in LEADS])
        g0(pairs)
    elif a.action == 'g7':
        g7()
    else:
        score(a.workers)


if __name__ == '__main__':
    main()
