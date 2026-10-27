"""Addendum 81: the cross-task matrix of the eight pretrained arms, read from the three score files.

Reads the scorers' outputs and the run checks they cite, and nothing else: no prediction is re-read and
nothing is resampled.

    runs/v2_fm_zoo/score_6h.json           tasks A (Dev-2025, 6 h, six phases), B (Reserved-2026, 6 h), F (AOU)
    runs/v2_fm_zoo/score_dist_bridge.json  tasks C (distribution, 1-72 h), D (silent probe, 6-72 h)
    runs/v2_fm_zoo/score_gaps.json         task E (gaps, four cohorts, forward and bidirectional)
    runs/v2_addendum82/C/pretrained.json   task E on Gap-2026's gaps of 1, 2 and 3 h (addendum 82, part C,
                                           scored by the same gap scorer), and the 1-6 h band over 1, 2, 3
                                           and 6 h, which replaces the band of 6 h alone

and writes runs/v2_fm_zoo.json with

  gates      every gate the three files record, and the determinism and rerun checks of the run; a failed
             deciding gate stops the script before anything is read (declaration, gates.on_failure)
  matrix     one headline cell per (task, arm) and variable: the metric, the contrast against the task's
             main reference (and a secondary one) with its 95 % interval, its interval status, its
             Benjamini-Hochberg and Benjamini-Yekutieli flags and its size against the floor
  k3, k4     the covariate mode (cov - uni) and the newer model of each family, in the same cells
  fdr        BH and BY at q = 0.05 over every K1-K4 interval the three files store, p from each interval
             by scripts/v2_fdr.p_side (p_mid where the estimate lies outside its own interval), as
             addendum 79 did for the tables; per contrast the flags, and the family counts
  rankings   per headline cell, the arms by metric and those not distinguishable from the best, from the
             stored contrasts only (the scorers keep no per-row errors, so no other pair is computed)

Task A's contrast is the phase mean of arm - per-fold tree with the verdict over the six phases (Table 4.1,
src/experiments/sweep_verdict.py, phases only: the models are deterministic, so there is no seed replicate).
Floors are those Table 6.1 reads (scripts/v2_summary_table.floors): 0.029 degC for a contrast on one
probe's record; 0.2028 degC where an arm reads other instruments (the covariate arms of tasks D and E,
the anchored continuation, the Gaussian process and the published Chronos-2 covariate fill); 0.059 mg/l
for oxygen. A size is 'below' or 'exceeds' by its point and 'clears' when the whole interval lies beyond
the floor on one side.

Oxygen in mg/l throughout: the scorers' own mg/l fields where they store them, else the stored value x c
(src/units.factor(), checked against the c each file records). Values to six decimals; p-values to nine
significant figures, since the step-up procedures rank values near 1e-6.

Declared in config/bay_reserved_protocol_v2_addendum81.json (reading, contrasts K1-K4) and its amendments.
Usage: python -m scripts.v2_fm_zoo_summary
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import scripts.v2_fdr as FDR
from src.experiments.sweep_verdict import label as verdict_label, verdict as sweep_verdict
from src.units import factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
ZOO = RUNS / 'v2_fm_zoo'
SOURCES = {'6h': ZOO / 'score_6h.json', 'dist_bridge': ZOO / 'score_dist_bridge.json',
           'gaps': ZOO / 'score_gaps.json'}
CHECKS = {'model_checks': ZOO / 'model_checks.json', 'rerun_check_a3': ZOO / 'preds/rerun_check_a3.json',
          'declared_check_a3': ZOO / 'preds/declared_check_a3.json',
          'rerun_check_superseded': ZOO / 'preds/rerun_check.json'}
#: addendum 82, part C: Gap-2026 holds gaps of 1 to 3 h as well; their cells (and the 1-6 h band they
#: complete) enter this comparison's false-discovery family, as that part declared
PART_C = RUNS / 'v2_addendum82/C/pretrained.json'
PART_C_COHORT = 'Gap-2026'
PART_C_CELLS = ('1', '2', '3', '1-6')
OUT = RUNS / 'v2_fm_zoo.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum81.json'
AMENDMENTS = [f'config/bay_reserved_protocol_v2_addendum81_amendment{i}.json' for i in range(1, 6)]
Q = FDR.Q

ARMS = ['chronos_bolt_small_uni', 'chronos_bolt_base_uni', 'chronos2_uni', 'chronos2_cov',
        'timesfm25_uni', 'timesfm25_cov', 'timesfm3_uni', 'timesfm3_cov']
VARIABLES = ('temperature', 'oxygen')
COV = {a for a in ARMS if a.endswith('_cov')}
#: references whose values come from other instruments than the scored probe
READS_OTHERS = {'anchored_direct', 'neighbour_change', 'gpr', 'chronos2_cov_bidir'}
K3_PAIRS = [('chronos2_cov', 'chronos2_uni'), ('timesfm25_cov', 'timesfm25_uni'), ('timesfm3_cov', 'timesfm3_uni')]
K4_PAIRS = [('timesfm3_uni', 'timesfm25_uni'), ('chronos2_uni', 'chronos_bolt_base_uni')]
#: a new fill against the published column of the same model and mode: a reproduction difference (read by
#: gate G1), not a comparison of two methods, so it stays out of the false-discovery family
REPRODUCTION = {('chronos_bolt_base_uni.bidir', 'chronos_bidir'), ('chronos2_cov.bidir', 'chronos2_cov_bidir'),
                ('chronos2_uni.bidir', 'chronos2_bidir')}
C_LEADS = ('6', '24', '72')
D_LEADS = ('6', '24', '48', '72')
E_STRICT, E_BAND = ('Gap-Dev', 'Gap-2026'), '12-72'
E_LONG, E_LENGTHS = ('Gap-Long-Dev', 'Gap-Long-2026'), ('168', '336', '720')
NE = 'n.e.'


def r6(x):
    return None if x is None else round(float(x), 6)


def g9(p):
    return None if p is None else float(f'{p:.9g}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def base(name):
    """An arm or fill name without its fill direction."""
    return name.split('.')[0]


# ------------------------------------------------------------------------------------------- gates

def walk(o, path, out, skip=()):
    """Every 'passed' flag (and every 'same' flag of a reproduction) under `o`, with its path."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in skip:
                continue
            p = path + (str(k),)
            if k in ('passed', 'same') and isinstance(v, bool):
                out.append({'gate': '/'.join(path if k == 'passed' else p), 'passed': v})
            elif k == 'status' and v == 'pending':
                out.append({'gate': '/'.join(p), 'passed': False, 'note': 'pending'})
            else:
                walk(v, p, out, skip)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + (str(i),), out, skip)


def flag(out, name, ok, note=None):
    e = {'gate': name, 'passed': bool(ok)}
    if note:
        e['note'] = note
    out.append(e)


def gates(s6, sd, sg, checks):
    deciding = {'6h': [], 'dist_bridge': [], 'gaps': [], 'run_checks': []}
    informative = []
    g = deciding['6h']
    walk(s6['gates'], ('gates',), g)
    flag(g, 'status complete', s6.get('status') == 'complete')
    pc = s6['preconditions']
    flag(g, 'preconditions/prediction_files == expected', pc['prediction_files'] == pc['expected'])
    flag(g, 'preconditions/rerun_check_a3', pc['rerun_check_a3'] is True)

    g = deciding['dist_bridge']
    walk(sd['C']['gates'], ('C', 'gates'), g)
    walk(sd['D']['gates'], ('D', 'gates'), g)
    flag(g, 'complete', sd.get('complete') is True)
    pr = sd['preds']
    flag(g, 'preds/npz_files == declared_files', pr['npz_files'] == pr['declared_files'])
    flag(g, 'preds/ready', pr['ready'] is True)
    flag(g, 'preds/rerun_check_a3', pr['rerun_check_a3'] is True)
    for v in VARIABLES:
        ew = sd['C']['info'][v]['ewma_gate']
        bad = sum(e['violations'] for e in ew.values())
        flag(g, f'C/info/{v}/ewma_gate violations == 0', bad == 0, f'{len(ew)} leads checked')

    g = deciding['gaps']
    walk(sg['gates'], ('gates',), g, skip=('G1_informative',))
    flag(g, 'complete_run', sg.get('complete_run') is True)
    flag(g, 'preds_status/complete', sg['preds_status']['complete'] is True)
    flag(g, 'preds_status/rerun_check_a3', sg['preds_status']['rerun_check_a3'] is True)
    flag(g, "reading == 'allowed: every gate passed'", sg.get('reading') == 'allowed: every gate passed')
    flag(g, 'gates/G2_rows/problems empty', not sg['gates']['G2_rows']['problems'])
    flag(g, 'gates/R_paired_replica/disagreeing empty', not sg['gates']['R_paired_replica']['disagreeing'])
    walk(sg['gates']['G1_informative'], ('gates', 'G1_informative'), informative)
    for e in informative:
        e['note'] = 'recorded beside G1 by the scorer, not deciding it'

    g = deciding['run_checks']
    mc = checks['model_checks']
    flag(g, 'model_checks/G1_pass (harness on the compute host)', mc['G1_pass'] is True)
    flag(g, 'model_checks/G3_pass (determinism, 256 contexts per variable)', mc['G3_pass'] is True)
    g3 =[a for v in mc['G3'].values() for a in v['arms'].values()]
    flag(g, 'model_checks/G3 every arm identical on rerun', all(a.get('identical') and a.get('pass') for a in g3),
         f'{len(g3)} arm-mode runs')
    ra = checks['rerun_check_a3']
    flag(g, 'preds/rerun_check_a3 all_identical (amendment 3 outputs)', all(ra['all_identical'].values()),
         f"{ra['outputs_checked']} outputs, {ra['rows']} rows each")
    dc = checks['declared_check_a3']
    flag(g, 'preds/declared_check_a3 ok (every declared output present and citing its amendment)',
         dc['ok'] is True and dc['present'] == dc['expected'] and not dc['missing'] and not dc['problems'],
         f"{dc['present']} of {dc['expected']}")
    rc = checks['rerun_check_superseded']
    informative.append({'gate': 'preds/rerun_check (before amendment 3)', 'passed': bool(all(rc['all_identical'].values())),
                        'note': 'the batch dependence of TimesFM 3.0 that amendment 3 records; superseded by '
                                'preds/rerun_check_a3 on the declared one-series-per-call outputs'})
    failed = [dict(e, file=f) for f, lst in deciding.items() for e in lst if not e['passed']]
    return deciding, informative, failed


# --------------------------------------------------------------------------- the stored contrasts

class Family:
    """Every stored K1-K4 contrast interval, keyed (task, variable, cohort, cell, a, b)."""

    def __init__(self, c):
        self.c = c
        self.items = {}
        self.merged = []
        self.excluded = {'D without_sparse_cells (a sensitivity reading of the same contrast)': 0}

    def add(self, key, ks, point, lo, hi, units, rows, unit, mgl=None, extra=None):
        """`point`, `lo`, `hi` in the stored unit; `mgl` the scorer's own (point, lo, hi) in mg/l."""
        ks = sorted(set(ks))
        if key in self.items:
            old = self.items[key]
            same = all((x is None and y is None) or (x is not None and y is not None and abs(x - y) <= 1e-12)
                       for x, y in zip(old['stored'], (point, lo, hi)))
            if not same:
                raise SystemExit(f'two different intervals stored for {key}: {old["stored"]} and {(point, lo, hi)}')
            old['K'] = sorted(set(old['K']) | set(ks))
            self.merged.append('|'.join(key))
            return old
        task, variable = key[0], key[1]
        if variable == 'oxygen' and unit != 'mg/l':
            if mgl is None:
                mgl = tuple(None if v is None else v * self.c for v in (point, lo, hi))
        else:
            mgl = (point, lo, hi)
        estimable = lo is not None and hi is not None
        rec = {'id': '|'.join(key), 'task': task, 'variable': variable, 'cohort': key[2], 'cell': key[3],
               'a': key[4], 'b': key[5], 'K': ks, 'unit_stored': unit, 'stored': [point, lo, hi],
               'point': mgl[0], 'lo': mgl[1], 'hi': mgl[2], 'units': units, 'rows': rows,
               'estimable': estimable,
               'excludes_zero': bool(estimable and (lo > 0 or hi < 0)),
               'status': (('excludes zero' if (lo > 0 or hi < 0) else 'not distinguishable') if estimable else NE)}
        if extra:
            rec.update(extra)
        if (key[4], key[5]) in REPRODUCTION:
            rec['reproduction'] = True
        self.items[key] = rec
        return rec

    def get(self, key):
        return self.items.get(key)

    def pair(self, task, variable, cohort, cell, a, b):
        """The stored contrast a - b, from a - b or (sign flipped) b - a; None if neither is stored."""
        r = self.items.get((task, variable, cohort, cell, a, b))
        if r is not None:
            return r, 1
        r = self.items.get((task, variable, cohort, cell, b, a))
        if r is not None:
            return r, -1
        return None, 0


def split(name):
    a, b = name.split(' - ')
    return a.strip(), b.strip()


def collect(s6, sd, sg, fam):
    # tasks A, B, F: entries carry their K list; 'value' [point, lo, hi] in 'unit', oxygen also 'value_mgl'
    for v in VARIABLES:
        for o, cell in s6['A'][v]['offsets'].items():
            for name, e in cell['contrasts'].items():
                a, b = split(name)
                fam.add(('A', v, f'offset{o}', '6', a, b), e['K'], *e['value'], e['units'], e['rows'], e['unit'],
                        mgl=tuple(e['value_mgl']) if 'value_mgl' in e else None)
        for name, e in s6['B'][v]['contrasts'].items():
            a, b = split(name)
            fam.add(('B', v, 'Reserved-2026', '6', a, b), e['K'], *e['value'], e['units'], e['rows'], e['unit'],
                    mgl=tuple(e['value_mgl']) if 'value_mgl' in e else None)
    for name, e in s6['F']['oxygen']['contrasts'].items():
        a, b = split(name)
        fam.add(('F', 'oxygen', 'Reserved-2026', '6', a, b), e['K'], *e['value'], e['units'], e['rows'], e['unit'])

    # tasks C and D: 'point', 'ci' in the stored unit, oxygen also 'point_mgl', 'ci_mgl'; K from 'families'
    stored_unit = {'temperature': 'degC', 'oxygen': 'umol/kg (stored sonde unit)'}
    for task, block, leads in (('C', sd['C']['scores'], None), ('D', sd['D']['scores'], 'by_lead')):
        for v in VARIABLES:
            for cohort, sc in block[v].items():
                cells = sc[leads] if leads else sc
                for lead, cell in cells.items():
                    ks = {}
                    for k, names in cell['families'].items():
                        for n in names:
                            ks.setdefault(n, []).append(k)
                    for name, e in cell['contrasts'].items():
                        if e is None:
                            continue
                        a, b = split(name)
                        ci = e.get('ci') or (None, None)
                        mgl = None
                        if v == 'oxygen':
                            cm = e.get('ci_mgl') or (None, None)
                            mgl = (e['point_mgl'], cm[0], cm[1])
                        extra = {}
                        if 'without_sparse_cells' in e:
                            w = e['without_sparse_cells']
                            extra['without_sparse_cells'] = {'point': w['point'], 'ci': w.get('ci'),
                                                             'excludes_zero': w.get('excludes_zero')}
                            fam.excluded['D without_sparse_cells (a sensitivity reading of the same contrast)'] += 1
                        if 'reading' in e:
                            extra['scorer_reading'] = e['reading']
                        units = e.get('units_fold_week')
                        rows = e.get('rows', e.get('paired_episodes'))
                        fam.add((task, v, cohort, lead, a, b), ks.get(name, []), e['point'], ci[0], ci[1],
                                units, rows, stored_unit[v], mgl=mgl, extra=extra)

    # task E: K1[fill][minus_<ref>], K2[fwd fill] (minus chronos_bolt_base_uni.fwd), K3/K4['a - b']
    for cohort, byvar in sg['results'].items():
        for v, res in byvar.items():
            for cl, cell in res['cells'].items():
                def add(a, b, k, e):
                    mg = e.get('mgl')
                    mgl = None if mg is None else (mg['delta'], mg['ci_low'], mg['ci_high'])
                    extra = {'jackknife_t': e.get('jackknife_t'),
                             'jackknife_excludes_zero': e.get('jackknife_excludes_zero'),
                             'robust': e.get('robust'), 'robust_chapter4': e.get('robust_chapter4'),
                             'unit_of_resampling': cell.get('unit')}
                    if mg is not None and mg.get('jackknife_t') is not None:
                        extra['jackknife_t_mgl'] = mg['jackknife_t']
                    unit = 'degC' if v == 'temperature' else 'umol/kg (stored sonde unit)'
                    return fam.add(('E', v, cohort, cl, a, b), [k], e['delta'], e['ci_low'], e['ci_high'],
                                   e['units'], e['rows'], unit, mgl=mgl, extra=extra)
                for fill, refs in cell['K1'].items():
                    for rk, e in refs.items():
                        if isinstance(e, str):
                            continue
                        add(fill, rk[len('minus_'):], 'K1', e)
                for fill, e in cell['K2'].items():
                    add(fill, 'chronos_bolt_base_uni.fwd', 'K2', e)
                for k in ('K3', 'K4'):
                    for name, e in cell[k].items():
                        a, b = split(name)
                        add(a, b, k, e)
                # a bidirectional fill's K2 is its K1 contrast against Chronos-Bolt base bidirectional
                for fill in cell['K1']:
                    r = fam.get(('E', v, cohort, cl, fill, 'chronos_bolt_base_uni.bidir'))
                    if r is not None and 'K2' not in r['K']:
                        r['K'] = sorted(set(r['K']) | {'K2'})


def adjust(fam):
    recs = [r for r in fam.items.values() if r['estimable'] and not r.get('reproduction')]
    repro = [r for r in fam.items.values() if r.get('reproduction')]
    fam.excluded['a new fill against the published fill of the same model (reproduction difference)'] = {
        'contrasts': len(repro), 'with_an_interval': sum(r['estimable'] for r in repro),
        'interval_excludes_zero': sum(r['excludes_zero'] for r in repro),
        'largest_abs_point': r6(max([abs(r['stored'][0]) for r in repro], default=None))}
    mismatch = []
    for r in recs:
        p, lo, hi = r['stored']
        ps = FDR.p_side(p, lo, hi)
        r['p_side_outside'] = ps is None
        r['p_mid'] = FDR.p_mid(lo, hi)
        r['p'] = ps if ps is not None else r['p_mid']
        if r['p'] is None:
            # an interval of zero width: neither rule defines p; it is decided by its own position
            r['degenerate_interval'] = True
            r['p'] = 0.0 if r['excludes_zero'] else 1.0
            r['p_mid'] = r['p']
        if (r['p'] < 0.05) != r['excludes_zero'] and not r['p_side_outside']:
            mismatch.append(r['id'])
    counts = {}
    for key in ('p', 'p_mid'):
        p = [r[key] if r[key] is not None else 1.0 for r in recs]
        bh, by = FDR.step_up(p), FDR.step_up(p, dependence_robust=True)
        for r, x, y in zip(recs, bh, by):
            suffix = '' if key == 'p' else '_mid'
            r['bh' + suffix] = bool(x and r['excludes_zero'])
            r['by' + suffix] = bool(y and r['excludes_zero'])
        counts[key] = {'intervals': len(recs), 'exclude_zero': sum(r['excludes_zero'] for r in recs),
                       'bh_retained': sum(r['bh' + ('' if key == 'p' else '_mid')] for r in recs),
                       'by_retained': sum(r['by' + ('' if key == 'p' else '_mid')] for r in recs)}

    def tally(rs):
        return {'intervals': len(rs), 'exclude_zero': sum(r['excludes_zero'] for r in rs),
                'bh_retained': sum(r['bh'] for r in rs), 'by_retained': sum(r['by'] for r in rs)}
    by_k = {k: tally([r for r in recs if k in r['K']]) for k in ('K1', 'K2', 'K3', 'K4')}
    by_task = {t: tally([r for r in recs if r['task'] == t]) for t in 'ABCDEF'}
    by_var = {v: tally([r for r in recs if r['variable'] == v]) for v in VARIABLES}
    thresholds = {'bh': max([r['p'] for r in recs if r['bh']], default=None),
                  'by': max([r['p'] for r in recs if r['by']], default=None)}
    ne = [r for r in fam.items.values() if not r['estimable'] and not r.get('reproduction')]
    return {'q': Q, 'p_rule': 'scripts/v2_fdr.p_side (estimate and the interval end towards zero); p_mid where '
                              'the estimate lies outside its own interval',
            'counts': counts['p'], 'counts_midpoint_rule': counts['p_mid'],
            'by_family': by_k, 'by_task': by_task, 'by_variable': by_var,
            'p_threshold': {k: g9(v) for k, v in thresholds.items()},
            'not_estimable_no_interval': len(ne),
            'merged_duplicates': len(fam.merged),
            'excluded': fam.excluded,
            'p_disagrees_with_interval': mismatch,
            'estimate_outside_its_interval': [r['id'] for r in recs if r['p_side_outside']],
            'sensitive_to_the_p_rule': [r['id'] for r in recs if r['bh'] != r['bh_mid'] or r['by'] != r['by_mid']]}


# ------------------------------------------------------------------------------ the headline cells

def floors():
    f = FDR.S.floors()
    return {'temperature_paired': float(f['quant']), 'temperature_other_instruments': float(f['coloc_low']),
            'oxygen_mgl': float(f['sonde'])}


def floor_for(fl, task, variable, a, b):
    if variable == 'oxygen':
        return fl['oxygen_mgl'], 'oxygen (mg/l)'
    others = {base(a), base(b), a, b}
    if task in ('D', 'E') and (others & COV or others & READS_OTHERS):
        return fl['temperature_other_instruments'], 'an arm reads other instruments'
    return fl['temperature_paired'], "one probe's record"


def against(point, lo, hi, floor):
    if point is None:
        return None
    if lo is not None and hi is not None and (lo > floor or hi < -floor):
        return 'clears'
    return 'exceeds' if abs(point) > floor else 'below'


def view(rec, fl, sign=1):
    """A stored contrast as a headline cell reads it (mg/l for oxygen)."""
    if rec is None:
        return None
    p, lo, hi = rec['point'], rec['lo'], rec['hi']
    if sign < 0:
        p, lo, hi = -p, (None if hi is None else -hi), (None if lo is None else -lo)
    floor, basis = floor_for(fl, rec['task'], rec['variable'], rec['a'], rec['b'])
    out = {'contrast': f"{rec['a']} - {rec['b']}" if sign > 0 else f"{rec['b']} - {rec['a']}",
           'point': r6(p), 'ci': None if lo is None else [r6(lo), r6(hi)], 'status': rec['status'],
           'units': rec['units'], 'rows': rec['rows'], 'K': rec['K'],
           'p': g9(rec.get('p')), 'bh': rec.get('bh', False), 'by': rec.get('by', False),
           'floor': r6(floor), 'floor_basis': basis, 'floor_reading': against(p, lo, hi, floor)}
    for k in ('robust', 'jackknife_excludes_zero', 'robust_chapter4', 'reproduction'):
        if rec.get(k) is not None:
            out[k] = rec[k]
    if rec.get('jackknife_t') is not None:
        jk = rec.get('jackknife_t_mgl') or rec['jackknife_t']
        out['jackknife_t'] = [r6(x) for x in (jk if sign > 0 else [-jk[1], -jk[0]])]
    if 'without_sparse_cells' in rec:
        w = rec['without_sparse_cells']
        out['without_sparse_cells_excludes_zero'] = w['excludes_zero']
    return out


def mg(variable, x, c):
    return None if x is None else (x * c if variable == 'oxygen' else x)


def task_a(s6, fam, fl, c):
    out, k34, rank = {}, {}, {}
    for v in VARIABLES:
        A = s6['A'][v]
        offsets = sorted(A['offsets'], key=int)
        cells, ks = {}, {}

        def phases(a, b):
            name = f'{a} - {b}'
            vd = A['verdicts'].get(name)
            recs = [fam.get(('A', v, f'offset{o}', '6', a, b)) for o in offsets]
            if vd is None or any(r is None for r in recs):
                return None
            triples = [(r['stored'][0], r['stored'][1], r['stored'][2]) for r in recs]
            check = sweep_verdict([triples[0]], triples)
            if check != vd['verdict_code']:
                raise SystemExit(f'A {v} {name}: stored verdict {vd["verdict_code"]}, recomputed {check}')
            pm = float(np.mean([r['point'] for r in recs]))
            floor, basis = floor_for(fl, 'A', v, a, b)
            return {'contrast': name, 'phase_mean': r6(pm), 'phase_range': r6(max(r['point'] for r in recs)
                                                                              - min(r['point'] for r in recs)),
                    'verdict_code': vd['verdict_code'], 'verdict': verdict_label(vd['verdict_code']),
                    'phases_excluding_zero': vd['phases_excluding_zero'],
                    'bh_phases_retained': sum(r.get('bh', False) for r in recs),
                    'by_phases_retained': sum(r.get('by', False) for r in recs),
                    'by_phase': [[r6(r['point']), r6(r['lo']), r6(r['hi'])] for r in recs],
                    'floor': r6(floor), 'floor_basis': basis, 'floor_reading': against(pm, None, None, floor),
                    'offset0': view(recs[0], fl)}
        for arm in ARMS:
            maes = [mg(v, A['offsets'][o]['mae'][arm], c) for o in offsets]
            e = {'metric': 'MAE', 'unit': 'degC' if v == 'temperature' else 'mg/l',
                 'mae_offset0': r6(maes[0]), 'mae_phase_mean': r6(np.mean(maes)),
                 'mae_by_phase': [r6(x) for x in maes],
                 'vs_tree': phases(arm, 'tree'), 'vs_tree_shared': phases(arm, 'tree_shared')}
            if arm.startswith('timesfm'):
                pcs = [A['offsets'][o]['timesfm_positivity_clamp'][arm] for o in offsets]
                e['positivity_clamp'] = {'clamp_on': sorted({bool(p['clamp_on']) for p in pcs}),
                                         'zero_outputs': sum(p['record']['of_them_with_a_zero_output'] for p in pcs)}
            cells[arm] = e
        refs = {r: {'mae_offset0': r6(mg(v, A['offsets'][offsets[0]]['mae'][r], c)),
                    'mae_phase_mean': r6(np.mean([mg(v, A['offsets'][o]['mae'][r], c) for o in offsets]))}
                for r in ('tree', 'tree_shared', 'persistence', 'clim_hour')}
        out[v] = {'six_phases': {'arms': cells, 'references': refs}}
        for a, b in K3_PAIRS + K4_PAIRS:
            ks[f'{a} - {b}'] = phases(a, b)
        k34[v] = {'six_phases': ks}
        order = sorted(ARMS, key=lambda a: cells[a]['mae_phase_mean'])
        best = order[0]
        detail = {}
        for arm in order[1:]:
            ph = phases(best, arm) or phases(arm, best)
            if ph is None:
                detail[arm] = {'reading': 'no stored contrast (per-row errors are not stored by the scorer)'}
                continue
            code = ph['verdict_code']
            detail[arm] = {'contrast': ph['contrast'], 'verdict': ph['verdict'],
                           'reading': ('not distinguishable' if code == 'not distinguishable' else
                                       'choice-sensitive' if code == 'not estimable' else 'distinguishable')}
        o0 = sorted(ARMS, key=lambda a: cells[a]['mae_offset0'])
        rank[v] = {'six_phases': {'metric': 'MAE, mean over the six phases', 'order': order, 'best': best,
                                  'versus_best': detail,
                                  'not_distinguishable_from_best': [a for a, d in detail.items()
                                                                    if d['reading'] in ('not distinguishable',
                                                                                        'choice-sensitive')],
                                  'rule': 'verdict over the six phases of the stored contrast best - arm '
                                          '(or arm - best): no demonstrated difference or choice-sensitive'},
                   'offset0': ranking(fam, fl, 'A', v, f'offset0', '6', o0, {a: cells[a]['mae_offset0'] for a in ARMS})}
    return out, k34, rank


def ranking(fam, fl, task, v, cohort, cell, order, metric, suffix=''):
    best = order[0]
    detail, nd, missing = {}, [], []
    for arm in order[1:]:
        rec, sign = fam.pair(task, v, cohort, cell, best + suffix, arm + suffix)
        if rec is None:
            detail[arm] = {'reading': 'no stored contrast (per-row errors are not stored by the scorer)'}
            missing.append(arm)
            continue
        vw = view(rec, fl, sign)
        reading = {'not distinguishable': 'not distinguishable', NE: NE}.get(rec['status'], 'distinguishable')
        detail[arm] = {'contrast': vw['contrast'], 'point': vw['point'], 'ci': vw['ci'], 'reading': reading}
        if reading in ('not distinguishable', NE):
            nd.append(arm)
    return {'order': order, 'metric': {a: r6(metric[a]) for a in order}, 'best': best, 'versus_best': detail,
            'not_distinguishable_from_best': nd, 'not_stored': missing,
            'rule': 'interval of the stored contrast best - arm (or arm - best) spans zero'}


def simple_cell(fam, fl, task, v, cohort, cell, arm, metric, main, secondary, extra=None, suffix=''):
    e = {'metric': metric}
    e.update(extra or {})
    for tag, ref in (('vs_main', main), ('vs_secondary', secondary)):
        if ref is None:
            continue
        rec = fam.get((task, v, cohort, cell, arm + suffix, ref))
        e[tag] = view(rec, fl)
        if rec is None:
            e[tag] = {'contrast': f'{arm}{suffix} - {ref}', 'reading': 'not stored'}
    return e


def k_cells(fam, fl, task, v, cohort, cell, suffix=''):
    return {f'{a}{suffix} - {b}{suffix}': view(fam.get((task, v, cohort, cell, a + suffix, b + suffix)), fl)
            for a, b in K3_PAIRS + K4_PAIRS}


def task_bf(s6, fam, fl):
    out, k34, rank = {}, {}, {}
    for task, variables in (('B', VARIABLES), ('F', ('oxygen',))):
        out[task], k34[task], rank[task] = {}, {}, {}
        for v in variables:
            blk = s6[task][v]
            if task == 'B':
                maes = blk['mae_mgl'] if v == 'oxygen' else blk['mae']
            else:
                maes = blk['mae']   # mg/l as stored
            cells = {a: simple_cell(fam, fl, task, v, 'Reserved-2026', '6', a, {'MAE': r6(maes[a])},
                                    'MODEL_v2', 'B1_clim_increment')
                     for a in ARMS}
            refs = {r: r6(maes[r]) for r in ('MODEL_v2', 'B0_persistence', 'B1_clim_increment') if r in maes}
            out[task][v] = {'Reserved-2026 6 h': {'arms': cells, 'references_MAE': refs, 'rows': blk['rows'],
                                                  'unit': 'degC' if v == 'temperature' else 'mg/l'}}
            k34[task][v] = {'Reserved-2026 6 h': k_cells(fam, fl, task, v, 'Reserved-2026', '6')}
            order = sorted(ARMS, key=lambda a: maes[a])
            rank[task][v] = {'Reserved-2026 6 h': ranking(fam, fl, task, v, 'Reserved-2026', '6', order, maes)}
    return out, k34, rank


def task_c(sd, fam, fl):
    out, k34, rank = {}, {}, {}
    C = sd['C']['scores']
    for v in VARIABLES:
        out[v], k34[v], rank[v] = {}, {}, {}
        for cohort in C[v]:
            for lead in C_LEADS:
                cell = C[v][cohort][lead]
                name = f'{cohort} {lead} h'

                def stats(s):
                    m = s.get('mgl', {}) if v == 'oxygen' else s
                    return {'quantile_score9': r6(m['quantile_score9']), 'coverage80': r6(s['coverage80']),
                            'median_mae': r6(m['median_mae']), 'rows': s['rows']}
                cells = {a: simple_cell(fam, fl, 'C', v, cohort, lead, a, stats(cell['arms'][a]),
                                        'clim_quantiles', 'quantile_tree') for a in ARMS}
                refs = {r: stats(cell['published_arms'][r]) for r in ('clim_quantiles', 'quantile_tree')}
                out[v][name] = {'arms': cells, 'references': refs, 'rows_common': cell['rows_common'],
                                'units_common': cell['units_common'],
                                'unit': 'degC' if v == 'temperature' else 'mg/l'}
                k34[v][name] = k_cells(fam, fl, 'C', v, cohort, lead)
                qs = {a: cells[a]['metric']['quantile_score9'] for a in ARMS}
                rank[v][name] = ranking(fam, fl, 'C', v, cohort, lead, sorted(ARMS, key=qs.get), qs)
    return out, k34, rank


def task_d(sd, fam, fl):
    out, k34, rank = {}, {}, {}
    D = sd['D']
    for v in VARIABLES:
        main = D['main_reference'][v]
        out[v], k34[v], rank[v] = {}, {}, {}
        for cohort in D['scores'][v]:
            for lead in D_LEADS:
                cell = D['scores'][v][cohort]['by_lead'][lead]
                key = 'mae_mgl' if v == 'oxygen' else 'mae'
                maes = {a: cell['mae'][a][key] for a in cell['mae']}
                name = f'{cohort} {lead} h'
                cells = {a: simple_cell(fam, fl, 'D', v, cohort, lead, a, {'MAE': r6(maes[a])}, main, 'persistence')
                         for a in ARMS}
                out[v][name] = {'arms': cells, 'main_reference': main,
                                'references_MAE': {r: r6(maes[r]) for r in D['references'][v] if r in maes},
                                'episodes': cell['n'], 'unit': 'degC' if v == 'temperature' else 'mg/l'}
                k34[v][name] = k_cells(fam, fl, 'D', v, cohort, lead)
                rank[v][name] = ranking(fam, fl, 'D', v, cohort, lead, sorted(ARMS, key=maes.get), maes)
    return out, k34, rank


def task_e(sg, fam, fl):
    out, k34, rank = {}, {}, {}
    R = sg['results']
    for v in VARIABLES:
        out[v], k34[v], rank[v] = {}, {}, {}
        for cohort, cl in [(c, E_BAND) for c in E_STRICT] + [(c, L) for c in E_LONG for L in E_LENGTHS]:
            cell = R[cohort][v]['cells'].get(cl)
            name = f'{cohort} {cl} h'
            if cell is None:
                out[v][name] = {'reading': 'cell not in the score file'}
                continue
            key = 'mae_mgl' if v == 'oxygen' else 'mae'
            maes = {f: cell['mae'][f][key] for f in cell['mae']}
            inc_alias = cell['K1']['chronos_bolt_base_uni.bidir']['minus_incumbent']
            incumbent = cell['incumbent']
            if inc_alias.startswith('minus_linear'):
                incumbent = 'linear'
            elif inc_alias != f'minus_{incumbent}':
                raise SystemExit(f'E {cohort} {v} {cl}: incumbent {incumbent}, alias {inc_alias}')
            strict = cohort in E_STRICT
            cells = {a: simple_cell(fam, fl, 'E', v, cohort, cl, a, {'MAE': r6(maes[a + '.bidir'])}, 'linear',
                                    incumbent if strict else None, suffix='.bidir',
                                    extra={'fill': 'bidirectional'})
                     for a in ARMS}
            if not strict and incumbent != 'linear':
                for a in ARMS:
                    rec = fam.get(('E', v, cohort, cl, a + '.bidir', incumbent))
                    cells[a]['vs_incumbent_stored'] = view(rec, fl)
            refs = {r: r6(maes[r]) for r in ('linear', 'gpr', 'chronos_bidir', 'chronos2_cov_bidir') if r in maes}
            out[v][name] = {'arms': cells, 'incumbent': incumbent, 'references_MAE': refs,
                            'units': cell.get('units'), 'unit_of_resampling': cell.get('unit'),
                            'episodes': cell.get('episodes'), 'unit': 'degC' if v == 'temperature' else 'mg/l'}
            k34[v][name] = k_cells(fam, fl, 'E', v, cohort, cl, suffix='.bidir')
            m = {a: maes[a + '.bidir'] for a in ARMS}
            rank[v][name] = ranking(fam, fl, 'E', v, cohort, cl, sorted(ARMS, key=m.get), m, suffix='.bidir')
    return out, k34, rank


def main():
    srcs = {k: load(p) for k, p in SOURCES.items()}
    checks = {k: load(p) for k, p in CHECKS.items()}
    s6, sd, sg = srcs['6h'], srcs['dist_bridge'], srcs['gaps']
    deciding, informative, failed = gates(s6, sd, sg, checks)
    if failed:
        raise SystemExit('v2_fm_zoo_summary: a gate failed, nothing is read: '
                         + json.dumps(failed, indent=1))
    c = factor()
    cs = {'6h': s6['c_mg_per_l_per_umol_kg'], 'dist_bridge': sd['oxygen_mg_per_l_per_stored_unit'],
          'gaps': sg['rules']['oxygen_factor_mg_per_l_per_umol_kg']}
    if any(abs(x - c) > 1e-15 for x in cs.values()):
        raise SystemExit(f'oxygen factor differs from src.units.factor() = {c}: {cs}')
    for name, d in (('6h', s6['arms_scored']), ('dist_bridge', sd['arms']), ('gaps', sg['arms'])):
        if list(d) != ARMS:
            raise SystemExit(f'{name}: arms {d}, expected {ARMS}')
    fl = floors()

    # addendum 82, part C: the Gap-2026 cells of 1, 2 and 3 h and the 1-6 h band over 1, 2, 3 and 6 h, scored
    # by the same gap scorer after its own gates passed; the band replaces the published band of 6 h alone
    pc = load(PART_C)
    if not (pc['gates'].get('passed') is True and pc['gates'].get('R_replica') is True):
        raise SystemExit('v2_fm_zoo_summary: part C\'s gap cells did not pass their gates')
    for v in VARIABLES:
        if sorted(pc['cells'][v]) != sorted(PART_C_CELLS):
            raise SystemExit(f'v2_fm_zoo_summary: part C {v}: cells {sorted(pc["cells"][v])}')
        if pc['cells'][v]['1-6']['gap_lengths'] != [1, 2, 3, 6]:
            raise SystemExit(f'v2_fm_zoo_summary: part C {v}: the 1-6 h band is not over 1, 2, 3 and 6 h')
        sg['results'][PART_C_COHORT][v]['cells'].update(pc['cells'][v])
    flag(deciding['gaps'], 'v2_addendum82/C/pretrained.json/gates', pc['gates']['passed'],
         'part C: the 1-3 h fills reproduce its Chronos columns (G1), every file present (G2), replica (R)')

    fam = Family(c)
    collect(s6, sd, sg, fam)
    fdr = adjust(fam)

    a, ak, ar = task_a(s6, fam, fl, c)
    bf, bfk, bfr = task_bf(s6, fam, fl)
    cc, ck, cr = task_c(sd, fam, fl)
    dd, dk, dr = task_d(sd, fam, fl)
    ee, ek, er = task_e(sg, fam, fl)

    def contrast_out(r):
        o = {k: r[k] for k in ('id', 'task', 'variable', 'cohort', 'cell', 'a', 'b', 'K', 'status', 'units', 'rows',
                               'unit_stored')}
        o['stored'] = [r6(x) for x in r['stored']]
        o['point_mgl_or_degC'] = r6(r['point'])
        o['ci_mgl_or_degC'] = None if r['lo'] is None else [r6(r['lo']), r6(r['hi'])]
        for k in ('p', 'p_mid'):
            o[k] = g9(r.get(k))
        for k in ('bh', 'by', 'bh_mid', 'by_mid', 'robust', 'jackknife_excludes_zero', 'reproduction',
                  'degenerate_interval', 'p_side_outside'):
            if k in r and r[k] is not None:
                o[k] = r[k]
        return o

    report = {
        'declared_in': {DECLARED: sha256(ROOT / DECLARED), **{p: sha256(ROOT / p) for p in AMENDMENTS}},
        'producer': 'scripts/v2_fm_zoo_summary.py',
        'sources': {rel(p): sha256(p) for p in list(SOURCES.values()) + list(CHECKS.values()) + [PART_C]},
        'gap_cells_added': {'source': rel(PART_C), 'cohort': PART_C_COHORT, 'cells': list(PART_C_CELLS),
                            'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json',
                            'rule': 'gaps of 1, 2 and 3 h on Gap-2026 scored by the same gap scorer; the 1-6 h '
                                    'band over 1, 2, 3 and 6 h replaces the band of 6 h alone'},
        'arms': ARMS,
        'oxygen': {'unit': 'mg/l', 'c_mg_per_l_per_umol_kg': c,
                   'rule': "the scorer's own mg/l fields where stored (A, B, C, D, E contrasts; B, C, D, E "
                           "errors), else the stored value x c (task A errors); task F is scored in mg/l"},
        'floors': {**{k: r6(v) for k, v in fl.items()},
                   'source': 'scripts/v2_summary_table.floors (the floors Table 6.1 reads)',
                   'rule': '0.2028 degC where an arm of the contrast reads other instruments in tasks D and E '
                           '(covariate arms, the anchored continuation, the Gaussian process, the published '
                           'Chronos-2 covariate fill); 0.029 degC otherwise; 0.059 mg/l for oxygen. '
                           "'below' and 'exceeds' read the point, 'clears' the whole interval on one side",
                   'note': 'the task C scorer reads its covariate arms against 0.2028 degC; this summary reads '
                           'tasks A, B, C and F against 0.029 degC throughout'},
        'gates': {'all_deciding_passed': True,
                  'counts': {k: len(v) for k, v in deciding.items()},
                  'deciding': deciding, 'informative': informative},
        'headline_cells': {
            'A': 'Dev-2025 6 h at six phases: MAE at offset 0 and over the phases; arm - per-fold tree as the phase '
                 'mean with the verdict over the six phases (and arm - shared tree)',
            'B': 'Reserved-2026 6 h: MAE; arm - released model (main), arm - climatological increment',
            'F': 'Reserved-2026 6 h AOU (mg/l): MAE; arm - released model (main), arm - climatological increment',
            'C': 'distribution at 6, 24 and 72 h per cohort: nine-level quantile score, 80 % coverage; arm - '
                 'climatological quantiles (main), arm - quantile tree at seed 42',
            'D': 'silent probe at 6, 24, 48 and 72 h per cohort: MAE; arm - anchored continuation (temperature) or '
                 'hour-of-day increment (oxygen) (main), arm - last value',
            'E': 'bidirectional fill; strict cohorts the 12-72 h band, arm - linear (main) and arm - incumbent; long '
                 'cohorts at 168, 336 and 720 h, arm - linear; bootstrap with the delete-one-unit jackknife'},
        'matrix': {'A': a, 'B': bf['B'], 'F': bf['F'], 'C': cc, 'D': dd, 'E': ee},
        'k3_k4': {'A': ak, 'B': bfk['B'], 'F': bfk['F'], 'C': ck, 'D': dk, 'E': ek},
        'rankings': {'A': ar, 'B': bfr['B'], 'F': bfr['F'], 'C': cr, 'D': dr, 'E': er},
        'fdr': {**fdr, 'contrasts': [contrast_out(r) for r in fam.items.values()]},
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    n = sum(len(v) for v in deciding.values())
    print(f'gates: {n} deciding, all passed; {len(informative)} informative')
    k = fdr['counts']
    print(f"family: {k['intervals']} intervals, {k['exclude_zero']} exclude zero, BH keeps {k['bh_retained']}, "
          f"BY keeps {k['by_retained']}; {fdr['not_estimable_no_interval']} n.e.; "
          f"{fdr['merged_duplicates']} duplicates merged")
    for f, t in fdr['by_family'].items():
        print(f'   {f}: {t}')
    print('wrote', rel(OUT))


if __name__ == '__main__':
    main()
