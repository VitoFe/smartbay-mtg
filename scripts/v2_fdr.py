"""Addendum 79: false-discovery control over the answers of Table 6.1 and over every interval in the tables.

Each interval in the thesis is a 95 % interval for one contrast, reported without correction for
multiplicity. Over many contrasts with no real effect about one in twenty excludes zero by chance, so
this script asks which of the starred results survive a false-discovery procedure, over two families
fixed by the declaration before anything adjusted was computed:

  F1  the estimates printed in Table 6.1, read through the row builders of scripts/v2_summary_table.py
      (one test per printed estimate; a row over leads or gap lengths gives one test per cell that has an
      interval);
  F2  every interval printed in a table file that thesis.tex or supplement.tex inputs, except the
      declared intervals of a level, counted once each (F2t: the thesis's tables alone).

A p-value comes from the interval itself, by a normal approximation to the bootstrap tail: for F1 from
the estimate and the interval's end towards zero, for F2 from the interval's midpoint and half-width,
since many table cells print an interval without its estimate. Both give p < 0.05 exactly when the
interval excludes zero, so the unadjusted reading is the printed star. Benjamini-Hochberg at q = 0.05
is the primary procedure and Benjamini-Yekutieli, valid under any dependence, is reported beside it.

Declared in config/bay_reserved_protocol_v2_addendum79.json. Nothing is refitted.
Usage: python -m scripts.v2_fdr
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import norm

import scripts.v2_summary_table as S

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'thesis_template'
OUT = ROOT / 'reports/bay_twin/runs/v2_fdr.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum79.json'
Q = 0.05
Z = float(norm.ppf(0.975))
TOLERANCE = 1e-12

#: the rows of Table 6.1 that span leads or gap lengths: each is the selected method of one task and band minus
#: the task's reference, and each cell's interval is the one runs/v2_best_by_task.json stores for the 2026
#: cohort (oxygen in mg/l, as the row prints it); a cell without an interval is not a test
RANGED = {'forecast_t': ('forecast', 'temperature', '1-72'),
          'forecast_o': ('forecast', 'oxygen', '1-72'),
          'bridge_last': ('silent_probe', 'temperature', '6-72'),
          'silent_o': ('silent_probe', 'oxygen', '6-72'),
          'gap_long_t': ('gaps', 'temperature', '168-1440'),
          'gap_long_o': ('gaps', 'oxygen', '168-1440')}


def ranged_cells(key):
    """(cell label, point, lo, hi) for each lead or gap length a ranged row of Table 6.1 spans, in the
    order the row builder passes them to span()."""
    task, variable, band = RANGED[key]
    b = S.at('runs/v2_best_by_task.json', 'tasks', task, variable, band)
    out = []
    for cell in b['cells']:
        e = b['per_cell'][cell]['2026']['contrast']
        if e.get('ci') is not None:
            out.append((f'{cell} h', float(e['point']), float(e['ci'][0]), float(e['ci'][1])))
    return out


def capture_table():
    """Run the summary table's row builders with contrast(), span() and row() wrapped, and return the
    printed rows with what each passed to contrast() or span()."""
    calls, rows = [], {}
    contrast, span, row = S.contrast, S.span, S.row

    def wrapped_contrast(point, lo, hi, d, m=''):
        calls.append(('contrast', point, lo, hi))
        return contrast(point, lo, hi, d, m)

    def wrapped_span(points, d, marks=None):
        calls.append(('span', list(points)))
        return span(points, d, marks)

    def wrapped_row(key, result, cohort, text, change, verdict):
        r = row(key, result, cohort, text, change, verdict)
        rows[key] = {'calls': list(calls), 'text': text, 'label': result, 'cohort': cohort}
        calls.clear()
        return r

    S.contrast, S.span, S.row = wrapped_contrast, wrapped_span, wrapped_row
    try:
        S._ROW.clear()
        groups = S.printed([(title, build()) for title, build in S.GROUPS])
    finally:
        S.contrast, S.span, S.row = contrast, span, row
    printed = [r['key'] for _, rs in groups for r in rs]
    return printed, rows


def family_answers():
    """F1 and gate GA."""
    printed, rows = capture_table()
    tests, gate = [], []
    for key in printed:
        r = rows[key]
        con = [c for c in r['calls'] if c[0] == 'contrast' and c[2] is not None]
        spans = [c for c in r['calls'] if c[0] == 'span']
        if key in RANGED:
            cells = ranged_cells(key)
            ok = (len(spans) == 1 and len(spans[0][1]) == len(cells) and not con
                  and all(abs(a - c[1]) <= TOLERANCE for a, c in zip(spans[0][1], cells)))
            gate.append({'row': key, 'kind': 'range', 'cells': len(cells), 'passed': bool(ok)})
            for cell, point, lo, hi in cells:
                tests.append({'row': key, 'label': r['label'], 'cell': cell, 'point': point, 'lo': lo, 'hi': hi})
        elif con:
            if len(con) != 1:
                gate.append({'row': key, 'kind': 'single', 'passed': False, 'reason': f'{len(con)} contrast calls'})
                continue
            _, point, lo, hi = con[0]
            gate.append({'row': key, 'kind': 'single', 'passed': True})
            tests.append({'row': key, 'label': r['label'], 'cell': None, 'point': point, 'lo': lo, 'hi': hi})
        else:
            gate.append({'row': key, 'kind': 'no interval', 'passed': True, 'excluded': True})
    return tests, gate


# ---------------------------------------------------------------------------------- p-values

def p_side(point, lo, hi):
    """Two-sided p from the estimate and the interval's end towards zero; None if the estimate lies
    outside its own interval on that side."""
    if point == 0:
        return 1.0
    s = (hi - point) / Z if point < 0 else (point - lo) / Z
    if s <= 0:
        return None
    return float(2 * norm.sf(abs(point) / s))


def p_mid(lo, hi):
    """Two-sided p from the interval's midpoint and half-width."""
    if hi <= lo:
        return None
    m, se = (lo + hi) / 2, (hi - lo) / (2 * Z)
    return float(2 * norm.sf(abs(m) / se))


def step_up(p, dependence_robust=False):
    """Benjamini-Hochberg (or Benjamini-Yekutieli) rejections at Q, as a boolean list in input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return []
    c = np.sum(1.0 / np.arange(1, m + 1)) if dependence_robust else 1.0
    order = np.argsort(p, kind='stable')
    thresholds = Q * np.arange(1, m + 1) / (m * c)
    below = np.nonzero(p[order] <= thresholds)[0]
    reject = np.zeros(m, dtype=bool)
    if len(below):
        reject[order[:below[-1] + 1]] = True
    return reject.tolist()


def excludes(lo, hi):
    return lo > 0 or hi < 0


# ------------------------------------------------------------------------- F2: the table files

BRACKET = re.compile(r'\[\s*\$?\s*([-+]?\d+(?:\.\d+)?)\s*\$?\s*,\s*\$?\s*([-+]?\d+(?:\.\d+)?)\s*\$?\s*\]')
COMMENT = re.compile(r'(?<!\\)%.*')
#: unsigned intervals of a contrast, printed in a note's sentence ("worse by 0.277 [0.154, 0.404]")
UNSIGNED_CONTRASTS = {('table_bay_placement.tex', 0.154, 0.404), ('table_bay_placement.tex', 0.103, 0.287),
                      ('table_bay_physics_comparisons.tex', 0.00123, 0.01058)}
#: tables whose unsigned intervals are of a level: the floor's medians and means, recall and precision
LEVEL_TABLES = {'table_bay_oxygen_floor.tex', 'table_bay_warning_episodes.tex'}


def table_files():
    """The table files each document inputs, following \\input as check_references.py does."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('check_references', TEX / 'check_references.py')
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    tables = lambda files: [p for p in files if p.parent.name == 'tables']
    thesis = tables(cr.walk(cr.resolve('thesis.tex')))
    supplement = tables(cr.walk(cr.resolve('supplement.tex')))
    return thesis, supplement


def family_tables():
    thesis, supplement = table_files()
    in_thesis = {p.name for p in thesis}
    seen, kept, levels, unknown, reversed_ = {}, [], [], [], []
    for path in list(dict.fromkeys(thesis + supplement)):
        body = COMMENT.sub('', path.read_text(encoding='utf-8'))
        for m in BRACKET.finditer(body):
            a_txt, b_txt = m.group(1), m.group(2)
            a, b = float(a_txt), float(b_txt)
            signed = a_txt[0] in '+-' or b_txt[0] in '+-'
            entry = {'file': path.name, 'lo': a, 'hi': b, 'text': m.group(0)}
            if not signed:
                if (path.name, a, b) in UNSIGNED_CONTRASTS:
                    pass
                elif path.name in LEVEL_TABLES:
                    levels.append(entry)
                    continue
                else:
                    unknown.append(entry)
                    continue
            if a > b:
                reversed_.append(entry)
                continue
            if (a, b) in seen:
                seen[(a, b)]['copies'].append(path.name)
                continue
            entry['copies'] = [path.name]
            seen[(a, b)] = entry
            kept.append(entry)
    for e in kept:
        e['in_thesis'] = any(f in in_thesis for f in e['copies'])
    return kept, {'levels_excluded': levels, 'unsigned_unlisted_excluded': unknown,
                  'reversed_excluded': reversed_, 'table_files': {'thesis': len(thesis),
                                                                   'supplement': len(supplement)}}


def adjust(entries, key):
    """Add each procedure's decision under p-value `key` and return the counts."""
    p = [e[key] for e in entries]
    bh, by = step_up(p), step_up(p, dependence_robust=True)
    for e, r1, r2 in zip(entries, bh, by):
        e[f'bh_{key}'] = bool(r1 and e['excludes_zero'])
        e[f'by_{key}'] = bool(r2 and e['excludes_zero'])
    return {'tests': len(entries), 'exclude_zero': sum(e['excludes_zero'] for e in entries),
            'bh_retained': sum(e[f'bh_{key}'] for e in entries), 'by_retained': sum(e[f'by_{key}'] for e in entries)}


def main():
    tests, gate = family_answers()
    ga = all(g['passed'] for g in gate)
    if not ga:
        raise SystemExit(f'v2_fdr: gate GA failed, nothing is read: {[g for g in gate if not g["passed"]]}')
    for t in tests:
        t['excludes_zero'] = excludes(t['lo'], t['hi'])
        ps = p_side(t['point'], t['lo'], t['hi'])
        t['p_side_outside'] = ps is None
        t['p_side'] = ps if ps is not None else p_mid(t['lo'], t['hi'])
        t['p_mid'] = p_mid(t['lo'], t['hi'])
        assert (t['p_side'] < 0.05) == t['excludes_zero'] or t['p_side_outside'], t
    f1_side, f1_mid = adjust(tests, 'p_side'), adjust(tests, 'p_mid')
    sensitive = [f"{t['row']} {t['cell'] or ''}".strip() for t in tests
                 if t['bh_p_side'] != t['bh_p_mid'] or t['by_p_side'] != t['by_p_mid']]
    lost_bh = [t for t in tests if t['excludes_zero'] and not t['bh_p_side']]
    lost_by = [t for t in tests if t['excludes_zero'] and not t['by_p_side']]

    entries, excluded = family_tables()
    for e in entries:
        e['excludes_zero'] = excludes(e['lo'], e['hi'])
        e['p_mid'] = p_mid(e['lo'], e['hi'])
    f2 = adjust(entries, 'p_mid')
    thesis_entries = [dict(e) for e in entries if e['in_thesis']]
    f2t = adjust(thesis_entries, 'p_mid')

    report = {
        'declared_in': DECLARED, 'q': Q,
        'gates': {'GA': {'passed': ga, 'rows': gate}},
        'F1_answers': {
            'counts_side': f1_side, 'counts_mid': f1_mid,
            'bh_not_retained': [{'row': t['row'], 'cell': t['cell'], 'label': t['label'], 'p': t['p_side']}
                                for t in lost_bh],
            'by_not_retained': [{'row': t['row'], 'cell': t['cell'], 'label': t['label'], 'p': t['p_side']}
                                for t in lost_by],
            'sensitive_to_the_approximation': sensitive,
            'estimate_outside_its_interval': [t['row'] for t in tests if t['p_side_outside']],
            'tests': tests},
        'F2_tables': {'counts': f2, 'counts_thesis_tables': f2t, 'excluded': excluded,
                      'p_threshold_bh': max([e['p_mid'] for e in entries if e['bh_p_mid']], default=None),
                      'p_threshold_by': max([e['p_mid'] for e in entries if e['by_p_mid']], default=None),
                      'intervals': entries},
    }
    starred = f1_side['exclude_zero']
    report['registered_predictions'] = {
        'FP1': {'held': starred - f1_side['bh_retained'] <= 2,
                'statement': 'BH over F1 retains every starred estimate but at most two'},
        'FP2': {'held': f1_side['by_retained'] < f1_side['bh_retained'],
                'statement': 'BY over F1 retains at least one starred estimate fewer than BH'},
        'FP3': {'held': f2['bh_retained'] >= 0.8 * f2['exclude_zero'],
                'statement': 'BH over F2 retains at least 80 % of the intervals that exclude zero'}}
    OUT.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f"F1: {f1_side['tests']} tests, {starred} exclude zero, BH keeps {f1_side['bh_retained']}, "
          f"BY keeps {f1_side['by_retained']}; midpoint rule: BH {f1_mid['bh_retained']}, BY {f1_mid['by_retained']}")
    for t in lost_bh:
        print(f"   BH drops {t['row']} {t['cell'] or ''}  p = {t['p_side']:.4f}")
    for t in lost_by:
        print(f"   BY drops {t['row']} {t['cell'] or ''}  p = {t['p_side']:.4f}")
    print(f"   sensitive to the approximation: {sensitive}")
    print(f"F2: {f2['tests']} intervals, {f2['exclude_zero']} exclude zero, BH keeps {f2['bh_retained']}, "
          f"BY keeps {f2['by_retained']}; thesis tables {f2t}")
    print(f"   excluded: {len(excluded['levels_excluded'])} levels, "
          f"{len(excluded['unsigned_unlisted_excluded'])} unlisted unsigned, {len(excluded['reversed_excluded'])} reversed")
    print('predictions', {k: v['held'] for k, v in report['registered_predictions'].items()})
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
