"""The best model per task, chosen by one rule on the development cohorts and read on 2026.

Four tasks, each in temperature and oxygen, each against its reference:

  station        a station without a probe, Field-57 per outer fold and Field-2026, held-out stations;
                 reference: the spatial estimator without a depth offset (A2)
  forecast       forecasting at a probe, 1-72 h, Dev-2025 and Reserved-2026 on addendum 85's common rows
                 (part B); reference: the last value
  silent_probe   a probe that falls silent, 6-72 h after onset, Bridge-Dev and Bridge-2026 on addendum 85's
                 common rows (part A); reference: the last value for temperature, the hour-of-day increment
                 for oxygen
  gaps           a gap filled once both edges are known, bands 1-6 h and 12-72 h (Gap-Dev and Gap-2026) and
                 168-1440 h (Gap-Long-Dev and Gap-Long-2026) on addendum 85's common rows (part D); reference:
                 linear interpolation

The rule, the same for every task, variable and band: the candidates are the arms that have a value in
every cell of the band in both the development and the 2026 cohort (a core arm of the cell, scored on its
common rows, where the source defines common rows), other than the reference, arms that read observations a
live system would not have, arms the thesis does not report, the shared selection of an arm that is also
selected per outer fold, and the version of the cast-derived depth offset fitted on all training
stations (its offset is fitted on labels inside the inner split, so the version fitted inside each inner
split is the primary one). A candidate's score is the mean over the band's cells of its development MAE
over the reference's development MAE in the same cell. The named arm is the candidate with the lowest
score, so it is never the reference; where its score is not below one, no candidate beats the reference on
development and the record says so.

For the named arm, per cell: the development MAE and the difference from the runner-up (the next candidate
whose values are not identical to the named arm's in every cell), the 2026 MAE, the reference's 2026 MAE,
the 2026 contrast against the reference exactly as the source stores it (point, 95 % interval, jackknife
where stored; null where the source stores none; no interval is computed here), the margin that applies,
whether the interval excludes zero, and the lowest 2026 MAE among the candidates and the reference with the
named arm's shortfall to it. Every candidate's score and per-cell MAEs are kept.

Margins: the stored contrast's own floor for parts A, B and D (the quantisation uncertainty, 0.029 degC, for
arms that read the probe's own record and forcing; the co-located margin, 0.2028 degC, for temperature arms
that read other probes; the oxygen floor, 0.059 mg/l); for the station task the cast floor of each cohort
(the Gen. 2 casts before 2026 for Field-57, the whole archive for Field-2026), as Chapter 4's margins table
assigns them.

Station sources. Temperature: the per-fold Field-57 arms (runs/v2_perfold_a69.json and
runs/v2_perfold_rest_a69.json) and the Field-2026 reports as published (runs/v2_field_reserved2026/innerprior
for the cast-derived depth offset fitted inside each inner split, runs/v2_field_reserved2026_copernicus for
the Copernicus arms; four decimals, as published). Oxygen: the mg/l reruns of the same analyses
(runs/v2_oxygen_mgl/C1 and C2), because the casts mix per-kilogram and per-litre sources. The one exception
is the cast-derived depth offset fitted inside each inner split on Field-2026, of which no mg/l rerun
exists: its oxygen MAE and contrast are c times the stored values, as the thesis table prints them, and the
record says so. A second station entry applies the rule without the arms that read the Copernicus product,
which is not available after 16 January 2026 without the extraction rule.

Gates, refused unless all hold: addendum 85's gates for the parts read (as the summary and the part files
record them); the reference's MAEs read here equal the values the same source stores elsewhere (the
summary's compact tables for parts A, B and D; the published and reproduced A2 for the station); every
stored contrast of a candidate against the reference equals the difference of the two MAEs read here, to
the stored rounding; the oxygen values in mg/l equal c times the stored values where the source keeps both;
and the named arm is present in every cell of its band in both cohorts.

2026 is the test period. As a check of how the development choice generalises, the same candidates are
also ranked on 2026 by the same score taken on 2026 (the mean over the band's cells of the arm's 2026 MAE
over the reference's 2026 MAE), leaving out any cell with fewer than five independent units on 2026,
which is listed. Each band records the 2026 winner, its development rank, the development winner's 2026
rank and score, whether the two coincide, and the gap between their 2026 scores; a development winner
ranked low on 2026 was good on development and worse on 2026. The 2026 winner is chosen on 2026, so its
own 2026 error is not an unbiased estimate, and it does not replace the development winner. Oxygen is
printed in mg/l; temperature in degC. Values are rounded to six decimals once, here.

Writes reports/bay_twin/runs/v2_best_by_task.json.
Usage: python -m scripts.v2_best_by_task
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_gap_tables import FAMILIES as GAP_FAMILIES  # noqa: E402
from scripts.v2_gap_tables import NAMES as GAP_NAMES  # noqa: E402
from src.units import UNITS as UNITS_FILE, factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
A85 = RUNS / 'v2_addendum85'
MGL = RUNS / 'v2_oxygen_mgl'
OUT = RUNS / 'v2_best_by_task.json'
SCRIPT = 'scripts/v2_best_by_task.py'
DECLARED_85 = 'config/bay_reserved_protocol_v2_addendum85.json'

SRC = {
    'summary_85': RUNS / 'v2_addendum85.json',
    'part_A': A85 / 'A.json',
    'part_B': A85 / 'B.json',
    'part_D': A85 / 'D.json',
    'field57_temperature': RUNS / 'v2_perfold_a69.json',
    'field57_rest_temperature': RUNS / 'v2_perfold_rest_a69.json',
    'field57_oxygen': MGL / 'C1/v2_perfold_a69.json',
    'field57_rest_oxygen': MGL / 'C1/v2_perfold_rest_a69.json',
    'field2026_inner': RUNS / 'v2_field_reserved2026/innerprior/report.json',
    'field2026_inner_copernicus': RUNS / 'v2_field_reserved2026/innerprior/copernicus_contrast.json',
    'field2026_copernicus_temperature': RUNS / 'v2_field_reserved2026_copernicus/report.json',
    'field2026_copernicus_oxygen': MGL / 'C1/v2_field_reserved2026_copernicus/report.json',
    'field2026_cast_oxygen': MGL / 'C2/v2_field_reserved2026/report.json',
    'cast_floor_temperature': RUNS / 'v2_ctd_floor.json',
    'cast_floor_oxygen': MGL / 'C2/v2_ctd_floor.json',
    'fm_zoo': RUNS / 'v2_fm_zoo.json',
    'gap_variants': RUNS / 'v2_addendum82/C/rescored/v2_gap_variants.json',
    'oxygen_units': UNITS_FILE,
}

C = factor()
VARIABLES = ('temperature', 'oxygen')
UNIT = {'temperature': 'degC', 'oxygen': 'mg/l'}
CORE_SHARE = 0.95
MIN_UNITS = 5
#: stored values carry six decimals: a difference of two of them and a stored point differ by at most 1e-6
TOL6 = 2e-6
#: the published Field-2026 reports keep four decimals
TOL4 = 1.5e-4

#: arms that read observations a live system would not have at the target hour (the perfect-weather arms
#: of the distribution study); none is among the sources read here, and any that appears is excluded
INADMISSIBLE = {'quantile_tree_weather_future': 'reads the observed weather after issuance (a perfect forecast)',
                'chronos2_net_weather_future': 'reads the observed weather after issuance (a perfect forecast)'}
#: scored but not reported: neither had the 100-trial study its free parameter needs
NOT_REPORTED = {'lstm': 'not reported: no 100-trial study for its free parameter',
                'compressed_sensing': 'not reported: run at a fixed penalty, no 100-trial study'}
#: the shared selection where the same arm is also selected per outer fold (the per-fold one is primary)
SHARED = {'tree_shared': 'tree', 'weather_tree_shared': 'weather_tree'}

# --------------------------------------------------------------------------------------------- station
#: Field-57 per-fold arm -> (label, Field-2026 report, its name there, reads the Copernicus product)
STATION_ARMS = {
    'A2_none_direct': ('no depth offset', 'inner', 'A2_none', False),
    'A1_both_direct': ('Copernicus depth offset and regressors (the spatial estimator with the Copernicus '
                       'depth offset)', 'copernicus', 'A1_both', True),
    'A4_base_only_direct': ('Copernicus depth offset only', 'copernicus', 'A4_base_only', True),
    'A7_ctdprior_direct': ('cast-derived depth offset, fitted inside each inner split', 'inner', 'A7o', False),
    'A3_design_only_direct': ('regressors only, no depth offset', None, None, False),
    'A5_intercept_direct': ('Copernicus depth offset and regressors with a fitted intercept', None, None, True),
    'A6_ctdprior_direct': ('cast-derived depth offset by month and depth', None, None, False),
    'P1_placebo_direct': ('fitted depth basis (placebo)', None, None, False),
    'P2_placebo_direct': ('random depth basis (placebo)', None, None, False),
    'A1_both_aou': ('Copernicus depth offset and regressors, oxygen as apparent utilisation', None, None, True),
}
#: Field-2026 arms with no per-fold Field-57 counterpart: name here -> (label, report, name there)
STATION_2026_ONLY = {
    'A7o_all_stations': ('cast-derived depth offset fitted on all training stations',
                         'copernicus', 'A7o'),
    'seasonal': ('seasonal climatology of the pre-2026 archive', 'copernicus', 'seasonal'),
    'background_A2_none': ('the background alone, no depth offset', 'inner', 'background_A2_none'),
    'background_A7o_inner': ('the background alone with the cast-derived depth offset fitted inside each '
                             'inner split', 'inner', 'background_A7o'),
}
STATION_EXCLUDED = {'A7o_all_stations': 'the version of the cast-derived depth offset fitted on all '
                                        'training stations and so on labels inside the inner split; the '
                                        'version fitted inside each inner split is the primary one'}
STATION_REF = 'A2_none_direct'
STATION_CELL = 'held_station'
TARGET = {'temperature': 'temp_c', 'oxygen': 'do_umol_l'}
#: cast floor per cohort: (source, cohort key in it)
CAST_FLOOR = {'Field-57': 'gen2_before_2026', 'Field-2026': 'CTD-722'}

# --------------------------------------------------------------------------------------------- parts
B_COHORTS = ('Dev-2025', 'Reserved-2026')
B_LEADS = (1, 3, 6, 12, 24, 36, 48, 72)
A_COHORTS = ('Bridge-Dev', 'Bridge-2026')
A_LEADS = tuple(range(6, 73, 6))
A_REF = {'temperature': 'persistence', 'oxygen': 'clim_hour'}
GAP_BANDS = (('1-6', ('Gap-Dev', 'Gap-2026'), (1, 2, 3, 6)),
             ('12-72', ('Gap-Dev', 'Gap-2026'), (12, 18, 24, 48, 72)),
             ('168-1440', ('Gap-Long-Dev', 'Gap-Long-2026'), (168, 336, 504, 720, 1440)))
#: the four references of the forecasting cells; the blended weather tree is stored against each
B_STORED_REFS = ('persistence', 'clim_month_hour', 'clim_hour', 'weather_tree')

RULE = ("For each task, variable and band of cells, the candidates are the arms that have a value in every cell "
        "of the band in both the development cohort and the 2026 cohort (on the cell's common rows where the "
        "source defines them), other than the task's reference, arms that read observations a live system "
        "would not have, arms the thesis does not report, the shared selection of an arm that is also selected "
        "per outer fold, and the declared all-stations reading of the cast-derived depth offset. A candidate's "
        "score is the mean over the band's cells of its development MAE divided by the reference's development "
        "MAE in the same cell. The named arm is the candidate with the lowest score, so it is never the "
        "reference; where its score is not below one, no candidate beats the reference on development and the "
        "reference stays. 2026 is the test period: the named arm's 2026 values are read against the reference "
        "through the contrast the source stores, against the margin that applies; no interval is computed "
        "here. As a check of how the choice generalises, and to detect an arm that is good on development and "
        "worse on 2026, the same candidates are scored on 2026 in the same way (the mean over the band's cells "
        "of the arm's 2026 MAE divided by the reference's 2026 MAE, leaving out any cell with fewer than five "
        "independent units on 2026), and the 2026 winner is the candidate with the lowest 2026 score. The 2026 "
        "winner is chosen on 2026, so its own 2026 error is not an unbiased estimate, and it does not replace "
        "the development winner.")
CHECK_2026 = ("2026 is the test period. Beside each band's development winner (the named arm), the same candidates "
              "are ranked on 2026 by the development score's analogue; winner_2026 is the lowest, same_winner says "
              "whether it is the named arm (or an arm identical to it in every cell of both cohorts), and "
              "score_2026_gap is the named arm's 2026 score minus the 2026 winner's. The 2026 winner is chosen on "
              "2026, so its own 2026 error is not an unbiased estimate, and it does not replace the development "
              "winner.")


def fail(message):
    raise SystemExit(f'v2_best_by_task: {message}')


def load(path):
    path = Path(path)
    if not path.exists():
        fail(f'source missing: {rel(path)}')
    return json.loads(path.read_text(encoding='utf-8'))


def rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


def rounded(obj):
    """Every float in a nested structure rounded to six decimals (once, at the end)."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return obj


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def dig(obj, keys, where):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            fail(f'{where}: key {k!r} missing')
        obj = obj[k]
    return obj


# ------------------------------------------------------------------------------------------ the rule
def choose(dev, new, cells, reference, excluded=(), ratios=None):
    """The rule on one band. `dev` and `new` map each cell to {arm: MAE} over the arms scored there (core
    arms on the common rows where the source has them). Refused unless every cell has values in both
    cohorts and the reference is in every cell. `ratios` optionally maps a cell to {arm: development MAE
    over the reference's}, for an arm whose MAE and the reference's are on one basis only in the stored unit.
    Returns the named arm, the candidates in score order (ties broken by name), the scores and the arms
    present in every cell of both cohorts."""
    for name, block in (('development', dev), ('2026', new)):
        missing = [c for c in cells if c not in block or not block[c]]
        if missing:
            fail(f'the {name} cohort has no values for cell(s) {missing}')
        for c in cells:
            bad = [a for a, v in block[c].items() if not finite(v)]
            if bad:
                fail(f'the {name} cohort, cell {c}: no finite MAE for {bad}')
    present = set.intersection(*(set(block[c]) for block in (dev, new) for c in cells))
    if reference not in present:
        fail(f'the reference {reference} is not scored in every cell of both cohorts')
    for c in cells:
        if not dev[c][reference] > 0:
            fail(f'cell {c}: the reference has no positive development MAE')
    candidates = sorted(present - {reference} - set(excluded))
    if not candidates:
        fail('no candidate is scored in every cell of both cohorts')
    ratios = ratios or {}

    def ratio(c, a):
        r = ratios.get(c, {}).get(a)
        return dev[c][a] / dev[c][reference] if r is None else r

    scores = {a: sum(ratio(c, a) for c in cells) / len(cells) for a in candidates}
    order = sorted(candidates, key=lambda a: (scores[a], a))
    return {'named': order[0], 'order': order, 'scores': scores, 'present': sorted(present)}


def rank_2026(new, cells, reference, candidates, units=None, ratios=None):
    """The 2026 check on one band: the same candidates scored on 2026 exactly as the rule scores them on
    development (the mean over the cells of the arm's 2026 MAE over the reference's), over the cells with at
    least MIN_UNITS independent units on 2026 (`units` maps a cell to its count; None counts as estimable).
    `ratios` optionally maps a cell to {arm: 2026 MAE over the reference's} on another basis, as in choose().
    Returns the winner, the order, the scores, the cells scored and the cells left out. Chosen on 2026, the
    winner's own 2026 error is not an unbiased estimate; it never replaces the development choice."""
    units = units or {}
    ratios = ratios or {}
    if not candidates:
        fail('no candidate to rank on 2026')
    scored = [c for c in cells if units.get(c) is None or units[c] >= MIN_UNITS]
    left_out = [c for c in cells if c not in scored]
    if not scored:
        return {'winner': None, 'order': [], 'scores': {a: None for a in candidates}, 'cells': [],
                'not_estimable': left_out}
    for c in scored:
        if c not in new or reference not in new[c] or not new[c][reference] > 0:
            fail(f'2026 cell {c}: the reference has no positive MAE')
        missing = [a for a in candidates if a not in new[c] or not finite(new[c][a])]
        if missing:
            fail(f'2026 cell {c}: no finite MAE for candidate(s) {missing}')

    def ratio(c, a):
        r = ratios.get(c, {}).get(a)
        return new[c][a] / new[c][reference] if r is None else r

    scores = {a: sum(ratio(c, a) for c in scored) / len(scored) for a in candidates}
    order = sorted(candidates, key=lambda a: (scores[a], a))
    return {'winner': order[0], 'order': order, 'scores': scores, 'cells': scored, 'not_estimable': left_out}


def identical(a, b, dev, new, cells):
    """True when two arms hold the same value in every cell of both cohorts (one model under two names)."""
    return all(dev[c][a] == dev[c][b] and new[c][a] == new[c][b] for c in cells)


def margin_reading(point, ci, margin):
    """The reading of a stored contrast against a margin, by the rule the stored readings follow: an
    interval spanning zero is not read against it."""
    if ci is None:
        return 'n.e. (no interval)'
    lo, hi = ci
    if not (lo > 0 or hi < 0):
        return 'interval spans zero'
    if lo > margin or hi < -margin:
        return 'clears floor'
    return 'exceeds floor' if abs(point) > margin else 'below floor'


# ------------------------------------------------------------------------------- addendum 85's gates
def addendum85_gates(summary, parts):
    """The gates the common-row parts rest on, refused unless every one passed."""
    for p, d in parts.items():
        if d.get('declared_in') != DECLARED_85 or d.get('part') != p:
            fail(f'{rel(SRC["part_" + p])} is not part {p} of {DECLARED_85}')
    g = summary['gates']
    g14 = g['A85-G1_to_G4']
    out = {'B': {f'A85-G5_rows_B|{v}': g[f'A85-G5_rows_B|{v}'] for v in VARIABLES},
           'A': {f'A85-G5_rows_A|{v}': g[f'A85-G5_rows_A|{v}'] for v in VARIABLES},
           'D': {'A85-G6_gaps': g['A85-G6_gaps']}}
    for key in ('A85-G1_objective_reproduction', 'A85-G2_prediction_reproduction', 'A85-G3_2026_path'):
        out['B'][key] = g14[key]
    for v in VARIABLES:
        g4 = g14['A85-G4_addendum82_study'][v]
        out['B'][f'A85-G4_addendum82_study|{v}'] = bool(g4['inner_window_identical'] and g4['reuse']
                                                       and g4['abs_diff'] <= 1e-9)
        out['B'][f'joins|{v}'] = all(j['passed'] for j in parts['B']['records'][v]['joins'].values())
        out['A'][f'joins|{v}'] = all(j.get('passed', True) for j in parts['A']['records'][v]['joins'].values())
        rows = parts['A']['records'][v]['A85-G5_rows']
        out['A'][f'rows|{v}'] = (rows['missing_otherwise'] == 0 and rows['max_abs_truth_difference'] <= 1e-9
                                 and rows['max_abs_anchor_difference'] <= 1e-9)
    out['A']['A85-G6_bridge'] = g['A85-G6_bridge']['passed']
    cells = parts['D']['records']['gate_cells']
    out['D']['gate_cells'] = cells['passed'] is True and all(c['same'] for k, c in cells.items() if k != 'passed')
    for part, block in out.items():
        bad = [k for k, ok in block.items() if ok is not True]
        if bad:
            fail(f'part {part}: gate(s) {bad} did not pass')
    return out


# ------------------------------------------------------------------------------------- normalising
def stored_contrast(c, label, oxygen):
    """A contrast as the source stores it, in the printed unit (oxygen: the mg/l the source stores beside;
    a jackknife stored only per kilogram is c times it, the sondes' exact conversion), or None."""
    if c is None:
        return None
    if oxygen and 'point_mgl' not in c:
        fail(f'{label}: no mg/l value stored beside the oxygen contrast')
    ci = c.get('ci_mgl') if oxygen else c.get('ci')
    out = {'contrast': label, 'point': c['point_mgl'] if oxygen else c['point'],
           'ci': None if ci is None else list(ci), 'excludes_zero': c.get('excludes_zero'),
           'interval': '95 % percentile bootstrap over the source\'s independent unit, as stored'}
    if 'jackknife_t' in c:
        jk = c['jackknife_t']
        out['jackknife_t'] = None if jk is None else ([x * C for x in jk] if oxygen else list(jk))
        out['jackknife_excludes_zero'] = c.get('jackknife_excludes_zero')
    for k in ('estimable', 'rows', 'rows_kind', 'units', 'units_fold_week', 'episodes', 'bh', 'by'):
        if k in c:
            out[k] = c[k]
    out['margin'] = {'value': c.get('floor'), 'basis': c.get('floor_basis')}
    out['reading'] = c.get('reading')
    if 'reading_against_other_floor' in c:
        out['reading_against_other_margin'] = c['reading_against_other_floor']
    if oxygen:
        out['stored_umol_kg'] = {'point': c['point'], 'ci': c.get('ci'), 'jackknife_t': c.get('jackknife_t')}
    return out


def label_of(arm, labels):
    return labels.get(arm, arm)


def gap_label(arm, labels):
    if '.' in arm:
        base, mode = arm.split('.')
        name = labels.get(base, base).replace('+ network', '+ four nearest probes')
        return f'{name}, {"both edges" if mode == "bidir" else "forward"}'
    families = {a: lab for _, members in GAP_FAMILIES for a, lab in members}
    extra = {'nearest_live_probe': 'nearest reporting probe'}
    return GAP_NAMES.get(arm) or families.get(arm) or extra.get(arm) or labels.get(arm, arm)


# ---------------------------------------------------------------------------- generic band record
def band_record(task, variable, band, cells, cohorts, data, reference, excluded, labels, notes=None):
    """`data[cohort][cell]` = {'score': {arm: MAE in the stored unit}, 'shown': {arm: MAE printed},
    'contrast': {arm: stored contrast vs the reference, printed unit, or None}, 'units', 'unit',
    'common_rows', 'source_best', 'margin' (station only)}."""
    dev_name, new_name = cohorts
    dev = {c: data[dev_name][c]['score'] for c in cells}
    new = {c: data[new_name][c]['score'] for c in cells}
    union = set().union(*(set(dev[c]) | set(new[c]) for c in cells))
    reasons = {a: why for a, why in excluded.items() if a in union}
    ratios = {c: data[dev_name][c].get('ratio', {}) for c in cells}
    got = choose(dev, new, cells, reference, reasons, ratios)
    order, scores, named = got['order'], got['scores'], got['named']
    for c in cells:
        for cohort in cohorts:
            if named not in data[cohort][c]['score']:
                fail(f'{task} {variable} {band}: the named arm {named} is missing from {cohort} {c}')
    present = set(got['present'])
    not_every = sorted(union - present - set(reasons))
    for a in not_every:
        lacking = [name for name, block in ((dev_name, dev), (new_name, new))
                   if any(a not in block[c] for c in cells)]
        reasons[a] = f'not scored in every cell of the band in {" and ".join(lacking)}'
    if reference in union:
        reasons.pop(reference, None)
    aliases = [a for a in order[1:] if identical(a, named, dev, new, cells)]
    runner = next((a for a in order[1:] if a not in aliases), None)
    # the 2026 check: the same candidates, the same score taken on 2026
    check = rank_2026(new, cells, reference, order,
                      units={c: data[new_name][c]['units'] for c in cells},
                      ratios={c: data[new_name][c].get('ratio', {}) for c in cells})
    scores_new, order_new, winner = check['scores'], check['order'], check['winner']
    rank_new = {a: i + 1 for i, a in enumerate(order_new)}
    rank_dev = {a: i + 1 for i, a in enumerate(order)}

    per_cell = {}
    tally = {'cells': len(cells), 'below_reference': 0, 'interval_excludes_zero_favouring_named': 0,
             'interval_excludes_zero_favouring_reference': 0, 'interval_spans_zero': 0, 'no_stored_interval': 0,
             'lowest_2026': 0, 'not_estimable': 0}
    for c in cells:
        d, n = data[dev_name][c], data[new_name][c]
        pool = [a for a in order] + [reference]
        lowest = min(pool, key=lambda a: (n['shown'][a], a))
        con = n['contrast'].get(named)
        dev_con = d['contrast'].get(named)
        margin = con['margin'] if con is not None else n.get('margin')
        estimable = n['units'] is None or n['units'] >= MIN_UNITS
        if con is not None and con.get('estimable') is False:
            estimable = False
        entry = {
            'development': {'mae': d['shown'][named], 'reference_mae': d['shown'][reference],
                            'ratio_to_reference': d.get('ratio', {}).get(named,
                                                                        d['score'][named] / d['score'][reference]),
                            'runner_up_mae': None if runner is None else d['shown'][runner],
                            'named_minus_runner_up': None if runner is None else d['shown'][named] - d['shown'][runner],
                            'contrast': dev_con, 'units': d['units'], 'unit': d['unit'],
                            'common_rows': d['common_rows']},
            '2026': {'mae': n['shown'][named], 'reference_mae': n['shown'][reference],
                     'named_minus_reference': n['shown'][named] - n['shown'][reference],
                     'contrast': con,
                     'margin': margin,
                     'excludes_zero': None if con is None else con.get('excludes_zero'),
                     'reading': None if con is None else con.get('reading'),
                     'estimable': estimable,
                     'runner_up_mae': None if runner is None else n['shown'][runner],
                     'lowest': {'arm': lowest, 'label': labels(lowest), 'mae': n['shown'][lowest],
                                'is_reference': lowest == reference},
                     'shortfall_to_lowest': n['shown'][named] - n['shown'][lowest],
                     'source_lowest': {**n['source_best'], 'label': labels(n['source_best']['arm'])},
                     'units': n['units'], 'unit': n['unit'], 'common_rows': n['common_rows']},
        }
        note = n.get('arm_notes', {}).get(named) or n.get('note')
        if note:
            entry['2026']['note'] = note
        if named in d.get('record_unit', {}):
            entry['development']['note'] = ('no mg/l rerun of this arm exists: its MAE is c times the stored value, '
                                            'and its ratio to the reference is taken in the stored unit')
        if d.get('note'):
            entry['development'].setdefault('note', d['note'])
        per_cell[str(c)] = entry
        tally['below_reference'] += n['shown'][named] < n['shown'][reference]
        tally['lowest_2026'] += lowest == named
        tally['not_estimable'] += not estimable
        if con is None or con.get('ci') is None:
            tally['no_stored_interval'] += 1
        elif con['excludes_zero']:
            key = 'interval_excludes_zero_favouring_named' if con['point'] < 0 else \
                'interval_excludes_zero_favouring_reference'
            tally[key] += 1
        else:
            tally['interval_spans_zero'] += 1

    candidates = [{'arm': a, 'label': labels(a), 'score': scores[a], 'rank': i + 1,
                   'beats_reference_on_development': scores[a] < 1,
                   'score_2026': scores_new[a], 'rank_2026': rank_new.get(a),
                   'mae_development': {str(c): data[dev_name][c]['shown'][a] for c in cells},
                   'mae_2026': {str(c): data[new_name][c]['shown'][a] for c in cells}}
                  for i, a in enumerate(order)]
    same = winner is not None and (winner == named or winner in aliases)
    out = {'task': task, 'variable': variable, 'band': band, 'unit': UNIT[variable],
           'cohorts': {'development': dev_name, '2026': new_name}, 'cells': [str(c) for c in cells],
           'reference': {'arm': reference, 'label': labels(reference),
                         'mae_development': {str(c): data[dev_name][c]['shown'][reference] for c in cells},
                         'mae_2026': {str(c): data[new_name][c]['shown'][reference] for c in cells}},
           'named': {'arm': named, 'label': labels(named), 'score': scores[named],
                     'beats_reference_on_development': scores[named] < 1,
                     'score_2026': scores_new[named], 'rank_2026': rank_new.get(named),
                     'beats_reference_on_2026': None if scores_new[named] is None else scores_new[named] < 1},
           'reference_is_best_on_development': not scores[named] < 1,
           'runner_up': None if runner is None else {'arm': runner, 'label': labels(runner),
                                                      'score': scores[runner]},
           'identical_to_named': aliases,
           'n_candidates': len(order),
           'winner_2026': None if winner is None else {
               'arm': winner, 'label': labels(winner), 'score_2026': scores_new[winner],
               'rank_development': rank_dev[winner], 'score_development': scores[winner],
               'beats_reference_on_2026': scores_new[winner] < 1},
           'same_winner': same,
           'score_2026_gap': None if winner is None else scores_new[named] - scores_new[winner],
           'cells_2026_scored': [str(c) for c in check['cells']],
           'cells_2026_not_estimable': [{'cell': str(c), 'units': data[new_name][c]['units'],
                                         'unit': data[new_name][c]['unit']} for c in check['not_estimable']],
           'candidates': candidates,
           'excluded': dict(sorted(reasons.items())),
           'per_cell': per_cell,
           'tally_2026': tally}
    if notes:
        out['notes'] = notes
    return out


def check_contrasts(data, cohorts, cells, reference, oxygen, where, tol=TOL6, field='score'):
    """Every stored contrast of an arm against the reference equals the difference of the two MAEs read
    here (in the stored unit), to the stored rounding."""
    n = 0
    for cohort in cohorts:
        for c in cells:
            cell = data[cohort][c]
            for arm, con in cell['contrast'].items():
                if con is None or arm == reference:
                    continue
                point = con['stored_umol_kg']['point'] if oxygen and 'stored_umol_kg' in con else con['point']
                diff = cell[field][arm] - cell[field][reference]
                if abs(point - diff) > tol:
                    fail(f'{where} {cohort} {c}: stored {arm} - {reference} {point} is not the MAE difference {diff}')
                n += 1
    return n


# ------------------------------------------------------------------------------------- parts A, B
def forecast_data(part, variable, cohorts, cells, reference, summary_block):
    """Per cohort and cell, the core arms' common-row MAE (stored unit) and printed value, and every stored
    contrast against the reference; refused unless every core arm covers the declared share, oxygen in
    mg/l equals c times the stored value, and the reference equals the summary's compact value."""
    res = part['results'][variable]
    oxygen = variable == 'oxygen'
    data, checked = {}, 0
    for cohort in cohorts:
        data[cohort] = {}
        for c in cells:
            cell = res[cohort][str(c)]
            where = f'{part["part"]} {variable} {cohort} {c} h'
            if not cell['common_rows'] > 0:
                fail(f'{where}: no common rows')
            score, shown, contrast = {}, {}, {}
            for a in cell['core_arms']:
                e = cell['mae'][a]
                m = e.get('mae_common')
                if not finite(m) or cell['coverage'][a] < CORE_SHARE:
                    fail(f'{where}: core arm {a} has no common-row MAE or covers less than {CORE_SHARE}')
                score[a] = float(m)
                if oxygen:
                    mg = e['mgl']['mae_common']
                    if abs(mg - m * C) > TOL6:
                        fail(f'{where}: {a} mg/l {mg} is not c x {m}')
                    shown[a] = float(mg)
                else:
                    shown[a] = float(m)
                if a != reference:
                    contrast[a] = stored_contrast(cell['contrasts'].get(f'{a} - {reference}'),
                                                  f'{a} - {reference}', oxygen)
            if reference not in score:
                fail(f'{where}: the reference {reference} is not a core arm')
            block = summary_block[cohort].get(str(c))
            if block and reference in block.get('mae', {}):
                if abs(block['mae'][reference] - shown[reference]) > TOL6:
                    fail(f'{where}: reference {shown[reference]} here, {block["mae"][reference]} in the summary')
                checked += 1
            data[cohort][c] = {'score': score, 'shown': shown, 'contrast': contrast,
                               'units': cell['fold_week_units'], 'unit': 'fold x week',
                               'common_rows': cell['common_rows'],
                               'source_best': {'arm': cell['best'], 'mae': shown[cell['best']],
                                               'among': 'every core arm of the cell'}}
    return data, checked


def blended_six_hours(part_b, variable, named):
    """The blended weather tree on the 2026 six-hour cell, with every contrast the source stores for it."""
    cell = part_b['results'][variable]['Reserved-2026']['6']
    if 'blended' not in cell['core_arms']:
        fail(f'B {variable} Reserved-2026 6 h: the blended weather tree is not a core arm')
    oxygen = variable == 'oxygen'
    e = cell['mae']['blended']
    join = part_b['records'][variable]['joins']['blended']
    if abs(join['published_mae'] - e['mae_common']) > TOL6 or not join['passed']:
        fail(f'B {variable}: the blended weather tree {e["mae_common"]} is not its published {join["published_mae"]}')
    mae = e['mgl']['mae_common'] if oxygen else e['mae_common']
    named_mae = cell['mae'][named]['mgl']['mae_common'] if oxygen else cell['mae'][named]['mae_common']
    return {'label': part_b['labels'].get('blended', 'blended'), 'mae_2026': mae,
            'published_mae_stored_unit': join['published_mae'],
            'named_mae_2026': named_mae, 'named_minus_blended': named_mae - mae,
            'contrasts': {ref: stored_contrast(cell['contrasts'].get(f'blended - {ref}'), f'blended - {ref}', oxygen)
                          for ref in B_STORED_REFS},
            'named_vs_blended_stored': stored_contrast(cell['contrasts'].get(f'{named} - blended'),
                                                       f'{named} - blended', oxygen),
            'common_rows': cell['common_rows']}


# -------------------------------------------------------------------------------------------- part D
def gap_data(part, variable, cohorts, cells, summary_block):
    res = part['results']
    oxygen = variable == 'oxygen'
    data, checked = {}, 0
    for cohort in cohorts:
        data[cohort] = {}
        for c in cells:
            cell = res[cohort][variable].get(str(c))
            where = f'D {cohort} {variable} {c} h'
            if cell is None:
                fail(f'{where}: missing')
            if not cell['common_rows'] > 0:
                fail(f'{where}: no common rows')
            score, shown, contrast = {}, {}, {}
            for a in cell['core_arms']:
                e = cell['mae'][a]
                if e.get('rows') != 'common' or not finite(e.get('mae')) or cell['coverage'][a] < CORE_SHARE:
                    fail(f'{where}: core arm {a} is not scored on the common rows')
                score[a] = float(e['mae'])
                if oxygen:
                    if abs(e['mae_mgl'] - e['mae'] * C) > TOL6:
                        fail(f'{where}: {a} mg/l {e["mae_mgl"]} is not c x {e["mae"]}')
                    shown[a] = float(e['mae_mgl'])
                else:
                    shown[a] = score[a]
                if a != 'linear':
                    con = cell['minus_linear'].get(a)
                    if con is not None and con.get('rows_kind') != 'common rows':
                        fail(f'{where}: {a} - linear is not on the common rows')
                    contrast[a] = stored_contrast(con, f'{a} - linear', oxygen)
            if 'linear' not in score:
                fail(f'{where}: linear interpolation is not a core arm')
            block = summary_block.get(cohort, {}).get(variable, {}).get(str(c))
            if block and block.get('linear') is not None:
                if abs(block['linear'] - shown['linear']) > TOL6:
                    fail(f'{where}: linear {shown["linear"]} here, {block["linear"]} in the summary')
                checked += 1
            data[cohort][c] = {'score': score, 'shown': shown, 'contrast': contrast, 'units': cell['units'],
                               'unit': cell['unit'], 'common_rows': cell['common_rows'],
                               'source_best': {'arm': cell['best'], 'mae': shown[cell['best']],
                                               'among': 'every core arm of the cell'},
                               'incumbent': cell['incumbent']}
            if cell['units'] < MIN_UNITS:
                data[cohort][c]['note'] = f'fewer than {MIN_UNITS} independent units: not estimable'
    return data, checked


def gap_band_contrasts(named, variable, band, zoo, variants):
    """The band-level contrasts the thesis tables print for the named gap method, where they exist: the
    pretrained arms' 12-72 h band from runs/v2_fm_zoo.json (matrix E), every other method's 1-6 h and
    12-72 h bands from the rescored gap variants. Nothing is recomputed; oxygen from the gap variants is c
    times the stored value (the sondes' exact conversion)."""
    oxygen = variable == 'oxygen'
    out = {}
    if band == '168-1440':
        # past a week the thesis reads each length on its own: there is no band-level contrast
        return None
    if '.' in named:
        base, mode = named.split('.')
        if band != '12-72':
            return None
        for cohort in ('Gap-Dev', 'Gap-2026'):
            blk = zoo['matrix']['E'][variable].get(f'{cohort} 12-72 h')
            arm = None if blk is None else blk['arms'].get(base)
            if arm is None or arm.get('fill') != ('bidirectional' if mode == 'bidir' else 'forward'):
                out[cohort] = None
                continue
            # the p-values are left out: six decimals would round most of them to zero
            out[cohort] = {'source': rel(SRC['fm_zoo']), 'key': f'matrix.E.{variable}.{cohort} 12-72 h.arms.{base}',
                           'unit': blk['unit'], 'mae': arm['metric']['MAE'], 'units': blk['units'],
                           'unit_of_resampling': blk['unit_of_resampling'],
                           'vs_linear': {k: x for k, x in (arm.get('vs_main') or {}).items() if k != 'p'},
                           'vs_incumbent': {k: x for k, x in (arm.get('vs_secondary') or {}).items() if k != 'p'},
                           'incumbent': blk['incumbent'], 'references_mae': blk.get('references_MAE')}
        return out
    for cohort in ('Gap-Dev', 'Gap-2026'):
        blk = variants['results'].get(cohort, {}).get(variable, {}).get(band, {})
        arm = blk.get(named)
        if arm is None:
            out[cohort] = None
            continue
        scale = C if oxygen else 1.0
        entry = {'source': rel(SRC['gap_variants']), 'key': f'results.{cohort}.{variable}.{band}.{named}',
                 'mae': arm['mae'] * scale, 'contrasts': {}}
        for k, con in arm.items():
            if not k.startswith('minus_'):
                continue
            entry['contrasts'][f'{named} - {k[6:]}'] = {
                'point': con['delta'] * scale, 'ci': [con['ci_low'] * scale, con['ci_high'] * scale],
                'mae_arm': None if con.get('mae_arm') is None else con['mae_arm'] * scale,
                'mae_reference': None if con.get('mae_reference') is None else con['mae_reference'] * scale,
                'excludes_zero': con.get('excludes_zero'), 'estimable': con.get('estimable'),
                'jackknife_t': None if con.get('jackknife_t') is None else [x * scale for x in con['jackknife_t']],
                'jackknife_excludes_zero': con.get('jackknife_excludes_zero'), 'clusters': con.get('clusters'),
                'episodes': con.get('episodes')}
        out[cohort] = entry
    return out


# ------------------------------------------------------------------------------------------- station
def station_values(sources, variable):
    """Field-57 per fold and Field-2026 held-station MAEs and contrasts against A2, in the printed unit."""
    t = TARGET[variable]
    oxygen = variable == 'oxygen'
    f57 = sources['field57_oxygen' if oxygen else 'field57_temperature']['field57']
    f57r = sources['field57_rest_oxygen' if oxygen else 'field57_rest_temperature']['field57']
    inner = sources['field2026_inner']['held_station']
    cop = sources['field2026_copernicus_oxygen' if oxygen else 'field2026_copernicus_temperature']['held_station']
    cast = sources['field2026_cast_oxygen']['held_station']

    # Field-57 per outer fold
    dev_score, dev_contrast, dev_ratio, record_unit = {}, {}, {}, {}
    published = {}
    if oxygen:
        for key in ('field57_temperature', 'field57_rest_temperature'):
            for arm, e in sources[key]['field57']['arms'].items():
                published[arm] = float(e['nested_mae']['perfold'][t])
    for block in (f57, f57r):
        for arm, e in block['arms'].items():
            if arm not in STATION_ARMS:
                fail(f'station: Field-57 arm {arm} has no entry in STATION_ARMS')
            dev_score[arm] = float(dig(e, ('nested_mae', 'perfold', t), f'Field-57 {arm}'))
        for con in block['contrasts']:
            if con['reference'] == STATION_REF and con['target'] == t:
                point, lo, hi = con['perfold']['value']
                dev_contrast[con['arm']] = {'contrast': f'{con["arm"]} - {STATION_REF}', 'point': point,
                                            'ci': [lo, hi], 'excludes_zero': con['perfold']['excludes_zero'],
                                            'interval': '95 % bootstrap over cast dates, as stored (per fold)',
                                            'description': con['contrast']}
    if oxygen:
        # the cast-derived offsets have no mg/l rerun: the mg/l file holds c times their stored values, while
        # the reference is the mg/l rerun; their ratio to the reference is taken in the stored unit, where
        # both are on one basis
        for arm, mg in dev_score.items():
            if arm != STATION_REF and arm in published and abs(mg - published[arm] * C) <= 1e-9:
                dev_ratio[arm] = published[arm] / published[STATION_REF]
                record_unit[arm] = {'arm': published[arm], 'reference': published[STATION_REF]}
                if arm in dev_contrast:
                    dev_contrast[arm]['note'] = ('no mg/l rerun of this arm exists: c times the stored contrast, '
                                                 'as the mg/l file holds it')
    # Field-2026, held station
    new_score, new_contrast, notes, new_ratio = {}, {}, {}, {}
    reports = {'inner': inner, 'copernicus': cop}
    for arm, (_, src, name, _) in STATION_ARMS.items():
        if src is None:
            continue
        if oxygen and src == 'inner':
            if arm == STATION_REF:
                new_score[arm] = float(cop['nested_mae']['A2_none'][t])
                continue
            # no mg/l rerun of this arm exists: c times the stored value, as the thesis table prints it;
            # its ratio to the reference is taken in the stored unit, where both are on one basis (as on
            # Field-57)
            new_score[arm] = float(inner['nested_mae'][name][t]) * C
            new_ratio[arm] = float(inner['nested_mae'][name][t]) / float(inner['nested_mae']['A2_none'][t])
            notes[arm] = ('no mg/l rerun of this arm exists: its MAE and contrast are c times the stored values, '
                          'as the thesis table prints them, so the contrast is not the difference of the two '
                          'printed MAEs (the reference is the mg/l rerun)')
        else:
            new_score[arm] = float(reports[src]['nested_mae'][name][t])
    for arm, (_, src, name) in STATION_2026_ONLY.items():
        if oxygen and src == 'inner':
            if arm == 'background_A2_none':
                new_score[arm] = float(cast['nested_mae']['background_A2_none'][t])
            else:
                new_score[arm] = float(inner['nested_mae'][name][t]) * C
        else:
            new_score[arm] = float(reports[src]['nested_mae'][name][t])
    pairs = {'A1_both_direct': (cop, 'A1_both - A2_none', 1.0),
             'A4_base_only_direct': (cop, 'A4_base_only - A2_none', 1.0),
             'A7_ctdprior_direct': (inner, 'A7o - A2_none', C if oxygen else 1.0)}
    for arm, (rep, key, scale) in pairs.items():
        con = rep['contrasts'][key][t]
        entry = {'contrast': f'{arm} - {STATION_REF}', 'point': con['point'] * scale,
                 'ci': [x * scale for x in con['ci']], 'excludes_zero': con['excludes_zero'],
                 'jackknife_t': [x * scale for x in con['jackknife_t']],
                 'jackknife_excludes_zero': con['jackknife_excludes_zero'], 'units': con['dates'],
                 'interval': '95 % bootstrap over cast dates (each drawn date relabelled) and the delete-one-date '
                             'jackknife, as stored', 'stored_key': f'held_station.contrasts.{key}.{t}'}
        if scale != 1.0:
            entry['stored_umol_kg'] = {'point': con['point'], 'ci': con['ci'], 'jackknife_t': con['jackknife_t']}
            entry['note'] = notes[arm]
        new_contrast[arm] = entry
    if not oxygen:
        six = sources['field2026_inner_copernicus']['A7o_inner_prior - A2_none_six_decimals'][t]
        new_contrast['A7_ctdprior_direct']['six_decimals'] = {'point': six['point'], 'ci': six['ci'],
                                                              'source': rel(SRC['field2026_inner_copernicus'])}
    return dev_score, dev_contrast, dev_ratio, record_unit, new_score, new_contrast, notes, new_ratio


def station_gates(sources):
    """The A2 reference the station task reads equals what the same sources store for it elsewhere."""
    out = {}
    inner = sources['field2026_inner']
    shared = sources['field57_temperature']['field57']['arms'][STATION_REF]['nested_mae']['shared']['temp_c']
    out['inner_split_study_reproduces_A2'] = bool(inner['development']['A2_reproduces_published'])
    out['A2_development_equals_shared_Field57'] = abs(inner['development']['A2_none']['nested_mae']['temp_c']
                                                      - shared) <= 1e-9
    out['A2_2026_temperature_inner_equals_copernicus'] = (
        inner['held_station']['nested_mae']['A2_none']['temp_c']
        == sources['field2026_copernicus_temperature']['held_station']['nested_mae']['A2_none']['temp_c'])
    out['A2_2026_oxygen_C1_equals_C2'] = (
        sources['field2026_copernicus_oxygen']['held_station']['nested_mae']['A2_none']['do_umol_l']
        == sources['field2026_cast_oxygen']['held_station']['nested_mae']['A2_none']['do_umol_l'])
    out['copernicus_reports_reproduce_Field57'] = all(
        sources[k]['G2'][a]['passed'] for k in ('field2026_copernicus_temperature', 'field2026_copernicus_oxygen')
        for a in ('A1_both', 'A4_base_only'))
    bad = [k for k, ok in out.items() if ok is not True]
    if bad:
        fail(f'station: gate(s) {bad} did not pass')
    return out


def station_floors(sources):
    out = {}
    for variable, src in (('temperature', 'cast_floor_temperature'), ('oxygen', 'cast_floor_oxygen')):
        for cohort, key in CAST_FLOOR.items():
            value = dig(sources[src], ('cohorts', key, variable, 'mean_over_station_days'), f'cast floor {key}')
            out[(variable, cohort)] = {'value': float(value),
                                       'basis': f'cast floor: mean disagreement of two casts minutes apart at one '
                                                f'station ({key})',
                                       'source': rel(SRC[src]),
                                       'key': f'cohorts.{key}.{variable}.mean_over_station_days'}
    return out


def station_data(sources, variable, floors):
    (dev_score, dev_contrast, dev_ratio, record_unit, new_score, new_contrast, notes,
     new_ratio) = station_values(sources, variable)
    cell = STATION_CELL
    f57 = sources['field57_temperature']['field57']
    held = sources['field2026_inner']['held_station']
    data = {'Field-57': {cell: {'score': dev_score, 'shown': dict(dev_score),
                                'contrast': {a: dev_contrast.get(a) for a in dev_score if a != STATION_REF},
                                'ratio': dev_ratio, 'record_unit': record_unit,
                                'units': f57['dates'], 'unit': 'cast date', 'common_rows': f57['casts'],
                                'source_best': None, 'margin': floors[(variable, 'Field-57')]}},
            'Field-2026': {cell: {'score': new_score, 'shown': dict(new_score),
                                  'contrast': {a: new_contrast.get(a) for a in new_score if a != STATION_REF},
                                  'ratio': new_ratio,
                                  'units': held['dates'], 'unit': 'cast date', 'common_rows': held['casts'],
                                  'source_best': None, 'margin': floors[(variable, 'Field-2026')]}}}
    for cohort in data:
        c = data[cohort][cell]
        low = min(c['shown'], key=lambda a: (c['shown'][a], a))
        c['source_best'] = {'arm': low, 'mae': c['shown'][low], 'among': 'every arm the cohort stores'}
        for arm, con in c['contrast'].items():
            if con is not None:
                con['margin'] = dict(c['margin'])
                con['reading'] = margin_reading(con['point'], con['ci'], c['margin']['value'])
                con['reading_rule'] = 'computed here from the stored contrast and the margin'
    for arm, note in notes.items():
        data['Field-2026'][cell].setdefault('arm_notes', {})[arm] = note
    return data


def station_contrast_gate(data, variable):
    """Stored contrasts against A2 equal the MAE differences: exactly on Field-57 per fold, to the four
    decimals of the published Field-2026 reports for temperature and six for the mg/l reruns; the arm with
    no mg/l rerun is checked in its stored unit."""
    n = 0
    for cohort, tol in (('Field-57', 1e-9), ('Field-2026', TOL4 if variable == 'temperature' else TOL6)):
        cell = data[cohort][STATION_CELL]
        for arm, con in cell['contrast'].items():
            if con is None or 'stored_umol_kg' in con:
                # the arm without an mg/l rerun on Field-2026 is checked in its stored unit (station_records)
                continue
            ru = cell.get('record_unit', {}).get(arm)
            if ru is not None:
                # c times the stored contrast against the stored difference, on one basis
                diff = (ru['arm'] - ru['reference']) * C
            else:
                diff = cell['score'][arm] - cell['score'][STATION_REF]
            if abs(con['point'] - diff) > tol:
                fail(f'station {variable} {cohort}: stored {arm} - A2 {con["point"]} is not the MAE '
                     f'difference {diff}')
            n += 1
    return n


def station_records(sources, floors):
    out, checked = {}, 0
    labels = lambda a: (STATION_ARMS.get(a) or STATION_2026_ONLY.get(a) or (a,))[0]  # noqa: E731
    inner_umol = sources['field2026_inner']['held_station']['nested_mae']
    for variable in VARIABLES:
        data = station_data(sources, variable, floors)
        checked += station_contrast_gate(data, variable)
        if variable == 'oxygen':
            # the arm without an mg/l rerun: its stored contrast is the difference of the stored MAEs
            con = sources['field2026_inner']['held_station']['contrasts']['A7o - A2_none']['do_umol_l']
            diff = inner_umol['A7o']['do_umol_l'] - inner_umol['A2_none']['do_umol_l']
            if abs(con['point'] - diff) > TOL4:
                fail(f'station oxygen Field-2026: stored A7o - A2 {con["point"]} is not {diff} in the stored unit')
            checked += 1
        excluded = dict(STATION_EXCLUDED)
        main = band_record('station', variable, 'held_station', [STATION_CELL], ('Field-57', 'Field-2026'), data,
                           STATION_REF, excluded, labels,
                           notes=['Field-57 is scored per outer fold; Field-2026 by the single selection on all of '
                                  'Field-57 that the 2026 reading uses.'])
        product = {a for a, v in STATION_ARMS.items() if v[3]}
        excluded_pf = {**excluded, **{a: 'reads the Copernicus product, not available after 16 January 2026 '
                                         'without the extraction rule' for a in product}}
        free = band_record('station', variable, 'held_station_product_free', [STATION_CELL],
                           ('Field-57', 'Field-2026'), data, STATION_REF, excluded_pf, labels,
                           notes=['The rule without the arms that read the Copernicus product.'])
        out[variable] = {'held_station': main, 'held_station_product_free': free}
    return out, checked


# ---------------------------------------------------------------------------------------------- main
def summary_of(tasks):
    """How many task bands keep their development winner on 2026, and which do not. A band with a single
    candidate (the station task without the product) keeps it by construction, so it is listed apart and
    not counted."""
    same, differ, single = [], [], []
    for task, block in tasks.items():
        for variable, bands in block.items():
            for band, rec in bands.items():
                w = rec['winner_2026']
                entry = {'task': task, 'variable': variable, 'band': band,
                         'development_winner': rec['named']['arm'], 'its_rank_2026': rec['named']['rank_2026'],
                         'winner_2026': None if w is None else w['arm'],
                         'its_rank_development': None if w is None else w['rank_development'],
                         'n_candidates': rec['n_candidates'], 'score_2026_gap': rec['score_2026_gap']}
                if rec['n_candidates'] < 2:
                    single.append(entry)
                else:
                    (same if rec['same_winner'] else differ).append(entry)
    return {'task_bands': len(same) + len(differ), 'same_winner': len(same), 'different_winner': len(differ),
            'same': same, 'different': differ, 'single_candidate_bands': single}


def build():
    sources = {k: load(p) for k, p in SRC.items() if k != 'oxygen_units'}
    if not Path(SRC['oxygen_units']).exists():
        fail(f'source missing: {rel(SRC["oxygen_units"])}')
    parts = {'A': sources['part_A'], 'B': sources['part_B'], 'D': sources['part_D']}
    summary = sources['summary_85']
    gates = {'addendum85': addendum85_gates(summary, parts)}
    checked = {}
    floors = station_floors(sources)

    tasks = {'station': {}, 'forecast': {}, 'silent_probe': {}, 'gaps': {}}
    station, checked['station_contrasts'] = station_records(sources, floors)
    gates['station'] = station_gates(sources)
    tasks['station'] = station

    excluded_common = {**INADMISSIBLE, **NOT_REPORTED,
                       **{a: f'shared selection; the per-fold selection ({b}) is primary' for a, b in SHARED.items()}}

    labels_b = parts['B']['labels']
    checked['forecast_reference'] = checked['forecast_contrasts'] = 0
    for v in VARIABLES:
        data, n = forecast_data(parts['B'], v, B_COHORTS, B_LEADS, 'persistence', summary['compact']['B'][v])
        checked['forecast_reference'] += n
        checked['forecast_contrasts'] += check_contrasts(data, B_COHORTS, B_LEADS, 'persistence', v == 'oxygen',
                                                         f'B {v}')
        rec = band_record('forecast', v, '1-72', B_LEADS, B_COHORTS, data, 'persistence', excluded_common,
                          lambda a: label_of(a, labels_b))
        rec['blended_weather_tree_6h'] = blended_six_hours(parts['B'], v, rec['named']['arm'])
        tasks['forecast'][v] = {'1-72': rec}

    labels_a = parts['A']['labels']
    checked['silent_reference'] = checked['silent_contrasts'] = 0
    for v in VARIABLES:
        ref = A_REF[v]
        data, n = forecast_data(parts['A'], v, A_COHORTS, A_LEADS, ref, summary['compact']['A'][v])
        checked['silent_reference'] += n
        checked['silent_contrasts'] += check_contrasts(data, A_COHORTS, A_LEADS, ref, v == 'oxygen', f'A {v}')
        rec = band_record('silent_probe', v, '6-72', A_LEADS, A_COHORTS, data, ref, excluded_common,
                          lambda a: label_of(a, labels_a))
        tasks['silent_probe'][v] = {'6-72': rec}

    labels_d = parts['D']['labels']
    checked['gap_reference'] = checked['gap_contrasts'] = 0
    for v in VARIABLES:
        tasks['gaps'][v] = {}
        for band, cohorts, cells in GAP_BANDS:
            data, n = gap_data(parts['D'], v, cohorts, cells, summary['compact']['D'])
            checked['gap_reference'] += n
            checked['gap_contrasts'] += check_contrasts(data, cohorts, cells, 'linear', v == 'oxygen',
                                                        f'D {v} {band}')
            rec = band_record('gaps', v, band, cells, cohorts, data, 'linear', excluded_common,
                              lambda a: gap_label(a, labels_d))
            rec['incumbent'] = {str(c): data[cohorts[0]][c]['incumbent'] for c in cells}
            rec['band_contrasts'] = gap_band_contrasts(rec['named']['arm'], v, band, sources['fm_zoo'],
                                                       sources['gap_variants'])
            tasks['gaps'][v][band] = rec

    out = {'script': SCRIPT,
           'rule': RULE,
           'unit': f'temperature degC; oxygen mg/l (the sondes alone: the mg/l the source stores beside the '
                   f'per-kilogram value, equal to c times it, c = {C} mg/l per umol/kg, src/units.py; the station '
                   f'task: the mg/l reruns of the cast analyses)',
           'rounding': 'six decimals, once, here; the published Field-2026 temperature reports keep four',
           'check_2026': CHECK_2026,
           'summary': summary_of(tasks),
           'references': {'station': STATION_REF, 'forecast': 'persistence',
                          'silent_probe': dict(A_REF), 'gaps': 'linear'},
           'margins': {'station': {f'{v}|{c}': f for (v, c), f in floors.items()},
                       'parts_A_B_D': 'the floor each stored contrast carries (floor, floor_basis)'},
           'contrast_sign': 'named arm minus reference: negative favours the named arm',
           'gates': gates,
           'checked': checked,
           'tasks': tasks,
           'sources': {rel(p): sha256(p) for p in SRC.values()}}
    return rounded(out)


def main():
    out = build()
    OUT.write_bytes(json.dumps(out, indent=1).encode('utf-8'))
    for task, block in out['tasks'].items():
        for v, bands in block.items():
            for band, rec in bands.items():
                n = rec['named']
                ru = rec['runner_up']
                flag = '' if n['beats_reference_on_development'] else '  (no candidate beats the reference)'
                print(f'{task:12s} {v:11s} {band:26s} {n["arm"]:28s} {n["score"]:.6f}'
                      f'  runner-up {ru["arm"] if ru else None}{flag}')
                w = rec['winner_2026']
                print(f'{"":51s} 2026: named rank {n["rank_2026"]}/{rec["n_candidates"]} score {n["score_2026"]:.6f};'
                      f' winner {w["arm"]} (development rank {w["rank_development"]}) {w["score_2026"]:.6f}'
                      f'{"" if rec["same_winner"] else "  DIFFERENT"}')
    s = out['summary']
    print(f'summary: {s["same_winner"]} of {s["task_bands"]} task bands keep their development winner on 2026')
    print('checked:', out['checked'])
    print('wrote', rel(OUT))


if __name__ == '__main__':
    main()
