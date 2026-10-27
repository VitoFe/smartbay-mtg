"""Chapter 6 summary table: one row per answer to the four research questions.

Every number is read from the artifacts that Chapters 5 and 6 cite, never from the .tex, and the
table is written to thesis_template/content/tables/table_bay_summary.tex. A missing artifact or key
stops the script with the artifact and the key it needed.

Under the first three questions each task's rows give the task's selected method minus the task's
reference, as runs/v2_best_by_task.json (scripts.v2_best_by_task) stores the contrast: the method with
the lowest development error among the arms scored in both cohorts, read on the 2026 cohort (and on
Field-57 for the station). A row over leads or gap lengths spans the band's cells. A selected arm that
changes in the artifact stops the script, so that the row is relabelled.

Under the fourth question each printed row also gives, after its verdict, its change and interval status on the
other period (runs/v2_addendum86.json): the 2026 test period for a development row, the casts of 2025 for the
ridge scored on the casts of 2026.

Each row leads with its change: 100 x (the first arm's error - the second arm's) / the second arm's, from
the two errors stored in the artifact the row's contrast is read from (refused when they do not give the
stored difference); a row over leads or gap lengths gives the range of the changes of its cells.

Each verdict is computed from the artifacts by the rules of Chapter 4:
  - Section 4.9.2 (subsec:meth_uncertainty): over three sampler seeds and six issuance grids a
    contrast is *stable* when every seed and phase keeps its sign with an interval excluding zero,
    *sign-stable* when the seeds and the phase mean share a sign and the only intervals excluding
    zero point that way, *choice-sensitive* when intervals exclude zero in both directions or the
    seeds and the phase mean disagree in sign, and *no demonstrated difference* when the signs agree
    but no interval excludes zero; *seeds only* reads the three seeds alone, and *not swept* is a
    point estimate on the published grid and seed. The artifacts store the codes 'result',
    'direction', 'not estimable' and 'not distinguishable'; src.experiments.sweep_verdict.label
    prints them. Below about twenty independent units a contrast is robust only where the bootstrap
    and the delete-one-unit jackknife both exclude zero.
  - Chapter 4 (tab:bay_verdicts): a single unswept interval *excludes zero* or is *not
    distinguishable*; it is an interval's status, not a sweep verdict. Where the jackknife was
    computed and does not exclude zero the status is *bootstrap alone*. A row over leads or gap
    lengths names the cells whose interval does not exclude zero.
  - Section 4.9.2 (sec:meth_multiplicity): a starred estimate that false-discovery control over the
    table's own estimates does not retain carries a further mark, read from runs/v2_fdr.json
    (scripts.v2_fdr), which is written from the row builders here and so is read after them.
The verdicts the chapters state on 2026-09-28 are kept in EXPECTED. A computed verdict that differs is
printed as DRIFT, so that the prose resting on a regenerated artifact is re-read; it does not stop the
script, because the evidence wins over the prose.

Usage: python -m scripts.v2_summary_table [--out PATH]
"""
import argparse
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from src.experiments.sweep_verdict import label as verdict_label
from src.experiments.sweep_verdict import verdict as sweep_verdict
from src.units import OXYGEN_UNIT_TEX, column_decimals, factor, mg_decimals

ROOT = Path(__file__).resolve().parents[1]
BAY = ROOT / 'reports/bay_twin'
OUT = ROOT / 'thesis_template/content/tables/table_bay_summary.tex'
#: the artifacts behind each row, printed in the summary-sources appendix (app:summary_sources; the
#: provenance check reads a table's citations from this companion when the table itself cites none)
SOURCES = OUT.with_name('table_bay_summary_sources.tex')

#: Dissolved oxygen is printed in mg/l. An analysis of the sondes alone prints its published value times
#: c; an analysis that reads casts or a per-litre source (Field-57, Field-2026, the floors, the
#: utilisation arms) reads its rerun in mg/l (addendum 60 part C), in the same schema. Temperature and
#: salinity always come from the published artifacts.
C = factor()
MGL = 'runs/v2_oxygen_mgl/'
#: the per-fold utilisation section (addendum 60 part C3, amendment 4)
AOU_REST = MGL + 'C3/v2_perfold_rest.json'

#: artifact keys (relative to reports/bay_twin, as the thesis cites them) in the order first read
CITED = []
#: artifacts read since the last row was built; row() claims them
_ROW = []
_CACHE = {}
STAR, DAGGER = r'\ast', r'\dagger'
#: the method selected for each task and its 2026 result against the task's reference
BEST = 'runs/v2_best_by_task.json'
#: how far the difference of two stored errors may sit from the stored difference: the station's 2026
#: errors are stored to four decimals, every other to six or more
TOLERANCE = 2e-4


def load(key, cite=None):
    """An artifact's JSON; `cite` is the path the note cites when it differs (its folder)."""
    if key not in _CACHE:
        path = BAY / key
        if not path.exists():
            raise SystemExit(f'v2_summary_table: missing artifact {key}')
        _CACHE[key] = json.loads(path.read_text(encoding='utf-8'))
    name = cite or key
    if name not in CITED:
        CITED.append(name)
    _ROW.append(name)
    return _CACHE[key]


def at(key, *path, cite=None):
    """The value at `path` inside artifact `key`, or a stop naming both."""
    node = load(key, cite)
    for i, step in enumerate(path):
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            raise SystemExit(f"v2_summary_table: {key} has no key {'/'.join(map(str, path[:i + 1]))}")
    return node


def pick(key, path, **match):
    """The one entry of the list at `path` whose fields equal `match`."""
    found = [e for e in at(key, *path) if all(e.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise SystemExit(f"v2_summary_table: {key} {'/'.join(path)} has {len(found)} entries "
                         f'matching {match}')
    return found[0]


def triple(entry, where):
    """A [point, lo, hi] list as the forecasting artifacts store it."""
    if not (isinstance(entry, list) and len(entry) >= 3):
        raise SystemExit(f'v2_summary_table: {where}: expected [point, lo, hi], found {entry!r}')
    return tuple(float(x) for x in entry[:3])


def need(entry, keys, where):
    missing = [k for k in keys if k not in entry]
    if missing:
        raise SystemExit(f"v2_summary_table: {where} has no key {', '.join(missing)}")
    return entry


# ------------------------------------------------------------------------------------ formatting

def signed(x, d):
    s = f'{x:+.{d}f}'
    return '+' + s[1:] if s.startswith('-') and float(s) == 0 else s


def sup(m):
    return f'^{{{m}}}' if m else ''


def contrast(point, lo, hi, d, m=''):
    """`$-0.190^{\\ast}$ $[-0.389, -0.051]$`; the point alone when there is no interval."""
    head = f'${signed(point, d)}{sup(m)}$'
    return head if lo is None else f'{head} $[{signed(lo, d)}, {signed(hi, d)}]$'


def span(points, d, marks=None):
    """A range over leads or lengths, from the smallest to the largest magnitude."""
    order = sorted(range(len(points)), key=lambda i: abs(points[i]))
    marks = marks or [''] * len(points)
    a, b = order[0], order[-1]
    return f'${signed(points[a], d)}{sup(marks[a])}$ to ${signed(points[b], d)}{sup(marks[b])}$'


def starred(point, lo, hi, d):
    return f'${signed(point, d)}{sup(STAR if excludes(lo, hi) else "")}$'


def excludes(lo, hi):
    return lo > 0 or hi < 0


def mark(excl, jack=None):
    """Star: the interval excludes zero, and so does the jackknife where one was computed.
    Dagger: the bootstrap alone."""
    if not excl:
        return ''
    return STAR if jack is None or jack else DAGGER


def count(n):
    """A whole number with a thin space between thousands: 1\\,440."""
    return f'{int(n):,}'.replace(',', r'\,')


def sig2(x):
    """A change in per cent, signed, with at most two significant figures, rounded half away from zero."""
    if x == 0:
        return '0'
    d = Decimal(repr(float(x)))
    e = math.floor(math.log10(abs(x))) - 1
    q = d.quantize(Decimal(1).scaleb(e), rounding=ROUND_HALF_UP)
    if abs(q) >= Decimal(10) ** (e + 2):          # 9.96 rounds to 10.0: one figure too many
        e += 1
        q = d.quantize(Decimal(1).scaleb(e), rounding=ROUND_HALF_UP)
    digits = format(abs(q), 'f') if e < 0 else str(int(abs(q)))
    return ('+' if x > 0 else '-') + digits


def pct(arm, ref, point=None, where=''):
    """The change: 100 x (the arm's error - the reference's) / the reference's, from two errors an artifact
    stores. Refused when their difference is not the stored difference `point` (both in one unit)."""
    arm, ref = float(arm), float(ref)
    if point is not None and abs((arm - ref) - float(point)) > TOLERANCE:
        raise SystemExit(f'v2_summary_table: {where}: the stored errors {arm} and {ref} do not give the stored '
                         f'difference {point}')
    return 100 * (arm - ref) / ref


def change_text(x):
    """`$-74\\,\\%$`, or --- where the row's artifact stores no errors."""
    return '---' if x is None else f'${sig2(x)}\\,\\%$'


def change_span(changes):
    """The range of the changes of a row's cells, from the smallest in magnitude to the largest."""
    if len(changes) == 1:
        return change_text(changes[0])
    order = sorted(changes, key=abs)
    # on two lines, so that the column keeps the width of a single change
    return f'\\makecell[tr]{{${sig2(order[0])}$ to\\\\${sig2(order[-1])}\\,\\%$}}'


def mgd(value_mg, decimals):
    """Decimals for an oxygen value in mg/l whose published form printed `decimals` in its own unit:
    the significant figures it had there, at least two (src.units)."""
    return mg_decimals(value_mg / C, decimals)


def mg_span(points_mg, decimals):
    """One number of decimals for a range of oxygen values in mg/l (src.units.column_decimals)."""
    return column_decimals([p / C for p in points_mg], decimals)


def times_c(*values):
    """A sonde-only oxygen quantity, published per kilogram, in mg/l."""
    return tuple(None if v is None else C * v for v in values)


# --------------------------------------------------------------------------------- the two rules

def status(excl, jack=None):
    """The status of one interval (tab:bay_verdicts), worded apart from the sweep verdict 'no demonstrated
    difference': it excludes zero, by the bootstrap alone where the jackknife was computed and does not, or
    it is not distinguishable."""
    if not excl:
        return 'not distinguishable'
    return 'excludes zero' if jack is None or jack else 'excludes zero, bootstrap alone'


def against(point, lo, hi, floor, harm=False):
    """The reading of a difference against a floor, which the pretrained-forecaster and recalibration
    producers store beside their contrasts (scripts.v2_fm_zoo_score_dist_bridge,
    src.experiments.v2_recalibration import it). Table 6.1 reads no contrast against a floor."""
    if lo is not None and not excludes(lo, hi):
        return 'interval spans zero'
    prefix = 'deficit ' if harm else ''
    if lo is not None and (lo > floor or hi < -floor):
        return prefix + 'clears floor'
    return prefix + ('exceeds floor' if abs(point) > floor else 'below floor')


def floors():
    """The measurement floors, for the producers that import them (scripts.v2_fm_zoo_score_dist_bridge);
    Table 6.1 no longer reads them."""
    ctd, ctd_mgl = 'runs/v2_ctd_floor.json', MGL + 'C2/v2_ctd_floor.json'
    mads = [float(need(p, ('mad',), 'runs/v2_colocated_depth.json by_pair')['mad'])
            for p in at('runs/v2_colocated_depth.json', 'by_pair')]
    return {
        'quant': at('runs/v2_variance_components.json', 'budget', 'u_quantisation'),
        'coloc_low': min(mads),
        # the oxygen floors in mg/l, from the casts' rerun
        'sonde': at(MGL + 'C2/v2_oxygen_floor.json', 'by_cohort', 'pre_2026', '30min', 'pooled', 'oxygen',
                    'median_abs'),
        'cast_T_gen2': at(ctd, 'cohorts', 'gen2_before_2026', 'temperature', 'mean_over_station_days'),
        'cast_O_gen2': at(ctd_mgl, 'cohorts', 'gen2_before_2026', 'oxygen', 'mean_over_station_days'),
        'cast_T_all': at(ctd, 'cohorts', 'CTD-722', 'temperature', 'mean_over_station_days'),
        'cast_O_all': at(ctd_mgl, 'cohorts', 'CTD-722', 'oxygen', 'mean_over_station_days'),
    }


def rule(seeds, phases=None):
    """Section 4.9.2 over (point, lo, hi) triples. `seeds` are 42, 7 and 123; `phases` are offsets
    0 to 5, of which offset 0 is seed 42 on the published grid and is not counted twice."""
    return sweep_verdict(seeds, phases)


# ---------------------------------------------------------------------------- the selected methods

def selected(task, variable, band, arm):
    """One band of one task in runs/v2_best_by_task.json, whose selected method must still be `arm`: a row's
    label names the method, so a change of selected arm stops the script rather than mislabel it."""
    b = at(BEST, 'tasks', task, variable, band)
    if b['named']['arm'] != arm:
        raise SystemExit(f"v2_summary_table: {BEST} now selects {b['named']['arm']} for {task} {variable} {band}, "
                         f'not {arm}; relabel the row')
    return b


def stored(entry, where):
    """(point, lo, hi, excludes zero, jackknife excludes zero or None) of a contrast as the artifact stores it,
    or None where it has no interval."""
    if entry.get('ci') is None:
        return None
    point, lo, hi = float(entry['point']), float(entry['ci'][0]), float(entry['ci'][1])
    flag = entry.get('excludes_zero')
    if flag is not None and bool(flag) != excludes(lo, hi):
        raise SystemExit(f'v2_summary_table: {where}: the stored flag disagrees with its interval')
    jack = entry.get('jackknife_excludes_zero') if entry.get('jackknife_t') is not None else None
    return point, lo, hi, excludes(lo, hi), jack


def cell_change(e, where):
    """The change of one cell of a band in one cohort, from the selected method's and the reference's errors
    the cell stores beside its contrast."""
    need(e, ('mae', 'reference_mae'), where)
    return pct(e['mae'], e['reference_mae'], e['contrast'].get('point'), where)


def cells(b, cohort):
    """[(cell, stored contrast or None, change)] for each cell of the band in `cohort`, against the reference."""
    out = []
    for cell in b['cells']:
        where = f"{BEST} {b['task']} {b['variable']} {b['band']} {cohort} {cell}"
        e = need(b['per_cell'][cell][cohort], ('contrast',), where)
        out.append((cell, stored(e['contrast'], where), cell_change(e, where)))
    return out


def band_contrast(b, cohort):
    """The band's pooled contrast against linear interpolation in `cohort` and its change, in either schema
    the artifact keeps (the arm's error beside the references', or both errors inside the contrast)."""
    where = f"{BEST} {b['band']} {cohort}"
    block = need(b, ('band_contrasts',), f"{BEST} {b['task']} {b['variable']} {b['band']}")['band_contrasts'][cohort]
    if 'vs_linear' in block:
        entry = block['vs_linear']
        change = pct(block['mae'], need(block, ('references_mae',), where)['references_mae']['linear'],
                     entry['point'], where)
    else:
        entry = block['contrasts'][f"{b['named']['arm']} - linear"]
        change = pct(entry['mae_arm'], entry['mae_reference'], entry['point'], where)
    return stored(entry, where), change


def hours(names, order):
    """'6, 12 and 36\\,h' for cells named by their lead or gap length in hours; three or more cells that
    follow each other in the band's `order` print as one run, '168 to 504\\,h'."""
    runs = []
    for n in names:
        if runs and order.index(n) == order.index(runs[-1][-1]) + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    words = []
    for run in runs:
        words += [f'{count(run[0])} to {count(run[-1])}'] if len(run) >= 3 else [count(n) for n in run]
    return (words[0] if len(words) == 1 else ', '.join(words[:-1]) + ' and ' + words[-1]) + r'\,h'


def exceptions(kept, order):
    """The cells of a band whose interval does not exclude zero, and those where only the bootstrap does."""
    words = []
    flat = [cell for cell, c, _ in kept if not c[3]]
    boot = [cell for cell, c, _ in kept if c[3] and c[4] is False]
    if flat:
        words.append(f'not distinguishable at {hours(flat, order)}')
    if boot:
        words.append(f'bootstrap alone at {hours(boot, order)}')
    return words


def over_band(b, d, unit, oxygen=False):
    """The text, the range of changes and the verdict of a row spanning a band's 2026 cells: the cells whose
    interval does not exclude zero (or excludes it by the bootstrap alone), and the cells with no interval."""
    found, order = cells(b, '2026'), list(b['cells'])
    kept = [(cell, c, ch) for cell, c, ch in found if c is not None]
    missing = [cell for cell, c, _ in found if c is None]
    pts = [c[0] for _, c, _ in kept]
    text = span(pts, mg_span(pts, d) if oxygen else d, [mark(c[3], c[4]) for _, c, _ in kept])
    one = unit[:-1]
    if not any(c[3] for _, c, _ in kept):
        words = [f'not distinguishable at any {one}']
    else:
        words = exceptions(kept, order) or [f'excludes zero at every {one}']
    if missing:
        words.append(f'n.e. at {hours(missing, order)}')
    return text, change_span([ch for _, _, ch in kept]), '; '.join(words), kept


# ---------------------------------------------------------------------------------------- rows

def row(key, result, cohort, text, change, verdict):
    sources = list(dict.fromkeys(_ROW))
    _ROW.clear()
    if not sources:
        raise SystemExit(f'v2_summary_table: row {key} read no artifact')
    return {'key': key, 'result': result, 'cohort': rf'\textsc{{{cohort}}}', 'contrast': text,
            'change': change, 'verdict': verdict, 'sources': sources}


def state_estimation():
    """Question 1: the method selected for a station without a probe, the spatial estimator with the Copernicus
    depth offset, against no depth offset, and the cast-derived depth offset where the product is not read
    (Sections 5.1 and 5.4.2)."""
    rows = []
    # addendum 69: the comparators of the per-fold model and the cast prior rebuilt per inner split
    fc = 'runs/v2_field_contrasts_a69.json'

    def field57(arm, ref, target):
        src = MGL + 'C1/v2_field_contrasts_a69.json' if target == 'do_umol_l' else fc
        e = pick(src, ('contrasts',), arm=arm, reference=ref, target=target)
        return need(e, ('date',), f'{src} {arm} - {ref} {target}')['date']

    # the selected method, its depth offset chosen per outer fold (addendum 47), on Field-57; the shared
    # selection beside it
    b = selected('station', 'temperature', 'held_station', 'A1_both_direct')
    cell = b['per_cell']['held_station']['development']
    per = stored(cell['contrast'], f'{BEST} station Field-57')
    e = field57('A1_both_direct', 'A2_none_direct', 'temp_c')
    verdict = status(per[3], per[4])
    if e['excludes_zero']:
        verdict += f"; {starred(e['point'], *e['ci'], 3)} with one penalty"
    rows.append(row('prior57_T', r'Copernicus depth offset $-$ none, $T$, per fold', 'Field-57',
                    contrast(*per[:3], 3, mark(per[3], per[4])),
                    change_text(cell_change(cell, f'{BEST} station Field-57')), verdict))
    # the same on the 2026 casts, and, where the product is not read, the cast-derived depth offset fitted
    # inside each inner split (addendum 69)
    for key, label, name, arm in (
            ('cop26_T', r'Copernicus depth offset $-$ none, $T$', 'held_station', 'A1_both_direct'),
            ('cast26_T', r'Cast-derived depth offset $-$ none, $T$, inner split', 'held_station_product_free',
             'A7_ctdprior_direct')):
        band = selected('station', 'temperature', name, arm)
        cell = band['per_cell']['held_station']['2026']
        e = stored(cell['contrast'], f'{BEST} station {name} 2026')
        rows.append(row(key, label, 'Field-2026', contrast(*e[:3], 3, mark(e[3], e[4])),
                        change_text(cell_change(cell, f'{BEST} station {name} 2026')), status(e[3], e[4])))
    # the spatial estimator (the per-fold model) against the oracle climatology comparator; the contrast
    # file stores no errors
    e = field57('naive_model', 'naive_oracle', 'temp_c')
    rows.append(row('oracle_T', r'Spatial estimator $-$ \textsc{oracle} climatology, $T$', 'Field-57',
                    contrast(e['point'], *e['ci'], 3, mark(e['excludes_zero'])), change_text(None),
                    status(e['excludes_zero'])))

    pf = pick(MGL + 'C1/v2_perfold_a69.json', ('field57', 'contrasts'), contrast='CTD prior, harmonic, vs none',
              target='do_umol_l')
    per = triple(pf['perfold']['value'], 'v2_perfold.json cast prior per fold')
    pub = triple(pf['published']['value'], 'v2_perfold.json cast prior published')
    verdict = status(excludes(*per[1:]))
    if excludes(*pub[1:]):
        verdict += f'; {starred(*pub, mgd(pub[0], 2))} with one penalty'
    rows.append(row('cast57_O', r'Cast-derived depth offset $-$ none, DO, per fold', 'Field-57',
                    contrast(*per, mgd(per[0], 2), mark(excludes(*per[1:]))), change_text(None), verdict))
    # on oxygen no depth offset beats none on development, so the reference is kept; the row gives what the
    # Copernicus depth offset does on the 2026 casts
    b = selected('station', 'oxygen', 'held_station', 'A1_both_direct')
    if not b['reference_is_best_on_development']:
        raise SystemExit(f'v2_summary_table: {BEST} station oxygen: a depth offset now beats none; reword the row')
    cell = b['per_cell']['held_station']['2026']
    e = stored(cell['contrast'], f'{BEST} station oxygen 2026')
    rows.append(row('cop26_O', r'Copernicus depth offset $-$ none, DO', 'Field-2026',
                    contrast(*e[:3], mgd(e[0], 3), mark(e[3], e[4])),
                    change_text(cell_change(cell, f'{BEST} station oxygen 2026')),
                    status(e[3], e[4]) + r'; none is selected on \textsc{Field-57}'))
    e = field57('naive_model', 'naive_oracle', 'do_umol_l')
    rows.append(row('oracle_O', r'Spatial estimator $-$ \textsc{oracle} climatology, DO', 'Field-57',
                    contrast(e['point'], *e['ci'], mgd(e['point'], 2), mark(e['excludes_zero'])),
                    change_text(None), status(e['excludes_zero'])))
    return rows


def forecasting():
    """Question 2: the forecaster selected for each variable against the last value from 1 to 72 h, on the
    common rows of every forecaster (Section 5.2); the rows after them are not swept over seed or grid."""
    rows = []
    for key, variable, arm, label, d in (
            ('forecast_t', 'temperature', 'quantile_tree_weather',
             r'Quantile tree given the weather, median $-$ last value, $T$, 1~to~72\,h', 3),
            ('forecast_o', 'oxygen', 'timesfm3_cov_past',
             r"TimesFM 3.0 given the other probes' past $-$ last value, DO, 1~to~72\,h", 2)):
        b = selected('forecast', variable, '1-72', arm)
        text, change, verdict, _ = over_band(b, d, 'leads', oxygen=variable == 'oxygen')
        rows.append(row(key, label, 'Reserved-2026', text, change, verdict))

    # addenda 64-65: a forecast of the distribution against the climatology of the change, at six hours,
    # from the run that scores every forecaster on the same rows to 72 h (Table bay_quantile)
    pr = 'runs/v2_weather_oracle/report.json'
    for variable, label, d in (
            ('temperature', r'Chronos-2 $-$ climatological quantiles, quantile score, $T$', 3),
            ('oxygen', r'Chronos-2 $-$ climatological quantiles, quantile score, DO', 2)):
        c = need(at(pr, 'scores', variable, 'Reserved-2026', '6', 'contrasts', 'chronos2_net - clim_quantiles'),
                 ('point', 'ci'), f'{pr} {variable}')
        point, (lo, hi) = c['point'], c['ci']
        if variable == 'oxygen':
            point, lo, hi = times_c(point, lo, hi)
            d = mgd(point, d)
        rows.append(row(f'quantile_{variable[0]}', label, 'Reserved-2026',
                        contrast(point, lo, hi, d, mark(excludes(lo, hi))), change_text(None),
                        status(excludes(lo, hi))))

    # addendum 69: the tree of the sequence comparison selected per outer fold, on every grid and seed
    seq = 'runs/v2_sequence_arms/perfold_tree/sequence_arms_table.json'
    seq_cite = 'runs/v2_sequence_arms/perfold_tree/'
    rec = 'runs/v2_perfold_neural_a69.json'
    sw = 'runs/v2_sequence_sweep_a69.json'
    for variable, d in (('temperature', 3), ('oxygen', 3)):
        rules = at(seq, 'variables', variable, 'rules', cite=seq_cite)
        primary = [k for k in rules if k.startswith('fold')]
        if len(primary) != 1:
            raise SystemExit(f'v2_summary_table: {seq} {variable} has no single fold rule: {list(rules)}')
        pairs = {k: triple(v, f'{seq} {variable} {k}') for k, v in rules[primary[0]]['contrasts'].items()}
        # addendum 49, rule D4: the table carries the recurrent network selected on the calendar inner
        # window, so every contrast with it comes from that design
        calendar = at(rec, 'recurrent', variable, 'contrasts', cite=rec)
        for k in [k for k in pairs if 'lstm' in k]:
            if k not in calendar:
                raise SystemExit(f'v2_summary_table: {rec} has no calendar contrast {variable} {k}')
            pairs[k] = triple(calendar[k]['by_design']['lstm@calendar'], f'{rec} {variable} {k}')
        if variable == 'temperature':
            name, (point, lo, hi) = max(pairs.items(), key=lambda kv: abs(kv[1][0]))
            n = sum(excludes(a, b) for _, a, b in pairs.values())
            verdict = 'not distinguishable' if n == 0 else f'{n} of {len(pairs)} exclude zero'
            a, b = name.split(' - ')
            label = rf'{ARM_NAMES[a]} $-$ {ARM_NAMES[b]}, $T$ (largest of six gaps)'
        else:
            name, (point, lo, hi) = max(((k, v) for k, v in pairs.items() if k.startswith('tree - ')),
                                        key=lambda kv: kv[1][0])
            if name != 'tree - chronos_zs':
                raise SystemExit(f'v2_summary_table: the largest oxygen tree gap is now {name}; relabel the row')
            swept = at(sw, 'variables', variable, 'designs', 'calendar', 'contrasts', name, 'verdict', cite=sw)
            point, lo, hi = times_c(point, lo, hi)
            d = mgd(point, d)
            verdict = verdict_label(swept)
            label = r'Tree $-$ zero-shot Chronos-Bolt, DO'
        rows.append(row(f'saturation_{variable[0]}', label, 'Dev-2025',
                        contrast(point, lo, hi, d, mark(excludes(lo, hi))), change_text(None), verdict))
    return rows


#: How Table 6.1 names the arms of the sequence comparison.
ARM_NAMES = {'tree': 'Tree', 'lstm': 'Recurrent network', 'chronos_zs': 'zero-shot Chronos-Bolt',
             'timesfm_zs': 'zero-shot TimesFM'}


def data_loss():
    """Question 3: the method selected for each variable while a probe is silent, from 6 to 72 h on the common
    rows of every silent-probe arm, and for a gap filled once both edges are known, by band of length
    (Sections 5.3 and 5.4)."""
    rows = []
    b = selected('silent_probe', 'temperature', '6-72', 'anchored_direct')
    if any(c is None for _, c, _ in cells(b, '2026')):
        raise SystemExit(f'v2_summary_table: {BEST} silent probe temperature: a lead has no interval')
    text, change, verdict, _ = over_band(b, 3, 'leads')
    rows.append(row('bridge_last', r'Network-aided forecast $-$ last value, $T$, 6~to~72\,h', 'Bridge-2026',
                    text, change, verdict))

    b = selected('silent_probe', 'oxygen', '6-72', 'timesfm3_cov')
    text, change, verdict, _ = over_band(b, 2, 'leads', oxygen=True)
    rows.append(row('silent_o', r'TimesFM 3.0 given the other probes $-$ hour-of-day increment, DO, 6~to~72\,h',
                    'Bridge-2026', text, change, verdict))

    # A gap filled after the fact, against linear interpolation: the bands up to three days carry one pooled
    # contrast over the band's overlap clusters, with the lengths whose own interval does not exclude zero
    # named beside it; past a week each length is its own cell.
    for key, variable, band, arm, label, oxygen in (
            ('gap_short_t', 'temperature', '1-6', 'ens_stack', r'Stacked ensemble $-$ linear, $T$, 1~to~6\,h', False),
            ('gap_mid_t', 'temperature', '12-72', 'timesfm3_cov.bidir',
             r'TimesFM 3.0 given the four nearest probes $-$ linear, $T$, 12~to~72\,h', False),
            ('gap_short_o', 'oxygen', '1-6', 'chronos2_bidir', r'Chronos-2 run from both edges $-$ linear, DO, 1~to~6\,h',
             True),
            ('gap_mid_o', 'oxygen', '12-72', 'timesfm3_cov.bidir',
             r'TimesFM 3.0 given the four nearest sondes $-$ linear, DO, 12~to~72\,h', True)):
        b = selected('gaps', variable, band, arm)
        e, change = band_contrast(b, 'Gap-2026')
        dev, _ = band_contrast(b, 'Gap-Dev')
        kept = [(cell, c, ch) for cell, c, ch in cells(b, '2026') if c is not None]
        each = exceptions(kept, list(b['cells']))
        verdict = status(e[3], e[4]) + ('; by length, ' + ', '.join(each) if each else '')
        if dev[3] and dev[4] and dev[0] < 0:
            verdict += r'; also on \textsc{Gap-Dev}'
        d = mgd(e[0], 2) if oxygen else 3
        rows.append(row(key, label, 'Gap-2026', contrast(*e[:3], d, mark(e[3], e[4])), change_text(change),
                        verdict))

    for key, variable, arm, label, oxygen in (
            ('gap_long_t', 'temperature', 'ens_mean_c2',
             r'Mean of the Gaussian process and Chronos-2 given the four nearest probes $-$ linear, $T$, '
             r'a week to 60~days', False),
            ('gap_long_o', 'oxygen', 'timesfm3_cov.bidir',
             r'TimesFM 3.0 given the four nearest sondes $-$ linear, DO, a week to 60~days', True)):
        b = selected('gaps', variable, '168-1440', arm)
        text, change, verdict, kept = over_band(b, 2 if oxygen else 3, 'lengths', oxygen=oxygen)
        both = all(c is not None and c[3] and c[0] < 0
                   for cohort in ('development', '2026') for _, c, _ in cells(b, cohort))
        if both:
            verdict += ' in both cohorts' if verdict.startswith('excludes zero at every') else ''
        rows.append(row(key, label, 'Gap-Long-2026', text, change, verdict))
    return rows


def features():
    """Question 4, physical features (Section 5.4.1)."""
    rows = []
    pm = 'runs/v2_phase_matched.json'

    def swept(h, variable):
        phases = [triple(at(pm, h, 'phases', str(i), variable, 'contrast'), f'{pm} {h} {i}') for i in range(6)]
        seeds = [phases[0]] + [triple(at(pm, h, 'seeds', sd, variable, 'contrast'), f'{pm} {h} seed {sd}')
                               for sd in ('7', '123')]
        return phases, seeds

    # The value shown has every arm selected per outer fold (addenda 47 and 48); the verdict is read over
    # the phases and seeds with that value as a further replicate, as the artifacts record it. The change is
    # read from the two arms' per-fold errors in the same cell.
    cells_ = 'runs/v2_perfold.json'
    for key, label, h, variable, d in (
            ('weather_T6', r'Weather $-$ plain tree, $T$, 6\,h', 'h6', 'temperature', 3),
            ('weather_O24', r'Weather $-$ plain tree, DO, 24\,h', 'h24', 'oxygen', 3)):
        phases, seeds = swept(h, variable)
        cell = at(cells_, 'forecast', 'cells', f'{variable}|{h[1:]}')
        p = triple(cell['contrast']['perfold']['value'], f'{cells_} {variable} {h}')
        change = pct(cell['arms']['weather_tree']['perfold']['mae'], cell['arms']['tree']['perfold']['mae'], p[0],
                     f'{cells_} {variable} {h}')
        verdict = verdict_label(cell['verdict']['with_perfold'])
        if rule(seeds, phases) != cell['verdict']['without_perfold']:
            print(f"DRIFT {key}: the rule gives '{rule(seeds, phases)}', the artifact '{cell['verdict']['without_perfold']}'")
        if variable == 'oxygen':
            p = times_c(*p)
            d = mgd(p[0], d)
        rows.append(row(key, label, 'Dev-2025', contrast(*p, d, mark(excludes(*p[1:]))), change_text(change),
                        verdict))

    pr = 'runs/v2_phase_river.json'
    phases = [triple(at(pr, 'phases', str(i), 'contrasts', 'river_tree - tree'), f'{pr} {i}') for i in range(6)]
    seeds = [phases[0]] + [triple(at(pr, 'seeds', sd, 'contrasts', 'river_tree - tree'), f'{pr} seed {sd}')
                           for sd in ('7', '123')]
    verdict = rule(seeds, phases)
    if verdict != at(pr, 'decision'):
        print(f"DRIFT river: the rule gives '{verdict}', the artifact's decision is '{at(pr, 'decision')}'")
    rest = 'runs/v2_perfold_rest.json'
    swept_river = at(rest, 'forecast', 'swept', 'river_tree - tree')
    raw = triple(swept_river['perfold'], f'{rest} river per fold')
    arms = at(rest, 'forecast', 'cells', 'oxygen|24', 'arms')
    change = pct(arms['river_tree']['perfold']['mae'], arms['tree']['perfold']['mae'], raw[0], f'{rest} river per fold')
    p = times_c(*raw)
    rows.append(row('river_O24', r'River level $-$ plain tree, DO, 24\,h', 'Dev-2025',
                    contrast(*p, mgd(p[0], 3), mark(excludes(*p[1:]))), change_text(change),
                    verdict_label(swept_river['with_perfold'])))
    return rows


def priors():
    """Question 4, priors and transformations (Section 5.4.2); the vertical prior is under question 1."""
    rows = []
    au = MGL + 'C3/v2_bottom_aou/aou_units/summary.json'

    def ridge(cohort, a, b):
        e = pick(au, ('contrasts',), cohort=cohort, a=a, b=b)
        errors = at(au, 'mae')
        change = pct(errors[f'{cohort}|{a}']['nested'], errors[f'{cohort}|{b}']['nested'], e['diff'],
                     f'{au} {cohort} {a} - {b}')
        return e, change

    # both ridge rows with the per-litre saturation, so that the second is part of the first
    e, change = ridge('gen2_164', 'aou_l', 'direct_do')
    rows.append(row('aou_ridge', r'AOU $-$ concentration, ridge, DO', 'Field-2026',
                    contrast(e['diff'], *e['ci'], mgd(e['diff'], 2), mark(e['excludes_zero'], e['jackknife_excludes_zero'])),
                    change_text(change), status(e['excludes_zero'], e['jackknife_excludes_zero'])))
    e, change = ridge('gen2_164', 'aou_l', 'direct_do_ts')
    full = pick(au, ('contrasts',), cohort='full_722', a='aou_l', b='direct_do_ts')
    verdict = status(e['excludes_zero'], e['jackknife_excludes_zero'])
    verdict += (r'; not when trained on \textsc{CTD-722}' if not full['excludes_zero']
                else f"; {starred(full['diff'], *full['ci'], mgd(full['diff'], 2))} trained on " + r'\textsc{CTD-722}')
    rows.append(row('aou_ridge_ts', r'AOU $-$ concentration, ridge given $T$ and $S$, DO', 'Field-2026',
                    contrast(e['diff'], *e['ci'], mgd(e['diff'], 2), mark(e['excludes_zero'], e['jackknife_excludes_zero'])),
                    change_text(change), verdict))
    rest = AOU_REST
    errors = at(rest, 'aou', 'mae', 'perfold')
    pf = pick(rest, ('aou', 'contrasts'), contrast='aou_tree - weather_tree')
    e = triple(pf['perfold']['value'], f'{rest} aou tree per fold')
    change = pct(errors['aou_tree'], errors['weather_tree'], e[0], f'{rest} aou tree per fold')
    rows.append(row('aou_tree', r'AOU tree $-$ weather tree, DO, 6\,h', 'Dev-2025',
                    contrast(*e, mgd(e[0], 3), mark(excludes(*e[1:]))), change_text(change),
                    status(excludes(*e[1:]))))
    # The thermodynamic baseline: persisted utilisation, concentration recomputed from the forecast
    # temperature; per fold, as the other Dev-2025 rows.
    pf = pick(rest, ('aou', 'contrasts'), contrast='thermodynamic - weather_tree')
    e = triple(pf['perfold']['value'], f'{rest} thermodynamic per fold')
    change = pct(errors['thermodynamic'], errors['weather_tree'], e[0], f'{rest} thermodynamic per fold')
    rows.append(row('thermo_tree', r'Thermodynamic baseline $-$ weather tree, DO, 6\,h', 'Dev-2025',
                    contrast(*e, mgd(e[0], 2), mark(excludes(*e[1:]))), change_text(change),
                    status(excludes(*e[1:]))))
    return rows


def operators():
    """Question 4, hard constraints, dynamical operators and output gates (Section 5.4.3)."""
    rows = []
    sc = 'runs/v2_seed_contrasts.json'

    def by_seed(name):
        cc = at(sc, 'contrasts', name, 'by_seed')
        return [(cc[sd]['point'], cc[sd]['ci'][0], cc[sd]['ci'][1])
                for sd in ('42', '7', '123') if need(cc, (sd,), f'{sc} {name}')]

    # Graph arms and the tree selected per outer fold (addendum 49), in the seed-spread stream; the reading
    # counts the per-fold value as a fourth replicate beside the three seeds (rule D3).
    pn = 'runs/v2_perfold_neural.json'

    def perfold(name):
        return triple(at(pn, 'contrasts', f'seed|{name}', 'values', 'perfold', cite=pn), f'{pn} {name}')

    def perfold_change(name, point):
        """The change from the two arms' per-fold errors, which the same artifact stores per arm."""
        variable, h, pair = name.split('|')
        arm, ref = pair.split(' - ')
        err = lambda a: at(pn, 'arms', f'{variable}|{a}|h{h}', 'values', 'perfold', cite=pn)[0]
        return pct(err(arm), err(ref), point, f'{pn} {name}')

    # The two graph contrasts are swept over the six issuance grids as well (addendum 82, part E,
    # runs/v2_addendum82_E.json): seeds 42, 7 and 123 on the reference grid, offsets 0 to 5 at seed 42,
    # and the per-fold value as one more replicate. The seeds read here must equal the seed spread's.
    ev = 'runs/v2_addendum82_E.json'

    def reading(key, name, seeds, per):
        entry = at(ev, 'contrasts', key, cite=ev)
        if f"{entry['variable']}|{entry['horizon_hours']}|{entry['arm']} - {entry['reference']}" != name:
            raise SystemExit(f'v2_summary_table: {ev} {key} is not {name}')
        cells = {sd: (c['point'], c['lo'], c['hi']) for sd, c in entry['cells'].items()}
        swept_seeds = [cells['0|42'], cells['0|7'], cells['0|123']]
        # the seed spread stores four decimals, the sweep six
        if max(abs(a[0] - b[0]) for a, b in zip(seeds, swept_seeds)) > 5e-5:
            raise SystemExit(f'v2_summary_table: {name}: the sweep and the seed spread disagree on the seeds')
        phases = [cells[f'{k}|42'] for k in range(6)]
        code = rule(swept_seeds + [per], phases)
        if code != entry['verdict']:
            print(f"DRIFT {name}: the rule gives {code}, the sweep records {entry['verdict']}")
        return verdict_label(code)

    name = 'oxygen|6|conservative_exchange - learned_source'
    seeds, p = by_seed(name), perfold(name)
    reading_o = reading('E3', name, seeds, p)      # unit-free: signs and intervals
    change = perfold_change(name, p[0])
    p = times_c(*p)
    rows.append(row('constraint_O6', r'Conservative $-$ learned graph, DO, 6\,h', 'Dev-2025',
                    contrast(*p, mgd(p[0], 3), mark(excludes(*p[1:]))), change_text(change), reading_o))

    name = 'temperature|6|learned_source - tree'
    seeds, p = by_seed(name), perfold(name)
    rows.append(row('operator_T6', r'Learned graph $-$ tree, $T$, 6\,h', 'Dev-2025',
                    contrast(*p, 3, mark(excludes(*p[1:]))), change_text(perfold_change(name, p[0])),
                    reading('E1', name, seeds, p)))

    rc = 'runs/v2_horizon_reserved_cells.json'
    e = times_c(*triple(at(rc, 'variables', 'oxygen', 'contrasts', 'learned_source - tree', 'published'), rc))
    dev = times_c(*perfold('oxygen|24|learned_source - tree'))
    verdict = status(excludes(*e[1:])) + r'; \textsc{Dev-2025}: ' + starred(*dev, mgd(dev[0], 3))
    rows.append(row('operator_O24', r'Learned graph $-$ tree, DO, 24\,h', 'Reserved-2026',
                    contrast(*e, mgd(e[0], 3), mark(excludes(*e[1:]))), change_text(None), verdict))

    # Output gates on the recursive rollout (addendum 5): the empirical range bound against no gate on
    # temperature in the network blackout, and whether the thermodynamic and rate bounds ever act.
    # The gate rollouts read the neighbours at each step's issuance hour (addendum 59, part C2);
    # the range bound's effect carries a paired onset-week interval (range_bound_effect.json).
    folder = 'runs/v2_physics_gate/perfold/'      # addendum 69: the rollouts with every tree selected per fold
    eff = at(folder + 'range_bound_effect.json', 'results', 'temperature|network', 'by_lead', '6', cite=folder)
    point, (lo, hi) = eff['point'], eff['ci']
    change = pct(eff['mae_range'], eff['mae_none'], point, f'{folder}range_bound_effect.json')
    rows.append(row('gate_range', r'Range bound $-$ none, $T$, blackout', 'Dev-2025',
                    contrast(point, lo, hi, 3, mark(excludes(lo, hi))) + r' at 6\,h', change_text(change),
                    status(excludes(lo, hi))))
    # Each gate's firings against the steps on which it could act: the rate
    # bound alone, and the thermodynamic bound inside the full set; "inert" means no error moves.
    rate_fires = rate_steps = sat_fires = sat_steps = 0
    moved = 0.0
    for name in ('gate_temperature_network_issuance_summary.json', 'gate_temperature_single_issuance_summary.json',
                 'gate_oxygen_network_issuance_summary.json', 'gate_oxygen_single_issuance_summary.json'):
        summ = at(folder + name, cite=folder)
        rate_fires += summ['fire_counts'].get('rate', 0)
        rate_steps += summ['evaluable_steps']['rate']['rate']
        sat_fires += summ['fire_by_type'].get('saturation', 0)
        sat_steps += summ['evaluable_steps']['all']['saturation']
        moved = max(moved, max(abs(e['rate']['mae'] - e['none']['mae']) for e in summ['by_lead'].values()))
    rows.append(row('gate_inert', r'Saturation, rate bounds: fires', 'Dev-2025',
                    f'{count(sat_fires)}/{count(sat_steps)}, {count(rate_fires)}/{count(rate_steps)}', '---',
                    ('negligible error effect' if sat_fires == 0 and moved < 5e-5 else 'active')))
    return rows


# ------------------------------------------------------------------------------------ expected

#: The verdicts Chapters 5 and 6 state, in this script's words (the rows of the selected methods since
#: 2026-10-20; no contrast is read against a margin since then, and a row over leads or gap lengths names
#: the cells whose interval does not exclude zero).
EXPECTED = {
    'prior57_T': r'excludes zero; $-0.190^{\ast}$ with one penalty',
    'cop26_T': 'excludes zero, bootstrap alone',
    'cast26_T': 'excludes zero',
    'oracle_T': 'excludes zero',
    'cast57_O': 'not distinguishable',
    'cop26_O': r'excludes zero; none is selected on \textsc{Field-57}',
    'oracle_O': 'not distinguishable',
    'forecast_t': r'not distinguishable at 24\,h',
    'forecast_o': r'not distinguishable at 24 and 72\,h',
    'quantile_t': 'excludes zero',
    'quantile_o': 'excludes zero',
    'saturation_t': '4 of 6 exclude zero',
    'saturation_o': 'sign-stable',
    'bridge_last': 'excludes zero at every lead',
    'silent_o': 'not distinguishable at any lead',
    'gap_short_t': r'excludes zero; also on \textsc{Gap-Dev}',
    'gap_mid_t': r'excludes zero; also on \textsc{Gap-Dev}',
    'gap_short_o': r'excludes zero; by length, not distinguishable at 2 and 6\,h, bootstrap alone at 3\,h; '
                   r'also on \textsc{Gap-Dev}',
    'gap_mid_o': r'excludes zero; also on \textsc{Gap-Dev}',
    'gap_long_t': 'excludes zero at every length in both cohorts',
    'gap_long_o': r'not distinguishable at 720\,h; bootstrap alone at 504\,h; n.e. at 1\,440\,h',
    'weather_T6': 'stable',
    'weather_O24': 'sign-stable',
    'river_O24': 'sign-stable',
    'aou_ridge': 'excludes zero',
    'aou_ridge_ts': r'excludes zero; not when trained on \textsc{CTD-722}',
    'aou_tree': 'not distinguishable',
    'thermo_tree': 'excludes zero',
    'constraint_O6': 'choice-sensitive',
    'operator_T6': 'sign-stable',
    'operator_O24': r'not distinguishable; \textsc{Dev-2025}: $-0.0151^{\ast}$',
    'gate_range': 'excludes zero',
    'gate_inert': 'negligible error effect',
}


#: Rows computed and checked but not printed: Table 6.1 keeps the rows a reader needs to answer the
#: research questions, and these are reported in Chapter 5 or the supplement beside their tables. Under
#: questions 1 to 3 it prints only each task's selected method against the reference (2026-10-20), so the
#: comparisons with the oracle climatology, the climatological quantiles and zero-shot Chronos-Bolt are
#: kept here as checks.
NOT_PRINTED = {'oracle_T', 'oracle_O', 'cast57_O', 'quantile_t', 'quantile_o', 'saturation_t', 'saturation_o',
               'weather_O24', 'aou_ridge_ts', 'aou_tree', 'operator_O24', 'gate_inert'}


#: Each printed physical-guidance row read on the other period (addendum 86, runs/v2_addendum86.json): a row of
#: the development cohorts gains its reading on the 2026 test period, and the ridge row, scored on the casts of
#: 2026, its reading on the casts of 2025 with the ridges trained on the earlier casts. (the words printed,
#: the part and the contrast's key in the artifact)
GUIDANCE = 'runs/v2_addendum86.json'
OTHER_PERIOD = {
    'weather_T6': ('2026', 'A1', 'temperature|6|weather_tree - tree'),
    'river_O24': ('2026', 'B', 'oxygen|24|river_tree - tree'),
    'aou_ridge': ('casts of 2025', 'E1', 'gen2_164|aou_l - direct_do'),
    'thermo_tree': ('2026', 'A2', 'oxygen|6|thermodynamic_2026 - weather_tree'),
    'constraint_O6': ('2026', 'C', 'oxygen|6|conservative_exchange - learned_source'),
    'operator_T6': ('2026', 'C', 'temperature|6|learned_source - tree'),
    'gate_range': ('2026', 'D', 'temperature|network|6|range - none'),
}


def other_period(key):
    """'2026: $-3.5\\,\\%$, not distinguishable': the change of the row's contrast on the other period, from the
    two errors stored beside it, and its interval status, read from runs/v2_addendum86.json."""
    words, part, name = OTHER_PERIOD[key]
    found = [c for c in at(GUIDANCE, 'contrasts') if c['part'] == part and c['key'] == name]
    if len(found) != 1:
        raise SystemExit(f'v2_summary_table: {GUIDANCE} has {len(found)} contrasts {part} {name}')
    c = need(found[0], ('mae_x', 'mae_y', 'difference', 'ci', 'interval_status'), f'{GUIDANCE} {part} {name}')
    lo, hi = (float(x) for x in c['ci'])
    if (c['interval_status'] == 'excludes zero') != excludes(lo, hi):
        raise SystemExit(f'v2_summary_table: {GUIDANCE} {part} {name}: the stored status disagrees with its interval')
    jack = c.get('jackknife_t')
    change = pct(c['mae_x'], c['mae_y'], c['difference'], f'{GUIDANCE} {part} {name}')
    return f'{words}: {change_text(change)}, {status(excludes(lo, hi), None if jack is None else excludes(*jack))}'


def printed(groups):
    out = []
    for title, rows in groups:
        keep = []
        by_key = {r['key']: r for r in rows}
        for r in rows:
            if r['key'] in NOT_PRINTED:
                continue
            r = dict(r)
            if r['key'] == 'aou_ridge' and 'aou_ridge_ts' in by_key:
                ts = by_key['aou_ridge_ts']['contrast'].split(' [')[0].split(' $[')[0]
                r['verdict'] += rf'; {ts} given $T$ and $S$'
            if r['key'] in OTHER_PERIOD:
                r['verdict'] += '; ' + other_period(r['key'])
                r['sources'] = list(dict.fromkeys(r['sources'] + [GUIDANCE]))
            keep.append(r)
        out.append((title, keep))
    missing = set(OTHER_PERIOD) - {r['key'] for _, rows in out for r in rows}
    if missing:
        raise SystemExit(f'v2_summary_table: rows read on the other period but not printed: {sorted(missing)}')
    return out


#: false-discovery control over the table's own estimates (addendum 79), written by scripts.v2_fdr from
#: the row builders above, so it is read only after printed() and never by a builder
FDR = 'runs/v2_fdr.json'
BH_MARK, BY_MARK = r'\S', r'\ddagger'


def fdr_marks(groups):
    """Mark each starred estimate that false-discovery control over the table does not retain: BH_MARK
    when the Benjamini-Hochberg procedure drops it, BY_MARK when only the Benjamini-Yekutieli procedure,
    valid under any dependence, does. A row over leads or gap lengths is marked when it is so for one of its
    cells at least, which the note says. Returns the groups and the note's sentence."""
    report = json.loads((BAY / FDR).read_text(encoding='utf-8'))['F1_answers']
    bh = {e['row'] for e in report['bh_not_retained']}
    by = {e['row'] for e in report['by_not_retained']} - bh
    ranged = {e['row'] for e in report['bh_not_retained'] + report['by_not_retained'] if e['cell']}
    out = []
    for title, rows in groups:
        keep = []
        for r in rows:
            m = BH_MARK if r['key'] in bh else BY_MARK if r['key'] in by else ''
            if m:
                # on the row's label, which wraps, rather than on the estimate, whose column sets its width
                r = dict(r, result=r['result'] + f'$^{{{m}}}$')
            keep.append(r)
        out.append((title, keep))
    n = report['counts_side']['tests']
    words = []
    if bh:
        words.append(rf"$^{{{BH_MARK}}}$ not retained by the Benjamini--Hochberg procedure over the table's {n} estimates")
    if by:
        words.append(rf"$^{{{BY_MARK}}}$ not retained by false-discovery control valid under any dependence")
    note = '; '.join(words) + (' (on a range, at one of its leads or lengths at least)' if ranged else '')
    note = (note + r', Section~\ref{sec:meth_multiplicity}; ') if words else ''
    return out, note


def check(rows):
    drift = 0
    for r in rows:
        want = EXPECTED.get(r['key'])
        if want is None:
            print(f"NEW {r['key']}: no stated verdict to compare with; read '{r['verdict']}' against the chapters")
        elif r['verdict'] != want:
            drift += 1
            print(f"DRIFT {r['key']}: the artifacts now give '{r['verdict']}', the chapters say '{want}'. "
                  'Re-read the prose before using this table.')
    return drift


# ---------------------------------------------------------------------------------------- write

#: (heading or None to continue the previous group, row builder)
GROUPS = (
    (r'1.\ State estimation at an unmonitored station (Sections~\ref{sec:res_spatial} and~\ref{subsec:res_boundary_ablation})',
     state_estimation),
    (r'2.\ Forecasting at a monitored probe (Section~\ref{sec:res_forecast})', forecasting),
    (r'3.\ Behaviour under data loss (Sections~\ref{sec:res_outage} and~\ref{subsec:res_gap_recovery})', data_loss),
    (r'4.\ Physical guidance by kind (Section~\ref{sec:res_physics}); the depth offset is under question~1',
     features),
    (None, priors),
    (None, operators),
)


def stacked(text):
    """An estimate above its interval in one cell, so that the column needs only the interval's width and
    the row labels wrap less; a range over leads or lengths has no interval and stays on one line."""
    head, sep, rest = text.partition(' $[')
    return rf'\makecell[tl]{{{head}\\$[{rest}}}' if sep else text


def render(groups, fdr_note=''):
    printed_rows = [r for _, rows in groups for r in rows]
    dagger = (r'$^{\dagger}$ and \emph{bootstrap alone}: the bootstrap interval alone; '
              if any(r'\dagger' in r['contrast'] or 'bootstrap alone' in r['verdict'] for r in printed_rows) else '')
    lines = []
    for title, rows in groups:
        if title is not None:
            # no extra space above a question's heading: the italic heading row separates the groups, and the
            # float keeps to its page with the other period's readings under question 4
            lines.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
        for r in rows:
            lines.append(f"{r['result']} & {r['cohort']} & {stacked(r['contrast'])} & {r['change']} & {r['verdict']} \\\\")
    body = '\n'.join(lines)
    # 7.5 pt on an 8.8 pt line, the size of the note: the prose names of the selected methods wrap the first
    # column to three or four lines, and at 8 pt on 9.3 pt the float is taller than the page.
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{The answers to the research questions, one result per row.}}
\label{{tab:bay_summary}}
\fontsize{{7.5}}{{8.8}}\selectfont
\setlength{{\tabcolsep}}{{2pt}}
\renewcommand{{\arraystretch}}{{1.0}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}X l l r >{{\raggedright\arraybackslash}}p{{0.22\textwidth}}@{{}}}}
\toprule
\textbf{{Contrast}} & \textbf{{Cohort}} & \textbf{{Estimate and interval}} & \textbf{{Change}} & \textbf{{Verdict}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}

\vspace{{1pt}}
\fontsize{{7.5}}{{8.5}}\selectfont\justifying\noindent A contrast is the first-named arm minus the second,
so a negative value favours the first; $T$ in $^\circ$C, DO in {OXYGEN_UNIT_TEX}. Under questions 1 to 3 the
first arm is the task's selected method and the second the task's reference
(Section~\ref{{subsec:meth_best}}). \emph{{Change}}: the difference as a percentage of the second arm's
error. $^{{\ast}}$ the $95\,\%$
interval over the cohort's independent unit excludes zero, and below about twenty units so does the
delete-one-unit jackknife; {dagger}{fdr_note}a range runs over leads or gap
lengths, from the smallest value to the largest. The sweep verdicts and the interval
statuses \emph{{excludes zero}} and \emph{{not distinguishable}}: Table~\ref{{tab:bay_verdicts}}; a
forecasting contrast with no sweep verdict was not swept;
n.e.: fewer than five independent units;
\emph{{per fold}} and \emph{{with one penalty}}: Section~\ref{{sec:concl_synthesis}}.
\end{{table}}
"""


def render_sources(groups):
    """The sources of Table 6.1: each row with the artifacts it is read from."""
    lines = []
    for i, (title, rows) in enumerate(groups):
        if title is not None:
            if i:
                lines.append(r'\addlinespace[2pt]')
            lines.append(rf'\multicolumn{{2}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
        for r in rows:
            paths = ', '.join(rf'\path{{{k}}}' for k in r['sources'])
            lines.append(f"{r['result']}, {r['cohort']} & {paths} \\\\")
    body = '\n'.join(lines)
    return rf"""\begin{{table}}[htbp]
\centering
\caption[The artifacts behind each row of Table~\ref{{tab:bay_summary}}]{{The artifacts behind each row of
Table~\ref{{tab:bay_summary}}. Paths are relative to the repository's evidence folder, reports/bay\_twin.}}
\label{{tab:bay_summary_sources}}
\fontsize{{8}}{{9}}\selectfont
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.46\textwidth}} >{{\raggedright\arraybackslash}}X@{{}}}}
\toprule
\textbf{{Row}} & \textbf{{Artifacts}} \\
\midrule
{body}
\addlinespace[2pt]
\multicolumn{{2}}{{@{{}}l}}{{\textit{{False-discovery control over the table}}}} \\
Every row with an interval & \path{{{FDR}}} \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=OUT, help='where to write the table (default: the thesis)')
    args = ap.parse_args()
    _ROW.clear()
    groups = [(title, build()) for title, build in GROUPS]
    drift = check([r for _, rows in groups for r in rows])
    groups, fdr_note = fdr_marks(printed(groups))
    tex = render(groups, fdr_note)
    args.out.write_text(tex, encoding='utf-8')
    if args.out == OUT:
        SOURCES.write_text(render_sources(groups), encoding='utf-8')
    print(tex)
    print(f'wrote {args.out}: {sum(len(r) for _, r in groups)} rows, {len(CITED)} artifacts cited, '
          f'{drift} verdicts differ from the chapters')


if __name__ == '__main__':
    main()
