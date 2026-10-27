"""Write the gap-recovery tables of Chapter 5 and of the supplement's chapter S1 from their artifacts.

Generated rather than typed, because each carries some seventy intervals and a transcription slip
in one of them would be invisible. All read only published artifacts:

  table_bay_gap_pooled    every comparand against linear interpolation, pooled over gap lengths,
                          on the development record (runs/v2_gap_intervals.json, runs/v2_gap_splines.json)
  table_bay_gap_lengths   bidirectional Chronos against linear by gap length, on the development
                          record and on the untouched 2026 read (runs/v2_gap_intervals*.json)
  table_bay_gap_long      addendum 32: gaps of a week to sixty days, two cohorts (runs/v2_gap_long.json)
  table_bay_gap_improvements, _variants, _variants_long, _exogenous, _chronos_covariates_*
                          addenda 33-35 (runs/v2_gap_{variants,exogenous,chronos_covariates}.json)

Addendum 82, part C added the Gap-2026 episodes of 1 to 3 h and gave PCHIP an interval. The three
artifacts of addenda 33-35 are therefore read as rescored with those episodes
(runs/v2_addendum82/C/rescored/, every entry outside the Gap-2026 1-6 h band unchanged), PCHIP's
contrast and the Gap-2026 cells at 1 to 3 h come from runs/v2_addendum82_C.json, and each 6 h cell
of that file must equal the published one.

Two rules of Chapter 4 are applied mechanically. Where the percentile interval excludes zero and the
delete-one-unit jackknife does not, the cell carries a section sign, since below about twenty units
such a result is not robust. The long cohorts are tiled, so their windows are independent only
within one length: past a week every contrast is shown per length, never pooled across lengths.

Usage: python -m scripts.v2_gap_tables
"""
import json
from pathlib import Path

from src.units import OXYGEN_UNIT_TEX, column_decimals, factor, mg_decimals

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
TABLES = ROOT / 'thesis_template/content/tables'
#: decimals in each artifact's own unit; an oxygen column prints in mg/l with the decimals that keep
#: these significant figures (`digits`)
DIGITS = {'temperature': 3, 'oxygen': 2}
#: the gap artifacts hold oxygen per kilogram (the sondes' unit): the tables print c times it, in mg/l
SCALE = {'temperature': 1.0, 'oxygen': factor()}
#: the floor or threshold a gain is read against, in the printed unit
FLOOR = {'temperature': 0.20, 'oxygen': 0.059}
UNIT = {'temperature': r'$^\circ$C', 'oxygen': OXYGEN_UNIT_TEX}
#: addenda 33-35 rescored with the Gap-2026 episodes of 1 to 3 h (addendum 82, part C)
RESCORED = {name: f'v2_addendum82/C/rescored/{name}' for name in
            ('v2_gap_variants.json', 'v2_gap_exogenous.json', 'v2_gap_chronos_covariates.json')}
VARIANTS, EXOGENOUS, COVARIATES = (RESCORED[n] for n in ('v2_gap_variants.json', 'v2_gap_exogenous.json',
                                                          'v2_gap_chronos_covariates.json'))
#: addendum 82, part C: PCHIP with an interval, and the Gap-2026 cells by length
PART_C = 'v2_addendum82_C.json'


def part_c_cell(cohort, variable, length, arm='chronos_bidir', reference='linear', family='C1'):
    """One per-length cell of part C, on the overlap clusters of the whole strict cohort."""
    found = [c for c in json.loads((RUNS / PART_C).read_text(encoding='utf-8'))['cells']
             if (c['family'], c['cohort'], c['variable'], c['length'], c['arm'], c['reference'])
             == (family, cohort, variable, length, arm, reference)]
    if len(found) != 1:
        raise SystemExit(f'v2_gap_tables: {len(found)} part-C cells for {cohort} {variable} {length} h {arm}')
    return found[0]


def scaled(entry, variable):
    """A contrast entry with its estimate and interval in the printed unit (oxygen in mg/l)."""
    k = SCALE[variable]
    if entry is None or k == 1.0:
        return entry
    return {**entry, **{key: k * entry[key] for key in ('delta', 'ci_low', 'ci_high')
                        if entry.get(key) is not None}}


def digits(variable, values):
    """One number of decimals for a column of `variable` values in their artifact's unit (contrast
    entries or numbers): DIGITS for temperature; for oxygen the most any value needs in mg/l."""
    if variable != 'oxygen':
        return DIGITS[variable]
    found = [v['delta'] if isinstance(v, dict) else v for v in values if v is not None]
    return column_decimals(found, DIGITS[variable])

#: The bidirectional LSTM and compressed sensing were scored but are not reported. Each has a free
#: parameter the tuning protocol requires a 100-trial study for, and neither had one: the LSTM's
#: declared study tuned a tree whose settings were then discarded, and compressed sensing ran at a
#: fixed penalty. No conclusion rested on either.
FAMILIES = [
    ('Parameter-free', [('persistence', 'persistence'), ('spline', 'cubic spline'),
                        ('pchip', 'PCHIP'), ('climatology', 'climatology')]),
    ('Neighbouring probes', [('nearest_probe', 'nearest probe'),
                             ('gpr', 'Gaussian process (kriging)')]),
    # The "causal" arm keeps the gap length, and so the distance to the far edge: it is a one-edge
    # arm, labelled as such.
    ('Direct regression', [('regression_causal', 'gradient boosting, one edge'),
                           ('regression_retro', 'gradient boosting, both edges'),
                           ('polynomial', 'polynomial ridge')]),
    ('Recursive rollout', [('rolling_forward', 'forward'), ('rolling_backward', 'backward'),
                           ('hybrid', 'hybrid blend')]),
    ('In-context tabular', [('tabpfn', 'TabPFN'), ('tabicl', 'TabICLv2')]),
    ('Pretrained time series', [('chronos_forward', 'Chronos-Bolt, forward'),
                                ('chronos_bidir', 'Chronos-Bolt, bidirectional')]),
]
#: Methods that read no other probe, marked in the tables because they are what a probe with no
#: neighbour can use (Section 5.3). The tree, rollout and in-context families read the four
#: nearest probes' readings among their features; the polynomial ridge keeps the first fourteen
#: feature columns, all the probe's own (v2_gap_recovery.polynomial_predict); the Kalman bridge
#: reads the forcing at most.
NO_NEIGHBOUR = {'persistence', 'spline', 'pchip', 'climatology', 'chronos_forward', 'chronos_bidir',
                'polynomial',
                'linear', 'chronos2_bidir', 'arx_bridge', 'arx_bridge_exo', 'chronos2_exo_bidir',
                'bolt_resid_bidir', 'bolt_resid_lin_bidir'}
NO_NEIGHBOUR_MARK = r'$^{\circ}$'
NO_NEIGHBOUR_NOTE = (r"$^{\circ}$ reads no other probe, only the probe's own record and at most "
                     r'the weather station and the Magra level (Section~\ref{subsec:res_isolated}).')


def num(value, digits):
    return f'${value:+.{digits}f}$'.replace('+-', '-')


def hours(length):
    """A gap length as the prose writes it: 1\\,440, not 1440."""
    return f'{int(length):,}'.replace(',', r'\,')


def series(items):
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


#: The temperature threshold is the lower end of the disagreement between co-located probes, not a
#: floor of one instrument; oxygen has the cast-against-sonde floor.
FLOOR_LABEL = {'temperature': 'co-located disagreement from $0.20$', 'oxygen': 'floor $0.059$'}


#: the footnote for the section sign, shared by every table that can carry one
JACKKNIFE_NOTE = (r'$^{\S}$ the delete-one-unit jackknife interval includes zero, so the result is '
                  r'not robust (Section~\ref{subsec:meth_uncertainty}).')


def fragile(entry):
    """The percentile interval excludes zero and the jackknife does not."""
    return bool(entry and entry.get('estimable', True) and entry.get('excludes_zero')
                and entry.get('jackknife_excludes_zero') is False)


def cell(entry, digits, floor=None, ci_size=r'\footnotesize'):
    """Delta with its interval; starred if the interval excludes zero, bold if it favours the arm.

    A section sign joins the star where the jackknife interval includes zero. With a floor, daggers mark
    a gain beyond it (the supplement's tables); the interval is set at `ci_size`, or at the body's size
    when that is empty (the Appendix A tables, set at one size).
    """
    if entry is None:
        return '---'
    if not entry.get('estimable', True):
        return f'\\textcolor{{gray}}{{({num(entry["delta"], digits)})}} {{\\footnotesize n.e.}}'
    inner = f'{entry["delta"]:+.{digits}f}'.replace('+-', '-')
    star = entry['excludes_zero']
    if star:
        inner += '^{\\ast\\S}' if fragile(entry) else '^{\\ast}'
    if star and entry['delta'] < 0:
        inner = f'\\mathbf{{{inner}}}'
        # dagger: the point estimate is beyond the floor; double dagger: the whole interval is
        if floor is not None and entry['ci_high'] < -floor:
            inner += '^{\\ddagger}'
        elif floor is not None and -entry['delta'] > floor:
            inner += '^{\\dagger}'
    value = f'${inner}$'
    ci = f'$[{entry["ci_low"]:+.{digits}f}, {entry["ci_high"]:+.{digits}f}]$'.replace('+-', '-')
    if ci_size:
        ci = f'{{{ci_size}{ci}}}'
    return f'{value} {ci}'


def pooled():
    intervals = json.loads((RUNS / 'v2_gap_intervals.json').read_text())['variables']
    splines = json.loads((RUNS / 'v2_gap_splines.json').read_text())['variables']
    summary = json.loads((RUNS / 'v2_gap_recovery_summary.json').read_text())['variables']
    pchip = json.loads((RUNS / PART_C).read_text(encoding='utf-8'))['pchip']['Gap-Dev']
    for v in ('temperature', 'oxygen'):
        # the interval's point estimate is the aggregate the published spline run kept
        s = splines[v]['overall']
        if abs(pchip[v]['pooled']['delta'] - (s['pchip'] - s['linear'])) > 5e-4:
            raise SystemExit(f'v2_gap_tables: PCHIP {v}: the part-C contrast differs from v2_gap_splines.json')
    rows = []
    keys = [key for _, members in FAMILIES for key, _ in members]
    d = {v: digits(v, [intervals[v]['overall'].get(key) for key in keys if key != 'pchip']
                   + [pchip[v]['pooled']])
         for v in ('temperature', 'oxygen')}
    for family, members in FAMILIES:
        rows.append(f'\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{family}}}}} \\\\')
        for key, label in members:
            cells = []
            for variable in ('temperature', 'oxygen'):
                if key == 'pchip':
                    cells.append(cell(scaled(pchip[variable]['pooled'], variable), d[variable]))
                else:
                    cells.append(cell(scaled(intervals[variable]['overall'].get(key), variable), d[variable]))
            mark = NO_NEIGHBOUR_MARK if key in NO_NEIGHBOUR else ''
            rows.append(f'\\quad {label}{mark} & {cells[0]} & {cells[1]} \\\\')
        rows.append('\\addlinespace[2pt]')
    linear = {v: summary[v]['baselines']['linear'] for v in ('temperature', 'oxygen')}
    linear_o2 = f"{SCALE['oxygen'] * linear['oxygen']:.{mg_decimals(linear['oxygen'], 3)}f}"
    counts = {v: (intervals[v]['overall']['chronos_bidir']['episodes'], intervals[v]['clusters'])
              for v in ('temperature', 'oxygen')}
    body = '\n'.join(rows[:-1])
    jackknife = (' ' + JACKKNIFE_NOTE) if r'\S' in body else ''
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Every gap-filling method against linear interpolation on \textsc{{Gap-Dev}}, pooled over
gap lengths: the method's macro mean absolute error minus that of linear interpolation on the same
masked episodes; negative favours the method. Linear itself scores
${linear['temperature']:.4f}\,^\circ$C on {hours(counts['temperature'][0])} temperature episodes and
${linear_o2}\,${UNIT['oxygen']} on {counts['oxygen'][0]} oxygen episodes. Regression and
rollout methods were tuned by their own 100-trial studies inside calendar-block folds; the Gaussian
process fits its kernel to each episode by marginal likelihood; in-context and pretrained models ran
zero-shot.}}
\label{{tab:bay_gap_pooled}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{@{{}}lll@{{}}}}
\toprule
\textbf{{Method}} & \textbf{{Temperature}} ($^\circ$C) & \textbf{{Oxygen}} ({UNIT['oxygen']}) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{2pt}}
\scriptsize\justifying\noindent $^{{\ast}}$ interval excludes zero; bold where it does so in the
method's favour.{jackknife} Paired 95\,\% bootstrap intervals over overlap clusters (episodes on one
probe whose masked stretches share an hour), 600 draws: {counts['temperature'][1]} clusters for
temperature and {counts['oxygen'][1]} for oxygen. Climatology is fitted outside the masked stretch; the nearest probe falls back to
persistence when that probe is dark, and has no value when it is dark at the gap's edge, so its row
rests on slightly fewer episodes. The one-edge regression sees the left edge and the gap's
length. TabPFN and TabICLv2 receive missing features as zero and at most 8\,000 training rows.
{NO_NEIGHBOUR_NOTE}\artifact{{runs/v2_gap_intervals.json}}%
\artifact{{runs/v2_gap_splines.json}}\artifact{{runs/v2_gap_recovery_summary.json}}\artifact{{runs/{PART_C}}}
\end{{table}}
"""


def by_length():
    dev = json.loads((RUNS / 'v2_gap_intervals.json').read_text())['variables']
    confirm = json.loads((RUNS / 'v2_gap_intervals_reserved2026.json').read_text())['variables']
    summary = json.loads((RUNS / 'v2_gap_recovery_summary.json').read_text())['variables']
    rows = []
    for variable in ('oxygen', 'temperature'):
        floor, k = FLOOR[variable], SCALE[variable]
        rows.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{variable.capitalize()}}} '
                    f'({UNIT[variable]})}} \\\\')
        # Gap-Dev is defined at 1 to 72 h; gaps of a week and more are read on the long cohorts
        # (table_bay_gap_long), so the rows past 72 h that this artifact also holds are not printed
        lengths = sorted(int(L) for L in dev[variable]['by_gap_length'] if int(L) <= 72)
        found = [(L, dev[variable]['by_gap_length'][str(L)].get('chronos_bidir'),
                  confirm.get(variable, {}).get('by_gap_length', {}).get(str(L), {}).get('chronos_bidir'),
                  summary[variable]['by_gap_length'][str(L)]['linear']) for L in lengths]
        # Gap-2026 at 1 to 3 h from part C; its 6 h cell must be the published one
        six = part_c_cell('Gap-2026', variable, 6)
        published6 = confirm[variable]['by_gap_length']['6']['chronos_bidir']
        if abs(six['delta'] - published6['delta']) > 5e-5 or six['clusters'] != published6['clusters']:
            raise SystemExit(f'v2_gap_tables: {variable} Gap-2026 6 h: part C differs from the published cell')
        found = [(L, a, part_c_cell('Gap-2026', variable, L) if L in (1, 2, 3) else b, lin)
                 for L, a, b, lin in found]
        dl, da, db = (digits(variable, [row[j] for row in found]) for j in (3, 1, 2))
        for L, a, b, linear in found:
            rows.append(f'{hours(L)} & ${k * linear:.{dl}f}$ & {cell(scaled(a, variable), da, ci_size="")} & '
                        f'{a["clusters"]} & {cell(scaled(b, variable), db, ci_size="")} & '
                        f'{b["clusters"] if b else "---"} \\\\')
        rows.append('\\addlinespace')
    body = '\n'.join(rows[:-1])
    jackknife = (' ' + JACKKNIFE_NOTE) if r'\S' in body else ''
    unestimable = ('\nn.e.: fewer than five overlap clusters, so no interval is reported and the point '
                   'estimate, in grey,\ndescribes one to four stretches of record (the $k$ column) '
                   'rather than an estimate.') if 'n.e.' in body else ''
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Bidirectional Chronos-Bolt minus linear interpolation by length.}}
\label{{tab:bay_gap_lengths}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}rlllll@{{}}}}
\toprule
\makecell[r]{{\textbf{{Gap}}\\\textbf{{(h)}}}} & \makecell[l]{{\textbf{{Linear}}\\\textbf{{MAE}}}} & \textsc{{Gap-Dev}} & $k$ &
\textsc{{Gap-2026}} & $k$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{2pt}}
\scriptsize\justifying\noindent Negative favours Chronos; $k$ is the number of independent overlap
clusters behind each contrast. $^{{\ast}}$ interval excludes zero; bold where it favours Chronos.{jackknife}{unestimable}
The linear column is the development-record error.\artifact{{runs/v2_gap_intervals.json}}\artifact{{runs/v2_gap_intervals_reserved2026.json}}%
\artifact{{runs/v2_gap_recovery_summary.json}}\artifact{{runs/{PART_C}}}
\end{{table}}
"""


def long_gaps():
    """Addendum 32: gaps of a week to sixty days, on windows whose interior may hold real gaps."""
    long = json.loads((RUNS / 'v2_gap_long.json').read_text())
    arms = {'oxygen': [('chronos_bidir', 'Chronos'), ('gpr', 'GPR')],
            'temperature': [('gpr', 'GPR'), ('nearest_live_probe', 'nearest reporting probe')]}
    rows = []
    for variable in ('oxygen', 'temperature'):
        k = SCALE[variable]
        (a, a_label), (b, b_label) = arms[variable]
        rows.append(f'\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{variable.capitalize()}}} '
                    f'({UNIT[variable]}): {a_label} and {b_label} '
                    f'minus linear}} \\\\')
        dev = long['Gap-Long-Dev'][variable]['by_gap_length']
        new = long['Gap-Long-2026'][variable]['by_gap_length']
        columns = [(arm, cohort, block) for arm in (a, b)
                   for cohort, block in (('development', dev), ('2026', new))]
        # one number of decimals per column: the linear error, then each method and cohort
        dl = digits(variable, [dev[L]['mae']['linear'] for L in dev])
        dc = [digits(variable, [block.get(L, {}).get('minus_linear', {}).get(arm, {}).get('conservative')
                                for L in dev]) for arm, _, block in columns]
        for L in sorted(dev, key=int):
            units = f'{dev[L]["conservative_units"]}/{new[L]["conservative_units"]}' if L in new \
                else f'{dev[L]["conservative_units"]}/---'
            values, intervals = [], []
            for (arm, cohort, block), d in zip(columns, dc):
                got = block.get(L, {}).get('minus_linear', {}).get(arm, {})
                entry = scaled(got.get('conservative'), variable)
                # Value and interval on separate lines, so that eight columns of intervals fit the
                # text width at one type size.
                value, _, interval = cell(entry, d, ci_size='').partition(' ')
                values.append(value)
                intervals.append(interval)
            rows.append(f'{hours(L)} & {units} & ${k * dev[L]["mae"]["linear"]:.{dl}f}$ & '
                        + ' & '.join(values) + ' \\\\')
            rows.append(' & & & ' + ' & '.join(intervals) + ' \\\\[1pt]')
        rows.append('\\addlinespace')
    body = '\n'.join(rows[:-1])
    jackknife = (' ' + JACKKNIFE_NOTE) if r'\S' in body else ''
    # The registered predictions and their outcomes stay in the declaration and the artifact
    # (runs/v2_gap_long.json, registered_predictions); the printed note does not state them.
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{A week to sixty days: two methods minus linear interpolation.}}
\label{{tab:bay_gap_long}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}rrlllll@{{}}}}
\toprule
\textbf{{Gap (h)}} & $k$ & \textbf{{Linear}} & \multicolumn{{2}}{{l}}{{\textbf{{First method}}}} &
\multicolumn{{2}}{{l}}{{\textbf{{Second method}}}} \\
 & & & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{2pt}}
\scriptsize\justifying\noindent The windows' interiors may contain real short gaps, and only hours
that were observed are scored; on \textsc{{Gap-Long-2026}} fitted methods are applied frozen. $k$:
the independent units, a probe and calendar month of the gap start (development/2026), resampled
within each length and never pooled across lengths. $^{{\ast}}$ interval excludes zero; bold where it
favours the method.{jackknife} n.e.: fewer than five units. The linear column is the
development-record error.\artifact{{runs/v2_gap_long.json}}
\end{{table}}
"""


NAMES = {'linear': 'linear', 'chronos_bidir': 'Chronos-Bolt', 'gpr': 'GPR',
         'ens_stack': 'the stacked ensemble', 'ens_mean_c2': 'the mean of Chronos-2 with neighbours and GPR',
         'ens_mean': 'the mean of Chronos-Bolt and GPR', 'twin_rule': 'the twin rule',
         'twin_else_gpr': 'twin else GPR', 'twin_else_best': 'twin else best',
         'chronos2_cov_bidir': 'Chronos-2 with neighbours', 'chronos2_bidir': 'Chronos-2',
         'gpr_k8': 'GPR with eight neighbours',
         'anchored_gbm': 'anchored boosting',
         'anchored_gbm_exo': 'anchored boosting with exogenous series',
         'anchored_gbm_rel': 'edge-relative boosting',
         'anchored_gbm_rel_exo': 'edge-relative boosting with exogenous series',
         'arx_bridge': 'the Kalman bridge',
         'arx_bridge_exo': 'the Kalman bridge with exogenous series',
         'chronos2_exo_bidir': 'Chronos-2 with the exogenous series',
         'chronos2_all_bidir': 'Chronos-2 with neighbours and the exogenous series',
         'bolt_resid_bidir': 'Chronos-Bolt on a tree residual',
         'chronos2_cov_resid_bidir': 'Chronos-2 with neighbours on a tree residual',
         'bolt_resid_lin_bidir': 'Chronos-Bolt on a ridge residual',
         'chronos2_cov_resid_lin_bidir': 'Chronos-2 with neighbours on a ridge residual'}


def stored_replacements(dec, variable, arms):
    """{arm: (lengths, lengths the jackknife does not confirm)} from an artifact's own per-length
    decision (addenda 34 and 35 store one), or None if it has none. The
    stored decision also asks a candidate to beat the method addendum 33 leaves in place at a
    length where it differs from the declared incumbent."""
    decision = dec[f'{variable} {LONG[0]}']
    if 'replaced_at' not in decision:
        return None
    at, robust = decision['replaced_at'], decision.get('replaced_at_robust', {})
    return {a: (at[a], [L for L in at[a] if L not in robust.get(a, [])])
            for a in arms if at.get(a)}


def long_note(res, dec, arms, variables=('oxygen', 'temperature')):
    """Per variable, what beats the incumbent past a week at some length in both cohorts."""
    out = []
    for variable in variables:
        incumbent = dec[f'{variable} {LONG[0]}']['incumbent']
        found = stored_replacements(dec, variable, arms)
        if found is None:
            found = long_replacements(res, variable, arms, f'minus_{incumbent}')
        text = describe_replacements(found, NAMES[incumbent])
        out.append(f'on {variable}, {text}' if len(variables) > 1 else text)
    return '; '.join(out)


VARIANT_LABELS = [('chronos2_bidir', 'Chronos-2'), ('chronos2_cov_bidir', 'Chronos-2 + neighbours'),
                  ('gpr_k8', 'GPR, 8 neighbours'), ('ens_mean', 'mean: Chronos-Bolt, GPR'),
                  ('ens_mean_c2', 'mean: Chronos-2 + neighbours, GPR'), ('ens_stack', 'stacked'),
                  ('twin_else_gpr', 'twin, else GPR'), ('twin_else_best', 'twin, else best')]
LONG = ('168-1440', 'Gap-Long-Dev', 'Gap-Long-2026')
#: What a count cell in a long-gap column means, shared by the three appendix tables.
COUNT_NOTE = (r'Past a week the windows are tiled and independent only within one length, so each '
              r'model is contrasted at each of the five lengths separately, on probe-months. The '
              r'columns count the lengths, among those with at least five units, at which it is '
              r'better or worse than the reference with an interval excluding zero, written '
              r'better/worse; where fewer than five lengths have five units, their number follows '
              r'in brackets.')


def length_contrasts(res, cohort, variable, key, minus):
    """{length: contrast} for one model past a week, or None if the model was not run there."""
    arm = res.get(cohort, {}).get(variable, {}).get(LONG[0], {}).get(key)
    if not arm or 'by_gap_length' not in arm:
        return None
    return {int(L): row.get(minus) for L, row in arm['by_gap_length'].items()}


def long_replacements(res, variable, arms, minus):
    """{arm: (lengths, lengths the jackknife does not confirm)} past a week.

    A model replaces the incumbent at a length when it beats it with an interval excluding zero in
    both long cohorts at that length (the declared rule, applied per length). Below twenty units
    Chapter 4 also asks the jackknife to agree; the lengths where it does not are listed apart.
    `minus` is the key of the contrast with the incumbent in each length's entry.
    """
    out = {}
    for arm in arms:
        found, weak = [], []
        dev = length_contrasts(res, LONG[1], variable, arm, minus) or {}
        new = length_contrasts(res, LONG[2], variable, arm, minus) or {}
        for L in sorted(set(dev) & set(new)):
            pair = [dev[L], new[L]]
            if all(e and e.get('estimable') and e['excludes_zero'] and e['delta'] < 0
                   for e in pair):
                found.append(L)
                if not all(e['clusters'] >= 20 or e.get('jackknife_excludes_zero') for e in pair):
                    weak.append(L)
        if found:
            out[arm] = (found, weak)
    return out


def describe_replacements(found, incumbent):
    """Prose for `long_replacements`: which model beats the incumbent where, and how firmly."""
    if not found:
        return f'nothing beats {incumbent} at any length in both cohorts'
    parts = []
    for arm, (Ls, weak) in found.items():
        text = f'{NAMES.get(arm, arm)} beats {incumbent} at ' + series(hours(L) for L in Ls) + '\\,h'
        if weak == Ls:
            text += ', which the jackknife does not confirm'
        elif weak:
            text += ', which the jackknife does not confirm at ' + series(hours(L) for L in weak) + '\\,h'
        parts.append(text)
    return '; '.join(parts)


def count_cell(entries):
    """Lengths better/worse than the reference with an interval excluding zero."""
    if not entries or all(e is None for e in entries.values()):
        return '---'
    usable = [e for e in entries.values() if e and e.get('estimable')]
    better = sum(bool(e['excludes_zero'] and e['delta'] < 0) for e in usable)
    worse = sum(bool(e['excludes_zero'] and e['delta'] > 0) for e in usable)
    text = f'{better}/{worse}'
    if len(usable) < len(entries):
        text += f' {{\\footnotesize(of {len(usable)})}}'
    return text


def variants():
    """Addendum 33: variations and ensembles, each against its band's incumbent, both cohorts."""
    v = json.loads((RUNS / VARIANTS).read_text())
    res, dec = v['results'], v['decisions']
    rows = []
    for variable in ('oxygen', 'temperature'):
        floor = FLOOR[variable]
        incumbent = NAMES[dec[f'{variable} 12-72']['incumbent']]
        rows.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{variable.capitalize()}}} '
                    f'({UNIT[variable]}), minus the incumbent, {incumbent}}} \\\\')

        def entry_of(cohort, key):
            arm = res.get(cohort, {}).get(variable, {}).get('12-72', {}).get(key)
            return (arm or {}).get(f'minus_{dec[f"{variable} 12-72"]["incumbent"]}')
        # one number of decimals per cohort column, over the rows the table prints
        shown = [key for key, _ in VARIANT_LABELS if not (variable == 'oxygen' and key == 'gpr_k8')]
        d = {cohort: digits(variable, [entry_of(cohort, key) for key in shown])
             for cohort in ('Gap-Dev', 'Gap-2026')}
        for key, label in VARIANT_LABELS:
            values, intervals = [], []
            present = False
            for cohort in ('Gap-Dev', 'Gap-2026'):
                entry = entry_of(cohort, key)
                present |= entry is not None
                value, _, interval = cell(scaled(entry, variable), d[cohort], floor).partition(' ')
                values.append(value)
                intervals.append(interval)
            for cohort in LONG[1:]:
                values.append(count_cell(length_contrasts(res, cohort, variable, key,
                                                          'minus_incumbent')))
                intervals.append('')
            # On oxygen the eight-neighbour GPR is the four-neighbour one (five sondes in all), so
            # its row would only repeat the GPR; the note says so instead.
            if not present or (variable == 'oxygen' and key == 'gpr_k8'):
                continue
            rows.append(f'\\quad {label} & ' + ' & '.join(values) + ' \\\\')
            rows.append(' & ' + ' & '.join(intervals) + ' \\\\')
        rows.append('\\addlinespace')
    body = '\n'.join(rows[:-1])
    dagger = ((r' $^{\dagger}$ the point estimate of the gain exceeds the floor of its variable.'
               if r'^{\dagger}' in body else '')
              + (r' $^{\ddagger}$ the whole interval does.' if r'\ddagger' in body else '')
              + ((' ' + JACKKNIFE_NOTE) if r'\S' in body else ''))
    #: 1-6 h is not a column (linear is the incumbent there), so its decisions go in the note;
    #: the bold cells already show the decisions for the bands that are columns
    short = '; '.join(f'on {k.split()[0]}, '
                      + (', '.join(NAMES[a] for a in x['replaced_by']) or 'nothing')
                      for k, x in dec.items() if k.split()[1] == '1-6')
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Variants and ensembles of the best gap fillers. Each variant or ensemble minus the
incumbent of its variable: bidirectional Chronos-Bolt for oxygen, the Gaussian process over the
four nearest probes for temperature. A candidate replaces an incumbent only if it beats it with an
interval excluding zero in both the development and the 2026 cohort, within the same band up to
$72$\,h and at the same length past a week. Negative favours the candidate.}}
\label{{tab:bay_gap_variants}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{@{{}}lllll@{{}}}}
\toprule
 & \multicolumn{{2}}{{l}}{{\textbf{{12 to 72\,h}}}} & \multicolumn{{2}}{{l}}{{\textbf{{168 to 1\,440\,h, lengths}}}} \\
\textbf{{Candidate}} & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{2pt}}
\scriptsize\justifying\noindent $^{{\ast}}$ interval excludes zero; bold where it favours the
candidate.{dagger} Paired 95\,\% bootstrap intervals over overlap clusters, 600 draws.
{COUNT_NOTE} Table~\ref{{tab:bay_gap_variants_long}} gives each length. Stacked: convex weights
over linear, Chronos-Bolt, the Gaussian process and Chronos-2 with neighbours, chosen by a
100-trial study on the folds not scored and frozen before the 2026 read. A twin rule takes the
co-located twin's reading where the twin reports and its named method everywhere else. On oxygen
the Gaussian process with eight neighbours is the one with four, as there are five oxygen sondes,
and is not shown. At 1 to 6\,h, not shown, linear is the incumbent and is replaced by: {short}.\artifact{{runs/{VARIANTS}}}
\end{{table}}
"""


def variants_long():
    """Addendum 33 past a week, length by length: the decisions of Table tab:bay_gap_variants."""
    v = json.loads((RUNS / VARIANTS).read_text())
    res, dec = v['results'], v['decisions']
    lengths = sorted(int(L) for L in
                     res['Gap-Long-Dev']['temperature'][LONG[0]]['gpr']['by_gap_length'])
    width = 2 + len(lengths)
    cohorts = (('Gap-Long-Dev', '\\textsc{Dev}'), ('Gap-Long-2026', '\\textsc{2026}'))
    rows = []
    for variable in ('oxygen', 'temperature'):
        floor = FLOOR[variable]
        incumbent = dec[f'{variable} {LONG[0]}']['incumbent']
        rows.append(f'\\multicolumn{{{width}}}{{@{{}}l}}{{\\textbf{{{variable.capitalize()}}} '
                    f'({UNIT[variable]}), minus {NAMES[incumbent]}}} \\\\')
        for n, (cohort, tag) in enumerate(cohorts):
            got = length_contrasts(res, cohort, variable, 'linear', 'minus_incumbent') or {}
            units = [str(got[L]['clusters']) if got.get(L) else '---' for L in lengths]
            rows.append(f'\\quad {{\\footnotesize units}} & {{\\footnotesize {tag}}} & '
                        + ' & '.join(f'{{\\footnotesize {u}}}' for u in units) + ' \\\\')
        shown = [key for key, _ in [('linear', 'linear')] + VARIANT_LABELS
                 if not (key == incumbent or (variable == 'oxygen' and key == 'gpr_k8'))]
        # one number of decimals for the variable's block: the lengths are columns, and a row
        # compares one candidate across them
        d = digits(variable, [(length_contrasts(res, cohort, variable, key, 'minus_incumbent') or {}).get(L)
                              for key in shown for cohort, _ in cohorts for L in lengths])
        for key, label in [('linear', 'linear')] + VARIANT_LABELS:
            if key == incumbent or (variable == 'oxygen' and key == 'gpr_k8'):
                continue
            block = []
            for n, (cohort, tag) in enumerate(cohorts):
                got = length_contrasts(res, cohort, variable, key, 'minus_incumbent') or {}
                if not got:
                    continue
                cells = [cell(scaled(got.get(L), variable), d, floor).partition(' ')[0] for L in lengths]
                name = f'\\quad {label}' if not block else ''
                block.append(f'{name} & {tag} & ' + ' & '.join(cells) + ' \\\\')
            rows += block
        rows.append('\\addlinespace')
    body = '\n'.join(rows[:-1])
    head = ' & '.join(f'\\textbf{{{hours(L)}\\,h}}' for L in lengths)
    notes = ((r' $^{\dagger}$ the point estimate of the gain exceeds the floor or threshold of its '
              r'variable.' if r'^{\dagger}' in body else '')
             + (r' $^{\ddagger}$ the whole interval does.' if r'\ddagger' in body else '')
             + ((' ' + JACKKNIFE_NOTE) if r'\S' in body else ''))

    def listed(variable):
        at = dec[f'{variable} {LONG[0]}'].get('replaced_at', {})
        robust = dec[f'{variable} {LONG[0]}'].get('replaced_at_robust', {})
        if not at:
            return f'on {variable}, at no length'
        parts = []
        for arm, Ls in at.items():
            text = f'{NAMES[arm]} at ' + series(hours(L) for L in Ls) + '\\,h'
            weak = [L for L in Ls if L not in robust.get(arm, [])]
            if weak == Ls:
                text += ', which the jackknife does not confirm'
            elif weak:
                text += (', which the jackknife does not confirm at '
                         + series(hours(L) for L in weak) + '\\,h')
            parts.append(text)
        return f'on {variable}, ' + '; '.join(parts)

    replaced = '; '.join(listed(v_) for v_ in ('oxygen', 'temperature'))
    cols = 'll' + 'r' * len(lengths)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{The variants and ensembles of Table~\ref{{tab:bay_gap_variants}} past a week, length by
length: each minus the incumbent on \textsc{{Gap-Long-Dev}} (\textsc{{Dev}}) and
\textsc{{Gap-Long-2026}} (\textsc{{2026}}). Negative favours the candidate.}}
\label{{tab:bay_gap_variants_long}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{@{{}}{cols}@{{}}}}
\toprule
\textbf{{Candidate}} & & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{2pt}}
\scriptsize\justifying\noindent $^{{\ast}}$ interval excludes zero; bold where it favours the
candidate.{notes} Units: probe-months behind each contrast; fewer than five is not estimable
(grey). Paired 95\,\% bootstrap intervals, 600 draws, in the source. Replacing the incumbent at a
length in both cohorts: {replaced}.\artifact{{runs/{VARIANTS}}}
\end{{table}}
"""


def band_entry(res, cohort, variable, key, ref_of):
    """A model's contrast at 12 to 72 h with the reference `ref_of` names, or None."""
    arm = res[cohort][variable]['12-72'].get(key) or {}
    return arm.get(f'minus_{ref_of("12-72")}')


def band_digits(res, variable, pairs):
    """{cohort: decimals} of the 12-72 h columns over the (key, ref_of) rows of one table."""
    return {cohort: digits(variable, [band_entry(res, cohort, variable, key, ref_of) for key, ref_of in pairs])
            for cohort in ('Gap-Dev', 'Gap-2026')}


def band_row(res, variable, key, ref_of, label, against_incumbent=True, d=None):
    """One model's row: value and interval at 12 to 72 h, the count of lengths past a week.

    The floor mark is for a fill against the method it would replace. A row set against another
    version of the same model gets none: a large gain over a version that is itself far worse than
    linear interpolation is not a measurement worth marking. `d` gives each cohort column's decimals.
    """
    d = d or dict.fromkeys(('Gap-Dev', 'Gap-2026'), DIGITS[variable])
    floor = FLOOR[variable] if against_incumbent else None
    values, intervals = [], []
    for cohort in ('Gap-Dev', 'Gap-2026'):
        entry = scaled(band_entry(res, cohort, variable, key, ref_of), variable)
        value, _, interval = cell(entry, d[cohort], floor).partition(' ')
        values.append(value)
        intervals.append(interval)
    for cohort in LONG[1:]:
        values.append(count_cell(length_contrasts(res, cohort, variable, key,
                                                  f'minus_{ref_of(LONG[0])}')))
        intervals.append('')
    return [f'\\quad {label} & ' + ' & '.join(values) + ' \\\\',
            ' & ' + ' & '.join(intervals) + ' \\\\']


def incumbent_text(dec, variable, bands):
    """'the incumbent, X' when both bands share it, else each band's named."""
    names = [NAMES[dec[f'{variable} {b}']['incumbent']] for b, _, _ in bands]
    if len(set(names)) == 1:
        return f'the incumbent, {names[0]}'
    return 'the incumbent (' + ', '.join(
        f'{b.replace("-", " to ").replace("1440", hours(1440))}\\,h: {n}'
        for (b, _, _), n in zip(bands, names)) + ')'


BAND_HEAD = (r' & \multicolumn{2}{l}{\textbf{12 to 72\,h}} & '
             r'\multicolumn{2}{l}{\textbf{168 to 1\,440\,h, lengths}} \\')


A34_LABELS = [('anchored_gbm', 'anchored boosting'),
              ('anchored_gbm_exo', 'anchored boosting + exogenous'),
              ('anchored_gbm_rel', 'edge-relative boosting'),
              ('anchored_gbm_rel_exo', 'edge-relative + exogenous'),
              ('arx_bridge', 'Kalman bridge'),
              ('arx_bridge_exo', 'Kalman bridge + exogenous')]
#: (exogenous arm, the same model without the exogenous series, label)
A34_WEATHER = [('anchored_gbm_exo', 'anchored_gbm', 'anchored boosting'),
               ('anchored_gbm_rel_exo', 'anchored_gbm_rel', 'edge-relative boosting'),
               ('arx_bridge_exo', 'arx_bridge', 'Kalman bridge')]


def exogenous():
    """Addendum 34: anchored regressions and the Kalman bridge, and what the exogenous series add."""
    v = json.loads((RUNS / EXOGENOUS).read_text())
    res, dec = v['results'], v['decisions']
    bands = [('12-72', 'Gap-Dev', 'Gap-2026'), ('168-1440', 'Gap-Long-Dev', 'Gap-Long-2026')]

    def row(variable, key, ref_of, label, against_incumbent=True, d=None):
        # Each interval sits 4 pt closer under its value, and the rows are set at 0.9 of their
        # height: at full spacing this float ran taller than the page.
        value, interval = band_row(res, variable, key, ref_of, label, against_incumbent, d)
        return [value.removesuffix('\\\\') + '\\\\[-4pt]', interval]

    rows = []
    for variable in ('oxygen', 'temperature'):
        incumbent_of = lambda b, v=variable: dec[f'{v} {b}']['incumbent']
        d = band_digits(res, variable, [(key, incumbent_of) for key, _ in A34_LABELS]
                        + [(key, lambda b, p=plain: p) for key, plain, _ in A34_WEATHER])
        rows.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{variable.capitalize()}}} '
                    f'({UNIT[variable]}), minus {incumbent_text(dec, variable, bands)}}} \\\\')
        for key, label in A34_LABELS:
            rows += row(variable, key, incumbent_of, label, d=d)
        rows.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{with the exogenous series, minus '
                    f'the same model without them}}}} \\\\')
        for key, plain, label in A34_WEATHER:
            rows += row(variable, key, lambda b, p=plain: p, label, against_incumbent=False, d=d)
        rows.append('\\addlinespace')
    body = '\n'.join(rows[:-1])
    dagger = ((r' $^{\dagger}$ the point estimate of the gain exceeds the floor of its variable.'
               if r'^{\dagger}' in body else '')
              + (r' $^{\ddagger}$ the whole interval does.' if r'\ddagger' in body else '')
              + ((' ' + JACKKNIFE_NOTE) if r'\S' in body else ''))
    short_bands = {k: x for k, x in dec.items() if k.split()[1] == '1-6'}
    short = ('nothing is replaced' if not any(x['replaced_by'] for x in short_bands.values())
             else 'replaced by ' + '; '.join(
                 f'on {k.split()[0]}, '
                 + (', '.join(NAMES.get(a, a.replace('_', ' ')) for a in x['replaced_by'])
                    or 'nothing') for k, x in short_bands.items()))
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Regressions anchored to the edges of the gap, with and without the exogenous series. Upper
rows: each model minus the incumbent. Lower rows: each model with the series minus the same model
without them. Negative favours the first term.}}
\label{{tab:bay_gap_exogenous}}
\renewcommand{{\arraystretch}}{{0.86}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{@{{}}lllll@{{}}}}
\toprule
{BAND_HEAD}
\textbf{{Model}} & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{2pt}}
\scriptsize\justifying\noindent $^{{\ast}}$ interval excludes zero; bold where it favours the
first term.{dagger} n.e.: fewer than five independent units. Paired 95\,\% cluster-bootstrap
intervals, 600 draws. {COUNT_NOTE} Past a week, length by length:
{long_note(res, dec, [k for k, _ in A34_LABELS])}. Each model tuned by its own nested 100-trial
study; 2026 applies the development fit frozen. Edge-relative boosting, with no absolute level among
its features, was added after a look at one development fold. At 1 to 6\,h: {short}.\artifact{{runs/{EXOGENOUS}}}
\end{{table}}
"""


A35_LABELS = [('chronos2_exo_bidir', 'Chronos-2 + exogenous', 'chronos2_bidir'),
              ('chronos2_all_bidir', 'Chronos-2 + neighb. + exog.', 'chronos2_cov_bidir'),
              ('bolt_resid_bidir', 'Bolt, tree residual', 'chronos_bidir'),
              ('chronos2_cov_resid_bidir', 'Chronos-2 + neighb., tree res.',
               'chronos2_cov_bidir'),
              ('bolt_resid_lin_bidir', 'Bolt, ridge residual', 'chronos_bidir'),
              ('chronos2_cov_resid_lin_bidir', 'Chronos-2 + neighb., ridge res.',
               'chronos2_cov_bidir')]
#: The oxygen note also names its replacements past a week, and at the footnote size that float
#: ran taller than the page; the temperature table keeps the footnote size.
NOTE_SIZE = {'oxygen': r'\scriptsize', 'temperature': r'\footnotesize'}


def chronos_covariates(variable):
    """Addendum 35: exogenous covariates read by Chronos-2, and Chronos on a covariate residual.

    One table per variable: six models against two references each do not fit one page.
    """
    v = json.loads((RUNS / COVARIATES).read_text())
    res, dec = v['results'], v['decisions']
    bands = [('12-72', 'Gap-Dev', 'Gap-2026'), ('168-1440', 'Gap-Long-Dev', 'Gap-Long-2026')]

    incumbent_of = lambda b: dec[f'{variable} {b}']['incumbent']
    d = band_digits(res, variable, [(key, incumbent_of) for key, _, _ in A35_LABELS]
                    + [(key, lambda b, p=plain: p) for key, _, plain in A35_LABELS])

    def row(variable, key, ref_of, label, against_incumbent=True):
        return band_row(res, variable, key, ref_of, label, against_incumbent, d)

    rows = [f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{minus {incumbent_text(dec, variable, bands)}}}}} '
            '\\\\']
    for key, label, _ in A35_LABELS:
        rows += row(variable, key, incumbent_of, label)
    rows.append('\\addlinespace')
    rows.append('\\multicolumn{5}{@{}l}{\\textit{minus the same Chronos model without the '
                'addition}} \\\\')
    for key, label, plain in A35_LABELS:
        rows += row(variable, key, lambda b, p=plain: p, label, against_incumbent=False)
    body = '\n'.join(rows)
    dagger = ((r' $^{\dagger}$ the point estimate of the gain exceeds the floor of its variable.'
               if r'^{\dagger}' in body else '')
              + (r' $^{\ddagger}$ the whole interval does.' if r'\ddagger' in body else '')
              + ((' ' + JACKKNIFE_NOTE) if r'\S' in body else ''))
    replaced = dec[f'{variable} 1-6']['replaced_by']
    short = ('nothing is replaced' if not replaced
             else 'replaced by ' + ', '.join(NAMES.get(a, a) for a in replaced))
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Chronos given the exogenous series, {variable} ({UNIT[variable]}). Upper rows: each model
minus the incumbent. Lower rows: each model minus the same Chronos model without the exogenous
covariates or the residual scheme. Negative favours the first term.}}
\label{{tab:bay_gap_chronos_covariates_{variable}}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{@{{}}lllll@{{}}}}
\toprule
{BAND_HEAD}
\textbf{{Model}} & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}

\vspace{{2pt}}
{NOTE_SIZE[variable]}\justifying\noindent $^{{\ast}}$ interval excludes zero; bold where it favours the
first term.{dagger} n.e.: fewer than five independent units. Paired 95\,\% cluster-bootstrap
intervals, 600 draws. {COUNT_NOTE} Past a week, length by length:
{long_note(res, dec, [k for k, _, _ in A35_LABELS], (variable,))}. Chronos is zero-shot
throughout. Tree and ridge residual:
Chronos fills the reading minus a gradient-boosting or ridge covariate regressor, tuned by its own
nested 100-trial study, and the regressor is added back; 2026 applies the development fit frozen.
Neighb.: the four nearest probes as covariates. At 1 to 6\,h: {short}.\artifact{{runs/{COVARIATES}}}
\end{{table}}
"""


#: (source artifact, arm, label, the run that declared it); the chapter's one-table summary of the
#: three runs that tried to beat the incumbents, each method's best representative only
IMPROVEMENT_ROWS = [
    ('v2_gap_variants.json', 'linear', 'linear interpolation', ''),
    ('v2_gap_variants.json', 'chronos_bidir', 'Chronos-Bolt, bidirectional', ''),
    ('v2_gap_variants.json', 'gpr', 'Gaussian process, 4 nearest probes', ''),
    ('v2_gap_variants.json', 'chronos2_cov_bidir', 'Chronos-2 + neighbours', 'A'),
    ('v2_gap_variants.json', 'ens_mean_c2', 'mean: Gaussian process, Chronos-2 + neighb.', 'A'),
    ('v2_gap_variants.json', 'ens_stack', 'stacked ensemble', 'A'),
    ('v2_gap_exogenous.json', 'anchored_gbm_rel_exo', 'anchored boosting + forcing', 'B'),
    ('v2_gap_exogenous.json', 'arx_bridge_exo', 'Kalman bridge + forcing', 'B'),
    ('v2_gap_chronos_covariates.json', 'chronos2_all_bidir', 'Chronos-2 + neighbours + forcing',
     'C'),
    ('v2_gap_chronos_covariates.json', 'chronos2_cov_resid_lin_bidir',
     'Chronos-2 + neighbours on a forcing residual', 'C'),
]


def improvements():
    """One table for the chapter: every attempt's error beside the incumbents, both years. It prints
    the bands from 12 h on, which the Gap-2026 episodes of 1 to 3 h leave unchanged, so it reads the
    artifacts of addenda 33-35 themselves."""
    sources = {name: json.loads((RUNS / name).read_text())
               for name in {s for s, *_ in IMPROVEMENT_ROWS}}
    adopted = sources['v2_gap_variants.json']['decisions']
    cells = [('oxygen', '12-72', 'Gap-Dev', 'Gap-2026'),
             ('oxygen', '168-1440', 'Gap-Long-Dev', 'Gap-Long-2026'),
             ('temperature', '12-72', 'Gap-Dev', 'Gap-2026'),
             ('temperature', '168-1440', 'Gap-Long-Dev', 'Gap-Long-2026')]
    values = {}
    for source, arm, _, _ in IMPROVEMENT_ROWS:
        res = sources[source]['results']
        for variable, band, dev, new in cells:
            for cohort in (dev, new):
                values[(arm, variable, band, cohort)] = (res[cohort][variable][band].get(arm)
                                                         or {}).get('mae')
    best = {}
    for variable, band, dev, new in cells:
        for cohort in (dev, new):
            best[(variable, band, cohort)] = min(
                v for (a, vv, b, c), v in values.items()
                if (vv, b, c) == (variable, band, cohort) and v is not None)
    # one number of decimals per column
    column = {(variable, band, cohort): digits(variable, [values[(arm, variable, band, cohort)]
                                                          for _, arm, _, _ in IMPROVEMENT_ROWS])
              for variable, band, dev, new in cells for cohort in (dev, new)}
    rows = []
    for source, arm, label, run in IMPROVEMENT_ROWS:
        out = []
        for variable, band, dev, new in cells:
            for cohort in (dev, new):
                d = column[(variable, band, cohort)]
                v = values[(arm, variable, band, cohort)]
                text = '---' if v is None else f'{SCALE[variable] * v:.{d}f}'
                if v is not None and v == best[(variable, band, cohort)]:
                    text = f'\\textbf{{{text}}}'
                if arm in adopted[f'{variable} {band}']['replaced_by']:
                    text += '$^{\\dagger}$'
                out.append(text)
        tag = f' ({run})' if run else ''           # at the table's own size: a smaller tag prints below 7 pt
        mark = NO_NEIGHBOUR_MARK if arm in NO_NEIGHBOUR else ''
        rows.append(f'{label}{mark}{tag} & ' + ' & '.join(out) + ' \\\\')
        if arm == 'gpr':
            rows.append('\\addlinespace')
    body = '\n'.join(rows)

    def per_length(variable):
        """Past a week the decision is per length: the variants' own, and any from runs B and C."""
        decision = adopted[f'{variable} 168-1440']
        robust = decision.get('replaced_at_robust', {})
        found = {arm: (Ls, [L for L in Ls if L not in robust.get(arm, [])])
                 for arm, Ls in decision.get('replaced_at', {}).items()}
        text = describe_replacements(found, NAMES[decision['incumbent']])
        for source in ('v2_gap_exogenous.json', 'v2_gap_chronos_covariates.json'):
            other = sources[source]
            incumbent = other['decisions'][f'{variable} 168-1440']['incumbent']
            arms = [arm for s, arm, *_ in IMPROVEMENT_ROWS if s == source]
            more = stored_replacements(other['decisions'], variable, arms)
            if more is None:
                more = long_replacements(other['results'], variable, arms, f'minus_{incumbent}')
            if more:
                text += '; ' + describe_replacements(more, NAMES[incumbent])
        return f'on {variable}, {text}'

    past_week = '; '.join(per_length(v) for v in ('oxygen', 'temperature'))
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Variants, anchored regressions and Chronos given the forcing: each method's error by band.}}
\label{{tab:bay_gap_improvements}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}Xllllllll@{{}}}}
\toprule
 & \multicolumn{{4}}{{l}}{{\textbf{{Oxygen}} ({UNIT['oxygen']})}} & \multicolumn{{4}}{{l}}{{\textbf{{Temperature}} ($^\circ$C)}} \\
 & \multicolumn{{2}}{{l}}{{12 to 72\,h}} & \multicolumn{{2}}{{l}}{{168 to 1\,440\,h}} & \multicolumn{{2}}{{l}}{{12 to 72\,h}} & \multicolumn{{2}}{{l}}{{168 to 1\,440\,h}} \\
\textbf{{Method}} & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} & \textsc{{Dev}} & \textsc{{2026}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}

\vspace{{2pt}}
\scriptsize\justifying\noindent Macro mean absolute error on the same rows within each column,
over the $12$ to $72$\,h band and over every episode of the five lengths from a week, as a
description only; the lowest in each column is bold. $^{{\dagger}}$ beats the Gaussian process
(temperature) or bidirectional Chronos-Bolt (oxygen) at $12$ to $72$\,h with an interval excluding
zero in both cohorts. Past a week the comparison is length by length: {past_week}. (A) variations and ensembles, (B) regressions anchored to the
edges, (C) Chronos given the forcing, (B) and (C) each also given the forcing.
{NO_NEIGHBOUR_NOTE}\artifact{{runs/v2_gap_variants.json}}%
\artifact{{runs/v2_gap_exogenous.json}}\artifact{{runs/v2_gap_chronos_covariates.json}}
\end{{table}}
"""


def write(path, text):
    """UTF-8 with LF line ends on every host (Path.write_text writes CRLF on Windows)."""
    Path(path).write_bytes(text.replace('\r\n', '\n').encode('utf-8'))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    write(TABLES / 'table_bay_gap_improvements.tex', improvements())
    if (RUNS / COVARIATES).exists():
        for variable in ('oxygen', 'temperature'):
            write(TABLES / f'table_bay_gap_chronos_covariates_{variable}.tex',
                  chronos_covariates(variable))
    write(TABLES / 'table_bay_gap_pooled.tex', pooled())
    write(TABLES / 'table_bay_gap_lengths.tex', by_length())
    write(TABLES / 'table_bay_gap_long.tex', long_gaps())
    write(TABLES / 'table_bay_gap_variants.tex', variants())
    write(TABLES / 'table_bay_gap_variants_long.tex', variants_long())
    if (RUNS / EXOGENOUS).exists():
        write(TABLES / 'table_bay_gap_exogenous.tex', exogenous())
    print('wrote table_bay_gap_{improvements,chronos_covariates_*,pooled,lengths,long,variants,'
          'variants_long,exogenous}.tex')


if __name__ == '__main__':
    main()
