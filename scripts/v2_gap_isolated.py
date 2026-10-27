"""A probe with no neighbour: what the gap methods that read no other probe achieve.

Written 2026-09-28. No model is fitted and nothing new is scored. Every prediction was
published by addenda 30 to 35 and 41 on identical rows; this script groups the methods by what they
read and pairs the best of each group on those rows, with the resampling unit of its cohort:

  own record    linear interpolation, Chronos-Bolt and Chronos-2 from both edges, the Kalman bridge
  + forcing     the same families given the Santa Teresa weather station and the Magra level: the
                Kalman bridge, Chronos-2 with the forcing as covariates, and Chronos-Bolt on the
                residual of a forcing regression (tree or ridge)
  network       everything that reads another probe: the Gaussian process, Chronos-2 with the four
                nearest probes as covariates, the ensembles built on either, the twin rules, and the
                anchored and gradient-boosted regressions, whose features hold the four nearest
                probes' readings

The first two groups are what a probe with no neighbour has. The polynomial ridge of addendum 30
also reads only the probe's own record (its fourteen features are the first, all the probe's
own), but it was scored on Gap-Dev only and is worse than linear there, so it cannot
enter a comparison that needs both cohorts. The rules were written after the
per-method results had been published, so this is a re-reading, not a confirmatory test:

  isolated   per band up to 72 h and per length past a week, linear interpolation, unless a method
             of the first two groups beats it with an interval excluding zero in both cohorts,
             robustly (twenty units, or the delete-one-unit jackknife agreeing), in which case the
             one with the lower development error
  network    per band, the network method with the lowest development error
  margin     network minus isolated, paired, in each cohort; a contrast already published is read
             from its artifact, any other is computed here on the same rows and units

While the probe is silent the same split is read from the bridging artifacts: of the continuations
that read no other probe (the last value, the two climatological increments, zero-shot Chronos-Bolt)
and of those that do (the neighbour rule, the anchored continuation, the recursive rollout), the
one with the lowest development error averaged over the eight leads, and their published contrast at
each lead.

Writes runs/v2_gap_isolated.json and table_bay_gap_isolated.tex.
Usage: python -m scripts.v2_gap_isolated
       python -m scripts.v2_gap_isolated --table-from reports/bay_twin/runs/v2_addendum82/C/rescored/v2_gap_isolated.json
       (the thesis table, from the re-reading with the Gap-2026 episodes of 1 to 3 h; nothing recomputed)
"""
import json
import sys
from pathlib import Path

import pandas as pd

from scripts.v2_gap_intervals import clusters, paired
from scripts.v2_gap_tables import FLOOR, JACKKNIFE_NOTE, SCALE, UNIT, cell, digits, hours, scaled

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
TABLES = ROOT / 'thesis_template/content/tables'
OUT = RUNS / 'v2_gap_isolated.json'
TABLE = TABLES / 'table_bay_gap_isolated.tex'

#: artifact -> its per-row predictions; the variants rows are the base the others must match
SOURCES = {'v2_gap_variants.json': 'v2_gap_variants',
           'v2_gap_exogenous.json': 'v2_gap_exogenous',
           'v2_gap_chronos_covariates.json': 'v2_gap_chronos_covariates'}
OWN = ('linear', 'chronos_bidir', 'chronos2_bidir', 'arx_bridge')
FORCED = ('arx_bridge_exo', 'chronos2_exo_bidir', 'bolt_resid_bidir', 'bolt_resid_lin_bidir')
NETWORK = ('gpr', 'gpr_k8', 'chronos2_cov_bidir', 'ens_mean', 'ens_mean_c2', 'ens_stack',
           'twin_rule', 'twin_else_gpr', 'twin_else_best', 'anchored_gbm', 'anchored_gbm_exo',
           'anchored_gbm_rel', 'anchored_gbm_rel_exo', 'regression_retro', 'chronos2_all_bidir',
           'chronos2_cov_resid_bidir', 'chronos2_cov_resid_lin_bidir')
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'),
         '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
PER_LENGTH = ('168-1440',)
#: addendum 33 stores the contrast with its band's incumbent per length as `minus_incumbent`
VARIANT_INCUMBENT = {('oxygen', '1-6'): 'linear', ('oxygen', '12-72'): 'chronos_bidir',
                     ('oxygen', '168-1440'): 'chronos_bidir', ('temperature', '1-6'): 'linear',
                     ('temperature', '12-72'): 'gpr', ('temperature', '168-1440'): 'gpr'}
ROBUST_UNITS = 20

SILENT_OWN = ('persistence', 'clim_month_hour', 'clim_hour', 'chronos_zs')
SILENT_NETWORK = ('neighbour_change', 'anchored_direct', 'recursive_tree')
SILENT_COHORTS = ('Bridge-Dev', 'Bridge-2026')

LABEL = {'linear': 'linear', 'chronos_bidir': 'Chronos-Bolt', 'chronos2_bidir': 'Chronos-2',
         'arx_bridge': 'Kalman bridge', 'arx_bridge_exo': 'Kalman bridge + forcing',
         'chronos2_exo_bidir': 'Chronos-2 + forcing',
         'bolt_resid_bidir': 'Chronos-Bolt on a forcing residual',
         'bolt_resid_lin_bidir': 'Chronos-Bolt on a forcing residual',
         'gpr': 'Gaussian process', 'gpr_k8': 'Gaussian process, 8 probes',
         'chronos2_cov_bidir': 'Chronos-2 + neighbours', 'ens_mean': 'mean: Chronos-Bolt, GP',
         'ens_mean_c2': 'mean: GP, Chronos-2 + neighb.', 'ens_stack': 'stacked ensemble',
         'twin_rule': 'twin rule', 'twin_else_gpr': 'twin, else GP', 'twin_else_best': 'twin, else best',
         'anchored_gbm': 'anchored boosting', 'anchored_gbm_exo': 'anchored boosting + forcing',
         'anchored_gbm_rel': 'edge-relative boosting',
         'anchored_gbm_rel_exo': 'edge-relative boosting + forcing',
         'regression_retro': 'boosting, both edges',
         'chronos2_all_bidir': 'Chronos-2 + neighb. + forcing',
         'chronos2_cov_resid_bidir': 'Chronos-2 + neighb. on a residual',
         'chronos2_cov_resid_lin_bidir': 'Chronos-2 + neighb. on a residual',
         'persistence': 'last value', 'clim_month_hour': 'month-by-hour increment',
         'clim_hour': 'hour-of-day increment', 'chronos_zs': 'Chronos-Bolt',
         'neighbour_change': 'neighbour rule', 'anchored_direct': 'network-aided forecast',
         'recursive_tree': 'recursive rollout'}


def load_results():
    return {name: json.loads((RUNS / name).read_text())['results'] for name in SOURCES}


def rows(cohort, variable, band):
    """The published per-row predictions of every arm, merged on identical rows."""
    base = None
    for folder in SOURCES.values():
        frame = pd.read_csv(RUNS / folder / f'{cohort}__{variable}__{band}.csv.gz')
        if base is None:
            base = frame
            continue
        keys = ['episode', 'position']
        if not (frame[keys].equals(base[keys]) and
                (frame.truth - base.truth).abs().max() == 0):
            raise SystemExit(f'{folder} {cohort} {variable} {band}: rows differ from the variants rows')
        for column in frame.columns:
            if column not in base:
                base[column] = frame[column].to_numpy()
    return base


def arm_entry(results, cohort, variable, band, arm):
    for res in results.values():
        entry = res.get(cohort, {}).get(variable, {}).get(band, {}).get(arm)
        if entry and entry.get('mae') is not None:
            return entry
    return None


def mae(results, cohort, variable, band, arm, length=None):
    entry = arm_entry(results, cohort, variable, band, arm)
    if entry is None:
        return None
    if length is None:
        return entry['mae']
    return entry.get('by_gap_length', {}).get(str(length), {}).get('mae')


def published(results, cohort, variable, band, arm, ref, length=None):
    """A contrast an earlier artifact already computed, with its source, or (None, None)."""
    for name, res in results.items():
        entry = res.get(cohort, {}).get(variable, {}).get(band, {}).get(arm) or {}
        if length is None:
            found = entry.get(f'minus_{ref}')
        else:
            row = entry.get('by_gap_length', {}).get(str(length), {})
            found = row.get(f'minus_{ref}')
            if found is None and name == 'v2_gap_variants.json' \
                    and ref == VARIANT_INCUMBENT[(variable, band)]:
                found = row.get('minus_incumbent')
        if found:
            return found, f'runs/{name}'
    return None, None


def robust(entry):
    return entry['clusters'] >= ROBUST_UNITS or bool(entry.get('jackknife_excludes_zero'))


def beats(entry):
    return bool(entry and entry.get('estimable', True) and entry.get('excludes_zero')
                and entry['delta'] < 0)


def units_for(part, long_cohort, cluster_of):
    if not long_cohort:
        return cluster_of
    start = part.episode.str.rsplit('|', n=2).str[1].str[:7]
    return dict(zip(part.episode, part.sensor + '|' + start))


def after_the_fact(results):
    out = {}
    for variable in ('oxygen', 'temperature'):
        out[variable] = {}
        for band, (dev, new) in PAIRS.items():
            present = [a for a in OWN + FORCED + NETWORK
                       if all(arm_entry(results, c, variable, band, a) for c in (dev, new))]
            network = min((a for a in present if a in NETWORK),
                          key=lambda a: mae(results, dev, variable, band, a))
            lengths = ([None] if band not in PER_LENGTH else
                       sorted(int(L) for L in arm_entry(results, dev, variable, band,
                                                        'linear')['by_gap_length']))
            frames = {c: rows(c, variable, band) for c in (dev, new)}
            cluster_of = {c: clusters(f) for c, f in frames.items()}
            cells = []
            for length in lengths:
                # The isolated choice: linear unless a no-neighbour method robustly beats it in both.
                qualified = []
                for arm in (a for a in present if a in OWN + FORCED and a != 'linear'):
                    got = [published(results, c, variable, band, arm, 'linear', length)[0]
                           for c in (dev, new)]
                    if all(beats(e) and robust(e) for e in got):
                        qualified.append(arm)
                isolated = min(qualified, key=lambda a: mae(results, dev, variable, band, a, length)) \
                    if qualified else 'linear'
                cell_out = {'length': length, 'isolated': isolated, 'network': network,
                            'isolated_candidates_that_qualified': qualified}
                for cohort in (dev, new):
                    f = frames[cohort]
                    part = f if length is None else f[f.gap_length == length]
                    entry = {'linear': mae(results, cohort, variable, band, 'linear', length),
                             'isolated': mae(results, cohort, variable, band, isolated, length),
                             'network': mae(results, cohort, variable, band, network, length)}
                    if isolated != 'linear':
                        e, src = published(results, cohort, variable, band, isolated, 'linear',
                                           length)
                        entry['isolated_minus_linear'] = {**e, 'source': src}
                    e, src = published(results, cohort, variable, band, network, isolated, length)
                    if e is None:
                        label = f'isolated|{cohort}|{variable}|{band}|{network}|{isolated}'
                        if length is not None:
                            label += f'|{length}'
                        e = paired(part, network, isolated, label,
                                   units_for(part, band in PER_LENGTH, cluster_of[cohort]))
                        src = 'computed here'
                    entry['network_minus_isolated'] = {**e, 'source': src}
                    cell_out[cohort] = entry
                cells.append(cell_out)
            out[variable][band] = {'cohorts': [dev, new], 'network': network,
                                   'no_neighbour_methods_compared':
                                       [a for a in present if a in OWN + FORCED],
                                   'network_methods_compared': [a for a in present if a in NETWORK],
                                   'cells': cells}
    return out


def silent():
    bridge = json.loads((RUNS / 'v2_bridge/report.json').read_text())
    zs = json.loads((RUNS / 'v2_chronos_adapt/contrasts.json').read_text())['continuation']
    out = {}
    for variable in ('oxygen', 'temperature'):
        def maes(cohort, lead):
            m = dict(bridge[variable][cohort]['by_lead'][str(lead)]['mae'])
            m['chronos_zs'] = zs[variable][cohort]['by_lead'][str(lead)]['mae']['chronos_zs']
            return m

        leads = sorted(int(L) for L in bridge[variable]['Bridge-Dev']['by_lead'])

        def best(pool):
            usable = [a for a in pool if all(a in maes(c, L) for c in SILENT_COHORTS for L in leads)]
            return min(usable, key=lambda a: sum(maes('Bridge-Dev', L)[a] for L in leads) / len(leads))

        own, network = best(SILENT_OWN), best(SILENT_NETWORK)

        def contrast(cohort, lead):
            for source in (bridge[variable][cohort]['by_lead'][str(lead)]['contrasts'],
                           zs[variable][cohort]['by_lead'][str(lead)]['contrasts']):
                if f'{network} - {own}' in source:
                    e, sign = source[f'{network} - {own}'], 1
                elif f'{own} - {network}' in source:
                    e, sign = source[f'{own} - {network}'], -1
                else:
                    continue
                lo, hi = sorted(sign * x for x in e['ci'])
                return {'delta': round(sign * e['point'], 4), 'ci_low': round(lo, 4),
                        'ci_high': round(hi, 4), 'excludes_zero': bool(e['excludes_zero'])}
            raise SystemExit(f'no published contrast of {network} and {own} for {variable}')

        out[variable] = {'isolated': own, 'network': network,
                         'rule': 'lowest development error averaged over the eight leads',
                         'by_lead': {str(L): {c: {'isolated': round(maes(c, L)[own], 4),
                                                  'network': round(maes(c, L)[network], 4),
                                                  'network_minus_isolated': contrast(c, L)}
                                              for c in SILENT_COHORTS}
                                     for L in leads},
                         'sources': ['runs/v2_bridge/report.json',
                                     'runs/v2_chronos_adapt/contrasts.json']}
    return out


def fmt(value, digits):
    return f'{value:.{digits}f}'


def split_cell(entry, d):
    """`cell` as two lines, the estimate and below it the interval or n.e., both at the table's size."""
    text = cell(entry, d)
    value, _, rest = text.partition(' {\\footnotesize')
    return value, ('{' + rest.lstrip()) if rest else ''


def render(report, cite='runs/v2_gap_isolated.json'):
    """The gap-filling half. The silent half is already per lead in the Chronos-roles tables, which
    set zero-shot Chronos-Bolt against the anchored continuation, so the prose cites those. `cite` is
    the report the table is written from, as the thesis cites it."""
    body = []
    for variable in ('oxygen', 'temperature'):
        k = SCALE[variable]
        body.append(f'\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{variable.capitalize()}}} '
                    f'({UNIT[variable]})}} \\\\')
        # one number of decimals per column (the errors and the differences in each cohort)
        cells = [(b['cohorts'], c) for b in report['after_the_fact'][variable].values() for c in b['cells']]
        d_err = {i: digits(variable, [c[co[i]]['isolated'] for co, c in cells]) for i in (0, 1)}
        d_net = {i: digits(variable, [c[co[i]]['network_minus_isolated'] for co, c in cells]) for i in (0, 1)}
        previous = (None, None)
        for band in PAIRS:
            b = report['after_the_fact'][variable][band]
            dev, new = b['cohorts']
            for c in b['cells']:
                gap = (f'{band.replace("-", " to ")}\\,h' if c['length'] is None
                       else f'{hours(c["length"])}\\,h')
                # A blank method cell repeats the one above.
                iso = LABEL[c['isolated']] if c['isolated'] != previous[0] else ''
                net = LABEL[b['network']] if b['network'] != previous[1] else ''
                previous = (c['isolated'], b['network'])
                lines = [split_cell(scaled(c[co]['network_minus_isolated'], variable), d_net[i])
                         for i, co in enumerate((dev, new))]
                body.append(f'{gap} & {iso} & {fmt(k * c[dev]["isolated"], d_err[0])} & '
                            f'{fmt(k * c[new]["isolated"], d_err[1])} & {net} & {lines[0][0]} & '
                            f'{lines[1][0]} \\\\')
                body.append(f' & & & & & {lines[0][1]} & {lines[1][1]} \\\\[1pt]')
        body.append('\\addlinespace')
    text = '\n'.join(body[:-1])
    jackknife = (' ' + JACKKNIFE_NOTE) if r'\S' in text else ''
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Gaps on a probe with no neighbour, and what the network adds.}}
\label{{tab:bay_gap_isolated}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l>{{\raggedright\arraybackslash}}Xll>{{\raggedright\arraybackslash}}Xll@{{}}}}
\toprule
 & \multicolumn{{3}}{{l}}{{\textbf{{No neighbour}}}} & \multicolumn{{3}}{{l}}{{\textbf{{Network minus no neighbour}}}} \\
\textbf{{Gap}} & Method & \textsc{{Dev}} & \textsc{{2026}} & Method & \textsc{{Dev}} & \textsc{{2026}} \\
\midrule
{text}
\bottomrule
\end{{tabularx}}

\vspace{{2pt}}
\scriptsize\justifying\noindent For each gap length, the best method reading no other probe (the
probe's own record and at most the weather station and the Magra level), its error in each cohort,
and the best method reading the network minus it. Each group's method has the lowest
development error, except that a no-neighbour method replaces linear interpolation only where it
beats it robustly in both cohorts; a blank method cell repeats the one above. The eight pretrained
arms of Section~\ref{{subsec:meth_gap}} are not among the candidates; Figure~\ref{{fig:bay_gap_common}}
draws them. Cohorts: \textsc{{Gap-Dev}} and \textsc{{Gap-2026}} to $72$\,h, the two
\textsc{{Gap-Long}} cohorts from a week, resampled by overlap clusters and by probe-months.
$^{{\ast}}$ interval excludes zero; bold where it favours the network.{jackknife} n.e.: fewer
than five units.\artifact{{{cite}}}
\end{{table}}
"""


#: addendum 82, part C: the same re-reading with the Gap-2026 episodes of 1 to 3 h added, written by
#: scripts/v2_gap_short2026_score.py beside the published report; the thesis table is drawn from it
RESCORED = RUNS / 'v2_addendum82/C/rescored/v2_gap_isolated.json'


def table_from(path):
    """Write the thesis table from a report this module wrote, without recomputing it."""
    path = Path(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    cite = 'runs/' + path.resolve().relative_to(RUNS.resolve()).as_posix()
    TABLE.write_bytes(render(report, cite).encode('utf-8'))
    print('wrote', TABLE.relative_to(ROOT), 'from', cite)


def main():
    if '--table-from' in sys.argv[1:]:
        table_from(sys.argv[sys.argv.index('--table-from') + 1])
        return
    results = load_results()
    report = {'written': '2026-09-28',
              'what': 'a re-reading of published gap-recovery predictions by what each method reads; '
                      'no model fitted, nothing new scored',
              'groups': {'own_record': list(OWN), 'own_record_and_forcing': list(FORCED),
                         'network': list(NETWORK)},
              'rules': {'isolated': 'linear unless a no-neighbour method beats it with an interval '
                                    'excluding zero in both cohorts, robustly (>= 20 units or the '
                                    'jackknife agreeing); then the lowest development error',
                        'network': 'the network method with the lowest development error in the band',
                        'while_silent': 'in each group the lowest development error averaged over '
                                        'the eight leads'},
              'after_the_fact': after_the_fact(results),
              'while_silent': silent()}
    # LF on every host (Path.write_text writes CRLF on Windows)
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    # the thesis table is drawn from the re-reading with the Gap-2026 episodes of 1 to 3 h where it
    # exists (addendum 82, part C), else from this report
    if RESCORED.exists():
        table_from(RESCORED)
    else:
        TABLE.write_bytes(render(report).encode('utf-8'))
    for variable in ('oxygen', 'temperature'):
        s = report['while_silent'][variable]
        print(f'\n== {variable}: while silent {s["isolated"]} vs {s["network"]}')
        for L, r in s['by_lead'].items():
            print(f'   {L:>3} h', {c: (r[c]['isolated'], r[c]['network'],
                                     r[c]['network_minus_isolated']['delta'],
                                     r[c]['network_minus_isolated']['excludes_zero'])
                                 for c in SILENT_COHORTS})
        for band, b in report['after_the_fact'][variable].items():
            print(f'   {band}: network {b["network"]}')
            for c in b['cells']:
                print(f'     {c["length"] or band:>6} {c["isolated"]:<16}',
                      {co: (c[co]['linear'], c[co]['isolated'], c[co]['network'],
                            c[co]['network_minus_isolated']['delta'],
                            c[co]['network_minus_isolated'].get('excludes_zero'),
                            c[co]['network_minus_isolated']['source'])
                       for co in b['cohorts']})
    print('wrote', OUT.relative_to(ROOT), 'and', TABLE.relative_to(ROOT))


if __name__ == '__main__':
    main()
