"""Addendum 82, part C (C6): PCHIP against linear interpolation, with a paired cluster interval.

Addendum 30 kept only PCHIP's aggregate error (scripts/v2_gap_splines.py), so table_bay_gap_pooled
printed it without an interval. `v2_gap_recovery.build` is deterministic given its seed and computes
the shape-preserving spline on the same 72 h of knots either side, so the Gap-Dev episodes are rebuilt
here, checked row for row against the published addendum-30 run, and their per-row PCHIP is checked
against the column every later strict run carries on the identical rows (20260922T204658Z) and against
the published aggregates. Then PCHIP minus linear is bootstrapped exactly as every other gap contrast:
paired on the rows both carry, overlap clusters of the whole cohort (183 temperature, 73 oxygen), 600
draws from the stream of the label, the delete-one-cluster jackknife beside it.

  Gap-Dev         pooled ('{variable}|pchip|all') and per length ('{variable}|pchip|{L}'): printed
  Gap-2026        the published 6-72 h column (run 20260922T215413Z) with the 1-3 h rows of this
                  part's run, whole-cohort clusters: descriptive
  Gap-Long-*      the published columns (20260922T195727Z, 20260922T195745Z), per length on probe x
                  calendar month, the unit of the tiled cohorts: descriptive

Gate C-G6: the rebuilt PCHIP equals 20260922T204658Z's column on every Gap-Dev row to 1e-12, its
macro by length equals runs/v2_gap_splines.json, and spline minus linear from the same call reproduces
the published 'spline' entry of runs/v2_gap_intervals.json exactly.

Writes runs/v2_addendum82/C/v2_gap_pchip.json and the per-row sidecar
runs/v2_addendum82/C/pchip/Gap-Dev__{variable}.csv.gz.
Usage: python -m scripts.v2_gap_pchip
"""
import json

import numpy as np
import pandas as pd

from scripts.v2_gap_intervals import clusters, paired
from scripts.v2_gap_short2026_rows import PART, RUNS, append, paired6, r6
from src.experiments import v2_gap_recovery as gap

OUT = PART / 'v2_gap_pchip.json'
SIDECARS = PART / 'pchip'
GZIP = {'method': 'gzip', 'mtime': 0}
PUBLISHED_DEV = RUNS / '20260922T171859Z_v2_gaprecovery' / 'v2_gap_recovery'
LATER_DEV = RUNS / '20260922T204658Z_v2_gaprecovery' / 'v2_gap_recovery'
RES_2026 = RUNS / '20260922T215413Z_v2_gaprecovery' / 'v2_gap_recovery_reserved2026'
LONG = {'Gap-Long-Dev': RUNS / '20260922T195727Z_v2_gaprecovery' / 'v2_gap_recovery_long',
        'Gap-Long-2026': RUNS / '20260922T195745Z_v2_gaprecovery' / 'v2_gap_recovery_long_reserved2026'}
VARIABLES = ('temperature', 'oxygen')
SCALE_NOTE = 'oxygen in umol/kg as stored; tables print mg/l by c = 0.032844'


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    return float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean())


def dev_frame(variable):
    """The Gap-Dev rows rebuilt under the published environment, with every check of C-G6."""
    if gap.PERIOD != 'development' or gap.DESIGN != 'strict' or gap.SEED != 20260922:
        raise SystemExit('v2_gap_recovery must be at its published defaults here')
    frame = gap.build(variable, np.random.default_rng(gap.SEED))
    published = pd.read_csv(PUBLISHED_DEV / f'{variable}_chronos_bidir' / 'predictions.csv.gz')
    later = pd.read_csv(LATER_DEV / f'{variable}_chronos2_bidir' / 'predictions.csv.gz',
                        usecols=['episode', 'position', 'pchip'])
    keys = ['episode', 'position']
    m = published[keys + ['truth', 'linear', 'spline']].merge(
        frame[keys + ['truth', 'linear', 'spline', 'pchip', 'sensor', 'gap_length', 'time']],
        on=keys, how='outer', suffixes=('_pub', ''), indicator=True)
    rows_ok = bool((m._merge == 'both').all())
    exact = {c: float(np.nanmax(np.abs(m[c] - m[c + '_pub']))) for c in ('truth', 'linear', 'spline')}
    m2 = m.merge(later, on=keys, suffixes=('', '_later'))
    pchip_diff = float(np.nanmax(np.abs(m2.pchip - m2.pchip_later)))
    nan_same = bool((m2.pchip.isna() == m2.pchip_later.isna()).all())
    splines = json.loads((RUNS / 'v2_gap_splines.json').read_text())['variables'][variable]
    by_length = {str(int(L)): round(macro(part, 'pchip'), 4) for L, part in m.groupby('gap_length')}
    agg_ok = (round(macro(m, 'pchip'), 4) == splines['overall']['pchip']
              and all(by_length[L] == splines['by_gap_length'][L]['pchip'] for L in by_length))
    gate = {'rows_equal_published': rows_ok and len(m) == len(published),
            'max_abs_diff_truth_linear_spline': exact,
            'pchip_vs_20260922T204658Z_max_abs_diff': pchip_diff, 'pchip_nan_pattern_equal': nan_same,
            'pchip_macro_equals_v2_gap_splines': bool(agg_ok),
            'pchip_macro_overall': round(macro(m, 'pchip'), 4)}
    # a rebuild off the compute host matches the published columns to the last bits of the
    # hourly averaging, so rows are compared at the declared PCHIP tolerance; the contrasts read the
    # published truth, linear and spline columns and the rebuilt PCHIP
    gate['passed'] = bool(gate['rows_equal_published'] and max(exact.values()) <= 1e-12
                          and pchip_diff <= 1e-12 and nan_same and agg_ok)
    out = m[['episode', 'position', 'sensor', 'gap_length', 'time', 'truth_pub', 'linear_pub',
             'spline_pub', 'pchip']].rename(columns={'truth_pub': 'truth', 'linear_pub': 'linear',
                                                     'spline_pub': 'spline'})
    return out.reset_index(drop=True), gate


def cohort_block(frame, units, label, per_length_only=False):
    block = {}
    if not per_length_only:
        block['pooled'] = paired6(frame, 'pchip', 'linear', f'{label}|all', units)
    block['by_gap_length'] = {str(int(L)): paired6(part, 'pchip', 'linear', f'{label}|{int(L)}', units)
                              for L, part in frame.groupby('gap_length')}
    return block


def main():
    SIDECARS.mkdir(parents=True, exist_ok=True)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json', 'part': 'C',
              'contrast': 'C6: PCHIP minus linear interpolation, macro MAE over episodes, paired',
              'units': SCALE_NOTE, 'gate_C-G6': {}, 'Gap-Dev': {}, 'Gap-2026': {}, 'Gap-Long-Dev': {},
              'Gap-Long-2026': {}}
    published = json.loads((RUNS / 'v2_gap_intervals.json').read_text())['variables']
    for variable in VARIABLES:
        frame, gate = dev_frame(variable)
        frame.sort_values(['episode', 'position']).to_csv(
            SIDECARS / f'Gap-Dev__{variable}.csv.gz', index=False, compression=GZIP)
        cluster_of = clusters(frame)
        spline = paired(frame, 'spline', 'linear', f'{variable}|spline|all', cluster_of)
        want = published[variable]['overall']['spline']
        fields = ('delta', 'ci_low', 'ci_high', 'clusters', 'episodes', 'jackknife_t')
        gate['spline_minus_linear_reproduces_published'] = all(spline[f] == want[f] for f in fields)
        gate['clusters'] = len(set(cluster_of.values()))
        gate['passed'] = bool(gate['passed'] and gate['spline_minus_linear_reproduces_published'])
        report['gate_C-G6'][variable] = gate
        if not gate['passed']:
            continue
        report['Gap-Dev'][variable] = cohort_block(frame, cluster_of, f'{variable}|pchip')

        # Gap-2026: the published 6-72 h column with this part's 1-3 h rows (descriptive)
        pub = pd.read_csv(RES_2026 / f'{variable}_arx_bridge' / 'predictions.csv.gz')
        res = append(pub[['episode', 'sensor', 'gap_length', 'position', 'time', 'truth', 'linear',
                          'pchip', 'arx_bridge']], variable, 'arx_bridge')
        report['Gap-2026'][variable] = cohort_block(res, clusters(res), f'Gap-2026|{variable}|pchip')
        # long cohorts: per length on probe x calendar month (descriptive)
        for cohort, folder in LONG.items():
            f = pd.read_csv(folder / f'{variable}_chronos_bidir' / 'predictions.csv.gz')
            start = f.episode.str.rsplit('|', n=2).str[1].str[:7]
            units = dict(zip(f.episode, f.sensor + '|' + start))
            report[cohort][variable] = cohort_block(f, units, f'{cohort}|{variable}|pchip',
                                                    per_length_only=True)
    report['passed_C-G6'] = all(g['passed'] for g in report['gate_C-G6'].values())
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    for variable in VARIABLES:
        g = report['gate_C-G6'][variable]
        print(variable, 'C-G6', 'PASSED' if g['passed'] else 'FAILED', g)
    print('wrote', OUT.relative_to(RUNS.parents[2]))


if __name__ == '__main__':
    main()
