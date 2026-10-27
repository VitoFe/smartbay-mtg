"""Addendum 82, part C: the gates that must pass before any 1-3 h Gap-2026 row is scored.

  C-G0 episodes      the 1-3 h build holds 899 temperature episodes (300, 300, 299) on the five
                     published probes and 360 oxygen (120 x 3) on the two published sondes, the
                     episodes an availability-only replication of `build` selects; the gate run's
                     6-72 h episodes are the published ones; both runs share one snapshot
  C-G1 reproduction  the gate run (same snapshot, 6-72 h) reproduces the published Gap-2026 rows of
                     every arm: keys, truth, linear, persistence, spline, nearest probe (corrected)
                     and PCHIP exactly; the Chronos arms and the addendum-35 arms to 1e-4 per row;
                     the Gaussian processes and the anchored and ARX arms to 1e-9; then
                     v2_gap_intervals.paired on the gate run reproduces the published per-length
                     Chronos-Bolt bidirectional minus linear at four decimals
  C-G2 frozen        every tuned arm ran on the injected published selection, no trial run, and the
                     surviving published study databases agree with it
  C-G4 rows          every arm of each run scores identical rows; truth equals the raw export
  C-G5 leakage       no comparand equals the truth on a probe's rows; declared climatology; every
                     hour of every masked stretch scored and observed

C-G3 (nothing published moves) and the stack-weight part of C-G2 are checked by the scorer, which
recomputes the published entries. Writes runs/v2_addendum82/C/v2_gap_short2026_gate.json.
Usage: python -m scripts.v2_gap_short2026_gate
"""
import json
import sys

import numpy as np
import pandas as pd

from scripts.v2_gap_intervals import clusters, paired
from scripts.v2_gap_short2026_rows import GATE, PART, RUNS, SHORT, SHORT_RUN
from src.experiments import v2_gap_recovery as gap

OUT = PART / 'v2_gap_short2026_gate.json'
VARIABLES = ('temperature', 'oxygen')
PUBLISHED_RESULTS = 'v2_gap_recovery_reserved2026'
CHRONOS = ('chronos_bidir', 'chronos_forward', 'chronos2_bidir', 'chronos2_cov_bidir')
A35 = ('chronos2_exo_bidir', 'chronos2_all_bidir', 'bolt_resid_bidir', 'chronos2_cov_resid_bidir',
       'bolt_resid_lin_bidir', 'chronos2_cov_resid_lin_bidir')
GP = ('gpr', 'gpr_k8')
ANCHORED = ('anchored_gbm', 'anchored_gbm_exo', 'anchored_gbm_rel', 'anchored_gbm_rel_exo')
ARX = ('arx_bridge', 'arx_bridge_exo')
ARMS = CHRONOS + A35 + GP + ANCHORED + ARX
TOLERANCE = {**{a: 1e-4 for a in CHRONOS + A35}, **{a: 1e-9 for a in GP + ANCHORED + ARX}}
DECLARED_COUNTS = {'temperature': {'1': 300, '2': 300, '3': 299},
                   'oxygen': {'1': 120, '2': 120, '3': 120}}
DECLARED_ROWS = {'temperature': 1797, 'oxygen': 720}
A35_RUNS = ('20260922T224158Z_v2_gaprecovery', '20260922T225405Z_v2_gaprecovery',
            '20260922T225420Z_v2_gaprecovery')


def published_run(variable, arm):
    if arm in ('chronos_bidir', 'chronos_forward'):
        return '20260922T180814Z_v2_gaprecovery'
    if arm in ('chronos2_bidir', 'chronos2_cov_bidir'):
        return '20260922T204715Z_v2_gaprecovery'
    if arm in GP:
        return '20261001T110810Z_v2_gaprecovery'
    if arm in ARX or (arm in ANCHORED and variable == 'oxygen'):
        return '20260922T215413Z_v2_gaprecovery'
    if arm in ANCHORED:
        return '20260923T101736Z_v2_gaprecovery'
    done = [r for r in A35_RUNS
            if (RUNS / r / PUBLISHED_RESULTS / f'{variable}_{arm}' / 'report.json').exists()]
    if len(done) != 1:
        raise SystemExit(f'{variable} {arm}: {len(done)} published runs')
    return done[0]


def published_frame(variable, arm):
    return pd.read_csv(RUNS / published_run(variable, arm) / PUBLISHED_RESULTS / f'{variable}_{arm}'
                       / 'predictions.csv.gz')


def run_frame(where, variable, arm):
    return pd.read_csv(where / f'{variable}_{arm}' / 'predictions.csv.gz')


def max_diff(a, b):
    """(largest absolute difference, NaN pattern equal)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
    ok = ~np.isnan(a) & ~np.isnan(b)
    return (float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0), same_nan


def replicate_episodes(variable, lengths):
    """build's episode selection from availability alone: same draws, same fold-straddle drop."""
    meta, frames = gap.observations('reserved')
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    ok_all = np.column_stack([frames[n][variable].reindex(index).notna().to_numpy() for n in names])
    rng = np.random.default_rng(gap.SEED)
    rows = []
    for length in lengths:
        need = gap.MARGIN + length + gap.MARGIN
        per = gap.PER_LENGTH_PER_PROBE if length <= 24 else 20 if length <= 168 else 8
        for i, name in enumerate(names):
            ok = ok_all[:, i]
            cum = np.concatenate([[0], np.cumsum(ok)])
            if len(ok) < need:
                continue
            starts = np.flatnonzero((cum[need:] - cum[:-need]) == need)
            if not len(starts):
                continue
            chosen = rng.choice(starts, size=min(per, len(starts)), replace=False)
            spread = (np.arange(length) if length <= gap.MAX_ROWS_PER_EPISODE else
                      np.unique(np.linspace(0, length - 1, gap.MAX_ROWS_PER_EPISODE).astype(int)))
            for s in chosen:
                g0 = int(s) + gap.MARGIN
                key = f'{name}|{index[g0].isoformat()}|{length}'
                rows += [(key, index[g0 + int(k)]) for k in spread]
    f = pd.DataFrame(rows, columns=['episode', 'time'])
    edges = pd.date_range(f.time.min(), f.time.max(), periods=gap.FOLDS + 1)
    span = f.groupby('episode').time.agg(['min', 'max'])
    b0 = np.searchsorted(edges[1:-1], span['min'].to_numpy(), side='right')
    b1 = np.searchsorted(edges[1:-1], span['max'].to_numpy(), side='right')
    return set(span.index[b0 == b1]), int((b0 != b1).sum())


def raw_panel(variable):
    meta, frames = gap.observations('reserved')
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    return pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)


def truth_matches_raw(frame, panel):
    t = pd.to_datetime(frame.time, utc=True)
    at = panel.index.get_indexer(t)
    col = {n: i for i, n in enumerate(panel.columns)}
    raw = panel.to_numpy(float)[at, frame.sensor.map(col).to_numpy()]
    return bool((at >= 0).all()) and max_diff(raw, frame.truth)[0] == 0.0


def g0_episodes():
    out, ok = {}, True
    for variable in VARIABLES:
        short = run_frame(SHORT, variable, 'chronos_bidir')
        counts = {str(int(L)): int(n) for L, n in short.groupby('gap_length').episode.nunique().items()}
        expected, dropped = replicate_episodes(variable, [1, 2, 3])
        published = published_frame(variable, 'chronos_bidir')
        gate = run_frame(GATE, variable, 'chronos_bidir')
        entry = {'episodes_by_length': counts, 'declared': DECLARED_COUNTS[variable],
                 'rows': int(len(short)), 'declared_rows': DECLARED_ROWS[variable],
                 'probes': sorted(short.sensor.unique()),
                 'probes_are_the_published_ones': bool(set(short.sensor) == set(published.sensor)),
                 'equal_to_availability_replication': bool(set(short.episode) == expected),
                 'straddlers_dropped': dropped,
                 'gate_run_episodes_are_published': bool(set(gate.episode) == set(published.episode))}
        entry['passed'] = bool(counts == DECLARED_COUNTS[variable]
                               and entry['rows'] == DECLARED_ROWS[variable]
                               and entry['probes_are_the_published_ones']
                               and entry['equal_to_availability_replication']
                               and entry['gate_run_episodes_are_published'])
        ok &= entry['passed']
        out[variable] = entry
    # one launch, one snapshot: both kinds are folders of the same run
    out['one_snapshot'] = SHORT_RUN
    out['passed'] = bool(ok)
    return out


def g1_reproduction():
    out, ok = {}, True
    for variable in VARIABLES:
        block = {}
        published_base = published_frame(variable, 'chronos_bidir')
        gate_base = run_frame(GATE, variable, 'chronos_bidir')
        merged = published_base.merge(gate_base, on=['episode', 'position'], how='outer',
                                      suffixes=('_pub', ''), indicator=True)
        keys_ok = bool((merged._merge == 'both').all())
        cols = {}
        for c in ('truth', 'linear', 'persistence', 'spline', 'gap_length'):
            d, same = max_diff(merged[c + '_pub'], merged[c])
            cols[c] = {'max_abs_diff': d, 'nan_pattern_equal': same, 'exact': d == 0.0 and same}
        # nearest probe: the published column carried the self-neighbour leak on temperature; the
        # corrected column of runs/v2_gap_neighbours is the published reference
        if variable == 'temperature':
            ref = pd.read_csv(RUNS / 'v2_gap_neighbours' /
                              '20260922T180814Z_v2_gaprecovery__temperature.csv.gz',
                              usecols=['episode', 'position', 'nearest_probe'])
        else:
            ref = published_base[['episode', 'position', 'nearest_probe']]
        m = ref.merge(gate_base[['episode', 'position', 'nearest_probe']], on=['episode', 'position'],
                      suffixes=('_pub', ''))
        d, same = max_diff(m.nearest_probe_pub, m.nearest_probe)
        cols['nearest_probe'] = {'max_abs_diff': d, 'nan_pattern_equal': same,
                                 'exact': d == 0.0 and same and len(m) == len(gate_base)}
        pchip_ref = pd.read_csv(RUNS / '20260922T215413Z_v2_gaprecovery' / PUBLISHED_RESULTS /
                                f'{variable}_arx_bridge' / 'predictions.csv.gz',
                                usecols=['episode', 'position', 'pchip'])
        m = pchip_ref.merge(gate_base[['episode', 'position', 'pchip']], on=['episode', 'position'],
                            suffixes=('_pub', ''))
        d, same = max_diff(m.pchip_pub, m.pchip)
        cols['pchip'] = {'max_abs_diff': d, 'nan_pattern_equal': same,
                         'exact': d == 0.0 and same and len(m) == len(gate_base)}
        clim = pd.read_csv(RUNS / 'v2_gap_climatology' / f'20260922T180814Z_v2_gaprecovery__{variable}.csv.gz',
                           usecols=['episode', 'position', 'climatology'])
        m = clim.merge(gate_base[['episode', 'position', 'climatology']], on=['episode', 'position'],
                       suffixes=('_pub', ''))
        d, same = max_diff(m.climatology_pub, m.climatology)
        cols['climatology_declared'] = {'max_abs_diff': d, 'nan_pattern_equal': same,
                                        'exact': d == 0.0 and same, 'note': 'reported, not declared'}
        base_ok = keys_ok and len(merged) == len(published_base) and all(
            cols[c]['exact'] for c in ('truth', 'linear', 'persistence', 'spline', 'gap_length',
                                       'nearest_probe', 'pchip'))
        block['baselines'] = {'keys_identical': keys_ok, 'rows': int(len(gate_base)), 'columns': cols,
                              'passed': bool(base_ok)}
        ok &= base_ok
        arms = {}
        for arm in ARMS:
            pub = published_frame(variable, arm)
            new = run_frame(GATE, variable, arm)
            m = pub[['episode', 'position', arm]].merge(new[['episode', 'position', arm]],
                                                        on=['episode', 'position'], how='outer',
                                                        suffixes=('_pub', ''), indicator=True)
            d, same = max_diff(m[arm + '_pub'], m[arm])
            passed = bool((m._merge == 'both').all() and same and d <= TOLERANCE[arm])
            arms[arm] = {'published_run': published_run(variable, arm), 'rows': int(len(m)),
                         'max_abs_diff': d, 'nan_pattern_equal': same, 'tolerance': TOLERANCE[arm],
                         'passed': passed}
            ok &= passed
        block['arms'] = arms
        # the published per-length intervals of Chronos-Bolt bidirectional minus linear
        published = json.loads((RUNS / 'v2_gap_intervals_reserved2026.json').read_text())[
            'variables'][variable]['by_gap_length']
        cluster_of = clusters(gate_base)
        intervals = {}
        for L, part in gate_base.groupby('gap_length'):
            got = paired(part, 'chronos_bidir', 'linear', f'{variable}|chronos_bidir|{int(L)}',
                         cluster_of)
            want = published[str(int(L))]['chronos_bidir']
            fields = ('delta', 'ci_low', 'ci_high', 'clusters', 'episodes', 'jackknife_t',
                      'excludes_zero', 'jackknife_excludes_zero')
            same = all(got[f] == want[f] for f in fields)
            intervals[str(int(L))] = {'reproduced': {f: got[f] for f in fields},
                                      'published': {f: want[f] for f in fields}, 'passed': bool(same)}
            ok &= same
        block['intervals_chronos_bidir_minus_linear'] = intervals
        out[variable] = block
    out['passed'] = bool(ok)
    return out


def g2_frozen():
    selections = json.loads((PART / 'published_selections.json').read_text())['studies']
    cross_path = PART / 'remote' / SHORT_RUN / 'selection_crosscheck.json'
    cross = json.loads(cross_path.read_text()) if cross_path.exists() else {}
    out, ok = {'arms': {}}, True
    for where, kind in ((GATE, 'gate'), (SHORT, 'short')):
        for variable in VARIABLES:
            for arm in ANCHORED + ARX + A35:
                if arm not in ARMS:
                    continue
                report = json.loads((where / f'{variable}_{arm}' / 'report.json').read_text())
                for entry in report['selections']:
                    s = entry['selection']
                    if s.get('tuned') is False:
                        continue
                    want = selections[s['study']]
                    good = (s.get('frozen_from') == want['source_run'] and s['params'] == want['params']
                            and s.get('trials_run_here') == 0 and not s.get('smoke_only'))
                    out['arms'][f'{kind}|{variable}|{arm}|{s["study"]}'] = {
                        'frozen_from': s.get('frozen_from'), 'params_equal': s['params'] == want['params'],
                        'trials_run_here': s.get('trials_run_here'), 'passed': bool(good)}
                    ok &= good
    out['published_databases'] = {'studies': len(cross),
                                  'present': sum(v.get('database') == 'present' for v in cross.values()),
                                  'agree': sum(bool(v.get('agrees')) for v in cross.values())}
    ok &= out['published_databases']['agree'] == len(selections)
    out['passed'] = bool(ok)
    return out


def g4_g5_rows():
    out, ok = {}, True
    for variable in VARIABLES:
        panel = raw_panel(variable)
        block = {}
        for kind, where in (('gate', GATE), ('short', SHORT)):
            frames = {arm: run_frame(where, variable, arm) for arm in ARMS}
            first = frames['chronos_bidir'][['episode', 'position']]
            identical = all(f[['episode', 'position']].equals(first) for f in frames.values())
            base = frames['chronos_bidir']
            truth_raw = truth_matches_raw(base, panel)
            # every hour of every masked stretch scored and observed
            per = base.groupby('episode').agg(n=('position', 'size'), L=('gap_length', 'first'),
                                              lo=('position', 'min'), hi=('position', 'max'))
            complete = bool(((per.n == per.L) & (per.lo == 1) & (per.hi == per.L)).all()) \
                if kind == 'short' else bool((per.lo == 1).all())
            finite = bool(np.isfinite(base.truth).all())
            # leakage: share of a probe's rows on which a comparand equals the truth exactly
            worst = 0.0
            for arm, f in frames.items():
                for column in ('nearest_probe', 'nearest_live_probe', arm):
                    if column in f:
                        eq = (f[column] == f.truth).groupby(f.sensor).mean()
                        worst = max(worst, float(eq.max()))
            reports = {arm: json.loads((where / f'{variable}_{arm}' / 'report.json').read_text())
                       for arm in ARMS}
            declared = all(r.get('design') == 'strict' and r.get('period') == 'reserved'
                           and not r.get('smoke_only') for r in reports.values())
            passed = identical and truth_raw and complete and finite and worst < 1.0 and declared
            block[kind] = {'identical_rows_across_arms': identical, 'truth_equals_raw_export': truth_raw,
                           'every_hour_scored': complete, 'truth_finite': finite,
                           'largest_share_of_a_probes_rows_equal_to_truth': round(worst, 6),
                           'declared_climatology_strict_reserved_full': declared,
                           'passed': bool(passed)}
            ok &= passed
        out[variable] = block
    out['passed'] = bool(ok)
    return out


def main():
    global ARMS, OUT
    if '--preliminary' in sys.argv:
        # a check while the Gaussian processes are still running; never the gate of record
        ARMS = tuple(a for a in ARMS if a not in GP)
        OUT = PART / 'v2_gap_short2026_gate_preliminary.json'
    result = {'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json', 'part': 'C',
              'run': SHORT_RUN}
    result['C-G0_episodes'] = g0_episodes()
    result['C-G1_reproduction'] = g1_reproduction()
    result['C-G2_frozen'] = g2_frozen()
    result['C-G4_C-G5_rows_leakage'] = g4_g5_rows()
    result['passed'] = all(v['passed'] for k, v in result.items() if k.startswith('C-G'))
    OUT.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    for k, v in result.items():
        if k.startswith('C-G'):
            print(k, 'PASSED' if v['passed'] else 'FAILED')
    print('all gates', 'PASSED' if result['passed'] else 'FAILED', '->', OUT.relative_to(PART.parents[3]))


if __name__ == '__main__':
    main()
