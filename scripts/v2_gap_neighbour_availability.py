"""Addendum 59 part A: where the neighbour-reading methods were scored, and where real gaps fall.

A masked gap removes only the target probe, so every method that reads another probe was scored
while the network around the target was up. Real outages are not like that. This counts, for every
scored hour of the four gap cohorts and of the two Bridge cohorts, how many of the target's four
nearest other probes (`v2_gap_recovery.nearest_others`) report at that hour, and how far away the
nearest reporting other probe is; and the same over the hours of the real gaps of the development
record and of 2026, by gap length (1-6, 7-48, 49-167, 168 h or more) and, for the Bridge cohorts,
by hours elapsed since the gap began.

Every arm that reads another probe then has its error and its margin over linear interpolation
(over the last value on the Bridge cohorts, which have no far edge) stratified by that count, and
the cohort's contrast reweighted to the real-gap distribution of the count, band by band, with the
cohort's resampling unit: overlap clusters on Gap-Dev and Gap-2026, probe x calendar month of the
gap start on the tiled long cohorts, onset weeks on the Bridge cohorts. A stratum the cohort does
not populate cannot be reweighted; its real-gap weight is reported as unsupported and the estimate
is renormalised over the strata that are populated. Descriptive: no verdict rests on a reweighted
figure alone.

Also summarised here: on the rows of each corrected Gaussian-process run, how often the published
process entered a silent neighbour at its margin mean (`gp_diagnostics.csv.gz` of each rerun).

Declared in config/bay_reserved_protocol_v2_addendum59.json (part A, neighbour_availability).
Usage: python -m scripts.v2_gap_neighbour_availability
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import GP_RERUNS, fix_neighbours, load_predictions
from scripts.v2_gap_intervals import MIN_CLUSTERS, clusters

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_neighbour_availability.json'
DRAWS = 600
K = 4
BANDS = (('1-6', 1, 6), ('7-48', 7, 48), ('49-167', 49, 167), ('168+', 168, 10 ** 9))
#: the published run each gap cohort's scored hours come from, and its record
GAP_COHORTS = {'Gap-Dev': ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery', 'development'),
               'Gap-2026': ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026',
                            'reserved'),
               'Gap-Long-Dev': ('20260922T195727Z_v2_gaprecovery', 'v2_gap_recovery_long',
                                'development'),
               'Gap-Long-2026': ('20260922T195745Z_v2_gaprecovery',
                                 'v2_gap_recovery_long_reserved2026', 'reserved')}
LONG = ('Gap-Long-Dev', 'Gap-Long-2026')
ROW_BANDS = {'Gap-Dev': ('1-6', '12-72'), 'Gap-2026': ('1-6', '12-72'),
             'Gap-Long-Dev': ('168-1440',), 'Gap-Long-2026': ('168-1440',)}
BRIDGE_PERIOD = {'Bridge-Dev': 'development', 'Bridge-2026': 'reserved'}
#: every gap arm that reads another probe (v2_gap_isolated.NETWORK), and the two parameter-free
#: neighbour rules of the run itself
NETWORK = ('gpr', 'gpr_k8', 'chronos2_cov_bidir', 'ens_mean', 'ens_mean_c2', 'ens_stack',
           'twin_rule', 'twin_else_gpr', 'twin_else_best', 'anchored_gbm', 'anchored_gbm_exo',
           'anchored_gbm_rel', 'anchored_gbm_rel_exo', 'regression_retro', 'chronos2_all_bidir',
           'chronos2_cov_resid_bidir', 'chronos2_cov_resid_lin_bidir', 'nearest_probe',
           'nearest_live_probe')
#: the Bridge arms that read another probe: the neighbour fallback and the anchored continuation
BRIDGE_NETWORK = ('neighbour_change', 'anchored_direct')
STRATA = ('0', '1', '2', '3', '4')


def band_of_length(length):
    for name, lo, hi in BANDS:
        if lo <= length <= hi:
            return name
    raise ValueError(length)


def panels():
    """(period, variable) -> (names, hourly index, values, distance, the four nearest others)."""
    from src.experiments import v2_gap_recovery as G
    out = {}
    for period in ('development', 'reserved'):
        meta, frames = G.observations(period)
        for variable in ('temperature', 'oxygen'):
            names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
            index = pd.date_range(min(frames[n].index.min() for n in names),
                                  max(frames[n].index.max() for n in names), freq='h', tz='UTC')
            values = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names},
                                  index=index).to_numpy(float)
            xy = meta.loc[names, ['x_m', 'y_m']].to_numpy()
            distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
            nearest = {i: G.nearest_others(distance, i, K) for i in range(len(names))}
            out[(period, variable)] = (names, index, values, distance, nearest)
    return out


def availability(panel, sensors, times):
    """Per row: how many of the probe's four nearest others report, and the distance to the nearest
    other probe that reports (NaN where none does)."""
    names, index, values, distance, nearest = panel
    column = {n: i for i, n in enumerate(names)}
    pos = index.get_indexer(pd.DatetimeIndex(pd.to_datetime(times, utc=True)))
    count = np.full(len(pos), np.nan)
    near = np.full(len(pos), np.nan)
    sensor_idx = np.array([column[s] for s in sensors])
    ok = pos >= 0
    finite = np.isfinite(values)
    for i in np.unique(sensor_idx):
        rows = np.flatnonzero((sensor_idx == i) & ok)
        if not len(rows):
            continue
        at = pos[rows]
        count[rows] = finite[np.ix_(at, nearest[i])].sum(axis=1)
        others = np.array([j for j in range(len(names)) if j != i])
        d = np.where(finite[np.ix_(at, others)], distance[i, others][None, :], np.inf)
        m = d.min(axis=1)
        near[rows] = np.where(np.isfinite(m), m, np.nan)
    return count, near


def describe(count, near, extra=None):
    count = np.asarray(count, float)
    out = {'hours': int(len(count)),
           'share_by_count': {s: round(float(np.mean(count == int(s))), 4) for s in STRATA},
           'share_none_of_four': round(float(np.mean(count == 0)), 4),
           'mean_count': round(float(np.mean(count)), 3),
           'share_no_other_probe_reporting': round(float(np.mean(np.isnan(near))), 4),
           'median_distance_to_nearest_reporting_m': (round(float(np.nanmedian(near)), 1)
                                                      if np.isfinite(near).any() else None)}
    if extra:
        out.update(extra)
    return out


def real_gaps(panel):
    """Every hour of every real gap inside a probe's own record, with its gap's length band and the
    hours elapsed since the gap began."""
    names, index, values, distance, nearest = panel
    finite = np.isfinite(values)
    rows = []
    for i, name in enumerate(names):
        seen = np.flatnonzero(finite[:, i])
        if not len(seen):
            continue
        a, b = seen[0], seen[-1]
        miss = ~finite[a:b + 1, i]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], miss.astype(int), [0]])))
        for s, e in zip(edges[::2], edges[1::2]):
            length = int(e - s)
            hours = a + np.arange(s, e)
            rows.append(pd.DataFrame({'sensor': name, 'time': index[hours], 'gap_length': length,
                                      'elapsed': np.arange(1, length + 1), 'gap': f'{name}|{s}'}))
    frame = pd.concat(rows, ignore_index=True)
    frame['count'], frame['near'] = availability(panel, frame.sensor.to_numpy(), frame.time)
    frame['band'] = frame.gap_length.map(band_of_length)
    return frame


def real_summary(frame, observed_near):
    out = {'all': describe(frame['count'], frame['near'], {'gaps': int(frame.gap.nunique())}),
           'by_length_band': {}, 'by_elapsed_band': {},
           'observed_hours_median_distance_to_nearest_reporting_m': observed_near}
    for name, lo, hi in BANDS:
        part = frame[frame.band == name]
        if len(part):
            out['by_length_band'][name] = describe(part['count'], part['near'],
                                                   {'gaps': int(part.gap.nunique())})
        part = frame[(frame.elapsed >= lo) & (frame.elapsed <= hi)]
        if len(part):
            out['by_elapsed_band'][name] = describe(part['count'], part['near'])
    return out


def observed_near(panel):
    """Median distance to the nearest reporting other probe over the hours each probe reports."""
    names, index, values, distance, nearest = panel
    finite = np.isfinite(values)
    got = []
    for i in range(len(names)):
        at = np.flatnonzero(finite[:, i])
        others = np.array([j for j in range(len(names)) if j != i])
        d = np.where(finite[np.ix_(at, others)], distance[i, others][None, :], np.inf).min(axis=1)
        got.append(d[np.isfinite(d)])
    return round(float(np.median(np.concatenate(got))), 1)


def stratified(frame, arm, ref, unit_of, weights, label):
    """`arm` minus `ref` per stratum of the count, macro over episodes within the stratum, and the
    contrast reweighted to `weights` (stratum -> share of real-gap hours), with a bootstrap over
    `unit_of` (episode -> unit) and the delete-one-unit jackknife."""
    d = frame.dropna(subset=[arm, ref, 'truth', 'count'])
    if not len(d):
        return None
    d = d.assign(diff=(d.truth - d[arm]).abs() - (d.truth - d[ref]).abs(),
                 ea=(d.truth - d[arm]).abs(), eb=(d.truth - d[ref]).abs(),
                 stratum=d['count'].astype(int).astype(str))
    per = d.groupby(['episode', 'stratum'])[['diff', 'ea', 'eb']].mean().reset_index()
    per['unit'] = per.episode.map(unit_of)
    units = sorted(per.unit.unique())
    u_index = {u: k for k, u in enumerate(units)}
    s_index = {s: k for k, s in enumerate(STRATA)}
    S = np.zeros((len(units), len(STRATA)))
    N = np.zeros((len(units), len(STRATA)))
    np.add.at(S, (per.unit.map(u_index).to_numpy(), per.stratum.map(s_index).to_numpy()),
              per['diff'].to_numpy())
    np.add.at(N, (per.unit.map(u_index).to_numpy(), per.stratum.map(s_index).to_numpy()), 1)
    out = {'by_count': {}, 'units': len(units)}
    for s in STRATA:
        part = per[per.stratum == s]
        if not len(part):
            continue
        n_units = int(part.unit.nunique())
        out['by_count'][s] = {'rows': int((d.stratum == s).sum()), 'episodes': int(len(part)),
                              'units': n_units, 'mae': round(float(part.ea.mean()), 4),
                              f'mae_{ref}': round(float(part.eb.mean()), 4),
                              'minus_reference': round(float(part['diff'].mean()), 4)}
    for name, members in (('0', ('0',)), ('1', ('1',)), ('2+', ('2', '3', '4'))):
        part = per[per.stratum.isin(members)]
        if len(part):
            # an episode with rows in two of the grouped strata enters once, by its pooled rows
            rows = d[d.stratum.isin(members)].groupby('episode')['diff'].mean()
            out.setdefault('grouped', {})[name] = {
                'episodes': int(len(rows)), 'units': int(rows.index.map(unit_of).nunique()),
                'minus_reference': round(float(rows.mean()), 4)}
    if weights is None:
        return out
    w = np.array([weights.get(s, 0.0) for s in STRATA])
    w = w / w.sum()                       # the shares are rounded to four decimals
    present = N.sum(axis=0) > 0
    supported = min(1.0, float(w[present].sum()))
    out['reweighted'] = {'weights_real_gaps': {s: round(float(x), 4) for s, x in zip(STRATA, w)},
                         'real_gap_weight_without_support': round(1 - supported, 4)}
    if supported <= 0:
        return out
    wn = np.where(present, w, 0) / supported

    def estimate(mult):
        num, den = mult @ S, mult @ N
        with np.errstate(invalid='ignore', divide='ignore'):
            m = np.where(den > 0, num / den, np.nan)
        keep = present & np.isfinite(m)
        if not keep.any():
            return np.nan
        return float(np.sum(wn[keep] * m[keep]) / wn[keep].sum())

    point = estimate(np.ones(len(units)))
    # the cohort's own contrast over the same rows: each episode's mean over all its rows
    unweighted = float(d.groupby('episode')['diff'].mean().mean())
    out['reweighted'].update({'point': round(point, 4), 'unweighted_point': round(unweighted, 4)})
    if len(units) < MIN_CLUSTERS:
        out['reweighted'].update({'estimable': False})
        return out
    rng = np.random.default_rng([20261001, zlib.crc32(label.encode())])
    draws = np.array([estimate(np.bincount(rng.integers(0, len(units), len(units)),
                                           minlength=len(units)).astype(float))
                      for _ in range(DRAWS)])
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    leave = []
    for k in range(len(units)):
        mult = np.ones(len(units))
        mult[k] = 0
        leave.append(estimate(mult))
    leave = np.array(leave)
    leave = leave[np.isfinite(leave)]
    from scipy.stats import t as student
    n = len(leave)
    se = float(np.sqrt((n - 1) / n * np.sum((leave - leave.mean()) ** 2)))
    half = float(student.ppf(0.975, n - 1)) * se
    out['reweighted'].update({
        'estimable': True, 'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4),
        'excludes_zero': bool(lo > 0 or hi < 0),
        'jackknife_t': [round(point - half, 4), round(point + half, 4)],
        'jackknife_excludes_zero': bool(point - half > 0 or point + half < 0)})
    return out


def gap_rows(cohort, variable, panel):
    """The cohort's rows of every arm (variants, exogenous and covariate rows merged on identical
    rows, as v2_gap_isolated reads them) with the run's two neighbour rules attached."""
    from scripts.v2_gap_isolated import rows as merged_rows
    run, results, period = GAP_COHORTS[cohort]
    parts = [merged_rows(cohort, variable, band) for band in ROW_BANDS[cohort]]
    frame = pd.concat(parts, ignore_index=True)
    rules = load_predictions(run, results, variable, 'chronos_bidir')
    # the nearest reporting probe joined the run's columns with addendum 32
    rules = rules[['episode', 'position'] + [c for c in ('nearest_probe', 'nearest_live_probe')
                                             if c in rules]]
    frame = frame.drop(columns=[c for c in ('nearest_probe', 'nearest_live_probe') if c in frame])
    frame = frame.merge(rules, on=['episode', 'position'], how='left', validate='one_to_one')
    frame['count'], frame['near'] = availability(panel, frame.sensor.to_numpy(), frame.time)
    frame['band'] = frame.gap_length.map(band_of_length)
    return frame


def units_for(cohort, part):
    if cohort in LONG:
        start = part.episode.str.rsplit('|', n=2).str[1].str[:7]
        return dict(zip(part.episode, part.sensor + '|' + start))
    return clusters(part)


def gap_cohort(cohort, variable, panel, real):
    run, results, period = GAP_COHORTS[cohort]
    scored = load_predictions(run, results, variable, 'chronos_bidir')
    count, near = availability(panel, scored.sensor.to_numpy(), scored.time)
    bands = scored.gap_length.map(band_of_length).to_numpy()
    out = {'run': f'{run}/{results}', 'scored_hours': describe(count, near),
           'scored_hours_by_band': {b: describe(count[bands == b], near[bands == b])
                                    for b, _, _ in BANDS if (bands == b).any()},
           'arms': {}}
    frame = gap_rows(cohort, variable, panel)
    for band, part in frame.groupby('band'):
        unit_of = units_for(cohort, part)
        weights = real['by_length_band'].get(band, {}).get('share_by_count')
        for arm in NETWORK:
            if arm not in part or part[arm].isna().all():
                continue
            got = stratified(part, arm, 'linear', unit_of, weights,
                             f'availability|{cohort}|{variable}|{band}|{arm}')
            if got:
                out['arms'].setdefault(arm, {})[band] = got
    return out


def bridge_cohorts(panels_):
    out = {}
    for variable in ('temperature', 'oxygen'):
        frame = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz')
        frame['time'] = pd.to_datetime(frame.onset, utc=True) + pd.to_timedelta(frame.lead, unit='h')
        frame['episode'] = (frame.fold.astype(str) + '|' + frame.sensor + '|'
                            + frame.onset.astype(str))
        week = pd.to_datetime(frame.onset, utc=True).dt.tz_localize(None).dt.to_period('W')
        frame['week'] = week.astype(str)
        for cohort, period in BRIDGE_PERIOD.items():
            part = frame[frame.cohort == cohort].copy()
            panel = panels_[(period, variable)]
            part['count'], part['near'] = availability(panel, part.sensor.to_numpy(), part.time)
            part['band'] = part.lead.map(band_of_length)
            real = real_summary_cache[(period, variable)]
            entry = {'scored_hours': describe(part['count'], part['near']),
                     'scored_hours_by_lead_band': {b: describe(g['count'], g['near'])
                                                   for b, g in part.groupby('band')},
                     'reference': 'persistence (the last value)',
                     'weights_from': 'real-gap hours by hours elapsed since the gap began',
                     'arms': {}}
            unit_of = dict(zip(part.episode, part.week))
            for band, g in part.groupby('band'):
                weights = real['by_elapsed_band'].get(band, {}).get('share_by_count')
                for arm in BRIDGE_NETWORK:
                    got = stratified(g, arm, 'persistence', unit_of, weights,
                                     f'availability|{cohort}|{variable}|{band}|{arm}')
                    if got:
                        entry['arms'].setdefault(arm, {})[band] = got
            out.setdefault(cohort, {})[variable] = entry
    return out


def gp_substitution():
    """Share of scored rows at which the published process entered a silent neighbour, from the
    diagnostics each corrected run wrote."""
    out = {}
    for name, run in GP_RERUNS.items():
        for path in sorted((RUNS / run).glob('*/*_gpr*/gp_diagnostics.csv.gz')):
            d = pd.read_csv(path)
            out[f'{run}/{path.parent.parent.name}/{path.parent.name}'] = {
                'rows': int(len(d)),
                'share_rows_published_entered_a_silent_neighbour_in_the_gap': round(float(
                    d.gp_published_substituted.mean()), 4),
                'share_episodes_published_entered_one_in_the_margin': round(float(
                    d.groupby('episode').gp_published_margin_substituted.max().mean()), 4),
                'share_rows_corrected_used_fewer_neighbours_than_report': round(float(
                    (d.gp_used < d.gp_reporting).mean()), 4),
                'corrected_neighbours_used_by_row': {str(int(k)): int(v) for k, v in
                                                     d.gp_used.value_counts().sort_index().items()}}
    return out


def pa3(gaps, bridges):
    """Registered prediction PA3: on every cohort, arms that read other probes keep less of their
    margin over the reference on hours with none of the four nearest reporting than on hours with
    two or more reporting. Read per (cohort, variable, band, arm) where both groups have five units
    or more. The declaration does not say how a cell without a margin is read: the primary reading
    keeps only cells where the arm beats the reference on hours with two or more reporting (there
    is a margin to keep), and the reading over every cell is kept beside it."""
    def read(cells):
        contrary = [c for c in cells if not c['keeps_less_with_none']]
        return {'cells_read': len(cells), 'cells_contrary': len(contrary),
                'cohorts_read': sorted({c['cohort'] for c in cells}),
                'cohorts_with_a_contrary_cell': sorted({c['cohort'] for c in contrary}),
                'held': bool(cells) and not contrary, 'contrary': contrary}

    cells = []
    for cohort, block in list(gaps.items()) + list(bridges.items()):
        for variable, entry in block.items():
            for arm, by_band in entry['arms'].items():
                for band, got in by_band.items():
                    g = got.get('grouped', {})
                    if '0' in g and '2+' in g and g['0']['units'] >= MIN_CLUSTERS \
                            and g['2+']['units'] >= MIN_CLUSTERS:
                        cells.append({'cohort': cohort, 'variable': variable, 'band': band,
                                      'arm': arm, 'none': g['0']['minus_reference'],
                                      'two_or_more': g['2+']['minus_reference'],
                                      'keeps_less_with_none': bool(g['0']['minus_reference']
                                                                   > g['2+']['minus_reference'])})
    with_margin = [c for c in cells if c['two_or_more'] < 0]
    return {'rule': 'per cohort, variable, band and arm where both groups have five or more units: '
                    'the margin over the reference with none of the four nearest reporting is less '
                    'favourable than with two or more reporting',
            'primary': {'cells': 'those where the arm beats the reference with two or more reporting',
                        **read(with_margin)},
            'every_cell': read(cells),
            'held': read(with_margin)['held']}


real_summary_cache = {}


def main():
    panels_ = panels()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum59.json',
              'what': 'neighbour availability over scored hours and over real-gap hours, and the '
                      'neighbour-reading arms stratified by it; descriptive',
              'neighbours': 'the four nearest other probes, v2_gap_recovery.nearest_others',
              'draws': DRAWS, 'min_units': MIN_CLUSTERS,
              'units': {'Gap-Dev': 'overlap cluster', 'Gap-2026': 'overlap cluster',
                        'Gap-Long-Dev': 'probe x calendar month of the gap start',
                        'Gap-Long-2026': 'probe x calendar month of the gap start',
                        'Bridge-Dev': 'onset week', 'Bridge-2026': 'onset week'},
              'macro': 'episode mean within a stratum, then mean over episodes (Bridge episodes '
                       'are fold, probe and onset; not the fold-sensor macro of the bridge tables)',
              'real_gaps': {}, 'gap_cohorts': {}, 'bridge_cohorts': {}}
    for (period, variable), panel in panels_.items():
        frame = real_gaps(panel)
        real_summary_cache[(period, variable)] = real_summary(frame, observed_near(panel))
        report['real_gaps'].setdefault(period, {})[variable] = real_summary_cache[(period, variable)]
    for cohort, (_, _, period) in GAP_COHORTS.items():
        for variable in ('temperature', 'oxygen'):
            print(cohort, variable, flush=True)
            report['gap_cohorts'].setdefault(cohort, {})[variable] = gap_cohort(
                cohort, variable, panels_[(period, variable)],
                real_summary_cache[(period, variable)])
    report['bridge_cohorts'] = bridge_cohorts(panels_)
    report['registered_predictions'] = {'PA3': pa3(report['gap_cohorts'], report['bridge_cohorts'])}
    report['gp_published_substitution'] = gp_substitution()
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    for period, block in report['real_gaps'].items():
        for variable, r in block.items():
            print(f'real gaps {period} {variable}: none of four reporting by length band',
                  {b: (e['share_none_of_four'], e['hours'], e['gaps'])
                   for b, e in r['by_length_band'].items()})
    for cohort, block in report['gap_cohorts'].items():
        for variable, e in block.items():
            print(f'{cohort} {variable}: scored hours none of four {e["scored_hours"]["share_none_of_four"]}')
    pa = report['registered_predictions']['PA3']
    print('PA3 held:', pa['held'], {k: (pa[k]['cells_contrary'], pa[k]['cells_read'],
                                        pa[k]['cohorts_with_a_contrary_cell'])
                                    for k in ('primary', 'every_cell')})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
