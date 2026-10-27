"""Addendum 84: the month-by-hour climatology anchored to both edges of a gap.

Declared in config/bay_reserved_protocol_v2_addendum84.json before this module existed. The arm
`clim_anchored` fills a masked stretch of a probe's record with linear interpolation between the two
edge readings plus the climatology's departure from its own straight line between the edges:

    fill(t) = C(t) + a_L + (a_R - a_L) w,   a = y - C at each edge,   w = (t - t_L) / (t_R - t_L)

C is the probe's month-by-hour mean fitted outside the masked stretch, as v2_gap_recovery.build fits
the declared climatology column. Where C(t) has no reading outside the stretch, the departure is
taken as zero and the fill is linear interpolation on that row. The arm reads only the probe's own
record, so it belongs to the own-record class of the no-neighbour reading (scripts/v2_gap_isolated.py).

Nothing is refitted and no published file is touched:

  A84-G1   the published episodes of the four gap cohorts are rebuilt from their keys and the
           period's record, and must reproduce the published truth, linear interpolation and
           declared climatology on every row; every per-row comparator file read must score rows of
           the reconstruction with the same truth and linear interpolation
  A84-G3   on every episode, replacing the masked hours by noise or by nothing leaves the fill
           unchanged exactly (A84-G2, the definition, is unit-tested in tests/test_gap_climanchor.py)
  A84-G4   the plain climatology minus linear and the Kalman bridge minus linear are recomputed on
           the published units and seed labels and must equal their published entries

Only then is the arm contrasted, paired on rows where both arms exist, with linear interpolation,
the plain climatology, the Kalman bridge, every incumbent of its band or length and the method the
no-neighbour reading chooses there: per band up to 72 h on the band's overlap clusters (per length
on the whole cohort's clusters beside), per length past a week on probe x calendar month.

Writes runs/v2_gap_climanchor.json and the arm's per-row column under runs/v2_gap_climanchor/.
Usage: python -m scripts.v2_gap_climanchor [--gates-only]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_climanchor.json'
ROWS = RUNS / 'v2_gap_climanchor'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum84.json'
ARM = 'clim_anchored'
VARIABLES = ('oxygen', 'temperature')
KEYS = ['episode', 'position']
#: gzip writes the wall-clock time into its header; a fixed one keeps the digest stable
GZIP = {'method': 'gzip', 'mtime': 0}
TOLERANCE = 1e-9
#: month (1-12, index 0 unused) x hour of day, as v2_gap_recovery.build indexes its climatology
CELLS = 13 * 24

#: cohort -> the published run, its results folder, the record it was built from, and whether the
#: 1-3 h episodes of addendum 82 part C are appended (Gap-2026 only)
COHORTS = {
    'Gap-Dev': {'run': '20260922T171859Z_v2_gaprecovery', 'results': 'v2_gap_recovery',
                'period': 'development', 'short': False},
    'Gap-2026': {'run': '20260922T180814Z_v2_gaprecovery', 'results': 'v2_gap_recovery_reserved2026',
                 'period': 'reserved', 'short': True},
    'Gap-Long-Dev': {'run': '20260922T195727Z_v2_gaprecovery', 'results': 'v2_gap_recovery_long',
                     'period': 'development', 'short': False},
    'Gap-Long-2026': {'run': '20260922T195745Z_v2_gaprecovery',
                      'results': 'v2_gap_recovery_long_reserved2026', 'period': 'reserved',
                      'short': False},
}
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'),
         '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
LONG = '168-1440'
#: the rescored readings of addenda 33 and 82 part C, whose decisions and no-neighbour answers are
#: the incumbents and the own-record answers this arm is read against
VARIANTS = RUNS / 'v2_addendum82/C/rescored/v2_gap_variants.json'
EXOGENOUS = RUNS / 'v2_addendum82/C/rescored/v2_gap_exogenous.json'
ISOLATED = RUNS / 'v2_addendum82/C/rescored/v2_gap_isolated.json'
#: arms that read only the probe's own record (at most the forcing): between two of them the
#: temperature scale is the 0.029 degC quantisation, against any other the 0.20 degC disagreement
#: of co-located probes
OWN_RECORD = ('linear', 'climatology', 'pchip', 'chronos_bidir', 'chronos_forward', 'chronos2_bidir',
              'arx_bridge', 'arx_bridge_exo', 'chronos2_exo_bidir', 'bolt_resid_bidir',
              'bolt_resid_lin_bidir', ARM)


# ------------------------------------------------------------------------------------ the arm
def cells_of(index):
    """month * 24 + hour for each hour of a DatetimeIndex."""
    return index.month.to_numpy() * 24 + index.hour.to_numpy()


def climatology_outside(values, cells, g0, length):
    """The probe's month-by-hour mean with the masked stretch [g0, g0 + length) left out.

    The masked hours are blanked before anything is summed, so no masked reading can reach the
    mean whatever it holds; v2_gap_recovery.build reaches the same means by subtracting the
    stretch's sums from the whole record's (equal to rounding, gate A84-G1).
    """
    v = np.array(values, dtype=float)
    v[g0:g0 + length] = np.nan
    ok = np.isfinite(v)
    total = np.bincount(cells[ok], weights=v[ok], minlength=CELLS)
    count = np.bincount(cells[ok], minlength=CELLS)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / count, np.nan)


def anchored(y_left, y_right, c_t, c_left, c_right, position, length):
    """(fill, linear, defined) at `position` hours after the left edge (1 .. length inside the gap).

    Linear interpolation is written exactly as v2_gap_recovery.build writes it, and the climatology's
    chord between the edges the same way, so a row whose C(t) is undefined returns linear exactly.
    """
    position = np.asarray(position)
    linear = y_left + (y_right - y_left) * position / (length + 1)
    chord = c_left + (c_right - c_left) * position / (length + 1)
    departure = np.asarray(c_t, dtype=float) - chord
    defined = np.isfinite(departure)
    return linear + np.where(defined, departure, 0.0), linear, defined


def episode_fill(values, cells, g0, length, offsets):
    """(fill, linear, defined, C(t)) at the scored offsets (0-based) of one masked stretch."""
    clim = climatology_outside(values, cells, g0, length)
    offsets = np.asarray(offsets)
    t = g0 + offsets
    left, right = g0 - 1, g0 + length
    fill, linear, defined = anchored(values[left], values[right], clim[cells[t]], clim[cells[left]],
                                     clim[cells[right]], offsets + 1, length)
    return fill, linear, defined, clim[cells[t]]


# --------------------------------------------------------------------------- reconstruction
_RECORDS = {}


def record(period):
    if period not in _RECORDS:
        from src.experiments.v2_gap_recovery import observations
        _RECORDS[period] = observations(period)
    return _RECORDS[period]


def panel(period, variable):
    """{probe: hourly values}, index: the period's panel of `variable`, as build makes it."""
    _, frames = record(period)
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    return {n: frames[n][variable].reindex(index).to_numpy(float) for n in names}, index


def published_rows(cohort, variable):
    """The cohort's published rows with the declared climatology, read as every gap scorer reads them."""
    from scripts.v2_gap_climatology import load_predictions
    c = COHORTS[cohort]
    frame = load_predictions(c['run'], c['results'], variable, 'chronos_bidir')
    if c['short']:
        from scripts.v2_gap_short2026_rows import append
        frame = append(frame, variable, 'chronos_bidir')
    keep = ['episode', 'sensor', 'gap_length', 'position', 'time', 'truth', 'linear', 'climatology']
    return frame[keep].reset_index(drop=True)


def rebuild(cohort, variable):
    """The published rows with the arm, and gates A84-G1 (columns) and A84-G3 (no masked reading)."""
    c = COHORTS[cohort]
    series, index = panel(c['period'], variable)
    cells = cells_of(index)
    where = pd.Series(np.arange(len(index)), index=index)
    rows = published_rows(cohort, variable)
    rng = np.random.default_rng(84)
    n = len(rows)
    truth, linear, clim = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    fill, defined = np.full(n, np.nan), np.zeros(n, bool)
    g3_failed, episodes = [], 0
    for key, block in rows.groupby('episode', sort=False):
        sensor, start, length = key.rsplit('|', 2)
        length = int(length)
        g0 = int(where[pd.Timestamp(start)])
        values = series[sensor]
        offsets = block.position.to_numpy() - 1
        f, lin, ok, c_t = episode_fill(values, cells, g0, length, offsets)
        i = block.index.to_numpy()
        truth[i], linear[i], clim[i], fill[i], defined[i] = values[g0 + offsets], lin, c_t, f, ok
        noisy = values.copy()
        noisy[g0:g0 + length] = rng.normal(0.0, 100.0, length)
        blank = values.copy()
        blank[g0:g0 + length] = np.nan
        if not all(np.array_equal(f, episode_fill(other, cells, g0, length, offsets)[0], equal_nan=True)
                   for other in (noisy, blank)):
            g3_failed.append(key)
        episodes += 1
    finite = np.isfinite(rows.climatology.to_numpy()) & np.isfinite(clim)
    pattern = int((np.isfinite(rows.climatology.to_numpy()) != np.isfinite(clim)).sum())
    g1 = {'rows': int(n), 'episodes': int(episodes),
          'truth_max_abs_difference': float(np.nanmax(np.abs(rows.truth.to_numpy() - truth))),
          'truth_rows_identical': int((rows.truth.to_numpy() == truth).sum()),
          'linear_max_abs_difference': float(np.nanmax(np.abs(rows.linear.to_numpy() - linear))),
          'linear_rows_identical': int((rows.linear.to_numpy() == linear).sum()),
          'climatology_max_abs_difference': float(np.max(np.abs(rows.climatology.to_numpy()[finite]
                                                                - clim[finite]), initial=0.0)),
          'climatology_undefined_rows': int((~np.isfinite(clim)).sum()),
          'climatology_undefined_pattern_mismatches': pattern,
          'unmatched_truth': int(np.isnan(truth).sum())}
    g1['passed'] = bool(g1['truth_max_abs_difference'] <= TOLERANCE and g1['unmatched_truth'] == 0
                        and g1['linear_max_abs_difference'] <= TOLERANCE
                        and g1['climatology_max_abs_difference'] <= TOLERANCE and pattern == 0
                        and bool(defined.sum() == np.isfinite(clim).sum()))
    g3 = {'episodes': int(episodes), 'failed': len(g3_failed), 'first_failures': g3_failed[:5],
          'perturbations': 'masked hours replaced by N(0, 100) noise, and by missing values',
          'passed': not g3_failed}
    out = rows.assign(climatology_rebuilt=clim, **{ARM: fill}, climatology_defined=defined)
    return out, g1, g3


# --------------------------------------------------------------------------- rows and units
def band_frame(cohort, variable, band, arm_frame):
    """The published per-row files of addenda 33-35 for one band (rescored where part C rescored
    them), merged, with the arm and the declared climatology attached; the rest of A84-G1."""
    from scripts.v2_gap_short2026_score import ROW_FOLDERS, Rows
    base = None
    for name in ROW_FOLDERS:
        frame = pd.read_csv(Rows(name) / f'{cohort}__{variable}__{band}.csv.gz')
        if base is None:
            base = frame
            continue
        if not (frame[KEYS].equals(base[KEYS]) and (frame.truth - base.truth).abs().max() == 0):
            raise SystemExit(f'{name} {cohort} {variable} {band}: rows differ from the variants rows')
        for column in frame.columns:
            if column not in base:
                base[column] = frame[column].to_numpy()
    lo, hi = (int(x) for x in band.split('-'))
    mine = arm_frame[(arm_frame.gap_length >= lo) & (arm_frame.gap_length <= hi)]
    if 'climatology' in base or ARM in base:
        raise SystemExit(f'{cohort} {variable} {band}: the per-row files already hold the columns added here')
    merged = base.merge(mine[KEYS + ['truth', 'linear', 'climatology', ARM, 'climatology_defined']],
                        on=KEYS, how='outer', validate='one_to_one', suffixes=('', '_rebuilt'),
                        indicator=True)
    check = {'rows_in_files': int(len(base)), 'rows_rebuilt': int(len(mine)),
             'unmatched': int((merged._merge != 'both').sum()),
             'truth_max_abs_difference': float((merged.truth - merged.truth_rebuilt).abs().max()),
             'linear_max_abs_difference': float((merged.linear - merged.linear_rebuilt).abs().max())}
    check['passed'] = bool(check['unmatched'] == 0 and check['truth_max_abs_difference'] <= TOLERANCE
                           and check['linear_max_abs_difference'] <= TOLERANCE)
    merged = merged.drop(columns=['_merge', 'truth_rebuilt', 'linear_rebuilt'])
    merged['climatology_defined'] = merged.climatology_defined.astype(bool)
    return merged, check


def probe_month(part):
    """episode -> probe x calendar month of the gap start (addendum 32's conservative unit)."""
    start = part.episode.str.rsplit('|', n=2).str[1].str[:7]
    return dict(zip(part.episode, part.sensor + '|' + start))


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean())


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------------------------------------------ gate G4
SAME_FOUR = ('delta', 'ci_low', 'ci_high', 'clusters', 'episodes', 'jackknife_t')
SAME_SIX = SAME_FOUR + ('paired_rows',)


def compare(got, want, fields):
    if got is None or want is None:
        return {'passed': False, 'reason': 'missing', 'got': got is not None, 'published': want is not None}
    diffs = {f: [got.get(f), want.get(f)] for f in fields if got.get(f) != want.get(f)}
    return {'passed': not diffs, **({'differences': diffs} if diffs else {})}


def gate_g4(frames, bands):
    """The published climatology and Kalman-bridge contrasts, recomputed on the rebuilt rows."""
    from scripts.v2_gap_intervals import clusters, paired
    from scripts.v2_gap_short2026_rows import paired6
    checks = {}
    # (a) climatology minus linear by length on the strict cohorts' published units and labels
    for cohort, artifact in (('Gap-Dev', 'v2_gap_intervals.json'),
                             ('Gap-2026', 'v2_gap_intervals_reserved2026.json')):
        published = json.loads((RUNS / artifact).read_text(encoding='utf-8'))['variables']
        for variable in VARIABLES:
            f = frames[(cohort, variable)].assign(climatology=lambda d: d.climatology_rebuilt)
            if cohort == 'Gap-2026':
                f = f[f.gap_length >= 6]          # the published run; part C added 1-3 h later
            units = clusters(f)
            for length, part in f.groupby('gap_length'):
                got = paired(part, 'climatology', 'linear', f'{variable}|climatology|{int(length)}', units)
                want = published[variable]['by_gap_length'][str(int(length))].get('climatology')
                checks[f'{cohort} {variable} {int(length)} h climatology-linear'] = compare(got, want, SAME_FOUR)
    # (b) the same on the tiled cohorts, probe x month (runs/v2_gap_long.json)
    long = json.loads((RUNS / 'v2_gap_long.json').read_text(encoding='utf-8'))
    for cohort in ('Gap-Long-Dev', 'Gap-Long-2026'):
        for variable in VARIABLES:
            f = frames[(cohort, variable)].assign(climatology=lambda d: d.climatology_rebuilt)
            for length, part in f.groupby('gap_length'):
                got = paired(part, 'climatology', 'linear',
                             f'long|{cohort}|{variable}|climatology|{int(length)}|conservative',
                             probe_month(part))
                want = (long[cohort][variable]['by_gap_length'][str(int(length))]['minus_linear']
                        .get('climatology', {}).get('conservative'))
                checks[f'{cohort} {variable} {int(length)} h climatology-linear'] = compare(got, want, SAME_FOUR)
    # (c) the Kalman bridge minus linear, by band and by length (rescored addendum-34 reading)
    exo = json.loads(EXOGENOUS.read_text(encoding='utf-8'))['results']
    for (cohort, variable, band), f in bands.items():
        entry = exo[cohort][variable][band]['arx_bridge']
        units = clusters(f)
        got = paired6(f, 'arx_bridge', 'linear', f'{cohort}|{variable}|{band}|arx_bridge|linear', units,
                      check=False)
        checks[f'{cohort} {variable} {band} arx_bridge-linear'] = compare(got, entry['minus_linear'], SAME_SIX)
        for length, part in f.groupby('gap_length'):
            u = probe_month(part) if band == LONG else units
            got = paired6(part, 'arx_bridge', 'linear',
                          f'{cohort}|{variable}|{band}|arx_bridge|linear|{int(length)}', u, check=False)
            want = entry['by_gap_length'][str(int(length))]['minus_linear']
            checks[f'{cohort} {variable} {int(length)} h arx_bridge-linear'] = compare(got, want, SAME_SIX)
    failed = sorted(k for k, v in checks.items() if not v['passed'])
    return {'checks': len(checks), 'failed': failed, 'details_of_failures': {k: checks[k] for k in failed},
            'passed': not failed}


# ------------------------------------------------------------------------------- contrasts
FAMILY = []


def floor_of(variable, reference):
    from src.units import factor
    if variable == 'oxygen':
        return 0.059 / factor(), 'oxygen 0.059 mg/l (umol/kg as stored)'
    if reference in OWN_RECORD:
        return 0.029, '0.029 degC quantisation (both arms read only the probe\'s own record)'
    return 0.20, '0.20 degC co-located disagreement (the reference reads other probes)'


def contrast(frame, reference, label, units, variable, family=True):
    """The arm minus `reference`, paired, six decimals, with its floor reading."""
    from scripts.v2_gap_short2026_rows import paired6
    from src.units import factor
    if reference not in frame or reference == ARM:
        return None
    e = paired6(frame, ARM, reference, f'a84|{label}|{reference}', units, check=False)
    if e is None:
        return None
    floor, name = floor_of(variable, reference)
    e.update({'label': f'a84|{label}|{reference}', 'floor': name,
              'gain_beyond_floor': bool(-e['delta'] > floor),
              'deficit_beyond_floor': bool(e['delta'] > floor),
              'interval_beyond_floor': bool(e['estimable'] and e['ci_high'] < -floor)})
    if variable == 'oxygen':
        e['mg_per_l'] = {k: (None if e.get(k) is None else r6(factor() * e[k]))
                         for k in ('delta', 'ci_low', 'ci_high', 'mae_arm', 'mae_reference')}
        if e.get('jackknife_t'):
            e['mg_per_l']['jackknife_t'] = [r6(factor() * x) for x in e['jackknife_t']]
    if family:
        FAMILY.append(e)
    return e


def evaluate(frame, references, label, units, variable):
    """One cell: errors, the arm against every reference, and the rows where C(t) is undefined."""
    from src.units import factor
    out = {'rows': int(len(frame)), 'episodes': int(frame.episode.nunique()),
           'fallback_rows': int((~frame.climatology_defined).sum()),
           'fallback_episodes': int(frame.loc[~frame.climatology_defined, 'episode'].nunique()),
           'mae': {a: r6(macro(frame, a)) for a in (ARM, *references) if a in frame},
           'minus': {}}
    if variable == 'oxygen':
        out['mae_mg_per_l'] = {a: (None if v is None else r6(factor() * v)) for a, v in out['mae'].items()}
    for ref in references:
        got = contrast(frame, ref, label, units, variable)
        if got is not None:
            out['minus'][ref] = got
    if out['fallback_rows']:
        defined = frame[frame.climatology_defined]
        out['defined_rows_only'] = {
            'rows': int(len(defined)), 'episodes': int(defined.episode.nunique()),
            'mae': {a: r6(macro(defined, a)) for a in (ARM, *references) if a in defined},
            'minus': {ref: e for ref in references
                      if (e := contrast(defined, ref, label + '|defined', units, variable)) is not None}}
    return out


def beats(e, robust=True):
    return bool(e and e.get('estimable') and e.get('excludes_zero') and e['delta'] < 0
                and (not robust or e.get('robust')))


def loses(e):
    return bool(e and e.get('estimable') and e.get('excludes_zero') and e['delta'] > 0)


def decide(cohort_cells, references, incumbents):
    """Chapter 4's rule over the two cohorts of a band (or of a length past a week)."""
    minus = [c['minus'] for c in cohort_cells]

    def both(ref, robust=True):
        return all(beats(m.get(ref), robust) for m in minus)

    present = [r for r in references if all(r in m for m in minus)]
    members = [a for a in incumbents if a != ARM and all(a in m for m in minus)]
    estimable = all(m.get(r, {}).get('estimable') for m in minus for r in members) if members else False
    return {'beats_robustly_in_both': [r for r in present if both(r)],
            'beats_in_both_percentile_only': [r for r in present if both(r, False) and not both(r)],
            'worse_in_both': [r for r in present if all(loses(m.get(r)) for m in minus)],
            'replaces_linear': both('linear'),
            'incumbents_compared': members, 'incumbents_estimable_in_both': bool(estimable),
            'replaces_the_incumbent': bool(members) and all(both(a) for a in members),
            'replaces_the_incumbent_percentile_only': bool(members) and all(both(a, False) for a in members)}


def no_neighbour(own, decision, dev_frame):
    """The no-neighbour rule with the arm added to the own-record class."""
    before = list(own['isolated_candidates_that_qualified'])
    after = before + ([ARM] if decision['replaces_linear'] else [])
    errors = {a: r6(macro(dev_frame, a)) for a in after}
    answer_before_recomputed = min(before, key=lambda a: errors[a]) if before else 'linear'
    answer = min(after, key=lambda a: errors[a]) if after else 'linear'
    return {'answer_before': own['isolated'], 'answer_before_recomputed': answer_before_recomputed,
            'qualified_before': before, 'qualified_after': after,
            'development_error_of_qualified': errors, 'answer_after': answer,
            'changed': answer != own['isolated']}


def incumbent_set(variable, band, length=None):
    """The declared incumbent (addenda 33 and 34) and every method the rescored addendum-33 decision
    names as having replaced it (past a week only where the jackknife confirms it at that length)."""
    import scripts.v2_gap_exogenous as E
    import scripts.v2_gap_variants as V
    dec = json.loads(VARIANTS.read_text(encoding='utf-8'))['decisions'][f'{variable} {band}']
    members = [V.INCUMBENT[(variable, band)], E.INCUMBENT[(variable, band)]]
    if band == LONG:
        members += [a for a, at in dec.get('replaced_at_robust', {}).items() if length in at]
    else:
        members += list(dec['replaced_by'])
    return list(dict.fromkeys(members))


def own_record_cell(variable, band, length=None):
    cells = json.loads(ISOLATED.read_text(encoding='utf-8'))['after_the_fact'][variable][band]['cells']
    return next(c for c in cells if c['length'] == length)


def references_for(variable, band, length=None):
    inc = incumbent_set(variable, band, length)
    own = own_record_cell(variable, band, length)
    return list(dict.fromkeys(['linear', 'climatology', 'arx_bridge', *inc, own['isolated']])), inc, own


def score(frames, bands):
    from scripts.v2_gap_intervals import clusters
    from scripts.v2_gap_short2026_score import whole_cohort_clusters
    whole = whole_cohort_clusters()
    cells, by_length, decisions = {}, {}, {}
    for variable in VARIABLES:
        for band, (dev, new) in PAIRS.items():
            if band != LONG:
                refs, inc, own = references_for(variable, band)
                entry = {'cohorts': {}, 'references': refs, 'incumbent_set': inc,
                         'own_record_answer': own['isolated']}
                for cohort in (dev, new):
                    f = bands[(cohort, variable, band)]
                    entry['cohorts'][cohort] = evaluate(f, refs, f'{cohort}|{variable}|{band}', clusters(f),
                                                        variable)
                    for length, part in f.groupby('gap_length'):
                        by_length.setdefault(f'{variable} {int(length)} h', {'band': band, 'cohorts': {}})[
                            'cohorts'][cohort] = evaluate(part, refs, f'{cohort}|{variable}|{int(length)}',
                                                          whole[(cohort, variable)], variable)
                entry['decision'] = decide(list(entry['cohorts'].values()), refs, inc)
                entry['decision']['no_neighbour'] = no_neighbour(own, entry['decision'],
                                                                 bands[(dev, variable, band)])
                cells[f'{variable} {band}'] = entry
                continue
            lengths = sorted(set(bands[(dev, variable, band)].gap_length)
                             & set(bands[(new, variable, band)].gap_length))
            for length in lengths:
                refs, inc, own = references_for(variable, band, int(length))
                entry = {'cohorts': {}, 'references': refs, 'incumbent_set': inc,
                         'own_record_answer': own['isolated'], 'unit': 'probe x calendar month'}
                for cohort in (dev, new):
                    f = bands[(cohort, variable, band)]
                    part = f[f.gap_length == length]
                    entry['cohorts'][cohort] = evaluate(part, refs, f'{cohort}|{variable}|{int(length)}',
                                                        probe_month(part), variable)
                entry['decision'] = decide(list(entry['cohorts'].values()), refs, inc)
                dev_part = bands[(dev, variable, band)]
                entry['decision']['no_neighbour'] = no_neighbour(own, entry['decision'],
                                                                 dev_part[dev_part.gap_length == length])
                cells[f'{variable} {int(length)} h'] = entry
    for key, entry in by_length.items():
        variable = key.split()[0]
        refs = cells[f'{variable} {entry["band"]}']['references']
        inc = cells[f'{variable} {entry["band"]}']['incumbent_set']
        if len(entry['cohorts']) == 2:
            entry['decision'] = decide(list(entry['cohorts'].values()), refs, inc)
    for key, entry in cells.items():
        d = entry['decision']
        decisions[key] = {k: d[k] for k in ('replaces_linear', 'replaces_the_incumbent',
                                            'beats_robustly_in_both', 'worse_in_both')}
        decisions[key]['no_neighbour_answer'] = [d['no_neighbour']['answer_before'],
                                                 d['no_neighbour']['answer_after']]
    return cells, by_length, decisions


def strict_descriptive(frames):
    """Gap-Dev's strict lengths past 72 h and both strict cohorts pooled over their lengths, against
    linear and the plain climatology on whole-cohort clusters, as the thesis's pooled table reads them:
    descriptions that decide nothing."""
    from scripts.v2_gap_short2026_score import whole_cohort_clusters
    whole = whole_cohort_clusters()
    out = {}
    for cohort in ('Gap-Dev', 'Gap-2026'):
        for variable in VARIABLES:
            f = frames[(cohort, variable)]
            units = whole[(cohort, variable)]
            block = {'pooled': evaluate(f, ['linear', 'climatology'], f'{cohort}|{variable}|pooled', units,
                                        variable)}
            longer = {str(int(L)): evaluate(part, ['linear', 'climatology'],
                                            f'{cohort}|{variable}|strict|{int(L)}', units, variable)
                      for L, part in f[f.gap_length > 72].groupby('gap_length')}
            if longer:
                block['strict_lengths_past_72_h'] = longer
            out.setdefault(cohort, {})[variable] = block
    return out


def multiplicity():
    from scripts.v2_fdr import step_up
    fam = [e for e in FAMILY if e.get('p') is not None]
    p = [e['p'] for e in fam]
    bh, by = step_up(p), step_up(p, dependence_robust=True)
    for e, a, b in zip(fam, bh, by):
        e['bh'], e['by'] = bool(a), bool(b)
    return {'intervals': len(fam), 'excluding_zero': int(sum(e['excludes_zero'] for e in fam)),
            'retained_bh': int(sum(bh)), 'retained_by': int(sum(by)), 'q': 0.05,
            'not_estimable': int(sum(1 for e in FAMILY if not e.get('estimable')))}


def predictions(cells):
    """The declared predictions Q1-Q5 against the cells (recorded in the artifact, never printed)."""
    def m(key, cohort, ref):
        return cells[key]['cohorts'][cohort]['minus'].get(ref)
    q1 = cells['oxygen 12-72']['decision']
    q1_held = bool(q1['replaces_linear'] and any(loses(m('oxygen 12-72', c, 'chronos_bidir'))
                                                for c in ('Gap-Dev', 'Gap-2026')))
    q2 = {c: m('temperature 1-6', c, 'linear')['delta'] for c in ('Gap-Dev', 'Gap-2026')}
    long_keys = [k for k in cells if k.endswith(' h')]
    q3 = {k: {c: m(k, c, 'climatology') for c in ('Gap-Long-Dev', 'Gap-Long-2026')} for k in long_keys}
    q3_cells = {k: [beats(e, robust=False) for e in v.values() if e and e.get('estimable')]
                for k, v in q3.items()}
    q4 = {k: cells[k]['decision']['replaces_linear'] for k in long_keys if int(k.split()[1]) >= 336}
    q5 = [m('oxygen 168 h', c, 'arx_bridge') for c in ('Gap-Long-Dev', 'Gap-Long-2026')]
    q5_both = bool(all(e and e.get('excludes_zero') for e in q5)
                   and len({np.sign(e['delta']) for e in q5}) == 1)
    return {
        'Q1': {'held': q1_held, 'replaces_linear': q1['replaces_linear']},
        'Q2': {'held': all(abs(d) < 0.029 for d in q2.values()), 'minus_linear': q2},
        'Q3': {'held': all(all(v) for v in q3_cells.values()),
               'beats_climatology_where_estimable': q3_cells},
        'Q4': {'held': not any(q4.values()), 'replaces_linear': q4},
        'Q5': {'held': not q5_both},
    }


# ------------------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gates-only', action='store_true')
    args = parser.parse_args()
    decl = (ROOT / DECLARATION).read_bytes()
    frames, g1, g3 = {}, {}, {}
    for cohort in COHORTS:
        for variable in VARIABLES:
            frames[(cohort, variable)], g1[f'{cohort} {variable}'], g3[f'{cohort} {variable}'] = \
                rebuild(cohort, variable)
            print(f'{cohort} {variable}: G1 {"pass" if g1[f"{cohort} {variable}"]["passed"] else "FAIL"} '
                  f'{g1[f"{cohort} {variable}"]}; G3 {"pass" if g3[f"{cohort} {variable}"]["passed"] else "FAIL"}')
    bands, g1_files = {}, {}
    for variable in VARIABLES:
        for band, pair in PAIRS.items():
            for cohort in pair:
                bands[(cohort, variable, band)], g1_files[f'{cohort} {variable} {band}'] = band_frame(
                    cohort, variable, band, frames[(cohort, variable)])
    gates = {'A84-G1_reconstruction': {'published_rows': g1, 'comparator_files': g1_files,
                                       'tolerance': TOLERANCE},
             'A84-G2_definition': 'tests/test_gap_climanchor.py',
             'A84-G3_no_masked_reading': g3}
    gates['A84-G1_reconstruction']['passed'] = bool(all(v['passed'] for v in g1.values())
                                                    and all(v['passed'] for v in g1_files.values()))
    gates['A84-G3_no_masked_reading']['passed'] = bool(all(v['passed'] for v in g3.values()))
    print('G1 comparator files:', {k: v['passed'] for k, v in g1_files.items()})
    if not (gates['A84-G1_reconstruction']['passed'] and gates['A84-G3_no_masked_reading']['passed']):
        print(json.dumps(gates, indent=1, default=str)[:6000])
        raise SystemExit('A84-G1 or A84-G3 failed: nothing is scored')
    gates['A84-G4_resampling_reproduction'] = gate_g4(frames, bands)
    print('G4', gates['A84-G4_resampling_reproduction']['passed'], gates['A84-G4_resampling_reproduction']['failed'][:20])
    if not gates['A84-G4_resampling_reproduction']['passed']:
        print(json.dumps(gates['A84-G4_resampling_reproduction']['details_of_failures'], indent=1)[:6000])
        raise SystemExit('A84-G4 failed: nothing is scored')
    gates['passed'] = True
    if args.gates_only:
        print('gates passed; --gates-only: nothing scored')
        return
    from src.units import factor
    cells, by_length, decisions = score(frames, bands)
    report = {
        'addendum': 84, 'declared_in': DECLARATION,
        'declaration_sha256': hashlib.sha256(decl).hexdigest(),
        'arm': ARM,
        'definition': 'C(t) + a_L + (a_R - a_L)(t - t_L)/(t_R - t_L), a = y - C at each edge; C the probe\'s '
                      'month-by-hour mean fitted outside the masked stretch; linear interpolation where '
                      'C(t) is undefined',
        'units': f'oxygen umol/kg as stored (mg_per_l beside, c = {factor()}); temperature degC',
        'floors': {'temperature_own_record': 0.029, 'temperature_against_network': 0.20,
                   'oxygen_mg_per_l': 0.059, 'oxygen_umol_kg': r6(0.059 / factor())},
        'cohorts': {k: {'run': v['run'], 'results': v['results'], 'period': v['period'],
                        **({'appended_1_3_h': 'runs/v2_addendum82/C/remote/20261012T075809Z_v2_add82C'}
                           if v['short'] else {})} for k, v in COHORTS.items()},
        'comparator_rows': 'runs/v2_gap_variants/, runs/v2_gap_exogenous/, runs/v2_gap_chronos_covariates/ '
                           '(Gap-2026 1-6 h: their rescored forms under runs/v2_addendum82/C/rescored/)',
        'reading': 'cells: per band up to 72 h on the band\'s overlap clusters, per length past a week on '
                   'probe x calendar month; by_length_up_to_72_h: per length on the whole cohort\'s '
                   'overlap clusters (secondary); a decision needs the interval excluding zero in both '
                   'cohorts, robustly (twenty units or the jackknife agreeing)',
        'gates': gates,
        'decisions': decisions,
        'cells': cells,
        'by_length_up_to_72_h': by_length,
        'strict_descriptive': strict_descriptive(frames),
    }
    report['multiplicity'] = multiplicity()
    report['registered_predictions'] = predictions(cells)
    report['per_row'] = 'runs/v2_gap_climanchor/<cohort>__<variable>.csv.gz'
    ROWS.mkdir(parents=True, exist_ok=True)
    for (cohort, variable), f in frames.items():
        (f[['episode', 'position', 'gap_length', ARM, 'climatology_defined']]
         .sort_values(KEYS).to_csv(ROWS / f'{cohort}__{variable}.csv.gz', index=False,
                                   compression=GZIP, lineterminator='\n'))
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    summarise(report)


def summarise(report):
    for key, entry in report['cells'].items():
        variable = key.split()[0]
        k = 0.032844 if variable == 'oxygen' else 1.0
        print(f'\n== {key}  incumbents {entry["incumbent_set"]}  no-neighbour {entry["own_record_answer"]}')
        for cohort, c in entry['cohorts'].items():
            parts = []
            for ref, e in c['minus'].items():
                mark = (('*' if e['excludes_zero'] else '') + ('' if e['robust'] or not e['excludes_zero'] else '§')
                        if e['estimable'] else ' n.e.')
                parts.append(f'{ref} {k * e["delta"]:+.4f}{mark}')
            print(f'   {cohort:<14} arm {k * c["mae"][ARM]:.4f} linear {k * c["mae"]["linear"]:.4f} '
                  f'fallback {c["fallback_rows"]}  ' + '  '.join(parts))
        print('   decision', report['decisions'][key])
    print('\nmultiplicity', report['multiplicity'])
    print('predictions', {k: v['held'] for k, v in report['registered_predictions'].items()})
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
