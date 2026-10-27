"""Addendum 33: the corrected Gaussian process on 2026, variations on the best fillers, ensembles.

For each of the four cohorts the addendum-33 arms are merged with the published predictions of the
incumbents, and the merge is refused unless both score exactly the same rows. Then:

  ensembles     ens_mean (chronos_bidir, gpr), ens_mean_c2 (chronos2_cov_bidir, gpr), and ens_stack,
                convex weights over linear, chronos_bidir, gpr and chronos2_cov_bidir chosen per
                variable and band by a 100-trial study on the calendar folds other than the one
                scored; for the 2026 cohorts, fitted on every fold of the matching development
                cohort and frozen
  twin rules    twin_else_gpr, and twin_else_best (gpr or ens_stack, whichever the development
                cohort prefers where there is no twin reading), temperature only. A twin rule
                returns the twin's reading only where the twin reports, and its declared method
                everywhere else: on a probe without a twin, and on a probe whose twin is dark at
                that hour or at the gap's edge. `nearest_probe`, which the rules first read, is
                persistence wherever the twin is dark.
  contrasts     every candidate against its band's incumbent, linear and both published fillers,
                paired; overlap clusters (addendum 30) for the 1-6 and 12-72 h bands. The long
                cohorts are tiled, so their windows are independent only within one length
                (addendum 32): they are contrasted per length on probe x calendar month. The pooled
                168-1440 h contrasts are kept in the artifact as descriptions and decide nothing.
  decision      a candidate replaces an incumbent only if it beats it with an interval excluding
                zero in the development and the 2026 cohort of that band; past a week, of that
                length. The registered predictions are read the same way: one cell per band up to
                72 h, one per length past a week.

Usage: python -m scripts.v2_gap_variants
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import arm_dirs, fix_neighbours, load_predictions, read_arm
from scripts.v2_gap_intervals import MIN_CLUSTERS, clusters, paired
from scripts.v2_gap_neighbours import with_twin
from src.tuning.protocol_v2 import run_study

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_variants.json'
#: The stacked weights are Optuna studies, and `run_study` resumes a study it finds rather than
#: refitting it. Its members include the Gaussian process, which addendum 59 corrected, so the
#: weights are refitted in a folder of their own; the studies fitted on the published process stay
#: in v2_gap_variants_studies.
STUDIES = RUNS / 'v2_gap_variants_studies_gp_by_pattern'
#: every candidate's per-row predictions, so figures can draw the ensembles without refitting them
ROWS = RUNS / 'v2_gap_variants'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
FLOOR = {'temperature': 0.20, 'oxygen': 1.8}

#: cohort -> (published run, its results dir, the addendum-33 run, band names it serves)
COHORTS = {
    'Gap-Dev': ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery',
                '20260922T204658Z_v2_gaprecovery', ('1-6', '12-72')),
    'Gap-2026': ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026',
                 '20260922T204715Z_v2_gaprecovery', ('1-6', '12-72')),
    'Gap-Long-Dev': ('20260922T195727Z_v2_gaprecovery', 'v2_gap_recovery_long',
                     '20260922T204732Z_v2_gaprecovery', ('168-1440',)),
    'Gap-Long-2026': ('20260922T195745Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026',
                      '20260922T204749Z_v2_gaprecovery', ('168-1440',)),
}
#: the record each cohort was built from, which is where a twin's reading is looked up
PERIOD = {'Gap-Dev': 'development', 'Gap-2026': 'reserved', 'Gap-Long-Dev': 'development',
          'Gap-Long-2026': 'reserved'}
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'),
         '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
#: bands whose cohorts are tiled, so that contrasts and decisions are per length only
PER_LENGTH = ('168-1440',)
BANDS = {'1-6': (1, 6), '12-72': (12, 72), '168-1440': (168, 1440)}
INCUMBENT = {('oxygen', '1-6'): 'linear', ('oxygen', '12-72'): 'chronos_bidir',
             ('oxygen', '168-1440'): 'chronos_bidir', ('temperature', '1-6'): 'linear',
             ('temperature', '12-72'): 'gpr', ('temperature', '168-1440'): 'gpr'}
PUBLISHED_ARMS = ('chronos_bidir', 'gpr')
NEW_ARMS = ('chronos2_bidir', 'chronos2_cov_bidir', 'gpr_k8')
STACK_MEMBERS = ('linear', 'chronos_bidir', 'gpr', 'chronos2_cov_bidir')


def load(cohort, variable):
    published, results, new, _ = COHORTS[cohort]
    frames = []
    for run, arms in ((published, PUBLISHED_ARMS), (new, NEW_ARMS + ('gpr',))):
        for arm in arms:
            path = RUNS / run / results / f'{variable}_{arm}' / 'predictions.csv.gz'
            if not path.exists():
                continue
            if run == published:
                frame = load_predictions(run, results, variable, arm)
            else:
                # an arm rerun since (the Gaussian process, addendum 59) is read from its rerun,
                # which must score exactly these rows
                frame = fix_neighbours(read_arm(run, results, variable, arm), run, variable)
            frames.append((arm, frame))
    base = None
    for arm, frame in frames:
        keys = frame[['episode', 'position']]
        if base is None:
            base = frame.drop(columns=[arm]).copy()
        elif not keys.reset_index(drop=True).equals(base[['episode', 'position']]
                                                     .reset_index(drop=True)):
            raise SystemExit(f'{cohort} {variable} {arm}: rows differ from the published episodes')
        base[arm] = frame[arm].to_numpy()
    if variable == 'temperature':
        with_twin(base, variable, PERIOD[cohort])
    return base


def twin_rules(f, fallback_best=None):
    """The twin's reading where it reports, the declared method everywhere else."""
    f['twin_rule'] = f.twin.where(f.twin_reports, f.chronos_bidir)
    f['twin_else_gpr'] = f.twin.where(f.twin_reports, f.gpr)
    if fallback_best is not None:
        f['twin_else_best'] = f.twin.where(f.twin_reports, f[fallback_best])


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def band_of(lengths):
    out = pd.Series('', index=lengths.index)
    for name, (lo, hi) in BANDS.items():
        out[(lengths >= lo) & (lengths <= hi)] = name
    return out


def weights_from(trial_or_params, members):
    raw = np.array([trial_or_params[f'w_{m}'] for m in members], float)
    return raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / len(raw))


def stack(frame, members, label):
    """Out-of-fold convex blend: weights for each fold chosen on the other folds."""
    cols = list(members)
    complete = frame.dropna(subset=cols + ['truth'])
    X, y, episodes = complete[cols].to_numpy(float), complete.truth.to_numpy(float), complete.episode
    out = pd.Series(np.nan, index=frame.index)
    chosen = {}

    def fit(rows, name):
        def objective(trial):
            w = weights_from({f'w_{m}': trial.suggest_float(f'w_{m}', 0.0, 1.0) for m in cols},
                             cols)
            err = np.abs(y[rows] - X[rows] @ w)
            return float(pd.Series(err).groupby(episodes.to_numpy()[rows]).mean().mean())
        selection = run_study(name, objective, STUDIES)
        return weights_from(selection['params'], cols), selection['completed_trials']

    for fold in sorted(complete.fold.unique()):
        train = (complete.fold != fold).to_numpy()
        w, trials = fit(train, f'{label}_f{int(fold)}')
        chosen[str(int(fold))] = {'weights': dict(zip(cols, np.round(w, 4))), 'trials': trials}
        test = (complete.fold == fold).to_numpy()
        out.loc[complete.index[test]] = X[test] @ w
    w_all, trials = fit(np.ones(len(complete), bool), f'{label}_all')
    return out, chosen, dict(zip(cols, np.round(w_all, 4))), w_all


def contrasts(frame, candidates, incumbent, label, long_cohort):
    """Band contrasts on overlap clusters, per-length ones on the cohort's own unit.

    On a long (tiled) cohort the band contrasts pool lengths whose windows overlap, so they are
    descriptions only; the per-length ones, on probe x calendar month, are what decisions and
    registered predictions read there. Per length every candidate is contrasted with the same
    references as the band. The seed label of each existing per-length contrast is unchanged.
    """
    cluster_of = clusters(frame)
    out = {}
    # The incumbent decides; linear and both published fillers are the references the registered
    # predictions are stated against, so every candidate is contrasted with all of them.
    references = list(dict.fromkeys((incumbent, 'linear', 'chronos_bidir', 'gpr')))
    for arm in candidates:
        entry = {'mae': macro(frame, arm)}
        for ref in references:
            if arm == ref:
                continue
            entry[f'minus_{ref}'] = paired(frame, arm, ref, f'{label}|{arm}|{ref}', cluster_of)
        if long_cohort:
            entry['band_contrasts'] = ('pooled across lengths of a tiled cohort on overlap '
                                       'clusters: descriptive only; see by_gap_length')
        by_length = {}
        for length, part in frame.groupby('gap_length'):
            if long_cohort:
                start = part.episode.str.rsplit('|', n=2).str[1].str[:7]
                units = dict(zip(part.episode, part.sensor + '|' + start))
            else:
                units = cluster_of
            row = {'mae': macro(part, arm)}
            if arm != incumbent:
                row['minus_incumbent'] = paired(part, arm, incumbent,
                                                f'{label}|{arm}|{int(length)}', units)
            if long_cohort:
                for ref in references:
                    if ref not in (arm, incumbent):
                        row[f'minus_{ref}'] = paired(part, arm, ref,
                                                     f'{label}|{arm}|{ref}|{int(length)}', units)
            by_length[str(int(length))] = row
        entry['by_gap_length'] = by_length
        out[arm] = entry
    return out


def contrast_of(res, cohort, variable, band, arm, ref, length=None):
    """One contrast: the band's, or with `length` the per-length one (the incumbent's is stored
    as `minus_incumbent`)."""
    entry = res[cohort][variable][band].get(arm, {})
    if length is None:
        return entry.get(f'minus_{ref}') or {}
    row = entry.get('by_gap_length', {}).get(str(length), {})
    key = 'minus_incumbent' if ref == INCUMBENT[(variable, band)] else f'minus_{ref}'
    return row.get(key) or {}


def cells_of(res):
    """(cohort, variable, band, length) cells: the band up to 72 h, each length past a week."""
    out = []
    for co in res:
        for v in res[co]:
            for b in res[co][v]:
                if b in PER_LENGTH:
                    lengths = next(iter(res[co][v][b].values()))['by_gap_length']
                    out += [(co, v, b, int(L)) for L in lengths]
                else:
                    out.append((co, v, b, None))
    return out


def main():
    STUDIES.mkdir(parents=True, exist_ok=True)
    frames = {(c, v): load(c, v) for c in COHORTS for v in ('oxygen', 'temperature')}
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum33.json',
              'cohorts': {c: {'published': p, 'results': r, 'addendum33_run': n}
                          for c, (p, r, n, _) in COHORTS.items()},
              'arm_sources': arm_sources(),
              'stack_studies': STUDIES.relative_to(RUNS).as_posix(),
              'incumbents': {f'{v} {b}': a for (v, b), a in INCUMBENT.items()},
              'stack_weights': {}, 'results': {}, 'decisions': {}}

    # Ensembles and twin rules, built per band, the stacked weights nested within development.
    for variable in ('oxygen', 'temperature'):
        for band, (dev, confirm) in PAIRS.items():
            lo, hi = BANDS[band]
            parts = {}
            for cohort in (dev, confirm):
                f = frames[(cohort, variable)]
                parts[cohort] = f[(f.gap_length >= lo) & (f.gap_length <= hi)].copy()
            if not len(parts[confirm]):
                continue
            label = f'stack_{variable}_{band}'
            members = [m for m in STACK_MEMBERS if m in parts[dev] and m in parts[confirm]]
            oof, per_fold, frozen, w_all = stack(parts[dev], members, label)
            parts[dev]['ens_stack'] = oof
            c = parts[confirm].dropna(subset=members)
            parts[confirm]['ens_stack'] = np.nan
            parts[confirm].loc[c.index, 'ens_stack'] = c[members].to_numpy(float) @ w_all
            report['stack_weights'][f'{variable} {band}'] = {'members': members,
                                                            'per_fold': per_fold,
                                                            'frozen_for_2026': frozen}
            for cohort, f in parts.items():
                f['ens_mean'] = (f.chronos_bidir + f.gpr) / 2
                if 'chronos2_cov_bidir' in f:
                    f['ens_mean_c2'] = (f.chronos2_cov_bidir + f.gpr) / 2
            if variable == 'temperature':
                # The better fallback, chosen on development only, on the rows it would fill:
                # those with no twin reading.
                choice = min(('gpr', 'ens_stack'),
                             key=lambda a: macro(parts[dev][~parts[dev].twin_reports], a) or 1e9)
                for f in parts.values():
                    twin_rules(f, choice)
                report['stack_weights'][f'{variable} {band}']['twin_else_best_fallback'] = choice
            candidates = [a for a in ('linear', 'chronos_bidir', 'gpr', 'chronos2_bidir',
                                      'chronos2_cov_bidir', 'gpr_k8', 'ens_mean', 'ens_mean_c2',
                                      'ens_stack', 'twin_rule', 'twin_else_gpr', 'twin_else_best')
                          if a in parts[dev] and a in parts[confirm]]
            incumbent = INCUMBENT[(variable, band)]
            for cohort, f in parts.items():
                report['results'].setdefault(cohort, {}).setdefault(variable, {})[band] = contrasts(
                    f, candidates, incumbent, f'{cohort}|{variable}|{band}',
                    cohort.startswith('Gap-Long'))
                keep = ['episode', 'sensor', 'has_twin', 'gap_length', 'position', 'fold', 'time',
                        'truth'] + (['twin_reports'] if variable == 'temperature' else []) \
                    + candidates
                ROWS.mkdir(parents=True, exist_ok=True)
                f[keep].to_csv(ROWS / f'{cohort}__{variable}__{band}.csv.gz', index=False,
                               compression={'method': 'gzip', 'mtime': 0})
            # Decision: beats the incumbent with an interval excluding zero in both cohorts; past
            # a week, at that length.
            res = report['results']

            def beats_in_both(arm, length=None):
                return all(contrast_of(res, c, variable, band, arm, incumbent, length)
                           .get('excludes_zero') and
                           contrast_of(res, c, variable, band, arm, incumbent, length)['delta'] < 0
                           for c in (dev, confirm))

            others = [a for a in candidates if a != incumbent]
            if band in PER_LENGTH:
                lengths = sorted({int(L) for c in (dev, confirm)
                                  for L in res[c][variable][band][incumbent]['by_gap_length']})
                both = [L for L in lengths
                        if all(str(L) in res[c][variable][band][incumbent]['by_gap_length']
                               for c in (dev, confirm))]
                at = {L: [a for a in others if beats_in_both(a, L)] for L in both}
                estimable = [L for L in both
                             if all(contrast_of(res, c, variable, band, others[0], incumbent, L)
                                    .get('estimable') for c in (dev, confirm))]
                def robust(arm, L):
                    """Chapter 4: below about twenty units the jackknife must agree."""
                    return all(e['clusters'] >= 20 or e.get('jackknife_excludes_zero')
                               for e in (contrast_of(res, c, variable, band, arm, incumbent, L)
                                         for c in (dev, confirm)))

                replaced_at = {a: [L for L in both if a in at[L]] for a in others
                               if any(a in at[L] for L in both)}
                report['decisions'][f'{variable} {band}'] = {
                    'incumbent': incumbent,
                    'rule': 'per length: beats the incumbent with a probe-month interval excluding '
                            'zero in both cohorts at that length',
                    'replaced_by': [a for a in others if estimable
                                    and all(a in at[L] for L in estimable)],
                    'replaced_at': replaced_at,
                    'replaced_at_robust': {a: [L for L in Ls if robust(a, L)]
                                           for a, Ls in replaced_at.items()},
                    'lengths_estimable_in_both': estimable}
            else:
                report['decisions'][f'{variable} {band}'] = {
                    'incumbent': incumbent, 'replaced_by': [a for a in others if beats_in_both(a)]}
            if variable == 'oxygen' and band == '168-1440':
                exo = {}
                for cohort, f in parts.items():
                    e = f[(f.sensor == EXO2) & (f.gap_length == 720)]
                    exo[cohort] = {a: macro(e, a) for a in candidates}
                report['exo2_720h'] = exo
    report['registered_predictions'] = verdicts(report)
    OUT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    summarise(report)


def arm_sources():
    """cohort -> variable -> arm -> the run folder its predictions are read from."""
    out = {}
    for cohort, (published, results, new, _) in COHORTS.items():
        for run, arms in ((published, PUBLISHED_ARMS), (new, NEW_ARMS + ('gpr',))):
            dirs = arm_dirs(run, results)
            for (variable, arm), folder in dirs.items():
                if arm in arms:
                    out.setdefault(cohort, {}).setdefault(variable, {}).setdefault(
                        arm, folder.relative_to(RUNS).as_posix())
    return out


def verdicts(r):
    """The six registered predictions. A cell is a band up to 72 h and a length past a week,
    because the long cohorts' windows are independent only within one length."""
    res = r['results']

    def c(cohort, variable, band, arm, ref, length=None):
        return contrast_of(res, cohort, variable, band, arm, ref, length)

    def beats(e):
        return bool(e.get('excludes_zero') and e['delta'] < 0)

    def name(co, v, b, L):
        return f'{co} {v} {b}' if L is None else f'{co} {v} {L} h'

    twin_vs_rule = paired_twin(r)
    cells = cells_of(res)
    # (2) Chronos-2 univariate never separates from Chronos-Bolt.
    differs = [{'cell': name(co, v, b, L), 'delta': c(co, v, b, 'chronos2_bidir',
                                                      'chronos_bidir', L)['delta']}
               for co, v, b, L in cells
               if c(co, v, b, 'chronos2_bidir', 'chronos_bidir', L).get('excludes_zero')]
    # (3) With covariates it beats Chronos-Bolt on temperature from 12 h, is level with the
    # Gaussian process there, and does not beat Chronos-Bolt on oxygen at 12-72 h.
    temp_cells = [(co, b, L) for co, v, b, L in cells if v == 'temperature' and b != '1-6']
    not_beating = [name(co, 'temperature', b, L) for co, b, L in temp_cells
                   if not beats(c(co, 'temperature', b, 'chronos2_cov_bidir', 'chronos_bidir', L))]
    not_level = [{'cell': name(co, 'temperature', b, L),
                  'delta': c(co, 'temperature', b, 'chronos2_cov_bidir', 'gpr', L)['delta']}
                 for co, b, L in temp_cells
                 if c(co, 'temperature', b, 'chronos2_cov_bidir', 'gpr', L).get('excludes_zero')]
    oxy_beats = [co for co in ('Gap-Dev', 'Gap-2026')
                 if beats(c(co, 'oxygen', '12-72', 'chronos2_cov_bidir', 'chronos_bidir'))]
    # (4) ens_mean beats both components on development temperature at 12-720 h, by < 0.05 C over
    # the better one, and does not beat Chronos-Bolt on oxygen.
    four = {}
    for co, b, L in [('Gap-Dev', '12-72', None)] + [('Gap-Long-Dev', '168-1440', L)
                                                     for L in (168, 336, 504, 720)]:
        vs_c = c(co, 'temperature', b, 'ens_mean', 'chronos_bidir', L)
        vs_g = c(co, 'temperature', b, 'ens_mean', 'gpr', L)
        four[name(co, 'temperature', b, L)] = {
            'beats_chronos': beats(vs_c), 'beats_gpr': beats(vs_g),
            'margin_over_better': min(-vs_c.get('delta', 0), -vs_g.get('delta', 0))}
    four_oxy = [name(co, v, b, L) for co, v, b, L in cells
                if v == 'oxygen' and beats(c(co, v, b, 'ens_mean', 'chronos_bidir', L))]
    p4 = all(e['beats_chronos'] and e['beats_gpr'] and e['margin_over_better'] < 0.05
             for e in four.values()) and not four_oxy
    # (5) No stacked ensemble beats its incumbent by more than 5 % of the incumbent's error.
    over = []
    for co, v, b, L in cells:
        inc = INCUMBENT[(v, b)]
        e = c(co, v, b, 'ens_stack', inc, L)
        entry = res[co][v][b].get(inc, {})
        base = entry.get('mae') if L is None else entry['by_gap_length'][str(L)]['mae']
        if e and base and beats(e) and -e['delta'] > 0.05 * base:
            over.append({'cell': name(co, v, b, L), 'gain': -e['delta'],
                         'share': round(-e['delta'] / base, 3)})
    exo = r.get('exo2_720h', {})
    best = min((v, a, co) for co, m in exo.items() for a, v in m.items() if v is not None)
    return {'1': {'gpr_minus_linear_2026_12_72': c('Gap-2026', 'temperature', '12-72', 'gpr',
                                                     'linear'),
                  'twin_else_gpr_minus_twin_rule_2026_12_72': twin_vs_rule,
                  'held': bool(beats(c('Gap-2026', 'temperature', '12-72', 'gpr', 'linear'))
                               and beats(twin_vs_rule))},
            '2': {'held': not differs, 'cells': len(cells), 'cells_where_it_differs': differs},
            '3': {'held': not (not_beating or not_level or oxy_beats),
                  'temperature_cells': len(temp_cells),
                  'temperature_cells_not_beating_chronos': not_beating,
                  'temperature_cells_not_level_with_gpr': not_level,
                  'oxygen_12_72_cohorts_where_it_beats_chronos': oxy_beats},
            '4': {'held': bool(p4), 'temperature': four, 'oxygen_cells_beating_chronos': four_oxy},
            '5': {'held': not over, 'cells': len(cells), 'cells_over_5_percent': over},
            '6': {'held': bool(best[0] > 5.0), 'best_at_720h_on_exo2': {'mae': best[0],
                                                                        'arm': best[1],
                                                                        'cohort': best[2]}}}


def paired_twin(r):
    """twin_else_gpr minus the addendum-31 twin rule on Gap-2026 at 12-72 h, computed directly."""
    f = load('Gap-2026', 'temperature')
    f = f[(f.gap_length >= 12) & (f.gap_length <= 72)].copy()
    twin_rules(f)
    return paired(f, 'twin_else_gpr', 'twin_rule', 'twin_else_gpr|2026|12-72', clusters(f))


def summarise(r):
    for cohort, by_var in r['results'].items():
        for variable, by_band in by_var.items():
            for band, arms in by_band.items():
                inc = INCUMBENT[(variable, band)]
                print(f'\n== {cohort} {variable} {band}  (incumbent {inc}, floor {FLOOR[variable]})')
                for arm, e in sorted(arms.items(), key=lambda kv: kv[1]['mae'] or 1e9):
                    d = e.get(f'minus_{inc}')
                    mark = ''
                    if d:
                        mark = (f"{d['delta']:+.4f}" + ('*' if d['excludes_zero'] else '')
                                + ('' if d['estimable'] else ' n.e.'))
                    print(f'   {arm:<20} {e["mae"]:.4f}   vs incumbent {mark}')
    print('\ndecisions:', json.dumps(r['decisions'], indent=1))
    print('predictions:', {k: v.get('held') for k, v in r['registered_predictions'].items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
