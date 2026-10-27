"""Addendum 35: Chronos with exogenous covariates, and Chronos on a covariate regressor's residual.

For each of the four cohorts the addendum-35 arms are merged with the addendum-33 per-row
predictions of that cohort and band, which carry every plain model these arms extend, and the merge
is refused unless both score exactly the same rows with the same truth. Each arm is contrasted,
paired, with its band's incumbent, with linear, and with its own plain model, which is the only
contrast that credits the covariates or the residual scheme:

  chronos2_exo_bidir        against chronos2_bidir       (exogenous and calendar covariates)
  chronos2_all_bidir        against chronos2_cov_bidir   (the same, on top of the neighbours)
  bolt_resid_bidir          against chronos_bidir        (the covariate-regressor residual)
  chronos2_cov_resid_bidir  against chronos2_cov_bidir   (the same, for Chronos-2 with neighbours)

Decision rule as addenda 33 and 34. Past a week the cohorts are tiled, so contrasts, decisions and
registered predictions are read per length on probe x calendar month, the band contrasts pooled
across lengths kept as descriptions only, and the jackknife reading kept beside the declared one,
exactly as in scripts/v2_gap_exogenous.py.

Usage: python -m scripts.v2_gap_chronos_covariates
"""
import json
from pathlib import Path

import pandas as pd

from scripts.v2_gap_exogenous import contrast_at, firm, name_of, per_length_incumbents
from scripts.v2_gap_intervals import clusters, paired

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_chronos_covariates.json'
ROWS = RUNS / 'v2_gap_chronos_covariates'
VARIANTS = RUNS / 'v2_gap_variants'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
FLOOR = {'temperature': 0.20, 'oxygen': 1.8}

#: cohort -> (addendum-35 run, its results dir)
COHORTS = {
    'Gap-Dev': ('20260922T223901Z_v2_gaprecovery', 'v2_gap_recovery'),
    'Gap-2026': ('20260922T224158Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026'),
    'Gap-Long-Dev': ('20260922T223918Z_v2_gaprecovery', 'v2_gap_recovery_long'),
    'Gap-Long-2026': ('20260922T224215Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026'),
}
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'),
         '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
BANDS = {'1-6': (1, 6), '12-72': (12, 72), '168-1440': (168, 1440)}
INCUMBENT = {('oxygen', '1-6'): 'linear', ('oxygen', '12-72'): 'chronos_bidir',
             ('oxygen', '168-1440'): 'chronos_bidir'}
INCUMBENT.update({('temperature', b): 'ens_stack' for b in BANDS})
PLAIN = {'chronos2_exo_bidir': 'chronos2_bidir', 'chronos2_all_bidir': 'chronos2_cov_bidir',
         'bolt_resid_bidir': 'chronos_bidir', 'chronos2_cov_resid_bidir': 'chronos2_cov_bidir',
         # amendment 1: the linear covariate regressor
         'bolt_resid_lin_bidir': 'chronos_bidir',
         'chronos2_cov_resid_lin_bidir': 'chronos2_cov_bidir'}
#: amendment 1's arms are also contrasted with the tree version of the same scheme
TREE = {'bolt_resid_lin_bidir': 'bolt_resid_bidir',
        'chronos2_cov_resid_lin_bidir': 'chronos2_cov_resid_bidir'}
NEW = tuple(PLAIN)
REFERENCES = ('linear', 'chronos_bidir', 'chronos2_bidir', 'chronos2_cov_bidir', 'gpr',
              'ens_stack', 'ens_mean_c2')
#: bands whose cohorts are tiled, so that contrasts, decisions and predictions are per length only
PER_LENGTH = ('168-1440',)


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


#: Jobs that failed for want of GPU memory were relaunched as runs of their own; each arm is read
#: from the one run of its cohort that completed it.
RELAUNCHES = {'Gap-Dev': ['20260922T225335Z_v2_gaprecovery'],
              'Gap-Long-Dev': ['20260922T225350Z_v2_gaprecovery'],
              'Gap-2026': ['20260922T225405Z_v2_gaprecovery', '20260922T225420Z_v2_gaprecovery']}


def arm_folder(cohort, variable, arm):
    run, results = COHORTS[cohort]
    done = [RUNS / r / results / f'{variable}_{arm}' for r in [run] + RELAUNCHES.get(cohort, [])
            if (RUNS / r / results / f'{variable}_{arm}' / 'report.json').exists()]
    if len(done) != 1:
        raise SystemExit(f'{cohort} {variable} {arm}: completed in {len(done)} runs, expected one')
    return done[0]


def load(cohort, variable, band):
    """The addendum-33 rows of a band with every addendum-35 arm merged on, refusing a mismatch."""
    base = pd.read_csv(VARIANTS / f'{cohort}__{variable}__{band}.csv.gz')
    lo, hi = BANDS[band]
    reports = {}
    for arm in NEW:
        folder = arm_folder(cohort, variable, arm)
        reports[arm] = json.loads((folder / 'report.json').read_text())
        frame = pd.read_csv(folder / 'predictions.csv.gz')
        frame = frame[(frame.gap_length >= lo) & (frame.gap_length <= hi)]
        if len(frame) != len(base) or set(zip(frame.episode, frame.position)) != set(
                zip(base.episode, base.position)):
            raise SystemExit(f'{cohort} {variable} {band} {arm}: rows differ from addendum 33')
        merged = base[['episode', 'position', 'truth']].merge(
            frame[['episode', 'position', 'truth', arm]], on=['episode', 'position'],
            suffixes=('', '_new'), validate='one_to_one')
        if float((merged.truth - merged.truth_new).abs().max()) > 1e-9:
            raise SystemExit(f'{cohort} {variable} {band} {arm}: truth differs from addendum 33')
        base[arm] = merged[arm].to_numpy()
    return base, reports


def contrasts(frame, incumbent, label, long_cohort):
    """Band contrasts on overlap clusters, per-length ones on the cohort's own unit; on a long
    (tiled) cohort the band contrasts are descriptions only, and each arm is also contrasted per
    length with linear, Chronos-2 with neighbours and, on temperature, the Gaussian process.
    Existing contrasts keep their seed labels, so no value computed before moves."""
    cluster_of = clusters(frame)
    out = {}
    for arm in NEW + REFERENCES:
        if arm not in frame:
            continue
        entry = {'mae': macro(frame, arm)}
        if arm in NEW:
            refs = (incumbent, 'linear', PLAIN[arm], 'chronos2_cov_bidir') + (
                (TREE[arm],) if arm in TREE else ())
            for ref in dict.fromkeys(refs):
                if ref in frame and ref != arm:
                    entry[f'minus_{ref}'] = paired(frame, arm, ref, f'{label}|{arm}|{ref}',
                                                   cluster_of)
            by_length = {}
            for length, part in frame.groupby('gap_length'):
                if long_cohort:
                    start = part.episode.str.rsplit('|', n=2).str[1].str[:7]
                    units = dict(zip(part.episode, part.sensor + '|' + start))
                else:
                    units = cluster_of
                row = {'mae': macro(part, arm), PLAIN[arm]: macro(part, PLAIN[arm]),
                       'linear': macro(part, 'linear')}
                for ref in dict.fromkeys((incumbent, PLAIN[arm]) + ((TREE[arm],) if arm in TREE
                                                                    else ())):
                    row[f'minus_{ref}'] = paired(part, arm, ref,
                                                 f'{label}|{arm}|{ref}|{int(length)}', units)
                if long_cohort:
                    more = ['linear', 'chronos2_cov_bidir'] + (['gpr'] if incumbent == 'ens_stack'
                                                               else [])
                    for ref in more:
                        if ref in part and ref != arm and f'minus_{ref}' not in row:
                            row[f'minus_{ref}'] = paired(part, arm, ref,
                                                         f'{label}|{arm}|{ref}|{int(length)}', units)
                by_length[str(int(length))] = row
            entry['by_gap_length'] = by_length
            if long_cohort:
                entry['band_contrasts'] = ('pooled across lengths of a tiled cohort on overlap '
                                           'clusters: descriptive only; see by_gap_length')
        elif long_cohort:
            entry['by_gap_length'] = {str(int(length)): {'mae': macro(part, arm)}
                                      for length, part in frame.groupby('gap_length')}
        out[arm] = entry
    return out


def long_lengths(res, cohort, variable, band):
    return sorted(int(L) for L in res[cohort][variable][band][NEW[0]]['by_gap_length'])


def decide_per_length(res, variable, band, dev, confirm):
    """The declared rule applied at each length past a week (as scripts/v2_gap_exogenous.py)."""
    lengths = [L for L in long_lengths(res, dev, variable, band)
               if L in long_lengths(res, confirm, variable, band)]
    must = per_length_incumbents(variable, band, lengths)

    def cells(arm, L):
        return [contrast_at(res, c, variable, band, arm, ref, L) for c in (dev, confirm)
                for ref in must[L]]

    estimable = [L for L in lengths if all(e.get('estimable') for e in cells(NEW[0], L))]
    at, robust = {}, {}
    for arm in NEW:
        for L in estimable:
            got = cells(arm, L)
            if all(e.get('excludes_zero') and e['delta'] < 0 for e in got):
                at.setdefault(arm, []).append(L)
                if all(firm(e) for e in got):
                    robust.setdefault(arm, []).append(L)
    return {'incumbent': INCUMBENT[(variable, band)],
            'rule': 'per length: beats every method in incumbents_at_length with a probe-month '
                    'interval excluding zero in both cohorts at that length',
            'incumbents_at_length': {str(L): m for L, m in must.items()},
            'replaced_by': [a for a in NEW if estimable and at.get(a) == estimable],
            'replaced_at': at,
            'replaced_at_robust': {a: robust.get(a, []) for a in at},
            'lengths_estimable_in_both': estimable}


def main():
    ROWS.mkdir(parents=True, exist_ok=True)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum35.json',
              'cohorts': {c: {'run': r, 'results': d, 'relaunches': RELAUNCHES.get(c, [])}
                          for c, (r, d) in COHORTS.items()},
              'incumbents': {f'{v} {b}': a for (v, b), a in INCUMBENT.items()},
              'plain_models': PLAIN, 'results': {}, 'decisions': {}, 'regressor': {}}
    for variable in ('oxygen', 'temperature'):
        for band, (dev, confirm) in PAIRS.items():
            incumbent = INCUMBENT[(variable, band)]
            for cohort in (dev, confirm):
                frame, reports = load(cohort, variable, band)
                report['results'].setdefault(cohort, {}).setdefault(variable, {})[band] = \
                    contrasts(frame, incumbent, f'{cohort}|{variable}|{band}',
                              cohort.startswith('Gap-Long'))
                report['regressor'].setdefault(cohort, {})[variable] = {
                    arm: [{'fold': s['fold'], 'trials': s['selection'].get('completed_trials'),
                           'inner_mae': s['selection'].get('best_value'),
                           'params': s['selection'].get('params'),
                           'unseen_probes': s['selection'].get('unseen_probes')}
                          for s in reports[arm]['selections']]
                    for arm in A35_RESIDUAL + tuple(TREE)}
                frame.to_csv(ROWS / f'{cohort}__{variable}__{band}.csv.gz', index=False,
                             compression={'method': 'gzip', 'mtime': 0})
                if variable == 'oxygen' and band == '168-1440':
                    e = frame[(frame.sensor == EXO2) & (frame.gap_length == 720)]
                    report.setdefault('exo2_720h', {})[cohort] = {
                        a: macro(e, a) for a in NEW + REFERENCES if a in e}
            if band in PER_LENGTH:
                report['decisions'][f'{variable} {band}'] = decide_per_length(
                    report['results'], variable, band, dev, confirm)
                continue
            won = []
            for arm in NEW:
                ok = all((report['results'][c][variable][band][arm].get(f'minus_{incumbent}')
                          or {}).get('excludes_zero')
                         and report['results'][c][variable][band][arm][f'minus_{incumbent}']
                         ['delta'] < 0 for c in (dev, confirm))
                if ok:
                    won.append(arm)
            report['decisions'][f'{variable} {band}'] = {'incumbent': incumbent,
                                                        'replaced_by': won}
    report['registered_predictions'] = verdicts(report)
    report['registered_predictions_reading'] = (
        'a cell is a band up to 72 h and a length past a week (the long cohorts are tiled); '
        '`held` is the declared percentile reading, `held_jackknife` also asks the jackknife to '
        'agree below twenty units (Chapter 4)')
    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))  # LF on every host
    summarise(report)


def cells_of(res):
    """(cohort, variable, band, length) cells: the band up to 72 h, each length past a week."""
    out = []
    for co in res:
        for v in res[co]:
            for b in res[co][v]:
                if b in PER_LENGTH:
                    out += [(co, v, b, L) for L in long_lengths(res, co, v, b)]
                else:
                    out.append((co, v, b, None))
    return out


def verdicts(r):
    """The ten registered predictions, each cell a band up to 72 h and a length past a week.

    `held` reads the declared percentile interval; `held_jackknife` also asks the jackknife to agree
    below twenty units (Chapter 4).
    """
    declared = evaluate(r, robust=False)
    jack = evaluate(r, robust=True)
    for key, got in declared.items():
        got['held_jackknife'] = jack[key]['held']
        for field in ('cells', 'cells_where_it_helps_in_both', 'cells_where_it_holds', 'replaced'):
            if field in jack[key] and jack[key][field] != got.get(field):
                got[f'{field}_jackknife'] = jack[key][field]
    return declared


def evaluate(r, robust):
    res = r['results']

    def c(cohort, variable, band, arm, ref, length=None):
        return contrast_at(res, cohort, variable, band, arm, ref, length)

    def ok(e):
        return not robust or firm(e)

    def beats(e, by=0.0):
        return bool(e and e.get('excludes_zero') and e['delta'] < 0 and -e['delta'] >= by
                    and ok(e))

    def loses(e):
        return bool(e and e.get('excludes_zero') and e['delta'] > 0 and ok(e))

    def held_where(contrasts, extra=lambda e: True):
        usable = {k: e for k, e in contrasts.items() if e.get('estimable')}
        return {'held': bool(usable) and all(beats(e) and extra(e) for e in usable.values()),
                'cells_where_it_holds': [k for k, e in usable.items() if beats(e) and extra(e)],
                'cells_not_estimable': [k for k, e in contrasts.items() if not e.get('estimable')]}

    cells = cells_of(res)
    # pairs of cells read together: a band in both of its cohorts, or one length in both long ones
    pairs = []
    for b, (dev, new) in PAIRS.items():
        if b in PER_LENGTH:
            for v in ('oxygen', 'temperature'):
                for L in long_lengths(res, dev, v, b):
                    if L in long_lengths(res, new, v, b):
                        pairs.append((v, b, dev, new, L))
        else:
            pairs += [(v, b, dev, new, None) for v in ('oxygen', 'temperature')]
    long_dev = long_lengths(res, 'Gap-Long-Dev', 'oxygen', '168-1440')
    # (1) exogenous and calendar covariates never help Chronos-2 in both cohorts of a band
    both = [f'{v} {b if L is None else f"{L} h"} {a}' for v, b, dev, new, L in pairs
            for a in A35_COVARIATE if beats(c(dev, v, b, a, PLAIN[a], L))
            and beats(c(new, v, b, a, PLAIN[a], L))]
    # (2) the residual scheme helps long oxygen gaps on development, by less than the floor, at
    # every length of Gap-Long-Dev
    two = {f'{a} {L} h': c('Gap-Long-Dev', 'oxygen', '168-1440', a, PLAIN[a], L)
           for a in A35_RESIDUAL for L in long_dev}
    # (3) it does not help at 1-6 h on development
    three = [f'{v} {a}' for v in ('oxygen', 'temperature') for a in A35_RESIDUAL
             if beats(c('Gap-Dev', v, '1-6', a, PLAIN[a]))]
    # (4) no new arm beats the ensemble on temperature by 0.02 C
    four = [f'{name_of(co, "temperature", b, L)} {a}' for co, v, b, L in cells if v == 'temperature'
            for a in A35_COVARIATE + A35_RESIDUAL
            if beats(c(co, 'temperature', b, a, 'ens_stack', L), 0.02)]
    # (5) nothing declared replaces an incumbent: past a week, at any length
    declared = A35_COVARIATE + A35_RESIDUAL
    five = replaced(r, declared, robust)
    # (6) nothing declared reaches 5 umol/l on Baia EXO2 at 720 h
    exo = r.get('exo2_720h', {})
    best = min(((m.get(a), a, co) for co, m in exo.items() for a in declared
                if m.get(a) is not None), default=(None, None, None))
    # amendment 1
    lin = tuple(TREE)
    l1 = {f'{v} {a}': c('Gap-Dev', v, '1-6', a, TREE[a]) for v in ('oxygen', 'temperature')
          for a in lin}
    l2 = [f'{v} {a}' for v in ('oxygen', 'temperature') for a in lin
          if loses(c('Gap-Dev', v, '1-6', a, PLAIN[a]))]
    l3 = {f'{a} {L} h': c('Gap-Long-Dev', 'oxygen', '168-1440', a, PLAIN[a], L)
          for a in lin for L in long_dev}
    l4 = replaced(r, lin, robust)
    amendment = {'L1': {'held': all(beats(e) for e in l1.values()), 'contrasts': l1},
                 'L2': {'held': not l2, 'cells': l2},
                 'L3': {**held_where(l3), 'contrasts': l3},
                 'L4': {'held': not l4, 'replaced': l4}}
    return {**amendment,
            '1': {'held': not both, 'cells_where_it_helps_in_both': both},
            '2': {**held_where(two, lambda e: -e['delta'] < FLOOR['oxygen']), 'contrasts': two},
            '3': {'held': not three, 'cells': three},
            '4': {'held': not four, 'cells': four},
            '5': {'held': not five, 'replaced': five},
            '6': {'held': bool(best[0] is None or best[0] > 5.0),
                  'best_at_720h_on_exo2': {'mae': best[0], 'arm': best[1], 'cohort': best[2]}}}


def replaced(r, arms, robust):
    """{decision: arms that replace its incumbent}: the band's `replaced_by` up to 72 h, and past a
    week the lengths at which each arm replaces it (`replaced_at`, or `replaced_at_robust`)."""
    out = {}
    for k, v in r['decisions'].items():
        if 'replaced_at' in v:
            at = v['replaced_at_robust' if robust else 'replaced_at']
            got = {a: Ls for a, Ls in at.items() if a in arms and Ls}
        else:
            got = [a for a in v['replaced_by'] if a in arms]
        if got:
            out[k] = got
    return out


A35_COVARIATE = ('chronos2_exo_bidir', 'chronos2_all_bidir')
A35_RESIDUAL = ('bolt_resid_bidir', 'chronos2_cov_resid_bidir')


def summarise(r):
    for cohort, by_var in r['results'].items():
        for variable, by_band in by_var.items():
            for band, arms in by_band.items():
                inc = INCUMBENT[(variable, band)]
                print(f'\n== {cohort} {variable} {band}  (incumbent {inc}, floor {FLOOR[variable]})')
                for arm, e in sorted(arms.items(), key=lambda kv: kv[1]['mae'] or 1e9):
                    marks = []
                    for ref in dict.fromkeys((inc, 'linear', PLAIN.get(arm, inc))):
                        d = e.get(f'minus_{ref}')
                        if d:
                            marks.append(f"vs {ref} {d['delta']:+.4f}"
                                         + ('*' if d['excludes_zero'] else '')
                                         + ('' if d['estimable'] else ' n.e.'))
                    print(f'   {arm:<26} {e["mae"]:.4f}   ' + '  '.join(marks))
    print('\nEXO2 720 h:', json.dumps(r.get('exo2_720h'), indent=1))
    print('\ndecisions:', json.dumps(r['decisions'], indent=1))
    print('predictions:', {k: (v.get('held'), v.get('held_jackknife'))
                           for k, v in r['registered_predictions'].items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
