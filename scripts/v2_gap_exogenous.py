"""Addendum 34 and its amendment: gap-filling regressions anchored to the edges, with exogenous series.

For each of the four cohorts the addendum-34 arms are merged with the addendum-33 per-row
predictions of that cohort and band, which carry linear interpolation, the incumbents and Chronos-2
with neighbours. The merge is refused unless both score exactly the same rows with the same truth.
Then every arm is contrasted, paired, with its band's incumbent, with linear, with Chronos-2 with
neighbours and with its declared counterpart:

  anchored_gbm_exo      against anchored_gbm        (does the weather add anything?)
  arx_bridge_exo        against arx_bridge          (the same question, other model family)
  anchored_gbm_rel      against anchored_gbm        (amendment 1: edge-relative features)
  anchored_gbm_rel_exo  against anchored_gbm_rel and anchored_gbm

The anchored arms are also contrasted with the published direct regression on Gap-Dev, the only
cohort that has its per-row predictions. The decision rule is addendum 33's: an arm replaces an
incumbent only if it beats it with an interval excluding zero in the development and the 2026
cohort of the band.

Past a week the cohorts are tiled, so their windows are independent only within one length
(addendum 32). There every contrast, decision and registered prediction is read per
length on probe x calendar month, and the contrasts pooled across lengths stay in the artifact as
descriptions that decide nothing. Per length, a candidate must beat the
declared incumbent and, where addendum 33 read per length leaves another method in place (the
Gaussian process on temperature at the lengths where the stack did not replace it), that method
too. Below twenty units Chapter 4 also asks the delete-one-unit jackknife to agree: decisions carry
`replaced_at_robust` beside `replaced_at`, and every registered prediction `held_jackknife` beside
`held`, the declared percentile reading.

Usage: python -m scripts.v2_gap_exogenous
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import load_predictions
from scripts.v2_gap_intervals import clusters, paired

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_exogenous.json'
ROWS = RUNS / 'v2_gap_exogenous'
VARIANTS = RUNS / 'v2_gap_variants'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
FLOOR = {'temperature': 0.20, 'oxygen': 1.8}

#: cohort -> (addendum-34 run, its results dir, bands it serves)
COHORTS = {
    'Gap-Dev': ('20260922T215308Z_v2_gaprecovery', 'v2_gap_recovery', ('1-6', '12-72')),
    'Gap-2026': ('20260922T215413Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026',
                 ('1-6', '12-72')),
    'Gap-Long-Dev': ('20260922T215328Z_v2_gaprecovery', 'v2_gap_recovery_long', ('168-1440',)),
    'Gap-Long-2026': ('20260922T215431Z_v2_gaprecovery', 'v2_gap_recovery_long_reserved2026',
                      ('168-1440',)),
}
#: The anchored trees read the neighbour features, and a probe is never its own neighbour: co-located
#: probes share coordinates, so the temperature arms were rerun on identical episodes with the probe
#: excluded from its own neighbours by index.
#: (cohort, variable, arm) -> the run that supersedes the one in COHORTS; ARX arms read no
#: neighbours and oxygen had no twin, so neither is replaced.
REPLACED = {(cohort, 'temperature', arm): run
            for cohort, run in (('Gap-Dev', '20260923T101654Z_v2_gaprecovery'),
                                ('Gap-Long-Dev', '20260923T101716Z_v2_gaprecovery'),
                                ('Gap-2026', '20260923T101736Z_v2_gaprecovery'),
                                ('Gap-Long-2026', '20260923T101758Z_v2_gaprecovery'))
            for arm in ('anchored_gbm', 'anchored_gbm_exo', 'anchored_gbm_rel',
                        'anchored_gbm_rel_exo')}
#: the published run whose direct regression the anchored arms are compared with
PUBLISHED_DEV = ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery')
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'),
         '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
BANDS = {'1-6': (1, 6), '12-72': (12, 72), '168-1440': (168, 1440)}
INCUMBENT = {('oxygen', '1-6'): 'linear', ('oxygen', '12-72'): 'chronos_bidir',
             ('oxygen', '168-1440'): 'chronos_bidir'}
INCUMBENT.update({('temperature', b): 'ens_stack' for b in BANDS})
DECLARED = ('anchored_gbm', 'anchored_gbm_exo', 'arx_bridge', 'arx_bridge_exo')
AMENDED = ('anchored_gbm_rel', 'anchored_gbm_rel_exo')
NEW = DECLARED + AMENDED
COUNTERPARTS = {'anchored_gbm_exo': ('anchored_gbm',), 'arx_bridge_exo': ('arx_bridge',),
                'anchored_gbm_rel': ('anchored_gbm',),
                'anchored_gbm_rel_exo': ('anchored_gbm_rel', 'anchored_gbm')}
#: bands whose cohorts are tiled, so that contrasts, decisions and predictions are per length only
PER_LENGTH = ('168-1440',)
#: Chapter 4: below about twenty units a percentile interval is robust only where the jackknife agrees
ROBUST_UNITS = 20
REFERENCES = ('linear', 'chronos_bidir', 'gpr', 'chronos2_cov_bidir', 'ens_stack', 'ens_mean_c2',
              'regression_retro')


def per_length_incumbents(variable, band, lengths):
    """{length: methods a candidate must beat there}: the declared incumbent, and the method that
    addendum 33, read per length on the same cohorts, leaves in place where that differs (only
    replacements the jackknife confirms count). Read from runs/v2_gap_variants.json."""
    declared = INCUMBENT[(variable, band)]
    a33 = json.loads((RUNS / 'v2_gap_variants.json').read_text())['decisions'][f'{variable} {band}']
    robust = a33.get('replaced_at_robust', {})
    out = {}
    for L in lengths:
        held = a33['incumbent']
        for arm, at in robust.items():
            if L in at and arm == declared:
                held = arm
        out[L] = list(dict.fromkeys((declared, held)))
    return out


def arm_rows(cohort, variable, arm):
    run, results, _ = COHORTS[cohort]
    run = REPLACED.get((cohort, variable, arm), run)
    folder = RUNS / run / results / f'{variable}_{arm}'
    report = json.loads((folder / 'report.json').read_text())
    frame = pd.read_csv(folder / 'predictions.csv.gz')
    return frame, report


def load(cohort, variable, band):
    """The addendum-33 rows of a band with every addendum-34 arm merged on, refusing a mismatch."""
    base = pd.read_csv(VARIANTS / f'{cohort}__{variable}__{band}.csv.gz')
    lo, hi = BANDS[band]
    reports = {}
    for arm in NEW:
        frame, reports[arm] = arm_rows(cohort, variable, arm)
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
    if cohort == 'Gap-Dev':
        published = load_predictions(*PUBLISHED_DEV, variable, 'regression_retro')
        base = base.merge(published[['episode', 'position', 'regression_retro']],
                          on=['episode', 'position'], how='left', validate='one_to_one')
    return base, reports


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return None
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def contrasts(frame, incumbent, label, long_cohort):
    """Band contrasts on overlap clusters, per-length ones on the cohort's own unit.

    On a long (tiled) cohort the band contrasts pool lengths whose windows overlap, so they are
    descriptions only; per length, on probe x calendar month, each arm is also contrasted with
    every reference of its band and, on temperature, with the Gaussian process, which is what the
    decisions and registered predictions read there. Existing contrasts keep their seed labels, so
    no value computed before moves.
    """
    cluster_of = clusters(frame)
    out = {}
    for arm in NEW + REFERENCES:
        if arm not in frame:
            continue
        entry = {'mae': macro(frame, arm)}
        if arm in NEW:
            refs = [incumbent, 'linear', 'chronos2_cov_bidir', *COUNTERPARTS.get(arm, ())]
            if 'regression_retro' in frame and arm.startswith('anchored'):
                refs.append('regression_retro')
            for ref in dict.fromkeys(refs):
                if ref in frame and ref != arm:
                    entry[f'minus_{ref}'] = paired(frame, arm, ref, f'{label}|{arm}|{ref}',
                                                   cluster_of)
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
                row = {'mae': macro(part, arm), 'linear': macro(part, 'linear'),
                       incumbent: macro(part, incumbent)}
                for ref in dict.fromkeys((incumbent, 'linear', *COUNTERPARTS.get(arm, ()))):
                    if ref != arm:
                        row[f'minus_{ref}'] = paired(part, arm, ref,
                                                     f'{label}|{arm}|{ref}|{int(length)}', units)
                if long_cohort:
                    more = ['chronos2_cov_bidir'] + (['gpr'] if incumbent == 'ens_stack' else [])
                    for ref in more:
                        if ref in part and ref != arm and f'minus_{ref}' not in row:
                            row[f'minus_{ref}'] = paired(part, arm, ref,
                                                         f'{label}|{arm}|{ref}|{int(length)}', units)
                by_length[str(int(length))] = row
            entry['by_gap_length'] = by_length
        elif long_cohort:
            entry['by_gap_length'] = {str(int(length)): {'mae': macro(part, arm)}
                                      for length, part in frame.groupby('gap_length')}
        out[arm] = entry
    return out


def contrast_at(res, cohort, variable, band, arm, ref, length=None):
    """One contrast: the band's, or with `length` the per-length one."""
    entry = res[cohort][variable][band].get(arm, {})
    if length is None:
        return entry.get(f'minus_{ref}') or {}
    return entry.get('by_gap_length', {}).get(str(length), {}).get(f'minus_{ref}') or {}


def firm(e):
    """Chapter 4: the percentile interval counts below twenty units only if the jackknife agrees."""
    return bool(e.get('clusters', 0) >= ROBUST_UNITS or e.get('jackknife_excludes_zero'))


def long_lengths(res, cohort, variable, band):
    arm = res[cohort][variable][band][NEW[0]]
    return sorted(int(L) for L in arm['by_gap_length'])


def decide_per_length(res, variable, band, dev, confirm):
    """The declared rule applied at each length past a week: beats every method it must replace,
    with a probe-month interval excluding zero, in both cohorts at that length."""
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


def pooled_dev(variable):
    """Every Gap-Dev length at once, for the registered contrasts stated 'pooled over lengths'."""
    frames = {arm: arm_rows('Gap-Dev', variable, arm)[0] for arm in NEW}
    base = frames['anchored_gbm'].copy()
    for arm in NEW[1:]:
        f = frames[arm]
        if not f[['episode', 'position']].equals(base[['episode', 'position']]):
            raise SystemExit(f'Gap-Dev {variable} {arm}: rows differ between addendum-34 arms')
        base[arm] = f[arm].to_numpy()
    published = load_predictions(*PUBLISHED_DEV, variable, 'regression_retro')
    base = base.merge(published[['episode', 'position', 'regression_retro']],
                      on=['episode', 'position'], how='left', validate='one_to_one')
    if base.regression_retro.isna().any():
        raise SystemExit(f'Gap-Dev {variable}: published regression rows do not cover addendum 34')
    cluster_of = clusters(base)
    return {f'{a}_minus_{b}': paired(base, a, b, f'pooled|{variable}|{a}|{b}', cluster_of)
            for a, b in (('anchored_gbm', 'regression_retro'), ('anchored_gbm', 'linear'),
                         ('anchored_gbm_rel', 'anchored_gbm'), ('anchored_gbm_rel', 'linear'),
                         ('anchored_gbm_rel', 'regression_retro'))}


def tuned(reports):
    """What each study chose, per fold, and how often the state-space arm fell back to linear."""
    out = {}
    for arm, report in reports.items():
        out[arm] = [{'fold': s['fold'], 'trials': s['selection'].get('completed_trials'),
                     'params': s['selection'].get('params'),
                     'linear_fallback_rows': s['selection'].get('linear_fallback_rows')}
                    for s in report['selections']]
    return out


def main():
    ROWS.mkdir(parents=True, exist_ok=True)
    report = {'declared_in': ['config/bay_reserved_protocol_v2_addendum34.json',
                              'config/bay_reserved_protocol_v2_addendum34_amendment1.json'],
              'cohorts': {c: {'run': r, 'results': d} for c, (r, d, _) in COHORTS.items()},
              'incumbents': {f'{v} {b}': a for (v, b), a in INCUMBENT.items()},
              'results': {}, 'decisions': {}, 'tuning': {}, 'pooled_gap_dev': {}}
    for variable in ('oxygen', 'temperature'):
        report['pooled_gap_dev'][variable] = pooled_dev(variable)
        for band, (dev, confirm) in PAIRS.items():
            incumbent = INCUMBENT[(variable, band)]
            for cohort in (dev, confirm):
                frame, reports = load(cohort, variable, band)
                report['results'].setdefault(cohort, {}).setdefault(variable, {})[band] = \
                    contrasts(frame, incumbent, f'{cohort}|{variable}|{band}',
                              cohort.startswith('Gap-Long'))
                report['tuning'].setdefault(cohort, {})[variable] = tuned(reports)
                frame.to_csv(ROWS / f'{cohort}__{variable}__{band}.csv.gz', index=False,
                             compression={'method': 'gzip', 'mtime': 0})
                if variable == 'oxygen' and band == '168-1440':
                    e = frame[(frame.sensor == EXO2) & (frame.gap_length == 720)]
                    report.setdefault('exo2_720h', {})[cohort] = {
                        a: macro(e, a) for a in NEW + ('linear', 'chronos_bidir', 'gpr',
                                                       'chronos2_cov_bidir', 'ens_stack')}
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


def name_of(co, v, b, L):
    return f'{co} {b}' if L is None else f'{co} {L} h'


def verdicts(r):
    """The ten registered predictions, each cell a band up to 72 h and a length past a week.

    `held` reads the declared percentile interval; `held_jackknife` also asks the jackknife to agree
    below twenty units (Chapter 4), which can only remove a cell that a prediction names.
    """
    declared = evaluate(r, robust=False)
    jack = evaluate(r, robust=True)
    for key, got in declared.items():
        got['held_jackknife'] = jack[key]['held']
        for field in ('cells', 'cells_where_it_holds'):
            if field in jack[key] and jack[key][field] != got.get(field):
                got[f'{field}_jackknife'] = jack[key][field]
    return declared


def evaluate(r, robust):
    res, pooled = r['results'], r['pooled_gap_dev']

    def c(cohort, variable, band, arm, ref, length=None):
        return contrast_at(res, cohort, variable, band, arm, ref, length)

    def ok(e):
        return not robust or firm(e)

    def beats(e, by=0.0):
        return bool(e and e.get('excludes_zero') and e['delta'] < 0 and -e['delta'] >= by
                    and ok(e))

    def loses(e):
        return bool(e and e.get('excludes_zero') and e['delta'] > 0 and ok(e))

    cells = cells_of(res)
    temp = [(co, b, L) for co, v, b, L in cells if v == 'temperature']
    long_dev = long_lengths(res, 'Gap-Long-Dev', 'oxygen', '168-1440')
    # (1) anchoring repairs the regression; at 1-6 h it is not worse than linear (Gap-Dev only)
    p1 = {v: {'minus_regression_retro': pooled[v]['anchored_gbm_minus_regression_retro'],
              'minus_linear_1_6': c('Gap-Dev', v, '1-6', 'anchored_gbm', 'linear')}
          for v in ('oxygen', 'temperature')}
    held1 = all(beats(p1[v]['minus_regression_retro']) and not loses(p1[v]['minus_linear_1_6'])
                for v in p1)
    # (2) no declared arm beats ens_stack on temperature by 0.02 C or more
    over2 = [f'{name_of(co, "temperature", b, L)} {a}' for co, b, L in temp for a in DECLARED
             if beats(c(co, 'temperature', b, a, 'ens_stack', L), 0.02)]
    # (3) the exogenous series add less than 0.02 C on temperature
    over3 = [f'{name_of(co, "temperature", b, L)} {a}' for co, b, L in temp
             for a in ('anchored_gbm_exo', 'arx_bridge_exo')
             if beats(c(co, 'temperature', b, a, COUNTERPARTS[a][0], L), 0.02)]
    # (4) on oxygen the exogenous series help, both families, at 12-72 h on Gap-Dev and past a week
    # on Gap-Long-Dev, there at every length
    named = [('Gap-Dev', '12-72', None)] + [('Gap-Long-Dev', '168-1440', L) for L in long_dev]
    four = {f'{name_of(co, "oxygen", b, L)} {a}': c(co, 'oxygen', b, a, COUNTERPARTS[a][0], L)
            for co, b, L in named for a in ('anchored_gbm_exo', 'arx_bridge_exo')}
    # (5) the best declared arm on long oxygen gaps is within 1 umol/l of Chronos-2 with neighbours,
    # at each length of Gap-Long-Dev
    arms = res['Gap-Long-Dev']['oxygen']['168-1440']
    five = {}
    for L in long_dev:
        mae = {a: arms[a]['by_gap_length'][str(L)]['mae'] for a in DECLARED}
        best = min(DECLARED, key=lambda a: mae[a])
        ref = arms['chronos2_cov_bidir']['by_gap_length'][str(L)]['mae']
        five[str(L)] = {'best_declared_arm': best, 'best_minus_chronos2_cov_bidir':
                        round(mae[best] - ref, 4)}
    # (6) nothing replaces Chronos-Bolt on oxygen at 12-72 h
    replaced6 = r['decisions']['oxygen 12-72']['replaced_by']
    # (7) no arm reaches 5 umol/l on Baia EXO2 at 720 h
    exo = r.get('exo2_720h', {})
    best7 = min(((m.get(a), a, co) for co, m in exo.items() for a in NEW if m.get(a) is not None),
                default=(None, None, None))
    # amendment 1
    a1 = {v: {'rel_minus_anchored': pooled[v]['anchored_gbm_rel_minus_anchored_gbm'],
              'rel_minus_linear_1_6': c('Gap-Dev', v, '1-6', 'anchored_gbm_rel', 'linear')}
          for v in ('oxygen', 'temperature')}
    held_a1 = all(beats(a1[v]['rel_minus_anchored']) and not loses(a1[v]['rel_minus_linear_1_6'])
                  for v in a1)
    a2 = {name_of(co, 'oxygen', b, L): c(co, 'oxygen', b, 'anchored_gbm_rel_exo',
                                         'anchored_gbm_rel', L) for co, b, L in named}
    over_a3 = [f'{name_of(co, "temperature", b, L)} {a}' for co, b, L in temp for a in AMENDED
               if beats(c(co, 'temperature', b, a, 'ens_stack', L), 0.02)]

    def held_where(contrasts):
        usable = {k: e for k, e in contrasts.items() if e.get('estimable')}
        return {'held': bool(usable) and all(beats(e) for e in usable.values()),
                'cells_where_it_holds': [k for k, e in usable.items() if beats(e)],
                'cells_not_estimable': [k for k, e in contrasts.items() if not e.get('estimable')]}

    return {
        '1': {'held': held1, 'blind': 'part (b) was not blind for anchored_gbm on temperature; see '
                                      'amendment 1', 'contrasts': p1},
        '2': {'held': not over2, 'cells': over2},
        '3': {'held': not over3, 'cells': over3},
        '4': {**held_where(four), 'contrasts': four},
        '5': {'held': all(abs(x['best_minus_chronos2_cov_bidir']) <= 1.0 for x in five.values()),
              'by_length': five},
        '6': {'held': not replaced6, 'replaced_by': replaced6},
        '7': {'held': bool(best7[0] is None or best7[0] > 5.0),
              'best_at_720h_on_exo2': {'mae': best7[0], 'arm': best7[1], 'cohort': best7[2]}},
        'A1': {'held': held_a1, 'contrasts': a1},
        'A2': {**held_where(a2), 'contrasts': a2},
        'A3': {'held': not over_a3, 'cells': over_a3},
    }


def summarise(r):
    for cohort, by_var in r['results'].items():
        for variable, by_band in by_var.items():
            for band, arms in by_band.items():
                inc = INCUMBENT[(variable, band)]
                print(f'\n== {cohort} {variable} {band}  (incumbent {inc}, floor {FLOOR[variable]})')
                for arm, e in sorted(arms.items(), key=lambda kv: kv[1]['mae'] or 1e9):
                    marks = []
                    for ref in (inc, 'linear', *COUNTERPARTS.get(arm, ())):
                        d = e.get(f'minus_{ref}')
                        if d:
                            marks.append(f"vs {ref} {d['delta']:+.4f}"
                                         + ('*' if d['excludes_zero'] else '')
                                         + ('' if d['estimable'] else ' n.e.'))
                    print(f'   {arm:<22} {e["mae"]:.4f}   ' + '  '.join(marks))
    print('\npooled Gap-Dev:', json.dumps(r['pooled_gap_dev'], indent=1))
    print('\nEXO2 720 h:', json.dumps(r.get('exo2_720h'), indent=1))
    print('\ndecisions:', json.dumps(r['decisions'], indent=1))
    print('predictions:', {k: (v.get('held'), v.get('held_jackknife'))
                           for k, v in r['registered_predictions'].items()})
    print('wrote', OUT)


if __name__ == '__main__':
    main()
