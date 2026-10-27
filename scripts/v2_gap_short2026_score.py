"""Addendum 82, part C: scoring the 1-3 h Gap-2026 rows beside the published 6-72 h ones.

Refuses to start unless runs/v2_addendum82/C/v2_gap_short2026_gate.json passed. Every published gap
scorer is rerun unchanged (imported, never edited), with three things redirected inside this module:
its row reader appends the 1-3 h rows of each arm to the published Gap-2026 rows
(scripts/v2_gap_short2026_rows.append), its `paired` keeps six decimals with the published
function's four-decimal result beside it (`paired6`), and its outputs go to
runs/v2_addendum82/C/rescored/ instead of the published artifacts. Of the per-row files the scorers
write there, only those that differ from the published ones under runs/<scorer>/ are kept (every band
but Gap-2026 1-6 h is byte-identical, since the rows are written with a fixed gzip time); the rest is
written to a temporary folder, checked against the published file and read from the published folder
(`Sink`, `Rows`).

  variants     scripts/v2_gap_variants (C2: candidates, ensembles with the stacked weights resumed
               from a copy of the published studies, twin rules; band decisions)
  exogenous    scripts/v2_gap_exogenous (C3, addendum 34)
  covariates   scripts/v2_gap_chronos_covariates (C3, addendum 35)
  isolated     scripts/v2_gap_isolated (C4), its table written beside, never into the thesis
  cells        the declared per-length cells (C1-C4) on overlap clusters of each WHOLE strict cohort
               (Gap-Dev: its 14 strict lengths; Gap-2026: the published 6-72 h and the new 1-3 h
               episodes together), the per-length decisions, C-G3 (every published entry outside the
               Gap-2026 1-6 h band unchanged), and Benjamini-Hochberg and -Yekutieli over part C's own
               intervals; writes runs/v2_addendum82_C.json
Run in that order (exogenous, covariates in parallel after variants; isolated after both); 'all', the
default, runs them in that order one after another.
Usage: python -m scripts.v2_gap_short2026_score [<action>]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_intervals import clusters
from scripts.v2_gap_short2026_rows import (PART, RUNS, SHORT, SHORT_RUN, append, paired6, r6,
                                           short_rows)

RESCORED = PART / 'rescored'
GATE = PART / 'v2_gap_short2026_gate.json'
SUMMARY = RUNS / 'v2_addendum82_C.json'
VARIABLES = ('oxygen', 'temperature')
COHORTS = ('Gap-Dev', 'Gap-2026')
LENGTHS = (1, 2, 3, 6)
FLOOR = {'temperature_own': 0.029, 'temperature_other': 0.20, 'oxygen_umol_kg': 0.059 / 0.032844}
C_MG = 0.032844
#: arms that read only the probe's own record (and at most the forcing): the 0.029 degC scale applies
OWN_RECORD = ('linear', 'pchip', 'chronos_bidir', 'chronos_forward', 'chronos2_bidir', 'arx_bridge',
              'arx_bridge_exo', 'chronos2_exo_bidir', 'bolt_resid_bidir', 'bolt_resid_lin_bidir')


#: the scorers whose per-row files the rescoring writes, in the order v2_gap_isolated merges them
ROW_FOLDERS = ('v2_gap_variants', 'v2_gap_exogenous', 'v2_gap_chronos_covariates')


def gate_passed():
    if not GATE.exists() or not json.loads(GATE.read_text())['passed']:
        raise SystemExit('the reproduction gates have not passed: nothing is scored')


def rescored_band(file):
    """The per-row files the 1-3 h rows change: Gap-2026, band 1-6 h."""
    return file.startswith('Gap-2026__') and file.endswith('__1-6.csv.gz')


class Rows:
    """A rescored per-row folder that reads a file it does not hold from the published folder."""

    def __init__(self, name):
        self.own, self.published = RESCORED / name, RUNS / name

    def __truediv__(self, file):
        own = self.own / file
        return own if own.exists() else self.published / file


class Sink:
    """Where a scorer writes its per-row files: the Gap-2026 1-6 h files under rescored/, every other one
    into a temporary folder, checked byte for byte against the published file after the scorer returns."""

    def __init__(self, name, scratch):
        self.name, self.own, self.published, self.scratch = name, RESCORED / name, RUNS / name, scratch

    def mkdir(self, **kw):
        self.own.mkdir(**kw)

    def __truediv__(self, file):
        return (self.own if rescored_band(file) else self.scratch) / file

    def check(self):
        written = sorted(p.name for p in self.scratch.glob('*.csv.gz'))
        differ = [f for f in written if not (self.published / f).exists()
                  or (self.published / f).read_bytes() != (self.scratch / f).read_bytes()]
        if differ:
            raise SystemExit(f'{self.name}: per-row files outside Gap-2026 1-6 h differ from the published: {differ}')
        print(f'{self.name}: {len(written)} per-row files identical to runs/{self.name}/ (read from there)')


def six(frame, arm, reference, label, cluster_of):
    return paired6(frame, arm, reference, label, cluster_of, check=True)


def copy_studies():
    src = RUNS / 'v2_gap_variants_studies_gp_by_pattern'
    dst = RESCORED / 'stack_studies'
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)
    return dst


# --------------------------------------------------------------------------------- the scorers
def variants():
    import scripts.v2_gap_variants as V
    from scripts.v2_gap_climatology import fix_neighbours, load_predictions, read_arm
    from scripts.v2_gap_neighbours import with_twin

    def load(cohort, variable):
        """v2_gap_variants.load, with the 1-3 h rows appended to every Gap-2026 arm."""
        published, results, new, _ = V.COHORTS[cohort]
        frames = []
        for run, arms in ((published, V.PUBLISHED_ARMS), (new, V.NEW_ARMS + ('gpr',))):
            for arm in arms:
                path = RUNS / run / results / f'{variable}_{arm}' / 'predictions.csv.gz'
                if not path.exists():
                    continue
                if run == published:
                    frame = load_predictions(run, results, variable, arm)
                else:
                    frame = fix_neighbours(read_arm(run, results, variable, arm), run, variable)
                if cohort == 'Gap-2026':
                    frame = append(frame, variable, arm)
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
            with_twin(base, variable, V.PERIOD[cohort])
        return base

    RESCORED.mkdir(parents=True, exist_ok=True)
    V.load = load
    V.paired = six
    V.OUT = RESCORED / 'v2_gap_variants.json'
    V.STUDIES = copy_studies()
    with tempfile.TemporaryDirectory() as tmp:
        V.ROWS = Sink('v2_gap_variants', Path(tmp))
        V.main()
        V.ROWS.check()


def exogenous():
    import scripts.v2_gap_exogenous as E
    original = E.arm_rows

    def arm_rows(cohort, variable, arm):
        frame, report = original(cohort, variable, arm)
        if cohort == 'Gap-2026':
            frame = append(frame, variable, arm)
        return frame, report

    E.arm_rows = arm_rows
    E.paired = six
    E.VARIANTS = Rows('v2_gap_variants')
    E.OUT = RESCORED / 'v2_gap_exogenous.json'
    with tempfile.TemporaryDirectory() as tmp:
        E.ROWS = Sink('v2_gap_exogenous', Path(tmp))
        E.main()
        E.ROWS.check()


def covariates():
    import scripts.v2_gap_chronos_covariates as C

    def load(cohort, variable, band):
        """v2_gap_chronos_covariates.load, with the 1-3 h rows appended to every Gap-2026 arm."""
        base = pd.read_csv(C.VARIANTS / f'{cohort}__{variable}__{band}.csv.gz')
        lo, hi = C.BANDS[band]
        reports = {}
        for arm in C.NEW:
            folder = C.arm_folder(cohort, variable, arm)
            reports[arm] = json.loads((folder / 'report.json').read_text())
            frame = pd.read_csv(folder / 'predictions.csv.gz')
            if cohort == 'Gap-2026':
                frame = append(frame, variable, arm)
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

    C.load = load
    C.paired = six
    C.VARIANTS = Rows('v2_gap_variants')
    C.OUT = RESCORED / 'v2_gap_chronos_covariates.json'
    with tempfile.TemporaryDirectory() as tmp:
        C.ROWS = Sink('v2_gap_chronos_covariates', Path(tmp))
        C.main()
        C.ROWS.check()


def isolated():
    import scripts.v2_gap_isolated as I

    def rows(cohort, variable, band):
        """v2_gap_isolated.rows (same merge and checks), each rescored folder read through `Rows`."""
        base = None
        for name in ROW_FOLDERS:
            frame = pd.read_csv(Rows(name) / f'{cohort}__{variable}__{band}.csv.gz')
            if base is None:
                base = frame
                continue
            keys = ['episode', 'position']
            if not (frame[keys].equals(base[keys]) and
                    (frame.truth - base.truth).abs().max() == 0):
                raise SystemExit(f'{name} {cohort} {variable} {band}: rows differ from the variants rows')
            for column in frame.columns:
                if column not in base:
                    base[column] = frame[column].to_numpy()
        return base

    I.SOURCES = {'v2_addendum82/C/rescored/v2_gap_variants.json': 'v2_addendum82/C/rescored/v2_gap_variants',
                 'v2_addendum82/C/rescored/v2_gap_exogenous.json': 'v2_addendum82/C/rescored/v2_gap_exogenous',
                 'v2_addendum82/C/rescored/v2_gap_chronos_covariates.json':
                     'v2_addendum82/C/rescored/v2_gap_chronos_covariates'}
    I.rows = rows
    I.paired = six
    I.OUT = RESCORED / 'v2_gap_isolated.json'
    I.TABLE = RESCORED / 'table_bay_gap_isolated.preview.tex'
    I.main()


# ------------------------------------------------------------------------------ the declared cells
def whole_cohort_clusters():
    """episode -> cluster over each whole strict cohort (the Gap-Dev convention of v2_gap_intervals)."""
    out = {}
    for variable in VARIABLES:
        d = pd.read_csv(RUNS / '20260922T171859Z_v2_gaprecovery' / 'v2_gap_recovery' /
                        f'{variable}_chronos_bidir' / 'predictions.csv.gz',
                        usecols=['episode', 'sensor', 'time', 'gap_length'])
        r = pd.read_csv(RUNS / '20260922T180814Z_v2_gaprecovery' / 'v2_gap_recovery_reserved2026' /
                        f'{variable}_chronos_bidir' / 'predictions.csv.gz',
                        usecols=['episode', 'sensor', 'time', 'gap_length'])
        s = short_rows(variable, 'chronos_bidir')[['episode', 'sensor', 'time', 'gap_length']]
        out[('Gap-Dev', variable)] = clusters(d)
        out[('Gap-2026', variable)] = clusters(pd.concat([r, s], ignore_index=True))
        # the published 6-72 h partition must survive the union (C-G3)
        alone = clusters(r)
        union = out[('Gap-2026', variable)]
        part = pd.DataFrame({'e': list(alone)})
        a = part.e.map(alone)
        b = part.e.map(union)
        if not (a.groupby(b).nunique().max() == 1 and b.groupby(a).nunique().max() == 1):
            raise SystemExit(f'{variable}: the 1-3 h episodes change the published 6-72 h clusters')
    return out


def rows_frame(kind, cohort, variable):
    return pd.read_csv(Rows(kind) / f'{cohort}__{variable}__1-6.csv.gz')


def forward_frame(cohort, variable):
    from scripts.v2_gap_climatology import load_predictions
    run, results = (('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery') if cohort == 'Gap-Dev'
                    else ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026'))
    f = load_predictions(run, results, variable, 'chronos_forward')
    if cohort == 'Gap-2026':
        f = append(f, variable, 'chronos_forward')
    return f[f.gap_length <= 6]


def floor_for(variable, arm, reference):
    if variable == 'oxygen':
        return 'oxygen 0.059 mg/l'
    reads_other = not (arm in OWN_RECORD and reference in OWN_RECORD)
    return '0.20 degC co-located disagreement' if reads_other else '0.029 degC quantisation'


def cell(frame, arm, ref, label, units, family, cohort, variable, length, reproduces=None):
    e = six(frame, arm, ref, label, units)
    if e is None:
        return None
    e.update({'family': family, 'cohort': cohort, 'variable': variable, 'length': length, 'arm': arm,
              'reference': ref, 'label': label, 'floor': floor_for(variable, arm, ref),
              'reproduces_published': reproduces})
    if variable == 'oxygen':
        e['mg_per_l'] = {k: (None if e.get(k) is None else r6(C_MG * e[k]))
                         for k in ('delta', 'ci_low', 'ci_high', 'mae_arm', 'mae_reference')}
    return e


def published_interval(cohort, variable, arm, length):
    name = 'v2_gap_intervals.json' if cohort == 'Gap-Dev' else 'v2_gap_intervals_reserved2026.json'
    got = json.loads((RUNS / name).read_text())['variables'][variable]['by_gap_length']
    return got.get(str(length), {}).get(arm)


def matches_published(e, published):
    if not published:
        return None
    pf = e.get('published_function') or {}
    return all(pf.get(f) == published.get(f) for f in ('delta', 'ci_low', 'ci_high', 'clusters',
                                                       'episodes', 'jackknife_t'))


def cells():
    gate_passed()
    import scripts.v2_gap_chronos_covariates as C
    import scripts.v2_gap_exogenous as E
    import scripts.v2_gap_variants as V
    units = whole_cohort_clusters()
    out = []
    for variable in VARIABLES:
        for cohort in COHORTS:
            u = units[(cohort, variable)]
            va = rows_frame('v2_gap_variants', cohort, variable)
            ex = rows_frame('v2_gap_exogenous', cohort, variable)
            co = rows_frame('v2_gap_chronos_covariates', cohort, variable)
            fw = forward_frame(cohort, variable)
            candidates = [a for a in ('chronos_bidir', 'gpr', 'chronos2_bidir', 'chronos2_cov_bidir',
                                      'gpr_k8', 'ens_mean', 'ens_mean_c2', 'ens_stack', 'twin_rule',
                                      'twin_else_gpr', 'twin_else_best') if a in va]
            for L in LENGTHS:
                # C1: Chronos-Bolt bidirectional and forward minus linear at 1-3 h (and 6 h beside)
                for arm, frame in (('chronos_bidir', va), ('chronos_forward', fw)):
                    part = frame[frame.gap_length == L]
                    pub = published_interval(cohort, variable, arm, L)
                    e = cell(part, arm, 'linear', f'{variable}|{arm}|{L}', u, 'C1', cohort, variable, L)
                    e['reproduces_published'] = matches_published(e, pub)
                    out.append(e)
                # C2: every variants candidate minus linear, Chronos-Bolt bidirectional and the GP
                part = va[va.gap_length == L]
                for arm in candidates:
                    for ref in ('linear', 'chronos_bidir', 'gpr'):
                        if arm == ref or (arm == 'chronos_bidir' and ref == 'linear'):
                            continue
                        label = (f'{variable}|{arm}|{L}' if ref == 'linear'
                                 else f'{cohort}|{variable}|{arm}|{ref}|{L}')
                        e = cell(part, arm, ref, label, u, 'C2', cohort, variable, L)
                        if ref == 'linear':
                            e['reproduces_published'] = matches_published(
                                e, published_interval(cohort, variable, arm, L))
                        out.append(e)
                # C3: addendum 34 and 35 arms against their incumbent, linear and their counterparts
                inc = E.INCUMBENT[(variable, '1-6')]
                for frame, arms, refs_of in (
                        (ex, E.NEW, lambda a: (inc, 'linear', *E.COUNTERPARTS.get(a, ()))),
                        (co, C.NEW, lambda a: (inc, 'linear', C.PLAIN[a]) + ((C.TREE[a],) if a in C.TREE
                                                                            else ()))):
                    part = frame[frame.gap_length == L]
                    for arm in arms:
                        for ref in dict.fromkeys(refs_of(arm)):
                            if ref == arm or ref not in part:
                                continue
                            out.append(cell(part, arm, ref, f'{cohort}|{variable}|{arm}|{ref}|{L}', u,
                                            'C3', cohort, variable, L))
    # C4: the no-neighbour reading per length, with the band's network method (development error)
    iso = json.loads((RESCORED / 'v2_gap_isolated.json').read_text())['after_the_fact']
    own = [a for a in ('chronos_bidir', 'chronos2_bidir', 'arx_bridge', 'arx_bridge_exo',
                       'chronos2_exo_bidir', 'bolt_resid_bidir', 'bolt_resid_lin_bidir')]
    for variable in VARIABLES:
        band = iso[variable]['1-6']
        network = band['network']
        for L in LENGTHS:
            qualified = []
            for arm in own:
                got = [e for e in out if e and e['family'] in ('C1', 'C2', 'C3') and e['arm'] == arm
                       and e['reference'] == 'linear' and e['variable'] == variable and e['length'] == L]
                if len(got) == 2 and all(e['excludes_zero'] and e['delta'] < 0 and e['robust'] for e in got):
                    qualified.append(arm)
            isolated_arm = 'linear'
            if qualified:
                dev_mae = {e['arm']: e['mae_arm'] for e in out if e and e['cohort'] == 'Gap-Dev'
                           and e['variable'] == variable and e['length'] == L and e['reference'] == 'linear'}
                isolated_arm = min(qualified, key=lambda a: dev_mae[a])
            for cohort in COHORTS:
                frames = [rows_frame(k, cohort, variable) for k in
                          ('v2_gap_variants', 'v2_gap_exogenous', 'v2_gap_chronos_covariates')]
                base = frames[0]
                for f in frames[1:]:
                    for c in f.columns:
                        if c not in base:
                            base[c] = f[c].to_numpy()
                part = base[base.gap_length == L]
                e = cell(part, network, isolated_arm,
                         f'isolated|{cohort}|{variable}|1-6|{network}|{isolated_arm}|{L}',
                         units[(cohort, variable)], 'C4', cohort, variable, L)
                e['isolated_method'] = isolated_arm
                e['isolated_candidates_that_qualified'] = qualified
                out.append(e)
    out = [e for e in out if e is not None]
    (RESCORED / 'cells.json').write_bytes((json.dumps(out, indent=2) + '\n').encode('utf-8'))
    print(len(out), 'cells')


# ------------------------------------------------------------------------------ C-G3 and summary
def walk(new, old, path, diffs, allowed):
    """Compare a recomputed artifact with the published one: a recomputed interval through its
    published-function values, everything else by value."""
    if isinstance(new, dict) and 'published_function' in new and isinstance(old, dict):
        pf = new['published_function']
        for f in ('delta', 'ci_low', 'ci_high', 'clusters', 'episodes', 'jackknife_t'):
            if f in old and pf.get(f) != old.get(f):
                diffs.append(('/'.join(path + [f]), allowed(path)))
        return
    if isinstance(new, dict) and isinstance(old, dict):
        for k in set(new) | set(old):
            if k not in old or k not in new:
                if k not in ('published_function', 'agrees_at_four_decimals', 'mae_arm', 'mae_reference',
                             'paired_rows', 'status', 'robust', 'p'):
                    diffs.append(('/'.join(path + [str(k)]) + (' (new key)' if k not in old else ' (gone)'),
                                  allowed(path + [str(k)])))
                continue
            walk(new[k], old[k], path + [str(k)], diffs, allowed)
        return
    if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        for i, (a, b) in enumerate(zip(new, old)):
            walk(a, b, path + [str(i)], diffs, allowed)
        return
    same = new == old
    if not same and isinstance(new, float) and isinstance(old, float):
        same = round(new, 4) == round(old, 4)
    if not same:
        diffs.append(('/'.join(path), allowed(path)))


def g3():
    """Every published entry outside the Gap-2026 1-6 h band, and outside what is derived from it,
    must be value-identical after the merge."""
    def allowed(path):
        p = '/'.join(path)
        return (('Gap-2026' in path and '1-6' in path)
                or p.startswith('decisions') or p.startswith('registered_predictions')
                or ('1-6' in path and p.startswith('after_the_fact')))
    out, ok = {}, True
    for name in ('v2_gap_variants.json', 'v2_gap_exogenous.json', 'v2_gap_chronos_covariates.json',
                 'v2_gap_isolated.json'):
        new = json.loads((RESCORED / name).read_text())
        old = json.loads((RUNS / name).read_text())
        keep = ['results', 'decisions', 'stack_weights', 'registered_predictions', 'exo2_720h',
                'pooled_gap_dev', 'after_the_fact', 'while_silent']
        diffs = []
        for k in keep:
            if k in old or k in new:
                walk(new.get(k), old.get(k), [k], diffs, allowed)
        unexpected = sorted({d for d, a in diffs if not a})
        out[name] = {'differences': len(diffs), 'allowed': sorted({d for d, a in diffs if a})[:400],
                     'unexpected': unexpected[:200], 'passed': not unexpected}
        ok &= not unexpected
    # Table 6.1's two short-gap rows read 12-72 h cells that must be untouched
    var = json.loads((RESCORED / 'v2_gap_variants.json').read_text())
    pub = json.loads((RUNS / 'v2_gap_variants.json').read_text())
    a = var['results']['Gap-2026']['temperature']['12-72']['gpr']['minus_linear']['published_function']
    b = pub['results']['Gap-2026']['temperature']['12-72']['gpr']['minus_linear']
    out['table_6_1_gap_gpr_short_unchanged'] = all(a[f] == b[f] for f in ('delta', 'ci_low', 'ci_high'))
    ok &= out['table_6_1_gap_gpr_short_unchanged']
    out['stack_weights_unchanged'] = var['stack_weights'] == pub['stack_weights']
    ok &= out['stack_weights_unchanged']
    out['passed'] = bool(ok)
    return out


def multiplicity(entries):
    from scripts.v2_fdr import step_up
    fam = [e for e in entries if e.get('p') is not None]
    p = [e['p'] for e in fam]
    bh, by = step_up(p), step_up(p, dependence_robust=True)
    for e, a, b in zip(fam, bh, by):
        e['bh'], e['by'] = bool(a), bool(b)
    return {'intervals': len(fam), 'excluding_zero': int(sum(e['excludes_zero'] for e in fam)),
            'retained_bh': int(sum(bh)), 'retained_by': int(sum(by)), 'q': 0.05}


def decisions(entries):
    """Per length: a candidate replaces the incumbent only if it beats it with an interval excluding
    zero in both cohorts at that length, the jackknife agreeing below twenty clusters."""
    out = {}
    for variable in VARIABLES:
        for family, members, inc in (('C1-C2', ('C1', 'C2'), 'linear'),
                                     ('C3', ('C3',), 'linear' if variable == 'oxygen' else 'ens_stack')):
            for L in LENGTHS:
                arms = sorted({e['arm'] for e in entries if e['family'] in members
                               and e['variable'] == variable and e['length'] == L and e['reference'] == inc})
                won = []
                for arm in arms:
                    got = [e for e in entries if e['family'] in members and e['variable'] == variable
                           and e['length'] == L and e['arm'] == arm and e['reference'] == inc]
                    if len(got) == 2 and all(e['excludes_zero'] and e['delta'] < 0 and e['robust']
                                             for e in got):
                        won.append(arm)
                out[f'{family} {variable} {L} h'] = {'incumbent': inc, 'replaced_by': won}
    return out


def summary():
    gate_passed()
    entries = json.loads((RESCORED / 'cells.json').read_text())
    pchip = json.loads((PART / 'v2_gap_pchip.json').read_text())
    new_cells = [e for e in entries if not e.get('reproduces_published')]
    pchip_cells = []
    for cohort in ('Gap-Dev', 'Gap-2026', 'Gap-Long-Dev', 'Gap-Long-2026'):
        for variable, block in pchip[cohort].items():
            if 'pooled' in block and block['pooled']:
                pchip_cells.append({**block['pooled'], 'family': 'C6', 'cohort': cohort,
                                    'variable': variable, 'length': 'pooled', 'arm': 'pchip',
                                    'reference': 'linear'})
            for L, e in block['by_gap_length'].items():
                if e:
                    pchip_cells.append({**e, 'family': 'C6', 'cohort': cohort, 'variable': variable,
                                        'length': int(L), 'arm': 'pchip', 'reference': 'linear'})
    # band cells of the rescored scorers: the Gap-2026 1-6 h band is new, Gap-Dev's reproduces
    band = []
    for name, family in (('v2_gap_variants.json', 'C2'), ('v2_gap_exogenous.json', 'C3'),
                         ('v2_gap_chronos_covariates.json', 'C3')):
        res = json.loads((RESCORED / name).read_text())['results']
        for cohort in COHORTS:
            for variable in VARIABLES:
                for arm, entry in res[cohort][variable]['1-6'].items():
                    for k, e in entry.items():
                        if k.startswith('minus_') and isinstance(e, dict) and e.get('delta') is not None:
                            band.append({**e, 'family': family + ' band', 'cohort': cohort,
                                         'variable': variable, 'length': '1-6', 'arm': arm,
                                         'reference': k[len('minus_'):], 'source': f'rescored/{name}',
                                         'reproduces_published': cohort == 'Gap-Dev'})
    # the no-neighbour band cells computed by the isolated reading itself (others are read from the
    # scorers' artifacts and already counted there)
    iso = json.loads((RESCORED / 'v2_gap_isolated.json').read_text())['after_the_fact']
    for variable in VARIABLES:
        for c in iso[variable]['1-6']['cells']:
            for cohort in COHORTS:
                e = c[cohort]['network_minus_isolated']
                if e.get('source') == 'computed here' and e.get('delta') is not None:
                    band.append({**e, 'family': 'C4 band', 'cohort': cohort, 'variable': variable,
                                 'length': '1-6', 'arm': c['network'], 'reference': c['isolated'],
                                 'source': 'rescored/v2_gap_isolated.json',
                                 'reproduces_published': cohort == 'Gap-Dev'})
    family = ([e for e in new_cells if e['family'] != 'C5'] + pchip_cells
              + [e for e in band if not e['reproduces_published']])
    report = {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum82.json', 'part': 'C',
        'run': SHORT_RUN,
        'units': 'oxygen umol/kg as stored (mg_per_l beside, c = 0.032844); temperature degC',
        'gates': {'C-G0..C-G2, C-G4, C-G5': 'runs/v2_addendum82/C/v2_gap_short2026_gate.json',
                  'C-G3_nothing_published_moves': g3(),
                  'C-G6_pchip': pchip['gate_C-G6'],
                  'C-G7_pretrained': 'runs/v2_addendum82/C/pretrained.json'},
        'reading': 'per length (1, 2, 3, 6 h) on overlap clusters of the whole strict cohort; the 1-6 h '
                   'band on band clusters beside; decisions per length need an interval excluding zero '
                   'in both cohorts, the jackknife agreeing below twenty clusters',
        'decisions_per_length': decisions(entries),
        'band_decisions': {name: json.loads((RESCORED / name).read_text())['decisions']
                           for name in ('v2_gap_variants.json', 'v2_gap_exogenous.json',
                                        'v2_gap_chronos_covariates.json')},
        'multiplicity_part_C': multiplicity(family),
        'cells': entries, 'band_cells': band, 'pchip': {k: pchip[k] for k in
                                                        ('Gap-Dev', 'Gap-2026', 'Gap-Long-Dev',
                                                         'Gap-Long-2026')},
        'rescored_artifacts': [f'runs/v2_addendum82/C/rescored/{n}' for n in
                               ('v2_gap_variants.json', 'v2_gap_exogenous.json',
                                'v2_gap_chronos_covariates.json', 'v2_gap_isolated.json')],
    }
    pre = PART / 'pretrained.json'
    if pre.exists():
        report['pretrained_addendum81_family'] = json.loads(pre.read_text()).get('summary')
    SUMMARY.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print('C-G3', 'PASSED' if report['gates']['C-G3_nothing_published_moves']['passed'] else 'FAILED')
    print('family', report['multiplicity_part_C'])
    print('wrote', SUMMARY.relative_to(RUNS.parents[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', nargs='?', default='all',
                   choices=['variants', 'exogenous', 'covariates', 'isolated', 'cells', 'summary', 'all'])
    a = p.parse_args()
    gate_passed()
    steps = {'variants': variants, 'exogenous': exogenous, 'covariates': covariates, 'isolated': isolated,
             'cells': cells, 'summary': summary}
    # 'all': every step in the declared order, as the pipeline runs it (after scripts.v2_gap_pchip and the
    # pretrained arms' scoring, whose outputs the summary reads)
    for name in (steps if a.action == 'all' else [a.action]):
        steps[name]()


if __name__ == '__main__':
    main()
