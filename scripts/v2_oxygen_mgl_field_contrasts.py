"""Addendum 60 parts C1 and C2: the Field-57 comparators and every Field-57 contrast, in mg/l.

Reads the arms scripts/v2_oxygen_mgl_field.py refitted in mg/l and recomputes, with the published
generators, streams and draws:

  C1/v2_field_naive/            the naive comparators (climatologies, the physical background, the
                                released model at its penalty), as src/experiments/v2_field_naive.py
  C1/v2_field_contrasts.json    every paired Field-57 contrast, as scripts/v2_field_contrasts.py
  C1/v2_perfold.json            the Field-57 section of addendum 47 (scripts/v2_perfold_contrasts.field)
  C1/v2_perfold_rest.json       the Field-57 section of addendum 48 (scripts/v2_perfold_rest_contrasts.field)
  C1/v2_background_bias.json    the background's mean signed residual (scripts/v2_background_bias.py)
  C1/20260919T145045Z_v2_field/v2_field_summary.json   the campaign's per-arm errors and penalties

Every step runs first on the published files in the published unit and must give the published
artifact (to the precision it stores) before it is run on the mg/l files. The two cast bootstraps are
computed from per-cast mean errors with the published generators (the identity is checked by that
reproduction), because the pandas loops of the originals take hours for both units.

Usage: python -m scripts.v2_oxygen_mgl_field_contrasts
"""
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import OUT, ROOT, RUNS, harmonised, lf_rewrite, write_json

C1, C2 = OUT / 'C1', OUT / 'C2'
SCRATCH = OUT / '_reproduction'
TARGETS = ('temp_c', 'salinity_psu', 'do_umol_l')


# ---------------------------------------------------------------- fast nested bootstraps

class Nested:
    """Triple-nested macro MAE from per-cast mean errors, with weights per cast (a drawn unit's multiplicity)."""

    def __init__(self, keys):
        casts = keys.cast_id.astype(str).to_numpy()
        self.codes, self.cast = np.unique(casts, return_inverse=True)
        first = (pd.DataFrame({'c': self.cast, 'fold': keys.fold.to_numpy(), 'station': keys.station.to_numpy()})
                 .drop_duplicates('c').sort_values('c'))
        cells = list(dict.fromkeys(zip(first.fold, first.station)))
        self.cell = np.array([cells.index(c) for c in zip(first.fold, first.station)])
        self.cell_fold = np.array([c[0] for c in cells])
        self.n = np.bincount(self.cast, minlength=len(self.codes))

    def cast_means(self, err):
        return np.bincount(self.cast, weights=err, minlength=len(self.codes)) / self.n

    def value(self, w, e):
        tot = np.bincount(self.cell, weights=w * e, minlength=self.cell.max() + 1)
        cnt = np.bincount(self.cell, weights=w, minlength=self.cell.max() + 1)
        present = cnt > 0
        return float(np.mean([np.mean(tot[present & (self.cell_fold == f)] / cnt[present & (self.cell_fold == f)])
                              for f in np.unique(self.cell_fold) if (present & (self.cell_fold == f)).any()]))


def field_contrast(keys, ea, eb, unit, label, t, draws=2000, digits=6):
    """scripts/v2_field_contrasts.contrast: a drawn unit relabelled, so a repeat counts twice."""
    ok = np.isfinite(ea) & np.isfinite(eb)
    k = keys[ok].reset_index(drop=True)
    ea, eb = ea[ok], eb[ok]
    nest = Nested(k)
    ma, mb = nest.cast_means(ea), nest.cast_means(eb)
    ones = np.ones(len(nest.codes))
    point = nest.value(ones, ma) - nest.value(ones, mb)
    col = 'day' if unit == 'date' else 'cast_id'
    groups = k.groupby(col).indices
    names = list(groups)
    unit_of_cast = pd.Series(k[col].to_numpy(), index=nest.cast).groupby(level=0).first()
    unit_index = {n: i for i, n in enumerate(names)}
    cast_unit = np.array([unit_index[unit_of_cast[c]] for c in range(len(nest.codes))])
    rng = np.random.default_rng([20260923, zlib.crc32(f'{label}|{t}|{col}'.encode())])
    out = np.empty(draws)
    for i in range(draws):
        m = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)).astype(float)
        w = m[cast_unit]
        out[i] = nest.value(w, ma) - nest.value(w, mb)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {'point': round(point, digits), 'ci': [round(float(lo), digits), round(float(hi), digits)],
            'excludes_zero': bool(lo > 0 or hi < 0), 'units': len(names)}


def field_contrasts(sources, out_path, digits=6):
    """scripts/v2_field_contrasts.main on the given prediction files."""
    from scripts import v2_field_contrasts as V
    frames, keys, truth = {}, None, None
    for arm, path in sources.items():
        f = pd.read_csv(path).sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
        k = f[['fold', 'station', 'cast_id', 'depth_m']]
        if keys is None:
            keys, truth = k, f[list(TARGETS)].to_numpy(float)
            day = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        elif not k.equals(keys) or not np.allclose(f[list(TARGETS)].to_numpy(float), truth, equal_nan=True):
            raise SystemExit(f'{arm} does not score the rows of the others')
        frames[arm] = f[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
    keys = keys.assign(day=day)
    report = {'rows': int(len(keys)), 'casts': int(keys.cast_id.nunique()), 'days': int(keys.day.nunique()),
              'draws': V.DRAWS,
              'metric': 'nested macro MAE: depth bins within cast, casts within (fold, station), stations within fold, folds',
              'units': {'date': 'calendar day of the cast (primary)', 'cast': 'the cast'}, 'contrasts': []}
    for label, arm, ref, targets in V.CONTRASTS:
        for t in targets:
            j = TARGETS.index(t)
            ea = np.abs(truth[:, j] - frames[arm][:, j])
            eb = np.abs(truth[:, j] - frames[ref][:, j])
            entry = {'contrast': label, 'arm': arm, 'reference': ref, 'target': t}
            for unit in ('date', 'cast'):
                entry[unit] = field_contrast(keys, ea, eb, unit, label, t, digits=digits)
            report['contrasts'].append(entry)
    write_json(out_path, report)
    return report


def naive_paired_fast(frames, x, y, keys):
    """src/experiments/v2_field_naive.paired: casts drawn with rng.choice and merged back by cast_id."""
    from src.experiments.v2_field_naive import DRAWS, nested_mae
    rng = np.random.default_rng([20260920, zlib.crc32(f'{x}|{y}'.encode())])
    dx, dy = frames[x], frames[y]
    casts = keys.cast_id.unique()
    nest = Nested(dx[['fold', 'station', 'cast_id']])
    pos = {c: i for i, c in enumerate(nest.codes)}
    order = np.array([pos[c] for c in casts])
    point = {t: nested_mae(dx)[t] - nested_mae(dy)[t] for t in TARGETS}
    ex = {t: nest.cast_means(np.abs(dx[t] - dx['pred_' + t]).to_numpy()) for t in TARGETS}
    ey = {t: nest.cast_means(np.abs(dy[t] - dy['pred_' + t]).to_numpy()) for t in TARGETS}
    draws = {t: [] for t in TARGETS}
    for _ in range(DRAWS):
        picked = rng.choice(casts, len(casts), replace=True)
        w = np.zeros(len(nest.codes))
        w[[pos[c] for c in np.unique(picked)]] = 1.0
        for t in TARGETS:
            draws[t].append(nest.value(w, ex[t]) - nest.value(w, ey[t]))
    del order
    return {t: [point[t], *np.percentile(draws[t], [2.5, 97.5]).tolist()] for t in TARGETS}


def naive(run, out, mgl):
    """v2_field_naive.main with its run folder and output redirected."""
    from src.experiments import v2_field_naive as N
    import src.experiments.v2_field  # noqa: F401
    saved = (N.RUN, N.OUT, N.paired, N.save)
    N.RUN, N.OUT, N.paired = run, out, naive_paired_fast
    N.save = lambda path, obj: write_json(path, obj)
    try:
        with harmonised(mgl):
            N.main()
    finally:
        N.RUN, N.OUT, N.paired, N.save = saved
    return finish_naive(out)


def finish_naive(out):
    """The prediction files v2_field_naive wrote with the host's line ending, rewritten with LF."""
    report = json.loads((out / 'field_naive_summary.json').read_text())
    for name in list(report.get('prediction_sha256', {})):
        report['prediction_sha256'][name] = lf_rewrite(out / name)
    write_json(out / 'field_naive_summary.json', report)
    return report


def background_bias(out, mgl):
    from scripts import v2_background_bias as B
    saved = B.OUT
    B.OUT = out
    try:
        with harmonised(mgl):
            B.main()
    finally:
        B.OUT = saved
    write_json(out, json.loads(out.read_text()))
    return json.loads(out.read_text())


def perfold47(runs, work, field_source, out=None):
    from scripts import v2_perfold_contrasts as P
    saved = (P.RUNS, P.WORK, P.field_source)
    P.RUNS, P.WORK, P.field_source = runs, work, field_source
    try:
        return P.field()
    finally:
        P.RUNS, P.WORK, P.field_source = saved


def perfold48(runs, work, a47, field_file):
    from scripts import v2_perfold_rest_contrasts as P
    saved = (P.RUNS, P.WORK, P.A47, P.field_file)
    P.RUNS, P.WORK, P.A47, P.field_file = runs, work, a47, field_file
    try:
        return P.field()
    finally:
        P.RUNS, P.WORK, P.A47, P.field_file = saved


def gate_root(srcs, naive_dir, dst):
    """A runs folder for the addendum-47/48 gate G4b, which compares its bootstrap at four decimals with
    v2_field_contrasts.json. That artifact now stores six (scripts/v2_field_contrasts.py, after the
    addendum-47 run), and rounding six to four can land on the other side of a 5, so the gate reads
    the same contrasts recomputed at the four decimals it was written against; the physical
    background's predictions sit beside them as the gate expects."""
    import shutil
    field_contrasts(srcs, dst / 'v2_field_contrasts.json', digits=4)
    (dst / 'v2_field_naive').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(naive_dir / 'pred_background.csv.gz', dst / 'v2_field_naive/pred_background.csv.gz')
    return dst


def same(a, b, path='', tol=1e-9, rel=False):
    """Every number of `a` equals `b` (structure too); returns the differing paths."""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f'{path}: keys {sorted(map(str, set(a) ^ set(b)))}']
        for k in a:
            bad += same(a[k], b[k], f'{path}/{k}', tol, rel)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f'{path}: length']
        for i, (x, y) in enumerate(zip(a, b)):
            bad += same(x, y, f'{path}[{i}]', tol, rel)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        if a != b:
            bad.append(f'{path}: {a!r} != {b!r}')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(a - b) / (max(abs(b), 1e-12) if rel else 1.0)
        if d > tol:
            bad.append(f'{path}: {a} != {b}')
    elif a != b:
        bad.append(f'{path}: {a!r} != {b!r}')
    return bad


def sources(campaign, local_c1, local_c2, naive_dir):
    S = {}
    for a in ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct', 'A4_base_only_direct', 'A1_both_aou'):
        S[a] = campaign / f'v2_field_held_site_{a}/selected_predictions.csv.gz'
    for a in ('A5_intercept_direct', 'A6_ctdprior_direct', 'A7_ctdprior_direct', 'P1_placebo_direct', 'P2_placebo_direct'):
        root = local_c1 if a == 'A5_intercept_direct' else local_c2
        S[a] = root / f'v2_field_held_site_{a}/selected_predictions.csv.gz'
    for n in ('model', 'background', 'climatology', 'seasonal', 'oracle', 'no_copernicus'):
        S[f'naive_{n}'] = naive_dir / f'pred_{n}.csv.gz'
    return S


def main():
    t0 = time.time()
    record = {'what': 'each Field-57 comparator and contrast recomputed from the published files in the published '
                      'unit, against the published artifact, before the mg/l run', 'steps': {}}
    camp_pub = RUNS / '20260919T145045Z_v2_field/results'
    camp_mgl = C1 / '20260919T145045Z_v2_field/results'

    # 1. the naive comparators
    pub = json.loads((RUNS / 'v2_field_naive/field_naive_summary.json').read_text())
    got = naive(RUNS / '20260919T145045Z_v2_field', SCRATCH / 'v2_field_naive', False)
    diff = same({k: got[k] for k in ('nested_mae', 'contrasts', 'month_cell_coverage_total')},
                {k: pub[k] for k in ('nested_mae', 'contrasts', 'month_cell_coverage_total')}, tol=1e-9, rel=True)
    record['steps']['v2_field_naive'] = {'pass': not diff, 'differences': diff[:20]}
    print('naive reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        raise SystemExit('v2_field_naive does not reproduce')
    naive(C1 / '20260919T145045Z_v2_field', C1 / 'v2_field_naive', True)

    # 2. every paired contrast
    pub = json.loads((RUNS / 'v2_field_contrasts.json').read_text())
    got = field_contrasts(sources(camp_pub, ROOT / 'results', ROOT / 'results', RUNS / 'v2_field_naive'),
                          SCRATCH / 'v2_field_contrasts.json')
    diff = same(got, pub, tol=0)
    record['steps']['v2_field_contrasts'] = {'pass': not diff, 'differences': diff[:20]}
    print('contrast reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        raise SystemExit('v2_field_contrasts does not reproduce')
    field_contrasts(sources(camp_mgl, C1 / 'results', C2 / 'results', C1 / 'v2_field_naive'),
                    C1 / 'v2_field_contrasts.json')

    # 3. addendum 47, Field-57 section
    def src47(root_campaign, root_local, work):
        def f(design, arm):
            if design == 'published':
                return (root_local if arm == 'A7_ctdprior_direct' else root_campaign) / f'v2_field_held_site_{arm}/selected_predictions.csv.gz'
            return work / 'field' / ('published' if design == 'refit' else design) / arm / 'selected_predictions.csv.gz'
        return f
    pub = json.loads((RUNS / 'v2_perfold.json').read_text())['field57']
    got = perfold47(gate_root(sources(camp_pub, ROOT / 'results', ROOT / 'results', RUNS / 'v2_field_naive'),
                              RUNS / 'v2_field_naive', SCRATCH / 'gate_pub'),
                     RUNS / 'v2_perfold', src47(camp_pub, ROOT / 'results', RUNS / 'v2_perfold'))
    got = json.loads(json.dumps(got))
    diff = same({k: got[k] for k in ('arms', 'contrasts', 'dates', 'casts')},
                {k: pub[k] for k in ('arms', 'contrasts', 'dates', 'casts')}, tol=1e-9)
    record['steps']['v2_perfold.field57'] = {'pass': not diff, 'differences': diff[:20]}
    print('addendum 47 reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        raise SystemExit('the addendum-47 Field-57 section does not reproduce')
    fd = perfold47(gate_root(sources(camp_mgl, C1 / 'results', C2 / 'results', C1 / 'v2_field_naive'),
                             C1 / 'v2_field_naive', SCRATCH / 'gate_mgl'),
                   C1 / 'v2_perfold', src47(camp_mgl, C2 / 'results', C1 / 'v2_perfold'))
    write_json(C1 / 'v2_perfold.json', {'declared_in': 'config/bay_reserved_protocol_v2_addendum47.json',
                                         'harmonised': 'addendum 60 part C: the Field-57 section only, every oxygen '
                                                       'input in mg/l, penalties as published',
                                         'field57': fd})

    # 4. addendum 48, Field-57 section
    def src48(root_campaign, root_c1, root_c2, a47, work):
        def f(design, arm):
            if arm in ('A1_both_direct', 'A2_none_direct'):
                if design == 'published':
                    return root_campaign / f'v2_field_held_site_{arm}/selected_predictions.csv.gz'
                return a47 / 'field' / design / arm / 'selected_predictions.csv.gz'
            if design == 'published':
                if arm == 'A1_both_aou':
                    return root_campaign / 'v2_field_held_site_A1_both_aou/selected_predictions.csv.gz'
                root = root_c1 if arm == 'A5_intercept_direct' else root_c2
                return root / f'v2_field_held_site_{arm}' / 'selected_predictions.csv.gz'
            return work / 'field' / ('published' if design == 'refit' else design) / arm / 'selected_predictions.csv.gz'
        return f
    pub = json.loads((RUNS / 'v2_perfold_rest.json').read_text())['field57']
    got = perfold48(SCRATCH / 'gate_pub', RUNS / 'v2_perfold_rest', RUNS / 'v2_perfold',
                    src48(camp_pub, ROOT / 'results', ROOT / 'results', RUNS / 'v2_perfold', RUNS / 'v2_perfold_rest'))
    got = json.loads(json.dumps(got))
    diff = same({k: got[k] for k in ('arms', 'contrasts', 'dates', 'casts')},
                {k: pub[k] for k in ('arms', 'contrasts', 'dates', 'casts')}, tol=1e-9)
    record['steps']['v2_perfold_rest.field57'] = {'pass': not diff, 'differences': diff[:20]}
    print('addendum 48 reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        raise SystemExit('the addendum-48 Field-57 section does not reproduce')
    fd = perfold48(SCRATCH / 'gate_mgl', C1 / 'v2_perfold_rest', C1 / 'v2_perfold',
                   src48(camp_mgl, C1 / 'results', C2 / 'results', C1 / 'v2_perfold', C1 / 'v2_perfold_rest'))
    write_json(C1 / 'v2_perfold_rest.json', {'declared_in': 'config/bay_reserved_protocol_v2_addendum48.json',
                                              'harmonised': 'addendum 60 part C: the Field-57 section only (the '
                                                            'utilisation section is in C3/v2_perfold_rest.json)',
                                              'field57': fd})

    # 5. the background's signed residual
    pub = json.loads((RUNS / 'v2_background_bias.json').read_text())
    got = background_bias(SCRATCH / 'v2_background_bias.json', False)
    diff = same(got, pub, tol=0)
    record['steps']['v2_background_bias'] = {'pass': not diff, 'differences': diff[:20]}
    print('background bias reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        raise SystemExit('v2_background_bias does not reproduce')
    background_bias(C1 / 'v2_background_bias.json', True)

    # 6. the campaign summary's per-arm errors and penalties
    pub = json.loads((RUNS / '20260919T145045Z_v2_field/v2_field_summary.json').read_text())
    arms = {}
    for arm, entry in pub['arms'].items():
        folder = camp_mgl / f'v2_field_held_site_{arm}'
        if not folder.exists():
            continue
        r = json.loads((folder / 'report.json').read_text())
        arms[arm] = {**r['nested_mae'], 'alpha': entry['alpha'], 'trials': entry['trials']}
    write_json(C1 / '20260919T145045Z_v2_field/v2_field_summary.json',
               {'arms': arms, 'contrasts': 'see C1/v2_field_contrasts.json (cast-date bootstrap, the one the thesis prints)',
                'not_rerun': sorted(set(pub['arms']) - set(arms)),
                'why_not_rerun': 'A2-A4 with the utilisation target appear in no thesis table or sentence'})
    record['pass'] = all(s['pass'] for s in record['steps'].values())
    write_json(C1 / 'field57_contrasts_reproduction.json', record)
    print(f'done in {time.time() - t0:.0f} s')


#: the Field-57 oxygen contrasts that read the cast prior
CAST_PRIOR = ('A6_ctdprior_direct', 'A7_ctdprior_direct')


def status(c):
    """Interval status of a contrast on the primary unit: the side of zero it excludes, or none."""
    d = c['date']
    return (1 if d['ci'][0] > 0 else -1) if d['excludes_zero'] else 0


def a69():
    """Addendum 69: every Field-57 contrast with the naive comparators of the per-fold model and the
    inner-split cast prior, in the record unit (runs/v2_field_contrasts_a69.json, the sources of
    scripts/v2_field_contrasts) and in mg/l (C1/v2_field_contrasts_a69.json).

    The fast bootstrap first reproduces runs/v2_field_contrasts.json from the published files
    (tolerance zero). In mg/l the per-fold model and no-boundary ridge are addendum 60's per-fold
    reruns (C1/v2_perfold/field/perfold/), the parameter-free comparators are unchanged. The cast-prior
    oxygen contrasts stay the mg/l reruns of the shared-prior design unless one of them changes
    interval status in the record unit with the inner-split prior (the addendum's decision); then
    every cast-prior oxygen contrast is the record-unit value times c, flagged as converted."""
    from scripts import v2_field_contrasts as V
    from src.units import factor
    camp_pub = RUNS / '20260919T145045Z_v2_field/results'
    pub = json.loads((RUNS / 'v2_field_contrasts.json').read_text())
    got = field_contrasts(sources(camp_pub, ROOT / 'results', ROOT / 'results', RUNS / 'v2_field_naive'),
                          SCRATCH / 'v2_field_contrasts.json')
    diff = same(got, pub, tol=0)
    print('contrast reproduction', 'ok' if not diff else diff[:5], flush=True)
    if diff:
        raise SystemExit('v2_field_contrasts does not reproduce')
    if V.SHARED:
        raise SystemExit('run without --shared')
    rec = field_contrasts(V.SOURCES, SCRATCH / 'v2_field_contrasts_a69.json')
    rec['addendum69'] = {'naive_comparators': 'per-fold model and no-boundary ridge (runs/v2_field_naive/perfold/)',
                         'cast_prior': 'inner-split prior (results/v2_field_held_site_A{6,7}_ctdprior_direct_innerprior/)',
                         'placebo': 'addendum 71: P1 fitted inside each inner split '
                                    '(results/v2_field_held_site_P1_placebo_direct_innerprior/)',
                         'sources': {k: Path(v).relative_to(ROOT).as_posix() for k, v in V.SOURCES.items()}}
    write_json(V.OUT, rec)
    old = {(c['arm'], c['reference'], c['target']): c for c in pub['contrasts']}
    new = {(c['arm'], c['reference'], c['target']): c for c in rec['contrasts']}
    prior_o2 = [k for k in new if k[2] == 'do_umol_l' and (k[0] in CAST_PRIOR or k[1] in CAST_PRIOR)]
    changed = [k for k in prior_o2 if status(new[k]) != status(old[k])]

    camp_mgl = C1 / '20260919T145045Z_v2_field/results'
    S = sources(camp_mgl, C1 / 'results', C2 / 'results', C1 / 'v2_field_naive')
    S['naive_model'] = C1 / 'v2_perfold/field/perfold/A1_both_direct/selected_predictions.csv.gz'
    S['naive_no_copernicus'] = C1 / 'v2_perfold/field/perfold/A2_none_direct/selected_predictions.csv.gz'
    mgl = field_contrasts(S, SCRATCH / 'v2_field_contrasts_a69_mgl.json')
    c = factor()
    if changed:
        for e in mgl['contrasts']:
            k = (e['arm'], e['reference'], e['target'])
            if k in prior_o2:
                for unit in ('date', 'cast'):
                    r = new[k][unit]
                    e[unit] = {**r, 'point': round(r['point'] * c, 6), 'ci': [round(x * c, 6) for x in r['ci']]}
                e['converted_from_record_unit'] = True
    mgl['addendum69'] = {
        'naive_comparators': 'per-fold model and no-boundary ridge in mg/l (C1/v2_perfold/field/perfold/)',
        'cast_prior_oxygen': ('the inner-split record-unit contrasts times c' if changed else
                              'the mg/l reruns of the shared-prior design (C2): no cast-prior oxygen contrast '
                              'changes interval status with the inner-split prior'),
        'status_changes': [{'contrast': list(k), 'shared_prior': old[k]['date'], 'inner_prior': new[k]['date']}
                           for k in changed],
        'factor': c, 'sources': {k: Path(v).relative_to(ROOT).as_posix() for k, v in S.items()}}
    write_json(C1 / 'v2_field_contrasts_a69.json', mgl)
    print('cast-prior oxygen status changes:', changed or 'none')
    print('wrote', V.OUT.relative_to(ROOT), (C1 / 'v2_field_contrasts_a69.json').relative_to(ROOT))


def scaled_oxygen(entry, c):
    """A Field-57 arm or contrast entry of addenda 47/48 with every oxygen quantity times c."""
    def scale(x):
        if isinstance(x, bool) or x is None or isinstance(x, str):
            return x
        if isinstance(x, (int, float)):
            return x * c
        if isinstance(x, list):
            return [scale(v) for v in x]
        if isinstance(x, dict):
            return {k: scale(v) for k, v in x.items()}
        return x

    def walk(x, oxygen=False):
        if isinstance(x, dict):
            return {k: (walk(v, True) if k == 'do_umol_l' and not isinstance(v, str) else walk(v, oxygen))
                    for k, v in x.items()}
        if oxygen:
            return scale(x)
        return x
    out = json.loads(json.dumps(entry))
    if out.get('target') == 'do_umol_l':          # a contrast: every interval is oxygen
        for k, v in out.items():
            if isinstance(v, dict) and 'value' in v:
                v['value'] = scale(v['value'])
            elif k.startswith('movement'):
                out[k] = scale(v)
        return out
    return walk(out)                               # an arm: the oxygen sub-entries only


#: addendum 71: what the per-fold Field-57 section now reads for the fitted depth control
A71_NOTE = ('Addendum 71: P1 (the fitted depth control) under every design is read from its runs with the basis '
            'fitted inside each inner split (results/v2_field_held_site_P1_placebo_direct_innerprior/, '
            'runs/v2_perfold_rest/field_innerprior/{shared,perfold}/P1_placebo_direct/).')


def a69_perfold():
    """Addendum 69: the Field-57 sections of addenda 47 and 48 with the cast prior rebuilt per inner
    split (A7 in v2_perfold_contrasts, A6 in v2_perfold_rest_contrasts), every other section copied.

    Gate G4b reproduces runs/v2_field_contrasts_a69.json's sources at four decimals, the precision it
    was written against (gate_root). In mg/l (C1/v2_perfold_a69.json, C1/v2_perfold_rest_a69.json)
    the cast-prior arms and contrasts are the record-unit values times c when a cast-prior oxygen
    contrast changed interval status (C1/v2_field_contrasts_a69.json records it), every other entry
    the mg/l rerun's."""
    from scripts import v2_field_contrasts as V
    from scripts import v2_perfold_contrasts as P
    from scripts import v2_perfold_rest_contrasts as Q
    from src.units import factor
    if V.SHARED:
        raise SystemExit('run without --shared')
    gdir = SCRATCH / 'gate_a69'
    # the gate reproduces the 'published' design: the inner-split cast prior, and the naive comparators
    # of the released model at its one shared penalty (the per-fold model is a design of its own here)
    srcs = {**V.SOURCES, **{k: RUNS / 'v2_field_naive' / f"pred_{k.removeprefix('naive_')}.csv.gz"
                            for k in V.SOURCES if k.startswith('naive_')}}
    gate_root(srcs, RUNS / 'v2_field_naive', gdir)
    converted = bool(json.loads((C1 / 'v2_field_contrasts_a69.json').read_text())['addendum69']['status_changes'])
    c = factor()
    note = {'cast_prior': 'rebuilt per inner split (v2_field_ctdprior, v2_perfold, v2_perfold_rest --inner-prior)',
            'gate_G4b_reference': (gdir / 'v2_field_contrasts.json').relative_to(ROOT).as_posix()}
    # addendum 71: the fitted depth control P1 is read from its inner-split runs beside A6
    for module, arm, name, old_mgl in ((P, 'A7_ctdprior_direct', 'v2_perfold', C1 / 'v2_perfold.json'),
                                       (Q, 'A6_ctdprior_direct', 'v2_perfold_rest', C1 / 'v2_perfold_rest.json')):
        inner = {arm, 'P1_placebo_direct'} if module is Q else {arm}
        saved = (module.INNER_PRIOR, module.FIELD_CONTRASTS_PATH)
        module.INNER_PRIOR, module.FIELD_CONTRASTS_PATH = inner, gdir / 'v2_field_contrasts.json'
        try:
            fd = module.field()
        finally:
            module.INNER_PRIOR, module.FIELD_CONTRASTS_PATH = saved
        fd = json.loads(json.dumps(fd))
        if not all(g['pass'] for g in fd['gates'].values()):
            print(json.dumps(fd['gates'], indent=1)[:3000])
            raise SystemExit(f'{name}: a Field-57 gate failed')
        old = json.loads((RUNS / f'{name}.json').read_text())
        new = {**old, 'field57': fd, 'addendum69': {**note, 'arm': arm}}
        if module is P:
            new['registered_predictions'] = P.predictions(old['forecast'], fd)
            rules = dict(old['decision_rules'])
            rules['D4_triggered'] = [f"{e['contrast']}|{e['target']}" for e in fd['contrasts'] if e['D4_triggered']]
            rules['D5_table'] = bool(rules['D2_triggered'] or rules['D3_triggered'] or rules['D4_triggered'])
            new['decision_rules'] = rules
        else:
            new['registered_predictions'] = Q.predictions(old['forecast'], old['aou'], fd)
            changed = [f"{e['variable']}|{e['horizon']}|{e['contrast']}|{e['subgroup']}"
                       for e in old['forecast']['contrasts'] if e['D2_verdict_changed']]
            changed += [f"aou|{e['contrast']}" for e in old['aou']['contrasts'] if e['D2_verdict_changed']]
            changed += [f"field|{e['contrast']}|{e['target']}" for e in fd['contrasts'] if e['D2_verdict_changed']]
            new['decision_rules'] = {**old['decision_rules'], 'D2_changed': changed}
            new['gates'] = {**{k: v for k, v in old['gates'].items() if not k.startswith('field ')},
                            **{f'field {k}': v['pass'] for k, v in fd['gates'].items()}}
            new['addendum71'] = A71_NOTE
        write_json(RUNS / f'{name}_a69.json', new)
        mgl = json.loads(old_mgl.read_text())
        if converted:
            prior = ('A6_ctdprior_direct', 'A7_ctdprior_direct')
            mgl['field57']['arms'][arm] = scaled_oxygen(fd['arms'][arm], c)
            mine = {(e['contrast'], e['target']): e for e in fd['contrasts']}
            mgl['field57']['contrasts'] = [
                scaled_oxygen(mine[(e['contrast'], e['target'])], c)
                if e['target'] == 'do_umol_l' and (e['arm'] in prior or e['reference'] in prior) else e
                for e in mgl['field57']['contrasts']]
        mgl['addendum69'] = {**note, 'arm': arm, 'cast_prior_oxygen': (
            'the inner-split record-unit entries times c' if converted else 'the mg/l reruns of the shared-prior design'),
            'factor': c}
        if module is Q:
            mgl['addendum71'] = A71_NOTE + (' In this mg/l file the P1 entries remain the mg/l rerun of the '
                                            'published construction: P1 is compared on temperature only, which '
                                            'the tables read from the record-unit file.')
        write_json(C1 / f'{name}_a69.json', mgl)
        print('wrote', (RUNS / f'{name}_a69.json').relative_to(ROOT), (C1 / f'{name}_a69.json').relative_to(ROOT))


if __name__ == '__main__':
    import sys
    if '--a69' in sys.argv[1:]:
        a69()
    elif '--a69-perfold' in sys.argv[1:]:
        a69_perfold()
    else:
        main()
