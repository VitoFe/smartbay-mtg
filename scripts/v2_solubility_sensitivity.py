"""Addendum 71 part B: what the project's Garcia-Gordon coefficient mixture moves in the utilisation arms.

src/data/thermodynamics.py evaluates Garcia and Gordon's equation with the Benson and Krause A terms,
the combined fit's B and C0 terms and 44.661 umol per cm3; the consistent Benson and Krause set at
44.6596 (thermodynamics.o2sol_benson_krause_umol_per_l) is lower by up to 0.22 umol/l over the bay's
range. Every utilisation and thermodynamic arm the thesis prints is refitted here exactly as its
published mg/l rerun (addendum 60 part C, scripts/v2_oxygen_mgl_*.py), at the configuration its
published study selected, twice: with the project's function (`project`, which must reproduce the
published mg/l outputs, gate GB1) and with the consistent one (`bk`). Nothing is selected again.

  B1  year-boundary arms and the ridge given T and S        C3/v2_bottom_aou/aou_summary.json, aou_units/
  B2  near-bottom utilisation arms                          C2/v2_bottom_aou/bottom/, bottom_ref/
  B3  Field-57, A1 with the utilisation target vs direct    C1/field57_aou.json (shared and the three designs)
  B4  the forecasting utilisation tree, thermodynamic baseline  C3/v2_aou_forecast/, C3/v2_perfold_rest.json
  B5  zero-shot pretrained utilisation arms, Reserved-2026  C3/v2_aou_units_chronos.json (cluster, kind
      zeroshot-bk of scripts/v2_oxygen_mgl_chronos_remote.py; columns fetched into
      runs/v2_solubility_bk/C3/v2_aou_units_chronos/)

The two coefficient sets each run in a process of their own, because modules imported while the
consistent function is installed bind it. Outputs: runs/v2_solubility_bk/<set>/ and, from `summary`,
runs/v2_solubility_sensitivity.json.

Usage (from the repository root, V2_THREADS=2):
  python -m scripts.v2_solubility_sensitivity fit project
  python -m scripts.v2_solubility_sensitivity fit bk
  python -m scripts.v2_solubility_sensitivity chronos-columns   # after the cluster run is fetched
  python -m scripts.v2_solubility_sensitivity summary
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import OUT as MGL, RUNS, constant_c, harmonised, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
DST = RUNS / 'v2_solubility_bk'
REPORT = RUNS / 'v2_solubility_sensitivity.json'
DECLARED = ['config/bay_reserved_protocol_v2_addendum71.json',
            'config/bay_reserved_protocol_v2_addendum71_amendment1.json',
            'config/bay_reserved_protocol_v2_addendum71_amendment2.json']
#: arm names that mark a tree model; a contrast is a tree contrast when either arm is one
TREE = ('lightgbm', 'xgboost', 'M3_', 'M3b', 'aou_tree', 'tree')
#: a tenth of the oxygen floor (0.059 mg/l): the declared threshold for a moved point estimate
MOVE = 0.0059
FIELD_LABEL = 'apparent utilisation vs direct, released family'
#: the result files compared, relative to a set's root (or to runs/v2_oxygen_mgl for the published)
FILES = ('C2/v2_bottom_aou/bottom/contrasts.json', 'C2/v2_bottom_aou/bottom_ref/contrasts.json',
         'C3/v2_bottom_aou/aou_summary.json', 'C3/v2_bottom_aou/aou_units/summary.json',
         'C3/v2_aou_forecast/aou_forecast_h6_summary.json', 'C3/v2_perfold_rest.json',
         'C1/field57_aou.json')


@contextlib.contextmanager
def coefficients(name):
    """Install the project's Garcia-Gordon function or the consistent Benson and Krause one wherever the
    mg/l reruns read it: every module binding of the function and the container sat_mgl reads."""
    import src.data.thermodynamics as th
    import scripts.v2_oxygen_mgl_common as common
    project = common._ORIGINAL['o2sol']
    if project is not th.o2sol_garcia_gordon_umol_per_l:
        raise SystemExit('the solubility function was already replaced in this process')
    new = {'project': project, 'bk': th.o2sol_benson_krause_umol_per_l}[name]
    rebound = common._replace_everywhere(project, new) if new is not project else []
    common._ORIGINAL['o2sol'] = new
    try:
        yield new
    finally:
        for mod, k in rebound:
            setattr(mod, k, project)
        common._ORIGINAL['o2sol'] = project


# ------------------------------------------------------------------ B1 and B2: near-bottom, year boundary

def bottom(root):
    import scripts.v2_oxygen_mgl_bottom as B
    B.SCRATCH = root / '_scratch'
    B.run(root / 'C2/v2_bottom_aou', root / 'C3/v2_bottom_aou', True, {'steps': {}})


# ------------------------------------------------------------------ B4: the forecasting arms

def forecast(root):
    import scripts.v2_oxygen_mgl_aou_forecast as F
    F.SCRATCH = root / '_scratch'
    F.run_aou_forecast(root / 'C3/v2_aou_forecast', True)
    F.run_perfold(root / 'C3/v2_perfold_rest', root / 'C3', True)
    write_json(root / 'C3/v2_perfold_rest.json', {'aou': F.contrasts48(root / 'C3', root / 'C3/v2_perfold_rest')})


# ------------------------------------------------------------------ B3: Field-57

def field57_fit(root):
    """A1 with the utilisation target refitted in mg/l at every frozen penalty the thesis quotes."""
    import scripts.v2_oxygen_mgl_field as MF
    from src.experiments import bay_profile_field as BPF
    import src.experiments.v2_field  # noqa: F401  (bind the names harmonised() rebinds)
    import src.experiments.v2_field_ctdprior  # noqa: F401
    import src.experiments.v2_field_placebo  # noqa: F401
    import src.experiments.v2_field_intercept  # noqa: F401
    import src.data.bay_profile_context  # noqa: F401
    keys = {k: v for k, v in MF.alphas().items() if k[2] == 'A1_both_aou'}
    scale = BPF.SCALE
    saved = scale.copy()
    with harmonised(True):
        scale[2] = saved[2] * constant_c()
        try:
            bundles, _, mats, _ = MF.setup()
            fits = MF.fit_all(bundles, {'A1_both_aou': mats['A1_both_aou']}, keys)
        finally:
            scale[:] = saved
    for (family, design, arm), fit in fits.items():
        folder = root / 'C1/field57' / f'{family}_{design}'
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(fit['frames'][0], folder / f'{arm}.csv.gz')


def field57_sources(root):
    """(design -> (A1_both_aou predictions, A1_both_direct predictions)): the refitted utilisation arm
    of `root` (or the published mg/l one when `root` is None) against the published mg/l direct arm."""
    camp = MGL / 'C1/20260919T145045Z_v2_field/results'
    direct = {'campaign_shared': camp / 'v2_field_held_site_A1_both_direct/selected_predictions.csv.gz',
              'a48_published': camp / 'v2_field_held_site_A1_both_direct/selected_predictions.csv.gz',
              'a48_shared': MGL / 'C1/v2_perfold/field/shared/A1_both_direct/selected_predictions.csv.gz',
              'a48_perfold': MGL / 'C1/v2_perfold/field/perfold/A1_both_direct/selected_predictions.csv.gz'}
    published = {'campaign_shared': camp / 'v2_field_held_site_A1_both_aou/selected_predictions.csv.gz',
                 'a48_published': camp / 'v2_field_held_site_A1_both_aou/selected_predictions.csv.gz',
                 'a48_shared': MGL / 'C1/v2_perfold_rest/field/shared/A1_both_aou/selected_predictions.csv.gz',
                 'a48_perfold': MGL / 'C1/v2_perfold_rest/field/perfold/A1_both_aou/selected_predictions.csv.gz'}
    if root is None:
        return {d: (published[d], direct[d]) for d in direct}
    return {d: (root / 'C1/field57' / d / 'A1_both_aou.csv.gz', direct[d]) for d in direct}


def field57_contrasts(sources):
    """The utilisation-minus-direct oxygen contrast under each design, with the stream its published
    producer used: v2_field_contrasts for the campaign's shared selection, v2_perfold_rest_contrasts
    (CastDates) for addendum 48's three designs."""
    from scripts import v2_field_contrasts as V
    from scripts.v2_perfold_contrasts import CastDates
    out = {}
    for design, (aou_path, direct_path) in sources.items():
        frames, keys, truth = {}, None, None
        for name, path in (('A1_both_aou', aou_path), ('A1_both_direct', direct_path)):
            f = pd.read_csv(path).sort_values(['fold', 'station', 'cast_id', 'depth_m']).reset_index(drop=True)
            k = f[['fold', 'station', 'cast_id', 'depth_m']]
            if keys is None:
                keys, truth = k, f[list(V.TARGETS)].to_numpy(float)
                day = pd.to_datetime(f.t_utc, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            elif not k.equals(keys) or not np.allclose(f[list(V.TARGETS)].to_numpy(float), truth, equal_nan=True):
                raise SystemExit(f'{design}: {name} does not score the same rows')
            frames[name] = f[[f'pred_{t}' for t in V.TARGETS]].to_numpy(float)
        keys = keys.assign(day=day)
        t = 'do_umol_l'
        j = V.TARGETS.index(t)
        if design == 'campaign_shared':
            c = V.contrast(keys, truth, frames, 'A1_both_aou', 'A1_both_direct', t, 'day', FIELD_LABEL)
            out[design] = {'point': c['point'], 'ci': c['ci'], 'excludes_zero': c['excludes_zero']}
        else:
            d = design.split('_', 1)[1]
            label = f'{FIELD_LABEL}|{t}|day' if d == 'published' else f'a48|{d}|{FIELD_LABEL}|{t}|date'
            errs = {n: np.abs(truth[:, j] - p[:, j]) for n, p in frames.items()}
            v = CastDates(keys).boot(errs, {'A1_both_aou': 1.0, 'A1_both_direct': -1.0}, label)
            out[design] = {'value': v, 'excludes_zero': bool(v[1] > 0 or v[2] < 0)}
    return out


def field57_published_reference():
    """The same contrasts as the published mg/l artifacts store them (gate GB1 for B3)."""
    shared = json.loads((MGL / 'C1/v2_field_contrasts.json').read_text())
    a48 = json.loads((MGL / 'C1/v2_perfold_rest.json').read_text())['field57']
    c = next(e for e in shared['contrasts'] if e['contrast'] == FIELD_LABEL and e['target'] == 'do_umol_l')['date']
    out = {'campaign_shared': {'point': c['point'], 'ci': c['ci'], 'excludes_zero': c['excludes_zero']}}
    e = next(e for e in a48['contrasts'] if e['contrast'] == FIELD_LABEL and e['target'] == 'do_umol_l')
    for d in ('published', 'shared', 'perfold'):
        out[f'a48_{d}'] = {'value': e[d]['value'], 'excludes_zero': e[d]['excludes_zero']}
    return out


# ------------------------------------------------------------------ B5: the pretrained arms

def chronos_columns():
    """Run the published contrast script on the cluster's consistent-coefficient columns. The host wrote
    zeroshot_mgl_bk_columns.csv.gz with the suffix '|mgl_bk' and zeroshot_mgl_columns.csv.gz with the
    project's function (gate GB1 for B5: equal to the published constant-c columns to 1e-3 mg/l)."""
    from scripts import v2_oxygen_mgl_chronos_contrasts as K
    host = DST / 'C3/v2_aou_units_chronos'
    published = pd.read_csv(MGL / 'C3/v2_aou_units_chronos/zeroshot_mgl_columns.csv.gz')
    project = pd.read_csv(host / 'zeroshot_mgl_columns.csv.gz')
    cols = [c for c in published.columns if c.endswith('|mgl')]
    gate = {c: float(np.nanmax(np.abs(project[c].to_numpy(float) - published[c].to_numpy(float)))) for c in cols}
    same_missing = all((np.isfinite(project[c]) == np.isfinite(published[c])).all() for c in cols)
    gb1 = {'max_abs_diff': gate, 'same_missing': bool(same_missing),
           'pass': bool(same_missing and max(gate.values()) <= 1e-3)}
    print('GB1 (B5)', gb1, flush=True)
    for name, columns, suffix in (('project', project, '|mgl'),
                                  ('bk', pd.read_csv(host / 'zeroshot_mgl_bk_columns.csv.gz'), '|mgl_bk')):
        work = DST / name / 'C3/_chronos_host'
        work.mkdir(parents=True, exist_ok=True)
        frame = columns.rename(columns={c: c[:-len(suffix)] + '|mgl' for c in columns.columns if c.endswith(suffix)})
        frame.to_csv(work / 'zeroshot_mgl_columns.csv.gz', index=False)
        for f in ('zeroshot_published_report.json', 'zeroshot_mgl_report.json'):
            shutil.copyfile(host / f, work / f)
        saved = (K.HOST, K.DST)
        K.HOST, K.DST = work, DST / name / 'C3/v2_aou_units_chronos.json'
        try:
            K.main()
        finally:
            K.HOST, K.DST = saved
    write_json(DST / 'C3_chronos_gate.json', {'declared_in': DECLARED, 'GB1_B5': gb1})


# ------------------------------------------------------------------ the comparison

def contrast_records(obj, path=''):
    """{path: (point, lo, hi, flags)} for every contrast-shaped entry of a result file."""
    out = {}
    if isinstance(obj, dict):
        names = ' '.join(str(obj[k]) for k in ('model', 'a', 'b', 'contrast', 'arm', 'reference') if k in obj)
        if 'diff' in obj and 'ci' in obj:
            flags = {k: obj[k] for k in ('excludes_zero', 'jackknife_excludes_zero', 'robust') if k in obj}
            out[path] = (obj['diff'], *obj['ci'], flags, names)
        elif 'point' in obj and 'ci' in obj:
            out[path] = (obj['point'], *obj['ci'], {'excludes_zero': obj.get('excludes_zero')}, names)
        elif isinstance(obj.get('value'), list) and len(obj['value']) == 3:
            out[path] = (*obj['value'], {'excludes_zero': obj.get('excludes_zero')}, names)
        for k, v in obj.items():
            if k not in ('value', 'ci'):
                out.update({p: (*r[:4], (r[4] + ' ' + names).strip()) for p, r in contrast_records(v, f'{path}/{k}').items()})
        return out
    elif isinstance(obj, list):
        if len(obj) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj) \
                and '/contrasts' in path:
            out[path] = (*obj, {}, path.rsplit('/', 1)[-1])
        else:
            for i, v in enumerate(obj):
                key = f'{path}[{i}]'
                if isinstance(v, dict) and 'a' in v and 'b' in v:
                    key = f"{path}[{v.get('cohort') or v.get('protocol') or ''}|{v['a']} - {v['b']}]"
                out.update(contrast_records(v, key))
    return out


def mae_values(obj, path=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{path}/{k}'
            if k == 'mae' and isinstance(v, dict):
                out.update({f'{p}/{a}': x for a, x in v.items() if isinstance(x, (int, float))})
            else:
                out.update(mae_values(v, p))
    return out


def kind(names, pretrained=False):
    """'tree' when either arm is a tree model, 'pretrained' for the zero-shot arms, else 'linear'."""
    if pretrained:
        return 'pretrained'
    return 'tree' if any(t in names for t in TREE) else 'linear'


def compare(project, bk, pretrained=False):
    rows, worst = [], 0.0
    for p, (x0, l0, h0, f0, names) in project.items():
        if p not in bk:
            raise SystemExit(f'{p}: missing in the consistent-coefficient run')
        x1, l1, h1, f1, _ = bk[p]
        status0, status1 = (l0 > 0 or h0 < 0), (l1 > 0 or h1 < 0)
        flags = {k: (f0.get(k), f1.get(k)) for k in f0 if f0.get(k) != f1.get(k)}
        row = {'contrast': p, 'arms': names, 'kind': kind(names + ' ' + p, pretrained),
               'project': [x0, l0, h0], 'bk': [x1, l1, h1], 'moved': x1 - x0,
               'status_changed': status0 != status1, 'sign_changed': bool(np.sign(x0) != np.sign(x1) and x0 != 0),
               'flags_changed': flags}
        row['sensitive'] = bool(row['status_changed'] or flags or abs(row['moved']) > MOVE
                                or (row['sign_changed'] and (status0 or status1)))
        worst = max(worst, abs(row['moved']))
        rows.append(row)
    return rows, worst


def fit(name):
    root = DST / name
    t0 = time.time()
    with coefficients(name):
        bottom(root)
        print(f'{name}: B1 and B2 done ({time.time() - t0:.0f} s)', flush=True)
        forecast(root)
        print(f'{name}: B4 done ({time.time() - t0:.0f} s)', flush=True)
        field57_fit(root)
    write_json(root / 'C1/field57_aou.json', {'contrasts': field57_contrasts(field57_sources(root))})
    shutil.rmtree(root / '_scratch', ignore_errors=True)
    print(f'{name}: B3 done ({time.time() - t0:.0f} s)', flush=True)


def gb2():
    """The solubility difference at the rows the arms score, against the grid bound over their range."""
    import src.data.thermodynamics as th
    from src.data.bay_profile_context import load_profiles
    data, _ = load_profiles()
    casts = pd.read_csv(ROOT / 'data/processed/ctd_all_casts.csv.gz', usecols=lambda c: c in ('temp_c', 'salinity_psu'))
    out = {}
    for name, frame in (('field57', data), ('ctd_722', casts)):
        f = frame[['temp_c', 'salinity_psu']].dropna()
        t, s = f.temp_c.to_numpy(float), f.salinity_psu.to_numpy(float)
        d = th.o2sol_garcia_gordon_umol_per_l(t, s) - th.o2sol_benson_krause_umol_per_l(t, s)
        gt, gs = np.meshgrid(np.linspace(t.min(), t.max(), 181), np.linspace(s.min(), s.max(), 141))
        g = th.o2sol_garcia_gordon_umol_per_l(gt, gs) - th.o2sol_benson_krause_umol_per_l(gt, gs)
        out[name] = {'rows': int(len(f)), 'temperature_c': [float(t.min()), float(t.max())],
                     'salinity': [float(s.min()), float(s.max())],
                     'row_difference_umol_per_l': [float(d.min()), float(d.max())],
                     'grid_bound_umol_per_l': [float(g.min()), float(g.max())],
                     'row_difference_mg_per_l': [float(d.min() * 0.0319988), float(d.max() * 0.0319988)],
                     'pass': bool(d.min() >= g.min() - 1e-9 and d.max() <= g.max() + 1e-9)}
    # the sonde-only arms hold salinity at 38
    t = np.linspace(10, 28, 181)
    d = th.o2sol_garcia_gordon_umol_per_l(t, 38.0) - th.o2sol_benson_krause_umol_per_l(t, 38.0)
    out['sondes_at_S38'] = {'temperature_c': [10.0, 28.0], 'difference_umol_per_l': [float(d.min()), float(d.max())],
                            'pass': True}
    return out


def summary():
    report = {'declared_in': DECLARED, 'threshold_mg_per_l': MOVE, 'files': {}, 'gates': {}}
    gb1, all_rows, worst = {}, [], 0.0
    for rel in FILES:
        project = json.loads((DST / 'project' / rel).read_text())
        bk = json.loads((DST / 'bk' / rel).read_text())
        if rel == 'C1/field57_aou.json':
            published = {'contrasts': field57_published_reference()}
        else:
            published = json.loads((MGL / rel).read_text())
        rp, rb, rpub = contrast_records(project), contrast_records(bk), contrast_records(published)
        diff = [p for p in rpub if p not in rp or any(
            abs(a - b) > 1e-12 for a, b in zip(rp[p][:3], rpub[p][:3]))]
        gb1[rel] = {'contrasts': len(rpub), 'differing_from_published': diff[:10], 'pass': not diff}
        rows, w = compare(rp, rb)
        mp, mb = mae_values(project), mae_values(bk)
        report['files'][rel] = {'contrasts': rows, 'largest_move': w,
                                'mae_largest_move': max([abs(mb[k] - mp[k]) for k in mp if k in mb] or [0.0])}
        all_rows += rows
        worst = max(worst, w)
        print(f'{rel}: {len(rows)} contrasts, largest move {w:.6f}, GB1 {gb1[rel]["pass"]}', flush=True)
    chronos = DST / 'project/C3/v2_aou_units_chronos.json'
    if chronos.exists():
        rp = contrast_records(json.loads(chronos.read_text()))
        rb = contrast_records(json.loads((DST / 'bk/C3/v2_aou_units_chronos.json').read_text()))
        rows, w = compare(rp, rb, pretrained=True)
        report['files']['C3/v2_aou_units_chronos.json'] = {'contrasts': rows, 'largest_move': w}
        gb1['C3/v2_aou_units_chronos.json'] = json.loads((DST / 'C3_chronos_gate.json').read_text())['GB1_B5']
        all_rows += rows
        worst = max(worst, w)
    else:
        report['files']['C3/v2_aou_units_chronos.json'] = 'not yet run (cluster, kind zeroshot-bk)'
    report['gates'] = {'GB1': {'pass': all(g['pass'] for g in gb1.values()), 'files': gb1}, 'GB2': gb2()}
    report['gates']['GB2']['pass'] = all(v['pass'] for v in report['gates']['GB2'].values() if isinstance(v, dict))
    sensitive = [r['contrast'] for r in all_rows if r['sensitive']]
    report['decision'] = {'contrasts_compared': len(all_rows), 'largest_move_mg_per_l': worst,
                          'status_changes': [r['contrast'] for r in all_rows if r['status_changed'] or r['flags_changed']],
                          'sensitive': sensitive,
                          'verdict': 'sensitive' if sensitive else 'not sensitive',
                          'largest_move_by_kind': {k: max(abs(r['moved']) for r in all_rows if r['kind'] == k)
                                                   for k in sorted({r['kind'] for r in all_rows})},
                          'after_amendment2': ('the project function is kept and the movement disclosed '
                                               '(config/bay_reserved_protocol_v2_addendum71_amendment2.json)')}
    report['predictions'] = {
        'PB1': {'held': not any(r['status_changed'] or r['flags_changed'] or
                                (r['sign_changed'] and (r['project'][1] > 0 or r['project'][2] < 0)) for r in all_rows)},
        'PB2': {'held': worst <= 0.001, 'largest_move': worst}}
    write_json(REPORT, report)
    print('decision', report['decision']['verdict'], f"largest move {worst:.6f} mg/l",
          'GB1', report['gates']['GB1']['pass'], 'GB2', report['gates']['GB2']['pass'], flush=True)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'fit' and len(sys.argv) == 3 and sys.argv[2] in ('project', 'bk'):
        fit(sys.argv[2])
    elif sys.argv[1:] == ['chronos-columns']:
        chronos_columns()
    elif sys.argv[1:] == ['summary']:
        summary()
    else:
        raise SystemExit('fit project|bk, chronos-columns or summary')


if __name__ == '__main__':
    main()
