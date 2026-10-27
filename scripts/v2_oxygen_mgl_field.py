"""Addendum 60 parts C1 and C2: the Field-57 arms refitted with every oxygen input in mg/l.

Field-57 scores casts of 2024-07 to 2025-12 (56 per-kilogram casts and the one per-litre '%' cast)
from moored sondes (per kilogram) and, in every arm but A2, the Copernicus product (per litre); the
cast-derived prior (A6, A7) also reads the 722-cast archive (219 per-kilogram casts, 502 per-litre).
Each arm is refitted here at the penalty its published study selected, under every design the thesis
quotes (the shared selection of the campaign, and the published / shared / per-fold designs of
addenda 47 and 48), first in the published unit, where it must reproduce the published predictions,
and then with every oxygen input converted to mg/l by its own source unit
(scripts/v2_oxygen_mgl_common.py). Nothing is selected again. The objective's oxygen scale (20 umol/L)
is carried into mg/l by c; it does not enter a ridge prediction, and it sets the amplitude of the
random placebo P2.

Arms: A1-A4 direct and A1 with the utilisation target (the campaign), A5 (fitted intercept),
A6 and A7 (the cast prior), P1 and P2 (the placebo). Outputs mirror the published files under
reports/bay_twin/runs/v2_oxygen_mgl/ (C1: the Copernicus arms, the per-fold designs and the
reproduction record; C2: the cast-prior and placebo arms):

  C1/20260919T145045Z_v2_field/results/v2_field_held_site_<arm>/   report.json, selected_predictions.csv.gz
  C1/results/v2_field_held_site_A5_intercept_direct/
  C2/results/v2_field_held_site_{A6,A7}_ctdprior_direct/, {P1,P2}_placebo_direct/
  C1/v2_perfold/field/<design>/<arm>/                               addendum 47 arms
  C1/v2_perfold_rest/field/<design>/<arm>/                          addendum 48 arms
  C1/field57_reproduction.json                                      the published-unit gate

Usage: python -m scripts.v2_oxygen_mgl_field
"""
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from scripts.v2_oxygen_mgl_common import (OUT, ROOT, RUNS, constant_c, harmonised, write_csv, write_json)

FIELD_RUN = RUNS / '20260919T145045Z_v2_field/results'
LOCAL = ROOT / 'results'
A47 = RUNS / 'v2_perfold'
A48 = RUNS / 'v2_perfold_rest'
C1, C2 = OUT / 'C1', OUT / 'C2'
CAMPAIGN = ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct', 'A4_base_only_direct', 'A1_both_aou')
LOCAL_ARMS = ('A5_intercept_direct', 'A6_ctdprior_direct', 'A7_ctdprior_direct', 'P1_placebo_direct',
              'P2_placebo_direct')
A47_ARMS = ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct', 'A4_base_only_direct',
            'A7_ctdprior_direct')
A48_ARMS = ('A5_intercept_direct', 'A6_ctdprior_direct', 'P1_placebo_direct', 'P2_placebo_direct', 'A1_both_aou')
DESIGNS = ('published', 'shared', 'perfold')
TOL_REL = 1e-5          # the published files were written on the cluster, in float32 designs


def published_source(arm):
    return (FIELD_RUN if arm in CAMPAIGN else LOCAL) / f'v2_field_held_site_{arm}'


def mirror_source(arm):
    if arm in CAMPAIGN:
        return C1 / '20260919T145045Z_v2_field/results' / f'v2_field_held_site_{arm}'
    return (C1 if arm == 'A5_intercept_direct' else C2) / 'results' / f'v2_field_held_site_{arm}'


def alphas():
    """{(family, design, arm): {fold: alpha}} from the published selections and design reports."""
    out = {}
    for arm in CAMPAIGN + LOCAL_ARMS:
        sel = json.loads((published_source(arm) / f'held_site_{arm}_selection.json').read_text())
        out[('campaign', 'shared', arm)] = {0: sel['params']['alpha'], 1: sel['params']['alpha']}
    for family, root, arms in (('a47', A47, A47_ARMS), ('a48', A48, A48_ARMS)):
        for d in DESIGNS:
            for arm in arms:
                r = json.loads((root / 'field' / d / arm / 'report.json').read_text())
                out[(family, d, arm)] = {int(k): float(v) for k, v in r['alpha_by_fold'].items()}
    return out


def setup():
    """Bundles and every arm's matrices (per draw), as the published producers build them."""
    from src.data.bay_profile_context import TARGETS, load_profiles
    from src.experiments.bay_field_export import EXTRAS
    from src.experiments.bay_profile_field import prepare
    from src.experiments import v2_field as F
    from src.experiments import v2_field_ctdprior as Cp
    from src.experiments import v2_field_placebo as P
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, _ = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                on=['cast_id', 'depth_m'], validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    truth = data[TARGETS].to_numpy(float)
    mats = {}
    for arm in ('A1_both_direct', 'A2_none_direct', 'A3_design_only_direct', 'A4_base_only_direct', 'A1_both_aou'):
        ub, ud, aou = F.ARMS[arm]
        mats[arm] = [[F.matrices(b, extras, ub, ud, aou) for b in bundles]]
    mats['A5_intercept_direct'] = mats['A1_both_direct']
    ub, ud, aou = F.ARMS['A1_both_direct']
    times = pd.to_datetime(data.t_utc, utc=True)
    query_bin = np.floor(data.depth_m.to_numpy() / Cp.BIN_M).astype(int)
    full = Cp.archive()
    coverage = {}
    for arm, prior in (('A6_ctdprior_direct', 'month'), ('A7_ctdprior_direct', 'harmonic')):
        m, cov = [], []
        for b, contract in zip(bundles, contracts):
            cutoff = pd.Timestamp(contract['last_training'])
            cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff
            train = full[full.station.isin(set(contract['training_stations'])) & (full.t_utc <= cutoff)]
            if prior == 'month':
                delta, fell = Cp.month_depth_prior(train, query_bin, times.dt.month.to_numpy())
            else:
                delta, fell = Cp.harmonic_prior(train, query_bin, times.dt.dayofyear.to_numpy())
            e = extras.copy()
            e[:, 3:6] = delta
            m.append(F.matrices(b, e, ub, ud, aou))
            cov.append({'fold': contract['fold'], 'held_station': contract['held_station'],
                        'climatology_casts': int(train.cast_id.nunique()), 'climatology_rows': int(len(train)),
                        'cutoff': str(cutoff), 'fallback_bin_only': int((fell == 1).sum()),
                        'fallback_zero': int((fell == 2).sum()), 'rows': int(len(fell))})
        mats[arm] = [m]
        coverage[arm] = cov
    depth = data.depth_m.to_numpy(float)
    for arm in ('P1_placebo_direct', 'P2_placebo_direct'):
        kind = arm.split('_')[0]
        rng = np.random.default_rng(P.SEED)
        draws = []
        for _ in range(1 if kind == 'P1' else P.DRAWS):
            m = []
            for b in bundles:
                mask = np.zeros(len(depth), dtype=bool)
                mask[b['tr']] = True
                e = extras.copy()
                e[:, 3:6] = P.placebo_offsets(kind, depth, mask, truth - b['base'].astype(float), rng)
                m.append(F.matrices(b, e, ub, ud, aou))
            draws.append(m)
        mats[arm] = draws
    return bundles, contracts, mats, coverage


def fit_all(bundles, mats, alpha_table):
    """{(family, design, arm): {'frames': [per draw], 'intercepts': [...]}} at the frozen penalties."""
    from src.experiments.bay_profile_field import prediction_frame
    from src.experiments import v2_field as F
    from src.experiments import v2_field_intercept as I
    out = {}
    for key, alpha in alpha_table.items():
        arm = key[2]
        aou = arm == 'A1_both_aou'
        frames, intercepts = [], []
        for draw in mats[arm]:
            rows = []
            for b, (base, y, feats) in zip(bundles, draw):
                w = np.asarray(b['weights'], float)
                if arm == 'A5_intercept_direct':
                    pred, model, _ = I.fit_predict(feats, base, y, b['tr'], b['va'], w, alpha[b['fold']])
                    intercepts.append([float(v) * s for v, s in zip(np.atleast_1d(model.intercept_), I.SCALE)])
                else:
                    pred, _, _ = F.fit_predict(feats, base, y, b['tr'], b['va'], w, alpha[b['fold']], aou)
                rows.append(prediction_frame(b, pred))
            frames.append(pd.concat(rows, ignore_index=True))
        out[key] = {'frames': frames, 'intercepts': intercepts}
    return out


def run_mode(mode):
    """Everything in one unit: 'published' (reproduction) or 'mgl'."""
    from src.experiments import bay_profile_field as BPF
    import src.experiments.v2_field  # noqa: F401  (bind the names harmonised() rebinds)
    import src.experiments.v2_field_ctdprior  # noqa: F401
    import src.experiments.v2_field_placebo  # noqa: F401
    import src.experiments.v2_field_intercept  # noqa: F401
    import src.data.bay_profile_context  # noqa: F401
    scale = BPF.SCALE            # one array object shared by every module that imported it
    saved = scale.copy()
    seen = {}
    with harmonised(mode == 'mgl', log=seen):
        if mode == 'mgl':
            scale[2] = saved[2] * constant_c()
        try:
            bundles, contracts, mats, coverage = setup()
            fits = fit_all(bundles, mats, alphas())
        finally:
            scale[:] = saved
    return bundles, contracts, fits, coverage, seen


def nested(d):
    from src.experiments.v2_field import nested_mae
    return nested_mae(d)


def compare(frame, path):
    pub = pd.read_csv(path)
    cols = [c for c in pub.columns if c.startswith('pred_')]
    same = (list(pub.columns) == list(frame.columns) and len(pub) == len(frame) and
            (pub[['cast_id', 'depth_m', 'fold']].to_numpy() == frame[['cast_id', 'depth_m', 'fold']].to_numpy()).all())
    a, b = pub[cols].to_numpy(float), frame[cols].to_numpy(float)
    diff = float(np.nanmax(np.abs(a - b))) if same else None
    rel = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))) if same else None
    truth = float(np.nanmax(np.abs(pub[['temp_c', 'salinity_psu', 'do_umol_l']].to_numpy(float)
                                   - frame[['temp_c', 'salinity_psu', 'do_umol_l']].to_numpy(float)))) if same else None
    return same, diff, rel, truth


def reproduce(fits):
    """The published-unit refits against every published prediction file and nested MAE."""
    rows = []
    for (family, design, arm), fit in sorted(fits.items()):
        if family == 'campaign':
            src = published_source(arm)
            report = json.loads((src / 'report.json').read_text())
        else:
            src = (A47 if family == 'a47' else A48) / 'field' / design / arm
            report = json.loads((src / 'report.json').read_text())
        same, diff, rel, tdiff = compare(fit['frames'][0], src / 'selected_predictions.csv.gz')
        got = [nested(f) for f in fit['frames']]
        if 'per_draw_nested_mae' in report and report['per_draw_nested_mae']:
            want = report['per_draw_nested_mae']
        elif 'nested_mae' in report:
            want = [report['nested_mae']]
        else:
            want = [{t: v['mean'] for t, v in report['summary'].items()}]
        mae = max(abs(g[t] - w[t]) / abs(w[t]) for g, w in zip(got, want) for t in w)
        ok = bool(same and rel <= TOL_REL and tdiff <= 1e-9 and mae <= TOL_REL and len(got) == len(want))
        rows.append({'family': family, 'design': design, 'arm': arm, 'rows_identical': bool(same),
                     'max_abs_prediction_diff': diff, 'max_relative_prediction_diff': rel,
                     'truth_max_abs_diff': tdiff, 'draws': len(got),
                     'max_relative_nested_mae_diff': mae, 'pass': ok})
        print(f'reproduce {family:8s} {design:9s} {arm:22s} rel {rel:.1e}  mae {mae:.1e}  {"ok" if ok else "FAIL"}', flush=True)
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(key):
    return {'declared_in': ['config/bay_reserved_protocol_v2_addendum60.json',
                            'config/bay_reserved_protocol_v2_addendum60_amendment1.json'],
            'unit': 'mg/l (columns keep their published names)',
            'configuration': f'penalty of the published {key[0]} study, design {key[1]}, not re-selected',
            'producer': 'scripts/v2_oxygen_mgl_field.py'}


def write(fits, coverage):
    from src.data.bay_profile_context import TARGETS
    # the campaign's shared selection and the addendum-12/14/19 arms, in their published folders
    for arm in CAMPAIGN + LOCAL_ARMS:
        key = ('campaign', 'shared', arm)
        fit, src, dst = fits[key], published_source(arm), mirror_source(arm)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f'held_site_{arm}_selection.json', dst / f'held_site_{arm}_selection.json')
        path = dst / 'selected_predictions.csv.gz'
        write_csv(fit['frames'][0], path)
        report = json.loads((src / 'report.json').read_text())
        per_draw = [nested(f) for f in fit['frames']]
        if 'summary' in report:
            frame = pd.DataFrame(per_draw)
            report['per_draw_nested_mae'] = per_draw
            report['summary'] = {t: {'mean': float(frame[t].mean()), 'min': float(frame[t].min()),
                                     'max': float(frame[t].max()),
                                     'sd': float(frame[t].std(ddof=1)) if len(per_draw) > 1 else 0.0}
                                 for t in TARGETS}
        else:
            report['nested_mae'] = per_draw[0]
        if 'rows' in report:
            report['rows'] = int(len(fit['frames'][0]))
            report['casts'] = int(fit['frames'][0].cast_id.nunique())
        if 'climatology_coverage' in report:
            report['climatology_coverage'] = coverage[arm]
        if 'fitted_intercepts_physical_units' in report:
            report['fitted_intercepts_physical_units'] = fit['intercepts']
        report['prediction_sha256'] = sha(path)
        report['harmonised'] = provenance(key)
        write_json(dst / 'report.json', report)
    # the per-fold designs
    for family, root, arms in (('a47', A47, A47_ARMS), ('a48', A48, A48_ARMS)):
        for d in DESIGNS:
            for arm in arms:
                key = (family, d, arm)
                fit = fits[key]
                dst = C1 / ('v2_perfold' if family == 'a47' else 'v2_perfold_rest') / 'field' / d / arm
                dst.mkdir(parents=True, exist_ok=True)
                path = dst / 'selected_predictions.csv.gz'
                write_csv(fit['frames'][0], path)
                report = json.loads((root / 'field' / d / arm / 'report.json').read_text())
                per_draw = [nested(f) for f in fit['frames']]
                report['nested_mae'] = per_draw[0]
                if 'per_draw_nested_mae' in report:
                    report['per_draw_nested_mae'] = per_draw if len(per_draw) > 1 else None
                if 'nested_mae_mean_over_draws' in report:
                    report['nested_mae_mean_over_draws'] = {t: float(np.mean([x[t] for x in per_draw])) for t in per_draw[0]}
                report['rows'] = int(len(fit['frames'][0]))
                report['casts'] = int(fit['frames'][0].cast_id.nunique())
                if d == 'published':
                    # the refit and the 'published' file are one computation in mg/l
                    camp = mirror_source(arm) / 'selected_predictions.csv.gz'
                    same, diff, rel, _ = compare(fit['frames'][0], camp)
                    mine = nested(pd.read_csv(camp)) if arm not in ('P2_placebo_direct',) else per_draw[0]
                    mae_diff = max(abs(per_draw[0][t] - mine[t]) for t in mine)
                    report.update(gate_G4_rows_identical=bool(same), gate_G4_max_abs_prediction_diff=diff,
                                  gate_G4_nested_mae_abs_diff=mae_diff,
                                  gate_G4_pass=bool(same and diff <= 1e-9 and mae_diff <= 1e-9))
                    if 'gate_G4_max_relative_prediction_diff' in report:
                        report['gate_G4_max_relative_prediction_diff'] = rel
                    for k in ('objective_here', 'objective_abs_diff', 'published_best_value'):
                        report.pop(k, None)
                report['prediction_sha256'] = sha(path)
                report['harmonised'] = provenance(key)
                write_json(dst / 'report.json', report)


def main():
    t0 = time.time()
    _, _, fits, _, seen = run_mode('published')
    rows = reproduce(fits)
    record = {'what': 'Field-57 arms refitted in the published unit at their published penalties, against the '
                      'published prediction files and nested MAEs (gate before any mg/l refit)',
              'tolerance_relative': TOL_REL, 'pass': all(r['pass'] for r in rows), 'fits': rows}
    write_json(C1 / 'field57_reproduction.json', record)
    if not record['pass']:
        raise SystemExit('a Field-57 arm does not reproduce in the published unit; nothing converted')
    print(f'reproduced {len(rows)} fits in {time.time() - t0:.0f} s', flush=True)
    _, _, fits, coverage, seen = run_mode('mgl')
    print('conversions applied:', seen, flush=True)
    for need in ('ctd_all_casts.csv.gz', 'profile_boundary_context.csv.gz', 'read_export', 'o2sol'):
        if not seen.get(need):
            raise SystemExit(f'the mg/l run did not convert {need}')
    write(fits, coverage)
    record['mgl_conversions_applied'] = seen
    write_json(C1 / 'field57_reproduction.json', record)
    for arm in CAMPAIGN + LOCAL_ARMS:
        r = json.loads((mirror_source(arm) / 'report.json').read_text())
        m = r.get('nested_mae') or {t: v['mean'] for t, v in r['summary'].items()}
        print(f'{arm:22s} mg/l  T {m["temp_c"]:.4f}  S {m["salinity_psu"]:.4f}  O {m["do_umol_l"]:.4f}')
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
