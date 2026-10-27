"""One ridge penalty per target instead of one shared across temperature, salinity and oxygen.

Addendum-2 amendment 1. The released recipe selects a single penalty on an objective averaged over
the three targets; the v1 43-point sweep shows temperature flat above alpha 3e4 while oxygen has a
clear optimum near 5e3, so the shared penalty is set where temperature wants it and oxygen pays.
This runs one >=100-trial nested study per target -- selection is an inner GroupKFold over each
outer fold's *training* stations, never the held station, so the headroom seen in that sweep (which
was scored on the held stations) is not simply read off.

Reported as a recommended simplification, exactly as arm A4 was. Nothing released changes.
"""
import hashlib, json, zlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from src.experiments.bay_profile_field import ROOT, SCALE, prepare, prediction_frame
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_field import matrices, fit_predict, nested_mae
from src.experiments.v2_field_naive import DRAWS
from src.data.bay_profile_context import load_profiles, TARGETS
from src.tuning.protocol_v2 import run_study, PROTOCOL

RUN = ROOT / 'reports/bay_twin/runs/20260919T145045Z_v2_field'
OUT = ROOT / 'reports/bay_twin/runs/v2_field_pertarget'


def single_target_inner(b, mats, alpha, target, stations, casts, truth):
    """Inner GroupKFold over the outer fold's training stations, scored on one target only."""
    base, y, feats = mats; tr = b['tr']; w = np.asarray(b['weights'], float); groups = stations[tr]
    k = min(4, len(np.unique(groups)))
    if k < 2: raise ValueError('fewer than two training stations')
    scores = []
    for itr, iva in GroupKFold(n_splits=k).split(tr, groups=groups):
        pred, _, _ = fit_predict(feats, base, y, tr[itr], tr[iva], w[itr], alpha, False)
        e = pd.Series(np.abs(pred[:, target] - truth[tr[iva], target])).groupby(casts[tr[iva]]).mean()
        scores.append(e.mean() / SCALE[target])
    return float(np.mean(scores))


def predict_per_target(feats, base, y, tr, va, weights, alphas):
    """Each target predicted by a ridge at its own penalty; identical design and background."""
    scaler = StandardScaler().fit(feats[tr]); design = scaler.transform(feats).astype('float32')
    out = np.empty((len(va), len(TARGETS)))
    for j, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha, fit_intercept=False).fit(design[tr], ((y - base) / SCALE)[tr], sample_weight=weights)
        out[:, j] = base[va, j] + model.predict(design[va])[:, j] * SCALE[j]
    out[:, 0] = np.clip(out[:, 0], -2, 40); out[:, 1] = np.clip(out[:, 1], 0, 42)
    return out


def paired(a, b, label):
    rng = np.random.default_rng([20260920, zlib.crc32(label.encode())])
    casts = a.cast_id.unique()
    index = {c: np.flatnonzero(a.cast_id.to_numpy() == c) for c in casts}
    point = {t: nested_mae(a)[t] - nested_mae(b)[t] for t in TARGETS}
    draws = {t: [] for t in TARGETS}
    for _ in range(DRAWS):
        take = np.concatenate([index[c] for c in rng.choice(casts, len(casts), replace=True)])
        x, y = nested_mae(a.iloc[take]), nested_mae(b.iloc[take])
        for t in TARGETS: draws[t].append(x[t] - y[t])
    return {t: [point[t], *np.percentile(draws[t], [2.5, 97.5]).tolist()] for t in TARGETS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, excluded = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                 on=['cast_id', 'depth_m'], validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    stations = data.station.to_numpy(); casts = data.cast_id.to_numpy(); truth = data[TARGETS].to_numpy(float)
    mats = [matrices(b, extras, True, True, False) for b in bundles]
    shared = json.loads((RUN / 'results/v2_field_held_site_A1_both_direct/report.json').read_text())['selection']['params']['alpha']

    save(OUT / 'contract.json', {'protocol': PROTOCOL['version'], 'declared_in': 'config/bay_reserved_protocol_v2_addendum2.json amendment_1',
                                 'arm': 'A1_both direct, one penalty per target', 'shared_alpha': shared,
                                 'folds': contracts, 'excluded_casts': excluded, 'search_space': 'ridge alpha 1e-1 to 1e8 (log)',
                                 'selection': 'inner GroupKFold (k<=4) over each outer fold training stations, scored on ONE target',
                                 'no_2026': True})
    selections = {}
    for j, target in enumerate(TARGETS):
        def objective(trial, j=j):
            alpha = trial.suggest_float('alpha', .1, 1e8, log=True)
            return float(np.mean([single_target_inner(b, m, alpha, j, stations, casts, truth) for b, m in zip(bundles, mats)]))
        selections[target] = run_study(f'field_pertarget_{target}', objective, OUT)
    alphas = [selections[t]['params']['alpha'] for t in TARGETS]

    rows, ref = [], []
    for b, (base, y, feats) in zip(bundles, mats):
        w = np.asarray(b['weights'], float)
        rows.append(prediction_frame(b, predict_per_target(feats, base, y, b['tr'], b['va'], w, alphas)))
        ref.append(prediction_frame(b, fit_predict(feats, base, y, b['tr'], b['va'], w, shared, False)[0]))
    per, shared_frame = pd.concat(rows, ignore_index=True), pd.concat(ref, ignore_index=True)
    naive = pd.read_csv(ROOT / 'reports/bay_twin/runs/v2_field_naive/pred_background.csv.gz')

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum2.json amendment_1',
              'shared_alpha': shared, 'per_target_alpha': dict(zip(TARGETS, alphas)),
              'trials': {t: selections[t]['completed_trials'] for t in TARGETS},
              'casts': int(per.cast_id.nunique()), 'rows': len(per), 'draws': DRAWS,
              'nested_mae': {'per_target': nested_mae(per), 'shared': nested_mae(shared_frame),
                             'background': nested_mae(naive)},
              'contrasts': {'per_target - shared': paired(per, shared_frame, 'pertarget-shared'),
                            'per_target - background': paired(per, naive, 'pertarget-background')},
              'adopted': False,
              'why_not_adopted': 'Selected on Field-57; the spatial interface abstains throughout 2026, so no independent cohort exists to confirm it. Recommended for the next version, like arm A4.'}
    path = OUT / 'pred_per_target.csv.gz'; per.to_csv(path, index=False)
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT / 'field_pertarget_summary.json', report)

    print('alphas', {t: f'{a:.4g}' for t, a in zip(TARGETS, alphas)}, ' shared', f'{shared:.4g}')
    for k, m in report['nested_mae'].items():
        print(f"{k:12s} " + '  '.join(f'{t} {m[t]:.4f}' for t in TARGETS))
    for k, v in report['contrasts'].items():
        print(k)
        for t in TARGETS:
            d, lo, hi = v[t]
            print(f'   {t:14s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    main()
