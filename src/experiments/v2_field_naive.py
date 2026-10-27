"""Is the spatial field model better than no model at all? (Field-57, development only)

A leave-one-station-out month x depth climatology reaches 10.70 umol/L on the 164-cast Gen2 cohort
while the released field model reports 12.23 umol/L on Field-57, and nothing follows from that
because the cohorts differ. This runs the comparison on the identical Field-57 casts, rows and
held-station folds.

Three predictors on exactly the same endpoints and the same frozen nested macro rule:
  climatology  month x 1 m depth-bin mean, built from the fold's *training* stations only
  background   the model's own physically constructed background with the ridge residual zeroed
  model        the v2 released recipe (A1 both, direct oxygen) at its already-selected penalty

Both comparators are parameter-free, so they need no Optuna study, and the model is not
re-tuned: this study adds comparators, it does not change anything the thesis released.
Declared in config/bay_reserved_protocol_v2_addendum2.json before it was run.

`--perfold` (review R3, 2026-10-05): the model's penalty, one for both folds in the published run,
becomes each calendar fold's own (addendum 47, `v2_perfold_select.field('A1_both_direct', k)`), and
the no-Copernicus comparator, a tuned ridge too, is read from its per-fold predictions
(PERFOLD_RUN/A2_none_direct). The parameter-free comparators do not change. The refitted model is
checked against addendum 47's per-fold A1 rows as the published one is against the released run.
Everything goes to OUT/'perfold' under the default names; the default writes what it always wrote.
"""
import argparse, hashlib, json, zlib
import numpy as np
import pandas as pd
from src.experiments.bay_profile_field import ROOT, prepare, prediction_frame
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_field import matrices, fit_predict, nested_mae
from src.data.bay_profile_context import load_profiles, TARGETS

RUN = ROOT / 'reports/bay_twin/runs/20260919T145045Z_v2_field'
OUT = ROOT / 'reports/bay_twin/runs/v2_field_naive'
#: addendum 47's per-fold Field-57 runs (a caller rerunning in other units may point this elsewhere)
PERFOLD_RUN = ROOT / 'reports/bay_twin/runs/v2_perfold/field/perfold'
DRAWS = 2000
GZIP = {'method': 'gzip', 'mtime': 0}


def save_lf(path, obj):
    """JSON with LF line endings whatever the host."""
    path.write_bytes(json.dumps(obj, indent=2, allow_nan=False).encode('utf-8'))


def _source(path):
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        name = path.as_posix()
    return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def climatology_frame(bundle, data):
    """Month x 1 m depth-bin mean of each target over the fold's training casts only.

    This is the climatology quoted in the module docstring. On these folds it is weaker than it
    looks: fold 0 trains on 2024-07..2024-12 and is scored on 2025-01..2025-06, so *no* validation
    row has its (month, depth) cell in training and the table degenerates to a depth profile with
    no seasonal information. The hit rate is reported alongside the result, and `seasonal_frame`
    gives the naive comparator a form that can extrapolate across the calendar.
    """
    tr, va = bundle['tr'], bundle['va']
    month = data.t_utc.dt.month.to_numpy(); depth = np.floor(data.depth_m.to_numpy()).astype(int)
    train = pd.DataFrame({'month': month[tr], 'depth': depth[tr]}).join(
        data.iloc[tr][TARGETS].reset_index(drop=True))
    cell = train.groupby(['month', 'depth'])[TARGETS].mean()
    by_depth = train.groupby('depth')[TARGETS].mean()
    overall = train[TARGETS].median()
    want = pd.MultiIndex.from_arrays([month[va], depth[va]])
    pred = cell.reindex(want).to_numpy()
    hit = int(np.isfinite(pred[:, 0]).sum())
    fill = by_depth.reindex(depth[va]).to_numpy()
    pred = np.where(np.isfinite(pred), pred, fill)
    return np.where(np.isfinite(pred), pred, overall.to_numpy()[None, :]), hit


def oracle_frame(bundle, data):
    """The leave-one-station-out construction: month x depth bin from every *other* station's casts,
    with no temporal holdout at all. The baseline may therefore use casts from the scoring period,
    an advantage the model is denied. Reported so the comparison cannot be dismissed as an artefact
    of giving the naive comparator less data than the model.
    """
    va = bundle['va']
    month = data.t_utc.dt.month.to_numpy(); depth = np.floor(data.depth_m.to_numpy()).astype(int)
    held = data.station.to_numpy()[va][0]
    other = np.flatnonzero(data.station.to_numpy() != held)
    train = pd.DataFrame({'month': month[other], 'depth': depth[other]}).join(
        data.iloc[other][TARGETS].reset_index(drop=True))
    cell = train.groupby(['month', 'depth'])[TARGETS].mean()
    by_depth = train.groupby('depth')[TARGETS].mean()
    pred = cell.reindex(pd.MultiIndex.from_arrays([month[va], depth[va]])).to_numpy()
    pred = np.where(np.isfinite(pred), pred, by_depth.reindex(depth[va]).to_numpy())
    return np.where(np.isfinite(pred), pred, train[TARGETS].median().to_numpy()[None, :])


def harmonic(day):
    return np.column_stack([np.ones(len(day)), np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)])


def seasonal_frame(bundle, data):
    """Annual-harmonic climatology per 1 m depth bin, fitted on the fold's training casts only.

    Three coefficients per depth bin by ordinary least squares on a fixed functional form: no
    hyperparameter is selected, so no Optuna study is needed. Bins with fewer than six rows or three
    distinct days borrow the depth-pooled fit, which is itself the fallback for unseen bins.
    """
    tr, va = bundle['tr'], bundle['va']
    day = data.t_utc.dt.dayofyear.to_numpy(); depth = np.floor(data.depth_m.to_numpy()).astype(int)
    y = data[TARGETS].to_numpy(float)

    def solve(rows):
        return np.linalg.lstsq(harmonic(day[rows]), y[rows], rcond=None)[0]
    pooled = solve(tr)
    fits = {}
    for b in np.unique(depth[tr]):
        rows = tr[depth[tr] == b]
        fits[b] = solve(rows) if len(rows) >= 6 and len(np.unique(day[rows])) >= 3 else pooled
    design = harmonic(day[va])
    coefficients = np.stack([fits.get(b, pooled) for b in depth[va]])
    return np.einsum('ij,ijk->ik', design, coefficients)


def paired(frames, x, y, keys):
    """Cast-level paired bootstrap of nested-macro(x) - nested-macro(y); own stream per pair."""
    rng = np.random.default_rng([20260920, zlib.crc32(f'{x}|{y}'.encode())])
    dx, dy = frames[x], frames[y]
    casts = keys.cast_id.unique()
    index = {c: np.flatnonzero(keys.cast_id.to_numpy() == c) for c in casts}
    point = {t: nested_mae(dx)[t] - nested_mae(dy)[t] for t in TARGETS}
    draws = {t: [] for t in TARGETS}
    for _ in range(DRAWS):
        take = np.concatenate([index[c] for c in rng.choice(casts, len(casts), replace=True)])
        a, b = nested_mae(dx.iloc[take]), nested_mae(dy.iloc[take])
        for t in TARGETS: draws[t].append(a[t] - b[t])
    return {t: [point[t], *np.percentile(draws[t], [2.5, 97.5]).tolist()] for t in TARGETS}


def main(perfold=False):
    """The published comparison; with `perfold` the two tuned ridges at their per-fold penalties
    (module docstring), written to OUT/'perfold'."""
    out = OUT / 'perfold' if perfold else OUT
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
    bundles, contracts, excluded = prepare('held_site', cfg, False)
    data, _ = load_profiles()
    context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                 on=['cast_id', 'depth_m'], validate='one_to_one')
    extras = context[EXTRAS].to_numpy(dtype=float)
    if perfold:
        from src.experiments import v2_perfold_select as pfs
        records = {k: pfs.field('A1_both_direct', k) for k in (0, 1)}
        alpha = {k: r['params']['alpha'] for k, r in records.items()}
    else:
        alpha = json.loads((RUN / 'results/v2_field_held_site_A1_both_direct/report.json').read_text())['selection']['params']['alpha']

    rows = {'climatology': [], 'seasonal': [], 'oracle': [], 'background': [], 'model': []}
    coverage = {}
    for b in bundles:
        base, y, feats = matrices(b, extras, use_base=True, use_design=True, aou=False)
        va = b['va']
        month_cells, hit = climatology_frame(b, data)
        coverage[b['identity']] = {'rows': len(va), 'month_cell_hits': hit}
        rows['climatology'].append(prediction_frame(b, month_cells))
        rows['seasonal'].append(prediction_frame(b, seasonal_frame(b, data)))
        rows['oracle'].append(prediction_frame(b, oracle_frame(b, data)))
        background = base[va].copy()
        background[:, 0] = np.clip(background[:, 0], -2, 40); background[:, 1] = np.clip(background[:, 1], 0, 42)
        rows['background'].append(prediction_frame(b, background))
        pred, _, _ = fit_predict(feats, base, y, b['tr'], va, np.asarray(b['weights'], float),
                                 alpha[b['fold']] if perfold else alpha, False)
        rows['model'].append(prediction_frame(b, pred))
    frames = {k: pd.concat(v, ignore_index=True) for k, v in rows.items()}
    # The degraded mode, reported as a first-class comparator: no Copernicus anywhere. Its
    # predictions already exist from the v2 field campaign and align row-for-row with these frames.
    frames['no_copernicus'] = pd.read_csv(PERFOLD_RUN / 'A2_none_direct/selected_predictions.csv.gz' if perfold else
                                          RUN / 'results/v2_field_held_site_A2_none_direct/selected_predictions.csv.gz')
    keys = frames['model'][['fold', 'station', 'cast_id']]
    for k, f in frames.items():
        assert f[['fold', 'station', 'cast_id', 'depth_m']].reset_index(drop=True).equals(
            frames['model'][['fold', 'station', 'cast_id', 'depth_m']].reset_index(drop=True)), k

    released = pd.read_csv(PERFOLD_RUN / 'A1_both_direct/selected_predictions.csv.gz' if perfold else
                           RUN / 'results/v2_field_held_site_A1_both_direct/selected_predictions.csv.gz')
    reproduction = {t: [float(nested_mae(frames['model'])[t]),
                        float(nested_mae(released.assign(t_utc=released.t_utc))[t])] for t in TARGETS}

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum2.json',
              'alpha': {str(k): v for k, v in alpha.items()} if perfold else alpha,
              'casts': int(frames['model'].cast_id.nunique()), 'rows': len(frames['model']),
              'folds': [c['id'] for c in contracts], 'excluded_casts': excluded, 'draws': DRAWS,
              'reproduction_check_model_vs_released_run': reproduction,
              'month_cell_coverage': coverage,
              'month_cell_coverage_total': [sum(v['month_cell_hits'] for v in coverage.values()),
                                            sum(v['rows'] for v in coverage.values())],
              'nested_mae': {k: nested_mae(f) for k, f in frames.items()},
              'contrasts': {f'{x} - {y}': paired(frames, x, y, keys)
                            for x, y in [('model', 'climatology'), ('model', 'seasonal'),
                                         ('model', 'oracle'), ('model', 'background'),
                                         ('background', 'oracle'), ('seasonal', 'climatology'),
                                         ('no_copernicus', 'oracle'), ('no_copernicus', 'model')]}}
    if perfold:
        report['selection'] = {
            'design': 'perfold', 'objective': "mean over that calendar fold's own checkpoints of the inner station-fold score",
            'model': {str(k): {f: r[f] for f in ('source', 'sha256', 'study', 'completed_trials', 'params')}
                      for k, r in records.items()},
            'no_copernicus': _source(PERFOLD_RUN / 'A2_none_direct/selected_predictions.csv.gz')}
        report['reproduction_source'] = _source(PERFOLD_RUN / 'A1_both_direct/selected_predictions.csv.gz')
    for k, f in frames.items():
        p = out / f'pred_{k}.csv.gz'
        if perfold:
            f.to_csv(p, index=False, lineterminator='\n', compression=GZIP)
        else:
            f.to_csv(p, index=False)
        report.setdefault('prediction_sha256', {})[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (save_lf if perfold else save)(out / 'field_naive_summary.json', report)
    for k, m in report['nested_mae'].items():
        print(f"{k:12s} " + '  '.join(f'{t} {m[t]:.4f}' for t in TARGETS))
    for k, v in report['contrasts'].items():
        print(k)
        for t in TARGETS:
            d, lo, hi = v[t]
            print(f'   {t:14s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--perfold', action='store_true',
                    help="the model and the no-Copernicus ridge at each fold's own penalty; writes OUT/perfold")
    main(ap.parse_args().perfold)
