"""Addenda 49 and 51: the recurrent arm of addendum 23 on a calendar split, and the sequence
comparison swept over issuance phase and sampler seed.

Addendum 49. The published recurrent arm is selected per fold on a random
80/20 split of that fold's training windows, which overlap their neighbours by 506 of 512 hours,
by plain mean absolute error. Here it is also selected per outer fold on the calendar inner window
every other tuned forecasting arm uses (`v2_horizon.inner_fold`), by the published ratio objective
(fold-macro forecast MAE over fold-macro persistence MAE). Three designs:

  published     each fold's published selection refitted (gate G4)
  random_split  the published design re-run here (the control)
  calendar      one study per outer fold on its calendar inner window, ratio objective

Addendum 51. The same comparison on the issuance grid advanced by 1 to 5 hours and, on the
published grid, under sampler seeds 7 and 123: the recurrent arm re-tuned in every cell under both
designs; the tree refitted without per-probe weights, as addendum 23 fits it, at the configuration
the matched-objective phase sweep selected for that grid and seed; Chronos-Bolt, TimesFM 2.5 and
(new, descriptive) Chronos-2 zero-shot on the same contexts.

Nothing in `v2_sequence_arms` is edited; its functions are imported and its published path is
unchanged. Declared in config/bay_reserved_protocol_v2_addendum49.json and _addendum51.json.

Usage (repository root, V2_THREADS=2):
  python -m src.experiments.v2_sequence_sweep gru --variable oxygen --offset 0 --sampler-seed 42 --design published random_split calendar
  python -m src.experiments.v2_sequence_sweep panel --variable oxygen --offset 3

Review R3 (2026-10-05). The panel's tree takes the configuration the phase sweep selected on both
folds' inner windows together for that grid and seed (`tree_selection`). `panel-perfold` refits the
tree columns per outer fold at that fold's own selection and reads every other column from the
saved panel (the tree reaches nothing else), after refitting the shared configuration and requiring
the saved column back (gate G1). The reference cell (offset 0, seed 42) uses addendum 47's per-fold
study; the seven other cells (offsets 1-5 at seed 42, seeds 7 and 123 at offset 0) have per-fold
studies of exactly this tree (data, design, search space, ratio objective on the fold's own inner
window, sampler seed) from addendum 50 (`runs/v2_xgb_sweep/<variable>_tree_h6_shift<o>_seed<s>/`),
which `--studies reuse` (the default) reads. `--studies run` runs those studies afresh with
`protocol_v2.run_study` (100 completed trials each) into `runs/v2_sequence_sweep/perfold_tree_studies/`
and scores with them: 2 folds x 7 cells x 2 variables = 28 studies (`perfold-plan` lists them).
Outputs: `runs/v2_sequence_sweep/panel_perfold/` (`_smoke` with `--quick`), under the panel's names.
  python -m src.experiments.v2_sequence_sweep panel-perfold --variable oxygen --offset 3 [--studies run]
  python -m src.experiments.v2_sequence_sweep perfold-plan
"""
import os
os.environ.setdefault('V2_THREADS', '2')

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments import v2_sequence_arms as sa

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_sequence_sweep'
PUBLISHED = RUNS / 'v2_sequence_arms'
PHASEFIX = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs/v2_phase_shift'
CHRONOS2_REPO = 'amazon/chronos-2'
DESIGNS = ('published', 'random_split', 'calendar')
OFFSETS = (0, 1, 2, 3, 4, 5)
SAMPLER_SEEDS = (42, 7, 123)
KEY = ['fold', 'sensor', 'time']


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_tag(variable, offset, seed):
    return f'{variable}_offset{offset}_seed{seed}'


def declared_in(offset, seed):
    return ('config/bay_reserved_protocol_v2_addendum49.json' if (offset, seed) == (0, 42)
            else 'config/bay_reserved_protocol_v2_addendum51.json')


def setup(variable, offset):
    """The six-hour data of that grid, the hourly panel behind it and the outer folds."""
    from src.experiments.v2_horizon import build, folds
    from src.experiments.v2_phase_shift import build_shifted
    data = build(variable, sa.HORIZON) if offset == 0 else build_shifted(variable, sa.HORIZON, offset)
    names, index, z, scale = sa.hourly_panel(variable)
    assert names == data['names'], 'sensor order differs from build'
    q = sa.assert_aligned(data, z, index)
    return data, z, q, folds(data, sa.HORIZON), scale


def rows_of(data, fold_like, which):
    """Eligible (endpoint, sensor) rows of a fold or an inner fold, as v2_sequence_arms.eligible_rows."""
    mask = fold_like['eligible']
    return [(i, s) for i in fold_like[which] for s in range(len(data['names']))
            if mask[s] and data['valid'][i, s]]


# ---------------------------------------------------------------- the recurrent arm

def _fold_macro(sensors, err):
    return float(pd.Series(err).groupby(np.asarray(sensors)).mean().mean())


def run_gru(variable, offset, seed, designs, quick=False):
    from src.experiments.v2_forecast import SEEDS
    from src.experiments.v2_horizon import inner_fold
    from src.tuning import protocol_v2
    torch.set_num_threads(sa.THREADS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    data, z, q, outer, scale = setup(variable, offset)
    tag = cell_tag(variable, offset, seed)
    init_seeds = [42] if quick else list(SEEDS)
    for design in designs:
        if design == 'published' and (offset, seed) != (0, 42):
            raise ValueError('the published selection exists only on the published grid and seed')
        folder = OUT / ('gru_smoke' if quick else 'gru') / design / tag
        if (folder / 'report.json').exists():
            print(f'{tag} {design}: already done', flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        report = {'declared_in': declared_in(offset, seed), 'design': design, 'variable': variable,
                  'offset_hours': offset, 'sampler_seed': seed, 'device': device, 'smoke_only': quick,
                  'threads': sa.THREADS, 'initialisation_seeds': init_seeds, 'folds': {}}
        rows = []
        for fi, fold in enumerate(outer):
            va_ctx, va_keep = sa.gather(data, z, q, rows_of(data, fold, 'va'))
            tr_ctx, tr_keep = sa.gather(data, z, q, rows_of(data, fold, 'tr'))
            if not len(va_keep) or not len(tr_keep):
                continue
            tr_y = np.array([data['truth'][i, s] for i, s in tr_keep], 'float32')
            entry = {'training_windows': len(tr_keep), 'scored_windows': len(va_keep)}
            if design == 'published':
                rec = PUBLISHED / f'{variable}_rnn_fold{fi}_selection.json'
                sel = json.loads(rec.read_text())
                entry.update(selection_source=rec.relative_to(ROOT).as_posix(), selection_sha256=_sha(rec))
            elif design == 'random_split':
                cut = int(0.8 * len(tr_ctx))
                order = np.random.default_rng(20260921 + fi).permutation(len(tr_ctx))
                a_, b_ = order[:cut], order[cut:]

                def objective(trial, _a=a_, _b=b_):
                    p = sa.rnn_space(trial)
                    m = sa.fit_recurrent(tr_ctx[_a], tr_y[_a], p, 42, device, quick)
                    score = float(np.abs(sa.recurrent_predict(m, tr_ctx[_b], device) - tr_y[_b]).mean())
                    return score if np.isfinite(score) else 1e6
                sel = sa.run_study(f'{variable}_rnn_random_fold{fi}', objective, folder, quick)
                entry.update(objective='plain MAE on a random fifth of the fold training windows (the published design)')
            elif design == 'calendar':
                inner = inner_fold(data, fold, sa.HORIZON)
                itr_ctx, itr_keep = sa.gather(data, z, q, rows_of(data, inner, 'tr'))
                iva_ctx, iva_keep = sa.gather(data, z, q, rows_of(data, inner, 'va'))
                itr_y = np.array([data['truth'][i, s] for i, s in itr_keep], 'float32')
                iva_y = np.array([data['truth'][i, s] for i, s in iva_keep], 'float32')
                iva_s = [s for _, s in iva_keep]
                iva_p = np.array([data['x'][i, s] for i, s in iva_keep], 'float32')
                persistence = _fold_macro(iva_s, np.abs(iva_p - iva_y))
                entry.update(inner=inner['audit'], inner_training_windows=len(itr_keep),
                             inner_validation_windows=len(iva_keep), inner_persistence_mae_z=persistence)

                def objective(trial):
                    p = sa.rnn_space(trial)
                    m = sa.fit_recurrent(itr_ctx, itr_y, p, 42, device, quick)
                    err = np.abs(sa.recurrent_predict(m, iva_ctx, device) - iva_y)
                    if not np.isfinite(err).all():
                        return 1e6
                    return _fold_macro(iva_s, err) / persistence
                sel = sa.run_study(f'{variable}_rnn_calendar_fold{fi}', objective, folder, quick)
                entry.update(objective="fold-macro forecast MAE / persistence MAE on that fold's calendar inner window")
            else:
                raise ValueError(design)
            if sel['completed_trials'] < 100 and not quick:
                raise RuntimeError(f'{tag} {design} fold {fi}: {sel["completed_trials"]} completed trials')
            preds = [sa.recurrent_predict(sa.fit_recurrent(tr_ctx, tr_y, sel['params'], s, device, quick), va_ctx, device)
                     for s in init_seeds]
            pred = np.mean(preds, axis=0)
            entry['selection'] = sel
            report['folds'][str(fi)] = entry
            for k, (i, s) in enumerate(va_keep):
                rows.append({'fold': fi, 'sensor': data['names'][s], 'time': str(data['times'][i]),
                             'quarter': pd.Timestamp(data['times'][i]).quarter,
                             'truth': float(data['truth'][i, s]), 'lstm': float(pred[k]),
                             **{f'lstm_init{s_}': float(p_[k]) for s_, p_ in zip(init_seeds, preds)}})
        frame = pd.DataFrame(rows)
        path = folder / 'predictions.csv.gz'
        frame.to_csv(path, index=False)
        if design == 'published':
            pub = pd.read_csv(PUBLISHED / f'sequence_arms_{variable}.csv.gz')
            j = pub.merge(frame, on=KEY, suffixes=('_pub', ''), how='inner')
            diff = float(np.abs(j.lstm - j.lstm_pub).max()) * scale
            report.update(published_sha256=_sha(PUBLISHED / f'sequence_arms_{variable}.csv.gz'),
                          gate_G4_rows_matched=int(len(j)), gate_G4_published_rows=int(len(pub)),
                          gate_G4_max_abs_diff=diff,
                          gate_G4_pass=bool(len(j) == len(pub) and diff <= 1e-4))
        report.update(seconds=round(time.time() - t0, 1), prediction_sha256=_sha(path),
                      epochs={k: v['selection']['params']['epochs'] for k, v in report['folds'].items()},
                      mae_fold_macro=float(frame.assign(e=(frame.lstm - frame.truth).abs())
                                           .groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean()) * scale)
        _json(folder / 'report.json', report)
        print(f'{tag} {design}: MAE {report["mae_fold_macro"]:.4f} epochs {report["epochs"]} in {report["seconds"]} s'
              + (f'  G4 {report["gate_G4_pass"]} ({report["gate_G4_max_abs_diff"]:.1e})' if design == 'published' else ''),
              flush=True)


# ---------------------------------------------------------------- the rest of the panel

def chronos2_predict(ctxs, batch=128):
    """Chronos-2 zero-shot, univariate: the call of v2_chronos2.forecast without covariates."""
    from chronos import BaseChronosPipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS2_REPO, device_map=device, torch_dtype=torch.float32)
    qs = list(pipe.quantiles)
    qi = int(np.argmin(np.abs(np.asarray(qs) - .5)))
    out = []
    for k in range(0, len(ctxs), batch):
        res = pipe.predict([{'target': c.astype(np.float32)} for c in ctxs[k:k + batch]],
                           prediction_length=sa.HORIZON)
        for o in res:
            a = o.detach().cpu().float().numpy() if torch.is_tensor(o) else np.asarray(o)
            out.append(float(a.reshape(-1, len(qs), a.shape[-1])[0, qi, sa.HORIZON - 1]))
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(out, 'float32')


def tree_selection(variable, offset, seed):
    """The phase sweep's tree for that grid and seed (ratio objective, run 20260923T103430Z)."""
    name = f'phase_shift_offset{offset}_ratio' + ('' if seed == 42 else f'_seed{seed}') + '_summary.json'
    path = PHASEFIX / name
    params = json.loads(path.read_text())['variables'][variable]['params']['tree']
    # Addendum 23 refits a selection's parameters as its record stores them, without the
    # subsample_freq that tree_space adds, so LightGBM's default applies; kept here so that offset 0
    # reproduces the published tree column (gate G2).
    return ({k: v for k, v in params.items() if k != 'subsample_freq'}, path.relative_to(ROOT).as_posix(),
            _sha(path))


def run_panel(variable, offset, seeds=SAMPLER_SEEDS, zero_shot=True, quick=False):
    """Tree (per sampler seed), zero-shot models and baselines on one grid's paired contexts."""
    from src.experiments.v2_horizon import climatology
    torch.set_num_threads(sa.THREADS)
    data, z, q, outer, scale = setup(variable, offset)
    folder = OUT / ('panel_smoke' if quick else 'panel')
    path = folder / f'panel_{variable}_offset{offset}.csv.gz'
    if (folder / f'panel_{variable}_offset{offset}.json').exists():
        print(f'panel {variable} offset {offset}: already done', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [42] if offset else list(seeds)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum51.json',
              'variable': variable, 'offset_hours': offset, 'issuance_hours_utc':
              sorted(set(pd.DatetimeIndex(data['times']).hour)), 'tree_selections': {}, 'folds': {}}
    rows = []
    for fi, fold in enumerate(outer):
        va_ctx, va_keep = sa.gather(data, z, q, rows_of(data, fold, 'va'))
        if quick:
            va_ctx, va_keep = va_ctx[:64], va_keep[:64]
        month_hour, hour_only = climatology(data, fold)
        cols = {}
        for s_ in seeds:
            params, src, sha = tree_selection(variable, offset, s_)
            report['tree_selections'][str(s_)] = {'source': src, 'sha256': sha, 'params': params}
            grid = sa.tree_predict(data, fold, params, quick)
            cols[f'tree_seed{s_}'] = np.array([grid[i, s] for i, s in va_keep], 'float32')
        if zero_shot:
            cols['chronos_zs'] = sa.chronos_predict(va_ctx)
            cols['timesfm_zs'] = sa.timesfm_predict(va_ctx)
            cols['chronos2_zs'] = chronos2_predict(va_ctx)
        report['folds'][str(fi)] = {'scored_windows': len(va_keep)}
        for k, (i, s) in enumerate(va_keep):
            rows.append({'fold': fi, 'sensor': data['names'][s], 'time': str(data['times'][i]),
                         'quarter': pd.Timestamp(data['times'][i]).quarter,
                         'truth': float(data['truth'][i, s]),
                         'persistence': float(data['x'][i, s]), 'clim_hour': float(hour_only[i, s]),
                         **{c: float(v[k]) for c, v in cols.items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    if offset == 0 and not quick:
        pub = pd.read_csv(PUBLISHED / f'sequence_arms_{variable}.csv.gz')
        j = pub.merge(frame, on=KEY, how='outer', indicator=True, suffixes=('_pub', ''))
        both = j[j['_merge'] == 'both']
        gate = {'rows_identical': bool((j['_merge'] == 'both').all()), 'published_rows': int(len(pub)),
                'rows_here': int(len(frame)),
                'tree_max_abs_diff': float(np.abs(both.tree - both.tree_seed42).max()) * scale}
        if zero_shot:
            gate.update(chronos_max_abs_diff=float(np.abs(both.chronos_zs - both.chronos_zs_pub).max()) * scale,
                        timesfm_max_abs_diff=float(np.abs(both.timesfm_zs - both.timesfm_zs_pub).max()) * scale)
        gate['passed'] = bool(gate['rows_identical'] and gate['tree_max_abs_diff'] <= 1e-9 and
                              (not zero_shot or max(gate['chronos_max_abs_diff'], gate['timesfm_max_abs_diff']) <= 1e-4))
        report['gate_G2'] = gate
        report['published_sha256'] = _sha(PUBLISHED / f'sequence_arms_{variable}.csv.gz')
    report.update(rows=int(len(frame)), seconds=round(time.time() - t0, 1), prediction_sha256=_sha(path))
    _json(folder / f'panel_{variable}_offset{offset}.json', report)
    print(f'panel {variable} offset {offset}: {len(frame)} rows in {report["seconds"]} s'
          + (f'  G2 {report["gate_G2"]}' if 'gate_G2' in report else ''), flush=True)


# ---------------------------------------------------------------- the tree per outer fold (review R3)

GZIP = {'method': 'gzip', 'mtime': 0}
OFF_REFERENCE = [(o, 42) for o in range(1, 6)] + [(0, 7), (0, 123)]
#: the shared tree refitted here against the saved panel column
REPLAY_TOLERANCE = 1e-6


def _rel(path):
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:      # a redirected OUT (smoke runs)
        return Path(path).as_posix()


def perfold_plan():
    """Every per-fold tree study the off-reference cells need, and whether addendum 50 has it."""
    from src.experiments import v2_perfold_select as pfs
    out = []
    for variable in ('temperature', 'oxygen'):
        for offset, seed in OFF_REFERENCE:
            for k in (0, 1):
                try:
                    rec = pfs.sweep_tree(variable, offset, seed, k)
                    on_disk = rec['source']
                except (FileNotFoundError, ValueError) as e:
                    on_disk = f'missing ({e})'
                out.append({'variable': variable, 'offset': offset, 'sampler_seed': seed, 'fold': k,
                            'study': f'{variable}_tree_h6_shift{offset}_seed{seed}_perfold_fold{k}',
                            'addendum50_record': on_disk})
    return out


def perfold_tree_studies(variable, offset, seed, quick=False):
    """One protocol-v2 study per outer fold for the plain tree on that grid under that sampler seed,
    as addendum 50 ran its LightGBM arm (`v2_xgb_sweep.run_cell`): `build_shifted` (equal to `build`
    at offset 0), the plain design, `tree_space`, and the fold-macro MAE over persistence MAE on
    that fold's own inner window. Written to OUT/'perfold_tree_studies'/<tag>/; an existing study
    there is resumed, not restarted."""
    from src.experiments.v2_forecast import tree_space
    from src.experiments.v2_horizon import climatology, folds, inner_fold, ratio, tree_fit_score
    from src.experiments.v2_phase_shift import build_shifted, design_for
    from src.tuning import protocol_v2
    data = build_shifted(variable, sa.HORIZON, offset)
    outer = folds(data, sa.HORIZON)
    inner = [inner_fold(data, f, sa.HORIZON) for f in outer]
    clim = {id(f): climatology(data, f) for f in inner}
    cache = {id(f): design_for(data, f, False) for f in inner}
    tag = f'{variable}_tree_h6_shift{offset}_seed{seed}'
    folder = OUT / ('perfold_tree_studies_smoke' if quick else 'perfold_tree_studies') / tag
    saved = protocol_v2.PROTOCOL['sampler_seed']
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    records = {}
    try:
        for f, fi in zip(outer, inner):
            assert f['number'] == fi['number']

            def objective(trial, _f=fi):
                return ratio(tree_fit_score(data, cache[id(_f)], _f, tree_space(trial), clim[id(_f)])[0])
            name = f'{tag}_perfold_fold{f["number"]}'
            sel = protocol_v2.run_study(name, objective, folder, quick)
            path = folder / f'{name}_selection.json'
            records[f['number']] = {'params': dict(sel['params']), 'source': _rel(path), 'sha256': _sha(path),
                                    'study': name, 'completed_trials': sel['completed_trials']}
    finally:
        protocol_v2.PROTOCOL['sampler_seed'] = saved
    return records


def run_panel_perfold(variable, offset, studies='reuse', quick=False):
    """The panel of one grid with each tree column refitted per outer fold (module docstring)."""
    from src.experiments import v2_perfold_select as pfs
    if studies not in ('reuse', 'run'):
        raise ValueError(studies)
    t0 = time.time()
    data, z, q, outer, scale = setup(variable, offset)
    source = OUT / 'panel' / f'panel_{variable}_offset{offset}.csv.gz'
    published = pd.read_csv(source, float_precision='round_trip')    # every other column, digit for digit
    folder = OUT / ('panel_perfold_smoke' if quick else 'panel_perfold')
    folder.mkdir(parents=True, exist_ok=True)
    tindex = {str(t): i for i, t in enumerate(data['times'])}
    sindex = {n: j for j, n in enumerate(data['names'])}
    assert published.time.map(tindex).notna().all() and published.sensor.map(sindex).notna().all()
    seeds = [42] if offset else list(SAMPLER_SEEDS)
    frame = published.copy()
    gate, records = {}, {}
    for s_ in seeds:
        shared, shared_src, shared_sha = tree_selection(variable, offset, s_)
        ran = (perfold_tree_studies(variable, offset, s_, quick)
               if studies == 'run' and (offset, s_) != (0, 42) else None)
        for fi, fold in enumerate(outer):
            assert fold['number'] == fi
            rows = published[published.fold == fi]
            if rows.empty:
                continue
            replay = sa.tree_on_rows(data, fold, shared, rows, tindex, sindex)
            gate[f'seed{s_}|fold{fi}'] = {'rows': int(len(rows)), 'max_abs_diff': float(
                np.abs(replay - rows[f'tree_seed{s_}'].to_numpy()).max() * scale)}
            rec = ran[fi] if ran is not None else pfs.sweep_tree(variable, offset, s_, fi)
            # refitted as the panel refits a selection: its parameters as stored, LightGBM's default
            # subsample_freq (tree_selection)
            params = {k: v for k, v in rec['params'].items() if k != 'subsample_freq'}
            records[f'seed{s_}|fold{fi}'] = {k: rec[k] for k in ('source', 'sha256', 'study', 'completed_trials')} | {
                'params': params, 'shared_source': shared_src, 'shared_sha256': shared_sha}
            frame.loc[rows.index, f'tree_seed{s_}'] = sa.tree_on_rows(data, fold, params, rows, tindex, sindex)
    tolerance = REPLAY_TOLERANCE * scale
    passed = bool(gate) and all(g['max_abs_diff'] <= tolerance for g in gate.values())
    if not passed and not quick:
        raise SystemExit(f'G1: the shared tree does not reproduce the saved panel {gate}; nothing written')
    path = folder / f'panel_{variable}_offset{offset}.csv.gz'
    frame.to_csv(path, index=False, lineterminator='\n', compression=GZIP)
    mae = {c: float(frame.assign(e=(frame[c] - frame.truth).abs()).groupby(['fold', 'sensor']).e.mean()
                    .groupby('fold').mean().mean() * scale)
           for c in [f'tree_seed{s_}' for s_ in seeds]}
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum51.json',
              'review': 'R3 (2026-10-05): tree columns selected per outer fold; every other column from the saved panel',
              'variable': variable, 'offset_hours': offset, 'smoke_only': quick, 'studies': studies,
              'tree_selection': {'design': 'perfold', 'objective':
                                 "fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only",
                                 'records': records},
              'other_columns_from': {'path': _rel(source), 'sha256': _sha(source)},
              'gate_G1_shared_tree_replay': {'by_cell': gate, 'tolerance_physical': tolerance, 'pass': passed},
              'tree_mae_fold_macro': mae, 'rows': int(len(frame)), 'seconds': round(time.time() - t0, 1),
              'prediction_sha256': _sha(path)}
    (folder / f'panel_{variable}_offset{offset}.json').write_bytes(
        json.dumps(report, indent=2, allow_nan=False).encode('utf-8'))
    print(f'panel-perfold {variable} offset {offset}: {len(frame)} rows, tree MAE {mae}, G1 {passed} '
          f'in {report["seconds"]} s', flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('panel-perfold')
    pp.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    pp.add_argument('--offset', type=int, choices=OFFSETS, required=True)
    pp.add_argument('--studies', choices=['reuse', 'run'], default='reuse',
                    help="reuse addendum 50's per-fold studies, or run them afresh into perfold_tree_studies/")
    pp.add_argument('--quick', action='store_true')
    sub.add_parser('perfold-plan')
    g = sub.add_parser('gru')
    g.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    g.add_argument('--offset', type=int, choices=OFFSETS, default=0)
    g.add_argument('--sampler-seed', type=int, choices=SAMPLER_SEEDS, default=42)
    g.add_argument('--design', nargs='+', choices=DESIGNS, required=True)
    g.add_argument('--quick', action='store_true')
    p = sub.add_parser('panel')
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--offset', type=int, choices=OFFSETS, required=True)
    p.add_argument('--no-zero-shot', action='store_true')
    p.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'panel-perfold':
        run_panel_perfold(a.variable, a.offset, a.studies, a.quick)
        return
    if a.cmd == 'perfold-plan':
        plan = perfold_plan()
        for p_ in plan:
            print(f"{p_['study']:44s} {p_['addendum50_record']}")
        print(f'{len(plan)} studies; {sum(not p_["addendum50_record"].startswith("missing") for p_ in plan)} on disk')
        return
    if a.cmd == 'gru':
        if a.offset and a.sampler_seed != 42:
            raise SystemExit('undeclared cell: seeds 7 and 123 are run on the published grid only')
        run_gru(a.variable, a.offset, a.sampler_seed, a.design, a.quick)
    else:
        run_panel(a.variable, a.offset, zero_shot=not a.no_zero_shot, quick=a.quick)


if __name__ == '__main__':
    main()
