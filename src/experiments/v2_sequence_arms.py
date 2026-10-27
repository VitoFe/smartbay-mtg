"""Addendum-23: is the six-hour task saturated, or does pretraining carry something?

The thesis reports that a zero-shot pretrained forecaster matches the bay-specific pipeline at six
hours. Two explanations fit and have opposite consequences: pretrained models carry transferable
temporal priors worth as much as local fitting, or the task is saturated and any competent sequence
model reaches the same band. The evidence cannot currently separate them, because every pretrained
arm is Chronos lineage and the only local learner compared against them is a feature-engineered
tree, which differs in two ways at once.

Four arms on identical endpoints:

  tree        the plain LightGBM residual tree, feature-engineered as the thesis builds it
  lstm        a recurrent net trained here, univariate on the probe's own history
  chronos_zs  Chronos-Bolt, zero-shot
  timesfm_zs  TimesFM 2.5 200M, zero-shot, an independent pretraining corpus

The three sequence arms receive exactly the same input: 512 hours of the probe's own z-scored
history, univariate. Giving the recurrent arm the tree's engineered features would confound
sequence modelling with feature engineering and would not answer the question. The tree is carried
alongside as the feature-engineered reference, so the table reads two ways: pretraining against
local fitting on equal information, and feature engineering against raw history.

Development folds only. No reserved-period endpoint is read: the reserved battery already carries
Chronos arms, and its one-shot guarantee is the stated reason the AOU result sits outside the
supported claims. Chronos is therefore re-scored here rather than quoted from there, so that every
contrast is paired on identical rows.

Declared in config/bay_reserved_protocol_v2_addendum23.json.
Usage: python -m src.experiments.v2_sequence_arms --variable temperature [--quick]

`--perfold-tree` (review R3, 2026-10-05): the tree is the one arm whose configuration was shared
across the two outer folds (`tree_params`, one selection on both folds' inner windows); the
recurrent arm is already selected per fold and the pretrained arms select nothing. This path refits
only the tree, per fold, at that fold's own selection (addendum 47, `v2_perfold_select`), in the
same way (`tree_predict`), and reads every other arm from the published rows, which the tree cannot
reach: same endpoints, contexts and predictions. It first refits the shared configuration and
requires the published tree column back (gate G1). Writes `runs/v2_sequence_arms/perfold_tree/`
(`perfold_tree_smoke/` with `--quick`) under the published file names; no GPU is needed.
  python -m src.experiments.v2_sequence_arms --variable oxygen --perfold-tree
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

from src.data.bay_profile_context import ROOT, load_observations
from src.experiments.bay_dynamics_tree import features
from src.experiments.v2_forecast import SEEDS, THREADS, make_tree, tree_space
from src.experiments.v2_horizon import build, folds
from src.tuning.protocol_v2 import run_study

warnings.filterwarnings('ignore')
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')

OUT = ROOT / 'reports/bay_twin/runs/v2_sequence_arms'
CTX, HORIZON = 512, 6          # the context and lead the existing Chronos work used
MIN_FINITE = 48                # same admissibility rule as exp_chronos.predict_at
CHRONOS_REPO = 'amazon/chronos-bolt-base'
TIMESFM_REPO = 'google/timesfm-2.5-200m-pytorch'
DRAWS = 600                    # paired week-block bootstrap, the forecast convention


def hourly_panel(variable):
    """The z-scored hourly panel `build` works from, rebuilt so history windows can be sliced.

    `build` returns only the values at issuance, not the series behind them. The construction here
    is copied from it deliberately; `assert_aligned` then checks the two agree on every endpoint,
    so a drift between them fails loudly instead of silently mis-pairing the arms.
    """
    meta, frames = load_observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range('2024-07-24', '2025-12-31T23:00:00', freq='h', tz='UTC')
    raw = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    center, scale = (20., 10.) if variable == 'temperature' else (230., 50.)
    return names, index, ((raw - center) / scale).to_numpy().astype('float32'), scale


def assert_aligned(data, z, index):
    """The panel must reproduce `build`'s targets exactly, or the endpoints are not the same ones."""
    q = np.searchsorted(index, data['times'])
    assert (index[q] == data['times']).all(), 'issuance times do not line up'
    got = z[q + HORIZON]
    finite = np.isfinite(data['truth']) & np.isfinite(got)
    assert finite.sum() > 0, 'no overlapping finite targets'
    assert np.allclose(got[finite], data['truth'][finite], atol=1e-6), 'targets disagree with build'
    return q


def context_window(z, q_i, s):
    """The 512 hours ending at the issuance, for one sensor, with interior gaps interpolated.

    Chronos accepts a mask and the other two arms do not, so an identical imputed context is given
    to all three. Parity matters more here than matching the reserved battery's convention, and the
    two are not placed in the same table for exactly this reason.
    """
    lo = max(0, q_i - CTX + 1)
    hist = z[lo:q_i + 1, s]
    if np.isfinite(hist).sum() < MIN_FINITE:
        return None
    s_ = pd.Series(hist).interpolate(limit_direction='both')
    return s_.to_numpy(dtype='float32') if np.isfinite(s_).all() else None


def gather(data, z, q, fold_rows):
    """Contexts and targets for a list of (endpoint index, sensor index) pairs."""
    ctxs, keep = [], []
    for i, s in fold_rows:
        w = context_window(z, q[i], s)
        if w is None:
            continue
        if len(w) < CTX:                      # pad the earliest windows on the left with the edge
            w = np.concatenate([np.full(CTX - len(w), w[0], dtype='float32'), w])
        ctxs.append(w)
        keep.append((i, s))
    return (np.stack(ctxs) if ctxs else np.zeros((0, CTX), 'float32')), keep


def eligible_rows(data, fold, which):
    mask = fold['eligible']
    return [(i, s) for i in fold[which] for s in range(len(data['names']))
            if mask[s] and data['valid'][i, s]]


# ---------------------------------------------------------------- zero-shot arms

def chronos_predict(ctxs, batch=256):
    from chronos import BaseChronosPipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS_REPO, device_map=device,
                                               torch_dtype=torch.float32)
    out = []
    for k in range(0, len(ctxs), batch):
        chunk = [torch.tensor(c) for c in ctxs[k:k + batch]]
        q, _ = pipe.predict_quantiles(inputs=chunk, prediction_length=HORIZON,
                                      quantile_levels=[0.1, 0.5, 0.9])
        out.append(q[:, HORIZON - 1, 1].float().cpu().numpy())
    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.concatenate(out) if out else np.zeros(0, 'float32')


def timesfm_predict(ctxs, batch=256):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
    model.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=HORIZON,
                                         normalize_inputs=True))
    out = []
    for k in range(0, len(ctxs), batch):
        point, _ = model.forecast(horizon=HORIZON, inputs=list(ctxs[k:k + batch]))
        out.append(np.asarray(point)[:, HORIZON - 1])
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.concatenate(out) if out else np.zeros(0, 'float32')


# ---------------------------------------------------------------- the local recurrent arm

class Recurrent(torch.nn.Module):
    """A GRU over the context, predicting the change from the last observed value.

    Predicting the increment rather than the level is a reparameterisation, not extra information:
    the last value is the final element of the context the other arms also receive.
    """

    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.rnn = torch.nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1]).squeeze(-1) + x[:, -1]


def rnn_space(trial):
    return {'hidden': trial.suggest_int('hidden', 16, 128, log=True),
            'layers': trial.suggest_int('layers', 1, 2),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'epochs': trial.suggest_int('epochs', 4, 30),
            'batch': trial.suggest_categorical('batch', [64, 128, 256])}


def fit_recurrent(train_x, train_y, params, seed, device, quick=False):
    torch.manual_seed(seed)
    model = Recurrent(params['hidden'], params['layers'], params['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                            weight_decay=params['weight_decay'])
    x = torch.tensor(train_x, device=device)
    y = torch.tensor(train_y, device=device)
    n, batch = len(x), params['batch']
    rng = np.random.default_rng(seed)
    for _ in range(2 if quick else params['epochs']):
        for k in range(0, n, batch):
            sel = torch.tensor(rng.choice(n, min(batch, n), replace=False), device=device)
            loss = torch.nn.functional.l1_loss(model(x[sel]), y[sel])
            if not torch.isfinite(loss):
                # This configuration has diverged. Stopping and letting the caller score it as bad
                # keeps it a *completed* trial: a diverged configuration genuinely is a bad one, and
                # failing it instead would silently shrink the search the protocol asks for.
                model.eval()
                return model
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def recurrent_predict(model, ctxs, device, batch=512):
    out = []
    with torch.no_grad():
        for k in range(0, len(ctxs), batch):
            out.append(model(torch.tensor(ctxs[k:k + batch], device=device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, 'float32')


# ---------------------------------------------------------------- the tree reference

def tree_predict(data, fold, params, quick=False):
    """The plain residual tree, fitted exactly as the horizon sweep fits it."""
    x_feat, anchor = features(data, {'fallback': fold['fallback']}, 'forecast')
    flat = x_feat.reshape(-1, x_feat.shape[-1])
    tr_mask = np.zeros(x_feat.shape[:2], bool)
    tr_mask[np.ix_(fold['tr'], np.arange(x_feat.shape[1]))] = True
    tr_mask &= data['valid'] & fold['eligible'][None, :]
    model = make_tree(params)
    model.fit(flat[tr_mask.reshape(-1)],
              (data['truth'] - data['x'])[tr_mask],
              sample_weight=None)
    pred = np.full(data['truth'].shape, np.nan, 'float32')
    va_mask = np.zeros(x_feat.shape[:2], bool)
    va_mask[np.ix_(fold['va'], np.arange(x_feat.shape[1]))] = True
    va_mask &= data['valid'] & fold['eligible'][None, :]
    pred[va_mask] = model.predict(flat[va_mask.reshape(-1)]) + anchor[va_mask]
    return pred


TREE_SELECTION = ROOT / 'reports/bay_twin/runs/20260919T144822Z_v2_forecast/results'


def tree_params(variable):
    """The plain tree's selection from the bound forecast run: same data, same folds, same
    protocol, one hundred completed trials. Re-running an identical study would spend GPU hours
    to reproduce it, so it is loaded and its provenance recorded instead."""
    p = TREE_SELECTION / f'v2_forecast_{variable}_tree_nested/{variable}_tree_nested_selection.json'
    sel = json.loads(p.read_text())
    assert sel['completed_trials'] >= 100 and not sel.get('smoke_only'), 'tree selection is not v2'
    return sel['params'], p.relative_to(ROOT).as_posix()


def macro(frame, column):
    """Mean over quarters of the per-sensor mean absolute error, in physical units."""
    f = frame.dropna(subset=[column])
    cell = f.assign(e=(f[column] - f.truth).abs()).groupby(['quarter', 'sensor']).e.mean()
    return float(cell.groupby('quarter').mean().mean())


def paired_bootstrap(frame, a, b, scale, seed=20260921, draws=DRAWS):
    """Week-block resampling within fold, the forecast convention."""
    rng = np.random.default_rng(seed)
    f = frame.dropna(subset=[a, b]).copy()
    f['block'] = (f.fold.astype(str) + '|' +
                  pd.to_datetime(f.time, utc=True).dt.isocalendar().week.astype(str))
    blocks = f.block.unique()
    by_block = {k: g for k, g in f.groupby('block')}
    out = []
    for _ in range(draws):
        pick = rng.choice(blocks, len(blocks), replace=True)
        draw = pd.concat([by_block[k] for k in pick], ignore_index=True)
        out.append((macro(draw, a) - macro(draw, b)) * scale)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [round((macro(f, a) - macro(f, b)) * scale, 4), round(float(lo), 4), round(float(hi), 4)]


def run(variable, quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = build(variable, HORIZON)
    names, index, z, scale = hourly_panel(variable)
    assert names == data['names'], 'sensor order differs from build'
    q = assert_aligned(data, z, index)
    outer = folds(data, HORIZON)
    tp, tp_source = tree_params(variable)

    rows = []
    for fi, fold in enumerate(outer):
        va = eligible_rows(data, fold, 'va')
        tr = eligible_rows(data, fold, 'tr')
        va_ctx, va_keep = gather(data, z, q, va)
        tr_ctx, tr_keep = gather(data, z, q, tr)
        if not len(va_keep) or not len(tr_keep):
            continue
        va_y = np.array([data['truth'][i, s] for i, s in va_keep], 'float32')
        tr_y = np.array([data['truth'][i, s] for i, s in tr_keep], 'float32')

        # The recurrent arm: one study per fold, selected on an inner split of the training rows.
        cut = int(0.8 * len(tr_ctx))
        order = np.random.default_rng(20260921 + fi).permutation(len(tr_ctx))
        inner_tr, inner_va = order[:cut], order[cut:]

        def objective(trial, _x=tr_ctx, _y=tr_y, _a=inner_tr, _b=inner_va):
            p = rnn_space(trial)
            m = fit_recurrent(_x[_a], _y[_a], p, 42, device, quick)
            score = float(np.abs(recurrent_predict(m, _x[_b], device) - _y[_b]).mean())
            # A diverged configuration scores as bad rather than failing, so the study still
            # completes the hundred trials the protocol requires. Optuna minimises, so any finite
            # value far above the achievable error removes it from contention without ending it.
            return score if np.isfinite(score) else 1e6

        sel = run_study(f'{variable}_rnn_fold{fi}', objective, OUT, quick)
        seeds = [42] if quick else SEEDS
        preds = [recurrent_predict(fit_recurrent(tr_ctx, tr_y, sel['params'], s, device, quick),
                                   va_ctx, device) for s in seeds]
        rnn_pred = np.mean(preds, axis=0)

        chr_pred = chronos_predict(va_ctx)
        tfm_pred = timesfm_predict(va_ctx)
        tree_pred = tree_predict(data, fold, tp, quick)

        for k, (i, s) in enumerate(va_keep):
            rows.append({'fold': fi, 'sensor': data['names'][s],
                         'time': str(data['times'][i]),
                         'quarter': pd.Timestamp(data['times'][i]).quarter,
                         'truth': float(data['truth'][i, s]),
                         'tree': float(tree_pred[i, s]),
                         'lstm': float(rnn_pred[k]),
                         'chronos_zs': float(chr_pred[k]),
                         'timesfm_zs': float(tfm_pred[k])})

    frame = pd.DataFrame(rows)
    arms = ['tree', 'lstm', 'chronos_zs', 'timesfm_zs']
    paired = frame.dropna(subset=arms)           # every contrast on identical rows
    report = {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum23.json',
        'cohort': 'Forecast-2025 development folds; NO reserved-period endpoint is read',
        'variable': variable, 'horizon_hours': HORIZON, 'context_hours': CTX,
        'quick': quick, 'device': device,
        'tree_params_from': tp_source,
        'endpoints_scored': int(len(paired)),
        'endpoints_dropped_for_pairing': int(len(frame) - len(paired)),
        'mae': {a: round(macro(paired, a) * scale, 4) for a in arms},
        'contrasts': {}, 'floors': {'temperature': 0.029, 'oxygen': 1.8}[variable],
    }
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            report['contrasts'][f'{a} - {b}'] = paired_bootstrap(paired, a, b, scale)

    tag = f'{variable}{"_quick" if quick else ""}'
    (OUT / f'sequence_arms_{tag}.json').write_text(json.dumps(report, indent=2))
    paired.to_csv(OUT / f'sequence_arms_{tag}.csv.gz', index=False, compression='gzip')
    print(json.dumps({k: v for k, v in report.items() if k != 'contrasts'}, indent=2))
    for k, v in report['contrasts'].items():
        star = '*' if v[1] * v[2] > 0 else ' '
        print(f'  {k:28s} {v[0]:+8.4f} [{v[1]:+.4f}, {v[2]:+.4f}] {star}')
    return report


GZIP = {'method': 'gzip', 'mtime': 0}
#: the replayed shared tree against the published column: the same fit on another host
REPLAY_TOLERANCE = 1e-6


def _sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:      # a redirected OUT (smoke runs)
        return path.as_posix()


def tree_on_rows(data, fold, params, rows, tindex, sindex):
    """tree_predict's grid at the (time, sensor) of each published row, as run() stores it."""
    grid = tree_predict(data, fold, params)
    i = rows.time.map(tindex).to_numpy()
    s = rows.sensor.map(sindex).to_numpy()
    return np.array([float(v) for v in grid[i, s]])


def run_perfold_tree(variable, quick=False):
    """The tree arm per outer fold, every other arm from the published rows (module docstring)."""
    from src.experiments import v2_perfold_select as pfs
    out = OUT / ('perfold_tree_smoke' if quick else 'perfold_tree')
    out.mkdir(parents=True, exist_ok=True)
    data = build(variable, HORIZON)
    names, index, z, scale = hourly_panel(variable)
    assert names == data['names'], 'sensor order differs from build'
    assert_aligned(data, z, index)
    outer = folds(data, HORIZON)
    source = OUT / f'sequence_arms_{variable}.csv.gz'
    published = pd.read_csv(source, float_precision='round_trip')   # the other arms, digit for digit
    tindex = {str(t): i for i, t in enumerate(data['times'])}
    sindex = {n: j for j, n in enumerate(data['names'])}
    assert published.time.map(tindex).notna().all() and published.sensor.map(sindex).notna().all()
    tp, tp_source = tree_params(variable)

    frame = published.copy()
    gate, records = {}, {}
    for fi, fold in enumerate(outer):
        assert fold['number'] == fi, 'run() labels folds by position'
        rows = published[published.fold == fi]
        if rows.empty:
            continue
        replay = tree_on_rows(data, fold, tp, rows, tindex, sindex)
        d = np.abs(replay - rows.tree.to_numpy()) * scale
        gate[str(fi)] = {'rows': int(len(rows)), 'max_abs_diff': float(d.max())}
        rec = pfs.forecast(variable, 'tree', HORIZON, fi)
        records[str(fi)] = {k: rec[k] for k in ('source', 'sha256', 'study', 'completed_trials', 'params')}
        frame.loc[rows.index, 'tree'] = tree_on_rows(data, fold, rec['params'], rows, tindex, sindex)
    gate_pass = bool(gate) and all(g['max_abs_diff'] <= REPLAY_TOLERANCE for g in gate.values())
    if not gate_pass and not quick:
        raise SystemExit(f'G1: the shared tree does not reproduce the published column {gate}; nothing written')

    arms = ['tree', 'lstm', 'chronos_zs', 'timesfm_zs']
    paired = frame.dropna(subset=arms)
    report = {
        'declared_in': 'config/bay_reserved_protocol_v2_addendum23.json',
        'review': 'R3 (2026-10-05): the tree selected per outer fold; the other arms are the published rows',
        'cohort': 'Forecast-2025 development folds; NO reserved-period endpoint is read',
        'variable': variable, 'horizon_hours': HORIZON, 'context_hours': CTX, 'quick': quick,
        'tree_selection': {'design': 'perfold',
                           'objective': "fold-macro forecast MAE / persistence MAE on that outer fold's own inner window only",
                           'records': records, 'refit': 'tree_predict, as the published tree (no per-probe weights)'},
        'shared_tree_params_from': tp_source,
        'other_arms_from': {'path': _rel(source), 'sha256': _sha(source)},
        'gate_G1_shared_tree_replay': {'by_fold': gate, 'tolerance_physical': REPLAY_TOLERANCE * scale,
                                       'pass': gate_pass},
        'endpoints_scored': int(len(paired)),
        'endpoints_dropped_for_pairing': int(len(frame) - len(paired)),
        'mae': {a: round(macro(paired, a) * scale, 4) for a in arms},
        'mae_published_tree': round(macro(published.dropna(subset=arms), 'tree') * scale, 4),
        'contrasts': {}, 'floors': {'temperature': 0.029, 'oxygen': 1.8}[variable],
    }
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            report['contrasts'][f'{a} - {b}'] = paired_bootstrap(paired, a, b, scale)
    path = out / f'sequence_arms_{variable}.csv.gz'
    paired.to_csv(path, index=False, lineterminator='\n', compression=GZIP)
    report['prediction_sha256'] = _sha(path)
    (out / f'sequence_arms_{variable}.json').write_bytes(json.dumps(report, indent=2).encode('utf-8'))
    print(json.dumps({k: v for k, v in report.items() if k not in ('contrasts', 'tree_selection')}, indent=2))
    for k, v in report['contrasts'].items():
        star = '*' if v[1] * v[2] > 0 else ' '
        print(f'  {k:28s} {v[0]:+8.4f} [{v[1]:+.4f}, {v[2]:+.4f}] {star}')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variable', default='temperature', choices=['temperature', 'oxygen'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--perfold-tree', action='store_true',
                    help='refit the tree per outer fold, read the other arms from the published rows')
    a = ap.parse_args()
    if a.perfold_tree:
        run_perfold_tree(a.variable, a.quick)
    else:
        run(a.variable, a.quick)


if __name__ == '__main__':
    main()
