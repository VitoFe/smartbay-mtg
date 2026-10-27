"""Does physical guidance earn its place at longer lead times? (protocol v2, development only)

Every comparison in Chapter 5 is made at a six-hour horizon, where persistence is already a strong
predictor of a diurnal signal. This module repeats the Table 5.8 contrast at 24 h and 48 h, so the
question "when does physical guidance help?" is answered as a function of lead time rather than at
one point. Same folds, same nested selection, same architectures; only the horizon changes.

Two reporting changes relative to `v2_forecast.py`:
  * the `baseline` column carries the **climatological-increment** baseline,
    fitted on each outer fold's training rows only, instead of repeating persistence;
  * neural arms are scored in forecast mode only, which is the mode the contrast is about.

No 2026 value is loaded: the hourly index ends 2025-12-31T23:00 and every target lies inside it.
"""
import argparse, hashlib, json, os, traceback
import numpy as np
import pandas as pd
import torch
from src.experiments.bay_graph_dynamics import ROOT, metrics, save
from src.data.bay_profile_context import load_observations
from src.experiments.bay_dynamics_tree import features
from src.experiments.v2_forecast import (TREE_ARMS, NEURAL_ARMS, OXYGEN_ONLY, SEEDS, THREADS,
                                         graphs, tree_space, neural_space, xgb_space, make_tree)
from src.models.bay_graph_dynamics import BayGraphDynamics
from src.data.bay_forcing import weather
from src.tuning.protocol_v2 import run_study, PROTOCOL

HORIZONS = [1, 3, 6, 12, 24, 36, 48]  # 1/3/12/36 added under addendum-3 amendment_1
STRIDE = 6  # issuance cadence, held fixed so the endpoint grid is comparable across horizons


def build(variable, horizon):
    """`bay_graph_dynamics.build` generalised to an arbitrary horizon; identical at horizon 6."""
    meta, frames = load_observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    meta = meta.loc[names]
    index = pd.date_range('2024-07-24', '2025-12-31T23:00:00', freq='h', tz='UTC')
    raw = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    center, scale = (20., 10.) if variable == 'temperature' else (230., 50.)
    latest = (raw.ffill(limit=6) - center) / scale
    z = (raw - center) / scale
    q = np.arange(24, len(index) - horizon, STRIDE)
    x = latest.to_numpy()[q]; available = np.isfinite(x)
    truth = z.to_numpy()[q + horizon]; valid = np.isfinite(z.to_numpy()[q]) & np.isfinite(truth)
    trend = z.to_numpy()[q] - z.to_numpy()[q - 6]
    xy = meta[['x_m', 'y_m']].to_numpy(); distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    adjacency = np.exp(-distance / 1500); np.fill_diagonal(adjacency, 0)
    static = np.column_stack([(xy[:, 0] - 568000) / 5000, (xy[:, 1] - 4880000) / 5000, meta.nominal_depth_m / 10])
    calendar = np.column_stack([np.sin(index[q].dayofyear * 2 * np.pi / 365.25), np.cos(index[q].dayofyear * 2 * np.pi / 365.25),
                                np.sin(index[q].hour * 2 * np.pi / 24), np.cos(index[q].hour * 2 * np.pi / 24)])
    feats = [np.nan_to_num(trend)[..., None], np.isfinite(trend)[..., None], available[..., None],
             np.broadcast_to(static, (len(q), *static.shape)), np.broadcast_to(calendar[:, None, :], (len(q), len(names), 4))]
    for col, offset, divisor in [('temperature', 20, 10), ('salinity', 35, 5)]:
        a = pd.DataFrame({n: frames[n][col].reindex(index) if col in frames[n] else np.nan for n in names},
                         index=index).ffill(limit=6).to_numpy()[q]
        feats.extend([np.nan_to_num((a - offset) / divisor)[..., None], np.isfinite(a)[..., None]])
    forcing = np.concatenate(feats, axis=-1).astype('float32')
    future = np.stack([z.to_numpy()[q + h] for h in range(1, horizon + 1)], axis=1)
    return dict(names=names, times=index[q], x=x, available=available, truth=truth, valid=valid, forcing=forcing,
                future=future, adjacency=adjacency, near=distance <= 250, distance=distance,
                center=center, scale=scale, horizon=horizon)


def folds(data, horizon):
    """The two v1 development folds, with the training gap widened to the horizon."""
    t = data['times']; out = []
    for number, (start, end) in enumerate([('2025-07-01', '2025-10-01'), ('2025-10-01', '2026-01-01')]):
        begin = pd.Timestamp(start, tz='UTC'); finish = pd.Timestamp(end, tz='UTC')
        tr = np.flatnonzero(t + pd.Timedelta(hours=horizon + 72) < begin)
        va = np.flatnonzero((t >= begin) & (t + pd.Timedelta(hours=horizon) < finish))
        eligible = (data['valid'][tr].sum(0) >= 100) & (data['valid'][va].sum(0) >= 30)
        if not eligible.any(): continue
        fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
        out.append(dict(number=number, tr=tr, va=va, eligible=eligible, fallback=fallback,
                        audit={'fold': number, 'horizon_hours': horizon,
                               'last_training_outcome': str(t[tr].max() + pd.Timedelta(hours=horizon)),
                               'first_selection': str(t[va].min()), 'selection_end_exclusive': end,
                               'eligible_sensors': [n for n, e in zip(data['names'], eligible) if e]}))
    return out


def inner_fold(data, fold, horizon):
    t = data['times']; begin = t[fold['va']].min()
    end = begin - pd.Timedelta(hours=72); start = end - pd.Timedelta(days=45)
    tr = np.flatnonzero(t + pd.Timedelta(hours=horizon + 72) < start)
    va = np.flatnonzero((t >= start) & (t + pd.Timedelta(hours=horizon) < end))
    eligible = fold['eligible'] & (data['valid'][tr].sum(0) >= 100) & (data['valid'][va].sum(0) >= 10)
    fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
    return dict(number=fold['number'], tr=tr, va=va, eligible=eligible, fallback=fallback,
                audit={'fold': fold['number'], 'horizon_hours': horizon,
                       'inner_selection_start': str(t[va].min()), 'inner_selection_end': str(end),
                       'last_inner_training_outcome': str(t[tr].max() + pd.Timedelta(hours=horizon))})


def climatology(data, fold):
    """Persistence plus the mean horizon-length increment for that sensor x month x issuance hour.

    Same construction as the reserved-period baseline `B1_clim_increment`
    (`scripts/experiments/exp_forecast.py`), but fitted on this fold's training rows only
    rather than on all pre-2026 data. Cells absent from training fall back to that sensor's mean
    increment and then to the global mean. Parameter-free, so it needs no Optuna study.
    """
    t = data['times']; n = data['x'].shape[1]
    increment = data['truth'] - data['x']
    tr = fold['tr']; tm = data['valid'][tr]
    j, s = np.nonzero(tm)
    frame = pd.DataFrame({'sensor': s, 'month': t.month.to_numpy()[tr][j], 'hour': t.hour.to_numpy()[tr][j],
                          'd': increment[tr][tm]})
    per_sensor = frame.groupby('sensor').d.mean().reindex(np.arange(n)).to_numpy()
    overall = float(frame.d.mean())
    fallback = np.where(np.isfinite(per_sensor), per_sensor, overall)
    grid = np.broadcast_to(np.arange(n), (len(t), n)).ravel()
    month = np.repeat(t.month.to_numpy(), n); hour = np.repeat(t.hour.to_numpy(), n)

    def lookup(table, keys):
        d = table.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy().reshape(len(t), n)
        return data['x'] + np.where(np.isfinite(d), d, fallback[None, :])
    # month x hour is the baseline Chapter 5 uses; on these folds each month is seen about once,
    # so the better-fitted hour-only table is carried alongside it rather than instead of it.
    return (lookup(frame.groupby(['sensor', 'month', 'hour']).d.mean(), [grid, month, hour]),
            lookup(frame.groupby(['sensor', 'hour']).d.mean(), [grid, hour]))


def rows_from(data, fold, predicted, clim):
    month_hour, hour_only = clim
    mask = data['valid'][fold['va']] & fold['eligible'][None, :]; out = []
    for j, s in np.argwhere(mask):
        i = fold['va'][j]
        out.append({'fold': fold['number'], 'sensor': data['names'][s], 'time': str(data['times'][i]), 'mode': 'forecast',
                    'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                    'prediction': float(predicted[j, s] * data['scale'] + data['center']),
                    'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                    'baseline': float(month_hour[i, s] * data['scale'] + data['center']),
                    'clim_hour': float(hour_only[i, s] * data['scale'] + data['center'])})
    return pd.DataFrame(out)


def tree_fit_score(data, x, fold, params, clim):
    mask = data['valid'] & fold['eligible'][None, :]; tr, va = fold['tr'], fold['va']; tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]; w = w / w.mean()
    model = make_tree(params)
    model.fit(x[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    grid = np.full(data['x'][va].shape, np.nan); vm = mask[va]
    grid[vm] = model.predict(x[va][vm]) + data['x'][va][vm]
    return rows_from(data, fold, grid, clim), model


def neural_forecast(data, fold, params, exchange, relaxation, seed, device, clim, quick=False):
    torch.manual_seed(seed)
    arrays = {k: data[k] for k in ['available', 'forcing', 'truth']}
    arrays['x'] = np.where(np.isfinite(data['x']), data['x'], fold['fallback'])
    arrays['valid'] = data['valid'] & fold['eligible'][None, :]
    a = {k: torch.as_tensor(np.nan_to_num(v), device=device) for k, v in arrays.items()}
    a = {k: v.float() if v.dtype != torch.bool else v for k, v in a.items()}
    model = BayGraphDynamics(data['adjacency'], data['forcing'].shape[-1], params['hidden'], exchange, relaxation).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    weights = 1 / a['valid'][fold['tr']].sum(0).clamp_min(1).float(); weights /= weights.mean()
    rng = np.random.default_rng(seed)
    for _ in range(3 if quick else params['steps']):
        ids = rng.choice(fold['tr'], min(128, len(fold['tr'])), replace=False)
        pred, _ = model.rollout(a['x'][ids], a['available'][ids], a['forcing'][ids], hours=data['horizon'])
        mask = a['valid'][ids]
        loss = (torch.nn.functional.smooth_l1_loss(pred, a['truth'][ids], reduction='none') * mask * weights).sum() / (mask * weights).sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
        if not torch.isfinite(loss): raise ValueError('non-finite loss')
    with torch.no_grad():
        pred, _ = model.rollout(a['x'][fold['va']], a['available'][fold['va']], a['forcing'][fold['va']], hours=data['horizon'])
    return rows_from(data, fold, pred.cpu().numpy(), clim)


def ratio(d):
    m = metrics(d); return m['forecast_prediction_mae'] / m['forecast_persistence_mae']


def run(variable, arm, horizon, quick=False):
    output = ROOT / 'results'; tag = f'v2_horizon_{variable}_{arm}_h{horizon}'
    folder = output / tag; folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_{tag}.json'
    save(status, {'state': 'RUNNING', 'variable': variable, 'arm': arm, 'horizon': horizon, 'pid': os.getpid()})
    try:
        if arm in OXYGEN_ONLY and variable != 'oxygen': raise ValueError('saturation arms are oxygen-only')
        if horizon not in HORIZONS: raise ValueError(f'undeclared horizon {horizon}')
        torch.set_num_threads(THREADS); device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = build(variable, horizon); outer = folds(data, horizon)
        if quick:
            for f in outer: f['tr'] = f['tr'][-300:]; f['va'] = f['va'][:40]
        inner = [inner_fold(data, f, horizon) for f in outer]
        if quick:
            for f in inner: f['tr'] = f['tr'][-300:]; f['va'] = f['va'][:40]
        clim = {id(f): climatology(data, f) for f in outer + inner}
        contract = {'protocol': PROTOCOL['version'], 'variable': variable, 'arm': arm, 'horizon_hours': horizon,
                    'issuance_stride_hours': STRIDE, 'smoke_only': quick, 'device': device,
                    'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
                    'selection': PROTOCOL['selection']['comparisons'],
                    'objective': 'mean over folds of fold-macro forecast MAE / persistence MAE',
                    'baseline_column': 'climatological increment, month x issuance hour, fitted on outer training rows only',
                    'scope': 'Development folds only; hourly index ends 2025-12-31T23:00, so no 2026 value is loaded.',
                    'no_2026': True}
        if arm in TREE_ARMS:
            met = weather(data['times']).to_numpy() if arm.endswith('weather_tree') else None
            xgbarm = arm.startswith('xgb_')

            def design(fold):
                base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
                if met is None: return base
                return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))], axis=-1)
            cache = {id(f): design(f) for f in outer + inner}

            def objective(trial):
                p = xgb_space(trial) if xgbarm else tree_space(trial)
                return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))
            save(folder / 'contract.json', contract)
            sel = run_study(f'{variable}_{arm}_h{horizon}', objective, folder, quick)
            params = dict(sel['params']) if xgbarm else {**sel['params'], 'subsample_freq': 1}
            frames = []
            for f in outer:
                d, model = tree_fit_score(data, cache[id(f)], f, params, clim[id(f)]); frames.append(d)
            d = pd.concat(frames, ignore_index=True); seeds = [42]
        else:
            graph, exchange, relaxation = NEURAL_ARMS[arm]; data['adjacency'] = graphs(data)[graph]

            def objective(trial):
                p = neural_space(trial)
                return float(np.mean([ratio(neural_forecast(data, f, p, exchange, relaxation, 42, device, clim[id(f)], quick)) for f in inner]))
            contract.update(graph=graph, exchange=exchange, relaxation=relaxation)
            save(folder / 'contract.json', contract)
            sel = run_study(f'{variable}_{arm}_h{horizon}', objective, folder, quick); params = sel['params']
            seeds = [42] if quick else SEEDS
            frames = [neural_forecast(data, f, params, exchange, relaxation, seed, device, clim[id(f)], quick).assign(seed=seed)
                      for seed in seeds for f in outer]
            d = pd.concat(frames, ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'; d.to_csv(path, index=False)
        per_seed = {int(s): metrics(g.drop(columns='seed')) for s, g in d.groupby('seed')} if 'seed' in d else {42: metrics(d)}
        report = {'selection': sel, 'seeds': seeds, 'horizon_hours': horizon, 'per_seed_metrics': per_seed,
                  'endpoints': int(len(d) / len(seeds)),
                  'mean_forecast_mae': float(np.mean([m['forecast_prediction_mae'] for m in per_seed.values()])),
                  'mean_persistence_mae': float(np.mean([m['forecast_persistence_mae'] for m in per_seed.values()])),
                  'mean_climatology_mae': float(np.mean([m['forecast_baseline_mae'] for m in per_seed.values()])),
                  'mean_climatology_hour_mae': float(d.assign(e=(d.truth - d.clim_hour).abs())
                                                     .groupby(['fold', 'sensor']).e.mean().groupby('fold').mean().mean()),
                  'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        save(folder / 'report.json', report)
        save(status, {'state': 'COMPLETED', 'variable': variable, 'arm': arm, 'horizon': horizon,
                      'result_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'variable': variable, 'arm': arm, 'horizon': horizon, 'error': str(e)})
        traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=TREE_ARMS + list(NEURAL_ARMS), required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--quick', action='store_true'); a = p.parse_args()
    run(a.variable, a.arm, a.horizon, a.quick)
