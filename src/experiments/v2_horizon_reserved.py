"""Addendum-3 study 3: the 24 h horizon on Reserved-2026, read once.

The crossover reported in addendum 2 is a development-fold result. This scores it on held-out time.
Reserved-2026 has been read twice at six hours; the 24 h targets are a different endpoint set that
has never been predicted, scored or inspected, so this is a first reading for that horizon, frozen
in config/bay_reserved_protocol_v2_addendum3.json before anything below was run.

Nothing here selects anything on 2026. Every arm reuses the configuration chosen by its own
>=100-trial nested study on pre-2026 development windows, and every model is fitted on issuances
whose target falls before 2026 with a 72 h gap. The 2026 rows are predicted once and scored once.

Baselines are persistence and the 24 h climatological increment (sensor x month x hour) fitted on
the same pre-2026 rows, matching the six-hour reading's comparator set.
"""
import argparse, hashlib, json, os, zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments import bay_reserved_evaluation as v1
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_horizon import STRIDE, graphs
from src.experiments.v2_forecast import NEURAL_ARMS, SEEDS, make_tree
from src.models.bay_graph_dynamics import BayGraphDynamics

ROOT = v1.ROOT
DEV_RUN = ROOT / 'reports/bay_twin/runs/20260920T083322Z_v2_horizon/results'
OUT = ROOT / 'reports/bay_twin/runs/v2_horizon_reserved_h24'
RESERVED_START = pd.Timestamp('2026-01-01', tz='UTC')
RESERVED_END = pd.Timestamp('2026-06-19T18:00:00', tz='UTC')
DRAWS = 600


def build(variable, horizon):
    """v2_horizon.build extended through the reserved period, using the 2026-inclusive loader."""
    meta, frames = v1.load_all_observations()
    meta = meta.loc[[n for n in meta.index if n in frames]]
    names = [n for n in meta.index if variable in frames[n] and frames[n][variable].notna().any()]
    meta = meta.loc[names]
    index = pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), RESERVED_END, freq='h')
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
    return dict(names=names, times=index[q], x=x, available=available, truth=truth, valid=valid,
                forcing=forcing, adjacency=adjacency, distance=distance, near=distance <= 250,
                center=center, scale=scale, horizon=horizon)


def split(data, horizon):
    """Train on issuances whose target precedes 2026 with a 72 h gap; score the reserved period."""
    t = data['times']
    target = t + pd.Timedelta(hours=horizon)
    tr = np.flatnonzero(target + pd.Timedelta(hours=72) < RESERVED_START)
    va = np.flatnonzero((t >= RESERVED_START) & (target <= RESERVED_END))
    eligible = (data['valid'][tr].sum(0) >= 100) & (data['valid'][va].sum(0) >= 1)
    fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
    return dict(tr=tr, va=va, eligible=eligible, fallback=fallback,
                audit={'horizon_hours': horizon, 'training_issuances': len(tr), 'reserved_issuances': len(va),
                       'last_training_target': str((target[tr]).max()), 'first_reserved_issuance': str(t[va].min()),
                       'last_reserved_target': str((target[va]).max()),
                       'eligible_sensors': [n for n, e in zip(data['names'], eligible) if e]})


def climatology(data, bundle):
    t = data['times']; n = data['x'].shape[1]
    increment = data['truth'] - data['x']
    tr = bundle['tr']; tm = data['valid'][tr]
    j, s = np.nonzero(tm)
    frame = pd.DataFrame({'sensor': s, 'month': t.month.to_numpy()[tr][j], 'hour': t.hour.to_numpy()[tr][j],
                          'd': increment[tr][tm]})
    per_sensor = frame.groupby('sensor').d.mean().reindex(np.arange(n)).to_numpy()
    overall = float(frame.d.mean())
    fallback = np.where(np.isfinite(per_sensor), per_sensor, overall)
    cell = frame.groupby(['sensor', 'month', 'hour']).d.mean()
    grid = np.broadcast_to(np.arange(n), (len(t), n)).ravel()
    look = pd.MultiIndex.from_arrays([grid, np.repeat(t.month.to_numpy(), n), np.repeat(t.hour.to_numpy(), n)])
    delta = cell.reindex(look).to_numpy().reshape(len(t), n)
    return data['x'] + np.where(np.isfinite(delta), delta, fallback[None, :])


def rows_from(data, bundle, predicted, clim, arm, seed):
    mask = data['valid'][bundle['va']] & bundle['eligible'][None, :]
    t = data['times']
    out = []
    for j, s in np.argwhere(mask):
        i = bundle['va'][j]
        out.append({'arm': arm, 'seed': seed, 'sensor': data['names'][s], 'time': str(t[i]),
                    'quarter': 'Q1' if t[i].month <= 3 else 'Q2', 'week': int(t[i].isocalendar().week),
                    'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                    'prediction': float(predicted[j, s] * data['scale'] + data['center']),
                    'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                    'baseline': float(clim[i, s] * data['scale'] + data['center'])})
    return pd.DataFrame(out)


def tree_arm(data, design, bundle, params, clim, arm):
    mask = data['valid'] & bundle['eligible'][None, :]
    tr, va = bundle['tr'], bundle['va']; tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]; w = w / w.mean()
    model = make_tree(params)
    model.fit(design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    grid = np.full(data['x'][va].shape, np.nan); vm = mask[va]
    grid[vm] = model.predict(design[va][vm]) + data['x'][va][vm]
    return rows_from(data, bundle, grid, clim, arm, 42)


def neural_arm(data, bundle, params, exchange, relaxation, seed, device, clim, arm):
    torch.manual_seed(seed)
    arrays = {k: data[k] for k in ['available', 'forcing', 'truth']}
    arrays['x'] = np.where(np.isfinite(data['x']), data['x'], bundle['fallback'])
    arrays['valid'] = data['valid'] & bundle['eligible'][None, :]
    a = {k: torch.as_tensor(np.nan_to_num(v), device=device) for k, v in arrays.items()}
    a = {k: v.float() if v.dtype != torch.bool else v for k, v in a.items()}
    model = BayGraphDynamics(data['adjacency'], data['forcing'].shape[-1], params['hidden'], exchange, relaxation).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    weights = 1 / a['valid'][bundle['tr']].sum(0).clamp_min(1).float(); weights /= weights.mean()
    rng = np.random.default_rng(seed)
    for _ in range(params['steps']):
        ids = rng.choice(bundle['tr'], min(128, len(bundle['tr'])), replace=False)
        pred, _ = model.rollout(a['x'][ids], a['available'][ids], a['forcing'][ids], hours=data['horizon'])
        m = a['valid'][ids]
        loss = (torch.nn.functional.smooth_l1_loss(pred, a['truth'][ids], reduction='none') * m * weights).sum() / (m * weights).sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
        if not torch.isfinite(loss): raise ValueError('non-finite loss')
    with torch.no_grad():
        pred, _ = model.rollout(a['x'][bundle['va']], a['available'][bundle['va']], a['forcing'][bundle['va']], hours=data['horizon'])
    return rows_from(data, bundle, pred.cpu().numpy(), clim, arm, seed)


def macro(d, col='prediction'):
    d = d.dropna(subset=[col])
    return float(d.assign(e=(d.truth - d[col]).abs()).groupby(['quarter', 'sensor']).e.mean().groupby('quarter').mean().mean())


def paired(frames, x, y, col_x='prediction', col_y='prediction'):
    a, b = frames[x], frames[y]
    j = a[['quarter', 'week', 'sensor', 'time', 'truth', col_x]].merge(
        b[['sensor', 'time', col_y]], on=['sensor', 'time'], suffixes=('', '_y'), validate='one_to_one')
    rng = np.random.default_rng([20260920, zlib.crc32(f'{x}|{y}'.encode())])
    blocks = j[['quarter', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in j.groupby(['quarter', 'week'])}
    right = col_y if col_y != col_x else col_y + '_y'
    point = macro(j, col_x) - macro(j, right)
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(macro(s, col_x) - macro(s, right))
    return [point, *np.percentile(draws, [2.5, 97.5]).tolist()]


def selected(variable, arm, horizon):
    report = json.loads((DEV_RUN / f'v2_horizon_{variable}_{arm}_h{horizon}' / 'report.json').read_text())
    return report['selection']['params'], report['selection']['completed_trials']


def run(variable, horizon=24):
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(int(os.environ.get('V2_THREADS', '4')))
    data = build(variable, horizon)
    bundle = split(data, horizon)
    clim = climatology(data, bundle)
    met = v1.load_weather_2026(data['times']).to_numpy()
    base, _ = features(data, {'fallback': bundle['fallback']}, 'forecast')
    design_plain = base
    design_weather = np.concatenate(
        [base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))], axis=-1)

    frames, provenance = {}, {}
    for arm, design in [('tree', design_plain), ('weather_tree', design_weather)]:
        params, trials = selected(variable, arm, horizon)
        provenance[arm] = {'trials': trials, 'params': params}
        frames[arm] = tree_arm(data, design, bundle, {**params, 'subsample_freq': 1}, clim, arm)
    for arm in ['learned_source', 'conservative_exchange']:
        params, trials = selected(variable, arm, horizon)
        provenance[arm] = {'trials': trials, 'params': params}
        graph, exchange, relaxation = NEURAL_ARMS[arm]
        data['adjacency'] = graphs(data)[graph]
        seeds = [neural_arm(data, bundle, params, exchange, relaxation, s, device, clim, arm) for s in SEEDS]
        stacked = pd.concat(seeds, ignore_index=True)
        frames[arm] = stacked.groupby(['sensor', 'time'], as_index=False).agg(
            arm=('arm', 'first'), quarter=('quarter', 'first'), week=('week', 'first'),
            truth=('truth', 'first'), persistence=('persistence', 'first'), baseline=('baseline', 'first'),
            prediction=('prediction', 'mean'))
        frames[arm].to_csv(OUT / f'per_seed_{variable}_{arm}.csv.gz', index=False)

    reference = frames['tree']
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum3.json study_3',
              'reading': 'FIRST reading of the 24 h horizon on Reserved-2026', 'variable': variable,
              'horizon_hours': horizon, 'device': device, 'draws': DRAWS,
              'split': bundle['audit'], 'endpoints': int(len(reference)),
              'selection_provenance': provenance,
              'mae': {'persistence': macro(reference, 'persistence'),
                      'climatological_increment': macro(reference, 'baseline'),
                      **{a: macro(f) for a, f in frames.items()}},
              'contrasts': {}}
    for x, y in [('learned_source', 'weather_tree'), ('learned_source', 'tree'),
                 ('conservative_exchange', 'learned_source'), ('conservative_exchange', 'tree'),
                 ('weather_tree', 'tree')]:
        report['contrasts'][f'{x} - {y}'] = paired(frames, x, y)
    for arm in frames:
        report['contrasts'][f'{arm} - persistence'] = paired(frames, arm, arm, 'prediction', 'persistence')
        report['contrasts'][f'{arm} - climatology'] = paired(frames, arm, arm, 'prediction', 'baseline')

    everything = pd.concat(frames.values(), ignore_index=True)
    path = OUT / f'reserved_h{horizon}_{variable}.csv.gz'; everything.to_csv(path, index=False)
    report['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT / f'reserved_h{horizon}_{variable}_summary.json', report)

    print(f"== {variable} h{horizon}  n={report['endpoints']}  "
          f"sensors={len(bundle['audit']['eligible_sensors'])}  train={len(bundle['tr'])} reserved={len(bundle['va'])}")
    for k, v in report['mae'].items():
        print(f'   {k:26s} {v:.4f}')
    for k, (d, lo, hi) in report['contrasts'].items():
        print(f'   {k:44s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--horizon', type=int, default=24)
    a = p.parse_args(); run(a.variable, a.horizon)
