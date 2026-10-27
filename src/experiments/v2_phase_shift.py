"""Addendum-16 (E2): does the six-hour harm move when the issuance grid moves?

At six hours on oxygen, adding meteorological forcing to the residual tree makes it worse by
0.217 umol/L with an interval excluding zero, while at 24 h and beyond the same features help.
The thesis reads that spike as a property of the six-hour lead. An alternative reading, which two
audits proposed, is that it is a property of WHICH points of the diurnal cycle the grid samples:
issuance at 00, 06, 12, 18 UTC means that at a six-hour horizon every target lands on the same
four hours.

Shifting the grid by three hours separates the two. The lead stays at six hours - a quarter of a
diurnal cycle - while the clock hours sampled all change.

`build` is reproduced here with a phase offset rather than edited in place, because
v2_horizon.build is contract-bound and identical output at offset 0 is the point.

Development folds only. Declared in config/bay_reserved_protocol_v2_addendum16.json.
Usage: python -m src.experiments.v2_phase_shift --offset 3
"""
import argparse
import hashlib
import zlib

import numpy as np
import pandas as pd

from src.data.bay_forcing import weather
from src.data.bay_profile_context import load_observations
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import make_tree, tree_space
from src.experiments.v2_horizon import ROOT, STRIDE, climatology, folds, inner_fold
from src.tuning.protocol_v2 import run_study

OUT = ROOT / 'reports/bay_twin/runs/v2_phase_shift'
HORIZON = 6
DRAWS = 600


def output_dir(horizon):
    """Six-hour results keep their published path; any other horizon gets its own.

    Addendum 29 sweeps the same six phases at a 24 h lead, where the geometry differs because
    the target falls on the same clock hour as the issuance. Writing beside the published
    six-hour artifact would change its digest for no reason.
    """
    return OUT if horizon == HORIZON else ROOT / f'reports/bay_twin/runs/v2_phase_shift_h{horizon}'


def build_shifted(variable, horizon, offset):
    """v2_horizon.build with the issuance grid advanced by `offset` hours."""
    meta, frames = load_observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    meta = meta.loc[names]
    index = pd.date_range('2024-07-24', '2025-12-31T23:00:00', freq='h', tz='UTC')
    raw = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    center, scale = (20., 10.) if variable == 'temperature' else (230., 50.)
    latest = (raw.ffill(limit=6) - center) / scale
    z = (raw - center) / scale
    q = np.arange(24 + offset, len(index) - horizon, STRIDE)
    x = latest.to_numpy()[q]
    available = np.isfinite(x)
    truth = z.to_numpy()[q + horizon]
    valid = np.isfinite(z.to_numpy()[q]) & np.isfinite(truth)
    trend = z.to_numpy()[q] - z.to_numpy()[q - 6]
    xy = meta[['x_m', 'y_m']].to_numpy()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    adjacency = np.exp(-distance / 1500)
    np.fill_diagonal(adjacency, 0)
    static = np.column_stack([(xy[:, 0] - 568000) / 5000, (xy[:, 1] - 4880000) / 5000,
                              meta.nominal_depth_m / 10])
    calendar = np.column_stack([np.sin(index[q].dayofyear * 2 * np.pi / 365.25),
                                np.cos(index[q].dayofyear * 2 * np.pi / 365.25),
                                np.sin(index[q].hour * 2 * np.pi / 24),
                                np.cos(index[q].hour * 2 * np.pi / 24)])
    feats = [np.nan_to_num(trend)[..., None], np.isfinite(trend)[..., None], available[..., None],
             np.broadcast_to(static, (len(q), *static.shape)),
             np.broadcast_to(calendar[:, None, :], (len(q), len(names), 4))]
    for col, offs, div in [('temperature', 20, 10), ('salinity', 35, 5)]:
        a = pd.DataFrame({n: frames[n][col].reindex(index) if col in frames[n] else np.nan
                          for n in names}, index=index).ffill(limit=6).to_numpy()[q]
        feats.extend([np.nan_to_num((a - offs) / div)[..., None], np.isfinite(a)[..., None]])
    forcing = np.concatenate(feats, axis=-1).astype('float32')
    future = np.stack([z.to_numpy()[q + h] for h in range(1, horizon + 1)], axis=1)
    return dict(names=names, times=index[q], x=x, available=available, truth=truth, valid=valid,
                forcing=forcing, future=future, adjacency=adjacency, near=distance <= 250,
                distance=distance, center=center, scale=scale, horizon=horizon)


def design_for(data, fold, with_weather):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    if not with_weather:
        return base
    met = weather(data['times']).to_numpy()
    n = len(data['names'])
    return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))],
                          axis=-1)


def fit_grid(data, fold, design, params):
    mask = data['valid'] & fold['eligible'][None, :]
    tr, va = fold['tr'], fold['va']
    tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]
    w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})
    model.fit(design[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    grid = np.full(data['x'][va].shape, np.nan)
    vm = mask[va]
    grid[vm] = model.predict(design[va][vm]) + data['x'][va][vm]
    return grid


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def paired(frame, a, b, label):
    rng = np.random.default_rng([20260921, zlib.crc32(label.encode())])
    f = frame.assign(week=pd.to_datetime(frame.time, utc=True).dt.isocalendar().week.astype(int))
    blocks = f[['fold', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in f.groupby(['fold', 'week'])}
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(macro(s, a) - macro(s, b))
    return [macro(frame, a) - macro(frame, b), *np.percentile(draws, [2.5, 97.5]).tolist()]


#: The selection objective. The published arms (v2_forecast, v2_horizon) are tuned on the mean over
#: inner folds of fold-macro MAE divided by persistence MAE; addenda 16, 17 and 29 tuned the shifted
#: phases on a flat pooled MAE instead, so the one published phase was also the one tuned on the
#: published objective. 'ratio' removes that confound; 'flat' reproduces the sweep
#: as first run, and keeps its file names.
OBJECTIVES = ('flat', 'ratio')


def run(offset, quick=False, horizon=HORIZON, objective_kind='flat', seed=None):
    # a smoke run writes beside the published files, never over them
    out = output_dir(horizon) / 'smoke' if quick else output_dir(horizon)
    out.mkdir(parents=True, exist_ok=True)
    declared = ('config/bay_reserved_protocol_v2_addendum16.json' if horizon == HORIZON
                else 'config/bay_reserved_protocol_v2_addendum29.json')
    if objective_kind not in OBJECTIVES:
        raise ValueError(objective_kind)
    # A second sampler seed changes nothing but the TPE stream; run_study reads the protocol's seed
    # at call time, so the bound tuning module is left as it is.
    suffix = ('' if objective_kind == 'flat' else f'_{objective_kind}') + \
             ('' if seed is None else f'_seed{seed}')
    if seed is not None:
        from src.tuning import protocol_v2
        protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    report = {'declared_in': declared, 'objective': objective_kind,
              'sampler_seed': None if seed is None else int(seed),
              'offset_hours': offset, 'horizon_hours': horizon, 'variables': {}}
    for variable in ['oxygen', 'temperature']:
        data = build_shifted(variable, horizon, offset)
        outer = folds(data, horizon)
        inner = [inner_fold(data, f, horizon) for f in outer]
        hours = sorted(set(pd.DatetimeIndex(data['times']).hour))
        cache = {(id(f), w): design_for(data, f, w) for f in outer + inner for w in (True, False)}
        clim = {id(f): climatology(data, f) for f in outer + inner}

        params = {}
        for with_weather, arm in [(False, 'tree'), (True, 'weather_tree')]:
            def objective(trial, w=with_weather):
                p = tree_space(trial)
                if objective_kind == 'ratio':
                    from src.experiments.v2_horizon import ratio, tree_fit_score
                    return float(np.mean([ratio(tree_fit_score(data, cache[(id(f), w)], f, p,
                                                               clim[id(f)])[0]) for f in inner]))
                scores = []
                for f in inner:
                    grid = fit_grid(data, f, cache[(id(f), w)], p)
                    m = data['valid'][f['va']] & f['eligible'][None, :]
                    scores.append(float(np.nanmean(np.abs(grid - data['truth'][f['va']])[m])))
                return float(np.mean(scores))
            sel = run_study(f'{variable}_{arm}_h{horizon}_shift{offset}{suffix}', objective, out,
                            quick)
            params[arm] = {**sel['params'], 'subsample_freq': 1}

        rows = []
        for fold in outer:
            grids = {a: fit_grid(data, fold, cache[(id(fold), a == 'weather_tree')], params[a])
                     for a in params}
            cl, _ = clim[id(fold)]
            mask = data['valid'][fold['va']] & fold['eligible'][None, :]
            for j, i in enumerate(fold['va']):
                for s, name in enumerate(data['names']):
                    if not mask[j, s]:
                        continue
                    rows.append({
                        'fold': fold['number'], 'sensor': name, 'time': str(data['times'][i]),
                        'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                        'tree': float(grids['tree'][j, s] * data['scale'] + data['center']),
                        'weather_tree': float(grids['weather_tree'][j, s] * data['scale'] + data['center']),
                        'persistence': float(data['x'][i, s] * data['scale'] + data['center']),
                        'climatology': float(cl[i, s] * data['scale'] + data['center'])})

        frame = pd.DataFrame(rows)
        path = out / f'phase_shift_{variable}_offset{offset}{suffix}.csv.gz'
        frame.to_csv(path, index=False)
        entry = {'issuance_hours_utc': hours, 'endpoints': int(len(frame)),
                 'sensors': int(frame.sensor.nunique()), 'params': params,
                 'mae': {a: macro(frame, a) for a in ['tree', 'weather_tree', 'persistence', 'climatology']},
                 'contrasts': {
                     'weather_tree - tree': paired(frame, 'weather_tree', 'tree', f'{variable}|w-t|{offset}'),
                     'tree - persistence': paired(frame, 'tree', 'persistence', f'{variable}|t-p|{offset}')},
                 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        report['variables'][variable] = entry
        print(f'== {variable}  issuance {hours}  n={entry["endpoints"]}  sensors={entry["sensors"]}')
        for a, v in entry['mae'].items():
            print(f'   {a:14s} {v:.4f}')
        for k, (d, lo, hi) in entry['contrasts'].items():
            print(f'   {k:24s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))

    save(out / f'phase_shift_offset{offset}{suffix}_summary.json', report)
    print('\nwrote', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--offset', type=int, default=3)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--horizon', type=int, default=HORIZON)
    p.add_argument('--objective', choices=OBJECTIVES, default='flat')
    p.add_argument('--seed', type=int, default=None)
    a = p.parse_args()
    run(a.offset, a.quick, a.horizon, a.objective, a.seed)
