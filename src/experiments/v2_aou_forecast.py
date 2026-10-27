"""Addendum-15 (E5): the apparent-utilisation target inside the delivered forecaster.

The reparameterisation was tested inside a ridge, inside the spatial residual model and inside
Chronos, and never inside the released LightGBM forecaster - the one arm an operator would
actually get. Section 6.1.4 concludes from that incomplete set that oxygen skill lies outside the
pipeline. This builds the missing arm.

It also builds the baseline Section 4.5 promises and no table reports: persist apparent
utilisation, recompute dissolved oxygen from the forecast temperature. That baseline is only
distinct from oxygen persistence when the temperature at the target time is a genuine forecast -
with persistence temperature the solubility term cancels exactly - so it is given the released
temperature model's own forecast on the matched endpoints, which is also what an operator has.

    AOU(t)      = O2sol(T_obs(t), 38)   - DO(t)
    AOU(t+h)    = O2sol(T_obs(t+h), 38) - DO(t+h)      <- the training target
    DO_hat(t+h) = O2sol(T_hat(t+h), 38) - AOU_hat(t+h) <- the inversion an operator runs

The temperature build and the oxygen build share an hourly index, a stride and a horizon, so their
issuance times align exactly and their columns are matched by sensor name.

Development folds only. Declared in config/bay_reserved_protocol_v2_addendum15.json.
Usage: python -m src.experiments.v2_aou_forecast
"""
import hashlib
import zlib

import numpy as np
import pandas as pd

from src.data.bay_forcing import weather
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_closed_loop import selected
from src.experiments.v2_forecast import make_tree, tree_space
from src.experiments.v2_horizon import ROOT, build, climatology, folds, inner_fold
from src.tuning.protocol_v2 import run_study

OUT = ROOT / 'reports/bay_twin/runs/v2_aou_forecast'
HORIZON = 6
SALINITY_PSU = 38.0
DRAWS = 600


def plain_design(data, fold):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    return base


def weather_design(data, fold):
    base = plain_design(data, fold)
    met = weather(data['times']).to_numpy()
    n = len(data['names'])
    return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))],
                          axis=-1)


def fit_grid(data, fold, design, params, truth_z):
    """One residual tree: predict truth_z - data['x'], return the validation grid in z units."""
    mask = data['valid'] & fold['eligible'][None, :]
    tr, va = fold['tr'], fold['va']
    tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]
    w = w / w.mean()
    model = make_tree({**params, 'subsample_freq': 1})
    model.fit(design[tr][tm], (truth_z[tr] - data['x'][tr])[tm], sample_weight=w)
    grid = np.full(data['x'][va].shape, np.nan)
    vm = mask[va]
    grid[vm] = model.predict(design[va][vm]) + data['x'][va][vm]
    return grid


def temperature_fields(names, horizon=HORIZON):
    """Observed target-time temperature everywhere, and the released model's forecast on folds."""
    t = build('temperature', horizon)
    params = selected('temperature', 'weather_tree', horizon)
    column = {n: i for i, n in enumerate(t['names'])}
    take = [column[n] for n in names]

    observed = t['truth'][:, take] * t['scale'] + t['center']          # T_obs(t+h)
    anchor = t['x'][:, take] * t['scale'] + t['center']                # T_obs(t)
    forecast = np.full_like(observed, np.nan)
    for fold in folds(t, horizon):
        grid = fit_grid(t, fold, weather_design(t, fold), params, t['truth'])
        forecast[fold['va']] = (grid * t['scale'] + t['center'])[:, take]
    return anchor, observed, forecast


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


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data = build('oxygen', HORIZON)
    outer = folds(data, HORIZON)
    inner = [inner_fold(data, f, HORIZON) for f in outer]

    t_anchor, t_observed, t_forecast = temperature_fields(data['names'])
    salt = np.full_like(t_anchor, SALINITY_PSU)

    do_anchor = data['x'] * data['scale'] + data['center']
    do_truth = data['truth'] * data['scale'] + data['center']
    aou_anchor = solubility(t_anchor, salt) - do_anchor
    aou_truth = solubility(t_observed, salt) - do_truth

    # The AOU arm is the same residual construction on a transformed anchor and target.
    aou_data = dict(data)
    aou_data['x'] = (aou_anchor - data['center']) / data['scale']
    aou_truth_z = (aou_truth - data['center']) / data['scale']
    aou_data['valid'] = data['valid'] & np.isfinite(aou_anchor) & np.isfinite(aou_truth)

    wcache = {id(f): weather_design(data, f) for f in outer + inner}
    pcache = {id(f): plain_design(data, f) for f in outer + inner}
    clim = {id(f): climatology(data, f) for f in outer + inner}

    def objective(trial):
        p = tree_space(trial)
        scores = []
        for f in inner:
            grid = fit_grid(aou_data, f, wcache[id(f)], p, aou_truth_z)
            m = aou_data['valid'][f['va']] & f['eligible'][None, :]
            e = np.abs(grid - aou_truth_z[f['va']])[m] * data['scale']
            scores.append(float(np.nanmean(e)))
        return float(np.mean(scores))

    sel = run_study('oxygen_aou_tree_h6', objective, OUT, quick)
    params = {**sel['params'], 'subsample_freq': 1}
    weather_params = selected('oxygen', 'weather_tree', HORIZON)
    plain_params = selected('oxygen', 'tree', HORIZON)

    rows = []
    for fold in outer:
        aou_grid = fit_grid(aou_data, fold, wcache[id(fold)], params, aou_truth_z)
        do_grid = fit_grid(data, fold, wcache[id(fold)], weather_params, data['truth'])
        pl_grid = fit_grid(data, fold, pcache[id(fold)], plain_params, data['truth'])
        cl, _ = clim[id(fold)]
        mask = data['valid'][fold['va']] & fold['eligible'][None, :]
        for j, i in enumerate(fold['va']):
            for s, name in enumerate(data['names']):
                if not mask[j, s] or not np.isfinite(t_forecast[i, s]) or not np.isfinite(aou_anchor[i, s]):
                    continue
                sol_hat = float(solubility(t_forecast[i, s], SALINITY_PSU))
                rows.append({
                    'fold': fold['number'], 'sensor': name, 'time': str(data['times'][i]),
                    'truth': float(do_truth[i, s]),
                    'aou_tree': sol_hat - float(aou_grid[j, s] * data['scale'] + data['center']),
                    'weather_tree': float(do_grid[j, s] * data['scale'] + data['center']),
                    'plain_tree': float(pl_grid[j, s] * data['scale'] + data['center']),
                    'thermodynamic': sol_hat - float(aou_anchor[i, s]),
                    'persistence': float(do_anchor[i, s]),
                    'climatology': float(cl[i, s] * data['scale'] + data['center'])})

    frame = pd.DataFrame(rows)
    path = OUT / 'aou_forecast_h6.csv.gz'
    frame.to_csv(path, index=False)

    arms = ['aou_tree', 'weather_tree', 'plain_tree', 'thermodynamic', 'persistence', 'climatology']
    contrasts = [('aou_tree', 'weather_tree'), ('aou_tree', 'plain_tree'),
                 ('aou_tree', 'persistence'), ('thermodynamic', 'persistence'),
                 ('thermodynamic', 'weather_tree'), ('thermodynamic', 'climatology')]
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum15.json',
              'horizon_hours': HORIZON, 'salinity_psu': SALINITY_PSU, 'selection': sel,
              'endpoints': int(len(frame)), 'sensors': int(frame.sensor.nunique()),
              'mae': {a: macro(frame, a) for a in arms},
              'contrasts': {f'{a} - {b}': paired(frame, a, b, f'{a}|{b}') for a, b in contrasts},
              'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(OUT / 'aou_forecast_h6_summary.json', report)

    print(f"endpoints={report['endpoints']}  sensors={report['sensors']}")
    for a in arms:
        print(f'   {a:16s} {report["mae"][a]:.4f}')
    for k, (d, lo, hi) in report['contrasts'].items():
        print(f'   {k:34s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))


if __name__ == '__main__':
    main()
