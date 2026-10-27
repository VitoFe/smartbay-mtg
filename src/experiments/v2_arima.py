"""Addendum 66: the ARIMA family, to 72 hours.

Three forecasters of the distribution on addendum 65's rows:
  linear_ar        a seasonal linear autoregression fitted per probe and lead as a direct forecast: the
                   change to the target hour on the probe's own recent changes, optionally its level,
                   daily and annual harmonics; quantiles from its training residuals;
  linear_ar_joint  oxygen only: the same with the sonde's own temperature changes;
  linear_ar_tree   the hybrid: linear_ar plus quantile trees fitted to its training residuals.
Each is selected by its own nested study (protocol v2) on the quantile tree's objective. The
GARCH-type bands on linear_ar's and Chronos-2's medians are computed by the report
(v2_arima_report), from the rows written here and the starting sigma the seed-42 linear_ar job writes.

Declared in config/bay_reserved_protocol_v2_addendum66.json before anything was computed. The 2026
rows are read once, at sampler seed 42; seeds 7 and 123 re-select on Dev-2025 only.

Producers (Trinity) and the report (local):
  python -m src.experiments.v2_arima --variable temperature --arm linear_ar [--seed 7] [--quick]
  python -m src.experiments.v2_arima --report
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.experiments import v2_horizon, v2_horizon_reserved
from src.experiments import v2_probabilistic as P
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_forecast import THREADS, tree_space
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

OUT = P.RUNS / 'v2_arima'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum66.json'
LEADS = list(P.LEADS) + [72]
LAGS = (1, 2, 3, 6, 12, 24, 48, 72)
ARMS = ('linear_ar', 'linear_ar_joint', 'linear_ar_tree')
SCALING = {'temperature': (20., 10.), 'oxygen': (230., 50.)}     # v2_horizon.build


# --------------------------------------------------------------------------------------------------
# The hourly series the lags are read from
# --------------------------------------------------------------------------------------------------
def hourly(data, variable, lead, reserved):
    """Each probe's last reading at most 6 h old, scaled as the builders scale it, on the builders' hourly
    index, and the issuance positions in it. For oxygen also each sonde's own temperature, scaled as
    temperature. G2: the series reproduces the builders' last value at every issuance."""
    if reserved:
        _, frames = v2_horizon_reserved.v1.load_all_observations()
        index = pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), v2_horizon_reserved.RESERVED_END, freq='h')
    else:
        _, frames = v2_horizon.load_observations()
        index = pd.date_range('2024-07-24', '2025-12-31T23:00:00', freq='h', tz='UTC')
    names = data['names']

    def latest(col, scaling):
        c, k = scaling
        raw = pd.DataFrame({n: (frames[n][col].reindex(index) if col in frames[n] else np.nan) for n in names},
                           index=index)
        return ((raw.ffill(limit=6) - c) / k).to_numpy(float)
    own = latest(variable, SCALING[variable])
    q = np.arange(24, len(index) - lead, v2_horizon.STRIDE)
    same_times = bool(len(q) == len(data['times']) and (index[q] == data['times']).all())
    same_x = bool(same_times and np.array_equal(own[q], data['x'], equal_nan=True))
    gate = {'issuances': int(len(q)), 'same_times': same_times, 'same_last_value': same_x,
            'passed': bool(same_times and same_x)}
    if not gate['passed']:
        raise AssertionError(f'G2 failed: {gate}')
    temp = latest('temperature', SCALING['temperature']) if variable == 'oxygen' else None
    return own, temp, q, gate


def lag_changes(series, q, lags=LAGS):
    """[issuances, sensors, lags]: series(q) - series(q - k), NaN where either end is missing or q - k < 0."""
    cur = series[q]
    out = np.full((len(q), series.shape[1], len(lags)), np.nan)
    for n, k in enumerate(lags):
        ok = q - k >= 0
        out[ok, :, n] = cur[ok] - series[q[ok] - k]
    return out


def base_features(data, variable, lead, reserved):
    own, temp, q, gate = hourly(data, variable, lead, reserved)
    t = data['times']
    feats = {'chg': lag_changes(own, q), 'level': own[q], 'hour': t.hour.to_numpy(), 'doy': t.dayofyear.to_numpy()}
    if temp is not None:
        feats['t_chg'] = lag_changes(temp, q)
        feats['t_level'] = temp[q]
    return feats, gate


def linear_space(trial):
    return {'log_alpha': trial.suggest_float('log_alpha', -3.0, 3.0),
            'max_lag': trial.suggest_categorical('max_lag', [6, 12, 24, 48, 72]),
            'daily': trial.suggest_int('daily', 0, 3),
            'annual': trial.suggest_int('annual', 0, 2),
            'level': trial.suggest_categorical('level', [False, True]),
            'by_hour': trial.suggest_categorical('by_hour', [False, True])}


def design(feats, cfg, joint):
    """[issuances, sensors, columns]: the changes (zero where missing) and their indicators up to the maximum
    lag, the level, the harmonics; with `joint` the sonde's own temperature the same way."""
    keep = [n for n, k in enumerate(LAGS) if k <= cfg['max_lag']]
    blocks = []
    sources = [('chg', 'level')] + ([('t_chg', 't_level')] if joint else [])
    for chg, level in sources:
        c = feats[chg][:, :, keep]
        blocks += [np.nan_to_num(c), np.isnan(c).astype(float)]
        if cfg['level']:
            lv = feats[level]
            blocks += [np.nan_to_num(lv)[..., None], np.isnan(lv).astype(float)[..., None]]
    n, s = feats['level'].shape
    cal = []
    for m in range(1, cfg['daily'] + 1):
        w = 2 * np.pi * m * feats['hour'] / 24
        cal += [np.sin(w), np.cos(w)]
    for m in range(1, cfg['annual'] + 1):
        w = 2 * np.pi * m * feats['doy'] / 365.25
        cal += [np.sin(w), np.cos(w)]
    if cal:
        blocks.append(np.broadcast_to(np.column_stack(cal)[:, None, :], (n, s, len(cal))))
    return np.concatenate(blocks, axis=-1)


def fit_linear(data, X, tr, eligible, cfg):
    """Per probe: ridge on standardised columns, unpenalised intercept, residual quantiles (per probe, or per
    probe and issuance hour where that cell holds at least P.MIN_CELL rows)."""
    mask = data['valid'] & eligible[None, :]
    alpha = 10.0 ** cfg['log_alpha']
    hours = data['times'].hour.to_numpy()
    models = {}
    for s in np.flatnonzero(eligible):
        rows = tr[mask[tr, s]]
        if len(rows) < 2:
            # only under --quick, whose cut training rows can leave a scored probe without any; the folds
            # require 100 training rows for a probe to be scored. The last value, recorded as such.
            models[int(s)] = {'mu': np.zeros(X.shape[-1]), 'sd': np.ones(X.shape[-1]), 'beta': np.zeros(X.shape[-1]),
                              'mean': 0.0, 'own': np.zeros(len(P.LEVELS)), 'table': {}, 'rows': int(len(rows))}
            continue
        A, y = X[rows, s], data['truth'][rows, s] - data['x'][rows, s]
        mu, sd = A.mean(0), A.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (A - mu) / sd
        ym = y.mean()
        n = len(rows)
        beta = np.linalg.solve(Z.T @ Z / n + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym) / n)
        fitted = Z @ beta + ym
        resid = y - fitted
        own = np.quantile(resid, P.LEVELS)
        table = {}
        if cfg['by_hour']:
            hr = hours[rows]
            for h in np.unique(hr):
                cell = resid[hr == h]
                if len(cell) >= P.MIN_CELL:
                    table[int(h)] = np.quantile(cell, P.LEVELS)
        models[int(s)] = {'mu': mu, 'sd': sd, 'beta': beta, 'mean': ym, 'own': own, 'table': table, 'rows': n}
    return models


def predict_change(models, X, rows, n_sensors):
    """[len(rows), sensors]: each probe's predicted change, NaN where it has no model."""
    out = np.full((len(rows), n_sensors), np.nan)
    for s, m in models.items():
        out[:, s] = ((X[rows, s] - m['mu']) / m['sd']) @ m['beta'] + m['mean']
    return out


def linear_quantiles(data, models, change, va, eligible):
    """The scored rows' quantiles (scaled units), in P.scored's order."""
    mask, i, s = P.scored(data, va, eligible)
    j, _ = np.nonzero(mask)
    hours = data['times'].hour.to_numpy()[i]
    Q = np.empty((len(i), len(P.LEVELS)))
    for n, (jj, ss, h) in enumerate(zip(j, s, hours)):
        m = models[int(ss)]
        Q[n] = data['x'][i[n], ss] + change[jj, ss] + m['table'].get(int(h), m['own'])
    return Q


def hybrid_fit_predict(data, tree_X, tr, va, eligible, change_tr, change_va, params):
    """Quantile trees on linear_ar's in-sample residuals, reading the plain tree's features and the linear
    forecast change; one model per level, shared configuration, sorted; scaled units, P.scored's order."""
    import lightgbm as lgb
    mask = data['valid'] & eligible[None, :]
    tm, vm = mask[tr], mask[va]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]
    w = w / w.mean()
    Xtr = np.concatenate([tree_X[tr], change_tr[..., None]], axis=-1)[tm]
    ytr = (data['truth'][tr] - data['x'][tr] - change_tr)[tm]
    Xva = np.concatenate([tree_X[va], change_va[..., None]], axis=-1)[vm]
    Q = np.column_stack([lgb.LGBMRegressor(**params, objective='quantile', alpha=a, verbosity=-1, n_jobs=THREADS,
                                           random_state=42, deterministic=True, force_col_wise=True)
                         .fit(Xtr, ytr, sample_weight=w).predict(Xva) for a in P.LEVELS])
    Q.sort(axis=1)
    return Q + (data['x'][va] + change_va)[vm][:, None]


# --------------------------------------------------------------------------------------------------
# One arm at one lead
# --------------------------------------------------------------------------------------------------
def ratio(data, f, Q, clim):
    mask, i, s = P.scored(data, f['va'], f['eligible'])
    y = data['truth'][i, s]
    C = P.clim_predict(data, clim, f['va'], mask)
    return P.sensor_macro(P.quantile_score(y, Q), s) / P.sensor_macro(P.quantile_score(y, C), s)


def linear_study(data, feats, inner, clim, name, folder, quick, joint):
    def objective(trial):
        cfg = linear_space(trial)
        X = design(feats, cfg, joint)
        out = []
        for f in inner:
            models = fit_linear(data, X, f['tr'], f['eligible'], cfg)
            change = predict_change(models, X, f['va'], data['x'].shape[1])
            out.append(ratio(data, f, linear_quantiles(data, models, change, f['va'], f['eligible']), clim[id(f)]))
        return float(np.mean(out))
    return run_study(name, objective, folder, quick)


def lead_run(variable, arm, lead, seed, quick, folder, quantiles, studies, gates, sigma0):
    joint = arm == 'linear_ar_joint'
    data = v2_horizon.build(variable, lead)
    outer = v2_horizon.folds(data, lead)
    if quick:
        for f in outer:
            f['tr'], f['va'] = f['tr'][-300:], f['va'][:40]
    inner = [v2_horizon.inner_fold(data, f, lead) for f in outer]
    if quick:
        for f in inner:
            f['tr'], f['va'] = f['tr'][-300:], f['va'][:40]
    clim = {id(f): P.clim_table(data, f['tr'], P.LEVELS) for f in outer + inner}
    feats, gates[f'G2|Dev|{lead}'] = base_features(data, variable, lead, reserved=False)
    S = data['x'].shape[1]

    # the linear model: its own study (the hybrid repeats linear_ar's, same seed, in its own folder)
    lin_name = 'linear_ar_joint' if joint else 'linear_ar'
    lin_folder = folder / ('studies' if arm != 'linear_ar_tree' else 'studies_linear')
    sel = linear_study(data, feats, inner, clim, f'{variable}_{lin_name}_h{lead}', lin_folder, quick, joint)
    cfg = sel['params']
    studies[f'{lin_name}|{lead}'] = {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}
    X = design(feats, cfg, joint)

    if arm == 'linear_ar_tree':
        trees = {id(f): P.tree_design(data, f['fallback'], lead, False) for f in outer + inner}
        fitted = {}
        for f in outer + inner:
            models = fit_linear(data, X, f['tr'], f['eligible'], cfg)
            fitted[id(f)] = (predict_change(models, X, f['tr'], S), predict_change(models, X, f['va'], S))

        def objective(trial):
            p = tree_space(trial)
            out = []
            for f in inner:
                ctr, cva = fitted[id(f)]
                Q = hybrid_fit_predict(data, trees[id(f)], f['tr'], f['va'], f['eligible'], ctr, cva, p)
                out.append(ratio(data, f, Q, clim[id(f)]))
            return float(np.mean(out))
        hsel = run_study(f'{variable}_linear_ar_tree_h{lead}', objective, folder / 'studies', quick)
        params = {**hsel['params'], 'subsample_freq': 1}
        studies[f'linear_ar_tree|{lead}'] = {k: hsel[k] for k in ('completed_trials', 'best_value', 'params')}

    def forecast(d, X_, tr, va, eligible, fallback, label):
        models = fit_linear(d, X_, tr, eligible, cfg)
        gates.setdefault('min_fit_rows', {})[f'{label}|{lead}'] = min(m['rows'] for m in models.values())
        if arm != 'linear_ar_tree':
            change = predict_change(models, X_, va, d['x'].shape[1])
            return linear_quantiles(d, models, change, va, eligible)
        S_ = d['x'].shape[1]
        return hybrid_fit_predict(d, P.tree_design(d, fallback, lead, False), tr, va, eligible,
                                  predict_change(models, X_, tr, S_), predict_change(models, X_, va, S_), params)

    for f in outer:
        frame, _ = P.frame_of(data, f['va'], f['eligible'], 'Dev-2025', f['number'], lead)
        quantiles.append(P.with_quantiles(frame, arm, forecast(data, X, f['tr'], f['va'], f['eligible'],
                                                               f['fallback'], f'Dev-2025 fold {f["number"]}'), data))
    if sigma0 is not None:
        # the bands' starting sigma: the last value's RMS error on the training rows of the first split in
        # which the probe is scored (fold 0, then fold 1), physical units
        k = data['scale']
        block = {}
        for f in outer:
            m = data['valid'] & f['eligible'][None, :]
            for s in np.flatnonzero(f['eligible']):
                name = data['names'][s]
                rows = f['tr'][m[f['tr'], s]]
                if name not in block and len(rows):
                    e = (data['truth'][rows, s] - data['x'][rows, s]) * k
                    block[name] = {'sigma0': float(np.sqrt(np.mean(e ** 2))), 'from': f'Dev-2025 fold {f["number"]}'}
        sigma0[str(lead)] = block
    if seed == 42:
        data26 = v2_horizon_reserved.build(variable, lead)
        bundle = v2_horizon_reserved.split(data26, lead)
        if quick:
            bundle['tr'], bundle['va'] = bundle['tr'][-300:], bundle['va'][:40]
        feats26, gates[f'G2|2026|{lead}'] = base_features(data26, variable, lead, reserved=True)
        X26 = design(feats26, cfg, joint)
        frame, _ = P.frame_of(data26, bundle['va'], bundle['eligible'], 'Reserved-2026', None, lead)
        quantiles.append(P.with_quantiles(frame, arm, forecast(data26, X26, bundle['tr'], bundle['va'],
                                                               bundle['eligible'], bundle['fallback'],
                                                               'Reserved-2026'), data26))
        if sigma0 is not None:
            k = data26['scale']
            m = data26['valid'] & bundle['eligible'][None, :]
            for s in np.flatnonzero(bundle['eligible']):
                name = data26['names'][s]
                rows = bundle['tr'][m[bundle['tr'], s]]
                if name not in sigma0[str(lead)] and len(rows):
                    e = (data26['truth'][rows, s] - data26['x'][rows, s]) * k
                    sigma0[str(lead)][name] = {'sigma0': float(np.sqrt(np.mean(e ** 2))), 'from': 'pre-2026 training'}


def run(variable, arm, seed=42, quick=False):
    if arm not in ARMS or (arm == 'linear_ar_joint' and variable != 'oxygen'):
        raise ValueError(f'{variable} {arm}')
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    leads = [6, 72] if quick else LEADS
    quantiles, studies, gates = [], {}, {}
    sigma0 = {} if (seed == 42 and arm == 'linear_ar') else None
    for lead in leads:
        t0 = time.time()
        lead_run(variable, arm, lead, seed, quick, folder, quantiles, studies, gates, sigma0)
        print(variable, arm, 'seed', seed, 'lead', lead, f'{time.time() - t0:.0f} s', flush=True)
    q = pd.concat(quantiles, ignore_index=True)
    num = ['truth', 'last'] + P.QCOLS
    q[num] = q[num].round(P.DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_{arm}.csv.gz', index=False)
    if sigma0 is not None:
        save(folder / f'ewma_init_{variable}.json', {'declared_in': DECLARATION, 'variable': variable,
                                                     'units': 'physical (degC, umol/l)', 'sigma0': sigma0})
    save(folder / f'arima_{variable}_{arm}.json', {'declared_in': DECLARATION, 'variable': variable, 'arm': arm,
                                                   'seed': seed, 'smoke_only': quick, 'leads': leads,
                                                   'studies': studies, 'gates': gates, 'threads': THREADS})
    print(variable, arm, 'seed', seed, 'G2', all(g['passed'] for k, g in gates.items() if k.startswith('G2')),
          'fewest fit rows', min(gates['min_fit_rows'].values()), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=list(P.VARIABLES))
    p.add_argument('--arm', choices=list(ARMS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    if a.report:
        from src.experiments.v2_arima_report import report
        report()
    else:
        run(a.variable, a.arm, a.seed, a.quick)
