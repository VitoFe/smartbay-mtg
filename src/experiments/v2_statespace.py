"""Addendum 67: state-space ARIMA and N-BEATS as forecasters of the distribution, to 72 hours.

  arima_ss    ARIMA(p, d, q) with Fourier regressors, fitted per probe by maximum likelihood on the hourly
              record before the cut and filtered with fixed parameters through every later hour; the
              h-step mean and variance from the filtered state at each issuance, with bands from the
              standardised h-step errors of the 120 days before the cut. Orders and harmonics by a nested
              study (protocol v2) over the development inner folds.
  auto_arima  the same, the order chosen per probe and split by pmdarima's auto_arima (AIC).
  nbeats      generic N-BEATS across probes, eleven quantiles of the next 72 hourly changes, its own study.

Declared in config/bay_reserved_protocol_v2_addendum67.json before anything was computed on this
protocol; an exploratory sweep on the development folds preceded it and is listed there. The 2026 rows
are read once, at sampler seed 42; seeds 7 and 123 re-select on Dev-2025 only.

Producers (Trinity) and the report (local):
  python -m src.experiments.v2_statespace --arm arima_ss --variable temperature [--seed 7] [--procs 7]
  python -m src.experiments.v2_statespace --arm auto_arima --variable oxygen --procs 5
  python -m src.experiments.v2_statespace --arm nbeats --variable temperature [--seed 7]
  python -m src.experiments.v2_statespace --report
"""
import argparse
import json
import os
import warnings
from multiprocessing import Pool

# one BLAS thread per process: the ARIMA jobs fan out over probes, and the campaign launcher exports the
# job's thread count, which every forked worker would otherwise inherit
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments import v2_horizon, v2_horizon_reserved
from src.experiments import v2_probabilistic as P
from src.experiments.bay_twin_sweep import save
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

OUT = P.RUNS / 'v2_statespace'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum67.json'
LEADS = list(P.LEADS) + [72]
HMAX = 72
ARMS = ('arima_ss', 'auto_arima', 'nbeats')
SCALING = {'temperature': (20., 10.), 'oxygen': (230., 50.)}          # v2_horizon.build
ERR_DAYS, MIN_ERR, MIN_READINGS = 120, 50, 200
DEV_INDEX = pd.date_range('2024-07-24', '2025-12-31T23:00:00', freq='h', tz='UTC')
RES_INDEX = pd.date_range(pd.Timestamp('2024-07-24', tz='UTC'), v2_horizon_reserved.RESERVED_END, freq='h')
FOLDS = (('2025-07-01', '2025-10-01'), ('2025-10-01', '2026-01-01'))
GRID_HOURS = (0, 6, 12, 18)
warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------------------------------
# Splits and series
# --------------------------------------------------------------------------------------------------
def ts(x):
    return pd.Timestamp(x, tz='UTC')


def splits():
    """The two development folds and 2026: the hourly index, the cut (training readings end before it),
    the window whose issuances are forecast, and the inner window the study scores."""
    out = []
    for number, (a, b) in enumerate(FOLDS):
        begin, end = ts(a), ts(b)
        inner_end = begin - pd.Timedelta(hours=72)
        inner_start = inner_end - pd.Timedelta(days=45)
        out.append({'name': f'Dev-2025 fold {number}', 'cohort': 'Dev-2025', 'fold': number, 'reserved': False,
                    'index': DEV_INDEX, 'cut': begin - pd.Timedelta(hours=72), 'window': (begin, end),
                    'inner': {'cut': inner_start - pd.Timedelta(hours=72), 'window': (inner_start, inner_end)}})
    begin = v2_horizon_reserved.RESERVED_START
    out.append({'name': 'Reserved-2026', 'cohort': 'Reserved-2026', 'fold': None, 'reserved': True,
                'index': RES_INDEX, 'cut': begin - pd.Timedelta(hours=72),
                'window': (begin, v2_horizon_reserved.RESERVED_END + pd.Timedelta(hours=1)), 'inner': None})
    return out


def series(variable, reserved):
    """{probe: hourly readings scaled as the builders scale them} on the split's index (unfilled)."""
    if reserved:
        _, frames = v2_horizon_reserved.v1.load_all_observations()
        index = RES_INDEX
    else:
        _, frames = v2_horizon.load_observations()
        index = DEV_INDEX
    c, k = SCALING[variable]
    return {n: (f[variable].reindex(index).to_numpy(float) - c) / k
            for n, f in frames.items() if variable in f and f[variable].notna().any()}


def position(index, t):
    return int(index.get_indexer([t])[0])


def grid(index, lo, hi, end=None):
    """Positions of the issuance hours in [lo, hi), with a full 72 h horizon before `end` if given."""
    pos = np.flatnonzero(np.isin(index.hour, GRID_HOURS) & (index >= lo) & (index < hi))
    return pos if end is None else pos[pos + HMAX < end]


def fourier(index, daily, annual, constant):
    hours, doy = index.hour.to_numpy(), index.dayofyear.to_numpy()
    cols = [np.ones(len(index))] if constant else []
    for m in range(1, daily + 1):
        cols += [np.sin(2 * np.pi * m * hours / 24), np.cos(2 * np.pi * m * hours / 24)]
    for m in range(1, annual + 1):
        cols += [np.sin(2 * np.pi * m * doy / 365.25), np.cos(2 * np.pi * m * doy / 365.25)]
    return np.column_stack(cols) if cols else None


# --------------------------------------------------------------------------------------------------
# State-space forecasts
# --------------------------------------------------------------------------------------------------
def fit_filter(y, cut, order, X, maxiter=200):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    mod = SARIMAX(y[:cut], exog=None if X is None else X[:cut], order=order)
    res = mod.fit(disp=False, maxiter=maxiter, method='lbfgs')
    if not np.all(np.isfinite(res.params)):
        raise ValueError('non-finite parameters')
    return res.apply(y, exog=X)


def multistep(full, idx, nobs):
    """Mean and variance of the reading h = 1..HMAX hours after each position in idx, from the filtered
    state at that position (readings up to and including it) and the fitted parameters."""
    ssm = full.model.ssm

    def fixed(name):
        m = np.asarray(ssm[name])
        if m.ndim == 3:
            if m.shape[2] != 1:
                raise ValueError(f'{name} is time-varying')
            m = m[:, :, 0]
        return m
    T, R, Q = fixed('transition'), fixed('selection'), fixed('state_cov')
    Z = np.asarray(fixed('design')).reshape(-1)
    H = float(np.asarray(ssm['obs_cov']).reshape(-1)[0])
    c = np.asarray(ssm['state_intercept'])
    c = c[:, 0] if c.ndim == 2 else c
    d = np.asarray(ssm['obs_intercept']).reshape(-1)
    d = np.broadcast_to(d[:1], (nobs,)) if len(d) != nobs else d
    fr = full.filter_results
    A = fr.filtered_state[:, idx]
    Pm = np.transpose(fr.filtered_state_cov[:, :, idx], (2, 0, 1))
    RQR = R @ Q @ R.T
    mean = np.full((len(idx), HMAX), np.nan)
    var = np.full((len(idx), HMAX), np.nan)
    for h in range(1, HMAX + 1):
        A = T @ A + c[:, None]
        Pm = T @ Pm @ T.T + RQR
        tgt = idx + h
        ok = tgt < nobs
        mean[ok, h - 1] = (Z @ A)[ok] + d[tgt[ok]]
        var[:, h - 1] = np.einsum('i,nij,j->n', Z, Pm, Z) + H
    return mean, np.maximum(var, 1e-12)


def multipliers(y, mean, var, positions):
    """Empirical quantiles of the standardised h-step errors at `positions` (rows of mean and var), per lead;
    normal quantiles where fewer than MIN_ERR."""
    out = {}
    for h in LEADS:
        u = (y[np.minimum(positions + h, len(y) - 1)] - mean[:, h - 1]) / np.sqrt(var[:, h - 1])
        u = u[np.isfinite(u) & (positions + h < len(y))]
        out[h] = np.quantile(u, P.LEVELS) if len(u) >= MIN_ERR else stats.norm.ppf(P.LEVELS)
    return out


def auto_order(y, cut, index):
    """pmdarima's auto_arima on the longest gap-free stretch of the 60 days before the cut (gaps of at most
    6 h interpolated, for the selection only), with two daily harmonics."""
    import pmdarima
    lo = max(0, cut - 60 * 24)
    s = pd.Series(y[lo:cut]).interpolate(limit=6, limit_area='inside').to_numpy()
    ok = np.r_[np.isfinite(s), False]
    best, start = (0, 0), None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    a, b = best
    if b - a < 24 * 7:
        raise ValueError('no stretch of a week')
    X = fourier(index[lo + a:lo + b], 2, 0, False)
    model = pmdarima.auto_arima(s[a:b], X=X, seasonal=False, stepwise=True, max_p=5, max_q=5, max_d=1,
                                information_criterion='aic', suppress_warnings=True, error_action='ignore')
    return tuple(int(v) for v in model.order)


def arima_probe(args):
    """One probe in one split: fit, filter, forecast at `issue` and band from `err`. cfg is a configuration
    or 'auto'. Falls back to ARIMA(0,1,0) with the same regressors, recorded."""
    y, index, cut, nobs, issue, err, cfg, check = args
    y, index = y[:nobs], index[:nobs]
    info = {'readings_before_cut': int(np.isfinite(y[:cut]).sum())}
    if cfg == 'auto':
        try:
            order = auto_order(y, cut, index)
        except Exception as e:                                   # recorded
            order, info['auto_error'] = (0, 1, 0), repr(e)[:160]
        daily, annual = 2, 1
    else:
        order, daily, annual = (cfg['p'], cfg['d'], cfg['q']), cfg['daily'], cfg['annual']
    info['order'] = order
    X = fourier(index, daily, annual, order[1] == 0)
    full = None
    if info['readings_before_cut'] >= MIN_READINGS:
        try:
            full = fit_filter(y, cut, order, X)
        except Exception as e:
            info['fit_error'] = repr(e)[:160]
    if full is None:
        info['fallback'] = True
        info['order'] = (0, 1, 0)
        Xf = fourier(index, daily, annual, False)
        full = fit_filter(y, cut, (0, 1, 0), Xf)
    idx = np.r_[err, issue]
    mean, var = multistep(full, idx, nobs)
    mult = multipliers(y, mean[:len(err)], var[:len(err)], err)
    if check:
        # G2: against statsmodels' own forecast from the same state, at two issuances
        diffs = []
        for t in issue[[0, len(issue) // 2]] if len(issue) else []:
            ex = full.model.exog
            r = full.model.clone(y[:t + 1], exog=None if ex is None else ex[:t + 1]).filter(full.params)
            h = min(HMAX, nobs - t - 1)
            if h < 1:
                continue
            f = r.get_forecast(h, exog=None if ex is None else ex[t + 1:t + 1 + h])
            k = len(err) + int(np.flatnonzero(issue == t)[0])
            diffs.append(max(np.abs(f.predicted_mean - mean[k, :h]).max(), np.abs(f.var_pred_mean - var[k, :h]).max()))
        info['g2_max_difference'] = float(max(diffs)) if diffs else None
    return mean[len(err):], var[len(err):], mult, info


def quantiles_at(mean, var, mult, rows, lead):
    """Scaled-unit quantiles for `rows` (row numbers into mean/var) at one lead."""
    m, s = mean[rows, lead - 1], np.sqrt(var[rows, lead - 1])
    Q = m[:, None] + s[:, None] * mult[lead][None, :]
    Q.sort(axis=1)
    return Q


# --------------------------------------------------------------------------------------------------
# The rows the study scores: the builders' inner-fold rows, with the climatological quantiles
# --------------------------------------------------------------------------------------------------
def inner_rows(variable):
    """{fold: {lead: (sensor names, issuance times, truth, clim quantiles)}} in scaled units."""
    out = {0: {}, 1: {}}
    for lead in LEADS:
        data = v2_horizon.build(variable, lead)
        for f in v2_horizon.folds(data, lead):
            g = v2_horizon.inner_fold(data, f, lead)
            mask, i, s = P.scored(data, g['va'], g['eligible'])
            C = P.clim_predict(data, P.clim_table(data, g['tr'], P.LEVELS), g['va'], mask)
            out[f['number']][lead] = (np.asarray(data['names'])[s], data['times'][i], data['truth'][i, s], C)
    return out


def objective_value(rows, forecasts):
    """Mean over folds and leads of the sensor-macro quantile score divided by the climatology's.
    forecasts: {fold: {probe: (mean, var, mult, positions)}}."""
    ratios = []
    for fold, by_lead in rows.items():
        for lead, (names, times, y, C) in by_lead.items():
            if not len(y):
                continue
            pos = DEV_INDEX.get_indexer(times)
            Q = np.empty_like(C)
            for n in np.unique(names):
                mean, var, mult, issue = forecasts[fold][n]
                at = np.searchsorted(issue, pos[names == n])
                Q[names == n] = quantiles_at(mean, var, mult, at, lead)
            if not np.isfinite(Q).all():
                raise ValueError(f'non-finite forecast, fold {fold}, lead {lead}')
            ratios.append(P.sensor_macro(P.quantile_score(y, Q), names) / P.sensor_macro(P.quantile_score(y, C), names))
    return float(np.mean(ratios))


def arima_space(trial):
    return {'p': trial.suggest_int('p', 0, 5), 'd': trial.suggest_int('d', 0, 1), 'q': trial.suggest_int('q', 0, 3),
            'daily': trial.suggest_int('daily', 0, 3), 'annual': trial.suggest_int('annual', 0, 1)}


def inner_jobs(data, rows, split, cfg):
    """One worker task per probe that the inner rows of this fold score."""
    idx = split['index']
    lo, hi = split['inner']['window']
    cut = position(idx, split['inner']['cut'])
    nobs = position(idx, hi)
    issue = grid(idx, lo, hi)
    err = grid(idx, idx[max(0, cut - ERR_DAYS * 24)], idx[cut], end=cut)
    names = sorted({n for _, (ns, _, _, _) in rows[split['fold']].items() for n in ns})
    return names, [(data[n], idx, cut, nobs, issue, err, cfg, False) for n in names], issue


def outer_jobs(data, split, cfg, check):
    idx = split['index']
    cut = position(idx, split['cut'])
    lo, hi = split['window']
    issue = grid(idx, lo, hi)
    err = grid(idx, idx[max(0, cut - ERR_DAYS * 24)], idx[cut], end=cut)
    names = [n for n, y in data.items() if np.isfinite(y[issue]).any()]
    return names, [(data[n], idx, cut, len(idx), issue, err, cfg, check) for n in names], issue


def frames_from(variable, arm, split, names, results, issue, data):
    c, k = SCALING[variable]
    blocks, infos = [], {}
    for n, (mean, var, mult, info) in zip(names, results):
        infos[n] = info
        keep = np.flatnonzero(np.isfinite(data[n][issue]))
        for lead in LEADS:
            Q = quantiles_at(mean, var, mult, keep, lead) * k + c
            f = pd.DataFrame(Q, columns=P.QCOLS)
            f.insert(0, 'lead', lead)
            f.insert(0, 'time', split['index'][issue[keep]])
            f.insert(0, 'sensor', n)
            f.insert(0, 'fold', split['fold'])
            f.insert(0, 'cohort', split['cohort'])
            f.insert(0, 'arm', arm)
            blocks.append(f)
    return blocks, infos


def run_arima(variable, arm, seed, procs, quick):
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    dev = series(variable, reserved=False)
    sp = splits()
    if quick:
        names = list(dev)[:3]
        dev = {n: dev[n] for n in names}
    meta = {'declared_in': DECLARATION, 'variable': variable, 'arm': arm, 'seed': seed, 'smoke_only': quick}
    with Pool(procs) as pool:
        if arm == 'arima_ss':
            rows = inner_rows(variable)
            if quick:
                rows = {f: {L: tuple(a[np.isin(r[0], list(dev))] for a in r) for L, r in by.items() if L in (6, 72)}
                        for f, by in rows.items()}

            def objective(trial):
                cfg = arima_space(trial)
                forecasts = {}
                for split in sp[:2]:
                    names, jobs, issue = inner_jobs(dev, rows, split, cfg)
                    got = pool.map(arima_probe, jobs)
                    forecasts[split['fold']] = {n: (m, v, mu, issue) for n, (m, v, mu, _) in zip(names, got)}
                return objective_value(rows, forecasts)
            sel = run_study(f'{variable}_arima_ss', objective, folder / 'studies', quick)
            cfg = sel['params']
            meta['study'] = {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}
        else:
            cfg = 'auto'
        blocks, infos = [], {}
        for split in sp:
            if split['reserved'] and (seed != 42):
                continue
            data = series(variable, reserved=True) if split['reserved'] else dev
            if quick:
                data = {n: data[n] for n in list(dev) if n in data}
            names, jobs, issue = outer_jobs(data, split, cfg, check=True)
            got = pool.map(arima_probe, jobs)
            b, i = frames_from(variable, arm, split, names, got, issue, data)
            blocks += b
            infos[split['name']] = i
            print(variable, arm, 'seed', seed, split['name'], len(names), 'probes', flush=True)
    q = pd.concat(blocks, ignore_index=True)
    q[P.QCOLS] = q[P.QCOLS].round(P.DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_{arm}.csv.gz', index=False)
    g2 = [x['g2_max_difference'] for s in infos.values() for x in s.values() if x.get('g2_max_difference') is not None]
    meta.update({'probes': infos, 'gates': {'G2': {'checked': len(g2), 'max_difference': max(g2) if g2 else None,
                                                   'passed': bool(g2) and max(g2) <= 1e-8}}})
    save(folder / f'statespace_{variable}_{arm}.json', meta)
    print(variable, arm, 'seed', seed, 'G2', meta['gates']['G2'], flush=True)


# --------------------------------------------------------------------------------------------------
# N-BEATS
# --------------------------------------------------------------------------------------------------
def nbeats_space(trial):
    return {'width': trial.suggest_categorical('width', [256, 512, 1024]), 'blocks': trial.suggest_int('blocks', 2, 8),
            'L': trial.suggest_categorical('L', [72, 168, 336]), 'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'steps': trial.suggest_categorical('steps', [2000, 4000, 8000]),
            'calendar': trial.suggest_categorical('calendar', [False, True])}


def windows(data, positions, L, calendar, index):
    """Inputs (changes from the last reading over the last L hours, missingness, calendar), targets (the next
    HMAX changes) and their masks. G3: every window ends at its issuance."""
    xs, ys, ms, keys = [], [], [], []
    for name, y in data.items():
        pos = positions.get(name, np.array([], int))
        if not len(pos):
            continue
        filled = pd.Series(y).ffill().to_numpy()
        win = np.stack([filled[p - L + 1:p + 1] for p in pos])
        assert win.shape[1] == L
        obs = np.stack([np.isfinite(y[p - L + 1:p + 1]) for p in pos]).astype(np.float32)
        last = y[pos][:, None]
        win = np.where(np.isfinite(win), win, last) - last
        feats = [win, obs]
        if calendar:
            t = index[pos]
            feats.append(np.column_stack([np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24),
                                          np.sin(2 * np.pi * t.dayofyear / 365.25), np.cos(2 * np.pi * t.dayofyear / 365.25)]))
        xs.append(np.hstack(feats).astype(np.float32))
        tgt = pos[:, None] + np.arange(1, HMAX + 1)[None, :]
        fut = np.where(tgt < len(y), y[np.minimum(tgt, len(y) - 1)], np.nan) - last
        ms.append(np.isfinite(fut).astype(np.float32))
        ys.append(np.nan_to_num(fut).astype(np.float32))
        keys += [(name, int(p)) for p in pos]
    return np.vstack(xs), np.vstack(ys), np.vstack(ms), keys


def training_positions(data, cut, L):
    out = {}
    for name, y in data.items():
        p = np.arange(max(L, 336), cut - HMAX)          # the same windows whatever L
        if not len(p):
            continue
        frac = pd.Series(np.isfinite(y).astype(float)).rolling(L).mean().to_numpy()[p]
        fin = np.isfinite(y).astype(int)
        cs = np.r_[0, np.cumsum(fin)]
        has_target = cs[np.minimum(p + HMAX + 1, len(y))] - cs[p + 1] > 0
        out[name] = p[np.isfinite(y[p]) & (frac >= 0.5) & has_target]
    return out


def nbeats_fit_predict(data, index, cut, issue_by_probe, cfg, device, seed=0):
    import torch
    torch.manual_seed(seed)
    L, calendar, width, blocks = cfg['L'], cfg['calendar'], cfg['width'], cfg['blocks']
    X, Y, M, keys = windows(data, training_positions(data, cut, L), L, calendar, index)
    order = np.argsort([p for _, p in keys], kind='stable')
    n_val = max(1, len(X) // 7)
    val, tr = order[-n_val:], order[:-n_val]
    nq, D = len(P.LEVELS), X.shape[1]

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Sequential(torch.nn.Linear(D, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                                          torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU(),
                                          torch.nn.Linear(width, width), torch.nn.ReLU())
            self.back, self.fore = torch.nn.Linear(width, D), torch.nn.Linear(width, HMAX * nq)

        def forward(self, x):
            h = self.fc(x)
            return self.back(h), self.fore(h)

    class NBeats(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(blocks)])

        def forward(self, x):
            out = 0
            for b in self.blocks:
                back, fore = b(x)
                x, out = x - back, out + fore
            return out.view(-1, HMAX, nq)

    model = NBeats().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['steps'])
    lv = torch.tensor(P.LEVELS, dtype=torch.float32, device=device)
    Xt, Yt, Mt = (torch.tensor(a, device=device) for a in (X, Y, M))

    def loss_of(i):
        q = model(Xt[i])
        diff = Yt[i][:, :, None] - q
        pin = torch.maximum(lv * diff, (lv - 1) * diff).mean(-1)
        return (pin * Mt[i]).sum() / Mt[i].sum().clamp_min(1)

    rng = np.random.default_rng(seed)
    best, best_state = np.inf, None
    val_t = torch.tensor(val, device=device)
    for step in range(cfg['steps']):
        loss = loss_of(torch.tensor(rng.choice(tr, min(1024, len(tr))), device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 199 or step == cfg['steps'] - 1:
            with torch.no_grad():
                v = float(loss_of(val_t))
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    Xe, _, _, ekeys = windows(data, issue_by_probe, L, calendar, index)
    with torch.no_grad():
        Q = torch.cat([model(torch.tensor(Xe[i:i + 4096], device=device)) for i in range(0, len(Xe), 4096)]).cpu().numpy()
    Q.sort(axis=-1)
    last = np.array([data[n][p] for n, p in ekeys])
    return Q + last[:, None, None], ekeys, {'windows': int(len(tr)), 'best_val_pinball': best}


def run_nbeats(variable, seed, quick):
    import torch
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    dev = series(variable, reserved=False)
    sp = splits()
    rows = inner_rows(variable)

    def objective(trial):
        cfg = nbeats_space(trial)
        if quick:
            cfg['steps'] = 50
        forecasts = {}
        for split in sp[:2]:
            idx = split['index']
            lo, hi = split['inner']['window']
            cut, nobs = position(idx, split['inner']['cut']), position(idx, hi)
            issue = grid(idx, lo, hi)
            names = sorted({n for _, (ns, _, _, _) in rows[split['fold']].items() for n in ns})
            data = {n: dev[n][:nobs] for n in names}
            by_probe = {n: issue[np.isfinite(data[n][issue])] for n in names}
            Q, keys, _ = nbeats_fit_predict(data, idx[:nobs], cut, by_probe, cfg, device)
            forecasts[split['fold']] = {}
            for n in names:
                sel = np.array([j for j, (m, _) in enumerate(keys) if m == n], int)
                pos = np.array([p for m, p in keys if m == n], int)
                forecasts[split['fold']][n] = (Q[sel], pos)
        ratios = []
        for fold, by_lead in rows.items():
            for lead, (names, times, y, C) in by_lead.items():
                if not len(y):
                    continue
                pos = DEV_INDEX.get_indexer(times)
                Qr = np.empty_like(C)
                for n in np.unique(names):
                    Qn, pn = forecasts[fold][n]
                    Qr[names == n] = Qn[np.searchsorted(pn, pos[names == n]), lead - 1]
                ratios.append(P.sensor_macro(P.quantile_score(y, Qr), names) / P.sensor_macro(P.quantile_score(y, C), names))
        return float(np.mean(ratios))
    sel = run_study(f'{variable}_nbeats', objective, folder / 'studies', quick)
    cfg = sel['params']
    if quick:
        cfg['steps'] = 50
    meta = {'declared_in': DECLARATION, 'variable': variable, 'arm': 'nbeats', 'seed': seed, 'smoke_only': quick,
            'study': {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}, 'device': device, 'splits': {}}
    c, k = SCALING[variable]
    blocks = []
    for split in sp:
        if split['reserved'] and seed != 42:
            continue
        data = series(variable, reserved=True) if split['reserved'] else dev
        idx = split['index']
        cut = position(idx, split['cut'])
        lo, hi = split['window']
        issue = grid(idx, lo, hi)
        by_probe = {n: issue[np.isfinite(y[issue])] for n, y in data.items() if np.isfinite(y[issue]).any()}
        Q, keys, info = nbeats_fit_predict(data, idx, cut, by_probe, cfg, device)
        meta['splits'][split['name']] = info
        for lead in LEADS:
            f = pd.DataFrame(Q[:, lead - 1, :] * k + c, columns=P.QCOLS)
            f.insert(0, 'lead', lead)
            f.insert(0, 'time', idx[[p for _, p in keys]])
            f.insert(0, 'sensor', [n for n, _ in keys])
            f.insert(0, 'fold', split['fold'])
            f.insert(0, 'cohort', split['cohort'])
            f.insert(0, 'arm', 'nbeats')
            blocks.append(f)
        print(variable, 'nbeats seed', seed, split['name'], info, flush=True)
    q = pd.concat(blocks, ignore_index=True)
    q[P.QCOLS] = q[P.QCOLS].round(P.DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_nbeats.csv.gz', index=False)
    save(folder / f'statespace_{variable}_nbeats.json', meta)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=list(ARMS))
    p.add_argument('--variable', choices=list(P.VARIABLES))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--procs', type=int, default=4)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    if a.report:
        from src.experiments.v2_statespace_report import report
        report()
    elif a.arm == 'nbeats':
        run_nbeats(a.variable, a.seed, a.quick)
    else:
        run_arima(a.variable, a.arm, a.seed, a.procs, a.quick)
