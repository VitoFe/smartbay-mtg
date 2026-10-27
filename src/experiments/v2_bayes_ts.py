"""Addendum 72: Bayesian dynamic linear models (DLM) and Bayesian structural time series (BSTS) as forecasters
of the distribution, to 72 hours, against the state-space ARIMA of addendum 67 on the same rows.

  dlm       a Bayesian dynamic linear model in the sense of West and Harrison: a local level (or a local linear
            trend), K_d daily and K_a annual harmonics in Fourier form, and optionally an AR(1) component, with
            component discount factors for the evolution and conjugate (normal-gamma) learning of the
            observational variance with a variance discount. It is filtered through every hourly reading from
            the start of the record; the forecast from an issuance reads every reading up to and including it
            and is the model's own h-step Student-t predictive. The structure and the four discount factors are
            chosen by a nested study (protocol v2, at least 100 trials) on addendum 67's objective and rows,
            one configuration for both development folds and 2026, as for the state-space ARIMA.
  dlm_emp   the same means and scales with addendum 67's band construction: empirical quantiles of the
            standardised h-step errors at the issuances of the 120 days before the cut. It separates the
            DLM's point forecast from its own uncertainty.
  bsts      a Bayesian structural time series in the sense of Scott and Varian: the DLM study's structure as an
            unobserved-components model (statsmodels), its disturbance variances and AR coefficient sampled by
            Gibbs sampling (Durbin-Koopman simulation smoother for the states, inverse-gamma and truncated-
            normal conditionals) on the record before the cut, two chains; the forecast is the posterior
            predictive mixture of the filtered h-step Gaussians over 100 posterior draws.

Declared in config/bay_reserved_protocol_v2_addendum72.json before anything was computed on this protocol.
The 2026 rows are read once, at sampler seed 42; seeds 7 and 123 re-select the DLM on Dev-2025 only.

Producers (Trinity) and the report (local):
  python -m src.experiments.v2_bayes_ts --arm dlm --variable temperature [--seed 7] [--procs 7]
  python -m src.experiments.v2_bayes_ts --arm bsts --variable oxygen --procs 6
  python -m src.experiments.v2_bayes_ts_report
"""
import argparse
import json
import os
import warnings
from multiprocessing import Pool

# one BLAS thread per process: the jobs fan out over probes
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments import v2_probabilistic as P
from src.experiments import v2_statespace as S
from src.experiments.bay_twin_sweep import save
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

OUT = P.RUNS / 'v2_bayes_ts'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum72.json'
LEADS, HMAX = S.LEADS, S.HMAX
ARMS = ('dlm', 'dlm_emp', 'bsts')
HOURS_PER_YEAR = 24 * 365.25
#: BSTS sampler as declared (Gibbs): two chains, burn-in and kept draws per chain, and the draws the forecast mixes
CHAINS, BURN, KEEP, MIX = 2, 300, 300, 100
#: amendment 2: the sampler that replaced Gibbs after G4 failed, adaptive random-walk Metropolis on the
#: Kalman-filter likelihood (states integrated out); chains, adaptive burn-in and kept draws per chain
MH_CHAINS, MH_BURN, MH_KEEP = 4, 1000, 1000
#: inverse-gamma prior on every disturbance variance (scaled units), and the R-hat the gate requires
IG_A, IG_B, RHAT_MAX = 2.0, 1e-4, 1.05
warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------------------------------
# The DLM: structure, filtering and h-step predictive
# --------------------------------------------------------------------------------------------------
def rotation(omega):
    c, s = np.cos(omega), np.sin(omega)
    return np.array([[c, s], [-s, c]])


def structure(cfg):
    """(G, F, blocks): the evolution and observation matrices and, per discount group, the state indices."""
    G_blocks, F, groups = [], [], {'trend': [], 'seasonal': [], 'ar': []}
    i = 0
    if cfg['trend'] == 'linear':
        G_blocks.append(np.array([[1., 1.], [0., 1.]])); F += [1., 0.]; groups['trend'] += [0, 1]; i = 2
    else:
        G_blocks.append(np.array([[1.]])); F += [1.]; groups['trend'] += [0]; i = 1
    for j in range(1, cfg['daily'] + 1):
        G_blocks.append(rotation(2 * np.pi * j / 24)); F += [1., 0.]; groups['seasonal'] += [i, i + 1]; i += 2
    for j in range(1, cfg['annual'] + 1):
        G_blocks.append(rotation(2 * np.pi * j / HOURS_PER_YEAR)); F += [1., 0.]; groups['seasonal'] += [i, i + 1]; i += 2
    if cfg['ar']:
        G_blocks.append(np.array([[cfg['phi']]])); F += [1.]; groups['ar'] += [i]; i += 1
    n = sum(b.shape[0] for b in G_blocks)
    G = np.zeros((n, n))
    k = 0
    for b in G_blocks:
        m = b.shape[0]
        G[k:k + m, k:k + m] = b
        k += m
    return G, np.array(F), {g: np.array(v, dtype=int) for g, v in groups.items() if v}


def discount_W(P_, groups, cfg):
    """Component discounting: W = block-diag((1/delta_g - 1) P_gg) over the discount groups."""
    W = np.zeros_like(P_)
    for g, idx in groups.items():
        d = cfg[f'delta_{g}']
        W[np.ix_(idx, idx)] = (1. / d - 1.) * P_[np.ix_(idx, idx)]
    return W


def dlm_filter(y, cfg, keep, known_variance=None):
    """Filter y (NaN where missing) through every hour and return, at each position in `keep`, the h-step
    predictive (mean, scale^2, degrees of freedom) for h = 1..HMAX from the posterior after that reading.

    With `known_variance` the observational variance is fixed (no learning, no variance discount) and the
    predictive is Gaussian (degrees of freedom inf): the case gate G2 checks against a Kalman filter."""
    G, F, groups = structure(cfg)
    n_state = len(F)
    first = np.flatnonzero(np.isfinite(y))
    m = np.zeros(n_state)
    m[0] = y[first[0]] if len(first) else 0.
    C = np.eye(n_state)
    if known_variance is None:
        n_dof, S_est = 1., 0.01
    else:
        n_dof, S_est = np.inf, float(known_variance)
    beta = cfg['beta'] if known_variance is None else 1.
    keep = np.asarray(keep, dtype=int)
    want = np.zeros(len(y), dtype=bool)
    want[keep[keep < len(y)]] = True
    out_mean = np.full((len(keep), HMAX), np.nan)
    out_var = np.full((len(keep), HMAX), np.nan)
    out_dof = np.full(len(keep), np.nan)
    row_of = {int(p): r for r, p in enumerate(keep)}
    Gpow = [np.eye(n_state)]
    for _ in range(HMAX):
        Gpow.append(G @ Gpow[-1])
    for t in range(len(y)):
        # evolve to t, then update with the reading at t
        a = G @ m
        P_ = G @ C @ G.T
        R = P_ + discount_W(P_, groups, cfg)
        yt = y[t]
        if np.isfinite(yt):
            f = F @ a
            RF = R @ F
            q = F @ RF + S_est
            e = yt - f
            A = RF / q
            if known_variance is None:
                n_prior, d_prior = beta * n_dof, beta * n_dof * S_est
                n_dof = n_prior + 1.
                S_new = (d_prior + S_est * e * e / q) / n_dof
                C = (S_new / S_est) * (R - np.outer(A, A) * q)
                S_est = S_new
            else:
                C = R - np.outer(A, A) * q
            m = a + A * e
        else:
            m, C = a, R
        C = 0.5 * (C + C.T)
        if want[t]:
            r = row_of[t]
            # h-step: R_h = G R_{h-1} G' + W, W held at the one-step value from this posterior
            P1 = G @ C @ G.T
            W = discount_W(P1, groups, cfg)
            Rh = C
            for h in range(1, HMAX + 1):
                Rh = G @ Rh @ G.T + W
                out_mean[r, h - 1] = F @ (Gpow[h] @ m)
                out_var[r, h - 1] = F @ Rh @ F + S_est
            out_dof[r] = n_dof
    return out_mean, np.maximum(out_var, 1e-12), out_dof


def t_quantiles(mean, var, dof, rows, lead):
    """The Student-t predictive's quantiles at P.LEVELS for `rows` at one lead (scaled units)."""
    m, s = mean[rows, lead - 1], np.sqrt(var[rows, lead - 1])
    z = np.stack([stats.t.ppf(P.LEVELS, df) if np.isfinite(df) else stats.norm.ppf(P.LEVELS) for df in dof[rows]])
    Q = m[:, None] + s[:, None] * z
    Q.sort(axis=1)
    return Q


def dlm_probe(args):
    """One probe in one split: filter, forecast at `issue`, and the empirical multipliers from `err`."""
    y, index, cut, nobs, issue, err, cfg, check = args
    y = y[:nobs]
    idx = np.r_[err, issue]
    mean, var, dof = dlm_filter(y, cfg, idx)
    mult = S.multipliers(y, mean[:len(err)], var[:len(err)], err)
    info = {'readings_before_cut': int(np.isfinite(y[:cut]).sum()),
            'dof_at_issuance': [float(np.nanmin(dof[len(err):])), float(np.nanmax(dof[len(err):]))] if len(issue) else None}
    if check:
        info['g2_max_difference'] = gate_g2(y, cfg, issue)
    return mean[len(err):], var[len(err):], dof[len(err):], mult, info


def gate_g2(y, cfg, issue):
    """G2: with every discount factor 1 and the observational variance known, the DLM is a static Gaussian
    state-space model; its h-step means and variances must equal statsmodels' Kalman filter forecasts from the
    same prior, at two issuances."""
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    fixed = dict(cfg, delta_trend=1., delta_seasonal=1., delta_ar=1.)
    G, F, _ = structure(fixed)
    V = 0.01
    diffs = []
    for t in issue[[0, len(issue) // 2]] if len(issue) else []:
        if t + 1 >= len(y):
            continue
        # the same prior as the filter on y[:t + 1]: the level starts at the first reading up to t
        first = np.flatnonzero(np.isfinite(y[:t + 1]))
        a0 = np.zeros(len(F)); a0[0] = y[first[0]] if len(first) else 0.
        mean, var, _ = dlm_filter(y[:t + 1], fixed, [t], known_variance=V)
        mod = MLEModel(y[:t + 1], k_states=len(F))
        mod['design'] = F[None, :]
        mod['transition'] = G
        mod['selection'] = np.eye(len(F))
        mod['state_cov'] = np.zeros((len(F), len(F)))
        mod['obs_cov'] = np.array([[V]])
        # the DLM's prior for the first state is a0, C0 = I, evolved by G: statsmodels' initial state is the
        # predicted state at t = 0, i.e. G a0 with covariance G G'
        mod.ssm.initialize_known(G @ a0, G @ G.T)
        res = mod.ssm.filter()
        k = np.asarray(res.filtered_state)[:, -1]
        Pk = np.asarray(res.filtered_state_cov)[:, :, -1]
        h_mean, h_var = [], []
        A, Pm = k, Pk
        for h in range(1, HMAX + 1):
            A, Pm = G @ A, G @ Pm @ G.T
            h_mean.append(F @ A); h_var.append(F @ Pm @ F + V)
        diffs.append(max(np.abs(np.array(h_mean) - mean[0]).max(), np.abs(np.array(h_var) - var[0]).max()))
    return float(max(diffs)) if diffs else None


# --------------------------------------------------------------------------------------------------
# The study: addendum 67's objective on its inner rows
# --------------------------------------------------------------------------------------------------
def dlm_space(trial):
    cfg = {'trend': trial.suggest_categorical('trend', ['level', 'linear']),
           'daily': trial.suggest_int('daily', 0, 4), 'annual': trial.suggest_int('annual', 0, 1),
           'ar': trial.suggest_categorical('ar', [False, True]),
           'delta_trend': trial.suggest_float('delta_trend', 0.90, 1.0),
           'delta_seasonal': trial.suggest_float('delta_seasonal', 0.95, 1.0),
           'beta': trial.suggest_float('beta', 0.95, 1.0)}
    cfg['phi'] = trial.suggest_float('phi', 0.0, 0.99) if cfg['ar'] else 0.
    cfg['delta_ar'] = trial.suggest_float('delta_ar', 0.90, 1.0) if cfg['ar'] else 1.
    return cfg


def complete(params):
    """A study's selected parameters with the conditional ones filled in."""
    cfg = dict(params)
    cfg.setdefault('phi', 0.)
    cfg.setdefault('delta_ar', 1.)
    return cfg


def objective_value(rows, forecasts, own_bands):
    """Addendum 67's objective: mean over folds and leads of the sensor-macro quantile score over the
    climatology's. forecasts: {fold: {probe: (mean, var, dof, mult, positions)}}."""
    ratios = []
    for fold, by_lead in rows.items():
        for lead, (names, times, y, C) in by_lead.items():
            if not len(y):
                continue
            pos = S.DEV_INDEX.get_indexer(times)
            Q = np.empty_like(C)
            for n in np.unique(names):
                mean, var, dof, mult, issue = forecasts[fold][n]
                at = np.searchsorted(issue, pos[names == n])
                Q[names == n] = (t_quantiles(mean, var, dof, at, lead) if own_bands
                                 else S.quantiles_at(mean, var, mult, at, lead))
            if not np.isfinite(Q).all():
                raise ValueError(f'non-finite forecast, fold {fold}, lead {lead}')
            ratios.append(P.sensor_macro(P.quantile_score(y, Q), names) / P.sensor_macro(P.quantile_score(y, C), names))
    return float(np.mean(ratios))


def frames_from(variable, arm, split, names, quantiles_by_probe, issue, data):
    c, k = S.SCALING[variable]
    blocks = []
    for n in names:
        keep = np.flatnonzero(np.isfinite(data[n][issue]))
        for lead in LEADS:
            Q = quantiles_by_probe[n](keep, lead) * k + c
            f = pd.DataFrame(Q, columns=P.QCOLS)
            f.insert(0, 'lead', lead)
            f.insert(0, 'time', split['index'][issue[keep]])
            f.insert(0, 'sensor', n)
            f.insert(0, 'fold', split['fold'])
            f.insert(0, 'cohort', split['cohort'])
            f.insert(0, 'arm', arm)
            blocks.append(f)
    return blocks


def run_dlm(variable, seed, procs, quick):
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    dev = S.series(variable, reserved=False)
    sp = S.splits()
    if quick:
        dev = {n: dev[n] for n in list(dev)[:3]}
    meta = {'declared_in': DECLARATION, 'variable': variable, 'arm': 'dlm', 'seed': seed, 'smoke_only': quick}
    rows = S.inner_rows(variable)
    if quick:
        rows = {f: {L: tuple(a[np.isin(r[0], list(dev))] for a in r) for L, r in by.items() if L in (6, 72)}
                for f, by in rows.items()}
    with Pool(procs) as pool:
        def objective(trial):
            cfg = dlm_space(trial)
            forecasts = {}
            for split in sp[:2]:
                names, jobs, issue = S.inner_jobs(dev, rows, split, cfg)
                got = pool.map(dlm_probe, jobs)
                forecasts[split['fold']] = {n: (m, v, d, mu, issue) for n, (m, v, d, mu, _) in zip(names, got)}
            return objective_value(rows, forecasts, own_bands=True)
        sel = run_study(f'{variable}_dlm', objective, folder / 'studies', quick)
        cfg = complete(sel['params'])
        meta['study'] = {k: sel[k] for k in ('completed_trials', 'best_value', 'params')}
        meta['config'] = cfg
        blocks, infos = {'dlm': [], 'dlm_emp': []}, {}
        for split in sp:
            # 2026 is read once, at seed 42 of the full protocol; a smoke run never reads it
            if split['reserved'] and (seed != 42 or quick):
                continue
            data = S.series(variable, reserved=True) if split['reserved'] else dev
            if quick:
                data = {n: data[n] for n in list(dev) if n in data}
            names, jobs, issue = S.outer_jobs(data, split, cfg, check=True)
            got = pool.map(dlm_probe, jobs)
            own = {n: (lambda rows_, lead, g=g: t_quantiles(g[0], g[1], g[2], rows_, lead)) for n, g in zip(names, got)}
            emp = {n: (lambda rows_, lead, g=g: S.quantiles_at(g[0], g[1], g[3], rows_, lead)) for n, g in zip(names, got)}
            blocks['dlm'] += frames_from(variable, 'dlm', split, names, own, issue, data)
            blocks['dlm_emp'] += frames_from(variable, 'dlm_emp', split, names, emp, issue, data)
            infos[split['name']] = {n: g[4] for n, g in zip(names, got)}
            print(variable, 'dlm seed', seed, split['name'], len(names), 'probes', flush=True)
    for arm, b in blocks.items():
        q = pd.concat(b, ignore_index=True)
        q[P.QCOLS] = q[P.QCOLS].round(P.DIGITS[variable])
        q.to_csv(folder / f'quantiles_{variable}_{arm}.csv.gz', index=False)
    g2 = [x['g2_max_difference'] for s in infos.values() for x in s.values() if x.get('g2_max_difference') is not None]
    meta.update({'probes': infos, 'gates': {'G2': {'checked': len(g2), 'max_difference': max(g2) if g2 else None,
                                                   'passed': bool(g2) and max(g2) <= 1e-8}}})
    save(folder / f'bayes_{variable}_dlm.json', meta)
    print(variable, 'dlm seed', seed, 'G2', meta['gates']['G2'], flush=True)


# --------------------------------------------------------------------------------------------------
# BSTS: Gibbs sampling of the unobserved-components model's variances
# --------------------------------------------------------------------------------------------------
def uc_model(y, cfg):
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    freq = [{'period': 24, 'harmonics': cfg['daily']}] if cfg['daily'] else []
    if cfg['annual']:
        freq.append({'period': HOURS_PER_YEAR, 'harmonics': cfg['annual']})
    return UnobservedComponents(y, level='local linear trend' if cfg['trend'] == 'linear' else 'local level',
                                freq_seasonal=freq or None, autoregressive=1 if cfg['ar'] else None)


def state_layout(mod, cfg):
    """{parameter name: (state indices whose disturbances it scales)} for the model's variances."""
    names = mod.param_names
    layout, i = {}, 0
    if cfg['trend'] == 'linear':
        layout['sigma2.level'] = [0]; layout['sigma2.trend'] = [1]; i = 2
    else:
        layout['sigma2.level'] = [0]; i = 1
    for name in names:
        if name.startswith('sigma2.freq_seasonal_'):
            # 'sigma2.freq_seasonal_24(3)': one variance shared by the 2 x 3 states of that frequency block
            harmonics = int(name[name.rindex('(') + 1:-1])
            layout[name] = list(range(i, i + 2 * harmonics)); i += 2 * harmonics
    if cfg['ar']:
        layout['sigma2.ar'] = [i]; i += 1
    if i != mod.k_states:
        raise ValueError(f'layout covers {i} states, the model has {mod.k_states}')
    for name in layout:
        if name not in names:
            raise ValueError(f'unexpected layout: {name} not in {names}')
    return layout


def gibbs_chain(y, cfg, seed, burn=BURN, keep=KEEP):
    """One Gibbs chain over the variances (and the AR coefficient) of the UC model on y; returns kept draws."""
    rng = np.random.default_rng(seed)
    mod = uc_model(y, cfg)
    names = list(mod.param_names)
    layout = state_layout(mod, cfg)
    obs = np.isfinite(y)
    start = mod.fit(disp=False, maxiter=100).params                 # a starting point only
    params = np.asarray(start, float).copy()
    params = params * np.exp(rng.normal(0, 0.5, len(params))) if seed % 2 else params
    if cfg['ar']:
        params[names.index('ar.L1')] = np.clip(start[names.index('ar.L1')], -0.95, 0.95)
    sim = mod.simulation_smoother()
    draws = []
    for it in range(burn + keep):
        mod.update(params)
        sim.simulate(random_state=rng)
        alpha = np.asarray(sim.simulated_state)                      # (states, nobs)
        T = np.asarray(mod.ssm['transition'])
        T = T[:, :, 0] if T.ndim == 3 else T
        Z = np.asarray(mod.ssm['design'])
        Z = Z[:, :, 0] if Z.ndim == 3 else Z
        # observation variance
        resid = y[obs] - (Z @ alpha)[0, obs]
        params[names.index('sigma2.irregular')] = 1. / rng.gamma(IG_A + resid.size / 2, 1. / (IG_B + resid @ resid / 2))
        # state disturbance variances
        incr = alpha[:, 1:] - T @ alpha[:, :-1]
        for name, idx in layout.items():
            u = incr[idx].ravel()
            params[names.index(name)] = 1. / rng.gamma(IG_A + u.size / 2, 1. / (IG_B + u @ u / 2))
        # AR coefficient: normal conditional, truncated to the stationary region
        if cfg['ar']:
            i = layout['sigma2.ar'][0]
            x0, x1 = alpha[i, :-1], alpha[i, 1:]
            s2 = params[names.index('sigma2.ar')]
            prec = x0 @ x0 / s2
            mu = (x0 @ x1 / s2) / prec
            phi = stats.truncnorm.rvs((-0.99 - mu) * np.sqrt(prec), (0.99 - mu) * np.sqrt(prec),
                                      loc=mu, scale=1 / np.sqrt(prec), random_state=rng)
            params[names.index('ar.L1')] = float(phi)
        if it >= burn:
            draws.append(params.copy())
    return names, np.array(draws)


AR_BOUND = 0.99


def from_theta(names, theta):
    """theta -> the model's parameters: exp of a log variance, and AR_BOUND tanh(z) for the AR coefficient."""
    return np.array([AR_BOUND * np.tanh(t) if n == 'ar.L1' else np.exp(t) for n, t in zip(names, theta)])


def to_theta(names, params):
    return np.array([np.arctanh(np.clip(p / AR_BOUND, -0.999, 0.999)) if n == 'ar.L1' else np.log(max(p, 1e-7))
                     for n, p in zip(names, params)])


def log_posterior(mod, names, theta):
    """The declared posterior on the sampler's scale: theta holds the log of every variance and, since amendment 4,
    z = artanh(phi / 0.99) for the AR coefficient phi. Inverse-gamma(IG_A, IG_B) on each variance times the
    Jacobian of its log, uniform phi on (-0.99, 0.99) times the Jacobian of z, and the Kalman-filter likelihood
    with the states integrated out."""
    params = from_theta(names, theta)
    lp = 0.
    for j, n in enumerate(names):
        if n == 'ar.L1':
            lp += np.log1p(-np.tanh(theta[j]) ** 2)
        else:
            lp += -IG_A * theta[j] - IG_B / params[j]
    ll = mod.loglike(params) if np.all(np.isfinite(params)) else np.nan
    return (ll + lp, params) if np.isfinite(ll + lp) else (-np.inf, None)


def laplace(mod, names, theta0):
    """Amendments 3 and 4: the posterior mode on theta (L-BFGS-B from theta0) and the inverse of the negative log
    posterior's Hessian there, by central differences with steps of 1e-2. An eigenvalue of the Hessian below 1
    is raised to 1, so no direction gets a spread wider than one unit of theta; how many were raised is
    returned."""
    from scipy.optimize import minimize

    def f(t):
        v = -log_posterior(mod, names, t)[0]
        return v if np.isfinite(v) else 1e12
    mode = minimize(f, theta0, method='L-BFGS-B').x
    d = len(names)
    h = np.full(d, 1e-2)
    H = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.eye(d)[i] * h[i], np.eye(d)[j] * h[j]
            H[i, j] = H[j, i] = (f(mode + ei + ej) - f(mode + ei - ej) - f(mode - ei + ej)
                                 + f(mode - ei - ej)) / (4 * h[i] * h[j])
    w, V = np.linalg.eigh(H)
    raised = int((w < 1).sum())
    cov = V @ np.diag(1 / np.maximum(w, 1.)) @ V.T
    return mode, cov, raised


def mh_chain(mod, names, start, cov, seed, burn=MH_BURN, keep=MH_KEEP):
    """One random-walk Metropolis chain on theta, proposal covariance s^2 2.38^2/d times the Laplace covariance.
    During burn-in s is multiplied by exp(2 (a - 0.25)) after every 100 iterations, a being the acceptance rate
    of those 100 (amendment 4); it is fixed for the kept draws, so those are an ordinary Metropolis chain.
    Returns the kept draws on the parameters' own scale, the acceptance rate of the kept draws and s."""
    rng = np.random.default_rng(seed)
    d = len(names)
    theta = start.copy()
    lp, params = log_posterior(mod, names, theta)
    if params is None:
        raise ValueError('the starting point has zero posterior density')
    chol = np.linalg.cholesky(2.38 ** 2 / d * cov)
    s, window = 1., 0
    draws, accepted = [], 0
    for it in range(burn + keep):
        prop = theta + s * (chol @ rng.standard_normal(d))
        lp_new, p_new = log_posterior(mod, names, prop)
        if np.log(rng.uniform()) < lp_new - lp:
            theta, lp, params = prop, lp_new, p_new
            accepted += it >= burn
            window += 1
        if it < burn and (it + 1) % 100 == 0:
            s *= np.exp(2 * (window / 100 - 0.25))
            window = 0
        if it >= burn:
            draws.append(params.copy())
    return np.array(draws), accepted / max(keep, 1), s


def mh_posterior(y, cfg, seed, burn=MH_BURN, keep=MH_KEEP):
    """MH_CHAINS chains started at the posterior mode plus twice a draw from the Laplace approximation, so the
    starts are overdispersed relative to the posterior. Returns names, (chains, keep, params), the acceptance
    rates and the Laplace step's record."""
    mod = uc_model(y, cfg)
    names = list(mod.param_names)
    mle = to_theta(names, np.asarray(mod.fit(disp=False, maxiter=200).params, float))
    mode, cov, raised = laplace(mod, names, mle)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    chains, rates, scales = [], [], []
    for c in range(MH_CHAINS):
        start = mode + 2 * L @ rng.standard_normal(len(names))
        d, rate, s = mh_chain(mod, names, start, cov, seed * 100 + c, burn, keep)
        chains.append(d); rates.append(rate); scales.append(float(s))
    record = {'mode': dict(zip(names, map(float, from_theta(names, mode)))),
              'laplace_sd_theta': dict(zip(names, map(float, np.sqrt(np.diag(cov))))),
              'eigenvalues_raised': raised, 'proposal_scale': scales}
    return names, np.stack(chains), rates, record


def split_rhat(chains):
    """Split R-hat (Gelman et al.) per parameter over chains of equal length: (chains, draws, params)."""
    c, n, k = chains.shape
    half = n // 2
    parts = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, l = parts.shape[0], parts.shape[1]
    means = parts.mean(axis=1)
    W = parts.var(axis=1, ddof=1).mean(axis=0)
    B = l * means.var(axis=0, ddof=1)
    var = (l - 1) / l * W + B / l
    return np.sqrt(var / np.maximum(W, 1e-300))


def mixture_quantiles(means, variances, levels):
    """Quantiles of an equal-weight Gaussian mixture, row by row: means and variances are (rows, components)."""
    sd = np.sqrt(variances)
    lo = (means - 8 * sd).min(axis=1)
    hi = (means + 8 * sd).max(axis=1)
    out = np.empty((means.shape[0], len(levels)))
    for j, a in enumerate(levels):
        a_lo, a_hi = lo.copy(), hi.copy()
        for _ in range(60):
            mid = 0.5 * (a_lo + a_hi)
            cdf = stats.norm.cdf((mid[:, None] - means) / sd).mean(axis=1)
            a_lo = np.where(cdf < a, mid, a_lo)
            a_hi = np.where(cdf >= a, mid, a_hi)
        out[:, j] = 0.5 * (a_lo + a_hi)
    return out


def bsts_probe(args):
    """One probe in one split: the posterior of the parameters on the record before the cut (Metropolis since
    amendment 2), then the posterior predictive mixture at `issue` from the filter run with each of MIX
    posterior draws through every hour."""
    y, index, cut, nobs, issue, cfg, seed, quick, pooled = args
    y = y[:nobs]
    burn, keep = (300, 300) if quick else (MH_BURN, MH_KEEP)
    info = {'readings_before_cut': int(np.isfinite(y[:cut]).sum())}
    if info['readings_before_cut'] < S.MIN_READINGS:
        # amendment 1: the median of the sampled probes' posterior medians in the same split (their records
        # end at the same cut), since maximum likelihood is undefined on a probe with no reading before it
        info['fallback'] = 'too few readings: the median over the split of the sampled probes posterior medians'
        draws = np.asarray(pooled, float)[None, :]
    else:
        names, chains, rates, info['laplace'] = mh_posterior(y[:cut], cfg, seed, burn, keep)
        info['acceptance'] = [float(r) for r in rates]
        rhat = split_rhat(chains)
        info['rhat'] = {n: float(r) for n, r in zip(names, rhat)}
        info['rhat_max'] = float(np.max(rhat))
        info['posterior_median'] = {n: float(v) for n, v in zip(names, np.median(chains.reshape(-1, len(names)), axis=0))}
        flat = chains.reshape(-1, len(names))
        pick = np.random.default_rng(seed).choice(len(flat), size=min(MIX, len(flat)), replace=False)
        draws = flat[pick]
    full = uc_model(y, cfg)
    means, varis = [], []
    for p in draws:
        res = full.filter(p)
        m, v = S.multistep(res, issue, nobs)
        means.append(m); varis.append(v)
    means, varis = np.stack(means, axis=-1), np.stack(varis, axis=-1)       # (issuances, HMAX, draws)
    return means, varis, info


def run_bsts(variable, seed, procs, quick):
    folder = OUT / ('smoke' if quick else '') / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    # the structure is the DLM study's selection: that job's output in this run, or (amendment 2) the full
    # run's selection shipped with the snapshot
    sources = [folder / f'bayes_{variable}_dlm.json', OUT / f'seed{seed}' / f'bayes_{variable}_dlm.json']
    import time
    for _ in range(24 * 60):
        if any(f.exists() for f in sources):
            break
        time.sleep(60)
    source = next(f for f in sources if f.exists())
    dlm_meta = json.loads(source.read_text())
    cfg = dlm_meta['config']
    meta = {'declared_in': DECLARATION, 'amended_by': ['config/bay_reserved_protocol_v2_addendum72_amendment2.json',
                                                                   'config/bay_reserved_protocol_v2_addendum72_amendment3.json',
                                                                   'config/bay_reserved_protocol_v2_addendum72_amendment4.json'],
            'variable': variable, 'arm': 'bsts', 'seed': seed, 'smoke_only': quick,
            'structure_from': source.relative_to(P.RUNS).as_posix(), 'config': cfg,
            'sampler': {'kind': 'random-walk Metropolis on the Kalman-filter likelihood, Laplace proposal at the mode',
                        'chains': MH_CHAINS, 'burn': MH_BURN, 'keep': MH_KEEP, 'mixture_draws': MIX,
                        'prior': f'inverse-gamma({IG_A}, {IG_B}) on every variance, uniform AR coefficient on (-0.99, 0.99)'}}
    dev = S.series(variable, reserved=False)
    if quick:
        dev = {n: dev[n] for n in list(dev)[:3]}
    blocks, infos = [], {}
    with Pool(procs) as pool:
        for split in S.splits():
            if split['reserved'] and quick:          # a smoke run never reads 2026
                continue
            data = S.series(variable, reserved=True) if split['reserved'] else dev
            if quick:
                data = {n: data[n] for n in list(dev) if n in data}
            idx = split['index']
            cut = S.position(idx, split['cut'])
            lo, hi = split['window']
            issue = S.grid(idx, lo, hi)
            names = [n for n, y in data.items() if np.isfinite(y[issue]).any()]
            sampled = [n for n in names if np.isfinite(data[n][:cut]).sum() >= S.MIN_READINGS]
            sparse = [n for n in names if n not in sampled]
            got = dict(zip(sampled, pool.map(bsts_probe, [(data[n], idx, cut, len(idx), issue, cfg, seed, quick, None)
                                                         for n in sampled])))
            if sparse:
                pooled = np.median([list(got[n][2]['posterior_median'].values()) for n in sampled], axis=0)
                got.update(zip(sparse, pool.map(bsts_probe, [(data[n], idx, cut, len(idx), issue, cfg, seed, quick,
                                                              pooled) for n in sparse])))
            got = [got[n] for n in names]
            quant = {n: (lambda rows_, lead, g=g: mixture_quantiles(g[0][rows_, lead - 1, :], g[1][rows_, lead - 1, :],
                                                                    P.LEVELS)) for n, g in zip(names, got)}
            blocks += frames_from(variable, 'bsts', split, names, quant, issue, data)
            infos[split['name']] = {n: g[2] for n, g in zip(names, got)}
            print(variable, 'bsts', split['name'], len(names), 'probes', flush=True)
    q = pd.concat(blocks, ignore_index=True)
    q[P.QCOLS] = q[P.QCOLS].round(P.DIGITS[variable])
    q.to_csv(folder / f'quantiles_{variable}_bsts.csv.gz', index=False)
    rh = [x['rhat_max'] for s in infos.values() for x in s.values() if 'rhat_max' in x]
    meta.update({'probes': infos, 'gates': {'G4': {'probe_splits': len(rh), 'rhat_max': max(rh) if rh else None,
                                                   'passed': bool(rh) and max(rh) <= RHAT_MAX}}})
    save(folder / f'bayes_{variable}_bsts.json', meta)
    print(variable, 'bsts G4', meta['gates']['G4'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['dlm', 'bsts'], required=True)
    p.add_argument('--variable', choices=list(P.VARIABLES), required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--procs', type=int, default=4)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.arm == 'dlm':
        run_dlm(a.variable, a.seed, a.procs, a.quick)
    else:
        run_bsts(a.variable, a.seed, a.procs, a.quick)


if __name__ == '__main__':
    main()
