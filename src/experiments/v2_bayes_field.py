"""Addendum-1 PyMC experiment: hierarchical Bayesian version of the Field-57 spatial residual model.

Declared in config/bay_reserved_protocol_v2_addendum.json before fitting. Same held-station bundles,
background and design as v2 arm A1, except that the Copernicus vertical offset enters as a regressor
with prior Normal(1, 0.5) instead of a fixed coefficient of 1, the regression coefficients share a
learned shrinkage scale (half-normal hyperprior) instead of a selected ridge penalty, and the
likelihood is Student-t with the ridge's sample weights. Point estimate: posterior mean; intervals:
90 % posterior predictive. Development data only; no 2026.
"""
import argparse, hashlib, json, os, traceback
# Trinity has no Python headers for PyTensor's C backend: disable C compilation and sample with
# nutpie (NUTS in Rust, logp compiled with numba). Model and sampler settings are as declared.
os.environ.setdefault('PYTENSOR_FLAGS', 'cxx=')
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler
from src.experiments.bay_profile_field import ROOT, SCALE, prepare, prediction_frame
from src.experiments.bay_field_export import EXTRAS
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_field import nested_mae
from src.data.bay_profile_context import load_profiles, TARGETS

SAMPLER = dict(draws=1000, tune=1000, chains=4, target_accept=.9, random_seed=20260919, progressbar=False, nuts_sampler='nutpie')


def split_rhat(a):
    """Classic split-R-hat (Gelman et al.), max over parameters; a has shape (chain, draw, ...)."""
    a = a.reshape(a.shape[0], a.shape[1], -1); n = a.shape[1] // 2
    s = np.concatenate([a[:, :n], a[:, n:2 * n]], axis=0)                        # 2*chains x n x params
    W = s.var(axis=1, ddof=1).mean(0); B = n * s.mean(axis=1).var(axis=0, ddof=1)
    return float(np.nanmax(np.sqrt(((n - 1) / n * W + B / n) / np.where(W > 0, W, np.nan))))


def fit_target(X, off, r, w, Xv, offv, quick):
    """One target: r = gamma*off + X beta + e, e ~ StudentT; weights enter the log-likelihood."""
    with pm.Model() as m:
        tau = pm.HalfNormal('tau', .5)
        z = pm.Normal('z', 0., 1., shape=X.shape[1])            # non-centred (addendum amendment 1)
        beta = pm.Deterministic('beta', tau * z)
        gamma = pm.Normal('gamma', 1., .5)
        sigma = pm.HalfNormal('sigma', 1.)
        nu = pm.Gamma('nu', 2., .1)
        mu = gamma * off + pm.math.dot(X, beta)
        pm.Potential('lik', (w * pm.logp(pm.StudentT.dist(nu=nu, mu=mu, sigma=sigma), r)).sum())
        idata = pm.sample(**({**SAMPLER, 'draws': 50, 'tune': 50, 'chains': 2} if quick else SAMPLER))
    post = idata.posterior
    b = post['beta'].values.reshape(-1, X.shape[1]); g = post['gamma'].values.ravel()
    s = post['sigma'].values.ravel(); n = post['nu'].values.ravel()
    mu_v = g[:, None] * offv[None, :] + b @ Xv.T                                   # draws x rows
    rng = np.random.default_rng(20260919)
    y_v = mu_v + s[:, None] * rng.standard_t(n[:, None], size=mu_v.shape)            # posterior predictive
    rhat = max(split_rhat(post[v].values) for v in ['tau', 'z', 'gamma', 'sigma', 'nu'])
    dv = np.asarray(idata.sample_stats['diverging'].values); div = int(dv.sum()); total = int(dv.size)
    return mu_v.mean(0), np.quantile(y_v, .05, axis=0), np.quantile(y_v, .95, axis=0), {
        'rhat_max': rhat, 'divergent': div, 'transitions': total, 'gamma_mean': float(g.mean()), 'gamma_sd': float(g.std()), 'tau_mean': float(post['tau'].values.mean())}


def run(quick=False):
    output = ROOT / 'results'; folder = output / 'v2_bayes_field'; folder.mkdir(parents=True, exist_ok=True)
    status = output / 'status_v2_bayes_field.json'; save(status, {'state': 'RUNNING', 'pid': os.getpid()})
    try:
        cfg = json.loads((ROOT / 'config/bay_profile_field.json').read_text())
        bundles, contracts, excluded = prepare('held_site', cfg, quick)
        data, _ = load_profiles()
        context = data[['cast_id', 'depth_m']].merge(pd.read_csv(ROOT / 'reports/bay_twin/profile_boundary_context.csv.gz'),
                                                    on=['cast_id', 'depth_m'], validate='one_to_one')
        extras = context[EXTRAS].to_numpy(dtype=float)
        rows, lo_rows, hi_rows, diag = [], [], [], []
        for b in bundles:
            tr, va = b['tr'], b['va']; w = np.asarray(b['weights'], float); w = w / w.mean()
            base = b['base'].astype(float).copy(); base[:, 0] = np.clip(base[:, 0], -2, 40); base[:, 1] = np.clip(base[:, 1], 0, 42)
            # Vertical-offset columns (extras[:, 3:6]) enter only through gamma, never the design (amendment 1).
            feats = np.concatenate([b['sx'], np.delete(extras, [3, 4, 5], axis=1)], axis=1); sc = StandardScaler().fit(feats[tr]); X = sc.transform(feats)
            y = b['y'].astype(float); pred = np.zeros((len(va), 3)); lo = pred.copy(); hi = pred.copy()
            for j, t in enumerate(TARGETS):
                off = extras[:, 3 + j] / SCALE[j]; r = (y[:, j] - base[:, j]) / SCALE[j]
                m, l, h, d = fit_target(X[tr], off[tr], r[tr], w, X[va], off[va], quick)
                pred[:, j] = base[va, j] + m * SCALE[j]; lo[:, j] = base[va, j] + l * SCALE[j]; hi[:, j] = base[va, j] + h * SCALE[j]
                diag.append({'bundle': b['identity'], 'target': t, **d}); print(b['identity'], t, d, flush=True)
            for a in (pred, lo, hi): a[:, 0] = np.clip(a[:, 0], -2, 40); a[:, 1] = np.clip(a[:, 1], 0, 42)
            rows.append(prediction_frame(b, pred)); lo_rows.append(prediction_frame(b, lo)); hi_rows.append(prediction_frame(b, hi))
        d = pd.concat(rows, ignore_index=True); L = pd.concat(lo_rows, ignore_index=True); H = pd.concat(hi_rows, ignore_index=True)
        for t in TARGETS: d['lo_' + t] = L['pred_' + t].values; d['hi_' + t] = H['pred_' + t].values
        path = folder / 'predictions.csv.gz'; d.to_csv(path, index=False)
        cov = {}
        for t in TARGETS:
            z = d[['fold', 'station', 'cast_id']].assign(c=((d[t] >= d['lo_' + t]) & (d[t] <= d['hi_' + t])).astype(float))
            cov[t] = float(z.groupby(['fold', 'station', 'cast_id']).c.mean().groupby(['fold', 'station']).mean().groupby('fold').mean().mean())
        dg = pd.DataFrame(diag)
        save(folder / 'report.json', {'addendum': 'config/bay_reserved_protocol_v2_addendum.json', 'sampler': SAMPLER, 'smoke_only': quick,
                                      'nested_mae': nested_mae(d), 'nested_coverage_90': cov, 'rows': len(d), 'casts': int(d.cast_id.nunique()),
                                      'rhat_max': float(dg.rhat_max.max()), 'divergent_fraction': float(dg.divergent.sum() / dg.transitions.sum()),
                                      'diagnostics': diag, 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        save(status, {'state': 'COMPLETED'})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'error': str(e)}); traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--quick', action='store_true'); run(p.parse_args().quick)
