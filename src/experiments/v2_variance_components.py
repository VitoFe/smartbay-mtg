"""Addendum-22: decompose the co-located disagreement into variance components.

The thesis quotes a co-located agreement of 0.20 to 0.53 degC and uses it as the interpretive
floor for every temperature contrast. A range over pairs summarises a distribution; it does not
say whether the disagreement is a fixed offset between two instruments, a slow drift that
recalibration would remove, or short-term noise that averaging would reduce. Those three have
different engineering consequences.

Delta(t) = b + d(t) + eps(t), where b is the mean over the overlap, d is a centred 30-day rolling
median minus b, and eps is the remainder. A difference of two exchangeable probes carries twice
the variance of one, so each pooled component is divided by sqrt(2) to express it per probe.

The pooled offset is the root mean square of b over pairs. The declaration asks for "the standard
deviation of b across pairs" under the stated assumption that the two probes of a pair are
exchangeable, which gives b a population mean of zero; the sample standard deviation about the
sample mean instead depends on which probe of each pair is labelled a (alphabetical here), ranges
from 0.093 to 0.199 degC over the 2^11 labellings, and with these labels removes a bias of about
+0.13 degC between the WNode and EXO2 families. The labelled standard deviation and the family bias
are reported beside it.

The three terms are an additive identity of time series, not an orthogonal decomposition: the
drift is a rolling median, so it is not uncorrelated with the remainder, and the offset is taken
over every overlap hour while the drift and the remainder exist only where the rolling window is
full. Summing the three standard deviations in quadrature therefore omits the drift-remainder
covariance and mixes supports. Each pair also reports, on the hours where all three terms exist,
the exact identity mean(Delta^2) = mean(Delta)^2 + var(d) + var(eps) + 2 cov(d, eps), and the
budget reports the root-mean-square pair difference measured directly on those hours beside the
quadrature sum. Expressing a pair figure per probe by sqrt(2) assumes
the two probes' deviations are independent with equal variance; probes shared between pairs and
common deviations are outside that assumption.

This is a REPRODUCIBILITY budget, not an accuracy budget. Nothing here is traceable to a national
standard, no probe was ever swapped between positions, and instrument and position effects are
confounded within each pair. It describes how well two nominally identical deployments agree with
each other, not how far either sits from the true temperature. A pair difference also cannot see
an error common to both probes of a pair.

Declared in config/bay_reserved_protocol_v2_addendum22.json.
Usage: python -m src.experiments.v2_variance_components
"""
import json
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import stats

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT

OUT = ROOT / 'reports/bay_twin/runs/v2_variance_components.json'
SEPARATION_M = 60.0       # the radius Table 3.2 treats as co-located
MIN_OVERLAP_H = 50
DRIFT_WINDOW = '30D'      # slow component; anything faster is short-term
QUANT_STEP = 0.1          # degC, the WNode export resolution
QUANT_U = QUANT_STEP / np.sqrt(12)
EXO2_STEP = 0.01          # degC, the EXO2 export resolution (Chapter 3, reporting resolution)
PRE_2026 = pd.Timestamp('2026-01-01', tz='UTC')


def step_of(name):
    """Reporting step of a probe's temperature export: WNode loggers write one decimal place."""
    return QUANT_STEP if name.startswith('Nodo_') else EXO2_STEP


def hourly(series):
    """Reindex onto a uniform hourly grid so that time windows span hours, not observations."""
    s = series.dropna()
    s = s[~s.index.duplicated(keep='first')].sort_index()
    if s.empty:
        return s
    grid = pd.date_range(s.index[0].ceil('h'), s.index[-1].floor('h'), freq='h')
    return s.reindex(grid)


def decompose(delta, how='median'):
    """Split a difference series into a fixed offset, a slow drift and a short-term residual.

    The declaration specifies a rolling median. A median of a series quantised at 0.1 degC snaps
    to grid values and can return a step function with zero variance, which understates drift on
    WNode-against-WNode pairs. The rolling mean is computed alongside as a stated deviation; it
    averages hundreds of values and is not resolution-limited.
    """
    b = float(delta.mean())
    roll = delta.rolling(DRIFT_WINDOW, center=True, min_periods=24)
    slow = roll.median() if how == 'median' else roll.mean()
    d = slow - b
    eps = delta - b - d
    return b, d.dropna(), eps.dropna()


def main():
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    series = {}
    for name in meta.index:
        _, frame, _ = read_export(ROOT / 'data/raw' / f'{name}.txt')
        series[name] = hourly(frame['temperature'])

    rows = []
    for a, b_name in combinations(meta.index, 2):
        dx = np.hypot(meta.loc[a].x_m - meta.loc[b_name].x_m,
                      meta.loc[a].y_m - meta.loc[b_name].y_m)
        if dx > SEPARATION_M:
            continue
        joined = pd.concat([series[a].rename('a'), series[b_name].rename('b')], axis=1).dropna()
        if len(joined) < MIN_OVERLAP_H:
            continue
        delta = joined.a - joined.b
        offset, drift, resid = decompose(delta)
        _, drift_m, resid_m = decompose(delta, how='mean')
        # The same decomposition on hours before 2026 only, for the post-hoc failure ranking: probes
        # that stopped in 2025 have no 2026 hours, so a whole-overlap figure would compare them with
        # survivors measured partly over the period whose outcome is being ranked.
        before = delta[delta.index < PRE_2026]
        resid_pre = decompose(before)[2] if len(before) >= MIN_OVERLAP_H else None
        # The short-term term is autocorrelated, so averaging over a day shrinks it by much less than
        # the square root of 24 that white noise would give; measure it rather than assume it.
        # The exact second-moment identity on the hours where all three terms exist (common support).
        common = pd.concat([delta.rename('x'), drift.rename('d'), resid.rename('e')], axis=1).dropna()
        if len(common) > 1:
            mean_c = float(common.x.mean())
            var_d = float(common.d.var(ddof=0)); var_e = float(common.e.var(ddof=0))
            cov_de = float(((common.d - common.d.mean()) * (common.e - common.e.mean())).mean())
            m2 = float((common.x ** 2).mean())
            identity_error = m2 - (mean_c ** 2 + var_d + var_e + 2 * cov_de)
        else:
            mean_c = var_d = var_e = cov_de = m2 = identity_error = None
        daily = resid.resample('D').agg(['mean', 'count'])
        daily = daily.loc[daily['count'] >= 18, 'mean']
        lag1 = float(resid.autocorr(1)) if len(resid) > 2 else None
        # Sensitivity of the drift/short-term split to the drift window, which was fixed a priori.
        windows = {}
        for w in ('7D', '14D', '60D'):
            roll = delta.rolling(w, center=True, min_periods=24).median()
            eps_w = (delta - roll).dropna()
            d_w = (roll - delta.mean()).dropna()
            windows[w] = {'drift_sd': round(float(d_w.std()), 4) if len(d_w) > 1 else None,
                          'residual_sd': round(float(eps_w.std()), 4) if len(eps_w) > 1 else None}
        span_d = (joined.index[-1] - joined.index[0]).total_seconds() / 86400.0
        rows.append({
            'a': a, 'b': b_name,
            'separation_m': round(float(dx), 2),
            'depth_separation_m': round(abs(float(meta.loc[a].nominal_depth_m)
                                            - float(meta.loc[b_name].nominal_depth_m)), 2),
            'overlap_h': int(len(joined)), 'span_days': round(span_d, 1),
            'mean_abs_difference': round(float(delta.abs().mean()), 4),
            'offset_b': round(offset, 4),
            'drift_sd': round(float(drift.std()), 4) if len(drift) > 1 else None,
            'drift_range': round(float(drift.max() - drift.min()), 4) if len(drift) > 1 else None,
            'residual_sd': round(float(resid.std()), 4) if len(resid) > 1 else None,
            'residual_sd_before_2026': (round(float(resid_pre.std()), 4)
                                        if resid_pre is not None and len(resid_pre) > 1 else None),
            'overlap_h_before_2026': int(len(before)),
            'quantisation_step_a': step_of(a), 'quantisation_step_b': step_of(b_name),
            'drift_sd_mean_filter': round(float(drift_m.std()), 4) if len(drift_m) > 1 else None,
            'residual_sd_mean_filter': round(float(resid_m.std()), 4) if len(resid_m) > 1 else None,
            'residual_daily_mean_sd': round(float(daily.std()), 4) if len(daily) > 1 else None,
            'residual_lag1_autocorrelation': round(lag1, 3) if lag1 is not None else None,
            'drift_window_sensitivity': windows,
            'common_support_h': int(len(common)),
            'mean_common_support': round(mean_c, 6) if mean_c is not None else None,
            'drift_var_common_support': round(var_d, 6) if var_d is not None else None,
            'residual_var_common_support': round(var_e, 6) if var_e is not None else None,
            'cov_drift_residual': round(cov_de, 6) if cov_de is not None else None,
            'second_moment_common_support': round(m2, 6) if m2 is not None else None,
            'identity_error': identity_error,
            'strictly_colocated': bool(dx < 0.01 and abs(float(meta.loc[a].nominal_depth_m)
                                                         - float(meta.loc[b_name].nominal_depth_m)) < 1e-9),
        })

    frame = pd.DataFrame(rows)
    usable = frame.dropna(subset=['drift_sd', 'residual_sd'])
    last = {name: str(s.dropna().index.max()) for name, s in series.items() if s.notna().any()}
    report_out(frame, usable, rows, last)


def budget_for(usable, drift_col='drift_sd', resid_col='residual_sd'):
    # Pooled components, expressed per probe. Var(a - b) = 2 Var(probe) under exchangeability.
    root2 = np.sqrt(2.0)
    n = int(len(usable))
    # Exchangeable probes give the offset a mean of zero, so its standard deviation is the root
    # mean square of b, which does not depend on which probe of a pair is called a.
    u_offset = float(np.sqrt((usable.offset_b ** 2).mean())) / root2
    u_drift = float(np.sqrt((usable[drift_col] ** 2).mean())) / root2
    u_resid = float(np.sqrt((usable[resid_col] ** 2).mean())) / root2
    # Quantisation inside the residual, pair by pair: a WNode reads in 0.1 degC steps and an EXO2
    # in 0.01, so a WNode-EXO2 difference carries less of it than a WNode-WNode one.
    q2 = (usable.quantisation_step_a ** 2 + usable.quantisation_step_b ** 2) / 12.0
    u_quant_pairs = float(np.sqrt(q2.mean() / 2.0))
    # Remove it from the residual to report the short-term scatter beyond resolution.
    u_noise = float(np.sqrt(max(u_resid ** 2 - u_quant_pairs ** 2, 0.0)))
    u_total = float(np.sqrt(u_offset ** 2 + u_drift ** 2 + u_resid ** 2))
    # Effective degrees of freedom (Welch-Satterthwaite), each component counted on its pairs less
    # one. Probes are shared between pairs, so the true figure is lower; the offset alone at n - 1
    # degrees of freedom would need k = t(0.975, n - 1) for 95 % coverage.
    nu = max(n - 1, 1)
    nu_eff = u_total ** 4 / sum(u ** 4 / nu for u in (u_offset, u_drift, u_resid))
    # What the quadrature sum leaves out: the root-mean-square pair difference measured directly on
    # each pair's common support, and the drift-remainder covariance, both per probe.
    moment = usable.dropna(subset=['second_moment_common_support'])
    u_moment = float(np.sqrt(moment.second_moment_common_support.mean() / 2.0))
    cov_term = float((2 * moment.cov_drift_residual).mean() / 2.0)
    identity = float(moment.identity_error.abs().max())

    budget = {
        'pairs_used': n,
        'u_offset': round(u_offset, 4),
        'u_drift': round(u_drift, 4),
        'u_residual': round(u_resid, 4),
        'u_quantisation': round(float(QUANT_U), 4),
        'u_quantisation_pairs': round(u_quant_pairs, 4),
        'u_noise_excl_quantisation': round(u_noise, 4),
        'u_reproducibility_combined': round(u_total, 4),
        'expanded_k2': round(2 * u_total, 4),
        'combination': 'quadrature sum of the three standard deviations; omits the drift-remainder '
                       'covariance and mixes supports (the offset over all hours, the others where '
                       'the rolling window is full)',
        'u_rms_pair_difference_common_support': round(u_moment, 4),
        'expanded_k2_rms': round(2 * u_moment, 4),
        'covariance_term_per_probe': round(cov_term, 5),
        'cov_drift_residual_range_pairs': [round(float(moment.cov_drift_residual.min()), 5),
                                           round(float(moment.cov_drift_residual.max()), 5)],
        'identity_max_abs_error': identity,
        'pairs_with_common_support': int(len(moment)),
        'effective_degrees_of_freedom': round(float(nu_eff), 1),
        'coverage_at_k2': round(float(2 * stats.t.cdf(2.0, nu_eff) - 1), 3),
        'k_for_95': round(float(stats.t.ppf(0.975, nu_eff)), 3),
        'k_for_95_offset_alone': round(float(stats.t.ppf(0.975, nu)), 3),
        'share_of_variance': {
            'offset': round(u_offset ** 2 / u_total ** 2, 3),
            'drift': round(u_drift ** 2 / u_total ** 2, 3),
            'residual': round(u_resid ** 2 / u_total ** 2, 3),
        },
    }
    return budget


def offset_labelling(usable):
    """How the declared offset estimator behaves under the arbitrary labelling of each pair."""
    b = usable.offset_b.to_numpy()
    labelled_sd = float(np.std(b, ddof=1)) / np.sqrt(2.0)
    sds = [float(np.std(b * np.array(s), ddof=1)) / np.sqrt(2.0)
           for s in product((1.0, -1.0), repeat=len(b))]
    # WNode minus EXO2 on the mixed pairs: the family bias the labelled standard deviation removes
    mixed = []
    for _, r in usable.iterrows():
        wa, wb = r.a.startswith('Nodo_'), r.b.startswith('Nodo_')
        if wa != wb:
            mixed.append(r.offset_b if wa else -r.offset_b)
    return {'estimator': 'root mean square of b (standard deviation about zero under exchangeability)',
            'sample_sd_as_labelled_per_probe': round(labelled_sd, 4),
            'sample_sd_over_all_labellings_per_probe': [round(min(sds), 4), round(max(sds), 4)],
            'wnode_minus_exo2_pairs': len(mixed),
            'wnode_minus_exo2_mean_offset': round(float(np.mean(mixed)), 4) if mixed else None}


def failure_indicator(rows, last, column='residual_sd', seed=20260921):
    """POST HOC, not declared. Does short-term disagreement rank the probes that report nothing
    scorable in 2026 above those that survived? Probes share pairs, so the units are not
    independent and the permutation p is optimistic; it is reported as a ranking statistic.
    Every such probe's record ends in 2025, so `residual_sd_before_2026` is the like-for-like
    column; `residual_sd` uses each pair's whole overlap.
    """
    import collections
    missing = json.loads((ROOT / 'reports/bay_twin/runs/v2_missingness.json').read_text())
    # by name, not position: the cohort list of v2_missingness has been relabelled before
    reserved = [c for c in missing['cohorts'] if c.get('cohort') == 'Reserved-2026 temperature']
    dead = set(reserved[0]['sensors_never_scorable'])
    per = collections.defaultdict(list)
    for r in rows:
        if r[column] is None:
            continue
        per[r['a']].append(r[column])
        per[r['b']].append(r[column])
    probes = sorted(per)
    vals = np.array([float(np.mean(per[k])) for k in probes])
    lab = np.array([k in dead for k in probes])
    if lab.sum() == 0 or lab.all():
        return None
    obs = float(vals[lab].mean() - vals[~lab].mean())
    rng = np.random.default_rng(seed)
    draws = np.array([float(vals[s].mean() - vals[~s].mean())
                      for s in (rng.permutation(lab) for _ in range(20000))])
    order = sorted(zip(probes, vals, lab), key=lambda r: -r[1])
    return {'status': 'POST HOC, not declared in addendum 22',
            'column': column,
            'probes': len(probes), 'later_never_scorable': int(lab.sum()),
            'mean_residual_failed': round(float(vals[lab].mean()), 4),
            'mean_residual_surviving': round(float(vals[~lab].mean()), 4),
            'difference': round(obs, 4),
            'permutation_p_one_sided': round(float((draws >= obs).mean()), 4),
            'caveat': 'probes share pairs, so units are not independent and p is optimistic',
            'ranking': [{'probe': k, 'mean_residual_sd': round(float(x), 4),
                         'never_scorable_2026': bool(d), 'last_record': last.get(k)}
                        for k, x, d in order]}


def report_out(frame, usable, rows, last):
    budget = budget_for(usable)
    strict = usable[usable.strictly_colocated]
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum22.json',
              'nature': 'reproducibility between deployments; NOT accuracy against a standard',
              'drift_window': DRIFT_WINDOW,
              'pairs': rows,
              'budget': budget,
              'budget_mean_filter': budget_for(usable, 'drift_sd_mean_filter',
                                               'residual_sd_mean_filter'),
              'budget_strictly_colocated': (budget_for(strict) if len(strict) > 1 else None),
              'strictly_colocated_pairs': int(len(strict)),
              'offset_labelling': offset_labelling(usable),
              'observed_mean_abs_difference': round(float(frame.mean_abs_difference.mean()), 4),
              # Under a normal model with mean zero, a standard deviation s implies a mean absolute
              # difference of s sqrt(2/pi); the quadrature sum and the directly measured RMS differ
              # (covariance, support), so both conversions are reported.
              'implied_mean_abs_difference': round(
                  float(budget['u_reproducibility_combined'] * np.sqrt(2) * np.sqrt(2 / np.pi)), 4),
              'implied_mean_abs_difference_rms': round(
                  float(budget['u_rms_pair_difference_common_support'] * np.sqrt(2) * np.sqrt(2 / np.pi)), 4),
              'implied_pair_difference_sd': round(float(budget['u_reproducibility_combined'] * np.sqrt(2)), 4),
              'exploratory_failure_indicator': failure_indicator(rows, last),
              'exploratory_failure_indicator_before_2026': failure_indicator(
                  rows, last, column='residual_sd_before_2026')}
    # Per probe, as the budget is: what a day of averaging actually leaves of the short-term term.
    daily = usable.residual_daily_mean_sd.dropna()
    report['short_term_after_daily_averaging'] = {
        'u_per_probe': round(float(np.sqrt((daily ** 2).mean()) / np.sqrt(2.0)), 4),
        'white_noise_expectation': round(budget['u_residual'] / np.sqrt(24), 4),
        'median_lag1_autocorrelation': round(float(usable.residual_lag1_autocorrelation.median()), 3)}
    sens = {}
    for w in ('7D', '14D', '60D'):
        d = usable.drift_window_sensitivity.map(lambda x: x[w]['drift_sd'])
        r = usable.drift_window_sensitivity.map(lambda x: x[w]['residual_sd'])
        sens[w] = {'u_drift': round(float(np.sqrt((d.dropna() ** 2).mean()) / np.sqrt(2.0)), 4),
                   'u_residual': round(float(np.sqrt((r.dropna() ** 2).mean()) / np.sqrt(2.0)), 4)}
    sens[DRIFT_WINDOW] = {'u_drift': budget['u_drift'], 'u_residual': budget['u_residual']}
    report['drift_window_sensitivity'] = sens
    print('short-term after daily averaging:', report['short_term_after_daily_averaging'])
    print('drift window sensitivity:', sens)
    OUT.write_text(json.dumps(report, indent=2))

    pd.set_option('display.width', 200)
    print(frame[['a', 'b', 'separation_m', 'depth_separation_m', 'span_days',
                 'mean_abs_difference', 'offset_b', 'drift_sd', 'residual_sd']].to_string(index=False))
    print()
    for label, key in [('declared (rolling median)', 'budget'),
                       ('deviation (rolling mean)', 'budget_mean_filter'),
                       ('strictly co-located only', 'budget_strictly_colocated')]:
        b = report[key]
        if b is None:
            continue
        print(f"  {label}:  n={b['pairs_used']}  offset {b['u_offset']}  drift {b['u_drift']}"
              f"  residual {b['u_residual']}  combined {b['u_reproducibility_combined']}"
              f"  shares {b['share_of_variance']}")
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
