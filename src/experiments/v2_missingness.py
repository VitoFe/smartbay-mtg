"""Addendum-21: is the scorable part of the endpoint grid representative of the whole of it?

Only 33.8 per cent of scheduled temperature endpoints and 29.4 per cent of oxygen endpoints are
scorable, and every reported error is computed on that subset. If what goes missing is
systematically harder, the headline figures describe easy conditions.

Three stages. A model of P(scorable) over the scheduled grid; error stratified by whatever that
model finds predictive; and an inverse-probability-weighted (IPW) error that reweights the scorable
subset to resemble the grid. Two model-free corrections sit beside the IPW figure: averaging over
calendar months instead of quarters (month balancing), and post-stratifying each probe's months to
the scheduled grid within every (quarter, probe) cell. Month balancing also gives a short month the
weight of a full one and changes the probe mix month by month; post-stratification corrects only the
availability imbalance within each probe.

The propensity model is a logistic regression (L2, C = 1) on centred and scaled covariates, fitted
out of fold over five stratified folds SHUFFLED with a fixed seed. Folds in calendar order would
leave each held-out block's month dummies without an estimate, and the extrapolated month effect
drives scored endpoints to the probability floor; folds grouped by probe would do the same to the
probe dummies and cannot be formed for two or three oxygen sondes. Shuffled folds are checked
against folds grouped by week-long calendar block, which remove the within-week autocorrelation of
the lagged-availability covariates. Weights are 1/P with P clipped to [0.001, 0.999] and trimmed at
the 1st and 99th percentiles of the scored rows, as addendum 21 declared.

Uncertainty. Every correction is re-estimated, propensity model included, on a block bootstrap
over week-long calendar blocks nested in months and resampled within each month, and checked with
the delete-one-block jackknife. A block drawn twice enters a draw as two copies of its rows, so
inside each draw the propensity is cross-fitted over five shuffled stratified group folds grouped by
the original row: every copy of a row falls in one fold, and no row's held-out probability is fitted
on a copy of itself. The point estimate, which has no copies, uses the shuffled stratified folds
above. `--published-draws` cross-fits each draw as the point estimate does, which puts copies of one
row on both sides of a fold, and reproduces the earlier intervals.

Both cohorts use month, issuance hour, sensor, recent reporting history and the five meteorological
channels, as the declaration lists, and each model is also fitted without the weather. On Dev-2025
(the two outer forecasting folds, persistence error) the channels come from bay_forcing.weather. On
Reserved-2026 they come from bay_reserved_evaluation.load_weather_2026, the reader the reserved
evaluation uses, because bay_forcing.weather stops at 2025 by design and is left as it is; the two
readers give the same causal six-hour means (the 2026 reader also averages duplicated time stamps).
Until 2026-09-27 the reserved model had no weather. Recent history is derived from the frozen
endpoint grid rather than from a fresh read of the archive.

Inverse-probability weighting corrects for missingness explained by the recorded covariates. If an
endpoint is missing for a reason correlated with the error and absent from them, no reweighting
recovers it.

Declared in config/bay_reserved_protocol_v2_addendum21.json.
Usage: python -m src.experiments.v2_missingness
"""
import json
import math
import zlib

import numpy as np
import pandas as pd
from scipy.stats import t as student
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, column_or_1d

from src.data.bay_forcing import weather
from src.data.bay_profile_context import ROOT
from src.experiments.v2_horizon import build, folds

OUT = ROOT / 'reports/bay_twin/runs/v2_missingness.json'
RESERVED = ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'
TRIM = (1, 99)          # addendum 21, stage three: 1st and 99th percentiles
CLIP = 1e-3
SEED = 20260921
FOLDS = 5
DRAWS = 1000
CORRECTIONS = ('ipw', 'month_balanced', 'post_stratified')


def lagged_availability(frame, column, lags=(4, 12)):
    """Fraction of the previous `lag` endpoints at this sensor that had an anchor."""
    out = {}
    frame = frame.sort_values(['sensor', 'time'])
    for lag in lags:
        out[f'avail_prev_{lag}'] = (frame.groupby('sensor')[column]
                                    .transform(lambda s: s.shift(1).rolling(lag, min_periods=1).mean()))
    return pd.DataFrame(out, index=frame.index)


def calendar_block(t):
    """Week-long calendar blocks nested in months: days 1-7, 8-14, 15-21 and 22 to month end."""
    week = np.minimum((t.dt.day.to_numpy() - 1) // 7, 3)
    return t.dt.strftime('%Y-%m').to_numpy() + '-' + week.astype(str)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def shuffled_folds(seed=SEED):
    return StratifiedKFold(FOLDS, shuffle=True, random_state=seed)


class GroupedFolds(StratifiedGroupKFold):
    """scikit-learn's StratifiedGroupKFold with its greedy fold search in plain floats.

    The preamble, the shuffle, the stable sort and the selection rule (lowest mean over classes of
    the standard deviation of class shares across folds, ties within np.isclose to the emptier fold)
    are the library's, computed in the library's order of operations; only the per-group arithmetic
    runs on Python floats instead of five-element arrays, and the test indices come from an array
    lookup instead of a set scan. The library takes seconds per split at these sizes, a thousand
    draws per cohort. `intervals` checks on the first draws of every cohort that the splits equal
    the library's, and records it."""

    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = column_or_1d(np.asarray(y))
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(y_cnt)
        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        np.add.at(y_counts_per_group, (groups_inv, y_inv), 1)
        if self.shuffle:
            perm = np.arange(len(groups_cnt))
            rng.shuffle(perm)
            y_counts_per_group = y_counts_per_group[perm]
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(perm.size)
            groups_inv = inv_perm[groups_inv]
        order = np.argsort(-np.std(y_counts_per_group, axis=1), kind='stable')
        k = self.n_splits
        totals = [float(v) for v in y_cnt]
        per_fold = [[0.0] * n_classes for _ in range(k)]
        counts_of = y_counts_per_group.tolist()
        fold_of = np.empty(len(groups_cnt), dtype=int)
        for g in order.tolist():
            counts = counts_of[g]
            best, low, fewest = None, math.inf, math.inf
            for i in range(k):
                # the library's np.std over folds of the class shares, class by class, then the mean
                # over classes, in the same order of operations
                spread = 0.0
                for c in range(n_classes):
                    shares = [(per_fold[j][c] + (counts[c] if j == i else 0.0)) / totals[c] for j in range(k)]
                    mean = 0.0
                    for v in shares:
                        mean += v
                    mean /= k
                    square = 0.0
                    for v in shares:
                        square += (v - mean) * (v - mean)
                    spread += math.sqrt(square / k)
                score = spread / n_classes
                size = 0.0
                for v in per_fold[i]:
                    size += v
                close = abs(score - low) <= 1e-08 + 1e-05 * abs(low) if math.isfinite(low) else False
                if score < low or (close and size < fewest):
                    best, low, fewest = i, score, size
            for c in range(n_classes):
                per_fold[best][c] += counts[c]
            fold_of[g] = best
        assignment = fold_of[groups_inv]
        for i in range(k):
            yield np.flatnonzero(assignment == i)


def grouped_folds(seed=SEED):
    return GroupedFolds(FOLDS, shuffle=True, random_state=seed)


def fit_missingness(design, scorable, cv=None, groups=None, coefficients=True):
    """Out-of-fold P(scorable), its AUC, and coefficients per standard deviation of each covariate."""
    cv = shuffled_folds() if cv is None else cv
    proba = cross_val_predict(model(), design, scorable, cv=cv, groups=groups,
                              method='predict_proba')[:, 1]
    auc = float(roc_auc_score(scorable, proba))
    if not coefficients:
        return proba, auc, None
    fitted = model().fit(design, scorable)
    coef = fitted[-1].coef_[0]  # on standardised covariates: log-odds per SD, comparable across units
    return proba, auc, dict(sorted(zip(design.columns, coef), key=lambda kv: -abs(kv[1])))


def macro(frame, error='error', weight=None, outer='quarter'):
    f = frame.dropna(subset=[error]).copy()
    f['w'] = 1.0 if weight is None else f[weight]
    cell = (f.assign(num=f[error] * f.w)
             .groupby([outer, 'sensor'])[['num', 'w']].sum())
    per = (cell.num / cell.w).groupby(outer).mean()
    return float(per.mean())


def implied_weights(scored, outer='quarter', weight=None):
    """Per-row weight of the nested macro (mean over outer of the mean over sensors of cell means)."""
    w = pd.Series(1.0, index=scored.index) if weight is None else scored[weight]
    cell = w.groupby([scored[outer], scored.sensor]).transform('sum')
    sensors = scored.groupby(outer).sensor.transform('nunique')
    return w / cell / sensors / scored[outer].nunique()


def kish_fraction(a):
    return float(a.sum() ** 2 / (a ** 2).sum() / len(a))


def ipw(frame, proba):
    """Scored rows with clipped, trimmed inverse-probability weights, and the weight diagnostics."""
    p = np.clip(proba, CLIP, 1 - CLIP)
    scored = frame.assign(p_scorable=p)[frame.scorable.to_numpy() == 1].copy()
    w = 1.0 / scored.p_scorable
    lo, hi = np.percentile(w, TRIM)
    scored['ipw'] = np.clip(w, lo, hi)
    return scored, {'weight_range': [round(float(lo), 2), round(float(hi), 2)],
                    'weight_max_before_trim': round(float(w.max()), 2),
                    'scored_at_probability_floor': int((scored.p_scorable <= CLIP).sum())}


def estimates(frame, design, cv=None, groups=None, coefficients=True):
    """Raw headline and the three corrections, each recomputed from scratch (propensity included)."""
    proba, auc, coef = fit_missingness(design, frame.scorable.to_numpy(), cv=cv, groups=groups,
                                       coefficients=coefficients)
    scored, diag = ipw(frame, proba)
    counts = frame.groupby(['month', 'sensor']).scorable.agg(['size', 'sum'])
    post = (counts['size'] / counts['sum']).replace(np.inf, np.nan).rename('post_weight')
    scored = scored.join(post, on=['month', 'sensor'])
    raw = macro(scored)
    value = {'raw': raw,
             'ipw': macro(scored, weight='ipw'),
             'month_balanced': macro(scored, outer='month'),
             'post_stratified': macro(scored, weight='post_weight')}
    change = {k: 100 * (value[k] - raw) / raw for k in CORRECTIONS}
    return {'value': value, 'change': change, 'auc': auc, 'coef': coef, 'diag': diag, 'scored': scored}


CHECKED_DRAWS = 3


def intervals(frame, design, name, grouped=True):
    """Block bootstrap (within month) and delete-one-block jackknife.

    With `grouped` each draw is cross-fitted over stratified group folds grouped by the original
    row, so the copies a repeated block brings in share a fold; without it, as the point estimate.
    """
    blocks = frame.block.to_numpy()
    months = frame.t.dt.strftime('%Y-%m').to_numpy()
    rows = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    strata = {m: sorted({b for b in rows if b.startswith(m)}) for m in np.unique(months)}
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
    draws = {k: np.empty(DRAWS) for k in CORRECTIONS}
    duplicated, identical = [], []
    for i in range(DRAWS):
        pick = np.concatenate([rows[b] for m in sorted(strata)
                               for b in rng.choice(strata[m], size=len(strata[m]), replace=True)])
        f = frame.iloc[pick].reset_index(drop=True)
        d = design.iloc[pick].reset_index(drop=True)
        duplicated.append(1 - len(np.unique(pick)) / len(pick))
        if grouped:
            cv = grouped_folds()
            if i < CHECKED_DRAWS:
                ours = [t for _, t in cv.split(d, f.scorable.to_numpy(), groups=pick)]
                library = [t for _, t in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
                           .split(d, f.scorable.to_numpy(), groups=pick)]
                identical.append(all(np.array_equal(a, b) for a, b in zip(ours, library)))
            c = estimates(f, d, cv=cv, groups=pick, coefficients=False)['change']
        else:
            c = estimates(f, d, coefficients=False)['change']
        for k in CORRECTIONS:
            draws[k][i] = c[k]
    point = estimates(frame, design, coefficients=False)['change']
    leave = {k: [] for k in CORRECTIONS}
    for b in sorted(rows):
        keep = np.flatnonzero(blocks != b)
        c = estimates(frame.iloc[keep].reset_index(drop=True), design.iloc[keep].reset_index(drop=True),
                      coefficients=False)['change']
        for k in CORRECTIONS:
            leave[k].append(c[k])
    n = len(rows)
    out = {'unit': 'week-long calendar block nested in a month, resampled within its month',
           'blocks': n, 'draws': DRAWS,
           'cross_fitting_in_draws': ('five shuffled stratified group folds grouped by the original row, '
                                      'so every copy of a row falls in one fold' if grouped else
                                      'five shuffled stratified folds over the drawn rows, as the point estimate'),
           'repeated_row_share_in_draws': {'median': round(float(np.median(duplicated)), 4),
                                           'max': round(float(np.max(duplicated)), 4)}}
    if grouped:
        out['folds_equal_library_split'] = {'draws_checked': len(identical), 'identical': bool(all(identical))}
    for k in CORRECTIONS:
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        theta = np.asarray(leave[k])
        se = float(np.sqrt((n - 1) / n * ((theta - theta.mean()) ** 2).sum()))
        half = float(student.ppf(0.975, n - 1)) * se
        out[k] = {'change_pct': round(point[k], 2),
                  'bootstrap_95': [round(float(lo), 2), round(float(hi), 2)],
                  'excludes_zero': bool(lo > 0 or hi < 0),
                  'jackknife_t_95': [round(point[k] - half, 2), round(point[k] + half, 2)],
                  'jackknife_excludes_zero': bool(point[k] - half > 0 or point[k] + half < 0)}
    return out


def prepare(frame, with_weather, weather_source=weather):
    """The in-support frame and its covariate design. Returns (frame, design, weather columns)."""
    frame = frame.copy()
    t = pd.to_datetime(frame.time, utc=True)
    frame['t'] = t
    frame['month'] = t.dt.month
    frame['hour'] = t.dt.hour
    frame['block'] = calendar_block(t)
    frame['scorable'] = frame.error.notna().astype(int)
    frame = frame.join(lagged_availability(frame, 'has_anchor'))

    cols = {}
    for m in sorted(frame.month.unique()):
        cols[f'month_{m}'] = (frame.month == m).astype(float)
    for h in sorted(frame.hour.unique()):
        cols[f'hour_{h:02d}'] = (frame.hour == h).astype(float)
    for s in sorted(frame.sensor.unique()):
        cols[f'sensor_{s[:14]}'] = (frame.sensor == s).astype(float)
    for lag in ('avail_prev_4', 'avail_prev_12'):
        cols[lag] = frame[lag].fillna(frame[lag].mean()).astype(float)
    met_cols = []
    if with_weather:
        met = weather_source(pd.DatetimeIndex(t))
        assert met.notna().any(axis=None), 'no weather reading in this cohort'
        for c in met.columns:
            v = met[c].to_numpy()
            cols[f'met_{c}'] = pd.Series(v, index=frame.index).fillna(np.nanmean(v))
            met_cols.append(f'met_{c}')
    return frame, pd.DataFrame(cols, index=frame.index), met_cols


def analyse(frame, name, with_weather, bootstrap=True, weather_source=weather, grouped=True):
    # Common support. A sensor that is never scorable contributes no rows to reweight, so it lies
    # outside the support of the scorable data and no weighting reaches it. Those sensors are
    # removed from the target population and counted, because an unreachable part of the scheduled
    # grid is a finding rather than something to extrapolate into.
    scorable_any = frame.assign(ok=frame.error.notna()).groupby('sensor').ok.any()
    unreachable = sorted(scorable_any.index[~scorable_any])
    unreachable_rows = int(frame.sensor.isin(unreachable).sum())
    whole = None
    if unreachable:
        wf, wd, _ = prepare(frame, with_weather, weather_source)
        proba, auc, _ = fit_missingness(wd, wf.scorable.to_numpy(), coefficients=False)
        whole = {'missingness_auc': round(auc, 4),
                 'note': 'the same model fitted over the whole scheduled grid, never-scorable sensors '
                         'included; their rows are all unscored, so part of this AUC is the tautology '
                         'that a silent probe is silent'}
    frame, design, met_cols = prepare(frame[~frame.sensor.isin(unreachable)], with_weather, weather_source)
    y = frame.scorable.to_numpy()

    est = estimates(frame, design)
    scored = est['scored']
    ess = {'raw': kish_fraction(implied_weights(scored)),
           'ipw': kish_fraction(implied_weights(scored, weight='ipw')),
           'month_balanced': kish_fraction(implied_weights(scored, outer='month')),
           'post_stratified': kish_fraction(implied_weights(scored, weight='post_weight'))}
    ess_ipw_weights = float(scored.ipw.sum() ** 2 / (scored.ipw ** 2).sum())

    # Cross-fitting checks: folds grouped by calendar block, and folds in calendar order.
    checks = {}
    for label, cv, groups in [('week_grouped_folds', GroupKFold(FOLDS), frame.block.to_numpy()),
                              ('time_ordered_folds', StratifiedKFold(FOLDS), None)]:
        e = estimates(frame, design, cv=cv, groups=groups, coefficients=False)
        floor = e['scored'][e['scored'].p_scorable <= CLIP]
        checks[label] = {'missingness_auc': round(e['auc'], 4),
                         'ipw_change_pct': round(e['change']['ipw'], 2),
                         **e['diag'],
                         'floor_rows_by_month': {str(k): int(v) for k, v in
                                                 floor.month.value_counts().sort_index().items()}}
    without_weather = None
    if met_cols:
        _, auc_nw, _ = fit_missingness(design.drop(columns=met_cols), y, coefficients=False)
        without_weather = round(auc_nw, 4)

    strata = {}
    for key, label in [('hour', 'issuance hour'), ('month', 'month')]:
        g = frame.groupby(key).agg(scheduled=('scorable', 'size'),
                                   scorable_rate=('scorable', 'mean'))
        g['mae'] = scored.groupby(key).error.mean()
        strata[label] = {str(k): {'scheduled': int(v.scheduled),
                                  'scorable_rate': round(float(v.scorable_rate), 3),
                                  'mae': None if pd.isna(v.mae) else round(float(v.mae), 4)}
                         for k, v in g.iterrows()}

    value, change = est['value'], est['change']
    out = {'cohort': name, 'with_weather': with_weather,
           'sensors_never_scorable': unreachable,
           'endpoints_outside_support': unreachable_rows,
           'scheduled': int(len(frame)), 'scorable': int(y.sum()),
           'scorable_rate': round(float(y.mean()), 4),
           'propensity_model': 'logistic regression, L2 C=1, centred and scaled covariates; out of fold '
                               f'over {FOLDS} stratified folds shuffled with seed {SEED}',
           'missingness_auc': round(est['auc'], 4),
           'missingness_auc_without_weather': without_weather,
           'missingness_auc_whole_grid': whole,
           'top_coefficients_standardised': {k: round(float(v), 3)
                                             for k, v in list(est['coef'].items())[:8]},
           'trim_percentiles': list(TRIM),
           **est['diag'],
           'effective_sample_size': round(ess_ipw_weights, 1),
           'ess_fraction': round(ess_ipw_weights / len(scored), 3),
           'ess_fraction_of_estimator': {k: round(v, 3) for k, v in ess.items()},
           'macro_mae_raw': round(value['raw'], 4), 'macro_mae_ipw': round(value['ipw'], 4),
           'macro_mae_month_balanced': round(value['month_balanced'], 4),
           'macro_mae_post_stratified': round(value['post_stratified'], 4),
           'relative_change_pct': round(change['ipw'], 2),
           'month_balanced_change_pct': round(change['month_balanced'], 2),
           'post_stratified_change_pct': round(change['post_stratified'], 2),
           'intervals': intervals(frame, design, name, grouped) if bootstrap else None,
           'cross_fitting_checks': checks,
           'strata': strata}
    return out


def development(variable, horizon=6):
    """Rebuild the scheduled grid for the development folds, scorable or not."""
    data = build(variable, horizon)
    rows = []
    for fold in folds(data, horizon):
        mask = fold['eligible']
        for j, i in enumerate(fold['va']):
            for s, sensor in enumerate(data['names']):
                if not mask[s]:
                    continue
                ok = bool(data['valid'][i, s])
                err = (abs(data['truth'][i, s] - data['x'][i, s]) * data['scale']) if ok else np.nan
                rows.append({'sensor': sensor, 'time': str(data['times'][i]),
                             'quarter': pd.Timestamp(data['times'][i]).quarter,
                             'has_anchor': bool(np.isfinite(data['x'][i, s])),
                             'error': err})
    return pd.DataFrame(rows)


def cohort(job):
    """One cohort of the report: (source, variable, grouped) -> analyse(...)."""
    source, variable, grouped = job
    if source == 'Reserved-2026':
        from src.experiments.bay_reserved_evaluation import load_weather_2026
        e = pd.read_csv(RESERVED)
        sub = e[e.variable == variable][['sensor', 'time', 'quarter', 'has_anchor', 'error']]
        out = analyse(sub, f'Reserved-2026 {variable}', with_weather=True, weather_source=load_weather_2026,
                      grouped=grouped)
    else:
        out = analyse(development(variable), f'Dev-2025 {variable}', with_weather=True, grouped=grouped)
    print('done', out['cohort'], flush=True)
    return out


def main(grouped=True, workers=1):
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum21.json',
              'note': 'IPW corrects only for missingness explained by the recorded covariates',
              'cohorts': []}
    jobs = [(source, variable, grouped) for source in ('Reserved-2026', 'Dev-2025')
            for variable in ('temperature', 'oxygen')]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            report['cohorts'] = list(pool.map(cohort, jobs))
    else:
        report['cohorts'] = [cohort(job) for job in jobs]

    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))
    for c in report['cohorts']:
        print(f"== {c['cohort']}  scheduled {c['scheduled']}  scorable {c['scorable_rate']:.3f}"
              f"  weather={c['with_weather']}")
        print(f"   missingness AUC {c['missingness_auc']} (without weather "
              f"{c['missingness_auc_without_weather']})   top: "
              f"{list(c['top_coefficients_standardised'].items())[:4]}")
        print(f"   outside support: {len(c['sensors_never_scorable'])} sensors, "
              f"{c['endpoints_outside_support']} endpoints")
        print(f"   weights {c['weight_range']} (max before trim {c['weight_max_before_trim']}, "
              f"{c['scored_at_probability_floor']} at the floor)  ESS {c['effective_sample_size']} "
              f"({c['ess_fraction']:.0%})  estimator ESS {c['ess_fraction_of_estimator']}")
        for k in CORRECTIONS:
            iv = c['intervals'][k]
            print(f"   {k:16s} {iv['change_pct']:+.2f}%  boot {iv['bootstrap_95']}  "
                  f"jackknife {iv['jackknife_t_95']}")
        print(f"   checks: {json.dumps(c['cross_fitting_checks'])}")
    print('\nwrote', OUT)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--published-draws', action='store_true',
                   help='cross-fit each bootstrap draw as the point estimate, copies of a row in any fold')
    p.add_argument('--workers', type=int, default=1, help='cohorts run in parallel processes')
    a = p.parse_args()
    main(grouped=not a.published_draws, workers=a.workers)
