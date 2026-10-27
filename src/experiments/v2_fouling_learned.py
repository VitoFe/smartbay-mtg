"""Addendum-28: can a tuned model recover fouling state from the signal at all?

Three addenda have scored hand-built statistics. Each null says that quantity carries no fouling
signature; none says that no quantity does, and that is the claim needed before the topic leaves
the manuscript.

So the statistic is learned rather than chosen. A gradient-boosted tree gets twenty features of the
probe's own oxygen and temperature record and has to separate a freshly cleaned probe from one at
least ten days into its interval. Three of those features are new here, and they are the ones a
diffusion barrier should move most directly: a barrier caps how fast a reading can change, so the
distribution of the hourly rate of change, the lag-1 autocorrelation and the ratio of
high-frequency to low-frequency variance all respond to it. Addendum 27 tested only the diurnal
harmonic.

Two things keep this honest. Whole inter-cleaning intervals are held out together, because
splitting by day would let the model memorise an interval's level from its own neighbours and score
that as detection. And the placebo re-runs the same pipeline against 200 circularly shifted
calendars with the selected configuration held fixed, which is generous to the real result, since
that configuration was tuned on the true calendar and on none of the shifted ones.

Declared in config/bay_reserved_protocol_v2_addendum28.json.
Usage: python -m src.experiments.v2_fouling_learned
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data.bay_profile_context import ROOT
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.experiments.v2_biofouling import SALINITY_PSU, read_export
from src.experiments.v2_fouling_response import (WIPED, harmonic, transplanted)
from src.experiments.v2_logbook import NODES, read_logbook
from src.tuning.protocol_v2 import run_study

OUT = ROOT / 'reports/bay_twin/runs/v2_fouling_learned.json'
STUDIES = ROOT / 'reports/bay_twin/runs/v2_fouling_learned'
RECENT_DAYS, STALE_DAYS = 3, 10   # the two classes; the band between is dropped
MAX_DOSE_DAYS = 60                # beyond this is a redeployment gap, not accumulation
PLACEBO_DRAWS = 200
FOLDS = 5
SEED = 20260922
MIN_HOURS = 18
# LightGBM sizes its thread pool from n_jobs alone (OMP_NUM_THREADS does not bound it), so the count
# is passed to every estimator; left unset it takes every core of a shared machine.
THREADS = int(os.environ.get('V2_THREADS', '4'))


def day_features(full, previous):
    """Twenty features of one day, given that day's hourly frame and the three before it."""
    o, t = full.oxygen.to_numpy(float), full.temperature.to_numpy(float)
    if np.isfinite(o).sum() < MIN_HOURS or np.isfinite(t).sum() < MIN_HOURS:
        return None
    sol = solubility(t, np.full(len(t), SALINITY_PSU))
    deficit = sol - o
    ao, po = harmonic(o)
    at, pt = harmonic(t)
    do = np.diff(o[np.isfinite(o)])
    prev = previous.oxygen.to_numpy(float)
    prev = prev[np.isfinite(prev)]
    # A barrier caps how fast a reading can move, so the rate distribution and the smoothness of
    # the series carry it more directly than the level does.
    feat = {
        'oxygen_mean': np.nanmean(o), 'oxygen_sd': np.nanstd(o),
        'oxygen_range': np.nanmax(o) - np.nanmin(o),
        'temperature_mean': np.nanmean(t), 'temperature_sd': np.nanstd(t),
        'deficit_mean': np.nanmean(deficit), 'deficit_sd': np.nanstd(deficit),
        'amp_oxygen': ao, 'amp_temperature': at,
        'amp_ratio': ao / at if at > 0.02 else np.nan,
        'phase_lag': (po - pt + 12) % 24 - 12,
        'rate_p50': np.nanpercentile(np.abs(do), 50) if len(do) > 4 else np.nan,
        'rate_p95': np.nanpercentile(np.abs(do), 95) if len(do) > 4 else np.nan,
        'rate_max': np.nanmax(np.abs(do)) if len(do) else np.nan,
        'autocorr_lag1': autocorr(o, 1), 'autocorr_lag3': autocorr(o, 3),
        'hf_lf_ratio': hf_lf(o),
        'prev3_oxygen_mean': np.nanmean(prev) if len(prev) else np.nan,
        'prev3_oxygen_sd': np.nanstd(prev) if len(prev) else np.nan,
        'oxygen_minus_prev3': np.nanmean(o) - (np.nanmean(prev) if len(prev) else np.nan),
    }
    return feat


def autocorr(x, lag):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < MIN_HOURS:
        return np.nan
    y = x[ok] - x[ok].mean()
    if len(y) <= lag or not np.any(y):
        return np.nan
    denom = float(y @ y)
    return float(y[lag:] @ y[:-lag] / denom) if denom > 0 else np.nan


def hf_lf(x):
    """High-frequency energy over low-frequency energy: a barrier suppresses the numerator."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < MIN_HOURS:
        return np.nan
    y = x[ok] - x[ok].mean()
    power = np.abs(np.fft.rfft(y)) ** 2
    if len(power) < 6:
        return np.nan
    lo, hi = power[1:3].sum(), power[4:].sum()
    return float(hi / lo) if lo > 0 else np.nan


def feature_table(stem):
    """One row per day of a probe's record."""
    frame = read_export(stem)
    days = sorted(set(frame.index.floor('D')))
    rows, index = [], []
    for day in days:
        full = frame.reindex(pd.date_range(day, periods=24, freq='h', tz='UTC'))
        previous = frame[(frame.index >= day - pd.Timedelta(days=3)) & (frame.index < day)]
        feat = day_features(full, previous)
        if feat is not None:
            rows.append(feat)
            index.append(day)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name='day'))


def labelled(table, cleanings):
    """Attach the class label and the interval index used as the CV group."""
    if len(cleanings) < 2:
        return None
    marks = np.sort(np.asarray([c.value for c in cleanings]))
    days = np.asarray([d.value for d in table.index])
    idx = np.searchsorted(marks, days, side='right') - 1
    elapsed = np.where(idx >= 0, (days - marks[np.clip(idx, 0, None)]) / 86400e9, np.nan)
    out = table.assign(days_since=np.where(elapsed <= MAX_DOSE_DAYS, elapsed, np.nan),
                       interval=np.where(idx >= 0, idx, -1))
    out = out.dropna(subset=['days_since'])
    keep = (out.days_since <= RECENT_DAYS) | (out.days_since >= STALE_DAYS)
    out = out[keep].assign(y=(out[keep].days_since <= RECENT_DAYS).astype(int))
    if out.y.nunique() < 2 or out.interval.nunique() < FOLDS:
        return None
    return out


def scored(data, params, columns):
    """Macro AUC over held-out inter-cleaning intervals."""
    import lightgbm as lgb
    x, y, g = data[columns].to_numpy(float), data.y.to_numpy(), data.interval.to_numpy()
    aucs = []
    for tr, va in GroupKFold(n_splits=FOLDS).split(x, y, g):
        if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
            continue
        model = lgb.LGBMClassifier(**params, verbose=-1, random_state=SEED, n_jobs=THREADS)
        model.fit(x[tr], y[tr])
        aucs.append(roc_auc_score(y[va], model.predict_proba(x[va])[:, 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)}


def analyse(stem, cleanings, label, rng, quick=False):
    table = feature_table(stem)
    if not len(table):
        return {'probe': stem, 'role': label, 'error': 'no scorable day'}
    first, last = table.index[0], table.index[-1]
    if label == 'wiped' and cleanings:
        cleanings = transplanted(cleanings, min(cleanings).floor('D'), first)
    inside = [c for c in cleanings if first <= c <= last]
    data = labelled(table, inside)
    if data is None:
        return {'probe': stem, 'role': label, 'error': 'too few labelled intervals'}
    columns = [c for c in table.columns]
    entry = {'probe': stem, 'role': label, 'days_scored': int(len(data)),
             'recent_days': int(data.y.sum()), 'stale_days': int((1 - data.y).sum()),
             'intervals': int(data.interval.nunique()), 'features': len(columns),
             'first': str(first.date()), 'last': str(last.date())}

    STUDIES.mkdir(parents=True, exist_ok=True)
    folder = STUDIES / f'{label}_{stem}'.replace('(', '').replace(')', '')
    folder.mkdir(parents=True, exist_ok=True)
    sel = run_study(f'fouling_{label}_{stem}',
                    lambda t: -scored(data, space(t), columns), folder, quick)
    params = {k: v for k, v in sel['params'].items()}
    entry['selection'] = {'trials': sel['completed_trials'], 'params': params}
    entry['auc'] = round(scored(data, params, columns), 4)

    # The placebo re-labels the same days from a shifted calendar and re-scores with the
    # configuration already selected on the true one.
    span = (last - first).total_seconds() / 86400
    offsets = np.array([(c - first).total_seconds() / 86400 for c in sorted(inside)])
    draws = []
    for _ in range(3 if quick else PLACEBO_DRAWS):
        shifted = [first + pd.Timedelta(days=float(o))
                   for o in np.sort((offsets + rng.uniform(0, span)) % span)]
        fake = labelled(table, shifted)
        if fake is not None:
            draws.append(scored(fake, params, columns))
    if draws:
        entry['placebo'] = {'draws': len(draws), 'mean': round(float(np.mean(draws)), 4),
                            'p97_5': round(float(np.percentile(draws, 97.5)), 4),
                            'p02_5': round(float(np.percentile(draws, 2.5)), 4)}
        entry['above_placebo'] = bool(entry['auc'] > entry['placebo']['p97_5'])
    return entry


def main(quick=False):
    rng = np.random.default_rng(SEED)
    log = read_logbook()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum28.json',
              'recent_days': RECENT_DAYS, 'stale_days': STALE_DAYS,
              'placebo_draws': PLACEBO_DRAWS, 'folds': FOLDS, 'seed': SEED,
              'smoke_only': quick, 'nodes': {}}
    for node, stem in NODES.items():
        cleanings = sorted(log[(log.node == node) & (log.action == 'Pulizia')].time)
        report['nodes'][node] = analyse(stem, cleanings, 'hand_cleaned', rng, quick)
        report['nodes'][f'{node}_wiped_control'] = analyse(WIPED[node], cleanings, 'wiped',
                                                           rng, quick)
    real = [e for k, e in report['nodes'].items() if e.get('role') == 'hand_cleaned' and 'auc' in e]
    controls = [e for e in report['nodes'].values() if e.get('role') == 'wiped' and 'auc' in e]
    report['verdict'] = {
        'rule': 'addendum 28: AUC above the 97.5th percentile of the shifted-calendar placebo at '
                'either hand-cleaned node, with the wiped control inside its own',
        'detects': bool(any(e.get('above_placebo') for e in real)
                        and not any(c.get('above_placebo') for c in controls)),
        'hand_cleaned_auc': {e['probe']: e['auc'] for e in real},
        'wiped_control_auc': {e['probe']: e['auc'] for e in controls},
    }
    report['verdict']['decision'] = (
        'return fouling to the thesis as a learned detector' if report['verdict']['detects']
        else 'reduce biofouling to the minimum the delivered system requires')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['verdict'], indent=2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    main(p.parse_args().quick)
