"""Addendum 56: a clean-state reference model for biofouling (Vito's idea, 2026-09-29).

Addenda 25, 27 and 28 read the probe's own oxygen, and addendum 27 named why that cannot work: a
fouled optical window and genuinely undersaturated water look the same. This supplies the missing
reference. A model fitted only on freshly cleaned hours (6 to 72 h after a logbook cleaning), from
inputs that do not pass through the optical window (the probe's own thermistor and the solubility it
implies, the Santa Teresa weather, the Magra level, the clock, and in arm B the other hand-cleaned
probe), predicts what a clean probe would read. Fouling should then appear as a residual that
drifts, or grows, with days since cleaning. Days are compared within the same interval, so the
reference model's interval-level errors cancel.

Whole intervals are held out (GroupKFold), so no interval is scored by a model trained on any of its
hours. The placebo reruns the pipeline on 200 circularly shifted calendars with each fold's selected
configuration held fixed; the negative control is the wiped second generation at the same site with
the calendar transplanted onto it, before 2026.

Declared in config/bay_reserved_protocol_v2_addendum56.json.
Usage: python -m src.experiments.v2_fouling_reference [--quick] [--workers N]
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data.bay_profile_context import ROOT
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as solubility
from src.experiments.v2_biofouling import PAIRS, SALINITY_PSU, read_export
from src.experiments.v2_fouling_response import fake_calendars, transplanted
from src.experiments.v2_logbook import NODES, deployed_spans, read_logbook
from src.tuning.protocol_v2 import run_study

OUT = ROOT / 'reports/bay_twin/runs/v2_fouling_reference.json'
FOLDER = ROOT / 'reports/bay_twin/runs/v2_fouling_reference'
CLEAN_H = (6, 72)            # the clean window, hours after a cleaning
MAX_DOSE_DAYS = 60           # beyond this is a redeployment gap, as in addenda 27-28
FRESH_DAYS, STALE_DAYS = 3, 5
MIN_HOURS = 18
FOLDS, INNER = 5, 3
PLACEBO_DRAWS = 200
SEED = 20260922
CUTOFF = pd.Timestamp('2026-01-01', tz='UTC')    # the wiped control never reads 2026
#: amendment 1: node5's hand-cleaned record is also scored without every hour from this cleaning on
TRIM = pd.Timestamp('2023-04-15', tz='UTC')
THREADS = int(os.environ.get('V2_THREADS', '1'))
WARM = (5, 6, 7, 8, 9, 10)
#: node -> (Gen. 1 export, wiped Gen. 2 export at the same site and depth, the pair's other node)
SITES = {'node1': (NODES['node1'], PAIRS['Baia S. Teresa 1 m'][1], 'node5'),
         'node5': (NODES['node5'], PAIRS['Mitili Est 5 m'][1], 'node1')}


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def features(frame, exo, other=None):
    """Inputs that do not pass through the target's optical window (arm A), plus the other probe (B)."""
    t = frame.temperature
    doy = frame.index.dayofyear.to_numpy()
    hour = frame.index.hour.to_numpy()
    x = pd.DataFrame({
        'temp': t, 'temp_24': t.rolling(24, min_periods=12).mean(), 'temp_d6': t - t.shift(6),
        'solubility': solubility(t.to_numpy(float), np.full(len(t), SALINITY_PSU)),
        'solar': exo.solar, 'solar_24': exo.solar.rolling(24, min_periods=12).mean(),
        'air': exo.air, 'air_24': exo.air.rolling(24, min_periods=12).mean(),
        'wind': exo.wind, 'wind_24': exo.wind.rolling(24, min_periods=12).mean(),
        'pressure': exo.pressure, 'river_24': exo.river.rolling(24, min_periods=12).mean(),
        'hour_s': np.sin(2 * np.pi * hour / 24), 'hour_c': np.cos(2 * np.pi * hour / 24),
        'doy_s': np.sin(2 * np.pi * doy / 365.25), 'doy_c': np.cos(2 * np.pi * doy / 365.25),
    }, index=frame.index)
    if other is not None:
        o = other.oxygen.reindex(frame.index)
        x['other_o2'] = o
        x['other_o2_24'] = o.rolling(24, min_periods=12).mean()
    return x


def calendar(index, cleanings, deployed):
    """Interval id and hours since the most recent cleaning for every hour; -1 where not scored."""
    marks = np.sort(np.asarray([c.value for c in cleanings], dtype='int64'))
    t = index.as_unit('ns').asi8
    idx = np.searchsorted(marks, t, side='right') - 1
    hours = np.where(idx >= 0, (t - marks[np.clip(idx, 0, None)]) / 3.6e12, np.nan)
    ok = (idx >= 0) & (hours <= MAX_DOSE_DAYS * 24) & deployed
    return np.where(ok, idx, -1), np.where(ok, hours, np.nan)


def fit(params, x, y):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective='regression_l1', **params, subsample_freq=1, n_jobs=THREADS,
                              random_state=SEED, verbose=-1)
    model.fit(x, y)
    return model


def space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)}


def outer_folds(y, interval, hours):
    """(train, test) masks per outer fold: the clean hours of the training intervals, and every
    scored hour of the held-out ones. Folds with too little to train on are left out."""
    scored = (interval >= 0) & np.isfinite(y)
    clean = scored & (hours >= CLEAN_H[0]) & (hours <= CLEAN_H[1])
    groups = np.unique(interval[scored])
    out = {}
    for k, (tr_g, te_g) in enumerate(GroupKFold(n_splits=FOLDS).split(groups, groups=groups)):
        train = clean & np.isin(interval, groups[tr_g])
        test = scored & np.isin(interval, groups[te_g])
        if train.sum() >= 50 and test.any():
            out[k] = (train, test)
    return out


def tune_fold(x, y, interval, train, name, folder, quick):
    """One study for one outer fold, selection on inner held-out intervals' clean hours."""
    rows = np.flatnonzero(train)

    def objective(trial):
        p = space(trial)
        errs = []
        for a, b in GroupKFold(n_splits=INNER).split(rows, groups=interval[rows]):
            m = fit(p, x[rows[a]], y[rows[a]])
            errs.append(float(np.mean(np.abs(m.predict(x[rows[b]]) - y[rows[b]]))))
        return float(np.mean(errs))
    sel = run_study(name, objective, folder, quick)
    return {'params': dict(sel['params']), 'trials': sel['completed_trials']}


def residuals(x, y, interval, hours, params_by_fold):
    """Out-of-interval residuals at every scored hour, each fold with its selected configuration."""
    res = np.full(len(y), np.nan)
    for k, (train, test) in outer_folds(y, interval, hours).items():
        model = fit(params_by_fold[k]['params'], x[train], y[train])
        res[test] = y[test] - model.predict(x[test])
    return res


def daily(index, res, interval, hours):
    d = pd.DataFrame({'day': index.floor('D'), 'interval': interval, 'r': res, 'dose': hours / 24})
    d = d[(d.interval >= 0) & np.isfinite(d.r)]
    d['abs_r'] = d.r.abs()
    g = d.groupby(['interval', 'day'])
    out = pd.DataFrame({'r': g.r.mean(), 'abs_r': g.abs_r.mean(),
                        'dose': g.dose.mean(), 'n': g.r.size()}).reset_index()
    out = out[out.n >= MIN_HOURS]
    out['warm'] = out.day.dt.month.isin(WARM)
    return out


def within_slope(table, column):
    """Slope on days since cleaning with both demeaned within interval."""
    t = table[table.groupby('interval').dose.transform('size') >= 2]
    if len(t) < 10:
        return None
    x = t.dose - t.groupby('interval').dose.transform('mean')
    y = t[column] - t.groupby('interval')[column].transform('mean')
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 0 else None


def auc(table):
    t = table[(table.dose < FRESH_DAYS) | (table.dose >= STALE_DAYS)]
    label = (t.dose >= STALE_DAYS).astype(int)
    if label.nunique() < 2:
        return None
    return float(roc_auc_score(label, t.abs_r))


def statistics(table):
    out = {}
    for season, part in (('warm', table[table.warm]), ('cold', table[~table.warm]), ('all', table)):
        out[season] = {'S1_drift': within_slope(part, 'r'), 'S2_growth': within_slope(part, 'abs_r'),
                       'S3_auc': auc(part), 'days': int(len(part)), 'intervals': int(part.interval.nunique())}
    return out


def per_interval(table):
    """Amendment 1: each interval's own slope of the daily mean residual on days since cleaning."""
    out = []
    for k, g in table.groupby('interval'):
        if len(g) < 3 or g.dose.nunique() < 2:
            continue
        x = g.dose - g.dose.mean()
        out.append({'interval': int(k), 'start': str(g.day.min().date()), 'warm': bool(g.warm.mean() >= 0.5),
                    'days': int(len(g)), 'slope': round(float((x * (g.r - g.r.mean())).sum() / (x * x).sum()), 4),
                    'mean_residual': round(float(g.r.mean()), 3)})
    return out


def by_dose(table):
    """Mean residual and absolute residual by whole day since cleaning, per season (descriptive)."""
    t = table.assign(day_n=np.floor(table.dose).astype(int))
    return {season: {str(k): {'r': round(float(g.r.mean()), 3), 'abs_r': round(float(g.abs_r.mean()), 3),
                              'days': int(len(g))}
                     for k, g in part.groupby('day_n') if len(g) >= 3}
            for season, part in (('warm', t[t.warm]), ('cold', t[~t.warm]))}


def exogenous():
    """The weather station and the Magra, hourly, once for every job (the 10-minute station file is
    large, and parsed in every worker at once it exhausts memory)."""
    from src.experiments.v2_gap_recovery import exogenous_panel
    index = pd.date_range('2021-01-01', '2025-12-31 23:00', freq='h', tz='UTC')
    return exogenous_panel(index, 'development')


def prepare(node, arm, role, exo_all):
    """The record, its features, the calendar and the scoring mask of one (node, arm, role)."""
    log_rows = read_logbook()
    gen1, gen2, other_node = SITES[node]
    cleanings = sorted(log_rows[(log_rows.node == node) & (log_rows.action == 'Pulizia')].time)
    if role in ('hand-cleaned', 'hand-cleaned-trimmed'):
        frame = read_export(gen1)
        if role == 'hand-cleaned-trimmed':          # amendment 1: node5 without its last weeks
            frame = frame[frame.index < TRIM]
        other = read_export(SITES[other_node][0]) if arm == 'B' else None
        spans = deployed_spans(log_rows, node)
        deployed = np.zeros(len(frame), bool)
        for a, b in spans:
            deployed |= (frame.index >= a) & (frame.index <= b)
        cal = cleanings
    else:
        frame = read_export(gen2)
        frame = frame[frame.index < CUTOFF]
        other = read_export(SITES[other_node][1]) if arm == 'B' else None
        deployed = np.ones(len(frame), bool)
        cal = [c for c in transplanted(cleanings, min(cleanings).floor('D'), frame.index[0].floor('D'))
               if c < CUTOFF]
    exo = exo_all.reindex(frame.index)
    x = features(frame, exo, other).to_numpy(float)
    y = frame.oxygen.to_numpy(float)
    probe = gen2 if role == 'wiped' else gen1
    return frame.index, x, y, cal, deployed, probe


def base_folder(quick):
    return FOLDER / 'smoke' if quick else FOLDER      # a smoke run never seeds the real studies


def study_task(args):
    """The study of one outer fold of one (node, arm, role), on the real calendar."""
    node, arm, role, k, quick, exo_all = args
    os.environ['OMP_NUM_THREADS'] = '1'
    index, x, y, cal, deployed, _ = prepare(node, arm, role, exo_all)
    interval, hours = calendar(index, cal, deployed)
    folds = outer_folds(y, interval, hours)
    if k not in folds:
        return (node, arm, role), k, None
    t0 = time.time()
    sel = tune_fold(x, y, interval, folds[k][0], f'fouling_ref_{node}_{arm}_{role}_f{k}',
                    base_folder(quick) / 'studies', quick)
    log(f'{node} {arm} {role} fold {k}: study done in {time.time() - t0:.0f} s')
    return (node, arm, role), k, sel


def job(args):
    """One (node, arm, role) scored on the real calendar with each fold's selected configuration."""
    node, arm, role, quick, exo_all, chosen = args
    os.environ['OMP_NUM_THREADS'] = '1'
    index, x, y, cal, deployed, probe = prepare(node, arm, role, exo_all)
    interval, hours = calendar(index, cal, deployed)
    tag = f'{node}_{arm}_{role}'
    base = base_folder(quick)
    t0 = time.time()
    res = residuals(x, y, interval, hours, chosen)
    table = daily(index, res, interval, hours)
    base.mkdir(parents=True, exist_ok=True)
    table.to_csv(base / f'daily_{tag}.csv.gz', index=False)
    scored = (interval >= 0) & np.isfinite(res)
    clean = scored & (hours >= CLEAN_H[0]) & (hours <= CLEAN_H[1])
    entry = {'node': node, 'arm': arm, 'role': role, 'probe': probe,
             'first': str(index[0].date()), 'last': str(index[-1].date()),
             'cleanings': len(cal), 'intervals_scored': int(np.unique(interval[scored]).size),
             'hours_scored': int(scored.sum()), 'selections': chosen,
             'clean_window_mae': round(float(np.mean(np.abs(res[clean]))), 3),
             'mae_all_hours': round(float(np.mean(np.abs(res[scored]))), 3),
             'statistics': statistics(table), 'by_dose': by_dose(table),
             'per_interval': per_interval(table)}
    log(f'{tag}: scored in {time.time() - t0:.0f} s')
    return entry


def placebo_chunk(args):
    """Draws [start, stop) of the circularly shifted calendars (the same 200 in every chunk, since
    they come from one generator seeded identically), each fold's configuration fixed."""
    node, arm, role, exo_all, chosen, start, stop, total = args
    os.environ['OMP_NUM_THREADS'] = '1'
    index, x, y, cal, deployed, _ = prepare(node, arm, role, exo_all)
    calendars = fake_calendars(cal, index[0], index[-1], np.random.default_rng(SEED), draws=total)
    out = []
    for fake in calendars[start:stop]:
        fi, fh = calendar(index, fake, deployed)
        fr = residuals(x, y, fi, fh, chosen)
        out.append(statistics(daily(index, fr, fi, fh)))
    return (node, arm, role), out


def placebo_summary(real, draws):
    out = {'draws': len(draws)}
    for season in ('warm', 'cold', 'all'):
        out[season] = {}
        for stat, sided in (('S1_drift', 'two'), ('S2_growth', 'greater'), ('S3_auc', 'greater')):
            obs = real[season][stat]
            null = np.array([d[season][stat] for d in draws if d[season][stat] is not None], float)
            if obs is None or not len(null):
                out[season][stat] = None
                continue
            p = (np.mean(np.abs(null) >= abs(obs)) if sided == 'two' else np.mean(null >= obs))
            out[season][stat] = {'observed': round(obs, 5), 'p': round(float(p), 4),
                                 'null_p2_5': round(float(np.percentile(null, 2.5)), 5),
                                 'null_p97_5': round(float(np.percentile(null, 97.5)), 5)}
    return out


def decisions(entries):
    """D1-D3 of the declaration, per node."""
    out = {}
    for node in SITES:
        verdict = {}
        for arm in ('A', 'B'):
            hand = next(e for e in entries if (e['node'], e['arm'], e['role']) == (node, arm, 'hand-cleaned'))
            wiped = next(e for e in entries if (e['node'], e['arm'], e['role']) == (node, arm, 'wiped'))
            passed = []
            for stat in ('S1_drift', 'S2_growth'):
                h, w = hand['placebo']['warm'][stat], wiped['placebo']['warm'][stat]
                if h and h['p'] < 0.025:
                    if w and w['p'] < 0.05:
                        passed.append(f'{stat}: artefact (the wiped control passes too)')
                    else:
                        passed.append(stat)
            s3 = hand['placebo']['warm']['S3_auc']
            verdict[arm] = {'signature': [p for p in passed if 'artefact' not in p],
                            'artefacts': [p for p in passed if 'artefact' in p],
                            'detector': bool(s3 and s3['p'] < 0.025 and any('artefact' not in p for p in passed))}
        out[node] = verdict
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    exo_all = exogenous()
    keys = [(node, arm, role) for node in SITES for arm in ('A', 'B') for role in ('hand-cleaned', 'wiped')]
    keys += [('node5', arm, 'hand-cleaned-trimmed') for arm in ('A', 'B')]     # amendment 1
    total = 3 if a.quick else PLACEBO_DRAWS
    chunk = 1 if a.quick else 10
    with mp.get_context('spawn').Pool(a.workers) as pool:
        # one study per outer fold, all in parallel; then each (node, arm, role) scored with them
        chosen = {key: {} for key in keys}
        for key, k, sel in pool.imap_unordered(study_task, [(*key, k, a.quick, exo_all)
                                                            for key in keys for k in range(FOLDS)]):
            if sel is not None:
                chosen[key][k] = sel
        log('studies done; scoring the real calendars')
        entries = pool.map(job, [(*key, a.quick, exo_all, chosen[key]) for key in keys])
        log('real calendars scored; placebo draws next')
        tasks = [(e['node'], e['arm'], e['role'], exo_all, e['selections'], s, min(s + chunk, total), total)
                 for e in entries for s in range(0, total, chunk)]
        draws = {}
        for key, part in pool.imap_unordered(placebo_chunk, tasks):
            draws.setdefault(key, []).extend(part)
            done = sum(len(v) for v in draws.values())
            if done % 100 == 0:
                log(f'placebo draws done: {done} of {total * len(entries)}')
    for e in entries:
        e['placebo'] = placebo_summary(e['statistics'], draws[(e['node'], e['arm'], e['role'])])
    report = {'addendum': 56, 'declaration': 'config/bay_reserved_protocol_v2_addendum56.json',
              'amendment': 'config/bay_reserved_protocol_v2_addendum56_amendment1.json',
              'smoke_only': a.quick, 'entries': entries, 'decisions': decisions(entries),
              'note': 'decisions read the full records as declared; the node5 hand-cleaned-trimmed rows '
                      'are amendment 1\'s sensitivity and do not enter them'}
    out = OUT if not a.quick else OUT.with_name('v2_fouling_reference_smoke.json')
    out.write_bytes((json.dumps(report, indent=1, default=str) + '\n').encode('utf-8'))
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
