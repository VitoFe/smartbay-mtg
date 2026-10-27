"""Addendum 63: the silent-probe continuation to 72 hours.

Extends the Bridge cohorts of addendum 39 from 48 to 72 h without touching anything addendum 39
wrote: the same episodes (a real anchor and a real observation at every lead from 6 to 48 h),
scored at 54, 60, 66 and 72 h wherever the probe is observed at that lead. The parameter-free arms
are addendum 39's (the last reading, which is the delivered fallback; the nearest reporting probe's
change; the two climatological increments, their tables extended to twelve six-hour steps). The
anchored continuation is a second tree of addendum 39's form, trained on leads 54 to 72 h with its
own nested study of at least 100 completed trials, its inner window ending 72 h before each outer
fold.

Declared in config/bay_reserved_protocol_v2_addendum63.json before anything was computed. Bridge-2026
at these leads is read once, under sampler seed 42; seeds 7 and 123 re-select the temperature tree on
Bridge-Dev only.

Usage:
  python -m src.experiments.v2_bridge72 --variable temperature [--seed 7] [--quick]
  python -m src.experiments.v2_bridge72 --report
"""
import argparse
import json
import zlib

import numpy as np
import pandas as pd

from src.experiments import v2_bridge, v2_horizon
from src.experiments.v2_closed_loop import STEP, episodes, lead_climatology, climatological_value
from src.tuning import protocol_v2
from src.tuning.protocol_v2 import run_study

RUNS = v2_bridge.RUNS
OUT = RUNS / 'v2_bridge72'
PUBLISHED = RUNS / 'v2_bridge'
LEADS = [54, 60, 66, 72]
MAX_LEAD = LEADS[-1]
STEPS = MAX_LEAD // STEP
EPISODE_LEAD = v2_bridge.MAX_LEAD          # the eligibility rule of addendum 39, unchanged (48 h)
SEEDS = (7, 123)
KEY = ['cohort', 'fold', 'sensor', 'onset']


# --------------------------------------------------------------------------------------------------
# Climatology to twelve steps, rows at 54 to 72 h
# --------------------------------------------------------------------------------------------------
def climatology_lookup(data, fold):
    """(i0, s, h) -> the two climatological increments at that lead, in scaled units, to 72 h."""
    month_hour = lead_climatology(data, fold, STEPS)
    hour_only = v2_bridge.hour_climatology(data, fold, STEPS)

    def lookup(i0, s, h):
        k = h // STEP - 1
        x0 = data['x'][i0, s]
        return (climatological_value(month_hour, data, i0, s, k) - x0,
                v2_bridge.hour_value(hour_only, data, i0, s, k) - x0)
    return lookup


def target_rows(ctx, i0, s):
    """Rows (i0, h0, s, h, hT) at 54 to 72 h where the probe is observed inside the period."""
    h0 = ctx.pos[i0]
    out = []
    for h in LEADS:
        hT = h0 + h
        if hT >= len(ctx.index) or i0 + h // STEP - 1 >= len(ctx.data['times']):
            continue
        if np.isfinite(ctx.obs[hT, s]):
            out.append((i0, h0, s, h, hT))
    return out


def onset_rows(ctx, onsets, sensors_mask):
    """Training rows: every grid onset with a real reading, at each later lead that is observed."""
    out = []
    for i0 in onsets:
        if ctx.pos[i0] < 6:
            continue
        for s in np.flatnonzero(sensors_mask):
            if np.isfinite(ctx.obs[ctx.pos[i0], s]):
                out.extend(target_rows(ctx, i0, s))
    return out


def training_rows(ctx, onsets, eligible, first_scored):
    cutoff = ctx.index.get_indexer([first_scored - pd.Timedelta(hours=v2_bridge.GAP_H)])[0]
    return [r for r in onset_rows(ctx, onsets, eligible) if r[4] < cutoff]


def episode_rows(ctx, eps):
    return [r for i0, s in eps for r in target_rows(ctx, i0, s)]


# --------------------------------------------------------------------------------------------------
# Selection, fitting and scoring
# --------------------------------------------------------------------------------------------------
def select_and_fit(ctx, fold, label, folder, quick):
    data = ctx.data
    inner = v2_horizon.inner_fold(data, fold, MAX_LEAD)
    lookup_inner = climatology_lookup(data, inner)
    va_eps = episodes(data, inner, EPISODE_LEAD)
    first_inner = data['times'][min(i for i, _ in va_eps)]
    tr_rows = v2_bridge.subsample(training_rows(ctx, inner['tr'], inner['eligible'], first_inner),
                                  zlib.crc32(f'{label}|inner72'.encode()))
    va_rows = episode_rows(ctx, va_eps)
    if quick:
        tr_rows, va_rows = tr_rows[:4000], va_rows[:2000]
    Xv, yv, av = ctx.features(va_rows, lookup_inner)
    Xt, yt, _ = ctx.features(tr_rows, lookup_inner)
    sensors_t = np.array([r[2] for r in tr_rows])
    counts = np.bincount(sensors_t, minlength=len(ctx.names)).astype(float)
    wt = 1 / np.maximum(counts[sensors_t], 1); wt = wt / wt.mean()
    frame_v = pd.DataFrame({'sensor': [r[2] for r in va_rows], 'lead': [r[3] for r in va_rows],
                            'truth': av + yv, 'anchor': av})

    def objective(trial):
        params = v2_bridge.tree_space(trial)
        model = v2_bridge.make_tree(params)
        model.fit(Xt, yt, sample_weight=wt)
        return v2_bridge.lead_macro_ratio(frame_v.assign(pred=frame_v.anchor + model.predict(Xv)))

    summary = run_study(f'bridge72_{ctx.variable}_{label}', objective, folder / 'studies', quick=quick)
    lookup_outer = climatology_lookup(data, fold)
    first_outer = data['times'][fold['va'].min()]
    rows = v2_bridge.subsample(training_rows(ctx, fold['tr'], fold['eligible'], first_outer),
                               zlib.crc32(f'{label}|outer72'.encode()))
    if quick:
        rows = rows[:4000]
    model = v2_bridge.fit(ctx, rows, lookup_outer, summary['params'])
    return model, lookup_outer, summary, {'inner_training_rows': len(tr_rows), 'inner_validation_rows': len(va_rows),
                                          'outer_training_rows': len(rows),
                                          'inner_window': inner['audit']}


def score_rows(ctx, fold, eps, model, lookup):
    rows = episode_rows(ctx, eps)
    X, y, anchor = ctx.features(rows, lookup)
    pred = anchor + model.predict(X)
    sc, ce, data = ctx.scale, ctx.center, ctx.data
    out = []
    for n, (i0, h0, s, h, hT) in enumerate(rows):
        cm, ch = lookup(i0, s, h)
        x0 = data['x'][i0, s]
        truth_grid = data['truth'][i0 + h // STEP - 1, s]
        assert abs(truth_grid - ctx.obs[hT, s]) < 1e-9 and abs(x0 - ctx.obs[h0, s]) < 1e-9, 'grid and hourly disagree'
        out.append({'fold': fold['number'], 'sensor': ctx.names[s], 'onset': ctx.index[h0], 'lead': h,
                    'truth': ctx.obs[hT, s] * sc + ce, 'persistence': x0 * sc + ce,
                    'clim_month_hour': (x0 + cm) * sc + ce, 'clim_hour': (x0 + ch) * sc + ce,
                    'neighbour_change': v2_bridge.neighbour_change(ctx, i0, s, h) * sc + ce,
                    'anchored_direct': pred[n] * sc + ce})
    return pd.DataFrame(out)


def climatology_gate(ctx, fold, eps, lookup, published, cohort):
    """G2: the twelve-step tables reproduce the published climatology columns at 6 to 48 h."""
    sc, ce, data = ctx.scale, ctx.center, ctx.data
    pub = published[published.cohort == cohort].set_index(['fold', 'sensor', 'onset', 'lead'])
    worst, n = 0.0, 0
    for i0, s in eps:
        x0 = data['x'][i0, s]
        for h in v2_bridge.LEADS:
            key = (fold['number'], ctx.names[s], ctx.index[ctx.pos[i0]], h)
            if key not in pub.index:
                continue
            cm, ch = lookup(i0, s, h)
            row = pub.loc[key]
            worst = max(worst, abs((x0 + cm) * sc + ce - row.clim_month_hour), abs((x0 + ch) * sc + ce - row.clim_hour))
            n += 1
    return {'rows': n, 'max_abs_diff': worst, 'passed': bool(n > 0 and worst < 1e-9)}


def run(variable, seed=42, quick=False):
    seed = int(seed)
    protocol_v2.PROTOCOL['sampler_seed'] = seed
    folder = OUT / ('smoke' if quick else '') / ('' if seed == 42 else f'seed{seed}')
    folder.mkdir(parents=True, exist_ok=True)
    published = pd.read_csv(PUBLISHED / f'bridge_{variable}.csv.gz', parse_dates=['onset'])
    published_keys = {c: set(map(tuple, published[published.cohort == c][['fold', 'sensor', 'onset']]
                                 .drop_duplicates().itertuples(index=False, name=None)))
                      for c in ('Bridge-Dev', 'Bridge-2026')}
    frames, studies, gates = [], {}, {'G1': {}, 'G2': {}}

    ctx = v2_bridge.Context(variable, 'dev')
    closed = pd.read_csv(v2_bridge.CLOSED_LOOP / f'closed_loop_{variable}.csv.gz', usecols=['sensor', 'time'],
                         parse_dates=['time']).drop_duplicates()
    closed = set(zip(closed.sensor, closed.time))
    dev_keys = set()
    for fold in v2_horizon.folds(ctx.data, STEP):
        eps = [(i0, s) for i0, s in episodes(ctx.data, fold, EPISODE_LEAD)
               if (ctx.names[s], ctx.data['times'][i0]) in closed]
        dev_keys |= {(fold['number'], ctx.names[s], ctx.index[ctx.pos[i0]]) for i0, s in eps}
        model, lookup, summary, counts = select_and_fit(ctx, fold, f'fold{fold["number"]}', folder, quick)
        studies[f'fold{fold["number"]}'] = {**summary, **counts}
        gates['G2'][f'Bridge-Dev|fold{fold["number"]}'] = climatology_gate(ctx, fold, eps, lookup, published, 'Bridge-Dev')
        if quick:
            eps = eps[:40]
        frames.append(score_rows(ctx, fold, eps, model, lookup).assign(cohort='Bridge-Dev'))
    gates['G1']['Bridge-Dev'] = {'episodes': len(dev_keys), 'published': len(published_keys['Bridge-Dev']),
                                 'passed': dev_keys == published_keys['Bridge-Dev']}

    if seed == 42:
        rctx = v2_bridge.Context(variable, 'reserved')
        rfold = v2_bridge.fold_2026(rctx.data)
        reps = episodes(rctx.data, rfold, EPISODE_LEAD)
        res_keys = {(rfold['number'], rctx.names[s], rctx.index[rctx.pos[i0]]) for i0, s in reps}
        gates['G1']['Bridge-2026'] = {'episodes': len(res_keys), 'published': len(published_keys['Bridge-2026']),
                                      'passed': res_keys == published_keys['Bridge-2026']}
        model, lookup, summary, counts = select_and_fit(rctx, rfold, 'reserved2026', folder, quick)
        studies['reserved2026'] = {**summary, **counts}
        gates['G2']['Bridge-2026'] = climatology_gate(rctx, rfold, reps, lookup, published, 'Bridge-2026')
        if quick:
            reps = reps[:40]
        reserved = score_rows(rctx, rfold, reps, model, lookup).assign(cohort='Bridge-2026')
        if quick:
            print(f'smoke: Bridge-2026 pipeline produced {len(reserved)} rows (not written)')
        else:
            frames.append(reserved)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(folder / f'bridge72_{variable}.csv.gz', index=False)
    (folder / f'bridge72_{variable}_studies.json').write_text(json.dumps({'studies': studies, 'gates': gates},
                                                                        indent=2, default=str))
    print(variable, 'seed', seed, 'gates', json.dumps({g: {k: v['passed'] for k, v in d.items()} for g, d in gates.items()}))
    print(f'{variable}: {len(frame)} rows written to {folder}')


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def coverage(frame):
    """The band at 54 to 72 h, by v2_bridge.ledger_coverage."""
    return v2_bridge.ledger_coverage(frame, LEADS)

def with_weeks(frame):
    iso = frame.onset.dt.isocalendar()
    return frame.assign(week=iso.year.astype(str) + '-' + iso.week.astype(str))


def cohort_block(variable, name, f):
    arms = ['persistence', 'clim_month_hour', 'clim_hour', 'neighbour_change', 'anchored_direct']
    entry = {'episodes_scored_at_some_lead': int(f.groupby(['fold', 'sensor', 'onset']).ngroups),
             'cells': int(f.groupby(['fold', 'sensor']).ngroups), 'sensors': sorted(f.sensor.unique()),
             'weeks': int(f.week.nunique()), 'by_lead': {}}
    for lead, sub in f.groupby('lead'):
        e = {'n': int(len(sub)), 'mae': {a: round(v2_bridge.macro(sub, a), 6) for a in arms}, 'contrasts': {}}
        pairs = [('anchored_direct', other) for other in arms if other != 'anchored_direct']
        pairs += [('neighbour_change', 'persistence'), ('clim_hour', 'persistence'), ('clim_hour', 'clim_month_hour')]
        for x, y in pairs:
            label = f'bridge72|{variable}|{name}|{lead}|{x}|{y}'
            c = v2_bridge.week_blocks(sub, x, y, label)
            c.update(v2_bridge.sparse_cells(sub, x, y, label))
            if (name, variable) in v2_bridge.PER_SENSOR:
                c['per_sensor'] = v2_bridge.per_sensor(sub, x, y, label)
            e['contrasts'][f'{x} - {y}'] = c
        entry['by_lead'][str(int(lead))] = e
    return entry


def seed_block():
    """R2: the temperature tree on Bridge-Dev under seeds 42, 7 and 123, by lead."""
    out = {}
    frames = {42: OUT / 'bridge72_temperature.csv.gz', **{s: OUT / f'seed{s}' / 'bridge72_temperature.csv.gz' for s in SEEDS}}
    if not all(p.exists() for p in frames.values()):
        return {'missing': [str(p.relative_to(RUNS)) for p in frames.values() if not p.exists()]}
    base = with_weeks(pd.read_csv(frames[42], parse_dates=['onset']))
    base = base[base.cohort == 'Bridge-Dev']
    for seed, path in frames.items():
        f = with_weeks(pd.read_csv(path, parse_dates=['onset']))
        f = f[f.cohort == 'Bridge-Dev']
        joined = base.drop(columns=['anchored_direct']).merge(f[['fold', 'sensor', 'onset', 'lead', 'anchored_direct']],
                                                              on=['fold', 'sensor', 'onset', 'lead'], how='inner')
        assert len(joined) == len(base) == len(f), 'seed frames are not the same rows'
        out[str(seed)] = {}
        for lead, sub in joined.groupby('lead'):
            out[str(seed)][str(int(lead))] = {
                f'anchored_direct - {y}': v2_bridge.week_blocks(sub, 'anchored_direct', y,
                                                                 f'bridge72|seed{seed}|{lead}|ad|{y}')
                for y in ('persistence', 'neighbour_change')}
    # Chapter 4's rule over the three seeds: one sign and every interval excluding zero, a result;
    # one sign and some intervals excluding zero (necessarily that way), a direction; one sign and
    # none, not distinguishable; the seeds disagreeing in sign, not estimable
    verdict = {}
    for lead in LEADS:
        for y in ('persistence', 'neighbour_change'):
            cs = [out[str(s)][str(lead)][f'anchored_direct - {y}'] for s in (42, *SEEDS)]
            signs = {float(np.sign(c['point'])) for c in cs}
            excluding = sum(c['excludes_zero'] for c in cs)
            verdict[f'{lead}|anchored_direct - {y}'] = (
                'not estimable' if len(signs) > 1 else 'result' if excluding == len(cs) else
                'direction' if excluding else 'not distinguishable') + ' (seeds only)'
    return {'by_seed': out, 'verdict': verdict}


def report():
    rep = {'declared_in': 'config/bay_reserved_protocol_v2_addendum63.json', 'leads': LEADS,
           'draws': v2_bridge.DRAWS, 'aggregation': 'frozen (fold, sensor) macro', 'floors': v2_bridge.FLOORS}
    for variable in ('temperature', 'oxygen'):
        frame = with_weeks(pd.read_csv(OUT / f'bridge72_{variable}.csv.gz', parse_dates=['onset']))
        gates = json.loads((OUT / f'bridge72_{variable}_studies.json').read_text())['gates']
        block = {'gates': gates}
        if not all(v['passed'] for d in gates.values() for v in d.values()):
            rep[variable] = block
            print(variable, 'GATE FAILED', gates)
            continue
        for name in ('Bridge-Dev', 'Bridge-2026'):
            block[name] = cohort_block(variable, name, frame[frame.cohort == name])
        block['intervals'] = coverage(frame)
        rep[variable] = block
    rep['seeds_temperature_bridge_dev'] = seed_block()
    (OUT / 'report.json').write_bytes(json.dumps(rep, indent=2, default=str).encode('utf-8'))
    print('wrote', OUT / 'report.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    if a.report:
        report()
    else:
        run(a.variable, seed=a.seed, quick=a.quick)
