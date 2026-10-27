"""Addendum 45, part B: the learned temperature continuation re-tuned under further sampler seeds.

The anchored continuation of addendum 39 beats the parameter-free rule (last reading plus the
nearest reporting probe's change since onset) by 0.016 to 0.048 degC on Bridge-Dev, under sampler
seed 42 only. Whether that increment survives other seeds decides what the next release should
carry: the learned continuation, or the rule with nothing to retrain.

Everything is v2_bridge unchanged: the same episodes, features, row subsamples, nested studies and
scoring. Only the sampler seed differs, and the studies are written under this module's own folder
so no published study is resumed. Bridge-Dev only; 2026 is not read again.

Usage:
  python -m src.experiments.v2_bridge_seed --gate            G1: refit the published seed-42 configurations
  python -m src.experiments.v2_bridge_seed --seed 7 [--quick]
  python -m src.experiments.v2_bridge_seed --report          decision rule, predictions, runs/v2_bridge_seeds.json
Declared in config/bay_reserved_protocol_v2_addendum45.json (part B).
"""
import argparse
import json
import zlib

import numpy as np
import pandas as pd

from src.experiments import v2_bridge, v2_horizon
from src.experiments.v2_closed_loop import STEP, episodes
from src.tuning import protocol_v2

VARIABLE = 'temperature'
PUBLISHED = v2_bridge.RUNS / 'v2_bridge'
OUT = v2_bridge.RUNS / 'v2_bridge_seeds'
SUMMARY = v2_bridge.RUNS / 'v2_bridge_seeds.json'
KEY = ['fold', 'sensor', 'onset', 'lead']
FREE = ['truth', 'persistence', 'neighbour_change', 'clim_hour']
SEEDS = (7, 123)
TOLERANCE = 1e-9


def published_frame():
    frame = pd.read_csv(PUBLISHED / f'bridge_{VARIABLE}.csv.gz', parse_dates=['onset'])
    return frame[frame.cohort == 'Bridge-Dev'].reset_index(drop=True)


def dev_folds():
    """The Bridge-Dev context, folds and exactly the published episodes, as v2_bridge.run builds them."""
    ctx = v2_bridge.Context(VARIABLE, 'dev')
    closed = pd.read_csv(v2_bridge.CLOSED_LOOP / f'closed_loop_{VARIABLE}.csv.gz', usecols=['sensor', 'time'],
                         parse_dates=['time']).drop_duplicates()
    keep = set(zip(closed.sensor, closed.time))
    out = []
    for fold in v2_horizon.folds(ctx.data, STEP):
        eps = [(i0, s) for i0, s in episodes(ctx.data, fold, v2_bridge.MAX_LEAD)
               if (ctx.names[s], ctx.data['times'][i0]) in keep]
        out.append((fold, eps))
    return ctx, out


def gate():
    """G1: the published seed-42 configurations, refitted here, reproduce the published predictions."""
    ctx, folds = dev_folds()
    frames = []
    for fold, eps in folds:
        label = f'fold{fold["number"]}'
        selection = json.loads((PUBLISHED / 'studies' / f'bridge_{VARIABLE}_{label}_selection.json').read_text())
        lookup = v2_bridge.climatology_lookup(ctx.data, fold)
        first = ctx.data['times'][fold['va'].min()]
        rows = v2_bridge.subsample(v2_bridge.training_rows(ctx, fold['tr'], fold['eligible'], first),
                                   zlib.crc32(f'{label}|outer'.encode()))
        model = v2_bridge.fit(ctx, rows, lookup, selection['params'])
        frames.append(v2_bridge.score_rows(ctx, fold, eps, model, lookup))
    joined = pd.concat(frames).merge(published_frame(), on=KEY, how='outer', suffixes=('', '_pub'), indicator=True)
    result = {'unmatched': int((joined._merge != 'both').sum()),
              'anchored_direct_max_abs_diff': float((joined.anchored_direct - joined.anchored_direct_pub).abs().max())}
    result['passed'] = result['unmatched'] == 0 and result['anchored_direct_max_abs_diff'] <= TOLERANCE
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gate_G1.json').write_text(json.dumps(result, indent=2))
    print('G1', result)


def run(seed, quick=False):
    protocol_v2.PROTOCOL['sampler_seed'] = int(seed)
    folder = OUT / (f'smoke/seed{seed}' if quick else f'seed{seed}')
    v2_bridge.OUT = folder                  # select_and_fit writes its studies under v2_bridge.OUT
    ctx, folds = dev_folds()
    frames, studies = [], {}
    for fold, eps in folds:
        if quick:
            eps = eps[:40]
        label = f'fold{fold["number"]}'
        model, lookup, summary, counts = v2_bridge.select_and_fit(ctx, fold, label, quick)
        studies[label] = {**summary, **counts}
        frames.append(v2_bridge.score_rows(ctx, fold, eps, model, lookup).assign(cohort='Bridge-Dev'))
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(folder / f'bridge_{VARIABLE}.csv.gz', index=False)
    (folder / f'bridge_{VARIABLE}_studies.json').write_text(json.dumps(studies, indent=2, default=str))
    print(f'seed {seed}: written to {folder}')


def contrasts(frame):
    """anchored_direct - neighbour_change per lead, on v2_bridge's published bootstrap stream: the
    reported joint-week interval (`ci`) and the within-cell one beside it."""
    iso = frame.onset.dt.isocalendar()
    frame = frame.assign(week=iso.year.astype(str) + '-' + iso.week.astype(str))
    out = {}
    for lead, sub in frame.groupby('lead'):
        c = v2_bridge.week_blocks(sub, 'anchored_direct', 'neighbour_change',
                                  f'{VARIABLE}|Bridge-Dev|{lead}|ad|neighbour_change')
        out[str(int(lead))] = {k: c[k] for k in ('point', 'ci', 'excludes_zero', 'ci_within_cell',
                                                 'excludes_zero_within_cell')}
    return out


def report():
    g1 = json.loads((OUT / 'gate_G1.json').read_text())
    published = published_frame()
    rep = {'declared_in': 'config/bay_reserved_protocol_v2_addendum45.json', 'part': 'B',
           'variable': VARIABLE, 'cohort': 'Bridge-Dev', 'gates': {'G1': g1}, 'seeds': {}}
    frames = {42: published}
    for seed in SEEDS:
        frames[seed] = pd.read_csv(OUT / f'seed{seed}' / f'bridge_{VARIABLE}.csv.gz', parse_dates=['onset'])
        studies = json.loads((OUT / f'seed{seed}' / f'bridge_{VARIABLE}_studies.json').read_text())
        joined = frames[seed].merge(published, on=KEY, how='outer', suffixes=('', '_pub'), indicator=True)
        gap = max(float((joined[c] - joined[c + '_pub']).abs().max()) for c in FREE)
        rep['gates'][f'G2_seed{seed}'] = {'unmatched': int((joined._merge != 'both').sum()),
                                          'free_columns_max_abs_diff': gap,
                                          'passed': bool((joined._merge == 'both').all() and gap <= TOLERANCE)}
        rep['gates'][f'G3_seed{seed}'] = min(s['completed_trials'] for s in studies.values())
    failed = [k for k, g in rep['gates'].items() if isinstance(g, dict) and not g['passed']] + \
             [k for k, n in rep['gates'].items() if k.startswith('G3') and n < 100]
    if failed:
        SUMMARY.write_text(json.dumps(rep, indent=2)); raise SystemExit(f'gates failed, nothing reported: {failed}')
    for seed, frame in frames.items():
        by_lead = contrasts(frame)
        rep['seeds'][str(seed)] = {'by_lead': by_lead,
                                   'starred_leads': sum(v['ci'][1] < 0 for v in by_lead.values()),
                                   'starred_leads_within_cell': sum(v['ci_within_cell'][1] < 0 for v in by_lead.values()),
                                   'mean_over_leads': float(np.mean([v['point'] for v in by_lead.values()]))}
    reference = json.loads((PUBLISHED / 'report.json').read_text())[VARIABLE]['Bridge-Dev']['by_lead']
    rep['seed42_reproduces_published_intervals'] = all(
        rep['seeds']['42']['by_lead'][L][k] == reference[L]['contrasts']['anchored_direct - neighbour_change'][k]
        for L in reference for k in ('ci', 'ci_within_cell'))
    rep['intervals'] = ('ci: onset weeks drawn jointly within each fold, cells fixed (reported); '
                        'ci_within_cell: onset weeks resampled within each (fold, sensor) cell')
    keep = all(rep['seeds'][str(s)]['starred_leads'] >= 4 for s in SEEDS)
    rep['decision'] = 'keep' if keep else 'simplify'
    base = rep['seeds']['42']['mean_over_leads']
    rep['predictions'] = {'B1': keep, 'B2': all(abs(rep['seeds'][str(s)]['mean_over_leads'] - base) <= 0.015
                                                for s in SEEDS)}
    keep_within = all(rep['seeds'][str(s)]['starred_leads_within_cell'] >= 4 for s in SEEDS)
    rep['decision_within_cell'] = 'keep' if keep_within else 'simplify'
    SUMMARY.write_bytes(json.dumps(rep, indent=2).encode('utf-8'))
    print(json.dumps({s: {k: v for k, v in e.items() if k != 'by_lead'} for s, e in rep['seeds'].items()}, indent=1),
          rep['decision'], rep['predictions'], 'seed-42 intervals reproduce:', rep['seed42_reproduces_published_intervals'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int)
    p.add_argument('--gate', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.gate:
        gate()
    elif a.report:
        report()
    else:
        run(a.seed, a.quick)
