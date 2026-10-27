"""Addendum 41: score Chronos-Bolt, zero-shot and fine-tuned, as a continuation and as a gap filler.

Reads the traces the cluster run wrote (runs/v2_chronos_adapt/*.npz) and the inputs they were
computed from, and puts them on the published rows:

  continuation   the addendum-39 episodes (runs/v2_bridge/bridge_<variable>.csv.gz); the prediction
                 at lead h is step h of the trajectory from the onset. Contrasts by lead with
                 v2_bridge.week_blocks, the frozen (fold, sensor) macro, onset weeks drawn jointly
                 within each fold (reported) and resampled within each cell (kept beside). The
                 rollout set against it is v2_bridge.ROLLOUT, the one the bridge report reads.
  gap            the Gap-2026 rows of the published run; the forward and backward trajectories are
                 blended as v2_gap_recovery.chronos_fill blends them. Gate G2: the zero-shot blend
                 reproduces the published chronos_bidir column. Contrasts per gap length and over the
                 12-72 h band with v2_gap_intervals.paired (overlap clusters).

Writes runs/v2_chronos_adapt/contrasts.json with the gates and the registered verdicts.

Usage: python -m scripts.v2_chronos_adapt_contrasts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_intervals import clusters, paired
from src.experiments import v2_bridge

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
RUN = RUNS / 'v2_chronos_adapt'
INP = RUN / 'inputs'
GAP_RUN = RUNS / '20260922T180814Z_v2_gaprecovery' / 'v2_gap_recovery_reserved2026'
VARIABLES = ('temperature', 'oxygen')
METHODS = ('full', 'lora')
LEADS = [str(h) for h in v2_bridge.LEADS]
BAND = (12, 72)
PUBLISHED_BAND = {'temperature': -0.0771, 'oxygen': -1.8197}   # v2_gap_variants.json, Gap-2026 12-72 h


def continuation_frame(variable):
    frame = pd.read_csv(RUNS / 'v2_bridge' / f'bridge_{variable}.csv.gz', parse_dates=['onset'])
    frame['onset'] = pd.to_datetime(frame.onset, utc=True)
    iso = frame.onset.dt.isocalendar()
    frame['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
    inputs = np.load(INP / f'continuation_{variable}.npz', allow_pickle=True)
    key = pd.DataFrame({'cohort': inputs['cohort'], 'fold': inputs['fold'], 'sensor': inputs['sensor'],
                        'onset': pd.to_datetime(inputs['onset'], utc=True)})
    key['episode'] = np.arange(len(key))
    frame = frame.merge(key, on=['cohort', 'fold', 'sensor', 'onset'], how='left', validate='many_to_one')
    assert frame.episode.notna().all()
    step = frame.lead.to_numpy() - 1
    for arm, name in (('zs', 'chronos_zs'), *((m, f'chronos_{m}') for m in METHODS)):
        got = np.load(RUN / f'continuation_{variable}_{arm}.npz')
        trace = np.full((len(key), got['traces'].shape[1]), np.nan)
        trace[got['rows']] = got['traces']
        frame[name] = trace[frame.episode.to_numpy(), step]
    # the rollout of addendum 4, on the development episodes, as v2_bridge.report merges it
    published = pd.read_csv(v2_bridge.ROLLOUT[variable], parse_dates=['time'])
    published = published.rename(columns={'time': 'onset'})[['fold', 'sensor', 'onset', 'lead', 'recursive_tree']]
    published['onset'] = pd.to_datetime(published.onset, utc=True)
    return frame.merge(published, on=['fold', 'sensor', 'onset', 'lead'], how='left')


def continuation_contrasts(frame, variable):
    out = {}
    for cohort in ('Bridge-Dev', 'Bridge-2026'):
        f = frame[frame.cohort == cohort]
        pairs = [('chronos_zs', 'anchored_direct'), ('chronos_zs', 'persistence'), ('chronos_zs', 'clim_hour'),
                 ('chronos_zs', 'neighbour_change')]
        if cohort == 'Bridge-Dev':
            pairs.append(('chronos_zs', 'recursive_tree'))
        else:
            pairs += [(f'chronos_{m}', 'chronos_zs') for m in METHODS] + \
                     [(f'chronos_{m}', 'anchored_direct') for m in METHODS]
        block = {'episodes': int(f.groupby(['fold', 'sensor', 'onset']).ngroups), 'by_lead': {}}
        for lead in v2_bridge.LEADS:
            sub = f[f.lead == lead]
            arms = [a for a in ('chronos_zs', *(f'chronos_{m}' for m in METHODS), 'anchored_direct', 'persistence',
                                'clim_hour', 'neighbour_change', 'recursive_tree') if a in sub and sub[a].notna().all()]
            entry = {'mae': {a: round(v2_bridge.macro(sub, a), 6) for a in arms}, 'contrasts': {}}
            for x, y in pairs:
                if x in arms and y in arms:
                    entry['contrasts'][f'{x} - {y}'] = v2_bridge.week_blocks(sub, x, y, f'a41|{variable}|{cohort}|{lead}|{x}|{y}')
            block['by_lead'][str(lead)] = entry
        out[cohort] = block
    return out


def gap_frame(variable):
    frame = pd.read_csv(GAP_RUN / f'{variable}_chronos_bidir' / 'predictions.csv.gz')
    for arm, name in (('zs', 'chronos_bidir_zs'), *((m, f'chronos_bidir_{m}') for m in METHODS)):
        got = np.load(RUN / f'gap_{variable}_{arm}.npz', allow_pickle=True)
        row_of = {k: i for i, k in enumerate(got['keys'])}
        e = frame.episode.map(row_of).to_numpy()
        k = frame.position.to_numpy(int) - 1
        length = frame.gap_length.to_numpy(int)
        w = (length - k) / (length + 1)
        frame[name] = w * got['fwd'][e, k] + (1 - w) * got['bwd'][e, k]
    return frame


def gap_contrasts(frame, variable):
    g2 = float(np.nanmax(np.abs(frame.chronos_bidir_zs - frame.chronos_bidir)))
    band = frame[(frame.gap_length >= BAND[0]) & (frame.gap_length <= BAND[1])]
    cluster_of = clusters(band)
    # the label v2_gap_variants gives this pair, so the same resamples reproduce its interval too
    reproduced = paired(band, 'chronos_bidir', 'linear', f'Gap-2026|{variable}|12-72|chronos_bidir|linear', cluster_of)
    published = json.loads((RUNS / 'v2_gap_variants.json').read_text())['results']['Gap-2026'][variable]['12-72'][
        'chronos_bidir']['minus_linear']
    same = all(reproduced[k] == published[k] for k in ('delta', 'ci_low', 'ci_high', 'clusters', 'episodes'))
    out = {'G2': {'max_abs_diff': g2, 'passed': bool(g2 <= 1e-4 and frame.chronos_bidir_zs.notna().all())},
           'band_gate': {'published': published, 'reproduced': reproduced,
                         'passed': bool(same and abs(reproduced['delta'] - PUBLISHED_BAND[variable]) < 5e-5)},
           'band_12_72': {}, 'by_gap_length': {}}
    arms = ['chronos_bidir'] + [f'chronos_bidir_{m}' for m in METHODS]
    macro = lambda d, a: float(d.assign(e=(d.truth - d[a]).abs()).groupby('episode').e.mean().mean())
    out['band_12_72']['mae'] = {a: round(macro(band, a), 6) for a in arms + ['linear']}
    out['band_12_72']['chronos_bidir - linear'] = reproduced
    for m in METHODS:
        for ref in ('chronos_bidir', 'linear'):
            out['band_12_72'][f'chronos_bidir_{m} - {ref}'] = paired(band, f'chronos_bidir_{m}', ref,
                                                                    f'a41|{variable}|band|{m}|{ref}', cluster_of)
    full_clusters = clusters(frame)
    for length, part in frame.groupby('gap_length'):
        row = {'mae': {a: round(macro(part, a), 6) for a in arms + ['linear']}}
        for m in METHODS:
            row[f'chronos_bidir_{m} - chronos_bidir'] = paired(part, f'chronos_bidir_{m}', 'chronos_bidir',
                                                               f'a41|{variable}|{int(length)}|{m}', full_clusters)
        out['by_gap_length'][str(int(length))] = row
    return out


def verdicts(r, interval='ci'):
    """The registered predictions, read with the reported joint-week intervals (`ci`) or the
    within-cell ones (`ci_within_cell`)."""
    C, G = r['continuation'], r['gap']
    star_below = lambda c: c is not None and c[interval][1] < 0
    star_above = lambda c: c is not None and c[interval][0] > 0
    excludes = {'ci': 'excludes_zero', 'ci_within_cell': 'excludes_zero_within_cell'}[interval]
    lead = lambda v, co, L, k: C[v][co]['by_lead'][L]['contrasts'].get(k)
    p1 = all(sum(star_above(lead('temperature', co, L, 'chronos_zs - anchored_direct')) for L in LEADS) >= 6
             for co in ('Bridge-Dev', 'Bridge-2026'))
    p2 = all(star_below(lead('oxygen', co, L, 'chronos_zs - persistence')) for co in ('Bridge-Dev', 'Bridge-2026')
             for L in ('6', '12', '18'))
    p3 = all(sum(not lead('oxygen', co, L, 'chronos_zs - clim_hour')[excludes] for L in LEADS) >= 6
             for co in ('Bridge-Dev', 'Bridge-2026'))
    p4 = not any(star_below(lead(v, 'Bridge-2026', L, f'chronos_{m} - chronos_zs'))
                 for v in VARIABLES for m in METHODS for L in LEADS)
    p5 = not any(G[v]['band_12_72'][f'chronos_bidir_{m} - chronos_bidir']['excludes_zero']
                 and G[v]['band_12_72'][f'chronos_bidir_{m} - chronos_bidir']['delta'] < 0 for v in VARIABLES for m in METHODS)
    cont = {m: np.mean([abs(lead('temperature', 'Bridge-2026', L, f'chronos_{m} - chronos_zs')['point']) for L in LEADS])
            for m in METHODS}
    gapd = {m: abs(G['temperature']['band_12_72'][f'chronos_bidir_{m} - chronos_bidir']['delta']) for m in METHODS}
    p6 = bool(cont['lora'] < cont['full'] and gapd['lora'] < gapd['full'])
    return {'P1': bool(p1), 'P2': bool(p2), 'P3': bool(p3), 'P4': bool(p4), 'P5': bool(p5), 'P6': p6}


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum41.json',
              'configurations': json.loads((INP / 'configurations.json').read_text()),
              'inputs_manifest': json.loads((INP / 'manifest.json').read_text()),
              'training': {f'{v}|{m}': {t: {k: x for k, x in info.items() if k != 'curve'}
                                         for t, info in json.loads((RUN / f'train_{v}_{m}.json').read_text())['tasks'].items()}
                           for v in VARIABLES for m in METHODS},
              'continuation': {}, 'gap': {}}
    for variable in VARIABLES:
        report['continuation'][variable] = continuation_contrasts(continuation_frame(variable), variable)
        report['gap'][variable] = gap_contrasts(gap_frame(variable), variable)
    report['gates'] = {'G2': all(report['gap'][v]['G2']['passed'] for v in VARIABLES),
                       'band_reproduced': all(report['gap'][v]['band_gate']['passed'] for v in VARIABLES)}
    report['predictions'] = verdicts(report)
    report['predictions_within_cell'] = verdicts(report, 'ci_within_cell')
    (RUN / 'contrasts.json').write_bytes(json.dumps(report, indent=2, default=float).encode('utf-8'))
    print(json.dumps({'gates': report['gates'], 'predictions': report['predictions']}, indent=2))


if __name__ == '__main__':
    main()
