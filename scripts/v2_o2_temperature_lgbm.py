"""Addendum 55: score the sonde's own temperature as features of the tuned LightGBM oxygen gap filler
and forecaster (the declaration is config/bay_reserved_protocol_v2_addendum55.json; the runs,
scripts/experiments/v2_o2_temperature_lgbm.py through scripts/v2_o2_temperature_lgbm_remote.py).

Gap fillers: each cohort's rows are the published ones (reports/bay_twin/runs/v2_gap_variants) with the
published anchored_gbm_rel column beside them; contrasts use v2_gap_intervals.paired with the cohort
conventions of v2_gap_variants.contrasts, as addendum 54's scorer does. Forecasters: the published tree's
rows on the development folds, error as the benchmark computes it (mean over quarters of the per-sonde
mean), block bootstrap over ISO weeks within quarter, 600 draws, one generator seeded 20260919.

Writes reports/bay_twin/runs/v2_o2_temperature_lgbm.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'companion'))
from scripts.v2_gap_intervals import clusters, paired  # noqa: E402
from scripts.v2_o2_temperature_lgbm_remote import GAP, TREE, gap_dir, tree_dir  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
RESULTS = RUNS / 'v2_o2_temperature_lgbm' / 'results'
#: Addendum 59: the filled temperature reads the Gaussian process, so the rel_T_fill jobs were rerun on
#: addendum 54's rerun fills; where a rerun exists its predictions replace the published arm, refused
#: unless it scores exactly the published rows.
FILL_RESULTS = RUNS / 'v2_o2_temperature_lgbm' / 'results_gp_by_pattern'
OUT = RUNS / 'v2_o2_temperature_lgbm.json'


def arm_results(design, period, arm):
    """The folder an arm's predictions are read from: the addendum-59 rerun for the fill arm."""
    rerun = FILL_RESULTS / f'gap_{design}_{period}_{arm}'
    if arm == 'rel_T_fill' and (rerun / 'predictions.csv.gz').exists():
        keys = ['episode', 'position']
        a = pd.read_csv(RESULTS / f'gap_{design}_{period}_{arm}' / 'predictions.csv.gz', usecols=keys)
        if not pd.read_csv(rerun / 'predictions.csv.gz', usecols=keys).equals(a):
            raise SystemExit(f'{rerun} does not score the rows of the published {arm}')
        return rerun
    return RESULTS / f'gap_{design}_{period}_{arm}'
FLOOR = 1.8
TOLERANCE = 1e-3
COHORTS = {'Gap-Dev': ('strict', 'development', ('1-6', '12-72')),
           'Gap-2026': ('strict', 'reserved', ('1-6', '12-72')),
           'Gap-Long-Dev': ('relaxed', 'development', ('168-1440',)),
           'Gap-Long-2026': ('relaxed', 'reserved', ('168-1440',))}
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'),
         '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
INCUMBENT = {'1-6': 'linear', '12-72': 'chronos_bidir', '168-1440': 'chronos_bidir'}   # v2_gap_variants.INCUMBENT
GAP_ARMS = ('rel_T_obs', 'rel_T_fill')
FC_ARMS = ('tree_T_obs', 'tree_T_fc')


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def gap_frames():
    """cohort -> the published rows with the published reference, its refit (gate) and both arms."""
    out, gate = {}, {}
    for cohort, (design, period, bands) in COHORTS.items():
        pub = pd.concat([pd.read_csv(RUNS / 'v2_gap_variants' / f'{cohort}__oxygen__{b}.csv.gz') for b in bands],
                        ignore_index=True)
        ref = pd.read_csv(gap_dir(design, period) / 'predictions.csv.gz',
                          usecols=['episode', 'position', 'anchored_gbm_rel'])
        f = pub.merge(ref, on=['episode', 'position'], how='left')
        runs = {}
        for arm in ('gate',) + GAP_ARMS:
            r = pd.read_csv(arm_results(design, period, arm) / 'predictions.csv.gz',
                            usecols=['episode', 'position', arm])
            runs[arm] = r
            f = f.merge(r, on=['episode', 'position'], how='left')
        # G1 over every row the published run scored, not only the cohort's reported bands
        full = pd.read_csv(gap_dir(design, period) / 'predictions.csv.gz', usecols=['episode', 'position', 'anchored_gbm_rel'])
        g = full.merge(runs['gate'], on=['episode', 'position'], how='outer')
        gate[f'{design}/{period}'] = {'rows': int(len(g)), 'unmatched': int(g.isna().any(axis=1).sum()),
                                      'max_abs_difference': float((g.gate - g.anchored_gbm_rel).abs().max())}
        out[cohort] = f
    return out, gate


def gaps(frames, reference):
    out = {}
    for cohort, (design, period, bands) in COHORTS.items():
        f = frames[cohort]
        long_cohort = cohort.startswith('Gap-Long')
        by_band = {}
        for band in bands:
            lo, hi = (1, 6) if band == '1-6' else (12, 72) if band == '12-72' else (168, 1440)
            part = f[(f.gap_length >= lo) & (f.gap_length <= hi)].copy()
            if not len(part):
                continue
            cluster_of = clusters(part)
            entry = {'episodes': int(part.episode.nunique()),
                     'mae': {a: macro(part, a) for a in ('linear', 'chronos_bidir', 'anchored_gbm_rel', 'gate') + GAP_ARMS},
                     'contrasts': {}}
            refs = dict.fromkeys((reference, INCUMBENT[band], 'linear'))
            for arm in GAP_ARMS:
                entry['contrasts'][arm] = {f'minus_{r}': paired(part, arm, r, f'a55|{cohort}|{band}|{arm}|{r}', cluster_of)
                                           for r in refs}
                if long_cohort:
                    per_length = {}
                    for length, sub in part.groupby('gap_length'):
                        start = sub.episode.str.rsplit('|', n=2).str[1].str[:7]
                        units = dict(zip(sub.episode, sub.sensor + '|' + start))
                        per_length[str(int(length))] = {
                            'mae': macro(sub, arm), 'mae_reference': macro(sub, reference),
                            **{f'minus_{r}': paired(sub, arm, r, f'a55|{cohort}|{arm}|{r}|{int(length)}', units)
                               for r in refs}}
                    entry['contrasts'][arm]['by_gap_length'] = per_length
            if long_cohort:
                entry['note'] = 'pooled band contrasts of a tiled cohort are descriptive; by_gap_length decides'
            by_band[band] = entry
        out[cohort] = by_band
    return out


def robust(c):
    return bool(c and c.get('excludes_zero') and c['delta'] < 0
                and (c.get('clusters', 0) >= 20 or c.get('jackknife_excludes_zero')))


def decisions(res):
    """D1: rel_T_fill replaces the thesis's oxygen method at a band (1 to 72 h) or a length (past a
    week) only if it beats the incumbent robustly in both cohorts."""
    out = {}
    arm = 'rel_T_fill'
    for band, (dev, new) in PAIRS.items():
        inc = INCUMBENT[band]
        if band not in res[dev] or band not in res[new]:
            continue
        if band == '168-1440':
            for L in res[dev][band]['contrasts'][arm]['by_gap_length']:
                got = [res[c][band]['contrasts'][arm]['by_gap_length'].get(L, {}).get(f'minus_{inc}') for c in (dev, new)]
                out[f'{arm} {L} h'] = {'replaces': all(robust(g) for g in got),
                                       'minus_incumbent': {c: g and g['delta'] for c, g in zip((dev, new), got)}}
        else:
            got = [res[c][band]['contrasts'][arm][f'minus_{inc}'] for c in (dev, new)]
            out[f'{arm} {band}'] = {'replaces': all(robust(g) for g in got),
                                    'minus_incumbent': {c: g and g['delta'] for c, g in zip((dev, new), got)}}
    return out


def forecasts():
    out, gate, coverage = {}, {}, {}
    key = ['fold', 'sensor', 'time', 'mode']
    for lead in sorted(TREE):
        pub = pd.read_csv(tree_dir(lead) / 'selected_predictions.csv.gz')
        f = pub.rename(columns={'prediction': 'tree'})
        for arm in ('gate',) + FC_ARMS:
            r = pd.read_csv(RESULTS / f'horizon_h{lead}_{arm}' / 'selected_predictions.csv.gz')
            f = f.merge(r[key + ['prediction']].rename(columns={'prediction': arm}), on=key, how='left')
            if arm != 'gate':
                coverage[str(lead)] = coverage.get(str(lead), {}) | {arm: json.loads(
                    (RESULTS / f'horizon_h{lead}_{arm}' / 'report.json').read_text(encoding='utf-8'))
                    ['temperature_available_on_scored_rows']}
        gate[str(lead)] = {'rows': int(len(f)), 'unmatched': int(f.gate.isna().sum()),
                           'max_abs_difference': float((f.gate - f.tree).abs().max())}
        out[str(lead)] = f
    return out, gate, coverage


def forecast_contrasts(frames, reference):
    from forecast_bench import macro_arrays, macro_from, with_calendar
    rng = np.random.default_rng(20260919)
    arms = [reference, 'persistence', 'baseline'] + list(FC_ARMS)
    out = {}
    for lead, frame in frames.items():
        f = with_calendar(frame[frame['mode'] == 'forecast'].dropna(subset=arms + ['truth']))
        S, C, quarter = macro_arrays(f, arms)
        point = macro_from(S, C, quarter, np.ones(len(quarter)))
        draws = np.array([macro_from(S, C, quarter, np.bincount(rng.integers(0, len(quarter), len(quarter)),
                                                                   minlength=len(quarter)).astype(float))
                          for _ in range(600)])
        idx = {a: i for i, a in enumerate(arms)}
        entry = {'rows': int(len(f)), 'probes': int(f.sensor.nunique()),
                 'mae': {('tree' if a == reference else 'climatological_increment' if a == 'baseline' else a):
                         round(float(point[idx[a]]), 4) for a in arms}, 'contrasts': {}}
        for arm, ref in (('tree_T_obs', reference), ('tree_T_fc', reference), ('tree_T_fc', 'tree_T_obs')):
            d = draws[:, idx[arm]] - draws[:, idx[ref]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            name = f'{arm} - {"tree" if ref == reference else ref}'
            entry['contrasts'][name] = {'delta': round(float(point[idx[arm]] - point[idx[ref]]), 4),
                                        'ci': [round(float(lo), 4), round(float(hi), 4)],
                                        'excludes_zero': bool(lo > 0 or hi < 0)}
        out[lead] = entry
    return out


def predictions(g, fc):
    """The registered predictions H1-H6, read off the results (H3 is the decisions)."""
    def c(cohort, band, arm, ref):
        return g[cohort][band]['contrasts'][arm][f'minus_{ref}']
    ref = REFERENCE_NAME
    out = {}
    h1 = [c(co, '12-72', 'rel_T_fill', ref) for co in ('Gap-Dev', 'Gap-2026')]
    out['H1'] = not all(x['excludes_zero'] and x['delta'] < 0 for x in h1)
    out['H1_deltas'] = [x['delta'] for x in h1]
    obs, fill = c('Gap-Dev', '12-72', 'rel_T_obs', ref)['delta'], c('Gap-Dev', '12-72', 'rel_T_fill', ref)['delta']
    out['H2'] = obs < fill
    out['H2_deltas'] = {'rel_T_obs': obs, 'rel_T_fill': fill}
    h4 = {L: e['contrasts']['tree_T_obs - tree'] for L, e in fc.items() if int(L) >= 12}
    out['H4'] = any(d['excludes_zero'] and d['delta'] < 0 for d in h4.values())
    out['H4_deltas'] = {L: d['delta'] for L, d in h4.items()}
    gains = {L: (e['contrasts']['tree_T_obs - tree']['delta'], e['contrasts']['tree_T_fc - tree']['delta'])
             for L, e in fc.items() if e['contrasts']['tree_T_obs - tree']['delta'] < 0}
    out['H5'] = all(fcd > obsd / 2 for obsd, fcd in gains.values())
    out['H5_leads_with_an_observed_gain'] = {L: {'obs': o, 'fc': f} for L, (o, f) in gains.items()}
    beyond = []
    for co, bands in g.items():
        for band, e in bands.items():
            for arm in GAP_ARMS:
                d = e['contrasts'][arm][f'minus_{ref}']
                if d and abs(d['delta']) >= FLOOR:
                    beyond.append(f'{co} {band} {arm} ({d["delta"]:+.2f})')
                for L, row in e['contrasts'][arm].get('by_gap_length', {}).items():
                    d = row.get(f'minus_{ref}')
                    if d and abs(d['delta']) >= FLOOR:
                        beyond.append(f'{co} {L} h {arm} ({d["delta"]:+.2f}, {d["clusters"]} units'
                                      f'{"" if d.get("estimable", True) else ", not estimable"})')
    for L, e in fc.items():
        for name, d in e['contrasts'].items():
            if name.endswith('- tree') and abs(d['delta']) >= FLOOR:
                beyond.append(f'{L} h {name} ({d["delta"]:+.2f})')
    out['H6'] = not beyond
    out['H6_beyond_floor'] = beyond
    return out


REFERENCE_NAME = 'anchored_gbm_rel'


def main():
    global REFERENCE_NAME
    frames, gap_gate = gap_frames()
    g1 = all(v['max_abs_difference'] <= TOLERANCE and not v['unmatched'] for v in gap_gate.values())
    REFERENCE_NAME = 'anchored_gbm_rel' if g1 else 'gate'
    g = gaps(frames, REFERENCE_NAME)
    fc_frames, fc_gate, coverage = forecasts()
    g2 = all(v['max_abs_difference'] <= TOLERANCE and not v['unmatched'] for v in fc_gate.values())
    fc = forecast_contrasts(fc_frames, 'tree' if g2 else 'gate')
    d = decisions(g)
    fill_rerun = (FILL_RESULTS / 'tfill_extra.json').exists()
    tfill = json.loads(((FILL_RESULTS if fill_rerun else RESULTS) / 'tfill_extra.json')
                       .read_text(encoding='utf-8'))
    report = {'addendum': 55, 'declaration': 'config/bay_reserved_protocol_v2_addendum55.json',
              'gates': {'G1': {**gap_gate, 'passed': g1},
                        'G2': {**fc_gate, 'passed': g2},
                        'G3': {'what': "the filled temperature is addendum 54's (its gate G2 hid the sonde's temperature "
                                       'inside the gap and changed nothing), with the episodes it did not cover filled '
                                       'here by the same functions; each forecast temperature is the precompute\'s '
                                       'c2_net row whose issuance is the oxygen row\'s own (temperature_block)',
                               'filled_here': tfill, 'forecast_temperature_available_on_scored_rows': coverage,
                               'passed': True}},
              'reference': {'gaps': REFERENCE_NAME, 'forecasts': 'tree' if g2 else 'gate'},
              'gaps': g, 'forecasts': fc, 'decisions': d,
              'rel_T_fill_results': (FILL_RESULTS if fill_rerun else RESULTS).relative_to(RUNS).as_posix(),
              'rel_T_fill_note': ('rerun on the temperature filled with the corrected Gaussian process '
                                  '(config/bay_reserved_protocol_v2_addendum59.json)' if fill_rerun
                                  else 'the published run')}
    report['predictions'] = predictions(g, fc)
    # H3: no temperature arm beats Chronos-Bolt from both edges at 12 to 72 h in both cohorts
    beats = {arm: [g[co]['12-72']['contrasts'][arm]['minus_chronos_bidir'] for co in PAIRS['12-72']] for arm in GAP_ARMS}
    report['predictions']['H3'] = not any(all(x['excludes_zero'] and x['delta'] < 0 for x in v) for v in beats.values())         and not any(v['replaces'] for v in d.values())
    report['predictions']['H3_deltas'] = {arm: [x['delta'] for x in v] for arm, v in beats.items()}
    OUT.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))
    print('gates', {k: v['passed'] for k, v in report['gates'].items()})
    print('predictions', {k: v for k, v in report['predictions'].items() if not k.startswith(('H1_', 'H2_', 'H3_', 'H4_', 'H5_', 'H6_'))})
    print('replacements', [k for k, v in d.items() if v['replaces']])


if __name__ == '__main__':
    main()
