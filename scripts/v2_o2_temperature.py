"""Addendum 54: score the sonde's own temperature as a covariate for oxygen (the declaration is
config/bay_reserved_protocol_v2_addendum54.json; the runs, scripts/experiments/v2_o2_temperature.py
through scripts/v2_o2_temperature_remote.py).

Gap fillers: each cohort's rows are the published ones (reports/bay_twin/runs/v2_gap_variants), the
references their published columns; contrasts use v2_gap_intervals.paired with the cohort conventions of
v2_gap_variants.contrasts (overlap clusters on 1 to 72 h; per length on probe x calendar month on the
tiled long cohorts, whose pooled band is descriptive only). Forecasters: the benchmark's rows and error
(companion/forecast_bench.py), block bootstrap with one generator seeded 20260919 whose draws every
contrast shares.

Writes reports/bay_twin/runs/v2_o2_temperature.json.
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

RUNS = ROOT / 'reports/bay_twin/runs'
RESULTS = RUNS / 'v2_o2_temperature' / 'results'
#: Addendum 59: the gaps stage rerun with the corrected Gaussian process (the temperature fill reads
#: it). Its gap results replace the published ones, refused unless every column that reads no fill
#: (the truth, the members other than the process and the stack, the reference and observed-
#: temperature arms) equals the published run's exactly. The forecasts read no fill and stay.
GAP_RESULTS = RUNS / 'v2_o2_temperature' / 'results_gp_by_pattern'
NO_FILL = ('T_truth', 'T_linear', 'T_chronos_bidir', 'T_chronos2_cov_bidir', 'c2_ref', 'c2cov_ref',
           'c2_T_obs', 'c2cov_T_obs')
OUT = RUNS / 'v2_o2_temperature.json'


def gap_results(cohort):
    published = pd.read_csv(RESULTS / f'gaps_{cohort}.csv.gz')
    if not (GAP_RESULTS / f'gaps_{cohort}.csv.gz').exists():
        return published
    rerun = pd.read_csv(GAP_RESULTS / f'gaps_{cohort}.csv.gz')
    keys = ['episode', 'position', 'gap_length', 'sensor']
    if not rerun[keys].equals(published[keys]):
        raise SystemExit(f'{cohort}: the rerun scores other rows than the published run')
    for col in NO_FILL:
        a, b = published[col].to_numpy(float), rerun[col].to_numpy(float)
        if not np.array_equal(a, b, equal_nan=True):
            raise SystemExit(f'{cohort}: {col} differs between the published run and the rerun')
    return rerun
FLOOR = 1.8
BANDS = {'Gap-Dev': ('1-6', '12-72'), 'Gap-2026': ('1-6', '12-72'), 'Gap-Long-Dev': ('168-1440',),
         'Gap-Long-2026': ('168-1440',)}
PAIRS = {'1-6': ('Gap-Dev', 'Gap-2026'), '12-72': ('Gap-Dev', 'Gap-2026'), '168-1440': ('Gap-Long-Dev', 'Gap-Long-2026')}
INCUMBENT = {'1-6': 'linear', '12-72': 'chronos_bidir', '168-1440': 'chronos_bidir'}   # v2_gap_variants.INCUMBENT
REFERENCE = {'c2_T_obs': 'chronos2_bidir', 'c2_T_fill': 'chronos2_bidir',
             'c2cov_T_obs': 'chronos2_cov_bidir', 'c2cov_T_fill': 'chronos2_cov_bidir'}
FC_CONTRASTS = (('c2_T', 'c2'), ('c2_T_fc', 'c2_T'), ('c2_T_obs', 'c2_T'), ('c2_T_fc', 'c2'),
                ('c2net_T_fc', 'c2net'), ('c2net_T_obs', 'c2net'))
FC_ARMS = ('c2', 'c2_T', 'c2_T_fc', 'c2_T_obs', 'c2net', 'c2net_T_fc', 'c2net_T_obs')
LEADS = (1, 3, 6, 12, 24, 36, 48)


def macro(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    return round(float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean()), 4)


def gaps():
    out, gates = {}, {}
    for cohort, bands in BANDS.items():
        runs = gap_results(cohort)
        pub = pd.concat([pd.read_csv(RUNS / 'v2_gap_variants' / f'{cohort}__oxygen__{b}.csv.gz') for b in bands],
                        ignore_index=True)
        f = pub.merge(runs.drop(columns=['sensor', 'gap_length']), on=['episode', 'position'], how='left')
        gates[cohort] = {'rows': int(len(f)),
                         'c2_ref_vs_chronos2_bidir': float(f'{(f.c2_ref - f.chronos2_bidir).abs().max():.3g}'),
                         'c2cov_ref_vs_chronos2_cov_bidir': float(f'{(f.c2cov_ref - f.chronos2_cov_bidir).abs().max():.3g}')}
        long_cohort = cohort.startswith('Gap-Long')
        by_band = {}
        for band in bands:
            lo, hi = (1, 6) if band == '1-6' else (12, 72) if band == '12-72' else (168, 1440)
            part = f[(f.gap_length >= lo) & (f.gap_length <= hi)].copy()
            cluster_of = clusters(part)
            t = part.dropna(subset=['T_fill', 'T_truth'])
            entry = {'episodes': int(part.episode.nunique()),
                     'temperature_fill_mae': round(float((t.T_fill - t.T_truth).abs().groupby(t.episode).mean().mean()), 4),
                     'temperature_fill_method': part.drop_duplicates('episode').T_pick.value_counts().to_dict(),
                     'mae': {a: macro(part, a) for a in ('linear', 'chronos_bidir', 'chronos2_bidir', 'chronos2_cov_bidir')
                             + tuple(REFERENCE)},
                     'contrasts': {}}
            for arm, ref in REFERENCE.items():
                refs = dict.fromkeys((ref, INCUMBENT[band], 'linear'))
                entry['contrasts'][arm] = {f'minus_{r}': paired(part, arm, r, f'a54|{cohort}|{band}|{arm}|{r}', cluster_of)
                                           for r in refs}
                if long_cohort:
                    per_length = {}
                    for length, sub in part.groupby('gap_length'):
                        start = sub.episode.str.rsplit('|', n=2).str[1].str[:7]
                        units = dict(zip(sub.episode, sub.sensor + '|' + start))
                        per_length[str(int(length))] = {
                            'mae': macro(sub, arm),
                            **{f'minus_{r}': paired(sub, arm, r, f'a54|{cohort}|{arm}|{r}|{int(length)}', units)
                               for r in refs}}
                    entry['contrasts'][arm]['by_gap_length'] = per_length
            if long_cohort:
                entry['note'] = 'pooled band contrasts of a tiled cohort are descriptive; by_gap_length decides'
            by_band[band] = entry
        out[cohort] = by_band
    return out, gates


def robust(c):
    return bool(c and c.get('excludes_zero') and c['delta'] < 0
                and (c.get('clusters', 0) >= 20 or c.get('jackknife_excludes_zero')))


def decisions(res):
    """D1: a filled-temperature arm replaces the thesis's oxygen method at a band (1 to 72 h) or length
    (past a week) only if it beats the incumbent robustly in both cohorts."""
    out = {}
    for arm in ('c2_T_fill', 'c2cov_T_fill'):
        for band, (dev, new) in PAIRS.items():
            inc = INCUMBENT[band]
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


def forecast_rows():
    """(period, lead) -> frame of sensor, time, position, truth: the benchmark's oxygen rows."""
    from smartbay_companion.forecasting import published_horizon
    P = np.load(ROOT / 'companion/.cache/full/panels.npz')
    index = pd.DatetimeIndex(P['times']).tz_localize('UTC')
    pos = pd.Series(np.arange(len(index)), index=index)
    names = [str(n) for n in P['names_oxygen']]
    values = P['values_oxygen']
    horizon = published_horizon()
    rows = {}
    for L in LEADS:
        g = horizon[('oxygen', 'tree', L)]
        rows[('development', L)] = pd.DataFrame({'sensor': g.sensor.to_numpy(), 'time': g.time.to_numpy(),
                                                 'position': pos[g.time].to_numpy(), 'truth': g.truth.to_numpy(float)})
        parts = []
        for j, n in enumerate(names):
            t = np.flatnonzero((index >= pd.Timestamp('2026-01-01', tz='UTC')) & np.isin(index.hour, (0, 6, 12, 18)))
            t = t[t + L < len(index)]
            ok = np.isfinite(values[t, j]) & np.isfinite(values[t + L, j])
            parts.append(pd.DataFrame({'sensor': n, 'time': index[t[ok]], 'position': t[ok], 'truth': values[t[ok] + L, j]}))
        rows[('2026', L)] = pd.concat(parts, ignore_index=True)
    return rows


def forecasts():
    from forecast_bench import macro_arrays, macro_from, with_calendar
    F = np.load(RESULTS / 'forecasts.npz')
    gate = {k.split('|')[0]: float(F[k][0]) for k in F.files if k.endswith('|g3_c2')}
    gate_net = {k.split('|')[0]: float(F[k][0]) for k in F.files if k.endswith('|g3_c2net')}
    rng = np.random.default_rng(20260919)
    out = {}
    for (period, L), frame in forecast_rows().items():
        frame = frame.copy()
        for arm in FC_ARMS:
            col = np.full(len(frame), np.nan)
            for s, idx in frame.groupby('sensor').groups.items():
                if f'{s}|t' not in F.files:
                    continue
                row_of = {int(p): r for r, p in enumerate(F[f'{s}|t'])}
                A = F[f'{s}|{arm}']
                for i in idx:
                    r = row_of.get(int(frame.at[i, 'position']))
                    if r is not None:
                        col[i] = A[r, L - 1]
            frame[arm] = col
        f = with_calendar(frame.dropna(subset=list(FC_ARMS) + ['truth']))
        S, C, quarter = macro_arrays(f, list(FC_ARMS))
        point = macro_from(S, C, quarter, np.ones(len(quarter)))
        draws = np.array([macro_from(S, C, quarter, np.bincount(rng.integers(0, len(quarter), len(quarter)),
                                                                   minlength=len(quarter)).astype(float))
                          for _ in range(600)])
        idx = {a: i for i, a in enumerate(FC_ARMS)}
        entry = {'rows': int(len(f)), 'probes': int(f.sensor.nunique()),
                 'mae': {a: round(float(point[idx[a]]), 4) for a in FC_ARMS}, 'contrasts': {}}
        for arm, ref in FC_CONTRASTS:
            d = draws[:, idx[arm]] - draws[:, idx[ref]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            entry['contrasts'][f'{arm} - {ref}'] = {'delta': round(float(point[idx[arm]] - point[idx[ref]]), 4),
                                                    'ci': [round(float(lo), 4), round(float(hi), 4)],
                                                    'excludes_zero': bool(lo > 0 or hi < 0)}
        out.setdefault(period, {})[str(L)] = entry
    return out, {'c2': gate, 'c2net': gate_net}


def predictions(g, fc):
    """The registered predictions, read off the results (H3 is the decisions)."""
    def c(cohort, band, arm, ref):
        return g[cohort][band]['contrasts'][arm][f'minus_{ref}']
    out = {}
    h1 = [c(co, '12-72', 'c2_T_obs', 'chronos2_bidir') for co in ('Gap-Dev', 'Gap-2026')]
    out['H1'] = all(x['excludes_zero'] and x['delta'] < 0 and abs(x['delta']) < FLOOR for x in h1)
    h2 = [(c(co, '12-72', 'c2_T_fill', 'chronos2_bidir')['delta'], c(co, '12-72', 'c2_T_obs', 'chronos2_bidir')['delta'])
          for co in ('Gap-Dev', 'Gap-2026')]
    out['H2'] = all(obs < 0 and fill > obs / 2 for fill, obs in h2)
    per = [g[co]['168-1440']['contrasts']['c2cov_T_fill']['by_gap_length'] for co in ('Gap-Long-Dev', 'Gap-Long-2026')]
    out['H4'] = not any(all(p.get(L, {}).get('minus_chronos2_cov_bidir') and robust(p[L]['minus_chronos2_cov_bidir'])
                            for p in per) for L in per[0])
    out['H5'] = all(abs(fc[p][L]['contrasts']['c2_T - c2']['delta']) < 0.10 for p in fc for L in fc[p])
    long_leads = [L for L in fc['development'] if int(L) >= 12]
    obs_gain = {L: fc['development'][L]['contrasts']['c2_T_obs - c2_T'] for L in long_leads}
    fc_gain = {L: fc['development'][L]['contrasts']['c2_T_fc - c2_T'] for L in long_leads}
    out['H6'] = all(obs_gain[L]['excludes_zero'] and obs_gain[L]['delta'] < 0
                    and fc_gain[L]['delta'] > obs_gain[L]['delta'] / 2 for L in long_leads)
    beyond = []
    for co, bands in g.items():
        for band, e in bands.items():
            for arm, ref in REFERENCE.items():
                d = e['contrasts'][arm][f'minus_{ref}']
                if d and abs(d['delta']) >= FLOOR:
                    beyond.append(f'{co} {band} {arm}')
                # on the tiled long cohorts the per-length contrasts are the ones that count
                for L, row in e['contrasts'][arm].get('by_gap_length', {}).items():
                    d = row.get(f'minus_{ref}')
                    if d and abs(d['delta']) >= FLOOR:
                        beyond.append(f'{co} {L} h {arm} ({d["delta"]:+.2f}, {d["clusters"]} units'
                                      f'{"" if d.get("estimable", True) else ", not estimable"})')
    for p in fc:
        for L in fc[p]:
            for name, d in fc[p][L]['contrasts'].items():
                if abs(d['delta']) >= FLOOR:
                    beyond.append(f'{p} {L} h {name}')
    out['H7'] = not beyond
    out['H7_beyond_floor'] = beyond
    return out


def main():
    g, gap_gates = gaps()
    fc, fc_gates = forecasts()
    d = decisions(g)
    report = {'addendum': 54, 'declaration': 'config/bay_reserved_protocol_v2_addendum54.json',
              'gates': {'G1': {**gap_gates, 'passed': all(v['c2_ref_vs_chronos2_bidir'] <= 1e-3
                                                          and v['c2cov_ref_vs_chronos2_cov_bidir'] <= 1e-3
                                                          for v in gap_gates.values())},
                        'G2': json.loads(((GAP_RESULTS if (GAP_RESULTS / 'g2.json').exists() else RESULTS)
                                          / 'g2.json').read_text(encoding='utf-8')),
                        'G3': {**fc_gates, 'passed': all(v <= 1e-3 for k in fc_gates.values() for v in k.values())},
                        'G4': {'what': "by construction: the run takes each forecast temperature from the precompute's "
                                       "temperature c2_net row whose issuance is the oxygen row's own "
                                       '(v2_o2_temperature.forecasts_stage, rows_t[p]); an issuance without one '
                                       'leaves the arm, and the row leaves every arm', 'passed': True}},
              'gaps': g, 'forecasts': fc, 'decisions': d,
              'gap_results': (GAP_RESULTS if GAP_RESULTS.exists() else RESULTS).relative_to(RUNS).as_posix(),
              'gap_results_note': ('the gaps stage rerun with the corrected Gaussian process '
                                   '(config/bay_reserved_protocol_v2_addendum59.json); every column '
                                   'that reads no fill equals the published run exactly'
                                   if GAP_RESULTS.exists() else 'the published run')}
    worst = max(v for e in report['gates']['G2'] for v in e['max_abs_difference'].values() if v is not None)
    report['gates']['G2'] = {'episodes': len(report['gates']['G2']), 'max_abs_difference': worst, 'passed': worst <= 1e-6}
    report['predictions'] = predictions(g, fc)
    report['predictions']['H3'] = not any(v['replaces'] for v in d.values())
    OUT.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))  # LF on every host
    print('gates', {k: v['passed'] for k, v in report['gates'].items()})
    print('predictions', {k: v for k, v in report['predictions'].items() if k != 'H7_beyond_floor'})
    print('replacements', [k for k, v in d.items() if v['replaces']])


if __name__ == '__main__':
    main()
