"""Addendum 53: score Chronos-Bolt's forward forecasts under three treatments of its context's holes.

Reads the arrays scripts/v2_context_holes_inputs.py built and the predictions the compute host
returned (runs/v2_context_holes/results/, from scripts/experiments/v2_context_holes.py),
checks the reproduction gates, and writes runs/v2_context_holes.json with, per task and variable,
each treatment's error, the three paired contrasts with the declared intervals, and the same on
the contexts with any hole and with a hole of 24 h or more.

  A1  six-hour forecast, Forecast-2025 folds: rows of runs/v2_sequence_arms/sequence_arms_<v>.csv.gz,
      v2_sequence_arms.paired_bootstrap (week blocks within fold, 600 draws, seed 20260921)
  A2  six-hour forecast, Reserved-2026: the battery's 4,250 rows, its macro and its bootstrap, the
      draws replayed as scripts/v2_battery.py draws them (one generator seeded 20260919)
  B   forward gap filler, Gap-Dev and Gap-2026 at 1 to 72 h: v2_gap_intervals.paired over overlap
      clusters, with the jackknife

  python -m scripts.v2_context_holes [--smoke]      # --smoke: gates on the smoke run only
"""
import json
import sys

import numpy as np
import pandas as pd

from scripts.v2_gap_intervals import clusters, paired
from src.data.bay_profile_context import ROOT
from src.experiments import v2_sequence_arms as seq

RUNS = ROOT / 'reports/bay_twin/runs'
HERE = RUNS / 'v2_context_holes'
OUT = RUNS / 'v2_context_holes.json'
TRINITY = ROOT / 'reports/bay_twin/runs/battery/results'
GAP_RUNS = {'Gap-Dev': ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery'),
            'Gap-2026': ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026')}
TREATMENTS = ('linear', 'mask', 'bridge')
PAIRS = (('bridge', 'linear'), ('bridge', 'mask'), ('mask', 'linear'))
KEY = ['variable', 'sensor', 'time']
TOL = 1e-3


def load(task, smoke):
    inputs = dict(np.load(HERE / 'inputs' / f'{task}.npz'))
    results = dict(np.load(HERE / ('results_smoke' if smoke else 'results') / f'{task}.npz'))
    return inputs, results


def gate(values, reference, what):
    both = np.isfinite(values) & np.isfinite(reference)
    worst = float(np.max(np.abs(values[both] - reference[both]))) if both.any() else float('nan')
    return {'what': what, 'compared': int(both.sum()), 'missing': int((~both).sum()),
            'max_abs_diff': round(worst, 6), 'passed': bool(both.all() and worst <= TOL)}


def a1(variable, smoke):
    inputs, res = load(f'A1_{variable}', smoke)
    rows = res['rows']
    scale = float(inputs['scale'])
    got = pd.DataFrame({'fold': inputs['key_fold'][rows], 'sensor': inputs['key_sensor'][rows],
                        'time': inputs['key_time'][rows], 'hole_any': inputs['hole_any'][rows],
                        'hole_24h': inputs['hole_24h'][rows],
                        **{t: res[f'pred_{t}'] for t in TREATMENTS}})
    pub = pd.read_csv(RUNS / 'v2_sequence_arms' / f'sequence_arms_{variable}.csv.gz')
    frame = pub.merge(got, on=['fold', 'sensor', 'time'], how='inner' if smoke else 'left',
                      validate='one_to_one')
    out = {'gate_G1': gate(frame['linear'].to_numpy() * scale, frame['chronos_zs'].to_numpy() * scale,
                           'linear vs published chronos_zs, physical units, every endpoint')}
    if smoke:
        return out
    out['gate_G1']['mae_diff'] = round(abs(seq.macro(frame, 'linear') - seq.macro(frame, 'chronos_zs')) * scale, 6)
    out['gate_G1']['passed'] &= out['gate_G1']['mae_diff'] <= 1e-4
    out.update(summarise_a1(frame, scale))
    for subset, part in subsets(frame).items():
        out[f'on_{subset}'] = {'endpoints': int(len(part)), **summarise_a1(part, scale)}
    out['endpoints'] = int(len(frame))
    return out


def subsets(frame):
    """The declared descriptive subsets: contexts with any hole, with a hole of 24 h or more, and
    without one (the comparison registered prediction H4 makes)."""
    return {'hole_any': frame[frame.hole_any.astype(bool)], 'hole_24h': frame[frame.hole_24h.astype(bool)],
            'no_hole_24h': frame[~frame.hole_24h.astype(bool)]}


def summarise_a1(frame, scale):
    return {'mae': {t: round(seq.macro(frame, t) * scale, 4) for t in TREATMENTS},
            'contrasts': {f'{a} - {b}': seq.paired_bootstrap(frame, a, b, scale) for a, b in PAIRS}}


def battery_frame():
    """The battery's joined rows (scripts/v2_battery.py lines 14-17), the published ZS_raw columns."""
    f = pd.read_csv(TRINITY / 'forecast/forecast_predictions.csv.gz')[KEY + ['quarter', 'week', 'truth']]
    c = pd.read_csv(TRINITY / 'chronos/chronos_predictions.csv.gz')
    return f.merge(c[KEY + ['temperature|ZS_raw', 'oxygen|ZS_raw']], on=KEY, validate='one_to_one')


def battery_macro(s, col):
    s = s.dropna(subset=[col])
    return float(s.assign(e=(s[col] - s.truth).abs()).groupby(['quarter', 'sensor']).e.mean()
                 .groupby('quarter').mean().mean())


def a2(frames, smoke):
    """Both variables, the draws replayed in the battery's order so its intervals would reproduce."""
    out = {}
    for variable in ('temperature', 'oxygen'):
        inputs, res = load(f'A2_{variable}', smoke)
        rows = res['rows']
        got = pd.DataFrame({'variable': inputs['key_variable'][rows], 'sensor': inputs['key_sensor'][rows],
                            'time': inputs['key_time'][rows], 'hole_any': inputs['hole_any'][rows],
                            'hole_24h': inputs['hole_24h'][rows],
                            **{t: res[f'pred_{t}'] for t in TREATMENTS}})
        s0 = frames[frames.variable == variable].merge(got, on=KEY, how='inner' if smoke else 'left',
                                                        validate='one_to_one')
        out[variable] = {'gate_G2': gate(s0['mask'].to_numpy(), s0[f'{variable}|ZS_raw'].to_numpy(),
                                         'mask vs the battery\'s published ZS_raw, every endpoint'),
                         'frame': s0}
    if smoke:
        return {v: {'gate_G2': o['gate_G2']} for v, o in out.items()}
    rng = np.random.default_rng(20260919)
    for variable in ('temperature', 'oxygen'):
        s0 = out[variable].pop('frame')
        s0['no_hole_24h'] = ~s0.hole_24h.astype(bool)
        blocks = s0[['quarter', 'week']].drop_duplicates().values.tolist()
        groups = {tuple(k): g for k, g in s0.groupby(['quarter', 'week'])}
        draws = {p: [] for p in PAIRS}
        subsets = {'hole_any': s0.hole_any.astype(bool), 'hole_24h': s0.hole_24h.astype(bool),
                   'no_hole_24h': ~s0.hole_24h.astype(bool)}
        sub_draws = {(n, p): [] for n in subsets for p in PAIRS}
        for _ in range(600):
            s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
            for a, b in PAIRS:
                draws[(a, b)].append(battery_macro(s, a) - battery_macro(s, b))
            for n in subsets:
                part = s[s[n].astype(bool)]
                for a, b in PAIRS:
                    sub_draws[(n, (a, b))].append(battery_macro(part, a) - battery_macro(part, b)
                                                  if len(part) else np.nan)

        def entry(frame, values, a, b):
            lo, hi = np.nanpercentile(values, [2.5, 97.5])
            return [round(battery_macro(frame, a) - battery_macro(frame, b), 4), round(float(lo), 4),
                    round(float(hi), 4)]
        o = out[variable]
        o['endpoints'] = int(len(s0))
        o['mae'] = {t: round(battery_macro(s0, t), 4) for t in TREATMENTS}
        o['published_zs_raw_mae'] = round(battery_macro(s0, f'{variable}|ZS_raw'), 4)
        o['contrasts'] = {f'{a} - {b}': entry(s0, draws[(a, b)], a, b) for a, b in PAIRS}
        for n, mask in subsets.items():
            part = s0[mask]
            o[f'on_{n}'] = {'endpoints': int(len(part)),
                            'mae': {t: round(battery_macro(part, t), 4) for t in TREATMENTS},
                            'contrasts': {f'{a} - {b}': entry(part, sub_draws[(n, (a, b))], a, b)
                                          for a, b in PAIRS}}
    return out


def b(variable, cohort, smoke):
    inputs, res = load(f'B_{cohort}_{variable}', smoke)
    rows = res['rows']
    run, results = GAP_RUNS[cohort]
    pub = pd.read_csv(RUNS / run / results / f'{variable}_chronos_forward' / 'predictions.csv.gz')
    pub = pub[pub.gap_length <= 72].reset_index(drop=True)
    n_of = {k: i for i, k in enumerate(inputs['key_episode'])}
    ctx = pub.episode.map(n_of).to_numpy()
    keep = np.isin(ctx, rows)
    frame = pub[keep].copy()
    pos = {n: j for j, n in enumerate(rows)}
    j = np.array([pos[n] for n in ctx[keep]])
    k = frame.position.to_numpy(int) - 1
    for t in TREATMENTS:
        frame[t] = res[f'pred_{t}'][j, k]
    frame['hole_any'] = inputs['hole_any'][ctx[keep]]
    frame['hole_24h'] = inputs['hole_24h'][ctx[keep]]
    out = {'gate_G3': gate(frame['linear'].to_numpy(), frame['chronos_forward'].to_numpy(),
                           'linear vs the published chronos_forward, every row')}
    if smoke:
        return out
    cluster_of = clusters(frame)
    label = f'a53|{cohort}|{variable}'
    out['rows'] = int(len(frame))
    out['episodes'] = int(frame.episode.nunique())
    out['mae'] = {t: round(float(frame.dropna(subset=[t]).assign(e=lambda d, t=t: (d.truth - d[t]).abs())
                                 .groupby('episode').e.mean().mean()), 4) for t in TREATMENTS}
    out['contrasts'] = {f'{a} - {b_}': paired(frame, a, b_, f'{label}|{a}|{b_}', cluster_of)
                        for a, b_ in PAIRS}
    for subset, part in subsets(frame).items():
        out[f'on_{subset}'] = {'episodes': int(part.episode.nunique()),
                               'contrasts': {f'{a} - {b_}': paired(part, a, b_, f'{label}|{subset}|{a}|{b_}',
                                                                   cluster_of) for a, b_ in PAIRS}}
    out['by_gap_length'] = {str(int(L)): {f'{a} - {b_}': paired(part, a, b_, f'{label}|{a}|{b_}|{int(L)}',
                                                                 cluster_of) for a, b_ in PAIRS}
                            for L, part in frame.groupby('gap_length')}
    return out


BOUND = {'temperature': 0.005, 'oxygen': 0.10}          # H1, H2, H3, H5
MARGIN = {'temperature': 0.029, 'oxygen': 1.8}           # D4


def delta(c):
    return c[0] if isinstance(c, list) else c['delta']


def excludes(c):
    return (c[1] > 0 or c[2] < 0) if isinstance(c, list) else bool(c.get('excludes_zero'))


def beats(c):
    return excludes(c) and delta(c) < 0


def verdicts(r):
    """The registered predictions and decision rules of the declaration, read mechanically."""
    A1, A2, B = r['A1'], r['A2'], r['B']
    small = lambda c, v: abs(delta(c)) < BOUND[v]
    h = {
        'H1': {v: small(A1[v]['contrasts']['bridge - linear'], v)
                  and not excludes(A1[v]['contrasts']['bridge - linear']) for v in A1},
        'H2': {f'{task} {v}': small(src[v]['contrasts']['mask - linear'], v)
               for task, src in (('A1', A1), ('A2', A2)) for v in src},
        'H3': {v: small(A2[v]['contrasts']['bridge - mask'], v) for v in A2},
        'H4': {v: abs(delta(A1[v]['on_hole_24h']['contrasts']['bridge - linear']))
                  > abs(delta(A1[v]['on_no_hole_24h']['contrasts']['bridge - linear'])) for v in A1},
        'H5': {f'{c} {v}': small(B[c][v]['contrasts']['bridge - linear'], v) for c in B for v in B[c]},
    }
    out = {k: {'held': all(x.values()), 'cells': x} for k, x in h.items()}
    out['D1_bridge_replaces_the_six_hour_treatment'] = {
        v: all(beats(src[v]['contrasts'][p]) for src in (A1, A2) for p in ('bridge - linear', 'bridge - mask'))
        for v in A1}
    out['D2_mask_and_linear_differ_at_six_hours'] = {
        v: any(excludes(src[v]['contrasts']['mask - linear']) for src in (A1, A2)) for v in A1}
    out['D3_bridge_replaces_the_forward_filler_context'] = {
        v: all(beats(B[c][v]['contrasts']['bridge - linear']) for c in B) for v in ('temperature', 'oxygen')}
    out['D4_largest_difference_against_its_margin'] = {
        v: round(max(abs(delta(c)) for src in (A1, A2) for c in src[v]['contrasts'].values())
                 / MARGIN[v], 3) for v in A1}
    return out


def main():
    smoke = '--smoke' in sys.argv
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum53.json', 'smoke': smoke,
              'treatments': list(TREATMENTS), 'A1': {}, 'B': {}}
    for variable in ('temperature', 'oxygen'):
        report['A1'][variable] = a1(variable, smoke)
    report['A2'] = a2(battery_frame(), smoke)
    for cohort in GAP_RUNS:
        report['B'][cohort] = {v: b(v, cohort, smoke) for v in ('temperature', 'oxygen')}
    gates = {'G1': [report['A1'][v]['gate_G1'] for v in report['A1']],
             'G2': [report['A2'][v]['gate_G2'] for v in report['A2']],
             'G3': [report['B'][c][v]['gate_G3'] for c in report['B'] for v in report['B'][c]]}
    report['gates_passed'] = {g: all(x['passed'] for x in xs) for g, xs in gates.items()}
    print(json.dumps(report['gates_passed']))
    for g, xs in gates.items():
        for x in xs:
            print(g, x)
    if smoke:
        return
    report['registered_predictions_and_decisions'] = verdicts(report)
    print(json.dumps(report['registered_predictions_and_decisions'], indent=1))
    report['inputs_manifest'] = json.loads((HERE / 'inputs' / 'manifest.json').read_text())
    report['host_status'] = json.loads((HERE / 'results' / 'status.json').read_text())
    OUT.write_text(json.dumps(report, indent=2, default=float) + '\n', encoding='utf-8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
