"""Addendum 82, part E: gates, pairing, contrasts and verdicts of the graph-arm sweep.

For each graph contrast (E1-E6 and the derived ones) and each of its eight cells (phases 0-5 at sampler
seed 42, seeds 7 and 123 on the reference grid): the three-network mean of each graph arm against its
reference on identical endpoints. The offset-0 cells of the contrasts scripts/v2_seed_contrasts.py
already quotes are recomputed from the published predictions in that script's own frame, stream and
labels and stored to six decimals (E-G5 compares them with runs/v2_seed_contrasts.json at its four
decimals); every other cell draws from v2_phase_shift.paired with default_rng([20260921, crc32(label)]),
label '{variable}|graph|h{h}|{offset}|{seed}|{arm} - {reference}'. Chapter 4's verdict without and with
the per-fold replicate (runs/v2_perfold_neural.json), the oxygen operator's addendum-18 cells beside
the rerun, single-network points, BH and BY over the part's own intervals.
Writes reports/bay_twin/runs/v2_addendum82_E.json.

Declared in config/bay_reserved_protocol_v2_addendum82.json (part E).
Usage: python -m scripts.v2_phase_graph_contrasts
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.sweep_verdict import excludes, label, verdict  # noqa: E402
from src.units import factor  # noqa: E402

RUNS = ROOT / 'reports/bay_twin/runs'
HERE = RUNS / 'v2_addendum82/E'
OUT = RUNS / 'v2_addendum82_E.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum82.json'
PHASEFIX = RUNS / '20260923T103430Z_v2_phasefix/reports/bay_twin/runs'
HORIZON_RUN = RUNS / '20260920T083322Z_v2_horizon/results'
FORECAST_RUN = RUNS / '20260919T144822Z_v2_forecast/results'
SEEDREP = {7: RUNS / '20260923T110726Z_v2_seedrep/results', 123: RUNS / '20260923T110752Z_v2_seedrep/results'}
KEY = ['fold', 'sensor', 'time']
C = factor()                      # mg/l per umol/kg
FLOORS = {'temperature': 0.029, 'oxygen': 0.059 / C}
SEEDS = (42, 7, 123)
DRAWS = 600

#: id -> (variable, horizon, arm, reference, per-fold key in runs/v2_perfold_neural.json, seed-contrast key or None)
CONTRASTS = {
    'E1': ('temperature', 6, 'learned_source', 'tree', 'seed|temperature|6|learned_source - tree'),
    'E2': ('temperature', 6, 'conservative_exchange', 'learned_source',
           'seed|temperature|6|conservative_exchange - learned_source'),
    'E3': ('oxygen', 6, 'conservative_exchange', 'learned_source', 'seed|oxygen|6|conservative_exchange - learned_source'),
    'E4': ('oxygen', 24, 'learned_source', 'tree', 'seed|oxygen|24|learned_source - tree'),
    'E5': ('oxygen', 6, 'saturation_exchange', 'conservative_exchange',
           'forecast|oxygen|saturation_exchange - conservative_exchange'),
    'E6': ('oxygen', 6, 'saturation_source', 'learned_source', 'forecast|oxygen|saturation_source - learned_source'),
    'D1': ('temperature', 6, 'conservative_exchange', 'tree', 'seed|temperature|6|conservative_exchange - tree'),
    'D2': ('oxygen', 6, 'conservative_exchange', 'tree', 'seed|oxygen|6|conservative_exchange - tree'),
    'D3': ('oxygen', 6, 'saturation_source', 'tree', 'forecast|oxygen|saturation_source - tree'),
    'D4': ('oxygen', 6, 'saturation_exchange', 'tree', 'forecast|oxygen|saturation_exchange - tree'),
}
#: offset-0 cells recomputed in the seed-contrast stream (the contrasts runs/v2_seed_contrasts.json quotes)
SEED_STREAM = {'E1', 'E2', 'E3', 'E4', 'D1', 'D2'}
#: E-G5 reference points of the two prior contrasts (runs/v2_perfold_neural.json, 'published')
PUBLISHED_POINTS = {'E5': 0.0022113084358981183, 'E6': -0.028731255187098625}


def r6(x):
    return None if x is None else round(float(x), 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def macro(frame, col):
    f = frame.dropna(subset=[col])
    return float(f.assign(e=(f.truth - f[col]).abs()).groupby(['fold', 'sensor']).e.mean()
                 .groupby('fold').mean().mean())


def paired_seed_stream(frame, a, b, label_):
    """scripts/v2_seed_contrasts.paired without its four-decimal rounding (same frame, stream and draws)."""
    f = frame.dropna(subset=[a, b]).assign(
        week=lambda d: pd.to_datetime(d.time, utc=True).dt.isocalendar().week.astype(int))
    rng = np.random.default_rng([20260923, zlib.crc32(label_.encode())])
    blocks = f[['fold', 'week']].drop_duplicates().values.tolist()
    groups = {tuple(k): g for k, g in f.groupby(['fold', 'week'])}
    draws = []
    for _ in range(DRAWS):
        s = pd.concat([groups[tuple(blocks[i])] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(macro(s, a) - macro(s, b))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return macro(f, a) - macro(f, b), float(lo), float(hi), int(len(f))


def paired_new(frame, a, b, label_):
    from src.experiments.v2_phase_shift import paired
    f = frame.dropna(subset=[a, b])
    point, lo, hi = paired(f, a, b, label_)
    return point, lo, hi, int(len(f))


def cell_dict(point, lo, hi, n, label_, stream):
    return {'point': r6(point), 'lo': r6(lo), 'hi': r6(hi), 'excludes_zero': bool(excludes(lo, hi)),
            'paired_endpoints': n, 'label': label_, 'stream': stream}


# ------------------------------------------------------------------------- the arms of one cell

def published_graph(folder, arm_label):
    """A published graph arm: the three-network mean and each network, forecast rows only."""
    d = pd.read_csv(folder / 'selected_predictions.csv.gz')
    if 'mode' in d:
        d = d[d['mode'] == 'forecast']
    out = d.groupby(KEY, as_index=False).agg(truth=('truth', 'first'), **{arm_label: ('prediction', 'mean')})
    for s, g in d.groupby('seed'):
        out = out.merge(g[KEY + ['prediction']].rename(columns={'prediction': f'{arm_label}_init{int(s)}'}),
                        on=KEY, how='left', validate='one_to_one')
    return out


def digest_check(folder):
    """The published run's report records the prediction digest; None where it does not."""
    rep = folder / 'report.json'
    if not rep.exists():
        return None
    want = json.loads(rep.read_text(encoding='utf-8')).get('prediction_sha256')
    return None if want is None else want == sha(folder / 'selected_predictions.csv.gz')


def graph_arm(variable, arm, horizon, offset, seed, reuse_log):
    """The graph arm at one cell: new, reused published, or reused seed replicate."""
    s = 42 if seed is None else seed
    new = HERE / 'arms' / f'phase_graph_{variable}_{arm}_h{horizon}_offset{offset}' \
        f"{'' if seed is None else f'_seed{seed}'}.csv.gz"
    if offset != 0 or (seed is not None and arm in ('saturation_exchange', 'saturation_source')):
        summary = json.loads(new.with_name(new.name.replace('.csv.gz', '_summary.json')).read_text(encoding='utf-8'))
        if summary['prediction_sha256'] != sha(new) or summary['selection']['completed_trials'] < 100 \
                or summary.get('smoke_only'):
            raise SystemExit(f'{new.name}: digest, trials or smoke flag')
        f = pd.read_csv(new)
        cols = KEY + ['truth', arm] + [c for c in f.columns if c.startswith(arm + '_init')]
        return f[cols], 'new', {'file': new.relative_to(ROOT).as_posix(), 'sha256': sha(new),
                                'completed_trials': summary['selection']['completed_trials'],
                                'failed_trials': summary['selection']['failed_trials']}
    if s == 42:
        if arm == 'saturation_source':          # the horizon run has no saturation-source arm
            folder = FORECAST_RUN / f'v2_forecast_{variable}_{arm}_nested'
        else:
            folder = HORIZON_RUN / f'v2_horizon_{variable}_{arm}_h{horizon}'
    else:
        folder = SEEDREP[s] / f'v2_horizon_{variable}_{arm}_h{horizon}'
    ok = digest_check(folder)
    reuse_log.append({'file': (folder / 'selected_predictions.csv.gz').relative_to(ROOT).as_posix(),
                      'digest_matches_report': ok})
    return published_graph(folder, arm), 'published', {'file': (folder / 'selected_predictions.csv.gz')
                                                       .relative_to(ROOT).as_posix()}


def tree_arm(variable, horizon, offset, seed, reuse_log):
    s = 42 if seed is None else seed
    if offset == 0 and s == 42:
        folder = HORIZON_RUN / f'v2_horizon_{variable}_tree_h{horizon}'
        d = pd.read_csv(folder / 'selected_predictions.csv.gz')
        if 'mode' in d:
            d = d[d['mode'] == 'forecast']
        reuse_log.append({'file': (folder / 'selected_predictions.csv.gz').relative_to(ROOT).as_posix(),
                          'digest_matches_report': digest_check(folder)})
        return d[KEY + ['truth', 'prediction']].rename(columns={'prediction': 'tree'})
    folder = PHASEFIX / ('v2_phase_shift' if horizon == 6 else f'v2_phase_shift_h{horizon}')
    sfx = '' if seed is None else f'_seed{seed}'
    path = folder / f'phase_shift_{variable}_offset{offset}_ratio{sfx}.csv.gz'
    summ = json.loads((folder / f'phase_shift_offset{offset}_ratio{sfx}_summary.json').read_text(encoding='utf-8'))
    ok = summ['variables'][variable]['prediction_sha256'] == sha(path) and summ['horizon_hours'] == horizon \
        and summ['offset_hours'] == offset
    reuse_log.append({'file': path.relative_to(ROOT).as_posix(), 'digest_matches_report': bool(ok)})
    return pd.read_csv(path)[KEY + ['truth', 'tree']]


def pair_frames(a, b, gate):
    """Join two arms on identical endpoints; record whether they hold the same endpoints and truth."""
    j = a.merge(b, on=KEY, how='outer', suffixes=('', '_ref'), indicator=True)
    same = bool((j['_merge'] == 'both').all())
    gap = float((j.truth - j.truth_ref).abs().max()) if same else float('nan')
    gate.append({'same_endpoints': same, 'max_truth_difference': gap, 'pass': bool(same and gap <= 1e-9)})
    return j[j['_merge'] == 'both'].drop(columns=['_merge', 'truth_ref'])


def single_networks(frame, arm, ref):
    """Point contrasts with each graph arm taken one network at a time (paired by initialisation seed)."""
    out = {}
    for s in SEEDS:
        a = f'{arm}_init{s}'
        b = f'{ref}_init{s}' if ref != 'tree' else 'tree'
        if a in frame and b in frame:
            f = frame.dropna(subset=[a, b])
            out[str(s)] = r6(macro(f, a) - macro(f, b))
    return out


# ------------------------------------------------------------------------------------ verdicts

def triple(c):
    return (c['point'], c['lo'], c['hi'])


def reading(cells, perfold, floor):
    seeds = [triple(cells['0|42']), triple(cells['0|7']), triple(cells['0|123'])]
    phases = [triple(cells[f'{k}|42']) for k in range(6)]
    without = verdict(seeds, phases)
    with_pf = verdict(seeds + [perfold], phases) if perfold is not None else None
    printed = with_pf if with_pf is not None else without
    points = [p for p, _, _ in phases]
    out = {'verdict_without_perfold': without, 'verdict_with_perfold': with_pf, 'verdict': printed,
           'printed': label(printed), 'phase_mean': r6(np.mean(points)), 'phase_range': r6(max(points) - min(points)),
           'negative_phases': int(sum(p < 0 for p in points)),
           'cells_excluding_zero': int(sum(c['excludes_zero'] for c in cells.values())), 'floor': r6(floor)}
    if printed == 'result':
        allp = [c['point'] for c in cells.values()] + ([perfold[0]] if perfold is not None else [])
        out['size'] = {'per_fold_exceeds_floor': None if perfold is None else bool(abs(perfold[0]) > floor),
                       'phase_mean_exceeds_floor': bool(abs(np.mean(points)) > floor),
                       'cells_exceeding_floor': int(sum(abs(p) > floor for p in allp))}
    else:
        out['size'] = 'not read: the verdict is not stable (the verdict, phase mean and range are reported)'
    return out


def published_seed_label(cid, variable, horizon, arm, ref):
    """The seed-reading the thesis printed before this sweep (seeds and per fold)."""
    sc = json.loads((RUNS / 'v2_seed_contrasts.json').read_text(encoding='utf-8'))['contrasts']
    key = f'{variable}|{horizon}|{arm} - {ref}'
    if key not in sc:
        return None
    pn = json.loads((RUNS / 'v2_perfold_neural.json').read_text(encoding='utf-8'))['contrasts']
    seeds = [(sc[key]['by_seed'][s]['point'], *sc[key]['by_seed'][s]['ci']) for s in ('42', '7', '123')]
    pf = pn.get(f'seed|{key}', {}).get('values', {}).get('perfold')
    return label(verdict(seeds + ([tuple(pf)] if pf else [])))


def main():
    from scripts.v2_seed_contrasts import frame_for
    from scripts.v2_fdr import p_mid, p_side, step_up
    pn = json.loads((RUNS / 'v2_perfold_neural.json').read_text(encoding='utf-8'))['contrasts']
    sc = json.loads((RUNS / 'v2_seed_contrasts.json').read_text(encoding='utf-8'))['contrasts']
    report = {'declared_in': DECLARED, 'part': 'E',
              'unit': 'oxygen umol/kg as stored (print x 0.032844 mg/l), temperature degC',
              'cells': 'phases 0-5 at sampler seed 42 and seeds 7 and 123 at offset 0, keyed offset|seed',
              'graph_arms': 'mean of three networks at initialisation seeds 42, 7 and 123',
              'gates': {}, 'contrasts': {}}
    # gate files written by the gate jobs on the compute host and by E-G0 locally
    gates = report['gates']
    g0 = HERE / 'gates' / 'E-G0.json'
    gates['E-G0'] = {'pass': bool(g0.exists() and json.loads(g0.read_text(encoding='utf-8'))['pass']),
                     'file': g0.relative_to(ROOT).as_posix()}
    refits = sorted((HERE / 'gates').glob('refit_*.json'))
    gates['E-G1'] = {p.stem: {'pass': json.loads(p.read_text(encoding='utf-8'))['E-G1']['pass'],
                              'max_abs_prediction_diff': json.loads(p.read_text(encoding='utf-8'))['E-G1']['max_abs_prediction_diff']}
                     for p in refits}
    gates['E-G2'] = {p.stem: {'pass': json.loads(p.read_text(encoding='utf-8'))['E-G2']['pass'],
                              'abs_diff': json.loads(p.read_text(encoding='utf-8'))['E-G2']['abs_diff']} for p in refits}
    g3 = HERE / 'gates' / 'reproduce_temperature_learned_source_h6.json'
    g3r = json.loads(g3.read_text(encoding='utf-8')) if g3.exists() else None
    gates['E-G3'] = None if g3r is None else {k: g3r[k] for k in ('pass', 'trials_here', 'trials_published',
                                                                    'max_abs_value_diff', 'params_identical',
                                                                    'best_trial_here', 'best_trial_published',
                                                                    'optuna_here', 'optuna_published')}
    gates['E-G4'], gates['E-G5'] = {}, {}
    refit_pass = {(r.stem.split('_', 1)[1]): v['pass'] and gates['E-G2'][r.stem]['pass']
                  for r, v in zip(refits, gates['E-G1'].values())}

    entries = []
    for cid, (variable, horizon, arm, ref, pf_key) in CONTRASTS.items():
        cells, singles, reuse, g4, sources = {}, {}, [], [], {}
        for offset, seed in [(k, None) for k in range(6)] + [(0, 7), (0, 123)]:
            key = f'{offset}|{42 if seed is None else seed}'
            s = 42 if seed is None else seed
            if offset == 0 and cid in SEED_STREAM:
                frame = frame_for(variable, horizon, s)
                lab = f'{variable}|{horizon}|{arm} - {ref}|{s}'
                point, lo, hi, n = paired_seed_stream(frame, arm, ref, lab)
                cells[key] = cell_dict(point, lo, hi, n, lab, 'scripts/v2_seed_contrasts.py')
                pub = sc[f'{variable}|{horizon}|{arm} - {ref}']['by_seed'][str(s)]
                same = round(point, 4) == pub['point'] and [round(lo, 4), round(hi, 4)] == pub['ci']
                gates['E-G5'][f'{cid}|{key}'] = {'recomputed': [round(point, 4), round(lo, 4), round(hi, 4)],
                                                 'published': [pub['point'], *pub['ci']], 'pass': bool(same)}
                sources[key] = 'published predictions, frame of scripts/v2_seed_contrasts.frame_for'
                # single networks from the published files of this cell
                a_df, _, _ = graph_arm(variable, arm, horizon, 0, seed, reuse)
                b_df = (tree_arm(variable, horizon, 0, seed, reuse) if ref == 'tree'
                        else graph_arm(variable, ref, horizon, 0, seed, reuse)[0])
                singles[key] = single_networks(pair_frames(a_df, b_df, g4), arm, ref)
                continue
            a_df, a_kind, a_src = graph_arm(variable, arm, horizon, offset, seed, reuse)
            if ref == 'tree':
                b_df = tree_arm(variable, horizon, offset, seed, reuse)
                b_src = 'tree'
            else:
                b_df, _, b_src = graph_arm(variable, ref, horizon, offset, seed, reuse)
            frame = pair_frames(a_df, b_df, g4)
            lab = f'{variable}|graph|h{horizon}|{offset}|{s}|{arm} - {ref}'
            point, lo, hi, n = paired_new(frame, arm, ref, lab)
            cells[key] = cell_dict(point, lo, hi, n, lab, 'v2_phase_shift.paired')
            sources[key] = {'arm': a_src, 'reference': b_src}
            singles[key] = single_networks(frame, arm, ref)
            if offset == 0 and s == 42 and cid in PUBLISHED_POINTS:
                gates['E-G5'][f'{cid}|{key}'] = {'recomputed_point': point, 'published_point': PUBLISHED_POINTS[cid],
                                                 'abs_diff': abs(point - PUBLISHED_POINTS[cid]),
                                                 'pass': bool(abs(point - PUBLISHED_POINTS[cid]) <= 1e-9)}
        gates['E-G4'][cid] = {'files': reuse, 'pairs': g4,
                              'pass': all(x['digest_matches_report'] is not False for x in reuse)
                              and all(x['pass'] for x in g4)}
        pf = pn.get(pf_key, {}).get('values', {}).get('perfold')
        pf = tuple(pf) if pf else None
        entry = {'variable': variable, 'horizon_hours': horizon, 'arm': arm, 'reference': ref,
                 'kind': {'E1': 'operator', 'E2': 'constraint', 'E3': 'constraint', 'E4': 'operator', 'E5': 'prior',
                          'E6': 'prior'}.get(cid, 'derived'),
                 'cells': cells, 'per_fold': None if pf is None else [r6(x) for x in pf], 'per_fold_key': pf_key,
                 'single_network_points': singles, 'sources': sources,
                 'published_reading_before_sweep': published_seed_label(cid, variable, horizon, arm, ref)}
        entry.update(reading(cells, pf, FLOORS[variable]))
        needed = {f'{variable}_{arm}_h{horizon}'} | ({f'{variable}_{ref}_h{horizon}'} if ref != 'tree' else set())
        entry['refit_gates_pass'] = all(refit_pass.get(n, False) for n in needed)
        report['contrasts'][cid] = entry
        for key, c in cells.items():
            if c['stream'] == 'scripts/v2_seed_contrasts.py':
                continue       # reproduces a published interval: outside the family
            p = p_side(c['point'], c['lo'], c['hi'])
            how = 'estimate and end towards zero'
            if p is None:
                p, how = p_mid(c['lo'], c['hi']), 'midpoint'
            entries.append({'contrast': cid, 'cell': key, 'point': c['point'], 'lo': c['lo'], 'hi': c['hi'],
                            'excludes_zero': c['excludes_zero'], 'p': p, 'p_from': how})
        print(f"{cid} {variable} {horizon} h {arm} - {ref}: "
              + ' '.join(f"{k}:{c['point']:+.5f}{'*' if c['excludes_zero'] else ''}" for k, c in cells.items())
              + f"  per fold {pf[0] if pf else float('nan'):+.5f}  phase mean {entry['phase_mean']:+.6f} -> "
              f"{entry['printed']} (before: {entry['published_reading_before_sweep']})", flush=True)

    # the oxygen operator: addendum 18's cells (unchanged verdict) beside the rerun
    pm = json.loads((RUNS / 'v2_phase_matched.json').read_text(encoding='utf-8'))['operator_h6']
    rerun, check = {}, {}
    for k in range(1, 6):
        summ = json.loads((PHASEFIX / 'v2_phase_shift' / f'phase_operator_oxygen_offset{k}_treeratio_summary.json')
                          .read_text(encoding='utf-8'))
        check[str(k)] = bool(np.allclose(summ['contrasts']['learned_source - tree'], pm['phases'][str(k)]['contrast'],
                                         rtol=0, atol=1e-12))
        reuse = []
        a_df, _, _ = graph_arm('oxygen', 'learned_source', 6, k, None, reuse)
        frame = pair_frames(a_df, tree_arm('oxygen', 6, k, None, reuse), [])
        lab = f'oxygen|graph|h6|{k}|42|learned_source - tree'
        point, lo, hi, n = paired_new(frame, 'learned_source', 'tree', lab)
        rerun[f'{k}|42'] = cell_dict(point, lo, hi, n, lab, 'v2_phase_shift.paired')
        rerun[f'{k}|42']['addendum18_cell'] = [r6(x) for x in pm['phases'][str(k)]['contrast']]
        rerun[f'{k}|42']['difference_rerun_minus_addendum18'] = r6(point - pm['phases'][str(k)]['contrast'][0])
    gates['E-G5']['operator_phase_cells_equal_v2_phase_matched'] = {'cells': check, 'pass': all(check.values())}
    report['oxygen_operator_6h'] = {
        'cells_of_record': 'addendum 18 (runs/v2_phase_matched.json operator_h6); its verdict stays choice-sensitive',
        'rerun_on_compute_host_optuna_4_9_0': rerun}
    report['multiplicity'] = {'family': 'every new interval of part E (new streams); offset-0 cells recomputed in the '
                                        'seed-contrast stream reproduce published intervals and are excluded, as are '
                                        'per-fold values',
                              'q': 0.05}
    p = [e['p'] for e in entries]
    bh, by = step_up(p), step_up(p, dependence_robust=True)
    for e, r1, r2 in zip(entries, bh, by):
        e['bh'] = bool(r1 and e['excludes_zero'])
        e['by'] = bool(r2 and e['excludes_zero'])
    report['multiplicity'].update({'intervals': len(entries), 'exclude_zero': sum(e['excludes_zero'] for e in entries),
                                   'bh_retained': sum(e['bh'] for e in entries),
                                   'by_retained': sum(e['by'] for e in entries), 'entries': entries})
    gp = {'E-G0': gates['E-G0']['pass'], 'E-G1': all(v['pass'] for v in gates['E-G1'].values()) and len(refits) == 7,
          'E-G2': all(v['pass'] for v in gates['E-G2'].values()) and len(refits) == 7,
          'E-G3': bool(gates['E-G3'] and gates['E-G3']['pass']),
          'E-G4': {k: v['pass'] for k, v in gates['E-G4'].items()},
          'E-G5': all(v['pass'] for v in gates['E-G5'].values())}
    report['gates_pass'] = gp
    part_ok = gp['E-G0'] and gp['E-G3'] and gp['E-G5']
    report['readable'] = {cid: bool(part_ok and gp['E-G4'][cid] and report['contrasts'][cid]['refit_gates_pass'])
                          for cid in CONTRASTS}
    for cid, ok in report['readable'].items():
        if not ok:
            report['contrasts'][cid]['status'] = 'NOT READ: a gate failed'
    write_json(OUT, report)
    print('wrote', OUT.relative_to(ROOT).as_posix(), '| gates:', json.dumps(gp))
    print('multiplicity:', {k: v for k, v in report['multiplicity'].items() if k != 'entries'})


if __name__ == '__main__':
    main()
