"""Addendum 69 (review R3): the declared gates G1-G4, read from the per-fold and inner-prior runs.

  G1  each per-fold mode reproduces the shared-selection column or rows when given the shared
      configuration: the sequence tree (max difference 0, recorded by each run), and the spatial
      comparators (the refitted per-fold model against addendum 47's per-fold rows)
  G2  at six hours the per-fold rollout equals the per-fold open-loop weather tree on exactly the
      episodes where the shared rollout equals the shared tree (recorded by v2_closed_loop
      --perfold), and the gate rollout's ungated six-hour step equals the per-fold rollout's
      (v2_physics_gate --six-hour-check --perfold)
  G3  under the inner-split prior A2 reproduces its development study (v2_field_reserved2026
      --inner-prior), and the A7 objective at the published penalty, computed here on the full
      protocol, is recorded
  G4  every selection a per-fold or inner-prior run read or made is a completed study of at least 100
      trials, not a smoke run, and the file behind each digest still has that digest

Reads only; writes runs/v2_a69_gates.json.
Usage: python -m scripts.v2_a69_gates
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_a69_gates.json'
VARIABLES = ('temperature', 'oxygen')
OFFSETS = range(6)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


G4 = []


def record(rec, where):
    """One selection record of G4: its file, trial count, smoke flag and digest."""
    path = ROOT / rec['source']
    sel = load(path)
    entry = {'where': where, 'source': rec['source'], 'completed_trials': int(sel['completed_trials']),
             'smoke_only': bool(sel.get('smoke_only')), 'sha256_matches': sha(path) == rec['sha256']}
    entry['pass'] = entry['completed_trials'] >= 100 and not entry['smoke_only'] and entry['sha256_matches']
    G4.append(entry)


def study(sel, where, path=None):
    """G4 for a study a run made itself (its selection dict as stored in the report)."""
    entry = {'where': where, 'study': sel['study'], 'completed_trials': int(sel['completed_trials']),
             'smoke_only': bool(sel.get('smoke_only'))}
    if path is not None:
        entry['source'] = Path(path).relative_to(ROOT).as_posix()
        entry['matches_file'] = load(path)['params'] == sel['params']
    entry['pass'] = entry['completed_trials'] >= 100 and not entry['smoke_only'] and entry.get('matches_file', True)
    G4.append(entry)


def sequence():
    out = {}
    for v in VARIABLES:
        r = load(RUNS / f'v2_sequence_arms/perfold_tree/sequence_arms_{v}.json')
        out[f'arms|{v}'] = r['gate_G1_shared_tree_replay']
        for k, rec in r['tree_selection']['records'].items():
            record(rec, f'sequence arms {v} fold {k}')
        for o in OFFSETS:
            p = load(RUNS / f'v2_sequence_sweep/panel_perfold/panel_{v}_offset{o}.json')
            out[f'sweep|{v}|offset{o}'] = p['gate_G1_shared_tree_replay']
            for k, rec in p['tree_selection']['records'].items():
                record(rec, f'sweep {v} offset {o} {k}')
    return out


def rollout():
    out = {}
    for v, name in (('temperature', 'closed_loop_temperature'), ('oxygen', 'closed_loop_oxygen_rolled_neighbour')):
        s = load(RUNS / f'v2_closed_loop/perfold/{name}_summary.json')
        out[f'closed_loop|{v}'] = s['gate_open_loop']
        for arm, folds in s['selection']['records'].items():
            for k, rec in folds.items():
                record(rec, f'closed loop {v} {arm} fold {k}')
    checks = load(RUNS / 'v2_physics_gate/perfold/issuance_gate.json')
    for v in VARIABLES:
        c = checks[f'{v}|six_hour_vs_closed_loop']
        out[f'six_hour|{v}'] = {**c, 'pass': bool(c['rows_above_1e-9'] == 0
                                                  and c['episodes_compared'] == c['episodes_in_closed_loop'])}
    for v in VARIABLES:
        for regime in ('single', 'network'):
            for suffix in ('', '_issuance'):
                s = load(RUNS / f'v2_physics_gate/perfold/gate_{v}_{regime}{suffix}_summary.json')
                for arm, folds in s['selection']['records'].items():
                    for k, rec in folds.items():
                        record(rec, f'gate {v} {regime}{suffix} fold {k}')
    return out


def spatial():
    naive = load(RUNS / 'v2_field_naive/perfold/field_naive_summary.json')
    rep = naive['reproduction_check_model_vs_released_run']
    gap = max(abs(a - b) for a, b in rep.values())
    for k, rec in naive['selection']['model'].items():
        record(rec, f'field naive A1 fold {k}')
    return {'naive_model_vs_perfold_A1_rows': {'nested_mae_max_abs_diff': gap, 'pass': bool(gap <= 1e-9),
                                               'reference': naive['reproduction_source']}}


def inner_prior():
    out = {}
    for arm in ('A6', 'A7'):
        folder = ROOT / f'results/v2_field_held_site_{arm}_ctdprior_direct_innerprior'
        r = load(folder / 'report.json')
        study(r['selection'], f'v2_field_ctdprior {arm} inner prior',
              folder / f"{r['selection']['study']}_selection.json")
    for base, arm in (('v2_perfold', 'A7_ctdprior_direct'), ('v2_perfold_rest', 'A6_ctdprior_direct')):
        for design in ('shared', 'perfold'):
            r = load(RUNS / base / 'field_innerprior' / design / arm / 'report.json')
            sels = r['selection'] if design == 'perfold' else {'all': r['selection']}
            for k, s in sels.items():
                study(s, f'{base} {arm} {design} {k} inner prior',
                      RUNS / base / 'field_innerprior' / design / arm / f"{s['study']}_selection.json")
    r = load(RUNS / 'v2_field_reserved2026/innerprior/report.json')
    for a in ('A2_none', 'A7o'):
        s = r['development'][a]['selection']
        study(s, f'Field-2026 development {a} inner prior',
              RUNS / 'v2_field_reserved2026/innerprior/studies' / f"{s['study']}_selection.json")
    out['A2_reproduces_development_study'] = bool(r['development']['A2_reproduces_published'])
    # the A7 inner objective at the published penalty, on the full protocol
    from src.experiments import v2_perfold as P47
    from src.experiments.v2_field import inner_score
    bundles, _, mats, c = P47.field_setup(['A7_ctdprior_direct'], inner_prior=True)
    arm = 'A7_ctdprior_direct'
    path = ROOT / 'results/v2_field_held_site_A7_ctdprior_direct/held_site_A7_ctdprior_direct_selection.json'
    alpha = load(path)['params']['alpha']
    shared_prior = float(np.mean([inner_score(b, m, alpha, False, c['stations'], c['casts'], c['truth'])
                                  for b, m in zip(bundles, mats[arm])]))
    split_prior = float(np.mean([inner_score(b, m, alpha, False, c['stations'], c['casts'], c['truth'], split_mats=sm)
                                 for b, m, sm in zip(bundles, mats[arm], c['split_mats'][arm])]))
    out['A7_objective_at_published_penalty'] = {
        'alpha': alpha, 'published_best_value': load(path)['best_value'],
        'prior_from_all_training_stations': shared_prior, 'prior_rebuilt_per_inner_split': split_prior,
        'reproduces_published_objective': bool(abs(shared_prior - load(path)['best_value']) <= 1e-12)}
    out['prior_stations_are_inner_training_stations'] = all(
        s['prior_stations_are_inner_training_stations'] for x in c['split_coverage'][arm] for s in x)
    return out


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum69.json',
              'G1_and_G2_sequence': sequence(), 'G2_rollout_and_gates': rollout(), 'G1_spatial': spatial(),
              'G3_inner_prior': inner_prior()}
    report['G4'] = {'records': G4, 'checked': len(G4), 'pass': all(e['pass'] for e in G4)}
    passes = {
        'G1': all(g['pass'] for k, g in report['G1_and_G2_sequence'].items())
        and report['G1_spatial']['naive_model_vs_perfold_A1_rows']['pass'],
        'G2': all(g.get('passed', g.get('pass')) for g in report['G2_rollout_and_gates'].values()),
        'G3': bool(report['G3_inner_prior']['A2_reproduces_development_study']
                   and report['G3_inner_prior']['A7_objective_at_published_penalty']['reproduces_published_objective']),
        'G4': report['G4']['pass']}
    report['pass'] = passes
    OUT.write_bytes(json.dumps(report, indent=1, allow_nan=False).encode('utf-8'))
    print(json.dumps(passes), f"G4 records {len(G4)}")
    print(json.dumps(report['G3_inner_prior']['A7_objective_at_published_penalty']))
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
