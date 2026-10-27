"""Addendum 76: the conservation constraint isolated under feedback.

The per-fold closed-loop rollouts (v2_closed_loop --perfold) roll the conservation-constrained graph
operator in their graph_denied column. `v2_closed_loop --perfold --graph-arm learned_source` rolls the
unconstrained operator through the identical episodes into runs/v2_closed_loop_learned_source/perfold/.
This sets the two against each other, and the unconstrained one against the tree, lead by lead, with the
onset-week block bootstrap of scripts/v2_closed_loop_leads.

Gate GF: the new run's recursive_tree and persistence columns equal the published ones on every row.

Usage: python -m scripts.v2_closed_loop_constraint
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_closed_loop_leads import macro, week_blocks

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_closed_loop_constraint.json'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum76.json'
FILES = {'temperature': 'closed_loop_temperature.csv.gz', 'oxygen': 'closed_loop_oxygen_rolled_neighbour.csv.gz'}
KEY = ['fold', 'sensor', 'time', 'lead']
CONTRASTS = (('graph_conservative', 'graph_unconstrained'), ('graph_unconstrained', 'recursive_tree'),
             ('graph_conservative', 'recursive_tree'))


def main():
    rep = {'declared_in': DECLARATION, 'gate_GF': {}, 'variables': {}}
    for variable, name in FILES.items():
        pub = pd.read_csv(RUNS / 'v2_closed_loop/perfold' / name, parse_dates=['time']).sort_values(KEY).reset_index(drop=True)
        new = pd.read_csv(RUNS / 'v2_closed_loop_learned_source/perfold' / name, parse_dates=['time']).sort_values(KEY).reset_index(drop=True)
        same_rows = pub[KEY].equals(new[KEY])
        worst = {c: float(np.nanmax(np.abs(pub[c].to_numpy() - new[c].to_numpy()))) if same_rows else None
                 for c in ('recursive_tree', 'persistence', 'truth')}
        passed = bool(same_rows and all(v == 0.0 for v in worst.values()))
        rep['gate_GF'][variable] = {'same_rows': bool(same_rows), 'max_abs_difference': worst, 'passed': passed}
        if not passed:
            continue
        frame = pub.rename(columns={'graph_denied': 'graph_conservative'})
        frame['graph_unconstrained'] = new.graph_denied.to_numpy()
        iso = frame.time.dt.isocalendar()
        frame['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
        scale = 1.0
        if variable == 'oxygen':
            from src.units import factor
            scale = factor()
        by_lead = {}
        for lead, sub in frame.groupby('lead'):
            entry = {'mae': {a: round(scale * macro(sub, a), 4)
                             for a in ('recursive_tree', 'graph_conservative', 'graph_unconstrained', 'persistence')},
                     'contrasts': {}}
            for x, y in CONTRASTS:
                c = week_blocks(sub, x, y, f'constraint|{variable}|{lead}|{x}|{y}')
                entry['contrasts'][f'{x} - {y}'] = {'point': round(scale * c['point'], 4),
                                                    'ci': [round(scale * v, 4) for v in c['ci']],
                                                    'excludes_zero': c['excludes_zero']}
            by_lead[str(int(lead))] = entry
        rep['variables'][variable] = {'unit': 'mg/l' if variable == 'oxygen' else 'degC', 'by_lead': by_lead}
    v = rep['variables']
    if len(v) == 2:
        rep['predictions'] = {
            'PF1': not any(e['contrasts']['graph_conservative - graph_unconstrained']['excludes_zero']
                           and e['contrasts']['graph_conservative - graph_unconstrained']['point'] < 0
                           for b in v.values() for e in b['by_lead'].values()),
            'PF2': all(e['contrasts']['graph_unconstrained - recursive_tree']['point'] > 0
                       and e['contrasts']['graph_unconstrained - recursive_tree']['excludes_zero']
                       for e in v['temperature']['by_lead'].values())}
    OUT.write_bytes(json.dumps(rep, indent=1).encode('utf-8'))
    print(json.dumps(rep['gate_GF']))
    for var, b in v.items():
        for lead, e in b['by_lead'].items():
            print(var, lead, e['mae'], {k: (c['point'], c['ci'], '*' if c['excludes_zero'] else '') for k, c in e['contrasts'].items()})
    print(rep.get('predictions'))


if __name__ == '__main__':
    main()
