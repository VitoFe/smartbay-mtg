"""Addendum 61: the Arno arms with the lag searched up to seven days, against the same arms searched up to
four days and against the plain and weather trees, per design.

Reads runs/v2_arno_lag7/forecast/{shared,perfold}/riverlag_<variable>_<arm>_h<h>/ (written by
src/experiments/v2_arno_lag7.py), the same cells on the 0-96 h grid (runs/v2_perfold_rest/forecast/),
and the plain and weather trees of addendum 47 (runs/v2_perfold/forecast/), each under the matching
design. Every contrast is paired on identical (fold, sensor, time) rows with the forecast producers'
(fold, ISO week) block bootstrap (scripts/v2_perfold_rest_contrasts.Blocks), 600 draws, generator
[20261002, crc32('a61|<cell>|<design>|<contrast>')]. Oxygen is also given in mg/l (stored value x c).
Evaluates the declared predictions P1-P3.

  python -m scripts.v2_arno_lag7_contrasts   ->  reports/bay_twin/runs/v2_arno_lag7.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_perfold_rest_contrasts import Blocks
from src.units import factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
NEW = RUNS / 'v2_arno_lag7/forecast'
OLD = RUNS / 'v2_perfold_rest/forecast'
TREES = RUNS / 'v2_perfold/forecast'
OUT = RUNS / 'v2_arno_lag7.json'
DECLARED = 'config/bay_reserved_protocol_v2_addendum61.json'
KEY = ['fold', 'sensor', 'time']
SEED = 20261002
ARMS = ('bocca_lag', 'bocca_gated')
VARIABLES = ('temperature', 'oxygen')
HORIZONS = (6, 24)
DESIGNS = ('shared', 'perfold')
FLOOR = {'temperature': 0.029, 'oxygen_mgl': 0.059}


def rows(path, name):
    d = pd.read_csv(path)
    return d[KEY + ['truth', 'prediction']].rename(columns={'prediction': name, 'truth': f'truth_{name}'})


def contrast(blocks, a, b, stream):
    point, lo, hi = blocks.boot({a: 1.0, b: -1.0}, SEED, stream)
    return {'value': [point, lo, hi], 'excludes_zero': bool(lo > 0 or hi < 0)}


def main():
    c = factor()
    out = {'declared_in': DECLARED, 'seed': SEED, 'draws': 600, 'cells': {}}
    for v in VARIABLES:
        for arm in ARMS:
            for h in HORIZONS:
                cell = f'{v}_{arm}_h{h}'
                entry = {}
                for design in DESIGNS:
                    folder = NEW / design / f'riverlag_{cell}'
                    report = json.loads((folder / 'report.json').read_text())
                    frame = rows(folder / 'selected_predictions.csv.gz', 'arno7')
                    for name, path in (('arno4', OLD / design / f'riverlag_{cell}' / 'selected_predictions.csv.gz'),
                                       ('tree', TREES / design / f'{v}_tree_h{h}' / 'selected_predictions.csv.gz'),
                                       ('weather_tree', TREES / design / f'{v}_weather_tree_h{h}' / 'selected_predictions.csv.gz')):
                        frame = frame.merge(rows(path, name), on=KEY, how='inner', validate='one_to_one')
                    truth = frame[[col for col in frame if col.startswith('truth_')]].to_numpy()
                    assert np.allclose(truth, truth[:, :1], atol=1e-9), f'{cell} {design}: truths differ'
                    frame['truth'] = frame['truth_arno7']
                    frame = frame.sort_values(['fold', 'time', 'sensor']).reset_index(drop=True)
                    blocks = Blocks(frame)
                    mae = {k: blocks.value({k: 1.0}) for k in ('arno7', 'arno4', 'tree', 'weather_tree')}
                    res = {'rows': int(len(frame)),
                           'selected_lag_hours': report.get('selected_lag_hours'),
                           'mae': mae,
                           'contrasts': {
                               'arno7 - arno4': contrast(blocks, 'arno7', 'arno4', f'a61|{cell}|{design}|arno7-arno4'),
                               'arno7 - tree': contrast(blocks, 'arno7', 'tree', f'a61|{cell}|{design}|arno7-tree'),
                               'arno7 - weather_tree': contrast(blocks, 'arno7', 'weather_tree',
                                                                f'a61|{cell}|{design}|arno7-weather')}}
                    if v == 'oxygen':
                        res['mae_mgl'] = {k: m * c for k, m in mae.items()}
                        for k, val in res['contrasts'].items():
                            val['value_mgl'] = [x * c for x in val['value']]
                    entry[design] = res
                old_lags = {d: json.loads((OLD / d / f'riverlag_{cell}' / 'report.json').read_text()).get('selected_lag_hours')
                            for d in DESIGNS}
                entry['selected_lag_hours_0_96'] = old_lags
                out['cells'][cell] = entry
    # the declared predictions
    per = {cell: e['perfold'] for cell, e in out['cells'].items()}
    apart = [cell for cell, r in per.items()
             if abs(r['selected_lag_hours']['0'] - r['selected_lag_hours']['1']) > 24]
    beats_weather = [cell for cell, r in per.items()
                     if r['contrasts']['arno7 - weather_tree']['excludes_zero']
                     and r['contrasts']['arno7 - weather_tree']['value'][0] < 0]
    ext_moves = [cell for cell, r in per.items() if r['contrasts']['arno7 - arno4']['excludes_zero']]
    lags = [lag for e in out['cells'].values() for d in DESIGNS
            for lag in (e[d]['selected_lag_hours'] or {}).values()]
    out['selected_lags_all'] = {'count': len(lags), 'beyond_96h': int(sum(l > 96 for l in lags)), 'values': lags}
    out['predictions'] = {
        'P1': {'statement': 'in at least four of the eight cells the two per-fold studies select lags more '
                            'than 24 h apart', 'cells_apart': apart, 'held': len(apart) >= 4},
        'P2': {'statement': 'no Arno arm on the 0-168 h grid beats the weather tree per fold with an interval '
                            'excluding zero', 'cells_beating': beats_weather, 'held': not beats_weather},
        'P3': {'statement': 'arm(0-168) - arm(0-96) per fold spans zero in every cell',
               'cells_excluding_zero': ext_moves, 'held': not ext_moves}}
    OUT.write_bytes((json.dumps(out, indent=1, allow_nan=False) + '\n').encode('utf-8'))
    for cell, e in out['cells'].items():
        r = e['perfold']
        unit = c if cell.startswith('oxygen') else 1.0
        s = ', '.join(f"{k} {x['value'][0] * unit:+.5f} [{x['value'][1] * unit:+.5f}, {x['value'][2] * unit:+.5f}]"
                      + ('*' if x['excludes_zero'] else '') for k, x in r['contrasts'].items())
        print(f"{cell:30s} lags {r['selected_lag_hours']} (0-96: {e['selected_lag_hours_0_96']['perfold']})  {s}")
    print(json.dumps(out['predictions'], indent=1))
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
