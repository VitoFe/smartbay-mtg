"""Addendum-13: does depth separation explain the co-located disagreement?

Section 3.7.3 claimed it does, on the strength of a 0.051 degC cast-against-sonde control, and
that claim was never tested. It is testable directly: if depth separation drives the 0.20 to
0.53 degC spread between co-located probes, then pairs at the SAME nominal depth must agree far
better than pairs at different depths.

The split by instrument class is reported beside it, because the 0.051 degC control is an EXO2
against a CTD, both reporting at 0.01 degC, and most co-located pairs are WNode against WNode at
0.1 degC, where the quantisation of the difference alone contributes about 0.04 degC.

Declared in config/bay_reserved_protocol_v2_addendum13.json.
Usage: python -m src.experiments.v2_colocated_depth
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT

OUT = ROOT / 'reports/bay_twin/runs/v2_colocated_depth.json'
SEPARATION_M = 60.0      # the radius Table 3.2 treats as co-located
MIN_OVERLAP_H = 50


def instrument_class(name):
    return 'WNode' if name.startswith('Nodo_') else 'EXO2'


def main():
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    series = {}
    for name in meta.index:
        _, frame, _ = read_export(ROOT / 'data/raw' / f'{name}.txt')
        series[name] = frame['temperature']

    rows = []
    for a, b in combinations(meta.index, 2):
        dx = np.hypot(meta.loc[a].x_m - meta.loc[b].x_m, meta.loc[a].y_m - meta.loc[b].y_m)
        if dx > SEPARATION_M:
            continue
        joined = pd.concat([series[a].rename('a'), series[b].rename('b')], axis=1).dropna()
        if len(joined) < MIN_OVERLAP_H:
            continue
        d = joined.a - joined.b
        depth_a, depth_b = float(meta.loc[a].nominal_depth_m), float(meta.loc[b].nominal_depth_m)
        pair = tuple(sorted([instrument_class(a), instrument_class(b)]))
        rows.append({'a': a, 'b': b, 'separation_m': round(float(dx), 2),
                     'depth_a_m': depth_a, 'depth_b_m': depth_b,
                     'depth_separation_m': round(abs(depth_a - depth_b), 2),
                     'same_depth': abs(depth_a - depth_b) < 1e-9,
                     'classes': '-'.join(pair), 'overlap_h': int(len(joined)),
                     'mad': round(float(d.abs().mean()), 4),
                     'median_abs': round(float(d.abs().median()), 4),
                     'mean_signed': round(float(d.mean()), 4)})

    frame = pd.DataFrame(rows).sort_values(['same_depth', 'mad'], ascending=[False, True])
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum13.json',
              'separation_radius_m': SEPARATION_M, 'pairs': len(frame),
              'by_pair': frame.to_dict('records'), 'groups': {}}

    def summarise(sub, label):
        if not len(sub):
            return None
        return {'pairs': int(len(sub)), 'mad_min': float(sub.mad.min()),
                'mad_max': float(sub.mad.max()), 'mad_mean': round(float(sub.mad.mean()), 4),
                'overlap_h_total': int(sub.overlap_h.sum()), 'label': label}

    report['groups']['same_depth'] = summarise(frame[frame.same_depth], 'identical nominal depth')
    report['groups']['different_depth'] = summarise(frame[~frame.same_depth], 'different depth')
    for cls, sub in frame.groupby('classes'):
        report['groups'][f'class_{cls}'] = summarise(sub, cls)
        for same in (True, False):
            s = sub[sub.same_depth == same]
            if len(s):
                report['groups'][f'class_{cls}_{"same" if same else "diff"}_depth'] = summarise(
                    s, f'{cls}, {"same" if same else "different"} depth')

    OUT.write_text(json.dumps(report, indent=2))

    pd.set_option('display.width', 200)
    print(frame[['a', 'b', 'separation_m', 'depth_a_m', 'depth_b_m', 'same_depth', 'classes',
                 'overlap_h', 'mad', 'median_abs']].to_string(index=False))
    print()
    for key, g in report['groups'].items():
        if g:
            print(f'  {key:34s} n={g["pairs"]:2d}  MAD {g["mad_min"]:.4f}-{g["mad_max"]:.4f}  '
                  f'mean {g["mad_mean"]:.4f}  overlap {g["overlap_h_total"]:6d} h')
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
