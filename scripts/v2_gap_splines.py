"""The spline baselines, computed on the same episodes the arms were scored on.

`v2_gap_recovery` shipped one spline, an unconstrained cubic through the observed margins, and it
diverges: 280 umol/l across a thirty-day oxygen gap, on a signal that lives near 250. That is Runge
oscillation rather than a bug, and it is the reason nobody fills a long gap with a plain cubic. But
reporting only the diverging version would misrepresent what "spline interpolation" means in
practice, where the shape-preserving PCHIP is the usual choice because it cannot overshoot between
knots.

Both are computed here, along with a cubic restricted to a short window either side, which
separates the two things the plain cubic gets wrong: the polynomial degree, and the length of the
interval it has to span.

`build` is deterministic given its seed, so these land on exactly the episodes the arms were scored
on and can be read beside them.

Usage: python -m scripts.v2_gap_splines
"""
import json

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from src.data.bay_profile_context import load_observations
from src.experiments.v2_gap_recovery import MARGIN, OUT, SEED, build

REPORT = OUT.parent / 'v2_gap_splines.json'
#: knots either side for the "short window" variant; the full one uses all MARGIN hours
SHORT = 12


def variants(frame, values, names, index):
    column = {n: i for i, n in enumerate(names)}
    positions = {t: i for i, t in enumerate(index)}
    out = {k: np.full(len(frame), np.nan) for k in ('pchip', 'cubic_short', 'pchip_short')}
    for key, block in frame.groupby('episode', sort=False):
        sensor, start, length = key.rsplit('|', 2)
        length = int(length)
        g0 = positions[pd.Timestamp(start)]
        i = column[sensor]
        rows = frame.index.get_indexer(block.index)
        target = g0 + block.position.to_numpy(int) - 1
        for label, width, maker in (('pchip', MARGIN, PchipInterpolator),
                                    ('cubic_short', SHORT, CubicSpline),
                                    ('pchip_short', SHORT, PchipInterpolator)):
            left = np.arange(max(0, g0 - width), g0)
            right = np.arange(g0 + length, min(len(index), g0 + length + width))
            knots = np.concatenate([left, right])
            y = values[knots, i]
            ok = np.isfinite(y)
            if ok.sum() < 6:
                continue
            try:
                f = maker(knots[ok], y[ok], extrapolate=False)
            except ValueError:
                continue
            out[label][rows] = f(target)
    return out


def score(frame, column):
    d = frame.dropna(subset=[column, 'truth'])
    if not len(d):
        return float('nan')
    return float(d.assign(e=(d.truth - d[column]).abs()).groupby('episode').e.mean().mean())


def main():
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum30.json',
              'note': 'spline variants on the episodes v2_gap_recovery scored', 'variables': {}}
    for variable in ['temperature', 'oxygen']:
        rng = np.random.default_rng(SEED)
        frame = build(variable, rng)
        meta, frames = load_observations()
        names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
        index = pd.date_range(min(frames[n].index.min() for n in names),
                              max(frames[n].index.max() for n in names), freq='h', tz='UTC')
        values = pd.DataFrame({n: frames[n][variable].reindex(index)
                               for n in names}, index=index).to_numpy(float)
        for label, column in variants(frame, values, names, index).items():
            frame[label] = column
        columns = ['linear', 'spline', 'pchip', 'cubic_short', 'pchip_short', 'nearest_probe']
        block = {'overall': {c: round(score(frame, c), 4) for c in columns},
                 'by_gap_length': {}}
        for length, part in frame.groupby('gap_length'):
            block['by_gap_length'][str(int(length))] = {
                c: round(score(part, c), 4) for c in columns}
        report['variables'][variable] = block

        print(f'\n== {variable}')
        print(f"{'h':>6} " + ' '.join(f'{c[:12]:>12}' for c in columns))
        for length, row in block['by_gap_length'].items():
            print(f'{length:>6} ' + ' '.join(f'{row[c]:12.3f}' for c in columns))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print('\nwrote', REPORT)


if __name__ == '__main__':
    main()
