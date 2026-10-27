"""How much of the missing record is recoverable, and over what gap lengths?

A feasibility probe, run before any protocol is written, to answer a question that decides whether
gap recovery is worth a thesis chapter: is there a band of gap lengths where a model can do
something a trivial rule cannot?

Three facts already bound the answer and are worth stating before any model runs. Sixty per cent of
gaps are one hour and ninety-three per cent are six hours or less, so most *gaps* are short. But
only a third of *lost hours* sit in gaps of six hours or less, because a handful of multi-month
outages dominate the total: the longest temperature gap is 3237 hours. And co-located probes
disagree by 0.20 to 0.53 C over months, so any method that leans on a neighbour inherits that as a
floor.

Method. Real gaps have no ground truth, so nothing is scored on them. Instead, stretches that are
fully observed are masked to look like real gaps, drawn at the lengths the record actually
produces, and the hidden values are recovered and scored against what was really there. Only
baselines appear here, all parameter-free: the point is the shape of the curve against gap length,
not which model wins.

    persistence       hold the last value before the gap (what the outage fallback does today)
    linear            interpolate between the two edges, which a real-time system cannot do
    climatology       the month-by-hour mean, fitted outside the masked stretch
    nearest probe     the contemporaneous reading at the closest working probe, offset-corrected

Usage: python -m scripts.v2_gap_feasibility
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT, load_observations

OUT = ROOT / 'reports/bay_twin/runs/v2_gap_feasibility.json'
LENGTHS = [1, 2, 3, 6, 12, 18, 24, 48, 72, 168]
PER_LENGTH = 400          # masked episodes per (variable, gap length)
MARGIN = 72               # observed hours required either side, so every method has context
SEED = 20260922


def series_frame(variable):
    """One column per probe on a common hourly index, unfilled."""
    meta, frames = load_observations()
    names = [n for n, f in frames.items() if variable in f and f[variable].notna().any()]
    index = pd.date_range(min(frames[n].index.min() for n in names),
                          max(frames[n].index.max() for n in names), freq='h', tz='UTC')
    data = pd.DataFrame({n: frames[n][variable].reindex(index) for n in names}, index=index)
    return meta.loc[names], data


def episodes(column, length, rng, wanted=PER_LENGTH):
    """Start positions where `length` hours plus margins either side are all observed."""
    ok = np.isfinite(column.to_numpy())
    n = len(ok)
    need = MARGIN + length + MARGIN
    if n < need:
        return []
    # A run of `need` consecutive observed hours, found by a cumulative sum rather than a scan.
    cum = np.concatenate([[0], np.cumsum(ok)])
    full = (cum[need:] - cum[:-need]) == need
    starts = np.flatnonzero(full)
    if not len(starts):
        return []
    picked = rng.choice(starts, size=min(wanted, len(starts)), replace=False)
    return [int(s) + MARGIN for s in picked]


def recover(values, start, length, neighbour, month_hour):
    """The four baselines over one masked stretch. Returns name -> predicted array."""
    before = values[start - 1]
    after = values[start + length]
    steps = np.arange(1, length + 1)
    out = {
        'persistence': np.full(length, before),
        'linear': before + (after - before) * steps / (length + 1),
        'climatology': month_hour,
    }
    if neighbour is not None:
        # Offset-corrected: the neighbour's level differs, its variation is what carries.
        ref = neighbour[start - 1]
        out['nearest_probe'] = (neighbour[start:start + length] - ref + before
                                if np.isfinite(ref) else neighbour[start:start + length])
    return out


def run(variable, rng):
    meta, data = series_frame(variable)
    xy = meta[['x_m', 'y_m']].to_numpy()
    order = {n: [j for j in np.argsort(np.linalg.norm(xy - xy[i], axis=1), kind='stable') if j != i]
             for i, n in enumerate(data.columns)}
    hour = data.index.hour.to_numpy()
    month = data.index.month.to_numpy()

    scores = defaultdict(lambda: defaultdict(list))
    counts = {}
    for length in LENGTHS:
        made = 0
        for name in data.columns:
            column = data[name]
            values = column.to_numpy(float)
            starts = episodes(column, length, rng, wanted=max(PER_LENGTH // len(data.columns), 8))
            if not starts:
                continue
            # Climatology fitted on this probe outside the masked stretch, by month and hour.
            table = pd.DataFrame({'v': values, 'm': month, 'h': hour}).dropna()
            grid = table.groupby(['m', 'h']).v.mean()
            for start in starts:
                idx = slice(start, start + length)
                truth = values[idx]
                if not np.isfinite(truth).all():
                    continue
                key = list(zip(month[idx], hour[idx]))
                clim = np.array([grid.get(k, np.nan) for k in key])
                neighbour = None
                for other in order[name]:
                    candidate = data.iloc[:, other].to_numpy(float)
                    if np.isfinite(candidate[start - 1:start + length + 1]).all():
                        neighbour = candidate
                        break
                for method, pred in recover(values, start, length, neighbour, clim).items():
                    err = np.abs(pred - truth)
                    if np.isfinite(err).all():
                        scores[length][method].append(float(err.mean()))
                made += 1
        counts[length] = made

    table = {}
    for length in LENGTHS:
        row = {}
        for method, values in scores[length].items():
            if values:
                row[method] = {'mae': round(float(np.mean(values)), 4), 'episodes': len(values)}
        table[str(length)] = row
    return {'episodes_attempted': counts, 'by_gap_length': table,
            'probes': int(len(data.columns))}


def main():
    rng = np.random.default_rng(SEED)
    report = {'purpose': 'feasibility probe for gap recovery; no protocol, no tuning',
              'masking': f'{MARGIN} h observed either side; real gaps are never scored',
              'gap_lengths_hours': LENGTHS, 'seed': SEED, 'variables': {}}
    for variable in ['temperature', 'oxygen']:
        report['variables'][variable] = run(variable, rng)
        print(f'\n== {variable}')
        block = report['variables'][variable]['by_gap_length']
        methods = ['persistence', 'linear', 'climatology', 'nearest_probe']
        print(f"{'gap h':>6}  " + '  '.join(f'{m:>14}' for m in methods) + '   episodes')
        for length in LENGTHS:
            row = block[str(length)]
            if not row:
                continue
            cells = '  '.join(f"{row[m]['mae']:>14.4f}" if m in row else f"{'-':>14}"
                              for m in methods)
            n = max((row[m]['episodes'] for m in row), default=0)
            print(f'{length:>6}  {cells}   {n}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
