"""Every month-long test on the Baia EXO2 sonde: the real record against the best fill and linear.

A single example trace can mislead, and at 720 h there are only sixteen oxygen tests on this probe,
so none is chosen: every one is drawn, in date order. The real record is black, including the
masked month that the methods did not see; real short gaps inside it show as breaks. Drawn
against linear interpolation, from the addendum-33 predictions: for oxygen, bidirectional
Chronos-Bolt (the confirmed incumbent) and Chronos-2 with the nearest probes as covariates (best on
the development record, not confirmed on 2026); for temperature, the equal mean of the Gaussian
process and Chronos-2 with neighbours, the best filler confirmed in both years for a probe without a
twin. Fills are scored at 48 evenly spread hours, drawn as dots joined by a thin line.

Usage: python -m scripts.v2_gap_long_traces
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v2_timeseries_figures import BAND, BLUE, DECK, INK, MUTED, ORANGE, ROOT, THESIS
from src.data.bay_observations import read_export
from src.units import OXYGEN_UNIT_TEXT, factor, mg_decimals

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_long_traces.json'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
LENGTH, MARGIN = 720, 72
#: addendum-33 per-row predictions, which carry the ensembles as well as the single methods
COHORTS = {'development': RUNS / 'v2_gap_variants/Gap-Long-Dev__{variable}__168-1440.csv.gz',
           '2026': RUNS / 'v2_gap_variants/Gap-Long-2026__{variable}__168-1440.csv.gz'}
#: What each figure draws, first in blue: for oxygen the confirmed incumbent and the best method on
#: the development record, which 2026 did not confirm; for temperature the confirmed best without a
#: twin (this sonde has none), the equal mean of the Gaussian process and Chronos-2 with neighbours.
BEST = {'oxygen': [('chronos_bidir', 'Chronos-Bolt, bidirectional (B)', 'B'),
                   ('chronos2_cov_bidir', 'Chronos-2 with neighbours (C2)', 'C2')],
        'temperature': [('ens_mean_c2', 'mean of GPR and Chronos-2 with neighbours (best)',
                         'best')]}
UNITS = {'oxygen': OXYGEN_UNIT_TEXT, 'temperature': '°C'}
FLOOR = {'oxygen': f'0.059 {OXYGEN_UNIT_TEXT}', 'temperature': '0.20 °C'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': .6,
                     'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.frameon': False})


def windows(variable):
    """Every 720 h window on the sonde, both cohorts, with the drawn methods and linear."""
    out = []
    for cohort, path in COHORTS.items():
        frame = pd.read_csv(str(path).format(variable=variable))
        frame = frame[(frame.sensor == EXO2) & (frame.gap_length == LENGTH)]
        for episode, rows in frame.groupby('episode'):
            rows = rows.sort_values('position')
            out.append({'cohort': cohort, 'episode': episode,
                        'start': pd.Timestamp(episode.rsplit('|', 2)[1]), 'rows': rows})
    return sorted(out, key=lambda w: w['start'])


def mae(rows, column):
    return float((rows.truth - rows[column]).abs().mean())


def figure(variable, record):
    drawn = BEST[variable]
    arm = drawn[0][0]
    series = read_export(ROOT / 'data/raw' / (EXO2 + '.txt'), include_auxiliary=True)[1][variable]
    tests = windows(variable)
    cols = 4
    rows_n = int(np.ceil(len(tests) / cols))
    # The summary stays in the artifacts' units; oxygen is drawn and printed in mg/l.
    scale = factor() if variable == 'oxygen' else None
    digits = 1 if variable == 'oxygen' else 2
    # In mg/l the oxygen errors print with three figures, and one header line holding all three
    # methods is wider than a column: linear interpolation takes a third line, and each row grows by
    # part of that line (the whole line made the float taller than the supplement's page).
    split = scale is not None
    joint = '\n' if split else ' · '
    #: Sized for the thesis text block (5.8 in wide, 8.2 in tall with room for the caption), so that
    #: the figure is set at full width without a height cap and no label falls below 7 pt.
    row_height = (1.75 if rows_n <= 4 else 1.7) + (.08 if split else 0)
    fig, axes = plt.subplots(rows_n, cols, figsize=(9.0, row_height * rows_n + 1.0), squeeze=False)

    def shown(values):
        return values if scale is None else values * scale

    def number(value):
        if scale is None:
            return f'{value:.{digits}f}'
        return f'{value * scale:.{mg_decimals(value, digits)}f}'

    summary = []
    for ax, w in zip(axes.flat, tests):
        g0 = w['start']
        index = pd.date_range(g0 - pd.Timedelta(hours=MARGIN),
                              g0 + pd.Timedelta(hours=LENGTH + MARGIN - 1), freq='h')
        days = (np.arange(len(index)) - MARGIN) / 24
        ax.axvspan(0, LENGTH / 24, color=BAND, alpha=.55, lw=0, zorder=0)
        ax.plot(days, shown(series.reindex(index).to_numpy(float)), color=INK, lw=.7, zorder=3)
        at = (w['rows'].position.to_numpy(int) - 1) / 24
        ax.plot(at, shown(w['rows'].linear), color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=4)
        for (key, _, _), colour in zip(drawn, (BLUE, ORANGE)):
            ax.plot(at, shown(w['rows'][key]), color=colour, lw=1, marker='o', ms=3.0, zorder=5)
        best, line = mae(w['rows'], arm), mae(w['rows'], 'linear')
        tag = ' · 2026' if w['cohort'] == '2026' else ''
        parts = ' · '.join(f'{short} {number(mae(w["rows"], key))}' for key, _, short in drawn)
        ax.set_title(f'{g0:%d %b %Y}{tag}\n{parts}{joint}lin {number(line)}',
                     loc='left', color=INK, fontsize=11)
        ax.set_xlim(-MARGIN / 24, (LENGTH + MARGIN) / 24)
        summary.append({'episode': w['episode'], 'cohort': w['cohort'],
                        **{f'mae_{key}': round(mae(w['rows'], key), 4) for key, _, _ in drawn},
                        'mae_best': round(best, 4), 'mae_linear': round(line, 4)})
    for ax in axes.flat[len(tests):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(UNITS[variable])
    fig.supxlabel('days from the start of the hidden month', fontsize=13)
    handles = ([plt.Line2D([], [], color=INK, lw=1.2)]
               + [plt.Line2D([], [], color=colour, lw=1, marker='o', ms=4)
                  for _, colour in zip(drawn, (BLUE, ORANGE))]
               + [plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 2))),
                  plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=.55)])
    fig.legend(handles, ['real measurement'] + [label for _, label, _ in drawn]
               + ['linear interpolation (lin)', 'the hidden month, which no method saw'],
               loc='lower center', ncol=2, bbox_to_anchor=(.5, 1.0), fontsize=12)
    wins = sum(s['mae_best'] < s['mae_linear'] for s in summary)
    fig.tight_layout()
    name = f'fig_bay_gap_long_traces_{variable}'
    fig.savefig(THESIS / f'{name}.png', bbox_inches='tight', dpi=300)
    fig.savefig(DECK / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    record[variable] = {'best_arm': arm, 'windows': summary,
                        'best_ahead_of_linear': int(wins), 'of': len(summary)}


def main():
    for target in (THESIS, DECK):
        target.mkdir(parents=True, exist_ok=True)
    record = {"declared_in": "config/bay_reserved_protocol_v2_addendum33.json", "probe": EXO2,
              'length_hours': LENGTH, 'selection': 'none: every window on the probe is drawn'}
    for variable in ('oxygen', 'temperature'):
        figure(variable, record)
        r = record[variable]
        print(f'{variable}: best ahead of linear on {r["best_ahead_of_linear"]} of {r["of"]}')
    OUT.write_bytes(json.dumps(record, indent=2).encode('utf-8'))  # LF on every host
    print('wrote fig_bay_gap_long_traces_{oxygen,temperature} and', OUT)


if __name__ == '__main__':
    main()
