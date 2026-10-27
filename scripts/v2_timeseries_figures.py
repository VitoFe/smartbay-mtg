"""Time-series figures: bridging a telemetry gap, and what a six-hour forecast looks like.

Every figure in this thesis so far is a bar chart or a scatter of summary statistics, which shows
how large an effect is and never what the series actually does. Three traces are drawn here:

  fig_bay_gap_bridging     every continuation of a silent probe on the macro rule, out to 72 h,
                           on Bridge-Dev and Bridge-2026 (thesis only while DECK_BRIDGING is False)
  fig_bay_forecast_trace   a reserved-2026 window with the calibrated interval, so coverage is
                           visible rather than only tabulated
  fig_bay_saturation       the four addendum-23 arms over one window, which is what "one
                           measurement" looks like

Colours are #2a78d6 / #eb6834 / #7d55c7, validated for colour-vision deficiency (worst adjacent
pair ΔE 24.7 under protanopia, against a target of 8). Every series also carries a distinct dash
pattern, so identity never rests on hue alone in print or for a CVD reader.

Usage: python -m scripts.v2_timeseries_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.units import OXYGEN_UNIT_TEXT, oxygen_mg

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
THESIS = ROOT / 'thesis_template/images/figures'
DECK = ROOT / 'presentation/figures_v2'

BLUE, ORANGE, PURPLE = '#2a78d6', '#eb6834', '#7d55c7'
INK, MUTED, GRID, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#cfe0f5'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.titlesize': 10,
    'xtick.color': MUTED, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .7,
    'axes.axisbelow': True, 'figure.dpi': 150, 'legend.frameon': False,
})
manifest = {}
#: \textwidth of the thesis in inches (416.7 pt), and the type sizes of a figure drawn at its printed size
TEXTWIDTH_IN = 416.7 / 72
PRINT_STYLE = {'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
               'ytick.labelsize': 7, 'legend.fontsize': 7}


def tidy_dates(ax):
    """Thin and slant the date ticks. Left to itself matplotlib packs them until adjacent labels
    overlap, which happened at the month boundary in the reserved window."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    for label in ax.get_xticklabels():
        label.set_rotation(0)


def save(fig, name, sources, deck=True, crop=True):
    """The thesis PNG (cropped to its content unless `crop` is False, for a figure drawn at its printed
    size) and, with `deck`, a PDF for the deck."""
    for target in (THESIS, DECK):
        target.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS / f'{name}.png', bbox_inches='tight' if crop else None, dpi=300)
    if deck:
        fig.savefig(DECK / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    manifest[name] = [str(Path(s).relative_to(ROOT)).replace('\\', '/') for s in sources]
    print('wrote', name, '' if deck else '(thesis only)')


# The deck's slide on this figure still describes the rollout alone; its PDF is left as it is until
# that slide is redrawn, and then this switch writes it too.
DECK_BRIDGING = False
AQUA = '#1baf7a'     # fourth categorical slot; validated with the three above for CVD (worst ΔE 9.2)


def gap_bridging():
    """Carrying a silent probe's record: every continuation on the frozen (fold, sensor) macro rule.

    Both cohorts side by side, Bridge-Dev (the addendum-4 dropout episodes) and Bridge-2026: the
    anchored continuation, the last value, the hour-of-day climatological increment, the neighbour
    rule and the recursive rollout. Every point is a lead's macro MAE in runs/v2_bridge/report_a69.json,
    the numbers of the two continuation tables (addendum 69: every model selected per outer fold; every
    other arm as in report.json). Leads of 54 to 72 h come from runs/v2_bridge72/report.json (addendum
    63): a second tree, scored where the probe is observed at the lead, so each line breaks between 48
    and 54 h. The rollout is read from runs/v2_rollout_a82.json (addendum 82, part A), on both cohorts and
    at every lead to 72 h, on the continuation's rows; at 6 to 48 h on Bridge-Dev it must equal the
    report's. The neighbour rule is drawn for temperature only: for oxygen another sonde reports at
    onset and at the target in about a fifth of the development episodes, so its error would be on
    other episodes than the rest (the tables give its paired contrast).

    An earlier draft drew four individual episodes chosen by how much the truth moved. That is
    selection on the outcome, and the aggregate over every episode is the actual claim.
    """
    src = RUNS / 'v2_bridge/report_a69.json'
    report = json.loads(src.read_text())
    src72 = RUNS / 'v2_bridge72/report.json'
    report72 = json.loads(src72.read_text())
    src_rollout = RUNS / 'v2_rollout_a82.json'
    rollout = json.loads(src_rollout.read_text(encoding='utf-8'))['mae']
    for variable in ('temperature', 'oxygen'):
        for L, e in report[variable]['Bridge-Dev']['by_lead'].items():
            if abs(e['mae']['recursive_tree'] - rollout[f'{variable}|Bridge-Dev'][L]['recursive_tree']) > 1e-6:
                raise SystemExit(f'gap_bridging: the {variable} rollout at {L} h differs between the two reports')
    arms = [('anchored_direct', 'network-aided forecast', BLUE, '-', 'o', 2.2, 5),
            ('neighbour_change', "last value + nearest probe's change", AQUA, '-.', 'D', 1.8, 4),
            ('clim_hour', 'last value + hour-of-day increment', MUTED, ':', 's', 1.6, 2),
            ('persistence', 'hold last value (delivered for temperature)', ORANGE, '--', '^', 1.8, 3),
            ('recursive_tree', 'recursive rollout', PURPLE, (0, (5, 1.5, 1, 1.5)), 'v', 1.6, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), sharey='row')
    handles = {}
    for r, variable in enumerate(('temperature', 'oxygen')):
        for c, cohort in enumerate(('Bridge-Dev', 'Bridge-2026')):
            ax = axes[r, c]
            block = report[variable][cohort]
            leads = sorted(block['by_lead'], key=int)
            x = [int(L) for L in leads]
            for key, label, colour, ls, marker, lw, z in arms:
                if key == 'neighbour_change' and variable == 'oxygen':
                    continue
                if key == 'recursive_tree':
                    # the rollout on both cohorts, 6 to 48 h and then 54 to 72 h, on the continuation's rows
                    m = rollout[f'{variable}|{cohort}']
                    for part in (leads, ['54', '60', '66', '72']):
                        y = [m[L]['recursive_tree'] for L in part]
                        handles[label], = ax.plot([int(L) for L in part], oxygen_mg(y) if variable == 'oxygen' else y,
                                                  color=colour, lw=lw, ls=ls, marker=marker, ms=5, label=label,
                                                  zorder=z)
                    continue
                if key not in block['by_lead'][leads[0]]['mae']:
                    continue
                y = [block['by_lead'][L]['mae'][key] for L in leads]
                if variable == 'oxygen':
                    y = oxygen_mg(y)
                handles[label], = ax.plot(x, y, color=colour, lw=lw, ls=ls, marker=marker, ms=5,
                                          label=label, zorder=z)
                later = report72[variable][cohort]['by_lead']
                if key in later[min(later, key=int)]['mae']:
                    x2 = sorted(int(L) for L in later)
                    y2 = [later[str(L)]['mae'][key] for L in x2]
                    ax.plot(x2, oxygen_mg(y2) if variable == 'oxygen' else y2, color=colour, lw=lw, ls=ls,
                            marker=marker, ms=5, zorder=z)
            ax.axvline(51, color=GRID, lw=1.2, zorder=0)
            ax.set_xticks([6, 12, 24, 36, 48, 60, 72])
            unit = '°C' if variable == 'temperature' else OXYGEN_UNIT_TEXT
            if c == 0:
                ax.set_ylabel(f'{variable}: mean absolute error ({unit})')
            if r == 1:
                ax.set_xlabel('hours since the probe went silent')
            ax.set_title(f"{cohort} · {block['episodes']:,} episodes, {len(block['sensors'])} probes",
                         color=MUTED, loc='left')
    order = [a[1] for a in arms if a[1] in handles]
    fig.legend([handles[k] for k in order], order, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 'fig_bay_gap_bridging', [src, src72, src_rollout], deck=DECK_BRIDGING)


def forecast_trace():
    """A reserved-2026 window: the six-hour forecast, its calibrated interval and the truth, so
    that coverage can be seen. Persistence is drawn as the thing the forecast has to beat."""
    src = RUNS / 'v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'
    e = pd.read_csv(src)
    t = e[(e.variable == 'temperature') & e.truth.notna() & e.prediction.notna()].copy()
    t['target_time'] = pd.to_datetime(t.target_time, utc=True)
    sensor = t.groupby('sensor').size().idxmax()
    s = t[t.sensor == sensor].sort_values('target_time')
    # The densest four-week stretch, so the trace is continuous rather than a dotted scatter.
    s = s.set_index('target_time')
    counts = s.resample('7D').size()
    start = counts.rolling(4).sum().idxmax() - pd.Timedelta(days=21)
    w = s.loc[start:start + pd.Timedelta(days=28)].reset_index()

    # drawn at the size it prints at (\textwidth by the earlier drawing's height there) and saved
    # without cropping, so that no text prints below 7 pt and the page does not move
    with plt.rc_context(PRINT_STYLE):
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.039))
        if w.radius.notna().any():
            ax.fill_between(w.target_time, w.prediction - w.radius, w.prediction + w.radius,
                            color=BAND, alpha=.85, lw=0, label='90 % interval', zorder=1)
        ax.plot(w.target_time, w.persistence, color=ORANGE, lw=1.0, ls='--',
                label='last value', zorder=2)
        ax.plot(w.target_time, w.prediction, color=BLUE, lw=1.3, ls='-', label='forecast', zorder=3)
        ax.plot(w.target_time, w.truth, color=INK, lw=1.0, ls=':', label='observed', zorder=4)
        miss = w[w.covered == False]  # noqa: E712 - a NaN-bearing column, so `is False` will not do
        if len(miss):
            ax.scatter(miss.target_time, miss.truth, s=14, facecolor='none', edgecolor=PURPLE,
                       lw=1.0, zorder=5, label='outside the interval')
        ax.set_ylabel('Temperature (°C)')
        covered = float(w.covered.mean()) if w.covered.notna().any() else float('nan')
        # what the window is, as a title; the axis carries only its quantity and unit
        ax.set_title(f'{sensor}, Reserved-2026, six-hour lead: {len(w)} scored endpoints, '
                     f'{100 * covered:.0f} % inside the interval', loc='left')
        ax.set_xlabel('Target time (UTC), 2026')
        tidy_dates(ax)
        fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc='lower center', bbox_to_anchor=(.5, 0.0),
                   handlelength=2.4, columnspacing=1.6)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        save(fig, 'fig_bay_forecast_trace', [src], crop=False)


def saturation_trace():
    """The four addendum-23 arms over one window, the recurrent network as Table bay_sequence_arms
    carries it: selected on the calendar inner window (addendum 49, rule D4), joined on the same rows.
    The aggregate comparison belongs to the table and the caption, not to the image."""
    # addendum 69: the tree refitted per outer fold, as Table bay_sequence_arms carries it
    src = RUNS / 'v2_sequence_arms/perfold_tree/sequence_arms_temperature.csv.gz'
    cal = RUNS / 'v2_sequence_sweep/gru/calendar/temperature_offset0_seed42/predictions.csv.gz'
    d = pd.read_csv(src)
    keys = ['fold', 'sensor', 'time']
    calendar = pd.read_csv(cal, usecols=keys + ['lstm'])
    d = d.drop(columns='lstm').merge(calendar, on=keys, how='left', validate='one_to_one')
    if d.lstm.isna().any():
        raise SystemExit('calendar recurrent predictions do not cover the sequence-arm rows')
    d['time'] = pd.to_datetime(d.time, utc=True)
    sensor = d.groupby('sensor').size().idxmax()
    s = d[d.sensor == sensor].sort_values('time').set_index('time')
    counts = s.resample('7D').size()
    start = counts.rolling(2).sum().idxmax() - pd.Timedelta(days=7)
    w = s.loc[start:start + pd.Timedelta(days=14)].reset_index()
    scale = 10.0  # temperature z-scaling used by the forecast build

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.plot(w.time, 20 + w.truth * scale, color=INK, lw=2.4, label='observed', zorder=4)
    for col, colour, style, label in [('tree', ORANGE, '--', 'residual tree'),
                                      ('lstm', PURPLE, '-.', 'recurrent, trained here'),
                                      ('chronos_zs', BLUE, '-', 'Chronos, zero-shot'),
                                      ('timesfm_zs', '#1f9e8f', (0, (1, 1.6)), 'TimesFM, zero-shot')]:
        ax.plot(w.time, 20 + w[col] * scale, color=colour, lw=1.7, ls=style, label=label, zorder=3)
    ax.set_ylabel('temperature (°C)')
    ax.set_xlabel(f'{sensor}, development folds · six-hour lead')
    tidy_dates(ax)
    ax.legend(ncol=5, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.28))
    fig.tight_layout()
    save(fig, 'fig_bay_saturation_trace', [src, cal])


def main():
    gap_bridging()
    forecast_trace()
    saturation_trace()
    path = DECK / 'manifest_timeseries.json'
    path.write_text(json.dumps(manifest, indent=2))
    print('\nmanifest:', path.relative_to(ROOT))


if __name__ == '__main__':
    main()
