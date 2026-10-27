"""Can the Baia S. Teresa EXO2 sonde's real outages be filled? One probe, three questions.

The pooled gap-recovery results answer for the network. The operational question is asked about one
instrument: the EXO2 multiparameter sonde at Baia S. Teresa, one of the two probes that supply every
oxygen episode in addenda 30 and 31. Three things decide the answer, and this puts them side by side:

  where its lost hours are     the real gaps in its delivered record, by length
  what is validated there      its own recovery error against gap length, on Gap-Dev under two
                               episode seeds and on the untouched Gap-2026 read, with overlap
                               clusters counted
  what else was reporting      which neighbours were live during its two long outages

Descriptive for the record, and a re-read of published predictions for the errors: nothing is fitted
or tuned here. Climatology is the declared one, fitted outside the masked stretch.

Usage: python -m scripts.v2_gap_exo2
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import PUBLISHED, load_predictions
from scripts.v2_gap_intervals import MIN_CLUSTERS, clusters, paired
from scripts.v2_timeseries_figures import BAND, BLUE, DECK, GRID, INK, MUTED, ORANGE, ROOT, THESIS
from src.data.bay_observations import read_export
from src.units import OXYGEN_UNIT_TEXT, mg_decimals, oxygen_mg

RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_gap_exo2.json'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
FLOOR = 1.8     # in the artifacts' units, as the report records it; drawn in mg/l
AQUA = "#1baf7a"
RESERVED = pd.Timestamp('2026-01-01', tz='UTC')
#: which published run supplies which reading
READINGS = {'Gap-Dev': PUBLISHED[0], 'Gap-Dev, second seed': PUBLISHED[2], 'Gap-2026': PUBLISHED[1]}
#: the addendum-32 cohorts, as scripts/v2_gap_long.py reads them
LONG_COHORTS = {'Gap-Long-Dev': ('20260922T195727Z_v2_gaprecovery', 'v2_gap_recovery_long'),
                'Gap-Long-2026': ('20260922T195745Z_v2_gaprecovery',
                                  'v2_gap_recovery_long_reserved2026')}
SHORT_MAX = 72
#: the probes a fill could lean on during the long outages
NEIGHBOURS = {'temperature': ['Nodo_S07', 'Nodo_S05', 'Nodo_S08'],
              'oxygen': ['TPortovenereS_02m-CTD(calibrato)', 'TPortovenereP_02m-CTD(calibrato)']}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14.5, 'axes.titlesize': 14.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'legend.frameon': False, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
#: the figure is drawn at the size it prints at (\textwidth, 416.7 pt, by the earlier drawing's height there)
#: in these type sizes and saved without cropping
SIZE_IN = (416.7 / 72, 4.3846)
PRINT_STYLE = {'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
               'ytick.labelsize': 7, 'legend.fontsize': 7, 'grid.linewidth': .5}
#: a thin space, as the thesis groups thousands and sets a percent sign
THIN = ' '


def read(name):
    return read_export(ROOT / 'data/raw' / (name + '.txt'), include_auxiliary=True)[1]


def runs_of(present):
    """(start index, length) of every run of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], present.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2] - edges[::2]))


def record(series):
    grid = pd.date_range(series.dropna().index.min(), series.dropna().index.max(), freq='h')
    x = series.reindex(grid)
    present = x.notna().to_numpy()
    gaps = [(grid[s], int(n)) for s, n in runs_of(~present)]
    clean = sorted(((grid[s], int(n)) for s, n in runs_of(present)), key=lambda g: -g[1])
    hours = np.array([n for _, n in gaps])
    lost = int(hours.sum())
    return x, gaps, clean, {
        'first': grid[0].isoformat(), 'last': grid[-1].isoformat(), 'hours': int(len(grid)),
        'observed_hours': int(present.sum()), 'gaps': int(len(gaps)), 'lost_hours': lost,
        'gaps_longer_than': {str(b): int((hours > b).sum()) for b in (6, 24, 72, 168)},
        'share_of_lost_hours_in_gaps_longer_than': {
            str(b): round(float(hours[hours > b].sum() / lost), 4) for b in (6, 24, 72, 168)},
        'gaps_over_6h': [{'start': s.isoformat(), 'hours': n} for s, n in gaps if n > 6],
        'longest_clean_runs': [{'start': s.isoformat(), 'hours': n} for s, n in clean[:5]]}


def errors(run, results, variable='oxygen'):
    frame = load_predictions(run, results, variable, 'chronos_bidir')
    frame = frame[frame.sensor == EXO2]
    cluster_of = clusters(frame)
    out = {}
    for length, part in frame.groupby('gap_length'):
        row = {'episodes': int(part.episode.nunique())}
        for column in ('linear', 'chronos_bidir', 'climatology', 'nearest_probe'):
            d = part.dropna(subset=[column, 'truth'])
            row[column] = round(float(d.assign(e=(d.truth - d[column]).abs())
                                      .groupby('episode').e.mean().mean()), 4)
        row['chronos_bidir_minus_linear'] = paired(part, 'chronos_bidir', 'linear',
                                                   f'exo2|{run}|{int(length)}', cluster_of)
        row['clusters'] = row['chronos_bidir_minus_linear']['clusters']
        out[str(int(length))] = row
    return out


def main():
    exo = read(EXO2)
    oxygen, gaps, clean, stats = record(exo['oxygen'])
    _, _, _, temp_stats = record(exo['temperature'])
    long_outages = [(s, n) for s, n in gaps if n > 168]
    coverage = {}
    for start, n in long_outages:
        end = start + pd.Timedelta(hours=n - 1)
        entry = {}
        for variable, names in NEIGHBOURS.items():
            for name in names:
                s = read(name)[variable].reindex(pd.date_range(start, end, freq='h'))
                entry[f'{name} {variable}'] = int(s.notna().sum())
        coverage[f'{start:%Y-%m-%d} ({n} h)'] = entry
    o = oxygen.dropna()
    daily = o.groupby(o.index.floor('D')).agg(lambda v: v.max() - v.min() if len(v) > 20 else np.nan)
    monthly_sd = o.groupby(o.index.month).std()
    by_reading = {label: errors(run, results) for label, (run, results, _) in READINGS.items()}
    # Addendum 32: windows whose interior may hold short real gaps, scored only where observed.
    # This is where the gaps of a week or more on this probe became testable.
    long_all = json.loads((RUNS / 'v2_gap_long.json').read_text())
    long = {cohort: long_all[cohort]['oxygen']['exo2'] for cohort in LONG_COHORTS}
    # Addendum 33: Chronos-2 with the nearest probes as covariates, the best month-long oxygen filler
    # on the development record (not confirmed on 2026), taken from its per-row predictions.
    for cohort in LONG_COHORTS:
        rows = pd.read_csv(RUNS / 'v2_gap_variants' / f'{cohort}__oxygen__168-1440.csv.gz')
        rows = rows[rows.sensor == EXO2]
        for L, part in rows.groupby('gap_length'):
            err = (part.truth - part.chronos2_cov_bidir).abs().groupby(part.episode).mean().mean()
            long[cohort][str(int(L))]['mae']['chronos2_cov_bidir'] = round(float(err), 4)
    windows = {}
    for cohort, (run, results) in LONG_COHORTS.items():
        frame = pd.read_csv(RUNS / run / results / 'oxygen_chronos_bidir/predictions.csv.gz',
                            usecols=['episode', 'sensor', 'gap_length'])
        keys = frame[(frame.sensor == EXO2) & (frame.gap_length == 720)].episode.unique()
        windows[cohort] = sorted(k.rsplit('|', 2)[1] for k in keys)

    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum30.json',
              'long_gaps_declared_in': 'config/bay_reserved_protocol_v2_addendum32.json',
              'long_cohorts': long, 'tested_720h_windows': windows,
              'probe': EXO2, 'floor_umol_per_l': FLOOR, 'min_clusters': MIN_CLUSTERS,
              'oxygen_record': stats, 'temperature_record': temp_stats,
              'oxygen_level': {'mean': round(float(o.mean()), 1), 'sd': round(float(o.std()), 1),
                               'within_month_sd_range': [round(float(monthly_sd.min()), 1),
                                                         round(float(monthly_sd.max()), 1)],
                               'daily_range_median': round(float(daily.median()), 1)},
              'neighbour_hours_during_long_outages': coverage,
              'recovery_by_gap_length': by_reading,
              'readings': {k: {'run': v[0], 'results': v[1]} for k, v in READINGS.items()}}
    OUT.write_bytes(json.dumps(report, indent=2).encode('utf-8'))  # LF on every host

    # drawn at the size it prints at (\textwidth by the earlier drawing's height there) in these type sizes
    # and saved without cropping, so that no text prints below 7 pt and the page does not move
    plt.rcParams.update(PRINT_STYLE)
    fig = plt.figure(figsize=SIZE_IN)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, .8, 1.35], width_ratios=[3.4, 1.9], hspace=.62,
                            wspace=.04, left=.125, right=.955, top=.95, bottom=.085)
    ax = fig.add_subplot(grid[0, :])
    ax.plot(oxygen.index, oxygen_mg(oxygen.to_numpy()), color=INK, lw=.3)
    # The masked 720 h gaps actually tested on this probe (addendum 32), as bars along the floor.
    for starts in windows.values():
        for start in starts:
            t0 = pd.Timestamp(start)
            ax.plot([t0, t0 + pd.Timedelta(hours=720)], [.035, .035], color=BLUE, lw=3,
                    solid_capstyle='butt', transform=ax.get_xaxis_transform(), zorder=5)
    ax.text(pd.Timestamp(windows['Gap-Long-Dev'][0]), .07,
            f'{sum(map(len, windows.values()))} masked 720 h test gaps',
            transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=7, color=BLUE)
    for start, n in gaps:
        if n > 6:
            ax.axvspan(start, start + pd.Timedelta(hours=n), color=ORANGE, alpha=.35, lw=0)
        if n > 168:
            ax.text(start + pd.Timedelta(hours=n / 2), .96, f'{n} h\noutage',
                    transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7,
                    color=INK)
    ax.axvline(RESERVED, color=MUTED, lw=.8, ls=(0, (2, 2)))
    ax.text(RESERVED, .96, ' Reserved-2026', transform=ax.get_xaxis_transform(), va='top',
            fontsize=7, color=MUTED)
    ax.set_ylabel(f'Oxygen ({OXYGEN_UNIT_TEXT})')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    lost = f'{stats["lost_hours"]:,}'.replace(',', THIN)
    share_out = 100 * stats['share_of_lost_hours_in_gaps_longer_than']['168']
    ax.set_title(f'(a) {stats["gaps"]} gaps, {lost} lost hours; two outages hold {share_out:.0f}{THIN}% of them',
                 loc='left')

    hours = np.sort(np.array([n for _, n in gaps]))
    share = np.cumsum(hours) / hours.sum()
    lengths = sorted(int(L) for L in by_reading['Gap-Dev'])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_b.step(hours, share * 100, where='post', color=INK, lw=1.0)
    ax_b.set_ylabel('Lost hours in\ngaps ≤ length (%)')
    ax_b.set_ylim(0, 105)
    ax_b.set_title('(b) where the lost hours are', loc='left')

    ax_c = fig.add_subplot(grid[2, 0], sharex=ax_b)
    dev = by_reading['Gap-Dev']
    short = [L for L in lengths if L <= SHORT_MAX]
    # Up to 72 h the fully observed design has enough independent stretches; from a week on only the
    # addendum-32 windows do, so each part of the axis is drawn from the design that can support it.
    ax_c.axvline(110, color=MUTED, lw=.7, ls=(0, (1, 2)))
    ax_c.text(95, .97, 'fully observed\nwindows', transform=ax_c.get_xaxis_transform(),
              ha='right', va='top', fontsize=7, color=MUTED)
    ax_c.text(128, .97, 'windows with short\nreal gaps inside',
              transform=ax_c.get_xaxis_transform(), ha='left', va='top', fontsize=7, color=MUTED)
    # the oxygen floor on an axis of absolute errors: the scale an error is read against
    ax_c.axhline(oxygen_mg(FLOOR), color=MUTED, lw=.8, ls=(0, (6, 3)))
    # At the right-hand end, where every series sits well above it; at the left it ran into them.
    ax_c.text(1900, oxygen_mg(FLOOR), f'floor {oxygen_mg(FLOOR):.{mg_decimals(FLOOR, 1)}f}',
              ha='right', va='bottom', fontsize=7, color=MUTED)
    series = (('linear', 'linear interpolation', INK, 'o'),
              ('chronos_bidir', 'Chronos-Bolt, bidirectional', BLUE, 'D'),
              ('chronos2_cov_bidir', 'Chronos-2 with neighbours', ORANGE, '^'),
              ('gpr', 'Gaussian process', AQUA, 's'), ('climatology', 'climatology', MUTED, 'v'))
    for key, label, colour, marker in series:
        if key in dev[str(short[0])]:
            ax_c.plot(short, oxygen_mg([dev[str(L)][key] for L in short]), color=colour, lw=1.1,
                      marker=marker, ms=3.4, label=label, zorder=4)
        long_dev = long['Gap-Long-Dev']
        x = sorted(int(L) for L in long_dev)
        ax_c.plot(x, oxygen_mg([long_dev[str(L)]['mae'][key] for L in x]), color=colour, lw=1.1,
                  ls=(0, (5, 2)), marker=marker, ms=3.4, zorder=4,
                  label=None if key in dev[str(short[0])] else label)
    # The confirmatory reads, open markers, only where five or more independent units support them.
    for key, colour, marker in (('chronos_bidir', BLUE, 'D'), ('linear', INK, 'o')):
        confirm = by_reading['Gap-2026']
        x = sorted(int(L) for L in confirm if confirm[L]['clusters'] >= MIN_CLUSTERS)
        y = [confirm[str(L)][key] for L in x]
        long_2026 = long['Gap-Long-2026']
        for L, row in sorted(long_2026.items(), key=lambda kv: int(kv[0])):
            if row['conservative_units'] >= MIN_CLUSTERS:
                x.append(int(L))
                y.append(row['mae'][key])
        ax_c.plot(x, oxygen_mg(y), ls='none', marker=marker, ms=5, mfc='none', mec=colour, mew=.9,
                  zorder=5, label=f'{"Chronos-Bolt" if key == "chronos_bidir" else "linear interpolation"}, 2026')
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xticks([1, 3, 6, 12, 24, 72, 168, 336, 720, 1440])
    ax_c.set_xticklabels(['1', '3', '6', '12', '24', '72', '168', '336', '720', '1440'])
    ax_c.minorticks_off()
    ax_c.set_ylim(.03, 1.3)
    ax_c.set_yticks([.05, .1, .2, .5, 1])
    ax_c.set_yticklabels(['0.05', '0.1', '0.2', '0.5', '1'])
    ax_c.set_xlim(.8, 2000)
    ax_c.set_xlabel('Gap length (h)')
    ax_c.set_ylabel(f'Mean absolute error\n({OXYGEN_UNIT_TEXT})')
    ax_c.set_title('(c) what is validated there: this probe\'s own recovery error', loc='left')
    ax_c.legend(loc='upper left', bbox_to_anchor=(1.03, 1.0), fontsize=7, handlelength=2.2, borderaxespad=0,
                labelspacing=.6)
    # Panel (b) keeps its own tick labels: hidden, its gap lengths could only be read off panel (c).
    ax_b.tick_params(axis='x', labelbottom=True)
    ax_b.set_xlabel('Gap length (h)')
    fig.align_ylabels([ax, ax_b, ax_c])
    for target in (THESIS, DECK):
        target.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS / 'fig_bay_gap_exo2.png', dpi=300)
    fig.savefig(DECK / 'fig_bay_gap_exo2.pdf', bbox_inches='tight')
    plt.close(fig)
    manifest = DECK / 'manifest_gap.json'
    entries = json.loads(manifest.read_text()) if manifest.exists() else {}
    entries['fig_bay_gap_exo2'] = ['reports/bay_twin/runs/v2_gap_exo2.json',
                                   'reports/bay_twin/runs/v2_gap_long.json',
                                   'reports/bay_twin/runs/v2_gap_climatology',
                                   'reports/bay_twin/runs/v2_gap_variants.json'] + sorted(
        {f'reports/bay_twin/runs/{run}/{results}'
         for run, results, *_ in list(READINGS.values()) + list(LONG_COHORTS.values())})
    manifest.write_text(json.dumps(entries, indent=2))

    print(f'{EXO2}: {stats["gaps"]} oxygen gaps, {stats["lost_hours"]} lost hours, '
          f'{stats["share_of_lost_hours_in_gaps_longer_than"]["168"]:.1%} in gaps over a week')
    for label, block in by_reading.items():
        print(f'== {label}')
        for L, row in block.items():
            c = row['chronos_bidir_minus_linear']
            ci = (f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]{'*' if c['excludes_zero'] else ' '}"
                  if c['estimable'] else 'not estimable')
            print(f"   {L:>4} h  linear {row['linear']:6.2f}  chronos {row['chronos_bidir']:6.2f}"
                  f"  clim {row['climatology']:6.2f}  delta {c['delta']:+6.2f} {ci}"
                  f"  ({row['episodes']} ep, {row['clusters']} clusters)")
    print('neighbours during long outages:', json.dumps(coverage, indent=1))
    print('wrote fig_bay_gap_exo2 and', OUT)


if __name__ == '__main__':
    main()
