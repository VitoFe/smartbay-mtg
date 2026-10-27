"""What each family of gap-filling method actually draws across a masked gap.

The recovery tables report error against gap length, which says how far off a method is and never
what its fill looks like: whether it keeps the diurnal cycle, whether it drifts, whether it simply
draws a line. This draws the arms of addendum 30, grouped by family, over the same masked
episodes, with the truth that was hidden from them: every family for temperature, and for oxygen
the two families its text reads (`ROWS`). The errors of every arm are stored either way.

Episodes are chosen by a fixed-seed draw among every eligible episode of the stated probe and
length. They are never chosen by any method's error, by how variable the truth is, or by eye. A
figure that picks its example because a method did well on it shows the selection rather than the
method, and an earlier trace in this thesis was rebuilt for exactly that reason. Each panel is
therefore one draw, and the per-episode errors printed on it are anecdotes. The pooled numbers
are in `runs/v2_gap_intervals.json`.

The climatology drawn is the declared one, fitted outside the masked stretch.
PCHIP is recomputed here through the same 72 h of knots either side that `v2_gap_splines` uses.

Usage: python -m scripts.v2_gap_traces
"""
import json
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from scripts.v2_gap_climatology import arm_dirs, load_predictions
from scripts.v2_timeseries_figures import BAND, BLUE, DECK, INK, MUTED, ORANGE, ROOT, THESIS
from src.data.bay_profile_context import load_observations
from src.units import OXYGEN_UNIT_TEXT, factor, mg_decimals

RUNS = ROOT / 'reports/bay_twin/runs'
RUN = '20260922T171859Z_v2_gaprecovery'
RESULTS = RUNS / RUN / 'v2_gap_recovery'
OUT = RUNS / 'v2_gap_traces.json'
EXO2 = 'TBaiaP_01m-CTD(calibrato)'
DRAW_SEED = 20260922
MARGIN = 72
UNITS = {'temperature': '°C', 'oxygen': OXYGEN_UNIT_TEXT}
#: Third slot is the reference palette's aqua. Blue, orange and aqua pass the colour-vision check
#: on every pair (worst ΔE 9.2, deutan); a fourth hue does not, so no family draws more than three.
AQUA = '#1baf7a'
SLOTS = [(BLUE, '-'), (ORANGE, (0, (5, 2))), (AQUA, (0, (1.4, 1.2)))]

FAMILIES = [
    ('Interpolation', [('linear', 'linear'), ('pchip', 'PCHIP'), ('climatology', 'climatology')]),
    ('Neighbouring probes', [('nearest_probe', 'nearest probe'), ('gpr', 'Gaussian process')]),
    ('Direct regression', [('regression_retro', 'gradient boosting'),
                           ('polynomial', 'polynomial ridge')]),
    ('Recursive rollout', [('rolling_forward', 'forward'), ('rolling_backward', 'backward'),
                           ('hybrid', 'hybrid blend')]),
    ('In-context tabular', [('tabpfn', 'TabPFN'), ('tabicl', 'TabICL')]),
    # The LSTM and compressed sensing are not drawn: neither was tuned as the protocol requires,
    # and the thesis does not report them.
    ('Pretrained time series', [('chronos_forward', 'Chronos, forward'),
                                ('chronos_bidir', 'Chronos, bidirectional')]),
]

#: The families each figure draws. The oxygen figure keeps the two its text reads: the incumbent's
#: family, whose contrast with linear the header strip places, and the rollouts, which drift past a
#: week. Linear interpolation is the reference in every row. The temperature figure draws them all.
ROWS = {'oxygen': ('Pretrained time series', 'Recursive rollout')}

#: Sizes per figure. The oxygen figure is drawn at the size it prints at (\textwidth, 416.7 pt, by the
#: earlier drawing's height there) with no text below 7 pt and saved without cropping (`crop`), so the
#: page does not move; the temperature figure, in the supplement, keeps its earlier drawing.
LAYOUT = {
    'temperature': {'figsize': (10.2, 11.2), 'hspace': .3, 'rc': {}, 'title': '{title}\n{date}',
                    'title_size': 16, 'title_pad': 68, 'note': 13, 'legend': 15,
                    'legend_title': 16, 'mae': 14, 'mae_y': .905, 'mae_text':
                    'per-episode MAE ({unit})\nleft · right panel', 'xlabel': 17, 'xlabel_y': .05,
                    'xlabel_text': 'Time from the start of the masked gap (h)', 'ylabel': 'Temperature ({unit})',
                    'ms': 3.8, 'strip_s': (14, 70), 'lw': (1.2, 2.0), 'crop': True,
                    'grid': {'width_ratios': [1, 1, .95], 'wspace': .3}},
    'oxygen': {'figsize': (416.7 / 72, 2.888), 'hspace': .22,
               'rc': {'font.size': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7},
               'title': '{length} h gap · {date}', 'title_size': 7.5, 'title_pad': 4,
               'note': 7, 'legend': 7, 'legend_title': 7.5, 'mae': 7, 'mae_y': .85,
               'mae_text': 'per-episode MAE ({unit}),\nleft · right panel', 'xlabel': 7.5,
               'xlabel_y': .005, 'xlabel_text': 'Time from the start of the masked gap (h)',
               'ylabel': 'Oxygen ({unit})', 'ms': 2.0, 'strip_s': (4, 22), 'lw': (0.9, 1.3), 'crop': False,
               'grid': {'width_ratios': [1, 1, 1.9], 'wspace': .2, 'left': .085, 'right': .99, 'top': .79,
                        'bottom': .15}},
}

#: Oxygen is drawn on Baia EXO2 because that is the probe the question was asked about: one gap
#: inside the band where a method was confirmed, one the length of its real outages.
SPECS = {
    'oxygen': [{'title': 'Baia EXO2 · 24 h', 'sensor': EXO2, 'length': 24},
               {'title': 'Baia EXO2 · 720 h', 'sensor': EXO2, 'length': 720}],
    'temperature': [{'title': 'twin · 48 h', 'twin': True, 'length': 48},
                    {'title': 'no twin · 48 h', 'twin': False, 'length': 48}],
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 17,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.titlesize': 16,
    'xtick.color': MUTED, 'ytick.color': INK, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': .7,
    'axes.axisbelow': True, 'legend.frameon': False,
})


def predictions(variable):
    """All arms side by side, one row per scored position, with the declared climatology."""
    merged = None
    for (var, arm) in sorted(arm_dirs(RUN, RESULTS.name)):
        if var != variable:
            continue
        # The loader applies the declared climatology and resolves arms rerun since publication.
        frame = load_predictions(RUN, RESULTS.name, variable, arm)
        if merged is None:
            merged = frame[['episode', 'sensor', 'has_twin', 'gap_length', 'position', 'fold',
                            'time', 'truth', 'persistence', 'linear', 'spline', 'climatology',
                            'nearest_probe']]
        merged = merged.merge(frame[['episode', 'position', arm]], on=['episode', 'position'],
                              how='left', validate='one_to_one')
    return merged


def draw_key(variable, spec):
    """What the draw is seeded on: the eligibility rule, never the display title.

    The first version hashed the title, so shortening a title for layout silently re-drew the
    episode. A cosmetic edit must not be able to change which case is shown.
    """
    rule = (f'sensor={spec["sensor"]}' if 'sensor' in spec else f'twin={spec["twin"]}')
    return f'{variable}|{rule}|length={spec["length"]}'


def draw(frame, variable, spec):
    """One episode by a seeded draw over the eligible set, and how many it was drawn from."""
    pool = frame[frame.gap_length == spec['length']]
    if 'sensor' in spec:
        pool = pool[pool.sensor == spec['sensor']]
    if 'twin' in spec:
        pool = pool[pool.has_twin == spec['twin']]
    eligible = sorted(pool.episode.unique())
    rng = np.random.default_rng([DRAW_SEED, zlib.crc32(draw_key(variable, spec).encode())])
    return eligible[int(rng.integers(len(eligible)))], len(eligible)


def pchip(series, g0, length, positions):
    knots = np.concatenate([np.arange(g0 - MARGIN, g0), np.arange(g0 + length, g0 + length + MARGIN)])
    y = series[knots]
    ok = np.isfinite(y)
    return PchipInterpolator(knots[ok], y[ok], extrapolate=False)(g0 + positions - 1)


def mae(truth, estimate):
    ok = np.isfinite(truth) & np.isfinite(estimate)
    return float(np.mean(np.abs(truth[ok] - estimate[ok]))) if ok.any() else float('nan')


def episode_data(variable, frame, frames, spec):
    """Everything one column needs: the drawn episode, its truth window and every fill."""
    episode, pool = draw(frame, variable, spec)
    sensor, start, length = episode.rsplit('|', 2)
    length, g0t = int(length), pd.Timestamp(start)
    rows = frame[frame.episode == episode].sort_values('position')
    positions = rows.position.to_numpy(int)
    index = pd.date_range(g0t - pd.Timedelta(hours=MARGIN),
                          g0t + pd.Timedelta(hours=length + MARGIN - 1), freq='h')
    window = frames[sensor][variable].reindex(index).to_numpy(float)
    estimates = {key: rows[key].to_numpy(float) for key in rows.columns
                 if key not in ('episode', 'sensor', 'has_twin', 'gap_length', 'position',
                                'fold', 'time', 'truth')}
    estimates['pchip'] = pchip(window, MARGIN, length, positions)
    truth = rows.truth.to_numpy(float)
    errors = {key: mae(truth, y) for key, y in estimates.items()}
    # One draw cannot say whether it is typical, so the header places it in its own pool on the
    # contrast the text reports. This is computed after the draw and cannot influence it.
    same = frame[frame.gap_length == spec['length']]
    if 'sensor' in spec:
        same = same[same.sensor == spec['sensor']]
    if 'twin' in spec:
        same = same[same.has_twin == spec['twin']]
    per = (same.assign(a=(same.truth - same.chronos_bidir).abs(), b=(same.truth - same.linear).abs())
           .groupby('episode')[['a', 'b']].mean())
    contrast = per.a - per.b
    placement = {'here': float(contrast[episode]), 'pool_median': float(contrast.median()),
                 'rank': int((contrast < contrast[episode]).sum()) + 1, 'of': int(len(contrast)),
                 # every eligible episode's contrast, drawn as a strip beside the panel (review R15)
                 'pool': sorted(float(v) for v in contrast)}
    return {'spec': spec, 'episode': episode, 'pool': pool, 'length': length, 'start': g0t,
            'positions': positions, 'window': window, 'hours': np.arange(-MARGIN, length + MARGIN),
            'estimates': estimates, 'errors': errors, 'placement': placement}


def figure(variable, frame, frames, record):
    with plt.rc_context(LAYOUT[variable]['rc']):
        draw_figure(variable, frame, frames, record)


def draw_figure(variable, frame, frames, record):
    columns = [episode_data(variable, frame, frames, spec) for spec in SPECS[variable]]
    lay = LAYOUT[variable]
    order = ROWS.get(variable, tuple(name for name, _ in FAMILIES))
    families = [next(f for f in FAMILIES if f[0] == name) for name in order]
    digits = 2 if variable == 'oxygen' else 3
    # The record and every selection stay in the artifacts' units; oxygen is drawn in mg/l.
    scale = factor() if variable == 'oxygen' else None

    def shown(values):
        return values if scale is None else values * scale

    def number(value, sign=''):
        """A printed error or contrast, with the significant figures it has at `digits`, and a real minus
        sign."""
        if scale is None:
            text = f'{value:{sign}.{digits}f}'
        else:
            text = f'{value * scale:{sign}.{mg_decimals(value, digits)}f}'
        return text.replace('-', '−')

    fig = plt.figure(figsize=lay['figsize'])
    grid = GridSpec(len(families), 3, figure=fig, hspace=lay['hspace'], **lay['grid'])
    reference = [('linear', 'linear (reference)', MUTED, (0, (3, 2)))]
    for row, (family, members) in enumerate(families):
        # Linear is the reference in every row, drawn the same way everywhere, so the interpolation
        # row shows only what it adds to it.
        drawn = [(key, label, colour, style) for (key, label), (colour, style)
                 in zip([m for m in members if m[0] != 'linear'], SLOTS)]
        for col, c in enumerate(columns):
            ax = fig.add_subplot(grid[row, col])
            window = shown(c['window'])
            lo, hi = np.nanmin(window), np.nanmax(window)
            pad = .3 * (hi - lo)
            sampled = c['length'] > 48
            ax.axvspan(-.5, c['length'] - .5, color=BAND, alpha=.55, lw=0, zorder=0)
            ax.plot(c['hours'], window, color=INK, lw=.9, zorder=3)
            off = []
            for key, label, colour, style in reference + drawn:
                y = shown(c['estimates'][key])
                if sampled and key != 'linear':
                    ax.plot(c['positions'] - 1, y, ls='none', marker='o', ms=lay['ms'], color=colour,
                            mec='white', mew=.4, zorder=4)
                else:
                    ax.plot(c['positions'] - 1, y, color=colour, ls=style,
                            lw=lay['lw'][0] if key == 'linear' else lay['lw'][1], zorder=2 if key == 'linear' else 4)
                if np.mean((y < lo - pad) | (y > hi + pad)) > .5:
                    off.append(label)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-MARGIN, c['length'] + MARGIN - 1)
            # The nearest-probe comparand falls back to persistence when that probe is dark too,
            # which a flat line would otherwise leave unexplained.
            if any(key == 'nearest_probe' for key, *_ in drawn) and np.allclose(
                    c['estimates']['nearest_probe'], c['estimates']['persistence']):
                off.append('neighbour dark')
            if off:
                ax.text(.99, .05, '; '.join(o if o == 'neighbour dark' else f'{o} off scale'
                                            for o in off),
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=lay['note'],
                        color=MUTED)
            if col == 0:
                ax.set_ylabel(lay['ylabel'].format(unit=UNITS[variable]))
            if row < len(families) - 1:
                ax.tick_params(labelbottom=False)
            if row == 0:
                p = c['placement']
                ax.set_title(lay['title'].format(title=c['spec']['title'], length=c['length'],
                                                 date=f'{c["start"]:%d %b %Y}'),
                             loc='left', fontsize=lay['title_size'], color=INK, fontweight='bold',
                             pad=lay['title_pad'])
                ax.text(0, 1.04, f'drawn from {c["pool"]}\n'
                                 f'this {number(p["here"], "+")}\n'
                                 f'median {number(p["pool_median"], "+")}',
                        transform=ax.transAxes, fontsize=lay['note'], color=MUTED, va='bottom')
                # Every eligible episode's Chronos-minus-linear contrast, the drawn one ringed and
                # the median ticked, so the reader sees where the example sits (review R15).
                strip = ax.inset_axes([.62, 1.08, .38, .42])
                values = shown(np.asarray(p['pool']))
                jitter = np.random.default_rng(0).uniform(-.32, .32, len(values))
                strip.axvline(0, color=MUTED, lw=.9, ls=(0, (3, 2)))
                strip.scatter(values, jitter, s=lay['strip_s'][0], color=MUTED, alpha=.7, lw=0)
                strip.scatter([shown(p['here'])], [0], s=lay['strip_s'][1], facecolor='none',
                              edgecolor=ORANGE, lw=2.0, zorder=3)
                strip.plot([shown(p['pool_median'])] * 2, [-.5, .5], color=INK, lw=1.6)
                span = max(abs(values).max(), 1e-9) * 1.15
                strip.set_xlim(-span, span)
                strip.set_ylim(-.6, .6)
                strip.axis('off')
        legend_ax = fig.add_subplot(grid[row, 2])
        legend_ax.axis('off')
        # The reference is listed once, in the first row; it is drawn identically in all of them.
        entries = drawn + (reference if row == 0 else [])
        handles = [Line2D([], [], color=colour, ls=style, lw=2.2 if key != 'linear' else 1.4)
                   for key, _, colour, style in entries]
        labels = [f'{"linear" if key == "linear" else label}  '
                  + ' · '.join(number(c['errors'][key]) for c in columns)
                  for key, label, _, _ in entries]
        legend_ax.legend(handles, labels, loc='center left', fontsize=lay['legend'], title=family,
                         title_fontproperties={'weight': 'bold', 'size': lay['legend_title']},
                         alignment='left', handlelength=1.8, labelspacing=.3,
                         borderaxespad=0)
    top = fig.axes[-1].get_position()
    fig.text(top.x0, lay['mae_y'], lay['mae_text'].format(unit=UNITS[variable]),
             fontsize=lay['mae'], color=MUTED, va='bottom')
    fig.supxlabel(lay['xlabel_text'], x=.36, y=lay['xlabel_y'], fontsize=lay['xlabel'],
                  color=INK)
    for c in columns:
        record[variable].append({'draw_key': draw_key(variable, c['spec']),
                                 'episode': c['episode'], 'drawn_from': c['pool'],
                                 'gap_start': c['start'].isoformat(),
                                 'scored_positions': int(len(c['positions'])),
                                 'chronos_bidir_minus_linear': {k: round(v, 4) if
                                                                isinstance(v, float) else v
                                                                for k, v in c['placement'].items()},
                                 'mae': {k: round(v, 4) for k, v in c['errors'].items()}})
    fig.savefig(THESIS / f'fig_bay_gap_traces_{variable}.png', bbox_inches='tight' if lay['crop'] else None,
                dpi=300)
    fig.savefig(DECK / f'fig_bay_gap_traces_{variable}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    for target in (THESIS, DECK):
        target.mkdir(parents=True, exist_ok=True)
    _, frames = load_observations()
    record = {'declared_in': 'config/bay_reserved_protocol_v2_addendum30.json', 'run': RUN,
              'selection': f'np.random.default_rng([{DRAW_SEED}, crc32(draw_key)]) over the '
                           'sorted eligible episode ids; the key is the eligibility rule, never '
                           'the title, the error or the truth',
              'climatology': 'declared, fitted outside the masked stretch',
              'oxygen': [], 'temperature': []}
    for variable in ('oxygen', 'temperature'):
        figure(variable, predictions(variable), frames, record)
        for entry in record[variable]:
            print(variable, entry['episode'], 'from', entry['drawn_from'])
            print('   ', {k: v for k, v in sorted(entry['mae'].items(), key=lambda kv: kv[1])})
    OUT.write_bytes(json.dumps(record, indent=2).encode('utf-8'))  # LF on every host
    print('wrote fig_bay_gap_traces_{oxygen,temperature} and', OUT)


if __name__ == '__main__':
    main()
