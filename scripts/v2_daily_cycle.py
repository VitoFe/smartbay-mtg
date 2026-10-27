"""The average day by season at the moorings: what the morning casts do not see.

A companion to the time-depth sections (scripts/v2_ctd_sections.py), asked for by Vito on
2026-09-25: the casts start between 07:00 and 12:00 UTC, so a daily view has to come from the
moored probes, which record every hour at 1 to 5 m.

For each probe, each complete day (at least 20 of its 24 hourly values) is taken relative to its
own mean, which removes the seasonal level and any fixed offset between instruments, and the
median of those departures is taken hour by hour within each season. A probe counts in a season
only with 15 or more complete days. Probes are then averaged within a group of the same area and
depth, never across areas: a first draft averaged by depth alone, and because each depth was then
one area in summer, the gulf mouth's own summer cycle (warmest at night, at three instruments of two
types) read as a property of 2-3 m. Hours are the moorings' interval ends (read_export and the EDA
loader both label an hourly mean at its end); the figure places them at interval centres. The whole
record is used, July 2024 to June 2026. Descriptive: no model fitted, nothing scored.

Left out: the 8 m thermistors on the Mitili current meters record about ten hours a day,
fragmented, and almost nothing in summer; Nodo_S11 at the Santa Teresa buoy reports 6 % of hours.

Writes reports/bay_twin/runs/v2_daily_cycle.json and thesis_template/images/figures/
fig_bay_daily_cycle.png.
Usage: python -m scripts.v2_daily_cycle
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.eda_opportunities.eda_lib import load as load_raw
from scripts.v2_timeseries_figures import BLUE, GRID, MUTED, ORANGE, PURPLE, ROOT, THESIS
from src.data.bay_observations import read_export
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l

OUT = ROOT / 'reports/bay_twin/runs/v2_daily_cycle.json'
FIGURE = THESIS / 'fig_bay_daily_cycle.png'
#: the figure's oxygen and saturation rows are drawn from this analysis rerun with the sondes' oxygen
#: in mg/l and the solubility per litre in mg/l (addendum 60 amendment 3,
#: scripts.v2_oxygen_mgl_daily_cycle); that rerun sets it to None and draws what it computes
OXYGEN_FROM = ROOT / 'reports/bay_twin/runs/v2_oxygen_mgl/C5/v2_daily_cycle.json'
GEOMETRY = ROOT / 'reports/bay_twin/sensor_geometry.csv'
THERMISTORS = {'TMitiliE_02m-Corrente': ('Mitili_Est', 2.0), 'TMitiliO_02m-Corrente': ('Mitili_Ovest', 2.0)}
#: (label, area, line style, variables, rule on (site, nominal depth)); colour is the area, style
#: the depth. Portovenere, at the gulf mouth, is drawn for oxygen only: in summer its temperature
#: cycles disagree from probe to probe (warmest at 01:30-02:30 UTC at Pontile, 08:30 at Benzinaio,
#: 14:30 and 21:30 at the two Secco probes; per-probe cycles in the JSON), so their average would
#: describe no place.
BOTH, OXYGEN = ('temperature', 'oxygen', 'saturation'), ('oxygen', 'saturation')
GROUPS = [('Baia S. Teresa, 1 m', 'baia', '-', BOTH, lambda site, z: site == 'Baia_S.Teresa' and z == 1.0),
          ('Baia Pontone, 5 m', 'baia', '--', BOTH, lambda site, z: site == 'Baia_Pontone'),
          ('Mitili, 2 m', 'mitili', '-', BOTH, lambda site, z: site.startswith('Mitili') and z == 2.0),
          ('Mitili, 5 m', 'mitili', '--', BOTH, lambda site, z: site.startswith('Mitili') and z == 5.0),
          ('Portovenere, 2 m', 'portovenere', '-', OXYGEN, lambda site, z: site.startswith('Portovenere'))]
STYLE = {label: (area, ls) for label, area, ls, _, _ in GROUPS}
VARIABLES = {label: variables for label, _, _, variables, _ in GROUPS}
#: the same-area pairs whose difference is the daily part of the upper-column layering
PAIRS = {'Baia, 1 m minus 5 m': ('Baia S. Teresa, 1 m', 'Baia Pontone, 5 m'),
         'Mitili, 2 m minus 5 m': ('Mitili, 2 m', 'Mitili, 5 m')}
AREA_COLOUR = {'baia': BLUE, 'mitili': ORANGE, 'portovenere': PURPLE}
MIN_HOURS, MIN_DAYS = 20, 15
CAST_WINDOW = (7.0, 12.0)                 # UTC; 99 % of casts start inside it (v2_ctd_sections.json)
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
          6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
TITLE = {'DJF': 'December to February', 'MAM': 'March to May', 'JJA': 'June to August',
         'SON': 'September to November'}

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'legend.fontsize': 8.5,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
#: The figure is drawn at the size it prints at (\textwidth, 416.7 pt, by the earlier drawing's height there)
#: in these type sizes and saved without cropping, so that no text prints below 7 pt and the page does not move.
SIZE_IN = (416.7 / 72, 5.8141)
PRINT_STYLE = {'font.size': 7.5, 'axes.titlesize': 7.5, 'legend.fontsize': 7, 'xtick.labelsize': 7,
               'ytick.labelsize': 7, 'axes.labelsize': 7.5}


def group_of(site, depth):
    return next((label for label, _, _, _, rule in GROUPS if rule(site, depth)), None)


def records():
    """(name, variable, group, hourly series) for every moored record that belongs to a group."""
    geo = pd.read_csv(GEOMETRY).set_index('sensor')
    out, left_out = [], []
    for name, row in geo.iterrows():
        group = group_of(row.site, float(row.nominal_depth_m))
        if group is None:
            left_out.append(name)
            continue
        frame = read_export(ROOT / 'data/raw' / f'{name}.txt', include_auxiliary=True)[1]
        out.append((name, 'temperature', group, frame.temperature))
        if 'oxygen' in frame:
            out.append((name, 'oxygen', group, frame.oxygen))
            # percent saturation at the sonde's own temperature and salinity (Garcia & Gordon 1992,
            # practical salinity: the thesis's solubility, and no equation of state involved)
            solubility = o2sol_garcia_gordon_umol_per_l(frame.temperature.to_numpy(), frame.salinity.to_numpy())
            out.append((name, 'saturation', group, 100 * frame.oxygen / solubility))
    for name, (site, depth) in THERMISTORS.items():
        out.append((name, 'temperature', group_of(site, depth), load_raw(name + '.txt')[0]['Temperatura(°C)']))
    return out, left_out


def day_of(index):
    """An hourly mean labelled 00:00 closes the previous day."""
    return (index - pd.Timedelta(minutes=30)).floor('D')


def composite(series, level=False):
    """{season: (median by interval-centre hour, complete days)}: of each day's departures from its
    own mean, or of the values themselves when the level is the point (saturation against 100 %)."""
    s = series.dropna()
    day = day_of(s.index)
    n = s.groupby(day).size()
    s = s[day.isin(n[n >= MIN_HOURS].index)]
    day = day_of(s.index)
    departure = s if level else s - s.groupby(day).transform('mean')
    frame = pd.DataFrame({'v': departure.to_numpy(),
                          'hour': (s.index - pd.Timedelta(minutes=30)).hour + 0.5,
                          'season': pd.Series(day.month).map(SEASON).to_numpy(), 'day': day})
    out = {}
    for season, g in frame.groupby('season'):
        days = g.day.nunique()
        if days >= MIN_DAYS:
            out[season] = (g.groupby('hour').v.median().reindex(np.arange(0.5, 24, 1.0)), int(days))
    return out


def summary(curve):
    cast = curve[(curve.index > CAST_WINDOW[0]) & (curve.index < CAST_WINDOW[1])].mean()
    return {'mean': round(float(curve.mean()), 3), 'min': round(float(curve.min()), 3),
            'max': round(float(curve.max()), 3), 'range': round(float(curve.max() - curve.min()), 3),
            'hour_min_utc': float(curve.idxmin()), 'hour_max_utc': float(curve.idxmax()),
            'cast_window_mean': round(float(cast), 3),
            'cast_window_above_min': round(float(cast - curve.min()), 3),
            'cast_window_below_max': round(float(curve.max() - cast), 3)}


def by_group(recs, variable):
    """{season: {group: {'curve': mean of the probes' curves, 'probes': [...], 'days': [...]}}}"""
    out = {}
    for name, var, group, series in recs:
        if var != variable or variable not in VARIABLES[group]:
            continue
        for season, (curve, days) in composite(series, level=variable == 'saturation').items():
            slot = out.setdefault(season, {}).setdefault(group, {'curves': [], 'probes': [], 'days': []})
            slot['curves'].append(curve)
            slot['probes'].append(name)
            slot['days'].append(days)
    for season in out:
        for slot in out[season].values():
            slot['curve'] = pd.concat(slot.pop('curves'), axis=1).mean(axis=1)
    return out


def per_probe(recs):
    """Every probe's own daily cycle by season, grouped or not, so a group can be checked."""
    out = {}
    for name, var, group, series in recs:
        for season, (curve, days) in composite(series, level=var == 'saturation').items():
            out.setdefault(var, {}).setdefault(name, {})[season] = {**summary(curve), 'complete_days': days,
                                                                    'group': group}
    return out


def figure(temperature, oxygen, saturation):
    with plt.rc_context(PRINT_STYLE):
        draw(temperature, oxygen, saturation)


def draw(temperature, oxygen, saturation):
    fig, axes = plt.subplots(3, 4, figsize=SIZE_IN, sharex=True, sharey='row')
    order = [label for label, *_ in GROUPS]
    rows = ((temperature, 'Temperature (°C)\ndeparture from daily mean', 0),
            (oxygen, 'Oxygen (mg/ℓ)\ndeparture from daily mean', 0),
            (saturation, 'Oxygen saturation (%)', 100))
    for j, season in enumerate(SEASONS):
        for i, (data, name, reference) in enumerate(rows):
            ax = axes[i, j]
            ax.axvspan(*CAST_WINDOW, color=GRID, lw=0, zorder=0)
            ax.axhline(reference, color=MUTED, lw=0.5)
            for label in order:
                slot = data.get(season, {}).get(label)
                if slot is None:
                    continue
                area, style = STYLE[label]
                ax.plot(slot['curve'].index, slot['curve'].to_numpy(), color=AREA_COLOUR[area], ls=style,
                        lw=1.1, label=label)
            if j == 0:
                ax.set_ylabel(name)
            if i == 0:
                ax.set_title(TITLE[season])
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xlim(0, 24)
            ax.grid(False)
        axes[-1, j].set_xlabel('Hour (UTC)')
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labels = [k for k in order if k in handles]
    fig.legend([handles[k] for k in labels], labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)


def report_of(data):
    out = {}
    for season in SEASONS:
        for label, slot in data.get(season, {}).items():
            out.setdefault(season, {})[label] = {
                **summary(slot['curve']), 'probes': slot['probes'], 'complete_days': slot['days'],
                'curve': [round(float(v), 3) for v in slot['curve'].to_numpy()]}
    return out


def curves(block):
    """A report block's curves in the form by_group returns, for drawing."""
    return {season: {label: {'curve': pd.Series(v['curve'], index=np.arange(0.5, 24, 1.0), dtype=float)}
                     for label, v in groups.items()}
            for season, groups in block.items()}


def main():
    recs, left_out = records()
    temperature, oxygen = by_group(recs, 'temperature'), by_group(recs, 'oxygen')
    saturation = by_group(recs, 'saturation')
    drawn = (oxygen, saturation)
    if OXYGEN_FROM is not None:
        mgl = json.loads(OXYGEN_FROM.read_text())
        drawn = (curves(mgl['oxygen']), curves(mgl['saturation_percent']))
    figure(temperature, *drawn)
    t, o, sat = report_of(temperature), report_of(oxygen), report_of(saturation)
    # the daily part of the upper-minus-lower difference within one area: offset-free, since each
    # curve is already a departure from its own daily mean
    layering = {}
    for pair, (upper, lower) in PAIRS.items():
        for season in SEASONS:
            d = temperature.get(season, {})
            if upper in d and lower in d:
                layering.setdefault(pair, {})[season] = summary(d[upper]['curve'] - d[lower]['curve'])
    geo = pd.read_csv(GEOMETRY).set_index('sensor')
    xy = geo[['x_m', 'y_m']]
    report = {
        'question': 'Vito, 2026-09-25: a 24-hour view by season, which the casts (07:00-12:00 UTC) cannot give.',
        'descriptive': 'No model fitted, nothing scored.',
        'method': {'departure_from': 'each complete day\'s own mean', 'complete_day_min_hours': MIN_HOURS,
                   'min_complete_days_per_probe_and_season': MIN_DAYS, 'statistic': 'median by hour',
                   'averaged': 'within a group of one area and depth, never across areas',
                   'hours': 'interval centres, UTC', 'cast_window_utc': list(CAST_WINDOW),
                   'period': 'whole moored record, July 2024 to June 2026'},
        'groups': {label: {'variables': list(VARIABLES[label]),
                           'probes': sorted({name for name, _var, group, _s in recs if group == label})}
                   for label, *_rest in GROUPS},
        'left_out': left_out + ['TMitiliE_08m-Corrente', 'TMitiliO_08m-Corrente'],
        'baia_pair_distance_m': round(float(np.linalg.norm(
            xy.loc['TBaiaP_01m-CTD(calibrato)'].to_numpy() - xy.loc['Nodo_S08'].to_numpy())), 0),
        'temperature': t, 'oxygen': o, 'saturation_percent': sat,
        'upper_minus_lower_daily_part': layering,
        'per_probe': per_probe(recs),
    }
    OUT.write_text(json.dumps(report, indent=2))
    for name, block in (('temperature', t), ('oxygen', o), ('saturation', sat)):
        print(name)
        for season in SEASONS:
            for label, v in block.get(season, {}).items():
                print(f'  {season} {label:20s}: range {v["range"]:7.3f}  min {v["hour_min_utc"]:4.1f}  '
                      f'max {v["hour_max_utc"]:4.1f}  cast-min {v["cast_window_above_min"]:6.3f}  '
                      f'max-cast {v["cast_window_below_max"]:6.3f}  days {v["complete_days"]}')
    for pair, seasons in layering.items():
        print(pair)
        for season, v in seasons.items():
            print(f'  {season}: range {v["range"]:.3f}  max at {v["hour_max_utc"]}  '
                  f'below max at casts {v["cast_window_below_max"]:.3f}')
    print('Baia pair distance (m):', report['baia_pair_distance_m'])
    print('wrote', OUT.relative_to(ROOT), FIGURE.relative_to(ROOT))


if __name__ == '__main__':
    main()
