"""Addendum 37: rapid oxygen declines typed by the co-located salinity change.

Episodes come from src.experiments.v2_warning_episodes unchanged, with only its period moved:
development 2024-07-24 to 2025-12-31, and Warning-2026. An episode is a WATER-MASS change if the
co-located salinity moves by more than 0.5 PSU over the 6 h window ending at its onset, IN-PLACE if
by 0.5 PSU or less, UNTYPED if either end is missing. Three declared tests on the development years:

    T1  the total-scale pH falls over the window (>= 90 % overall, >= 80 % within each type)
    T2  12 h mean scalar wind before onset, as an anomaly from its month x hour median: positive for
        water-mass episodes and larger than for in-place ones (one-sided Mann-Whitney, p < 0.05)
    T3  24 h air-temperature change before onset, as the same anomaly: negative for water-mass
        episodes and more negative than for in-place ones (one-sided Mann-Whitney, p < 0.05)

Declared in config/bay_reserved_protocol_v2_addendum37.json.
Usage: python -m scripts.v2_decline_types [--tables]
  --tables  rewrite the two tables from the saved runs/v2_decline_types.json, recomputing nothing
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import src.experiments.v2_warning_episodes as W
from src.units import OXYGEN_UNIT_TEX, column_decimals, factor, mg_decimals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_decline_types.json'
THRESHOLD_PSU = 0.5
DEV = ('2024-07-24', '2026-01-01')
RESERVED = ('2026-01-01', '2026-06-20')


def named_export(path, columns):
    """Selected columns of a raw export by the names in its own header, hourly as read_export does."""
    with open(path, encoding='cp1252') as stream:
        head = stream.read(40000)
    names = next(l.lstrip('#').split() for l in head.splitlines() if l.startswith('#') and 'Data' in l and 'Ora' in l)
    f = pd.read_csv(path, comment='#', sep=r'\s+', header=None, encoding='cp1252', low_memory=False)
    f.columns = names[:f.shape[1]]
    t = pd.to_datetime(f.iloc[:, 0].astype(str) + ' ' + f.iloc[:, 1].astype(str),
                       format='%d/%m/%Y %H:%M:%S', errors='coerce', utc=True)
    out = f[columns].apply(pd.to_numeric, errors='coerce')
    out = out.where(out > -999)
    out.index = t
    out = out[out.index.notna()].sort_index().groupby(level=0).mean()
    return out.resample('1h', closed='left', label='right').mean()


def episodes(start, end):
    W.START, W.END = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    rows = []
    for name, frame in W.series().items():
        try:
            ph = named_export(ROOT / 'data/raw' / f'{name}.txt', ['pH_T'])['pH_T'].reindex(frame.index)
        except KeyError:
            ph = pd.Series(np.nan, index=frame.index)
        for ep in W.cluster(W.crossings_of(frame.oxygen), 12):
            t = ep[0]
            t0 = t - pd.Timedelta(hours=W.EVENT_WIN)
            ds = frame.salinity.get(t, np.nan) - frame.salinity.get(t0, np.nan) if 'salinity' in frame else np.nan
            kind = 'untyped' if not np.isfinite(ds) else ('water_mass' if abs(ds) > THRESHOLD_PSU else 'in_place')
            rows.append({'sensor': name, 'onset': t, 'hour': t.hour, 'month': t.month, 'dS': ds, 'type': kind,
                         'dT_water': frame.temperature.get(t, np.nan) - frame.temperature.get(t0, np.nan),
                         'dpH_T': ph.get(t, np.nan) - ph.get(t0, np.nan),
                         'drop': frame.oxygen.get(t0, np.nan) - frame.oxygen.get(t, np.nan)})
    return pd.DataFrame(rows)


SUMMER, SPRING = (6, 7, 8, 9, 10), (2, 3, 4)


def typing_power(start, end):
    """What the salinity criterion can see, and how large the declines are against the ordinary
    nocturnal fall (descriptive, not a declared test). For each sonde with a
    salinity record: the share of six-hour windows whose salinity changes by more than the typing
    threshold, in June to October and in February to April; and the median over June-October days
    of each day's largest six-hour oxygen fall, the scale a decline without a salinity change is
    measured against."""
    W.START, W.END = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    out = {}
    for name, frame in W.series().items():
        block = {}
        if 'salinity' in frame and frame.salinity.notna().any():
            ds = (frame.salinity - frame.salinity.shift(W.EVENT_WIN)).dropna()
            for label, months in (('june_october', SUMMER), ('february_april', SPRING)):
                sub = ds[ds.index.month.isin(months)]
                block[f'windows_{label}'] = int(len(sub))
                block[f'share_above_threshold_{label}'] = finite((sub.abs() > THRESHOLD_PSU).mean(), 4) if len(sub) else None
        drop = -(frame.oxygen - frame.oxygen.shift(W.EVENT_WIN)).dropna()
        daily = drop.groupby(drop.index.floor('D')).max()
        summer = daily[daily.index.month.isin(SUMMER)]
        block['summer_days'] = int(len(summer))
        block['summer_daily_largest_drop_median'] = finite(summer.median(), 2) if len(summer) else None
        out[name.split('_')[0]] = block
    return out


def forcing():
    m = named_export(ROOT / 'data/raw/T2-Meteo.txt', ["Velocita'_Scalare_del_Vento(m/s)", 'Temperatura(°C)'])
    m.columns = ['wind', 'air']
    f = pd.DataFrame({'wind12': m.wind.rolling(12, min_periods=9).mean(), 'dair24': m.air - m.air.shift(24)})
    dev = f.loc[DEV[0]:'2025-12-31 23:00']
    ref = dev.groupby([dev.index.month, dev.index.hour]).median()
    return f, ref


def anomaly(e, f, ref):
    out = []
    for _, r in e.iterrows():
        key = (r.onset.month, r.onset.hour)
        now = f.loc[r.onset] if r.onset in f.index else pd.Series({'wind12': np.nan, 'dair24': np.nan})
        out.append({'wind_anom': now.wind12 - ref.loc[key, 'wind12'], 'cool_anom': now.dair24 - ref.loc[key, 'dair24']})
    return pd.concat([e.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def finite(x, digits):
    """Rounded, or None where there is nothing to summarise (valid JSON has no NaN)."""
    return round(float(x), digits) if np.isfinite(x) else None


def summary(e):
    by = {}
    for kind, g in e.groupby('type'):
        by[kind] = {'n': int(len(g)), 'onset_18_06_share': round(float(g.hour.isin(list(range(18, 24)) + list(range(0, 7))).mean()), 3),
                    'onset_00_06': int(g.hour.isin(range(0, 7)).sum()),
                    'months': {int(k): int(v) for k, v in g.month.value_counts().sort_index().items()},
                    'dS_median': finite(g.dS.median(), 3), 'dT_water_median': finite(g.dT_water.median(), 3),
                    'dT_water_quartiles': [finite(g.dT_water.quantile(q), 3) for q in (.25, .75)],
                    'drop_median': finite(g['drop'].median(), 2),
                    'drop_rate_median_per_h': finite(g['drop'].median() / W.EVENT_WIN, 2)}
    return by


EVENING_TO_DAWN = list(range(18, 24)) + list(range(0, 7))


def timing(start, end, onsets):
    """Descriptive, already seen in the exploratory pass: onsets (first crossings, the END of the
    six-hour drop) in 00-06 UTC against the share of days whose largest 6 h drop ends there, on the
    same sondes and period (the published reference). The same reference is given for 18-06 UTC,
    the window the recommendations rest on, since a count there means nothing without it."""
    from scipy.stats import binomtest
    W.START, W.END = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    ends = []
    for frame in W.series().values():
        drop = -(frame.oxygen - frame.oxygen.shift(W.EVENT_WIN)).dropna()
        ends += [t.hour for t in drop.groupby(drop.index.floor('D')).idxmax()]
    share = float(np.isin(ends, range(0, 7)).mean())
    inside = int(np.isin(onsets, range(0, 7)).sum())
    share_night = float(np.isin(ends, EVENING_TO_DAWN).mean())
    night = int(np.isin(onsets, EVENING_TO_DAWN).sum())
    return {'episodes': len(onsets), 'onsets_00_06': inside, 'reference_share': round(share, 3),
            'p_greater': round(float(binomtest(inside, len(onsets), share, alternative='greater').pvalue), 6),
            'onsets_18_06': night, 'reference_share_18_06': round(share_night, 3),
            'p_greater_18_06': round(float(binomtest(night, len(onsets), share_night, alternative='greater').pvalue), 6)}


def main():
    f, ref = forcing()
    dev = anomaly(episodes(*DEV), f, ref)
    res = anomaly(episodes(*RESERVED), f, ref)
    seen = {'development': timing(*DEV, dev.hour.to_numpy()), 'reserved': timing(*RESERVED, res.hour.to_numpy())}
    typed = dev[dev.type != 'untyped']
    ph = typed.dropna(subset=['dpH_T'])
    t1 = {'with_pH': int(len(ph)), 'share_drop': round(float((ph.dpH_T < 0).mean()), 3),
          'by_type': {k: {'n': int(len(g)), 'share_drop': round(float((g.dpH_T < 0).mean()), 3),
                          'dpH_median': round(float(g.dpH_T.median()), 4)} for k, g in ph.groupby('type')}}
    t1['held'] = bool(t1['share_drop'] >= .9 and all(v['share_drop'] >= .8 for v in t1['by_type'].values()))
    wm, ip = typed[typed.type == 'water_mass'], typed[typed.type == 'in_place']
    t2p = mannwhitneyu(wm.wind_anom.dropna(), ip.wind_anom.dropna(), alternative='greater').pvalue
    t3p = mannwhitneyu(wm.cool_anom.dropna(), ip.cool_anom.dropna(), alternative='less').pvalue
    t2 = {'water_mass_median': round(float(wm.wind_anom.median()), 3), 'in_place_median': round(float(ip.wind_anom.median()), 3),
          'n': [int(wm.wind_anom.notna().sum()), int(ip.wind_anom.notna().sum())], 'p_one_sided': round(float(t2p), 4)}
    t2['held'] = bool(t2['water_mass_median'] > 0 and t2p < .05)
    t3 = {'water_mass_median': round(float(wm.cool_anom.median()), 3), 'in_place_median': round(float(ip.cool_anom.median()), 3),
          'n': [int(wm.cool_anom.notna().sum()), int(ip.cool_anom.notna().sum())], 'p_one_sided': round(float(t3p), 4)}
    t3['held'] = bool(t3['water_mass_median'] < 0 and t3p < .05)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum37.json', 'threshold_psu': THRESHOLD_PSU,
              'development': {'period': DEV, 'episodes': int(len(dev)), 'types': summary(dev),
                              'typing_power': typing_power(*DEV)},
              'reserved': {'period': RESERVED, 'episodes': int(len(res)), 'types': summary(res),
                           'rows': [{'sensor': r.sensor.split('_')[0], 'onset': str(r.onset)[:16], 'type': r.type,
                                     'dS': round(float(r.dS), 2) if np.isfinite(r.dS) else None,
                                     'dT': round(float(r.dT_water), 2) if np.isfinite(r.dT_water) else None,
                                     'dpH_T': round(float(r.dpH_T), 3) if np.isfinite(r.dpH_T) else None}
                                    for r in res.itertuples()]},
              'tests': {'T1_pH_witness': t1, 'T2_wind_before': t2, 'T3_cooling_before': t3},
              'timing_seen_in_exploration': seen}
    write_keeping_endings(OUT, json.dumps(report, indent=2, default=str))   # same bytes on every platform
    write_tables(report)
    print(json.dumps(report, indent=1, default=str))


def write_keeping_endings(path, text):
    """Write `text` with the line ending the file already has (CRLF or LF), so a regeneration
    changes only the lines that changed; a new file gets LF."""
    crlf = path.exists() and b'\r\n' in path.read_bytes()
    path.write_bytes((text.replace('\n', '\r\n') if crlf else text).encode('utf-8'))


TABLES = ROOT / 'thesis_template/content/tables'
NAMES = {'TBaiaP': 'Baia Santa Teresa', 'TMitiliE': 'Mitili Est'}
#: Table labels. A decline with no salinity change is named by what was measured: in summer, when the
#: column is nearly uniform in salinity, the criterion cannot exclude oxygen-poorer water of the same
#: salinity arriving at the sensor. JSON keys keep `in_place`.
KIND = {'in_place': 'no salinity change', 'water_mass': 'water mass', 'untyped': '---'}


def signed(v, d):
    return '---' if v is None else f'${v:+.{d}f}$'


#: the sondes' oxygen is per kilogram; the tables print c times it, in mg/l, at the decimals that keep
#: the significant figures it was printed with (src.units)
O2 = factor()


def mg(value, decimals):
    """An oxygen value printed at `decimals` in the sondes' unit, as mg/l keeping its figures."""
    return f'{O2 * value:.{mg_decimals(value, decimals)}f}'


def mg_span(values, decimals):
    """'a to b' in mg/l, both ends at the decimals the larger needs."""
    d = column_decimals(values, decimals)
    return f'{O2 * min(values):.{d}f} to {O2 * max(values):.{d}f}'


def write_tables(r):
    """The Warning-2026 episode table's hydrography columns, and the typing across both cohorts."""
    ep = json.loads((ROOT / 'reports/bay_twin/runs/v2_warning_episodes.json').read_text())
    by_onset = {e['onset'][:16]: e for e in ep['episodes']}
    shown = [by_onset[row['onset']] for row in r['reserved']['rows']]
    # one number of decimals per column, the most any of its values needs in mg/l
    d_drop = column_decimals([e['drop_umol_l'] for e in shown], 2)
    d_min = column_decimals([e['min_oxygen'] for e in shown], 1)
    lines = []
    for row in r['reserved']['rows']:
        e = by_onset[row['onset']]
        rec = '---' if e['recovery_h'] is None else f"{e['recovery_h']:.0f}"
        lines.append(f"{NAMES[row['sensor']]} & {row['onset']} & {e['crossings']} & {O2 * e['drop_umol_l']:.{d_drop}f} & "
                     f"{O2 * e['min_oxygen']:.{d_min}f} & {rec} & {signed(row['dS'], 2)} & {signed(row['dT'], 2)} & "
                     f"{signed(row['dpH_T'], 3)} & {KIND[row['type']]} \\\\")
    g = ep['by_gap']['12h']
    # the operating point and its 2026 crossing scores, from the one place the selection is recorded
    rate, consec, cooldown = W.operating_point()
    if (ep['operating_point']['rate'], ep['operating_point']['consecutive'],
            ep['operating_point']['cooldown_h']) != (rate, consec, cooldown):
        raise SystemExit('runs/v2_warning_episodes.json is not at the selected operating point: rerun '
                         'src.experiments.v2_warning_episodes first')
    sel = json.loads(W.POINT_SOURCE.read_bytes())['warning_2026']['crossing_level']['selected_point']
    mins = [e['min_oxygen'] for e in ep['episodes']]
    recs = [e['recovery_h'] for e in ep['episodes'] if e['recovery_h'] is not None]
    oh = ep['onset_hours']
    # the caption and the note spell these counts out in words: stop rather than print a stale one
    spelled = {gap: (ep['by_gap'][gap]['crossings'], ep['by_gap'][gap]['episodes']) for gap in ('6h', '12h', '24h')}
    if set(spelled.values()) != {(12, 7)} or (oh['episodes_inside'], oh['episodes']) != (7, 7):
        raise SystemExit(f'the Warning-2026 counts changed ({spelled}, {oh}): reword the episode table')
    rows_tex = '\n'.join(lines)
    # The episode artifact measures salinity from six hours before the first crossing to the last
    # crossing; the typing uses the six hours ending at the first crossing. Name any episode whose
    # type depends on which.
    window = []
    for row in r['reserved']['rows']:
        e = by_onset[row['onset']]
        ds = e['salinity_change']
        if ds is None or row['type'] == 'untyped':
            continue
        extended = 'water_mass' if abs(ds) > r['threshold_psu'] else 'in_place'
        if extended != row['type']:
            t = pd.Timestamp(row['onset'])
            as_type = {'water_mass': 'a change of water mass', 'in_place': 'having no salinity change'}[extended]
            window.append(f"Carried on to the last crossing, the {t.day} {t.strftime('%B')} decline at "
                          f"{NAMES[row['sensor']]} changes salinity by ${ds:+.2f}$ and would be typed as {as_type}.")
    window_tex = (' ' + ' '.join(window)) if window else ''
    episodes_tex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Rapid-decline episodes on \textsc{{Warning-2026}}, and the operating point selected on development.}}
\label{{tab:bay_warning_episodes}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}Xlrrrrrrr>{{\raggedright\arraybackslash}}X@{{}}}}
\toprule
\textbf{{Sensor}} & \makecell[l]{{\textbf{{First crossing}}\\\textbf{{(UTC)}}}} & \makecell[r]{{\textbf{{Cross-}}\\\textbf{{ings}}}} &
\makecell[r]{{\textbf{{Drop}}\\\textbf{{({OXYGEN_UNIT_TEX})}}}} & \makecell[r]{{\textbf{{Minimum}}\\\textbf{{({OXYGEN_UNIT_TEX})}}}} &
\makecell[r]{{\textbf{{Recovery}}\\\textbf{{(h)}}}} & \makecell[r]{{\textbf{{$\Delta S$}}}} &
\makecell[r]{{\textbf{{$\Delta T$}}\\\textbf{{($^\circ$C)}}}} & \textbf{{$\Delta$pH}} & \textbf{{Type}} \\
\midrule
{rows_tex}
\midrule
\multicolumn{{10}}{{@{{}}>{{\raggedright\arraybackslash}}p{{\textwidth}}@{{}}}}{{\textit{{Detector at the operating point selected on development data ($>{mg(rate, 1)}\,$mg/($\ell$\,h), {consec} consecutive, {cooldown}\,h cooldown)}}}} \\
\multicolumn{{5}}{{@{{}}l}}{{Alerts issued}} & \multicolumn{{5}}{{r}}{{{g['alerts']}}} \\
\multicolumn{{5}}{{@{{}}l}}{{Episodes matched, one-to-one}} & \multicolumn{{5}}{{r}}{{{g['matched_one_to_one']} of {g['episodes']}}} \\
\multicolumn{{5}}{{@{{}}l}}{{\textbf{{Episode recall / precision}}}} & \multicolumn{{5}}{{r}}{{\textbf{{{g['episode_recall']:.3f} / {g['episode_precision']:.3f}}}}} \\
\multicolumn{{5}}{{@{{}}l}}{{Crossing recall / precision, for comparison}} & \multicolumn{{5}}{{r}}{{{sel['recall_causal']:.3f} / {sel['precision_causal']:.3f}}} \\
\bottomrule
\end{{tabularx}}

\vspace{{2pt}}
\scriptsize\justifying\noindent The twelve crossings are seven episodes at clustering gaps of $6$,
$12$ and $24$\,h; matching is one-to-one and causal, by the smallest lead within $24$\,h, zero
included. $\Delta S$, $\Delta T$ and $\Delta$pH are the changes over the six hours ending at the
first crossing; the type follows Table~\ref{{tab:bay_decline_types}}. Counted by episode, precision
falls from ${sel['precision_causal']:.3f}$ to ${g['episode_precision']:.3f}$, because every alert
before a decline counts by crossing, one alert by episode. Exact $95\,\%$ intervals: recall
$[{g['episode_recall_exact_95'][0]:.2f}, {g['episode_recall_exact_95'][1]:.2f}]$ on seven episodes,
precision $[{g['episode_precision_exact_95'][0]:.2f}, {g['episode_precision_exact_95'][1]:.2f}]$ on
{g['alerts']} alerts. The minima, ${mg_span(mins, 0).replace(' to ', '$ to $')}\,${OXYGEN_UNIT_TEX},
stay far above the $2\,${OXYGEN_UNIT_TEX} hypoxia threshold. The three Mitili Est declines with a
salinity record came with salinity rising by more than a practical salinity unit; the fourth has no
salinity record.{window_tex} The declines recover in ${min(recs):.0f}$ to ${max(recs):.0f}$\,h;
with no wiper or maintenance record for these sondes, an instrument cause is not fully
excluded.\artifact{{runs/v2_warning_episodes.json}}%
\artifact{{runs/v2_decline_types.json}}\artifact{{runs/v2_warning_grid_ext.json}}
\end{{table}}
"""
    write_keeping_endings(TABLES / 'table_bay_warning_episodes.tex', episodes_tex)

    d, s = r['development']['types'], r['reserved']['types']
    t = r['tests']
    tim = r['timing_seen_in_exploration']

    def cell(block, kind, key, fmt='{}'):
        return fmt.format(block[kind][key]) if kind in block else '---'
    rows = []
    for kind in ('in_place', 'water_mass'):
        rows.append(f"{KIND[kind].capitalize()} & {cell(d, kind, 'n')} & {cell(d, kind, 'dS_median', '{:+.2f}')} & "
                    f"{cell(d, kind, 'dT_water_median', '{:+.2f}')} & "
                    f"{sum(v for k, v in d[kind]['months'].items() if int(k) in (6, 7, 8, 9, 10))} & "
                    f"{t['T1_pH_witness']['by_type'][kind]['share_drop']:.2f} & {cell(s, kind, 'n')} \\\\")
    body = '\n'.join(rows)
    tp = r['development']['typing_power']

    def span(key, scale=1.0, digits=0):
        v = [b[key] * scale for b in tp.values() if b.get(key) is not None]
        return f'{min(v):.{digits}f} to {max(v):.{digits}f}'
    ip = d['in_place']
    power = (rf"From June to October, when {sum(v for k, v in ip['months'].items() if int(k) in (6, 7, 8, 9, 10))} "
             rf"of the {ip['n']} declines with no salinity change occur, {span('share_above_threshold_june_october', 100)}\,\% "
             rf"of six-hour windows at each sonde change salinity by more than $0.5$ (February to April: "
             rf"{span('share_above_threshold_february_april', 100)}\,\%), so in summer the rule cannot separate a "
             rf"drawdown at the sensor from oxygen-poorer water of the same salinity. Those declines fall by a "
             rf"median of ${mg(ip['drop_median'], 1)}\,${OXYGEN_UNIT_TEX} in six hours (the median June-to-October day's "
             rf"largest six-hour fall: "
             rf"{mg_span([b['summer_daily_largest_drop_median'] for b in tp.values() if b.get('summer_daily_largest_drop_median') is not None], 1)}\,{OXYGEN_UNIT_TEX} "
             rf"by sonde), their temperature change's interquartile range, ${ip['dT_water_quartiles'][0]:+.2f}$ "
             rf"to ${ip['dT_water_quartiles'][1]:+.2f}\,^\circ$C, spanning zero.")
    tex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Rapid oxygen declines typed by the co-located salinity change.}}
\label{{tab:bay_decline_types}}
\footnotesize
\begin{{tabularx}}{{\textwidth}}{{@{{}}Xrrrrrr@{{}}}}
\toprule
& \multicolumn{{5}}{{c}}{{\textbf{{Development, {r['development']['episodes']} episodes}}}} & \textbf{{2026}} \\
\cmidrule(lr){{2-6}}\cmidrule(lr){{7-7}}
\textbf{{Type}} & $n$ & \makecell[r]{{median\\$\Delta S$}} & \makecell[r]{{median $\Delta T$\\($^\circ$C)}} &
\makecell[r]{{June to\\October}} & \makecell[r]{{share with\\a pH drop}} & $n$ \\
\midrule
{body}
\midrule
\multicolumn{{7}}{{@{{}}>{{\raggedright\arraybackslash}}p{{\textwidth}}@{{}}}}{{\textit{{Timing: first crossings in a window against the share of days whose largest six-hour drop ends there}}}} \\
\multicolumn{{3}}{{@{{}}l}}{{Development years, 00--06\,UTC}} & \multicolumn{{4}}{{r}}{{{tim['development']['onsets_00_06']} of {tim['development']['episodes']} against {tim['development']['reference_share']:.3f}, $p = {tim['development']['p_greater']:.2f}$}} \\
\multicolumn{{3}}{{@{{}}l}}{{\textsc{{Warning-2026}}, 00--06\,UTC}} & \multicolumn{{4}}{{r}}{{{tim['reserved']['onsets_00_06']} of {tim['reserved']['episodes']} against {tim['reserved']['reference_share']:.3f}, $p = {tim['reserved']['p_greater']:.3f}$}} \\
\multicolumn{{3}}{{@{{}}l}}{{Development years, 18--06\,UTC}} & \multicolumn{{4}}{{r}}{{{tim['development']['onsets_18_06']} of {tim['development']['episodes']} against {tim['development']['reference_share_18_06']:.3f}, $p = {tim['development']['p_greater_18_06']:.2f}$}} \\
\multicolumn{{3}}{{@{{}}l}}{{\textsc{{Warning-2026}}, 18--06\,UTC}} & \multicolumn{{4}}{{r}}{{{tim['reserved']['onsets_18_06']} of {tim['reserved']['episodes']} against {tim['reserved']['reference_share_18_06']:.3f}, $p = {tim['reserved']['p_greater_18_06']:.2f}$}} \\
\bottomrule
\end{{tabularx}}

\vspace{{2pt}}
\scriptsize\justifying\noindent Episodes follow the rule of Table~\ref{{tab:bay_warning_episodes}} on
every Gen.~2 oxygen sonde. A decline whose salinity changes by $0.5$ or less over the six hours ending
at its first crossing is labelled by what was measured, no salinity change. Every typed development
episode with a total-scale pH record, {t['T1_pH_witness']['with_pH']} of them, shows a pH drop. The
twelve hours before a water-mass change are no windier than usual for the month and hour (median
anomaly ${t['T2_wind_before']['water_mass_median']:+.2f}$\,m/s, $p = {t['T2_wind_before']['p_one_sided']:.2f}$),
and the day before is not colder (median anomaly ${t['T3_cooling_before']['water_mass_median']:+.2f}\,^\circ$C,
$p = {t['T3_cooling_before']['p_one_sided']:.2f}$).
{power}\artifact{{runs/v2_decline_types.json}}
\end{{table}}
"""
    write_keeping_endings(TABLES / 'table_bay_decline_types.tex', tex)


if __name__ == '__main__':
    import sys
    if '--tables' in sys.argv[1:]:
        write_tables(json.loads(OUT.read_bytes()))
    else:
        main()
