"""Addendum-20: cluster threshold crossings into episodes, and look at what they are.

The released warning evaluation counts crossings, meaning hours at which the six-hour drop exceeds
30 umol/L. A single physical decline lasting several hours produces several of them, and a
detector standing near it is credited once per crossing, so recall over crossings is not recall
over events. This clusters crossings into episodes, scores the detector one-to-one, and
characterises each episode against the concurrent salinity and temperature.

The detection target is rapid deoxygenation. No hypoxic or sub-lethal concentration occurs in this
record, so a concentration threshold would have nothing to fire on; the rate is what the data can
support, and no biological threshold is asserted for it.

Declared in config/bay_reserved_protocol_v2_addendum20.json.

The operating point (rate, consecutive hours, cooldown) is the one selected on development over the
extended grid of addendum 87 and is read from runs/v2_warning_grid_ext.json, its only source
(operating_point); the module attributes RATE, CONSEC and COOLDOWN read it too, on first use.

Usage: python -m src.experiments.v2_warning_episodes
"""
import json

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export
from src.data.bay_profile_context import ROOT

OUT = ROOT / 'reports/bay_twin/runs/v2_warning_episodes.json'
#: where the operating point selected on development is recorded (scripts/v2_warning_grid_ext.py)
POINT_SOURCE = ROOT / 'reports/bay_twin/runs/v2_warning_grid_ext.json'
SENSORS = ['TBaiaP_01m-CTD(calibrato)', 'TMitiliE_05m-CTD(calibrato)',
           'TPortovenereP_02m-CTD(calibrato)', 'TMitiliO_05m-CTD(calibrato)',
           'TPortovenereS_02m-CTD(calibrato)']
EVENT_DROP, EVENT_WIN, MATCH_H = 30.0, 6, 24
GAPS_H = [6, 12, 24]
START = pd.Timestamp('2026-01-01', tz='UTC')
END = pd.Timestamp('2026-06-20', tz='UTC')


def operating_point():
    """(rate in umol/kg/h, consecutive hours, cooldown in h): the point selected on development.

    Read from runs/v2_warning_grid_ext.json on every call; a missing file stops the caller rather
    than falling back to any other point."""
    if not POINT_SOURCE.exists():
        raise FileNotFoundError(f'{POINT_SOURCE.relative_to(ROOT).as_posix()} is missing: the warning operating '
                                'point is read only from there (python -m scripts.v2_warning_grid_ext writes it)')
    sel = json.loads(POINT_SOURCE.read_bytes())['development']['selected']
    return float(sel['rate_umol_per_kg_per_h']), int(sel['consecutive_hours']), int(sel['cooldown_h'])


def __getattr__(name):
    """RATE, CONSEC and COOLDOWN as module attributes, read from the single source when first asked for."""
    if name in ('RATE', 'CONSEC', 'COOLDOWN'):
        return operating_point()[('RATE', 'CONSEC', 'COOLDOWN').index(name)]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def series():
    """Strictly hourly-spaced records.

    The grid is reindexed to a uniform hourly axis before any difference is taken, so that
    `shift(6)` spans six hours rather than six observations. Without it a multi-hour telemetry gap
    makes two readings adjacent and manufactures a decline that never happened; that is the
    difference between the 26 crossings a naive pass finds here and the 12 that exist.
    """
    out = {}
    for name in SENSORS:
        _, frame, _ = read_export(ROOT / 'data/raw' / f'{name}.txt', include_auxiliary=True)
        frame = frame[(frame.index >= START) & (frame.index < END)]
        if 'oxygen' not in frame or frame.oxygen.notna().sum() < 24:
            continue
        valid = frame.oxygen.dropna()
        grid = pd.date_range(valid.index.min(), valid.index.max(), freq='1h', tz='UTC')
        out[name] = frame.reindex(grid)
    return out


def crossings_of(oxygen):
    drop = -(oxygen - oxygen.shift(EVENT_WIN))
    return list(drop[drop > EVENT_DROP].dropna().index)


def alerts_of(oxygen, point=None):
    """Alert hours at `point` (rate, consecutive hours, cooldown), by default the selected one."""
    rate, consec, cooldown = operating_point() if point is None else point
    d = -(oxygen - oxygen.shift(1))
    cond = d > rate
    for k in range(1, consec):
        cond = cond & (d.shift(k) > rate)
    out, last = [], None
    for t, flag in cond.dropna().items():
        if flag and (last is None or (t - last) >= pd.Timedelta(hours=cooldown)):
            out.append(t)
            last = t
    return out


def cluster(times, gap_h):
    """Consecutive crossings closer than gap_h belong to one episode."""
    groups = []
    for t in sorted(times):
        if groups and (t - groups[-1][-1]) <= pd.Timedelta(hours=gap_h):
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def one_to_one(alerts, episodes):
    """Greedy assignment by smallest positive lead; each alert and episode used at most once."""
    pairs = []
    for ei, ep in enumerate(episodes):
        onset = ep[0]
        for ai, a in enumerate(alerts):
            lead = (onset - a).total_seconds() / 3600.0
            if 0 <= lead <= MATCH_H:
                pairs.append((lead, ai, ei))
    pairs.sort()
    used_a, used_e, matched = set(), set(), []
    for lead, ai, ei in pairs:
        if ai in used_a or ei in used_e:
            continue
        used_a.add(ai)
        used_e.add(ei)
        matched.append({'lead_h': round(lead, 1), 'alert': str(alerts[ai]),
                        'episode_onset': str(episodes[ei][0])})
    return matched


def describe(frame, ep):
    """Magnitude, duration, recovery and the concurrent hydrography of one episode."""
    onset, last = ep[0], ep[-1]
    pre = onset - pd.Timedelta(hours=EVENT_WIN)
    window = frame.loc[pre:last]
    if window.oxygen.notna().sum() < 2:
        return None
    present = window.oxygen.dropna()
    start_o2 = float(present.iloc[0])
    min_o2 = float(window.oxygen.min())
    tail = frame.loc[last:last + pd.Timedelta(hours=72)].oxygen
    back = tail[tail >= start_o2 - 10.0]
    rec = float((back.index[0] - last).total_seconds() / 3600.0) if len(back) else None

    def delta(col):
        if col not in window or window[col].notna().sum() < 2:
            return None
        v = window[col].dropna()
        return round(float(v.iloc[-1] - v.iloc[0]), 3)

    return {'sensor': None, 'onset': str(onset), 'crossings': len(ep),
            'duration_h': round((last - onset).total_seconds() / 3600.0, 1),
            'onset_hour_utc': int(onset.hour),
            'drop_umol_l': round(start_o2 - min_o2, 2), 'min_oxygen': round(min_o2, 2),
            'recovery_h': rec, 'salinity_change': delta('salinity'),
            'temperature_change': delta('temperature')}


def main():
    data = series()
    point = operating_point()
    rate, consec, cooldown = point
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum20.json',
              'target': 'rapid deoxygenation; no hypoxic concentration occurs in this record',
              'event_drop_umol_l': EVENT_DROP, 'event_window_h': EVENT_WIN,
              'operating_point': {'rate': rate, 'consecutive': consec, 'cooldown_h': cooldown,
                                  'selected_in': 'runs/v2_warning_grid_ext.json'},
              'match_window_h': MATCH_H, 'by_gap': {}, 'episodes': []}

    all_cross, all_alerts = {}, {}
    for name, frame in data.items():
        oxy = frame.oxygen
        all_cross[name] = crossings_of(oxy)
        all_alerts[name] = alerts_of(oxy, point)

    total_cross = sum(len(v) for v in all_cross.values())
    total_alerts = sum(len(v) for v in all_alerts.values())

    for gap in GAPS_H:
        episodes, matched, n_ep = {}, 0, 0
        for name in data:
            eps = cluster(all_cross[name], gap)
            episodes[name] = eps
            n_ep += len(eps)
            matched += len(one_to_one(all_alerts[name], eps))
        report['by_gap'][f'{gap}h'] = {
            'episodes': n_ep, 'crossings': total_cross, 'alerts': total_alerts,
            'matched_one_to_one': matched,
            'episode_recall': round(matched / n_ep, 3) if n_ep else None,
            'episode_precision': round(matched / total_alerts, 3) if total_alerts else None}
        if gap == 12:
            for name, eps in episodes.items():
                for ep in eps:
                    d = describe(data[name], ep)
                    if d:
                        d['sensor'] = name
                        report['episodes'].append(d)

    # The clock-hour result needs the right null. A uniform clock (7 of 24 hours) ignores that the
    # diurnal cycle already puts large declines at night, so the reference is the hour at which
    # each day's largest six-hour drop ends, on the same sensors and period. An episode's first
    # crossing is the END of its six-hour window, so the drop itself began six hours earlier.
    window = set(range(0, 7))
    ends = []
    for name, frame in data.items():
        drop = -(frame.oxygen - frame.oxygen.shift(EVENT_WIN)).dropna()
        daily = drop.groupby(drop.index.floor('D')).idxmax()
        ends.extend(int(t.hour) for t in daily)
    onsets = [e['onset_hour_utc'] for e in report['episodes']]
    inside = sum(h in window for h in onsets)
    share = float(np.mean([h in window for h in ends]))
    from scipy.stats import beta, binomtest
    report['onset_hours'] = {
        'window_utc': [0, 6], 'episodes_inside': inside, 'episodes': len(onsets),
        'uniform_clock_share': round(len(window) / 24, 3),
        'uniform_clock_p': round(float(binomtest(inside, len(onsets), len(window) / 24,
                                                 alternative='greater').pvalue), 5),
        'daily_largest_drop_share': round(share, 3), 'days': len(ends),
        'daily_largest_drop_p': round(float(binomtest(inside, len(onsets), share,
                                                      alternative='greater').pvalue), 6),
        'note': 'onset is the first crossing hour, the end of the six-hour window'}

    def exact(k, n):
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
        return [round(lo, 3), round(hi, 3)]
    g = report['by_gap']['12h']
    g['episode_recall_exact_95'] = exact(g['matched_one_to_one'], g['episodes'])
    g['episode_precision_exact_95'] = exact(g['matched_one_to_one'], g['alerts'])

    OUT.write_text(json.dumps(report, indent=2))
    print(f"onset hours: {report['onset_hours']}")
    print(f"12 h gap exact 95%: recall {g['episode_recall_exact_95']}  precision "
          f"{g['episode_precision_exact_95']}")
    print(f'crossings {total_cross}   alerts {total_alerts}')
    for gap, v in report['by_gap'].items():
        print(f"  gap {gap:>4s}: episodes {v['episodes']:2d}  matched {v['matched_one_to_one']:2d}"
              f"  recall {v['episode_recall']}  precision {v['episode_precision']}")
    print('\nepisodes at 12 h gap:')
    e = pd.DataFrame(report['episodes'])
    if len(e):
        pd.set_option('display.width', 200)
        print(e[['sensor', 'onset', 'crossings', 'duration_h', 'onset_hour_utc', 'drop_umol_l',
                 'min_oxygen', 'recovery_h', 'salinity_change', 'temperature_change']]
              .assign(sensor=lambda x: x.sensor.str[:12]).to_string(index=False))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
