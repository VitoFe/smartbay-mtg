"""Addendum-25: validate the biofouling detector against the documented cleaning log.

Addendum 24 concluded the detector does not discriminate a manually cleaned generation from a
wiped one, and recorded that the servicing dates were absent. They are not: data/logbook.txt holds
159 dated entries from 2021-08 to 2023-08, 104 of them cleanings, covering exactly the
first-generation period. The Gen. 1 exports name their node in the header, so node1 is
TBaia_01m and node5 is Mitili Est 5 m.

Three questions, in order of what they would cost if answered no:

  A  Is a fouling signature present at all? A superposed-epoch average of the deficit around each
     cleaning; if biofilm accumulates and cleaning removes it, the deficit rises before and drops.
  B  Does the published detector find the cleanings? Precision and recall against them.
  C  Does a reworked detector? The slack is set from the record so the statistic has no drift under
     normal conditions, chosen on one node and evaluated on the other.

The comparison that decides it is a periodic alerter matched on rate. Cleanings at node1 are close
to weekly, so an alarm every seven days would match most of them while knowing nothing; a detector
that cannot beat that has no skill.

Declared in config/bay_reserved_protocol_v2_addendum25.json.
Usage: python -m src.experiments.v2_logbook
"""
import json
import re

import numpy as np
import pandas as pd

from src.data.bay_profile_context import ROOT
from src.experiments.v2_biofouling import H_DECISION, K_SLACK, deficit, read_export

OUT = ROOT / 'reports/bay_twin/runs/v2_logbook.json'
LOGBOOK = ROOT / 'data/logbook.txt'
MATCH_WINDOW_H = 72          # an alert is matched if a cleaning follows within three days
EPOCH_DAYS = 10              # superposed-epoch half-width

#: logbook node -> Gen. 1 export, from the Sonda_S/N field in each file's header.
NODES = {'node1': "TBaia_01m-Ossigeno(calibrato)",
         'node5': "TMitiliE_05m-Ossigeno(calibrato)"}


def read_logbook():
    """Parse `DD/MM/YYYY HH:MM UTC <action> <nodes...>` into per-node event lists."""
    events = []
    for line in LOGBOOK.read_text(encoding='latin-1').splitlines():
        m = re.match(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\s+UTC\s+(\w+)\s*(.*)', line.strip())
        if not m:
            continue
        stamp = pd.to_datetime(f'{m.group(1)} {m.group(2)}', dayfirst=True, utc=True)
        for node in m.group(4).split():
            events.append({'time': stamp, 'action': m.group(3), 'node': node})
    return pd.DataFrame(events)


def deployed_spans(log, node):
    """Intervals in which the probe was deployed, so a redeployment gap is not scored."""
    rows = log[log.node == node].sort_values('time')
    spans, start = [], None
    for _, r in rows.iterrows():
        if r.action == 'Start' and start is None:
            start = r.time
        elif r.action == 'Stop' and start is not None:
            spans.append((start, r.time))
            start = None
    if start is not None:
        spans.append((start, rows.time.max()))
    return spans


def superposed_epoch(series, cleanings):
    """Mean deficit as a function of hours relative to a cleaning."""
    grid = np.arange(-EPOCH_DAYS * 24, EPOCH_DAYS * 24 + 1, 6)
    stack = []
    for t in cleanings:
        window = series.reindex(t + pd.to_timedelta(grid, unit='h'), method='nearest',
                                tolerance=pd.Timedelta('3h'))
        if window.notna().sum() > len(grid) * 0.5:
            stack.append(window.to_numpy())
    if not stack:
        return None
    arr = np.vstack(stack)
    before = np.nanmean(arr[:, (grid >= -72) & (grid < 0)])
    after = np.nanmean(arr[:, (grid > 0) & (grid <= 72)])
    return {'episodes_used': len(stack),
            'offsets_hours': grid.tolist(),
            'mean_deficit': np.round(np.nanmean(arr, axis=0), 3).tolist(),
            'mean_72h_before': round(float(before), 3),
            'mean_72h_after': round(float(after), 3),
            'drop_at_cleaning': round(float(before - after), 3)}


def placebo_epoch(series, n, seed=20260922, draws=400):
    """The same before-minus-after statistic at random times.

    A monotone seasonal trend alone makes the window after any timestamp warmer, and therefore more
    undersaturated, than the window before it. Without this control a trend would be read as a
    reversed fouling signal. The cleaning statistic means something only in so far as it differs
    from this distribution.
    """
    rng = np.random.default_rng(seed)
    lo, hi = series.index[0] + pd.Timedelta(days=EPOCH_DAYS), series.index[-1] - pd.Timedelta(days=EPOCH_DAYS)
    pool = series.loc[lo:hi].index
    if len(pool) < n:
        return None
    out = []
    for _ in range(draws):
        picks = pd.DatetimeIndex(rng.choice(pool, n, replace=False))
        got = superposed_epoch(series, list(picks))
        if got:
            out.append(got['drop_at_cleaning'])
    if not out:
        return None
    out = np.asarray(out)
    return {'draws': len(out), 'mean': round(float(out.mean()), 3),
            'p2_5': round(float(np.percentile(out, 2.5)), 3),
            'p97_5': round(float(np.percentile(out, 97.5)), 3)}


def alerts_from(series, slack, threshold):
    """One-sided CUSUM with reset, returning alert timestamps."""
    s, out = 0.0, []
    values = series.to_numpy()
    for i, r in enumerate(values):
        if np.isfinite(r):
            s = max(0.0, s + (r - slack))
            if s >= threshold:
                out.append(series.index[i])
                s = 0.0
    return pd.DatetimeIndex(out)


def score(alerts, cleanings, span_hours):
    """An alert is matched if a cleaning follows it within the window; a cleaning is recalled if
    any alert precedes it within the window. Both are one-to-one."""
    if len(alerts) == 0:
        return {'alerts': 0, 'precision': None, 'recall': 0.0,
                'alerts_per_year': 0.0, 'matched': 0}
    window = pd.Timedelta(hours=MATCH_WINDOW_H)
    unused = list(cleanings)
    matched = 0
    for a in alerts:
        hit = next((c for c in unused if a <= c <= a + window), None)
        if hit is not None:
            unused.remove(hit)
            matched += 1
    return {'alerts': len(alerts), 'matched': matched,
            'precision': round(matched / len(alerts), 4),
            'recall': round(matched / len(cleanings), 4) if len(cleanings) else None,
            'alerts_per_year': round(len(alerts) / (span_hours / (24 * 365.25)), 1)}


def periodic_baseline(series, n_alerts, cleanings, span_hours):
    """A rate-matched alarm on a fixed period, phase chosen to favour it. The detector must beat
    this to have shown anything beyond reproducing the maintenance calendar."""
    if n_alerts < 1:
        return None
    step = len(series) / n_alerts
    best = None
    for phase in range(0, int(step), max(1, int(step) // 24)):
        idx = series.index[np.clip((phase + np.arange(n_alerts) * step).astype(int),
                                   0, len(series) - 1)]
        got = score(pd.DatetimeIndex(idx), cleanings, span_hours)
        if best is None or (got['precision'] or 0) > (best['precision'] or 0):
            best = got
    return best


def main():
    log = read_logbook()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum25.json',
              'logbook': {'entries': int(len(log)), 'file': 'data/logbook.txt',
                          'actions': {k: int(v) for k, v in log.action.value_counts().items()}},
              'match_window_hours': MATCH_WINDOW_H, 'nodes': {}}

    prepared = {}
    for node, stem in NODES.items():
        frame = read_export(stem)
        d = deficit(frame).dropna()
        cleanings = [t for t in log[(log.node == node) & (log.action == 'Pulizia')].time
                     if d.index[0] <= t <= d.index[-1]]
        prepared[node] = (d, cleanings)
        spans = deployed_spans(log, node)
        report['nodes'][node] = {
            'probe': stem, 'series_hours': int(d.notna().sum()),
            'first': str(d.index[0].date()), 'last': str(d.index[-1].date()),
            'cleanings_in_series': len(cleanings),
            'median_days_between_cleanings': round(float(
                pd.Series(cleanings).diff().dt.total_seconds().median() / 86400), 2)
            if len(cleanings) > 1 else None,
            'deployed_spans': len(spans),
            'mean_deficit': round(float(d.mean()), 3),
            'A_superposed_epoch': superposed_epoch(d, cleanings),
            'A_placebo_epoch': placebo_epoch(d, len(cleanings)),
        }

    # B: the published parameterisation, and C: a slack that removes the drift, held out.
    for node, (d, cleanings) in prepared.items():
        span_h = float(d.notna().sum())
        entry = report['nodes'][node]
        published = alerts_from(d, K_SLACK, H_DECISION)
        entry['B_published'] = score(published, cleanings, span_h)
        entry['B_published']['slack'] = K_SLACK

        other = next(k for k in prepared if k != node)
        held_out_slack = round(float(prepared[other][0].median()), 3)
        reworked = alerts_from(d, held_out_slack, H_DECISION)
        entry['C_reworked'] = score(reworked, cleanings, span_h)
        entry['C_reworked'].update({'slack': held_out_slack, 'slack_chosen_on': other})
        entry['C_periodic_baseline'] = periodic_baseline(d, len(reworked), cleanings, span_h)

    OUT.write_text(json.dumps(report, indent=2))
    for node, e in report['nodes'].items():
        print(f"== {node}  {e['probe'][:34]}  {e['first']}..{e['last']}")
        print(f"   {e['cleanings_in_series']} cleanings, median {e['median_days_between_cleanings']} d apart"
              f"   mean deficit {e['mean_deficit']}")
        a = e['A_superposed_epoch']
        if a:
            print(f"   A  deficit 72 h before {a['mean_72h_before']}  after {a['mean_72h_after']}"
                  f"   drop {a['drop_at_cleaning']:+.3f}  (n={a['episodes_used']})")
            pl = e.get('A_placebo_epoch')
            if pl:
                inside = pl['p2_5'] <= a['drop_at_cleaning'] <= pl['p97_5']
                print(f"      placebo at random times: {pl['mean']:+.3f} "
                      f"[{pl['p2_5']:+.3f}, {pl['p97_5']:+.3f}]  "
                      f"-> cleaning drop is {'INSIDE (no effect)' if inside else 'OUTSIDE'}")
        for key in ('B_published', 'C_reworked', 'C_periodic_baseline'):
            s = e.get(key)
            if s:
                extra = f" slack={s['slack']}" if 'slack' in s else ''
                print(f"   {key:22s} alerts {s['alerts']:5d} ({s['alerts_per_year']:7.1f}/yr)"
                      f"  precision {s['precision']}  recall {s['recall']}{extra}")
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
