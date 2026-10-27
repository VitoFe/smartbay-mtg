"""Reserved-2026 availability over the scheduled roster and over the probes actually deployed.

The frozen Reserved-2026 grid schedules 680 six-hour endpoints (issuances 2026-01-01 00 UTC to
2026-06-19 18 UTC) at every sensor the released forecaster supports: 15 temperature probes and 4
oxygen sondes, the sensors with at least 100 valid training endpoints before the release's training
cutoff (src.experiments.v2_release.refit_forecast). Its scorable share is therefore a property of
that roster as well as of the instruments. This script reads the raw exports once and separates
the two:

  * when every probe last reported, and whether it reported at all inside the reserved window;
  * the scorable rate over the whole scheduled grid, and over the scheduled probes that were
    deployed (at least one reading in the window);
  * the probes that reported in 2026 but are not on the grid, with the share of the 680 issuances
    at which the raw record holds a reading at the issuance hour and at the target hour, which is
    the grid's own scorability rule (bay_reserved_evaluation / v2_reserved: the anchor is the
    reading AT the issuance hour, the target the reading at issuance + 6 h). That share describes
    the record; no model is scored on those probes and the frozen grid is not changed.

The raw rule is first checked against the frozen grid's own has_anchor / has_target flags on every
scheduled probe, so the figures for the unscheduled probes come from a rule that reproduces the
grid where the grid exists.

Usage: python -m scripts.v2_availability_deployed
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bay_observations import read_export

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bay_twin/runs/v2_availability_deployed.json'
GRID = ROOT / 'reports/bay_twin/runs/v2_reserved_reading_v2/forecast_endpoints_2026.csv.gz'
REFIT = ROOT / 'reports/bay_twin/runs/v2_release_v2/refit_forecast'
GEOMETRY = ROOT / 'reports/bay_twin/sensor_geometry.csv'
ISSUANCES = pd.date_range('2026-01-01T00:00:00', '2026-06-19T18:00:00', freq='6h', tz='UTC')
WINDOW = (pd.Timestamp('2026-01-01T00:00:00', tz='UTC'), pd.Timestamp('2026-06-20T00:00:00', tz='UTC'))
HORIZON = pd.Timedelta(hours=6)
ELIGIBLE_MIN = 100  # v2_release.refit_forecast: valid training endpoints needed to be supported


def pct(num, den):
    return None if den == 0 else round(100 * num / den, 1)


def release_counts(variable):
    """Valid training endpoints per probe under the release's own rule, for every probe it built."""
    from src.experiments.bay_graph_dynamics import build
    data = build(variable)
    t = data['times']
    cal_start = pd.Timestamp('2025-11-14T00:00:00', tz='UTC')
    train_end = cal_start - pd.Timedelta(hours=78)
    tr = np.flatnonzero(t + pd.Timedelta(hours=6) <= train_end)
    return {n: int(data['valid'][tr, s].sum()) for s, n in enumerate(data['names'])}


def main():
    meta = pd.read_csv(GEOMETRY).set_index('sensor')
    grid = pd.read_csv(GRID)
    grid['time'] = pd.to_datetime(grid.time, utc=True)
    frames = {}
    for name in meta.index:
        path = ROOT / 'data/raw' / (name + '.txt')
        if path.exists():
            _, frame, _ = read_export(path, include_auxiliary=True)
            frames[name] = frame

    report = {'window': {'first_issuance': str(ISSUANCES[0]), 'last_issuance': str(ISSUANCES[-1]),
                         'issuances': len(ISSUANCES), 'horizon_hours': 6},
              'rule': 'scorable = a reading at the issuance hour and at issuance + 6 h (hourly '
                      'means labelled at the interval end, as read_export returns them)',
              'grid': str(GRID.relative_to(ROOT / 'reports/bay_twin')),
              'variables': {}}
    for variable in ['temperature', 'oxygen']:
        cp = json.loads((REFIT / f'refit_{variable}_model.json').read_text())
        supported = cp['contract']['supported_sensors']
        counts = release_counts(variable)
        g = grid[grid.variable == variable]
        roster = sorted(g.sensor.unique())
        assert roster == sorted(supported), 'frozen grid roster differs from the release contract'
        probes = {}
        check = {'probes': 0, 'anchor_mismatches': 0, 'target_mismatches': 0}
        for name, frame in frames.items():
            if variable not in frame or not frame[variable].notna().any():
                continue
            s = frame[variable].dropna()
            inside = s[(s.index >= WINDOW[0]) & (s.index < WINDOW[1])]
            anchor = frame[variable].reindex(ISSUANCES).notna().to_numpy()
            target = frame[variable].reindex(ISSUANCES + HORIZON).notna().to_numpy()
            pair = int((anchor & target).sum())
            row = {'site': meta.loc[name, 'site'],
                   'first_reading': str(s.index.min()), 'last_reading': str(s.index.max()),
                   'readings_in_window': int(len(inside)),
                   'first_in_window': str(inside.index.min()) if len(inside) else None,
                   'last_in_window': str(inside.index.max()) if len(inside) else None,
                   'deployed_in_window': bool(len(inside) > 0),
                   'on_grid': name in roster,
                   'release_training_endpoints': counts.get(name),
                   'raw_scorable_issuances': pair,
                   'raw_scorable_rate_pct': pct(pair, len(ISSUANCES))}
            if name in roster:
                gs = g[g.sensor == name].set_index('time').reindex(ISSUANCES)
                row['grid_scorable'] = int((gs.has_anchor.astype(bool) & gs.has_target.astype(bool)).sum())
                row['grid_scorable_rate_pct'] = pct(row['grid_scorable'], len(ISSUANCES))
                check['probes'] += 1
                check['anchor_mismatches'] += int((gs.has_anchor.astype(bool).to_numpy() != anchor).sum())
                check['target_mismatches'] += int((gs.has_target.astype(bool).to_numpy() != target).sum())
            probes[name] = row

        on = [n for n in roster]
        deployed_on = [n for n in on if probes[n]['deployed_in_window']]
        silent_on = [n for n in on if not probes[n]['deployed_in_window']]
        off = sorted(n for n, r in probes.items() if not r['on_grid'] and r['deployed_in_window'])
        sched = len(on) * len(ISSUANCES)
        scor = sum(probes[n]['grid_scorable'] for n in on)
        scor_dep = sum(probes[n]['grid_scorable'] for n in deployed_on)
        raw_all = sum(probes[n]['raw_scorable_issuances'] for n in deployed_on + off)
        report['variables'][variable] = {
            'raw_rule_reproduces_grid': check,
            'scheduled_grid': {'probes': len(on), 'endpoints': sched, 'scorable': scor,
                               'scorable_rate_pct': pct(scor, sched)},
            'scheduled_and_deployed': {'probes': len(deployed_on),
                                       'endpoints': len(deployed_on) * len(ISSUANCES),
                                       'scorable': scor_dep,
                                       'scorable_rate_pct': pct(scor_dep, len(deployed_on) * len(ISSUANCES))},
            'scheduled_but_silent_all_window': {
                n: {'last_reading': probes[n]['last_reading'], 'site': probes[n]['site']} for n in silent_on},
            'deployed_not_scheduled': {
                n: {'site': probes[n]['site'], 'first_reading': probes[n]['first_reading'],
                    'last_in_window': probes[n]['last_in_window'],
                    'readings_in_window': probes[n]['readings_in_window'],
                    'release_training_endpoints': probes[n]['release_training_endpoints'],
                    'raw_scorable_issuances': probes[n]['raw_scorable_issuances'],
                    'raw_scorable_rate_pct': probes[n]['raw_scorable_rate_pct']} for n in off},
            'all_deployed_raw_record': {
                'probes': len(deployed_on) + len(off), 'issuances': (len(deployed_on) + len(off)) * len(ISSUANCES),
                'raw_scorable': raw_all,
                'raw_scorable_rate_pct': pct(raw_all, (len(deployed_on) + len(off)) * len(ISSUANCES)),
                'note': 'descriptive: the scheduled probes by the frozen grid rule plus the unscheduled '
                        'ones by the same rule on the raw record; nothing is scored on the latter'},
            'probes': probes}

    OUT.write_text(json.dumps(report, indent=2) + '\n')
    for variable, v in report['variables'].items():
        print(f"== {variable}: raw rule vs grid {v['raw_rule_reproduces_grid']}")
        print(f"   scheduled grid {v['scheduled_grid']}")
        print(f"   scheduled and deployed {v['scheduled_and_deployed']}")
        print('   scheduled, silent all window: ' + ', '.join(
            f"{n} (last {r['last_reading'][:10]})" for n, r in v['scheduled_but_silent_all_window'].items()))
        print('   deployed, not scheduled: ' + ', '.join(
            f"{n} ({r['raw_scorable_rate_pct']} %, {r['release_training_endpoints']} training endpoints)"
            for n, r in v['deployed_not_scheduled'].items()))
        print(f"   all deployed, raw record: {v['all_deployed_raw_record']['raw_scorable_rate_pct']} %")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
