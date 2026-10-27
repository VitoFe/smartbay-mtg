"""Addendum-7 study 3: how much of a real outage record does an 18 h bridge cover?

Addendum 4 put the operational coherence limit at about eighteen hours, beyond which holding the
last real value beats feeding a model its own output. That is a lead time, not a coverage figure.
This measures the distribution of contiguous missing-hour runs in the delivered record, so the
threshold can be stated as a fraction of the interruptions an operator actually sees.

Descriptive only: no model is fitted and nothing is scored.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bay_profile_context import ROOT, load_observations

LIMIT = 18
BUCKETS = [1, 3, 6, 12, 18, 24, 48, 72, 168]
OUT = ROOT / 'reports/bay_twin/runs/v2_gap_durations.json'


def main():
    meta, frames = load_observations()
    records = []
    for name, frame in frames.items():
        for variable in ['temperature', 'oxygen']:
            if variable not in frame:
                continue
            grid = pd.date_range(frame.index.min(), frame.index.max(), freq='h')
            present = frame[variable].reindex(grid).notna().to_numpy()
            if present.sum() < 100:
                continue
            index = np.flatnonzero(present)
            runs = np.diff(index) - 1
            for length in runs[runs > 0]:
                records.append({'sensor': name, 'variable': variable, 'hours': int(length)})

    gaps = pd.DataFrame(records)
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum7.json',
              'coherence_limit_hours': LIMIT, 'by_variable': {}}
    for variable, group in gaps.groupby('variable'):
        hours = group.hours.to_numpy()
        covered = hours <= LIMIT
        report['by_variable'][variable] = {
            'sensors': int(group.sensor.nunique()), 'gaps': int(len(hours)),
            'lost_hours': int(hours.sum()),
            'median_gap_hours': float(np.median(hours)),
            'p90_gap_hours': float(np.percentile(hours, 90)),
            'longest_gap_hours': int(hours.max()),
            'share_of_gaps_within_limit': float(covered.mean()),
            'share_of_lost_hours_within_limit': float(hours[covered].sum() / hours.sum()),
            'cumulative_share_of_gaps': {str(b): float((hours <= b).mean()) for b in BUCKETS},
            'cumulative_share_of_lost_hours': {str(b): float(hours[hours <= b].sum() / hours.sum())
                                               for b in BUCKETS}}
    OUT.write_text(json.dumps(report, indent=2))

    for variable, entry in report['by_variable'].items():
        print(f"{variable:12s} {entry['gaps']:6d} gaps over {entry['sensors']:2d} sensors, "
              f"{entry['lost_hours']:7d} lost hours, median {entry['median_gap_hours']:.0f} h, "
              f"longest {entry['longest_gap_hours']} h")
        print(f"{'':12s} within {LIMIT} h: {entry['share_of_gaps_within_limit']:.1%} of gaps "
              f"but only {entry['share_of_lost_hours_within_limit']:.1%} of lost hours")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
