"""Addendum 77, the reading: weight interpolation between zero-shot and adapted Chronos-Bolt.

The compute host wrote, per addendum-42 selection, the held-out quarter's validation loss and the 2026
column at a in {0, 0.25, 0.5, 0.75, 1} (scripts/experiments/v2_chronos_wiseft.py). The selected a is the
one with the lowest validation loss. Where it is 1, the selected model is the addendum-42 adaptation itself:
this checks that its column equals the published one (v2_chronos_timeval_predictions.csv.gz) on every
endpoint, so its contrasts are those of runs/v2_chronos_timeval/contrasts.json, and records every a's
validation loss and 2026 error as description (oxygen in mg/l).

Usage: python -m scripts.v2_chronos_wiseft_contrasts
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.units import factor

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
RUN = RUNS / 'v2_chronos_wiseft'
OUT = RUN / 'summary.json'
DECLARATION = 'config/bay_reserved_protocol_v2_addendum77.json'


def main():
    report = json.loads((RUN / 'v2_chronos_wiseft_report.json').read_text())
    new = pd.read_csv(RUN / 'v2_chronos_wiseft_predictions.csv.gz')
    published = pd.read_csv(RUNS / 'v2_chronos_timeval/v2_chronos_timeval_predictions.csv.gz')

    out = {'declared_in': DECLARATION, 'selections': {}}
    for job, rec in report.items():
        if not isinstance(rec, dict) or 'by_alpha' not in rec:
            continue
        method, variable, space = rec['method'], rec['variable'], rec['space']
        k = factor() if variable == 'oxygen' else 1.0
        entry = {'gates': {'GE1': rec['GE1']['passed'], 'GE2': rec['GE2']['passed']},
                 'chosen_alpha': rec['chosen_alpha'],
                 'zero_shot_mae_2026': round(k * rec['mae_zero_shot_2026'], 6),
                 'by_alpha': {a: {'val_loss': round(v['val_loss'], 6), 'mae_2026': round(k * v['mae_2026'], 6),
                                  'minus_zero_shot': round(k * (v['mae_2026'] - rec['mae_zero_shot_2026']), 6)}
                              for a, v in rec['by_alpha'].items()},
                 'unit': 'mg/l' if variable == 'oxygen' else 'degC'}
        if rec['chosen_alpha'] == 1.0:
            mine = new[f'WISE_{method}_{variable}_{space}'].to_numpy(float)
            theirs = published[f'V2TV_{method}_{variable}_{space}'].to_numpy(float)
            same = bool((np.isfinite(mine) == np.isfinite(theirs)).all()
                        and np.nanmax(np.abs(mine - theirs)) <= 1e-9)
            entry['selected_equals_published_adaptation'] = same
            # the selected model is the published adaptation endpoint for endpoint, so its contrasts are
            # the published ones, column V2TV_<method>_<variable>_<space>
            entry['contrasts_in'] = ('runs/v2_chronos_timeval/contrasts.json, column '
                                     f'V2TV_{method}_{variable}_{space}') if same else None
        out['selections'][job] = entry
    out['predictions'] = {
        'PE1': all(not (s['chosen_alpha'] > 0 and s['by_alpha'][str(s['chosen_alpha'])]['minus_zero_shot'] < 0)
                   or s['chosen_alpha'] == 0 for j, s in out['selections'].items() if 'oxygen' in j),
        'PE2': all(abs(s['by_alpha'][str(s['chosen_alpha'])]['minus_zero_shot']) <= 0.005
                   for j, s in out['selections'].items() if 'temperature' in j)}
    OUT.write_bytes(json.dumps(out, indent=1).encode('utf-8'))
    for job, s in out['selections'].items():
        print(job, 'chosen', s['chosen_alpha'], 'equals published', s.get('selected_equals_published_adaptation'),
              {a: v['minus_zero_shot'] for a, v in s['by_alpha'].items()})
    print(out['predictions'])


if __name__ == '__main__':
    main()
