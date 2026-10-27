"""Could the delivered temperature fallback use the neighbours? A causal check on the masked gaps.

The delivered outage fallback holds the probe's last real reading. The nearest-reporting-probe fill
of the gap cohorts is causal in construction: it adds to that last reading the change the nearest
probe that is reporting has seen since the gap opened, and it never reads the silent probe after
onset. Scored on the masked gaps of Gap-Dev and Gap-2026 at 24, 48 and 72 h, it answers whether
the fallback would be better with the neighbours, on data where the truth is known. Paired
cluster bootstrap over overlap clusters, as every gap contrast.

It also scores the delivered fallback itself on its two weekly outage cohorts under one rule
(`outage`): Outage-2025, the Monday outages of the thermal-memory pilot, and Outage-2026, the same
schedule on 2026 from the first reserved evaluation. The pilot aggregated by quarter, spatial group
and sensor and the 2026 evaluation by quarter and sensor; both are put here on the second, the
forecast rule of Chapter 4 (mean over quarters of the per-sensor mean), with the pilot's own figure
kept beside it. The pilot's selected physics blend is scored under the same rule, with its paired
difference from persistence resampling whole weeks within each quarter, as the pilot did.

Usage: python -m scripts.v2_neighbour_fallback [--outage-only]
  --outage-only   recompute only the `outage` block and keep the rest of the file as it is
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2_gap_climatology import load_predictions
from scripts.v2_gap_intervals import clusters, paired

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
OUT = RUNS / 'v2_neighbour_fallback.json'
COHORTS = {'Gap-Dev': ('20260922T171859Z_v2_gaprecovery', 'v2_gap_recovery'),
           'Gap-2026': ('20260922T180814Z_v2_gaprecovery', 'v2_gap_recovery_reserved2026')}
LENGTHS = (24, 48, 72)
PILOT = RUNS / '20260917T184229Z_thermal_memory_pilot/results/thermal_memory/predictions.csv.gz'
OUTAGE_2026 = RUNS / '20260918_reserved_evaluation/outage_endpoints_2026.csv.gz'
OUTAGE_DRAWS = 500


def nested(frame, error='error'):
    """Chapter 4's forecast rule: the per-sensor mean within each quarter, then the mean over quarters."""
    return float(frame.groupby(['quarter', 'sensor'])[error].mean().groupby('quarter').mean().mean())


def week_paired(frame, label):
    """Blend minus persistence, whole weeks resampled within each quarter (the pilot's scheme)."""
    rng = np.random.default_rng([20260926, zlib.crc32(label.encode())])
    point = nested(frame, 'e_blend') - nested(frame, 'e_persistence')
    weeks = {q: [g for _, g in part.groupby('episode')] for q, part in frame.groupby('quarter')}
    draws = []
    for _ in range(OUTAGE_DRAWS):
        s = pd.concat([blocks[i] for blocks in weeks.values() for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(nested(s, 'e_blend') - nested(s, 'e_persistence'))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'delta': round(point, 4), 'ci': [round(float(lo), 4), round(float(hi), 4)],
            'excludes_zero': bool(lo > 0 or hi < 0), 'draws': OUTAGE_DRAWS,
            'weeks': int(sum(len(b) for b in weeks.values()))}


def outage():
    out = {'rule': 'mean over quarters of the per-sensor mean absolute error (Chapter 4 forecast rule)',
           'sources': {'Outage-2025': str(PILOT.relative_to(RUNS.parent)).replace('\\', '/'),
                       'Outage-2026': str(OUTAGE_2026.relative_to(RUNS.parent)).replace('\\', '/')},
           'cohorts': {'Outage-2025': {}, 'Outage-2026': {}}}
    pilot = pd.read_csv(PILOT).rename(columns={'fold': 'quarter'})
    for h in (24, 72):
        part = pilot[pilot.horizon == h]
        p = part[part.family == 'persistence']
        b = part[part.family == 'selected_physics']
        key = ['quarter', 'sensor', 'episode']
        both = p.merge(b[key + ['prediction']], on=key, suffixes=('', '_blend'), validate='one_to_one')
        both = both[both.target_temperature.notna() & both.prediction.notna() & both.prediction_blend.notna()]
        both = both.assign(e_persistence=(both.target_temperature - both.prediction).abs(),
                           e_blend=(both.target_temperature - both.prediction_blend).abs())
        pilot_rule = lambda f, c: float(f.groupby(['quarter', 'group', 'sensor'])[c].mean()
                                        .groupby(['quarter', 'group']).mean().groupby('quarter').mean().mean())
        out['cohorts']['Outage-2025'][f'{h}h'] = {
            'scheduled': int(len(p)), 'observed_targets': int(p.target_temperature.notna().sum()),
            'valid_anchors': int(p.initial_temperature.notna().sum()), 'scored': int(len(both)),
            'weeks': int(p.episode.nunique()), 'sensors_scored': int(both.sensor.nunique()),
            'macro_mae': round(nested(both, 'e_persistence'), 4),
            'blend_macro_mae': round(nested(both, 'e_blend'), 4),
            'blend_minus_persistence': week_paired(both, f'outage|2025|{h}'),
            'pilot_rule_quarter_group_sensor': {'persistence': round(pilot_rule(both, 'e_persistence'), 4),
                                                'blend': round(pilot_rule(both, 'e_blend'), 4)}}
    reserved = pd.read_csv(OUTAGE_2026)
    for h in (24, 72):
        sub = reserved[reserved.horizon_h == h]
        scored = sub[sub.error.notna()]
        out['cohorts']['Outage-2026'][f'{h}h'] = {
            'scheduled': int(len(sub)), 'observed_targets': int(sub.has_target.sum()),
            'valid_anchors': int(sub.has_anchor.sum()), 'scored': int(len(scored)),
            'weeks': int(sub.outage_start.nunique()), 'sensors_scored': int(scored.sensor.nunique()),
            'macro_mae': round(nested(scored), 4)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outage-only', action='store_true')
    if p.parse_args().outage_only:
        report = json.loads(OUT.read_text())
        report['outage'] = outage()
        OUT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['outage'], indent=1))
        print('wrote', OUT)
        return
    report = {'fill': 'last real reading plus the nearest reporting probe\'s change since onset',
              'causal': True, 'variable': 'temperature', 'cohorts': {}}
    from src.experiments.v2_gap_recovery import SEED, build
    for cohort, (run, results) in COHORTS.items():
        # The addendum-30 runs predate the nearest-reporting-probe column; rebuild the episodes
        # with the published seed and lengths, which reproduces the published rows exactly
        # (v2_gap_neighbours checks the same rebuild row for row).
        published = load_predictions(run, results, 'temperature', 'chronos_bidir')
        lengths = sorted(int(x) for x in published.gap_length.unique())
        period = 'reserved' if cohort.endswith('2026') else 'development'
        frame = build('temperature', np.random.default_rng(SEED), period=period, lengths=lengths)
        keys = frame[['episode', 'position']].merge(published[['episode', 'position']],
                                                     how='outer', indicator=True)
        if (keys._merge != 'both').any():
            raise SystemExit(f'{cohort}: rebuilt episodes differ from the published ones')
        frame = frame[frame.gap_length.isin(LENGTHS)].dropna(subset=['nearest_live_probe'])
        cluster_of = clusters(frame)
        block = {}
        for length, part in frame.groupby('gap_length'):
            block[str(int(length))] = {
                'live_minus_persistence': paired(part, 'nearest_live_probe', 'persistence',
                                                 f'fallback|{cohort}|{length}', cluster_of),
                'mae': {c: round(float((part[c] - part.truth).abs().groupby(part.episode).mean()
                                       .mean()), 4)
                        for c in ('persistence', 'nearest_live_probe', 'linear')}}
        pooled = paired(frame, 'nearest_live_probe', 'persistence', f'fallback|{cohort}|all',
                        cluster_of)
        report['cohorts'][cohort] = {'by_length': block, 'pooled_24_72': pooled}
        print(cohort, 'pooled', pooled)
        for L, b in block.items():
            print(f"  {L} h  {b['live_minus_persistence']}  mae {b['mae']}")
    report['outage'] = outage()
    OUT.write_text(json.dumps(report, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
