"""How much of the boundary product's temperature gain each cast-derived prior recovers, by half-year.

Chapter 5 says the annual-harmonic cast prior recovers the Copernicus offset's gain in January to June
and none of it in July to December. Each Field-57 outer fold is one half-year (fold 0 scores January
to June 2025, fold 1 July to December), so the statement is the per-fold gain over the prior-free arm,
A2 - A1 (Copernicus) and A2 - A7 (the cast prior), under the triple-nested rule within a fold (casts,
then stations). Read from the stored held-station predictions: the published one-penalty design, the
per-fold selection (addendum 47) and the per-fold selection with the cast prior fitted inside each
inner station split (addendum 69).
Usage: python -m scripts.v2_prior_halfyear
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'
PERFOLD = RUNS / 'v2_perfold'
OUT = RUNS / 'v2_prior_halfyear.json'
HALF = {0: 'January to June 2025', 1: 'July to December 2025'}


def by_fold(path):
    d = pd.read_csv(path)
    d['e'] = (d.temp_c - d.pred_temp_c).abs()
    return (d.groupby(['fold', 'station', 'cast_id']).e.mean()
            .groupby(['fold', 'station']).mean().groupby('fold').mean())


def main():
    designs = {'published': ('field/published', 'field/published'),
               'perfold': ('field/perfold', 'field/perfold'),
               'perfold_inner_prior': ('field/perfold', 'field_innerprior/perfold')}
    out = {'question': 'gain over the prior-free arm by half-year (outer fold), temperature, degC',
           'folds': {str(k): v for k, v in HALF.items()}, 'designs': {}}
    for name, (base, a7dir) in designs.items():
        a2 = by_fold(PERFOLD / base / 'A2_none_direct/selected_predictions.csv.gz')
        a1 = by_fold(PERFOLD / base / 'A1_both_direct/selected_predictions.csv.gz')
        a7 = by_fold(PERFOLD / a7dir / 'A7_ctdprior_direct/selected_predictions.csv.gz')
        out['designs'][name] = {
            'copernicus_gain_A2_minus_A1': {str(f): round(float(v), 6) for f, v in (a2 - a1).items()},
            'cast_prior_gain_A2_minus_A7': {str(f): round(float(v), 6) for f, v in (a2 - a7).items()},
            'sources': [str((PERFOLD / p).relative_to(ROOT)).replace('\\', '/') for p in
                        (f'{base}/A2_none_direct', f'{base}/A1_both_direct', f'{a7dir}/A7_ctdprior_direct')]}
    OUT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out['designs'], indent=1))


if __name__ == '__main__':
    main()
