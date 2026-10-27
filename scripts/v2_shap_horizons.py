"""Addendum-3 study 4: why does the weather-feature contrast reverse with horizon?

Exact TreeSHAP attribution for the selected weather tree at each horizon, on the outer development
folds. LightGBM computes TreeSHAP natively through `pred_contrib=True`, so this adds no dependency
and uses the same algorithm the `shap` package would call.

The model is refitted from the penalty recorded in each study's report.json. LightGBM is configured
deterministically in `make_tree`, so the refit reproduces the scored model rather than approximating
it; the script checks this by comparing its predictions with the stored ones.

Features are grouped into six blocks. The tree predicts the *increment* from the issuance value, so
"own history" covers the anchor and its recent trend, not the target.

Usage: python scripts/v2_shap_horizons.py <run-id> [<run-id> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.v2_horizon import build, folds, inner_fold, climatology, tree_fit_score
from src.experiments.bay_dynamics_tree import features
from src.data.bay_forcing import weather

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports/bay_twin/runs'

# design column -> block. Columns 0-1 are the anchor and neighbour mean, 2-15 the forcing stack
# assembled in v2_horizon.build, 16-20 the five meteorological channels from bay_forcing.weather.
BLOCKS = {
    'own history': [0, 2, 3, 4],
    'neighbour context': [1],
    'geometry': [5, 6, 7],
    'calendar': [8, 9, 10, 11],
    'co-located T/S': [12, 13, 14, 15],
    'meteorology': [16, 17, 18, 19, 20],
}
NAMES = ['anchor', 'neighbour', 'trend', 'trend_valid', 'available', 'x_m', 'y_m', 'depth',
         'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour', 'temperature', 'temperature_valid',
         'salinity', 'salinity_valid', 'air', 'solar', 'wind', 'humidity', 'pressure']


def study(variable, horizon, runs):
    for r in runs:
        folder = RUNS / r / 'results' / f'v2_horizon_{variable}_weather_tree_h{horizon}'
        if (folder / 'report.json').exists():
            return folder
    return None


def main():
    runs = sys.argv[1:]
    out = {}
    for variable in ['temperature', 'oxygen']:
        out[variable] = {}
        for horizon in [1, 3, 6, 12, 24, 36, 48]:
            folder = study(variable, horizon, runs)
            if folder is None:
                continue
            report = json.loads((folder / 'report.json').read_text())
            params = {**report['selection']['params'], 'subsample_freq': 1}
            data = build(variable, horizon)
            outer = folds(data, horizon)
            met = weather(data['times']).to_numpy()
            clim = {id(f): climatology(data, f) for f in outer}

            share = np.zeros(len(NAMES))
            rows = 0
            stored = pd.read_csv(folder / 'selected_predictions.csv.gz')
            reproduced = []
            for f in outer:
                base, _ = features(data, {'fallback': f['fallback']}, 'forecast')
                design = np.concatenate(
                    [base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))],
                    axis=-1)
                frame, model = tree_fit_score(data, design, f, params, clim[id(f)])
                reproduced.append(frame)
                mask = data['valid'][f['va']] & f['eligible'][None, :]
                x = design[f['va']][mask]
                contrib = model.predict(x, pred_contrib=True)[:, :len(NAMES)]
                share += np.abs(contrib).sum(0)
                rows += len(x)

            check = pd.concat(reproduced, ignore_index=True)
            agree = bool(len(check) == len(stored) and
                         np.allclose(check.prediction.to_numpy(), stored.prediction.to_numpy()))
            total = share.sum()
            blocks = {name: float(share[cols].sum() / total) for name, cols in BLOCKS.items()}
            top = sorted(zip(NAMES, share / total), key=lambda kv: -kv[1])[:4]
            out[variable][horizon] = {
                'endpoints': rows, 'refit_matches_scored_model': agree,
                'blocks': blocks, 'top_features': [[n, float(v)] for n, v in top],
                'mae': report['mean_forecast_mae']}
            print(f'{variable:12s} h{horizon:<3d} n={rows:5d} refit_ok={agree}  ' +
                  '  '.join(f'{k} {v:5.1%}' for k, v in blocks.items()))
    path = RUNS / 'v2_shap_horizons.json'
    path.write_text(json.dumps(out, indent=2))
    print('\nwrote', path)


if __name__ == '__main__':
    main()
