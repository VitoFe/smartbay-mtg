"""Addendum-6: does Magra river level improve known-sensor forecasting?

A controlled ablation in the Table 5.8 setting, so it slots into the thesis taxonomy as another
test of physical *features* rather than as a new capability. Two new arms, each with its own
>=100-trial nested study so neither inherits a penalty tuned in its absence:

    river_tree           plain residual tree plus the three causal river features
    weather_river_tree   released weather tree plus the same three

The decisive contrast is weather_river_tree against weather_tree, which asks whether river level
adds anything beyond the released feature set. The released weather channels carry no precipitation,
so the river is an integrated catchment rainfall signal the estimator otherwise has no access to.

Development folds only; the hourly index ends 2025-12-31, so no 2026 value is loadable.
Declared in config/bay_reserved_protocol_v2_addendum6.json.
"""
import argparse, hashlib, json, os, traceback

import numpy as np
import pandas as pd

from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_twin_sweep import save
from src.experiments.v2_horizon import (ROOT, build, folds, inner_fold, climatology, tree_fit_score,
                                        ratio, HORIZONS)
from src.experiments.v2_forecast import tree_space
from src.data.bay_forcing import weather
from src.data.bay_river import river, COLUMNS
from src.tuning.protocol_v2 import run_study, PROTOCOL

ARMS = {'river_tree': False, 'weather_river_tree': True}


def design_for(data, fold, with_weather):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    blocks = [base]
    n = len(data['names'])
    if with_weather:
        met = weather(data['times']).to_numpy()
        blocks.append(np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1])))
    flow = river(data['times'])[COLUMNS].to_numpy()
    blocks.append(np.broadcast_to(flow[:, None], (len(data['times']), n, flow.shape[1])))
    return np.concatenate(blocks, axis=-1)


def run(variable, arm, horizon, quick=False):
    output = ROOT / 'results'
    tag = f'v2_river_{variable}_{arm}_h{horizon}'
    folder = output / tag; folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_{tag}.json'
    save(status, {'state': 'RUNNING', 'variable': variable, 'arm': arm, 'horizon': horizon, 'pid': os.getpid()})
    try:
        if horizon not in HORIZONS: raise ValueError(f'undeclared horizon {horizon}')
        data = build(variable, horizon)
        outer = folds(data, horizon)
        if quick:
            for f in outer: f['tr'] = f['tr'][-300:]; f['va'] = f['va'][:40]
        inner = [inner_fold(data, f, horizon) for f in outer]
        if quick:
            for f in inner: f['tr'] = f['tr'][-300:]; f['va'] = f['va'][:40]
        clim = {id(f): climatology(data, f) for f in outer + inner}
        cache = {id(f): design_for(data, f, ARMS[arm]) for f in outer + inner}

        contract = {'protocol': PROTOCOL['version'], 'declared_in': 'config/bay_reserved_protocol_v2_addendum6.json',
                    'variable': variable, 'arm': arm, 'horizon_hours': horizon, 'smoke_only': quick,
                    'river_features': COLUMNS, 'with_weather': ARMS[arm],
                    'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
                    'selection': PROTOCOL['selection']['comparisons'],
                    'objective': 'mean over folds of fold-macro forecast MAE / persistence MAE',
                    'source': 'ARPAL AMEGLIA FOCE MAGRA hourly stage height (m), not volumetric discharge',
                    'no_2026': True}
        save(folder / 'contract.json', contract)

        def objective(trial):
            p = tree_space(trial)
            return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))

        sel = run_study(f'{variable}_{arm}_h{horizon}', objective, folder, quick)
        params = {**sel['params'], 'subsample_freq': 1}
        frames = [tree_fit_score(data, cache[id(f)], f, params, clim[id(f)])[0] for f in outer]
        d = pd.concat(frames, ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'; d.to_csv(path, index=False)

        from src.experiments.bay_graph_dynamics import metrics
        m = metrics(d)
        save(folder / 'report.json', {'selection': sel, 'horizon_hours': horizon, 'endpoints': len(d),
                                      'per_seed_metrics': {42: m},
                                      'mean_forecast_mae': m['forecast_prediction_mae'],
                                      'mean_persistence_mae': m['forecast_persistence_mae'],
                                      'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        save(status, {'state': 'COMPLETED', 'variable': variable, 'arm': arm, 'horizon': horizon,
                      'result_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'variable': variable, 'arm': arm, 'horizon': horizon, 'error': str(e)})
        traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--quick', action='store_true'); a = p.parse_args()
    run(a.variable, a.arm, a.horizon, a.quick)
