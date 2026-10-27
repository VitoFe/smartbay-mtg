"""Addendum-8 part B: seabed depth as a forecasting feature, pooled and held-sensor.

The surveyed seabed depth exists for every probe in sensor_geometry.csv and has never been a model
feature: the static block carries the probe's coordinates and its own mooring depth only, so the
estimator does not know that a probe at 1 m sits over an 8 m seabed while another at 5 m sits over
11 m. Three features are added, all derived from data already bound:

    seabed_m        surveyed seabed depth
    clearance_m     seabed minus mooring depth, the sensor's height above the bed
    relative_depth  mooring depth over seabed depth, the fraction of water column above the sensor

The point of the study is that these are evaluated twice. Coordinates already identify every probe
exactly, so the pooled model may gain nothing; but coordinates do not transfer to a probe the model
has never seen, whereas a fraction of the water column does. The held-sensor protocol of addendum 7
is what separates the two.

Development folds only. Declared in config/bay_reserved_protocol_v2_addendum8.json.
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
from src.tuning.protocol_v2 import run_study, PROTOCOL

COLUMNS = ['seabed_m', 'clearance_m', 'relative_depth']


def bathymetric_block(data):
    """Per-sensor static features, broadcast over time. No new data source is introduced."""
    meta = pd.read_csv(ROOT / 'reports/bay_twin/sensor_geometry.csv').set_index('sensor')
    meta = meta.loc[list(data['names'])]
    seabed = meta.metadata_bottom_m.to_numpy(dtype=float)
    mooring = meta.nominal_depth_m.to_numpy(dtype=float)
    block = np.column_stack([seabed / 10.0, (seabed - mooring) / 10.0,
                             np.divide(mooring, seabed, out=np.zeros_like(mooring), where=seabed > 0)])
    return np.broadcast_to(block, (len(data['times']), *block.shape))


def design_for(data, fold):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    met = weather(data['times']).to_numpy()
    n = len(data['names'])
    return np.concatenate([base,
                           np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1])),
                           bathymetric_block(data)], axis=-1)


def run(variable, horizon, quick=False):
    output = ROOT / 'results'
    tag = f'v2_bathy_{variable}_bathy_tree_h{horizon}'
    folder = output / tag; folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_{tag}.json'
    save(status, {'state': 'RUNNING', 'variable': variable, 'horizon': horizon, 'pid': os.getpid()})
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
        cache = {id(f): design_for(data, f) for f in outer + inner}

        save(folder / 'contract.json', {
            'protocol': PROTOCOL['version'], 'declared_in': 'config/bay_reserved_protocol_v2_addendum8.json',
            'variable': variable, 'arm': 'bathy_tree', 'horizon_hours': horizon, 'smoke_only': quick,
            'features': COLUMNS, 'source': 'reports/bay_twin/sensor_geometry.csv, already bound',
            'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
            'selection': PROTOCOL['selection']['comparisons'],
            'objective': 'mean over folds of fold-macro forecast MAE / persistence MAE', 'no_2026': True})

        def objective(trial):
            p = tree_space(trial)
            return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))

        sel = run_study(f'{variable}_bathy_tree_h{horizon}', objective, folder, quick)
        params = {**sel['params'], 'subsample_freq': 1}
        d = pd.concat([tree_fit_score(data, cache[id(f)], f, params, clim[id(f)])[0] for f in outer],
                      ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'; d.to_csv(path, index=False)

        from src.experiments.bay_graph_dynamics import metrics
        m = metrics(d)
        save(folder / 'report.json', {'selection': sel, 'horizon_hours': horizon, 'endpoints': len(d),
                                      'per_seed_metrics': {42: m},
                                      'mean_forecast_mae': m['forecast_prediction_mae'],
                                      'mean_persistence_mae': m['forecast_persistence_mae'],
                                      'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        save(status, {'state': 'COMPLETED', 'variable': variable, 'horizon': horizon,
                      'result_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'variable': variable, 'horizon': horizon, 'error': str(e)})
        traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--quick', action='store_true'); a = p.parse_args()
    run(a.variable, a.horizon, a.quick)
