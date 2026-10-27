"""Addendum-7 studies 1 and 2: an external reference station as forecasting context.

Two arms, each added to the released weather tree and each with its own >=100-trial nested study
so that neither inherits a penalty tuned in its absence:

    sealevel_tree      + sea level, its 6 h change, and its 24 h range (tidal amplitude proxy)
    extboundary_tree   + external water temperature, its 24 h change, and the difference between
                         it and the target probe's own last observed value

The second asks whether an in-situ measurement can do the job the 4.2 km Copernicus background
offset currently does for temperature. The first asks whether tidal forcing is informative at all;
the published modelling of this gulf says semidiurnal tides drive almost no net exchange here, so a
null is the predicted outcome rather than a disappointment.

Development folds only; the hourly index ends 2025-12-31. Declared in
config/bay_reserved_protocol_v2_addendum7.json; the station's hourly means are labelled at the end of
their hour (src/data/bay_ispra.py) under config/bay_reserved_protocol_v2_addendum59.json, part B, and
each run records the binning it used in its contract.json.
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
from src.data.bay_ispra import (sea_level, external_boundary, LEVEL_COLUMNS, BOUNDARY_COLUMNS, BINNINGS,
                                binning)
from src.tuning.protocol_v2 import run_study, PROTOCOL

ARMS = ['sealevel_tree', 'extboundary_tree']


def design_for(data, fold, arm):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    n = len(data['names'])
    met = weather(data['times']).to_numpy()
    blocks = [base, np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))]
    if arm == 'sealevel_tree':
        extra = sea_level(data['times'])[LEVEL_COLUMNS].to_numpy()
        blocks.append(np.broadcast_to(extra[:, None], (len(data['times']), n, extra.shape[1])))
    else:
        station = external_boundary(data['times'])[BOUNDARY_COLUMNS].to_numpy()
        blocks.append(np.broadcast_to(station[:, None], (len(data['times']), n, station.shape[1])))
        # the difference between the external station and each probe's own anchor, in physical
        # units: this is the quantity a boundary condition is actually informative about
        anchor = data['x'] * data['scale'] + data['center']
        gap = station[:, 0:1] - anchor
        blocks.append(gap[..., None])
    return np.concatenate(blocks, axis=-1)


def run(variable, arm, horizon, quick=False):
    output = ROOT / 'results'
    tag = f'v2_ispra_{variable}_{arm}_h{horizon}'
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
        cache = {id(f): design_for(data, f, arm) for f in outer + inner}

        save(folder / 'contract.json', {
            'protocol': PROTOCOL['version'], 'declared_in': 'config/bay_reserved_protocol_v2_addendum7.json',
            'variable': variable, 'arm': arm, 'horizon_hours': horizon, 'smoke_only': quick,
            'source': 'ISPRA La Spezia station; position relative to the breakwater not established '
                      'in this repository, so it is an external reference rather than a verified '
                      'outer-gulf boundary',
            'features': LEVEL_COLUMNS if arm == 'sealevel_tree' else BOUNDARY_COLUMNS + ['ext_minus_anchor'],
            'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
            'selection': PROTOCOL['selection']['comparisons'],
            'objective': 'mean over folds of fold-macro forecast MAE / persistence MAE', 'no_2026': True,
            'hourly_binning': binning(), 'hourly_binning_rule': BINNINGS[binning()],
            'binning_declared_in': 'config/bay_reserved_protocol_v2_addendum59.json'})

        def objective(trial):
            p = tree_space(trial)
            return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p, clim[id(f)])[0]) for f in inner]))

        sel = run_study(f'{variable}_{arm}_h{horizon}', objective, folder, quick)
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
        save(status, {'state': 'COMPLETED', 'variable': variable, 'arm': arm, 'horizon': horizon,
                      'result_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'variable': variable, 'arm': arm, 'horizon': horizon, 'error': str(e)})
        traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--quick', action='store_true'); a = p.parse_args()
    run(a.variable, a.arm, a.horizon, a.quick)
