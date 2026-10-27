"""Addendum-26: a tuned lag for the river signal, and a gated second catchment.

Addendum 6 gave the Magra gauge three features on fixed windows: a 6 h mean, a 24 h difference and
a 72 h maximum. The windows were chosen a priori from advective reasoning and never varied, so the
0.250 umol/l it found at 24 h rests on one alignment among many. Two things follow.

First, the lag becomes a tuned hyperparameter rather than an assumption. It is selected inside the
nested study alongside the tree parameters, on inner folds, so no reported number comes from an
alignment chosen by looking at the reported rows. A sweep whose best value is read off the results
would be a forking path; a sweep inside the selection is not.

Second, a second catchment enters, gated. Bocca d'Arno is 54-55 km south-east and a different river; the
argument for it is that it reaches this bay only when the local water is already high, so its
features are zeroed outside that condition and the gate is carried as its own feature.

Arms, all against the plain tree as the reference the augmented-arm rule requires:

    ameglia_lag   the Magra triple with a tuned lag
    bocca_lag     the Arno triple with a tuned lag, ungated, to show what it does alone
    bocca_gated   the Arno triple zeroed unless the Magra gauge is high, plus the gate

Declared in config/bay_reserved_protocol_v2_addendum26.json.
Usage: python -m src.experiments.v2_river_lag --variable oxygen --arm ameglia_lag --horizon 24
"""
import argparse
import hashlib
import json
import os
import traceback

import numpy as np
import pandas as pd

from src.data.bay_bocca import LAG_COLUMNS, ameglia, bocca, features_at_lag, gate_from_ameglia
from src.data.bay_forcing import weather
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_graph_dynamics import ROOT, save
from src.experiments.v2_forecast import tree_space
from src.experiments.v2_horizon import (HORIZONS, build, climatology, folds, inner_fold, ratio,
                                        tree_fit_score)
from src.tuning.protocol_v2 import PROTOCOL, run_study

ARMS = ['ameglia_lag', 'bocca_lag', 'bocca_gated']
LAG_GRID = list(range(0, 97, 6))   # hours; 0 reproduces the addendum-6 alignment
GATE_QUANTILE = 0.75               # declared before scoring; 'the local water is high'


def river_block(arm, times, lag):
    """The three river features for this arm at this lag, as a (time, feature) array."""
    if arm == 'ameglia_lag':
        return features_at_lag(ameglia(), times, lag)[LAG_COLUMNS].to_numpy()
    block = features_at_lag(bocca(), times, lag)[LAG_COLUMNS].to_numpy()
    if arm == 'bocca_lag':
        return block
    gate, _ = gate_from_ameglia(times, GATE_QUANTILE)
    # Zeroed outside the gate, with the gate itself carried so the tree can tell the two apart.
    return np.concatenate([block * gate[:, None], gate[:, None].astype(float)], axis=1)


def design_for(data, fold, arm, lag):
    base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
    n = len(data['names'])
    met = weather(data['times']).to_numpy()
    blocks = [base, np.broadcast_to(met[:, None], (len(data['times']), n, met.shape[1]))]
    flow = np.nan_to_num(river_block(arm, data['times'], lag))
    blocks.append(np.broadcast_to(flow[:, None], (len(data['times']), n, flow.shape[1])))
    return np.concatenate(blocks, axis=-1)


def run(variable, arm, horizon, quick=False):
    output = ROOT / 'results'
    tag = f'v2_riverlag_{variable}_{arm}_h{horizon}'
    folder = output / tag
    folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_{tag}.json'
    save(status, {'state': 'RUNNING', 'variable': variable, 'arm': arm, 'horizon': horizon,
                  'pid': os.getpid()})
    try:
        if horizon not in HORIZONS:
            raise ValueError(f'undeclared horizon {horizon}')
        data = build(variable, horizon)
        outer = folds(data, horizon)
        if quick:
            for f in outer:
                f['tr'] = f['tr'][-300:]
                f['va'] = f['va'][:40]
        inner = [inner_fold(data, f, horizon) for f in outer]
        if quick:
            for f in inner:
                f['tr'] = f['tr'][-300:]
                f['va'] = f['va'][:40]
        clim = {id(f): climatology(data, f) for f in outer + inner}
        grid = LAG_GRID[:3] if quick else LAG_GRID

        # Designs are cached per (fold, lag): the lag changes the features, so a single per-fold
        # cache would silently reuse one alignment for every trial.
        cache = {}

        def design(fold, lag):
            key = (id(fold), lag)
            if key not in cache:
                cache[key] = design_for(data, fold, arm, lag)
            return cache[key]

        save(folder / 'contract.json', {
            'protocol': PROTOCOL['version'],
            'declared_in': 'config/bay_reserved_protocol_v2_addendum26.json',
            'variable': variable, 'arm': arm, 'horizon_hours': horizon, 'smoke_only': quick,
            'lag_grid_hours': grid, 'gate_quantile': GATE_QUANTILE,
            'selection': 'lag tuned inside the nested study with the tree parameters',
            'sources': {'ameglia': 'ARPAL AMEGLIA FOCE MAGRA hourly stage (m)',
                        'bocca': "Bocca d'Arno daily stage (m), held forward causally"},
            'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
            'no_2026': True})

        def objective(trial):
            lag = trial.suggest_categorical('river_lag_h', grid)
            p = tree_space(trial)
            return float(np.mean([ratio(tree_fit_score(data, design(f, lag), f, p, clim[id(f)])[0])
                                  for f in inner]))

        sel = run_study(f'{variable}_{arm}_h{horizon}', objective, folder, quick)
        lag = sel['params']['river_lag_h']
        params = {**{k: v for k, v in sel['params'].items() if k != 'river_lag_h'},
                  'subsample_freq': 1}
        frames = [tree_fit_score(data, design(f, lag), f, params, clim[id(f)])[0] for f in outer]
        d = pd.concat(frames, ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'
        d.to_csv(path, index=False)

        from src.experiments.bay_graph_dynamics import metrics
        m = metrics(d)
        save(folder / 'report.json', {
            'selection': sel, 'selected_lag_hours': lag, 'horizon_hours': horizon,
            'endpoints': len(d), 'mean_forecast_mae': m['forecast_prediction_mae'],
            'mean_persistence_mae': m['forecast_persistence_mae'],
            'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        save(status, {'state': 'COMPLETED', 'variable': variable, 'arm': arm, 'horizon': horizon,
                      'selected_lag_hours': lag,
                      'result_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as exc:
        save(status, {'state': 'FAILED', 'variable': variable, 'arm': arm, 'horizon': horizon,
                      'error': str(exc)})
        traceback.print_exc()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run(a.variable, a.arm, a.horizon, a.quick)


if __name__ == '__main__':
    main()
