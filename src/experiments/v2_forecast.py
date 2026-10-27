"""Tuning protocol v2 for six-hour known-sensor forecasts (Table 5.8 arms and the released weather tree).

Each arm gets its own >=100-trial TPE study. Comparison arms use nested selection: for each outer
development fold the study scores an inner window (45 days ending 72 h before the fold opens) with a
model trained strictly before it; the outer fold is scored once, after selection, by a model trained
on everything before the fold. The release study selects on the outer development windows (the v1
procedure) because its independent test is the 2026 reserved period. No 2026 value is loaded.
"""
import argparse, hashlib, json, os, traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from src.experiments.bay_graph_dynamics import ROOT, build, folds, fit, metrics, save, tensors
from src.experiments.bay_dynamics_tree import features
from src.models.bay_graph_dynamics import BayGraphDynamics
from src.data.bay_forcing import weather
from src.tuning.protocol_v2 import run_study, PROTOCOL

TREE_ARMS = ['tree', 'weather_tree', 'xgb_tree', 'xgb_weather_tree']  # xgb_*: addendum-1, same features as the LightGBM arm
NEURAL_ARMS = {  # arm -> (graph, exchange, relaxation)
    'learned_source': ('base', False, False), 'conservative_exchange': ('base', True, False),
    'saturation_source': ('base', False, True), 'saturation_exchange': ('base', True, True),
    'coast_exchange': ('coast', True, False), 'coast_breakwater_exchange': ('coast_breakwater', True, False),
    'coast_depth_exchange': ('coast_depth', True, False)}
OXYGEN_ONLY = {'saturation_source', 'saturation_exchange'}
SEEDS = [42, 7, 123]
THREADS = int(os.environ.get('V2_THREADS', '4'))


def graphs(data):
    geometry = json.loads((ROOT / 'reports/bay_twin/exchange_geometry.json').read_text())
    ids = [geometry['sensors'].index(n) for n in data['names']]
    land = np.array(geometry['land_crossing'])[np.ix_(ids, ids)]
    barrier = np.array(geometry['breakwater_crossing'])[np.ix_(ids, ids)]
    depth = np.array(geometry['nominal_depth_m'])[ids]
    base = data['adjacency'].copy(); coast = base * ~land
    return {'base': base, 'coast': coast, 'coast_breakwater': coast * ~barrier,
            'coast_depth': coast * np.exp(-np.abs(depth[:, None] - depth[None, :]) / 3)}


def inner_fold(data, fold):
    """Inner selection window strictly inside the outer fold's training period."""
    t = data['times']; begin = t[fold['va']].min()
    end = begin - pd.Timedelta(hours=72); start = end - pd.Timedelta(days=45)
    tr = np.flatnonzero(t + pd.Timedelta(hours=78) < start)
    va = np.flatnonzero((t >= start) & (t + pd.Timedelta(hours=6) < end))
    eligible = fold['eligible'] & (data['valid'][tr].sum(0) >= 100) & (data['valid'][va].sum(0) >= 10)
    fallback = np.nan_to_num(pd.DataFrame(data['x'][tr]).median().to_numpy())
    return dict(number=fold['number'], tr=tr, va=va, eligible=eligible, fallback=fallback,
                audit={'fold': fold['number'], 'inner_selection_start': str(t[va].min()), 'inner_selection_end': str(end),
                       'last_inner_training_outcome': str(t[tr].max() + pd.Timedelta(hours=6))})


def tree_space(trial):
    return {'num_leaves': trial.suggest_int('num_leaves', 4, 63, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, .2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100., log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .5, 1.),
            'subsample': trial.suggest_float('subsample', .5, 1.), 'subsample_freq': 1}


def neural_space(trial):
    return {'hidden': trial.suggest_categorical('hidden', [16, 32, 64, 128]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, .1, log=True),
            'steps': trial.suggest_categorical('steps', [150, 300, 600])}


def xgb_space(trial):
    return {'max_depth': trial.suggest_int('max_depth', 2, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, .2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100., log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1., 100., log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .5, 1.),
            'subsample': trial.suggest_float('subsample', .5, 1.)}


def make_tree(params):
    if 'max_depth' in params:
        import xgboost as xgb
        return xgb.XGBRegressor(**params, objective='reg:absoluteerror', tree_method='hist', n_jobs=THREADS, random_state=42)
    return lgb.LGBMRegressor(**params, objective='regression_l1', verbosity=-1, n_jobs=THREADS, random_state=42,
                             deterministic=True, force_col_wise=True)


def tree_fit_score(data, x, fold, params):
    """Train on fold['tr'], score forecast on fold['va']; returns prediction rows."""
    mask = data['valid'] & fold['eligible'][None, :]; tr, va = fold['tr'], fold['va']; tm = mask[tr]
    w = np.broadcast_to(1 / np.maximum(tm.sum(0), 1), tm.shape)[tm]; w = w / w.mean()
    model = make_tree(params)
    model.fit(x[tr][tm], (data['truth'][tr] - data['x'][tr])[tm], sample_weight=w)
    vm = mask[va]; p = model.predict(x[va][vm]) + data['x'][va][vm]; rows = []
    for (j, s), value in zip(np.argwhere(vm), p):
        i = va[j]
        rows.append({'fold': fold['number'], 'sensor': data['names'][s], 'time': str(data['times'][i]), 'mode': 'forecast',
                     'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                     'prediction': float(value * data['scale'] + data['center']),
                     'persistence': float(data['x'][i, s] * data['scale'] + data['center'])})
    return pd.DataFrame(rows).assign(baseline=lambda f: f.persistence), model


def neural_forecast_only(data, fold, params, exchange, relaxation, seed, device, quick=False):
    """Same training loop as bay_graph_dynamics.fit; scores the forecast mode only (cheap inner loop)."""
    torch.manual_seed(seed); a = tensors(data, fold, device)
    model = BayGraphDynamics(data['adjacency'], data['forcing'].shape[-1], params['hidden'], exchange, relaxation).to(device).float()
    a = {k: v.float() if v.dtype != torch.bool else v for k, v in a.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    weights = 1 / a['valid'][fold['tr']].sum(0).clamp_min(1).float(); weights /= weights.mean()
    rng = np.random.default_rng(seed)
    for _ in range(3 if quick else params['steps']):
        ids = rng.choice(fold['tr'], min(128, len(fold['tr'])), replace=False)
        pred, _ = model.rollout(a['x'][ids], a['available'][ids], a['forcing'][ids]); mask = a['valid'][ids]
        loss = (torch.nn.functional.smooth_l1_loss(pred, a['truth'][ids], reduction='none') * mask * weights).sum() / (mask * weights).sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
        if not torch.isfinite(loss): raise ValueError('non-finite loss')
    with torch.no_grad():
        pred, _ = model.rollout(a['x'][fold['va']], a['available'][fold['va']], a['forcing'][fold['va']])
    p = pred.cpu().numpy(); valid = data['valid'][fold['va']] & fold['eligible'][None, :]; rows = []
    for j, s in np.argwhere(valid):
        i = fold['va'][j]
        rows.append({'fold': fold['number'], 'sensor': data['names'][s], 'mode': 'forecast',
                     'truth': float(data['truth'][i, s] * data['scale'] + data['center']),
                     'prediction': float(p[j, s] * data['scale'] + data['center']),
                     'persistence': float(data['x'][i, s] * data['scale'] + data['center'])})
    return pd.DataFrame(rows).assign(baseline=lambda f: f.persistence)


def ratio(d):
    m = metrics(d); return m['forecast_prediction_mae'] / m['forecast_persistence_mae']


def run(variable, arm, mode, quick=False):
    output = ROOT / 'results'; folder = output / f'v2_forecast_{variable}_{arm}_{mode}'; folder.mkdir(parents=True, exist_ok=True)
    status = output / f'status_v2_forecast_{variable}_{arm}_{mode}.json'
    save(status, {'state': 'RUNNING', 'variable': variable, 'arm': arm, 'mode': mode, 'pid': os.getpid()})
    try:
        if arm in OXYGEN_ONLY and variable != 'oxygen': raise ValueError('saturation arms are oxygen-only')
        if mode == 'release' and arm != 'weather_tree': raise ValueError('only the released family (LightGBM weather tree, protocol v2) has a release study')
        torch.set_num_threads(THREADS); device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = build(variable); outer = folds(data, False)
        if quick:
            for f in outer: f['tr'] = f['tr'][-300:]; f['va'] = f['va'][:40]
        inner = [inner_fold(data, f) for f in outer]
        if quick:
            for f in inner: f['tr'] = f['tr'][-300:]; f['va'] = f['va'][:40]
        select_on = outer if mode == 'release' else inner
        contract = {'protocol': PROTOCOL['version'], 'variable': variable, 'arm': arm, 'mode': mode, 'smoke_only': quick, 'device': device,
                    'outer_folds': [f['audit'] for f in outer], 'inner_folds': [f['audit'] for f in inner],
                    'selection': PROTOCOL['selection']['release' if mode == 'release' else 'comparisons'],
                    'objective': 'mean over folds of fold-macro forecast MAE / persistence MAE', 'no_2026': True}
        if arm in TREE_ARMS:
            met = weather(data['times']).to_numpy() if arm.endswith('weather_tree') else None; xgbarm = arm.startswith('xgb_')

            def design(fold):
                base, _ = features(data, {'fallback': fold['fallback']}, 'forecast')
                if met is None: return base
                return np.concatenate([base, np.broadcast_to(met[:, None], (len(data['times']), len(data['names']), met.shape[1]))], axis=-1)
            cache = {(id(f)): design(f) for f in outer + inner}

            def objective(trial):
                p = xgb_space(trial) if xgbarm else tree_space(trial)
                return float(np.mean([ratio(tree_fit_score(data, cache[id(f)], f, p)[0]) for f in select_on]))
            contract['search_space'] = 'XGBoost: max_depth 2-10, n_estimators 100-1000, learning_rate 5e-3-0.2, reg_lambda 1e-3-100, min_child_weight 1-100 (log); colsample_bytree, subsample 0.5-1; reg:absoluteerror' if xgbarm else 'LightGBM: num_leaves 4-63, n_estimators 100-1000, learning_rate 5e-3-0.2, reg_lambda 1e-3-100 (log); min_child_samples 5-200; colsample_bytree, subsample 0.5-1'
            save(folder / 'contract.json', contract)
            sel = run_study(f'{variable}_{arm}_{mode}', objective, folder, quick)
            params = dict(sel['params']) if xgbarm else {**sel['params'], 'subsample_freq': 1}
            frames = []
            for f in outer:
                d, model = tree_fit_score(data, cache[id(f)], f, params); frames.append(d)
                save(folder / f'selected_fold{f["number"]}_model.json', {'model_string': model.get_booster().save_raw('json').decode() if xgbarm else model.booster_.model_to_string(), 'params': params})
            d = pd.concat(frames, ignore_index=True); seeds = [42]
        else:
            graph, exchange, relaxation = NEURAL_ARMS[arm]; data['adjacency'] = graphs(data)[graph]

            def objective(trial):
                p = neural_space(trial)
                return float(np.mean([ratio(neural_forecast_only(data, f, p, exchange, relaxation, 42, device, quick)) for f in select_on]))
            contract['search_space'] = 'hidden {16,32,64,128}; lr 1e-4-1e-2 (log); weight_decay 1e-6-0.1 (log); steps {150,300,600}'
            contract['graph'] = graph; contract['exchange'] = exchange; contract['relaxation'] = relaxation
            save(folder / 'contract.json', contract)
            sel = run_study(f'{variable}_{arm}_{mode}', objective, folder, quick); params = sel['params']
            seeds = [42] if quick else SEEDS; frames = []
            for seed in seeds:  # full v1 evaluation (forecast and remote-update modes) at the selected configuration
                for f in outer:
                    df, _, state = fit(data, f, params, exchange, seed, device, quick, relaxation); frames.append(df.assign(seed=seed))
            d = pd.concat(frames, ignore_index=True)
        path = folder / 'selected_predictions.csv.gz'; d.to_csv(path, index=False)
        per_seed = {int(s): metrics(g.drop(columns='seed')) for s, g in d.groupby('seed')} if 'seed' in d else {42: metrics(d)}
        report = {'selection': sel, 'seeds': seeds, 'per_seed_metrics': per_seed,
                  'mean_forecast_mae': float(np.mean([m['forecast_prediction_mae'] for m in per_seed.values()])),
                  'mean_persistence_mae': float(np.mean([m['forecast_persistence_mae'] for m in per_seed.values()])),
                  'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        save(folder / 'report.json', report)
        save(status, {'state': 'COMPLETED', 'variable': variable, 'arm': arm, 'mode': mode,
                      'result_sha256': {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix != '.sqlite3'}})
    except BaseException as e:
        save(status, {'state': 'FAILED', 'variable': variable, 'arm': arm, 'mode': mode, 'error': str(e)}); traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variable', choices=['temperature', 'oxygen'], required=True)
    p.add_argument('--arm', choices=TREE_ARMS + list(NEURAL_ARMS), required=True)
    p.add_argument('--mode', choices=['nested', 'release'], default='nested')
    p.add_argument('--quick', action='store_true'); a = p.parse_args(); run(a.variable, a.arm, a.mode, a.quick)
