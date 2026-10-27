"""Tuning protocol v2 for two experiments whose results the thesis cites, near-bottom oxygen and the
utilisation target. Outputs go to reports/bay_twin/runs/v2_bottom_aou/<task>/.

  bottom : near-bottom oxygen from surface information (exp_bottom), P1 held-station and
           P2 chronological (train < 2026, test on the 2026 casts -- part of the second,
           labelled 2026 reading).
  aou    : direct-DO vs AOU target for a year-boundary generalisation (exp_spatial, P2 only),
           ridge+telemetry and LightGBM, both cohorts.

Loaders, features and folds are imported unchanged from scripts/experiments so the rows
are identical to v1. Only the estimators change: RidgeCV grids and fixed LightGBM settings become
per-(fold, model, target) TPE studies of >=100 trials, selected on inner splits of the outer
training rows (P1: GroupKFold over training stations; P2: the latest 25 % of training casts).
"""
import argparse, json, os, shutil, sys, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tuning.protocol_v2 import run_study, PROTOCOL

warnings.filterwarnings('ignore')
THREADS = int(os.environ.get('V2_THREADS', '4'))


def content_digest(raw):
    """SHA-256 of a gzip file's decompressed text with line endings normalised to LF: what the file
    holds, whatever the compressor stamped into its header and whichever line ending wrote it."""
    import gzip, hashlib
    return hashlib.sha256(gzip.decompress(raw).replace(b'\r\n', b'\n')).hexdigest()


def import_experiments(output):
    """exp_* modules read inputs from $SMARTBAY_BASE/review_20260919/inputs, the layout of the compute
    host; stage hash-identical copies in that layout under the run's staged_inputs/ (git-ignored).

    Each input must match its manifest entry: its bytes `sha256`, or else its content `sha256_content`
    (content_digest). pandas writes the time of writing and the file name into a gzip header, so a
    data-stage rerun over the same record gives the same content in different bytes; the content is
    what the experiments read."""
    base = output / 'staged_inputs'; inputs = base / 'review_20260919' / 'inputs'; inputs.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'reports/bay_twin/runs/battery/input_manifest.json').read_text())
    sources = {'ctd_all_casts.csv.gz': ROOT / 'data/processed/ctd_all_casts.csv.gz',
               'hourly_observations.csv.gz': ROOT / 'reports/bay_twin/hourly_observations.csv.gz'}
    import hashlib
    for name, src in sources.items():
        entry = manifest['review_20260919/inputs/' + name]
        raw = src.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256'] and entry.get('sha256_content') != content_digest(raw):
            raise ValueError('input differs from its manifest: ' + name)
        shutil.copyfile(src, inputs / name)
    os.environ['SMARTBAY_BASE'] = str(base)
    sys.path.insert(0, str(ROOT / 'scripts/experiments'))
    import exp_bottom, exp_spatial
    return exp_bottom, exp_spatial


def impute(tr, te, cols):
    xtr = tr[cols].to_numpy(float); xte = te[cols].to_numpy(float)
    mu = np.nanmean(xtr, axis=0); mu = np.where(np.isnan(mu), 0., mu)
    return np.where(np.isnan(xtr), mu, xtr), np.where(np.isnan(xte), mu, xte)


def make(name, params):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
    if name == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(**params, objective='reg:squarederror', tree_method='hist', random_state=0, n_jobs=THREADS)
    return lgb.LGBMRegressor(**params, subsample_freq=1, random_state=0, verbose=-1, n_jobs=THREADS)


def space(name, trial):
    if name == 'ridge':
        return {'alpha': trial.suggest_float('alpha', 1e-3, 1e7, log=True)}
    if name == 'xgboost':
        return {'max_depth': trial.suggest_int('max_depth', 2, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 5e-3, .2, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 1., 100., log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100., log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', .5, 1.),
                'subsample': trial.suggest_float('subsample', .5, 1.)}
    return {'num_leaves': trial.suggest_int('num_leaves', 4, 63, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, .2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100., log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .5, 1.),
            'subsample': trial.suggest_float('subsample', .5, 1.)}


def inner_splits(tr, protocol):
    if protocol == 'P1_held_station':
        g = tr.station.to_numpy(); k = min(4, len(np.unique(g)))
        return list(GroupKFold(n_splits=k).split(tr, groups=g))
    order = tr.groupby('cast_id').t_utc.min().sort_values(); cut = order.index[int(len(order) * .75):]
    va = np.flatnonzero(tr.cast_id.isin(cut)); return [(np.setdiff1d(np.arange(len(tr)), va), va)]


def tuned_predict(name, tr, te, cols, ycol, to_score, protocol, folder, study, quick):
    """Tune on inner splits of tr (scored in the final DO/temperature space), refit on tr, predict te."""
    splits = inner_splits(tr, protocol)

    def objective(trial):
        p = space(name, trial); errs = []
        for i, v in splits:
            a, b = tr.iloc[i], tr.iloc[v]; xa, xb = impute(a, b, cols)
            pred = make(name, p).fit(xa, a[ycol].to_numpy(float)).predict(xb)
            errs.append(np.abs(to_score(pred, b)).mean())
        return float(np.mean(errs))
    sel = run_study(study, objective, folder, quick)
    params = sel['params']; xtr, xte = impute(tr, te, cols)
    return make(name, params).fit(xtr, tr[ycol].to_numpy(float)).predict(xte), sel


def bottom(exp_bottom, folder, quick):
    d = exp_bottom.build(); rows, preds, sels = [], [], []
    for protocol in ('P1_held_station', 'P2_chronological'):
        for fold, tr, te in exp_bottom.folds(d, protocol):
            tr = tr.reset_index(drop=True); te = te.reset_index(drop=True); truth = te.bot_DO.to_numpy(float)
            cand = {'B_surface': te.surf_DO.to_numpy(float), 'B_climatology': np.full(len(te), tr.bot_DO.mean())}
            for rg, cols in exp_bottom.REGIMES.items():
                targets = [('direct', 'bot_DO', lambda p, x: p - x.bot_DO.to_numpy(float), lambda p, x: p),
                           ('aou', 'bot_AOU', lambda p, x: (x.bot_sol.to_numpy(float) - p) - x.bot_DO.to_numpy(float), lambda p, x: x.bot_sol.to_numpy(float) - p)]
                if rg == 'C1_cast_surface':
                    targets.append(('gradient', 'delta_DO', lambda p, x: (x.surf_DO.to_numpy(float) + p) - x.bot_DO.to_numpy(float), lambda p, x: x.surf_DO.to_numpy(float) + p))
                for mdl in ('ridge', 'lightgbm'):
                    for tname, ycol, err, back in targets:
                        key = f'{rg}|{mdl}|{tname}'
                        p, sel = tuned_predict(mdl, tr, te, cols, ycol, err, protocol, folder / 'studies',
                                               f'bottom_{protocol}_{fold}_{rg}_{mdl}_{tname}', quick)
                        cand[key] = back(p, te); sels.append({'protocol': protocol, 'fold': str(fold), 'model': key, **sel})
            for k, v in cand.items():
                rows.append(dict(protocol=protocol, fold=str(fold), model=k, n=len(te), mae=float(np.abs(v - truth).mean())))
                preds.append(pd.DataFrame(dict(protocol=protocol, fold=str(fold), model=k, cast_id=te.cast_id.values,
                                               truth=truth, pred=v, abs_delta=np.abs(te.delta_DO.values))))
            print(f'bottom {protocol}/{fold} done', flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), sels


def aou(exp_spatial, folder, quick, models=(('M2_ridge_tel', 'ridge'), ('M3_lightgbm', 'lightgbm'))):
    c = exp_spatial.featurise(exp_spatial.load()); rows, preds, sels = [], [], []
    for cname, cdf in {'gen2_164': c[c.t_utc >= '2024-07-24'], 'full_722': c}.items():
        for fold, tr, te in exp_spatial.folds(cdf, 'P2_chronological'):
            tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
            for mdl, name in models:
                cols = exp_spatial.FEATS
                for sp, ycol in [('direct_do', 'do_umol_l'), ('aou', 'aou')]:
                    err = (lambda p, x, sp=sp: exp_spatial.to_do(sp, p, x) - x.do_umol_l.to_numpy(float))
                    p, sel = tuned_predict(name, tr, te, cols, ycol, err, 'P2_chronological', folder / 'studies',
                                           f'aou_{cname}_{mdl}_{sp}', quick)
                    do = exp_spatial.to_do(sp, p, te)
                    rows.append(dict(cohort=cname, protocol='P2_chronological', fold=fold, space=sp, model=mdl, n=len(te),
                                     n_casts=te.cast_id.nunique(), mae=float(np.abs(do - te.do_umol_l.values).mean())))
                    preds.append(pd.DataFrame(dict(cohort=cname, space=sp, model=mdl, cast_id=te.cast_id.values,
                                                   truth=te.do_umol_l.values, pred=do)))
                    sels.append({'cohort': cname, 'model': mdl, 'space': sp, **sel})
            print(f'aou {cname} done', flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), sels


def run(task, quick=False):
    folder = ROOT / 'reports/bay_twin/runs/v2_bottom_aou' / (task + ('_smoke' if quick else '')); folder.mkdir(parents=True, exist_ok=True)
    (folder / 'status.json').write_text(json.dumps({'state': 'RUNNING', 'task': task}))
    try:
        exp_bottom, exp_spatial = import_experiments(folder)
        res, preds, sels = (bottom(exp_bottom, folder, quick) if task == 'bottom' else
                            aou(exp_spatial, folder, quick, (('M3b_xgboost', 'xgboost'),)) if task == 'aou_xgb' else aou(exp_spatial, folder, quick))
        res.to_csv(folder / 'folds.csv', index=False); preds.to_csv(folder / 'predictions.csv.gz', index=False)
        (folder / 'selections.json').write_text(json.dumps(sels, indent=2, default=str))
        short = [s['study'] for s in sels if s['completed_trials'] < PROTOCOL['min_completed_trials']]
        (folder / 'status.json').write_text(json.dumps({'state': 'COMPLETED', 'task': task, 'protocol': PROTOCOL['version'], 'studies': len(sels),
                                                        'studies_below_minimum': short, 'smoke_only': quick}, indent=2))
    except BaseException as e:
        (folder / 'status.json').write_text(json.dumps({'state': 'FAILED', 'task': task, 'error': str(e)})); traceback.print_exc(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--task', choices=['bottom', 'aou', 'aou_xgb'], required=True); p.add_argument('--quick', action='store_true')
    a = p.parse_args(); run(a.task, a.quick)
