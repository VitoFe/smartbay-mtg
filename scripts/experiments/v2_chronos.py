"""Tuning protocol v2 for Chronos-Bolt fine-tuning (Table 5.5): >=100-trial TPE study per task.

Same data, windows, validation split, early-stopping rule and prediction code as
exp_chronos_valtest.py; v1 compared four hand-picked (lr, steps) pairs. Selection uses the
pre-2026 validation loss ONLY. Each trial's 2026 macro MAE is recorded as a post-hoc diagnostic
(Optuna user attribute) and never enters the objective; the selected trial's 2026 predictions are
the reported fine-tuned result. Runs from review_20260919/v2 on Trinity.
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'code'))
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
import torch
from chronos import BaseChronosPipeline
import exp_chronos_valtest as vt
from exp_chronos import o2_sol, ep, sensors_for, predict_at, macro_mae
from src.tuning.protocol_v2 import run_study, PROTOCOL

OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
QUICK = '--quick' in sys.argv
ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
FINALIZE = '--finalize' in sys.argv          # study workers (default) run trials; --finalize merges and writes outputs
SUFFIX = f'_{ONLY}' if ONLY else ''


def train(ctxs, tgts, ti, vi, lr, max_steps, weight_decay, batch, rng):
    pipe = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    model = pipe.model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_on(sel):
        c = torch.tensor(ctxs[sel], device='cuda'); t = torch.tensor(tgts[sel], device='cuda')
        return model(context=torch.nan_to_num(c), mask=torch.isfinite(c), target=t, target_mask=torch.isfinite(t)).loss
    best, best_state, patience = float('inf'), None, 0
    for step in range(20 if QUICK else max_steps):
        l = loss_on(rng.choice(ti, batch, replace=False)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % (10 if QUICK else 100) == 0:
            model.eval()
            with torch.no_grad(): vl = float(np.mean([loss_on(vi[i:i + 64]).item() for i in range(0, len(vi), 64)]))
            model.train()
            if not np.isfinite(vl): raise ValueError('non-finite validation loss')
            if vl < best - 1e-4: best, patience, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 4: break
    if best_state is None: raise ValueError('no validation improvement recorded')
    model.load_state_dict(best_state); model.eval(); return pipe, best


def to_column(pipe, variable, space, eval_times, temp_zs):
    preds = predict_at(pipe, variable, space, eval_times); col = np.full(len(ep), np.nan)
    for i, r in ep.iterrows():
        if r.variable != variable or (r.sensor, r.time) not in preds: continue
        v = preds[(r.sensor, r.time)]
        if space == 'aou':
            T = temp_zs.get((r.sensor, r.time), np.nan)
            if not np.isfinite(T): continue
            v = o2_sol(T) - v
        col[i] = v
    return col


def main():
    eval_times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), 'time'])) for s in sensors_for(v)} for v in ('temperature', 'oxygen')}
    zs = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = predict_at(zs, 'temperature', 'raw', eval_times['temperature']); del zs; torch.cuda.empty_cache()
    report = {}
    for variable, space in [('temperature', 'raw'), ('oxygen', 'aou')]:
        ctxs, tgts = vt.windows(variable, space)                 # drawn even when skipped: keeps the shared RNG stream identical to v1
        n_val = max(128, int(.1 * len(ctxs))); idx = vt.RNG.permutation(len(ctxs)); vi, ti = idx[:n_val], idx[n_val:]
        if ONLY and variable != ONLY: continue
        name = f'chronos_{variable}_{space}'; coldir = OUT / 'columns' / name; coldir.mkdir(parents=True, exist_ok=True)

        def fit_config(p, number):
            pipe, val = train(ctxs, tgts, ti, vi, rng=np.random.default_rng(number), **p)
            col = to_column(pipe, variable, space, eval_times[variable], temp_zs); del pipe; torch.cuda.empty_cache()
            return val, col

        def objective(trial):
            p = dict(lr=trial.suggest_float('lr', 1e-6, 1e-3, log=True), max_steps=trial.suggest_categorical('max_steps', [400, 800, 1500, 2500]),
                     weight_decay=trial.suggest_float('weight_decay', 1e-5, .3, log=True), batch=trial.suggest_categorical('batch', [16, 32, 64]))
            val, col = fit_config(p, trial.number)
            trial.set_user_attr('mae_2026_posthoc', float(macro_mae(ep.assign(_c=col)[ep.variable == variable], '_c')))
            np.save(coldir / f'{trial.number}.npy', col)          # on disk at once: several workers share the study
            return val
        if not FINALIZE:
            run_study(name, objective, OUT / 'studies', QUICK)
            continue
        import optuna
        study = optuna.load_study(study_name=name, storage='sqlite:///' + str(OUT / 'studies' / f'{name}.sqlite3'))
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not QUICK and len(complete) < PROTOCOL['min_completed_trials']: raise RuntimeError(f'{name}: only {len(complete)} completed trials')
        best_t = study.best_trial
        sel = {'study': name, 'protocol': PROTOCOL['version'], 'completed_trials': len(complete), 'best_trial': best_t.number,
               'best_value': best_t.value, 'params': best_t.params, 'smoke_only': QUICK}
        cols = {int(f.stem): np.load(f) for f in coldir.glob('*.npy')}
        if best_t.number not in cols:   # trial ran in a worker stopped before columns were saved to disk: refit its configuration
            _, cols[best_t.number] = fit_config(best_t.params, best_t.number); sel['selected_column'] = 'refit of the selected configuration (same seed)'
            sel['refit_mae_2026'] = float(macro_mae(ep.assign(_c=cols[best_t.number])[ep.variable == variable], '_c'))
        curve = [{'trial': t.number, **t.params, 'val_loss': t.value, 'mae_2026_posthoc': t.user_attrs.get('mae_2026_posthoc')}
                 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pd.DataFrame(curve).to_csv(OUT / f'{name}_curve.csv', index=False)
        np.savez_compressed(OUT / f'{name}_2026_columns.npz', **{str(k): v for k, v in cols.items()})
        best = sel['best_trial']
        if best in cols: ep[f'V2FT_{variable}_{space}'] = cols[best]
        c = pd.DataFrame(curve); rho = float(c[['val_loss', 'mae_2026_posthoc']].corr(method='spearman').iloc[0, 1]) if len(c) > 2 else None
        report[name] = {'selection': sel, 'selected_mae_2026': next(r['mae_2026_posthoc'] for r in curve if r['trial'] == best),
                        'best_posthoc_mae_2026': float(c.mae_2026_posthoc.min()), 'spearman_val_vs_2026': rho,
                        'note': 'Selected on pre-2026 validation loss only; 2026 MAEs of other trials are post-hoc diagnostics.'}
        print(json.dumps(report[name], indent=2, default=str), flush=True)
    if not FINALIZE: return
    ep.to_csv(OUT / f'v2_chronos_predictions{SUFFIX}.csv.gz', index=False)
    (OUT / f'v2_chronos_report{SUFFIX}.json').write_text(json.dumps({'protocol': PROTOCOL['version'], 'smoke_only': QUICK, **report}, indent=2, default=str))


if __name__ == '__main__':
    main()
