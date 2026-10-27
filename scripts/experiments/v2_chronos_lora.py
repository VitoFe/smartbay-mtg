"""Addendum 40: low-rank adapters for Chronos-Bolt, under the full fine-tuning study's protocol.

Everything that v2_chronos.py fixes is kept: the model, the pre-2026 windows and their validation
split (drawn through the unmodified exp_chronos_valtest in the same order), the early-stopping rule,
the step budgets, the selection on validation loss alone, and a >=100-trial study per task. Only the
trainable parameters change: every base weight is frozen and low-rank adapters are added to the T5
attention (and optionally feed-forward) projections, with B = 0 so the untrained model is zero-shot.
Each trial's Reserved-2026 error and trainable-parameter count are recorded as post-hoc diagnostics
and never enter the objective.

  python v2_chronos_lora.py --gates                 # G1 (zero-shot reproduced) and the window digests
  python v2_chronos_lora.py --only temperature      # a study worker; several may share one study
  python v2_chronos_lora.py --finalize              # merge both studies and write the outputs
  add --quick for a smoke run into results_smoke/ (never the real studies)

Runs from a snapshot directory on Trinity with code/ (exp_chronos.py, exp_chronos_valtest.py),
src/tuning/protocol_v2.py and config/tuning_protocol_v2.json beside it.
"""
import hashlib, json, math, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'code'))
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
import torch
torch.set_num_threads(int(os.environ.get('V2_THREADS', '3')))
from chronos import BaseChronosPipeline
import exp_chronos_valtest as vt
from exp_chronos import o2_sol, ep, sensors_for, predict_at, macro_mae
from src.tuning.protocol_v2 import run_study, PROTOCOL

QUICK = '--quick' in sys.argv
ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
OUT = HERE / ('results_smoke' if QUICK else 'results'); OUT.mkdir(exist_ok=True)
TASKS = [('temperature', 'raw'), ('oxygen', 'aou')]
TARGETS = {'qv': ('q', 'v'), 'qkvo': ('q', 'k', 'v', 'o'), 'qkvo_ffn': ('q', 'k', 'v', 'o', 'wi', 'wo')}
PUBLISHED_ZERO_SHOT = {'temperature': 0.1724, 'oxygen': 5.3911}      # battery_full_v2.json, 4 decimals


class LoRALinear(torch.nn.Module):
    """W x + (alpha / r) B A dropout(x), with W frozen and B = 0 at the start (Hu et al., 2022)."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base, self.scale, self.drop = base, alpha / rank, torch.nn.Dropout(dropout)
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, **kw))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, **kw))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):                    # T5's feed-forward reads wo.weight.dtype to cast its input
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_a.T @ self.lora_b.T) * self.scale


def attach(model, targets, rank, alpha, dropout):
    """Freeze every weight, wrap the targeted Linear layers inside the T5 stacks, return the count."""
    for p in model.parameters():
        p.requires_grad_(False)
    names, n = TARGETS[targets], 0
    for stack in (model.encoder, model.decoder):
        for parent in list(stack.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in names and isinstance(child, torch.nn.Linear):
                    setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout)); n += 1
    return n


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def train(ctxs, tgts, ti, vi, lr, max_steps, weight_decay, batch, rank, alpha_over_rank, dropout, targets, rng, seed):
    pipe = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    torch.manual_seed(seed)
    model = pipe.model
    wrapped = attach(model, targets, rank, alpha_over_rank * rank, dropout)
    params = trainable(model); n_trainable = sum(p.numel() for p in params)
    model.train(); opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def loss_on(sel):
        c = torch.tensor(ctxs[sel], device='cuda'); t = torch.tensor(tgts[sel], device='cuda')
        return model(context=torch.nan_to_num(c), mask=torch.isfinite(c), target=t, target_mask=torch.isfinite(t)).loss
    best, best_state, patience, stopped = float('inf'), None, 0, None
    for step in range(20 if QUICK else max_steps):
        l = loss_on(rng.choice(ti, batch, replace=False)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (step + 1) % (10 if QUICK else 100) == 0:
            model.eval()
            with torch.no_grad(): vl = float(np.mean([loss_on(vi[i:i + 64]).item() for i in range(0, len(vi), 64)]))
            model.train()
            if not np.isfinite(vl): raise ValueError('non-finite validation loss')
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            else:
                patience += 1
                if patience >= 4: stopped = step + 1; break
    if best_state is None: raise ValueError('no validation improvement recorded')
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state: p.copy_(best_state[n])
    model.eval()
    return pipe, best, {'trainable_params': int(n_trainable), 'wrapped_layers': wrapped, 'stopped_at': stopped}


def to_column(pipe, variable, space, eval_times, temp_zs):
    """The 2026 prediction column, as v2_chronos.to_column: oxygen through solubility at the zero-shot temperature."""
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


def mae(variable, col):
    return float(macro_mae(ep.assign(_c=col)[ep.variable == variable], '_c'))


def draw_windows():
    """The windows and split of v2_chronos.main, drawn in its order from the same RNG stream."""
    out = {}
    for variable, space in TASKS:
        ctxs, tgts = vt.windows(variable, space)
        n_val = max(128, int(.1 * len(ctxs))); idx = vt.RNG.permutation(len(ctxs)); vi, ti = idx[:n_val], idx[n_val:]
        out[variable] = (ctxs, tgts, ti, vi)
    return out


def digest(ctxs, tgts, ti, vi):
    h = hashlib.sha256()
    for a in (ctxs, tgts, ti, vi): h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def gates(eval_times):
    zs = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = predict_at(zs, 'temperature', 'raw', eval_times['temperature'])
    plain = {v: to_column(zs, v, s, eval_times[v], temp_zs) for v, s in TASKS}
    torch.manual_seed(0); attach(zs.model, 'qkvo_ffn', 8, 16, 0.0); zs.model.eval()
    wrapped = {v: to_column(zs, v, s, eval_times[v], temp_zs) for v, s in TASKS}
    windows = draw_windows()
    g1 = {v: {'max_abs_diff': float(np.nanmax(np.abs(plain[v] - wrapped[v]))),
              'same_missing': bool((np.isnan(plain[v]) == np.isnan(wrapped[v])).all()),
              'mae_plain': mae(v, plain[v]), 'mae_wrapped': mae(v, wrapped[v]), 'published': PUBLISHED_ZERO_SHOT[v]}
          for v, _ in TASKS}
    ok = all(g['max_abs_diff'] <= 1e-5 and g['same_missing'] and abs(g['mae_wrapped'] - g['published']) <= 1e-4 for g in g1.values())
    report = {'G1': {'passed': ok, **g1},
              'windows': {v: {'n_windows': int(len(w[0])), 'n_train': int(len(w[2])), 'n_val': int(len(w[3])), 'sha256': digest(*w)}
                          for v, w in windows.items()}}
    (OUT / 'gates.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return ok


def main():
    eval_times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), 'time'])) for s in sensors_for(v)} for v, _ in TASKS}
    if '--gates' in sys.argv:
        raise SystemExit(0 if gates(eval_times) else 1)
    zs = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = predict_at(zs, 'temperature', 'raw', eval_times['temperature'])
    zero_shot = {v: mae(v, to_column(zs, v, s, eval_times[v], temp_zs)) for v, s in TASKS}
    del zs; torch.cuda.empty_cache()
    windows = draw_windows()                           # both tasks, always: keeps the RNG stream identical
    report, cols_out = {}, {}
    for variable, space in TASKS:
        if ONLY and variable != ONLY: continue
        ctxs, tgts, ti, vi = windows[variable]
        name = f'lora_{variable}_{space}'; coldir = OUT / 'columns' / name; coldir.mkdir(parents=True, exist_ok=True)

        def fit_config(p, number):
            pipe, val, info = train(ctxs, tgts, ti, vi, rng=np.random.default_rng(number), seed=number, **p)
            col = to_column(pipe, variable, space, eval_times[variable], temp_zs); del pipe; torch.cuda.empty_cache()
            return val, col, info

        def objective(trial):
            p = dict(lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                     max_steps=trial.suggest_categorical('max_steps', [400, 800, 1500, 2500]),
                     weight_decay=trial.suggest_float('weight_decay', 1e-5, .3, log=True),
                     batch=trial.suggest_categorical('batch', [16, 32, 64]),
                     rank=trial.suggest_categorical('rank', [1, 2, 4, 8, 16, 32]),
                     alpha_over_rank=trial.suggest_categorical('alpha_over_rank', [1, 2]),
                     dropout=trial.suggest_float('dropout', 0.0, 0.1),
                     targets=trial.suggest_categorical('targets', list(TARGETS)))
            val, col, info = fit_config(p, trial.number)
            trial.set_user_attr('mae_2026_posthoc', mae(variable, col))
            for k, v in info.items(): trial.set_user_attr(k, v)
            np.save(coldir / f'{trial.number}.npy', col)
            return val
        if '--finalize' not in sys.argv:
            run_study(name, objective, OUT / 'studies', QUICK)
            continue
        import optuna
        study = optuna.load_study(study_name=name, storage='sqlite:///' + str(OUT / 'studies' / f'{name}.sqlite3'))
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not QUICK and len(complete) < PROTOCOL['min_completed_trials']: raise RuntimeError(f'{name}: only {len(complete)} completed trials')
        best_t = study.best_trial
        cols = {int(f.stem): np.load(f) for f in coldir.glob('*.npy')}
        sel = {'study': name, 'protocol': PROTOCOL['version'], 'completed_trials': len(complete),
               'failed_trials': len(study.trials) - len(complete), 'best_trial': best_t.number,
               'best_value': best_t.value, 'params': best_t.params, 'smoke_only': QUICK}
        if best_t.number not in cols:
            raise RuntimeError(f'{name}: the selected trial {best_t.number} has no saved column')
        curve = pd.DataFrame([{'trial': t.number, **t.params, 'val_loss': t.value, **t.user_attrs} for t in complete])
        curve.to_csv(OUT / f'{name}_curve.csv', index=False)
        np.savez_compressed(OUT / f'{name}_2026_columns.npz', **{str(k): v for k, v in cols.items()})
        cols_out[f'V2LORA_{variable}_{space}'] = cols[best_t.number]
        rho = lambda a, b: float(curve[[a, b]].corr(method='spearman').iloc[0, 1]) if len(curve) > 2 else None
        best_post = curve.loc[curve.mae_2026_posthoc.idxmin()]
        report[name] = {'selection': sel, 'selected_mae_2026': float(curve.set_index('trial').loc[best_t.number, 'mae_2026_posthoc']),
                        'zero_shot_mae_2026': zero_shot[variable],
                        'spearman_val_vs_2026': rho('val_loss', 'mae_2026_posthoc'),
                        'spearman_params_vs_2026': rho('trainable_params', 'mae_2026_posthoc'),
                        'n_better_than_zero_shot': int((curve.mae_2026_posthoc < zero_shot[variable]).sum()),
                        'best_posthoc': {'trial': int(best_post.trial), 'mae_2026': float(best_post.mae_2026_posthoc),
                                         'trainable_params': int(best_post.trainable_params)},
                        'mae_2026_range': [float(curve.mae_2026_posthoc.min()), float(curve.mae_2026_posthoc.max())],
                        'trainable_params_range': [int(curve.trainable_params.min()), int(curve.trainable_params.max())],
                        'note': 'Selected on pre-2026 validation loss only; 2026 errors of other trials are post-hoc diagnostics.'}
        print(json.dumps(report[name], indent=2, default=str), flush=True)
    if '--finalize' not in sys.argv: return
    ep.assign(**cols_out).to_csv(OUT / 'v2_chronos_lora_predictions.csv.gz', index=False)
    (OUT / 'v2_chronos_lora_report.json').write_text(json.dumps(
        {'declared_in': 'config/bay_reserved_protocol_v2_addendum40.json', 'protocol': PROTOCOL['version'],
         'smoke_only': QUICK, **report}, indent=2, default=str))


if __name__ == '__main__':
    main()
