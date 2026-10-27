"""Addendum 42: select the adaptation of Chronos-Bolt on a validation block held out in time.

The six-hour fine-tuning (v2_chronos.py) and adapter (v2_chronos_lora.py) studies selected on
windows drawn at random from the same record they trained on, and the validation loss ranked the
2026 error backwards. Here the windows are split by time instead: validation windows have all six
target hours in 2025-10-01 to 2025-12-31, training windows end before 2025-10-01, and windows across
the boundary are dropped. The pretrained model is the step-0 checkpoint, so a trial that never
improves on it restores the untouched model. Everything else is the earlier studies' protocol.

  python v2_chronos_timeval.py --gates                              # G1 (split) and G2 (zero-shot)
  python v2_chronos_timeval.py --only temperature --method lora     # a study worker
  python v2_chronos_timeval.py --finalize                           # all four studies, outputs
  add --quick for a smoke run into results_smoke/

Runs from a snapshot directory on Trinity with code/ (exp_chronos.py, exp_chronos_valtest.py),
src/tuning/protocol_v2.py and config/tuning_protocol_v2.json beside it.
"""
import json, math, os, sys, warnings, zlib
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
from exp_chronos import o2_sol, ep, sensors_for, series_for, predict_at, macro_mae
from src.tuning.protocol_v2 import run_study, PROTOCOL

QUICK = '--quick' in sys.argv
ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
METHOD = sys.argv[sys.argv.index('--method') + 1] if '--method' in sys.argv else None
OUT = HERE / ('results_smoke' if QUICK else 'results'); OUT.mkdir(exist_ok=True)
TASKS = [('temperature', 'raw'), ('oxygen', 'aou')]
METHODS = ('full', 'lora')
VAL_START = pd.Timestamp('2025-10-01', tz='UTC')
CUT = pd.Timestamp('2026-01-01', tz='UTC')
N_WINDOWS = 24000
SEED = 20260924
TARGETS = {'qv': ('q', 'v'), 'qkvo': ('q', 'k', 'v', 'o'), 'qkvo_ffn': ('q', 'k', 'v', 'o', 'wi', 'wo')}
PUBLISHED_ZERO_SHOT = {'temperature': 0.1724, 'oxygen': 5.3911}      # battery_full_v2.json, 4 decimals


class LoRALinear(torch.nn.Module):
    """W x + (alpha / r) B A dropout(x), with W frozen and B = 0 at the start (addendum 40)."""

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
    for p in model.parameters():
        p.requires_grad_(False)
    names, n = TARGETS[targets], 0
    for stack in (model.encoder, model.decoder):
        for parent in list(stack.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in names and isinstance(child, torch.nn.Linear):
                    setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout)); n += 1
    return n


def windows(variable, space):
    """vt.windows' sampling rule, with the split made by time. Depends on the variable only, so both
    methods train and validate on the same windows."""
    rng = np.random.default_rng([SEED, zlib.crc32(variable.encode())])
    ss = sensors_for(variable)
    out = {'train': ([], [], []), 'val': ([], [], [])}
    for s in ss:
        ser = series_for(s, variable, space)
        if ser is None:
            continue
        ser = ser[ser.index < CUT]
        v, idx = ser.to_numpy(float), ser.index
        if len(v) < vt.CTX + vt.H + 24:
            continue
        ends = np.arange(vt.CTX, len(v) - vt.H)
        take = min(len(ends), max(1, N_WINDOWS // max(1, len(ss))))
        for e in rng.choice(ends, take, replace=False):
            c, t = v[e - vt.CTX:e], v[e:e + vt.H]
            if not (np.isfinite(c).sum() >= 96 and np.isfinite(t).all()):
                continue
            first, last = idx[e], idx[e + vt.H - 1]
            part = 'val' if first >= VAL_START else 'train' if last < VAL_START else None
            if part:
                out[part][0].append(c); out[part][1].append(t); out[part][2].append(last.value)
    arr = {k: (np.asarray(c, np.float32), np.asarray(t, np.float32), np.asarray(e, np.int64)) for k, (c, t, e) in out.items()}
    return arr


def split_gate(w):
    tr_last, va_last = w['train'][2], w['val'][2]
    va_first = va_last - (vt.H - 1) * 3600 * 10 ** 9
    ok = bool(len(tr_last) and len(va_last) and tr_last.max() < VAL_START.value
              and va_first.min() >= VAL_START.value and va_last.max() < CUT.value)
    return {'passed': ok, 'n_train': int(len(tr_last)), 'n_val': int(len(va_last)),
            'train_last_target': str(pd.Timestamp(int(tr_last.max()), tz='UTC')),
            'val_first_target': str(pd.Timestamp(int(va_first.min()), tz='UTC')),
            'val_last_target': str(pd.Timestamp(int(va_last.max()), tz='UTC'))}


def train(method, params, w, rng, seed):
    pipe = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    torch.manual_seed(seed)
    model = pipe.model
    if method == 'lora':
        attach(model, params['targets'], params['rank'], params['alpha_over_rank'] * params['rank'], params['dropout'])
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in trainable))
    opt = torch.optim.AdamW(trainable, lr=params['lr'], weight_decay=params['weight_decay'])
    tc, tt, _ = w['train']; vc, vtg, _ = w['val']
    ti = np.arange(len(tc))

    def loss_on(ctxs, tgts, sel):
        c = torch.tensor(ctxs[sel], device='cuda'); t = torch.tensor(tgts[sel], device='cuda')
        return model(context=torch.nan_to_num(c), mask=torch.isfinite(c), target=t, target_mask=torch.isfinite(t)).loss

    def validate():
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(vc, vtg, np.arange(i, min(i + 64, len(vc)))).item() for i in range(0, len(vc), 64)]))
        model.train()
        if not np.isfinite(vl): raise ValueError('non-finite validation loss')
        return vl
    model.train()
    best = step0 = validate()                                   # the pretrained model is a candidate
    best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    improved, patience, stopped = False, 0, None
    for step in range(20 if QUICK else params['max_steps']):
        l = loss_on(tc, tt, rng.choice(ti, params['batch'], replace=False)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if (step + 1) % (10 if QUICK else 100) == 0:
            vl = validate()
            if vl < best - 1e-4:
                best, patience, improved = vl, 0, True
                best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            else:
                patience += 1
                if patience >= 4: stopped = step + 1; break
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state: p.copy_(best_state[n])
    model.eval()
    return pipe, best, {'trainable_params': n_trainable, 'stopped_at': stopped, 'val_loss_step0': step0,
                        'zero_shot_restored': not improved}


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


def mae(variable, col):
    return float(macro_mae(ep.assign(_c=col)[ep.variable == variable], '_c'))


def space_for(method, trial):
    p = dict(max_steps=trial.suggest_categorical('max_steps', [400, 800, 1500, 2500]),
             weight_decay=trial.suggest_float('weight_decay', 1e-5, .3, log=True),
             batch=trial.suggest_categorical('batch', [16, 32, 64]))
    if method == 'full':      # the search space of v2_chronos.py
        p['lr'] = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    else:                     # the search space of addendum 40
        p.update(lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                 rank=trial.suggest_categorical('rank', [1, 2, 4, 8, 16, 32]),
                 alpha_over_rank=trial.suggest_categorical('alpha_over_rank', [1, 2]),
                 dropout=trial.suggest_float('dropout', 0.0, 0.1),
                 targets=trial.suggest_categorical('targets', list(TARGETS)))
    return p


def main():
    eval_times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), 'time'])) for s in sensors_for(v)} for v, _ in TASKS}
    zs = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = predict_at(zs, 'temperature', 'raw', eval_times['temperature'])
    zero_shot = {v: mae(v, to_column(zs, v, s, eval_times[v], temp_zs)) for v, s in TASKS}
    del zs; torch.cuda.empty_cache()
    if '--gates' in sys.argv:
        g = {'G1': {v: split_gate(windows(v, s)) for v, s in TASKS},
             'G2': {v: {'mae': zero_shot[v], 'published': PUBLISHED_ZERO_SHOT[v],
                        'passed': abs(zero_shot[v] - PUBLISHED_ZERO_SHOT[v]) <= 1e-4} for v, _ in TASKS}}
        g['passed'] = all(x['passed'] for x in g['G1'].values()) and all(x['passed'] for x in g['G2'].values())
        (OUT / 'gates.json').write_text(json.dumps(g, indent=2)); print(json.dumps(g, indent=2), flush=True)
        raise SystemExit(0 if g['passed'] else 1)
    report, cols_out = {}, {}
    for variable, space in TASKS:
        if ONLY and variable != ONLY: continue
        w = windows(variable, space)
        assert split_gate(w)['passed'], 'the time split failed its gate'
        for method in METHODS:
            if METHOD and method != METHOD: continue
            name = f'timeval_{method}_{variable}_{space}'
            coldir = OUT / 'columns' / name; coldir.mkdir(parents=True, exist_ok=True)

            def objective(trial, method=method, variable=variable, space=space, w=w, coldir=coldir):
                p = space_for(method, trial)
                pipe, val, info = train(method, p, w, np.random.default_rng(trial.number), trial.number)
                col = to_column(pipe, variable, space, eval_times[variable], temp_zs); del pipe; torch.cuda.empty_cache()
                trial.set_user_attr('mae_2026_posthoc', mae(variable, col))
                for k, x in info.items(): trial.set_user_attr(k, x)
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
            if best_t.number not in cols: raise RuntimeError(f'{name}: the selected trial has no saved column')
            curve = pd.DataFrame([{'trial': t.number, **t.params, 'val_loss': t.value, **t.user_attrs} for t in complete])
            curve.to_csv(OUT / f'{name}_curve.csv', index=False)
            np.savez_compressed(OUT / f'{name}_2026_columns.npz', **{str(k): x for k, x in cols.items()})
            cols_out[f'V2TV_{method}_{variable}_{space}'] = cols[best_t.number]
            rho = lambda a, b: float(curve[[a, b]].corr(method='spearman').iloc[0, 1]) if len(curve) > 2 else None
            best_post = curve.loc[curve.mae_2026_posthoc.idxmin()]
            report[name] = {
                'selection': {'study': name, 'protocol': PROTOCOL['version'], 'completed_trials': len(complete),
                              'failed_trials': len(study.trials) - len(complete), 'best_trial': best_t.number,
                              'best_value': best_t.value, 'params': best_t.params, 'smoke_only': QUICK,
                              'zero_shot_restored': bool(best_t.user_attrs.get('zero_shot_restored')),
                              'trainable_params': int(best_t.user_attrs.get('trainable_params', 0))},
                'windows': split_gate(w),
                'selected_mae_2026': float(curve.set_index('trial').loc[best_t.number, 'mae_2026_posthoc']),
                'zero_shot_mae_2026': zero_shot[variable],
                'spearman_val_vs_2026': rho('val_loss', 'mae_2026_posthoc'),
                'n_better_than_zero_shot': int((curve.mae_2026_posthoc < zero_shot[variable]).sum()),
                'n_zero_shot_restored': int(curve.zero_shot_restored.astype(bool).sum()),
                'best_posthoc': {'trial': int(best_post.trial), 'mae_2026': float(best_post.mae_2026_posthoc)},
                'mae_2026_range': [float(curve.mae_2026_posthoc.min()), float(curve.mae_2026_posthoc.max())],
                'note': 'Selected on the held-out 2025-Q4 validation loss only; 2026 errors are post-hoc diagnostics.'}
            print(json.dumps(report[name], indent=2, default=str), flush=True)
    if '--finalize' not in sys.argv: return
    ep.assign(**cols_out).to_csv(OUT / 'v2_chronos_timeval_predictions.csv.gz', index=False)
    (OUT / 'v2_chronos_timeval_report.json').write_text(json.dumps(
        {'declared_in': 'config/bay_reserved_protocol_v2_addendum42.json', 'protocol': PROTOCOL['version'],
         'validation_block': [str(VAL_START), str(CUT)], 'smoke_only': QUICK, **report}, indent=2, default=str))


if __name__ == '__main__':
    main()
