"""Addendum 43: refit the addendum-42 selections on the whole pre-2026 record.

For each selected configuration: rerun the selected trial exactly as its study ran it, logging the
validation curve, to recover the step its restored checkpoint came from (gate G1: the rerun must
reproduce the trial's validation loss and 2026 error); then train the same configuration on every
window of the pre-2026 record, the held-out quarter included, for exactly that many steps, and
score it on Reserved-2026. Windows, adapters and scoring come from v2_chronos_timeval, imported
unchanged from the same snapshot directory.

  python v2_chronos_refit.py --job timeval_lora_temperature_raw     # one selection
  python v2_chronos_refit.py --finalize                             # combine the four
  add --quick for a smoke run (20 steps) into results_smoke/
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_chronos_timeval as tv                     # windows, attach, to_column, mae, ep, vt, torch
torch, vt, BaseChronosPipeline = tv.torch, tv.vt, tv.BaseChronosPipeline

QUICK = '--quick' in sys.argv
OUT = HERE / ('results_smoke' if QUICK else 'results'); OUT.mkdir(exist_ok=True)
SELECTIONS = json.loads((HERE / 'selections.json').read_text())
PARTS = {'timeval_full_temperature_raw': ('full', 'temperature', 'raw'),
         'timeval_lora_temperature_raw': ('lora', 'temperature', 'raw'),
         'timeval_full_oxygen_aou': ('full', 'oxygen', 'aou'),
         'timeval_lora_oxygen_aou': ('lora', 'oxygen', 'aou')}


def build(method, params, number):
    """The start of v2_chronos_timeval.train, in its order: load, seed torch, wrap, optimiser."""
    pipe = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    torch.manual_seed(number)
    model = pipe.model
    if method == 'lora':
        tv.attach(model, params['targets'], params['rank'], params['alpha_over_rank'] * params['rank'], params['dropout'])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=params['lr'], weight_decay=params['weight_decay'])
    return pipe, model, trainable, opt


def loss_on(model, ctxs, tgts, sel):
    c = torch.tensor(ctxs[sel], device='cuda'); t = torch.tensor(tgts[sel], device='cuda')
    return model(context=torch.nan_to_num(c), mask=torch.isfinite(c), target=t, target_mask=torch.isfinite(t)).loss


def rerun(method, params, w, number):
    """v2_chronos_timeval.train line for line, with the validation curve and the restored step kept."""
    pipe, model, trainable, opt = build(method, params, number)
    tc, tt, _ = w['train']; vc, vtg, _ = w['val']
    ti = np.arange(len(tc)); rng = np.random.default_rng(number)

    def validate():
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(model, vc, vtg, np.arange(i, min(i + 64, len(vc)))).item() for i in range(0, len(vc), 64)]))
        model.train()
        return vl
    model.train()
    best = validate(); best_step, curve = 0, [[0, best]]
    best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    patience = 0
    for step in range(20 if QUICK else params['max_steps']):
        l = loss_on(model, tc, tt, rng.choice(ti, params['batch'], replace=False)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if (step + 1) % (10 if QUICK else 100) == 0:
            vl = validate(); curve.append([step + 1, vl])
            if vl < best - 1e-4:
                best, patience, best_step = vl, 0, step + 1
                best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            else:
                patience += 1
                if patience >= 4: break
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state: p.copy_(best_state[n])
    model.eval()
    return pipe, best, best_step, curve


def refit(method, params, w, number, steps):
    """The same configuration on every window, the held-out quarter included, for a fixed number of steps."""
    pipe, model, trainable, opt = build(method, params, number)
    ac = np.concatenate([w['train'][0], w['val'][0]]); at = np.concatenate([w['train'][1], w['val'][1]])
    ai = np.arange(len(ac)); rng = np.random.default_rng(number)
    model.train()
    for _ in range(steps):
        l = loss_on(model, ac, at, rng.choice(ai, params['batch'], replace=False)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    model.eval()
    return pipe, int(len(ac))


def run(job):
    method, variable, space = PARTS[job]
    sel = SELECTIONS[job]
    params, number = sel['params'], sel['best_trial']
    eval_times = {v: {s: pd.DatetimeIndex(sorted(tv.ep.loc[(tv.ep.variable == v) & (tv.ep.sensor == s), 'time']))
                      for s in tv.sensors_for(v)} for v, _ in tv.TASKS}
    zs = BaseChronosPipeline.from_pretrained(vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = tv.predict_at(zs, 'temperature', 'raw', eval_times['temperature']); del zs; torch.cuda.empty_cache()
    w = tv.windows(variable, space)
    split = tv.split_gate(w)
    pipe, best, step, curve = rerun(method, params, w, number)
    col_rerun = tv.to_column(pipe, variable, space, eval_times[variable], temp_zs); del pipe; torch.cuda.empty_cache()
    mae_rerun = tv.mae(variable, col_rerun)
    g1 = {'val_loss_study': sel['best_value'], 'val_loss_rerun': best, 'mae_study': sel['selected_mae_2026'],
          'mae_rerun': mae_rerun, 'passed': bool(abs(best - sel['best_value']) <= 1e-4 and abs(mae_rerun - sel['selected_mae_2026']) <= 1e-4)}
    steps = max(step, 1) if QUICK else step
    pipe, n_windows = refit(method, params, w, number, steps)
    col = tv.to_column(pipe, variable, space, eval_times[variable], temp_zs); del pipe; torch.cuda.empty_cache()
    out = {'job': job, 'trial': number, 'params': params, 'restored_step': step, 'curve': curve, 'G1': g1,
           'G2': {'passed': bool(split['passed'] and split['val_last_target'] < '2026'), **split},
           'refit_windows': n_windows, 'refit_steps': steps, 'mae_refit_2026': tv.mae(variable, col),
           'zero_shot_mae_2026': None}
    np.save(OUT / f'refit_{job}.npy', col)
    (OUT / f'refit_{job}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'curve'}, indent=2), flush=True)


def finalize():
    cols, report = {}, {}
    for job, (method, variable, space) in PARTS.items():
        rec = json.loads((OUT / f'refit_{job}.json').read_text())
        cols[f'V2RF_{method}_{variable}_{space}'] = np.load(OUT / f'refit_{job}.npy')
        report[job] = {k: v for k, v in rec.items() if k != 'curve'}
    tv.ep.assign(**cols).to_csv(OUT / 'v2_chronos_refit_predictions.csv.gz', index=False)
    (OUT / 'v2_chronos_refit_report.json').write_text(json.dumps(
        {'declared_in': 'config/bay_reserved_protocol_v2_addendum43.json', 'smoke_only': QUICK, **report}, indent=2))
    print('finalized', list(cols))


if __name__ == '__main__':
    if '--finalize' in sys.argv:
        finalize()
    else:
        run(sys.argv[sys.argv.index('--job') + 1])
