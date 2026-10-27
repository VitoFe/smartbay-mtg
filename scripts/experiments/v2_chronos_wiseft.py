"""Addendum 77: interpolating the weights of zero-shot and adapted Chronos-Bolt (WiSE-FT).

For each addendum-42 selection (full fine-tune or low-rank adapter, temperature raw or oxygen as apparent
utilisation): rerun the selected trial exactly as v2_chronos_refit.rerun does, restoring its selected step
(gate GE1: the study's validation loss and 2026 error to 1e-4); then, for a in {0, 0.25, 0.5, 0.75, 1},
set the weights between the pretrained model (a = 0) and the adapted one (a = 1):

  full  theta(a) = theta_0 + a (theta_ft - theta_0), every weight
  lora  B scaled by a with A kept, so the update B A becomes a B A (B is zero at the pretrained end)

and record the held-out quarter's validation loss and the 2026 column at each a. The a with the lowest
validation loss is the selection (ties to the smaller a); 2026 is not used to choose it. Gate GE2: a = 1
reproduces the rerun's column exactly and a = 0 the zero-shot column to 1e-6.

  python v2_chronos_wiseft.py --job timeval_lora_oxygen_aou     # one selection (run from the a52/ folder)
  python v2_chronos_wiseft.py --finalize                        # combine the four
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_chronos_refit as rf                       # PARTS, SELECTIONS, rerun, loss_on, QUICK
tv, torch, BaseChronosPipeline = rf.tv, rf.torch, rf.BaseChronosPipeline
import pandas as pd

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
OUT = HERE / ('results_smoke' if rf.QUICK else 'results')
OUT.mkdir(exist_ok=True)


def validation_loss(model, w):
    vc, vtg, _ = w['val']
    model.eval()
    with torch.no_grad():
        return float(np.mean([rf.loss_on(model, vc, vtg, np.arange(i, min(i + 64, len(vc)))).item()
                              for i in range(0, len(vc), 64)]))


def same(a, b, tol):
    both = np.isfinite(a) & np.isfinite(b)
    return bool((np.isfinite(a) == np.isfinite(b)).all() and (np.abs(a[both] - b[both]).max() <= tol if both.any() else True))


def run(job):
    method, variable, space = rf.PARTS[job]
    sel = rf.SELECTIONS[job]
    params, number = sel['params'], sel['best_trial']
    eval_times = {v: {s: pd.DatetimeIndex(sorted(tv.ep.loc[(tv.ep.variable == v) & (tv.ep.sensor == s), 'time']))
                      for s in tv.sensors_for(v)} for v, _ in tv.TASKS}
    zs = BaseChronosPipeline.from_pretrained(tv.vt.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = tv.predict_at(zs, 'temperature', 'raw', eval_times['temperature'])
    col_zs = tv.to_column(zs, variable, space, eval_times[variable], temp_zs)
    theta0 = {n: p.detach().clone() for n, p in zs.model.named_parameters()}
    del zs
    torch.cuda.empty_cache()
    w = tv.windows(variable, space)
    pipe, best, step, _ = rf.rerun(method, params, w, number)
    model = pipe.model
    col_rerun = tv.to_column(pipe, variable, space, eval_times[variable], temp_zs)
    mae_rerun = tv.mae(variable, col_rerun)
    ge1 = {'val_loss_study': sel['best_value'], 'val_loss_rerun': best, 'mae_study': sel['selected_mae_2026'],
           'mae_rerun': mae_rerun,
           'passed': bool(abs(best - sel['best_value']) <= 1e-4 and abs(mae_rerun - sel['selected_mae_2026']) <= 1e-4)}
    ft = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rows, cols = {}, {}
    for a in ALPHAS:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in ft:
                    continue
                if a == 1.0:
                    p.copy_(ft[n])
                elif method == 'full':
                    p.copy_(theta0[n] + a * (ft[n] - theta0[n]))
                else:
                    p.copy_(ft[n] * a if n.endswith('lora_b') else ft[n])
        vl = validation_loss(model, w)
        col = tv.to_column(pipe, variable, space, eval_times[variable], temp_zs)
        rows[str(a)] = {'val_loss': vl, 'mae_2026': tv.mae(variable, col)}
        cols[str(a)] = col
        print(job, a, rows[str(a)], flush=True)
    chosen = min(ALPHAS, key=lambda a: (rows[str(a)]['val_loss'], a))
    ge2 = {'a1_equals_rerun': same(cols['1.0'], col_rerun, 0.0), 'a0_equals_zero_shot': same(cols['0.0'], col_zs, 1e-6)}
    ge2['passed'] = bool(ge2['a1_equals_rerun'] and ge2['a0_equals_zero_shot'])
    out = {'job': job, 'method': method, 'variable': variable, 'space': space, 'trial': number, 'params': params,
           'restored_step': step, 'GE1': ge1, 'GE2': ge2, 'by_alpha': rows, 'chosen_alpha': chosen,
           'mae_zero_shot_2026': tv.mae(variable, col_zs), 'smoke_only': rf.QUICK}
    np.savez(OUT / f'wiseft_{job}.npz', zero_shot=col_zs, **{f'a{k}': v for k, v in cols.items()})
    (OUT / f'wiseft_{job}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def finalize():
    cols, report = {}, {}
    for job, (method, variable, space) in rf.PARTS.items():
        rec = json.loads((OUT / f'wiseft_{job}.json').read_text())
        z = np.load(OUT / f'wiseft_{job}.npz')
        cols[f'WISE_{method}_{variable}_{space}'] = z[f"a{rec['chosen_alpha']}"]
        for a in ALPHAS:
            cols[f'WISE{a}_{method}_{variable}_{space}'] = z[f'a{a}']
        report[job] = rec
    tv.ep.assign(**cols).to_csv(OUT / 'v2_chronos_wiseft_predictions.csv.gz', index=False)
    (OUT / 'v2_chronos_wiseft_report.json').write_text(json.dumps(
        {'declared_in': 'config/bay_reserved_protocol_v2_addendum77.json', 'smoke_only': rf.QUICK, **report}, indent=2))
    print('finalized', sorted(cols))


if __name__ == '__main__':
    if '--finalize' in sys.argv:
        finalize()
    else:
        run(sys.argv[sys.argv.index('--job') + 1])
