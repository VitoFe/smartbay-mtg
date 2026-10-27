"""Addendum 60 part C3, on the compute host: the adapted pretrained utilisation arms in mg/l.

The six adapted arms of the reserved-period table (fine-tuned and low-rank adapted Chronos-Bolt,
random split, best transfer and held-out quarter, with the held-out-quarter refits) are retrained
exactly as scripts/experiments/v2_aou_units_chronos.py.adapted retrains them, at the configuration
and seed their studies chose, once per mode:

  published   the original inputs and the per-litre saturation (unit 'l' of addendum 52, the columns
              the thesis prints): must reproduce runs/v2_aou_units_chronos/adapted_<arm>_columns.csv.gz
  mgl         the sondes in mg/l hour by hour and the saturation in mg/l (per-litre form x 0.0319988,
              T clipped to [-2, 40], S = 38)

No study is run. Writes results/adapted_<arm>_<mode>_columns.csv.gz and _report.json.

  python v2_oxygen_mgl_chronos_adapted.py <arm> published|mgl
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
ARM, MODE = sys.argv[1], sys.argv[2]
if MODE not in ('published', 'mgl'):
    raise SystemExit('<arm> published|mgl')
os.environ['SMARTBAY_BASE'] = str(HERE / f'base_{MODE}')
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'code'))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get('V2_THREADS', '2')))
import exp_chronos as xc  # noqa: E402
import v2_aou_units_chronos as au  # noqa: E402
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l  # noqa: E402

MG_PER_UMOL_L = 0.0319988
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def o2_sol(T, S=xc.SAL):
    T = np.clip(np.asarray(T, float), -2, 40)
    sat = o2sol_garcia_gordon_umol_per_l(T, S)
    return sat * MG_PER_UMOL_L if MODE == 'mgl' else sat


def main():
    import exp_chronos_valtest as vt
    import v2_chronos as v2c
    import v2_chronos_lora as lora
    import v2_chronos_timeval as tv
    import v2_chronos_refit as rf
    from chronos import BaseChronosPipeline
    if ARM not in au.ADAPTED:
        raise SystemExit(f'unknown arm {ARM}')
    REF = au.REF
    t0 = time.time()
    ep = xc.ep
    times = au.eval_times(ep)
    zs = BaseChronosPipeline.from_pretrained(xc.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = xc.predict_at(zs, 'temperature', 'raw', times['temperature'])
    del zs
    torch.cuda.empty_cache()
    report = {'declared_in': 'config/bay_reserved_protocol_v2_addendum60.json', 'arm': ARM, 'mode': MODE}
    if ARM.startswith('ft_'):
        rep = json.loads((REF / 'v2_chronos_report_oxygen.json').read_text())['chronos_oxygen_aou']
        curve = pd.read_csv(REF / 'chronos_oxygen_aou_curve.csv')
        if ARM == 'ft_random':
            number, params = rep['selection']['best_trial'], rep['selection']['params']
        else:
            row = curve.loc[curve.mae_2026_posthoc.idxmin()]
            number = int(row.trial)
            params = {'lr': float(row.lr), 'max_steps': int(row.max_steps), 'weight_decay': float(row.weight_decay),
                      'batch': int(row.batch)}
    elif ARM.startswith('lora_'):
        rep = json.loads((REF / 'v2_chronos_lora_report.json').read_text())['lora_oxygen_aou']
        number = rep['selection']['best_trial'] if ARM == 'lora_random' else rep['best_posthoc']['trial']
        curve = pd.read_csv(REF / 'lora_oxygen_aou_curve.csv').set_index('trial')
        if ARM == 'lora_random':
            params = dict(rep['selection']['params'])
        else:
            keys = ('lr', 'max_steps', 'weight_decay', 'batch', 'rank', 'alpha_over_rank', 'dropout', 'targets')
            params = {k: curve.loc[number, k] for k in keys}
            params = {k: (int(v) if k in ('max_steps', 'batch', 'rank', 'alpha_over_rank') else
                          (str(v) if k == 'targets' else float(v))) for k, v in params.items()}
    else:
        method = ARM.removeprefix('tv_')
        sel = json.loads((HERE / 'selections.json').read_text())[f'timeval_{method}_oxygen_aou']
        number, params = sel['best_trial'], sel['params']
    report.update(trial=int(number), params=params)
    for m in (xc, vt, v2c, lora, tv, rf):
        if hasattr(m, 'o2_sol'):
            m.o2_sol = o2_sol
    frame = ep[['variable', 'sensor', 'time']].copy()
    if ARM.startswith('ft_'):
        vt.RNG = np.random.default_rng(20260919)
        w = {}
        for variable, space in (('temperature', 'raw'), ('oxygen', 'aou')):
            ctxs, tgts = vt.windows(variable, space)
            n_val = max(128, int(.1 * len(ctxs)))
            idx = vt.RNG.permutation(len(ctxs))
            w[variable] = (ctxs, tgts, idx[n_val:], idx[:n_val])
        ctxs, tgts, ti, vi = w['oxygen']
        torch.manual_seed(int(number))
        pipe, val = v2c.train(ctxs, tgts, ti, vi, rng=np.random.default_rng(int(number)), **params)
        col = v2c.to_column(pipe, 'oxygen', 'aou', times['oxygen'], temp_zs)
        report['val_loss'] = val
    elif ARM.startswith('lora_'):
        vt.RNG = np.random.default_rng(20260919)
        ctxs, tgts, ti, vi = lora.draw_windows()['oxygen']
        pipe, val, info = lora.train(ctxs, tgts, ti, vi, rng=np.random.default_rng(int(number)), seed=int(number), **params)
        col = lora.to_column(pipe, 'oxygen', 'aou', times['oxygen'], temp_zs)
        report.update(val_loss=val, **info)
    else:
        w = tv.windows('oxygen', 'aou')
        pipe, best, step, curve_ = rf.rerun(method, params, w, int(number))
        col = tv.to_column(pipe, 'oxygen', 'aou', times['oxygen'], temp_zs)
        del pipe
        torch.cuda.empty_cache()
        pipe_r, n_windows = rf.refit(method, params, w, int(number), step)
        frame[f'refit_{method}|{MODE}'] = tv.to_column(pipe_r, 'oxygen', 'aou', times['oxygen'], temp_zs)
        del pipe_r
        torch.cuda.empty_cache()
        report.update(val_loss=best, restored_step=step, refit_windows=n_windows)
        pipe = None
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
    frame[f'{ARM}|{MODE}'] = col
    if MODE == 'published':
        ref = pd.read_csv(HERE / f'ref/adapted_{ARM}_columns.csv.gz')
        gate = {}
        for k in [c for c in frame.columns if c.endswith('|published')]:
            want = k.replace('|published', '|l')
            a, b = frame[k].to_numpy(float), ref[want].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            gate[k] = {'reference': want, 'max_abs_diff': float(np.max(np.abs(a[both] - b[both]))),
                       'same_missing': bool((np.isfinite(a) == np.isfinite(b)).all())}
            gate[k]['passed'] = bool(gate[k]['max_abs_diff'] <= 1e-3 and gate[k]['same_missing'])
        report['gate'] = gate
        report['passed'] = all(g['passed'] for g in gate.values())
    report['seconds'] = round(time.time() - t0, 1)
    frame.to_csv(OUT / f'adapted_{ARM}_{MODE}_columns.csv.gz', index=False)
    (OUT / f'adapted_{ARM}_{MODE}_report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str), flush=True)


if __name__ == '__main__':
    main()
