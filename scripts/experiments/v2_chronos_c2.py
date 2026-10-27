"""Tuning protocol v2 for the Chronos residual corrector (C2) on the zero-shot arms (Table 5.5).

v1 (exp_chronos.fit_c2) fitted RidgeCV over a 30-point alpha grid with leave-one-out CV. v2 keeps
the same design, the same pre-2026 issuance sample and the same 2026 application, and selects alpha
with a 100-trial TPE study on a chronological inner split (latest 25 % of the pre-2026 samples).
Zero-shot arms only: the corrector on fine-tuned arms is not reported.
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import exp_chronos as xc
from src.tuning.protocol_v2 import run_study, PROTOCOL

OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)


def samples(pipe, variable, space):
    """exp_chronos.fit_c2 data assembly, returned with issuance times for a chronological split."""
    by = xc.pre2026_issuance(variable)
    by = {s: (t[xc.RNG.choice(len(t), min(len(t), 1500), replace=False)] if len(t) > 1500 else t) for s, t in by.items() if len(t) > 0}
    preds = xc.predict_at(pipe, variable, space, by); X, y, when = [], [], []
    for s, times in by.items():
        ser = xc.obs[f'{s}::{variable}']; keep = [t for t in times if (s, t) in preds]
        if not keep: continue
        D = xc.c2_design(s, variable, keep)
        for i, t in enumerate(keep):
            truth = ser.get(t + pd.Timedelta(hours=xc.H), np.nan); p = preds[(s, t)]
            if space == 'aou': truth = xc.o2_sol(xc.obs[f'{s}::temperature'].get(t + pd.Timedelta(hours=xc.H), np.nan)) - truth
            if not np.isfinite(truth) or not np.isfinite(D[i]).all(): continue
            X.append(D[i]); y.append(truth - p); when.append(t)
    return np.asarray(X), np.asarray(y), pd.DatetimeIndex(when)


def main():
    ep = xc.ep
    eval_times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), 'time'])) for s in xc.sensors_for(v)} for v in ('temperature', 'oxygen')}
    zs = BaseChronosPipeline.from_pretrained(xc.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = xc.predict_at(zs, 'temperature', 'raw', eval_times['temperature']); report = {}
    for variable, space in [('temperature', 'raw'), ('oxygen', 'raw'), ('oxygen', 'aou')]:
        X, y, when = samples(zs, variable, space)
        order = np.argsort(when.asi8); cut = int(len(order) * .75); itr, iva = order[:cut], order[cut:]

        def objective(trial):
            m = make_pipeline(StandardScaler(), Ridge(alpha=trial.suggest_float('alpha', 1e-3, 1e6, log=True))).fit(X[itr], y[itr])
            return float(np.abs(m.predict(X[iva]) - y[iva]).mean())
        name = f'c2_{variable}_{space}'; sel = run_study(name, objective, OUT / 'studies')
        c2 = make_pipeline(StandardScaler(), Ridge(alpha=sel['params']['alpha'])).fit(X, y)
        base = xc.predict_at(zs, variable, space, eval_times[variable]); col = np.full(len(ep), np.nan)
        for i, r in ep.iterrows():
            if r.variable != variable or (r.sensor, r.time) not in base: continue
            v = base[(r.sensor, r.time)]; D = xc.c2_design(r.sensor, variable, [r.time])
            if np.isfinite(D).all(): v = v + float(c2.predict(D)[0])
            if space == 'aou':
                T = temp_zs.get((r.sensor, r.time), np.nan)
                if not np.isfinite(T): continue
                v = xc.o2_sol(T) - v
            col[i] = v
        ep[f'V2C2_{variable}_{space}'] = col
        report[name] = {'selection': sel, 'samples': len(y), 'inner_validation': len(iva), 'mae_2026': xc.macro_mae(ep[ep.variable == variable], f'V2C2_{variable}_{space}')}
        print(json.dumps(report[name], indent=2, default=str), flush=True)
    ep.to_csv(OUT / 'v2_chronos_c2_predictions.csv.gz', index=False)
    (OUT / 'v2_chronos_c2_report.json').write_text(json.dumps({'protocol': PROTOCOL['version'], **report}, indent=2, default=str))


if __name__ == '__main__':
    main()
