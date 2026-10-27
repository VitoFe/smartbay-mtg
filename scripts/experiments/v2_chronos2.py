"""Addendum-1: Chronos-2 zero-shot on the Reserved-2026 endpoints, univariate and with past covariates.

Declared in config/bay_reserved_protocol_v2_addendum.json before scoring. Same endpoints, 512 h context
and 6 h horizon as the Chronos-Bolt study (exp_chronos.predict_at); the median at hour 6 is the point
forecast. Covariates are past-only (up to issuance): network mean temperature, cross-probe spread,
and for oxygen the same sensor's temperature. No hyperparameters, no fine-tuning, no future values.
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
import exp_chronos as xc

OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
REPO = 'amazon/chronos-2'
QUICK = '--quick' in sys.argv


def window(series, t):
    return series.loc[:t].to_numpy(float)[-xc.CTX:]


def covs(sensor, variable, t, n):
    c = {'net_T': xc.NET.net_T, 'net_sd': xc.NET.net_sd}
    if variable == 'oxygen' and f'{sensor}::temperature' in xc.obs: c['own_T'] = xc.obs[f'{sensor}::temperature']
    out = {}
    for k, s in c.items():
        v = pd.Series(window(s, t)[-n:]).ffill().bfill().to_numpy(float)
        out[k] = np.nan_to_num(v, nan=float(np.nanmean(s.loc[:t].to_numpy(float)[-24 * 30:])) if np.isfinite(s.loc[:t]).any() else 0.)
    return out


def forecast(pipe, variable, space, use_cov, times_by_sensor):
    q = list(pipe.quantiles); qi = int(np.argmin(np.abs(np.asarray(q) - .5))); res = {}
    for s, times in times_by_sensor.items():
        ser = xc.series_for(s, variable, space)
        if ser is None: continue
        items, keys = [], []
        for t in times:
            hist = window(ser, t)
            if np.isfinite(hist).sum() < 48: continue
            it = {'target': hist.astype(np.float32)}
            if use_cov: it['past_covariates'] = {k: v.astype(np.float32) for k, v in covs(s, variable, t, len(hist)).items()}
            items.append(it); keys.append((s, t))
        for k in range(0, len(items), 128):
            out = pipe.predict(items[k:k + 128], prediction_length=xc.H)
            for j, o in enumerate(out):
                a = o.detach().cpu().float().numpy() if torch.is_tensor(o) else np.asarray(o)
                res[keys[k + j]] = float(a.reshape(-1, len(q), a.shape[-1])[0, qi, xc.H - 1])
    return res


def main():
    ep = xc.ep
    if QUICK: ep = ep.groupby('variable').head(40).reset_index(drop=True)
    times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), 'time'])) for s in xc.sensors_for(v)} for v in ('temperature', 'oxygen')}
    pipe = BaseChronosPipeline.from_pretrained(REPO, device_map='cuda', torch_dtype=torch.float32)
    report = {'model': REPO, 'pipeline': type(pipe).__name__, 'quantiles': list(pipe.quantiles), 'smoke_only': QUICK}
    for tag, use_cov in [('uni', False), ('cov', True)]:
        temp = forecast(pipe, 'temperature', 'raw', use_cov, times['temperature'])
        for variable, space in [('temperature', 'raw'), ('oxygen', 'raw'), ('oxygen', 'aou')]:
            pred = temp if variable == 'temperature' else forecast(pipe, variable, space, use_cov, times[variable])
            col = np.full(len(ep), np.nan)
            for i, r in ep.iterrows():
                k = (r.sensor, r.time)
                if r.variable != variable or k not in pred: continue
                v = pred[k]
                if space == 'aou':
                    T = temp.get(k, np.nan)
                    if not np.isfinite(T): continue
                    v = xc.o2_sol(T) - v
                col[i] = v
            name = f'CH2_{variable}_{space}_{tag}'; ep[name] = col
            report[name] = {'predictions': int(np.isfinite(col).sum()), 'mae_2026': xc.macro_mae(ep[ep.variable == variable], name)}
            print(name, report[name], flush=True)
    ep.to_csv(OUT / ('v2_chronos2_predictions' + ('_smoke' if QUICK else '') + '.csv.gz'), index=False)
    (OUT / ('v2_chronos2_report' + ('_smoke' if QUICK else '') + '.json')).write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
