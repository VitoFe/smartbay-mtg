"""Addendum 60 part C3, on the compute host: the zero-shot pretrained utilisation arms in mg/l.

The published utilisation arms of the reserved-period table (v2_aou_units_chronos.py, unit 'l')
subtract the sondes' per-kilogram oxygen from the Garcia-Gordon per-litre saturation at S = 38, so
the series Chronos reads mixes the two. This driver runs the same zero-shot code path twice, with no
study run again (the C2 corrector keeps the penalty its published 'l' study selected):

  published   the original inputs and the per-litre saturation: must reproduce the published
              '|l' columns of runs/v2_aou_units_chronos/zeroshot_columns.csv.gz
  mgl         the sondes in mg/l hour by hour (base_mgl/.../hourly_observations.csv.gz, written by
              scripts/v2_oxygen_mgl_chronos_remote.py) and the saturation in mg/l (per-litre form
              x 0.0319988, T clipped to [-2, 40] as exp_chronos clips it, S = 38)

Columns written: oxygen|ZS_aou, V2C2_oxygen_aou, CH2_oxygen_aou_uni, CH2_oxygen_aou_cov, each with a
'|<mode>' suffix, on the endpoint rows of exp_chronos.ep, in results/zeroshot_<mode>_columns.csv.gz.
Temperature forecasts (Chronos-Bolt zero-shot) invert the utilisation, as published.

A third mode, mgl_bk (addendum 71 part B), is mgl with the saturation from the consistent Benson and
Krause set of Garcia and Gordon (thermodynamics.o2sol_benson_krause_umol_per_l) in place of the
project's coefficient mixture; it reads base_mgl/.

Runs from a snapshot directory holding code/ (exp_chronos.py), v2_chronos2.py, v2_chronos_c2.py,
v2_aou_units_chronos.py, ref/ and base_<mode>/review_20260919/inputs/, with the repository root
(src/) on PYTHONPATH:  python v2_oxygen_mgl_chronos.py published|mgl|mgl_bk
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
MODE = sys.argv[1]
if MODE not in ('published', 'mgl', 'mgl_bk'):
    raise SystemExit('published | mgl | mgl_bk')
os.environ['SMARTBAY_BASE'] = str(HERE / ('base_mgl' if MODE.startswith('mgl') else f'base_{MODE}'))
os.environ.setdefault('HF_HOME', '/home/vito/smartbay_do/.hf')
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'code'))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get('V2_THREADS', '2')))
import exp_chronos as xc  # noqa: E402
import v2_aou_units_chronos as au  # noqa: E402  (column(), eval_times(); imported before o2_sol is rebound)
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l, o2sol_benson_krause_umol_per_l  # noqa: E402

MG_PER_UMOL_L = 0.0319988
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def o2_sol(T, S=xc.SAL):
    T = np.clip(np.asarray(T, float), -2, 40)
    sat = (o2sol_benson_krause_umol_per_l if MODE == 'mgl_bk' else o2sol_garcia_gordon_umol_per_l)(T, S)
    return sat * MG_PER_UMOL_L if MODE.startswith('mgl') else sat


def main():
    from chronos import BaseChronosPipeline
    import v2_chronos2 as c2m
    import v2_chronos_c2
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    published = json.loads((HERE / 'ref/zeroshot_report.json').read_text())
    alpha = published['c2_l']['selection']['params']['alpha']
    ep = xc.ep
    times = au.eval_times(ep)
    cols = {}
    # the published random stream at the start of the utilisation C2 step (au.zeroshot replays it)
    for variable in ('temperature', 'oxygen'):
        by = xc.pre2026_issuance(variable)
        {s: (t[xc.RNG.choice(len(t), min(len(t), 1500), replace=False)] if len(t) > 1500 else t)
         for s, t in by.items() if len(t) > 0}
    rng_state = xc.RNG.bit_generator.state
    xc.o2_sol = o2_sol
    zs = BaseChronosPipeline.from_pretrained(xc.REPO, device_map='cuda', torch_dtype=torch.float32)
    temp_zs = xc.predict_at(zs, 'temperature', 'raw', times['temperature'])
    base = xc.predict_at(zs, 'oxygen', 'aou', times['oxygen'])
    cols[f'oxygen|ZS_aou|{MODE}'] = au.column(ep, base, temp_zs, 'oxygen', 'aou')
    xc.RNG.bit_generator.state = rng_state
    X, y, when = v2_chronos_c2.samples(zs, 'oxygen', 'aou')
    c2 = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y)
    corrected = {}
    for k, v in base.items():
        D = xc.c2_design(k[0], 'oxygen', [k[1]])
        corrected[k] = v + float(c2.predict(D)[0]) if np.isfinite(D).all() else v
    cols[f'V2C2_oxygen_aou|{MODE}'] = au.column(ep, corrected, temp_zs, 'oxygen', 'aou')
    del zs
    torch.cuda.empty_cache()
    pipe = BaseChronosPipeline.from_pretrained(c2m.REPO, device_map='cuda', torch_dtype=torch.float32)
    for tag, use_cov in (('uni', False), ('cov', True)):
        temp = c2m.forecast(pipe, 'temperature', 'raw', use_cov, times['temperature'])
        cols[f'CH2_oxygen_aou_{tag}|{MODE}'] = au.column(ep, c2m.forecast(pipe, 'oxygen', 'aou', use_cov, times['oxygen']),
                                                        temp, 'oxygen', 'aou')
    del pipe
    torch.cuda.empty_cache()
    frame = ep[['variable', 'sensor', 'time']].copy()
    for k, v in cols.items():
        frame[k] = v
    frame.to_csv(OUT / f'zeroshot_{MODE}_columns.csv.gz', index=False)
    report = {'declared_in': ('config/bay_reserved_protocol_v2_addendum71.json' if MODE == 'mgl_bk' else
                              'config/bay_reserved_protocol_v2_addendum60.json'), 'mode': MODE,
              'c2_alpha': alpha, 'c2_samples': int(len(y)), 'seconds': round(time.time() - t0, 1),
              'columns': {k: int(np.isfinite(v).sum()) for k, v in cols.items()},
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if MODE == 'published':
        ref = pd.read_csv(HERE / 'ref/zeroshot_columns.csv.gz')
        gate = {}
        for k in cols:
            want = k.replace('|published', '|l')
            a, b = frame[k].to_numpy(float), ref[want].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            gate[k] = {'reference': want, 'max_abs_diff': float(np.max(np.abs(a[both] - b[both]))),
                       'same_missing': bool((np.isfinite(a) == np.isfinite(b)).all())}
            gate[k]['passed'] = bool(gate[k]['max_abs_diff'] <= 1e-3 and gate[k]['same_missing'])
        report['gate'] = gate
        report['passed'] = all(g['passed'] for g in gate.values())
    (OUT / f'zeroshot_{MODE}_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    main()
