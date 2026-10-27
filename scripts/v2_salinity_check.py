"""Addendum-3 study 2: what does the constant 38 PSU in the AOU arms actually cost?

Chapter 5 and scripts/experiments/exp_chronos.py both state that salinity is not recorded
hourly and evaluate Garcia-Gordon solubility at a constant 38 PSU. The statement is false: all five
oxygen-bearing probes carry a co-located conductivity channel with matching hourly coverage.

Before re-running any pretrained model, this measures whether the approximation can matter. The AOU
arm applies solubility twice, once to build the context and once to invert the forecast:

    AOU(t)   = sol(T_obs(t), S) - DO_obs(t)
    DO_pred  = sol(T_pred, S)  - AOU_pred

so a constant S cancels to first order and survives only through the interaction between the
salinity error and the temperature change over the forecast interval. The residual is therefore

    [sol(T_pred, S_true) - sol(T_anchor, S_true)] - [sol(T_pred, 38) - sol(T_anchor, 38)]

which this script evaluates on the real anchor/target temperature pairs and real salinity.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bay_profile_context import ROOT, load_observations
from src.data.thermodynamics import o2sol_garcia_gordon_umol_per_l as sol

CONSTANT = 38.0
HORIZON = 6


def main():
    meta, frames = load_observations()
    rows, report = [], {}
    for name, f in frames.items():
        if 'oxygen' not in f or 'salinity' not in f or 'temperature' not in f:
            continue
        d = f[['temperature', 'salinity', 'oxygen']].dropna()
        if d.empty:
            continue
        future = d.reindex(d.index + pd.Timedelta(hours=HORIZON))
        pair = pd.DataFrame({'t_anchor': d.temperature.to_numpy(), 's': d.salinity.to_numpy(),
                             't_target': future.temperature.to_numpy()},
                            index=d.index).dropna()
        if pair.empty:
            continue
        pair['sensor'] = name
        rows.append(pair)
    obs = pd.concat(rows)

    s, ta, tt = obs.s.to_numpy(), obs.t_anchor.to_numpy(), obs.t_target.to_numpy()
    direct = sol(ta, s) - sol(ta, np.full_like(s, CONSTANT))
    residual = ((sol(tt, s) - sol(ta, s)) -
                (sol(tt, np.full_like(s, CONSTANT)) - sol(ta, np.full_like(s, CONSTANT))))

    report['endpoints'] = int(len(obs))
    report['sensors'] = sorted(obs.sensor.unique().tolist())
    report['salinity'] = {k: float(v) for k, v in
                          obs.s.describe(percentiles=[.01, .5, .99]).items()}
    report['solubility_offset_if_used_once'] = {
        'mean_abs': float(np.abs(direct).mean()), 'p99_abs': float(np.percentile(np.abs(direct), 99)),
        'note': 'what a constant S would cost if solubility were applied only once; it is not'}
    report['residual_after_cancellation'] = {
        'mean_abs': float(np.abs(residual).mean()), 'p99_abs': float(np.percentile(np.abs(residual), 99)),
        'max_abs': float(np.abs(residual).max()),
        'note': 'the quantity that actually reaches the reconstructed oxygen forecast'}
    report['reference_scale'] = {
        'best_oxygen_arm_mae_umol_per_l': 5.3911,
        'sonde_accuracy_umol_per_l': '2-3',
        'oxygen_quantisation_umol_per_l': '0.01-0.03'}

    print(f"{report['endpoints']} endpoint pairs across {len(report['sensors'])} oxygen sensors")
    print(f"salinity: median {report['salinity']['50%']:.3f} PSU, "
          f"1st-99th percentile {report['salinity']['1%']:.3f}-{report['salinity']['99%']:.3f}, "
          f"min {report['salinity']['min']:.3f}, max {report['salinity']['max']:.3f}")
    print(f"offset if solubility were applied once: mean |.| "
          f"{report['solubility_offset_if_used_once']['mean_abs']:.4f} umol/L, "
          f"p99 {report['solubility_offset_if_used_once']['p99_abs']:.4f}")
    print(f"residual that survives the forward/inverse cancellation: mean |.| "
          f"{report['residual_after_cancellation']['mean_abs']:.5f} umol/L, "
          f"p99 {report['residual_after_cancellation']['p99_abs']:.5f}, "
          f"max {report['residual_after_cancellation']['max_abs']:.5f}")

    path = ROOT / 'reports/bay_twin/runs/v2_salinity_check.json'
    path.write_text(json.dumps(report, indent=2))
    print('wrote', path)


if __name__ == '__main__':
    main()
