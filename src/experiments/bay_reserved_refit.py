"""Implement frozen 2025-only refits and separate policy calibration for 2026 evaluation.

Strict temporal causal separation: no 2026 observations enter training,
feature imputation, scaling, hyperparameter tuning, or blend calibration.
"""
from pathlib import Path
import hashlib
import json
import os
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src.data.bay_profile_context import load_observations, load_profiles, TARGETS
from src.data.bay_forcing import weather
from src.experiments.bay_graph_dynamics import build
from src.experiments.bay_dynamics_tree import features
from src.experiments.bay_field_export import EXTRAS, SCALE
from src.protocols.bay_calibration import finite_radius

REFIT_RUN = "20260918_reserved_refit"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def choose_weight(cal, grid):
    """Equal site MAE on calibration window; ties favor smaller learned increment."""
    eligible = cal.groupby("sensor").size()
    names = eligible[eligible >= 30].index
    c = cal[cal.sensor.isin(names)]
    if c.empty:
        return 0.0, []
    scores = []
    for alpha in grid:
        err = abs(c.truth - (c.persistence + alpha * (c.prediction - c.persistence)))
        scores.append({"alpha": float(alpha), "mae": float(err.groupby(c.sensor).mean().mean())})
    best = min(scores, key=lambda r: (r["mae"], r["alpha"]))["alpha"]
    return best, scores


def refit_forecast():
    """Fit 2025-only weather tree with Q4-2025 calibration window for global blend."""
    out_dir = ROOT / f"reports/bay_twin/runs/{REFIT_RUN}/results/forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for variable in ["temperature", "oxygen"]:
        data = build(variable)
        t = data["times"]
        met = weather(t).to_numpy()

        params = json.loads(
            (ROOT / f"reports/bay_twin/runs/20260910T120301Z_dynamics_tree_pilot/results/dynamics_{variable}/selection_summary.json").read_text()
        )["params"]

        cal_end = pd.Timestamp("2025-12-28T18:00:00", tz="UTC")
        cal_start = pd.Timestamp("2025-11-14T00:00:00", tz="UTC")
        train_end = cal_start - pd.Timedelta(hours=78)

        tr = np.flatnonzero(t + pd.Timedelta(hours=6) <= train_end)
        ca = np.flatnonzero((t >= cal_start) & (t + pd.Timedelta(hours=6) <= cal_end))

        eligible = data["valid"][tr].sum(0) >= 100
        fallback = np.nan_to_num(pd.DataFrame(data["x"][tr]).median().to_numpy())
        base, _ = features(data, {"fallback": fallback}, "forecast")
        x_weather = np.concatenate([base, np.broadcast_to(met[:, None], (len(t), len(data["names"]), met.shape[1]))], axis=-1)

        mask = data["valid"] & eligible[None]
        tm = mask[tr]
        weight = 1 / np.maximum(tm.sum(0), 1)
        w = np.broadcast_to(weight, tm.shape)[tm]
        w /= w.mean()

        contract = {
            "model_family": "weather_tree",
            "variable": variable,
            "train_start": str(t[tr].min()),
            "last_training_outcome": str(t[tr].max() + pd.Timedelta(hours=6)),
            "first_calibration": str(t[ca].min()),
            "last_calibration_outcome": str(t[ca].max() + pd.Timedelta(hours=6)),
            "supported_sensors": [n for n, e in zip(data["names"], eligible) if e],
            "training_counts": {n: int(data["valid"][tr, s].sum()) for s, n in enumerate(data["names"]) if eligible[s]}
        }

        model = lgb.LGBMRegressor(
            **params, objective="regression_l1", verbosity=-1, n_jobs=4,
            random_state=42, deterministic=True, force_col_wise=True
        )
        model.fit(x_weather[tr][tm], (data["truth"][tr] - data["x"][tr])[tm], sample_weight=w)

        # Predict on calibration window
        cm = mask[ca]
        pred_inc = model.predict(x_weather[ca][cm])
        cal_records = []
        for (j, s), val in zip(np.argwhere(cm), pred_inc):
            i = ca[j]
            anchor = float(data["x"][i, s] * data["scale"] + data["center"])
            truth_val = float(data["truth"][i, s] * data["scale"] + data["center"])
            cal_records.append({
                "sensor": data["names"][s],
                "time": str(t[i]),
                "variable": variable,
                "truth": truth_val,
                "prediction": float(anchor + val * data["scale"]),
                "persistence": anchor
            })
        cal_df = pd.DataFrame(cal_records)
        cal_df.to_csv(out_dir / f"calibration_{variable}.csv.gz", index=False, compression="gzip")

        best_alpha, grid_scores = choose_weight(cal_df, [0.0, 0.25, 0.5, 0.75, 1.0])

        model_payload = {
            "model_string": model.booster_.model_to_string(),
            "params": params,
            "fallback": fallback.tolist(),
            "names": data["names"],
            "center": data["center"],
            "scale": data["scale"],
            "feature_count": x_weather.shape[-1],
            "contract": contract,
            "global_blend_weight": best_alpha,
            "calibration_grid_scores": grid_scores
        }
        save_json(out_dir / f"refit_{variable}_model.json", model_payload)
        summary[variable] = {
            "supported_sensors": contract["supported_sensors"],
            "calibration_records": len(cal_df),
            "global_blend_weight": best_alpha,
            "calibration_scores": grid_scores
        }
        print(f"[{variable}] Refit complete: blend weight {best_alpha}, supported: {len(contract['supported_sensors'])} sensors")

    save_json(out_dir / "forecast_refit_summary.json", summary)
    return summary


def refit_field():
    """Fit 2025-only linear+boundary spatial field model for 2026 evaluation."""
    out_dir = ROOT / f"reports/bay_twin/runs/{REFIT_RUN}/results/field"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, _ = load_profiles()  # pre-2026 casts
    boundary_df = pd.read_csv(ROOT / "reports/bay_twin/profile_boundary_context.csv.gz")
    merged = data[["cast_id", "depth_m"]].merge(boundary_df, on=["cast_id", "depth_m"], validate="one_to_one")
    extras = merged[EXTRAS].to_numpy(dtype=float)

    # Use frozen alpha
    alpha = 65648.17611753459
    meta, frames = load_observations()

    # Features across all 2024-2025 casts
    cfg = json.loads((ROOT / "config/bay_profile_field.json").read_text())
    from src.experiments.bay_profile_field import prepare
    bundles, contracts, excluded = prepare("temporal", cfg, False)

    # Use bundle for training (all casts up to 2025-12-31)
    b = bundles[-1]  # latest temporal partition covers 2025
    tr = np.arange(len(data))  # train on all valid 2024-2025 casts

    base = (b["base"] + extras[:, 3:6]).astype("float32")
    base[:, 0] = np.clip(base[:, 0], -2, 40)
    base[:, 1] = np.clip(base[:, 1], 0, 42)

    features_all = np.concatenate([b["sx"], extras], axis=1)
    scaler = StandardScaler().fit(features_all[tr])
    design = scaler.transform(features_all).astype("float32")
    target = (data[TARGETS].to_numpy().astype(float) - base.astype(float)) / SCALE

    model = Ridge(alpha=alpha, fit_intercept=False).fit(design[tr], target[tr])

    contract = {
        "model_family": "linear_ridge_vertical_prior",
        "last_training": str(data.t_utc.max()),
        "training_casts": int(data.cast_id.nunique()),
        "training_points": len(data),
        "alpha": alpha,
        "selection_window": {"train_end": "2026-01-01", "selection_end": "2026-07-01"}
    }

    cp = {
        k: b[k] for k in ["raw_features", "imputation_statistics", "indicator_features", "scaler_mean", "scaler_scale"]
    }
    cp.update(
        coef=model.coef_.tolist(),
        boundary_scaler_mean=scaler.mean_.tolist(),
        boundary_scaler_scale=scaler.scale_.tolist(),
        boundary_features=EXTRAS,
        alpha=alpha,
        contract=contract,
        selection_window=contract["selection_window"],
        units=["degC", "PSU", "umol/L"],
        limits="2025 full-period refit. 48h boundary proxy, 3m anchor, depth clamping."
    )
    save_json(out_dir / "refit_field_model.json", cp)
    print(f"[field] Refit complete across {len(data)} CTD points ({data.cast_id.nunique()} casts)")
    return contract


def main():
    print(f"--- Running 2025-Only Refits: {REFIT_RUN} ---")
    f_res = refit_forecast()
    s_res = refit_field()
    print("--- 2025 Refits Successfully Created and Frozen ---")


if __name__ == "__main__":
    main()
