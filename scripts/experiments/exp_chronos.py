"""Chronos 2x2 -- pretrained forecaster as a component, not a rival.

The earlier zero-shot observation is too weak a basis for a claim about what the bespoke
pipeline's exogenous features contribute. This experiment asks the three questions that would
settle it:

  1. Does fine-tuning on the LOCAL pre-2026 record improve the pretrained model?
  2. Can the physical guidance identified elsewhere in this thesis be COMPOSED with it rather
     than competing with it?
  3. Does adding exogenous context close the gap univariate operation leaves?

Axes
  training     ZS  zero-shot            | FT  fine-tuned on pre-2026 windows only
  composition  raw target               | C1  AOU reparameterisation (oxygen only)
                                        | C2  exogenous residual corrector (both variables)

C1: model apparent oxygen utilisation instead of concentration, then reconstruct
    DO = sol(That, S) - AOUhat using the SAME arm's temperature forecast.
C2: ridge on causal network context, fitted on pre-2026 residuals of the SAME arm, predicting
    the arm's own 6 h residual. Never sees a 2026 value.

Leakage control
  - every window used for fine-tuning ends strictly before 2026-01-01;
  - the fine-tuning validation split is the last 10 % of the pre-2026 period, still pre-2026;
  - C2 is fitted only on pre-2026 issuance points;
  - scoring uses the identical rows of the frozen reserved-period record, and the frozen
    aggregation rule (mean over quarters of per-sensor mean MAE), which reproduces the published
    headline exactly.

Salinity is not recorded hourly, so solubility uses S = 38 PSU throughout; this is a constant
offset in the reconstruction and is stated as an approximation, not a measurement.
"""
import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", "/home/vito/smartbay_do/.hf")
BASE = Path(os.environ.get("SMARTBAY_BASE", "/home/vito/smartbay_do"))
INP = BASE / "review_20260919" / "inputs"
OUT = BASE / "review_20260919" / "results" / "chronos"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260919)

CTX, H = 512, 6
REPO = "amazon/chronos-bolt-base"
CUT = pd.Timestamp("2026-01-01", tz="UTC")
SAL = 38.0

import torch
from chronos import BaseChronosPipeline


def o2_sol(T, S=SAL):
    T = np.clip(np.asarray(T, float), -2, 40)
    Ts = np.log((298.15 - T) / (273.15 + T))
    A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    return np.exp(A[0] + A[1]*Ts + A[2]*Ts**2 + A[3]*Ts**3 + A[4]*Ts**4 + A[5]*Ts**5
                  + S*(B[0] + B[1]*Ts + B[2]*Ts**2 + B[3]*Ts**3) - 2.75915e-7*S**2)


# --------------------------------------------------------------------- data ----
obs = pd.read_csv(INP / "hourly_observations.csv.gz", index_col=0)
obs.index = pd.to_datetime(obs.index, utc=True)
obs = obs.sort_index().reindex(pd.date_range(obs.index.min(), obs.index.max(), freq="h"))

ep = pd.read_csv(INP / "forecast_endpoints_2026.csv.gz", parse_dates=["time", "target_time"])
ep["time"] = pd.to_datetime(ep.time, utc=True)
ep = ep[ep.has_anchor & ep.has_target].reset_index(drop=True)

TCOLS = [c for c in obs.columns if c.endswith("::temperature")]
NET = pd.DataFrame({
    "net_T": obs[TCOLS].mean(axis=1),
    "net_sd": obs[TCOLS].std(axis=1),
})
NET["net_d24"] = NET.net_T.diff(24)
NET["net_d6"] = NET.net_T.diff(6)


def series_for(sensor, variable, space):
    """Return the hourly series this arm models."""
    col = f"{sensor}::{variable}"
    if col not in obs.columns:
        return None
    s = obs[col].astype(float)
    if space == "aou" and variable == "oxygen":
        tcol = f"{sensor}::temperature"
        if tcol not in obs.columns:
            return None
        return pd.Series(o2_sol(obs[tcol].to_numpy(float)) - s.to_numpy(), index=obs.index)
    return s


def sensors_for(variable):
    return sorted(ep.loc[ep.variable == variable, "sensor"].unique())


# ------------------------------------------------------------- fine-tuning ----
def sample_windows(variable, space, n_windows=24000):
    """Context+target windows ending strictly before the 2026 cutoff."""
    ctxs, tgts = [], []
    for s in sensors_for(variable):
        ser = series_for(s, variable, space)
        if ser is None:
            continue
        pre = ser[ser.index < CUT]
        v = pre.to_numpy(float)
        if len(v) < CTX + H + 24:
            continue
        ends = np.arange(CTX, len(v) - H)
        if len(ends) == 0:
            continue
        take = min(len(ends), max(1, n_windows // max(1, len(sensors_for(variable)))))
        for e in RNG.choice(ends, take, replace=False):
            c = v[e - CTX:e]; t = v[e:e + H]
            if np.isfinite(c).sum() < 96 or not np.isfinite(t).all():
                continue
            ctxs.append(c); tgts.append(t)
    if not ctxs:
        return None, None
    return np.asarray(ctxs, dtype=np.float32), np.asarray(tgts, dtype=np.float32)


def finetune(variable, space, steps=400, bs=32, lr=1e-4, tag=""):
    ctxs, tgts = sample_windows(variable, space)
    if ctxs is None or len(ctxs) < 200:
        print(f"  [{tag}] insufficient windows ({0 if ctxs is None else len(ctxs)}) -> skip FT", flush=True)
        return None
    n_val = max(64, int(0.1 * len(ctxs)))
    idx = RNG.permutation(len(ctxs))
    vi, ti = idx[:n_val], idx[n_val:]
    pipe = BaseChronosPipeline.from_pretrained(REPO, device_map="cuda", torch_dtype=torch.float32)
    model = pipe.model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, best_state, patience = float("inf"), None, 0

    def batch_loss(sel, train=True):
        c = torch.tensor(ctxs[sel], device="cuda")
        t = torch.tensor(tgts[sel], device="cuda")
        m = torch.isfinite(c)
        c = torch.nan_to_num(c)
        out = model(context=c, mask=m, target=t, target_mask=torch.isfinite(t))
        return out.loss

    for step in range(steps):
        sel = RNG.choice(ti, bs, replace=False)
        loss = batch_loss(sel)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([batch_loss(vi[i:i+64], False).item()
                                    for i in range(0, len(vi), 64)]))
            model.train()
            print(f"  [{tag}] step {step+1:4d}  train {loss.item():.4f}  val {vl:.4f}", flush=True)
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 3:
                    print(f"  [{tag}] early stop at step {step+1}", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  [{tag}] fine-tuned on {len(ti)} windows, best val {best:.4f}", flush=True)
    return pipe


# ---------------------------------------------------------------- inference ----
def predict_at(pipe, variable, space, times_by_sensor):
    """times_by_sensor: {sensor: DatetimeIndex of issuance times}. Returns {(sensor,t): value}."""
    out = {}
    for s, times in times_by_sensor.items():
        ser = series_for(s, variable, space)
        if ser is None or len(times) == 0:
            continue
        ctxs, keys = [], []
        for t in times:
            hist = ser.loc[:t].to_numpy(float)[-CTX:]
            if np.isfinite(hist).sum() < 48:
                continue
            ctxs.append(torch.tensor(hist, dtype=torch.float32)); keys.append((s, t))
        for k in range(0, len(ctxs), 256):
            q, _ = pipe.predict_quantiles(inputs=ctxs[k:k+256], prediction_length=H,
                                          quantile_levels=[0.1, 0.5, 0.9])
            med = q[:, H-1, 1].float().cpu().numpy()
            for j, v in enumerate(med):
                out[keys[k+j]] = float(v)
    return out


def pre2026_issuance(variable):
    """Causal issuance grid before the cutoff, for fitting C2."""
    grid = obs.index[(obs.index < CUT) & obs.index.hour.isin([0, 6, 12, 18])]
    by = {}
    for s in sensors_for(variable):
        col = f"{s}::{variable}"
        if col not in obs.columns:
            continue
        ser = obs[col]
        ok = [t for t in grid if np.isfinite(ser.get(t, np.nan))
              and np.isfinite(ser.get(t + pd.Timedelta(hours=H), np.nan))]
        by[s] = pd.DatetimeIndex(ok)
    return by


C2_FEATS = ["net_T", "net_sd", "net_d24", "net_d6", "own_d6", "own_d24",
            "hs1", "hc1", "ms1", "mc1"]


def c2_design(sensor, variable, times):
    col = f"{sensor}::{variable}"
    ser = obs[col]
    rows = []
    for t in times:
        h = t.hour; m = t.month
        rows.append([
            NET.net_T.get(t, np.nan), NET.net_sd.get(t, np.nan),
            NET.net_d24.get(t, np.nan), NET.net_d6.get(t, np.nan),
            ser.get(t, np.nan) - ser.get(t - pd.Timedelta(hours=6), np.nan),
            ser.get(t, np.nan) - ser.get(t - pd.Timedelta(hours=24), np.nan),
            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
            np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
        ])
    return np.asarray(rows, dtype=float)


def fit_c2(pipe, variable, space):
    """Ridge on pre-2026 residuals of THIS arm. Returns fitted model or None."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    by = pre2026_issuance(variable)
    by = {s: (t[RNG.choice(len(t), min(len(t), 1500), replace=False)] if len(t) > 1500 else t)
          for s, t in by.items() if len(t) > 0}
    preds = predict_at(pipe, variable, space, by)
    X, y = [], []
    for s, times in by.items():
        col = f"{s}::{variable}"
        ser = obs[col]
        keep = [t for t in times if (s, t) in preds]
        if not keep:
            continue
        D = c2_design(s, variable, keep)
        for i, t in enumerate(keep):
            truth = ser.get(t + pd.Timedelta(hours=H), np.nan)
            p = preds[(s, t)]
            if space == "aou":
                # residual is fitted in the arm's own space; reconstruct later
                tc = f"{s}::temperature"
                truth = o2_sol(obs[tc].get(t + pd.Timedelta(hours=H), np.nan)) - truth
            if not np.isfinite(truth) or not np.isfinite(D[i]).all():
                continue
            X.append(D[i]); y.append(truth - p)
    if len(X) < 200:
        return None
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30)))
    m.fit(np.asarray(X), np.asarray(y))
    return m


def macro_mae(d, col):
    dd = d.dropna(subset=[col, "truth"])
    if dd.empty:
        return np.nan
    cell = dd.groupby(["quarter", "sensor"]).apply(
        lambda x: (x[col] - x.truth).abs().mean(), include_groups=False)
    return float(cell.groupby("quarter").mean().mean())


# --------------------------------------------------------------------- run ----
def main():
    results = {}
    ep["B0_persistence"] = ep["persistence"]
    ep["MODEL"] = ep["prediction"]

    eval_times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), "time"]))
                      for s in sensors_for(v)} for v in ("temperature", "oxygen")}

    pipes = {}
    pipes[("ZS", "raw")] = BaseChronosPipeline.from_pretrained(
        REPO, device_map="cuda", torch_dtype=torch.float32)
    print("loaded zero-shot pipeline", flush=True)

    # fine-tuned variants: temperature(raw), oxygen(raw), oxygen(aou)
    ft = {}
    ft[("temperature", "raw")] = finetune("temperature", "raw", tag="FT temp/raw")
    ft[("oxygen", "raw")] = finetune("oxygen", "raw", tag="FT oxy/raw")
    ft[("oxygen", "aou")] = finetune("oxygen", "aou", tag="FT oxy/aou")

    # temperature predictions per training regime (needed to reconstruct AOU arms)
    temp_pred = {}
    for reg in ("ZS", "FT"):
        p = pipes[("ZS", "raw")] if reg == "ZS" else ft[("temperature", "raw")]
        if p is None:
            continue
        temp_pred[reg] = predict_at(p, "temperature", "raw", eval_times["temperature"])
        print(f"temperature predictions [{reg}]: {len(temp_pred[reg])}", flush=True)

    arms = []
    for reg in ("ZS", "FT"):
        for variable in ("temperature", "oxygen"):
            spaces = ("raw",) if variable == "temperature" else ("raw", "aou")
            for space in spaces:
                p = pipes[("ZS", "raw")] if reg == "ZS" else ft.get((variable, space))
                if p is None:
                    continue
                base = predict_at(p, variable, space, eval_times[variable])
                c2 = fit_c2(p, variable, space)
                for use_c2 in (False, True):
                    if use_c2 and c2 is None:
                        continue
                    name = f"{reg}_{space}" + ("_C2" if use_c2 else "")
                    col = np.full(len(ep), np.nan)
                    for i, r in ep.iterrows():
                        if r.variable != variable:
                            continue
                        k = (r.sensor, r.time)
                        if k not in base:
                            continue
                        v = base[k]
                        if use_c2:
                            D = c2_design(r.sensor, variable, [r.time])
                            if np.isfinite(D).all():
                                v = v + float(c2.predict(D)[0])
                        if space == "aou":
                            That = temp_pred.get(reg, {}).get((r.sensor, r.time), np.nan)
                            if not np.isfinite(That):
                                continue
                            v = o2_sol(That) - v
                        col[i] = v
                    ep[f"{variable}|{name}"] = col
                    arms.append((variable, name))
                    print(f"  arm {variable}/{name}: {int(np.isfinite(col).sum())} predictions", flush=True)

    ep.to_csv(OUT / "chronos_predictions.csv.gz", index=False, compression="gzip")

    summary, boot = {}, {}
    for variable in ("temperature", "oxygen"):
        d = ep[ep.variable == variable]
        cols = ["B0_persistence", "MODEL"] + [f"{variable}|{n}" for v, n in arms if v == variable]
        summary[variable] = {c: macro_mae(d, c) for c in cols if c in d}
        summary[variable]["_n"] = int(len(d))
        blocks = d[["quarter", "week"]].drop_duplicates().values.tolist()
        diffs = {c: [] for c in cols if c in d and c != "MODEL"}
        for _ in range(600):
            pick = [blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]
            s = pd.concat([d[(d.quarter == q) & (d.week == w)] for q, w in pick])
            ref = macro_mae(s, "MODEL")
            for c in diffs:
                diffs[c].append(macro_mae(s, c) - ref)
        boot[variable] = {c: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                          for c, v in diffs.items()}

    payload = {"macro_mae": summary, "ci_arm_minus_MODEL": boot,
               "notes": {"salinity": "constant 38 PSU in solubility",
                         "context": CTX, "horizon": H, "repo": REPO,
                         "finetune": "pre-2026 windows only, 10% pre-2026 val split, early stop"}}
    (OUT / "chronos_summary.json").write_text(json.dumps(payload, indent=2))
    print("\n" + json.dumps(payload, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
