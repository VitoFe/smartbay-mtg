"""POST-HOC diagnostic: validation-versus-test curve -- is "fine-tuning hurts" a finding or an artifact?

The 2x2 experiment capped fine-tuning at 400 steps with lr 1e-4, and the oxygen validation loss
was still decreasing when it stopped. That is not enough to conclude anything about fine-tuning.

This sweep varies learning rate and step budget, and -- critically -- selects the configuration
on the PRE-2026 validation split alone. The selected configuration's 2026 score is then read
once. No 2026 value influences the choice, so the reported number is not test-set tuned.

If the best pre-2026 configuration still underperforms zero-shot on 2026, that is a genuine
result about the data-scarce regime: it means fine-tuning improves fit to the pre-2026
distribution while degrading transfer across the year boundary.
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
CTX, H, REPO = 512, 6, "amazon/chronos-bolt-base"
CUT = pd.Timestamp("2026-01-01", tz="UTC")

import torch
from chronos import BaseChronosPipeline

import sys
sys.path.insert(0, str(BASE / "review_20260919" / "code"))
from exp_chronos import (o2_sol, obs, ep, series_for, sensors_for, predict_at, macro_mae)

GRID = [(1e-5, 1500), (3e-5, 1500), (1e-4, 1500), (3e-4, 800)]


def windows(variable, space, n=24000):
    ctxs, tgts = [], []
    ss = sensors_for(variable)
    for s in ss:
        ser = series_for(s, variable, space)
        if ser is None:
            continue
        v = ser[ser.index < CUT].to_numpy(float)
        if len(v) < CTX + H + 24:
            continue
        ends = np.arange(CTX, len(v) - H)
        take = min(len(ends), max(1, n // max(1, len(ss))))
        for e in RNG.choice(ends, take, replace=False):
            c, t = v[e - CTX:e], v[e:e + H]
            if np.isfinite(c).sum() >= 96 and np.isfinite(t).all():
                ctxs.append(c); tgts.append(t)
    return np.asarray(ctxs, np.float32), np.asarray(tgts, np.float32)


def train_one(variable, space, lr, steps, ctxs, tgts, ti, vi, tag):
    pipe = BaseChronosPipeline.from_pretrained(REPO, device_map="cuda", torch_dtype=torch.float32)
    model = pipe.model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def loss_on(sel):
        c = torch.tensor(ctxs[sel], device="cuda"); t = torch.tensor(tgts[sel], device="cuda")
        m = torch.isfinite(c)
        return model(context=torch.nan_to_num(c), mask=m, target=t,
                     target_mask=torch.isfinite(t)).loss

    best, best_state, patience = float("inf"), None, 0
    for step in range(steps):
        l = loss_on(RNG.choice(ti, 32, replace=False))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_on(vi[i:i+64]).item() for i in range(0, len(vi), 64)]))
            model.train()
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 4:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  [{tag}] lr={lr:g} steps<={steps} -> best pre-2026 val {best:.5f}", flush=True)
    return pipe, best


def main():
    eval_times = {v: {s: pd.DatetimeIndex(sorted(ep.loc[(ep.variable == v) & (ep.sensor == s), "time"]))
                      for s in sensors_for(v)} for v in ("temperature", "oxygen")}
    zs = BaseChronosPipeline.from_pretrained(REPO, device_map="cuda", torch_dtype=torch.float32)
    temp_zs = predict_at(zs, "temperature", "raw", eval_times["temperature"])

    report = {}
    for variable, space in [("temperature", "raw"), ("oxygen", "aou")]:
        ctxs, tgts = windows(variable, space)
        n_val = max(128, int(0.1 * len(ctxs)))
        idx = RNG.permutation(len(ctxs)); vi, ti = idx[:n_val], idx[n_val:]
        print(f"\n=== {variable}/{space}: {len(ti)} train / {len(vi)} val windows ===", flush=True)

        trials = []
        for lr, steps in GRID:
            pipe, val = train_one(variable, space, lr, steps, ctxs, tgts, ti, vi,
                                  f"{variable}/{space}")
            trials.append(dict(lr=lr, steps=steps, val=val, pipe=pipe))
        sel = min(trials, key=lambda r: r["val"])
        print(f"  SELECTED on pre-2026 val: lr={sel['lr']:g} (val {sel['val']:.5f})", flush=True)

        curve = []
        for t in trials:
            preds = predict_at(t["pipe"], variable, space, eval_times[variable])
            col = np.full(len(ep), np.nan)
            for i, r in ep.iterrows():
                if r.variable != variable:
                    continue
                k = (r.sensor, r.time)
                if k not in preds:
                    continue
                v = preds[k]
                if space == "aou":
                    That = temp_zs.get(k, np.nan)
                    if not np.isfinite(That):
                        continue
                    v = o2_sol(That) - v
                col[i] = v
            nm = f"VT_{variable}_{space}_lr{t['lr']:g}"
            ep[nm] = col
            d = ep[ep.variable == variable]
            mae = macro_mae(d, nm)
            curve.append(dict(lr=t["lr"], steps=t["steps"], val_loss=t["val"],
                              mae_2026=mae, selected=bool(t["lr"] == sel["lr"])))
            print(f"    lr={t['lr']:<8g} val={t['val']:.5f}  ->  2026 MAE={mae:.4f}"
                  f"{'   <- selected on val' if t['lr']==sel['lr'] else ''}", flush=True)
            del t["pipe"]
        torch.cuda.empty_cache()
        d = ep[ep.variable == variable]
        report[f"{variable}/{space}"] = {
            "note": "POST-HOC DIAGNOSTIC. Selection was made on pre-2026 val loss alone; the "
                    "2026 column is reported afterwards to expose distribution shift and must "
                    "not be used to choose a configuration.",
            "selected_lr": sel["lr"],
            "curve": curve,
            "reference_MODEL": macro_mae(d, "prediction"),
        }

    ep.to_csv(OUT / "chronos_valtest_predictions.csv.gz", index=False, compression="gzip")
    (OUT / "chronos_valtest_curve.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
