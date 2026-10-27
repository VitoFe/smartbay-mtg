"""Experiment A+B — spatial cohort expansion and AOU target reframing.

A: the frozen field model uses 57 casts because the loader keeps only Gen2 casts from
   2024-07-24 (the Copernicus coverage window). The archive holds 722 QC-passing casts,
   all with oxygen. Here Copernicus is NOT an eligibility gate; it is simply absent, so
   the question is what the in-archive data alone supports at each cohort size.

B: DO concentration is dominated by temperature-driven solubility. Predicting AOU
   (apparent oxygen utilisation) or percent saturation isolates the biogeochemical
   signal. Predictions are mapped back to DO so every model is scored on one scale.

Protocols
  P1 held_station  leave-one-station-out  (spatial generalisation; matches held-site)
  P2 chronological train < 2026, test 2026 (matches the reserved-period protocol)

Models
  M0 climatology     month x 1 m depth bin, from training rows only
  M1 ridge           geometry + seasonal harmonics
  M2 ridge+telemetry M1 plus nearest-in-time moored surface temperature
  M3 lightgbm        same features as M2
  M4 tabpfn          same features as M2 (in-context learner, small-data regime)

Scoring: MAE per held fold, macro-averaged over folds; paired bootstrap over whole
casts (2000 resamples) for model-vs-model differences. Denominators always reported.
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("SMARTBAY_BASE", "/home/vito/smartbay_do"))
INP = BASE / "review_20260919" / "inputs"
OUT = BASE / "review_20260919" / "results" / "spatial"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260919)
N_BOOT = 2000

# ----------------------------------------------------------------- helpers ----
def o2_solubility(T, S=38.0):
    """Garcia & Gordon (1992) combined fit, umol/kg."""
    Ts = np.log((298.15 - T) / (273.15 + T))
    A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    lnC = (A[0] + A[1]*Ts + A[2]*Ts**2 + A[3]*Ts**3 + A[4]*Ts**4 + A[5]*Ts**5
           + S*(B[0] + B[1]*Ts + B[2]*Ts**2 + B[3]*Ts**3) - 2.75915e-7*S**2)
    return np.exp(lnC)


def load():
    c = pd.read_csv(INP / "ctd_all_casts.csv.gz", parse_dates=["t_utc"])
    c = c[c.qc_flag == 0].dropna(subset=["temp_c", "do_umol_l", "depth_m"]).copy()
    c["t_utc"] = pd.to_datetime(c.t_utc, utc=True)
    c["db"] = np.floor(c.depth_m)
    if "salinity_psu" not in c or c.salinity_psu.isna().all():
        c["salinity_psu"] = 38.0
    c["salinity_psu"] = c.salinity_psu.fillna(38.0)
    c["sol"] = o2_solubility(c.temp_c.values, c.salinity_psu.values)
    c["aou"] = c["sol"] - c["do_umol_l"]
    c["pctsat"] = 100.0 * c["do_umol_l"] / c["sol"]

    obs = pd.read_csv(INP / "hourly_observations.csv.gz", index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    tcols = [x for x in obs.columns if x.endswith("::temperature")]
    net = obs[tcols].mean(axis=1).rename("net_T")          # bay-wide surface mean
    net = net.to_frame().sort_index()
    # nearest moored surface temperature within 3 h of each cast
    cast_t = c[["cast_id", "t_utc"]].drop_duplicates().sort_values("t_utc")
    merged = pd.merge_asof(cast_t, net, left_on="t_utc", right_index=True,
                           direction="nearest", tolerance=pd.Timedelta("3h"))
    c = c.merge(merged[["cast_id", "net_T"]], on="cast_id", how="left")
    return c


FEATS = ["depth_m", "seabed_m", "depth_frac", "lat", "lon",
         "doy_s1", "doy_c1", "doy_s2", "doy_c2", "hour", "net_T"]


def featurise(df):
    x = df.copy()
    x["depth_frac"] = x.depth_m / x.seabed_m.replace(0, np.nan)
    doy = x.t_utc.dt.dayofyear.values.astype(float)
    for k in (1, 2):
        x[f"doy_s{k}"] = np.sin(2*np.pi*k*doy/365.25)
        x[f"doy_c{k}"] = np.cos(2*np.pi*k*doy/365.25)
    x["hour"] = x.t_utc.dt.hour + x.t_utc.dt.minute/60.0
    return x


# ------------------------------------------------------------------ models ----
def fit_predict(name, tr, te, target):
    """Return predictions for `target` on te. All models see identical rows."""
    ycol = target
    if name == "M0_climatology":
        clim = tr.groupby([tr.t_utc.dt.month, "db"])[ycol].mean()
        fb = tr.groupby(tr.t_utc.dt.month)[ycol].mean()
        gm = tr[ycol].mean()
        return np.array([clim.get((m, d), fb.get(m, gm))
                         for m, d in zip(te.t_utc.dt.month, te.db)], dtype=float)

    cols = [c for c in FEATS if c != "net_T"] if name in ("M1_ridge",) else FEATS
    Xtr = tr[cols].to_numpy(dtype=float); Xte = te[cols].to_numpy(dtype=float)
    ytr = tr[ycol].to_numpy(dtype=float)
    mu = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)

    if name in ("M1_ridge", "M2_ridge_tel"):
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-3, 6, 40)))
        m.fit(Xtr, ytr); return m.predict(Xte)

    if name == "M3_lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              random_state=0, verbose=-1, n_jobs=8)
        m.fit(Xtr, ytr); return m.predict(Xte)

    if name == "M4_tabpfn":
        from tabpfn import TabPFNRegressor
        idx = np.arange(len(Xtr))
        if len(idx) > 8000:                      # in-context budget
            idx = RNG.choice(idx, 8000, replace=False)
        m = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
        m.fit(Xtr[idx], ytr[idx]); return m.predict(Xte)

    raise ValueError(name)


MODELS = ["M0_climatology", "M1_ridge", "M2_ridge_tel", "M3_lightgbm", "M4_tabpfn"]
# target space -> how to map a prediction back to DO umol/L
SPACES = {"direct_do": "do_umol_l", "aou": "aou", "pctsat": "pctsat"}


def to_do(space, pred, te):
    if space == "direct_do": return pred
    if space == "aou":       return te["sol"].to_numpy() - pred
    if space == "pctsat":    return te["sol"].to_numpy() * pred / 100.0
    raise ValueError(space)


def folds(df, protocol):
    if protocol == "P1_held_station":
        for s in sorted(df.station.unique()):
            tr, te = df[df.station != s], df[df.station == s]
            if len(te) >= 20 and len(tr) >= 100:
                yield s, tr, te
    else:
        tr, te = df[df.t_utc < "2026-01-01"], df[df.t_utc >= "2026-01-01"]
        if len(te) >= 20:
            yield "2026", tr, te


def run():
    c = featurise(load())
    cohorts = {"gen2_164": c[c.t_utc >= "2024-07-24"], "full_722": c}
    rows = []
    preds = []
    for cname, cdf in cohorts.items():
        for protocol in ("P1_held_station", "P2_chronological"):
            for fold, tr, te in folds(cdf, protocol):
                # --- temperature (direct only) ---
                for m in MODELS:
                    try:
                        p = fit_predict(m, tr, te, "temp_c")
                        rows.append(dict(cohort=cname, protocol=protocol, fold=fold,
                                         variable="temperature", space="direct",
                                         model=m, n=len(te),
                                         n_casts=te.cast_id.nunique(),
                                         mae=float(np.abs(p - te.temp_c.values).mean())))
                        preds.append(pd.DataFrame(dict(
                            cohort=cname, protocol=protocol, fold=fold, variable="temperature",
                            space="direct", model=m, cast_id=te.cast_id.values,
                            truth=te.temp_c.values, pred=p)))
                    except Exception as e:
                        rows.append(dict(cohort=cname, protocol=protocol, fold=fold,
                                         variable="temperature", space="direct",
                                         model=m, n=len(te), error=repr(e)[:200]))
                # --- oxygen in three target spaces ---
                for space, col in SPACES.items():
                    for m in MODELS:
                        try:
                            p = fit_predict(m, tr, te, col)
                            do_pred = to_do(space, p, te)
                            rows.append(dict(cohort=cname, protocol=protocol, fold=fold,
                                             variable="oxygen", space=space, model=m,
                                             n=len(te), n_casts=te.cast_id.nunique(),
                                             mae=float(np.abs(do_pred - te.do_umol_l.values).mean())))
                            preds.append(pd.DataFrame(dict(
                                cohort=cname, protocol=protocol, fold=fold, variable="oxygen",
                                space=space, model=m, cast_id=te.cast_id.values,
                                truth=te.do_umol_l.values, pred=do_pred)))
                        except Exception as e:
                            rows.append(dict(cohort=cname, protocol=protocol, fold=fold,
                                             variable="oxygen", space=space, model=m,
                                             n=len(te), error=repr(e)[:200]))
                print(f"done {cname}/{protocol}/{fold}  n={len(te)}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "spatial_folds.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "spatial_predictions.csv.gz",
                                               index=False, compression="gzip")
    ok = res[res.mae.notna()] if "mae" in res else res
    macro = (ok.groupby(["cohort", "protocol", "variable", "space", "model"])
               .agg(macro_mae=("mae", "mean"), folds=("fold", "nunique"),
                    n_points=("n", "sum")).reset_index())
    macro.to_csv(OUT / "spatial_macro.csv", index=False)
    print("\n=== MACRO (MAE; oxygen always scored in umol/L DO space) ===")
    for (co, pr, va) in macro[["cohort", "protocol", "variable"]].drop_duplicates().itertuples(index=False):
        sub = macro[(macro.cohort == co) & (macro.protocol == pr) & (macro.variable == va)]
        print(f"\n-- {co} | {pr} | {va} --")
        print(sub.pivot_table(index="model", columns="space", values="macro_mae").round(4).to_string())
    (OUT / "status.json").write_text(json.dumps(
        {"state": "complete", "rows": len(res),
         "errors": int(res.get("error", pd.Series(dtype=object)).notna().sum())}, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    run()
