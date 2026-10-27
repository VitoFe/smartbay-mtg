"""Experiment C — near-bottom oxygen inference from surface information.

Motivation: every moored probe sits at 1-5 m over a 7-11 m seabed, so the network
samples the surface mixed layer. Hypoxia, if it developed, would develop at the bed.
The operational question is therefore: given surface information, what can be said
about near-bottom oxygen?

Two predictor regimes
  C1 cast_surface   surface bins of the SAME cast (T, S, DO, chlorophyll)
                    -> upper bound: what vertical structure is predictable at all
  C2 moored_only    moored telemetry + geometry + season, NO cast surface values
                    -> the true operational case (no CTD available at query time)

The decisive baseline is B_surface: "assume the bottom equals the surface", which is
what an operator with only a surface sonde does today. A model must beat that or it
adds nothing.

Targets: near-bottom DO directly; near-bottom AOU; and the surface-to-bottom DO
difference (predict the gradient, add it to the observed surface value).
Protocols: P1 held-station, P2 chronological (train <2026, test 2026).
"""
import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("SMARTBAY_BASE", "/home/vito/smartbay_do"))
INP = BASE / "review_20260919" / "inputs"
OUT = BASE / "review_20260919" / "results" / "bottom"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260919)


def o2_solubility(T, S=38.0):
    Ts = np.log((298.15 - T) / (273.15 + T))
    A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    lnC = (A[0] + A[1]*Ts + A[2]*Ts**2 + A[3]*Ts**3 + A[4]*Ts**4 + A[5]*Ts**5
           + S*(B[0] + B[1]*Ts + B[2]*Ts**2 + B[3]*Ts**3) - 2.75915e-7*S**2)
    return np.exp(lnC)


def build():
    c = pd.read_csv(INP / "ctd_all_casts.csv.gz", parse_dates=["t_utc"])
    c = c[c.qc_flag == 0].dropna(subset=["temp_c", "do_umol_l", "depth_m"]).copy()
    c["t_utc"] = pd.to_datetime(c.t_utc, utc=True)
    c["salinity_psu"] = c.get("salinity_psu", pd.Series(38.0, index=c.index)).fillna(38.0)
    if "chlorophyll_ug_l" not in c: c["chlorophyll_ug_l"] = np.nan

    recs = []
    for cid, g in c.groupby("cast_id"):
        g = g.sort_values("depth_m")
        surf = g[g.depth_m <= 2.0]
        bot = g.tail(3)
        if len(surf) == 0 or len(bot) == 0:
            continue
        if bot.depth_m.min() - surf.depth_m.max() < 2.0:   # need real vertical span
            continue
        recs.append(dict(
            cast_id=cid, station=g.station.iloc[0], t_utc=g.t_utc.iloc[0],
            lat=g.lat.iloc[0], lon=g.lon.iloc[0], seabed_m=g.seabed_m.iloc[0],
            bot_depth=float(bot.depth_m.mean()),
            surf_T=float(surf.temp_c.mean()), surf_S=float(surf.salinity_psu.mean()),
            surf_DO=float(surf.do_umol_l.mean()),
            surf_chl=float(surf.chlorophyll_ug_l.mean()) if surf.chlorophyll_ug_l.notna().any() else np.nan,
            bot_T=float(bot.temp_c.mean()), bot_S=float(bot.salinity_psu.mean()),
            bot_DO=float(bot.do_umol_l.mean()),
        ))
    d = pd.DataFrame(recs)
    d["bot_sol"] = o2_solubility(d.bot_T.values, d.bot_S.values)
    d["bot_AOU"] = d.bot_sol - d.bot_DO
    d["delta_DO"] = d.bot_DO - d.surf_DO          # target for the gradient model
    d["strat_T"] = d.surf_T - d.bot_T             # thermal stratification proxy

    obs = pd.read_csv(INP / "hourly_observations.csv.gz", index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    tcols = [x for x in obs.columns if x.endswith("::temperature")]
    ocols = [x for x in obs.columns if x.endswith("::oxygen")]
    net = pd.DataFrame({"net_T": obs[tcols].mean(axis=1),
                        "net_T_sd": obs[tcols].std(axis=1),
                        "net_DO": obs[ocols].mean(axis=1)}).sort_index()
    net["net_T_24h"] = net.net_T.rolling(24, min_periods=6).mean()
    net["net_DO_24h"] = net.net_DO.rolling(24, min_periods=6).mean()
    d = pd.merge_asof(d.sort_values("t_utc"), net, left_on="t_utc", right_index=True,
                      direction="nearest", tolerance=pd.Timedelta("3h"))
    doy = d.t_utc.dt.dayofyear.values.astype(float)
    for k in (1, 2):
        d[f"doy_s{k}"] = np.sin(2*np.pi*k*doy/365.25)
        d[f"doy_c{k}"] = np.cos(2*np.pi*k*doy/365.25)
    return d.dropna(subset=["bot_DO", "surf_DO"]).reset_index(drop=True)


GEO_SEASON = ["lat", "lon", "seabed_m", "bot_depth", "doy_s1", "doy_c1", "doy_s2", "doy_c2"]
MOORED = ["net_T", "net_T_sd", "net_DO", "net_T_24h", "net_DO_24h"]
CAST_SURF = ["surf_T", "surf_S", "surf_DO", "surf_chl", "strat_T"]
REGIMES = {"C1_cast_surface": GEO_SEASON + MOORED + CAST_SURF,
           "C2_moored_only":  GEO_SEASON + MOORED}

# Reference arms (2026-09-27). The `aou` arms form their target and map their prediction back
# at `bot_sol`, the saturation at the temperature and salinity the cast measured at the bed, which
# neither the naive assumption nor an operator without a bottom sensor has. The reference arms map
# back at what each regime may read instead: the mean-state regime at the moored network's mean
# temperature (the training mean where the network is dark, as its features are imputed) and a
# fixed 38, the salinity of the forecasting arms; the profile regime at the deepest reading of the
# surface profile it is given (the deepest bin within 2 m). Two targets per regime:
#   aou_ref      saturation at that reference minus near-bottom oxygen, formed and inverted there
#   aou_bed_ref  the near-bottom utilisation of the `aou` arm (a training label, which the bed's
#                measured T and S may form), inverted at the reference
# build(), folds() and run() are unchanged, so the published arms reproduce.
REFERENCE_SALINITY = 38.0
REFERENCE_TARGETS = ("aou_ref", "aou_bed_ref")


def profile_reading():
    """The deepest reading of each cast's surface window (depth <= 2 m), filtered as build() does."""
    c = pd.read_csv(INP / "ctd_all_casts.csv.gz", parse_dates=["t_utc"])
    c = c[c.qc_flag == 0].dropna(subset=["temp_c", "do_umol_l", "depth_m"]).copy()
    c["salinity_psu"] = c.get("salinity_psu", pd.Series(38.0, index=c.index)).fillna(38.0)
    s = (c[c.depth_m <= 2.0].sort_values(["cast_id", "depth_m"], kind="mergesort")
         .groupby("cast_id").tail(1).set_index("cast_id"))
    return s[["depth_m", "temp_c", "salinity_psu"]].rename(
        columns={"depth_m": "prof_depth", "temp_c": "prof_T", "salinity_psu": "prof_S"})


def with_reference(tr, te):
    """One outer fold's rows with the reference saturation of each regime (`ref_sol_<regime>`) and
    the `aou_ref` target (`aou_ref_<regime>`). Needs the columns of profile_reading() joined on."""
    fill = float(np.nanmean(tr.net_T.to_numpy(float)))
    out = []
    for x in (tr, te):
        x = x.copy()
        x["ref_sol_C2_moored_only"] = o2_solubility(x.net_T.fillna(fill).to_numpy(float),
                                                    REFERENCE_SALINITY)
        x["ref_sol_C1_cast_surface"] = o2_solubility(x.prof_T.to_numpy(float),
                                                     x.prof_S.to_numpy(float))
        for rg in REGIMES:
            x[f"aou_ref_{rg}"] = x[f"ref_sol_{rg}"] - x.bot_DO
        out.append(x)
    return out


def fit_predict(name, tr, te, cols, ycol):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    mu = np.nanmean(Xtr, axis=0); mu = np.where(np.isnan(mu), 0.0, mu)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    ytr = tr[ycol].to_numpy(float)
    if name == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 40)))
    elif name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=15,
                              min_child_samples=15, random_state=0, verbose=-1, n_jobs=8)
    else:
        raise ValueError(name)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def folds(d, protocol):
    if protocol == "P1_held_station":
        for s in sorted(d.station.unique()):
            tr, te = d[d.station != s], d[d.station == s]
            if len(te) >= 10 and len(tr) >= 60:
                yield s, tr, te
    else:
        tr, te = d[d.t_utc < "2026-01-01"], d[d.t_utc >= "2026-01-01"]
        if len(te) >= 10:
            yield "2026", tr, te


def run():
    d = build()
    print(f"usable casts with surface+bottom pair: {len(d)}", flush=True)
    print(f"bottom DO: min {d.bot_DO.min():.1f}  median {d.bot_DO.median():.1f}", flush=True)
    print(f"|surface-bottom DO| median {d.delta_DO.abs().median():.1f}  p90 {d.delta_DO.abs().quantile(.9):.1f}", flush=True)

    rows, preds = [], []
    for protocol in ("P1_held_station", "P2_chronological"):
        for fold, tr, te in folds(d, protocol):
            truth = te.bot_DO.to_numpy(float)
            cand = {"B_surface": te.surf_DO.to_numpy(float),
                    "B_climatology": np.full(len(te), tr.bot_DO.mean())}
            for rg, cols in REGIMES.items():
                for mdl in ("ridge", "lightgbm"):
                    cand[f"{rg}|{mdl}|direct"] = fit_predict(mdl, tr, te, cols, "bot_DO")
                    aou = fit_predict(mdl, tr, te, cols, "bot_AOU")
                    cand[f"{rg}|{mdl}|aou"] = te.bot_sol.to_numpy(float) - aou
                    if rg == "C1_cast_surface":
                        dlt = fit_predict(mdl, tr, te, cols, "delta_DO")
                        cand[f"{rg}|{mdl}|gradient"] = te.surf_DO.to_numpy(float) + dlt
            for k, v in cand.items():
                rows.append(dict(protocol=protocol, fold=fold, model=k, n=len(te),
                                 mae=float(np.abs(v - truth).mean())))
                preds.append(pd.DataFrame(dict(protocol=protocol, fold=fold, model=k,
                                               cast_id=te.cast_id.values, truth=truth, pred=v,
                                               abs_delta=np.abs(te.delta_DO.values))))
            print(f"  fold {protocol}/{fold} n={len(te)}", flush=True)

    res = pd.DataFrame(rows); res.to_csv(OUT / "bottom_folds.csv", index=False)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(OUT / "bottom_predictions.csv.gz", index=False, compression="gzip")
    macro = res.groupby(["protocol", "model"]).agg(
        macro_mae=("mae", "mean"), folds=("fold", "nunique"), n=("n", "sum")).reset_index()
    macro.to_csv(OUT / "bottom_macro.csv", index=False)
    for pr in macro.protocol.unique():
        print(f"\n=== {pr} (MAE umol/L on near-bottom DO) ===")
        print(macro[macro.protocol == pr].sort_values("macro_mae")
              [["model", "macro_mae", "folds", "n"]].round(3).to_string(index=False))
    # stratified-case error: the cases that actually matter
    strat = P[P.abs_delta >= 20]
    if len(strat):
        s = strat.assign(ae=(strat.pred-strat.truth).abs()).groupby(["protocol","model"]).ae.mean()
        print("\n=== stratified casts only (|surface-bottom DO| >= 20 umol/L) ===")
        print(s.round(3).to_string())
    (OUT / "status.json").write_text(json.dumps({"state": "complete", "casts": len(d)}, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    run()
