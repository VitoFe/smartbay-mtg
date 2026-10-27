"""Sensor placement: turning the observability limit into an instrumentation recommendation.

The observability analysis establishes what the network cannot see. The operator's question is
the constructive one: given a fixed budget of one additional probe, what should be instrumented,
and where? This script answers it from the data that exist, and refuses to answer it for
locations where no ground truth exists.

Two analyses.

  A  HORIZONTAL -- marginal value of each station to the rest. For every held-out station, the
     vertical profile is reconstructed from a month x depth-bin climatology built on the other
     stations, first with all of them and then with one removed. The degradation caused by the
     removal is that station's marginal contribution to estimating everywhere else. A station
     whose removal barely matters is redundant; one whose removal hurts is carrying the network.

  B  VERTICAL -- value of a near-bottom sensor. The thesis concludes that a surface-only network
     cannot recover near-bottom oxygen. This quantifies the converse: if k stations carried a
     near-bottom probe, how well could near-bottom oxygen be estimated at a station that does
     not? Near-bottom CTD bins stand in for a bottom probe. The comparison baseline is the
     operator's current default, "assume the bottom equals the surface".

Both use the full 722-cast archive. Neither extrapolates to unmeasured coordinates: a placement
claim about a location with no observations would not be testable, and none is made.
"""
import json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "bay_twin" / "runs" / "v2_placement"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260919)


def o2_sol(T, S=38.0):
    T = np.clip(np.asarray(T, float), -2, 40)
    Ts = np.log((298.15 - T) / (273.15 + T))
    A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    return np.exp(A[0] + A[1]*Ts + A[2]*Ts**2 + A[3]*Ts**3 + A[4]*Ts**4 + A[5]*Ts**5
                  + S*(B[0] + B[1]*Ts + B[2]*Ts**2 + B[3]*Ts**3) - 2.75915e-7*S**2)


def load():
    c = pd.read_csv(ROOT / "data/processed/ctd_all_casts.csv.gz", parse_dates=["t_utc"])
    c = c[c.qc_flag == 0].dropna(subset=["temp_c", "do_umol_l", "depth_m"]).copy()
    c["t_utc"] = pd.to_datetime(c.t_utc, utc=True)
    c["db"] = np.floor(c.depth_m)
    return c


def clim_predict(train, test, var):
    clim = train.groupby([train.t_utc.dt.month, "db"])[var].mean()
    fb = train.groupby(train.t_utc.dt.month)[var].mean()
    gm = train[var].mean()
    return np.array([clim.get((m, d), fb.get(m, gm))
                     for m, d in zip(test.t_utc.dt.month, test.db)], dtype=float)


# ---------------------------------------------------------------- analysis A ----
def horizontal(c):
    stations = sorted(c.station.unique())
    rows = []
    for held in stations:
        te = c[c.station == held]
        if len(te) < 20:
            continue
        pool = [s for s in stations if s != held]
        base = {v: float(np.abs(clim_predict(c[c.station.isin(pool)], te, v) - te[v].values).mean())
                for v in ("temp_c", "do_umol_l")}
        for drop in pool:
            sub = [s for s in pool if s != drop]
            tr = c[c.station.isin(sub)]
            if tr.cast_id.nunique() < 20:
                continue
            for v in ("temp_c", "do_umol_l"):
                mae = float(np.abs(clim_predict(tr, te, v) - te[v].values).mean())
                rows.append(dict(held_station=held, removed_station=drop, variable=v,
                                 mae_with=base[v], mae_without=mae,
                                 degradation=mae - base[v], n_points=len(te)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "placement_horizontal.csv", index=False)
    summary = (d.groupby(["removed_station", "variable"])
                 .agg(mean_degradation=("degradation", "mean"),
                      folds=("held_station", "nunique")).reset_index())
    summary.to_csv(OUT / "placement_horizontal_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------- analysis B ----
def build_pairs(c):
    recs = []
    for cid, g in c.groupby("cast_id"):
        g = g.sort_values("depth_m")
        surf, bot = g[g.depth_m <= 2.0], g.tail(3)
        if len(surf) == 0 or len(bot) == 0:
            continue
        if bot.depth_m.min() - surf.depth_m.max() < 2.0:
            continue
        recs.append(dict(cast_id=cid, station=g.station.iloc[0], t_utc=g.t_utc.iloc[0],
                         month=g.t_utc.iloc[0].month,
                         surf_DO=float(surf.do_umol_l.mean()), surf_T=float(surf.temp_c.mean()),
                         bot_DO=float(bot.do_umol_l.mean()), bot_T=float(bot.temp_c.mean())))
    d = pd.DataFrame(recs)
    d["bot_AOU"] = o2_sol(d.bot_T) - d.bot_DO
    return d


def vertical(pairs):
    """How well is near-bottom oxygen at a held station estimated when k other stations
    carry a near-bottom probe? Donors are matched to the held cast by nearest time."""
    stations = sorted(pairs.station.unique())
    rows = []
    for held in stations:
        te = pairs[pairs.station == held]
        if len(te) < 10:
            continue
        donors_all = [s for s in stations if s != held]
        for k in range(0, len(donors_all) + 1):
            combos = list(itertools.combinations(donors_all, k))
            if len(combos) > 12:
                combos = [tuple(donors_all[i] for i in RNG.choice(len(donors_all), k, replace=False))
                          for _ in range(12)]
            for combo in combos:
                preds = []
                for _, r in te.iterrows():
                    if k == 0:
                        preds.append(r.surf_DO)          # operator default: bottom = surface
                        continue
                    don = pairs[pairs.station.isin(combo)]
                    near = don.iloc[(don.t_utc - r.t_utc).abs().to_numpy().argsort()[:3]]
                    if len(near) == 0:
                        preds.append(r.surf_DO); continue
                    # transfer the donors' surface-to-bottom oxygen offset to the held site
                    offset = float((near.bot_DO - near.surf_DO).mean())
                    preds.append(r.surf_DO + offset)
                mae = float(np.abs(np.asarray(preds) - te.bot_DO.values).mean())
                rows.append(dict(held_station=held, n_donor_stations=k,
                                 donors=",".join(combo), mae=mae, n_casts=len(te)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "placement_vertical.csv", index=False)
    summary = (d.groupby("n_donor_stations")
                 .agg(mean_mae=("mae", "mean"), median_mae=("mae", "median"),
                      folds=("held_station", "nunique"), configs=("mae", "size")).reset_index())
    summary.to_csv(OUT / "placement_vertical_summary.csv", index=False)
    return d, summary


if __name__ == "__main__":
    c = load()
    print(f"casts {c.cast_id.nunique()}  stations {c.station.nunique()}")

    print("\n=== A. horizontal: mean degradation caused by removing each station ===")
    h = horizontal(c)
    for v, unit in (("temp_c", "degC"), ("do_umol_l", "umol/L")):
        s = h[h.variable == v].sort_values("mean_degradation", ascending=False)
        print(f"\n  {v} ({unit}) -- larger = more load-bearing")
        print(s[["removed_station", "mean_degradation", "folds"]].round(4).to_string(index=False))

    pairs = build_pairs(c)
    print(f"\n=== B. vertical: {len(pairs)} casts with a usable surface/bottom pair ===")
    dv, sv = vertical(pairs)
    print("\n  near-bottom oxygen MAE (umol/L) vs number of stations carrying a bottom probe")
    print("  (0 donors = the operator's current default, assume bottom equals surface)")
    print(sv.round(3).to_string(index=False))
    best1 = dv[dv.n_donor_stations == 1].groupby("donors").mae.mean().sort_values()
    print("\n  if exactly ONE station gets a bottom probe, mean held-station MAE by choice:")
    print(best1.round(3).to_string())

    (OUT / "placement_summary.json").write_text(json.dumps({
        "horizontal": h.to_dict("records"),
        "vertical_by_k": sv.to_dict("records"),
        "best_single_bottom_probe": {k: float(v) for k, v in best1.items()},
    }, indent=2, default=str))
    print("\nWROTE", OUT)
