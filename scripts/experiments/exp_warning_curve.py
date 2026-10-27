"""Operating curve for the rapid oxygen-decline warning policy.

The released evaluation fixes one operating point (rate > 1.5 umol/L/h sustained two hours,
6 h cooldown) and reports that it fails: ~6.5 % precision, 6.19 unmatched alerts per site-week.
(Those are its own figures, under its symmetric +/- 24 h matching. With the causal matching below,
the same point scores 0.041 and 7.14 on 2026: table_bay_warning_curve.)
That is true but uninformative -- it rejects the capability without characterising it. An
operator cannot tell from one point whether a usable trade-off exists elsewhere.

This script sweeps the policy and reports the achievable frontier. It also corrects a matching
definition: the frozen evaluation counts an alert as matched if a crossing occurs within +/- 24 h,
which credits alerts raised AFTER the event. For an early-warning claim only alerts that PRECEDE
the crossing count, so both definitions are reported.

Protocol: operating points are selected on 2025 development data, and the selected points are
then read once on 2026. The full 2026 frontier is also reported as a post-hoc diagnostic and is
explicitly not used for selection.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "bay_twin" / "runs" / "v2_warning_curve"
OUT.mkdir(parents=True, exist_ok=True)

SENSORS = ["TBaiaP_01m-CTD(calibrato)", "TMitiliE_05m-CTD(calibrato)",
           "TPortovenereP_02m-CTD(calibrato)", "TMitiliO_05m-CTD(calibrato)",
           "TPortovenereS_02m-CTD(calibrato)"]
EVENT_DROP, EVENT_WIN, MATCH_H = 30.0, 6, 24

RATES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
CONSEC = [1, 2, 3]
COOLDOWN = [3, 6, 12]


def load():
    obs = pd.read_csv(ROOT / "reports/bay_twin/hourly_observations.csv.gz", index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    out = {}
    for s in SENSORS:
        c = f"{s}::oxygen"
        if c not in obs.columns:
            continue
        v = obs[c].dropna()
        v = v[v.between(0, 600)]
        if len(v) < 24:
            continue
        out[s] = v
    return out


def period(series, lo, hi):
    v = series[(series.index >= lo) & (series.index < hi)]
    if len(v) < 24:
        return None
    idx = pd.date_range(v.index.min(), v.index.max(), freq="1h", tz="UTC")
    return v.reindex(idx), len(v)


def evaluate(hourly, n_obs, rate, consec, cooldown):
    d = -(hourly - hourly.shift(1))
    cond = d > rate
    for k in range(1, consec):
        cond = cond & (d.shift(k) > rate)
    alerts, last = [], None
    for t, val in cond.dropna().items():
        if val and (last is None or (t - last) >= pd.Timedelta(hours=cooldown)):
            alerts.append(t); last = t
    drop6 = -(hourly - hourly.shift(EVENT_WIN))
    crossings = list(drop6[drop6 > EVENT_DROP].dropna().index)

    def matched(sym):
        m = 0
        for a in alerts:
            for c in crossings:
                dt = (c - a).total_seconds() / 3600.0
                if (abs(dt) <= MATCH_H) if sym else (0 <= dt <= MATCH_H):
                    m += 1; break
        return m

    def caught(sym):
        n = 0
        for c in crossings:
            for a in alerts:
                dt = (c - a).total_seconds() / 3600.0
                if (abs(dt) <= MATCH_H) if sym else (0 <= dt <= MATCH_H):
                    n += 1; break
        return n

    leads = []
    for c in crossings:
        cand = [(c - a).total_seconds() / 3600.0 for a in alerts
                if 0 <= (c - a).total_seconds() / 3600.0 <= MATCH_H]
        if cand:
            leads.append(max(cand))
    weeks = n_obs / 168.0
    res = {}
    for tag, sym in (("sym", True), ("causal", False)):
        m, k = matched(sym), caught(sym)
        res[f"precision_{tag}"] = m / len(alerts) if alerts else np.nan
        res[f"recall_{tag}"] = k / len(crossings) if crossings else np.nan
        res[f"unmatched_per_week_{tag}"] = (len(alerts) - m) / weeks if weeks else np.nan
    res.update(alerts=len(alerts), crossings=len(crossings), weeks=weeks,
               median_lead_h=float(np.median(leads)) if leads else np.nan)
    return res


def sweep(data, lo, hi, label):
    rows = []
    for rate in RATES:
        for consec in CONSEC:
            for cd in COOLDOWN:
                agg = dict(alerts=0, crossings=0, weeks=0.0, matched_sym=0, matched_causal=0,
                           caught_sym=0, caught_causal=0, leads=[])
                for s, ser in data.items():
                    p = period(ser, lo, hi)
                    if p is None:
                        continue
                    hourly, n_obs = p
                    r = evaluate(hourly, n_obs, rate, consec, cd)
                    agg["alerts"] += r["alerts"]; agg["crossings"] += r["crossings"]
                    agg["weeks"] += r["weeks"]
                    for tag in ("sym", "causal"):
                        if r["alerts"]:
                            agg[f"matched_{tag}"] += r[f"precision_{tag}"] * r["alerts"]
                        if r["crossings"]:
                            agg[f"caught_{tag}"] += r[f"recall_{tag}"] * r["crossings"]
                    if np.isfinite(r["median_lead_h"]):
                        agg["leads"].append(r["median_lead_h"])
                row = dict(period=label, rate=rate, consec=consec, cooldown=cd,
                           alerts=agg["alerts"], crossings=agg["crossings"],
                           weeks=round(agg["weeks"], 1),
                           median_lead_h=float(np.median(agg["leads"])) if agg["leads"] else np.nan)
                for tag in ("sym", "causal"):
                    row[f"precision_{tag}"] = (agg[f"matched_{tag}"] / agg["alerts"]) if agg["alerts"] else np.nan
                    row[f"recall_{tag}"] = (agg[f"caught_{tag}"] / agg["crossings"]) if agg["crossings"] else np.nan
                    row[f"unmatched_per_week_{tag}"] = ((agg["alerts"] - agg[f"matched_{tag}"]) / agg["weeks"]) if agg["weeks"] else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load()
    print(f"oxygen sensors with data: {len(data)}")
    dev = sweep(data, pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"), "development")
    res = sweep(data, pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-06-20", tz="UTC"), "reserved_2026")
    dev.to_csv(OUT / "warning_curve_development.csv", index=False)
    res.to_csv(OUT / "warning_curve_reserved2026.csv", index=False)

    frozen = dev[(dev.rate == 1.5) & (dev.consec == 2) & (dev.cooldown == 6)]
    frozen26 = res[(res.rate == 1.5) & (res.consec == 2) & (res.cooldown == 6)]
    print("\n=== the released operating point ===")
    for nm, f in (("development", frozen), ("reserved 2026", frozen26)):
        if len(f):
            r = f.iloc[0]
            print(f"  {nm:14s} alerts={r.alerts:4.0f} crossings={r.crossings:3.0f} "
                  f"prec_sym={r.precision_sym:.3f} prec_causal={r.precision_causal:.3f} "
                  f"recall_causal={r.recall_causal:.3f} unmatched/wk={r.unmatched_per_week_causal:.2f}")

    # select on development only: highest causal precision among points with recall >= 0.5
    cand = dev[(dev.recall_causal >= 0.5) & (dev.alerts > 0)]
    print(f"\n=== development points with causal recall >= 0.50: {len(cand)} ===")
    if len(cand):
        best = cand.sort_values("precision_causal", ascending=False).head(5)
        print(best[["rate", "consec", "cooldown", "precision_causal", "recall_causal",
                    "unmatched_per_week_causal", "median_lead_h"]].round(3).to_string(index=False))
        sel = best.iloc[0]
        m = res[(res.rate == sel.rate) & (res.consec == sel.consec) & (res.cooldown == sel.cooldown)]
        print(f"\n=== selected on development (rate={sel.rate}, consec={sel.consec}, "
              f"cooldown={sel.cooldown}h) -> read once on reserved 2026 ===")
        if len(m):
            r = m.iloc[0]
            print(f"  precision_causal={r.precision_causal:.3f}  recall_causal={r.recall_causal:.3f}  "
                  f"unmatched/wk={r.unmatched_per_week_causal:.2f}  median lead={r.median_lead_h:.1f} h")
        sel_out = dict(selected_on="development", rate=float(sel.rate), consec=int(sel.consec),
                       cooldown=int(sel.cooldown),
                       reserved_2026={k: (float(m.iloc[0][k]) if len(m) and np.isfinite(m.iloc[0][k]) else None)
                                      for k in ["precision_causal", "recall_causal",
                                                "unmatched_per_week_causal", "median_lead_h",
                                                "alerts", "crossings"]})
    else:
        sel_out = {"selected_on": "development", "result": "no point reaches causal recall 0.50"}

    print("\n=== best achievable causal precision on RESERVED 2026 (post-hoc diagnostic) ===")
    r26 = res[res.alerts > 0].sort_values("precision_causal", ascending=False).head(5)
    print(r26[["rate", "consec", "cooldown", "alerts", "crossings", "precision_causal",
               "recall_causal", "unmatched_per_week_causal"]].round(3).to_string(index=False))

    (OUT / "warning_selection.json").write_text(json.dumps(sel_out, indent=2, default=str))
    print("\nWROTE", OUT)
