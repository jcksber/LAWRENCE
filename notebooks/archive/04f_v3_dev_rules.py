# Databricks notebook source
# MAGIC %md
# MAGIC # 04f — v3 development, part B: cost- and selectivity-aware RULES (branch `v3`)
# MAGIC
# MAGIC Same rows, folds and model as 04d's V2b (v2 features, v1 LightGBM spec,
# MAGIC 4 weekly walk-forward folds over 2026-09-13 → 10-10, rule 7). The model is
# MAGIC trained ONCE per fold; only the trade-selection rule changes. Data ≤ 2026-10-10
# MAGIC only, so the holdout window is untouched.
# MAGIC
# MAGIC **Every variant run here is logged in `docs/v3_variant_log.md`**, good or bad.
# MAGIC The v3 budget is ~10 variants in total across B/C/D, and these 8 rules count.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pyspark.sql import functions as F

sys.path.append("../../src")
from backtest.shadow_eval_v1 import block_bootstrap_lower_bound, non_overlapping  # noqa: E402
from models import v2_spec as v2  # noqa: E402
from serving.contract_v1 import POSTGRAD_OBSERVED_HOURS, round_trip_cost_v1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
N = 360
cols = list(dict.fromkeys(["token_id", "assessment_time", *[c for c in v2.V2_FEATURES if c not in v2.HOST_FEATURES],
                           "r_up_360m", "t_up_360m"]))
g = spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true AND r_up_360m IS NOT NULL").select(*cols)
first_pg = (spark.table("lawrence.silver.token_price_1m").filter("phase = 'postgrad'")
            .groupBy("token_id").agg(F.min("bar_start").alias("first_pg")))
df = g.join(first_pg, "token_id", "left").join(spark.table("lawrence.gold.social_token_features_v2"), "token_id", "left").toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df["first_pg"] = pd.to_datetime(df["first_pg"], utc=True)
df = df[(df["assessment_time"] >= v2.TRAIN_START) & (df["assessment_time"] < v2.TRAIN_END)
        & (df["assessment_time"] + pd.Timedelta(minutes=N) <= df["first_pg"] + pd.Timedelta(hours=POSTGRAD_OBSERVED_HOURS))]
df = df.sort_values("assessment_time").reset_index(drop=True)
categories = {c: sorted(str(v) for v in df[c].dropna().unique()) for c in v2.CATEGORICAL}
X_all = v2.to_frame(df, categories)
venue = df["dex_venue"].astype(str)
df["cost_base"] = [round_trip_cost_v1(v, True) for v in venue]
df["cost_stress"] = [round_trip_cost_v1(v, True, 100.0) for v in venue]
print(f"{len(df):,} rows; cost base by venue:", df.groupby(venue)["cost_base"].first().to_dict())

# COMMAND ----------

S = v2.TRAIN_START
edges = [S + pd.Timedelta(days=7 * k) for k in range(5)]
folds = [(S + pd.Timedelta(days=3), edges[1])] + [(edges[i], edges[i + 1]) for i in range(1, 4)]
scored = []  # one frame per fold: test rows + score + train cutoffs
for fi, (lo, hi) in enumerate(folds, start=1):
    tr = df["assessment_time"] < lo
    te = (df["assessment_time"] >= lo) & (df["assessment_time"] < hi)
    reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X_all[tr], df.loc[tr, "r_up_360m"])
    p_tr, p_te = reg.predict(X_all[tr]), reg.predict(X_all[te])
    net_tr = p_tr - df.loc[tr, "cost_base"].to_numpy()
    t = df[te].copy()
    t["score"], t["net_score"], t["fold"] = p_te, p_te - t["cost_base"], fi
    t.attrs["cut"] = {q: float(np.quantile(p_tr, q)) for q in (0.95, 0.98, 0.99)}
    t.attrs["net_cut95"] = float(np.quantile(net_tr, 0.95))
    scored.append(t)
    print(f"fold {fi} {lo:%m-%d}→{hi:%m-%d}: train {tr.sum():,} test {te.sum():,}")

# COMMAND ----------

def _top_k(t, k, col="score", min_score=None):
    s = t if min_score is None else t[t[col] >= min_score]
    s = s.assign(_d=s["assessment_time"].dt.floor("D")).sort_values(["_d", col], ascending=[True, False])
    s = s.drop_duplicates(["_d", "token_id"])
    return s.groupby("_d", group_keys=False).head(k).drop(columns="_d")


RULES = {  # name -> fn(fold frame) -> selected signals (before one-position-per-token)
    "R0 v2 rule (p95, all venues)": lambda t: t[t["score"] >= t.attrs["cut"][0.95]],
    "R1 p98": lambda t: t[t["score"] >= t.attrs["cut"][0.98]],
    "R2 p99": lambda t: t[t["score"] >= t.attrs["cut"][0.99]],
    "R3 LaunchLab only, p95": lambda t: t[(t["score"] >= t.attrs["cut"][0.95]) & (t["dex_venue"].astype(str) == "raydium_launchlab")],
    "R4 pump.fun only, p95": lambda t: t[(t["score"] >= t.attrs["cut"][0.95]) & (t["dex_venue"].astype(str) == "pumpdotfun")],
    "R5 cost-aware: EV-cost >= train p95 of EV-cost": lambda t: t[t["net_score"] >= t.attrs["net_cut95"]],
    "R6 top-2/day by EV-cost (EV-cost > 0)": lambda t: _top_k(t[t["net_score"] > 0], 2, "net_score"),
    "R7 top-5/day by EV-cost (EV-cost > 0)": lambda t: _top_k(t[t["net_score"] > 0], 5, "net_score"),
}

rows, pooled = [], {}
for name, rule in RULES.items():
    parts = []
    for t in scored:
        sel = non_overlapping(rule(t))
        sel = sel.assign(net=sel["r_up_360m"] - sel["cost_base"], net_s=sel["r_up_360m"] - sel["cost_stress"])
        parts.append(sel)
        rows.append({"rule": name, "fold": int(t["fold"].iloc[0]), "trades": len(sel),
                     "mean_net": float(sel["net"].mean()) if len(sel) else np.nan})
    p = pd.concat(parts)
    pooled[name] = p
res = pd.DataFrame(rows)
fold_tbl = res.pivot(index="rule", columns="fold", values="mean_net").round(4)
summary = pd.DataFrame([{
    "rule": n, "trades": len(p), "trades_per_day": round(len(p) / 25, 1),
    "mean_net_base": p["net"].mean(), "mean_net_stress": p["net_s"].mean(),
    "lower_bound_90": block_bootstrap_lower_bound(p, p["net"]) if len(p) >= 30 else np.nan,
    "hit_2x": (p["r_up_360m"] >= 0.999).mean(), "folds_positive": int((fold_tbl.loc[n] > 0).sum()),
    "launchlab_share": (p["dex_venue"].astype(str) == "raydium_launchlab").mean(),
} for n, p in pooled.items()]).round(4)
display(summary)
display(fold_tbl)

# COMMAND ----------

# Paste-ready block for docs/v3_variant_log.md
print("| rule | trades | /day | mean net base | mean net stress | LB90 | folds>0 |")
print("|---|---|---|---|---|---|---|")
for r in summary.itertuples(index=False):
    print(f"| {r.rule} | {r.trades} | {r.trades_per_day} | {r.mean_net_base:+.4f} | {r.mean_net_stress:+.4f} | "
          f"{r.lower_bound_90:+.4f} | {r.folds_positive}/4 |")

# COMMAND ----------

# MAGIC %md
# MAGIC ## R9 — causal version of R7 (variant 9 of the budget)
# MAGIC R7 ranked against the whole UTC day, including later candidates, so it's not tradeable.
# MAGIC R9: trade if EV−cost ≥ the 5th-highest per-token EV−cost of the PREVIOUS 24h (hourly
# MAGIC refresh) and > 0, first-come up to 5 per day, one position per token. History for each
# MAGIC fold's first day = the last 24h of its training rows, scored by the same fold model.
# MAGIC k = 5 is fixed (no sweep). Also prints concentration diagnostics for R7 and R9.

# COMMAND ----------

from backtest.causal_rules import rolling_top_k  # noqa: E402

r9_parts = []
for fi, (lo, hi) in enumerate(folds, start=1):
    tr = df["assessment_time"] < lo
    te = (df["assessment_time"] >= lo) & (df["assessment_time"] < hi)
    tail = tr & (df["assessment_time"] >= lo - pd.Timedelta(hours=24))
    reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X_all[tr], df.loc[tr, "r_up_360m"])
    hist = df[tail | te][["token_id", "assessment_time", "cost_base"]].copy()
    hist["net_score"] = reg.predict(X_all[tail | te]) - hist["cost_base"]
    cands = df[te].copy()
    cands["net_score"] = reg.predict(X_all[te]) - cands["cost_base"]
    sel = rolling_top_k(cands, hist, k=5)
    sel = sel.assign(net=sel["r_up_360m"] - sel["cost_base"], net_s=sel["r_up_360m"] - sel["cost_stress"], fold=fi)
    r9_parts.append(sel)
    print(f"fold {fi}: R9 trades {len(sel)}, mean net {sel['net'].mean():+.4f}")
r9 = pd.concat(r9_parts)
pooled["R9 causal top-5 (prev-24h threshold)"] = r9


def _conc(p):
    per_tok = p.groupby("token_id")["net"].sum().sort_values(ascending=False)
    total = per_tok.sum()
    return {"trades": len(p), "tokens": p["token_id"].nunique(), "mean_net": p["net"].mean(),
            "mean_net_stress": p["net_s"].mean(),
            "lb90": block_bootstrap_lower_bound(p, p["net"]) if len(p) >= 30 else np.nan,
            "mean_net_ex_top1_token": p[p["token_id"] != per_tok.index[0]]["net"].mean(),
            "mean_net_ex_top5_tokens": p[~p["token_id"].isin(per_tok.index[:5])]["net"].mean(),
            "top5_tokens_share_of_pnl": per_tok.head(5).sum() / total if total > 0 else np.nan,
            "positive_days_share": (p.groupby(p["assessment_time"].dt.floor("D"))["net"].sum() > 0).mean(),
            "folds_positive": int((p.groupby("fold")["net"].mean() > 0).sum())}


display(pd.DataFrame({n: _conc(pooled[n]) for n in ("R7 top-5/day by EV-cost (EV-cost > 0)",
                                                     "R9 causal top-5 (prev-24h threshold)")}).T.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variant 10 — V2b + part C features (creator + early buyers), rule R1 fixed
# MAGIC Declared in `docs/v3_variant_log.md` before this cell was run. Same rows, folds, LightGBM
# MAGIC params and R1 (p98 of the fold's own train predictions). Only change: the C feature
# MAGIC columns from 05b are added (NaN for LaunchLab rows and tokens without early-buyer data).
# MAGIC **Adopt only if** pooled mean net AND its 95% lower bound beat V2b's R1, and ≥3 of 4
# MAGIC folds improve. No feature subset is re-fit here.

# COMMAND ----------

C_TOKEN = ["eb_buyers_slot0", "eb_non_creator_buyers_slot0", "eb_buyers_first3", "eb_supply_share_slot0",
           "eb_supply_share_first3", "eb_creator_share_slot0", "cr_launches_7d", "cr_grads_7d",
           "cr_grad_rate_7d", "launch_is_mayhem", "launch_non_sol_quote"]
C_FLOW = ["ct_buys_n", "ct_sells_n", "ct_bought_share", "ct_sold_share", "ct_postgrad_sells_n",
          "ct_net_share", "ct_min_since_last_sell"]
C_FEATURES = C_TOKEN + C_FLOW

tokf = spark.table("lawrence.gold.creator_token_features_v3").select("token_id", *C_TOKEN).toPandas()
flowf = spark.table("lawrence.gold.creator_flow_features_v3").select("token_id", "assessment_time", *C_FLOW).toPandas()
tokf.attrs = {}
flowf.attrs = {}
flowf["assessment_time"] = pd.to_datetime(flowf["assessment_time"], utc=True)
dfc = df.merge(tokf, on="token_id", how="left").merge(flowf, on=["token_id", "assessment_time"], how="left")
assert len(dfc) == len(df) and (dfc["token_id"].values == df["token_id"].values).all()
pf = dfc["dex_venue"].astype(str) == "pumpdotfun"
print(f"pump.fun rows with eb features: {dfc.loc[pf, 'eb_buyers_slot0'].notna().mean():.1%}; "
      f"with ct features: {dfc.loc[pf, 'ct_buys_n'].notna().mean():.1%}")
X_c = pd.concat([X_all, dfc[C_FEATURES].astype("float")], axis=1)

v10_parts, imps = [], []
for fi, (lo, hi) in enumerate(folds, start=1):
    tr = dfc["assessment_time"] < lo
    te = (dfc["assessment_time"] >= lo) & (dfc["assessment_time"] < hi)
    reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X_c[tr], dfc.loc[tr, "r_up_360m"])
    cut98 = float(np.quantile(reg.predict(X_c[tr]), 0.98))
    t = dfc[te].copy()
    t["score"] = reg.predict(X_c[te])
    sel = non_overlapping(t[t["score"] >= cut98])
    sel = sel.assign(net=sel["r_up_360m"] - sel["cost_base"], net_s=sel["r_up_360m"] - sel["cost_stress"], fold=fi)
    v10_parts.append(sel)
    imps.append(pd.Series(reg.booster_.feature_importance("gain"), index=X_c.columns, name=f"f{fi}"))
v10 = pd.concat(v10_parts)
base = pooled["R1 p98"].assign(fold=lambda d: d["fold"].astype(int))


def _row(name, p):
    return {"model": name, "trades": len(p), "mean_net_base": p["net"].mean(), "mean_net_stress": p["net_s"].mean(),
            "lb95": block_bootstrap_lower_bound(p, p["net"], alpha=0.05) if len(p) >= 30 else np.nan,
            "hit_2x": (p["r_up_360m"] >= 0.999).mean(),
            "launchlab_share": (p["dex_venue"].astype(str) == "raydium_launchlab").mean()}


cmp = pd.DataFrame([_row("V2b R1", base), _row("V10 = V2b + C, R1", v10)]).round(4)
folds_cmp = pd.DataFrame({"V2b": base.groupby("fold")["net"].mean(), "V10": v10.groupby("fold")["net"].mean(),
                          "V2b_n": base.groupby("fold").size(), "V10_n": v10.groupby("fold").size()}).round(4)
folds_cmp["improved"] = folds_cmp["V10"] > folds_cmp["V2b"]
display(cmp)
display(folds_cmp)
a, b = cmp.iloc[0], cmp.iloc[1]
adopt = (b.mean_net_base > a.mean_net_base) and (b.lb95 > a.lb95) and (folds_cmp["improved"].sum() >= 3)
print(f"DECISION (pre-declared rule): {'ADOPT' if adopt else 'REJECT'} variant 10 | "
      f"folds improved {int(folds_cmp['improved'].sum())}/4")
imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
display((imp / imp.sum()).head(20).round(4).rename("gain_share").to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exit variants E1 (ladder, v3 variant 11) and E2 (half at 2x, rest at 10x, variant 12)
# MAGIC Declared in `docs/v3_variant_log.md` before running. Same ENTRIES as V2b + R1 (553 trades above);
# MAGIC only the exit changes. E0 = current label (sell all at 2x / −50% / 6h). E1 = at each doubling of the
# MAGIC reference price sell half of what's left, then reset reference, stop (−50%) and the 6h clock.
# MAGIC Simulated from Silver 1-minute bars (`backtest.ladder_exit`, conservative fills, costs on every sale).
# MAGIC E0 is re-simulated too: it must reproduce `r_up_360m` (simulator check). Post-graduation bars stop
# MAGIC 24h after graduation, so a ladder still open then is closed at the last price (`truncated`).
# MAGIC **Adopt E1 only if** pooled mean net AND 95% LB beat E0 at base costs, mean net beats E0 at stress,
# MAGIC and ≥3 of 4 folds improve.

# COMMAND ----------

from backtest.ladder_exit import simulate  # noqa: E402

tr_ = pooled["R1 p98"].copy()
keys = spark.createDataFrame(tr_[["token_id"]].drop_duplicates())
gold_px = (spark.table("lawrence.gold.entry_candidates").join(keys, "token_id")
           .select("token_id", "assessment_time", "price_at_t", "price_unit").toPandas())
gold_px.attrs = {}
gold_px["assessment_time"] = pd.to_datetime(gold_px["assessment_time"], utc=True)
tr_ = tr_.merge(gold_px, on=["token_id", "assessment_time"], how="left")
bars = (spark.table("lawrence.silver.token_price_1m").join(keys, "token_id").filter("phase = 'postgrad'")
        .select("token_id", "price_unit", "bar_start", "high", "low", "close").toPandas())
bars.attrs = {}
bars["ts"] = pd.to_datetime(bars["bar_start"], utc=True).astype("int64") / 1e9 + 60  # bar CLOSE time
bars = bars.sort_values("ts")
by_key = {k: (g["ts"].to_numpy(), g["high"].to_numpy(float), g["low"].to_numpy(float), g["close"].to_numpy(float))
          for k, g in bars.groupby(["token_id", "price_unit"])}

rows = []
for r in tr_.itertuples(index=False):
    arr = by_key.get((r.token_id, r.price_unit))
    if arr is None or pd.isna(r.price_at_t):
        continue
    t0 = r.assessment_time.timestamp()
    end = (r.first_pg + pd.Timedelta(hours=24)).timestamp()
    for side_name, side in (("base", r.cost_base / 2), ("stress", r.cost_stress / 2)):
        e0 = simulate(*arr, t0, float(r.price_at_t), side, ladder=False, data_end=end)
        e1 = simulate(*arr, t0, float(r.price_at_t), side, ladder=True, data_end=end)
        e2 = simulate(*arr, t0, float(r.price_at_t), side, ladder=True, data_end=end, final_mult=10.0, final_stop_mult=1.1)
        rows.append({"token_id": r.token_id, "assessment_time": r.assessment_time, "fold": int(r.fold),
                     "costs": side_name, "r_label": r.r_up_360m, "e0_gross": e0["gross"], "e0_net": e0["net"],
                     "e1_net": e1["net"], "e1_rungs": e1["rungs"], "e1_trunc": e1["truncated"],
                     "e2_net": e2["net"], "e2_exit": e2["exit"], "e2_rungs": e2["rungs"], "e2_trunc": e2["truncated"]})
sim = pd.DataFrame(rows)
b = sim[sim["costs"] == "base"]
s = sim[sim["costs"] == "stress"]
print(f"simulated {len(b)} of {len(tr_)} trades; E0 reproduces the label on "
      f"{(abs(b['e0_gross'] - b['r_label']) < 1e-6).mean():.1%} (mean |diff| {abs(b['e0_gross'] - b['r_label']).mean():.4f})")
lb = lambda d, col: block_bootstrap_lower_bound(d, d[col], alpha=0.05)
exit_cmp = pd.DataFrame([
    {"exit": "E0 sell all at 2x", "mean_net_base": b["e0_net"].mean(), "lb95": lb(b, "e0_net"),
     "mean_net_stress": s["e0_net"].mean(), "median_net": b["e0_net"].median()},
    {"exit": "E1 ladder", "mean_net_base": b["e1_net"].mean(), "lb95": lb(b, "e1_net"),
     "mean_net_stress": s["e1_net"].mean(), "median_net": b["e1_net"].median()},
    {"exit": "E2 half at 2x, rest at 10x (stop 1.1x)", "mean_net_base": b["e2_net"].mean(), "lb95": lb(b, "e2_net"),
     "mean_net_stress": s["e2_net"].mean(), "median_net": b["e2_net"].median()},
]).round(4)
fold_exit = b.groupby("fold").agg(E0=("e0_net", "mean"), E1=("e1_net", "mean"), E2=("e2_net", "mean"),
                                  n=("e0_net", "size")).round(4)
fold_exit["E1_improved"] = fold_exit["E1"] > fold_exit["E0"]
fold_exit["E2_improved"] = fold_exit["E2"] > fold_exit["E0"]
display(exit_cmp)
display(fold_exit)
print("E1 rungs reached:", b["e1_rungs"].value_counts().sort_index().to_dict(),
      f"| truncated at 24h data end: {b['e1_trunc'].mean():.1%}")
print("E2 exits:", b["e2_exit"].value_counts().to_dict(), f"| reached 2x: {(b['e2_rungs'] >= 1).mean():.1%}",
      f"| truncated: {b['e2_trunc'].mean():.1%}")
a0 = exit_cmp.iloc[0]
passed = {}
for k, name in ((1, "E1"), (2, "E2")):
    a = exit_cmp.iloc[k]
    passed[name] = (a.mean_net_base > a0.mean_net_base) and (a.lb95 > a0.lb95) \
        and (a.mean_net_stress > a0.mean_net_stress) and (fold_exit[f"{name}_improved"].sum() >= 3)
    print(f"DECISION (pre-declared rule): {'PASS' if passed[name] else 'REJECT'} {name} | "
          f"folds improved {int(fold_exit[f'{name}_improved'].sum())}/4")
if all(passed.values()):
    print("Both pass -> adopt the one with the higher LB95 (pre-declared tie-break)")
