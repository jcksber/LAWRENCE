# Databricks notebook source
# MAGIC %md
# MAGIC # 04d — v2 development: do flow + metadata-host features beat v1? (branch `social`)
# MAGIC
# MAGIC **Development, not a test.** Uses only data ≤ 2026-10-10 (the v1 shadow window
# MAGIC is never touched). Whatever wins here gets frozen as contract v2 BEFORE
# MAGIC `SHADOW_START` and is then judged on the same unseen shadow window as v1.
# MAGIC
# MAGIC Fixed up front (no tuning on these results):
# MAGIC - rows: post-graduation, labelled 6h barrier, ≤ 18h after graduation (rule 7),
# MAGIC   `assessment_time` ≥ 2026-09-13 (the flow stats exist from then on);
# MAGIC - model: v1's LightGBM spec (seed 17, 300 trees, lr 0.05) on `r_up_360m`;
# MAGIC - rule: v1's (EV ≥ train-p95 cutoff, one position per token, base costs);
# MAGIC - folds: 4 weekly test blocks from 2026-09-13, each trained on ALL earlier
# MAGIC   rows (so fold 1 trains on 09-13 → 09-15 only — reported, not dropped);
# MAGIC - exactly 3 variants: **V1** (v1 features), **V2a** (+ flow), **V2b** (+ flow + host).
# MAGIC - decision rule: pick the variant with the highest pooled mean net per trade
# MAGIC   ONLY if it also beats V1 in ≥ 3 of 4 folds; otherwise v2 = no change.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pyspark.sql import functions as F

sys.path.append("../src")
from backtest.shadow_eval_v1 import block_bootstrap_lower_bound, non_overlapping  # noqa: E402
from models.barrier_model import FEATURE_COLUMNS as V1_FEATURES  # noqa: E402
from serving.contract_v1 import POSTGRAD_OBSERVED_HOURS, round_trip_cost_v1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
START = pd.Timestamp("2026-09-13", tz="UTC")
END = pd.Timestamp("2026-10-11", tz="UTC")
N = 360

FLOW = ["curve_buy_share", "curve_max_trade_share", "curve_top_trader_share", "curve_top_trader_share_max",
        "minutes_since_graduation", "flow_buy_share_15m", "flow_max_trade_share_15m", "flow_top_trader_share_15m"]
HOST = ["meta_host_group", "meta_is_launch_tool"]
VARIANTS = {"V1": V1_FEATURES, "V2a": V1_FEATURES + FLOW, "V2b": V1_FEATURES + FLOW + HOST}
CATEGORICAL = ["chain_family", "network", "dex_venue", "dex_protocol", "meta_host_group"]

cols = list(dict.fromkeys(["token_id", "assessment_time", *V1_FEATURES, *FLOW, "r_up_360m", "t_up_360m"]))
g = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true AND r_up_360m IS NOT NULL")
     .select(*cols))
first_pg = (spark.table("lawrence.silver.token_price_1m").filter("phase = 'postgrad'")
            .groupBy("token_id").agg(F.min("bar_start").alias("first_pg")))
host = spark.table("lawrence.gold.social_token_features_v2")
df = g.join(first_pg, "token_id", "left").join(host, "token_id", "left").toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df["first_pg"] = pd.to_datetime(df["first_pg"], utc=True)
df = df[(df["assessment_time"] >= START) & (df["assessment_time"] < END)
        & (df["assessment_time"] + pd.Timedelta(minutes=N) <= df["first_pg"] + pd.Timedelta(hours=POSTGRAD_OBSERVED_HOURS))]
df = df.sort_values("assessment_time").reset_index(drop=True)
for c in CATEGORICAL:
    df[c] = df[c].astype("category")
df["is_postgrad_at_t"] = df["is_postgrad_at_t"].astype(float)
df["meta_is_launch_tool"] = df["meta_is_launch_tool"].astype(float)
df["cost_base"] = [round_trip_cost_v1(v, True) for v in df["dex_venue"].astype(str)]
print(f"{len(df):,} rows, {df['token_id'].nunique():,} tokens, {df['assessment_time'].min()} -> {df['assessment_time'].max()}")
print(df.groupby(df["dex_venue"].astype(str)).size())

# COMMAND ----------

edges = [START + pd.Timedelta(days=7 * k) for k in range(5)]  # 09-13, 09-20, 09-27, 10-04, 10-11
# first test block starts after 3 days of training so fold 1 has something to learn from
folds = [(START + pd.Timedelta(days=3), edges[1])] + [(edges[i], edges[i + 1]) for i in range(1, 4)]

results, pooled = [], {k: [] for k in VARIANTS}
for fi, (lo, hi) in enumerate(folds, start=1):
    tr = df[df["assessment_time"] < lo]
    te = df[(df["assessment_time"] >= lo) & (df["assessment_time"] < hi)]
    for name, feats in VARIANTS.items():
        reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1)
        reg.fit(tr[feats], tr["r_up_360m"])
        cut = float(np.quantile(reg.predict(tr[feats]), 0.95))
        ev = reg.predict(te[feats])
        sig = te.assign(score=ev)[ev >= cut]
        t = non_overlapping(sig)
        net = t["r_up_360m"] - t["cost_base"]
        pooled[name].append(t.assign(net=net, fold=fi))
        rho = pd.Series(ev).corr(te["r_up_360m"].reset_index(drop=True), method="spearman")
        results.append({"fold": fi, "test": f"{lo:%m-%d}→{hi:%m-%d}", "variant": name, "train_rows": len(tr),
                        "trades": len(t), "mean_net": float(net.mean()) if len(t) else np.nan,
                        "hit_2x": float((t["r_up_360m"] >= 0.999).mean()) if len(t) else np.nan,
                        "spearman": rho, "cutoff": cut})
res = pd.DataFrame(results)
display(res.pivot(index="fold", columns="variant", values="mean_net").round(4))
display(res.round(4))

# COMMAND ----------

summary = []
for name in VARIANTS:
    t = pd.concat(pooled[name])
    lb = block_bootstrap_lower_bound(t, t["net"]) if len(t) else np.nan
    wins = (res[res.variant == name].set_index("fold")["mean_net"] > res[res.variant == "V1"].set_index("fold")["mean_net"]).sum()
    summary.append({"variant": name, "trades": len(t), "pooled_mean_net": float(t["net"].mean()),
                    "lower_bound_90": lb, "folds_beating_V1": int(wins) if name != "V1" else None,
                    "median_spearman": float(res[res.variant == name]["spearman"].median())})
summary = pd.DataFrame(summary)
display(summary.round(4))
best = summary.sort_values("pooled_mean_net", ascending=False).iloc[0]
choice = best["variant"] if best["variant"] == "V1" or best["folds_beating_V1"] >= 3 else "V1"
print("DECISION (pre-set rule):", choice)

# COMMAND ----------

# Feature importance for the chosen variant on all rows (descriptive only).
feats = VARIANTS[choice]
reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(df[feats], df["r_up_360m"])
display(pd.DataFrame({"feature": feats, "gain": reg.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False).round(1))
