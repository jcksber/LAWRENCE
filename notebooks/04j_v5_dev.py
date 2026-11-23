# Databricks notebook source
# MAGIC %md
# MAGIC # 04j — v5 development (plan + decision rule: `docs/v5_plan.md`, written before this ran)
# MAGIC Both models trade with exit **E3** and are judged on E3 net return (`gold.e3_labels_v5`).
# MAGIC - **B0 (baseline):** v2 features, target = the old label `r_up_360m`, entries = top 2% of its train predictions.
# MAGIC - **v5:** v2 features + 11 market-context features, target = **E3 net return**, same entry rule.
# MAGIC 6 weekly walk-forward folds, test weeks 09-20 → 11-01; a fold trains only on rows whose E3 outcome was known
# MAGIC before the fold starts. One position per token (until its E3 exit). No Dune credits.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pyspark.sql import functions as F

sys.path.append("../src")
from backtest.shadow_eval_v1 import block_bootstrap_lower_bound  # noqa: E402
from features.market_context import FEATURES as CTX  # noqa: E402
from features.social_features import LAUNCH_TOOL_HOSTS, host_group  # noqa: E402
from models import v2_spec as v2  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
m2 = json.load(open("../contracts/v2/manifest.json"))
groups = json.load(open("../contracts/v2/artifacts/" + m2["host_groups_file"]))
base_cols = [c for c in v2.V2_FEATURES if c not in v2.HOST_FEATURES]
g = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true")
     .select("token_id", "assessment_time", *base_cols, "r_up_360m", "t_up_360m")
     .join(spark.table("lawrence.gold.e3_labels_v5"), ["token_id", "assessment_time"])
     .withColumn("ctx_hour", F.date_trunc("hour", F.to_timestamp("assessment_time")))
     .join(spark.table("lawrence.gold.market_context_v5"), "ctx_hour", "left")
     .join(spark.table("lawrence.bronze.pumpfun_coin_uris").dropDuplicates(["mint"])
           .select(F.col("mint").alias("token_id"), "metadata_host"), "token_id", "left"))
df = g.toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df = df.sort_values("assessment_time").reset_index(drop=True)
df["meta_host_group"] = [host_group(h, groups) if isinstance(h, str) else None for h in df["metadata_host"]]
df["meta_is_launch_tool"] = [float(h in LAUNCH_TOOL_HOSTS) if isinstance(h, str) else None for h in df["metadata_host"]]
X0 = v2.to_frame(df, m2["categories"])
X5 = pd.concat([X0, df[CTX].astype(float)], axis=1)
df["e3_end"] = df["assessment_time"] + pd.to_timedelta(df["e3_hours"], unit="h")
df["e0_end"] = df["assessment_time"] + pd.to_timedelta(df["t_up_360m"].fillna(360), unit="m")
print(f"{len(df):,} rows, {df['token_id'].nunique():,} tokens, {df['assessment_time'].min()} → {df['assessment_time'].max()}; "
      f"context coverage {df['mkt_grads_24h'].notna().mean():.1%}")

# COMMAND ----------

def non_overlap(sig: pd.DataFrame) -> pd.DataFrame:
    sig = sig.sort_values(["token_id", "assessment_time"])
    keep, until = [], {}
    for idx, tok, ts, end in zip(sig.index, sig["token_id"], sig["assessment_time"], sig["e3_end"]):
        if tok in until and ts < until[tok]:
            continue
        keep.append(idx)
        until[tok] = end
    return sig.loc[keep]


S = pd.Timestamp("2026-09-13", tz="UTC")
folds = [(S + pd.Timedelta(days=7 * k), S + pd.Timedelta(days=7 * (k + 1))) for k in range(1, 7)]  # 09-20 … 11-01
models = {"B0 (v2 features, old target)": (X0, "r_up_360m", "e0_end"),
          "v5 (+ market context, E3 target)": (X5, "r_e3_net", "e3_end")}
picked, rows = {k: [] for k in models}, []
for fi, (lo, hi) in enumerate(folds, start=1):
    te = (df["assessment_time"] >= lo) & (df["assessment_time"] < hi)
    for name, (X, y, end_col) in models.items():
        tr = (df["assessment_time"] < lo) & (df[end_col] <= lo) & df[y].notna()
        reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X[tr], df.loc[tr, y])
        cut = float(np.quantile(reg.predict(X[tr]), 0.98))
        t = df[te].copy()
        t["score"] = reg.predict(X[te])
        sel = non_overlap(t[t["score"] >= cut]).assign(fold=fi)
        picked[name].append(sel)
        rows.append({"fold": fi, "model": name, "train": int(tr.sum()), "trades": len(sel), "mean_net": sel["r_e3_net"].mean()})
fold_tbl = pd.DataFrame(rows).pivot(index="fold", columns="model", values="mean_net").round(4)
display(pd.DataFrame(rows).round(4))

summ = []
for name, parts in picked.items():
    p = pd.concat(parts)
    summ.append({"model": name, "trades": len(p), "per_day": round(len(p) / 42, 1), "mean_net": p["r_e3_net"].mean(),
                 "lb95": block_bootstrap_lower_bound(p, p["r_e3_net"], alpha=0.05),
                 "median_net": p["r_e3_net"].median(), "share_positive": (p["r_e3_net"] > 0).mean(),
                 "folds_positive": int((p.groupby("fold")["r_e3_net"].mean() > 0).sum())})
summ = pd.DataFrame(summ).round(4)
display(summ)
b0, v5 = summ.iloc[0], summ.iloc[1]
improved = int((fold_tbl["v5 (+ market context, E3 target)"] > fold_tbl["B0 (v2 features, old target)"]).sum())
ok = (v5.mean_net > b0.mean_net) and (v5.lb95 > b0.lb95) and (improved >= 4) and (v5.mean_net > 0)
print(f"folds where v5 beats B0: {improved}/6")
print(f"DECISION (docs/v5_plan.md): {'FREEZE v5 for a forward paper test' if ok else 'DO NOT FREEZE — stop here'}")
