# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Exit E3 exploration (Jack's take-profit schedule) — EXPLORATORY, not a pass/fail test
# MAGIC **E3:** sell 50% of what's held at 2x, 4x, 6x and 8x the entry price, the rest at 10x; sell everything at
# MAGIC −20% from the ENTRY price (fixed); no time limit (post-graduation bars end 24h after graduation, so
# MAGIC whatever is still held then sells at the last price). **E0:** current exit (all at 2x / −50% / 6h).
# MAGIC Entries: the frozen v2 model at the v2-R1 cutoff (one position per token), 2026-09-13 → 11-01.
# MAGIC - Development period (09-13 → 10-10): the model was TRAINED on these days, so entries are optimistic.
# MAGIC - Holdout period (10-11 → 11-01): verdict already recorded (all FAIL), so this is no longer an untouched
# MAGIC   test either. Adopting E3 would need a NEW forward window (paper engine). One rule, no tuning.
# MAGIC No Dune credits.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

sys.path.append("../src")
from backtest.ladder_exit import simulate, simulate_schedule  # noqa: E402
from backtest.shadow_eval_v1 import block_bootstrap_lower_bound, frozen_trades  # noqa: E402
from features.social_features import LAUNCH_TOOL_HOSTS, host_group  # noqa: E402
from models import v2_spec as v2  # noqa: E402
from serving import contract_v1 as c1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
m2 = json.load(open("../contracts/v2/manifest.json"))
a1 = json.load(open("../contracts/v2/amendment_r1.json"))
p2 = "../contracts/v2/artifacts/" + m2["model_file"]
assert c1.sha256_file(p2) == m2["model_sha256"]
b2 = lgb.Booster(model_file=p2)
groups = json.load(open("../contracts/v2/artifacts/" + m2["host_groups_file"]))
START, SPLIT, END = pd.Timestamp("2026-09-13", tz="UTC"), pd.Timestamp("2026-10-11", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC")

cols = list(dict.fromkeys(["token_id", "assessment_time", "price_at_t", "price_unit",
                           *[c for c in v2.V2_FEATURES if c not in v2.HOST_FEATURES], "r_up_360m", "t_up_360m"]))
g = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true").select(*cols)
     .filter((F.to_timestamp("assessment_time") >= F.lit(START.to_pydatetime()))
             & (F.to_timestamp("assessment_time") < F.lit(END.to_pydatetime()))))
first_pg = (spark.table("lawrence.silver.token_price_1m").filter("phase = 'postgrad'")
            .groupBy("token_id").agg(F.min("bar_start").alias("first_pg")))
uris = spark.table("lawrence.bronze.pumpfun_coin_uris").dropDuplicates(["mint"]).select(F.col("mint").alias("token_id"), "metadata_host")
df = g.join(first_pg, "token_id", "left").join(uris, "token_id", "left").toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df["first_pg"] = pd.to_datetime(df["first_pg"], utc=True)
df = df[df["assessment_time"] + pd.Timedelta(minutes=c1.HORIZON_MINUTES)
        <= df["first_pg"] + pd.Timedelta(hours=c1.POSTGRAD_OBSERVED_HOURS)].reset_index(drop=True)
df["meta_host_group"] = [host_group(h, groups) if isinstance(h, str) else None for h in df["metadata_host"]]
df["meta_is_launch_tool"] = [float(h in LAUNCH_TOOL_HOSTS) if isinstance(h, str) else None for h in df["metadata_host"]]
df["score"] = b2.predict(v2.to_frame(df, m2["categories"]))
trades = frozen_trades(df, a1["cutoff"]).reset_index(drop=True)
trades["period"] = np.where(trades["assessment_time"] < SPLIT, "dev (in-sample)", "holdout (already judged)")
print(trades["period"].value_counts().to_dict())

# COMMAND ----------

keys = spark.createDataFrame(trades[["token_id"]].drop_duplicates())
bars = (spark.table("lawrence.silver.token_price_1m").join(keys, "token_id").filter("phase = 'postgrad'")
        .select("token_id", "price_unit", "bar_start", "high", "low", "close").toPandas())
bars.attrs = {}
bars["ts"] = pd.to_datetime(bars["bar_start"], utc=True).astype("int64") / 1e9 + 60  # bar CLOSE time
bars = bars.sort_values("ts")
by_key = {k: (x["ts"].to_numpy(), x["high"].to_numpy(float), x["low"].to_numpy(float), x["close"].to_numpy(float))
          for k, x in bars.groupby(["token_id", "price_unit"])}

rows = []
for r in trades.itertuples(index=False):
    arr = by_key.get((r.token_id, r.price_unit))
    if arr is None or pd.isna(r.price_at_t):
        continue
    t0, end = r.assessment_time.timestamp(), (r.first_pg + pd.Timedelta(hours=24)).timestamp()
    for costs, slip in (("base", c1.SLIPPAGE_BPS_BASE), ("stress", c1.SLIPPAGE_BPS_STRESS)):
        side = c1.round_trip_cost_v1(r.dex_venue, True, slip) / 2
        e0 = simulate(*arr, t0, float(r.price_at_t), side, ladder=False, data_end=end)
        e3 = simulate_schedule(*arr, t0, float(r.price_at_t), side, data_end=end)
        rows.append({"token_id": r.token_id, "assessment_time": r.assessment_time, "period": r.period, "costs": costs,
                     "r_label": r.r_up_360m, "e0_gross": e0["gross"], "e0_net": e0["net"], "e3_net": e3["net"],
                     "e3_exit": e3["exit"], "e3_rungs": e3["rungs"]})
sim = pd.DataFrame(rows)
b = sim[sim["costs"] == "base"]
print(f"simulated {b['token_id'].size} of {len(trades)} trades; E0 reproduces the label on "
      f"{(abs(b['e0_gross'] - b['r_label']) < 1e-6).mean():.1%}")

# COMMAND ----------

lb = lambda d, c: block_bootstrap_lower_bound(d, d[c], alpha=0.05) if len(d) >= 30 else np.nan
out = []
for period, d in sim.groupby("period"):
    bb, ss = d[d["costs"] == "base"], d[d["costs"] == "stress"]
    for name, col in (("E0 all at 2x / -50% / 6h", "e0_net"), ("E3 schedule, stop -20% from entry", "e3_net")):
        out.append({"period": period, "exit": name, "trades": len(bb), "mean_net_base": bb[col].mean(),
                    "lb95_base": lb(bb, col), "mean_net_stress": ss[col].mean(), "median_net": bb[col].median(),
                    "share_positive": (bb[col] > 0).mean()})
display(pd.DataFrame(out).round(4))
display(b.groupby(["period", "e3_exit"]).size().unstack(fill_value=0))
display(b.groupby(["period", "e3_rungs"]).size().unstack(fill_value=0))
