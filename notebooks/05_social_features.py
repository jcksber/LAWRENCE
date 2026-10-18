# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Social features (v2 candidates, branch `social`)
# MAGIC
# MAGIC **Checks first, features second.** Cells 2–4 decide whether each feature
# MAGIC family is usable at all; cell 5 builds `lawrence.gold.social_features_v2`
# MAGIC keyed `(token_id, assessment_time)` for pump.fun post-graduation rows.
# MAGIC Not part of contract v1. Don't run while 02 is running (cluster load only;
# MAGIC it reads Gold, so run it AFTER 02 finishes to get the rebuilt table).
# MAGIC
# MAGIC Uses training-period data only (≤ 2026-10-10). The v1 shadow window is
# MAGIC never looked at here.

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append("../src")
import importlib  # noqa: E402

import features.social_features as sf  # noqa: E402

importlib.reload(sf)
spark.conf.set("spark.sql.session.timeZone", "UTC")

COINS = "lawrence.bronze.pumpfun_coins"
CALLOUTS = "lawrence.bronze.pumpfun_callouts"
URIS = "lawrence.bronze.pumpfun_coin_uris"
PROGRESS = "lawrence.bronze._pumpfun_social_progress"
GOLD = "lawrence.gold.entry_candidates"
OUT_TABLE = "lawrence.gold.social_features_v2"
TRAIN_END = pd.Timestamp("2026-10-11", tz="UTC")

# latest progress row per mint; latest coin row per mint (prefer resolved metadata)
prog = (spark.table(PROGRESS).withColumn("_rn", F.row_number().over(
    Window.partitionBy("mint").orderBy(F.desc("completed_at")))).filter("_rn = 1").drop("_rn"))
coins = (spark.table(COINS).withColumn("_rn", F.row_number().over(
    Window.partitionBy("mint").orderBy(F.desc("has_launch_metadata"), F.desc("ingested_at")))).filter("_rn = 1").drop("_rn"))

# COMMAND ----------

# CHECK A -- mint counts (35,102 distinct coins vs ~34.9k expected).
print("progress mints:", prog.count(), "| coin rows:", spark.table(COINS).count(),
      "| distinct coin mints:", spark.table(COINS).select("mint").distinct().count(),
      "| uri mints:", spark.table(URIS).select("mint").distinct().count())
grad = (spark.table("lawrence.silver.token_price_1m").groupBy("token_id")
        .agg(F.min_by("venue", "bar_start").alias("lv"), F.max((F.col("phase") == "postgrad").cast("int")).alias("g"))
        .filter("lv = 'pumpdotfun' AND g = 1").select(F.col("token_id").alias("mint")))
c_m = spark.table(COINS).select("mint").distinct()
print("coin mints NOT in Silver graduated pump.fun:", c_m.join(grad, "mint", "left_anti").count(),
      "| graduated pump.fun NOT in coins:", grad.join(c_m, "mint", "left_anti").count())
display(spark.table(COINS).groupBy("mint").count().filter("count > 1").groupBy("count").count())

# COMMAND ----------

# CHECK B -- truncation simulation: are a coin's first 20 callouts (by time)
# kept when the API keeps only the top half by post-hoc performance?
co_pd = spark.table(CALLOUTS).select("mint", "callout_id", "created_at", "posthoc_max_multiplier") \
    .dropDuplicates(["mint", "callout_id"]).toPandas()
co_pd.attrs = {}
for frac in (0.5, 0.75):
    r = sf.truncation_check(co_pd, keep_frac=frac)
    print(f"keep top {frac:.0%}: coins={len(r):,}  first-20 kept share: mean={r['first_k_kept_share'].mean():.3f} "
          f"p10={r['first_k_kept_share'].quantile(0.1):.3f}  all-kept={(r['first_k_kept_share'] == 1).mean():.3f}")
# Pre-set rule: if mean >= 0.95 AND p10 >= 0.85 at keep 50%, capped counts
# are used for ALL coins (truncated ones included). Otherwise the callout
# family is DROPPED from v2 entirely -- setting it to NaN only for truncated
# coins would mark exactly the high-callout winners (truncation is post-hoc).

# COMMAND ----------

# CHECK C -- does metadata resolution (post-hoc) or host group track outcomes?
# Token-level outcome on its FIRST post-grad candidate row (one row per token).
g = (spark.table(GOLD).filter("is_postgrad_at_t = true AND dex_venue = 'pumpdotfun' AND r_up_360m IS NOT NULL")
     .withColumn("_rn", F.row_number().over(Window.partitionBy("token_id").orderBy("assessment_time")))
     .filter("_rn = 1").select(F.col("token_id").alias("mint"), "r_up_360m", "y_up_360m"))
uris = spark.table(URIS).dropDuplicates(["mint"]).select("mint", "metadata_host")
joined = g.join(coins.select("mint", "has_launch_metadata", "launch_twitter"), "mint", "left").join(uris, "mint", "left")
display(joined.groupBy("has_launch_metadata").agg(F.count("*").alias("tokens"), F.avg("y_up_360m").alias("hit_2x"),
                                                 F.avg("r_up_360m").alias("mean_r")))
display(joined.groupBy("metadata_host").agg(F.count("*").alias("tokens"), F.avg("y_up_360m").alias("hit_2x"),
                                           F.avg("r_up_360m").alias("mean_r")).orderBy(F.desc("tokens")).limit(15))
# Pre-set rule: if hit_2x for has_launch_metadata=false differs from =true by
# more than 3 points WITHIN the ipfs.io host (the one host with both), the
# meta_launch_has_* socials are dropped from v2 (resolution would leak through
# the NaN pattern). Host group itself is launch-time and stays.
display(joined.filter("metadata_host = 'ipfs.io'").groupBy("has_launch_metadata")
        .agg(F.count("*").alias("tokens"), F.avg("y_up_360m").alias("hit_2x"), F.avg("r_up_360m").alias("mean_r")))

# COMMAND ----------

# BUILD -- gates B and C both FAILED (docs/decisions.md 2026-10-18), so only
# the launch-time metadata host survives. One row per token (it doesn't change
# with T, and it's on-chain at creation, so it's known at every T):
# lawrence.gold.social_token_features_v2(token_id, meta_host_group, meta_is_launch_tool).
uri_pd = spark.table(URIS).dropDuplicates(["mint"]).join(coins.select("mint", "created_at"), "mint", "left") \
    .select("mint", "metadata_host", "created_at").toPandas()
uri_pd.attrs = {}
GROUPS = sf.build_host_groups(uri_pd["metadata_host"], uri_pd["created_at"], TRAIN_END)
print(len(GROUPS), "hosts mapped ->", sorted(set(GROUPS.values())))
rows = []
for r in uri_pd.itertuples(index=False):
    m = sf.meta_features(r.metadata_host, GROUPS, None, None, None, has_launch_metadata=False)
    rows.append({"token_id": r.mint, **{k: m[k] for k in sf.V2_SOCIAL_FEATURES}})
OUT_TABLE = "lawrence.gold.social_token_features_v2"
(spark.createDataFrame(pd.DataFrame(rows))
 .write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT_TABLE))
display(spark.table(OUT_TABLE).groupBy("meta_host_group", "meta_is_launch_tool").count().orderBy(F.desc("count")))
