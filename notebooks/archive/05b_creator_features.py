# Databricks notebook source
# MAGIC %md
# MAGIC # 05b — v3 part C: creator + early-buyer features (branch `v3`)
# MAGIC Builds two NEW Gold tables from 00l's Bronze pulls (nothing existing changes;
# MAGIC v1/v2 features and `r_up_360m` are untouched):
# MAGIC - `lawrence.gold.creator_token_features_v3` — one row per pump.fun token (`token_id` = mint).
# MAGIC   Known within 3 slots of creation, so causal at any post-graduation T:
# MAGIC   - early buyers: `eb_*` (slot-0 / first-3-slot buyers and supply shares, creator's slot-0 share)
# MAGIC   - launch: `launch_is_mayhem`, `launch_non_sol_quote`
# MAGIC   - creator history AT CREATION (7d lookback, strictly before this token's creation):
# MAGIC     `cr_launches_7d`, `cr_grads_7d` (graduated before this creation), `cr_grad_rate_7d`.
# MAGIC     7 days because launch history starts 08-30 and flow tokens start 09-06.
# MAGIC - `lawrence.gold.creator_flow_features_v3` — one row per (token_id, assessment_time):
# MAGIC   the creator's own trades in this token strictly BEFORE T (`ct_*`).
# MAGIC All inputs ≤ 2026-10-10 (00l caps every read below 10-11). Cell 4 is a coverage /
# MAGIC sanity check only -- no model is fit here and nothing here counts toward the
# MAGIC v3 variant budget.

# COMMAND ----------

from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")
SUPPLY = 1_000_000_000.0
LAUNCHES = spark.table("lawrence.bronze.pump_launches").dropDuplicates(["mint"])
EB = spark.table("lawrence.bronze.pump_early_buyers").dropDuplicates(["mint"])
CT = spark.table("lawrence.bronze.pump_creator_trades")
POOLS = spark.table("lawrence.bronze.pool_metadata").filter("venue = 'pumpswap'")
GOLD = (spark.table("lawrence.gold.entry_candidates")
        .filter("is_postgrad_at_t = true AND dex_venue = 'pumpdotfun' AND r_up_360m IS NOT NULL")
        .select("token_id", "assessment_time"))
TOKEN_OUT = "lawrence.gold.creator_token_features_v3"
FLOW_OUT = "lawrence.gold.creator_flow_features_v3"
LOOKBACK_DAYS = 7
# 2026-10-18: first run gave launch_non_sol_quote = 1.0 for EVERY token, so CreateEvent's
# quote_mint for SOL launches isn't the WSOL mint. Treat both SOL spellings as SOL and check
# the value counts printed in the token cell.
SOL_QUOTES = ["So11111111111111111111111111111111111111112", "11111111111111111111111111111111"]

# COMMAND ----------

# Token-level table
grad_at = POOLS.groupBy(F.col("base_mint").alias("mint")).agg(F.min("created_at").alias("grad_at"))
L = LAUNCHES.select("mint", "creator", "created_at", "quote_mint", "is_mayhem_mode")

# creator history at this token's creation: the creator's OTHER launches in [created_at - 7d, created_at)
# only graduates (tokens with early-buyer rows) on the left -- a launch-bot creator with
# thousands of launches would otherwise make an N^2 self-join
me = L.join(EB.select("mint"), "mint").select(F.col("mint").alias("token_id"), "creator", F.col("created_at").alias("t0"))
other = L.select(F.col("mint").alias("o_mint"), "creator", F.col("created_at").alias("o_created")) \
         .join(grad_at.withColumnRenamed("mint", "o_mint"), "o_mint", "left")
hist = (me.join(other, "creator")
          .filter((F.col("o_created") < F.col("t0"))
                  & (F.col("o_created") >= F.col("t0") - F.expr(f"INTERVAL {LOOKBACK_DAYS} DAYS")))
          .groupBy("token_id")
          .agg(F.count("*").alias("cr_launches_7d"),
               F.sum(F.when(F.col("grad_at") < F.col("t0"), 1).otherwise(0)).alias("cr_grads_7d")))

eb = EB.select(
    F.col("mint").alias("token_id"),
    F.col("buyers_slot0").cast("double").alias("eb_buyers_slot0"),
    F.col("non_creator_buyers_slot0").cast("double").alias("eb_non_creator_buyers_slot0"),
    F.col("buyers_first3slots").cast("double").alias("eb_buyers_first3"),
    F.coalesce("supply_share_slot0", F.lit(0.0)).alias("eb_supply_share_slot0"),
    F.coalesce("supply_share_first3slots", F.lit(0.0)).alias("eb_supply_share_first3"),
    F.coalesce("creator_share_slot0", F.lit(0.0)).alias("eb_creator_share_slot0"),
)
tok = (eb.join(L.select(F.col("mint").alias("token_id"), "creator", "created_at", "quote_mint", "is_mayhem_mode"),
               "token_id", "left")
         .join(hist, "token_id", "left")
         .withColumn("launch_is_mayhem", F.col("is_mayhem_mode").cast("double"))
         .withColumn("launch_non_sol_quote",
                     F.when(F.col("quote_mint").isNull(), None)
                      .otherwise((~F.col("quote_mint").isin(*SOL_QUOTES)).cast("double")))
         .fillna(0, subset=["cr_launches_7d", "cr_grads_7d"])
         .withColumn("cr_grad_rate_7d", F.when(F.col("cr_launches_7d") > 0,
                                               F.col("cr_grads_7d") / F.col("cr_launches_7d")))
         .drop("is_mayhem_mode", "quote_mint"))
tok.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TOKEN_OUT)
t = spark.table(TOKEN_OUT)
display(LAUNCHES.join(EB.select("mint"), "mint").groupBy("quote_mint").count().orderBy(F.desc("count")).limit(10))
print(f"{TOKEN_OUT}: {t.count():,} tokens | with launch row: {t.filter('created_at IS NOT NULL').count():,}")
display(t.describe())

# COMMAND ----------

# Row-level table: creator's trades in this token strictly before T
ct = CT.select(F.col("mint").alias("token_id"), F.to_timestamp("block_time").alias("block_time"),
               F.col("is_buy").cast("boolean").alias("is_buy"),
               (F.col("base_amt").cast("double") / SUPPLY).alias("share"), "project")
# Gold's assessment_time is an ISO string ('2026-09-06T20:16:00+00:00'); keep it as the join key
# and compare on a parsed copy `_t`.
rows = (GOLD.join(t.select("token_id").distinct(), "token_id")  # pump.fun graduates with early-buyer rows
            .withColumn("_t", F.to_timestamp("assessment_time")))
flow = (rows.join(ct, "token_id", "left")
            .withColumn("_in", F.col("block_time") < F.col("_t"))
            .groupBy("token_id", "assessment_time", "_t")
            .agg(F.sum(F.when(F.col("_in") & F.col("is_buy"), 1).otherwise(0)).cast("double").alias("ct_buys_n"),
                 F.sum(F.when(F.col("_in") & ~F.col("is_buy"), 1).otherwise(0)).cast("double").alias("ct_sells_n"),
                 F.sum(F.when(F.col("_in") & F.col("is_buy"), F.col("share")).otherwise(0.0)).alias("ct_bought_share"),
                 F.sum(F.when(F.col("_in") & ~F.col("is_buy"), F.col("share")).otherwise(0.0)).alias("ct_sold_share"),
                 F.sum(F.when(F.col("_in") & ~F.col("is_buy") & (F.col("project") == "pumpswap"), 1).otherwise(0))
                  .cast("double").alias("ct_postgrad_sells_n"),
                 F.max(F.when(F.col("_in") & ~F.col("is_buy"), F.col("block_time"))).alias("_last_sell"))
            .withColumn("ct_net_share", F.col("ct_bought_share") - F.col("ct_sold_share"))
            .withColumn("ct_min_since_last_sell",
                        (F.col("_t").cast("long") - F.col("_last_sell").cast("long")) / 60.0)
            .drop("_last_sell", "_t"))
flow.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(FLOW_OUT)
f = spark.table(FLOW_OUT)
print(f"{FLOW_OUT}: {f.count():,} rows | Gold pump.fun post-grad rows: {GOLD.count():,}")
display(f.describe())

# COMMAND ----------

# Coverage + sanity (training window only; no model fit). Rows in the v2 window
# that get token features, and mean r_up_360m by "creator bought the whole curve in slot 0".
g = (spark.table("lawrence.gold.entry_candidates")
     .filter("is_postgrad_at_t = true AND dex_venue = 'pumpdotfun' AND r_up_360m IS NOT NULL")
     .filter("assessment_time >= '2026-09-13' AND assessment_time < '2026-10-11'")
     .select("token_id", "assessment_time", "r_up_360m"))
j = g.join(t, "token_id", "left")
print(f"v2-window pump.fun rows: {g.count():,} | with eb features: {j.filter('eb_buyers_slot0 IS NOT NULL').count():,}")
display(j.withColumn("self_grad", F.col("eb_creator_share_slot0") >= 0.79)
         .groupBy("self_grad").agg(F.countDistinct("token_id").alias("tokens"), F.count("*").alias("rows"),
                                   F.avg("r_up_360m").alias("mean_r_up_360m"))
         .orderBy("self_grad"))
