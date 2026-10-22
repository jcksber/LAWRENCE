# Databricks notebook source
# MAGIC %md
# MAGIC # 05c — v4: creator history for EVERY pump.fun launch (branch `v3`)
# MAGIC v4 early track enters on the bonding curve 1–15 min after creation, so features must
# MAGIC exist for all launches, not just graduates (the `eb_*` early-buyer table covers graduates
# MAGIC only -- using it here would leak "this token graduates").
# MAGIC Writes `lawrence.gold.creator_launch_history_v4` (one row per mint), all known AT creation:
# MAGIC - `cr4_launches_7d`, `cr4_launches_1h`: creator's earlier launches in the window
# MAGIC - `cr4_grads_7d`: creator's tokens that graduated in [t0 - 7d, t0) (graduation time =
# MAGIC   first PumpSwap pool), `cr4_grad_rate_7d` = grads / launches
# MAGIC - `cr4_min_since_prev_launch`
# MAGIC - `launch_is_mayhem`, `launch_non_sol_quote`
# MAGIC Window functions per creator (no self-join: launch bots have thousands of launches).
# MAGIC Launch history starts 08-30, so the 7d features are complete for creations ≥ 09-06.
# MAGIC No Dune credits.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")
OUT = "lawrence.gold.creator_launch_history_v4"
SOL_QUOTES = ["So11111111111111111111111111111111111111112", "11111111111111111111111111111111"]
D7, H1 = 7 * 86400, 3600

L = (spark.table("lawrence.bronze.pump_launches").dropDuplicates(["mint"])
     .select("mint", "creator", F.to_timestamp("created_at").alias("t0"), "quote_mint", "is_mayhem_mode")
     .withColumn("ts", F.col("t0").cast("long")))
grad = (spark.table("lawrence.bronze.pool_metadata").filter("venue = 'pumpswap'")
        .groupBy(F.col("base_mint").alias("mint")).agg(F.min("created_at").alias("grad_at")))
# graduation events, attributed to the creator, at their graduation time
g_ev = (L.select("mint", "creator").join(grad, "mint")
         .select("creator", F.to_timestamp("grad_at").cast("long").alias("ts"),
                 F.lit(0).alias("is_launch"), F.lit(1).alias("is_grad"), F.lit(None).cast("string").alias("mint")))
l_ev = L.select("creator", "ts", F.lit(1).alias("is_launch"), F.lit(0).alias("is_grad"), "mint")
ev = l_ev.unionByName(g_ev)

# order: at equal timestamps put grads BEFORE launches, and exclude the row itself via -1 offsets
w = Window.partitionBy("creator").orderBy("ts")
w7 = w.rangeBetween(-D7, -1)
w1h = w.rangeBetween(-H1, -1)
feat = (ev.withColumn("cr4_launches_7d", F.coalesce(F.sum("is_launch").over(w7), F.lit(0)))
          .withColumn("cr4_launches_1h", F.coalesce(F.sum("is_launch").over(w1h), F.lit(0)))
          .withColumn("cr4_grads_7d", F.coalesce(F.sum("is_grad").over(w7), F.lit(0)))
          .withColumn("_prev_launch_ts", F.max(F.when(F.col("is_launch") == 1, F.col("ts"))).over(
              w.rangeBetween(Window.unboundedPreceding, -1)))
          .filter("is_launch = 1")
          .select("mint", "cr4_launches_7d", "cr4_launches_1h", "cr4_grads_7d",
                  ((F.col("ts") - F.col("_prev_launch_ts")) / 60.0).alias("cr4_min_since_prev_launch")))
# Same-second launches by one creator see each other only via the -1 offset rule (strictly earlier).

out = (L.join(feat, "mint", "left")
        .withColumn("cr4_grad_rate_7d", F.when(F.col("cr4_launches_7d") > 0,
                                               F.col("cr4_grads_7d") / F.col("cr4_launches_7d")))
        .withColumn("launch_is_mayhem", F.col("is_mayhem_mode").cast("double"))
        .withColumn("launch_non_sol_quote", F.when(F.col("quote_mint").isNull(), None)
                    .otherwise((~F.col("quote_mint").isin(*SOL_QUOTES)).cast("double")))
        .select(F.col("mint").alias("token_id"), "creator", "t0", "cr4_launches_7d", "cr4_launches_1h",
                "cr4_grads_7d", "cr4_grad_rate_7d", "cr4_min_since_prev_launch",
                "launch_is_mayhem", "launch_non_sol_quote"))
out.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT)
t = spark.table(OUT)
print(f"{OUT}: {t.count():,} launches")
display(t.filter("t0 >= '2026-09-06'").describe())
