# Databricks notebook source
# MAGIC %md
# MAGIC # 01b — Silver: `lawrence.silver.token_price_1m`
# MAGIC
# MAGIC One 1-minute price series per token, in ONE unit per token, stitched
# MAGIC from bonding-curve/AMM trades (`silver.token_trades`) and
# MAGIC post-graduation bars (`bronze.postgrad_bars_1m`, canonical pool only).
# MAGIC Logic + rationale: `src/features/silver_price_bars.py`. This is what
# MAGIC Gold will read for labels/features instead of raw trades.
# MAGIC
# MAGIC **Run after** `01_silver_token_trades.py` (MODE = "rebuild") so
# MAGIC token_trades already has the pool-metadata token-of-interest fix and
# MAGIC `price_quote`. Full rebuild every run (set-based, partitioned overwrite).

# COMMAND ----------

import sys

sys.path.append("../src")
# Force-reload (same reason as notebook 01): don't run stale cached code.
import importlib  # noqa: E402

import features.silver_price_bars  # noqa: E402

importlib.reload(features.silver_price_bars)
from features.silver_price_bars import USD_COVERAGE_MIN, build_token_price_1m  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")

TRADES_TABLE = "lawrence.silver.token_trades"
BARS_TABLE = "lawrence.bronze.postgrad_bars_1m"
POOLS_TABLE = "lawrence.bronze.pool_metadata"
PRICE_TABLE = "lawrence.silver.token_price_1m"

# COMMAND ----------

price_df = build_token_price_1m(
    spark.table(TRADES_TABLE),
    spark.table(BARS_TABLE),
    spark.table(POOLS_TABLE),
)
(
    price_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("trade_date")
    .saveAsTable(PRICE_TABLE)
)
print(f"Rebuilt {PRICE_TABLE} (USD_COVERAGE_MIN = {USD_COVERAGE_MIN}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks

# COMMAND ----------

p = spark.table(PRICE_TABLE)

# 1. Size and shape by unit and phase.
display(
    p.groupBy("price_unit", "phase")
    .agg(
        F.count("*").alias("bars"),
        F.countDistinct("token_id").alias("tokens"),
        F.sum("n_trades").alias("trades"),
        F.min("bar_start").alias("min_bar"),
        F.max("bar_start").alias("max_bar"),
    )
    .orderBy("price_unit", "phase")
)

# 2. The point of the whole exercise: how many tokens now have a
#    post-graduation tail (curve -> postgrad), i.e. no longer "vanish when
#    they win".
per_token = p.groupBy("token_id").agg(
    F.max(F.when(F.col("phase") == "curve", 1).otherwise(0)).alias("has_curve"),
    F.max(F.when(F.col("phase") == "postgrad", 1).otherwise(0)).alias("has_postgrad"),
)
display(per_token.groupBy("has_curve", "has_postgrad").count().orderBy("has_curve", "has_postgrad"))

# 3. Sanity: non-positive or absurd prices (should be ~0 rows). A unit mix-up
#    shows up as huge close-to-close jumps within a token.
display(
    p.agg(
        F.count(F.when(F.col("close") <= 0, 1)).alias("nonpositive_close"),
        F.count(F.when(F.col("high") < F.col("low"), 1)).alias("high_below_low"),
    )
)
