# Databricks notebook source
# MAGIC %md
# MAGIC # 02c — Gold: post-crash entry candidates (v4 post-crash track, branch `v3`)
# MAGIC Hourly assessment times for graduated pump.fun tokens 24h–14d after graduation, from
# MAGIC `bronze.postcrash_bars_1h` (00n) + each pool's first-24h high/low from `bronze.postgrad_bars_1m`.
# MAGIC Features + 72h barrier label (2x before −50%) from `features.postcrash_features` (7 tests).
# MAGIC Pools graduated ≥ 2026-09-12 only (hourly history complete from age 24h). Bars end 10-11, so the
# MAGIC last labelled T is 10-08. Writes NEW table `lawrence.gold.postcrash_candidates_v4`. No Dune credits.

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

sys.path.append("../src")
import importlib  # noqa: E402

import features.postcrash_features as pcf  # noqa: E402

importlib.reload(pcf)
spark.conf.set("spark.sql.session.timeZone", "UTC")
OUT = "lawrence.gold.postcrash_candidates_v4"
GRAD_FROM = "2026-09-12"
DATA_END = pd.Timestamp("2026-10-11", tz="UTC")

hb = (spark.table("lawrence.bronze.postcrash_bars_1h")
      .filter(F.col("pool_created_at") >= GRAD_FROM).filter("close_usd IS NOT NULL")
      .dropDuplicates(["pool_id", "bar_start"])
      .select("pool_id", "base_mint", F.col("pool_created_at").alias("grad_at"), "bar_start",
              F.col("high_usd").alias("high"), F.col("low_usd").alias("low"), F.col("close_usd").alias("close"),
              "volume_usd", "n_trades", "n_traders"))
first24 = (spark.table("lawrence.bronze.postgrad_bars_1m").filter("venue = 'pumpswap'")
           .join(hb.select("pool_id", "grad_at").distinct(), "pool_id")
           .filter(F.col("bar_start") < F.col("grad_at") + F.expr("INTERVAL 24 HOURS"))
           .groupBy("pool_id").agg(F.max("high_usd").alias("ath0"), F.min("low_usd").alias("low0")))
pumpfun = spark.table("lawrence.bronze.pump_launches").select(F.col("mint").alias("base_mint")).distinct()
src = hb.join(pumpfun, "base_mint").join(first24, "pool_id", "left")
print("pools:", src.select("pool_id").distinct().count(), "| bars:", src.count())

# COMMAND ----------

SCHEMA = ("pool_id string, token_id string, grad_at timestamp, assessment_time timestamp, price_at_t double, "
          + ", ".join(f"{c} double" for c in pcf.FEATURES)
          + ", r_pc double, outcome string, t_pc_h double, r_pc_next double")


def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
    g = pd.Timestamp(pdf["grad_at"].iloc[0])
    g = g.tz_localize("UTC") if g.tzinfo is None else g
    r = pcf.pool_rows(pdf, g, float(pdf["ath0"].iloc[0]) if pd.notna(pdf["ath0"].iloc[0]) else np.nan,
                      float(pdf["low0"].iloc[0]) if pd.notna(pdf["low0"].iloc[0]) else np.nan, DATA_END)
    if r.empty:
        return pd.DataFrame(columns=[c.split()[0] for c in SCHEMA.split(", ")])
    r["pool_id"], r["token_id"], r["grad_at"] = pdf["pool_id"].iloc[0], pdf["base_mint"].iloc[0], g
    return r[[c.split()[0] for c in SCHEMA.split(", ")]]


src.groupBy("pool_id").applyInPandas(_apply, schema=SCHEMA) \
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT)
t = spark.table(OUT)
print(f"{OUT}: {t.count():,} rows, {t.select('token_id').distinct().count():,} tokens")
display(t.groupBy("outcome").agg(F.count("*").alias("rows"), F.avg("r_pc").alias("mean_r")))
