# Databricks notebook source
# MAGIC %md
# MAGIC # 02d — Gold for v5: E3 net-return labels + market context (no Dune credits; new tables only)
# MAGIC - `lawrence.gold.e3_labels_v5`: for every post-graduation Gold candidate (rule 7, T in 09-13 → 11-01), what
# MAGIC   exit E3 would have earned net of that venue's base costs (sell half of what's held at 2x/4x/6x/8x, rest at
# MAGIC   10x; stop −20% from entry; no time limit → data ends 24h after graduation). `backtest.ladder_exit.simulate_schedule`.
# MAGIC - `lawrence.gold.market_context_v5`: hourly market state (`features.market_context`), joined on floor(T, hour);
# MAGIC   uses only hours that finished before T.
# MAGIC Frozen tables and columns are untouched.

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

sys.path.append("../src")
import importlib  # noqa: E402

import backtest.ladder_exit as lx  # noqa: E402
import features.market_context as mc  # noqa: E402

importlib.reload(lx)
importlib.reload(mc)
from serving import contract_v1 as c1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
START, END = "2026-09-13", "2026-11-01"
px = spark.table("lawrence.silver.token_price_1m")

# COMMAND ----------

# Market context (hourly)
tok = (px.groupBy("token_id")
       .agg(F.min(F.when(F.col("phase") == "curve", F.col("bar_start"))).alias("launch_time"),
            F.min(F.when(F.col("phase") == "postgrad", F.col("bar_start"))).alias("grad_time"),
            F.min_by(F.when(F.col("phase") == "postgrad", F.col("price_unit")),
                     F.when(F.col("phase") == "postgrad", F.col("bar_start"))).alias("pg_unit")))
pg = px.filter("phase = 'postgrad'").join(tok.select("token_id", "grad_time", "pg_unit"), "token_id") \
       .filter(F.col("price_unit") == F.col("pg_unit"))
first_close = pg.filter(F.col("bar_start") == F.col("grad_time")).groupBy("token_id").agg(F.first("close").alias("p_grad"))
hit = (pg.filter(F.col("bar_start") < F.col("grad_time") + F.expr("INTERVAL 6 HOURS"))
         .groupBy("token_id").agg(F.max("high").alias("max_high_6h"))
         .join(first_close, "token_id")
         .withColumn("hit2x_6h", F.col("max_high_6h") >= 2 * F.col("p_grad")))
events = (tok.join(hit.select("token_id", "hit2x_6h"), "token_id", "left")
             .select("launch_time", "grad_time", F.coalesce("hit2x_6h", F.lit(False)).alias("hit2x_6h")).toPandas())
events.attrs = {}
for c in ("launch_time", "grad_time"):
    events[c] = pd.to_datetime(events[c], utc=True)
hourly = (px.filter("phase = 'postgrad'").groupBy(F.date_trunc("hour", "bar_start").alias("hour"))
            .agg(F.sum("volume_usd").alias("volume_usd"), F.sum("n_trades").alias("n_trades")).toPandas())
hourly.attrs = {}
hours = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(days=1), freq="h", tz="UTC")
ctx = mc.market_context(events, hourly, hours)
spark.createDataFrame(ctx).write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("lawrence.gold.market_context_v5")
print(f"market_context_v5: {len(ctx):,} hours; {len(events):,} tokens, {events['grad_time'].notna().sum():,} graduates")
display(ctx.describe().T.round(3))

# COMMAND ----------

# E3 labels, one pass per token over its post-graduation bars
g = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true")
     .filter((F.to_timestamp("assessment_time") >= F.lit(START)) & (F.to_timestamp("assessment_time") < F.lit(END)))
     .select("token_id", "assessment_time", "price_at_t", "price_unit", "dex_venue")
     .join(tok.select("token_id", F.col("grad_time").alias("first_pg")), "token_id")
     .filter(F.to_timestamp("assessment_time") + F.expr(f"INTERVAL {c1.HORIZON_MINUTES} MINUTES")
             <= F.col("first_pg") + F.expr(f"INTERVAL {c1.POSTGRAD_OBSERVED_HOURS} HOURS")))
bars = px.filter("phase = 'postgrad'").select("token_id", "price_unit", "bar_start", "high", "low", "close")
SCHEMA = ("token_id string, assessment_time string, r_e3_gross double, r_e3_net double, e3_exit string, "
          "e3_rungs int, e3_hours double")


def _apply(key, rows: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = []
    if rows.empty or b.empty:
        return pd.DataFrame(columns=[c.split()[0] for c in SCHEMA.split(", ")])
    b = b.assign(ts=pd.to_datetime(b["bar_start"], utc=True).astype("int64") / 1e9 + 60).sort_values("ts")
    by_unit = {u: (x["ts"].to_numpy(), x["high"].to_numpy(float), x["low"].to_numpy(float), x["close"].to_numpy(float))
               for u, x in b.groupby("price_unit")}
    for r in rows.itertuples(index=False):
        arr = by_unit.get(r.price_unit)
        if arr is None or pd.isna(r.price_at_t):
            continue
        t0 = pd.Timestamp(r.assessment_time).tz_convert("UTC").timestamp() if pd.Timestamp(r.assessment_time).tzinfo \
            else pd.Timestamp(r.assessment_time, tz="UTC").timestamp()
        fp = pd.Timestamp(r.first_pg)
        fp = fp.tz_localize("UTC") if fp.tzinfo is None else fp
        side = c1.round_trip_cost_v1(r.dex_venue, True) / 2
        e = lx.simulate_schedule(*arr, t0, float(r.price_at_t), side, data_end=(fp + pd.Timedelta(hours=24)).timestamp())
        out.append((r.token_id, str(r.assessment_time), e["gross"], e["net"], e["exit"], int(e["rungs"]),
                    (e["exit_ts"] - t0) / 3600.0))
    return pd.DataFrame(out, columns=[c.split()[0] for c in SCHEMA.split(", ")])


labels = g.groupBy("token_id").cogroup(bars.groupBy("token_id")).applyInPandas(_apply, schema=SCHEMA)
labels.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("lawrence.gold.e3_labels_v5")
t = spark.table("lawrence.gold.e3_labels_v5")
print(f"e3_labels_v5: {t.count():,} rows (Gold candidates in window: {g.count():,})")
display(t.groupBy("e3_exit").agg(F.count("*").alias("rows"), F.avg("r_e3_net").alias("mean_net"),
                                 F.expr("percentile(r_e3_net, 0.5)").alias("median_net")))
