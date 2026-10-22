# Databricks notebook source
# MAGIC %md
# MAGIC # 00n — Bronze: post-crash HOURLY bars, pools aged 24h–14d (v4 post-crash track)
# MAGIC Full dev pull after the 00m cost test (2026-10-22: one trade day, compute + export = 25 credits).
# MAGIC Trade days 2026-09-13 → 10-10 (28 jobs). Writes `lawrence.bronze.postcrash_bars_1h`.
# MAGIC **Holdout rule:** every query is capped below 2026-10-11 (`postcrash_bars`, tested).
# MAGIC **Resumable:** one job per trade day; `lawrence.bronze._postcrash_progress` records successes and a
# MAGIC re-run skips them. Out of credits → stops; swap `dune_api_key` and re-run.
# MAGIC **First run: leave `MAX_DAYS = 1`** and note the credit counter; then `None`.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append("../src")
import importlib  # noqa: E402

import ingestion.postcrash_bars as pc  # noqa: E402

importlib.reload(pc)
from ingestion.dune_client import DuneClient, DuneCreditsExhausted  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
client = DuneClient(api_key=dbutils.secrets.get(scope="lawrence", key="dune_api_key"), poll_timeout_s=1800)
UTC = timezone.utc
CUTOFF = datetime(2026, 10, 11, tzinfo=UTC)
START, END = datetime(2026, 9, 13, tzinfo=UTC), datetime(2026, 10, 11, tzinfo=UTC)
assert END <= CUTOFF
MAX_DAYS = 1  # 1 = measure first; None = everything not yet done
TABLE = "lawrence.bronze.postcrash_bars_1h"
PROGRESS = "lawrence.bronze._postcrash_progress"
TS_COLS = ("bar_start", "pool_created_at")

# COMMAND ----------

done = set()
if spark.catalog.tableExists(PROGRESS):
    done = {r.day for r in spark.table(PROGRESS).filter("status = 'success'").collect()}

ran, d = 0, START
while d < END:
    day = d.strftime("%Y-%m-%d")
    nxt = d + timedelta(days=1)
    if day in done:
        d = nxt
        continue
    if MAX_DAYS is not None and ran >= MAX_DAYS:
        break
    try:
        rows = client.run_sql(pc.postcrash_hourly_bars_sql(d, nxt, CUTOFF))
    except DuneCreditsExhausted:
        print(f"{day}: OUT OF CREDITS -- nothing written for this day. Swap dune_api_key and re-run.")
        break
    if rows:
        pdf = pd.DataFrame(rows)
        pdf.attrs = {}
        for c in TS_COLS:
            pdf[c] = pd.to_datetime(pdf[c].astype(str).str.replace(" UTC", "", regex=False), utc=True)
        pdf["trade_day"] = day
        pdf["ingested_at"] = pd.Timestamp.now(tz="UTC")
        spark.createDataFrame(pdf).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(TABLE)
    spark.createDataFrame([(day, "success", len(rows), datetime.now(UTC))],
                          "day string, status string, n_rows long, logged_at timestamp") \
        .write.format("delta").mode("append").saveAsTable(PROGRESS)
    print(f"{day}: {len(rows):,} hourly bars")
    ran += 1
    d = nxt

# COMMAND ----------

display(spark.sql(f"SELECT count(*) AS days, sum(n_rows) AS rows, min(day) AS first, max(day) AS last "
                  f"FROM {PROGRESS} WHERE status = 'success'"))
