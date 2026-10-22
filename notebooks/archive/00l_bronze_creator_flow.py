# Databricks notebook source
# MAGIC %md
# MAGIC # 00l — Bronze: pump.fun launches, early buyers, creator trades (v3 part C)
# MAGIC Full pull for part C, after the 00k cost test (2026-10-18: 1 day = 30,783
# MAGIC launches, 1,141 graduates, 13,887 creator trades; cells 3-6 = 43 credits).
# MAGIC Adds three Bronze tables (never touches existing ones):
# MAGIC - `lawrence.bronze.pump_launches` — every pump.fun create (CreateEvent): mint,
# MAGIC   creator, created_at, slot, quote_mint, is_mayhem_mode. Window 08-30 → 10-10
# MAGIC   (history for "creator's prior launches / graduations before T").
# MAGIC - `lawrence.bronze.pump_early_buyers` — per graduated token: slot-0 / first-3-slot
# MAGIC   buyers and supply shares. Creation dates 09-06 → 10-10 (tokens graduating in the
# MAGIC   v2 training window 09-13 → 10-10 can be created a few days earlier).
# MAGIC - `lawrence.bronze.pump_creator_trades` — the creator's buys/sells of its own
# MAGIC   graduated token within 48h, same creation dates.
# MAGIC **Holdout rule:** every trade read is capped at `CUTOFF` = 2026-10-11 in the SQL
# MAGIC (`creator_flow`, tested). Nothing at/after it is read.
# MAGIC **Resumable:** one job per (table, day); `lawrence.bronze._creator_progress` records
# MAGIC successes and re-runs skip them. Out of credits → stops; swap key, re-run.
# MAGIC **First run: leave `MAX_DAYS_PER_JOB = 1`** and note the credit counter; then `None`.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append("../../src")
import importlib  # noqa: E402

import ingestion.creator_flow as cf  # noqa: E402

importlib.reload(cf)
from ingestion.dune_client import DuneClient, DuneCreditsExhausted  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
client = DuneClient(api_key=dbutils.secrets.get(scope="lawrence", key="dune_api_key"), poll_timeout_s=1800)

UTC = timezone.utc
CUTOFF = datetime(2026, 10, 11, tzinfo=UTC)        # holdout starts here -- never read
LAUNCH_START = datetime(2026, 8, 30, tzinfo=UTC)  # 2 weeks of history before 09-13 (budget ~1k credits)
FLOW_START = datetime(2026, 9, 6, tzinfo=UTC)
END = datetime(2026, 10, 11, tzinfo=UTC)           # exclusive; last day = 10-10
assert END <= CUTOFF

MAX_DAYS_PER_JOB = 1   # 1 = measure first; None = run everything not yet done

TABLES = {
    "launches": "lawrence.bronze.pump_launches",
    "early_buyers": "lawrence.bronze.pump_early_buyers",
    "creator_trades": "lawrence.bronze.pump_creator_trades",
}
PROGRESS_TABLE = "lawrence.bronze._creator_progress"
TS_COLS = {"created_at", "block_time"}


def _days(start, end):
    d = start
    while d < end:
        yield d, d + timedelta(days=1)
        d += timedelta(days=1)


# Priority order (2026-10-18, ~1k credits left): early buyers first (strongest
# signal in 00k), then creator trades, then launch history. If credits run out,
# the most useful data is already in; swap dune_api_key and re-run for the rest.
JOBS = (
    [("early_buyers", a, b, cf.early_buyers_sql(a, b, CUTOFF)) for a, b in _days(FLOW_START, END)]
    + [("creator_trades", a, b, cf.creator_trades_sql(a, b, CUTOFF)) for a, b in _days(FLOW_START, END)]
    + [("launches", a, b, cf.creator_launches_sql(a, b)) for a, b in _days(LAUNCH_START, END)]
)
print(f"{len(JOBS)} jobs")

# COMMAND ----------

done = set()
if spark.catalog.tableExists(PROGRESS_TABLE):
    done = {(r.job, r.day) for r in spark.table(PROGRESS_TABLE).filter("status = 'success'").collect()}


def _to_spark(rows, day):
    pdf = pd.DataFrame(rows)
    pdf.attrs = {}
    for c in pdf.columns:
        if c in TS_COLS:
            pdf[c] = pd.to_datetime(pdf[c].astype(str).str.replace(" UTC", "", regex=False), utc=True)
    pdf["source_day"] = day
    pdf["ingested_at"] = pd.Timestamp.now(tz="UTC")
    return spark.createDataFrame(pdf)


def _log(job, day, status, n):
    spark.createDataFrame(
        [(job, day, status, int(n), datetime.now(UTC))],
        "job string, day string, status string, n_rows long, logged_at timestamp",
    ).write.format("delta").mode("append").saveAsTable(PROGRESS_TABLE)


ran = {k: 0 for k in TABLES}
for job, a, b, sql in JOBS:
    day = a.strftime("%Y-%m-%d")
    if (job, day) in done:
        continue
    if MAX_DAYS_PER_JOB is not None and ran[job] >= MAX_DAYS_PER_JOB:
        continue
    try:
        rows = client.run_sql(sql)
    except DuneCreditsExhausted:
        print(f"{job} {day}: OUT OF CREDITS -- nothing written for this day. Swap dune_api_key and re-run.")
        break
    if rows:
        _to_spark(rows, day).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(TABLES[job])
    _log(job, day, "success", len(rows))
    ran[job] += 1
    print(f"{job} {day}: {len(rows):,} rows")

# COMMAND ----------

# Progress summary
display(spark.sql(f"""SELECT job, count(*) AS days, sum(n_rows) AS rows, min(day) AS first, max(day) AS last
FROM {PROGRESS_TABLE} WHERE status = 'success' GROUP BY job ORDER BY job"""))
