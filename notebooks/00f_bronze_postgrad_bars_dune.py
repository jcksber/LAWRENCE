# Databricks notebook source
# MAGIC %md
# MAGIC # 00f — Bronze: post-graduation 1-min bars + pool metadata (Dune)
# MAGIC
# MAGIC **Why (2026-09-26, see `docs/venue_mechanics.md` and the 00e entries
# MAGIC in `docs/decisions.md`).** Bronze's pump.fun / LaunchLab rows are
# MAGIC bonding-curve trades only; graduated tokens move to PumpSwap / Raydium
# MAGIC CPMM and vanish from our data at the moment they succeed. This notebook
# MAGIC ADDS (never re-fetches) two Bronze tables:
# MAGIC
# MAGIC - `lawrence.bronze.pool_metadata` — one row per pool: venue, pool_id,
# MAGIC   base_mint (the launched token), quote_mint, platform, created_at.
# MAGIC   PumpSwap pools + LaunchLab curve pools. Replaces Silver's static
# MAGIC   quote-denylist guess for LaunchLab (quote can be any token).
# MAGIC - `lawrence.bronze.postgrad_bars_1m` — per-pool 1-minute OHLCV bars in
# MAGIC   USD for each pool's first `HORIZON_HOURS` (24h, chosen 2026-09-26):
# MAGIC   PumpSwap pools created in the window, and CPMM pools of
# MAGIC   LaunchLab-launched mints. Aggregated ON Dune; only bars are exported
# MAGIC   (~6.5M rows/month vs ~336M raw trades).
# MAGIC
# MAGIC Pure logic (SQL builders, row mappers) is in
# MAGIC `src/ingestion/postgrad_bars.py`, tested in `tests/test_postgrad_bars.py`.
# MAGIC
# MAGIC **Cost + resumability — same rules as 00c.** Progress table
# MAGIC `lawrence.bronze._postgrad_progress` records every chunk that fully
# MAGIC succeeded; re-running (same key or a swapped `dune_api_key`) skips
# MAGIC them. A 402 writes whatever pages already arrived, logs the chunk
# MAGIC `partial`, and STOPS the run — swap the key and re-run. Partial chunks
# MAGIC are re-fetched whole next time, so Bronze can hold duplicate bars for
# MAGIC them; Silver dedupes on `(venue, pool_id, bar_start)`.
# MAGIC
# MAGIC **First run: leave `MAX_PUMPSWAP_CHUNKS = 1`.** Runs metadata, CPMM,
# MAGIC and ONE 6h PumpSwap chunk, so you can check real credits/rows against
# MAGIC 00e's estimate (~5.1k credits/month ≈ ~43/6h chunk) before running
# MAGIC the other ~119. Then set it to `None`.

# COMMAND ----------

import gc
import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from ingestion.dune_client import DuneClient, DuneCreditsExhausted  # noqa: E402
from ingestion.postgrad_bars import (  # noqa: E402
    cpmm_launchlab_bars_sql,
    launchlab_pools_sql,
    pumpswap_bars_sql,
    pumpswap_pools_sql,
    to_bar_row,
    to_pool_row,
)
from pyspark.sql import Row  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")

# Same FIXED window as 00c -- chunk boundaries must be identical across
# runs/keys for the progress skip-check to match.
BACKFILL_WINDOW_START = datetime(2026, 8, 14, tzinfo=timezone.utc)
BACKFILL_WINDOW_END = datetime(2026, 9, 13, tzinfo=timezone.utc)

HORIZON_HOURS = 24  # 2026-09-26 decision: 1m bars, 24h (see decisions.md)
PUMPSWAP_CHUNK_HOURS = 6  # ~54k bars/chunk at 00e's measured rate

# 1 = measure one chunk first. None = run everything not yet done.
MAX_PUMPSWAP_CHUNKS = 1

POOLS_TABLE = "lawrence.bronze.pool_metadata"
BARS_TABLE = "lawrence.bronze.postgrad_bars_1m"
PROGRESS_TABLE = "lawrence.bronze._postgrad_progress"

WRITE_BATCH_SIZE = 5_000  # same driver-heap reasoning as 00c

# 2026-09-29: first run's bars_raydium_cpmm hit DuneClient's default 180s
# poll timeout -- the notebook gave up waiting, but the Dune execution
# itself kept running and was billed. 30 min covers the whole-window CPMM
# scan and any slow PumpSwap chunk.
POLL_TIMEOUT_S = 1800

# Executions that already ran on Dune but whose results never reached
# Bronze (e.g. the poll timeout above). Fetching an existing execution's
# results re-uses the paid-for scan instead of re-running it. If the
# fetch fails (results expired, or the execution itself failed), the job
# falls back to a fresh run_sql. Remove an entry once its job is 'success'.
RECOVER_EXECUTION_IDS = {
    "bars_raydium_cpmm": "01M3R8JGTJ99M4XQB0JN56EK71",
}

# COMMAND ----------

_POOLS_SCHEMA = StructType([
    StructField("venue", StringType()),
    StructField("pool_id", StringType()),
    StructField("base_mint", StringType()),
    StructField("quote_mint", StringType()),
    StructField("platform_name", StringType()),
    StructField("created_at", TimestampType()),
    StructField("source", StringType()),
    StructField("ingested_at", TimestampType()),
])

_BARS_SCHEMA = StructType([
    StructField("venue", StringType()),
    StructField("pool_id", StringType()),
    StructField("base_mint", StringType()),
    StructField("quote_mint", StringType()),
    StructField("bar_start", TimestampType()),
    StructField("open_usd", DoubleType()),
    StructField("high_usd", DoubleType()),
    StructField("low_usd", DoubleType()),
    StructField("close_usd", DoubleType()),
    StructField("volume_usd", DoubleType()),
    StructField("n_trades", LongType()),
    StructField("n_buys", LongType()),
    StructField("n_traders", LongType()),
    StructField("n_unpriced", LongType()),
    StructField("source", StringType()),
    StructField("ingested_at", TimestampType()),
])


def _write(rows: list[dict], schema: StructType, table: str) -> int:
    for start in range(0, len(rows), WRITE_BATCH_SIZE):
        (
            spark.createDataFrame(rows[start : start + WRITE_BATCH_SIZE], schema=schema)
            .write.format("delta")
            .mode("append")
            .saveAsTable(table)
        )
    return len(rows)


def _log_progress(job: str, chunk_start: str, chunk_end: str, status: str, rows_written: int) -> None:
    try:
        (
            spark.createDataFrame([Row(
                job=job, chunk_start=chunk_start, chunk_end=chunk_end, status=status,
                rows_written=rows_written, completed_at=datetime.now(timezone.utc),
            )])
            .write.format("delta").mode("append").option("mergeSchema", "true")
            .saveAsTable(PROGRESS_TABLE)
        )
    except Exception as exc:  # same guard as 00c: never let bookkeeping kill the run
        print(f"    (could not log progress for {job} {chunk_start}: {exc})")


done: set[tuple[str, str, str]] = set()
if spark.catalog.tableExists(PROGRESS_TABLE):
    for r in spark.table(PROGRESS_TABLE).filter("status = 'success'").collect():
        done.add((r.job, r.chunk_start, r.chunk_end))
print(f"{len(done)} chunk(s) already done -- will be skipped.")

# COMMAND ----------

# Ordered cheapest-first. Each entry: (job, chunk_start, chunk_end, sql, mapper, schema, table)
jobs = [
    ("pools_pumpswap", BACKFILL_WINDOW_START, BACKFILL_WINDOW_END,
     pumpswap_pools_sql(BACKFILL_WINDOW_START, BACKFILL_WINDOW_END), to_pool_row, _POOLS_SCHEMA, POOLS_TABLE),
    ("pools_launchlab", BACKFILL_WINDOW_START, BACKFILL_WINDOW_END,
     launchlab_pools_sql(BACKFILL_WINDOW_START, BACKFILL_WINDOW_END), to_pool_row, _POOLS_SCHEMA, POOLS_TABLE),
    ("bars_raydium_cpmm", BACKFILL_WINDOW_START, BACKFILL_WINDOW_END,
     cpmm_launchlab_bars_sql(BACKFILL_WINDOW_START, BACKFILL_WINDOW_END, HORIZON_HOURS), to_bar_row, _BARS_SCHEMA, BARS_TABLE),
]
_s, _n = BACKFILL_WINDOW_START, 0
while _s < BACKFILL_WINDOW_END:
    _e = min(_s + timedelta(hours=PUMPSWAP_CHUNK_HOURS), BACKFILL_WINDOW_END)
    jobs.append(("bars_pumpswap", _s, _e, pumpswap_bars_sql(_s, _e, HORIZON_HOURS), to_bar_row, _BARS_SCHEMA, BARS_TABLE))
    _s = _e

client = DuneClient(api_key=DUNE_API_KEY, poll_timeout_s=POLL_TIMEOUT_S)
ingested_at = datetime.now(timezone.utc)
pumpswap_run = 0
total = 0

for i, (job, start, end, sql, mapper, schema, table) in enumerate(jobs, start=1):
    key = (job, start.isoformat(), end.isoformat())
    tag = f"[{i}/{len(jobs)}] {job} [{start:%Y-%m-%d %H:%M}, {end:%Y-%m-%d %H:%M})"
    if key in done:
        print(f"{tag}: already done -- skipping")
        continue
    if job == "bars_pumpswap":
        if MAX_PUMPSWAP_CHUNKS is not None and pumpswap_run >= MAX_PUMPSWAP_CHUNKS:
            print(f"Stopping: MAX_PUMPSWAP_CHUNKS={MAX_PUMPSWAP_CHUNKS} reached. Check credits, then set it to None.")
            break
        pumpswap_run += 1

    try:
        dune_rows = None
        if job in RECOVER_EXECUTION_IDS:
            exec_id = RECOVER_EXECUTION_IDS[job]
            try:
                client._wait_until_done(exec_id)
                dune_rows = client._fetch_all_rows(exec_id)
                print(f"{tag}: recovered {len(dune_rows):,} rows from existing execution {exec_id}")
            except DuneCreditsExhausted:
                raise
            except Exception as rec_exc:
                print(f"{tag}: could not recover execution {exec_id} ({rec_exc}) -- running fresh")
        if dune_rows is None:
            dune_rows = client.run_sql(sql)
    except DuneCreditsExhausted as exc:
        salvaged = _write([mapper(r, ingested_at) for r in exc.partial_rows], schema, table) if exc.partial_rows else 0
        _log_progress(job, *key[1:], "partial", salvaged)
        print(f"{tag}: OUT OF CREDITS after {salvaged} rows -- salvaged, logged 'partial'. Swap dune_api_key and re-run.")
        break
    except Exception as exc:
        _log_progress(job, *key[1:], "failed", 0)
        print(f"{tag}: FAILED: {exc}")
        continue

    try:
        n = _write([mapper(r, ingested_at) for r in dune_rows], schema, table)
    except Exception as exc:
        _log_progress(job, *key[1:], "failed", 0)
        print(f"{tag}: WRITE FAILED after a successful fetch ({len(dune_rows)} rows): {exc}")
        continue
    finally:
        del dune_rows
        gc.collect()

    _log_progress(job, *key[1:], "success", n)
    total += n
    print(f"{tag}: wrote {n:,} rows")

print(f"\nDone this run: {total:,} rows written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks (cheap aggregates over what's been written so far)

# COMMAND ----------

if spark.catalog.tableExists(POOLS_TABLE):
    display(spark.sql(f"""
        SELECT venue, COUNT(*) AS n_pools, COUNT(DISTINCT base_mint) AS n_base_mints,
               MIN(created_at) AS min_created, MAX(created_at) AS max_created
        FROM {POOLS_TABLE} GROUP BY venue ORDER BY venue"""))

if spark.catalog.tableExists(BARS_TABLE):
    display(spark.sql(f"""
        SELECT venue, COUNT(*) AS n_bars, COUNT(DISTINCT pool_id) AS n_pools,
               SUM(n_trades) AS n_trades, SUM(n_unpriced) AS n_unpriced,
               ROUND(SUM(n_unpriced) / SUM(n_trades), 4) AS unpriced_share,
               MIN(bar_start) AS min_bar, MAX(bar_start) AS max_bar
        FROM {BARS_TABLE} GROUP BY venue ORDER BY venue"""))

if spark.catalog.tableExists(PROGRESS_TABLE):
    display(spark.sql(f"""
        SELECT job, status, COUNT(*) AS n_chunks, SUM(rows_written) AS rows_written
        FROM {PROGRESS_TABLE} GROUP BY job, status ORDER BY job, status"""))
