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

# FIXED windows -- chunk boundaries must be identical across runs/keys for
# the progress skip-check to match. Window 1 = 00c's backfill window (all
# jobs done). Window 2 (2026-10-04) = the forward extension whose curve bars
# come from 00h; Dune had full days through 2026-10-02 (00g cell 1).
WINDOWS = [
    (datetime(2026, 8, 14, tzinfo=timezone.utc), datetime(2026, 9, 13, tzinfo=timezone.utc)),
    (datetime(2026, 9, 13, tzinfo=timezone.utc), datetime(2026, 10, 3, tzinfo=timezone.utc)),
]
BACKFILL_WINDOW_START, BACKFILL_WINDOW_END = WINDOWS[0]  # window 1 (REPLACES_VENUE applies only here)

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
# (Emptied 2026-09-30: that execution belonged to the previous Dune
# account; a different account's key can't read it, so it was re-run.)
RECOVER_EXECUTION_IDS: dict[str, str] = {}

# Jobs whose venue's existing rows must be DELETED before writing, because
# the job replaces an earlier pull rather than adding to it.
# bars_raydium_cpmm_v2 (2026-09-30) re-pulls CPMM with quote-unit prices;
# v1 rows (USD-only, 62% unpriced) are removed first so there's exactly one
# copy per bar. Note the progress key changed ("_v2"), so v1's 'success'
# row doesn't cause this to be skipped.
# pools_launchlab_v2 (2026-10-03) re-pulls LaunchLab pools with the
# platform_config account (per-pool platform/creator fee).
REPLACES_VENUE = {"bars_raydium_cpmm_v2": "raydium_cpmm", "pools_launchlab_v2": "raydium_launchlab"}


def _replaces(job: str, start: datetime) -> "str | None":
    """Venue whose rows a job deletes before writing -- ONLY for window 1,
    where the job replaces an earlier pull. Later windows ADD rows; deleting
    there would wipe window 1's data (2026-10-04)."""
    return REPLACES_VENUE.get(job) if start == WINDOWS[0][0] else None

# COMMAND ----------

_POOLS_SCHEMA = StructType([
    StructField("venue", StringType()),
    StructField("pool_id", StringType()),
    StructField("base_mint", StringType()),
    StructField("quote_mint", StringType()),
    StructField("platform_name", StringType()),
    StructField("platform_config", StringType()),  # 2026-10-03, LaunchLab only
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
    # 2026-09-30: price in QUOTE-token units (quote_amount / base_amount) --
    # exists for every trade, unlike USD. See postgrad_bars._bars_select.
    StructField("open_q", DoubleType()),
    StructField("high_q", DoubleType()),
    StructField("low_q", DoubleType()),
    StructField("close_q", DoubleType()),
    StructField("volume_quote", DoubleType()),
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


# 2026-09-30 schema change: bars tables written before quote-unit prices
# existed lack the *_q / volume_quote columns. Add them (nullable) so
# appends with the new schema don't fail. Rows already written keep NULLs
# there -- fine for PumpSwap (0% USD-unpriced); CPMM is re-pulled (v2).
# 2026-10-03: same for the pools table (platform_config).
for _table, _schema in ((BARS_TABLE, _BARS_SCHEMA), (POOLS_TABLE, _POOLS_SCHEMA)):
    if spark.catalog.tableExists(_table):
        _existing = set(spark.table(_table).columns)
        _missing = [f for f in _schema.fields if f.name not in _existing]
        if _missing:
            spark.sql(
                f"ALTER TABLE {_table} ADD COLUMNS ("
                + ", ".join(f"{f.name} {f.dataType.simpleString()}" for f in _missing)
                + ")"
            )
            print(f"Added columns to {_table}: {[f.name for f in _missing]}")

done: set[tuple[str, str, str]] = set()
if spark.catalog.tableExists(PROGRESS_TABLE):
    for r in spark.table(PROGRESS_TABLE).filter("status = 'success'").collect():
        done.add((r.job, r.chunk_start, r.chunk_end))
print(f"{len(done)} chunk(s) already done -- will be skipped.")

# COMMAND ----------

# Per window, ordered cheapest-first. Each entry: (job, chunk_start, chunk_end, sql, mapper, schema, table)
jobs = []
for _w_start, _w_end in WINDOWS:
    jobs += [
        ("pools_pumpswap", _w_start, _w_end,
         pumpswap_pools_sql(_w_start, _w_end), to_pool_row, _POOLS_SCHEMA, POOLS_TABLE),
        ("pools_launchlab_v2", _w_start, _w_end,
         launchlab_pools_sql(_w_start, _w_end), to_pool_row, _POOLS_SCHEMA, POOLS_TABLE),
        ("bars_raydium_cpmm_v2", _w_start, _w_end,
         cpmm_launchlab_bars_sql(_w_start, _w_end, HORIZON_HOURS), to_bar_row, _BARS_SCHEMA, BARS_TABLE),
    ]
for _w_start, _w_end in WINDOWS:
    _s = _w_start
    while _s < _w_end:
        _e = min(_s + timedelta(hours=PUMPSWAP_CHUNK_HOURS), _w_end)
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
            print(f"{tag}: running on Dune ({datetime.now(timezone.utc):%H:%M:%S} UTC)...", flush=True)
            dune_rows = client.run_sql(sql)
    except DuneCreditsExhausted as exc:
        if exc.partial_rows and _replaces(job, start):
            spark.sql(f"DELETE FROM {table} WHERE venue = '{_replaces(job, start)}'")
        salvaged = _write([mapper(r, ingested_at) for r in exc.partial_rows], schema, table) if exc.partial_rows else 0
        _log_progress(job, *key[1:], "partial", salvaged)
        print(f"{tag}: OUT OF CREDITS after {salvaged} rows -- salvaged, logged 'partial'. Swap dune_api_key and re-run.")
        break
    except Exception as exc:
        _log_progress(job, *key[1:], "failed", 0)
        print(f"{tag}: FAILED: {exc}")
        continue

    try:
        if _replaces(job, start):
            spark.sql(f"DELETE FROM {table} WHERE venue = '{_replaces(job, start)}'")
            print(f"{tag}: deleted previous '{_replaces(job, start)}' rows before replacing them")
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

# 2026-10-04: a LaunchLab pool that trades in two windows is returned by
# both windows' pools query (its created_at = first trade IN that window),
# so pool_metadata can hold two rows for one pool. Keep one per
# (venue, pool_id): the earliest created_at (closest to the real launch),
# with platform fields filled from any copy. PumpSwap pools are keyed by
# creation time, so they never overlap. No-op when there are no duplicates.
if spark.catalog.tableExists(POOLS_TABLE):
    _pools = spark.table(POOLS_TABLE)
    _n, _n_keys = _pools.count(), _pools.select("venue", "pool_id").distinct().count()
    if _n > _n_keys:
        # Delta reads a snapshot, so replacing a table from a query over
        # itself is safe and atomic.
        spark.sql(f"""
            CREATE OR REPLACE TABLE {POOLS_TABLE} AS
            SELECT venue, pool_id, base_mint, quote_mint, platform_name, platform_config,
                   created_at, source, ingested_at
            FROM (
                SELECT venue, pool_id, base_mint, quote_mint,
                       max(platform_name) OVER k AS platform_name,
                       max(platform_config) OVER k AS platform_config,
                       created_at, source, ingested_at,
                       row_number() OVER (PARTITION BY venue, pool_id
                                          ORDER BY created_at, ingested_at DESC) AS _rn
                FROM {POOLS_TABLE}
                WINDOW k AS (PARTITION BY venue, pool_id)
            )
            WHERE _rn = 1
        """)
        print(f"pool_metadata: removed {_n - _n_keys:,} duplicate pool rows ({_n_keys:,} pools kept)")
    else:
        print(f"pool_metadata: no duplicate pools ({_n:,} rows)")

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
               ROUND(AVG(CASE WHEN close_q IS NULL THEN 1.0 ELSE 0.0 END), 4) AS bars_missing_quote_price,
               MIN(bar_start) AS min_bar, MAX(bar_start) AS max_bar
        FROM {BARS_TABLE} GROUP BY venue ORDER BY venue"""))

if spark.catalog.tableExists(PROGRESS_TABLE):
    display(spark.sql(f"""
        SELECT job, status, COUNT(*) AS n_chunks, SUM(rows_written) AS rows_written
        FROM {PROGRESS_TABLE} GROUP BY job, status ORDER BY job, status"""))
