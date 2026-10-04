# Databricks notebook source
# MAGIC %md
# MAGIC # 00h — Bronze: bonding-curve 1-min bars from Dune (forward window)
# MAGIC
# MAGIC **Why (2026-10-04, see the 00g entries in `docs/decisions.md`).** More
# MAGIC months of data, without re-pulling raw trades: Dune turns pump.fun and
# MAGIC LaunchLab curve trades into 1-minute bars (`src/ingestion/curve_bars.py`,
# MAGIC validated within 1% of our Silver bars in 00g cell 5) and only the bars
# MAGIC are exported. Adds per-bar trader stats Bronze never had.
# MAGIC
# MAGIC Writes `lawrence.bronze.curve_bars_1m` for 2026-09-13 to 2026-10-03
# MAGIC (the old window's curve data stays in `bronze.dex_trades`; Silver
# MAGIC stitches the two). Estimated ~7k credits (pump.fun ~236/day,
# MAGIC LaunchLab ~120/day). The post-graduation bars for the same window come
# MAGIC from 00f (window 2).
# MAGIC
# MAGIC **Same resumability rules as 00f.** `lawrence.bronze._curve_bars_progress`
# MAGIC records finished chunks; re-running (same or a swapped `dune_api_key`)
# MAGIC skips them. On a 402 it writes the pages that arrived, logs `partial`,
# MAGIC and stops -- swap the key and re-run. Re-fetched chunks can leave
# MAGIC duplicate bars; Silver dedupes on `(venue, pool_id, bar_start)`.
# MAGIC
# MAGIC **First run: leave `MAX_CHUNKS = 2`** (one chunk per venue), note the
# MAGIC credit delta, then set it to `None`.

# COMMAND ----------

import gc
import importlib
import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
import ingestion.curve_bars as curve_bars  # noqa: E402
import ingestion.postgrad_bars as postgrad_bars  # noqa: E402

importlib.reload(postgrad_bars)
importlib.reload(curve_bars)
from ingestion.curve_bars import (  # noqa: E402
    launchlab_curve_bars_sql,
    pumpfun_curve_bars_sql,
    to_curve_bar_row,
)
from ingestion.dune_client import DuneClient, DuneCreditsExhausted  # noqa: E402
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

# FIXED window + chunk sizes: boundaries must not change between runs, or
# the progress skip-check won't match. Same window 2 as 00f.
WINDOW_START = datetime(2026, 9, 13, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 3, tzinfo=timezone.utc)
CHUNK_HOURS = {
    "pumpdotfun": 6,          # ~47k bars per chunk
    "raydium_launchlab": 24,  # ~81k bars/day; daily so the pool lookup runs once per day
}
SQL_BUILDERS = {
    "pumpdotfun": pumpfun_curve_bars_sql,
    "raydium_launchlab": launchlab_curve_bars_sql,
}

# 2 = one chunk per venue to check credits. None = run everything not done.
MAX_CHUNKS = 2

BARS_TABLE = "lawrence.bronze.curve_bars_1m"
PROGRESS_TABLE = "lawrence.bronze._curve_bars_progress"
WRITE_BATCH_SIZE = 5_000  # same driver-heap reasoning as 00c / 00f
POLL_TIMEOUT_S = 1800

_BARS_SCHEMA = StructType(
    [StructField(c, StringType()) for c in ("venue", "pool_id", "base_mint", "quote_mint")]
    + [StructField("bar_start", TimestampType())]
    + [StructField(c, DoubleType()) for c in (
        "open_usd", "high_usd", "low_usd", "close_usd", "volume_usd",
        "open_q", "high_q", "low_q", "close_q", "volume_quote",
        "volume_buy_q", "volume_sell_q", "max_trade_q", "top_trader_share",
    )]
    + [StructField(c, LongType()) for c in ("n_trades", "n_buys", "n_traders", "n_unpriced")]
    + [StructField("source", StringType()), StructField("ingested_at", TimestampType())]
)


def _write(rows: list[dict]) -> int:
    for start in range(0, len(rows), WRITE_BATCH_SIZE):
        (
            spark.createDataFrame(rows[start : start + WRITE_BATCH_SIZE], schema=_BARS_SCHEMA)
            .write.format("delta")
            .mode("append")
            .saveAsTable(BARS_TABLE)
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
    except Exception as exc:  # never let bookkeeping kill the run
        print(f"    (could not log progress for {job} {chunk_start}: {exc})")


done: set[tuple[str, str, str]] = set()
if spark.catalog.tableExists(PROGRESS_TABLE):
    for r in spark.table(PROGRESS_TABLE).filter("status = 'success'").collect():
        done.add((r.job, r.chunk_start, r.chunk_end))
print(f"{len(done)} chunk(s) already done -- will be skipped.")

# COMMAND ----------

# Interleave venues day by day, so a stop part-way leaves both venues
# covering the same dates.
jobs = []
_day = WINDOW_START
while _day < WINDOW_END:
    _day_end = min(_day + timedelta(days=1), WINDOW_END)
    for venue, hours in CHUNK_HOURS.items():
        _s = _day
        while _s < _day_end:
            _e = min(_s + timedelta(hours=hours), _day_end)
            jobs.append((f"curve_{venue}", _s, _e, SQL_BUILDERS[venue](_s, _e)))
            _s = _e
    _day = _day_end
print(f"{len(jobs)} chunks in the window")

client = DuneClient(api_key=DUNE_API_KEY, poll_timeout_s=POLL_TIMEOUT_S)
ingested_at = datetime.now(timezone.utc)
ran = 0
total = 0

for i, (job, start, end, sql) in enumerate(jobs, start=1):
    key = (job, start.isoformat(), end.isoformat())
    tag = f"[{i}/{len(jobs)}] {job} [{start:%Y-%m-%d %H:%M}, {end:%Y-%m-%d %H:%M})"
    if key in done:
        continue
    if MAX_CHUNKS is not None and ran >= MAX_CHUNKS:
        print(f"Stopping: MAX_CHUNKS={MAX_CHUNKS} reached. Check credits, then set it to None.")
        break
    ran += 1

    try:
        print(f"{tag}: running on Dune ({datetime.now(timezone.utc):%H:%M:%S} UTC)...", flush=True)
        dune_rows = client.run_sql(sql)
    except DuneCreditsExhausted as exc:
        salvaged = _write([to_curve_bar_row(r, ingested_at) for r in exc.partial_rows]) if exc.partial_rows else 0
        _log_progress(job, *key[1:], "partial", salvaged)
        print(f"{tag}: OUT OF CREDITS after {salvaged} rows -- salvaged, logged 'partial'. Swap dune_api_key and re-run.")
        break
    except Exception as exc:
        _log_progress(job, *key[1:], "failed", 0)
        print(f"{tag}: FAILED: {exc}")
        continue

    try:
        n = _write([to_curve_bar_row(r, ingested_at) for r in dune_rows])
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
# MAGIC ## Checks

# COMMAND ----------

if spark.catalog.tableExists(BARS_TABLE):
    display(spark.sql(f"""
        SELECT venue, to_date(bar_start) AS day, COUNT(*) AS n_bars,
               COUNT(DISTINCT base_mint) AS n_tokens, SUM(n_trades) AS n_trades,
               ROUND(SUM(n_unpriced) / SUM(n_trades), 4) AS unpriced_share
        FROM {BARS_TABLE} GROUP BY 1, 2 ORDER BY 2, 1"""))

if spark.catalog.tableExists(PROGRESS_TABLE):
    display(spark.sql(f"""
        SELECT job, status, COUNT(*) AS n_chunks, SUM(rows_written) AS rows_written
        FROM {PROGRESS_TABLE} GROUP BY job, status ORDER BY job, status"""))
