# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Gold: `lawrence.gold.entry_candidates` (training data)
# MAGIC
# MAGIC **This is a batch/backfill job, not a scheduled pipeline step.** It
# MAGIC regenerates the entire entry-model training table from whatever is
# MAGIC currently in Silver -- unlike Bronze/Silver, it isn't wired into
# MAGIC `resources/lawrence_job.yml`'s recurring schedule. Re-run it by hand
# MAGIC whenever Silver has enough new history to be worth retraining on.
# MAGIC (A production-serving path -- T = "now," incremental, per active
# MAGIC token -- is future work once there's a model to feed; see
# MAGIC docs/gold_spec.md.)
# MAGIC
# MAGIC The actual feature/label math lives in
# MAGIC `src/features/gold_features.py` (plain Python, no PySpark -- unit
# MAGIC tested, runs in CI). This notebook is the Spark glue: read all of
# MAGIC Silver, group by token, sample assessment points, call into that
# MAGIC module, write the result.
# MAGIC
# MAGIC **2026-09-17: streamed instead of collect()'d.** With real Dune
# MAGIC backfill volume in Silver (~10.75M rows), `silver_df.collect()` --
# MAGIC pulling every row into one Python list on the driver before doing
# MAGIC anything with it -- is the same driver-memory hazard that hung the
# MAGIC Silver catch-up notebook before it was chunked (see docs/decisions.md).
# MAGIC This notebook was written when Gold's own docs expected "few or zero
# MAGIC rows" (no real backfill existed yet), so it was never exercised at
# MAGIC this scale. Fixed by using `toLocalIterator()` (streams one partition
# MAGIC at a time instead of materializing the whole result) plus a
# MAGIC single streaming pass that relies on the existing `orderBy("token_id",
# MAGIC "trade_time")` -- a token's rows are contiguous in that order, so each
# MAGIC token's full trade history can be built and processed as soon as the
# MAGIC next token_id is seen, without ever holding more than one token's
# MAGIC trades (plus the much smaller accumulating Gold output) in memory.

# COMMAND ----------

import sys
import time

sys.path.append("../src")
from features.gold_features import build_entry_row, sample_assessment_times  # noqa: E402

# COMMAND ----------

SILVER_TABLE = "lawrence.silver.token_trades"
GOLD_TABLE = "lawrence.gold.entry_candidates"

# COMMAND ----------

silver_df = spark.table(SILVER_TABLE).orderBy("token_id", "trade_time").cache()
# 2026-09-17 fix (see docs/decisions.md): toLocalIterator() without a cache
# can force Spark to recompute the ENTIRE orderBy shuffle once per partition
# retrieved instead of once total -- for ~25 partitions that's ~25x the
# work, and matches exactly what a re-run showed (the underlying Spark
# statement itself hadn't finished after 18+ minutes, vs. 36s when this was
# a plain .collect()). .count() forces the cache to materialize eagerly
# (Spark caching is lazy otherwise) so the sort runs exactly once before any
# partition is streamed out.
_silver_row_count = silver_df.count()

gold_rows: list[dict] = []
total_silver_rows = 0
distinct_tokens = 0
current_token_id = None
current_trades: list[dict] = []


def _process_token(token_id, trades):
    if token_id is None or not trades:
        return
    for t in sample_assessment_times(trades):
        row = build_entry_row(token_id, t, trades)
        if row is not None:
            gold_rows.append(row)


# Single streaming pass over the sorted Silver rows -- see the note above on
# why this replaces a full .collect(). toLocalIterator() pulls partitions to
# the driver one at a time rather than materializing the entire 10M+ row
# result at once.
# 2026-09-17: progress logging (see docs/decisions.md) -- the earlier
# opaque-hang incidents (both the pre-fix .collect() and the pre-cache
# toLocalIterator() recompute) were hard to tell apart from "still working,
# just slow" without any signal between start and the final print. This
# gives a running count so a stalled run is visible immediately instead of
# discovered 18+ minutes in.
_PROGRESS_EVERY = 500_000
_progress_start = time.time()
for row in silver_df.toLocalIterator():
    r = row.asDict()
    total_silver_rows += 1
    if r["token_id"] != current_token_id:
        _process_token(current_token_id, current_trades)
        current_token_id = r["token_id"]
        current_trades = []
        distinct_tokens += 1
    current_trades.append(r)
    if total_silver_rows % _PROGRESS_EVERY == 0:
        _elapsed = time.time() - _progress_start
        print(
            f"... {total_silver_rows:,} rows scanned, {distinct_tokens:,} tokens, "
            f"{len(gold_rows):,} gold rows so far ({_elapsed:.0f}s elapsed)"
        )
_process_token(current_token_id, current_trades)  # flush the last token

if total_silver_rows != _silver_row_count:
    print(
        f"WARNING: scanned {total_silver_rows:,} rows via toLocalIterator() but "
        f"the cached count was {_silver_row_count:,} -- investigate before "
        "trusting this run's output."
    )
print(f"Silver rows scanned: {total_silver_rows} across {distinct_tokens} distinct tokens")
print(f"Entry candidate rows produced: {len(gold_rows)}")
# A low/zero count against a freshly-started Bronze/Silver history is
# expected, not a bug: build_entry_row needs both a trailing feature window
# AND a real trade near T + HOLD_MINUTES, which a just-started ingestion
# history mostly won't have yet. Re-run this after Bronze/Silver have
# accumulated more history.

# COMMAND ----------

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema, same rationale as Bronze/Silver's _BRONZE_SCHEMA/
# _SILVER_SCHEMA (see docs/decisions.md): avoids CANNOT_DETERMINE_TYPE if a
# nullable column (e.g. trailing_price_change_pct, network) happens to be
# all-None in a given run, and passing plain dicts (not Row(**r)) avoids any
# ambiguity between Row's internal alphabetical field order and this schema's
# authored order.
_ENTRY_SCHEMA = StructType([
    StructField("token_id", StringType()),
    StructField("chain_family", StringType()),
    StructField("network", StringType()),
    StructField("dex_venue", StringType()),
    StructField("dex_protocol", StringType()),
    StructField("assessment_time", StringType()),
    StructField("price_at_t", DoubleType()),
    StructField("hold_minutes", IntegerType()),
    StructField("y_entry", DoubleType()),
    StructField("is_rug_clamped", BooleanType()),
    StructField("trailing_volume_usd", DoubleType()),
    StructField("trailing_trade_count", IntegerType()),
    StructField("trailing_buy_count", IntegerType()),
    StructField("trailing_sell_count", IntegerType()),
    StructField("trailing_buy_sell_ratio", DoubleType()),
    StructField("trailing_price_change_pct", DoubleType()),
])

if gold_rows:
    df = spark.createDataFrame(gold_rows, schema=_ENTRY_SCHEMA)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_TABLE)
    )
    print(f"Wrote {len(gold_rows)} rows to {GOLD_TABLE} (full overwrite -- this "
          "is a batch regenerate, not an incremental append; re-running from "
          "scratch is the intended usage, unlike Bronze/Silver's MERGE)")
else:
    print("No entry candidate rows produced.")
