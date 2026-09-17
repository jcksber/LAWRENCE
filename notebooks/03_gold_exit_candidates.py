# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: `lawrence.gold.exit_candidates` (training data)
# MAGIC
# MAGIC Same batch/backfill nature as notebook 02 (`entry_candidates`) --
# MAGIC not wired into the recurring job schedule, re-run by hand. Position
# MAGIC context (entry_price, unrealized_return, holding_duration) is
# MAGIC **simulated**, not from real trading history: see docs/label_spec.md's
# MAGIC "Exit label: training data via simulated entries."
# MAGIC
# MAGIC Feature/label math lives in `src/features/gold_features.py` (shared
# MAGIC with notebook 02's entry build); this is the Spark read/write glue.
# MAGIC
# MAGIC **2026-09-17: streamed instead of collect()'d** -- same fix and same
# MAGIC reason as notebook 02 (see its header comment and docs/decisions.md):
# MAGIC `silver_df.collect()` on real ~10.75M-row Silver volume is the same
# MAGIC driver-memory hazard already hit and fixed once in Silver's own
# MAGIC catch-up notebook. Replaced with `toLocalIterator()` plus a single
# MAGIC streaming pass over the already-sorted (`token_id`, `trade_time`) rows.

# COMMAND ----------

import sys
import time

sys.path.append("../src")
from features.gold_features import build_exit_rows, sample_assessment_times  # noqa: E402

# COMMAND ----------

SILVER_TABLE = "lawrence.silver.token_trades"
GOLD_TABLE = "lawrence.gold.exit_candidates"

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
        gold_rows.extend(build_exit_rows(token_id, t, trades))


# Single streaming pass -- see notebook 02's header comment for why this
# replaces a full .collect(). toLocalIterator() pulls partitions to the
# driver one at a time; a token's rows are contiguous thanks to the orderBy,
# so each token can be fully processed as soon as the next token_id appears.
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
print(f"Exit candidate rows produced: {len(gold_rows)}")
# Same expected-low-count-on-fresh-history caveat as notebook 02, plus one
# more constraint: each row also needs a synthetic entry price at T-h for
# at least one h in EXIT_SIMULATED_HOLDS, so exit candidates need MORE
# trailing history per token than entry candidates do.

# COMMAND ----------

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema -- same rationale as notebook 02's _ENTRY_SCHEMA.
_EXIT_SCHEMA = StructType([
    StructField("token_id", StringType()),
    StructField("chain_family", StringType()),
    StructField("network", StringType()),
    StructField("dex_venue", StringType()),
    StructField("dex_protocol", StringType()),
    StructField("assessment_time", StringType()),
    StructField("price_at_t", DoubleType()),
    StructField("hold_minutes", IntegerType()),
    StructField("simulated_holding_minutes", DoubleType()),
    StructField("entry_price", DoubleType()),
    StructField("unrealized_return", DoubleType()),
    StructField("y_exit", DoubleType()),
    StructField("is_rug_clamped", BooleanType()),
    StructField("trailing_volume_usd", DoubleType()),
    StructField("trailing_trade_count", IntegerType()),
    StructField("trailing_buy_count", IntegerType()),
    StructField("trailing_sell_count", IntegerType()),
    StructField("trailing_buy_sell_ratio", DoubleType()),
    StructField("trailing_price_change_pct", DoubleType()),
])

if gold_rows:
    df = spark.createDataFrame(gold_rows, schema=_EXIT_SCHEMA)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_TABLE)
    )
    print(f"Wrote {len(gold_rows)} rows to {GOLD_TABLE} (full overwrite, "
          "same batch-regenerate rationale as notebook 02)")
else:
    print("No exit candidate rows produced.")
