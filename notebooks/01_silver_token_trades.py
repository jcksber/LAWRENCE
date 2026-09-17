# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Silver: `lawrence.silver.token_trades`
# MAGIC
# MAGIC Reads Bronze's raw two-sided swaps and writes one row per
# MAGIC (trade, token-of-interest) into `lawrence.silver.token_trades` --
# MAGIC see `docs/silver_spec.md` for the schema and rationale.
# MAGIC
# MAGIC The denylist + row-shaping logic lives in
# MAGIC `src/features/silver_token_trades.py` (plain Python, no PySpark, so
# MAGIC it's unit-testable and runs in CI) -- this notebook is just the Spark
# MAGIC read/merge glue.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from features.silver_token_trades import to_silver_rows  # noqa: E402

# COMMAND ----------

BRONZE_TABLE = "lawrence.bronze.dex_trades"
SILVER_TABLE = "lawrence.silver.token_trades"
# Only used in catch-up mode (CATCHUP_SINCE/UNTIL both set) -- same
# resumability idea as the Dune backfill notebook's own progress table.
# 2026-09-17: a 30-day catch-up run hit "IOStream.flush timed out" twice
# (a Databricks/Jupyter display-layer hiccup, not necessarily a Python
# crash) with no way to tell from notebook output alone whether the run
# actually finished. The MERGE below is already safe to re-run wholesale
# (keyed on dex_venue+tx_hash+token_id, so no duplicates), but re-running
# from scratch means re-collecting and re-transforming every prior day
# for nothing -- this table lets a re-run skip chunks already done.
CATCHUP_PROGRESS_TABLE = "lawrence.silver._catchup_progress"

# Bounded lookback, same idea as Bronze's own LOOKBACK_MINUTES: re-scan a
# window wider than the run cadence so a slow/late run doesn't lose rows,
# and rely on the MERGE below (keyed on dex_venue+tx_hash+token_id) to make
# re-processing the overlap a no-op rather than a duplicate.
# NOTE: this re-reads the whole window from Bronze every run -- fine at v1
# scale, but revisit with a real watermark once Bronze volume makes a
# 60-min scan expensive.
LOOKBACK_MINUTES = 60

# Optional manual catch-up overrides -- leave both None for normal live
# cadence (a single [now - LOOKBACK_MINUTES, now) pass, unchanged from
# before). Set both to process a bounded historical window in day-sized
# pieces instead of one shot.
#
# 2026-09-15 incident: widening LOOKBACK_MINUTES to 40 days to catch up
# the Dune backfill collected the whole ~9M+ row span into a single Python
# list on the driver in one go (see the loop below -- the old version had
# no chunking at all) and hung the kernel ("Python kernel is
# unresponsive"). Chunking bounds driver memory the same way the Dune
# backfill notebook itself already does successfully.
#
# Example -- catch up the Dune backfill window (Aug 14 - Sep 13):
#   CATCHUP_SINCE = datetime(2026, 8, 14, tzinfo=timezone.utc)
#   CATCHUP_UNTIL = datetime(2026, 9, 13, tzinfo=timezone.utc)
# NOTE: use "=" to set a real value, not ":" -- "CATCHUP_SINCE: datetime(...)"
# is a type annotation with no assignment (valid syntax, does nothing) and
# fails later with NameError, not at this line (confirmed live 2026-09-17).
CATCHUP_SINCE = None
CATCHUP_UNTIL = None
CATCHUP_CHUNK_HOURS = 24

# COMMAND ----------

IS_CATCHUP = CATCHUP_SINCE is not None and CATCHUP_UNTIL is not None

if IS_CATCHUP:
    windows = []
    chunk_start = CATCHUP_SINCE
    while chunk_start < CATCHUP_UNTIL:
        chunk_end = min(chunk_start + timedelta(hours=CATCHUP_CHUNK_HOURS), CATCHUP_UNTIL)
        windows.append((chunk_start, chunk_end))
        chunk_start = chunk_end
else:
    # Live cadence: single unbounded-above window, exactly as before.
    windows = [(datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES), None)]

from delta.tables import DeltaTable
from pyspark.sql import Row
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

completed_success: set[tuple[str, str]] = set()
if IS_CATCHUP and spark.catalog.tableExists(CATCHUP_PROGRESS_TABLE):
    for r in (
        spark.table(CATCHUP_PROGRESS_TABLE)
        .filter("status = 'success'")
        .select("chunk_start", "chunk_end")
        .collect()
    ):
        completed_success.add((r["chunk_start"], r["chunk_end"]))


def _log_catchup_progress(chunk_start_str: str, chunk_end_str: str, status: str, bronze_n: int, silver_n: int) -> None:
    row = Row(
        chunk_start=chunk_start_str,
        chunk_end=chunk_end_str,
        status=status,  # "success" | "failed"
        bronze_rows=bronze_n,
        silver_rows=silver_n,
        completed_at=datetime.now(timezone.utc),
    )
    (
        spark.createDataFrame([row])
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(CATCHUP_PROGRESS_TABLE)
    )

# Explicit schema, not inference -- same fix as Bronze's _BRONZE_SCHEMA
# (2026-09-15). tx_fee_usd/chain_native_fee are always None for a
# catch-up chunk that's 100% Dune-sourced rows (Dune has no usable
# transaction fee -- see silver_spec.md), which gives Spark's
# createDataFrame zero non-null samples to infer those columns' types
# from -- CANNOT_DETERMINE_TYPE. Live-cadence chunks (real Bitquery fees)
# never hit this, which is why it was invisible until the first
# all-Dune catch-up chunk actually ran.
_SILVER_SCHEMA = StructType(
    [
        StructField("token_id", StringType(), True),
        StructField("chain_family", StringType(), True),
        StructField("network", StringType(), True),
        StructField("dex_venue", StringType(), True),
        StructField("dex_protocol", StringType(), True),
        StructField("trade_time", StringType(), True),
        StructField("side", StringType(), True),
        StructField("price_usd", DoubleType(), True),
        StructField("amount_usd", DoubleType(), True),
        StructField("tx_fee_usd", DoubleType(), True),
        StructField("chain_native_fee", DoubleType(), True),
        StructField("tx_hash", StringType(), True),
        StructField("quote_token_id", StringType(), True),
    ]
)

total_bronze = 0
total_silver = 0

for i, (win_since, win_until) in enumerate(windows, start=1):
    since_str = win_since.isoformat()
    until_str = win_until.isoformat() if win_until is not None else ""

    if IS_CATCHUP and (since_str, until_str) in completed_success:
        print(f"[{i}/{len(windows)}] [{win_since}, {win_until}) already done -- skipping")
        continue

    filter_sql = f"block_time >= '{since_str}'"
    if win_until is not None:
        filter_sql += f" AND block_time < '{until_str}'"

    try:
        bronze_df = spark.table(BRONZE_TABLE).filter(filter_sql)
        bronze_rows = [row.asDict() for row in bronze_df.collect()]

        silver_rows = [
            silver_row
            for bronze_row in bronze_rows
            for silver_row in to_silver_rows(bronze_row)
        ]

        if silver_rows:
            # Plain dicts, not Row(**r) -- see 00c_bronze_backfill_dune.py's
            # matching comment (2026-09-15): dicts-with-schema is the
            # unambiguous, name-matched form; Row(**kwargs) sorts fields
            # alphabetically internally and this call's target schema field
            # order doesn't match that, so it isn't worth trusting to resolve
            # correctly by luck.
            new_df = spark.createDataFrame(silver_rows, schema=_SILVER_SCHEMA)

            if spark.catalog.tableExists(SILVER_TABLE):
                target = DeltaTable.forName(spark, SILVER_TABLE)
                (
                    target.alias("t")
                    .merge(
                        new_df.alias("s"),
                        "t.dex_venue = s.dex_venue AND t.tx_hash = s.tx_hash"
                        " AND t.token_id = s.token_id",
                    )
                    .whenNotMatchedInsertAll()
                    .execute()
                )
            else:
                new_df.write.format("delta").saveAsTable(SILVER_TABLE)
    except Exception as exc:
        if IS_CATCHUP:
            _log_catchup_progress(since_str, until_str, "failed", 0, 0)
        print(f"[{i}/{len(windows)}] [{win_since}, {win_until}) FAILED: {exc}")
        continue

    total_bronze += len(bronze_rows)
    total_silver += len(silver_rows)

    if IS_CATCHUP:
        _log_catchup_progress(since_str, until_str, "success", len(bronze_rows), len(silver_rows))

    print(
        f"[{i}/{len(windows)}] [{win_since}, {win_until}): "
        f"{len(bronze_rows)} Bronze rows -> {len(silver_rows)} Silver rows"
    )

print(
    f"Done. Bronze rows scanned: {total_bronze} -> "
    f"Silver candidate rows: {total_silver}"
)
