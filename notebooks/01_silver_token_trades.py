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

# Bounded lookback, same idea as Bronze's own LOOKBACK_MINUTES: re-scan a
# window wider than the run cadence so a slow/late run doesn't lose rows,
# and rely on the MERGE below (keyed on dex_venue+tx_hash+token_id) to make
# re-processing the overlap a no-op rather than a duplicate.
# NOTE: this re-reads the whole window from Bronze every run -- fine at v1
# scale, but revisit with a real watermark once Bronze volume makes a
# 60-min scan expensive.
LOOKBACK_MINUTES = 60

# COMMAND ----------

since = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)

bronze_df = spark.table(BRONZE_TABLE).filter(f"block_time >= '{since.isoformat()}'")
bronze_rows = [row.asDict() for row in bronze_df.collect()]

silver_rows = [
    silver_row
    for bronze_row in bronze_rows
    for silver_row in to_silver_rows(bronze_row)
]
print(f"Bronze rows scanned: {len(bronze_rows)} -> Silver candidate rows: {len(silver_rows)}")

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import Row

if silver_rows:
    new_df = spark.createDataFrame([Row(**r) for r in silver_rows])

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
        print(
            f"Merged {len(silver_rows)} candidate rows into {SILVER_TABLE} "
            "(deduped on dex_venue+tx_hash+token_id; matches were skipped)"
        )
    else:
        new_df.write.format("delta").saveAsTable(SILVER_TABLE)
        print(f"Created {SILVER_TABLE} with {len(silver_rows)} rows")
else:
    print("No Silver rows produced this run.")
