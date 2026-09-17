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
# MAGIC **2026-09-17: three attempts at real-scale volume, see docs/decisions.md.**
# MAGIC Against the real ~10.75M-row Dune backfill (this notebook was written
# MAGIC when Gold's own docs expected "few or zero rows" and was never
# MAGIC exercised at scale until now):
# MAGIC 1. The original `silver_df.collect()` -- everything into one Python
# MAGIC    list on the driver -- hit the same driver-memory hang already seen
# MAGIC    in Silver's own catch-up notebook.
# MAGIC 2. Replaced with `toLocalIterator()` (stream partitions to the driver
# MAGIC    one at a time) plus a manual per-token grouping loop. Ran into a
# MAGIC    suspected recompute-per-partition problem, for which the standard
# MAGIC    fix (`.cache()`) turned out to be unsupported on serverless compute
# MAGIC    (`PERSIST TABLE is not supported on serverless`).
# MAGIC 3. **Current approach:** stop routing 10M+ rows through the driver at
# MAGIC    all. `silver_df.groupBy("token_id").applyInPandas(...)` pushes the
# MAGIC    per-token feature/label computation onto the cluster as a real,
# MAGIC    distributed Spark operation -- each executor handles a subset of
# MAGIC    tokens in parallel, calling straight into the same
# MAGIC    `gold_features.py` functions. No driver memory pressure, no
# MAGIC    caching, no ordering assumptions to get right by hand -- Spark's
# MAGIC    own groupBy shuffle guarantees each token's rows land together.
# MAGIC    This also means the notebook's own Spark job progress (Tasks X/Y)
# MAGIC    is now an honest signal of real completion, since the heavy work
# MAGIC    happens as part of that job instead of invisibly afterward on the
# MAGIC    driver.

# COMMAND ----------

import sys

sys.path.append("../src")
from features.gold_features import build_entry_row, sample_assessment_times  # noqa: E402

# COMMAND ----------

SILVER_TABLE = "lawrence.silver.token_trades"
GOLD_TABLE = "lawrence.gold.entry_candidates"

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
# all-None, and doubles as applyInPandas's required return schema below.
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
_ENTRY_SCHEMA_COLUMNS = [f.name for f in _ENTRY_SCHEMA]

# COMMAND ----------

import pandas as pd


def _process_token_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs on an executor, once per distinct token_id (Spark's groupBy
    shuffle guarantees every row for a token lands in the same call).
    applyInPandas does NOT guarantee row order within a group, so sort
    locally first -- gold_features.py's whole contract depends on `trades`
    being sorted ascending by trade_time."""
    pdf = pdf.sort_values("trade_time")
    trades = pdf.to_dict("records")
    if not trades:
        return pd.DataFrame(columns=_ENTRY_SCHEMA_COLUMNS)

    token_id = trades[0]["token_id"]
    out_rows = []
    for t in sample_assessment_times(trades):
        row = build_entry_row(token_id, t, trades)
        if row is not None:
            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=_ENTRY_SCHEMA_COLUMNS)
    return pd.DataFrame(out_rows)[_ENTRY_SCHEMA_COLUMNS]


gold_df = (
    spark.table(SILVER_TABLE)
    .groupBy("token_id")
    .applyInPandas(_process_token_group, schema=_ENTRY_SCHEMA)
)

# COMMAND ----------

# 2026-09-17: the lawrence.gold SCHEMA itself (not just the table) had
# never been created -- bronze/silver's schemas exist from some earlier
# manual setup, gold's never did since no run had gotten this far before.
# Idempotent, so this is a no-op on every later run.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_TABLE.rsplit('.', 1)[0]}")

(
    gold_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)
_written = spark.table(GOLD_TABLE).count()
print(f"Wrote {_written} rows to {GOLD_TABLE} (full overwrite -- this "
      "is a batch regenerate, not an incremental append; re-running from "
      "scratch is the intended usage, unlike Bronze/Silver's MERGE)")
# A low/zero count against a freshly-started Bronze/Silver history is
# expected, not a bug: build_entry_row needs both a trailing feature window
# AND a real trade near T + HOLD_MINUTES, which a just-started ingestion
# history mostly won't have yet. Re-run this after Bronze/Silver have
# accumulated more history.
