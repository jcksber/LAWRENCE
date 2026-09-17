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
# MAGIC **2026-09-17: now a distributed groupBy/applyInPandas job** -- same
# MAGIC reasoning and same three-attempts history as notebook 02 (see its
# MAGIC header comment and docs/decisions.md). `.collect()` hung on real
# MAGIC volume; `toLocalIterator()` + a suspected per-partition recompute led
# MAGIC to trying `.cache()`, which isn't supported on serverless compute;
# MAGIC landed on pushing the per-token computation onto the cluster instead
# MAGIC of routing 10M+ rows through the driver at all.

# COMMAND ----------

import sys

sys.path.append("../src")
from features.gold_features import build_exit_rows, sample_assessment_times  # noqa: E402

# COMMAND ----------

SILVER_TABLE = "lawrence.silver.token_trades"
GOLD_TABLE = "lawrence.gold.exit_candidates"

# COMMAND ----------

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema -- same rationale as notebook 02's _ENTRY_SCHEMA; also
# doubles as applyInPandas's required return schema below.
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
_EXIT_SCHEMA_COLUMNS = [f.name for f in _EXIT_SCHEMA]

# COMMAND ----------

import pandas as pd


def _process_token_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs on an executor, once per distinct token_id. applyInPandas does
    NOT guarantee row order within a group, so sort locally first --
    gold_features.py's whole contract depends on `trades` being sorted
    ascending by trade_time."""
    pdf = pdf.sort_values("trade_time")
    trades = pdf.to_dict("records")
    if not trades:
        return pd.DataFrame(columns=_EXIT_SCHEMA_COLUMNS)

    token_id = trades[0]["token_id"]
    out_rows = []
    for t in sample_assessment_times(trades):
        out_rows.extend(build_exit_rows(token_id, t, trades))

    if not out_rows:
        return pd.DataFrame(columns=_EXIT_SCHEMA_COLUMNS)
    return pd.DataFrame(out_rows)[_EXIT_SCHEMA_COLUMNS]


gold_df = (
    spark.table(SILVER_TABLE)
    .groupBy("token_id")
    .applyInPandas(_process_token_group, schema=_EXIT_SCHEMA)
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
print(f"Wrote {_written} rows to {GOLD_TABLE} (full overwrite, "
      "same batch-regenerate rationale as notebook 02)")
# Same expected-low-count-on-fresh-history caveat as notebook 02, plus one
# more constraint: each row also needs a synthetic entry price at T-h for
# at least one h in EXIT_SIMULATED_HOLDS, so exit candidates need MORE
# trailing history per token than entry candidates do.
