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

# COMMAND ----------

import sys
from collections import defaultdict

sys.path.append("../src")
from features.gold_features import build_exit_rows, sample_assessment_times  # noqa: E402

# COMMAND ----------

SILVER_TABLE = "lawrence.silver.token_trades"
GOLD_TABLE = "lawrence.gold.exit_candidates"

silver_df = spark.table(SILVER_TABLE).orderBy("token_id", "trade_time")
silver_rows = [row.asDict() for row in silver_df.collect()]

by_token: dict[str, list[dict]] = defaultdict(list)
for row in silver_rows:
    by_token[row["token_id"]].append(row)

print(f"Silver rows: {len(silver_rows)} across {len(by_token)} distinct tokens")

# COMMAND ----------

gold_rows = []
for token_id, trades in by_token.items():
    for t in sample_assessment_times(trades):
        gold_rows.extend(build_exit_rows(token_id, t, trades))

print(f"Exit candidate rows produced: {len(gold_rows)}")
# Same expected-low-count-on-fresh-history caveat as notebook 02, plus one
# more constraint: each row also needs a synthetic entry price at T-h for
# at least one h in EXIT_SIMULATED_HOLDS, so exit candidates need MORE
# trailing history per token than entry candidates do.

# COMMAND ----------

from pyspark.sql import Row

if gold_rows:
    df = spark.createDataFrame([Row(**r) for r in gold_rows])
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
