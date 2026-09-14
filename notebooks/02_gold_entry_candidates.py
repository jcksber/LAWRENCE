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

# COMMAND ----------

import sys
from collections import defaultdict

sys.path.append("../src")
from features.gold_features import build_entry_row, sample_assessment_times  # noqa: E402

# COMMAND ----------

SILVER_TABLE = "lawrence.silver.token_trades"
GOLD_TABLE = "lawrence.gold.entry_candidates"

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
        row = build_entry_row(token_id, t, trades)
        if row is not None:
            gold_rows.append(row)

print(f"Entry candidate rows produced: {len(gold_rows)}")
# A low/zero count against a freshly-started Bronze/Silver history is
# expected, not a bug: build_entry_row needs both a trailing feature window
# AND a real trade near T + HOLD_MINUTES, which a just-started ingestion
# history mostly won't have yet. Re-run this after Bronze/Silver have
# accumulated more history.

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
    print(f"Wrote {len(gold_rows)} rows to {GOLD_TABLE} (full overwrite -- this "
          "is a batch regenerate, not an incremental append; re-running from "
          "scratch is the intended usage, unlike Bronze/Silver's MERGE)")
else:
    print("No entry candidate rows produced.")
