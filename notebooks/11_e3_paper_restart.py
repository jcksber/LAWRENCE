# Databricks notebook source
# MAGIC %md
# MAGIC # Restart after the first E3 half-sale
# MAGIC Run all twice on Python notebook compute with Structured Streaming. Keep the same fixture and trial on both runs. The first run records an uninterrupted reference, then aborts a streaming replay after the first 2× sale commits. The second starts a fresh Python session and retries the interrupted batch. Use a new trial name for a new drill.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import re
import sys
from pathlib import Path
from uuid import uuid4

src_dir = str(Path("../src").resolve())
contract_dir = str(Path("../contracts/v1").resolve())
sys.path.insert(0, src_dir)
from decision.e3 import E3PaperEngine
from decision.e3_restart import INJECTED, TABLE, check_research, inputs, load_record, save_record
from decision.replay import replay
from decision.types import RiskConfig

dbutils.widgets.text("fixture_id", "ef8bc69e4cc648cf95146d00191ae49b")
dbutils.widgets.text("trial", "trial-01")
fixture_id = dbutils.widgets.get("fixture_id")
trial_name = dbutils.widgets.get("trial")
if not re.fullmatch(r"[0-9a-f]{32}", fixture_id) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", trial_name):
    raise ValueError("Enter the fixture directory name and a short alphanumeric trial name.")
volume = "/Volumes/lawrence/paper_sprint/replay_fixtures"
fixture = f"{volume}/{fixture_id}"
trial = f"{fixture_id}/{trial_name}"
stream_checkpoint = f"{volume}/_e3_stream_v1/{trial}"
session_id = uuid4().hex
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE} (trial STRING, phase STRING, payload BINARY) USING DELTA")

# COMMAND ----------

if load_record(spark, trial, "baseline") is None:
    contract, tokens = inputs(fixture, contract_dir, Path(fixture) / "bars.parquet")
    control = E3PaperEngine(contract, RiskConfig.unlimited())
    replay(control, tokens)
    check_research(control, tokens)
    assert any(e["reason"] == "2x" for e in control.log.rows), "No half-sale in this fixture"
    save_record(spark, trial, "baseline", control.snapshot())
    print(f"E3 BASELINE | {len(control.closed)} closed | {len(control.open)} still open")

# COMMAND ----------

mode = "recover" if load_record(spark, trial, "half_sale") else "crash"


def handle_file_batch(batch_df, batch_id):
    # Standard compute can execute foreachBatch in an isolated Python interpreter.
    import sys
    sys.path.insert(0, src_dir)
    from decision.e3_restart import process_batch
    process_batch(batch_df, batch_id, fixture=fixture, contract_dir=contract_dir,
                  trial=trial, mode=mode, session_id=session_id)


# One immutable input file = one retryable micro-batch. Market time comes from its bars.
stream = (spark.readStream.format("cloudFiles").option("cloudFiles.format", "binaryFile")
          .option("pathGlobFilter", "bars.parquet")
          .schema("path STRING, modificationTime TIMESTAMP, length LONG, content BINARY")
          .load(fixture))
try:
    (stream.writeStream.foreachBatch(handle_file_batch).option("checkpointLocation", stream_checkpoint)
     .trigger(availableNow=True).start().awaitTermination())
except Exception as err:
    if mode != "crash" or INJECTED not in str(err):
        raise

half_sale = load_record(spark, trial, "half_sale")
assert half_sale is not None, "First half-sale was never saved"
if mode == "crash":
    print(f"E3 CRASH SAVED | {half_sale['tick']} | {half_sale['position']} | 50% remaining | Run all again")
else:
    final = load_record(spark, trial, "recovered")
    assert final is not None and final["batch"] == half_sale["batch"], "Interrupted batch did not recover"
    print(f"E3 RESTART PASS | {final['fills']} unique fills | {len(final['state']['closed'])} closed | "
          f"{len(final['state']['open'])} open | batch {final['batch']}")
