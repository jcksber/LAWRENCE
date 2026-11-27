# Databricks notebook source
# MAGIC %md
# MAGIC # Save an open paper position
# MAGIC Select the fixture from notebook 09 and run all cells. This replays the frozen input through the selected minute and saves the engine state, simulated fills and audit history in one Delta commit.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import sys
from dataclasses import asdict
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

sys.path.insert(0, "../src")
from decision.brokers import PaperBroker
from decision.contracts import ContractV1
from decision.engine import Engine
from decision.replay import replay
from decision.types import RiskConfig

dbutils.widgets.text("fixture_id", "ef8bc69e4cc648cf95146d00191ae49b")
fixture_id = dbutils.widgets.get("fixture_id")
if len(fixture_id) != 32 or any(c not in "0123456789abcdef" for c in fixture_id):
    raise ValueError("Use the 32-character fixture directory name from notebook 09.")
fixture = Path("/Volumes/lawrence/paper_sprint/replay_fixtures") / fixture_id
manifest = json.loads((fixture / "manifest.json").read_text())
boundary = pd.Timestamp(manifest["crash_tick_utc"])
contract = ContractV1("../contracts/v1")
assert contract.manifest["model_sha256"] == manifest["model_sha256"] and contract.cutoff == manifest["cutoff"]
ctx = pd.read_parquet(fixture / "token_context.parquet").set_index("token_id")
bars = pd.read_parquet(fixture / "bars.parquet")
assert len(ctx) == manifest["tokens"] and len(bars) == manifest["bars"]
tokens = {
    token: {"bars": group.drop(columns="token_id"),
            "launch_time": pd.to_datetime(ctx.at[token, "launch_time"], utc=True),
            "launch_venue": ctx.at[token, "launch_venue"]}
    for token, group in bars.groupby("token_id")
}

# COMMAND ----------

engine = Engine(contract, PaperBroker(0.0), RiskConfig.unlimited())
replay(engine, tokens, through=boundary)
target_id = manifest["open_position_at_crash"]
target, = [p for p in engine.open.values() if p.position_id == target_id]
reference = pd.read_parquet(fixture / "engine_trades.parquet").set_index("position_id").loc[target_id]
assert target.token_id == reference.token_id and target.entry_time == reference.entry_time
assert target.entry_price == reference.entry_price and target.entry_time <= boundary < reference.exit_time

audit = engine.log.frame()
baseline_audit = pd.read_parquet(fixture / "audit.parquet")
assert len(audit) < len(baseline_audit)
# Match timestamps, decision order, IDs, prices and returns up to the boundary.
columns = ["ts", "event", "reason", "token_id", "position_id", "fill_px", "r_gross", "r_net"]
pd.testing.assert_frame_equal(audit[columns], baseline_audit[columns].iloc[:len(audit)],
                              check_dtype=False, check_exact=True)

# COMMAND ----------

spark.conf.set("spark.sql.session.timeZone", "UTC")
table = "lawrence.paper_sprint.replay_checkpoints_v1"
checkpoint_id = f"{fixture_id}/{boundary.isoformat()}"
all_positions = [*engine.closed, *engine.open.values()]
# The engine allocates one sequential ID per entry and retains every position.
assert {p.position_id for p in all_positions} == {f"p{i}" for i in range(1, len(all_positions) + 1)}
state = {
    "format_version": 1,
    "fixture_id": fixture_id,
    "model_sha256": manifest["model_sha256"],
    "cutoff": contract.cutoff,
    "last_tick_utc": boundary.isoformat(),
    "open": [asdict(p) for p in engine.open.values()],
    "closed": [asdict(p) for p in engine.closed],
    "next_position_id": len(all_positions) + 1,
    "risk_config": asdict(engine.cfg),
    "risk_day": engine.risk._day,
    "entries_today": engine.risk.entries_today,
    "realized_today_usd": engine.risk.realized_today_usd,
    "paused": engine.paused,
    "killed": engine.killed,
    "paper_slippage_bps": engine.broker.slippage_bps,
}


def encode_date(value):
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Unexpected checkpoint value: {type(value)}")


state_json = json.dumps(state, default=encode_date, allow_nan=False, sort_keys=True)
audit_bytes = audit.to_parquet(index=False)
checkpoint = spark.createDataFrame([(checkpoint_id, state_json, audit_bytes)],
                                   "checkpoint_id string, state_json string, audit_parquet binary")
(checkpoint.write.format("delta").mode("append")
 .option("txnAppId", "lawrence-checkpoint-v1/" + checkpoint_id).option("txnVersion", 0)
 .saveAsTable(table))

# COMMAND ----------

stored, = spark.table(table).where(F.col("checkpoint_id") == checkpoint_id).collect()
assert json.loads(stored.state_json) == json.loads(state_json)
pd.testing.assert_frame_equal(pd.read_parquet(BytesIO(stored.audit_parquet))[columns],
                              audit[columns], check_dtype=False, check_exact=True)
print(f"CHECKPOINT PASS | {target_id} open | {len(audit)} audit rows | {checkpoint_id}")
