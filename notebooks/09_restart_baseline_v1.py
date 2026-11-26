# Databricks notebook source
# MAGIC %md
# MAGIC # LAWRENCE paper engine — uninterrupted baseline
# MAGIC Run all cells on Python notebook compute with access to the LAWRENCE catalog. This uses notebook 07’s replay and saves the inputs and expected results in a new run directory under the replay_fixtures volume. The final line gives the fixture path for the restart run.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pathlib import Path

spark.sql("CREATE SCHEMA IF NOT EXISTS lawrence.paper_sprint")
spark.sql("CREATE VOLUME IF NOT EXISTS lawrence.paper_sprint.replay_fixtures")
fixture_root = Path("/Volumes/lawrence/paper_sprint/replay_fixtures")

# COMMAND ----------

import sys
import time

import pandas as pd
from pyspark.sql import functions as F

sys.path.insert(0, "../src")
from backtest.shadow_eval_v1 import frozen_trades  # noqa: E402
from decision.brokers import PaperBroker  # noqa: E402
from decision.contracts import ContractV1  # noqa: E402
from decision.engine import Engine  # noqa: E402
from decision.replay import parity, replay  # noqa: E402
from decision.types import RiskConfig  # noqa: E402
from serving import contract_v1 as c1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
N_TOKENS = 300
GRAD_FROM, GRAD_TO = "2026-10-03", "2026-10-06"
contract = ContractV1("../contracts/v1")          # hash check + verify_startup
print("v1 cutoff", contract.cutoff)

px = spark.table("lawrence.silver.token_price_1m")
tok = (px.groupBy("token_id")
       .agg(F.min(F.when(F.col("phase") == "curve", F.col("bar_start"))).alias("launch_time"),
            F.min_by("venue", "bar_start").alias("launch_venue"),
            F.min(F.when(F.col("phase") == "postgrad", F.col("bar_start"))).alias("first_pg"))
       .filter(F.col("launch_time").isNotNull() & (F.col("first_pg") >= GRAD_FROM) & (F.col("first_pg") < GRAD_TO))
       .filter(F.col("launch_venue").isin(*c1.UNIVERSE_VENUES))
       .orderBy(F.xxhash64("token_id")).limit(N_TOKENS))
ctx = tok.toPandas()
ctx.attrs = {}
ids = ctx["token_id"].tolist()
bars = (px.filter(F.col("token_id").isin(ids)).select("token_id", *c1.BAR_COLUMNS).toPandas())
bars.attrs = {}
bars["bar_start"] = pd.to_datetime(bars["bar_start"], utc=True)
print(f"{len(ids)} tokens, {len(bars):,} bars")

# COMMAND ----------

# Engine replay (per token, no portfolio limits, so tokens are independent)
t0 = time.time()
engine = Engine(contract, PaperBroker(0.0), RiskConfig.unlimited())
tokens = {r.token_id: {"bars": bars[bars["token_id"] == r.token_id].drop(columns="token_id"),
                       "launch_time": pd.Timestamp(r.launch_time, tz="UTC") if pd.Timestamp(r.launch_time).tzinfo is None
                       else pd.Timestamp(r.launch_time), "launch_venue": r.launch_venue}
          for r in ctx.itertuples(index=False)}
eng_trades = replay(engine, tokens)
print(f"engine: {len(eng_trades)} trades in {time.time() - t0:.0f}s; audit rows {len(engine.log.rows):,}")

# COMMAND ----------

# Backtest trades: the same tokens' Gold rows, same model + cutoff, rule 7, one position per token
cols = list(dict.fromkeys(["token_id", "assessment_time", *c1.FEATURE_COLUMNS, "dex_venue", "r_up_360m", "t_up_360m"]))
g = (spark.table("lawrence.gold.entry_candidates").filter(F.col("token_id").isin(ids))
     .filter("is_postgrad_at_t = true").select(*cols).toPandas())
g.attrs = {}
g["assessment_time"] = pd.to_datetime(g["assessment_time"], utc=True)
g = g.merge(ctx[["token_id", "first_pg"]], on="token_id")
g["first_pg"] = pd.to_datetime(g["first_pg"], utc=True)
g = g[g["assessment_time"] + pd.Timedelta(minutes=c1.HORIZON_MINUTES)
      <= g["first_pg"] + pd.Timedelta(hours=c1.POSTGRAD_OBSERVED_HOURS)].reset_index(drop=True)
g["score"] = contract.booster.predict(c1.to_model_frame(g[c1.FEATURE_COLUMNS].to_dict("records"), contract.manifest["categories"]))
bt = frozen_trades(g, contract.cutoff)
print(f"backtest: {len(bt)} trades")

# COMMAND ----------

crash_tick = pd.Timestamp("2026-10-04 13:54:00", tz="UTC")
result = parity(eng_trades, bt)
print(result)
assert result == dict(matched=55, only_engine=0, only_backtest=0,
                      max_abs_return_diff=0.0, status="PASS"), result
assert (len(ctx), len(bars), len(eng_trades), len(bt)) == (300, 53290, 55, 55)
assert not eng_trades.duplicated(["token_id", "entry_time"]).any()
assert not bt.duplicated(["token_id", "assessment_time"]).any()

p16 = eng_trades.loc[eng_trades.position_id == "p16"].squeeze()
assert p16.entry_time <= crash_tick < p16.exit_time, "Crash point is outside p16"
display(eng_trades.loc[eng_trades.position_id == "p16",
                       ["position_id", "entry_time", "entry_price", "exit_time", "r_net"]])

# COMMAND ----------

import json
from uuid import uuid4

run_dir = fixture_root / uuid4().hex
run_dir.mkdir()
audit = engine.log.frame()
for name, frame in {
    "token_context": ctx,
    "bars": bars,
    "backtest_trades": bt,
    "engine_trades": eng_trades,
    "audit": audit,
}.items():
    # Encode in memory so the volume only receives sequential writes.
    (run_dir / f"{name}.parquet").write_bytes(frame.to_parquet(index=False))

manifest = {
    "model_sha256": contract.manifest["model_sha256"],
    "cutoff": contract.cutoff,
    "crash_tick_utc": crash_tick.isoformat(),
    "open_position_at_crash": "p16",
    "tokens": len(ctx), "bars": len(bars), "trades": len(eng_trades),
    "audit_rows": len(audit), "parity": result,
}
# A manifest marks a complete baseline. Partial runs never replace a good one.
(run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"BASELINE PASS | {len(eng_trades)} trades | fixture: {run_dir}")
