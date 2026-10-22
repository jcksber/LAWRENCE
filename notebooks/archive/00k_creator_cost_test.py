# Databricks notebook source
# MAGIC %md
# MAGIC # 00k — Cost test: creator + early-buyer features (v3 part C, branch `v3`)
# MAGIC
# MAGIC One day (2026-09-20) only. **Note the Dune credit counter before each cell** and
# MAGIC paste the deltas. Cell 2 is a schema probe (verifies the decoded pump.fun create
# MAGIC table); cells 3–5 compute on Dune but return ONE row each (compute cost only);
# MAGIC cell 6 exports one day of each to measure row counts. Nothing reads data
# MAGIC at/after 2026-10-11 (the holdout).

# COMMAND ----------

import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append("../../src")
import importlib  # noqa: E402

import ingestion.creator_flow as cf  # noqa: E402

importlib.reload(cf)
from ingestion.dune_client import DuneClient  # noqa: E402

client = DuneClient(api_key=dbutils.secrets.get(scope="lawrence", key="dune_api_key"), poll_timeout_s=1800)
D0, D1 = datetime(2026, 9, 20, tzinfo=timezone.utc), datetime(2026, 9, 21, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 10, 11, tzinfo=timezone.utc)

# COMMAND ----------

# CELL 2 -- probe: launches per create instruction (1 hour) + create_v2's columns.
# If create_v2 lacks account_mint / creator / call_block_slot, stop and paste this.
# (done 2026-10-18: tables = pump_call_create + pump_call_create_v2; columns verified)
display(pd.DataFrame(client.run_sql(cf.PROBE_COUNTS_SQL)))
v2_sample = pd.DataFrame(client.run_sql(cf.PROBE_SAMPLE_SQL))
print("create_v2 columns:", sorted(v2_sample.columns))
display(v2_sample[[c for c in ("account_mint", "account_user", "creator", "call_block_time", "call_block_slot", "name", "symbol")
                   if c in v2_sample.columns]])

# COMMAND ----------

# CELL 2b -- full-day coverage check (cells 3-5 found only 248 launches / 10 grads).
# Paste both outputs. Cheap: count-only scans of decoded tables.
display(pd.DataFrame(client.run_sql(cf.PROBE_DAY_SQL)))
evt = pd.DataFrame(client.run_sql(cf.PROBE_EVT_SAMPLE_SQL))
print("createevent columns:", sorted(evt.columns))

# COMMAND ----------

# CELL 3 -- launches (all pump.fun creates that day): compute only
print(client.run_sql(cf.summarize(cf.creator_launches_sql(D0, D1))))

# COMMAND ----------

# CELL 4 -- early buyers for that day's graduates: compute only
print(client.run_sql(cf.summarize(cf.early_buyers_sql(D0, D1, CUTOFF))))

# COMMAND ----------

# CELL 5 -- creator trades for that day's graduates: compute only
print(client.run_sql(cf.summarize(cf.creator_trades_sql(D0, D1, CUTOFF))))

# COMMAND ----------

# CELL 6 -- export one day of the two small outputs (early buyers, creator trades)
eb = pd.DataFrame(client.run_sql(cf.early_buyers_sql(D0, D1, CUTOFF)))
ct = pd.DataFrame(client.run_sql(cf.creator_trades_sql(D0, D1, CUTOFF)))
print(f"early buyers: {len(eb):,} graduates; creator trades: {len(ct):,} rows")
display(eb.describe(include="all").T)
display(ct.groupby(["project", "is_buy"]).size())
