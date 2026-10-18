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

sys.path.append("../src")
import importlib  # noqa: E402

import ingestion.creator_flow as cf  # noqa: E402

importlib.reload(cf)
from ingestion.dune_client import DuneClient  # noqa: E402

client = DuneClient(api_key=dbutils.secrets.get(scope="lawrence", key="dune_api_key"), poll_timeout_s=1800)
D0, D1 = datetime(2026, 9, 20, tzinfo=timezone.utc), datetime(2026, 9, 21, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 10, 11, tzinfo=timezone.utc)

# COMMAND ----------

# CELL 2 -- schema probe. Expect a table like pump_call_create with columns
# account_mint, account_user, call_block_time, call_block_slot. If names differ,
# paste this output before running anything else.
display(pd.DataFrame(client.run_sql(cf.PROBE_SQL)))

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
