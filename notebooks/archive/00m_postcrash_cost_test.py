# Databricks notebook source
# MAGIC %md
# MAGIC # 00m — Cost test: post-crash hourly bars (v4 post-crash track, branch `v3`)
# MAGIC One TRADE day (2026-09-27) of hourly bars for PumpSwap pools aged 24h–14d
# MAGIC (`ingestion.postcrash_bars`). **Note the Dune credit counter before each cell.**
# MAGIC Cell 2 = compute only (one row back); cell 3 = export the day. Nothing reads ≥ 2026-10-11.
# MAGIC Full dev pull would be ~28 trade days (09-13 → 10-10) — estimate = cell-3 credits × 28.

# COMMAND ----------

import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append("../../src")
import importlib  # noqa: E402

import ingestion.postcrash_bars as pc  # noqa: E402

importlib.reload(pc)
from ingestion.dune_client import DuneClient  # noqa: E402

client = DuneClient(api_key=dbutils.secrets.get(scope="lawrence", key="dune_api_key"), poll_timeout_s=1800)
UTC = timezone.utc
D0, D1 = datetime(2026, 9, 27, tzinfo=UTC), datetime(2026, 9, 28, tzinfo=UTC)
CUTOFF = datetime(2026, 10, 11, tzinfo=UTC)
SQL = pc.postcrash_hourly_bars_sql(D0, D1, CUTOFF)

# COMMAND ----------

# CELL 2 -- compute only: rows + pools for one trade day
print(client.run_sql(pc.summarize(SQL)))

# COMMAND ----------

# CELL 3 -- export the day; shape check
bars = pd.DataFrame(client.run_sql(SQL))
bars.attrs = {}
print(f"{len(bars):,} hourly bars, {bars['pool_id'].nunique():,} pools; "
      f"USD-priced share {bars['close_usd'].notna().mean():.1%}")
bars["age_d"] = (pd.to_datetime(bars["bar_start"].astype(str).str.replace(' UTC', ''), utc=True)
                 - pd.to_datetime(bars["pool_created_at"].astype(str).str.replace(' UTC', ''), utc=True)).dt.total_seconds() / 86400
display(bars.groupby(bars["age_d"].round()).agg(pools=("pool_id", "nunique"), bars=("pool_id", "size"),
                                                  median_trades=("n_trades", "median")).reset_index())
