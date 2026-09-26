# Databricks notebook source
# MAGIC %md
# MAGIC # 00e — Dune cost-test: post-graduation trades + pool metadata
# MAGIC
# MAGIC **Why (2026-09-22, see docs/venue_mechanics.md + docs/decisions.md).**
# MAGIC Bronze has pump.fun and LaunchLab *bonding-curve* trades only. When a
# MAGIC token graduates, trading moves to PumpSwap (pump.fun) or Raydium CPMM
# MAGIC (LaunchLab) -- neither ingested -- so winners vanish from our data the
# MAGIC moment they win. Fixing that is an ADDITIVE Bronze pull (nothing
# MAGIC existing is re-fetched), scoped to:
# MAGIC 1. **Pool metadata** (one row per pool): base mint, quote mint,
# MAGIC    platform, created_at. Replaces the quote-denylist guess with each
# MAGIC    pool's real base mint -- required for LaunchLab, whose quote can be
# MAGIC    any token since 2026-09-07.
# MAGIC 2. **Post-graduation trades, graduates only, bounded horizon**: trades
# MAGIC    on the graduation pool within `GRAD_HORIZON_HOURS` of its creation,
# MAGIC    for pools created inside the backfill window.
# MAGIC
# MAGIC Same discipline as 00d: every query is an aggregate, read-only, nothing
# MAGIC written anywhere. Cells are staged smallest-first. **Check your Dune
# MAGIC credit balance before and after EACH cell and note the delta** -- stop
# MAGIC and report back if any cell costs more than expected.
# MAGIC
# MAGIC Table names are from Dune's spellbook source (verified 2026-09-22):
# MAGIC `pumpswap_solana.pools`, `raydium_launchlab_v1.stg_swaps`,
# MAGIC `dex_solana.trades`. If a cell fails with "table not found", stop and
# MAGIC paste the error rather than guessing a replacement.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from ingestion.dune_client import DuneClient  # noqa: E402

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")
client = DuneClient(api_key=DUNE_API_KEY)

# Same fixed window as 00c_bronze_backfill_dune.py.
BACKFILL_WINDOW_START = datetime(2026, 8, 14, tzinfo=timezone.utc)
BACKFILL_WINDOW_END = datetime(2026, 9, 13, tzinfo=timezone.utc)

# Upper end of label_spec.md's 24-72h activity-window placeholder. Labels
# never look further past a pool's creation than this, so neither does
# the pull.
GRAD_HORIZON_HOURS = 72

# Stage-1 sample: one day, for per-day cost before running the full window.
SAMPLE_DAY_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
SAMPLE_DAY_END = SAMPLE_DAY_START + timedelta(days=1)


def _ts(dt: datetime) -> str:
    return f"TIMESTAMP '{dt.strftime('%Y-%m-%d %H:%M:%S')}'"


def _run(label: str, sql: str) -> None:
    print(f"--- {label} ---")
    for row in client.run_sql(sql):
        print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 (cheapest): confirm Dune's project/version names, ONE hour
# MAGIC
# MAGIC Which `project`/`version` values PumpSwap and Raydium CPMM trades
# MAGIC carry in `dex_solana.trades`. Cells 3-4 filter on these -- confirm
# MAGIC before trusting them.

# COMMAND ----------

_run(
    "project/version names, 1h",
    f"""
SELECT project, version, COUNT(*) AS n_trades
FROM dex_solana.trades
WHERE block_time >= {_ts(SAMPLE_DAY_START)}
  AND block_time <  {_ts(SAMPLE_DAY_START + timedelta(hours=1))}
  AND project IN ('pumpswap', 'raydium', 'pumpdotfun', 'raydium_launchlab')
GROUP BY 1, 2
ORDER BY n_trades DESC
""",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: pool metadata sizes, full window
# MAGIC
# MAGIC How many pools we'd pull metadata for, and what they're quoted in.
# MAGIC Both source tables are small pool-level/call-level tables, not trades.

# COMMAND ----------

_run(
    "PumpSwap pools created in window, by quote mint",
    f"""
SELECT quoteMint, COUNT(*) AS n_pools
FROM pumpswap_solana.pools
WHERE created_at >= {_ts(BACKFILL_WINDOW_START)}
  AND created_at <  {_ts(BACKFILL_WINDOW_END)}
GROUP BY 1
ORDER BY n_pools DESC
LIMIT 20
""",
)

_run(
    "LaunchLab pools active in window, by platform + quote mint",
    f"""
SELECT platform_name, quote_token_mint,
       COUNT(DISTINCT pool_id) AS n_pools,
       COUNT(*) AS n_curve_trades
FROM raydium_launchlab_v1.stg_swaps
WHERE block_date >= DATE '{BACKFILL_WINDOW_START.date()}'
  AND block_date <  DATE '{BACKFILL_WINDOW_END.date()}'
GROUP BY 1, 2
ORDER BY n_pools DESC
LIMIT 30
""",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: post-graduation PumpSwap trades — ONE-day sample first
# MAGIC
# MAGIC Trades on PumpSwap pools created on `SAMPLE_DAY`, within
# MAGIC `GRAD_HORIZON_HOURS` of pool creation. Joined on pool address
# MAGIC (`dex_solana.trades.project_program_id` = pool). This is the cost
# MAGIC driver -- multiply its credit delta by ~30 for the full-window
# MAGIC estimate before running cell 4.
# MAGIC
# MAGIC Note: PumpSwap pools aren't only pump.fun graduates (anyone can create
# MAGIC one). `n_pools_matching_curve_mints` counts only pools whose base mint
# MAGIC also traded on the pump.fun curve in our window -- the ones we
# MAGIC actually need.

# COMMAND ----------

_run(
    "PumpSwap post-graduation trades, 1-day sample",
    f"""
WITH pools AS (
  SELECT pool, baseMint, created_at
  FROM pumpswap_solana.pools
  WHERE created_at >= {_ts(SAMPLE_DAY_START)}
    AND created_at <  {_ts(SAMPLE_DAY_END)}
),
curve_mints AS (
  SELECT DISTINCT token_bought_mint_address AS mint
  FROM dex_solana.trades
  WHERE project = 'pumpdotfun'
    AND block_time >= {_ts(SAMPLE_DAY_START - timedelta(days=1))}
    AND block_time <  {_ts(SAMPLE_DAY_END)}
)
SELECT
  COUNT(*) AS n_trades,
  COUNT(DISTINCT p.pool) AS n_pools_traded,
  COUNT(DISTINCT CASE WHEN cm.mint IS NOT NULL THEN p.pool END) AS n_pools_matching_curve_mints,
  COUNT_IF(cm.mint IS NOT NULL) AS n_trades_matching_curve_mints
FROM dex_solana.trades t
JOIN pools p ON t.project_program_id = p.pool
LEFT JOIN curve_mints cm ON cm.mint = p.baseMint
WHERE t.project = 'pumpswap'
  AND t.block_time >= {_ts(SAMPLE_DAY_START)}
  AND t.block_time <  {_ts(SAMPLE_DAY_END + timedelta(hours=GRAD_HORIZON_HOURS))}
  AND t.block_time <  p.created_at + INTERVAL '{GRAD_HORIZON_HOURS}' HOUR
""",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: LaunchLab graduates on Raydium CPMM — ONE-day sample
# MAGIC
# MAGIC CPMM pool addresses differ from the LaunchLab curve pool, so this
# MAGIC joins on mint instead: CPMM trades touching a base mint launched on
# MAGIC LaunchLab. **Before running**, set `CPMM_VERSION` to whatever cell 1
# MAGIC showed for Raydium CPMM (Dune's spellbook models it as a separate
# MAGIC raydium version; don't assume the number).

# COMMAND ----------

CPMM_VERSION = None  # set from cell 1's output, e.g. 5

assert CPMM_VERSION is not None, "Set CPMM_VERSION from cell 1 first."
_run(
    "LaunchLab graduates' CPMM trades, 1-day sample",
    f"""
WITH ll_mints AS (
  SELECT base_token_mint AS mint, MIN(block_time) AS first_curve_trade
  FROM raydium_launchlab_v1.stg_swaps
  WHERE block_date >= DATE '{(SAMPLE_DAY_START - timedelta(days=1)).date()}'
    AND block_date <  DATE '{SAMPLE_DAY_END.date()}'
  GROUP BY 1
)
SELECT
  COUNT(*) AS n_trades,
  COUNT(DISTINCT m.mint) AS n_graduated_mints_traded
FROM dex_solana.trades t
JOIN ll_mints m
  ON m.mint IN (t.token_bought_mint_address, t.token_sold_mint_address)
WHERE t.project = 'raydium' AND t.version = {CPMM_VERSION}
  AND t.block_time >= {_ts(SAMPLE_DAY_START)}
  AND t.block_time <  {_ts(SAMPLE_DAY_END + timedelta(hours=GRAD_HORIZON_HOURS))}
  AND t.block_time <  m.first_curve_trade + INTERVAL '{GRAD_HORIZON_HOURS}' HOUR
""",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the results
# MAGIC
# MAGIC - Cell 2 row counts = size of the metadata pull (expect small).
# MAGIC - Cells 3-4 `n_trades` x ~30 days = size of the post-graduation pull;
# MAGIC   combine with 00c's measured ~780 credits / million rows fetched for a
# MAGIC   real credit estimate. Also note each cell's own scan cost.
# MAGIC - If the matching-curve-mints subset (cell 3) is much smaller than all
# MAGIC   PumpSwap pools, the real pull filters to that subset.
# MAGIC
# MAGIC Report back cell outputs + credit deltas; the numbers go into
# MAGIC docs/decisions.md before any real pull is written.
