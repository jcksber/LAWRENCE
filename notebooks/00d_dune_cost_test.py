# Databricks notebook source
# MAGIC %md
# MAGIC # 00d — Dune cost-test: is more history worth more Solana venues, or more calendar time?
# MAGIC
# MAGIC **Why this notebook exists (2026-09-18, see docs/decisions.md).** The v2
# MAGIC entry-model backtest still loses to Baseline B, but its ev/trade improves
# MAGIC every fold as the training window expands -- a data-starvation read, not a
# MAGIC "wrong model" read. Jack wants more training data faster than waiting weeks
# MAGIC for live ingestion to add it, spending as little as possible. Before
# MAGIC spending anything real, this notebook measures which of two levers actually
# MAGIC yields more distinct tokens per credit:
# MAGIC
# MAGIC 1. **More calendar time** -- extend `raydium_v4`'s backfill further back
# MAGIC    before `BACKFILL_WINDOW_START` (2026-08-14).
# MAGIC 2. **More Solana venues, same window** -- pump.fun/Orca/Meteora/etc. inside
# MAGIC    the window already paid for (2026-08-14 to 2026-09-13).
# MAGIC
# MAGIC **Cost discipline, staged on purpose:** every query below is a cheap
# MAGIC aggregate (`COUNT(*)`/`COUNT(DISTINCT ...)`/`GROUP BY`), never a full
# MAGIC trade-level `SELECT`. But an aggregate still has to SCAN every matching row
# MAGIC to compute it -- Dune bills "processing power, data scanned, and query-
# MAGIC engine time," not just what's returned (see docs/decisions.md's 2026-09-18
# MAGIC pricing research entry) -- and the discovery cell below is deliberately
# MAGIC UNFILTERED by project (that's the point, to see every Solana DEX in the
# MAGIC data), which scans more rows than any single-venue-filtered query this
# MAGIC backfill has run so far. **Check your Dune credit balance after EACH cell
# MAGIC before running the next one** -- this notebook is intentionally small,
# MAGIC narrow-window cells you can stop between, not one big batch job. If cell 1
# MAGIC (a 1-hour unfiltered window) costs more than expected, stop here and we
# MAGIC rethink before cell 2 (a 7-day unfiltered window) runs at all.
# MAGIC
# MAGIC Nothing here writes to Bronze or any other table -- this is read-only
# MAGIC measurement, not backfill. `docs/decisions.md` gets the real numbers once
# MAGIC you've run this and reported back what each cell actually cost.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from ingestion.dune_client import DuneClient  # noqa: E402

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")
client = DuneClient(api_key=DUNE_API_KEY)

# Same fixed backfill window as 00c_bronze_backfill_dune.py, for reference.
BACKFILL_WINDOW_START = datetime(2026, 8, 14, tzinfo=timezone.utc)
BACKFILL_WINDOW_END = datetime(2026, 9, 13, tzinfo=timezone.utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 (smallest, run first): what Solana DEXs exist, in ONE hour
# MAGIC
# MAGIC Deliberately a 1-hour, unfiltered-by-project window -- the same "tiny
# MAGIC window aggregate" size the very first (pre-incident) cost-test used. Not
# MAGIC meant to be a reliable venue ranking on its own (1 hour is noisy for
# MAGIC anything but the highest-volume venues) -- meant to confirm this kind of
# MAGIC query is cheap before cell 2 widens it to something actually useful.
# MAGIC **Check your Dune credit balance now, before running this cell, then
# MAGIC again right after, and note the delta before going any further.**

# COMMAND ----------

# One hour, starting exactly where the current backfill window ends --
# still within Dune's ~1-day curated-table lag (see docs/decisions.md),
# so this is real, queryable data, not "too recent to exist yet."
discovery_1h_start = BACKFILL_WINDOW_END
discovery_1h_end = discovery_1h_start + timedelta(hours=1)
discovery_1h_sql = f"""
SELECT project, version, COUNT(*) AS n_trades,
       COUNT(DISTINCT token_bought_mint_address) AS n_distinct_tokens_bought,
       SUM(amount_usd) AS total_volume_usd
FROM dex_solana.trades
WHERE block_time >= TIMESTAMP '{discovery_1h_start.strftime('%Y-%m-%d %H:%M:%S')}'
  AND block_time < TIMESTAMP '{discovery_1h_end.strftime('%Y-%m-%d %H:%M:%S')}'
GROUP BY project, version
ORDER BY n_trades DESC
"""
discovery_1h_rows = client.run_sql(discovery_1h_sql)
for row in discovery_1h_rows:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 (run only after checking cell 1's cost): same discovery, 7 days
# MAGIC
# MAGIC A 1-hour window is too noisy to size a real decision on -- this widens to
# MAGIC the last 7 days of the already-backfilled window (2026-09-06 to 09-13),
# MAGIC still unfiltered by project, so every real Solana DEX candidate shows up
# MAGIC with a large-enough sample to compare against `raydium_v4`'s own numbers
# MAGIC over the same 7 days (queried separately below, cell 3, for an apples-to-
# MAGIC apples comparison against what's already in Bronze).
# MAGIC **Check your Dune credit balance again after this cell.**

# COMMAND ----------

discovery_7d_start = datetime(2026, 9, 6, tzinfo=timezone.utc)
discovery_7d_sql = f"""
SELECT project, version, COUNT(*) AS n_trades,
       COUNT(DISTINCT token_bought_mint_address) AS n_distinct_tokens_bought,
       SUM(amount_usd) AS total_volume_usd
FROM dex_solana.trades
WHERE block_time >= TIMESTAMP '{discovery_7d_start.strftime('%Y-%m-%d %H:%M:%S')}'
  AND block_time < TIMESTAMP '{BACKFILL_WINDOW_END.strftime('%Y-%m-%d %H:%M:%S')}'
GROUP BY project, version
ORDER BY n_trades DESC
"""
discovery_7d_rows = client.run_sql(discovery_7d_sql)
for row in discovery_7d_rows:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: the "more calendar time" lever -- raydium_v4, 7 days BEFORE the
# MAGIC ## current backfill window starts
# MAGIC
# MAGIC Filtered by the same `project = 'raydium' AND version = 4` this backfill
# MAGIC already trusts (see 00c/decisions.md) -- a single-venue-filtered query,
# MAGIC cheaper than cells 1-2's unfiltered scan over the same span. Measures
# MAGIC "how many new distinct tokens does extending 7 more days backward buy,"
# MAGIC directly comparable to cell 2's `raydium` row for "how many new distinct
# MAGIC tokens does 7 days of a NEW venue buy, in a window already backfilled."

# COMMAND ----------

extend_time_start = datetime(2026, 8, 7, tzinfo=timezone.utc)  # 7 days before BACKFILL_WINDOW_START
extend_time_sql = f"""
SELECT COUNT(*) AS n_trades,
       COUNT(DISTINCT token_bought_mint_address) AS n_distinct_tokens_bought,
       SUM(amount_usd) AS total_volume_usd
FROM dex_solana.trades
WHERE block_time >= TIMESTAMP '{extend_time_start.strftime('%Y-%m-%d %H:%M:%S')}'
  AND block_time < TIMESTAMP '{BACKFILL_WINDOW_START.strftime('%Y-%m-%d %H:%M:%S')}'
  AND project = 'raydium' AND version = 4
"""
extend_time_rows = client.run_sql(extend_time_sql)
print(extend_time_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the results
# MAGIC
# MAGIC Compare, per 7-day window:
# MAGIC - Cell 3's `n_distinct_tokens_bought` (extend-time lever, raydium_v4 only)
# MAGIC - Cell 2's `n_distinct_tokens_bought` for whichever non-raydium/non-
# MAGIC   already-backfilled project(s) show real volume (add-venue lever, same
# MAGIC   window already in Bronze)
# MAGIC
# MAGIC Whichever gives more new distinct tokens for a comparable credit spend
# MAGIC (check your actual balance deltas, not just row/query counts -- Dune
# MAGIC bills compute, not rows) is the lever worth spending real money on. Once
# MAGIC you've run this and have real before/after credit numbers, report back
# MAGIC and this gets logged to docs/decisions.md as the actual (not estimated)
# MAGIC cost-per-token for each lever, before sizing a real backfill expansion.
