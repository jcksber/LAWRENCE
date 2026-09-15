# Databricks notebook source
# MAGIC %md
# MAGIC # 00b — Bronze: backfill COST TEST (run this, don't skip to a real backfill)
# MAGIC
# MAGIC **Purpose: measure real Bitquery credit cost of one small, bounded
# MAGIC historical query before committing to a real backfill.** Continuous
# MAGIC 15-min live polling was projected to exhaust API credits within days
# MAGIC (see docs/decisions.md, 2026-09-15) -- the recurring job is paused.
# MAGIC Bitquery's dashboard doesn't expose a per-query cost breakdown, so
# MAGIC this has to be measured empirically: check your credit balance/usage
# MAGIC right before running this cell, then again right after, and note the
# MAGIC delta before running anything bigger.
# MAGIC
# MAGIC Scope: ONE venue (raydium_v4), ONE bounded 1-hour window, `dataset:
# MAGIC archive`. This also validates the `till` + `archive` query shape
# MAGIC actually works against our `DEXTrades` root -- confirmed in Bitquery's
# MAGIC docs only against `DEXTradeByTokens`, not tested against ours yet.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from ingestion.bitquery_client import BitqueryClient, ChainFamily, Venue  # noqa: E402

# COMMAND ----------

API_KEY = dbutils.secrets.get(scope="lawrence", key="bitquery_api_key")

TEST_VENUE = Venue(
    name="raydium_v4",
    chain_family=ChainFamily.SOLANA,
    dex_program_or_address="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
)

# A 1-hour window from a few days ago -- old enough to plausibly need the
# archive dataset rather than the live path, recent enough to be within
# archive's "from 1 June 2024" coverage with room to spare.
till = datetime.now(timezone.utc) - timedelta(days=3)
since = till - timedelta(hours=1)

print(f"CHECK YOUR BITQUERY CREDIT BALANCE NOW, BEFORE RUNNING THE NEXT CELL.")
print(f"Test window: [{since.isoformat()}, {till.isoformat()})")

# COMMAND ----------

client = BitqueryClient(api_key=API_KEY)

trades = client.fetch_historical_trades(
    venue=TEST_VENUE,
    since_iso=since.isoformat(),
    till_iso=till.isoformat(),
    limit=1000,
)

print(f"Fetched {len(trades)} trades in this 1-hour window.")
if len(trades) == 1000:
    print("WARNING: hit the 1000-row cap -- this window has more trades than "
          "returned, would need to be bisected for a real backfill.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Now check your Bitquery credit balance again and compare to before
# MAGIC this notebook ran.** That delta is the real cost of one bounded
# MAGIC 1-hour/1-venue archive query -- report it back before we design the
# MAGIC actual backfill's chunk size and total scope.
