# Databricks notebook source
# MAGIC %md
# MAGIC # 00c — Bronze: historical backfill via Dune
# MAGIC
# MAGIC Replaces the abandoned Bitquery-archive backfill (`00b_bronze_backfill_test.py`
# MAGIC / `fetch_historical_trades` in `bitquery_client.py`) -- that path needs an
# MAGIC Enterprise plan, confirmed via a live 403. Dune's curated `dex.trades`
# MAGIC (EVM) / `dex_solana.trades` (Solana) tables cover the same historical
# MAGIC range without that gate, confirmed via real bounded queries against both
# MAGIC venues (see docs/decisions.md, 2026-09-15).
# MAGIC
# MAGIC **KNOWN GAP, accepted deliberately**: rows written by this notebook have
# MAGIC `tx_fee_native`/`tx_fee_usd` = NULL and `source = 'dune'` -- neither Dune
# MAGIC table carries a usable on-chain transaction fee. Live rows from
# MAGIC `00_bronze_dex_ingestion.py` (`source = 'bitquery'`) are unaffected. See
# MAGIC `dune_client.py`'s module docstring and `docs/decisions.md`.
# MAGIC
# MAGIC Writes straight into `lawrence.bronze.dex_trades` (append, same as the
# MAGIC live ingestion notebook) -- Silver already dedups on
# MAGIC `(dex_venue, tx_hash, token_id)`, so re-running a chunk here is a no-op
# MAGIC downstream, not a duplicate.
# MAGIC
# MAGIC Chunked by day to keep each Dune query's result set modest (unverified
# MAGIC at scale for our exact filtered venues -- daily chunks are a conservative
# MAGIC starting point, not a measured optimum). At the measured cost of the
# MAGIC cost-test queries (~0.08 credits/query), 30 days x 2 venues = 60 queries
# MAGIC is a few credits, trivial against the 2,500-credit free trial.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from ingestion.bitquery_client import ChainFamily, Venue  # noqa: E402
from ingestion.dune_client import DuneClient, DuneVenueFilter, to_bronze_row  # noqa: E402

# COMMAND ----------

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")

BACKFILL_DAYS = 30
BRONZE_TABLE = "lawrence.bronze.dex_trades"

# Same Venue identities as the live ingestion notebook (00_bronze_dex_ingestion.py)
# -- kept in sync manually since Databricks notebooks aren't import-friendly
# across files; if VENUES there changes, mirror it here.
VENUES: list[Venue] = [
    Venue(
        name="raydium_v4",
        chain_family=ChainFamily.SOLANA,
        dex_program_or_address="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    ),
    Venue(
        name="uniswap_v3_base",
        chain_family=ChainFamily.EVM,
        network="base",
        dex_program_or_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
    ),
]

# How to filter Dune's tables down to each Venue above -- see dune_client.py's
# DuneVenueFilter docstring for why this can't just reuse dex_program_or_address
# on the EVM side (Dune's EVM rows carry per-pool addresses, not the shared
# factory address).
DUNE_VENUE_FILTERS: dict[str, DuneVenueFilter] = {
    "raydium_v4": DuneVenueFilter(
        table="dex_solana.trades",
        where_sql="project_program_id = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'",
    ),
    "uniswap_v3_base": DuneVenueFilter(
        table="dex.trades",
        where_sql="blockchain = 'base' AND project = 'uniswap' AND version = '3'",
    ),
}

# COMMAND ----------

client = DuneClient(api_key=DUNE_API_KEY)
ingested_at = datetime.now(timezone.utc)

# Dune's curated tables lag real time by roughly a day (confirmed
# 2026-09-15) -- anchor the window a bit further back so every chunk is
# safely inside their actual coverage rather than racing the lag.
window_end = datetime.now(timezone.utc) - timedelta(days=2)
window_start = window_end - timedelta(days=BACKFILL_DAYS)

print(f"Backfill window: [{window_start.isoformat()}, {window_end.isoformat()})")
print(f"Venues: {[v.name for v in VENUES]}")

# COMMAND ----------

bronze_rows = []
chunk_start = window_start
chunk_n = 0
total_chunks = BACKFILL_DAYS * len(VENUES)

while chunk_start < window_end:
    chunk_end = min(chunk_start + timedelta(days=1), window_end)
    for venue in VENUES:
        chunk_n += 1
        venue_filter = DUNE_VENUE_FILTERS[venue.name]
        try:
            dune_rows = client.fetch_historical_trades(
                venue_filter=venue_filter,
                since_iso=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                till_iso=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            bronze_rows.extend(to_bronze_row(r, venue, ingested_at) for r in dune_rows)
            print(
                f"[{chunk_n}/{total_chunks}] {venue.name} "
                f"[{chunk_start.date()}, {chunk_end.date()}): {len(dune_rows)} rows"
            )
        except (RuntimeError, TimeoutError) as exc:
            # One chunk failing shouldn't take down the whole backfill --
            # log it and keep going; re-run this notebook afterward to
            # pick up any chunks that failed (harmless re-fetch of the
            # ones that succeeded, since Silver dedups downstream).
            print(f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start.date()}, {chunk_end.date()}) FAILED: {exc}")
    chunk_start = chunk_end

print(f"\nTotal rows fetched across all chunks: {len(bronze_rows)}")

# COMMAND ----------

if bronze_rows:
    from pyspark.sql import Row

    df = spark.createDataFrame([Row(**r) for r in bronze_rows])
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(BRONZE_TABLE)
    )
    print(f"Appended {df.count()} rows to {BRONZE_TABLE} (source=dune)")
else:
    print("No rows fetched -- nothing written.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next**: check your Dune credit balance to confirm real cost at this
# MAGIC scale, then run `01_silver_token_trades.py` (or the whole job) with a
# MAGIC wide enough lookback to pick up these backfilled rows -- they're
# MAGIC historical, so the usual 60-minute Silver lookback won't see them;
# MAGIC widen `LOOKBACK_MINUTES` there temporarily for one run, or query Bronze
# MAGIC directly for `source = 'dune'` rows if that's simpler.
