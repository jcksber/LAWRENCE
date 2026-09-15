# Databricks notebook source
# MAGIC %md
# MAGIC # 00c — Bronze: historical backfill via Dune
# MAGIC
# MAGIC Replaces the abandoned Bitquery-archive backfill (`00b_bronze_backfill_test.py`
# MAGIC / `fetch_historical_trades` in `bitquery_client.py`) -- that path needs an
# MAGIC Enterprise plan, confirmed via a live 403. Dune's curated `dex.trades`
# MAGIC (EVM) / `dex_solana.trades` (Solana) tables cover the same historical
# MAGIC range without that gate (see docs/decisions.md, 2026-09-15).
# MAGIC
# MAGIC **COST INCIDENT, 2026-09-15**: a first version of this notebook selected
# MAGIC all ~30 columns (`SELECT *`) chunked by day, and burned through a
# MAGIC 2,500-credit free trial partway through a SINGLE day/venue chunk
# MAGIC (paginated past row offset 415,000 before a 402). Worse, the 402 was an
# MAGIC unhandled `requests.HTTPError` that crashed the whole notebook before
# MAGIC the (single, end-of-run) Bronze write ever executed -- every credit
# MAGIC spent produced zero rows in Bronze. Fixed on both fronts:
# MAGIC - `dune_client.py` now selects only the columns `to_bronze_row` actually
# MAGIC   needs (not `SELECT *`), and raises `DuneCreditsExhausted` immediately
# MAGIC   on a 402 instead of a raw, unhandled `HTTPError`.
# MAGIC - This notebook now writes to Bronze after EVERY chunk, not once at the
# MAGIC   end, so a future failure can never again erase already-fetched data
# MAGIC   -- the failed run above actually fetched 7 good days before the 8th
# MAGIC   day's chunk drained the rest, but none of it was ever written, and
# MAGIC   it's confirmed gone.
# MAGIC - `BACKFILL_DAYS` reset to a real test value below with DAILY chunks
# MAGIC   (daily chunking itself was fine -- 7 of them succeeded; no need for
# MAGIC   finer than that). Confirm real per-query cost at the new narrower
# MAGIC   column set before raising it toward the original 30-day target.
# MAGIC
# MAGIC **KNOWN GAP, accepted deliberately**: rows written by this notebook have
# MAGIC `tx_fee_native`/`tx_fee_usd` = NULL and `source = 'dune'` -- neither Dune
# MAGIC table carries a usable on-chain transaction fee. Live rows from
# MAGIC `00_bronze_dex_ingestion.py` (`source = 'bitquery'`) are unaffected. See
# MAGIC `dune_client.py`'s module docstring and `docs/decisions.md`.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from ingestion.bitquery_client import ChainFamily, Venue  # noqa: E402
from ingestion.dune_client import (  # noqa: E402
    DuneClient,
    DuneCreditsExhausted,
    DuneVenueFilter,
    to_bronze_row,
)

# COMMAND ----------

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")

# START SMALL-ish. This is a re-run after a cost incident (see markdown
# above). Correction: the failed run actually got 7 full days through
# fine (daily chunking) before the 8th day's chunk spiked and drained the
# rest -- so daily chunking itself is fine, hourly was an overcorrection.
# That data never made it to Bronze though (crashed before the one-shot
# write ran, and this was BEFORE the per-chunk-write fix below existed) --
# confirmed gone, not recoverable. 5 days at the new narrower column set
# is a real test of whether that set meaningfully lowers cost vs. the
# SELECT * run, before raising this toward the 30-day target.
BACKFILL_DAYS = 5
CHUNK_HOURS = 24  # daily -- proven fine at real volume, no need for finer

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

# How to filter Dune's tables down to each Venue above, AND which columns
# to pull -- explicit, not `SELECT *` (see COST INCIDENT note). Only what
# dune_client.to_bronze_row actually reads for that chain family.
DUNE_VENUE_FILTERS: dict[str, DuneVenueFilter] = {
    "raydium_v4": DuneVenueFilter(
        table="dex_solana.trades",
        where_sql="project_program_id = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'",
        columns=[
            "block_time", "tx_id", "project", "version", "version_name",
            "token_bought_symbol", "token_sold_symbol",
            "token_bought_amount", "token_sold_amount", "amount_usd",
            "token_bought_mint_address", "token_sold_mint_address",
        ],
    ),
    "uniswap_v3_base": DuneVenueFilter(
        table="dex.trades",
        where_sql="blockchain = 'base' AND project = 'uniswap' AND version = '3'",
        columns=[
            "block_time", "tx_hash", "project", "version",
            "token_bought_symbol", "token_sold_symbol",
            "token_bought_amount", "token_sold_amount", "amount_usd",
            "token_bought_address", "token_sold_address",
        ],
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
print(f"Venues: {[v.name for v in VENUES]}, chunk size: {CHUNK_HOURS}h")

# COMMAND ----------

from pyspark.sql import Row  # noqa: E402

total_rows_written = 0
chunk_start = window_start
chunk_n = 0
total_chunks = int((window_end - window_start).total_seconds() // 3600 // CHUNK_HOURS) * len(VENUES)
aborted = False

while chunk_start < window_end and not aborted:
    chunk_end = min(chunk_start + timedelta(hours=CHUNK_HOURS), window_end)
    for venue in VENUES:
        chunk_n += 1
        venue_filter = DUNE_VENUE_FILTERS[venue.name]
        try:
            dune_rows = client.fetch_historical_trades(
                venue_filter=venue_filter,
                since_iso=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                till_iso=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except DuneCreditsExhausted as exc:
            # Every remaining chunk will fail the same way -- stop the
            # whole run now instead of burning retry time on each of
            # them. Whatever already got written (see below, written
            # per-chunk, not batched) is safe.
            print(f"[{chunk_n}/{total_chunks}] {venue.name} ABORTING RUN: {exc}")
            aborted = True
            break
        except (RuntimeError, TimeoutError) as exc:
            # One chunk failing for another reason shouldn't take down
            # the rest -- log and move on.
            print(f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start}, {chunk_end}) FAILED: {exc}")
            continue

        print(
            f"[{chunk_n}/{total_chunks}] {venue.name} "
            f"[{chunk_start}, {chunk_end}): {len(dune_rows)} rows"
        )
        if dune_rows:
            bronze_rows = [to_bronze_row(r, venue, ingested_at) for r in dune_rows]
            df = spark.createDataFrame([Row(**r) for r in bronze_rows])
            (
                df.write.format("delta")
                .mode("append")
                .option("mergeSchema", "true")
                .saveAsTable(BRONZE_TABLE)
            )
            total_rows_written += len(bronze_rows)
    chunk_start = chunk_end

print(f"\nTotal rows written to {BRONZE_TABLE} (source=dune) this run: {total_rows_written}")
if aborted:
    print("Run stopped early on credit exhaustion -- see message above.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next**: check your Dune credit balance and divide by the number of
# MAGIC chunks that actually completed above -- that's the real per-query cost
# MAGIC at this narrower column set. Report it back before raising
# MAGIC `BACKFILL_DAYS`/`CHUNK_HOURS` any further.
# MAGIC
# MAGIC Once you have real backfilled data, Silver's usual 60-minute lookback
# MAGIC won't see it -- widen `LOOKBACK_MINUTES` in `01_silver_token_trades.py`
# MAGIC for one run, or query Bronze directly for `source = 'dune'` rows.
