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
# MAGIC **COST INCIDENT, 2026-09-15, and the fixes below.** A first run
# MAGIC (`SELECT *`, daily chunks) fetched 7 real days fine, then hit a 402
# MAGIC mid-pagination on day 8 and crashed -- the whole 7 days was lost because
# MAGIC the notebook (then) wrote to Bronze only once, at the very end. Fixed
# MAGIC on every front that mattered:
# MAGIC - **Column cost**: queries now select only what `to_bronze_row` needs
# MAGIC   (`DuneVenueFilter.columns`), not `SELECT *`.
# MAGIC - **Mid-pagination loss**: `DuneCreditsExhausted` now carries whatever
# MAGIC   rows earlier pages of the SAME query already fetched
# MAGIC   (`partial_rows`) -- a 402 on page 4 no longer discards pages 1-3.
# MAGIC - **End-of-run loss**: Bronze is written after every chunk (and after
# MAGIC   a partial failure), never batched to one write at the end.
# MAGIC - **Re-run cost**: a small progress table
# MAGIC   (`lawrence.bronze._dune_backfill_progress`) records every chunk that
# MAGIC   fully succeeded. Re-running this notebook -- same API key, or a
# MAGIC   DIFFERENT one (e.g. after swapping in a friend's key when yours runs
# MAGIC   out) -- skips anything already done instead of re-paying for it.
# MAGIC   This is why the backfill window below is a FIXED pair of dates, not
# MAGIC   "now minus N days": chunk boundaries have to be identical across
# MAGIC   runs/sessions/accounts for the skip-check to match anything.
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
from pyspark.sql import Row  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# COMMAND ----------

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")
# ^ If you're swapping in a different Dune account's key partway through
# (e.g. your own ran out, using a friend's now), update this secret value
# and just re-run the whole notebook -- the progress table below makes
# that safe; nothing already-completed gets re-fetched or re-paid-for.

# FIXED window -- not "now minus N days". Chunk boundaries must be
# byte-identical across separate runs (different sessions, different API
# keys) for the progress-table skip-check to match anything. Dune's
# curated tables lag real time by ~1 day (confirmed 2026-09-15), hence
# ending a couple days back rather than at "now".
BACKFILL_WINDOW_START = datetime(2026, 8, 14, tzinfo=timezone.utc)
BACKFILL_WINDOW_END = datetime(2026, 9, 13, tzinfo=timezone.utc)
CHUNK_HOURS = 24  # daily -- proven fine at real volume (7 days succeeded
                  # before the incident); no need for anything finer.

BRONZE_TABLE = "lawrence.bronze.dex_trades"
PROGRESS_TABLE = "lawrence.bronze._dune_backfill_progress"

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
# to pull -- explicit, not `SELECT *` (see COST INCIDENT note above). Only
# what dune_client.to_bronze_row actually reads for that chain family.
DUNE_VENUE_FILTERS: dict[str, DuneVenueFilter] = {
    "raydium_v4": DuneVenueFilter(
        table="dex_solana.trades",
        # NOT an address match on project_program_id -- verified wrong live
        # (2026-09-15): that column returns ~0 rows for version=4 even
        # though Dune shows 6,064,410 real v4 trades / $406M volume for
        # August 2026 (likely because many v4 swaps route through an
        # aggregator, so Dune records the outer router program there
        # instead of Raydium's own address). Matches uniswap_v3_base's
        # existing project+version approach instead.
        # version is an integer column in dex_solana.trades (Trino/
        # DuneSQL doesn't implicitly coerce varchar '4' -> integer for
        # equality -- this is the likely cause of the 2026-09-15
        # QUERY_STATE_FAILED errors seen once this filter was deployed).
        where_sql="project = 'raydium' AND version = 4",
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

def _log_progress(venue_name: str, chunk_start_str: str, chunk_end_str: str, status: str, rows_written: int) -> None:
    """Appends one row to the progress table. Append-only, same as
    Bronze itself -- multiple rows for the same chunk are harmless (the
    skip-check below only cares whether ANY 'success' row exists for that
    key), and this is far simpler than a Delta MERGE for what's just a
    bookkeeping log."""
    row = Row(
        venue_name=venue_name,
        chunk_start=chunk_start_str,
        chunk_end=chunk_end_str,
        status=status,  # "success" | "partial" | "failed"
        rows_written=rows_written,
        completed_at=datetime.now(timezone.utc),
    )
    (
        spark.createDataFrame([row])
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(PROGRESS_TABLE)
    )


# Explicit schema, not inferred from the Row objects. Dune-sourced rows
# always have tx_fee_native/tx_fee_usd = None (the documented fee gap) --
# across a whole batch that's an all-null column, which Spark's type
# inference cannot assign a type to at all (CANNOT_DETERMINE_TYPE,
# confirmed live 2026-09-15). An explicit schema sidesteps inference
# entirely, which also makes the earlier CANNOT_MERGE_TYPE class of bug
# structurally impossible here going forward.
_BRONZE_SCHEMA = StructType([
    StructField("chain_family", StringType()),
    StructField("network", StringType()),
    StructField("dex_venue", StringType()),
    StructField("block_time", StringType()),
    StructField("tx_hash", StringType()),
    StructField("dex_protocol", StringType()),
    StructField("buy_token_id", StringType()),
    StructField("buy_symbol", StringType()),
    StructField("buy_amount", DoubleType()),
    StructField("buy_amount_usd", DoubleType()),
    StructField("sell_token_id", StringType()),
    StructField("sell_symbol", StringType()),
    StructField("sell_amount", DoubleType()),
    StructField("sell_amount_usd", DoubleType()),
    StructField("tx_fee_native", DoubleType()),
    StructField("tx_fee_usd", DoubleType()),
    StructField("raw_json", StringType()),
    StructField("source", StringType()),
    StructField("ingested_at", TimestampType()),
])


def _write_bronze(dune_rows: list[dict], venue: Venue, ingested_at: datetime) -> int:
    if not dune_rows:
        return 0
    bronze_rows = [to_bronze_row(r, venue, ingested_at) for r in dune_rows]
    # Plain dicts, not Row(**r) -- Row(**kwargs) is documented to sort its
    # fields alphabetically by name internally, and rather than rely on
    # unverified assumptions about how that resolves against a separately
    # authored explicit schema, dicts-with-schema is the unambiguous,
    # officially documented form (name-matched against schema.fields,
    # never positional). Verified 2026-09-15 after this exact ambiguity
    # was caught in the Silver notebook's copy of this pattern.
    df = spark.createDataFrame(bronze_rows, schema=_BRONZE_SCHEMA)
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(BRONZE_TABLE)
    )
    return len(bronze_rows)


# Only "success" blocks a re-fetch -- "partial"/"failed" chunks are
# retried wholesale next run (cheap: at most one page's worth of
# re-fetch for a partial chunk) rather than risk permanently losing the
# unfetched remainder by marking them done.
completed_success: set[tuple[str, str, str]] = set()
if spark.catalog.tableExists(PROGRESS_TABLE):
    for r in spark.table(PROGRESS_TABLE).filter("status = 'success'").select(
        "venue_name", "chunk_start", "chunk_end"
    ).collect():
        completed_success.add((r.venue_name, r.chunk_start, r.chunk_end))
print(f"{len(completed_success)} chunk(s) already completed in a previous run -- will be skipped.")

# COMMAND ----------

client = DuneClient(api_key=DUNE_API_KEY)
ingested_at = datetime.now(timezone.utc)

total_rows_written = 0
chunk_start = BACKFILL_WINDOW_START
chunk_n = 0
total_chunks = int(
    (BACKFILL_WINDOW_END - BACKFILL_WINDOW_START).total_seconds() // 3600 // CHUNK_HOURS
) * len(VENUES)
aborted = False

while chunk_start < BACKFILL_WINDOW_END and not aborted:
    chunk_end = min(chunk_start + timedelta(hours=CHUNK_HOURS), BACKFILL_WINDOW_END)
    chunk_start_str = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
    chunk_end_str = chunk_end.strftime("%Y-%m-%d %H:%M:%S")

    for venue in VENUES:
        chunk_n += 1
        key = (venue.name, chunk_start_str, chunk_end_str)
        if key in completed_success:
            print(f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start_str}, {chunk_end_str}) already done -- skipping")
            continue

        venue_filter = DUNE_VENUE_FILTERS[venue.name]
        try:
            dune_rows = client.fetch_historical_trades(
                venue_filter=venue_filter, since_iso=chunk_start_str, till_iso=chunk_end_str
            )
        except DuneCreditsExhausted as exc:
            # Salvage whatever pages already succeeded before this one
            # failed -- see DuneCreditsExhausted.partial_rows. Logged as
            # "partial", NOT "success", so a future run retries the
            # whole chunk rather than treating it as done.
            rows_salvaged = _write_bronze(exc.partial_rows, venue, ingested_at)
            total_rows_written += rows_salvaged
            _log_progress(venue.name, chunk_start_str, chunk_end_str, "partial", rows_salvaged)
            print(
                f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start_str}, {chunk_end_str}) "
                f"ABORTING RUN (salvaged {rows_salvaged} partial rows): {exc}"
            )
            aborted = True
            break
        except (RuntimeError, TimeoutError) as exc:
            # One chunk failing for another reason shouldn't take down
            # the rest -- log (as retryable, not done) and move on.
            _log_progress(venue.name, chunk_start_str, chunk_end_str, "failed", 0)
            print(f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start_str}, {chunk_end_str}) FAILED: {exc}")
            continue

        try:
            rows_written = _write_bronze(dune_rows, venue, ingested_at)
        except Exception as exc:
            # Anything at the write layer (a Spark type-inference error,
            # e.g. the 2026-09-15 CANNOT_MERGE_TYPE incident, or any
            # other surprise) shouldn't kill the whole run either --
            # this fetch is already paid for, so log it retryable and
            # move on rather than losing everything after this point.
            _log_progress(venue.name, chunk_start_str, chunk_end_str, "failed", 0)
            print(
                f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start_str}, {chunk_end_str}) "
                f"WRITE FAILED (fetched {len(dune_rows)} rows, not written): {exc}"
            )
            continue

        total_rows_written += rows_written
        _log_progress(venue.name, chunk_start_str, chunk_end_str, "success", rows_written)
        completed_success.add(key)
        print(f"[{chunk_n}/{total_chunks}] {venue.name} [{chunk_start_str}, {chunk_end_str}): {rows_written} rows written")

    chunk_start = chunk_end

print(f"\nTotal rows written to {BRONZE_TABLE} (source=dune) this run: {total_rows_written}")
if aborted:
    print("Run stopped early on credit exhaustion -- re-run (same key once topped up, or a different key) to continue; already-completed chunks will be skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next**: check your Dune credit balance and divide by the number of
# MAGIC chunks that *fully succeeded* above (see the progress table:
# MAGIC `SELECT * FROM lawrence.bronze._dune_backfill_progress WHERE status = 'success'`)
# MAGIC -- that's the real per-query cost at this narrower column set.
# MAGIC
# MAGIC Once you have real backfilled data, Silver's usual 60-minute lookback
# MAGIC won't see it -- widen `LOOKBACK_MINUTES` in `01_silver_token_trades.py`
# MAGIC for one run, or query Bronze directly for `source = 'dune'` rows.
