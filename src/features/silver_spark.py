"""
Spark-native Silver transform for `lawrence.silver.token_trades`.

Same row-shaping rules as `silver_token_trades.to_silver_rows()` (which
stays as the pure-Python reference implementation, runnable in CI without
PySpark), but expressed as DataFrame operations so it runs distributed.
The old notebook `.collect()`ed each Bronze chunk to the driver and looped
in Python -- fine at ~10M rows, a guaranteed driver OOM at the 131M-row
Dune backfill (see docs/decisions.md, 2026-09-22).

Differences from `to_silver_rows()`, all deliberate (docs/silver_spec.md):
- dedupes on a content-based `swap_id` (not `(dex_venue, tx_hash,
  token_id)`, which collapsed real multi-swap transactions);
- `trade_time` is a real TimestampType, parsed from Bronze's string
  `block_time` in either of its two formats;
- adds `trade_date` (partition column).
"""
from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from features.silver_token_trades import EVM_QUOTE_DENYLIST, SOLANA_QUOTE_DENYLIST

SILVER_COLUMNS = [
    "token_id",
    "chain_family",
    "network",
    "dex_venue",
    "dex_protocol",
    "trade_time",
    "trade_date",
    "side",
    "price_usd",
    "price_quote",
    "amount_usd",
    "quote_amount",
    "tx_fee_usd",
    "chain_native_fee",
    "tx_hash",
    "swap_id",
    "quote_token_id",
]

# Fields that define "the same swap". Two rows agreeing on all of these are
# a re-fetch duplicate (partial-chunk salvage + retry in the Dune backfill);
# two rows in one tx that differ on any of them are distinct real swaps
# (router multi-hops, bundled launch buys) and must both survive.
_SWAP_ID_FIELDS = [
    "dex_venue",
    "tx_hash",
    "buy_token_id",
    "sell_token_id",
    "buy_amount",
    "sell_amount",
]


def parse_block_time(col: Column) -> Column:
    """Bronze `block_time` is a string in two shapes:
      - "2026-08-16 00:00:01.000 UTC"  (Dune rows written before
        dune_client._normalize_block_time existed)
      - "2026-09-15T00:15:11Z" / "...+00:00"  (everything since)
    Rewrite the first into the second, then parse once. The zone is always
    explicit in the string, so the result doesn't depend on the session
    time zone. try_to_timestamp (not to_timestamp) so an unexpected third
    shape becomes NULL -- which the notebook counts and reports -- instead
    of failing the whole job under ANSI mode."""
    iso = F.regexp_replace(col, r" UTC$", "Z")
    iso = F.regexp_replace(iso, r"^(\d{4}-\d{2}-\d{2}) ", "$1T")
    return F.try_to_timestamp(iso)


def swap_id(df: DataFrame) -> Column:
    """sha2-256 over the swap's identifying fields. coalesce each to '' so
    a NULL doesn't shift later fields left (concat_ws skips NULLs)."""
    parts = [F.coalesce(F.col(c).cast("string"), F.lit("")) for c in _SWAP_ID_FIELDS]
    return F.sha2(F.concat_ws("|", *parts), 256)


def _is_quote(token_col: str) -> Column:
    """NULL for an unknown chain_family -> those rows are dropped by the
    side filters below (pure-Python version raises instead)."""
    return (
        F.when(F.col("chain_family") == "solana", F.col(token_col).isin(list(SOLANA_QUOTE_DENYLIST)))
        .when(F.col("chain_family") == "evm", F.lower(F.col(token_col)).isin(list(EVM_QUOTE_DENYLIST)))
    )


def _side(df: DataFrame, side: str, quote_side: str) -> DataFrame:
    amount = F.col(f"{side}_amount")
    amount_usd = F.col(f"{side}_amount_usd")
    quote_amount = F.col(f"{quote_side}_amount")
    # 2026-09-30: price_usd and price_quote computed separately; a row is
    # kept if EITHER exists (Dune can't USD-price many exotic quotes).
    # Same rule as to_silver_rows().
    price_usd = F.when(amount_usd.isNotNull(), amount_usd / amount)
    price_quote = F.when(quote_amount.isNotNull() & (quote_amount != 0), quote_amount / amount)
    return df.filter(
        (F.col(f"{side}_is_quote") == F.lit(False))
        & amount.isNotNull()
        & (amount != 0)
        & (amount_usd.isNotNull() | (quote_amount.isNotNull() & (quote_amount != 0)))
    ).select(
        F.col(f"{side}_token_id").alias("token_id"),
        "chain_family",
        "network",
        "dex_venue",
        "dex_protocol",
        "trade_time",
        F.to_date("trade_time").alias("trade_date"),
        F.lit(side).alias("side"),
        price_usd.alias("price_usd"),
        price_quote.alias("price_quote"),
        amount_usd.alias("amount_usd"),
        quote_amount.alias("quote_amount"),
        F.col("tx_fee_usd").cast("double").alias("tx_fee_usd"),
        F.col("tx_fee_native").cast("double").alias("chain_native_fee"),
        "tx_hash",
        "swap_id",
        F.col(f"{quote_side}_token_id").alias("quote_token_id"),
    )


def build_silver(bronze_df: DataFrame) -> DataFrame:
    """Bronze two-sided swaps -> Silver one-row-per-(swap, token-of-interest).

    0 rows for a quote/quote swap, 1 for the common token/quote case, 2 when
    neither side is a quote asset -- identical rule to to_silver_rows()."""
    swaps = (
        bronze_df.withColumn("swap_id", swap_id(bronze_df))
        .dropDuplicates(["swap_id"])
        .withColumn("trade_time", parse_block_time(F.col("block_time")))
        .withColumn("buy_is_quote", _is_quote("buy_token_id"))
        .withColumn("sell_is_quote", _is_quote("sell_token_id"))
    )
    return _side(swaps, "buy", "sell").unionByName(_side(swaps, "sell", "buy")).select(*SILVER_COLUMNS)
