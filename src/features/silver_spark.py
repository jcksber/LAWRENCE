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


# Venues whose token-of-interest comes from pool metadata (each pool's base
# mint) instead of the quote denylist. LaunchLab only: its quote mint is
# per-pool and, since 2026-09-07, can be ANY token (StonkFun pairs launches
# against xStocks, PreStocks, ZEC, other launched coins) -- see
# docs/venue_mechanics.md. pump.fun keeps the denylist (its quote is always
# SOL, verified on all 108M Bronze rows).
METADATA_BASE_VENUES = ["raydium_launchlab"]


def _pair_key(a: Column, b: Column) -> Column:
    """Order-independent key for a mint pair, so a trade (buy=A, sell=B) and
    a pool (base=B, quote=A) meet in a plain equi-join."""
    return F.concat_ws("|", F.least(a, b), F.greatest(a, b))


def pool_pairs(pool_metadata_df: DataFrame) -> DataFrame:
    """lawrence.bronze.pool_metadata -> one row per mint pair with its base
    mint, for METADATA_BASE_VENUES. Pairs whose pools disagree on which side
    is the base (someone created both A/B and B/A) are dropped as
    ambiguous; those trades fall back to the denylist rule."""
    return (
        pool_metadata_df.filter(F.col("venue").isin(METADATA_BASE_VENUES))
        .select(
            _pair_key(F.col("base_mint"), F.col("quote_mint")).alias("pair_key"),
            F.col("base_mint"),
        )
        .groupBy("pair_key")
        .agg(F.collect_set("base_mint").alias("bases"))
        .filter(F.size("bases") == 1)
        .select("pair_key", F.col("bases")[0].alias("pool_base_mint"))
    )


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


_NUMERIC_BRONZE_COLUMNS = [
    "buy_amount",
    "buy_amount_usd",
    "sell_amount",
    "sell_amount_usd",
    "tx_fee_native",
    "tx_fee_usd",
]


def build_silver(bronze_df: DataFrame, pool_pairs_df: DataFrame | None = None) -> DataFrame:
    """Bronze two-sided swaps -> Silver one-row-per-(swap, token-of-interest).

    Default (denylist) rule, identical to to_silver_rows(): 0 rows for a
    quote/quote swap, 1 for the common token/quote case, 2 when neither side
    is a quote asset.

    With `pool_pairs_df` (from `pool_pairs()`), trades on METADATA_BASE_VENUES
    whose mint pair matches a known pool use that pool's base mint as the
    token-of-interest -- exactly one row per swap, whatever the quote is.
    Unmatched/ambiguous pairs fall back to the denylist rule."""
    swaps = (
        bronze_df.withColumn("swap_id", swap_id(bronze_df))
        .dropDuplicates(["swap_id"])
        .withColumn("trade_time", parse_block_time(F.col("block_time")))
    )
    # 2026-10-01: Bronze's numeric columns are STRING in the real table --
    # the first rows came from Bitquery, whose GraphQL returns decimals as
    # JSON strings, and that fixed the table's column types. The old
    # row-by-row path hid this (to_silver_rows float()s everything); Spark
    # arithmetic on strings fails with DATATYPE_MISMATCH. Cast AFTER
    # swap_id so the hash still sees the exact stored text. try_cast, not
    # cast: an unparseable value becomes NULL (row then handled by the
    # normal NULL rules) instead of failing the whole rebuild under ANSI.
    for c in _NUMERIC_BRONZE_COLUMNS:
        if c in swaps.columns:
            swaps = swaps.withColumn(c, F.expr(f"try_cast({c} AS DOUBLE)"))
    if pool_pairs_df is not None:
        swaps = swaps.withColumn(
            "pair_key", _pair_key(F.col("buy_token_id"), F.col("sell_token_id"))
        ).join(F.broadcast(pool_pairs_df), on="pair_key", how="left")
        use_meta = F.col("dex_venue").isin(METADATA_BASE_VENUES) & F.col("pool_base_mint").isNotNull()
        swaps = swaps.withColumn(
            "buy_is_quote",
            F.when(use_meta, F.col("buy_token_id") != F.col("pool_base_mint")).otherwise(_is_quote("buy_token_id")),
        ).withColumn(
            "sell_is_quote",
            F.when(use_meta, F.col("sell_token_id") != F.col("pool_base_mint")).otherwise(_is_quote("sell_token_id")),
        )
    else:
        swaps = swaps.withColumn("buy_is_quote", _is_quote("buy_token_id")).withColumn(
            "sell_is_quote", _is_quote("sell_token_id")
        )
    return _side(swaps, "buy", "sell").unionByName(_side(swaps, "sell", "buy")).select(*SILVER_COLUMNS)
