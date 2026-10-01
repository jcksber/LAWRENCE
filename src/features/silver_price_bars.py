"""
`lawrence.silver.token_price_1m` -- ONE 1-minute price series per token,
stitching together everything we know about it:

- its trades in `silver.token_trades` (bonding curve on pumpdotfun /
  raydium_launchlab -> phase 'curve'; ordinary AMMs like raydium_v4 /
  uniswap_v3_base -> phase 'amm'), rolled up to 1-minute bars here; and
- its post-graduation bars from `bronze.postgrad_bars_1m` (PumpSwap /
  Raydium CPMM, already 1-minute on Dune) -> phase 'postgrad', taken from
  ONE canonical pool per token (the pool with the most trades; ~74k of
  109k PumpSwap pools never traded at all -- see docs/decisions.md).

Why one series per token matters (2026-10-01): Gold's labels are returns
over a hold window, so every price for a token must be in the SAME unit.
Each token gets one `price_unit`:
- 'usd'   if at least USD_COVERAGE_MIN of its trades have a Dune USD price;
- 'quote' otherwise -- prices in units of the token's PRIMARY quote token
  (its LaunchLab pool's quote mint if known, else the quote it traded
  against most). Rows quoted in any other token are left out, never mixed.

Spark-native (no driver collects). Orchestrated by
notebooks/01b_silver_token_price_1m.py; tested in
tests/test_silver_price_bars.py (skipped where PySpark isn't installed).
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

USD_COVERAGE_MIN = 0.95
CURVE_VENUES = ["pumpdotfun", "raydium_launchlab"]

PRICE_COLUMNS = [
    "token_id",
    "bar_start",
    "trade_date",
    "price_unit",
    "quote_token_id",
    "open",
    "high",
    "low",
    "close",
    "volume_usd",
    "volume_quote",
    "n_trades",
    "n_buys",
    "n_traders",
    "phase",
    "venue",
    "pool_id",
]


def dedupe_bars(bars_df: DataFrame) -> DataFrame:
    """One row per (venue, pool_id, bar_start): a partial Dune chunk that was
    later re-fetched left duplicates (~30k PumpSwap bars). Keep the newest."""
    w = Window.partitionBy("venue", "pool_id", "bar_start").orderBy(F.desc("ingested_at"))
    return bars_df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def canonical_bars(bars_df: DataFrame) -> DataFrame:
    """Keep only each base mint's busiest post-graduation pool."""
    per_pool = bars_df.groupBy("base_mint", "venue", "pool_id").agg(F.sum("n_trades").alias("_pool_trades"))
    w = Window.partitionBy("base_mint").orderBy(F.desc("_pool_trades"), "venue", "pool_id")
    best = (
        per_pool.withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .select("base_mint", "venue", "pool_id")
    )
    return bars_df.join(best, on=["base_mint", "venue", "pool_id"], how="inner")


def token_units(trades_df: DataFrame, canon_bars_df: DataFrame, pool_metadata_df: DataFrame) -> DataFrame:
    """Per token: primary quote token, USD coverage, and the chosen price_unit."""
    # Primary quote: LaunchLab pool's quote mint when known ...
    launch_quote = (
        pool_metadata_df.filter(F.col("venue") == "raydium_launchlab")
        .groupBy(F.col("base_mint").alias("token_id"))
        .agg(F.first("quote_mint").alias("launch_quote"))
    )
    # ... else the quote it traded against most.
    qcounts = trades_df.groupBy("token_id", "quote_token_id").count()
    wq = Window.partitionBy("token_id").orderBy(F.desc("count"), "quote_token_id")
    top_quote = (
        qcounts.withColumn("_rn", F.row_number().over(wq))
        .filter("_rn = 1")
        .select("token_id", F.col("quote_token_id").alias("top_quote"))
    )

    t_cov = trades_df.groupBy("token_id").agg(
        F.count("*").alias("t_total"),
        F.count("price_usd").alias("t_priced"),
    )
    b_cov = canon_bars_df.groupBy(F.col("base_mint").alias("token_id")).agg(
        F.sum("n_trades").alias("b_total"),
        F.sum(F.col("n_trades") - F.coalesce(F.col("n_unpriced"), F.lit(0))).alias("b_priced"),
        F.first("quote_mint").alias("bar_quote"),
    )
    units = (
        t_cov.join(b_cov, "token_id", "full_outer")
        .join(launch_quote, "token_id", "left")
        .join(top_quote, "token_id", "left")
        .withColumn(
            "usd_coverage",
            (F.coalesce("t_priced", F.lit(0)) + F.coalesce("b_priced", F.lit(0)))
            / (F.coalesce("t_total", F.lit(0)) + F.coalesce("b_total", F.lit(0))),
        )
        .withColumn("primary_quote", F.coalesce("launch_quote", "top_quote", "bar_quote"))
        .withColumn(
            "price_unit",
            F.when(F.col("usd_coverage") >= F.lit(USD_COVERAGE_MIN), F.lit("usd")).otherwise(F.lit("quote")),
        )
    )
    return units.select("token_id", "primary_quote", "usd_coverage", "price_unit")


def trade_bars(trades_df: DataFrame, units_df: DataFrame) -> DataFrame:
    """silver.token_trades -> 1-minute bars in each token's chosen unit."""
    t = trades_df.join(units_df, "token_id", "inner")
    price = F.when(F.col("price_unit") == "usd", F.col("price_usd")).otherwise(
        F.when(F.col("quote_token_id") == F.col("primary_quote"), F.col("price_quote"))
    )
    t = (
        t.withColumn("_price", price)
        .filter(F.col("_price").isNotNull())
        .withColumn("bar_start", F.date_trunc("minute", "trade_time"))
    )
    in_primary = F.col("quote_token_id") == F.col("primary_quote")
    return (
        t.groupBy("token_id", "bar_start", "price_unit", "primary_quote")
        .agg(
            F.min_by("_price", "trade_time").alias("open"),
            F.max("_price").alias("high"),
            F.min("_price").alias("low"),
            F.max_by("_price", "trade_time").alias("close"),
            F.sum("amount_usd").alias("volume_usd"),
            F.sum(F.when(in_primary, F.col("quote_amount"))).alias("volume_quote"),
            F.count("*").alias("n_trades"),
            F.count(F.when(F.col("side") == "buy", 1)).alias("n_buys"),
            F.concat_ws(",", F.array_sort(F.collect_set("dex_venue"))).alias("venue"),
        )
        .withColumn(
            "phase",
            F.when(F.col("venue").isin(CURVE_VENUES), F.lit("curve")).otherwise(F.lit("amm")),
        )
        .withColumn("n_traders", F.lit(None).cast("long"))  # Bronze trades carry no trader id
        .withColumn("pool_id", F.lit(None).cast("string"))
        .withColumnRenamed("primary_quote", "quote_token_id")
    )


def postgrad_bars(canon_bars_df: DataFrame, units_df: DataFrame) -> DataFrame:
    """Canonical post-graduation bars in each token's chosen unit."""
    b = canon_bars_df.withColumnRenamed("base_mint", "token_id").join(units_df, "token_id", "inner")
    is_usd = F.col("price_unit") == "usd"
    same_quote = F.col("quote_mint") == F.col("primary_quote")

    def pick(usd_col: str, q_col: str):
        return F.when(is_usd, F.col(usd_col)).otherwise(F.when(same_quote, F.col(q_col)))

    return (
        b.select(
            "token_id",
            "bar_start",
            "price_unit",
            F.col("primary_quote").alias("quote_token_id"),
            pick("open_usd", "open_q").alias("open"),
            pick("high_usd", "high_q").alias("high"),
            pick("low_usd", "low_q").alias("low"),
            pick("close_usd", "close_q").alias("close"),
            "volume_usd",
            F.when(same_quote, F.col("volume_quote")).alias("volume_quote"),
            "n_trades",
            "n_buys",
            "n_traders",
            F.lit("postgrad").alias("phase"),
            "venue",
            "pool_id",
        )
        .filter(F.col("close").isNotNull())
    )


def build_token_price_1m(
    trades_df: DataFrame,
    bars_df: DataFrame,
    pool_metadata_df: DataFrame,
) -> DataFrame:
    """All inputs -> lawrence.silver.token_price_1m (PRICE_COLUMNS).

    If a token has both a trade bar and a post-graduation bar in the same
    minute (graduation minute), the one with more trades wins."""
    canon = canonical_bars(dedupe_bars(bars_df))
    units = token_units(trades_df, canon, pool_metadata_df)
    both = trade_bars(trades_df, units).unionByName(postgrad_bars(canon, units))
    w = Window.partitionBy("token_id", "bar_start").orderBy(F.desc("n_trades"), "phase")
    return (
        both.withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .withColumn("trade_date", F.to_date("bar_start"))
        .select(*PRICE_COLUMNS)
    )
