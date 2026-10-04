"""
`lawrence.silver.token_price_1m` -- ONE 1-minute price series per token,
stitching together everything we know about it:

- its trades in `silver.token_trades` (bonding curve on pumpdotfun /
  raydium_launchlab -> phase 'curve'; ordinary AMMs like raydium_v4 /
  uniswap_v3_base -> phase 'amm'), rolled up to 1-minute bars here; and
- from 2026-09-13 on, its curve bars from `bronze.curve_bars_1m` (pump.fun /
  LaunchLab, aggregated to 1 minute ON Dune by notebook 00h -- raw curve
  trades past the first month were too expensive) -> phase 'curve';
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

from features.silver_token_trades import EVM_QUOTE_DENYLIST, SOLANA_QUOTE_DENYLIST

USD_COVERAGE_MIN = 0.95
CURVE_VENUES = ["pumpdotfun", "raydium_launchlab"]

# 2026-10-03: a token whose primary quote is one of these majors (SOL, USDC,
# USDT, WETH) is ALWAYS priced in USD. Previously the unit came only from
# USD coverage, and a graduated pump.fun token whose PumpSwap pool had
# enough USD-unpriced trades fell under 95%, was switched to quote units,
# and then lost all its PumpSwap bars (that pull has no quote-unit prices)
# -- i.e. vanished at graduation again (5,337 tokens).
USD_QUOTES = list(SOLANA_QUOTE_DENYLIST) + [a.lower() for a in EVM_QUOTE_DENYLIST]

# 2026-10-03: dust filters (see docs/decisions.md). Trades of a few
# lamports / raw token units give meaningless amount ratios -- e.g. exactly
# 0.001 SOL/token from 1 lamport for 1 raw unit -- and one such trade sets a
# bar's OHLC. 6.2% of pump.fun curve trades are under $0.01 (p05 $0.000135);
# at $0.01 a trade is ~67k lamports, so rounding error is < 0.01%. Real
# small trades above that (21% of pump.fun trades are under $1) are kept.
MIN_TRADE_USD = 0.01
# A pump.fun bonding-curve price is physically bounded: ~2.8e-8 SOL/token
# at launch (30 virtual SOL / 1.073B virtual tokens) up to ~4e-7 at
# graduation. Padded so only impossible prices are removed.
# DISABLED 2026-10-03: in real data 25.5% of graduated tokens' (and 15.4% of
# other tokens') non-dust curve trades fall BELOW this range -- p01 ~1e-10,
# 100-1000x under the assumed launch price. Either some tokens run on
# different curve parameters/decimals or Dune decodes some trades in other
# units; until that's understood, a fixed physical range is not safe. Set to
# a (low, high) tuple to re-enable.
PUMP_CURVE_PRICE_SOL: tuple[float, float] | None = None
WSOL = "So11111111111111111111111111111111111111112"
# Backstop for everything the two rules above can't see (quote-only trades,
# and post-graduation bars aggregated on Dune with dust inside them): drop
# prices more than this factor away from the token's own median. Dust spikes
# seen were 3,000-30,000x off; a genuine 100x post-graduation run still fits.
OUTLIER_BAND = 1000.0

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
    # 2026-10-04: per-bar flow stats. Quote units, only when the bar's quote
    # is the token's primary quote. Dune curve bars (from 2026-09-13) have
    # all four; trade bars from Silver trades have all but top_trader_share
    # (no trader id in Bronze); post-graduation bars have none yet.
    "volume_buy_q",
    "volume_sell_q",
    "max_trade_q",
    "top_trader_share",
    "phase",
    "venue",
    "pool_id",
]
FLOW_COLUMNS = ["volume_buy_q", "volume_sell_q", "max_trade_q", "top_trader_share"]


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
            F.when(
                F.col("primary_quote").isin(USD_QUOTES)
                | F.lower(F.col("primary_quote")).isin(USD_QUOTES)
                | (F.col("usd_coverage") >= F.lit(USD_COVERAGE_MIN)),
                F.lit("usd"),
            ).otherwise(F.lit("quote")),
        )
    )
    return units.select("token_id", "primary_quote", "usd_coverage", "price_unit")


def _within_band(df: DataFrame, col: str, band: float = OUTLIER_BAND) -> DataFrame:
    """Keep rows whose `col` is within `band`x of the token's own median."""
    med = df.groupBy("token_id").agg(F.percentile_approx(col, 0.5).alias("_med"))
    return (
        df.join(med, "token_id", "inner")
        .filter((F.col(col) <= F.col("_med") * band) & (F.col(col) >= F.col("_med") / band))
        .drop("_med")
    )


def trade_bars(trades_df: DataFrame, units_df: DataFrame) -> DataFrame:
    """silver.token_trades -> 1-minute bars in each token's chosen unit."""
    t = trades_df.join(units_df, "token_id", "inner")
    # Dust filters (2026-10-03, see constants above).
    pump_sol = (F.col("dex_venue") == "pumpdotfun") & (F.col("quote_token_id") == WSOL)
    t = t.filter(F.col("amount_usd").isNull() | (F.col("amount_usd") >= F.lit(MIN_TRADE_USD)))
    if PUMP_CURVE_PRICE_SOL is not None:
        t = t.filter(
            ~pump_sol
            | F.col("price_quote").isNull()
            | F.col("price_quote").between(PUMP_CURVE_PRICE_SOL[0], PUMP_CURVE_PRICE_SOL[1])
        )
    price = F.when(F.col("price_unit") == "usd", F.col("price_usd")).otherwise(
        F.when(F.col("quote_token_id") == F.col("primary_quote"), F.col("price_quote"))
    )
    # 2026-10-01: prices <= 0 are treated as missing. 01b's first run found
    # 306,057 bars with close <= 0 -- Dune reports amount_usd = 0 on dust
    # trades, so price_usd = 0 -- and a zero price breaks every return.
    t = t.withColumn("_price", F.when(price > 0, price)).filter(F.col("_price").isNotNull())
    t = _within_band(t, "_price")
    t = t.withColumn("bar_start", F.date_trunc("minute", "trade_time"))
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
            F.sum(F.when(in_primary & (F.col("side") == "buy"), F.col("quote_amount"))).alias("volume_buy_q"),
            F.sum(F.when(in_primary & (F.col("side") != "buy"), F.col("quote_amount"))).alias("volume_sell_q"),
            F.max(F.when(in_primary, F.col("quote_amount"))).alias("max_trade_q"),
            F.count("*").alias("n_trades"),
            F.count(F.when(F.col("side") == "buy", 1)).alias("n_buys"),
            F.concat_ws(",", F.array_sort(F.collect_set("dex_venue"))).alias("venue"),
        )
        .withColumn(
            "phase",
            F.when(F.col("venue").isin(CURVE_VENUES), F.lit("curve")).otherwise(F.lit("amm")),
        )
        .withColumn("n_traders", F.lit(None).cast("long"))  # Bronze trades carry no trader id
        .withColumn("top_trader_share", F.lit(None).cast("double"))
        .withColumn("pool_id", F.lit(None).cast("string"))
        .withColumnRenamed("primary_quote", "quote_token_id")
    )


def postgrad_bars(canon_bars_df: DataFrame, units_df: DataFrame, phase: str = "postgrad") -> DataFrame:
    """Dune-aggregated bars (canonical post-graduation pool, or -- with
    phase='curve' -- 00h's curve bars) in each token's chosen unit."""
    for c in FLOW_COLUMNS:  # postgrad_bars_1m has no flow stats (yet)
        if c not in canon_bars_df.columns:
            canon_bars_df = canon_bars_df.withColumn(c, F.lit(None).cast("double"))
    b = canon_bars_df.withColumnRenamed("base_mint", "token_id").join(units_df, "token_id", "inner")
    is_usd = F.col("price_unit") == "usd"
    same_quote = F.col("quote_mint") == F.col("primary_quote")

    def pick(usd_col: str, q_col: str):
        v = F.when(is_usd, F.col(usd_col)).otherwise(F.when(same_quote, F.col(q_col)))
        return F.when(v > 0, v)  # <= 0 -> missing, same rule as trade_bars

    b = b.select(
        "*",
        pick("open_usd", "open_q").alias("_o"),
        pick("high_usd", "high_q").alias("_h"),
        pick("low_usd", "low_q").alias("_l"),
        pick("close_usd", "close_q").alias("_c"),
    )
    # 2026-10-03: these bars were aggregated on Dune with dust trades inside,
    # so apply the per-token outlier band: drop bars whose close is out of
    # band, and treat out-of-band open/high/low as missing (falls back to
    # close-based values below).
    b = _within_band(b.filter(F.col("_c").isNotNull()), "_c")
    med = b.groupBy("token_id").agg(F.percentile_approx("_c", 0.5).alias("_med"))
    b = b.join(med, "token_id", "inner")
    for c in ("_o", "_h", "_l"):
        b = b.withColumn(
            c,
            F.when(
                (F.col(c) <= F.col("_med") * OUTLIER_BAND) & (F.col(c) >= F.col("_med") / OUTLIER_BAND),
                F.col(c),
            ),
        )
    # If a zero-priced dust trade knocked out open/high/low but close is
    # valid, fall back to close-based values rather than drop the bar.
    o = F.coalesce("_o", "_c")
    h = F.coalesce("_h", F.greatest(o, F.col("_c")))
    lo = F.coalesce("_l", F.least(o, F.col("_c")))

    return (
        b.select(
            "token_id",
            "bar_start",
            "price_unit",
            F.col("primary_quote").alias("quote_token_id"),
            o.alias("open"),
            h.alias("high"),
            lo.alias("low"),
            F.col("_c").alias("close"),
            "volume_usd",
            F.when(same_quote, F.col("volume_quote")).alias("volume_quote"),
            "n_trades",
            "n_buys",
            "n_traders",
            F.when(same_quote, F.col("volume_buy_q")).alias("volume_buy_q"),
            F.when(same_quote, F.col("volume_sell_q")).alias("volume_sell_q"),
            F.when(same_quote, F.col("max_trade_q")).alias("max_trade_q"),
            "top_trader_share",
            F.lit(phase).alias("phase"),
            "venue",
            "pool_id",
        )
        .filter(F.col("close").isNotNull())
    )


def build_token_price_1m(
    trades_df: DataFrame,
    bars_df: DataFrame,
    pool_metadata_df: DataFrame,
    curve_bars_df: DataFrame | None = None,
) -> DataFrame:
    """All inputs -> lawrence.silver.token_price_1m (PRICE_COLUMNS).

    `curve_bars_df` (optional, bronze.curve_bars_1m): curve bars built on
    Dune for dates after the trade-level backfill. If a token has two bars
    in the same minute (graduation minute, or a window boundary), the one
    with more trades wins."""
    canon = canonical_bars(dedupe_bars(bars_df))
    curve = dedupe_bars(curve_bars_df) if curve_bars_df is not None else None
    cov_bars = canon
    if curve is not None:
        _cols = ["base_mint", "n_trades", "n_unpriced", "quote_mint"]
        cov_bars = canon.select(*_cols).unionByName(curve.select(*_cols))
    units = token_units(trades_df, cov_bars, pool_metadata_df)
    both = trade_bars(trades_df, units).unionByName(postgrad_bars(canon, units))
    if curve is not None:
        both = both.unionByName(postgrad_bars(curve, units, phase="curve"))
    w = Window.partitionBy("token_id", "bar_start").orderBy(F.desc("n_trades"), "phase")
    return (
        both.withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .withColumn("trade_date", F.to_date("bar_start"))
        .select(*PRICE_COLUMNS)
    )
