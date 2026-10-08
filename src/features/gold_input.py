"""
Spark prep for Gold v2: `lawrence.silver.token_price_1m` -> the per-bar
records `gold_features.py` consumes, restricted to entry candidates.

Rules (2026-10-01, docs/decisions.md):
- Candidates = tokens with a `curve` phase (we saw them launch). Tokens with
  only `postgrad` and/or `amm` bars are excluded from Gold.
- `trade_time` = bar_start + 1 minute: a bar's close is only known once its
  minute is over, so features at T only see bars that had finished by T.
- `observed_until` per token (censoring bound, see gold_features):
    * token with post-graduation bars: first postgrad bar + POSTGRAD_HORIZON
      (00f pulls 24h of bars per graduation pool);
    * curve-only token: the end of the curve backfill (`curve_data_end`).
  Labels whose T + hold falls past it are skipped, not rug-clamped.
- `launch_time` = the token's first curve bar (a late proxy for tokens that
  launched just before the backfill window).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

POSTGRAD_HORIZON = timedelta(hours=24)  # must match 00f's HORIZON_HOURS

GOLD_INPUT_COLUMNS = [
    "token_id",
    "trade_time",
    "price",
    "high",  # 2026-10-08: barrier labels need each bar's range
    "low",
    "amount_usd",
    "n_trades",
    "n_buys",
    "phase",
    "chain_family",
    "network",
    "dex_venue",
    "dex_protocol",
    "price_unit",
    "launch_time",
    "observed_until",
]


def gold_input(price_df: DataFrame, curve_data_end: datetime) -> DataFrame:
    curve_stats = (
        price_df.filter(F.col("phase") == "curve")  # entry candidates only
        .groupBy("token_id")
        .agg(
            F.min("bar_start").alias("launch_time"),
            F.min_by("venue", "bar_start").alias("launch_venue"),
        )
    )
    postgrad_stats = (
        price_df.filter(F.col("phase") == "postgrad")
        .groupBy("token_id")
        .agg(F.min("bar_start").alias("first_postgrad"))
    )
    per_token = curve_stats.join(postgrad_stats, "token_id", "left").withColumn(
        "observed_until",
        F.when(
            F.col("first_postgrad").isNotNull(),
            F.col("first_postgrad") + F.expr(f"INTERVAL {int(POSTGRAD_HORIZON.total_seconds())} SECONDS"),
        ).otherwise(F.lit(curve_data_end)),
    )
    return (
        price_df.join(per_token, "token_id", "inner")
        .select(
            "token_id",
            (F.col("bar_start") + F.expr("INTERVAL 1 MINUTE")).alias("trade_time"),
            F.col("close").alias("price"),
            "high",
            "low",
            F.col("volume_usd").alias("amount_usd"),
            "n_trades",
            "n_buys",
            "phase",
            F.lit("solana").alias("chain_family"),  # every curve venue is Solana today
            F.lit(None).cast("string").alias("network"),
            F.col("launch_venue").alias("dex_venue"),
            F.lit(None).cast("string").alias("dex_protocol"),
            "price_unit",
            "launch_time",
            "observed_until",
        )
    )
