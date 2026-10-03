"""Unit tests for src/ingestion/curve_bars.py (pure Python, runs in CI)."""
from datetime import datetime, timezone

from ingestion.curve_bars import (
    MIN_TRADE_USD,
    WSOL,
    launchlab_curve_bars_sql,
    pumpfun_curve_bars_sql,
    summarize_sql,
    to_curve_bar_row,
)

T0 = datetime(2026, 9, 20, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 20, 6, tzinfo=timezone.utc)
INGESTED = datetime(2026, 10, 3, tzinfo=timezone.utc)


def test_dust_threshold_matches_silver():
    from features.silver_price_bars import MIN_TRADE_USD as SILVER_MIN

    assert MIN_TRADE_USD == SILVER_MIN


def test_pumpfun_sql_bounds_filters_and_sides():
    sql = pumpfun_curve_bars_sql(T0, T1)
    assert "t.block_time >= TIMESTAMP '2026-09-20 00:00:00'" in sql
    assert "t.block_time <  TIMESTAMP '2026-09-20 06:00:00'" in sql
    assert "project = 'pumpdotfun' AND t.version = 1" in sql
    assert f"t.amount_usd >= {MIN_TRADE_USD}" in sql
    # buy = SOL in, token out
    assert f"t.token_sold_mint_address = '{WSOL}' AS is_buy" in sql
    assert "'pumpdotfun' AS venue" in sql


def test_launchlab_sql_uses_pool_mints_both_directions():
    sql = launchlab_curve_bars_sql(T0, T1)
    assert "raydium_launchlab_v1.stg_swaps" in sql
    assert "t.token_bought_mint_address = p.base_mint AND t.token_sold_mint_address = p.quote_mint" in sql
    assert "t.token_sold_mint_address = p.base_mint AND t.token_bought_mint_address = p.quote_mint" in sql
    assert "project = 'raydium_launchlab' AND t.version = 1" in sql


def test_trader_stats_columns():
    for sql in (pumpfun_curve_bars_sql(T0, T1), launchlab_curve_bars_sql(T0, T1)):
        for col in ("volume_buy_q", "volume_sell_q", "max_trade_q", "top_trader_share", "n_traders", "close_q"):
            assert f"AS {col}" in sql
        # wallet share is per (pool, minute, wallet) -- chunk-safe
        assert "PARTITION BY pool_id, date_trunc('minute', block_time), trader_id" in sql


def test_summarize_wraps_one_row():
    s = summarize_sql(pumpfun_curve_bars_sql(T0, T1))
    assert s.strip().startswith("WITH bars AS (")
    assert "count(*) AS n_bars" in s


def test_to_curve_bar_row_types():
    row = to_curve_bar_row(
        {"venue": "pumpdotfun", "pool_id": "M", "base_mint": "M", "quote_mint": WSOL,
         "bar_start": "2026-09-20 00:01:00.000 UTC", "close_usd": 1, "close_q": 2,
         "top_trader_share": 1, "n_trades": 3.0, "volume_buy_q": None},
        INGESTED,
    )
    assert row["bar_start"] == datetime(2026, 9, 20, 0, 1, tzinfo=timezone.utc)
    assert isinstance(row["close_usd"], float) and isinstance(row["top_trader_share"], float)
    assert row["n_trades"] == 3 and isinstance(row["n_trades"], int)
    assert row["volume_buy_q"] is None and row["open_usd"] is None
