"""Unit tests for src/ingestion/postgrad_bars.py (pure Python, runs in CI)."""
from datetime import datetime, timezone

from ingestion.postgrad_bars import (
    cpmm_launchlab_bars_sql,
    launchlab_pools_sql,
    parse_dune_ts,
    pumpswap_bars_sql,
    pumpswap_pools_sql,
    to_bar_row,
    to_pool_row,
)

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 1, 6, tzinfo=timezone.utc)
INGESTED = datetime(2026, 9, 26, tzinfo=timezone.utc)


def test_parse_dune_ts_formats():
    want = datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc)
    assert parse_dune_ts("2026-09-01 00:01:00.000 UTC") == want
    assert parse_dune_ts("2026-09-01 00:01:00 UTC") == want
    assert parse_dune_ts("2026-09-01T00:01:00Z") == want
    assert parse_dune_ts(None) is None


def test_pumpswap_bars_sql_bounds_and_horizon():
    sql = pumpswap_bars_sql(T0, T1, horizon_hours=24)
    assert "created_at >= TIMESTAMP '2026-09-01 00:00:00'" in sql
    assert "created_at <  TIMESTAMP '2026-09-01 06:00:00'" in sql
    assert "p.created_at + INTERVAL '24' HOUR" in sql
    assert "t.project = 'pumpswap'" in sql
    assert "'pumpswap' AS venue" in sql
    # price is the BASE token's price, from whichever side the base is on
    assert "t.token_bought_mint_address = p.baseMint THEN t.amount_usd / NULLIF(t.token_bought_amount, 0)" in sql
    assert "t.token_sold_mint_address   = p.baseMint THEN t.amount_usd / NULLIF(t.token_sold_amount, 0)" in sql


def test_cpmm_sql_uses_version_5_and_launchlab_mints():
    sql = cpmm_launchlab_bars_sql(T0, T1, horizon_hours=24)
    assert "t.version = 5" in sql
    assert "raydium_launchlab_v1.stg_swaps" in sql
    assert "g.first_trade + INTERVAL '24' HOUR" in sql
    assert "'raydium_cpmm' AS venue" in sql


def test_unpriced_trades_counted_not_dropped():
    sql = pumpswap_bars_sql(T0, T1)
    assert "count(*) AS n_trades" in sql
    assert "count_if(price IS NULL) AS n_unpriced" in sql


def test_pool_sqls():
    assert "pumpswap_solana.pools" in pumpswap_pools_sql(T0, T1)
    ll = launchlab_pools_sql(T0, T1)
    assert "max(platform_name) AS platform_name" in ll
    assert "max(platform_config) AS platform_config" in ll
    assert "GROUP BY 1, 2, 3, 4" in ll
    # both pool queries return the same columns (one Bronze table)
    assert "AS platform_config" in pumpswap_pools_sql(T0, T1)


def test_to_bar_row_types():
    row = to_bar_row(
        {
            "venue": "pumpswap", "pool_id": "P", "base_mint": "B", "quote_mint": "Q",
            "bar_start": "2026-09-01 00:01:00.000 UTC",
            "open_usd": 1, "high_usd": 2.5, "low_usd": 1, "close_usd": 2,
            "volume_usd": 100, "n_trades": 3.0, "n_buys": 2, "n_traders": 2, "n_unpriced": 0,
        },
        INGESTED,
    )
    assert isinstance(row["open_usd"], float) and isinstance(row["volume_usd"], float)
    assert isinstance(row["n_trades"], int)
    assert row["bar_start"] == datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc)
    assert row["source"] == "dune"


def test_to_bar_row_all_unpriced_minute_keeps_nulls():
    row = to_bar_row(
        {
            "venue": "raydium_cpmm", "pool_id": "P", "base_mint": "B", "quote_mint": "Q",
            "bar_start": "2026-09-01 00:01:00.000 UTC",
            "open_usd": None, "high_usd": None, "low_usd": None, "close_usd": None,
            "volume_usd": None, "n_trades": 4, "n_buys": 1, "n_traders": 3, "n_unpriced": 4,
            "open_q": 0.002, "high_q": 0.003, "low_q": 0.002, "close_q": 0.003, "volume_quote": 12,
        },
        INGESTED,
    )
    assert row["close_usd"] is None and row["n_unpriced"] == 4
    # quote-unit prices survive when USD doesn't
    assert row["close_q"] == 0.003 and isinstance(row["volume_quote"], float)


def test_to_pool_row():
    row = to_pool_row(
        {"venue": "pumpswap", "pool_id": "P", "base_mint": "B", "quote_mint": "Q",
         "platform_name": None, "created_at": "2026-09-01 00:00:05.000 UTC"},
        INGESTED,
    )
    assert row["created_at"] == datetime(2026, 9, 1, 0, 0, 5, tzinfo=timezone.utc)
    assert row["platform_config"] is None
    ll = to_pool_row({"venue": "raydium_launchlab", "pool_id": "P", "base_mint": "B", "quote_mint": "Q",
                      "platform_name": "StonkFun", "platform_config": "CFG", "created_at": None}, INGESTED)
    assert ll["platform_config"] == "CFG"


def test_quote_unit_price_columns_in_both_sqls():
    for sql in (pumpswap_bars_sql(T0, T1), cpmm_launchlab_bars_sql(T0, T1)):
        assert "AS price_q" in sql
        assert "AS quote_amount" in sql
        assert "max_by(price_q, order_key) FILTER (WHERE price_q IS NOT NULL) AS close_q" in sql
        assert "sum(quote_amount) AS volume_quote" in sql
    # quote-per-base: the OTHER side's amount over the base side's amount
    ps = pumpswap_bars_sql(T0, T1)
    assert "WHEN t.token_bought_mint_address = p.baseMint THEN t.token_sold_amount / NULLIF(t.token_bought_amount, 0)" in ps
