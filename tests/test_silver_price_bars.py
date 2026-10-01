"""Spark tests for src/features/silver_price_bars.py. Skipped where PySpark
isn't installed (CI), same as tests/test_silver_spark.py."""
from datetime import datetime

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from features.silver_price_bars import build_token_price_1m  # noqa: E402

WSOL = "So11111111111111111111111111111111111111112"
SPYX = "XsoCS1TfEyfFhfvj8EtZ528L3CaKBDBRqRapnBbDF2W"
ZEC = "A7bdiYdS5GjqGFtxf17ppRHtDKPkkRqbKtR27dxvQXaS"


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()


def _trade(token, minute, sec, price_usd, price_quote, quote, venue="pumpdotfun", side="buy"):
    return dict(
        token_id=token, dex_venue=venue, side=side, quote_token_id=quote,
        trade_time=datetime(2026, 9, 1, 0, minute, sec),
        price_usd=price_usd, price_quote=price_quote,
        amount_usd=None if price_usd is None else 10.0, quote_amount=1.0,
    )


def _bar(base, pool, minute, close_usd, close_q, quote, n_trades=5, n_unpriced=0, venue="pumpswap"):
    return dict(
        venue=venue, pool_id=pool, base_mint=base, quote_mint=quote,
        bar_start=datetime(2026, 9, 1, 0, minute), ingested_at=datetime(2026, 9, 30),
        open_usd=close_usd, high_usd=close_usd, low_usd=close_usd, close_usd=close_usd,
        open_q=close_q, high_q=close_q, low_q=close_q, close_q=close_q,
        volume_usd=1.0, volume_quote=1.0, n_trades=n_trades, n_buys=1, n_traders=1,
        n_unpriced=n_unpriced,
    )


_TRADE_COLS = "token_id string, dex_venue string, side string, quote_token_id string, trade_time timestamp, price_usd double, price_quote double, amount_usd double, quote_amount double"
_BAR_COLS = ("venue string, pool_id string, base_mint string, quote_mint string, bar_start timestamp, "
             "ingested_at timestamp, open_usd double, high_usd double, low_usd double, close_usd double, "
             "open_q double, high_q double, low_q double, close_q double, volume_usd double, "
             "volume_quote double, n_trades long, n_buys long, n_traders long, n_unpriced long")
_POOL_COLS = "venue string, pool_id string, base_mint string, quote_mint string"


def _run(spark, trades, bars, pools):
    return build_token_price_1m(
        spark.createDataFrame(trades, _TRADE_COLS),
        spark.createDataFrame(bars, _BAR_COLS),
        spark.createDataFrame(pools, _POOL_COLS),
    ).collect()


def test_usd_token_curve_then_postgrad_from_busiest_pool(spark):
    trades = [_trade("PUMP1", 0, 5, 1.0, 0.01, WSOL), _trade("PUMP1", 0, 50, 2.0, 0.02, WSOL)]
    bars = [
        _bar("PUMP1", "canon", 5, 3.0, None, WSOL, n_trades=100),
        _bar("PUMP1", "copycat", 6, 999.0, None, WSOL, n_trades=1),
        _bar("PUMP1", "canon", 5, 3.0, None, WSOL, n_trades=100),  # duplicate (partial re-fetch)
    ]
    rows = sorted(_run(spark, trades, bars, []), key=lambda r: r.bar_start)
    assert [r.phase for r in rows] == ["curve", "postgrad"]
    assert rows[0].open == 1.0 and rows[0].close == 2.0 and rows[0].n_trades == 2
    assert rows[1].pool_id == "canon" and rows[1].close == 3.0
    assert all(r.price_unit == "usd" for r in rows)


def test_unpriced_stonkfun_token_uses_launch_quote_units_only(spark):
    trades = [
        _trade("STONK1", 0, 1, None, 0.5, SPYX, venue="raydium_launchlab"),
        _trade("STONK1", 0, 2, None, 0.6, SPYX, venue="raydium_launchlab"),
        _trade("STONK1", 1, 0, None, 77.0, ZEC, venue="raydium_launchlab"),  # other quote -> excluded
    ]
    bars = [_bar("STONK1", "cpmm1", 10, None, 0.9, SPYX, n_unpriced=5, venue="raydium_cpmm")]
    pools = [dict(venue="raydium_launchlab", pool_id="ll1", base_mint="STONK1", quote_mint=SPYX)]
    rows = sorted(_run(spark, trades, bars, pools), key=lambda r: r.bar_start)
    assert all(r.price_unit == "quote" and r.quote_token_id == SPYX for r in rows)
    assert [r.close for r in rows] == [0.6, 0.9]  # the ZEC-quoted minute is dropped, not mixed in
