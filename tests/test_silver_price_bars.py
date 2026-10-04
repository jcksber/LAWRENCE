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


def test_zero_prices_are_missing_not_zero(spark):
    trades = [_trade("PUMP2", 0, 1, 0.0, None, WSOL), _trade("PUMP2", 0, 2, 2.0, None, WSOL)]
    bars = [_bar("PUMP2", "p", 5, 0.0, None, WSOL), _bar("PUMP2", "p", 6, 3.0, None, WSOL)]
    rows = sorted(_run(spark, trades, bars, []), key=lambda r: r.bar_start)
    assert all(r.close > 0 and r.low > 0 for r in rows)
    assert [r.close for r in rows] == [2.0, 3.0]  # the all-zero bar at minute 5 is dropped


# ---- 2026-10-03: dust filters + unit-by-quote ----

def test_dust_and_impossible_curve_prices_are_dropped(spark):
    # price_quote here is SOL per token; pump.fun curve range ~2.8e-8 .. 4e-7
    trades = [
        _trade("PUMP3", 0, 1, 1.0, 3e-8, WSOL),
        _trade("PUMP3", 0, 2, 1.1, 3.3e-8, WSOL),
        dict(_trade("PUMP3", 1, 0, 9999.0, 1e-3, WSOL), amount_usd=0.000001),  # dust: < $0.01
        _trade("PUMP3", 2, 0, 9999.0, 1e-3, WSOL),  # not dust by size, but an impossible curve price
    ]
    rows = sorted(_run(spark, trades, [], []), key=lambda r: r.bar_start)
    # dust trade (minute 1) is dropped. The curve-range rule is disabled
    # (2026-10-03), so minute 2's price survives the trade filters; it is
    # ~9000x the token's median, inside the 1000x band only if the median
    # is high enough -- here the median is ~1.1, so 9999 is dropped too.
    assert [r.bar_start.minute for r in rows] == [0]
    assert rows[0].high == 1.1


def test_sol_quoted_graduate_stays_usd_and_keeps_postgrad_bars(spark):
    # curve trades fully USD-priced, but the PumpSwap pool is mostly USD-unpriced
    # -> coverage < 95%. Before 2026-10-03 this flipped the token to quote units
    # and dropped its PumpSwap bars (no quote-unit prices in that pull).
    trades = [_trade("GRAD1", 0, 1, 1.0, 3e-8, WSOL)]
    bars = [_bar("GRAD1", "pool", 5, 2.0, None, WSOL, n_trades=1000, n_unpriced=900)]
    rows = sorted(_run(spark, trades, bars, []), key=lambda r: r.bar_start)
    assert all(r.price_unit == "usd" for r in rows)
    assert [r.phase for r in rows] == ["curve", "postgrad"]


def test_postgrad_outlier_bar_dropped(spark):
    trades = [_trade("GRAD2", 0, 1, 1.0, 3e-8, WSOL)]
    bars = [_bar("GRAD2", "pool", m, 2.0, None, WSOL) for m in range(5, 10)] + [
        _bar("GRAD2", "pool", 10, 2.0 * 30000, None, WSOL)  # dust-driven spike
    ]
    rows = _run(spark, trades, bars, [])
    assert max(r.close for r in rows) == 2.0


_CURVE_COLS = _BAR_COLS + ", volume_buy_q double, volume_sell_q double, max_trade_q double, top_trader_share double"


def test_dune_curve_bars_continue_the_curve_with_flow_stats(spark):
    # 2026-10-04: a token with Silver curve trades in the old window and 00h
    # Dune curve bars after it -> one continuous 'curve' series, flow stats
    # from both sources (top_trader_share only from Dune).
    trades = [_trade("PUMP3", 0, 5, 1.0, 0.01, WSOL)]
    curve = [dict(_bar("PUMP3", "PUMP3", 3, 1.5, 0.015, WSOL, n_trades=4, venue="pumpdotfun"),
                  volume_buy_q=0.7, volume_sell_q=0.3, max_trade_q=0.5, top_trader_share=0.6)]
    rows = sorted(
        build_token_price_1m(
            spark.createDataFrame(trades, _TRADE_COLS),
            spark.createDataFrame([], _BAR_COLS),
            spark.createDataFrame([], _POOL_COLS),
            spark.createDataFrame(curve, _CURVE_COLS),
        ).collect(),
        key=lambda r: r.bar_start,
    )
    assert [r.phase for r in rows] == ["curve", "curve"]
    assert rows[0].volume_buy_q == 1.0 and rows[0].top_trader_share is None  # from Silver trades
    assert rows[1].close == 1.5 and rows[1].top_trader_share == 0.6 and rows[1].volume_buy_q == 0.7
    assert all(r.price_unit == "usd" for r in rows)
