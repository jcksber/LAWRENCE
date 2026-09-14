"""
Unit tests for src/features/gold_features.py -- the pure-Python trailing-
feature and label computation shared by the Gold entry/exit training-data
builds. No PySpark/Databricks dependency, so these run in plain CI.
"""
from datetime import datetime, timedelta, timezone

from features.gold_features import (
    build_entry_row,
    build_exit_rows,
    price_at,
    sample_assessment_times,
    trailing_activity_features,
)

T0 = datetime(2026, 9, 14, 0, 0, 0, tzinfo=timezone.utc)


def _trade(minutes_offset, side, price, amount_usd):
    return {
        "trade_time": (T0 + timedelta(minutes=minutes_offset)).isoformat(),
        "side": side,
        "price_usd": price,
        "amount_usd": amount_usd,
        "chain_family": "solana",
        "network": None,
        "dex_venue": "raydium_v4",
        "dex_protocol": "raydium_amm",
    }


def test_price_at_returns_last_trade_at_or_before_t():
    trades = [_trade(0, "buy", 1.0, 10), _trade(10, "sell", 1.5, 10), _trade(20, "buy", 2.0, 10)]
    assert price_at(trades, T0 + timedelta(minutes=10)) == 1.5
    assert price_at(trades, T0 + timedelta(minutes=15)) == 1.5  # no trade at 15, use last <= t
    assert price_at(trades, T0 + timedelta(minutes=20)) == 2.0


def test_price_at_never_looks_forward():
    trades = [_trade(0, "buy", 1.0, 10), _trade(100, "buy", 999.0, 10)]
    # at t=5, the only real candidate is the t=0 trade; the t=100 trade must
    # never be used even though it's "closest" in value terms
    assert price_at(trades, T0 + timedelta(minutes=5)) == 1.0


def test_price_at_returns_none_when_too_stale():
    trades = [_trade(0, "buy", 1.0, 10)]
    assert price_at(trades, T0 + timedelta(minutes=60), max_staleness=timedelta(minutes=30)) is None


def test_price_at_returns_none_with_no_trades_before_t():
    trades = [_trade(50, "buy", 1.0, 10)]
    assert price_at(trades, T0) is None


def test_trailing_activity_features_basic_math():
    trades = [
        _trade(0, "buy", 1.0, 100),
        _trade(5, "buy", 1.2, 50),
        _trade(10, "sell", 1.1, 30),
    ]
    feats = trailing_activity_features(trades, T0 + timedelta(minutes=10), window=timedelta(minutes=30))
    assert feats["trailing_volume_usd"] == 180
    assert feats["trailing_trade_count"] == 3
    assert feats["trailing_buy_count"] == 2
    assert feats["trailing_sell_count"] == 1
    assert feats["trailing_buy_sell_ratio"] == 3 / 2  # (2+1)/(1+1)
    assert feats["trailing_price_change_pct"] == (1.1 - 1.0) / 1.0


def test_trailing_activity_features_excludes_window_start_boundary_and_future():
    trades = [_trade(0, "buy", 1.0, 10), _trade(10, "buy", 1.0, 10), _trade(20, "buy", 1.0, 10)]
    # window=(t-10, t] at t=10min -> excludes the t=0 trade (not > window_start)
    feats = trailing_activity_features(trades, T0 + timedelta(minutes=10), window=timedelta(minutes=10))
    assert feats["trailing_trade_count"] == 1  # only the t=10 trade


def test_trailing_activity_features_returns_none_with_nothing_in_window():
    trades = [_trade(0, "buy", 1.0, 10)]
    assert trailing_activity_features(trades, T0 + timedelta(hours=100), window=timedelta(hours=1)) is None


def test_build_entry_row_computes_y_entry_and_carries_metadata():
    trades = [
        _trade(0, "buy", 1.0, 100),
        _trade(30, "buy", 1.0, 50),
        _trade(60, "sell", 1.5, 80),  # this is the t + HOLD_MINUTES(60) exit price
    ]
    row = build_entry_row("TOKEN123", T0, trades, hold_minutes=60)
    assert row is not None
    assert row["token_id"] == "TOKEN123"
    assert row["dex_venue"] == "raydium_v4"
    assert row["price_at_t"] == 1.0
    assert row["y_entry"] == (1.5 - 1.0) / 1.0


def test_build_entry_row_drops_when_no_exit_price_available():
    # only one trade, nothing near t + HOLD_MINUTES -- must drop, not guess
    trades = [_trade(0, "buy", 1.0, 100)]
    assert build_entry_row("TOKEN123", T0, trades, hold_minutes=60) is None


def test_build_exit_rows_only_for_holds_with_a_synthetic_entry_price():
    trades = [
        _trade(-15, "buy", 0.8, 10),  # only 15 min of history before T0
        _trade(0, "buy", 1.0, 100),
        _trade(60, "sell", 1.2, 80),
    ]
    rows = build_exit_rows(
        "TOKEN123", T0, trades, hold_minutes=60,
        simulated_holds=[timedelta(minutes=15), timedelta(minutes=120)],
    )
    # h=120 needs a trade at T0-120min, which doesn't exist -> dropped
    assert len(rows) == 1
    assert rows[0]["simulated_holding_minutes"] == 15
    assert rows[0]["entry_price"] == 0.8
    assert rows[0]["unrealized_return"] == (1.0 - 0.8) / 0.8
    assert rows[0]["y_exit"] == (1.2 - 1.0) / 1.0


def test_sample_assessment_times_grid_spans_first_to_last_trade():
    trades = [_trade(0, "buy", 1.0, 10), _trade(37, "buy", 1.0, 10)]
    times = sample_assessment_times(trades, interval=timedelta(minutes=15))
    assert times == [
        T0,
        T0 + timedelta(minutes=15),
        T0 + timedelta(minutes=30),
    ]


def test_sample_assessment_times_empty_for_no_trades():
    assert sample_assessment_times([]) == []
