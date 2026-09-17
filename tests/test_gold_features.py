"""
Unit tests for src/features/gold_features.py -- the pure-Python trailing-
feature and label computation shared by the Gold entry/exit training-data
builds. No PySpark/Databricks dependency, so these run in plain CI.
"""
from datetime import datetime, timedelta, timezone

from features.gold_features import (
    RUG_CLAMP_RETURN,
    _parse,
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


def test_parse_handles_iso_z_and_offset_and_dune_raw_shapes_identically():
    # 2026-09-17: real Silver data has a mix of trade_time shapes -- see
    # docs/decisions.md. A resumable backfill wrote some chunks before
    # dune_client.py's block_time normalization existed, so Bronze/Silver
    # genuinely contain both the clean ISO shape and Dune's raw
    # "YYYY-MM-DD HH:MM:SS.mmm UTC" shape for real, not hypothetically.
    expected = datetime(2026, 8, 18, 10, 16, 1, tzinfo=timezone.utc)
    assert _parse("2026-08-18T10:16:01Z") == expected
    assert _parse("2026-08-18T10:16:01+00:00") == expected
    assert _parse("2026-08-18 10:16:01.000 UTC") == expected
    assert _parse("2026-08-18 10:16:01 UTC") == expected  # no fractional seconds


def test_parse_always_returns_tz_aware_datetime():
    assert _parse("2026-08-18T10:16:01Z").tzinfo is not None
    assert _parse("2026-08-18 10:16:01.000 UTC").tzinfo is not None


def test_build_entry_row_handles_a_token_with_mixed_trade_time_shapes():
    # A token whose trades mix both real-world shapes must not crash on
    # datetime comparisons (aware vs. naive) and must compute correctly.
    trades = [
        _trade(0, "buy", 1.0, 100),
        {**_trade(30, "buy", 1.0, 50), "trade_time": "2026-09-14 00:30:00.000 UTC"},
        _trade(60, "sell", 1.5, 80),
    ]
    row = build_entry_row("TOKEN123", T0, trades, hold_minutes=60)
    assert row is not None
    assert row["y_entry"] == (1.5 - 1.0) / 1.0


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


def test_price_at_returns_none_for_non_positive_price():
    # 2026-09-17: a real Silver row had price_usd == 0.0 (a dust/degenerate
    # swap artifact) and crashed build_entry_row with ZeroDivisionError --
    # see docs/decisions.md. Treated the same as "no usable price."
    assert price_at([_trade(0, "buy", 0.0, 100)], T0) is None
    assert price_at([_trade(0, "buy", -0.5, 100)], T0) is None


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


def test_build_entry_row_drops_when_price_at_t_is_zero():
    # price_t == 0.0 must be treated like "no price at T" (dropped), not
    # used as a division denominator.
    trades = [_trade(0, "buy", 0.0, 100), _trade(30, "buy", 1.0, 50), _trade(60, "sell", 1.5, 80)]
    assert build_entry_row("TOKEN123", T0, trades, hold_minutes=60) is None


def test_build_entry_row_clamps_when_price_exit_is_zero():
    # price_exit == 0.0 must go through the same rug-clamp path as "no
    # exit-price trade found" -- not a literal (0 - price_t) / price_t.
    trades = [_trade(0, "buy", 1.0, 100), _trade(60, "sell", 0.0, 80)]
    row = build_entry_row("TOKEN123", T0, trades, hold_minutes=60)
    assert row is not None
    assert row["y_entry"] == RUG_CLAMP_RETURN
    assert row["is_rug_clamped"] is True


def test_build_entry_row_drops_when_no_price_anchor_at_t():
    # no trade at or before T at all -- a genuine data gap, still dropped
    trades = [_trade(50, "buy", 1.0, 10)]
    assert build_entry_row("TOKEN123", T0, trades, hold_minutes=60) is None


def test_build_entry_row_clamps_instead_of_dropping_when_exit_price_missing():
    # real price + features at T, but nothing trades near t + HOLD_MINUTES --
    # interim rug heuristic: keep the row, clamp the label, flag it
    trades = [_trade(0, "buy", 1.0, 100), _trade(5, "buy", 1.0, 50)]
    row = build_entry_row("TOKEN123", T0, trades, hold_minutes=60)
    assert row is not None
    assert row["y_entry"] == RUG_CLAMP_RETURN
    assert row["is_rug_clamped"] is True


def test_build_entry_row_marks_normal_rows_as_not_rug_clamped():
    trades = [
        _trade(0, "buy", 1.0, 100),
        _trade(30, "buy", 1.0, 50),
        _trade(60, "sell", 1.5, 80),
    ]
    row = build_entry_row("TOKEN123", T0, trades, hold_minutes=60)
    assert row["is_rug_clamped"] is False


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
    assert rows[0]["is_rug_clamped"] is False


def test_build_exit_rows_clamps_instead_of_dropping_when_exit_price_missing():
    # real price + features + a synthetic entry at T, but nothing trades
    # near t + HOLD_MINUTES -- interim rug heuristic: keep the row(s),
    # clamp y_exit, flag it (unrealized_return is unaffected -- it's a
    # real, already-observed quantity, not the forward-looking label)
    trades = [_trade(-15, "buy", 0.8, 10), _trade(0, "buy", 1.0, 100)]
    rows = build_exit_rows(
        "TOKEN123", T0, trades, hold_minutes=60,
        simulated_holds=[timedelta(minutes=15)],
    )
    assert len(rows) == 1
    assert rows[0]["y_exit"] == RUG_CLAMP_RETURN
    assert rows[0]["is_rug_clamped"] is True


def test_build_exit_rows_drops_when_no_price_anchor_at_t():
    trades = [_trade(50, "buy", 1.0, 10)]
    assert build_exit_rows("TOKEN123", T0, trades, hold_minutes=60) == []


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
