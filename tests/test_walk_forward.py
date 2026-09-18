import pandas as pd
import pytest

from backtest.walk_forward import (
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    round_trip_cost,
    simulate_trades,
    walk_forward_folds,
)


def _times(hours: list[int]) -> pd.Series:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.Series([base + pd.Timedelta(hours=h) for h in hours])


def test_walk_forward_folds_are_expanding_and_never_test_on_the_past():
    # 100 hours of evenly-spaced assessment points.
    t = _times(list(range(100)))
    folds = walk_forward_folds(t, n_folds=4)
    assert len(folds) == 4
    for train_mask, test_mask in folds:
        train_times = t[train_mask]
        test_times = t[test_mask]
        assert not train_times.empty
        assert not test_times.empty
        # Every training timestamp must be strictly before every test
        # timestamp in the same fold -- the whole point of walk-forward.
        assert train_times.max() < test_times.min()


def test_walk_forward_folds_train_window_expands_each_fold():
    t = _times(list(range(100)))
    folds = walk_forward_folds(t, n_folds=4)
    train_sizes = [train_mask.sum() for train_mask, _ in folds]
    assert train_sizes == sorted(train_sizes)  # non-decreasing


def test_walk_forward_folds_rejects_a_degenerate_single_instant():
    t = pd.Series([pd.Timestamp("2026-01-01T00:00:00Z")] * 5)
    with pytest.raises(ValueError):
        walk_forward_folds(t, n_folds=3)


def test_round_trip_cost_uses_default_fee_and_slippage_when_unspecified():
    assert round_trip_cost() == pytest.approx(
        2 * (DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) / 10_000
    )


def test_round_trip_cost_accepts_overrides():
    assert round_trip_cost(fee_bps=100.0, slippage_bps=0.0) == pytest.approx(0.02)


def test_simulate_trades_and_round_trip_cost_share_the_same_formula():
    # The whole reason round_trip_cost() is a shared function rather than
    # inlined separately in the notebook's decision rule and here: they
    # must never drift apart into two different cost numbers.
    y_true = pd.Series([0.10, -0.05, 0.20])
    enter_mask = pd.Series([True, True, False])
    result = simulate_trades(
        y_true, enter_mask, strategy="test", fee_bps=100.0, slippage_bps=0.0
    )
    cost = round_trip_cost(fee_bps=100.0, slippage_bps=0.0)
    assert result.ev_per_trade == pytest.approx(((0.10 - cost) + (-0.05 - cost)) / 2)


def test_simulate_trades_applies_round_trip_costs_to_entered_rows_only():
    y_true = pd.Series([0.10, -0.05, 0.20])
    enter_mask = pd.Series([True, True, False])
    result = simulate_trades(
        y_true, enter_mask, strategy="test", fee_bps=100.0, slippage_bps=0.0
    )
    # round_trip_cost = 2 * 100bps = 0.02
    assert result.n_candidates == 3
    assert result.n_trades == 2
    assert result.ev_per_trade == pytest.approx(((0.10 - 0.02) + (-0.05 - 0.02)) / 2)
    assert result.win_rate == pytest.approx(0.5)  # one of the two trades still wins


def test_simulate_trades_max_drawdown_is_the_worst_peak_to_trough_dip():
    # Cumulative curve (after a 0-cost strategy): 0.10, 0.15, 0.05, 0.25
    # Peaks at 0.15, troughs to 0.05 -> drawdown of -0.10 there.
    y_true = pd.Series([0.10, 0.05, -0.10, 0.20])
    enter_mask = pd.Series([True, True, True, True])
    result = simulate_trades(y_true, enter_mask, strategy="test", fee_bps=0.0, slippage_bps=0.0)
    assert result.max_drawdown == pytest.approx(-0.10)


def test_simulate_trades_handles_zero_trades_without_crashing():
    y_true = pd.Series([0.10, -0.05])
    enter_mask = pd.Series([False, False])
    result = simulate_trades(y_true, enter_mask, strategy="test")
    assert result.n_trades == 0
    assert result.n_candidates == 2
    assert result.total_return == 0.0
    import math

    assert math.isnan(result.ev_per_trade)
