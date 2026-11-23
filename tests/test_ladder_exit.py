import numpy as np
import pytest

from backtest.ladder_exit import simulate

T0 = 0.0


def bars(rows):
    """rows of (minutes_after_T, high, low, close) -> arrays; ts = bar close time."""
    a = np.array(rows, dtype=float)
    return a[:, 0] * 60, a[:, 1], a[:, 2], a[:, 3]


def test_e0_full_exit_at_2x():
    r = simulate(*bars([(1, 1.1, 0.9, 1.0), (2, 2.1, 1.5, 2.0)]), T0, 1.0, 0.0, ladder=False)
    assert r["gross"] == pytest.approx(1.0) and r["exit"] == "target"


def test_ladder_recovers_stake_then_worst_case_plus_50():
    # 2x at bar 2, then falls back to the entry price (= new stop at 1.0)
    r = simulate(*bars([(1, 1.2, 0.9, 1.1), (2, 2.05, 1.6, 2.0), (3, 2.0, 0.95, 1.0)]), T0, 1.0, 0.0, ladder=True)
    assert r["rungs"] == 1 and r["exit"] == "stop"
    assert r["gross"] == pytest.approx(0.5 * 2.0 + 0.5 * 1.0 - 1)  # +50%


def test_ladder_two_rungs_then_timeout():
    r = simulate(*bars([(1, 2.0, 1.5, 2.0), (2, 4.0, 3.0, 3.5), (3, 3.6, 3.0, 3.0)]), T0, 1.0, 0.0, ladder=True)
    # sell 0.5 @2, 0.25 @4, 0.25 @ last close 3.0
    assert r["rungs"] == 2 and r["gross"] == pytest.approx(0.5 * 2 + 0.25 * 4 + 0.25 * 3 - 1)


def test_both_barriers_same_bar_counts_as_stop():
    r = simulate(*bars([(1, 2.5, 0.4, 1.0)]), T0, 1.0, 0.0, ladder=True)
    assert r["exit"] == "stop" and r["gross"] == pytest.approx(-0.5)


def test_gap_through_stop_fills_at_close():
    r = simulate(*bars([(1, 0.3, 0.2, 0.25)]), T0, 1.0, 0.0, ladder=False)
    assert r["gross"] == pytest.approx(-0.75)


def test_costs_on_each_sale_notional():
    r = simulate(*bars([(1, 2.0, 1.5, 2.0)]), T0, 1.0, 0.01, ladder=False)
    assert r["net"] == pytest.approx(2.0 * 0.99 - 1.01)


def test_clock_resets_after_rung():
    # rung at minute 300; without a reset the 6h clock would end at 360 and the stop bar at 600 would be ignored
    rows = [(300, 2.0, 1.5, 2.0), (400, 2.1, 1.2, 1.2), (600, 2.1, 0.9, 1.0)]
    r = simulate(*bars(rows), T0, 1.0, 0.0, ladder=True)
    assert r["exit"] == "stop" and r["gross"] == pytest.approx(0.5 * 2 + 0.5 * 1.0 - 1)


def test_bars_before_entry_ignored():
    r = simulate(*bars([(-1, 5.0, 0.1, 1.0), (1, 1.1, 0.9, 1.05)]), T0, 1.0, 0.0, ladder=True)
    assert r["exit"] == "timeout" and r["gross"] == pytest.approx(0.05)


def test_e2_half_at_2x_rest_at_10x():
    rows = [(1, 2.0, 1.5, 2.0), (2, 4.0, 3.0, 4.0), (3, 10.5, 6.0, 9.0)]
    r = simulate(*bars(rows), T0, 1.0, 0.0, ladder=True, final_mult=10.0)
    assert r["exit"] == "target" and r["gross"] == pytest.approx(0.5 * 2 + 0.5 * 10 - 1)  # +500%


def test_e2_rest_breaks_even_at_entry_price():
    rows = [(1, 2.0, 1.5, 2.0), (2, 3.0, 2.0, 2.5), (3, 2.5, 0.9, 1.0)]
    r = simulate(*bars(rows), T0, 1.0, 0.0, ladder=True, final_mult=10.0)
    assert r["exit"] == "stop" and r["gross"] == pytest.approx(0.5 * 2 + 0.5 * 1 - 1)  # +50%


def test_e2_no_intermediate_rung_at_4x():
    rows = [(1, 2.0, 1.5, 2.0), (2, 4.2, 3.0, 4.0)]
    r = simulate(*bars(rows), T0, 1.0, 0.0, ladder=True, final_mult=10.0)
    assert r["rungs"] == 1 and r["exit"] == "timeout" and r["gross"] == pytest.approx(0.5 * 2 + 0.5 * 4 - 1)


def test_e2_stop_at_1_1x_covers_fees():
    rows = [(1, 2.0, 1.5, 2.0), (2, 2.5, 1.05, 1.2)]
    r = simulate(*bars(rows), T0, 1.0, 0.0, ladder=True, final_mult=10.0, final_stop_mult=1.1)
    assert r["exit"] == "stop" and r["gross"] == pytest.approx(0.5 * 2 + 0.5 * 1.1 - 1)  # +55%


from backtest.ladder_exit import simulate_schedule  # noqa: E402


def test_e3_full_ladder_to_10x():
    r = simulate_schedule(*bars([(1, 2.0, 1.5, 2.0), (2, 4.0, 3.0, 4.0), (3, 6.0, 5.0, 6.0), (4, 8.0, 7.0, 8.0),
                                 (5, 10.0, 9.0, 10.0)]), T0, 1.0, 0.0)
    assert r["exit"] == "target" and r["rungs"] == 5
    assert r["gross"] == pytest.approx(0.5 * 2 + 0.25 * 4 + 0.125 * 6 + 0.0625 * 8 + 0.0625 * 10 - 1)  # +287.5%


def test_e3_stop_from_entry_after_2x_keeps_40pct():
    r = simulate_schedule(*bars([(1, 2.0, 1.5, 2.0), (2, 1.9, 0.7, 0.8)]), T0, 1.0, 0.0)
    assert r["exit"] == "stop" and r["gross"] == pytest.approx(0.5 * 2 + 0.5 * 0.8 - 1)  # +40%


def test_e3_stop_wins_tie_and_gap_fills_at_close():
    assert simulate_schedule(*bars([(1, 2.5, 0.7, 1.0)]), T0, 1.0, 0.0)["gross"] == pytest.approx(-0.2)
    assert simulate_schedule(*bars([(1, 0.5, 0.3, 0.4)]), T0, 1.0, 0.0)["gross"] == pytest.approx(-0.6)


def test_e3_several_targets_in_one_bar_and_no_time_limit():
    r = simulate_schedule(*bars([(1, 4.5, 1.5, 4.0), (600, 4.2, 3.9, 4.1)]), T0, 1.0, 0.0)
    assert r["rungs"] == 2 and r["truncated"] and r["gross"] == pytest.approx(0.5 * 2 + 0.25 * 4 + 0.25 * 4.1 - 1)


def test_e3_exit_ts():
    r = simulate_schedule(*bars([(1, 1.1, 0.9, 1.0), (5, 1.0, 0.7, 0.75)]), T0, 1.0, 0.0)
    assert r["exit"] == "stop" and r["exit_ts"] == 5 * 60
    r2 = simulate_schedule(*bars([(1, 1.1, 0.9, 1.0)]), T0, 1.0, 0.0, data_end=3600.0)
    assert r2["exit"] == "end_of_data" and r2["exit_ts"] == 3600.0
