"""Contract v1 must reproduce Gold exactly on the same bars (no
train/serve skew from reimplementation). Pure pandas; runs in CI."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from features.gold_features import _barrier_labels, build_entry_row, sample_assessment_times
from serving.contract_v1 import (
    FEATURE_COLUMNS,
    assessment_grid,
    bars_to_records,
    compute_features,
    is_assessment_time,
    resolve_outcome,
    round_trip_cost_v1,
    select_row,
    to_model_frame,
)

L = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)  # launch = first curve bar_start


def _bars(n_curve=10, n_post=500, step_price=0.003):
    rows, p = [], 1.0
    for i in range(n_curve + n_post):
        p *= 1 + (step_price if i % 7 else -2 * step_price)
        rows.append({"bar_start": L + timedelta(minutes=i), "close": p, "high": p * 1.01, "low": p * 0.99,
                     "volume_usd": 100.0 + i, "n_trades": 3, "n_buys": 2 if i % 2 else 1,
                     "phase": "curve" if i < n_curve else "postgrad"})
    return pd.DataFrame(rows)


def test_features_match_gold_build_entry_row():
    bars = _bars()
    recs = bars_to_records(bars, "pumpdotfun")
    for t in (L + timedelta(minutes=5), L + timedelta(minutes=45), L + timedelta(minutes=180)):
        gold = build_entry_row("tok", t, recs, launch_time=L)
        live = compute_features(bars, t, launch_time=L, launch_venue="pumpdotfun")
        assert gold is not None and live is not None
        for k in FEATURE_COLUMNS + ["price_at_t"]:
            assert gold[k] == live[k], k


def test_future_bars_cannot_leak():
    bars = _bars()
    t = L + timedelta(minutes=30)
    a = compute_features(bars, t, launch_time=L, launch_venue="pumpdotfun")
    b = compute_features(bars[bars["bar_start"] + timedelta(minutes=1) <= t], t, launch_time=L, launch_venue="pumpdotfun")
    assert a == b


def test_grid_matches_gold_sampling():
    bars = _bars()
    recs = bars_to_records(bars, "pumpdotfun")
    gold = sample_assessment_times(recs, launch_time=L, active_window=None)
    live = assessment_grid(bars["bar_start"].min(), L, recs[-1]["trade_time"])
    assert gold == live
    assert is_assessment_time(L + timedelta(minutes=30), first_bar_start=L, launch_time=L)
    assert is_assessment_time(L + timedelta(minutes=75), first_bar_start=L, launch_time=L)
    assert not is_assessment_time(L + timedelta(minutes=76), first_bar_start=L, launch_time=L)


def test_select_row_rules():
    bars = _bars()
    ok, why = select_row(bars, L + timedelta(minutes=75), launch_time=L, launch_venue="pumpdotfun",
                         observed_from_launch=True)
    assert why == "candidate" and ok["is_postgrad_at_t"] is True
    assert select_row(bars, L + timedelta(minutes=5), launch_time=L, launch_venue="pumpdotfun",
                      observed_from_launch=True)[1] == "not_postgrad"
    assert select_row(bars, L + timedelta(minutes=76), launch_time=L, launch_venue="pumpdotfun",
                      observed_from_launch=True)[1] == "off_grid"
    assert select_row(bars, L + timedelta(minutes=75), launch_time=L, launch_venue="pumpdotfun",
                      observed_from_launch=False)[1] == "not_observed_from_launch"
    quiet = bars[(bars["bar_start"] < L + timedelta(minutes=60)) | (bars["bar_start"] > L + timedelta(minutes=80))]
    assert select_row(quiet, L + timedelta(minutes=75), launch_time=L, launch_venue="pumpdotfun",
                      observed_from_launch=True)[1] == "inactive"


def test_outcome_matches_training_label():
    bars = _bars(step_price=0.01)
    recs = bars_to_records(bars, "pumpdotfun")
    t = L + timedelta(minutes=30)
    row = compute_features(bars, t, launch_time=L, launch_venue="pumpdotfun")
    gold = _barrier_labels(recs, t, row["price_at_t"], None)
    live = resolve_outcome(bars, {"assessment_time": t, "price_at_t": row["price_at_t"]},
                           launch_venue="pumpdotfun", observed_until=recs[-1]["trade_time"])
    assert live is not None and abs(live["return"] - gold["r_up_360m"]) < 1e-12
    assert live["minutes"] == gold["t_up_360m"]
    assert (live["barrier_hit"] == "up") == (gold["y_up_360m"] == 1.0)


def test_unresolved_outcome_is_none():
    bars = _bars(n_post=40, step_price=0.0001)
    t = L + timedelta(minutes=20)
    row = compute_features(bars, t, launch_time=L, launch_venue="pumpdotfun")
    assert resolve_outcome(bars, {"assessment_time": t, "price_at_t": row["price_at_t"]},
                           launch_venue="pumpdotfun", observed_until=L + timedelta(minutes=50)) is None


def test_costs_and_frame():
    assert abs(round_trip_cost_v1("pumpdotfun", True) - 2 * (120 + 50) / 1e4) < 1e-12
    assert abs(round_trip_cost_v1("raydium_launchlab", True, 100) - 2 * (25 + 100) / 1e4) < 1e-12
    bars = _bars()
    row = compute_features(bars, L + timedelta(minutes=75), launch_time=L, launch_venue="pumpdotfun")
    X = to_model_frame([row], {"chain_family": ["solana"], "network": [], "dex_venue": ["pumpdotfun", "raydium_launchlab"],
                               "dex_protocol": []})
    assert list(X.columns) == FEATURE_COLUMNS and X["is_postgrad_at_t"].iloc[0] == 1.0
    assert str(X["dex_venue"].dtype) == "category"


def test_select_row_stops_past_training_horizon():
    bars = _bars(n_curve=10, n_post=1500)  # postgrad from minute 10 for 25h
    first_pg = L + timedelta(minutes=10)
    edge = first_pg + timedelta(hours=18)
    grid = assessment_grid(L, L, edge + timedelta(hours=1))
    inside = max(t for t in grid if t <= edge)
    outside = min(t for t in grid if t > edge)
    assert select_row(bars, inside, launch_time=L, launch_venue="pumpdotfun", observed_from_launch=True)[1] == "candidate"
    assert select_row(bars, outside, launch_time=L, launch_venue="pumpdotfun",
                      observed_from_launch=True)[1] == "past_training_horizon"
