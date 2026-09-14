"""
Pure-Python feature/label computation for the Gold training tables
(`lawrence.gold.entry_candidates`, `lawrence.gold.exit_candidates`).

Gold v1 is a TRAINING-DATA generation job: given a token's full trade
history from Silver, it samples assessment points T across that history
and computes trailing-activity features + the forward-return label at
each one. This is distinct from a future PRODUCTION-serving Gold path
(T = "now," incremental, per active token) -- that doesn't exist yet
because there's no trained model to serve. See docs/gold_spec.md.

No PySpark dependency, matching src/features/silver_token_trades.py's
split: this is the pure per-token time-series logic (unit-tested in
tests/test_gold_features.py, runs in CI); notebooks/02 and 03 are the
Spark glue that reads Silver, groups by token, calls this, and writes
the two Gold tables.

Scope (v1, per docs/label_spec.md and docs/decisions.md):
- Feature set limited to what Silver's token_trades actually has: trailing
  volume, buy/sell ratio, price action. Liquidity depth, holder
  concentration, and deployer wallet history are NOT included -- they need
  Bronze sources (pool-state snapshots, wallet data) that don't exist yet.
- KNOWN GAP, not silently handled: label_spec.md's rug-clamping rule ("if
  liquidity is pulled before T+HOLD_MINUTES, clamp the label at the
  realized loss at the pull event") is NOT implemented here. There's no
  liquidity/pool-state data to detect an actual liquidity-pull event, only
  a trade log. This module instead DROPS a (token, T) row when no trade
  exists near T+HOLD_MINUTES to serve as an exit price. That's a real
  survivorship-bias risk -- quietly excluding likely-rugged/dead tokens
  from training data, rather than labeling them as the large losses they
  are -- exactly what docs/game_plan.md's SS2 warns backtests lie about.
  Revisit once Bronze ingests pool-liquidity state.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

HOLD_MINUTES = 60  # placeholder -- tune once real price-decay distributions
                    # are visible in Gold; see docs/label_spec.md
TRAILING_WINDOW = timedelta(hours=48)  # placeholder, matches label_spec.md's
                                        # 24-72h candidate-activity window
PRICE_STALENESS = timedelta(minutes=30)  # how old the nearest trade can be
                                          # and still count as "the price at T"
T_SAMPLE_INTERVAL = timedelta(minutes=15)  # training-data T-sampling grid --
    # matches the deployed Bronze/Silver cadence, one of the two options
    # label_spec.md's "Training sample cadence" open question names, picked
    # as the simplest defensible default. Revisit if train/serve skew shows up.
EXIT_SIMULATED_HOLDS: list[timedelta] = [
    timedelta(minutes=15),
    timedelta(minutes=30),
    timedelta(minutes=60),
    timedelta(minutes=120),
]  # placeholder grid for simulated holding_duration h; see docs/label_spec.md

TokenTrades = list[dict[str, Any]]  # must be sorted ascending by trade_time


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def price_at(
    trades: TokenTrades, t: datetime, *, max_staleness: timedelta = PRICE_STALENESS
) -> float | None:
    """Last real trade price at or before `t`, within `max_staleness` of it.
    Never looks forward past `t` -- point-in-time correctness. Returns None
    (not a guess) when nothing recent enough exists."""
    best = None
    for trade in trades:
        ts = _parse(trade["trade_time"])
        if ts > t:
            break
        best = trade
    if best is None:
        return None
    if t - _parse(best["trade_time"]) > max_staleness:
        return None
    return float(best["price_usd"])


def trailing_activity_features(
    trades: TokenTrades, t: datetime, *, window: timedelta = TRAILING_WINDOW
) -> dict[str, Any] | None:
    """Volume/buy-sell/price-action features over (t - window, t].
    Shared by entry and exit builds -- must never see anything after t."""
    window_start = t - window
    windowed = [tr for tr in trades if window_start < _parse(tr["trade_time"]) <= t]
    if not windowed:
        return None

    buy_count = sum(1 for tr in windowed if tr["side"] == "buy")
    sell_count = sum(1 for tr in windowed if tr["side"] == "sell")
    first_price = float(windowed[0]["price_usd"])
    last_price = float(windowed[-1]["price_usd"])

    return {
        "trailing_volume_usd": sum(float(tr["amount_usd"]) for tr in windowed),
        "trailing_trade_count": len(windowed),
        "trailing_buy_count": buy_count,
        "trailing_sell_count": sell_count,
        # Laplace-smoothed so an all-buy or all-sell window gives a large
        # but finite ratio instead of a divide-by-zero / infinity.
        "trailing_buy_sell_ratio": (buy_count + 1) / (sell_count + 1),
        "trailing_price_change_pct": (
            (last_price - first_price) / first_price if first_price else None
        ),
    }


def _meta(trades: TokenTrades) -> dict[str, Any]:
    ref = trades[-1]
    return {
        "chain_family": ref["chain_family"],
        "network": ref.get("network"),
        "dex_venue": ref["dex_venue"],
        "dex_protocol": ref["dex_protocol"],
    }


def build_entry_row(
    token_id: str,
    t: datetime,
    trades: TokenTrades,
    *,
    hold_minutes: int = HOLD_MINUTES,
) -> dict[str, Any] | None:
    """One lawrence.gold.entry_candidates row for (token_id, t), or None if
    a price/feature endpoint isn't available (dropped -- see module
    docstring's KNOWN GAP note on survivorship bias)."""
    price_t = price_at(trades, t)
    price_exit = price_at(trades, t + timedelta(minutes=hold_minutes))
    if price_t is None or price_exit is None:
        return None

    features = trailing_activity_features(trades, t)
    if features is None:
        return None

    return {
        "token_id": token_id,
        **_meta(trades),
        "assessment_time": t.isoformat(),
        "price_at_t": price_t,
        "hold_minutes": hold_minutes,
        "y_entry": (price_exit - price_t) / price_t,
        **features,
    }


def build_exit_rows(
    token_id: str,
    t: datetime,
    trades: TokenTrades,
    *,
    hold_minutes: int = HOLD_MINUTES,
    simulated_holds: list[timedelta] = EXIT_SIMULATED_HOLDS,
) -> list[dict[str, Any]]:
    """0+ lawrence.gold.exit_candidates rows for (token_id, t) -- one per
    simulated holding duration h in `simulated_holds` for which a synthetic
    entry price is available. See docs/label_spec.md's "Exit label:
    training data via simulated entries" -- h is sampled independently of
    entry-model logic, by design."""
    price_t = price_at(trades, t)
    price_exit = price_at(trades, t + timedelta(minutes=hold_minutes))
    if price_t is None or price_exit is None:
        return []

    features = trailing_activity_features(trades, t)
    if features is None:
        return []

    y_exit = (price_exit - price_t) / price_t

    rows = []
    for h in simulated_holds:
        entry_price = price_at(trades, t - h)
        if not entry_price:  # None or 0.0
            continue
        rows.append(
            {
                "token_id": token_id,
                **_meta(trades),
                "assessment_time": t.isoformat(),
                "price_at_t": price_t,
                "hold_minutes": hold_minutes,
                "simulated_holding_minutes": h.total_seconds() / 60,
                "entry_price": entry_price,
                "unrealized_return": (price_t - entry_price) / entry_price,
                "y_exit": y_exit,
                **features,
            }
        )
    return rows


def sample_assessment_times(
    trades: TokenTrades, *, interval: timedelta = T_SAMPLE_INTERVAL
) -> list[datetime]:
    """Candidate T values for a token: a fixed grid from its first to last
    trade, at `interval` spacing. Training-data sampling only -- production
    scoring (T = "now") doesn't need this."""
    if not trades:
        return []
    start = _parse(trades[0]["trade_time"])
    end = _parse(trades[-1]["trade_time"])
    times = []
    cur = start
    while cur <= end:
        times.append(cur)
        cur += interval
    return times
