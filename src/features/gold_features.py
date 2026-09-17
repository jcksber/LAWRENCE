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
  Bronze sources (pool-state snapshots, wallet data) that don't exist yet
  (tracked in docs/game_plan.md's Phase 1 and docs/decisions.md).
- INTERIM rug-clamping heuristic (2026-09-17, see docs/decisions.md):
  label_spec.md's real rule is "if liquidity is pulled before
  T+HOLD_MINUTES, clamp the label at the realized loss at the pull event."
  That needs an actual liquidity-pull event, which needs pool-state data we
  don't have -- only a trade log. Until that data exists, a token that had
  a real price at T but no trade at all near T+HOLD_MINUTES (see
  `price_at`'s `max_staleness`) is treated as a proxy rug/dead-token signal:
  the label is clamped to RUG_CLAMP_RETURN instead of the row being dropped,
  and `is_rug_clamped=True` is carried on the row so it stays inspectable
  and filterable rather than hidden. This can't distinguish "rugged" from
  "just went quiet" -- it's a stand-in for real liquidity-pull detection,
  not the real thing -- but it removes the survivorship-bias risk that
  quietly excluding these rows would otherwise create (exactly what
  docs/game_plan.md's SS2 warns backtests lie about). Swap for real
  pull-event detection once Bronze ingests pool-liquidity state.
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

RUG_CLAMP_RETURN = -1.0  # interim heuristic label for a likely-rugged/dead
    # token (see module docstring) -- full realized loss, since we can't
    # observe the actual liquidity-pull price without pool-state data.

TokenTrades = list[dict[str, Any]]  # must be sorted ascending by trade_time


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _count_at_or_before(trades: TokenTrades, t: datetime) -> int:
    """Binary search: how many trades have trade_time <= t. Requires
    `trades` sorted ascending by trade_time (see TokenTrades) -- that's
    already a hard invariant every caller in this module relies on.

    2026-09-17 perf fix (see docs/decisions.md): price_at/
    trailing_activity_features used to do a full O(len(trades)) linear
    scan -- re-parsing every trade_time string along the way -- on EVERY
    call. That was invisible while Gold only ever ran against near-zero
    real history, but against the real Dune backfill (~10.75M Silver rows,
    some tokens with thousands of trades, sampled every 15 minutes across
    their whole active history) it made 02/03's driver-side Python loop the
    actual bottleneck: the Spark read itself finishes in well under a
    minute, but the notebook cell kept grinding for a long time after,
    since none of that per-token math shows up as a Spark job/statement.
    This turns each lookup into O(log n) instead of O(n)."""
    lo, hi = 0, len(trades)
    while lo < hi:
        mid = (lo + hi) // 2
        if _parse(trades[mid]["trade_time"]) <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def price_at(
    trades: TokenTrades, t: datetime, *, max_staleness: timedelta = PRICE_STALENESS
) -> float | None:
    """Last real trade price at or before `t`, within `max_staleness` of it.
    Never looks forward past `t` -- point-in-time correctness. Returns None
    (not a guess) when nothing recent enough exists."""
    idx = _count_at_or_before(trades, t) - 1
    if idx < 0:
        return None
    best = trades[idx]
    if t - _parse(best["trade_time"]) > max_staleness:
        return None
    return float(best["price_usd"])


def trailing_activity_features(
    trades: TokenTrades, t: datetime, *, window: timedelta = TRAILING_WINDOW
) -> dict[str, Any] | None:
    """Volume/buy-sell/price-action features over (t - window, t].
    Shared by entry and exit builds -- must never see anything after t."""
    window_start = t - window
    right = _count_at_or_before(trades, t)
    left = _count_at_or_before(trades, window_start)
    windowed = trades[left:right]
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
    there's no real price/feature anchor at T itself (genuine data gap --
    still dropped). If T has a real price but nothing trades near
    T+hold_minutes, the row is KEPT with a clamped loss label instead of
    being dropped -- see module docstring's interim rug-clamping note."""
    price_t = price_at(trades, t)
    if price_t is None:
        return None

    features = trailing_activity_features(trades, t)
    if features is None:
        return None

    price_exit = price_at(trades, t + timedelta(minutes=hold_minutes))
    if price_exit is not None:
        y_entry = (price_exit - price_t) / price_t
        is_rug_clamped = False
    else:
        y_entry = RUG_CLAMP_RETURN
        is_rug_clamped = True

    return {
        "token_id": token_id,
        **_meta(trades),
        "assessment_time": t.isoformat(),
        "price_at_t": price_t,
        "hold_minutes": hold_minutes,
        "y_entry": y_entry,
        "is_rug_clamped": is_rug_clamped,
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
    if price_t is None:
        return []

    features = trailing_activity_features(trades, t)
    if features is None:
        return []

    price_exit = price_at(trades, t + timedelta(minutes=hold_minutes))
    if price_exit is not None:
        y_exit = (price_exit - price_t) / price_t
        is_rug_clamped = False
    else:
        # Interim heuristic (see module docstring): T had a real price but
        # nothing trades near T+hold_minutes -- clamp instead of dropping.
        y_exit = RUG_CLAMP_RETURN
        is_rug_clamped = True

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
                "is_rug_clamped": is_rug_clamped,
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
