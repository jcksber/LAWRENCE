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

2026-10-01 -- Gold v2 input (see docs/decisions.md): records are now
1-minute BARS from `lawrence.silver.token_price_1m`, not raw trades. Each
record carries `trade_time` (= bar_start + 1 minute, i.e. when the bar's
close is known -- point-in-time safe), `price` (the bar close, in the
token's single `price_unit`), `amount_usd`, and `n_trades` / `n_buys`
(how many trades / buys the bar summarizes). Raw-trade records (no
n_trades, `price_usd` instead of `price`) still work, so existing tests and
callers are unchanged. Two v2 behaviours:
- CENSORING vs RUG: post-graduation bars only cover 24h after the pool is
  created and the backfill ends 2026-09-13, so "no price at T+hold" can
  mean "past the end of what we pulled", not "rugged". Callers pass each
  token's `observed_until`; any T with T+hold beyond it is skipped (a
  censored label), never clamped to RUG_CLAMP_RETURN. Before v2, graduated
  tokens vanished at graduation and were silently labelled as rugs.
- New features: `minutes_since_launch` and `is_postgrad_at_t`.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

HOLD_MINUTES = 15  # PRIMARY label horizon (y_entry / y_exit). 2026-10-01:
    # cut from 60 -- median pump.fun graduation is ~4.4 min after launch and
    # most tokens are dead within 15, so a 60-min hold measured long after
    # everything happened. Still a placeholder; LABEL_HOLD_MINUTES below
    # stores several horizons so the choice can be made from data.
LABEL_HOLD_MINUTES: tuple[int, ...] = (5, 15, 60)  # every Gold row also carries
    # y_entry_{h}m / y_exit_{h}m for each h here (NULL when censored for that h)
NO_RUG_PHASES = frozenset({"curve", "postgrad"})  # 2026-10-01: phases whose
    # liquidity CAN'T be pulled -- a bonding curve holds its SOL in the
    # program (you can always sell back at the curve price), and pump.fun /
    # LaunchLab burn or lock the graduation pool's LP. A quiet token in these
    # phases keeps its LAST price; it is never rug-clamped to -100%.
TRAILING_WINDOW = timedelta(hours=48)  # placeholder, matches label_spec.md's
                                        # 24-72h candidate-activity window
PRICE_STALENESS = timedelta(minutes=30)  # how old the nearest trade can be
                                          # and still count as "the price at T"
DENSE_SAMPLE_INTERVAL = timedelta(minutes=1)  # 2026-10-01: sample every minute
DENSE_SAMPLE_WINDOW = timedelta(minutes=60)    # for a token's first hour after
    # launch (where nearly all launch-token decisions and action happen), then
    # fall back to T_SAMPLE_INTERVAL. Only applies when launch_time is known.
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

# 2026-09-17 (see docs/decisions.md): real Gold data surfaced 779 rows
# (0.061% of entry_candidates) with y_entry in the hundreds-of-thousands
# to 16.7 BILLION range -- not real winners, an artifact of tokens priced
# at 1e-8 to 1e-15 USD (huge supply, near-zero market cap, all plausible
# real data, not a bug) where a tiny absolute price move produces an
# astronomical percentage "return." No real trade could capture that
# without knowing actual pool depth -- the same liquidity-data gap already
# tracked in gold_spec.md's feature-set section and game_plan.md's Phase
# 1. label_spec.md's original plan ("winsorize y for training stability,
# keep the raw value for backtest P&L simulation") assumed "raw" stayed
# meaningful for backtesting -- real data disproved that: an 18-billion-x
# "return" is exactly as uncapturable in a backtest P&L simulation as it
# is misleading during training, so this cap applies to BOTH, not just
# training. Deliberately a CLIP, not a drop -- see game_plan.md §2's
# survivorship-bias warning, which applies just as much to silently
# dropping extreme winners as it does to silently dropping rugs. Lives
# here, next to RUG_CLAMP_RETURN, rather than in src/backtest/ or
# src/models/, so both depend on one shared definition of "how this label
# behaves at the extremes" instead of a sideways dependency between them.
# 100.0 (+/-10,000%) matches label_spec.md's own "extreme (>100x)
# winners" framing -- a placeholder, not a measured number; tune once
# real liquidity-depth data exists to distinguish genuine big winners from
# unexecutable thin-pool noise. Gold's OWN stored y_entry/y_exit values
# are never clipped -- this is applied by callers (src/models/,
# src/backtest/) at consumption time, not written back to Gold.
RETURN_WINSORIZE_CAP = 100.0


def winsorize_return(y: Any) -> Any:
    """Clips a return (or a pandas Series/numpy array of returns) to
    [-RETURN_WINSORIZE_CAP, RETURN_WINSORIZE_CAP]. See
    RETURN_WINSORIZE_CAP's comment for why this exists and why it's a
    clip, not a drop. Works on a scalar or anything with `.clip()`
    (pandas Series, numpy array) via duck typing -- this module has no
    pandas/numpy dependency of its own (see module docstring), so it
    doesn't import either just to support this."""
    lo, hi = -RETURN_WINSORIZE_CAP, RETURN_WINSORIZE_CAP
    if hasattr(y, "clip"):
        return y.clip(lo, hi)
    return max(lo, min(hi, y))


TokenTrades = list[dict[str, Any]]  # must be sorted ascending by trade_time


def _parse(ts: str | datetime) -> datetime:
    """Parses a trade_time string. Must handle two real shapes (2026-09-17,
    see docs/decisions.md): the ISO-8601 "...Z"/"...+00:00" shape this
    codebase writes going forward, AND Dune's raw
    "YYYY-MM-DD HH:MM:SS.mmm UTC" shape, which turned up unnormalized in
    real Silver data -- a resumable-backfill side effect: `_normalize_
    block_time` in src/ingestion/dune_client.py was added to convert this
    at write time, but resumability means chunks already written+marked
    'success' before that fix never got reprocessed, so Bronze/Silver
    genuinely contain a mix of both shapes for `block_time`/`trade_time`,
    not just one. Always returns a UTC-aware datetime (never naive) so
    comparisons between differently-sourced rows never mix aware/naive --
    every timestamp in this codebase is UTC (see dune_client.py's own
    normalization comment)."""
    # 2026-09-22: Silver's trade_time is now a real TimestampType, so
    # applyInPandas hands over pandas Timestamps (a datetime subclass,
    # tz-naive in UTC since notebooks set spark.sql.session.timeZone=UTC).
    # Strings still accepted for tests and any pre-rebuild data.
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    s = ts.strip()
    if s.endswith(" UTC"):
        dt = datetime.fromisoformat(s[: -len(" UTC")].replace(" ", "T", 1))
    else:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _price(rec: dict[str, Any]) -> float | None:
    """v2 bar records carry `price` (in the token's price_unit); v1 raw
    trades carry `price_usd`."""
    v = rec.get("price", rec.get("price_usd"))
    return None if _missing(v) else float(v)


def _missing(v: Any) -> bool:
    """None, or a float NaN -- applyInPandas turns NULL doubles into NaN,
    and `NaN or 0.0` is still NaN, so a plain truthiness check isn't enough."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _n(rec: dict[str, Any], key: str, default: int) -> int:
    v = rec.get(key)
    return default if _missing(v) else int(v)


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
    (not a guess) when nothing recent enough exists -- including when the
    only candidate has a non-positive price_usd (2026-09-17, see
    docs/decisions.md: a real Silver row hit price_usd == 0.0 -- a dust/
    degenerate-swap data artifact, not a meaningful "worth $0" signal --
    and (price_exit - 0.0) / 0.0 crashed build_entry_row with
    ZeroDivisionError). Treating it as "no usable price" reuses the
    existing, already-tested genuine-data-gap handling in build_entry_row/
    build_exit_rows (drop the row) instead of adding a second code path."""
    idx = _count_at_or_before(trades, t) - 1
    if idx < 0:
        return None
    best = trades[idx]
    if t - _parse(best["trade_time"]) > max_staleness:
        return None
    price = _price(best)
    if price is None or price <= 0:
        return None
    return price


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

    # A raw trade counts as 1 trade (a buy if side == "buy"); a v2 bar
    # counts as its n_trades / n_buys.
    trade_count = sum(_n(tr, "n_trades", 1) for tr in windowed)
    buy_count = sum(_n(tr, "n_buys", 1 if tr.get("side") == "buy" else 0) for tr in windowed)
    sell_count = trade_count - buy_count
    first_price = _price(windowed[0])
    last_price = _price(windowed[-1])

    return {
        "trailing_volume_usd": sum(
            0.0 if _missing(tr.get("amount_usd")) else float(tr["amount_usd"]) for tr in windowed
        ),
        "trailing_trade_count": trade_count,
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
        "chain_family": ref.get("chain_family"),
        "network": ref.get("network"),
        "dex_venue": ref.get("dex_venue"),
        "dex_protocol": ref.get("dex_protocol"),
    }


def _v2_features(trades: TokenTrades, t: datetime, launch_time: Any) -> dict[str, Any]:
    """minutes_since_launch and is_postgrad_at_t (None when unknown)."""
    idx = _count_at_or_before(trades, t) - 1
    phase = trades[idx].get("phase") if idx >= 0 else None
    return {
        "minutes_since_launch": (
            (t - _parse(launch_time)).total_seconds() / 60 if launch_time is not None else None
        ),
        "is_postgrad_at_t": None if phase is None else phase == "postgrad",
    }


def _censored(t: datetime, hold_minutes: int, observed_until: Any) -> bool:
    return observed_until is not None and t + timedelta(minutes=hold_minutes) > _parse(observed_until)


def _exit_price(trades: TokenTrades, t_exit: datetime) -> float | None:
    """Price realizable at t_exit. Normal staleness rule first; if that finds
    nothing but the token's last record at/before t_exit is in a
    NO_RUG_PHASES phase, the token just went quiet -- its last price still
    stands (no LP to pull), so use it with no staleness limit."""
    p = price_at(trades, t_exit)
    if p is not None:
        return p
    idx = _count_at_or_before(trades, t_exit) - 1
    if idx >= 0 and trades[idx].get("phase") in NO_RUG_PHASES:
        last = _price(trades[idx])
        if last is not None and last > 0:
            return last
    return None


def _forward_return(
    trades: TokenTrades, t: datetime, price_t: float, hold_minutes: int, observed_until: Any
) -> tuple[float, bool] | None:
    """(return, is_rug_clamped) over `hold_minutes`, or None if censored."""
    if _censored(t, hold_minutes, observed_until):
        return None
    price_exit = _exit_price(trades, t + timedelta(minutes=hold_minutes))
    if price_exit is not None:
        return (price_exit - price_t) / price_t, False
    return RUG_CLAMP_RETURN, True


def _multi_horizon(
    prefix: str, trades: TokenTrades, t: datetime, price_t: float, observed_until: Any
) -> dict[str, Any]:
    out = {}
    for h in LABEL_HOLD_MINUTES:
        r = _forward_return(trades, t, price_t, h, observed_until)
        out[f"{prefix}_{h}m"] = None if r is None else r[0]
    return out


def build_entry_row(
    token_id: str,
    t: datetime,
    trades: TokenTrades,
    *,
    hold_minutes: int = HOLD_MINUTES,
    observed_until: Any = None,
    launch_time: Any = None,
) -> dict[str, Any] | None:
    """One lawrence.gold.entry_candidates row for (token_id, t), or None if
    there's no real price/feature anchor at T itself (genuine data gap --
    still dropped). If T has a real price but nothing trades near
    T+hold_minutes, the row is KEPT with a clamped loss label instead of
    being dropped -- see module docstring's interim rug-clamping note.
    If T+hold_minutes is past `observed_until`, the label is censored and
    the row is skipped (v2, see module docstring)."""
    if _censored(t, hold_minutes, observed_until):
        return None
    price_t = price_at(trades, t)
    if price_t is None:
        return None

    features = trailing_activity_features(trades, t)
    if features is None:
        return None

    y_entry, is_rug_clamped = _forward_return(trades, t, price_t, hold_minutes, None)

    return {
        "token_id": token_id,
        **_meta(trades),
        "assessment_time": t.isoformat(),
        "price_at_t": price_t,
        "hold_minutes": hold_minutes,
        "y_entry": y_entry,
        "is_rug_clamped": is_rug_clamped,
        **_multi_horizon("y_entry", trades, t, price_t, observed_until),
        **features,
        **_v2_features(trades, t, launch_time),
    }


def build_exit_rows(
    token_id: str,
    t: datetime,
    trades: TokenTrades,
    *,
    hold_minutes: int = HOLD_MINUTES,
    simulated_holds: list[timedelta] = EXIT_SIMULATED_HOLDS,
    observed_until: Any = None,
    launch_time: Any = None,
) -> list[dict[str, Any]]:
    """0+ lawrence.gold.exit_candidates rows for (token_id, t) -- one per
    simulated holding duration h in `simulated_holds` for which a synthetic
    entry price is available. See docs/label_spec.md's "Exit label:
    training data via simulated entries" -- h is sampled independently of
    entry-model logic, by design. Censored T (see build_entry_row) -> []."""
    if _censored(t, hold_minutes, observed_until):
        return []
    price_t = price_at(trades, t)
    if price_t is None:
        return []

    features = trailing_activity_features(trades, t)
    if features is None:
        return []

    # Rug-clamp only where liquidity can actually be pulled (see
    # _exit_price / NO_RUG_PHASES); censoring already handled above.
    y_exit, is_rug_clamped = _forward_return(trades, t, price_t, hold_minutes, None)
    horizons = _multi_horizon("y_exit", trades, t, price_t, observed_until)

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
                **horizons,
                **features,
                **_v2_features(trades, t, launch_time),
            }
        )
    return rows


def sample_assessment_times(
    trades: TokenTrades,
    *,
    interval: timedelta = T_SAMPLE_INTERVAL,
    launch_time: Any = None,
    dense_interval: timedelta = DENSE_SAMPLE_INTERVAL,
    dense_window: timedelta = DENSE_SAMPLE_WINDOW,
) -> list[datetime]:
    """Candidate T values for a token, from its first to its last record.

    Without `launch_time`: a fixed grid at `interval` (v1 behaviour).
    With `launch_time` (2026-10-01): every `dense_interval` for the first
    `dense_window` after launch, then every `interval` -- most launch tokens
    live and die within minutes, so a 15-minute grid gave many of them a
    single sample at their first bar. Training-data sampling only."""
    if not trades:
        return []
    start = _parse(trades[0]["trade_time"])
    end = _parse(trades[-1]["trade_time"])
    dense_until = _parse(launch_time) + dense_window if launch_time is not None else None
    times = []
    cur = start
    while cur <= end:
        times.append(cur)
        cur += dense_interval if dense_until is not None and cur < dense_until else interval
    return times
