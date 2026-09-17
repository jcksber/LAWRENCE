"""
Walk-forward time-based splitting and trade-simulation for backtesting
LAWRENCE's entry baselines/model (2026-09-17, see docs/game_plan.md Phase
3 and §5's "walk-forward validation only" / "backtest realism" rules).

Two responsibilities, deliberately separate:
- `walk_forward_folds`: turns a column of assessment timestamps into a
  sequence of (train_mask, test_mask) pairs, expanding-window style --
  fold i trains on everything strictly before its test window and tests
  on the next chronological slice. Never a random split: game_plan.md §5
  is explicit that a random train/test split on time-series token data is
  exactly how these backtests lie.
- `simulate_trades`: given which rows a strategy (baseline or model) would
  have entered and the real forward return for each, applies round-trip
  trading costs and reports the numbers game_plan.md §0/§2 actually care
  about (EV/trade, win rate, max drawdown) instead of a naive "average
  return," which game_plan.md itself warns is a misleading target.

KNOWN SIMPLIFICATION (v1, not yet addressed): `simulate_trades` treats
every entered candidate as an independent, equal-sized bet and does not
model concurrent open positions, capital constraints, or correlated
drawdowns across tokens trading at the same time. That's a real gap
before Phase 4/5 (real paper trading has exactly these constraints), but
position-sizing/portfolio simulation is future work, not this pass --
game_plan.md's Phase 3 asks for "start simple, walk-forward, judge
relative to baselines," not a full portfolio simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def walk_forward_folds(
    assessment_time: pd.Series, *, n_folds: int = 5
) -> list[tuple[pd.Series, pd.Series]]:
    """Expanding-window walk-forward folds over calendar time, not row
    count -- token trading activity is wildly non-uniform over time (see
    docs/decisions.md's outlier-token findings), so an equal-ROW split
    would badly skew which calendar period lands in which fold. Splits
    the observed [min, max] timestamp range into `n_folds + 1` equal-
    WIDTH time buckets instead: fold i's test window is bucket i+1, and
    its train window is every row strictly before that bucket's start --
    fold 0 has the least training history, the last fold the most,
    matching how this would actually play out in production (retrain on
    everything seen so far, test on what comes next).

    Returns up to `n_folds` boolean-mask pairs (train_mask, test_mask)
    aligned to `assessment_time`'s index -- caller applies them to
    whatever other columns (features, labels, baseline signals) share
    that index. Caller is responsible for keeping the underlying
    DataFrame sorted by assessment_time ascending -- same invariant as
    gold_features.py's TokenTrades -- so a mask's kept rows stay in
    chronological order for `simulate_trades`' cumulative-return curve to
    mean anything.

    2026-09-17 (see docs/decisions.md): the first implementation split
    [min, max] into `n_folds` buckets and used bucket 0 as BOTH a test
    window AND (via "everything before it") its own train window -- which
    is always empty, since bucket 0 starts at the very first observed
    timestamp. That silently dropped fold 0 every time, so `n_folds=4`
    only ever returned 3 folds -- caught by a standalone test asserting
    `len(folds) == n_folds`. Fixed by using `n_folds + 2` boundary points
    instead: bucket 0 is reserved purely as the initial training seed
    (never a test window itself), and test folds are buckets 1..n_folds --
    exactly matching this function's own "fold i trains on everything
    strictly before its test window" contract instead of contradicting it.
    """
    t = pd.to_datetime(assessment_time, utc=True)
    start, end = t.min(), t.max()
    if pd.isna(start) or pd.isna(end) or start == end:
        raise ValueError(
            "walk_forward_folds needs assessment_time to span a real "
            "range -- got a single instant or all-null input."
        )
    boundaries = pd.date_range(start, end, periods=n_folds + 2)

    folds = []
    for i in range(1, n_folds + 1):
        test_start, test_end = boundaries[i], boundaries[i + 1]
        is_last = i == n_folds
        train_mask = t < test_start
        test_mask = (t >= test_start) & (t <= test_end if is_last else t < test_end)
        if train_mask.any() and test_mask.any():
            folds.append((train_mask, test_mask))
    return folds


@dataclass
class BacktestResult:
    strategy: str
    n_candidates: int
    n_trades: int
    ev_per_trade: float  # mean realized return per trade, after costs
    win_rate: float  # fraction of trades with positive realized return
    max_drawdown: float  # <= 0, most negative peak-to-trough dip in cumulative return
    total_return: float  # sum of per-trade realized returns (equal-sized bets)


def simulate_trades(
    y_true: pd.Series,
    enter_mask: pd.Series,
    *,
    strategy: str,
    fee_bps: float = 30.0,
    slippage_bps: float = 50.0,
) -> BacktestResult:
    """Applies round-trip trading costs (paid on both entry and exit, each
    at fee_bps + slippage_bps) to every row `enter_mask` selects, then
    reports EV/trade, win rate, and max drawdown on the resulting
    cumulative-return curve. fee_bps/slippage_bps are placeholders (30bps
    DEX fee, 50bps slippage -- rough, not measured from real fills) per
    game_plan.md §5's "simulate actual slippage, DEX fees" -- tune once
    real fill data exists.

    `y_true` is the row's already point-in-time-correct forward return
    (`y_entry`, including the interim rug-clamp -- see gold_spec.md) --
    this function adds trading costs on top, it does not touch label
    correctness. Assumes `y_true` is already ordered chronologically by
    assessment_time (see walk_forward_folds) -- the cumulative curve
    otherwise doesn't mean anything.
    """
    round_trip_cost = 2 * (fee_bps + slippage_bps) / 10_000
    trades = y_true[enter_mask].astype(float) - round_trip_cost

    n_candidates = int(len(y_true))
    n_trades = int(len(trades))
    if n_trades == 0:
        return BacktestResult(
            strategy, n_candidates, 0, float("nan"), float("nan"), float("nan"), 0.0
        )

    cumulative = trades.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = float(drawdown.min())

    return BacktestResult(
        strategy=strategy,
        n_candidates=n_candidates,
        n_trades=n_trades,
        ev_per_trade=float(trades.mean()),
        win_rate=float((trades > 0).mean()),
        max_drawdown=max_drawdown,
        total_return=float(trades.sum()),
    )
