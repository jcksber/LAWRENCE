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
  have entered and the real forward return for each, winsorizes that
  return (see gold_features.py's RETURN_WINSORIZE_CAP -- real Gold data
  has ~0.06% of rows with "returns" in the billions-of-percent range,
  artifacts of near-zero-priced tokens, not real capturable winners),
  applies round-trip trading costs, and reports the numbers game_plan.md
  §0/§2 actually care about (EV/trade, win rate, max drawdown) instead of
  a naive "average return," which game_plan.md itself warns is a
  misleading target.

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

from features.gold_features import CAPTURABLE_RETURN_CAP, RETURN_WINSORIZE_CAP, winsorize_return  # noqa: F401

# 2026-09-18 (see docs/decisions.md): v1's notebook entry decision rule was
# simply "predicted > 0", which ignores the round-trip trading cost
# simulate_trades already applies below -- a model predicting, say, a 0.5%
# return would still "enter" even though that trade is a guaranteed loss
# once the ~1.6% round-trip cost (30bps fee + 50bps slippage, each way) is
# paid. Named here, not just inlined in simulate_trades, so the notebook's
# decision rule (`preds > round_trip_cost()`) uses the EXACT SAME cost
# assumptions the backtest itself scores that decision against -- two
# different cost numbers for "should I enter" vs. "what did entering cost
# me" would make the backtest lie about whether the threshold change
# actually helped.
DEFAULT_FEE_BPS = 30.0
DEFAULT_SLIPPAGE_BPS = 50.0

# 2026-10-02: per-row fees by venue + phase. The 30 bps default is an
# AMM-style fee (Raydium v4); Gold v2's candidates trade mostly on launchpad
# curves and fresh graduation pools, which charge far more. Verified
# against the venues' own docs on 2026-10-02 (see docs/decisions.md):
#   - pump.fun bonding curve: 1.25% per trade (0.30% creator + 0.95%
#     protocol), pump.fun/docs/fees.
#   - PumpSwap canonical pool after graduation: tiered by market cap; a
#     freshly graduated token (< ~1,470 SOL mcap) pays 1.20-1.25%, falling
#     to 0.30% only above ~98k SOL mcap. 120 bps used for the first 24h.
#   - Raydium LaunchLab curve: 1% base trade fee for LaunchLab-UI launches
#     (Raydium, 2025); third-party platforms (StonkFun, letsbonk.fun) can
#     add a platform fee on top -- not modelled, so this is a FLOOR.
#   - Raydium CPMM after LaunchLab graduation: 25 bps standard tier --
#     placeholder; platform fee tiers not verified.
VENUE_PHASE_FEE_BPS = {
    ("pumpdotfun", False): 125.0,
    ("pumpdotfun", True): 120.0,
    # 2026-10-03: 0.25% protocol + 1.0% platform fee (the near-universal
    # platform rate in the on-chain configs); creator fee unknown here. Only
    # a fallback -- notebook 04 passes each token's real config fee.
    ("raydium_launchlab", False): 125.0,
    ("raydium_launchlab", True): 25.0,
}


# LaunchLab's protocol trade fee (on-chain GlobalConfig trade_fee_rate =
# 2500 millionths, confirmed via Dune 2026-10-03). Each launch's
# PlatformConfig adds fee_rate + creator_fee_rate on top -- see
# launchlab_curve_fee_bps and data/reference/launchlab_platform_configs.csv.
LAUNCHLAB_PROTOCOL_FEE_BPS = 25.0


def launchlab_curve_fee_bps(fee_rate: "float | pd.Series", creator_fee_rate: "float | pd.Series") -> "float | pd.Series":
    """Per-side LaunchLab curve fee in bps from a PlatformConfig's rates
    (both in millionths, as stored on-chain: 10000 = 1%)."""
    return LAUNCHLAB_PROTOCOL_FEE_BPS + (fee_rate + creator_fee_rate) / 100


def venue_fee_bps(
    dex_venue: pd.Series, is_postgrad: pd.Series, launchlab_curve_bps: "pd.Series | None" = None
) -> pd.Series:
    """Per-row fee in bps from (launch venue, is_postgrad_at_t); anything
    unmapped falls back to DEFAULT_FEE_BPS. `launchlab_curve_bps` (optional,
    aligned to dex_venue's index) overrides the flat LaunchLab curve rate
    with the row's own platform fee where known (non-null); post-graduation
    rows and other venues ignore it."""
    postgrad = is_postgrad.fillna(False).astype(bool)
    fees = pd.Series(
        [
            VENUE_PHASE_FEE_BPS.get((v, p), DEFAULT_FEE_BPS)
            for v, p in zip(dex_venue.astype(object), postgrad)
        ],
        index=dex_venue.index,
        dtype=float,
    )
    if launchlab_curve_bps is not None:
        use = (dex_venue.astype(object) == "raydium_launchlab") & ~postgrad & launchlab_curve_bps.notna()
        fees[use] = launchlab_curve_bps[use].astype(float)
    return fees


def round_trip_cost(
    fee_bps: "float | pd.Series" = DEFAULT_FEE_BPS, slippage_bps: float = DEFAULT_SLIPPAGE_BPS
) -> "float | pd.Series":
    """Round-trip trading cost as a fraction of position size (fee +
    slippage, paid on both entry and exit) -- same formula `simulate_trades`
    applies internally, exposed here so callers (the notebook's entry
    decision rule) can share it instead of re-deriving or hard-coding
    `0.016`."""
    return 2 * (fee_bps + slippage_bps) / 10_000


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
    fee_bps: "float | pd.Series" = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    return_cap: float = CAPTURABLE_RETURN_CAP,
) -> BacktestResult:
    """Applies round-trip trading costs (paid on both entry and exit, each
    at fee_bps + slippage_bps) to every row `enter_mask` selects, then
    reports EV/trade, win rate, and max drawdown on the resulting
    cumulative-return curve. fee_bps/slippage_bps are placeholders (30bps
    DEX fee, 50bps slippage -- rough, not measured from real fills) per
    game_plan.md §5's "simulate actual slippage, DEX fees" -- tune once
    real fill data exists.

    `y_true` is the row's already point-in-time-correct forward return
    (`y_entry`, including the interim rug-clamp -- see gold_spec.md).
    Winsorized to `RETURN_WINSORIZE_CAP` before costs are applied
    (2026-09-17, see gold_features.py -- real Gold data has a long right
    tail of billions-of-percent "returns" from near-zero-priced tokens
    that would otherwise dominate every metric below; label_spec.md's
    original plan to keep backtest values unclipped didn't survive
    contact with real data). Assumes `y_true` is already ordered
    chronologically by assessment_time (see walk_forward_folds) -- the
    cumulative curve otherwise doesn't mean anything.
    """
    # 2026-10-02: scored at CAPTURABLE_RETURN_CAP (10x) by default, the same
    # cap the model trains on -- see gold_features.CAPTURABLE_RETURN_CAP.
    y_true = winsorize_return(y_true, return_cap)
    # fee_bps may be a per-row Series (venue_fee_bps) aligned to y_true.
    cost = round_trip_cost(fee_bps, slippage_bps)
    if isinstance(cost, pd.Series):
        cost = cost[enter_mask]
    trades = y_true[enter_mask].astype(float) - cost

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
