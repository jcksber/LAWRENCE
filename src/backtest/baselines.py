"""
Rules-based trading-decision baselines for the Phase 3 entry model to beat
(2026-09-17, see docs/game_plan.md Phase 3 and docs/decisions.md).

Baseline A mirrors BIA's actual current strategy ("buy everything new") --
the honest zero-signal floor game_plan.md §5 asks for. Baseline B is a
stand-in for game_plan.md §5's originally-envisioned "liquidity > X and no
prior rugs from this deployer" rule: that needs pool-liquidity-state and
deployer-wallet data that doesn't exist yet (the Phase 1 gap tracked in
game_plan.md and gold_spec.md's feature-set note). Until that data lands,
Baseline B substitutes the closest signal Gold v1 actually has: trailing
volume and buy-pressure, on the theory that a token with real trading
activity and more buying than selling is a weaker candidate for "already
dead" than one with none. This is explicitly a placeholder, not a
permanent design -- swap for the real liquidity/deployer version once
that data exists.

Both operate on Gold's entry_candidates schema (docs/gold_spec.md) and
return a boolean pd.Series aligned to the input DataFrame's index -- True
means "this baseline would enter the trade at this (token_id, T)."

No PySpark dependency, matching the rest of src/features/ and src/models/
-- plain pandas, unit-tested here, called from notebooks/04 for the
actual Spark-table read.
"""
from __future__ import annotations

import pandas as pd

# Placeholders, not measured numbers -- tune once real backtest results
# (this module feeding notebooks/04_entry_baseline_and_model.py) show what
# actually separates a "live" token from a dead one. Deliberately
# permissive rather than tight: the point of Baseline B is to be a LOW bar
# the model should clear, not a good trading rule in its own right.
BASELINE_B_MIN_TRAILING_VOLUME_USD = 1_000.0
BASELINE_B_MIN_BUY_SELL_RATIO = 1.0  # >= 1.0 means at least as many buys as sells


def baseline_a_buy_everything(candidates: pd.DataFrame) -> pd.Series:
    """The zero-signal floor: enter every candidate, no filtering at all --
    literally BIA's current rule ("buy everything new") applied to Gold's
    candidate set instead of a raw new-pool-creation feed."""
    return pd.Series(True, index=candidates.index)


def baseline_b_volume_momentum_filter(candidates: pd.DataFrame) -> pd.Series:
    """Enter only candidates with enough trailing volume and buy pressure
    to look like a still-live token -- a stand-in for real liquidity/
    deployer-history filtering, not a replacement for it (see module
    docstring)."""
    return (candidates["trailing_volume_usd"] >= BASELINE_B_MIN_TRAILING_VOLUME_USD) & (
        candidates["trailing_buy_sell_ratio"] >= BASELINE_B_MIN_BUY_SELL_RATIO
    )
