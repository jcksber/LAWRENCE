"""
First entry model (2026-09-17, Phase 3 -- see docs/game_plan.md §6). A
LightGBM regressor predicting `y_entry` directly from Gold's trailing-
activity features, per game_plan.md's "start simple -- gradient boosted
trees on tabular features before anything fancier."

Regression, not classification: y_entry is already a continuous forward
return (including the interim rug-clamp at RUG_CLAMP_RETURN -- see
gold_features.py) -- predicting it directly supports ranking candidates by
expected return, which is how an entry decision actually gets made, and
keeps a -100% rug loss as real magnitude instead of collapsing it into the
same "loss" bucket as a -5% one. Rug-clamped rows are NOT filtered out of
training -- they're real signal, not noise (see gold_spec.md's rug-
clamping rationale: dropping them is exactly the survivorship bias
game_plan.md §2 warns about).

v1 feature scope is deliberately narrow: only the numeric trailing-
activity columns Gold v1 actually has (see gold_spec.md's feature table).
chain_family/network/dex_venue/dex_protocol are real signal LightGBM could
use natively as categoricals, but adding them means an encoding decision
this first pass doesn't need to make yet -- tracked as a follow-up once
this numeric-only baseline is validated, not an oversight.
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

FEATURE_COLUMNS = [
    "trailing_volume_usd",
    "trailing_trade_count",
    "trailing_buy_count",
    "trailing_sell_count",
    "trailing_buy_sell_ratio",
    "trailing_price_change_pct",
]
TARGET_COLUMN = "y_entry"


def prepare_features(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Selects the model's input columns and target from a Gold
    entry_candidates DataFrame. Deliberately does NOT impute
    trailing_price_change_pct's occasional None (see
    trailing_activity_features -- it's None when the trailing window's
    first trade itself had price_usd == 0, a real "we don't actually
    know" gap, not a value to guess at). LightGBM handles missing numeric
    features natively (it learns a default split direction for them), so
    leaving it as NaN is the correct choice here, not a gap to fill in.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    y = candidates[TARGET_COLUMN].copy()
    return X, y


def train_entry_model(
    X_train: pd.DataFrame, y_train: pd.Series, **lgbm_params
) -> LGBMRegressor:
    """Fits a LightGBM regressor. `lgbm_params` passes through to
    LGBMRegressor for tuning without touching this function -- v1 uses
    its defaults plus a fixed random_state for reproducibility."""
    params = {"random_state": 17, **lgbm_params}
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def score_candidates(model: LGBMRegressor, X: pd.DataFrame) -> pd.Series:
    """Predicted y_entry for each candidate -- higher is a better entry.
    v1's decision rule (see notebooks/04) is simply "predicted > 0";
    ranking by this score and taking the top-K is an easy variant to try
    later without changing this function."""
    return pd.Series(model.predict(X), index=X.index)
