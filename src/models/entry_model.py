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

v1 was numeric-only (only the trailing-activity columns Gold v1 has --
see gold_spec.md's feature table), deliberately leaving out
chain_family/network/dex_venue/dex_protocol -- real signal, but adding it
meant an encoding decision that first pass didn't need to make yet.

2026-09-18 (v2, see docs/decisions.md): v1 lost to Baseline B (the plain
volume/momentum filter) on ev_per_trade in every walk-forward fold, so
this is the first of two "optimize the model further" changes.
CATEGORICAL_FEATURE_COLUMNS are now included, as pandas `category` dtype
-- LightGBM's sklearn API auto-detects category-dtype columns as
categorical splits with no one-hot encoding needed. `set_categorical_
dtypes` MUST run on the full dataset before walk-forward splitting, not
per-fold -- see its own docstring for why a per-fold cast would silently
break category consistency across folds.

2026-09-17 (see docs/decisions.md, gold_features.py's RETURN_WINSORIZE_CAP):
the raw y_entry a real Gold run produced included 779 rows in the
hundreds-of-thousands-to-16.7-billion range -- artifacts of near-zero-
priced tokens, not real winners a model should try to learn from.
`prepare_features` winsorizes the target before it ever reaches
`train_entry_model` so those 779 rows don't dominate training the way
they dominated this session's first (unwinsorized) walk-forward run.
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from features.gold_features import CAPTURABLE_RETURN_CAP, winsorize_return

NUMERIC_FEATURE_COLUMNS = [
    "trailing_volume_usd",
    "trailing_trade_count",
    "trailing_buy_count",
    "trailing_sell_count",
    "trailing_buy_sell_ratio",
    "trailing_price_change_pct",
]
CATEGORICAL_FEATURE_COLUMNS = ["chain_family", "network", "dex_venue", "dex_protocol"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
TARGET_COLUMN = "y_entry"


def set_categorical_dtypes(candidates: pd.DataFrame) -> pd.DataFrame:
    """Casts CATEGORICAL_FEATURE_COLUMNS to pandas `category` dtype --
    how LightGBM's sklearn API tells a categorical column from a numeric
    one (see module docstring). MUST be called ONCE on the full dataset
    BEFORE walk-forward splitting, never on a fold's train/test subset
    separately: pandas derives a column's category set from whatever
    values are actually present when `.astype("category")` runs, so
    casting two different subsets separately gives each one its own,
    possibly different category set for the same column -- a category
    that exists in a later fold's test data but never appeared in any
    earlier fold's train data would then look like a totally different
    (missing/unseen) thing to LightGBM than it should. Casting on the
    full, unsplit dataset first means every fold's train/test slice
    shares one consistent category set by construction.
    Returns a copy; does not mutate `candidates` in place.
    """
    df = candidates.copy()
    for col in CATEGORICAL_FEATURE_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def prepare_features(
    candidates: pd.DataFrame, cap: float = CAPTURABLE_RETURN_CAP
) -> tuple[pd.DataFrame, pd.Series]:
    """Selects the model's input columns and target from a Gold
    entry_candidates DataFrame. Assumes `set_categorical_dtypes` already
    ran on the full dataset (see its docstring) -- this function does not
    do that casting itself, so per-fold category consistency stays the
    caller's responsibility, not something this function could
    accidentally violate by re-deriving categories from a subset.
    Deliberately does NOT impute trailing_price_change_pct's occasional
    None (see trailing_activity_features -- it's None when the trailing
    window's first trade itself had price_usd == 0, a real "we don't
    actually know" gap, not a value to guess at). LightGBM handles
    missing numeric features natively (it learns a default split
    direction for them), so leaving it as NaN is the correct choice here,
    not a gap to fill in.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    # 2026-10-02: target capped at CAPTURABLE_RETURN_CAP (10x), not 100x --
    # see gold_features.CAPTURABLE_RETURN_CAP.
    y = winsorize_return(candidates[TARGET_COLUMN].copy(), cap)
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
    The notebook's decision rule (2026-09-18: `predicted > round_trip_cost()`,
    see walk_forward.py -- v1's plain "predicted > 0" ignored trading costs
    entirely) lives in notebooks/04, not here; ranking by this score and
    taking the top-K is an easy variant to try later without changing this
    function."""
    return pd.Series(model.predict(X), index=X.index)
