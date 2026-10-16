"""
Barrier entry model (2026-10-08, see docs/label_spec.md "Barrier label").

A LightGBM CLASSIFIER predicting P(price hits 2x before -50% within N)
-- `y_up_{N}m` in lawrence.gold.entry_candidates -- instead of the
regressor on the fixed 15-minute return (entry_model.py). Scored in
notebooks/04b on the rule's realized return `r_up_{N}m` after costs.

Why a classifier: the realized return of the barrier rule is bounded
(+100% at the target, about -50% at the stop, a timeout in between), so
the money depends on how often the target is hit first -- a probability,
which is what a classifier estimates. Rows with a NULL label (censored)
are excluded from training and scoring, never treated as losses.

Features: entry_model's columns plus two Gold v2 columns the regressor
never used -- minutes since launch and whether the token had graduated
at T (both known at T).
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

from models.entry_model import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS

EXTRA_FEATURE_COLUMNS = ["minutes_since_launch", "is_postgrad_at_t"]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS
# FEATURE_COLUMNS is FROZEN by the 2026-10-03 pre-registered test
# (docs/decisions.md) -- new features go in a separate list, compared
# against it out of sample, never edited into it.

# 2026-10-16: on-chain flow features (gold_features.FLOW_FEATURE_KEYS).
FLOW_FEATURE_COLUMNS = [
    "flow_buy_share_15m",
    "flow_max_trade_share_15m",
    "flow_top_trader_share_15m",
    "curve_buy_share",
    "curve_max_trade_share",
    "curve_top_trader_share",
    "curve_top_trader_share_max",
    "minutes_since_graduation",
]
FLOW_MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + FLOW_FEATURE_COLUMNS


def label_columns(horizon_minutes: int) -> tuple[str, str]:
    """(binary label, realized return) column names for a horizon."""
    return f"y_up_{horizon_minutes}m", f"r_up_{horizon_minutes}m"


def prepare_barrier_features(
    candidates: pd.DataFrame, horizon_minutes: int, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, r) for rows whose barrier label isn't censored. Assumes
    entry_model.set_categorical_dtypes already ran on the full dataset."""
    y_col, r_col = label_columns(horizon_minutes)
    df = candidates[candidates[y_col].notna()]
    X = df[feature_columns or FEATURE_COLUMNS].copy()
    # bool -> float so LightGBM treats it as numeric with NaN = unknown
    X["is_postgrad_at_t"] = X["is_postgrad_at_t"].astype("float")
    return X, df[y_col].astype(int), df[r_col].astype(float)


def train_barrier_model(X_train: pd.DataFrame, y_train: pd.Series, **lgbm_params) -> LGBMClassifier:
    params = {"random_state": 17, "n_estimators": 300, "learning_rate": 0.05, "verbose": -1, **lgbm_params}
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def predict_up_probability(model: LGBMClassifier, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
