"""Tests for src/models/barrier_model.py (needs LightGBM; skipped otherwise)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from models.barrier_model import (  # noqa: E402
    FEATURE_COLUMNS,
    predict_up_probability,
    prepare_barrier_features,
    train_barrier_model,
)
from models.entry_model import set_categorical_dtypes  # noqa: E402


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS})
    for c in ("chain_family", "network", "dex_venue", "dex_protocol"):
        df[c] = "x"
    df["is_postgrad_at_t"] = rng.random(n) > 0.5
    df["y_up_60m"] = (df["trailing_price_change_pct"] > 0.5).astype(float)
    df.loc[:9, "y_up_60m"] = np.nan  # censored
    df["r_up_60m"] = np.where(df["y_up_60m"] == 1, 1.0, -0.2)
    return set_categorical_dtypes(df)


def test_censored_rows_are_dropped_not_counted_as_losses():
    X, y, r = prepare_barrier_features(_frame(), 60)
    assert len(X) == 390 and len(y) == len(r) == 390
    assert set(y.unique()) <= {0, 1}


def test_classifier_learns_a_simple_signal():
    X, y, _ = prepare_barrier_features(_frame(), 60)
    p = predict_up_probability(train_barrier_model(X, y, n_estimators=50), X)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3


def test_flow_list_does_not_change_frozen_features():
    from models.barrier_model import FLOW_FEATURE_COLUMNS, FLOW_MODEL_FEATURE_COLUMNS
    from features.gold_features import FLOW_FEATURE_KEYS

    assert FLOW_MODEL_FEATURE_COLUMNS[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert not set(FLOW_FEATURE_COLUMNS) & set(FEATURE_COLUMNS)
    assert tuple(FLOW_FEATURE_COLUMNS) == FLOW_FEATURE_KEYS


def test_contract_v1_features_are_frozen_list():
    from serving.contract_v1 import FEATURE_COLUMNS as CONTRACT_COLUMNS

    assert CONTRACT_COLUMNS == FEATURE_COLUMNS
