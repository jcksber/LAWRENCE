"""
lightgbm/scikit-learn aren't installed in every environment this repo's
tests run in (see docs/decisions.md -- the device_bash sandbox used for
much of this session's verification has neither, no network to install
them either). Skip cleanly rather than error when they're missing, same
principle as test_label_point_in_time.py's existing scaffold-skip -- Jack
confirms these in his own terminal/Databricks, where requirements.txt's
scikit-learn/lightgbm are actually installed.
"""
import pandas as pd
import pytest

lightgbm = pytest.importorskip("lightgbm")

from models.entry_model import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_features,
    score_candidates,
    set_categorical_dtypes,
    train_entry_model,
)


def _synthetic_candidates(n: int = 200) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(17)
    trailing_volume_usd = rng.exponential(scale=5_000, size=n)
    trailing_buy_count = rng.integers(0, 50, size=n)
    trailing_sell_count = rng.integers(0, 50, size=n)
    # y_entry deliberately correlated with volume so a fitted model should
    # beat a constant-mean prediction -- not asserting a specific score,
    # just that the pipeline runs end-to-end and learns *something*.
    y_entry = 0.0001 * trailing_volume_usd + rng.normal(0, 0.05, size=n)
    return pd.DataFrame(
        {
            "trailing_volume_usd": trailing_volume_usd,
            "trailing_trade_count": trailing_buy_count + trailing_sell_count,
            "trailing_buy_count": trailing_buy_count,
            "trailing_sell_count": trailing_sell_count,
            "trailing_buy_sell_ratio": (trailing_buy_count + 1) / (trailing_sell_count + 1),
            # some real NaNs, matching trailing_activity_features' actual
            # behavior when a window's first trade had price_usd == 0.
            "trailing_price_change_pct": rng.choice(
                [rng.normal(), None], size=n, p=[0.9, 0.1]
            ),
            # 2026-09-18 (see docs/decisions.md): v2 feature set adds these
            # four categoricals -- synthetic data needs them too, or
            # prepare_features' FEATURE_COLUMNS selection raises a KeyError.
            "chain_family": rng.choice(["evm", "solana"], size=n),
            "network": rng.choice(["base", "ethereum", "solana"], size=n),
            "dex_venue": rng.choice(["uniswap", "raydium", "aerodrome"], size=n),
            "dex_protocol": rng.choice(["v2", "v3", "amm"], size=n),
            TARGET_COLUMN: y_entry,
        }
    )


def test_set_categorical_dtypes_casts_only_the_declared_categorical_columns():
    df = _synthetic_candidates(10)
    cast_df = set_categorical_dtypes(df)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        assert str(cast_df[col].dtype) == "category"
    for col in NUMERIC_FEATURE_COLUMNS:
        assert str(cast_df[col].dtype) != "category"


def test_set_categorical_dtypes_does_not_mutate_the_input_in_place():
    df = _synthetic_candidates(10)
    set_categorical_dtypes(df)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        assert str(df[col].dtype) != "category"


def test_set_categorical_dtypes_derives_consistent_categories_from_the_full_set():
    # The whole point of calling this once on the full dataset before
    # walk-forward splitting: a category present only in a later "fold"
    # must still be a known category, not silently unseen/missing, when
    # the cast happens on the full data up front.
    df = pd.DataFrame({"chain_family": ["evm", "evm", "solana"], "network": ["a", "a", "a"],
                        "dex_venue": ["x", "x", "x"], "dex_protocol": ["y", "y", "y"]})
    cast_df = set_categorical_dtypes(df)
    early_slice_categories = cast_df.iloc[:2]["chain_family"].cat.categories
    assert "solana" in early_slice_categories


def test_prepare_features_selects_only_the_declared_columns():
    df = set_categorical_dtypes(_synthetic_candidates(10))
    df["token_id"] = "not_a_feature"
    df["assessment_time"] = "not_a_feature_either"
    X, y = prepare_features(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(y) == len(df)


def test_prepare_features_keeps_real_nans_uninmputed():
    df = set_categorical_dtypes(_synthetic_candidates(50))
    X, _ = prepare_features(df)
    # The synthetic data deliberately includes some None -- prepare_features
    # must not silently fill them (see its own docstring: LightGBM handles
    # missing values natively).
    assert X["trailing_price_change_pct"].isna().any()


def test_prepare_features_keeps_categorical_dtype_for_lightgbm():
    df = set_categorical_dtypes(_synthetic_candidates(20))
    X, _ = prepare_features(df)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        assert str(X[col].dtype) == "category"


def test_prepare_features_winsorizes_the_target():
    df = set_categorical_dtypes(_synthetic_candidates(5))
    df.loc[0, TARGET_COLUMN] = 16_693_038_859.0  # a real observed extreme value
    _, y = prepare_features(df)
    assert y.iloc[0] == 100.0  # RETURN_WINSORIZE_CAP


def test_train_and_score_round_trip_runs_and_beats_a_constant_baseline():
    df = set_categorical_dtypes(_synthetic_candidates(300))
    X, y = prepare_features(df)
    X_train, y_train = X.iloc[:200], y.iloc[:200]
    X_test, y_test = X.iloc[200:], y.iloc[200:]

    model = train_entry_model(X_train, y_train, n_estimators=50)
    preds = score_candidates(model, X_test)

    assert list(preds.index) == list(X_test.index)

    model_mse = ((preds - y_test) ** 2).mean()
    constant_mse = ((y_train.mean() - y_test) ** 2).mean()
    # Not a strict requirement of every future model change, just a sanity
    # check that this synthetic, clearly-correlated setup actually trains
    # something better than predicting the training mean for everyone.
    assert model_mse < constant_mse


def test_train_entry_model_is_reproducible_with_the_same_random_state():
    df = set_categorical_dtypes(_synthetic_candidates(150))
    X, y = prepare_features(df)
    model_a = train_entry_model(X, y, n_estimators=20)
    model_b = train_entry_model(X, y, n_estimators=20)
    preds_a = score_candidates(model_a, X)
    preds_b = score_candidates(model_b, X)
    assert (preds_a == preds_b).all()
