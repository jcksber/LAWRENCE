import pandas as pd

from backtest.baselines import (
    BASELINE_B_MIN_BUY_SELL_RATIO,
    BASELINE_B_MIN_TRAILING_VOLUME_USD,
    baseline_a_buy_everything,
    baseline_b_volume_momentum_filter,
)


def _candidates(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_baseline_a_enters_every_row_regardless_of_features():
    df = _candidates(
        [
            {"trailing_volume_usd": 0.0, "trailing_buy_sell_ratio": 0.0},
            {"trailing_volume_usd": 1_000_000.0, "trailing_buy_sell_ratio": 5.0},
        ]
    )
    result = baseline_a_buy_everything(df)
    assert result.tolist() == [True, True]
    assert result.index.equals(df.index)


def test_baseline_b_requires_both_volume_and_ratio_thresholds():
    df = _candidates(
        [
            # neither threshold met
            {
                "trailing_volume_usd": BASELINE_B_MIN_TRAILING_VOLUME_USD - 1,
                "trailing_buy_sell_ratio": BASELINE_B_MIN_BUY_SELL_RATIO - 0.1,
            },
            # volume met, ratio not
            {
                "trailing_volume_usd": BASELINE_B_MIN_TRAILING_VOLUME_USD,
                "trailing_buy_sell_ratio": BASELINE_B_MIN_BUY_SELL_RATIO - 0.1,
            },
            # ratio met, volume not
            {
                "trailing_volume_usd": BASELINE_B_MIN_TRAILING_VOLUME_USD - 1,
                "trailing_buy_sell_ratio": BASELINE_B_MIN_BUY_SELL_RATIO,
            },
            # both met (at the boundary -- inclusive)
            {
                "trailing_volume_usd": BASELINE_B_MIN_TRAILING_VOLUME_USD,
                "trailing_buy_sell_ratio": BASELINE_B_MIN_BUY_SELL_RATIO,
            },
        ]
    )
    result = baseline_b_volume_momentum_filter(df)
    assert result.tolist() == [False, False, False, True]


def test_baseline_b_index_aligns_with_a_non_default_input_index():
    df = _candidates(
        [
            {
                "trailing_volume_usd": BASELINE_B_MIN_TRAILING_VOLUME_USD,
                "trailing_buy_sell_ratio": BASELINE_B_MIN_BUY_SELL_RATIO,
            },
        ]
    )
    df.index = [42]
    result = baseline_b_volume_momentum_filter(df)
    assert list(result.index) == [42]
