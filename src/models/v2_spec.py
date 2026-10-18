"""
Model v2 spec (2026-10-18, chosen by notebooks/04d's pre-set rule: "V2b").
Frozen by notebooks/04e_freeze_v2.py; evaluated on the pre-registered Dune
holdout (contracts/v2/PREREG.md). Do not edit after the freeze commit.
"""
from __future__ import annotations

import pandas as pd

from models.barrier_model import FEATURE_COLUMNS as V1_FEATURES

FLOW_FEATURES = [
    "curve_buy_share", "curve_max_trade_share", "curve_top_trader_share", "curve_top_trader_share_max",
    "minutes_since_graduation", "flow_buy_share_15m", "flow_max_trade_share_15m", "flow_top_trader_share_15m",
]
HOST_FEATURES = ["meta_host_group", "meta_is_launch_tool"]
V2_FEATURES = V1_FEATURES + FLOW_FEATURES + HOST_FEATURES
CATEGORICAL = ["chain_family", "network", "dex_venue", "dex_protocol", "meta_host_group"]
TRAIN_START = pd.Timestamp("2026-09-13", tz="UTC")   # flow stats complete from here
TRAIN_END = pd.Timestamp("2026-10-11", tz="UTC")     # exclusive
LGBM_PARAMS = {"random_state": 17, "n_estimators": 300, "learning_rate": 0.05, "verbose": -1}
TARGET = "r_up_360m"


def to_frame(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Model input with the TRAINING category sets (manifest), bool -> float.
    LaunchLab rows have no host by construction (training never had one)."""
    X = df.reindex(columns=V2_FEATURES).copy()
    X.loc[X["dex_venue"].astype(str) != "pumpdotfun", HOST_FEATURES] = None
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None), categories=categories.get(c, []))
    for c in ("is_postgrad_at_t", "meta_is_launch_tool"):
        X[c] = X[c].astype("float")
    return X
