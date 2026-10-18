"""Frozen holdout evaluator (contracts/v2/PREREG.md) + v2 frame builder."""
import numpy as np
import pandas as pd

from backtest.holdout_eval_2026_10 import WINDOW_START, evaluate_model


def _cands(n, r, venue="raydium_launchlab", offset_days=0):
    return pd.DataFrame({
        "token_id": [f"t{i}" for i in range(n)],
        "assessment_time": [WINDOW_START + pd.Timedelta(days=offset_days + i % 21, minutes=i % 600) for i in range(n)],
        "score": 1.0, "dex_venue": venue, "r_up_360m": r, "t_up_360m": 30.0,
    })


def test_pass_fail_inconclusive_invalid():
    assert evaluate_model(_cands(300, 0.5), 0.5)["status"] == "PASS"
    assert evaluate_model(_cands(300, -0.3), 0.5)["status"] == "FAIL"
    assert evaluate_model(_cands(50, 0.5), 0.5)["status"] == "INCONCLUSIVE"
    bad = [WINDOW_START + pd.Timedelta(days=i) for i in range(5)]
    assert evaluate_model(_cands(300, 0.5), 0.5, bad)["status"] == "INVALID"


def test_window_bounds_and_stricter_alpha():
    assert evaluate_model(_cands(300, 0.5, offset_days=-30), 0.5)["status"] == "INCONCLUSIVE"
    r = [1.0 if i % 25 < 10 else -0.5 for i in range(300)]  # ~+8.5% net, noisy
    res = evaluate_model(_cands(300, r), 0.5)
    assert res["alpha"] == 0.05 and res["lower_bound_95_base"] < res["mean_net_base"]


def test_v2_frame_blanks_host_for_launchlab():
    from models.v2_spec import V2_FEATURES, to_frame

    df = pd.DataFrame([{c: 1.0 for c in V2_FEATURES}, {c: 1.0 for c in V2_FEATURES}])
    df["chain_family"], df["network"], df["dex_protocol"] = "solana", None, None
    df["dex_venue"] = ["pumpdotfun", "raydium_launchlab"]
    df["meta_host_group"] = ["ipfs_standard", "ipfs_standard"]
    X = to_frame(df, {"chain_family": ["solana"], "dex_venue": ["pumpdotfun", "raydium_launchlab"],
                      "meta_host_group": ["ipfs_standard"], "network": [], "dex_protocol": []})
    assert list(X.columns) == V2_FEATURES
    assert X["meta_host_group"].iloc[0] == "ipfs_standard" and pd.isna(X["meta_host_group"].iloc[1])
    assert np.isnan(X["meta_is_launch_tool"].iloc[1])
