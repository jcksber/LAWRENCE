"""Frozen v1 shadow evaluator (CONTRACT.md §11)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backtest.shadow_eval_v1 import (
    MIN_TRADES,
    block_bootstrap_lower_bound,
    evaluate,
    frozen_trades,
    non_overlapping,
    top_k_per_day,
)

S = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _cands(n_tok=300, per_day=1, r=0.5, days=21, score=1.0, venue="raydium_launchlab", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tok):
        d = i % days
        rr = r if np.isscalar(r) else r[i]
        rows.append({"token_id": f"t{i}", "assessment_time": pd.Timestamp(S) + pd.Timedelta(days=d, minutes=int(rng.integers(0, 600))),
                     "score": score, "dex_venue": venue, "r_up_360m": rr, "t_up_360m": 30.0})
    return pd.DataFrame(rows)


def test_one_position_per_token():
    t0 = pd.Timestamp(S)
    df = pd.DataFrame({"token_id": ["a", "a", "a"], "assessment_time": [t0, t0 + pd.Timedelta(minutes=10), t0 + pd.Timedelta(minutes=40)],
                       "t_up_360m": [30.0, 30.0, 30.0]})
    assert list(non_overlapping(df).index) == [0, 2]


def test_cutoff_and_unlabelled_rows():
    df = _cands(10)
    df.loc[0, "score"] = 0.0
    df.loc[1, "r_up_360m"] = np.nan
    assert len(frozen_trades(df, 0.5)) == 8


def test_clear_edge_passes_and_clear_loss_fails():
    win = evaluate(_cands(300, r=0.5), 0.5, S)
    assert win["status"] == "PASS" and win["trades"] == 300 and win["lower_bound_90_base"] > 0
    lose = evaluate(_cands(300, r=-0.3), 0.5, S)
    assert lose["status"] == "FAIL"


def test_positive_mean_alone_does_not_pass():
    # mean slightly positive after costs but noisy: lower bound < 0 -> FAIL
    # 54 of 150 hit 2x (+100%), the rest stop out (-50%): mean gross +4%,
    # net of LaunchLab's 1.5% round trip +2.5%, but SE ~6% -> bound < 0
    r = [1.0 if i % 25 < 9 else -0.5 for i in range(150)]
    res = evaluate(_cands(150, r=r), 0.5, S)
    assert res["mean_net_base"] > 0 and res["lower_bound_90_base"] < 0 and res["status"] == "FAIL"


def test_floor_extend_then_inconclusive_and_invalid():
    few = _cands(MIN_TRADES - 1, r=0.5)
    assert evaluate(few, 0.5, S)["status"] == "EXTEND"
    assert evaluate(few, 0.5, S, extended=True)["status"] == "INCONCLUSIVE"
    bad = [S + timedelta(days=i) for i in range(5)]  # 5/21 > 20%
    assert evaluate(_cands(300), 0.5, S, excluded_days=bad)["status"] == "INVALID"


def test_rows_outside_window_ignored():
    df = _cands(300, r=0.5)
    df["assessment_time"] -= pd.Timedelta(days=30)
    assert evaluate(df, 0.5, S)["status"] == "EXTEND"


def test_block_bootstrap_deterministic_and_top_k():
    df = _cands(200, r=0.2)
    net = df["r_up_360m"] - 0.01
    a = block_bootstrap_lower_bound(df, net, n_boot=500)
    assert a == block_bootstrap_lower_bound(df, net, n_boot=500)
    assert len(top_k_per_day(_cands(210, score=1.0), 0.5, k=2)) == 21 * 2
