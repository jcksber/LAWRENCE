"""Causal top-k rule: never uses candidates from the future."""
import numpy as np
import pandas as pd

from backtest.causal_rules import hourly_thresholds, rolling_top_k

T0 = pd.Timestamp("2026-09-20 00:00", tz="UTC")


def _rows(spec):
    return pd.DataFrame([{"token_id": t, "assessment_time": T0 + pd.Timedelta(hours=h), "net_score": s, "t_up_360m": 30.0}
                         for t, h, s in spec])


def test_threshold_uses_only_prior_24h():
    hist = _rows([("a", -30, 9.0), ("b", -5, 0.3), ("c", -4, 0.2), ("d", 1, 99.0)])
    thr = hourly_thresholds(hist, pd.DatetimeIndex([T0]), k=2)
    assert thr[T0] == 0.2  # 'a' is older than 24h, 'd' is in the future


def test_future_candidates_cannot_change_past_decisions():
    hist = _rows([(f"h{i}", -10, 0.1) for i in range(5)])
    cands = _rows([("x", 1, 0.5), ("y", 2, 0.05)])
    a = rolling_top_k(cands, pd.concat([hist, cands]), k=5)
    more = pd.concat([cands, _rows([("z", 20, 5.0)])])
    b = rolling_top_k(more, pd.concat([hist, more]), k=5)
    assert list(a["token_id"]) == ["x"] and list(b["token_id"])[:1] == ["x"]


def test_daily_cap_positive_only_and_one_position_per_token():
    hist = _rows([(f"h{i}", -10, 0.01) for i in range(5)])
    cands = _rows([(f"t{i}", 1 + i * 0.01, 0.5) for i in range(8)] + [("neg", 2, -0.1), ("t0", 1.1, 0.6)])
    out = rolling_top_k(cands, pd.concat([hist, cands]), k=5)
    assert len(out) == 5 and "neg" not in set(out["token_id"]) and out["token_id"].is_unique


def test_no_trades_without_history():
    cands = _rows([("x", 1, 0.5)])
    assert rolling_top_k(cands, cands, k=5).empty
