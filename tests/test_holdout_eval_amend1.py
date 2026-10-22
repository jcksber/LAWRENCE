import numpy as np
import pandas as pd

from backtest import holdout_eval_2026_10 as H
from backtest import holdout_eval_2026_10_amend1 as A


def _cands(n=400, seed=3):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-10-11", tz="UTC")
    return pd.DataFrame({
        "token_id": [f"t{i % 150}" for i in range(n)],
        "assessment_time": [t0 + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 20 * 1440, n)],
        "score": rng.normal(0.05, 0.03, n), "dex_venue": "pumpdotfun",
        "r_up_360m": rng.choice([1.0, -0.5, 0.02], n, p=[0.3, 0.3, 0.4]), "t_up_360m": 60.0})


def test_alpha_is_family_over_three():
    assert np.isclose(A.ALPHA, 0.10 / 3) and A.MODELS == ("v1", "v2", "v2-R1")


def test_identical_to_frozen_evaluator_at_alpha_005():
    c = _cands()
    a, h = A.evaluate_model(c, 0.04, alpha=0.05), H.evaluate_model(c, 0.04)
    assert a["status"] == h["status"] and a["trades"] == h["trades"]
    assert np.isclose(a["lower_bound_base"], h["lower_bound_95_base"]) and np.isclose(a["mean_net_base"], h["mean_net_base"])


def test_stricter_alpha_gives_lower_bound():
    c = _cands()
    assert A.evaluate_model(c, 0.04)["lower_bound_base"] <= A.evaluate_model(c, 0.04, alpha=0.05)["lower_bound_base"]
