"""
Amendment 1 to the pre-registered Dune holdout 2026-10-11 -> 2026-11-01
(contracts/v2/PREREG_AMENDMENT_1.md), written 2026-10-22 BEFORE any holdout data was pulled.

Adds a third test, "v2-R1": the SAME frozen v2 model, with a stricter cutoff (98th percentile of its
predicted EV on the training rows, from notebooks/04i). Three tests on one window -> each uses
alpha = 0.10 / 3 (Bonferroni), which also TIGHTENS v1 and v2 from 0.05. Every other rule is identical
to holdout_eval_2026_10.evaluate_model (that frozen file is not edited; this one takes alpha as an argument).
"""
from __future__ import annotations

import pandas as pd

from backtest.holdout_eval_2026_10 import MAX_EXCLUDED_DAY_SHARE, WINDOW_END, WINDOW_START
from backtest.shadow_eval_v1 import MIN_TRADES, _net, block_bootstrap_lower_bound, frozen_trades, top_k_per_day
from serving.contract_v1 import SLIPPAGE_BPS_BASE, SLIPPAGE_BPS_STRESS

FAMILY_ALPHA = 0.10
MODELS = ("v1", "v2", "v2-R1")
ALPHA = FAMILY_ALPHA / len(MODELS)  # 0.0333...


def evaluate_model(cands: pd.DataFrame, cutoff: float, excluded_days: list | None = None,
                   alpha: float = ALPHA) -> dict:
    """Same as holdout_eval_2026_10.evaluate_model, with alpha as a parameter."""
    excluded = {pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d)
                for d in (excluded_days or [])}
    days = (WINDOW_END - WINDOW_START).days
    out = {"window": f"{WINDOW_START:%Y-%m-%d}..{WINDOW_END:%Y-%m-%d}", "excluded_days": len(excluded),
           "alpha": alpha}
    if len(excluded) / days > MAX_EXCLUDED_DAY_SHARE:
        return {**out, "status": "INVALID"}
    c = cands[(cands["assessment_time"] >= WINDOW_START) & (cands["assessment_time"] < WINDOW_END)]
    c = c[~c["assessment_time"].dt.floor("D").isin(excluded)]
    t = frozen_trades(c, cutoff)
    out.update({"trades": int(len(t)), "tokens": int(t["token_id"].nunique())})
    if len(t) < MIN_TRADES:
        return {**out, "status": "INCONCLUSIVE"}
    nb, ns = _net(t, SLIPPAGE_BPS_BASE), _net(t, SLIPPAGE_BPS_STRESS)
    lb = block_bootstrap_lower_bound(t, nb, alpha=alpha)
    per_tok = nb.groupby(t["token_id"].values).sum()
    k = top_k_per_day(c, cutoff)
    out.update({
        "mean_net_base": float(nb.mean()), "lower_bound_base": lb, "mean_net_stress": float(ns.mean()),
        "median_net_base": float(nb.median()), "hit_2x_share": float((t["r_up_360m"] >= 0.999).mean()),
        "top_token_share_of_pnl": float(per_tok.max() / per_tok.sum()) if per_tok.sum() > 0 else None,
        "exploratory_top2_per_day": {"trades": int(len(k)),
                                     "mean_net_base": float(_net(k, SLIPPAGE_BPS_BASE).mean()) if len(k) else None},
        "status": "PASS" if (lb > 0 and ns.mean() > 0) else "FAIL",
    })
    return out
