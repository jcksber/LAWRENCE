"""
Frozen evaluator for the pre-registered Dune holdout 2026-10-11 -> 2026-11-01
(contracts/v2/PREREG.md). Reuses the frozen v1 pieces (shadow_eval_v1) without
editing them; the only difference is alpha: two models are tested on one
window, so each uses alpha = 0.05 (Bonferroni of 0.10). Do not edit after the
v2 freeze commit.
"""
from __future__ import annotations

import pandas as pd

from backtest.shadow_eval_v1 import MIN_TRADES, _net, block_bootstrap_lower_bound, frozen_trades, top_k_per_day
from serving.contract_v1 import SLIPPAGE_BPS_BASE, SLIPPAGE_BPS_STRESS

WINDOW_START = pd.Timestamp("2026-10-11", tz="UTC")
WINDOW_END = pd.Timestamp("2026-11-01", tz="UTC")   # exclusive; 21 days
ALPHA = 0.05
MAX_EXCLUDED_DAY_SHARE = 0.20
MODELS = ("v1", "v2")


def evaluate_model(cands: pd.DataFrame, cutoff: float, excluded_days: list | None = None) -> dict:
    """cands: one row per holdout candidate (post-grad, rule 7, labelled or
    not) with token_id, assessment_time (UTC), score, dex_venue, r_up_360m,
    t_up_360m. Status: PASS / FAIL / INCONCLUSIVE / INVALID."""
    excluded = {pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d)
                for d in (excluded_days or [])}
    days = (WINDOW_END - WINDOW_START).days
    out = {"window": f"{WINDOW_START:%Y-%m-%d}..{WINDOW_END:%Y-%m-%d}", "excluded_days": len(excluded),
           "alpha": ALPHA}
    if len(excluded) / days > MAX_EXCLUDED_DAY_SHARE:
        return {**out, "status": "INVALID"}
    c = cands[(cands["assessment_time"] >= WINDOW_START) & (cands["assessment_time"] < WINDOW_END)]
    c = c[~c["assessment_time"].dt.floor("D").isin(excluded)]
    t = frozen_trades(c, cutoff)
    out.update({"trades": int(len(t)), "tokens": int(t["token_id"].nunique())})
    if len(t) < MIN_TRADES:
        return {**out, "status": "INCONCLUSIVE"}
    nb, ns = _net(t, SLIPPAGE_BPS_BASE), _net(t, SLIPPAGE_BPS_STRESS)
    lb = block_bootstrap_lower_bound(t, nb, alpha=ALPHA)
    per_tok = nb.groupby(t["token_id"].values).sum()
    k = top_k_per_day(c, cutoff)
    out.update({
        "mean_net_base": float(nb.mean()), "lower_bound_95_base": lb, "mean_net_stress": float(ns.mean()),
        "median_net_base": float(nb.median()), "hit_2x_share": float((t["r_up_360m"] >= 0.999).mean()),
        "top_token_share_of_pnl": float(per_tok.max() / per_tok.sum()) if per_tok.sum() > 0 else None,
        "exploratory_top2_per_day": {"trades": int(len(k)),
                                     "mean_net_base": float(_net(k, SLIPPAGE_BPS_BASE).mean()) if len(k) else None},
        "status": "PASS" if (lb > 0 and ns.mean() > 0) else "FAIL",
    })
    return out
