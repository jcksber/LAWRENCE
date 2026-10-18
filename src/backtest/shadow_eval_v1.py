"""
Frozen evaluator for the v1 shadow test (CONTRACT.md §11, approved
2026-10-17). Written and committed BEFORE the shadow window starts; the
result is whatever this code says on the pipeline-recomputed labels.
Do not edit after SHADOW_START -- a changed evaluator is a new test.

Input: one row per shadow CANDIDATE (decision true or false) joined to the
pipeline's labels on (token_id, assessment_time), with columns
  token_id, assessment_time (UTC), score, dex_venue,
  r_up_360m (gross realized barrier return), t_up_360m (minutes to resolution).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from serving.contract_v1 import HORIZON_MINUTES, SLIPPAGE_BPS_BASE, SLIPPAGE_BPS_STRESS, round_trip_cost_v1

WINDOW_DAYS = 21
EXTENDED_DAYS = 35
MIN_TRADES = 100
MAX_EXCLUDED_DAY_SHARE = 0.20
ALPHA = 0.10            # 90% one-sided lower bound
N_BOOT = 10_000
SEED = 17
TOP_K = 2               # exploratory variant only


def _net(df: pd.DataFrame, slippage_bps: float) -> pd.Series:
    cost = [round_trip_cost_v1(v, True, slippage_bps) for v in df["dex_venue"]]
    return df["r_up_360m"].astype(float) - np.array(cost)


def non_overlapping(sig: pd.DataFrame) -> pd.DataFrame:
    """One open position per token: skip a token's signals until its previous
    trade resolved (assessment_time + t_up_360m). Same rule as 04b."""
    sig = sig.sort_values(["token_id", "assessment_time"])
    keep, open_until = [], {}
    for idx, tok, ts, mins in zip(sig.index, sig["token_id"], sig["assessment_time"],
                                  sig["t_up_360m"].fillna(HORIZON_MINUTES)):
        if tok in open_until and ts < open_until[tok]:
            continue
        keep.append(idx)
        open_until[tok] = ts + pd.Timedelta(minutes=float(mins))
    return sig.loc[keep]


def frozen_trades(cands: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """The v1 rule: score >= cutoff, one position per token, labelled rows only."""
    sig = cands[(cands["score"] >= cutoff) & cands["r_up_360m"].notna()]
    return non_overlapping(sig)


def top_k_per_day(cands: pd.DataFrame, cutoff: float, k: int = TOP_K) -> pd.DataFrame:
    """Exploratory: per UTC day, the k highest-scoring rows >= cutoff (one per
    token, earliest resolution-safe), never judged."""
    sig = cands[(cands["score"] >= cutoff) & cands["r_up_360m"].notna()].copy()
    sig["_day"] = sig["assessment_time"].dt.floor("D")
    sig = sig.sort_values(["_day", "score"], ascending=[True, False]).drop_duplicates(["_day", "token_id"])
    return sig.groupby("_day", group_keys=False).head(k).drop(columns="_day")


def block_bootstrap_lower_bound(trades: pd.DataFrame, net: pd.Series, *, alpha: float = ALPHA,
                                n_boot: int = N_BOOT, seed: int = SEED) -> float:
    """alpha-quantile of the mean net per trade, resampling token-days (all of a
    token's trades on one UTC day move together)."""
    block = trades["token_id"].astype(str) + "|" + trades["assessment_time"].dt.floor("D").astype(str)
    g = pd.DataFrame({"b": block.values, "net": net.values}).groupby("b")["net"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, alpha))


def evaluate(cands: pd.DataFrame, cutoff: float, shadow_start: datetime, *, extended: bool = False,
             excluded_days: list | None = None) -> dict:
    """Status per §11: PASS / FAIL / EXTEND (first look, < MIN_TRADES) /
    INCONCLUSIVE (extended, still short) / INVALID (> 20% excluded days)."""
    days = EXTENDED_DAYS if extended else WINDOW_DAYS
    start = pd.Timestamp(shadow_start).tz_convert("UTC") if pd.Timestamp(shadow_start).tzinfo else pd.Timestamp(shadow_start, tz="UTC")
    end = start + timedelta(days=days)
    excluded = {pd.Timestamp(d).normalize().tz_localize("UTC") if pd.Timestamp(d).tzinfo is None
                else pd.Timestamp(d).normalize() for d in (excluded_days or [])}
    out = {"window_start": start.isoformat(), "window_end": end.isoformat(), "window_days": days,
           "excluded_days": len(excluded), "excluded_day_share": len(excluded) / days}
    if out["excluded_day_share"] > MAX_EXCLUDED_DAY_SHARE:
        return {**out, "status": "INVALID"}
    c = cands[(cands["assessment_time"] >= start) & (cands["assessment_time"] < end)]
    c = c[~c["assessment_time"].dt.floor("D").isin(excluded)]
    t = frozen_trades(c, cutoff)
    out["trades"] = int(len(t))
    out["tokens"] = int(t["token_id"].nunique())
    if len(t) < MIN_TRADES:
        return {**out, "status": "INCONCLUSIVE" if extended else "EXTEND"}
    nb, ns = _net(t, SLIPPAGE_BPS_BASE), _net(t, SLIPPAGE_BPS_STRESS)
    lb = block_bootstrap_lower_bound(t, nb)
    per_tok = nb.groupby(t["token_id"].values).sum()
    out.update({
        "mean_net_base": float(nb.mean()), "lower_bound_90_base": lb, "mean_net_stress": float(ns.mean()),
        "median_net_base": float(nb.median()), "hit_2x_share": float((t["r_up_360m"] >= 0.999).mean()),
        "top_token_share_of_pnl": float(per_tok.max() / per_tok.sum()) if per_tok.sum() > 0 else None,
        "status": "PASS" if (lb > 0 and ns.mean() > 0) else "FAIL",
    })
    k = top_k_per_day(c, cutoff)
    out["exploratory_top2_per_day"] = {"trades": int(len(k)),
                                       "mean_net_base": float(_net(k, SLIPPAGE_BPS_BASE).mean()) if len(k) else None}
    return out
