"""
Social features v2 candidates (2026-10-18, branch `social`): point-in-time
features at an assessment time T from pump.fun callouts and launch metadata.
Pure pandas/numpy (unit-tested); notebooks/05_social_features.py is the
Spark glue. NOT part of contract v1.

Sources (bronze, pulled 2026-10-17 by 00i):
- pumpfun_callouts: one row per callout with `created_at` (point-in-time).
  Engagement counts are as of the pull -> never used. `posthoc_max_multiplier`
  is FUTURE information -> never used.
- pumpfun_coin_uris: on-chain metadata URI host (set at creation).
- pumpfun_coins: launch-time IPFS metadata (socials). Whether it resolved is
  post-hoc; resolved-or-not is never a feature.

Truncation: /callout/coin returns at most 100 callouts RANKED BY post-hoc
performance, so for the ~8.5% of coins that hit 100 the kept set is biased.
Counts are therefore CAPPED at COUNT_CAP: below the cap a count is exact as
long as the coin's first COUNT_CAP callouts are all among the kept 100 --
checked empirically in notebook 05 (truncation simulation) before use.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

COUNT_CAP = 20
WINDOWS_MIN = (15, 60)
CALLOUT_FEATURES = [
    "co_n",              # callouts created <= T (capped)
    "co_n_15m",          # in (T-15m, T] (capped)
    "co_n_60m",
    "co_callers",        # distinct callers <= T (capped)
    "co_min_since_first",   # minutes since the first callout (NaN if none yet)
    "co_first_min_after_launch",  # launch -> first callout (NaN if none by T)
    "co_median_mcap_at_call",     # median market cap at the calls so far (NaN if none)
]
META_FEATURES = ["meta_host_group", "meta_is_launch_tool", "meta_launch_has_twitter",
                 "meta_launch_has_telegram", "meta_launch_has_website"]

# 2026-10-18, notebook 05 gates (pre-set, docs/decisions.md):
# - Check B FAILED (first-20 kept share at top-50% truncation: mean 0.855,
#   p10 0.55 vs >= 0.95 / >= 0.85) -> the CALLOUT family is dropped from v2.
# - Check C FAILED (ipfs.io coins: unresolved metadata hit 2x 29.4% vs
#   resolved 22.9%, 6.5 pts > 3) -> launch socials are dropped from v2.
# What survives into v2: the launch-time metadata host only.
V2_SOCIAL_FEATURES = ["meta_host_group", "meta_is_launch_tool"]

# Deployer / launch-tool hosts seen in the 2026-10-17 URI backfill. Frozen
# list; anything else with >= HOST_MIN_COINS in the TRAINING period becomes
# its own group, the rest -> other_custom.
LAUNCH_TOOL_HOSTS = frozenset({
    "metadata.j7tracker.io", "meta.uxento.io", "m.rapidlaunch.io", "usepaid.app",
    "api.vortexdeployer.com", "www.agencypad.fun", "ipfs.launchblitz.ai",
})
STANDARD_HOSTS = frozenset({"ipfs.io", "ipfs"})
HOST_MIN_COINS = 100


def callout_features(created_at: np.ndarray, user_ids: np.ndarray, mcap: np.ndarray,
                     t: pd.Timestamp, launch_time: pd.Timestamp) -> dict[str, Any]:
    """Features at T from ONE token's callouts. Arrays sorted by created_at
    (datetime64[ns, UTC] as int64 ns). Only callouts with created_at <= T."""
    tn = t.value
    k = int(np.searchsorted(created_at, tn, side="right"))
    out: dict[str, Any] = {"co_n": min(k, COUNT_CAP)}
    for w in WINDOWS_MIN:
        lo = int(np.searchsorted(created_at, tn - w * 60 * 1_000_000_000, side="right"))
        out[f"co_n_{w}m"] = min(k - lo, COUNT_CAP)
    if k == 0:
        out.update({"co_callers": 0, "co_min_since_first": np.nan,
                    "co_first_min_after_launch": np.nan, "co_median_mcap_at_call": np.nan})
        return out
    out["co_callers"] = min(len(set(user_ids[:k])), COUNT_CAP)
    out["co_min_since_first"] = (tn - created_at[0]) / 6e10
    out["co_first_min_after_launch"] = (created_at[0] - launch_time.value) / 6e10
    m = mcap[:k]
    m = m[~np.isnan(m)]
    out["co_median_mcap_at_call"] = float(np.median(m)) if len(m) else np.nan
    return out


def build_host_groups(hosts: pd.Series, created_at: pd.Series, train_end: pd.Timestamp,
                      min_coins: int = HOST_MIN_COINS) -> dict[str, str]:
    """host -> group, from coins CREATED before train_end only (no future
    hosts). Standard -> 'ipfs_standard'; known tools -> their host; other
    hosts with >= min_coins -> their host; Pinata gateways -> 'other_pinata'."""
    h = hosts[pd.to_datetime(created_at, utc=True) < train_end].dropna()
    counts = h.value_counts()
    groups = {}
    for host, n in counts.items():
        if host in STANDARD_HOSTS:
            groups[host] = "ipfs_standard"
        elif host in LAUNCH_TOOL_HOSTS or n >= min_coins:
            groups[host] = host
        elif "pinata" in host:
            groups[host] = "other_pinata"
    return groups


def host_group(host: str | None, groups: dict[str, str]) -> str | None:
    if host is None:
        return None
    if host in groups:
        return groups[host]
    return "other_pinata" if "pinata" in host else "other_custom"


def meta_features(host: str | None, groups: dict[str, str], launch_twitter: Any, launch_telegram: Any,
                  launch_website: Any, has_launch_metadata: bool) -> dict[str, Any]:
    """Launch-time metadata features. Socials are NaN (unknown) when the
    metadata didn't resolve -- never 0, and resolution itself is never a feature."""
    def flag(v):
        if not has_launch_metadata:
            return np.nan
        return float(isinstance(v, str) and v.strip() != "")
    return {
        "meta_host_group": host_group(host, groups),
        "meta_is_launch_tool": None if host is None else float(host in LAUNCH_TOOL_HOSTS),
        "meta_launch_has_twitter": flag(launch_twitter),
        "meta_launch_has_telegram": flag(launch_telegram),
        "meta_launch_has_website": flag(launch_website),
    }


def truncation_check(callouts: pd.DataFrame, keep_frac: float = 0.5, first_k: int = COUNT_CAP,
                     min_n: int = 40) -> pd.DataFrame:
    """Simulates the API's top-N-by-posthoc-performance truncation on coins we
    have IN FULL (fewer than 100 callouts, >= min_n): keep the top keep_frac by
    posthoc_max_multiplier, then check what share of each coin's first `first_k`
    callouts (by time) survived. ~1.0 means capped counts are safe for
    truncated coins; much lower means callout counts leak and must be dropped
    for those coins (or entirely)."""
    rows = []
    for mint, g in callouts.groupby("mint"):
        n = len(g)
        if n < min_n or n >= 100:
            continue
        keep = set(g.nlargest(int(n * keep_frac), "posthoc_max_multiplier")["callout_id"])
        first = g.nsmallest(min(first_k, n), "created_at")["callout_id"]
        rows.append({"mint": mint, "n": n, "first_k_kept_share": float(first.isin(keep).mean())})
    return pd.DataFrame(rows)
