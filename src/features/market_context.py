"""
v5 market-context features (2026-11-22): the state of the whole launchpad market at assessment time T.
Point-in-time by construction: a row at T uses only COMPLETED hours before floor(T to the hour), so every
input ended before T. Join key: `ctx_hour` = floor(T, hour).

Inputs (built in notebook 02d from Silver):
- events: one row per token with `launch_time` (first curve bar), `grad_time` (first post-graduation bar, NaT
  if none) and `hit2x_6h` (did it reach 2x its graduation price within 6h of graduating; known at grad+6h).
- hourly: post-graduation market activity per hour start: `hour`, `volume_usd`, `n_trades`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["mkt_launches_1h", "mkt_launches_6h", "mkt_grads_1h", "mkt_grads_6h", "mkt_grads_24h",
            "mkt_grad_rate_6h", "mkt_recent_2x_share_24h", "mkt_recent_resolved_24h",
            "mkt_postgrad_vol_1h", "mkt_postgrad_vol_6h", "mkt_postgrad_trades_1h"]


def _count_between(sorted_ts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """# of timestamps in [lo, hi)."""
    return np.searchsorted(sorted_ts, hi, side="left") - np.searchsorted(sorted_ts, lo, side="left")


def market_context(events: pd.DataFrame, hourly: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    H = hours.asi8                                   # ns, hour starts; features use [H - w, H)
    hr = np.int64(3_600_000_000_000)
    launches = np.sort(events["launch_time"].dropna().astype("datetime64[ns, UTC]").astype("int64").to_numpy())
    g = events.dropna(subset=["grad_time"])
    grads = np.sort(g["grad_time"].astype("datetime64[ns, UTC]").astype("int64").to_numpy())
    # outcome of a graduate is known at grad_time + 6h
    res = g.assign(_known=g["grad_time"] + pd.Timedelta(hours=6)).sort_values("_known")
    known = res["_known"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    hit_cum = np.concatenate([[0], np.cumsum(res["hit2x_6h"].astype(float).to_numpy())])

    out = pd.DataFrame({"ctx_hour": hours})
    out["mkt_launches_1h"] = _count_between(launches, H - hr, H)
    out["mkt_launches_6h"] = _count_between(launches, H - 6 * hr, H)
    out["mkt_grads_1h"] = _count_between(grads, H - hr, H)
    out["mkt_grads_6h"] = _count_between(grads, H - 6 * hr, H)
    out["mkt_grads_24h"] = _count_between(grads, H - 24 * hr, H)
    out["mkt_grad_rate_6h"] = np.where(out["mkt_launches_6h"] > 0, out["mkt_grads_6h"] / out["mkt_launches_6h"].clip(lower=1), np.nan)
    lo = np.searchsorted(known, H - 24 * hr, side="left")
    hi = np.searchsorted(known, H, side="left")
    n = hi - lo
    out["mkt_recent_resolved_24h"] = n
    out["mkt_recent_2x_share_24h"] = np.where(n > 0, (hit_cum[hi] - hit_cum[lo]) / np.maximum(n, 1), np.nan)

    hv = hourly.set_index(pd.to_datetime(hourly["hour"], utc=True))[["volume_usd", "n_trades"]].sort_index()
    full = hv.reindex(pd.date_range(min(hours.min() - pd.Timedelta(hours=7), hv.index.min()), hours.max(),
                                    freq="h", tz="UTC"), fill_value=0.0)
    prev1 = full.shift(1)                             # value of the hour ending at H
    prev6 = full.rolling(6, min_periods=1).sum().shift(1)
    out["mkt_postgrad_vol_1h"] = prev1["volume_usd"].reindex(hours).to_numpy()
    out["mkt_postgrad_vol_6h"] = prev6["volume_usd"].reindex(hours).to_numpy()
    out["mkt_postgrad_trades_1h"] = prev1["n_trades"].reindex(hours).to_numpy()
    return out
