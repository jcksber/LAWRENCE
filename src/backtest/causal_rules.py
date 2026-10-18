"""
Causal trade-selection rules (2026-10-18, v3 dev). 04f's R6/R7 ("top-k per
UTC day") ranked a candidate against the WHOLE day, including candidates that
appear later -- not tradeable live. This is the causal replacement:

rolling_top_k: a candidate at time t qualifies if its net score is >= the k-th
highest per-token net score seen in the 24h before t's HOUR (strictly earlier
data), and > 0. Qualifying candidates are taken in time order, at most k new
entries per UTC day, one open position per token (held until resolution).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from serving.contract_v1 import HORIZON_MINUTES


def hourly_thresholds(history: pd.DataFrame, hours: pd.DatetimeIndex, k: int,
                      lookback: pd.Timedelta = pd.Timedelta(hours=24)) -> pd.Series:
    """For each hour start h: the k-th largest per-token max `net_score` among
    rows with assessment_time in [h - lookback, h). NaN if fewer than k tokens."""
    h_ts = history["assessment_time"].to_numpy()
    out = {}
    for h in hours:
        m = (history["assessment_time"] >= h - lookback) & (history["assessment_time"] < h)
        per_tok = history.loc[m].groupby("token_id")["net_score"].max()
        out[h] = float(np.sort(per_tok.to_numpy())[-k]) if len(per_tok) >= k else np.nan
    return pd.Series(out)


def rolling_top_k(cands: pd.DataFrame, history: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """cands / history: token_id, assessment_time (UTC), net_score; cands also
    t_up_360m. `history` must include everything before the first candidate's
    hour that may be used (e.g. the end of the training period) plus cands."""
    c = cands.sort_values("assessment_time")
    hour = c["assessment_time"].dt.floor("h")
    thr = hourly_thresholds(history, pd.DatetimeIndex(hour.unique()), k)
    c = c.assign(_thr=hour.map(thr).to_numpy())
    keep, per_day, open_until = [], {}, {}
    for idx, tok, ts, ns, th, mins in zip(c.index, c["token_id"], c["assessment_time"], c["net_score"],
                                          c["_thr"], c["t_up_360m"].fillna(HORIZON_MINUTES)):
        if not (ns > 0) or np.isnan(th) or ns < th:
            continue
        if tok in open_until and ts < open_until[tok]:
            continue
        d = ts.floor("D")
        if per_day.get(d, 0) >= k:
            continue
        keep.append(idx)
        per_day[d] = per_day.get(d, 0) + 1
        open_until[tok] = ts + pd.Timedelta(minutes=float(mins))
    return c.loc[keep].drop(columns="_thr")
