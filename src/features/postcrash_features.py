"""
v4 post-crash track: entry candidates + labels for graduated pump.fun tokens 24h-14d after graduation,
from HOURLY PumpSwap bars (bronze.postcrash_bars_1h) plus each pool's first-24h high/low from the 1m bars
(bronze.postgrad_bars_1m). Pure pandas/numpy, one pool at a time (applyInPandas in notebook 02c).

Assessment time T = an hourly bar's CLOSE (bar_start + 1h); every feature uses bars that closed by T.
Label = the same barrier as v1/v2 (2x before -50%) over a longer horizon H, from hourly bars:
a bar touching both counts as the stop, a gapped stop fills at the close, time-out = last close by T+H.
Censored (row dropped) if T+H passes the data end or the pool's 14-day bar window.
`r_pc_next` = the same label if filled at the next hourly bar's close (delayed-entry check).
Activity filter at T (thin pools can't be traded): >= MIN_TRADES_24H trades and >= MIN_VOL_24H USD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H_HOURS = 72
MIN_TRADES_24H = 20
MIN_VOL_24H = 2_000.0
MIN_AGE_H, MAX_AGE_H = 24, 14 * 24
UP, DOWN = 2.0, 0.5

FEATURES = ["age_h", "dd_from_ath", "rebound_from_low", "hours_since_ath", "hours_since_low", "ret_6h", "ret_24h",
            "vol_24h", "trades_24h", "traders_24h", "active_hours_24h", "vol_6h_share", "ath_in_first_24h"]


def _sec(ts) -> float:
    return pd.Timestamp(ts).tz_localize("UTC").timestamp() if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).timestamp()


def pool_rows(bars: pd.DataFrame, grad_at: pd.Timestamp, ath0: float, low0: float, data_end: pd.Timestamp,
              h_hours: int = H_HOURS) -> pd.DataFrame:
    """bars: one pool's hourly bars (bar_start, high, low, close, volume_usd, n_trades, n_traders), USD.
    ath0 / low0: max high / min low over the first 24h after graduation (NaN if unknown).
    Times are handled as epoch seconds (UTC)."""
    b = bars.dropna(subset=["close"]).sort_values("bar_start").reset_index(drop=True)
    if b.empty:
        return pd.DataFrame()
    t = np.array([_sec(x) for x in b["bar_start"]]) + 3600.0  # close times
    hi, lo, cl = b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
    vol = b["volume_usd"].fillna(0).to_numpy(float)
    ntr, ntd = b["n_trades"].fillna(0).to_numpy(float), b["n_traders"].fillna(0).to_numpy(float)
    H = h_hours * 3600.0
    g = _sec(grad_at)
    end = min(_sec(data_end), g + MAX_AGE_H * 3600.0)
    out = []
    for i in range(len(b)):
        T = t[i]
        age_h = (T - g) / 3600.0
        if age_h < MIN_AGE_H or T + H > end:
            continue
        w24 = (t > T - 86400) & (t <= T)
        w6 = (t > T - 21600) & (t <= T)
        trades24, vol24 = ntr[w24].sum(), vol[w24].sum()
        if trades24 < MIN_TRADES_24H or vol24 < MIN_VOL_24H:
            continue
        j_ath = int(np.argmax(hi[: i + 1]))
        hourly_ath = hi[j_ath]
        ath_first = (not np.isnan(ath0)) and ath0 >= hourly_ath
        ath = ath0 if ath_first else hourly_ath
        since = 0 if ath_first else j_ath  # low since the ATH (hourly bars; plus the first-24h low if the ATH was there)
        j_low = since + int(np.argmin(lo[since: i + 1]))
        low_since = min(lo[j_low], low0) if (ath_first and not np.isnan(low0)) else lo[j_low]
        prev6 = np.flatnonzero(t <= T - 21600)
        prev24 = np.flatnonzero(t <= T - 86400)
        row = {
            "assessment_time": pd.Timestamp(T, unit="s", tz="UTC"), "price_at_t": cl[i], "age_h": age_h,
            "dd_from_ath": cl[i] / ath - 1, "rebound_from_low": cl[i] / low_since - 1 if low_since > 0 else np.nan,
            "hours_since_ath": np.nan if ath_first else (T - t[j_ath]) / 3600.0,
            "hours_since_low": (T - t[j_low]) / 3600.0,
            "ret_6h": cl[i] / cl[prev6[-1]] - 1 if prev6.size else np.nan,
            "ret_24h": cl[i] / cl[prev24[-1]] - 1 if prev24.size else np.nan,
            "vol_24h": vol24, "trades_24h": trades24, "traders_24h": ntd[w24].sum(),
            "active_hours_24h": float(w24.sum()), "vol_6h_share": vol[w6].sum() / vol24 if vol24 > 0 else np.nan,
            "ath_in_first_24h": float(ath_first),
        }
        row.update(_barrier(t, hi, lo, cl, i, H))
        # delayed-entry check: fill at the NEXT bar's close (if it closes within 2h), same barrier
        nxt = i + 1 if (i + 1 < len(t) and t[i + 1] <= T + 7200 and t[i + 1] + H <= end) else None
        row["r_pc_next"] = np.nan if nxt is None else _barrier(t, hi, lo, cl, nxt, H)["r_pc"]
        out.append(row)
    return pd.DataFrame(out)


def _barrier(t, hi, lo, cl, i, H) -> dict:
    p, T = cl[i], t[i]
    for k in range(i + 1, len(t)):
        if t[k] > T + H:
            break
        if lo[k] <= DOWN * p:
            r = (cl[k] / p - 1) if hi[k] < DOWN * p else DOWN - 1
            return {"r_pc": float(r), "outcome": "down", "t_pc_h": (t[k] - T) / 3600.0}
        if hi[k] >= UP * p:
            return {"r_pc": UP - 1, "outcome": "up", "t_pc_h": (t[k] - T) / 3600.0}
    last = np.flatnonzero(t <= T + H)[-1]
    return {"r_pc": float(cl[last] / p - 1), "outcome": "timeout", "t_pc_h": H / 3600.0}
