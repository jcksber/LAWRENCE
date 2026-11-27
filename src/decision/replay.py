"""
Replay harness + parity check (spec §7): feed historical 1-minute bars through the Engine minute by minute,
as a live feed would, then compare the engine's trades with the backtest's.
"""
from __future__ import annotations

import heapq

import pandas as pd

from decision.engine import Engine


TAIL = pd.Timedelta(minutes=360)  # keep ticking 6h past the last bar so time-outs resolve


def replay(engine: Engine, tokens: dict[str, dict], *, after: pd.Timestamp | None = None,
           through: pd.Timestamp | None = None,
           resume_at: tuple[pd.Timestamp, str] | None = None) -> pd.DataFrame:
    """tokens: token_id -> {"bars": DataFrame (BAR_COLUMNS), "launch_time", "launch_venue"}.
    Calls the engine for every token EVERY MINUTE from its first bar close to its last bar close + 6h, in
    global time order (ties by token id) -- as a live runtime ticking each minute would. (2026-10-24 fix:
    calling only at the token's own bar closes missed candidate times with no bar closing at that minute.)
    Optional UTC minute boundaries select ticks in (after, through]. resume_at retries the interrupted token tick inclusively. Bars remain available as lookback."""
    if after is not None and resume_at is not None:
        raise ValueError("Choose a minute boundary or an inclusive token cursor")
    for boundary in (after, through, resume_at[0] if resume_at else None):
        if boundary is not None and (boundary.tzinfo is None or boundary != boundary.floor("min")):
            raise ValueError("Replay boundaries must be timezone-aware whole minutes")
    heap = []
    for tok, d in tokens.items():
        b = d["bars"].sort_values("bar_start").reset_index(drop=True)
        d["bars"] = b
        closes = b["bar_start"] + pd.Timedelta(minutes=1)
        first, last = closes.min(), closes.max() + TAIL
        if after is not None:
            first = max(first, after + pd.Timedelta(minutes=1))
        if resume_at is not None:
            first = max(first, resume_at[0])
        if through is not None:
            last = min(last, through)
        for t in pd.date_range(first, last, freq="1min"):
            heap.append((t, tok))
    heapq.heapify(heap)
    while heap:
        now, tok = heapq.heappop(heap)
        if resume_at is not None and (now, tok) < resume_at:
            continue
        d = tokens[tok]
        engine.on_bar_close(tok, d["bars"], now.to_pydatetime(), launch_time=d["launch_time"],
                            launch_venue=d["launch_venue"], observed_from_launch=d.get("observed_from_launch", True))
    return engine.trades()


def parity(engine_trades: pd.DataFrame, backtest_trades: pd.DataFrame, tol: float = 1e-9) -> dict:
    """Both frames: token_id, entry time (engine: entry_time; backtest: assessment_time), gross return
    (engine: r_gross; backtest: r_up_360m). PASS = same set of (token, entry time) and returns within tol."""
    e = engine_trades.rename(columns={"entry_time": "t", "r_gross": "r"})[["token_id", "t", "r"]].copy()
    b = backtest_trades.rename(columns={"assessment_time": "t", "r_up_360m": "r"})[["token_id", "t", "r"]].copy()
    for f in (e, b):
        f["t"] = pd.to_datetime(f["t"], utc=True)
    m = e.merge(b, on=["token_id", "t"], how="outer", suffixes=("_engine", "_backtest"), indicator=True)
    both = m[m["_merge"] == "both"]
    max_diff = float((both["r_engine"] - both["r_backtest"]).abs().max()) if len(both) else 0.0
    out = {"matched": int(len(both)), "only_engine": int((m["_merge"] == "left_only").sum()),
           "only_backtest": int((m["_merge"] == "right_only").sum()), "max_abs_return_diff": max_diff}
    out["status"] = "PASS" if out["only_engine"] == 0 and out["only_backtest"] == 0 and max_diff <= tol else "FAIL"
    return out
