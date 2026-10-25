"""
Replay harness + parity check (spec §7): feed historical 1-minute bars through the Engine minute by minute,
as a live feed would, then compare the engine's trades with the backtest's.
"""
from __future__ import annotations

import heapq

import pandas as pd

from decision.engine import Engine


def replay(engine: Engine, tokens: dict[str, dict]) -> pd.DataFrame:
    """tokens: token_id -> {"bars": DataFrame (BAR_COLUMNS), "launch_time", "launch_venue"}.
    Calls engine.on_bar_close at every bar close time, in global time order (ties by token id)."""
    heap = []
    for tok, d in tokens.items():
        b = d["bars"].sort_values("bar_start").reset_index(drop=True)
        d["bars"] = b
        for t in b["bar_start"] + pd.Timedelta(minutes=1):
            heap.append((t, tok))
    heapq.heapify(heap)
    while heap:
        now, tok = heapq.heappop(heap)
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
