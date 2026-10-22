"""
Exit-rule simulator for the "recover the stake at 2x, then treat the rest as a new
position" ladder (Jack, 2026-10-22; first proposed 2026-10-02 in decisions.md).

Rules, per trade, from bars that CLOSE after the entry time T (point-in-time):
- E0 (current label): sell everything at 2x the entry price, or at the -50% stop, or at
  the price when the 6h clock runs out.
- E1 (ladder): at each doubling of the current reference price, sell HALF of the units
  still held (the first rung returns the whole stake), then reset: reference = that rung's
  price, stop = 50% of it, a fresh 6h clock. The rest is sold at the stop or at time-out.
  So after the first rung the worst case is +50% gross (the remaining half sells at the
  entry price); there is no cap on the upside.
- E2 (Jack, 2026-10-22): sell half at 2x, then the rest at 10x the ENTRY price, or at the stop
  (50% of the 2x level = the entry price, so that half breaks even), or at a fresh 6h time-out.
  Pass `final_mult=10.0` with `ladder=True`. `final_stop_mult` moves that stop: Jack chose 1.1x the entry
  (a 45% drop from the 2x level) so the second half also covers its fees -> +55% gross overall.

Conservative fills, same as gold_features.barrier_outcome:
- a bar touching both the target and the stop counts as the stop;
- after a rung fills inside a bar, the new stop is checked in that same bar (order unknown);
- one rung per bar at most;
- a stop gapped through (the whole bar below it) fills at the bar's close.
Costs: `side_cost` (fee + slippage, fraction of notional) is paid on the buy and on every
sale's notional. Net return = proceeds*(1 - side_cost)/p0 - (1 + side_cost).
"""
from __future__ import annotations

import numpy as np

HORIZON_S = 360 * 60


def simulate(ts: np.ndarray, hi: np.ndarray, lo: np.ndarray, close: np.ndarray, t0: float, p0: float,
             side_cost: float, *, ladder: bool, up: float = 2.0, down: float = 0.5,
             horizon_s: int = HORIZON_S, data_end: float | None = None,
             final_mult: float | None = None, final_stop_mult: float | None = None) -> dict:
    """ts = bar CLOSE times (epoch s, sorted). Returns {"net", "gross", "rungs", "exit", "truncated"}."""
    units, ref, deadline, proceeds, rungs = 1.0, p0, t0 + horizon_s, 0.0, 0
    last_px, exit_kind = p0, "timeout"
    i = int(np.searchsorted(ts, t0, side="right"))
    while i < len(ts):
        if ts[i] > deadline:
            break
        h, l, c = hi[i], lo[i], close[i]
        if not np.isnan(c):
            last_px = c
        stop = p0 * final_stop_mult if (rungs >= 1 and final_stop_mult is not None) else ref * down
        if l <= stop:  # stop (wins ties with the target)
            px = c if (h < stop and not np.isnan(c)) else stop
            proceeds += units * px
            return _out(proceeds, p0, side_cost, rungs, "stop", False)
        if rungs >= 1 and final_mult is not None:  # E2: one remaining sell-all target
            if h >= p0 * final_mult:
                proceeds += units * p0 * final_mult
                return _out(proceeds, p0, side_cost, rungs + 1, "target", False)
            i += 1
            continue
        if h >= ref * up:
            target = ref * up
            if not ladder:
                proceeds += units * target
                return _out(proceeds, p0, side_cost, rungs + 1, "target", False)
            proceeds += units / 2 * target
            units /= 2
            rungs += 1
            ref = target
            deadline = ts[i] + horizon_s
            stop = p0 * final_stop_mult if final_stop_mult is not None else ref * down
            if l <= stop:  # new stop in the same bar (conservative)
                px = c if (h < stop and not np.isnan(c)) else stop
                proceeds += units * px
                return _out(proceeds, p0, side_cost, rungs, "stop", False)
        i += 1
    truncated = data_end is not None and deadline > data_end
    proceeds += units * last_px
    return _out(proceeds, p0, side_cost, rungs, exit_kind, truncated)


def _out(proceeds: float, p0: float, side_cost: float, rungs: int, kind: str, truncated: bool) -> dict:
    gross = proceeds / p0 - 1
    net = proceeds * (1 - side_cost) / p0 - (1 + side_cost)
    return {"net": float(net), "gross": float(gross), "rungs": rungs, "exit": kind, "truncated": truncated}
