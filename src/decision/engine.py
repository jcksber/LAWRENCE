"""
Deterministic decision engine (docs/decision_agent_spec.md §3-§6).

Called once per token each time one of its 1-minute bars closes, with that token's bars CLOSED BY `now`.
- Open position: re-resolve its barrier with the contract's own label logic (2x / -50% / 6h, a bar touching
  both = stop). When resolved and its exit time has passed, close it at the label's return.
- No open position: ask the contract for an assessment; enter if score >= cutoff and the risk gate passes.
One open position per token, exactly as the training/backtest rule. Every decision is written to the audit log.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from decision.risk import RiskGate
from decision.types import Broker, OrderIntent, Position, RiskConfig, ScoringContract


class AuditLog:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, **row: Any) -> None:
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class Engine:
    def __init__(self, contract: ScoringContract, broker: Broker, risk: RiskConfig, log: AuditLog | None = None):
        self.contract, self.broker, self.risk = contract, broker, RiskGate(risk)
        self.cfg = risk
        self.log = log or AuditLog()
        self.open: dict[str, Position] = {}
        self.closed: list[Position] = []
        self.paused = False
        self.killed = False
        self._ids = itertools.count(1)

    # --- controls (the only calls the supervisor agent may make) ---------------------------------------
    def pause(self, now: datetime, reason: str) -> None:
        self.paused = True
        self.log.write(ts=now, event="pause", reason=reason)

    def resume(self, now: datetime) -> None:
        if not self.killed:
            self.paused = False
            self.log.write(ts=now, event="resume")

    def kill(self, now: datetime, marks: dict[str, float], reason: str) -> None:
        """Close everything at the given marks and stop for good."""
        for tok in list(self.open):
            p = self.open[tok]
            px = marks.get(tok, p.entry_price)
            self._close(p, now, "kill", px / p.entry_price - 1)
        self.paused = self.killed = True
        self.log.write(ts=now, event="kill", reason=reason)

    # --- main entry point ----------------------------------------------------------------------------
    def on_bar_close(self, token_id: str, bars: pd.DataFrame, now: datetime, *, launch_time: Any,
                     launch_venue: str, observed_from_launch: bool = True) -> None:
        closed = bars[bars["bar_start"] + pd.Timedelta(minutes=1) <= pd.Timestamp(now)]
        if closed.empty:
            return
        if token_id in self.open:
            self._check_exit(self.open[token_id], closed, now)
            return
        if self.killed:
            return
        a = self.contract.assess(token_id, closed, now, launch_time=launch_time, launch_venue=launch_venue,
                                 observed_from_launch=observed_from_launch)
        if not a.ok:
            return  # off-grid / not a candidate: not logged (the vast majority of calls)
        base = dict(ts=now, token_id=token_id, contract_version=self.contract.version,
                    cutoff=self.contract.cutoff, score=a.score, price_at_t=a.price_at_t)
        if a.score is None or a.score < self.contract.cutoff:
            self.log.write(**base, event="assess", reason="below_cutoff")
            return
        last_close = closed["bar_start"].max() + pd.Timedelta(minutes=1)
        ok, why = self.risk.check_entry(now, open_positions=len(self.open),
                                        bar_age_s=(pd.Timestamp(now) - last_close).total_seconds(),
                                        paused=self.paused)
        if not ok:
            self.log.write(**base, event="skip", reason=why)
            return
        intent = OrderIntent(token_id, "buy", self.cfg.position_usd, float(a.price_at_t), now, "entry")
        fill = self.broker.submit(intent, now)
        p = Position(f"p{next(self._ids)}", token_id, launch_venue, now, float(a.price_at_t), fill.price,
                     self.cfg.position_usd, float(a.score))
        self.open[token_id] = p
        self.risk.record_entry(now)
        self.log.write(**base, event="enter", reason="score_ge_cutoff", position_id=p.position_id,
                       fill_px=fill.price, slippage_bps=fill.slippage_bps, features=a.features)

    # --- internals -------------------------------------------------------------------------------------
    def _check_exit(self, p: Position, closed: pd.DataFrame, now: datetime) -> None:
        out = self.contract.resolve(closed, p.entry_time, p.entry_price, launch_venue=p.launch_venue,
                                    observed_until=now)
        if out is None or pd.Timestamp(out["exit_time"]) > pd.Timestamp(now):
            return
        self._close(p, out["exit_time"], out["barrier_hit"], out["return"])

    def _close(self, p: Position, ts: datetime, reason: str, r_gross: float) -> None:
        cost = self.contract.round_trip_cost(p.launch_venue)
        p.exit_time, p.exit_reason, p.r_gross, p.r_net = ts, reason, float(r_gross), float(r_gross) - cost
        intent = OrderIntent(p.token_id, "sell", p.notional_usd, p.entry_price * (1 + r_gross), ts, reason)
        fill = self.broker.submit(intent, ts)
        self.closed.append(p)
        del self.open[p.token_id]
        self.risk.record_exit(ts, p.r_net * p.notional_usd)
        self.log.write(ts=ts, token_id=p.token_id, event="exit", reason=reason, position_id=p.position_id,
                       r_gross=p.r_gross, r_net=p.r_net, fill_px=fill.price, slippage_bps=fill.slippage_bps)

    def trades(self) -> pd.DataFrame:
        return pd.DataFrame([vars(p) for p in self.closed])
