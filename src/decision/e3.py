"""Frozen v1 entries, E3 exits, and restorable state for the paper broker."""
from dataclasses import asdict, dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from backtest.ladder_exit import E3_SCHEDULE, E3_STOP
from decision.brokers import PaperBroker
from decision.engine import Engine
from decision.types import OrderIntent, Position, RiskConfig


@dataclass
class E3Position(Position):
    remaining: float = 1.0  # fraction of the original units
    next_rung: int = 0
    proceeds_usd: float = 0.0


class E3PaperEngine(Engine):
    position_type = E3Position

    def __init__(self, contract, risk, *, on_first_rung=None):
        # Costs (including the contract's slippage) are charged per side below.
        super().__init__(contract, PaperBroker(0.0), risk)
        self.on_first_rung = on_first_rung

    def _check_exit(self, p, closed, now):
        bar = closed.iloc[-1]
        if bar.bar_start + pd.Timedelta(minutes=1) != now or bar.phase != "postgrad":
            return False
        stop = p.entry_price * E3_STOP
        if bar.low <= stop:
            price = bar.close if bar.high < stop and pd.notna(bar.close) else stop
            self._sell(p, now, price, p.remaining, "stop")
            return True
        while p.next_rung < len(E3_SCHEDULE):
            multiple, share = E3_SCHEDULE[p.next_rung]
            price = p.entry_price * multiple
            if pd.isna(bar.high) or bar.high < price:
                break
            p.next_rung += 1
            self._sell(p, now, price, p.remaining * share, f"{multiple:g}x")
            if p.next_rung == 1 and self.on_first_rung is not None:
                # Called before any higher target on this very same bar.
                self.on_first_rung(self, p, now)
        return p.remaining == 0

    def _close(self, p, ts, reason, r_gross):
        # Engine.kill uses this path. Liquidate only the remaining units.
        self._sell(p, ts, p.entry_price * (1 + r_gross), p.remaining, reason)

    def _sell(self, p, ts, price, fraction, reason):
        basis = p.notional_usd * fraction
        units = basis / p.fill_price
        fill = self.broker.submit(OrderIntent(p.token_id, "sell", units * price, price, ts, reason), ts)
        proceeds = units * fill.price
        side_cost = self.contract.round_trip_cost(p.launch_venue) / 2
        net_usd = proceeds * (1 - side_cost) - basis * (1 + side_cost)
        p.proceeds_usd += proceeds
        p.remaining -= fraction
        self.risk.record_exit(ts, net_usd)
        if p.remaining == 0:
            p.exit_time = ts
            p.exit_reason = "target" if reason == "10x" else reason
            p.r_gross = p.proceeds_usd / p.notional_usd - 1
            p.r_net = p.proceeds_usd / p.notional_usd * (1 - side_cost) - (1 + side_cost)
            self.closed.append(p)
            del self.open[p.token_id]
        self.log.write(ts=ts, token_id=p.token_id, position_id=p.position_id,
                       event="exit" if p.remaining == 0 else "partial_exit", reason=reason,
                       fill_px=fill.price, units_sold=units, fraction_sold=fraction,
                       remaining=p.remaining, proceeds_usd=proceeds, realized_net_usd=net_usd,
                       r_gross=p.r_gross, r_net=p.r_net)

    def snapshot(self):
        return _portable({
            "version": 1, "contract": self.contract.version, "cutoff": self.contract.cutoff,
            "open": [asdict(p) for p in self.open.values()], "closed": [asdict(p) for p in self.closed],
            "next_position_id": self.next_position_id, "risk_config": asdict(self.cfg),
            "risk_day": self.risk._day, "entries_today": self.risk.entries_today,
            "realized_today_usd": self.risk.realized_today_usd, "paused": self.paused,
            "killed": self.killed, "audit": self.log.rows,
        })

    @classmethod
    def restore(cls, contract, state):
        if (state["version"], state["contract"], state["cutoff"]) != (1, contract.version, contract.cutoff):
            raise ValueError("Incompatible E3 checkpoint or scoring contract")
        engine = cls(contract, RiskConfig(**state["risk_config"]))

        def position(row):
            return E3Position(**{**row, "entry_time": pd.Timestamp(row["entry_time"]),
                                 "exit_time": pd.Timestamp(row["exit_time"]) if row["exit_time"] else None})

        engine.open = {p["token_id"]: position(p) for p in state["open"]}
        engine.closed = [position(p) for p in state["closed"]]
        engine.next_position_id = state["next_position_id"]
        engine.risk._day = date.fromisoformat(state["risk_day"]) if state["risk_day"] else None
        engine.risk.entries_today = state["entries_today"]
        engine.risk.realized_today_usd = state["realized_today_usd"]
        engine.paused, engine.killed = state["paused"], state["killed"]
        engine.log.rows = [{**row, "ts": pd.Timestamp(row["ts"])} for row in state["audit"]]
        return engine


def _portable(value):
    """Strict JSON round trip, including nullable v1 feature values and UTC timestamps."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    return None if pd.isna(value) else value
