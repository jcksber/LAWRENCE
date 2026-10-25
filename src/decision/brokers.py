"""Brokers. PaperBroker fills at the decision's reference price plus a fixed slippage (0 for replay parity)."""
from __future__ import annotations

from datetime import datetime

from decision.types import Fill, OrderIntent


class PaperBroker:
    def __init__(self, slippage_bps: float = 0.0):
        self.slippage_bps = float(slippage_bps)

    def submit(self, intent: OrderIntent, now: datetime) -> Fill:
        s = self.slippage_bps / 1e4
        px = intent.ref_price * (1 + s) if intent.side == "buy" else intent.ref_price * (1 - s)
        return Fill(intent.token_id, intent.side, px, intent.notional_usd, now, self.slippage_bps)
