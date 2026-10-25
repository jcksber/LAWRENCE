"""Risk gate (spec §5): every entry is checked; a failure means no order, with the reason logged."""
from __future__ import annotations

from datetime import datetime

from decision.types import RiskConfig


class RiskGate:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._day = None
        self.entries_today = 0
        self.realized_today_usd = 0.0

    def _roll(self, now: datetime) -> None:
        d = now.date()
        if d != self._day:
            self._day, self.entries_today, self.realized_today_usd = d, 0, 0.0

    def check_entry(self, now: datetime, *, open_positions: int, bar_age_s: float,
                    quote_slippage_bps: float = 0.0, paused: bool = False) -> tuple[bool, str]:
        self._roll(now)
        c = self.cfg
        if paused:
            return False, "paused"
        if self.realized_today_usd <= -c.daily_loss_limit_usd:
            return False, "daily_loss_limit"
        if open_positions >= c.max_open_positions:
            return False, "max_open_positions"
        if self.entries_today >= c.max_entries_per_day:
            return False, "max_entries_per_day"
        if bar_age_s > c.max_bar_age_s:
            return False, "stale_data"
        if quote_slippage_bps > c.max_quote_slippage_bps:
            return False, "quote_slippage"
        return True, "ok"

    def record_entry(self, now: datetime) -> None:
        self._roll(now)
        self.entries_today += 1

    def record_exit(self, now: datetime, pnl_usd: float) -> None:
        self._roll(now)
        self.realized_today_usd += pnl_usd
