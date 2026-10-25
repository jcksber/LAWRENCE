from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class Assessment:
    token_id: str
    as_of: datetime
    ok: bool                      # a valid candidate row (contract's row rules passed)
    reason: str                   # "candidate" or why not
    score: float | None = None
    price_at_t: float | None = None
    features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OrderIntent:
    token_id: str
    side: str                     # "buy" | "sell"
    notional_usd: float
    ref_price: float              # the price the decision assumed (contract price_at_t, or exit price)
    as_of: datetime
    reason: str                   # "entry" | "target" | "stop" | "timeout" | "kill"


@dataclass(frozen=True)
class Fill:
    token_id: str
    side: str
    price: float
    notional_usd: float
    ts: datetime
    slippage_bps: float           # vs ref_price, signed against us


@dataclass
class Position:
    position_id: str
    token_id: str
    launch_venue: str
    entry_time: datetime          # assessment time T (as in training labels)
    entry_price: float            # contract price_at_t
    fill_price: float
    notional_usd: float
    score: float
    exit_time: datetime | None = None
    exit_reason: str | None = None
    r_gross: float | None = None  # barrier return, as the label computes it
    r_net: float | None = None    # after round-trip cost


@dataclass(frozen=True)
class RiskConfig:
    position_usd: float = 100.0
    max_open_positions: int = 10
    max_entries_per_day: int = 30
    daily_loss_limit_usd: float = 1000.0      # 10 x position size
    max_quote_slippage_bps: float = 100.0     # 2 x the contract's 50 bps assumption
    max_bar_age_s: float = 120.0

    @staticmethod
    def unlimited(position_usd: float = 100.0) -> "RiskConfig":
        """For replay parity: the backtest has no portfolio limits."""
        inf = 10**9
        return RiskConfig(position_usd, inf, inf, float(inf), float(inf), float(inf))


class ScoringContract(Protocol):
    version: str
    cutoff: float
    def assess(self, token_id: str, bars: pd.DataFrame, as_of: datetime, *, launch_time: Any,
               launch_venue: str, observed_from_launch: bool) -> Assessment: ...
    def resolve(self, bars: pd.DataFrame, entry_time: datetime, entry_price: float, *, launch_venue: str,
                observed_until: datetime) -> dict | None: ...
    def round_trip_cost(self, launch_venue: str) -> float: ...


class Broker(Protocol):
    def submit(self, intent: OrderIntent, now: datetime) -> Fill: ...


def as_row(obj: Any) -> dict:
    return asdict(obj)
