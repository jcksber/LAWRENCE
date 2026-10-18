"""
Creator + early-buyer features for v3 part C (2026-10-18), pump.fun first.
Dune SQL builders only (pure Python, unit-tested). Every output is known at
the time it describes, so features at T only use rows with time <= T:

- creator_launches_sql: one row per pump.fun token CREATED in a chunk --
  mint, creator, create time/slot -- from the decoded create call. The
  creator's PRIOR launches/graduations are then computed in Spark with a
  strict "earlier than this token's creation" condition.
- early_buyers_sql: per graduated pump.fun token created in the chunk, the
  opening-block picture: buys in the creation slot and the first 3 slots,
  distinct buyers, share of the 1B supply taken -- all inside ~1-2 seconds of
  launch, so known at every post-graduation T.
- creator_trades_sql: the creator's own trades in its graduated token (curve +
  PumpSwap) for 48h after creation -- feature at T = what the creator did by T.

`grads_until` is also a HARD data cutoff: no query reads any trade at or after
it (2026-10-11 = the start of the reserved holdout window).

Table/column names for the decoded create call are VERIFIED by 00k's probe
cell before any of these run; CREATE_TABLE / columns below are the expected
Spellbook names and are overridden there if they differ.
"""
from __future__ import annotations

from datetime import datetime

from ingestion.postgrad_bars import _ts

WSOL = "So11111111111111111111111111111111111111112"
PUMP_SUPPLY = 1_000_000_000.0          # pump.fun tokens: 1B supply, 6 decimals (amounts are decimal-adjusted)
CREATE_TABLE = "pumpdotfun_solana.pump_call_create"
CREATE_MINT, CREATE_USER, CREATE_TIME, CREATE_SLOT = "account_mint", "account_user", "call_block_time", "call_block_slot"
EARLY_SLOTS = 3


def creator_launches_sql(start: datetime, end: datetime) -> str:
    return f"""
SELECT {CREATE_MINT} AS mint, {CREATE_USER} AS creator,
       {CREATE_TIME} AS created_at, {CREATE_SLOT} AS create_slot
FROM {CREATE_TABLE}
WHERE {CREATE_TIME} >= {_ts(start)} AND {CREATE_TIME} < {_ts(end)}
"""


def _graduates_created(start: datetime, end: datetime, grads_until: datetime) -> str:
    """CTE `c`: graduated pump.fun tokens (a PumpSwap pool exists by grads_until)
    created in [start, end)."""
    return f"""
c AS (
  SELECT k.{CREATE_MINT} AS mint, k.{CREATE_USER} AS creator, k.{CREATE_TIME} AS created_at, k.{CREATE_SLOT} AS s0
  FROM {CREATE_TABLE} k
  JOIN (SELECT DISTINCT baseMint FROM pumpswap_solana.pools WHERE created_at < {_ts(grads_until)}) p
    ON p.baseMint = k.{CREATE_MINT}
  WHERE k.{CREATE_TIME} >= {_ts(start)} AND k.{CREATE_TIME} < {_ts(end)}
)"""


def early_buyers_sql(start: datetime, end: datetime, grads_until: datetime) -> str:
    return f"""
WITH {_graduates_created(start, end, grads_until).strip()},
b AS (
  SELECT t.token_bought_mint_address AS mint, t.block_slot, t.trader_id, t.token_bought_amount AS base_amt, t.amount_usd
  FROM dex_solana.trades t
  WHERE t.project = 'pumpdotfun' AND t.version = 1
    AND t.block_time >= {_ts(start)} AND t.block_time < {_ts(end)} + INTERVAL '10' MINUTE
    AND t.block_time < {_ts(grads_until)}  -- never read past the data cutoff (holdout starts there)
    AND t.token_sold_mint_address = '{WSOL}'
)
SELECT c.mint, c.creator, c.created_at, c.s0,
       count_if(b.block_slot = c.s0) AS buys_slot0,
       count(DISTINCT CASE WHEN b.block_slot = c.s0 THEN b.trader_id END) AS buyers_slot0,
       count(DISTINCT CASE WHEN b.block_slot = c.s0 AND b.trader_id <> c.creator THEN b.trader_id END) AS non_creator_buyers_slot0,
       sum(CASE WHEN b.block_slot = c.s0 THEN b.base_amt END) / {PUMP_SUPPLY} AS supply_share_slot0,
       sum(CASE WHEN b.block_slot < c.s0 + {EARLY_SLOTS} THEN b.base_amt END) / {PUMP_SUPPLY} AS supply_share_first{EARLY_SLOTS}slots,
       count(DISTINCT CASE WHEN b.block_slot < c.s0 + {EARLY_SLOTS} THEN b.trader_id END) AS buyers_first{EARLY_SLOTS}slots,
       sum(CASE WHEN b.block_slot = c.s0 AND b.trader_id = c.creator THEN b.base_amt END) / {PUMP_SUPPLY} AS creator_share_slot0
FROM c LEFT JOIN b ON b.mint = c.mint AND b.block_slot < c.s0 + {EARLY_SLOTS}
GROUP BY 1, 2, 3, 4
"""


def creator_trades_sql(start: datetime, end: datetime, grads_until: datetime) -> str:
    """The creator's trades in its own graduated token, curve + PumpSwap,
    within 48h of creation. Two equi-joins (bought / sold) instead of an OR."""
    return f"""
WITH {_graduates_created(start, end, grads_until).strip()},
t AS (
  SELECT block_time, trader_id, token_bought_mint_address, token_sold_mint_address,
         token_bought_amount, token_sold_amount, amount_usd, project
  FROM dex_solana.trades
  WHERE project IN ('pumpdotfun', 'pumpswap')
    AND block_time >= {_ts(start)} AND block_time < {_ts(end)} + INTERVAL '48' HOUR
    AND block_time < {_ts(grads_until)}  -- never read past the data cutoff (holdout starts there)
)
SELECT c.mint, t.block_time, true AS is_buy, t.token_bought_amount AS base_amt, t.amount_usd, t.project
FROM t JOIN c ON t.token_bought_mint_address = c.mint AND t.trader_id = c.creator
WHERE t.block_time < c.created_at + INTERVAL '48' HOUR
UNION ALL
SELECT c.mint, t.block_time, false AS is_buy, t.token_sold_amount AS base_amt, t.amount_usd, t.project
FROM t JOIN c ON t.token_sold_mint_address = c.mint AND t.trader_id = c.creator
WHERE t.block_time < c.created_at + INTERVAL '48' HOUR
"""


def summarize(sql: str) -> str:
    """Compute on Dune but return one row -- for measuring credits."""
    return f"WITH q AS ({sql}) SELECT count(*) AS n_rows FROM q"


PROBE_SQL = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'pumpdotfun_solana' AND table_name LIKE '%create%'
ORDER BY table_name, ordinal_position
"""
