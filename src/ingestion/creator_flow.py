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
# 2026-10-18 probe (00k cell 2): pump.fun has TWO create instructions --
# `create` (legacy SPL) and `create_v2` (Token-2022). Both are read and
# unioned. Verified columns (create): account_mint, account_user, creator
# (instruction arg), call_block_time, call_block_slot -- same in create_v2.
CREATE_TABLES = ("pumpdotfun_solana.pump_call_create", "pumpdotfun_solana.pump_call_create_v2")
CREATE_TABLE = CREATE_TABLES[0]  # probe only
EVT_TABLE = "pumpdotfun_solana.pump_evt_createevent"  # launch source (see _creates)
CREATE_MINT, CREATE_USER, CREATE_TIME, CREATE_SLOT = "account_mint", "account_user", "call_block_time", "call_block_slot"


def _creates(start: datetime, end: datetime) -> str:
    """All pump.fun creations in [start, end) from the CreateEvent log, which
    fires on every create (both instructions) -- 2026-09-20 check: 30,783 events
    = 248 `create` + 30,535 `create_v2`. One scan instead of a two-table union."""
    return f"""SELECT mint, creator, evt_block_time AS created_at, evt_block_slot AS s0,
         quote_mint, is_mayhem_mode
  FROM {EVT_TABLE}
  WHERE evt_block_time >= {_ts(start)} AND evt_block_time < {_ts(end)}"""
EARLY_SLOTS = 3


def creator_launches_sql(start: datetime, end: datetime) -> str:
    return f"""
SELECT mint, creator, created_at, s0 AS create_slot, quote_mint, is_mayhem_mode FROM (
  {_creates(start, end)}
)
"""


def _graduates_created(start: datetime, end: datetime, grads_until: datetime) -> str:
    """CTE `c`: graduated pump.fun tokens (a PumpSwap pool exists by grads_until)
    created in [start, end)."""
    return f"""
c AS (
  SELECT k.mint, k.creator, k.created_at, k.s0
  FROM ({_creates(start, end)}) k
  JOIN (SELECT DISTINCT baseMint FROM pumpswap_solana.pools WHERE created_at < {_ts(grads_until)}) p
    ON p.baseMint = k.mint
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


# 2026-10-18: information_schema.columns fails on Dune (TypeNotFoundException in
# one catalog), so probe with SHOW TABLES + a 5-minute sample instead.
PROBE_TABLES_SQL = "SHOW TABLES FROM pumpdotfun_solana LIKE '%create%'"
PROBE_SAMPLE_SQL = f"""
SELECT * FROM {CREATE_TABLES[1]}
WHERE {CREATE_TIME} >= TIMESTAMP '2026-09-20 00:00:00' AND {CREATE_TIME} < TIMESTAMP '2026-09-20 01:00:00'
LIMIT 2
"""
# launches per instruction over one hour -- how much create_v2 matters
PROBE_COUNTS_SQL = f"""
SELECT 'create' AS ix, count(*) AS n FROM {CREATE_TABLES[0]}
WHERE {CREATE_TIME} >= TIMESTAMP '2026-09-20 00:00:00' AND {CREATE_TIME} < TIMESTAMP '2026-09-20 01:00:00'
UNION ALL
SELECT 'create_v2', count(*) FROM {CREATE_TABLES[1]}
WHERE {CREATE_TIME} >= TIMESTAMP '2026-09-20 00:00:00' AND {CREATE_TIME} < TIMESTAMP '2026-09-20 01:00:00'
"""

# 2026-10-18: one day of cells 3-5 gave 248 launches and 10 graduates for
# 2026-09-20 -- far too few for pump.fun. Full-day coverage check: each create
# table, the CreateEvent table (emitted on every create, whoever calls it), and
# PumpSwap pools created that day (a rough count of graduations) for reference.
_DAY = "TIMESTAMP '2026-09-20 00:00:00' AND {c} < TIMESTAMP '2026-09-21 00:00:00'"
PROBE_DAY_SQL = f"""
SELECT 'call_create' AS src, count(*) AS n, count(DISTINCT account_mint) AS mints FROM {CREATE_TABLES[0]}
WHERE call_block_time >= {_DAY.format(c='call_block_time')}
UNION ALL
SELECT 'call_create_v2', count(*), count(DISTINCT account_mint) FROM {CREATE_TABLES[1]}
WHERE call_block_time >= {_DAY.format(c='call_block_time')}
UNION ALL
SELECT 'evt_createevent', count(*), NULL FROM {EVT_TABLE}
WHERE evt_block_time >= {_DAY.format(c='evt_block_time')}
UNION ALL
SELECT 'pumpswap_pools', count(*), count(DISTINCT baseMint) FROM pumpswap_solana.pools
WHERE created_at >= {_DAY.format(c='created_at')}
"""
PROBE_EVT_SAMPLE_SQL = f"""
SELECT * FROM {EVT_TABLE}
WHERE evt_block_time >= TIMESTAMP '2026-09-20 00:00:00' AND evt_block_time < TIMESTAMP '2026-09-20 00:10:00'
LIMIT 2
"""
