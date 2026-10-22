"""
v4 post-crash track (2026-10-22): HOURLY bars for PumpSwap pools from 24h to 14 days after the
pool's creation (graduation), so we can test entries days-to-weeks after launch (e.g. a 70-80%
crash, then recovery). The existing post-graduation 1m bars (postgrad_bars) stop at 24h.

Chunked by TRADE day (not pool-creation day): each day of pumpswap trades is scanned once, for every
pool aged [MIN_AGE_H, MAX_AGE_D) on that day. Chunking by creation day would rescan each trade day
14 times. Hourly bars keep rows small (alive pools x 24 per day).
Every query is capped below the caller's `cutoff` (the holdout start) -- tested.
"""
from __future__ import annotations

from datetime import datetime

from ingestion.postgrad_bars import _ORDER_KEY, _ts

MIN_AGE_H = 24
MAX_AGE_D = 14


def postcrash_hourly_bars_sql(day_start: datetime, day_end: datetime, cutoff: datetime,
                              min_age_h: int = MIN_AGE_H, max_age_d: int = MAX_AGE_D) -> str:
    if day_end > cutoff:
        raise ValueError(f"day_end {day_end} is past the cutoff {cutoff} (holdout)")
    return f"""
WITH pools AS (
  SELECT pool, baseMint, quoteMint, created_at
  FROM pumpswap_solana.pools
  WHERE created_at >= {_ts(day_start)} - INTERVAL '{max_age_d}' DAY
    AND created_at <  {_ts(day_end)} - INTERVAL '{min_age_h}' HOUR
),
x AS (
  SELECT p.pool AS pool_id, p.baseMint AS base_mint, p.quoteMint AS quote_mint, p.created_at,
         t.block_time, {_ORDER_KEY} AS order_key,
         t.token_bought_mint_address = p.baseMint AS is_buy, t.trader_id, t.amount_usd,
         CASE WHEN t.token_bought_mint_address = p.baseMint THEN t.amount_usd / NULLIF(t.token_bought_amount, 0)
              WHEN t.token_sold_mint_address   = p.baseMint THEN t.amount_usd / NULLIF(t.token_sold_amount, 0) END AS price,
         CASE WHEN t.token_bought_mint_address = p.baseMint THEN t.token_sold_amount / NULLIF(t.token_bought_amount, 0)
              WHEN t.token_sold_mint_address   = p.baseMint THEN t.token_bought_amount / NULLIF(t.token_sold_amount, 0) END AS price_q
  FROM dex_solana.trades t
  JOIN pools p ON t.project_program_id = p.pool
  WHERE t.project = 'pumpswap'
    AND t.block_time >= {_ts(day_start)} AND t.block_time < {_ts(day_end)}
    AND t.block_time < {_ts(cutoff)}  -- never read past the data cutoff (holdout starts there)
    AND t.block_time >= p.created_at + INTERVAL '{min_age_h}' HOUR
    AND t.block_time <  p.created_at + INTERVAL '{max_age_d}' DAY
)
SELECT pool_id, base_mint, quote_mint, created_at AS pool_created_at,
  date_trunc('hour', block_time) AS bar_start,
  min_by(price, order_key) FILTER (WHERE price IS NOT NULL) AS open_usd,
  max(price) AS high_usd, min(price) AS low_usd,
  max_by(price, order_key) FILTER (WHERE price IS NOT NULL) AS close_usd,
  sum(amount_usd) FILTER (WHERE price IS NOT NULL) AS volume_usd,
  min_by(price_q, order_key) FILTER (WHERE price_q IS NOT NULL) AS open_q,
  max(price_q) AS high_q, min(price_q) AS low_q,
  max_by(price_q, order_key) FILTER (WHERE price_q IS NOT NULL) AS close_q,
  count(*) AS n_trades, count_if(is_buy) AS n_buys, count(DISTINCT trader_id) AS n_traders
FROM x
GROUP BY 1, 2, 3, 4, 5
"""


def summarize(sql: str) -> str:
    return f"WITH q AS ({sql}) SELECT count(*) AS n_rows, count(DISTINCT pool_id) AS n_pools FROM q"
