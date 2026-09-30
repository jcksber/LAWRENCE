"""
Post-graduation price bars + pool metadata, pulled from Dune.

Why this exists (2026-09-26, see docs/venue_mechanics.md and the 00e
cost-test entries in docs/decisions.md): Bronze's pump.fun and LaunchLab
rows are bonding-curve trades only. When a token graduates, trading moves
to PumpSwap (pump.fun) or Raydium CPMM (LaunchLab), so without this the
winners vanish from our data exactly when they win. Raw post-graduation
trades were measured at ~336M rows/month (~260k credits) -- unaffordable --
so Dune aggregates them into 1-minute OHLCV bars per pool and only the bars
are exported (~6.5M rows/month at a 24h horizon).

Pure Python, no PySpark -- SQL builders and row mappers only, unit-tested in
tests/test_postgrad_bars.py. The Spark/Dune orchestration lives in
notebooks/00f_bronze_postgrad_bars_dune.py.

Conventions:
- **base** = the launched token (what we score); **quote** = what it's priced
  in. For PumpSwap both come from `pumpswap_solana.pools`; for CPMM the base
  is the LaunchLab-launched mint and the quote is whatever it traded against.
- Prices are the BASE token's USD price per trade, `amount_usd / base_amount`
  (Dune's `amount_usd` is one trade-level notional -- see dune_client.py).
- Trades with no USD price (neither mint priced by Dune) are counted in
  `n_unpriced` but excluded from OHLC/volume, never silently dropped.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

HORIZON_HOURS_DEFAULT = 24
CPMM_VERSION = 5  # raydium v5 = CPMM (CPMMoo8L...), confirmed 2026-09-26

# Deterministic within-minute ordering for open/close: slot, then tx position
# in the slot, then instruction position in the tx.
_ORDER_KEY = (
    "(CAST(t.block_slot AS DOUBLE) * 1e9 + CAST(t.tx_index AS DOUBLE) * 1e3"
    " + CAST(t.outer_instruction_index AS DOUBLE))"
)


def _ts(dt: datetime) -> str:
    return f"TIMESTAMP '{dt.strftime('%Y-%m-%d %H:%M:%S')}'"


def _bars_select(venue_label: str) -> str:
    """Shared aggregation over a CTE `x` with columns: pool_id, base_mint,
    quote_mint, block_time, order_key, price, amount_usd, is_buy, trader_id."""
    return f"""
SELECT
  '{venue_label}' AS venue,
  pool_id,
  base_mint,
  quote_mint,
  date_trunc('minute', block_time) AS bar_start,
  min_by(price, order_key) FILTER (WHERE price IS NOT NULL) AS open_usd,
  max(price) AS high_usd,
  min(price) AS low_usd,
  max_by(price, order_key) FILTER (WHERE price IS NOT NULL) AS close_usd,
  sum(amount_usd) FILTER (WHERE price IS NOT NULL) AS volume_usd,
  count(*) AS n_trades,
  count_if(is_buy) AS n_buys,
  count(DISTINCT trader_id) AS n_traders,
  count_if(price IS NULL) AS n_unpriced
FROM x
GROUP BY 1, 2, 3, 4, 5
"""


def pumpswap_bars_sql(
    pools_created_since: datetime,
    pools_created_until: datetime,
    horizon_hours: int = HORIZON_HOURS_DEFAULT,
) -> str:
    """1m bars for PumpSwap pools CREATED in [since, until), covering each
    pool's first `horizon_hours`. Chunked by pool creation time so a chunk's
    boundaries are stable across re-runs (progress-table skip-check)."""
    return f"""
WITH pools AS (
  SELECT pool, baseMint, quoteMint, created_at
  FROM pumpswap_solana.pools
  WHERE created_at >= {_ts(pools_created_since)}
    AND created_at <  {_ts(pools_created_until)}
),
x AS (
  SELECT
    p.pool AS pool_id,
    p.baseMint AS base_mint,
    p.quoteMint AS quote_mint,
    t.block_time,
    {_ORDER_KEY} AS order_key,
    t.token_bought_mint_address = p.baseMint AS is_buy,
    t.trader_id,
    t.amount_usd,
    CASE
      WHEN t.token_bought_mint_address = p.baseMint THEN t.amount_usd / NULLIF(t.token_bought_amount, 0)
      WHEN t.token_sold_mint_address   = p.baseMint THEN t.amount_usd / NULLIF(t.token_sold_amount, 0)
    END AS price
  FROM dex_solana.trades t
  JOIN pools p ON t.project_program_id = p.pool
  WHERE t.project = 'pumpswap'
    AND t.block_time >= {_ts(pools_created_since)}
    AND t.block_time <  {_ts(pools_created_until)} + INTERVAL '{horizon_hours}' HOUR
    AND t.block_time <  p.created_at + INTERVAL '{horizon_hours}' HOUR
)
{_bars_select('pumpswap')}"""


def cpmm_launchlab_bars_sql(
    window_start: datetime,
    window_end: datetime,
    horizon_hours: int = HORIZON_HOURS_DEFAULT,
) -> str:
    """1m bars for Raydium CPMM pools trading a LaunchLab-launched mint,
    covering each pool's first `horizon_hours` after its first CPMM trade in
    the window (~= graduation). One query for the whole window: CPMM volume
    for these mints is small (00e cell 4: 1,624 trades/day pre-2026-09-07).

    Known approximation: a pool's first trade is taken from inside the
    window, so a pool that graduated BEFORE window_start gets a late anchor.
    Tolerable for tokens whose curve trading is in-window (the ones we need).
    """
    return f"""
WITH ll_mints AS (
  SELECT DISTINCT base_token_mint AS mint
  FROM raydium_launchlab_v1.stg_swaps
  WHERE block_date >= DATE '{window_start.date()}'
    AND block_date <  DATE '{window_end.date()}'
),
cp AS (
  SELECT t.*, m.mint AS base_mint
  FROM dex_solana.trades t
  JOIN ll_mints m ON m.mint IN (t.token_bought_mint_address, t.token_sold_mint_address)
  WHERE t.project = 'raydium' AND t.version = {CPMM_VERSION}
    AND t.block_time >= {_ts(window_start)}
    AND t.block_time <  {_ts(window_end)} + INTERVAL '{horizon_hours}' HOUR
),
grad AS (
  SELECT project_program_id AS pool_id, base_mint, min(block_time) AS first_trade
  FROM cp
  GROUP BY 1, 2
  HAVING min(block_time) < {_ts(window_end)}
),
x AS (
  SELECT
    g.pool_id,
    g.base_mint,
    CASE WHEN t.token_bought_mint_address = g.base_mint
         THEN t.token_sold_mint_address ELSE t.token_bought_mint_address END AS quote_mint,
    t.block_time,
    {_ORDER_KEY} AS order_key,
    t.token_bought_mint_address = g.base_mint AS is_buy,
    t.trader_id,
    t.amount_usd,
    CASE
      WHEN t.token_bought_mint_address = g.base_mint THEN t.amount_usd / NULLIF(t.token_bought_amount, 0)
      ELSE t.amount_usd / NULLIF(t.token_sold_amount, 0)
    END AS price
  FROM cp t
  JOIN grad g ON t.project_program_id = g.pool_id AND t.base_mint = g.base_mint
  WHERE t.block_time < g.first_trade + INTERVAL '{horizon_hours}' HOUR
)
{_bars_select('raydium_cpmm')}"""


def pumpswap_pools_sql(window_start: datetime, window_end: datetime) -> str:
    """One row per PumpSwap pool created in the window (~35k/month)."""
    return f"""
SELECT
  'pumpswap' AS venue,
  pool AS pool_id,
  baseMint AS base_mint,
  quoteMint AS quote_mint,
  CAST(NULL AS varchar) AS platform_name,
  created_at
FROM pumpswap_solana.pools
WHERE created_at >= {_ts(window_start)}
  AND created_at <  {_ts(window_end)}
"""


def launchlab_pools_sql(window_start: datetime, window_end: datetime) -> str:
    """One row per LaunchLab curve pool that traded in the window. created_at
    is the pool's first in-window curve trade (Dune's stg_swaps has no create
    event); for pools launched before window_start it's a late proxy."""
    return f"""
SELECT
  'raydium_launchlab' AS venue,
  pool_id,
  base_token_mint AS base_mint,
  quote_token_mint AS quote_mint,
  max(platform_name) AS platform_name,
  min(block_time) AS created_at
FROM raydium_launchlab_v1.stg_swaps
WHERE block_date >= DATE '{window_start.date()}'
  AND block_date <  DATE '{window_end.date()}'
GROUP BY 1, 2, 3, 4
"""


def parse_dune_ts(value: Any) -> datetime | None:
    """Dune returns timestamps as 'YYYY-MM-DD HH:MM:SS.mmm UTC' strings.
    Always returns a UTC-aware datetime (or None)."""
    if value is None:
        return None
    s = str(value).strip()
    if s.endswith(" UTC"):
        s = s[: -len(" UTC")]
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _f(x: Any) -> float | None:
    return None if x is None else float(x)


def _i(x: Any) -> int | None:
    return None if x is None else int(x)


def to_bar_row(r: dict[str, Any], ingested_at: datetime) -> dict[str, Any]:
    """Dune bar row -> Bronze `postgrad_bars_1m` row (explicit types; Dune's
    JSON mixes int/float for the same column -- see dune_client._to_float)."""
    return {
        "venue": r["venue"],
        "pool_id": r["pool_id"],
        "base_mint": r["base_mint"],
        "quote_mint": r["quote_mint"],
        "bar_start": parse_dune_ts(r["bar_start"]),
        "open_usd": _f(r.get("open_usd")),
        "high_usd": _f(r.get("high_usd")),
        "low_usd": _f(r.get("low_usd")),
        "close_usd": _f(r.get("close_usd")),
        "volume_usd": _f(r.get("volume_usd")),
        "n_trades": _i(r.get("n_trades")),
        "n_buys": _i(r.get("n_buys")),
        "n_traders": _i(r.get("n_traders")),
        "n_unpriced": _i(r.get("n_unpriced")),
        "source": "dune",
        "ingested_at": ingested_at,
    }


def to_pool_row(r: dict[str, Any], ingested_at: datetime) -> dict[str, Any]:
    return {
        "venue": r["venue"],
        "pool_id": r["pool_id"],
        "base_mint": r["base_mint"],
        "quote_mint": r["quote_mint"],
        "platform_name": r.get("platform_name"),
        "created_at": parse_dune_ts(r.get("created_at")),
        "source": "dune",
        "ingested_at": ingested_at,
    }
