"""
Bonding-curve 1-minute bars, aggregated on Dune (pump.fun + Raydium LaunchLab).

Why (2026-10-03, see docs/decisions.md "Cost estimate for more months"):
Gold reads only `silver.token_price_1m` (1-minute bars), never individual
trades, and trade-level Bronze has no trader_id anyway. Re-pulling ~108M raw
pump.fun curve trades per month costs ~75-90k Dune credits. Aggregating them
into 1m bars on Dune -- the same pattern as postgrad_bars.py -- exports far
fewer rows and adds per-bar trader stats Bronze never had.

Pure Python (SQL builders + row mapper), unit-tested in
tests/test_curve_bars.py. Orchestration: notebooks/00g (cost test).

Conventions (same as postgrad_bars.py):
- **base** = the launched token; **quote** = what it's priced in.
  pump.fun curve: quote is native SOL, which Dune labels with the WSOL mint;
  there is one curve per mint, so `pool_id` = the base mint.
  LaunchLab: base/quote per pool from `raydium_launchlab_v1.stg_swaps`
  (any quote token since 2026-09-07), one curve pool per mint.
- `price` = USD (amount_usd / base_amount), `price_q` = quote units.
- Dust trades (amount_usd < MIN_TRADE_USD, see silver_price_bars.py) are
  dropped here, since the bars can't be un-aggregated later. Trades with no
  USD price are kept (they still have price_q) and counted in n_unpriced.
- Chunked by trade time on minute boundaries, so no bar spans two chunks
  and every per-bar stat is chunk-safe.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ingestion.postgrad_bars import _ORDER_KEY, _f, _i, _ts, parse_dune_ts

WSOL = "So11111111111111111111111111111111111111112"
MIN_TRADE_USD = 0.01  # keep in sync with features.silver_price_bars.MIN_TRADE_USD


def _curve_bars_select(venue_label: str) -> str:
    """Aggregation over a CTE `x` with: pool_id, base_mint, quote_mint,
    block_time, order_key, price, price_q, quote_amount, amount_usd, is_buy,
    trader_id. Same OHLCV columns as postgrad_bars._bars_select, plus
    per-bar trader stats (2026-10-03):
    - volume_buy_q / volume_sell_q: quote volume by side.
    - max_trade_q: the bar's largest single trade (quote units).
    - top_trader_share: largest single wallet's share of the bar's quote
      volume (1.0 = one wallet did everything -- bot/wash signal)."""
    return f"""
, xt AS (
  SELECT
    x.*,
    sum(quote_amount) OVER (
      PARTITION BY pool_id, date_trunc('minute', block_time), trader_id
    ) AS trader_bar_q
  FROM x
)
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
  min_by(price_q, order_key) FILTER (WHERE price_q IS NOT NULL) AS open_q,
  max(price_q) AS high_q,
  min(price_q) AS low_q,
  max_by(price_q, order_key) FILTER (WHERE price_q IS NOT NULL) AS close_q,
  sum(quote_amount) AS volume_quote,
  sum(quote_amount) FILTER (WHERE is_buy) AS volume_buy_q,
  sum(quote_amount) FILTER (WHERE NOT is_buy) AS volume_sell_q,
  max(quote_amount) AS max_trade_q,
  max(trader_bar_q) / NULLIF(sum(quote_amount), 0) AS top_trader_share,
  count(*) AS n_trades,
  count_if(is_buy) AS n_buys,
  count(DISTINCT trader_id) AS n_traders,
  count_if(price IS NULL) AS n_unpriced
FROM xt
GROUP BY 1, 2, 3, 4, 5
"""


def _not_dust(alias: str = "t") -> str:
    return f"({alias}.amount_usd IS NULL OR {alias}.amount_usd >= {MIN_TRADE_USD})"


def pumpfun_curve_bars_sql(chunk_start: datetime, chunk_end: datetime) -> str:
    """1m bars for every pump.fun curve trade in [chunk_start, chunk_end).
    Trades where neither side is WSOL (none expected on the curve) are
    excluded rather than guessed."""
    return f"""
WITH x AS (
  SELECT
    CASE WHEN t.token_sold_mint_address = '{WSOL}' THEN t.token_bought_mint_address
         ELSE t.token_sold_mint_address END AS pool_id,
    CASE WHEN t.token_sold_mint_address = '{WSOL}' THEN t.token_bought_mint_address
         ELSE t.token_sold_mint_address END AS base_mint,
    '{WSOL}' AS quote_mint,
    t.block_time,
    {_ORDER_KEY} AS order_key,
    t.token_sold_mint_address = '{WSOL}' AS is_buy,
    t.trader_id,
    t.amount_usd,
    CASE
      WHEN t.token_sold_mint_address = '{WSOL}' THEN t.amount_usd / NULLIF(t.token_bought_amount, 0)
      ELSE t.amount_usd / NULLIF(t.token_sold_amount, 0)
    END AS price,
    CASE
      WHEN t.token_sold_mint_address = '{WSOL}' THEN t.token_sold_amount / NULLIF(t.token_bought_amount, 0)
      ELSE t.token_bought_amount / NULLIF(t.token_sold_amount, 0)
    END AS price_q,
    CASE
      WHEN t.token_sold_mint_address = '{WSOL}' THEN t.token_sold_amount
      ELSE t.token_bought_amount
    END AS quote_amount
  FROM dex_solana.trades t
  WHERE t.project = 'pumpdotfun' AND t.version = 1
    AND t.block_time >= {_ts(chunk_start)}
    AND t.block_time <  {_ts(chunk_end)}
    AND '{WSOL}' IN (t.token_bought_mint_address, t.token_sold_mint_address)
    AND {_not_dust()}
)
{_curve_bars_select('pumpdotfun')}"""


def launchlab_curve_bars_sql(chunk_start: datetime, chunk_end: datetime) -> str:
    """1m bars for every LaunchLab curve trade in [chunk_start, chunk_end).
    Base/quote come from the pool table, matched on BOTH mints (doesn't rely
    on dex_solana.trades.project_program_id being the pool account)."""
    return f"""
WITH pools AS (
  SELECT DISTINCT pool_id, base_token_mint AS base_mint, quote_token_mint AS quote_mint
  FROM raydium_launchlab_v1.stg_swaps
  WHERE block_date >= DATE '{chunk_start.date()}'
    AND block_date <= DATE '{chunk_end.date()}'
),
x AS (
  SELECT
    p.pool_id,
    p.base_mint,
    p.quote_mint,
    t.block_time,
    {_ORDER_KEY} AS order_key,
    t.token_bought_mint_address = p.base_mint AS is_buy,
    t.trader_id,
    t.amount_usd,
    CASE
      WHEN t.token_bought_mint_address = p.base_mint THEN t.amount_usd / NULLIF(t.token_bought_amount, 0)
      ELSE t.amount_usd / NULLIF(t.token_sold_amount, 0)
    END AS price,
    CASE
      WHEN t.token_bought_mint_address = p.base_mint THEN t.token_sold_amount / NULLIF(t.token_bought_amount, 0)
      ELSE t.token_bought_amount / NULLIF(t.token_sold_amount, 0)
    END AS price_q,
    CASE
      WHEN t.token_bought_mint_address = p.base_mint THEN t.token_sold_amount
      ELSE t.token_bought_amount
    END AS quote_amount
  FROM dex_solana.trades t
  JOIN pools p
    ON (t.token_bought_mint_address = p.base_mint AND t.token_sold_mint_address = p.quote_mint)
    OR (t.token_sold_mint_address = p.base_mint AND t.token_bought_mint_address = p.quote_mint)
  WHERE t.project = 'raydium_launchlab' AND t.version = 1
    AND t.block_time >= {_ts(chunk_start)}
    AND t.block_time <  {_ts(chunk_end)}
    AND {_not_dust()}
)
{_curve_bars_select('raydium_launchlab')}"""


def summarize_sql(bars_sql: str) -> str:
    """Wrap a bars query so Dune computes it but returns ONE summary row --
    for measuring compute cost without paying to export the bars."""
    return f"""
WITH bars AS ({bars_sql})
SELECT
  count(*) AS n_bars,
  count(DISTINCT base_mint) AS n_tokens,
  sum(n_trades) AS n_trades,
  sum(n_unpriced) AS n_unpriced,
  avg(top_trader_share) AS avg_top_trader_share
FROM bars
"""


def to_curve_bar_row(r: dict[str, Any], ingested_at: datetime) -> dict[str, Any]:
    """Dune curve-bar row -> Bronze row (explicit types)."""
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
        "open_q": _f(r.get("open_q")),
        "high_q": _f(r.get("high_q")),
        "low_q": _f(r.get("low_q")),
        "close_q": _f(r.get("close_q")),
        "volume_quote": _f(r.get("volume_quote")),
        "volume_buy_q": _f(r.get("volume_buy_q")),
        "volume_sell_q": _f(r.get("volume_sell_q")),
        "max_trade_q": _f(r.get("max_trade_q")),
        "top_trader_share": _f(r.get("top_trader_share")),
        "n_trades": _i(r.get("n_trades")),
        "n_buys": _i(r.get("n_buys")),
        "n_traders": _i(r.get("n_traders")),
        "n_unpriced": _i(r.get("n_unpriced")),
        "source": "dune",
        "ingested_at": ingested_at,
    }
