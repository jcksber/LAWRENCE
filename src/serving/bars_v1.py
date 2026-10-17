"""
Reference 1-minute bar builder for contract v1 (R2, 2026-10-17): raw swaps
(dex_solana.trades shape) -> the bars `contract_v1.compute_features` takes,
reproducing what training saw for 2026-09-13 on:
Dune curve bars (ingestion/curve_bars.py) + Dune post-graduation bars
(ingestion/postgrad_bars.py) -> Silver (features/silver_price_bars.py:
postgrad_bars() for both phases, same-minute merge).

The runtime should IMPORT this rather than reimplement it. Model-side owned.

Known training-side look-ahead (Silver uses the token's FULL 24h history):
1. outlier band median -> `band_mode="training"` reproduces it,
   `band_mode="live"` uses an expanding median (point-in-time). The runtime
   must use "live"; the gap is measured on the parity fixture.
2. canonical post-graduation pool = busiest pool over 24h -> live uses the
   migration pool (passed in as `grad_pool_id`); the fixture measures how often
   they differ.
3. price_unit from full-history USD coverage -> only matters for quotes
   outside USD_QUOTES; live rule: `price_unit_live()`.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

WSOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
USD_QUOTES = (WSOL, USDC, USDT)   # = silver_token_trades.SOLANA_QUOTE_DENYLIST (tested)
MIN_TRADE_USD = 0.01              # curve trades only (curve_bars.py); postgrad bars keep dust
OUTLIER_BAND = 1000.0             # silver_price_bars.OUTLIER_BAND
POSTGRAD_HORIZON_HOURS = 24       # 00f HORIZON_HOURS
CURVE_PROJECTS = {"pumpdotfun": ("pumpdotfun", 1), "raydium_launchlab": ("raydium_launchlab", 1)}
POSTGRAD_PROJECTS = {"pumpdotfun": ("pumpswap", None), "raydium_launchlab": ("raydium", 5)}

SWAP_COLUMNS = [
    "block_time", "block_slot", "tx_index", "outer_instruction_index", "project", "version",
    "project_program_id", "token_bought_mint_address", "token_sold_mint_address",
    "token_bought_amount", "token_sold_amount", "amount_usd", "trader_id",
]


def price_unit_live(primary_quote: str) -> str:
    """'usd' when the token's primary quote is SOL/USDC/USDT (every pump.fun
    token, most LaunchLab tokens), else 'quote'. Training also used 'usd' for
    other quotes at >= 95% full-history USD coverage -- not knowable live."""
    return "usd" if primary_quote in USD_QUOTES else "quote"


def _legs(sw: pd.DataFrame, mint: str) -> pd.DataFrame:
    """Per-trade values relative to the launched token (base = mint)."""
    buy = sw["token_bought_mint_address"] == mint
    sell = sw["token_sold_mint_address"] == mint
    sw = sw[buy | sell].copy()
    buy = sw["token_bought_mint_address"] == mint
    base_amt = np.where(buy, sw["token_bought_amount"], sw["token_sold_amount"]).astype(float)
    quote_amt = np.where(buy, sw["token_sold_amount"], sw["token_bought_amount"]).astype(float)
    sw["is_buy"] = buy.values
    sw["quote_mint"] = np.where(buy, sw["token_sold_mint_address"], sw["token_bought_mint_address"])
    sw["quote_amount"] = quote_amt
    with np.errstate(divide="ignore", invalid="ignore"):
        sw["price_usd"] = np.where(base_amt > 0, sw["amount_usd"].astype(float) / base_amt, np.nan)
        sw["price_q"] = np.where(base_amt > 0, quote_amt / base_amt, np.nan)
    sw["order_key"] = (sw["block_slot"].astype(float) * 1e9 + sw["tx_index"].astype(float) * 1e3
                       + sw["outer_instruction_index"].astype(float))
    sw["bar_start"] = pd.to_datetime(sw["block_time"], utc=True).dt.floor("min")
    return sw


def _agg(g: pd.DataFrame) -> pd.Series:
    """One bar = Dune's _bars_select / _curve_bars_select over one minute."""
    g = g.sort_values("order_key")
    pu, pq = g["price_usd"].dropna(), g["price_q"].dropna()
    priced = g["price_usd"].notna()
    return pd.Series({
        "open_usd": pu.iloc[0] if len(pu) else np.nan, "high_usd": pu.max() if len(pu) else np.nan,
        "low_usd": pu.min() if len(pu) else np.nan, "close_usd": pu.iloc[-1] if len(pu) else np.nan,
        "open_q": pq.iloc[0] if len(pq) else np.nan, "high_q": pq.max() if len(pq) else np.nan,
        "low_q": pq.min() if len(pq) else np.nan, "close_q": pq.iloc[-1] if len(pq) else np.nan,
        "volume_usd": g.loc[priced, "amount_usd"].sum() if priced.any() else np.nan,
        "n_trades": len(g), "n_buys": int(g["is_buy"].sum()),
    })


def _dune_bars(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    # Dune groups by (pool, base, quote, minute); one pool per phase here, and
    # a pool has one quote, so group by minute + quote.
    return trades.groupby(["bar_start", "quote_mint"], sort=True).apply(_agg, include_groups=False).reset_index()


def _expanding_median(x: pd.Series) -> pd.Series:
    return x.expanding().median()


def _silver_pick(b: pd.DataFrame, price_unit: str, primary_quote: str, band_mode: str) -> pd.DataFrame:
    """silver_price_bars.postgrad_bars for one token-phase."""
    if b.empty:
        return b
    b = b.sort_values("bar_start").copy()
    same = b["quote_mint"] == primary_quote
    for k in ("open", "high", "low", "close"):
        v = b[f"{k}_usd"] if price_unit == "usd" else b[f"{k}_q"].where(same)
        b[f"_{k[0]}"] = v.where(v > 0)
    b = b[b["_c"].notna()]
    if b.empty:
        return b
    # Silver: drop bars whose close is > 1000x from the median close, then
    # re-take the median on the KEPT bars and blank out-of-band open/high/low.
    # training = full-history medians (as Silver); live = expanding medians
    # (only bars up to and including this one -- point-in-time).
    med = b["_c"].median() if band_mode == "training" else _expanding_median(b["_c"])
    inband = (b["_c"] <= med * OUTLIER_BAND) & (b["_c"] >= med / OUTLIER_BAND)
    b = b[inband].copy()
    med = b["_c"].median() if band_mode == "training" else _expanding_median(b["_c"])
    for c in ("_o", "_h", "_l"):
        ok = (b[c] <= med * OUTLIER_BAND) & (b[c] >= med / OUTLIER_BAND)
        b[c] = b[c].where(ok)
    o = b["_o"].fillna(b["_c"])
    b["open"] = o
    b["high"] = b["_h"].fillna(np.maximum(o, b["_c"]))
    b["low"] = b["_l"].fillna(np.minimum(o, b["_c"]))
    b["close"] = b["_c"]
    return b


def build_bars(
    swaps: pd.DataFrame,
    *,
    mint: str,
    launch_venue: str,
    primary_quote: str,
    grad_pool_id: str | None = None,
    grad_time: Any = None,
    price_unit: str | None = None,
    band_mode: str = "live",
) -> pd.DataFrame:
    """Raw swaps for ONE token -> bars with contract_v1.BAR_COLUMNS (+ open,
    price_unit, venue, pool_id).

    - curve phase: trades on the launch venue's curve program whose quote is
      `primary_quote`; dust (amount_usd < $0.01) dropped, NULL USD kept.
    - postgrad phase: trades in `grad_pool_id` from `grad_time` (pool
      creation for PumpSwap; first CPMM trade for LaunchLab) for 24h.
      Dust is NOT dropped (Dune aggregated it in); the outlier band handles it.
    - Empty minutes produce no bar (no carry-forward).
    - A minute with both phases keeps the bar with more trades (tie -> curve).
    """
    if band_mode not in ("training", "live"):
        raise ValueError("band_mode must be 'training' or 'live'")
    unit = price_unit or price_unit_live(primary_quote)
    sw = swaps.copy()
    sw["block_time"] = pd.to_datetime(sw["block_time"], utc=True)
    proj, ver = CURVE_PROJECTS[launch_venue]
    cur = sw[(sw["project"] == proj) & ((sw["version"] == ver) if ver is not None else True)]
    cur = _legs(cur, mint)
    if launch_venue == "pumpdotfun":
        cur = cur[cur["quote_mint"] == WSOL]
    else:
        cur = cur[cur["quote_mint"] == primary_quote]
    cur = cur[cur["amount_usd"].isna() | (cur["amount_usd"] >= MIN_TRADE_USD)]
    parts = []
    cb = _silver_pick(_dune_bars(cur), unit, primary_quote, band_mode)
    if not cb.empty:
        parts.append(cb.assign(phase="curve", venue=launch_venue, pool_id=mint if launch_venue == "pumpdotfun" else None))
    if grad_pool_id is not None:
        gproj, gver = POSTGRAD_PROJECTS[launch_venue]
        g0 = pd.Timestamp(grad_time)
        g0 = g0.tz_localize("UTC") if g0.tzinfo is None else g0.tz_convert("UTC")
        pg = sw[(sw["project"] == gproj) & (sw["project_program_id"] == grad_pool_id)
                & ((sw["version"] == gver) if gver is not None else True)
                & (sw["block_time"] >= g0) & (sw["block_time"] < g0 + timedelta(hours=POSTGRAD_HORIZON_HOURS))]
        pb = _silver_pick(_dune_bars(_legs(pg, mint)), unit, primary_quote, band_mode)
        if not pb.empty:
            parts.append(pb.assign(phase="postgrad", venue=gproj if gproj != "raydium" else "raydium_cpmm",
                                   pool_id=grad_pool_id))
    if not parts:
        return pd.DataFrame(columns=["bar_start", "open", "high", "low", "close", "volume_usd",
                                     "n_trades", "n_buys", "phase", "price_unit", "venue", "pool_id"])
    allb = pd.concat(parts, ignore_index=True)
    allb = allb.sort_values(["bar_start", "n_trades", "phase"], ascending=[True, False, True])
    allb = allb.drop_duplicates("bar_start", keep="first")
    allb["price_unit"] = unit
    return allb[["bar_start", "open", "high", "low", "close", "volume_usd", "n_trades", "n_buys",
                 "phase", "price_unit", "venue", "pool_id"]].reset_index(drop=True)
