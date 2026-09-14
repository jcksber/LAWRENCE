"""
Pure-Python Silver transform for `lawrence.silver.token_trades`.

Turns one raw two-sided Bronze row (token A <-> token B) into zero, one, or
two Silver rows -- one per side that isn't a known quote/reference asset
("the token of interest"). See docs/silver_spec.md for the schema and the
dedup/denylist rationale.

Deliberately has no PySpark/Databricks dependency, matching
src/ingestion/bitquery_client.py's split: this module is the pure logic,
notebooks/01_silver_token_trades.py is the Spark glue that reads Bronze,
calls this, and merges into Silver. Keeps this unit-testable with plain
pytest (see tests/test_silver_token_trades.py) and runnable in CI, which
has no PySpark/Databricks runtime available.
"""
from __future__ import annotations

from typing import Any

# Phase 1 quote/reference-asset denylist. Scoped to the two live venues
# (raydium_v4/Solana, uniswap_v3_base/EVM) -- add an entry here whenever a
# new venue's quote asset shows up. Solana addresses are base58 and matched
# exact-case; EVM addresses are matched lowercase (Bitquery has returned
# them lowercase so far, but don't rely on that from an unverified source).
SOLANA_QUOTE_DENYLIST: set[str] = {
    "So11111111111111111111111111111111111111112",  # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}
EVM_QUOTE_DENYLIST: set[str] = {
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",  # USDC (Base)
    "0x4200000000000000000000000000000000000006",  # WETH (Base)
}


def _is_quote(chain_family: str, token_id: str) -> bool:
    if chain_family == "solana":
        return token_id in SOLANA_QUOTE_DENYLIST
    if chain_family == "evm":
        return token_id.lower() in EVM_QUOTE_DENYLIST
    raise ValueError(f"Unknown chain_family: {chain_family!r}")


def _to_float(x: Any) -> float | None:
    """Bitquery's GraphQL responses serialize decimals as JSON strings;
    Bronze carries them through as-is. Cast defensively rather than assume
    a type. None stays None (e.g. tx_fee columns on pre-fee-field Bronze
    rows written before this column existed)."""
    return None if x is None else float(x)


def to_silver_rows(bronze_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one Bronze row into 0, 1, or 2 Silver rows.

    - Exactly one side is a quote asset (the common case, e.g. TOKEN/WSOL):
      one row, for the non-quote side.
    - Both sides are quote assets (e.g. a USDC/WSOL swap): zero rows --
      there's no "token of interest" to score.
    - Neither side is on the denylist (two candidate tokens traded directly
      against each other): two rows, one per side, each treating the other
      as its quote asset. This case isn't spelled out verbatim in
      docs/silver_spec.md's schema table, but follows from applying the
      denylist rule per-side rather than per-trade, and is why the real
      dedup key is (dex_venue, tx_hash, token_id), not just
      (dex_venue, tx_hash) -- a single tx_hash can produce two rows here.
    """
    chain_family = bronze_row["chain_family"]
    buy_is_quote = _is_quote(chain_family, bronze_row["buy_token_id"])
    sell_is_quote = _is_quote(chain_family, bronze_row["sell_token_id"])

    sides: list[tuple[str, str]] = []
    if not buy_is_quote:
        sides.append(("buy", "sell"))
    if not sell_is_quote:
        sides.append(("sell", "buy"))

    tx_fee_usd = _to_float(bronze_row.get("tx_fee_usd"))
    chain_native_fee = _to_float(bronze_row.get("tx_fee_native"))

    rows = []
    for side, quote_side in sides:
        amount = _to_float(bronze_row[f"{side}_amount"])
        amount_usd = _to_float(bronze_row[f"{side}_amount_usd"])
        if not amount:  # None or 0.0 -- can't compute a price from it
            continue
        rows.append(
            {
                "token_id": bronze_row[f"{side}_token_id"],
                "chain_family": chain_family,
                "network": bronze_row.get("network"),
                "dex_venue": bronze_row["dex_venue"],
                "dex_protocol": bronze_row["dex_protocol"],
                "trade_time": bronze_row["block_time"],
                "side": side,
                "price_usd": amount_usd / amount,
                "amount_usd": amount_usd,
                "tx_fee_usd": tx_fee_usd,
                "chain_native_fee": chain_native_fee,
                "tx_hash": bronze_row["tx_hash"],
                "quote_token_id": bronze_row[f"{quote_side}_token_id"],
            }
        )
    return rows
