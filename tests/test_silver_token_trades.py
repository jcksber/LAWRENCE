"""
Unit tests for src/features/silver_token_trades.py -- the pure-Python
Bronze -> Silver row-shaping logic (quote-asset denylist, price/amount
normalization, fee passthrough). No PySpark/Databricks dependency, so
these run in plain CI.
"""
from features.silver_token_trades import to_silver_rows


def _bronze_row(**overrides):
    row = {
        "chain_family": "solana",
        "network": None,
        "dex_venue": "raydium_v4",
        "dex_protocol": "raydium_amm",
        "block_time": "2026-09-14T01:50:48Z",
        "tx_hash": "sig123",
        "buy_token_id": "So11111111111111111111111111111111111111112",  # WSOL
        "buy_amount": "0.153893717",
        "buy_amount_usd": "15.307589",
        "sell_token_id": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "sell_amount": "15.359219",
        "sell_amount_usd": "15.357402",
        "tx_fee_native": "0.000017540",
        "tx_fee_usd": "0.0017446788",
    }
    row.update(overrides)
    return row


def test_one_side_quote_one_side_token_of_interest():
    # WSOL (quote, buy side) <-> USDC (also quote, sell side) -- both are
    # on the denylist, so this specific fixture should yield zero rows.
    # Swap USDC out for a real candidate token to test the common case.
    row = _bronze_row(
        sell_token_id="SomeMintAddressForARealToken111111111111111",
        sell_symbol="MEME",
    )
    rows = to_silver_rows(row)
    assert len(rows) == 1
    r = rows[0]
    assert r["side"] == "sell"  # the non-quote side was the sell side
    assert r["token_id"] == "SomeMintAddressForARealToken111111111111111"
    assert r["quote_token_id"] == "So11111111111111111111111111111111111111112"
    assert r["price_usd"] == 15.357402 / 15.359219
    assert r["amount_usd"] == 15.357402
    assert r["tx_fee_usd"] == 0.0017446788
    assert r["chain_native_fee"] == 0.000017540
    assert r["tx_hash"] == "sig123"
    assert r["dex_venue"] == "raydium_v4"


def test_both_sides_quote_assets_yields_zero_rows():
    row = _bronze_row()  # WSOL <-> USDC, both on the denylist
    assert to_silver_rows(row) == []


def test_both_sides_non_quote_yields_two_rows():
    row = _bronze_row(
        buy_token_id="TokenA1111111111111111111111111111111111",
        sell_token_id="TokenB1111111111111111111111111111111111",
    )
    rows = to_silver_rows(row)
    assert len(rows) == 2
    sides = {r["side"] for r in rows}
    assert sides == {"buy", "sell"}
    token_ids = {r["token_id"] for r in rows}
    assert token_ids == {"TokenA1111111111111111111111111111111111", "TokenB1111111111111111111111111111111111"}
    # each treats the other as its quote
    by_side = {r["side"]: r for r in rows}
    assert by_side["buy"]["quote_token_id"] == "TokenB1111111111111111111111111111111111"
    assert by_side["sell"]["quote_token_id"] == "TokenA1111111111111111111111111111111111"


def test_evm_denylist_is_case_insensitive():
    row = _bronze_row(
        chain_family="evm",
        network="base",
        dex_venue="uniswap_v3_base",
        dex_protocol="uniswap_v3",
        buy_token_id="0x4200000000000000000000000000000000000006",  # WETH, lowercase
        sell_token_id="0xACFE6019ED1A7DC6F7B508C02D1B04EC88CC21BF",  # candidate token, uppercase hex
    )
    rows = to_silver_rows(row)
    assert len(rows) == 1
    assert rows[0]["token_id"] == "0xACFE6019ED1A7DC6F7B508C02D1B04EC88CC21BF"


def test_zero_amount_side_is_skipped_not_a_zero_division_error():
    row = _bronze_row(
        sell_token_id="SomeMintAddressForARealToken111111111111111",
        sell_amount="0",
        sell_amount_usd="0",
    )
    assert to_silver_rows(row) == []


def test_missing_fee_columns_pass_through_as_none():
    # Pre-fee-field Bronze rows (written before the fee columns existed)
    # have tx_fee_native/tx_fee_usd = NULL, not a KeyError.
    row = _bronze_row(sell_token_id="SomeMintAddressForARealToken111111111111111")
    del row["tx_fee_native"]
    del row["tx_fee_usd"]
    rows = to_silver_rows(row)
    assert len(rows) == 1
    assert rows[0]["tx_fee_usd"] is None
    assert rows[0]["chain_native_fee"] is None
