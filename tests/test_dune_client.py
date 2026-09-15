"""
Unit tests for src/ingestion/dune_client.py.

`to_bronze_row` is pure Python -- tested directly, no mocking needed.
The HTTP mechanics (_submit / _wait_until_done / _fetch_all_rows) are
tested with unittest.mock (stdlib, no network) since this sandbox has no
PyPI egress for a mocking library -- see docs/decisions.md re: pytest
unavailability here; both run for real under pytest in CI.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ingestion.bitquery_client import ChainFamily, Venue
from ingestion.dune_client import DuneClient, DuneVenueFilter, to_bronze_row

INGESTED_AT = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

RAYDIUM_V4 = Venue(
    name="raydium_v4",
    chain_family=ChainFamily.SOLANA,
    dex_program_or_address="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
)
UNISWAP_V3_BASE = Venue(
    name="uniswap_v3_base",
    chain_family=ChainFamily.EVM,
    network="base",
    dex_program_or_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
)


def _solana_row(**overrides):
    row = {
        "blockchain": "solana",
        "project": "raydium",
        "version": 4,
        "version_name": "amm",
        "block_time": "2026-09-13 00:58:59.000 UTC",
        "token_bought_symbol": "WSOL",
        "token_sold_symbol": "USDC",
        "token_bought_amount": 0.489187368,
        "token_sold_amount": 50.0,
        "amount_usd": 49.750355325600005,
        "fee_tier": None,
        "fee_usd": None,
        "token_bought_mint_address": "So11111111111111111111111111111111111111112",
        "token_sold_mint_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "project_program_id": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
        "tx_id": "6Vo3245eszAb5wuqEMw8mGdbfRUdKbHhDHP5LcaGuTAB",
    }
    row.update(overrides)
    return row


def _evm_row(**overrides):
    row = {
        "blockchain": "base",
        "project": "uniswap",
        "version": 3,
        "block_time": "2026-09-13 00:01:29.000 UTC",
        "tx_hash": "0x710318b621519cf51584dd90d1fce74d6214855aaa84aa0d1afccd92470f16d2",
        "token_bought_symbol": "WETH",
        "token_sold_symbol": "USDC",
        "token_bought_amount": 0.010060458242980321,
        "token_sold_amount": 25.422973,
        "amount_usd": 25.41181208052642,
        "token_bought_address": "0x4200000000000000000000000000000000000006",
        "token_sold_address": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
    }
    row.update(overrides)
    return row


def test_solana_row_maps_program_id_venue_correctly():
    bronze = to_bronze_row(_solana_row(), RAYDIUM_V4, INGESTED_AT)
    assert bronze["chain_family"] == "solana"
    assert bronze["network"] is None
    assert bronze["dex_venue"] == "raydium_v4"
    assert bronze["tx_hash"] == "6Vo3245eszAb5wuqEMw8mGdbfRUdKbHhDHP5LcaGuTAB"
    assert bronze["buy_token_id"] == "So11111111111111111111111111111111111111112"
    assert bronze["sell_token_id"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    assert bronze["dex_protocol"] == "raydium_amm"
    assert bronze["source"] == "dune"
    assert bronze["ingested_at"] == INGESTED_AT


def test_evm_row_maps_correctly():
    bronze = to_bronze_row(_evm_row(), UNISWAP_V3_BASE, INGESTED_AT)
    assert bronze["chain_family"] == "evm"
    assert bronze["network"] == "base"
    assert bronze["dex_venue"] == "uniswap_v3_base"
    assert bronze["dex_protocol"] == "uniswap_v3"
    assert bronze["buy_token_id"] == "0x4200000000000000000000000000000000000006"


def test_single_amount_usd_is_copied_to_both_sides():
    # Dune has one amount_usd per trade, not one per side -- see the
    # module's KNOWN APPROXIMATION.
    bronze = to_bronze_row(_solana_row(), RAYDIUM_V4, INGESTED_AT)
    assert bronze["buy_amount_usd"] == bronze["sell_amount_usd"] == 49.750355325600005


def test_fee_fields_are_always_none():
    # KNOWN GAP -- neither Dune table has a usable on-chain tx fee.
    solana_bronze = to_bronze_row(_solana_row(), RAYDIUM_V4, INGESTED_AT)
    evm_bronze = to_bronze_row(_evm_row(), UNISWAP_V3_BASE, INGESTED_AT)
    assert solana_bronze["tx_fee_native"] is None
    assert solana_bronze["tx_fee_usd"] is None
    assert evm_bronze["tx_fee_native"] is None
    assert evm_bronze["tx_fee_usd"] is None


def test_null_amount_usd_row_passes_through_as_none():
    # Confirmed on real sampled Base/Uniswap rows: illiquid/exotic tokens
    # can have no USD price at all. to_bronze_row shouldn't crash on it --
    # Silver's to_silver_rows is what's responsible for dropping the row.
    bronze = to_bronze_row(_evm_row(amount_usd=None), UNISWAP_V3_BASE, INGESTED_AT)
    assert bronze["buy_amount_usd"] is None
    assert bronze["sell_amount_usd"] is None


def _mock_response(json_body, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_body
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = Exception("http error")
    return resp


def test_wait_until_done_accepts_completed_state():
    client = DuneClient(api_key="fake", poll_interval_s=0, poll_timeout_s=5)
    with patch("ingestion.dune_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"state": "QUERY_STATE_COMPLETED"})
        client._wait_until_done("exec123")  # should not raise


def test_wait_until_done_accepts_plain_success_state():
    # Dune's own docs disagree on the enum spelling -- this must accept
    # both, see module docstring's UNVERIFIED note.
    client = DuneClient(api_key="fake", poll_interval_s=0, poll_timeout_s=5)
    with patch("ingestion.dune_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"state": "success"})
        client._wait_until_done("exec123")  # should not raise


def test_wait_until_done_raises_on_failed_state():
    client = DuneClient(api_key="fake", poll_interval_s=0, poll_timeout_s=5)
    with patch("ingestion.dune_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"state": "QUERY_STATE_FAILED"})
        try:
            client._wait_until_done("exec123")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


def test_wait_until_done_times_out_on_stuck_pending():
    client = DuneClient(api_key="fake", poll_interval_s=0, poll_timeout_s=0.05)
    with patch("ingestion.dune_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"state": "QUERY_STATE_PENDING"})
        try:
            client._wait_until_done("exec123")
            assert False, "expected TimeoutError"
        except TimeoutError:
            pass


def test_fetch_all_rows_paginates_past_page_size():
    client = DuneClient(api_key="fake", page_size=2)
    page1 = _mock_response({"result": {"rows": [{"a": 1}, {"a": 2}]}})
    page2 = _mock_response({"result": {"rows": [{"a": 3}]}})
    with patch("ingestion.dune_client.requests.get", side_effect=[page1, page2]):
        rows = client._fetch_all_rows("exec123")
    assert rows == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_fetch_all_rows_single_partial_page_stops_immediately():
    client = DuneClient(api_key="fake", page_size=5000)
    only_page = _mock_response({"result": {"rows": [{"a": 1}]}})
    with patch("ingestion.dune_client.requests.get", return_value=only_page):
        rows = client._fetch_all_rows("exec123")
    assert rows == [{"a": 1}]
