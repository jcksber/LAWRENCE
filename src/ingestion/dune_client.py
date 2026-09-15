"""
Dune Analytics client -- ONE-TIME HISTORICAL BACKFILL only, not live
ingestion. Bitquery's `dataset: archive` requires an Enterprise plan
(confirmed live, 403 on the self-serve tiers -- see docs/decisions.md,
2026-09-15); Dune's curated `dex.trades` (EVM) / `dex_solana.trades`
(Solana) tables cover historical DEX trades without that gate, confirmed
live against real bounded queries for both `raydium_v4` (exact
`project_program_id` match) and `uniswap_v3_base` (`project`+`version`
match -- EVM rows carry per-pool addresses, not the shared factory
address our `Venue.dex_program_or_address` stores, so filtering has to go
through the DEX/version instead of an address match on this side).

KNOWN GAP: neither table carries a usable on-chain transaction fee.
`dex.trades` has no fee column at all; `dex_solana.trades` has
`fee_tier`/`fee_usd`, but both were blank across every sampled row
(AMM/CLMM/CPMM versions alike), and even populated they'd be the DEX's own
swap fee, not the wallet's paid transaction/priority fee our Bronze schema
tracks. `to_bronze_row()` below sets `tx_fee_native`/`tx_fee_usd` = None
for every row it produces -- a documented gap, not a silent one. Live rows
from `bitquery_client.py` are unaffected.

KNOWN APPROXIMATION: Dune reports one `amount_usd` per trade, not a
separate USD value per side the way Bitquery does. `to_bronze_row()` uses
that same value for both `buy_amount_usd` and `sell_amount_usd` -- for a
real swap these should be nearly equal anyway, and Silver's
`price_usd = amount_usd / amount` is computed per side using each side's
own token amount, so this doesn't distort the derived price.

This client is not used for live/production ingestion -- that stays on
Bitquery's realtime tier. Run manually from
notebooks/00c_bronze_backfill_dune.py.

Auth: set DUNE_API_KEY as a Databricks secret
(`databricks secrets put-secret lawrence dune_api_key`).

UNVERIFIED: Dune's own docs disagree with themselves on the execution
`state` enum -- one page shows QUERY_STATE_PENDING/EXECUTING/COMPLETED/
FAILED, another shows plain pending/success/failure. `_wait_until_done`
below checks case-insensitive substrings ("COMPLET"/"SUCCESS" -> done,
"FAIL"/"CANCEL" -> error) rather than trusting one exact literal, precisely
because this hasn't been confirmed against a real response yet. Watch the
first real run's printed state values; tighten this if neither pattern
matches what actually comes back.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from .bitquery_client import ChainFamily, Venue

DUNE_API_BASE = "https://api.dune.com/api/v1"


@dataclass(frozen=True)
class DuneVenueFilter:
    """How to filter Dune's curated trade tables down to one Venue, keyed
    by Venue.name in the caller. A SQL WHERE fragment rather than a single
    generic filter, because Dune's schema differs fundamentally by chain
    (Solana carries an exact program address; EVM rows carry per-pool
    addresses only, never the shared factory address)."""

    table: str  # "dex_solana.trades" or "dex.trades"
    where_sql: str  # ANDed with the time-window filter; no trailing AND/OR


class DuneClient:
    """Submits one bounded SQL query via Dune's Execute SQL endpoint, polls
    until it finishes, and pages through all result rows. No bisection on
    a row cap the way bitquery_client's live fetch does -- caller is
    responsible for chunking the overall backfill window into pieces small
    enough to keep each query's result set reasonable."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        timeout_s: int = 30,
        poll_interval_s: float = 2.0,
        poll_timeout_s: float = 180.0,
        page_size: int = 5000,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self.page_size = page_size

    def _headers(self) -> dict[str, str]:
        return {"X-Dune-Api-Key": self.api_key, "Content-Type": "application/json"}

    def fetch_historical_trades(
        self,
        venue_filter: DuneVenueFilter,
        since_iso: str,
        till_iso: str,
        performance: str = "medium",
    ) -> list[dict[str, Any]]:
        """Runs one bounded [since_iso, till_iso) query and returns every
        result row (paginated automatically past `page_size`)."""
        sql = (
            f"SELECT * FROM {venue_filter.table} "
            f"WHERE block_time >= TIMESTAMP '{since_iso}' "
            f"AND block_time < TIMESTAMP '{till_iso}' "
            f"AND {venue_filter.where_sql}"
        )
        execution_id = self._submit(sql, performance)
        self._wait_until_done(execution_id)
        return self._fetch_all_rows(execution_id)

    def _submit(self, sql: str, performance: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{DUNE_API_BASE}/sql/execute",
                    json={"sql": sql, "performance": performance},
                    headers=self._headers(),
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                return resp.json()["execution_id"]
            except (requests.RequestException, KeyError) as exc:
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            f"Dune query submission failed after {self.max_retries} attempts"
        ) from last_err

    def _wait_until_done(self, execution_id: str) -> None:
        deadline = time.monotonic() + self.poll_timeout_s
        while time.monotonic() < deadline:
            resp = requests.get(
                f"{DUNE_API_BASE}/execution/{execution_id}/status",
                headers=self._headers(),
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            state = str(resp.json().get("state", "")).upper()
            if "COMPLET" in state or "SUCCESS" in state:
                return
            if "FAIL" in state or "CANCEL" in state:
                raise RuntimeError(f"Dune execution {execution_id} failed: state={state}")
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"Dune execution {execution_id} did not finish within {self.poll_timeout_s}s"
        )

    def _fetch_all_rows(self, execution_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            resp = requests.get(
                f"{DUNE_API_BASE}/execution/{execution_id}/results",
                headers=self._headers(),
                params={"limit": self.page_size, "offset": offset},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            page_rows = resp.json().get("result", {}).get("rows", [])
            rows.extend(page_rows)
            if len(page_rows) < self.page_size:
                break
            offset += self.page_size
        return rows


def to_bronze_row(
    dune_row: dict[str, Any], venue: Venue, ingested_at: datetime
) -> dict[str, Any]:
    """Maps one Dune `dex.trades`/`dex_solana.trades` row into Bronze's
    schema shape, so the existing Silver transform (`to_silver_rows`) runs
    on backfilled rows exactly as it does on live Bitquery rows. Pure
    Python, no PySpark -- see tests/test_dune_client.py."""
    is_evm = venue.chain_family == ChainFamily.EVM
    if is_evm:
        buy_token_id = dune_row["token_bought_address"]
        sell_token_id = dune_row["token_sold_address"]
        tx_hash = dune_row["tx_hash"]
        dex_protocol = f"{dune_row['project']}_v{dune_row['version']}"
    else:
        buy_token_id = dune_row["token_bought_mint_address"]
        sell_token_id = dune_row["token_sold_mint_address"]
        tx_hash = dune_row["tx_id"]
        version_label = dune_row.get("version_name") or dune_row.get("version")
        dex_protocol = f"{dune_row['project']}_{version_label}"

    # One `amount_usd` per trade on Dune, not a separate value per side --
    # see module docstring's KNOWN APPROXIMATION.
    amount_usd = dune_row.get("amount_usd")

    return {
        "chain_family": venue.chain_family.value,
        "network": venue.network,
        "dex_venue": venue.name,
        "block_time": dune_row["block_time"],
        "tx_hash": tx_hash,
        "dex_protocol": dex_protocol,
        "buy_token_id": buy_token_id,
        "buy_symbol": dune_row["token_bought_symbol"],
        "buy_amount": dune_row["token_bought_amount"],
        "buy_amount_usd": amount_usd,
        "sell_token_id": sell_token_id,
        "sell_symbol": dune_row["token_sold_symbol"],
        "sell_amount": dune_row["token_sold_amount"],
        "sell_amount_usd": amount_usd,
        # KNOWN GAP -- see module docstring. Always None for Dune-sourced
        # rows; live Bitquery rows still carry real fees.
        "tx_fee_native": None,
        "tx_fee_usd": None,
        "raw_json": json.dumps(dune_row, default=str),
        "source": "dune",
        "ingested_at": ingested_at,
    }
