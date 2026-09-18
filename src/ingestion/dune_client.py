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

COST INCIDENT, 2026-09-15: a first real run selected all ~30 columns
(`SELECT *`) chunked by day, and blew through a 2,500-credit free trial
partway through ONE day/venue chunk's pagination (past row offset
415,000 before a 402) -- 7 full prior days had actually fetched fine.
None of it reached Bronze: the notebook (then) wrote once at the end, and
the 402 crashed it first. The earlier cost-test that suggested this
would be cheap only measured tiny 1-hour-window aggregate queries
(COUNT(*)/GROUP BY), not full-column trade-level SELECTs at real
top-venue volume -- a materially different cost profile.

Changes as a result: `DuneVenueFilter` now carries an explicit `columns`
list (only what `to_bronze_row` needs) instead of `SELECT *`;
`DuneCreditsExhausted` is raised immediately on a 402 (never retried) so
a caller can abort the whole run instead of burning retry time on every
remaining chunk, AND carries whatever rows had already been fetched from
earlier pages of the SAME query (`partial_rows`) so a 402 mid-pagination
loses at most the one in-flight page, not the whole chunk. The backfill
notebook writes to Bronze after every chunk (and after a partial
failure) rather than once at the end, and tracks completed chunks in a
small progress table so a re-run -- on the same API key or a different
one -- skips whatever's already landed instead of re-paying for it.
Re-measure real per-query cost at this narrower column set on a small
window before sizing anything close to the original 30-day pull.

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
from datetime import datetime, timezone
from typing import Any

import requests

from .bitquery_client import ChainFamily, Venue

DUNE_API_BASE = "https://api.dune.com/api/v1"


class DuneCreditsExhausted(RuntimeError):
    """Raised immediately on a 402 Payment Required, never retried --
    retrying a balance failure wastes time without changing the outcome,
    and every other in-flight/queued chunk will fail the same way until
    credits are topped up. Callers (the backfill notebook) should catch
    this SEPARATELY from RuntimeError and abort the whole run, not just
    skip the one chunk -- see the module's COST INCIDENT note.

    `partial_rows` carries whatever rows `_fetch_all_rows` had already
    fetched from earlier PAGES of the same query before a later page hit
    402 -- e.g. pages 1-3 succeeded (15,000 rows) and page 4 failed: those
    15,000 rows are real, paid-for data and shouldn't be discarded just
    because the query didn't finish. Empty when nothing was fetched yet
    (submission itself failed, or the very first page did)."""

    def __init__(self, message: str, partial_rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.partial_rows: list[dict[str, Any]] = partial_rows or []


@dataclass(frozen=True)
class DuneVenueFilter:
    """How to filter Dune's curated trade tables down to one Venue, keyed
    by Venue.name in the caller. A SQL WHERE fragment rather than a single
    generic filter, because Dune's schema differs fundamentally by chain
    (Solana carries an exact program address; EVM rows carry per-pool
    addresses only, never the shared factory address).

    `columns` is explicit -- NOT `SELECT *` -- because pulling every
    column at real trade volume is a real, measured cost problem (see
    module docstring's COST INCIDENT). List only what `to_bronze_row`
    actually reads for this chain family."""

    table: str  # "dex_solana.trades" or "dex.trades"
    where_sql: str  # ANDed with the time-window filter; no trailing AND/OR
    columns: list[str]


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

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        """Shared HTTP-with-retry for every Dune API call this client
        makes. A 402 (out of credits) is raised immediately, never
        retried -- see DuneCreditsExhausted. Anything else transient
        (timeouts, connection errors, other non-2xx) gets the normal
        retry-with-backoff treatment, then wraps into RuntimeError so
        every call site raises the same, catchable error type instead of
        a raw requests exception leaking out uncaught (the actual bug
        behind the 2026-09-15 incident -- only _submit had this before)."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.request(
                    method, url, headers=self._headers(), timeout=self.timeout_s, **kwargs
                )
                if resp.status_code == 402:
                    raise DuneCreditsExhausted(
                        f"Dune returned 402 Payment Required for {method} {url} "
                        "-- credits exhausted, not retrying"
                    )
                resp.raise_for_status()
                return resp.json()
            except DuneCreditsExhausted:
                raise  # never retry a balance failure
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            f"Dune {method} {url} failed after {self.max_retries} attempts"
        ) from last_err

    def fetch_historical_trades(
        self,
        venue_filter: DuneVenueFilter,
        since_iso: str,
        till_iso: str,
        performance: str = "medium",
    ) -> list[dict[str, Any]]:
        """Runs one bounded [since_iso, till_iso) query and returns every
        result row (paginated automatically past `page_size`). Selects
        only `venue_filter.columns` -- not `SELECT *` -- see
        DuneVenueFilter's docstring."""
        columns_sql = ", ".join(venue_filter.columns)
        sql = (
            f"SELECT {columns_sql} FROM {venue_filter.table} "
            f"WHERE block_time >= TIMESTAMP '{since_iso}' "
            f"AND block_time < TIMESTAMP '{till_iso}' "
            f"AND {venue_filter.where_sql}"
        )
        return self.run_sql(sql, performance)

    def run_sql(self, sql: str, performance: str = "medium") -> list[dict[str, Any]]:
        """Runs arbitrary raw SQL and returns every result row -- the
        general-purpose escape hatch `fetch_historical_trades` builds on
        top of for the specific trade-row shape. Exists as its own public
        method (2026-09-18, see docs/decisions.md) for one-off diagnostic
        queries -- e.g. a cheap COUNT(*)/GROUP BY cost-test before sizing
        a real backfill expansion -- that don't fit `DuneVenueFilter`'s
        trade-row assumptions (a fixed column list, a `dex_protocol`-style
        `WHERE`). Caller is responsible for keeping the query itself cheap
        (aggregate, narrow time window) -- this method has no guardrail of
        its own beyond `DuneCreditsExhausted` surfacing a 402 immediately,
        same as `fetch_historical_trades`."""
        execution_id = self._submit(sql, performance)
        self._wait_until_done(execution_id)
        return self._fetch_all_rows(execution_id)

    def _submit(self, sql: str, performance: str) -> str:
        data = self._request(
            "POST",
            f"{DUNE_API_BASE}/sql/execute",
            json={"sql": sql, "performance": performance},
        )
        try:
            return data["execution_id"]
        except KeyError as exc:
            raise RuntimeError(f"Dune submit response missing execution_id: {data}") from exc

    def _wait_until_done(self, execution_id: str) -> None:
        deadline = time.monotonic() + self.poll_timeout_s
        while time.monotonic() < deadline:
            data = self._request("GET", f"{DUNE_API_BASE}/execution/{execution_id}/status")
            state = str(data.get("state", "")).upper()
            if "COMPLET" in state or "SUCCESS" in state:
                return
            if "FAIL" in state or "CANCEL" in state:
                # Surface Dune's actual error detail instead of discarding
                # it -- the status response carries an `error` object
                # (type/message/metadata.line/column) on failure, and
                # without it every failure looks identical ("state=FAILED")
                # with no way to tell a real SQL bug from a transient
                # engine issue. Confirmed via Dune's API docs (2026-09-15).
                error = data.get("error") or {}
                detail = error.get("message") or error.get("type") or "no error detail returned"
                raise RuntimeError(
                    f"Dune execution {execution_id} failed: state={state}: {detail}"
                )
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"Dune execution {execution_id} did not finish within {self.poll_timeout_s}s"
        )

    def _fetch_all_rows(self, execution_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            try:
                data = self._request(
                    "GET",
                    f"{DUNE_API_BASE}/execution/{execution_id}/results",
                    params={"limit": self.page_size, "offset": offset},
                )
            except DuneCreditsExhausted as exc:
                # Attach whatever earlier pages already succeeded before
                # re-raising -- see DuneCreditsExhausted's docstring. This
                # is the actual fix for the 2026-09-15 incident: that
                # failure happened mid-pagination (offset 415,000), and
                # every page before it was silently discarded.
                exc.partial_rows = rows
                raise
            page_rows = data.get("result", {}).get("rows", [])
            rows.extend(page_rows)
            # Pagination has no overall time cap (only each page's request
            # has one) -- a high-volume chunk can silently page for
            # minutes with nothing printed, indistinguishable from a hang
            # (2026-09-15: 10+ min of silence on a Sep-1 chunk during a
            # period of rising daily volume, never actually confirmed
            # either way). Print progress every few pages so this is never
            # ambiguous again.
            if offset % (self.page_size * 5) == 0:
                print(
                    f"    ...fetched {len(rows)} rows so far "
                    f"(execution {execution_id}, offset {offset})"
                )
            if len(page_rows) < self.page_size:
                break
            offset += self.page_size
        return rows


def _normalize_block_time(dune_block_time: str) -> str:
    """Dune's block_time comes back as `"YYYY-MM-DD HH:MM:SS.mmm UTC"`
    (confirmed on real rows, both chains) -- a different string shape
    than Bitquery's ISO-8601 `Z`-suffixed format Bronze already uses.
    Normalizing to match keeps `source='bitquery'` and `source='dune'`
    rows string-sortable/comparable the same way for any code doing
    range filtering on this column (e.g. Silver's lookback query)."""
    dt = datetime.strptime(dune_block_time.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(x: Any) -> float | None:
    """Dune's JSON responses mix Python int and float for the same
    numeric column depending on whether a given value happens to be a
    whole number (e.g. `50` vs `49.75`) -- a batch of rows with
    inconsistent int/float types for one column breaks Spark's schema
    inference (`CANNOT_MERGE_TYPE: DoubleType`/`LongType`, confirmed live
    2026-09-15). Cast every numeric field through this so the type is
    always float (or None), never left to whatever JSON happened to
    produce. Bitquery rows don't have this problem -- it serializes
    decimals as strings -- so this is Dune-specific."""
    return None if x is None else float(x)


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
    amount_usd = _to_float(dune_row.get("amount_usd"))

    return {
        "chain_family": venue.chain_family.value,
        "network": venue.network,
        "dex_venue": venue.name,
        "block_time": _normalize_block_time(dune_row["block_time"]),
        "tx_hash": tx_hash,
        "dex_protocol": dex_protocol,
        "buy_token_id": buy_token_id,
        "buy_symbol": dune_row["token_bought_symbol"],
        "buy_amount": _to_float(dune_row["token_bought_amount"]),
        "buy_amount_usd": amount_usd,
        "sell_token_id": sell_token_id,
        "sell_symbol": dune_row["token_sold_symbol"],
        "sell_amount": _to_float(dune_row["token_sold_amount"]),
        "sell_amount_usd": amount_usd,
        # KNOWN GAP -- see module docstring. Always None for Dune-sourced
        # rows; live Bitquery rows still carry real fees.
        "tx_fee_native": None,
        "tx_fee_usd": None,
        "raw_json": json.dumps(dune_row, default=str),
        "source": "dune",
        "ingested_at": ingested_at,
    }
