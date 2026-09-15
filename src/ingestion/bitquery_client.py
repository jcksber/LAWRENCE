"""
Chain-agnostic client for Bitquery's DEX Trades GraphQL API.

LAWRENCE is explicitly chain- and platform-agnostic: nothing in this module
assumes Solana, Raydium, or any single venue. A `Venue` just describes which
Bitquery query root + filter to use; add a new one to VENUES (in the calling
notebook/config) to pick up another chain or DEX with no code changes here.

Bitquery's schema uses one shared `DEXTrades` shape across chains, reached
through different root blocks:
  - EVM chains (ethereum, bsc, base, ...): `EVM(network: <name>) { DEXTrades }`
  - Solana:                                `Solana { DEXTrades }`
  - Others (Tron, etc.) follow the same per-family pattern; add a
    ChainFamily + template pair below when you need one.

Phase 1 scope: pull raw pool-creation + trade events into a Bronze-shaped
list of dicts. No cleaning, no joins, no feature engineering here.

Auth: set BITQUERY_API_KEY as a Databricks secret
(`databricks secrets put-secret lawrence bitquery_api_key`), read via
dbutils.secrets in the calling notebook, and pass it in here rather than
hardcoding it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

BITQUERY_ENDPOINT = "https://streaming.bitquery.io/graphql"


class ChainFamily(str, Enum):
    EVM = "evm"       # ethereum, bsc, base, polygon, arbitrum, ...
    SOLANA = "solana"
    # Add TRON, etc. here + a template below when a venue needs them.


@dataclass(frozen=True)
class Venue:
    """One tradeable venue: a chain (+ network, for EVM) and a DEX protocol
    or program to filter on. This is the *only* place venue identity lives —
    everything downstream (ingestion, Bronze schema, features) is generic."""

    name: str  # human label, e.g. "raydium_v4" or "uniswap_v3_base"
    chain_family: ChainFamily
    dex_program_or_address: str  # Solana: program address. EVM: DEX
                                   # protocol/factory address or name filter.
    network: str | None = None  # EVM only, e.g. "eth", "bsc", "base"


# EVM query: identical shape for every network, per Bitquery's cross-chain
# API — swap `$network` and you're on a different chain with no other change.
_EVM_QUERY = """
query NewPoolTrades($network: evm_network!, $program: String!, $since: DateTime!, $limit: Int!) {
  EVM(network: $network) {
    DEXTrades(
      limit: {count: $limit}
      orderBy: {descending: Block_Time}
      where: {
        Trade: {Dex: {OwnerAddress: {is: $program}}}
        Block: {Time: {since: $since}}
      }
    ) {
      Block { Time }
      Transaction { Hash }
      Trade {
        Buy  { Currency { SmartContract Symbol } Amount AmountInUSD }
        Sell { Currency { SmartContract Symbol } Amount AmountInUSD }
        Dex  { OwnerAddress ProtocolName }
      }
      joinTransactions(join: inner, Transaction_Hash: Transaction_Hash) {
        Fee { SenderFee SenderFeeInUSD }
      }
    }
  }
}
"""

_SOLANA_QUERY = """
query NewPoolTrades($program: String!, $since: DateTime!, $limit: Int!) {
  Solana {
    DEXTrades(
      limit: {count: $limit}
      orderBy: {descending: Block_Time}
      where: {
        Trade: {Dex: {ProgramAddress: {is: $program}}}
        Block: {Time: {since: $since}}
      }
    ) {
      Block { Time }
      Transaction { Signature Fee FeeInUSD }
      Trade {
        Buy  { Currency { MintAddress Symbol } Amount AmountInUSD }
        Sell { Currency { MintAddress Symbol } Amount AmountInUSD }
        Dex  { ProgramAddress ProtocolName }
        Market { MarketAddress }
      }
    }
  }
}
"""
# NOTE: verify field names against current Bitquery docs before relying on
# this in production — their schema shifts, and EVM `OwnerAddress` filtering
# in particular may need adjusting per-protocol (some are keyed by factory
# address, some by pool/pair address).

# Bounded historical backfill queries -- same shape as the live queries
# above (same response path, so the existing Bronze _normalize() works
# unchanged), but add `till` (a closed time window, not "everything since
# X") and `dataset: archive`. Confirmed via Bitquery's own docs examples
# (docs.bitquery.io/docs/schema/evm/dextrades/,
# .../Solana/historical-aggregate-data/) against DEXTradeByTokens, NOT yet
# confirmed against our actual DEXTrades root -- test with a small bounded
# window before trusting this for a real backfill; see notebooks/
# 00b_bronze_backfill_test.py.
_EVM_BACKFILL_QUERY = """
query HistoricalPoolTrades($network: evm_network!, $program: String!, $since: DateTime!, $till: DateTime!, $limit: Int!) {
  EVM(network: $network, dataset: archive) {
    DEXTrades(
      limit: {count: $limit}
      orderBy: {descending: Block_Time}
      where: {
        Trade: {Dex: {OwnerAddress: {is: $program}}}
        Block: {Time: {since: $since, till: $till}}
      }
    ) {
      Block { Time }
      Transaction { Hash }
      Trade {
        Buy  { Currency { SmartContract Symbol } Amount AmountInUSD }
        Sell { Currency { SmartContract Symbol } Amount AmountInUSD }
        Dex  { OwnerAddress ProtocolName }
      }
      joinTransactions(join: inner, Transaction_Hash: Transaction_Hash) {
        Fee { SenderFee SenderFeeInUSD }
      }
    }
  }
}
"""

_SOLANA_BACKFILL_QUERY = """
query HistoricalPoolTrades($program: String!, $since: DateTime!, $till: DateTime!, $limit: Int!) {
  Solana(dataset: archive) {
    DEXTrades(
      limit: {count: $limit}
      orderBy: {descending: Block_Time}
      where: {
        Trade: {Dex: {ProgramAddress: {is: $program}}}
        Block: {Time: {since: $since, till: $till}}
      }
    ) {
      Block { Time }
      Transaction { Signature Fee FeeInUSD }
      Trade {
        Buy  { Currency { MintAddress Symbol } Amount AmountInUSD }
        Sell { Currency { MintAddress Symbol } Amount AmountInUSD }
        Dex  { ProgramAddress ProtocolName }
        Market { MarketAddress }
      }
    }
  }
}
"""


@dataclass
class BitqueryClient:
    api_key: str
    endpoint: str = BITQUERY_ENDPOINT
    timeout_s: int = 30
    max_retries: int = 3

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def fetch_new_pool_trades(
        self, venue: Venue, since_iso: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Fetch raw trade records for a given venue since a timestamp.

        Dispatches to the right query shape based on `venue.chain_family`;
        callers never need to know the underlying GraphQL differs by chain.
        Returns the raw nested trade list as-is (Bronze: minimal
        transformation) — caller normalizes into the common Bronze schema.
        """
        if venue.chain_family == ChainFamily.EVM:
            if not venue.network:
                raise ValueError(f"Venue {venue.name} is EVM but has no network set")
            query, variables, path = (
                _EVM_QUERY,
                {
                    "network": venue.network,
                    "program": venue.dex_program_or_address,
                    "since": since_iso,
                    "limit": limit,
                },
                ("data", "EVM", "DEXTrades"),
            )
        elif venue.chain_family == ChainFamily.SOLANA:
            query, variables, path = (
                _SOLANA_QUERY,
                {
                    "program": venue.dex_program_or_address,
                    "since": since_iso,
                    "limit": limit,
                },
                ("data", "Solana", "DEXTrades"),
            )
        else:
            raise NotImplementedError(
                f"No query template for chain family {venue.chain_family}"
            )

        payload = {"query": query, "variables": variables}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    raise RuntimeError(f"Bitquery GraphQL errors: {data['errors']}")
                result = data
                for key in path:
                    result = result[key]
                return result
            except (requests.RequestException, RuntimeError, KeyError) as exc:
                last_err = exc
                time.sleep(2**attempt)  # simple exponential backoff

        raise RuntimeError(
            f"Bitquery fetch failed for venue={venue.name} after "
            f"{self.max_retries} attempts"
        ) from last_err

    def fetch_historical_trades(
        self, venue: Venue, since_iso: str, till_iso: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Bounded-window historical backfill fetch -- `[since_iso, till_iso)`
        against Bitquery's `archive` dataset, NOT the live/recent path
        `fetch_new_pool_trades` uses. Caller is responsible for sizing the
        window small enough to stay under `limit` (a response with exactly
        `limit` rows means the window has more data than was returned --
        bisect it and re-fetch both halves rather than trusting it's
        complete). Same response shape as `fetch_new_pool_trades`, so the
        same Bronze `_normalize()` works on either.
        """
        if venue.chain_family == ChainFamily.EVM:
            if not venue.network:
                raise ValueError(f"Venue {venue.name} is EVM but has no network set")
            query, variables, path = (
                _EVM_BACKFILL_QUERY,
                {
                    "network": venue.network,
                    "program": venue.dex_program_or_address,
                    "since": since_iso,
                    "till": till_iso,
                    "limit": limit,
                },
                ("data", "EVM", "DEXTrades"),
            )
        elif venue.chain_family == ChainFamily.SOLANA:
            query, variables, path = (
                _SOLANA_BACKFILL_QUERY,
                {
                    "program": venue.dex_program_or_address,
                    "since": since_iso,
                    "till": till_iso,
                    "limit": limit,
                },
                ("data", "Solana", "DEXTrades"),
            )
        else:
            raise NotImplementedError(
                f"No backfill query template for chain family {venue.chain_family}"
            )

        payload = {"query": query, "variables": variables}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    raise RuntimeError(f"Bitquery GraphQL errors: {data['errors']}")
                result = data
                for key in path:
                    result = result[key]
                return result
            except (requests.RequestException, RuntimeError, KeyError) as exc:
                last_err = exc
                time.sleep(2**attempt)

        raise RuntimeError(
            f"Bitquery historical fetch failed for venue={venue.name} "
            f"window=[{since_iso}, {till_iso}) after {self.max_retries} attempts"
        ) from last_err
