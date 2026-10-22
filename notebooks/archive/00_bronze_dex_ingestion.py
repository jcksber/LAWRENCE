# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Bronze: chain-agnostic DEX pool/trade ingestion
# MAGIC
# MAGIC LAWRENCE is chain- and platform-agnostic by design: this notebook loops
# MAGIC over a configurable list of `Venue`s (each just a chain + DEX filter) and
# MAGIC lands them all into one unified Bronze table. Adding a new chain or DEX
# MAGIC means adding a line to `VENUES` below — nothing else changes.
# MAGIC
# MAGIC Phase 1 scope: append-only raw trade events. No cleaning, no dedup, no
# MAGIC joins — that's Silver/Gold's job.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../../src")
from ingestion.bitquery_client import BitqueryClient, ChainFamily, Venue  # noqa: E402

# COMMAND ----------

API_KEY = dbutils.secrets.get(scope="lawrence", key="bitquery_api_key")

# Starting venues — deliberately more than one chain/DEX from day one, to
# prove the adapter is actually generic rather than "Solana with extra
# steps." Add/remove venues freely; nothing downstream is chain-specific.
VENUES: list[Venue] = [
    Venue(
        name="raydium_v4",
        chain_family=ChainFamily.SOLANA,
        dex_program_or_address="675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    ),
    Venue(
        name="uniswap_v3_base",
        chain_family=ChainFamily.EVM,
        network="base",
        dex_program_or_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
    ),
]

BRONZE_TABLE = "lawrence.bronze.dex_trades"
LOOKBACK_MINUTES = 20  # > job schedule interval, to tolerate late data / retries

# COMMAND ----------

client = BitqueryClient(api_key=API_KEY)
since = (datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)).isoformat()

raw_by_venue: dict[str, list[dict]] = {}
for venue in VENUES:
    try:
        raw_by_venue[venue.name] = client.fetch_new_pool_trades(
            venue=venue, since_iso=since, limit=1000
        )
        print(f"[{venue.name}] fetched {len(raw_by_venue[venue.name])} raw trades")
    except RuntimeError as exc:
        # One venue failing shouldn't take down ingestion for the rest —
        # log and move on; the scheduled job re-runs in 15 min anyway.
        print(f"[{venue.name}] FAILED: {exc}")
        raw_by_venue[venue.name] = []

# COMMAND ----------

# Normalize every venue's raw payload into one common Bronze shape. The
# per-chain field names differ (MintAddress vs SmartContract, Signature vs
# Hash, ...) — this is the one place that translation happens, so Silver/
# Gold and everything above never need to know which chain a row came from.
import json

from pyspark.sql import Row

ingested_at = datetime.now(timezone.utc)


def _normalize(venue: Venue, t: dict) -> Row:
    is_evm = venue.chain_family == ChainFamily.EVM
    buy_currency = t["Trade"]["Buy"]["Currency"]
    sell_currency = t["Trade"]["Sell"]["Currency"]
    # Fee shape differs by chain: EVM has no direct Transaction.Fee under
    # DEXTrades, so it comes via a joinTransactions join; Solana's Fee is a
    # plain Transaction sibling field. Both are native-currency decimals
    # (SOL / ETH), not raw base units (lamports / wei) -- verified against
    # live Bitquery responses, see docs/decisions.md.
    if is_evm:
        fee = t["joinTransactions"]["Fee"]
        tx_fee_native, tx_fee_usd = fee["SenderFee"], fee["SenderFeeInUSD"]
    else:
        tx_fee_native = t["Transaction"]["Fee"]
        tx_fee_usd = t["Transaction"]["FeeInUSD"]
    return Row(
        chain_family=venue.chain_family.value,
        network=venue.network,  # None for non-EVM
        dex_venue=venue.name,
        block_time=t["Block"]["Time"],
        tx_hash=(t["Transaction"]["Hash"] if is_evm else t["Transaction"]["Signature"]),
        dex_protocol=t["Trade"]["Dex"]["ProtocolName"],
        buy_token_id=(buy_currency.get("SmartContract") or buy_currency.get("MintAddress")),
        buy_symbol=buy_currency["Symbol"],
        buy_amount=t["Trade"]["Buy"]["Amount"],
        buy_amount_usd=t["Trade"]["Buy"]["AmountInUSD"],
        sell_token_id=(sell_currency.get("SmartContract") or sell_currency.get("MintAddress")),
        sell_symbol=sell_currency["Symbol"],
        sell_amount=t["Trade"]["Sell"]["Amount"],
        sell_amount_usd=t["Trade"]["Sell"]["AmountInUSD"],
        tx_fee_native=tx_fee_native,
        tx_fee_usd=tx_fee_usd,
        raw_json=json.dumps(t),
        source="bitquery",
        ingested_at=ingested_at,
    )


rows = [
    _normalize(venue, t)
    for venue in VENUES
    for t in raw_by_venue.get(venue.name, [])
]

if rows:
    df = spark.createDataFrame(rows)

    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(BRONZE_TABLE)
    )
    print(f"Appended {df.count()} rows to {BRONZE_TABLE} across {len(VENUES)} venues")
else:
    print("No rows fetched this run across any venue — nothing written.")
