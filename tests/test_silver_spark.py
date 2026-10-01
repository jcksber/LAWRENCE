"""
Parity tests: src/features/silver_spark.py::build_silver (Spark) vs
src/features/silver_token_trades.py::to_silver_rows (pure-Python reference).

Needs a local PySpark + Java; skipped automatically where PySpark isn't
installed (e.g. CI), so the pure-Python suite still runs there.
"""
from datetime import datetime, timezone

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import DoubleType, StringType, StructField, StructType  # noqa: E402

from features.silver_spark import build_silver, pool_pairs  # noqa: E402
from features.silver_token_trades import to_silver_rows  # noqa: E402

WSOL = "So11111111111111111111111111111111111111112"
USDC_SOL = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WETH_BASE = "0x4200000000000000000000000000000000000006"

_BRONZE_SCHEMA = StructType(
    [StructField(c, StringType()) for c in (
        "chain_family", "network", "dex_venue", "dex_protocol", "block_time", "tx_hash",
        "buy_token_id", "sell_token_id",
    )]
    + [StructField(c, DoubleType()) for c in (
        "buy_amount", "buy_amount_usd", "sell_amount", "sell_amount_usd", "tx_fee_native", "tx_fee_usd",
    )]
)


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("silver-parity")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()


def _row(**kw):
    base = dict(
        chain_family="solana", network=None, dex_venue="pumpdotfun", dex_protocol="pumpdotfun",
        block_time="2026-09-01T12:00:00Z", tx_hash="tx1",
        buy_token_id="MEMEmint111", sell_token_id=WSOL,
        buy_amount=1_000_000.0, buy_amount_usd=50.0, sell_amount=0.25, sell_amount_usd=50.0,
        tx_fee_native=None, tx_fee_usd=None,
    )
    base.update(kw)
    return base


def _build(spark, rows):
    return build_silver(spark.createDataFrame(rows, schema=_BRONZE_SCHEMA)).collect()


def _key(r):
    def _r(x):
        return None if x is None else round(x, 9)
    return (r["tx_hash"], r["token_id"], r["side"], _r(r["price_usd"]), _r(r["price_quote"]))


def test_parity_with_reference(spark):
    rows = [
        _row(),  # token/quote -> 1 row
        _row(tx_hash="tx2", buy_token_id=WSOL, sell_token_id=USDC_SOL),  # quote/quote -> 0
        _row(tx_hash="tx3", sell_token_id="OTHERmint222", sell_amount=10.0),  # neither quote -> 2
        _row(tx_hash="tx4", buy_amount=0.0),  # zero amount on token side -> 0
        _row(tx_hash="tx5", buy_amount_usd=None),  # no USD price -> kept, quote price only
        _row(tx_hash="tx6", buy_amount_usd=None, sell_amount=None),  # no price at all -> 0
        _row(  # EVM, mixed-case quote address must still match denylist
            tx_hash="0xabc", chain_family="evm", network="base", dex_venue="uniswap_v3_base",
            dex_protocol="uniswap_v3", buy_token_id="0xTOKEN", sell_token_id=WETH_BASE.upper().replace("0X", "0x"),
        ),
    ]
    expected = sorted(_key(r) for b in rows for r in to_silver_rows(b))
    got = sorted(_key(r) for r in _build(spark, rows))
    assert got == expected


def test_exact_duplicate_collapses_but_distinct_swaps_in_same_tx_survive(spark):
    rows = [
        _row(),
        _row(),  # byte-identical re-fetch -> deduped
        _row(buy_amount=900_000.0),  # 2nd bundled buy, same tx + token, different amount -> kept
    ]
    out = _build(spark, rows)
    assert len(out) == 2
    assert len({r["swap_id"] for r in out}) == 2


def test_both_block_time_formats_parse_to_same_instant(spark):
    rows = [
        _row(tx_hash="a", block_time="2026-08-16 00:00:01.000 UTC"),
        _row(tx_hash="b", block_time="2026-08-16T00:00:01Z"),
    ]
    out = {r["tx_hash"]: r for r in _build(spark, rows)}
    want = datetime(2026, 8, 16, 0, 0, 1)  # naive, in session tz (UTC)
    assert out["a"]["trade_time"] == want
    assert out["b"]["trade_time"] == want
    assert str(out["a"]["trade_date"]) == "2026-08-16"


def test_unknown_block_time_shape_is_null_not_crash(spark):
    out = _build(spark, [_row(block_time="not a time")])
    assert out[0]["trade_time"] is None


def test_launchlab_uses_pool_base_mint_not_denylist(spark):
    # StonkFun coin quoted in ANOTHER launched coin: neither side is on the
    # denylist, so the denylist rule would emit 2 rows. With pool metadata
    # it's exactly one row, for the pool's base.
    row = _row(
        tx_hash="ll1", dex_venue="raydium_launchlab", dex_protocol="raydium_launchlab_1",
        buy_token_id="QUOTECOIN", sell_token_id="BASECOIN", sell_amount=10.0,
    )
    pools = spark.createDataFrame(
        [("raydium_launchlab", "pool1", "BASECOIN", "QUOTECOIN")],
        "venue string, pool_id string, base_mint string, quote_mint string",
    )
    bronze = spark.createDataFrame([row], schema=_BRONZE_SCHEMA)
    with_meta = build_silver(bronze, pool_pairs(pools)).collect()
    without = build_silver(bronze).collect()
    assert [r["token_id"] for r in with_meta] == ["BASECOIN"]
    assert with_meta[0]["quote_token_id"] == "QUOTECOIN"
    assert len(without) == 2  # old behaviour, for contrast
