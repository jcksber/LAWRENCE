"""Reference bar builder (src/serving/bars_v1.py): synthetic checks of the
Dune/Silver rules. Parity against REAL Dune output lives in
tests/test_parity_fixture.py (runs once the fixture is exported)."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from features.silver_token_trades import SOLANA_QUOTE_DENYLIST
from serving.bars_v1 import USD_QUOTES, WSOL, build_bars, price_unit_live

M = "MintPump111"
POOL = "PumpSwapPool1"
T0 = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _sw(sec, project, buy, base_amt, sol_amt, usd, slot, pool=None, version=1, trader="w1"):
    return {
        "block_time": T0 + timedelta(seconds=sec), "block_slot": slot, "tx_index": 0, "outer_instruction_index": 0,
        "project": project, "version": version, "project_program_id": pool,
        "token_bought_mint_address": M if buy else WSOL, "token_sold_mint_address": WSOL if buy else M,
        "token_bought_amount": base_amt if buy else sol_amt, "token_sold_amount": sol_amt if buy else base_amt,
        "amount_usd": usd, "trader_id": trader,
    }


def test_usd_quotes_match_silver():
    assert set(USD_QUOTES) == set(SOLANA_QUOTE_DENYLIST)
    assert price_unit_live(WSOL) == "usd" and price_unit_live("OtherQuote") == "quote"


def test_ohlc_by_order_key_and_dust():
    sw = pd.DataFrame([
        _sw(5, "pumpdotfun", True, 1000, 1, 100.0, slot=3),     # price 0.10, later slot
        _sw(10, "pumpdotfun", True, 1000, 1, 120.0, slot=1),    # price 0.12, first by order key
        _sw(20, "pumpdotfun", False, 1000, 1, 90.0, slot=5),    # price 0.09, last
        _sw(30, "pumpdotfun", True, 10, 0.0001, 0.001, slot=9),  # dust -> dropped
    ])
    b = build_bars(sw, mint=M, launch_venue="pumpdotfun", primary_quote=WSOL)
    assert len(b) == 1
    r = b.iloc[0]
    assert abs(r.open - 0.12) < 1e-12 and abs(r.close - 0.09) < 1e-12
    assert abs(r.high - 0.12) < 1e-12 and abs(r.low - 0.09) < 1e-12
    assert r.n_trades == 3 and r.n_buys == 2 and abs(r.volume_usd - 310.0) < 1e-9 and r.phase == "curve"


def test_empty_minutes_have_no_bar_and_graduation_merge():
    rows = [_sw(0, "pumpdotfun", True, 1000, 1, 100.0, slot=1),
            _sw(180, "pumpdotfun", True, 1000, 1, 100.0, slot=2),           # minute 3, curve 1 trade
            _sw(185, "pumpswap", True, 1000, 1, 150.0, slot=3, pool=POOL, version=None),
            _sw(190, "pumpswap", False, 1000, 1, 150.0, slot=4, pool=POOL, version=None),  # minute 3, pool 2 trades
            _sw(300, "pumpswap", True, 1000, 1, 160.0, slot=5, pool=POOL, version=None),
            _sw(310, "pumpswap", True, 1000, 1, 999.0, slot=6, pool="OtherPool", version=None)]  # wrong pool
    b = build_bars(pd.DataFrame(rows), mint=M, launch_venue="pumpdotfun", primary_quote=WSOL,
                   grad_pool_id=POOL, grad_time=T0 + timedelta(seconds=180))
    assert list(b["bar_start"].dt.minute) == [0, 3, 5]          # no bars for minutes 1, 2, 4
    assert list(b["phase"]) == ["curve", "postgrad", "postgrad"]  # busier bar wins minute 3
    assert abs(b.iloc[2].close - 0.16) < 1e-12                    # OtherPool ignored


def test_live_band_is_point_in_time():
    # a later 5000x spike must not change whether an EARLY bar is kept in live mode
    rows = [_sw(60 * i, "pumpdotfun", True, 1000, 1, 100.0, slot=i) for i in range(5)]
    base = build_bars(pd.DataFrame(rows), mint=M, launch_venue="pumpdotfun", primary_quote=WSOL, band_mode="live")
    rows += [_sw(60 * i, "pumpdotfun", True, 1000, 1, 100.0 * 5000, slot=i) for i in range(5, 20)]
    more = build_bars(pd.DataFrame(rows), mint=M, launch_venue="pumpdotfun", primary_quote=WSOL, band_mode="live")
    pd.testing.assert_frame_equal(base, more.iloc[: len(base)])
    train = build_bars(pd.DataFrame(rows), mint=M, launch_venue="pumpdotfun", primary_quote=WSOL, band_mode="training")
    assert len(train) == 15  # training median (full history) drops the 5 early bars instead
