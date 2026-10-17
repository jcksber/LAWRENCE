"""Parity against REAL Dune data (contracts/v1/fixtures/parity_2026-10-05,
exported by notebooks/00j). Skips until the fixture is committed.

What must hold:
1. bars_v1.build_bars(band_mode="training") reproduces silver.token_price_1m
   from the raw swaps: same minutes, same n_trades, OHLC/volume within 1e-6.
2. contract_v1.compute_features on the fixture's Silver bars reproduces the
   exported expected features exactly (determinism across machines).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

FIX = Path(__file__).resolve().parents[1] / "contracts" / "v1" / "fixtures" / "parity_2026-10-05"
pytestmark = pytest.mark.skipif(not (FIX / "swaps.parquet").exists(), reason="parity fixture not exported yet")


def _load():
    ctx = pd.DataFrame(json.loads((FIX / "token_context.json").read_text()))
    return (ctx, pd.read_parquet(FIX / "swaps.parquet"), pd.read_parquet(FIX / "silver_bars.parquet"),
            pd.read_parquet(FIX / "expected_features.parquet"))


def test_reference_bars_match_silver():
    from serving import bars_v1

    ctx, swaps, silver, _ = _load()
    for r in ctx.itertuples(index=False):
        sb = silver[silver["token_id"] == r.token_id].sort_values("bar_start").reset_index(drop=True)
        pump = r.launch_venue == "pumpdotfun"
        grad = bool(r.graduated)
        rb = bars_v1.build_bars(
            swaps[(swaps["token_bought_mint_address"] == r.token_id) | (swaps["token_sold_mint_address"] == r.token_id)],
            mint=r.token_id, launch_venue=r.launch_venue,
            primary_quote=bars_v1.WSOL if pump else r.launchlab_quote,
            grad_pool_id=r.canonical_pool if grad else None,
            grad_time=(r.pumpswap_created_at if pump else r.first_postgrad_bar) if grad else None,
            price_unit=r.price_unit, band_mode="training")
        m = sb.merge(rb, on="bar_start", how="outer", suffixes=("_s", "_r"), indicator=True)
        assert (m["_merge"] == "both").mean() >= 0.99, r.token_id
        both = m[m["_merge"] == "both"]
        assert (both["n_trades_s"] == both["n_trades_r"]).mean() >= 0.99, r.token_id
        for c in ("open", "high", "low", "close", "volume_usd"):
            rel = ((both[f"{c}_r"] - both[f"{c}_s"]).abs() / both[f"{c}_s"].abs()).fillna(0)
            assert (rel <= 1e-6).mean() >= 0.99, (r.token_id, c)


def test_features_reproduce_expected():
    from serving import contract_v1

    ctx, _, silver, expected = _load()
    for r in ctx.itertuples(index=False):
        sb = silver[silver["token_id"] == r.token_id]
        ex = expected[expected["token_id"] == r.token_id]
        lt = pd.Timestamp(r.launch_time)
        for e in ex.head(50).itertuples(index=False):
            f = contract_v1.compute_features(sb, e.assessment_time, launch_time=lt, launch_venue=r.launch_venue)
            for k in contract_v1.FEATURE_COLUMNS:
                a, b = f[k], getattr(e, k)
                assert (pd.isna(a) and pd.isna(b)) or a == b or abs(float(a) - float(b)) < 1e-9, (r.token_id, k)


def test_reference_bars_reproduce_exported_modes():
    """R3: the decoder -> build_bars path is pinned in BOTH band modes."""
    from serving import bars_v1

    ctx, swaps, _, _ = _load()
    for mode in ("training", "live"):
        exp = pd.read_parquet(FIX / f"expected_bars_{mode}.parquet")
        for r in ctx.itertuples(index=False):
            pump = r.launch_venue == "pumpdotfun"
            grad = bool(r.graduated)
            got = bars_v1.build_bars(
                swaps[(swaps["token_bought_mint_address"] == r.token_id) | (swaps["token_sold_mint_address"] == r.token_id)],
                mint=r.token_id, launch_venue=r.launch_venue,
                primary_quote=bars_v1.WSOL if pump else r.launchlab_quote,
                grad_pool_id=r.canonical_pool if grad else None,
                grad_time=(r.pumpswap_created_at if pump else r.first_postgrad_bar) if grad else None,
                price_unit=r.price_unit, band_mode=mode)
            want = exp[exp["token_id"] == r.token_id].drop(columns="token_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(got.reset_index(drop=True), want, check_dtype=False)
