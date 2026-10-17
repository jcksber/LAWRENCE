"""
Scoring contract v1 (2026-10-17): what the product runtime imports to score
live tokens in SHADOW mode. Model-side owned; the product side wraps it and
never edits it. See contracts/v1/CONTRACT.md.

Pure pandas/numpy -- no Spark/Delta. Features come from the SAME functions
Gold uses (features.gold_features); the only new code is the pandas
bars -> records mapping, which mirrors features.gold_input.gold_input and
is tested against it.

STATUS: shadow only. This model spec failed its pre-registered OOS test on
2026-10-16 (docs/decisions.md). Never route real money through v1.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.walk_forward import VENUE_PHASE_FEE_BPS, DEFAULT_FEE_BPS, round_trip_cost
from features.gold_features import (
    ACTIVE_WINDOW,
    BARRIER_DOWN,
    BARRIER_UP,
    DENSE_SAMPLE_INTERVAL,
    DENSE_SAMPLE_WINDOW,
    PRICE_STALENESS,
    T_SAMPLE_INTERVAL,
    TRAILING_WINDOW,
    _is_active,
    _meta,
    _v2_features,
    barrier_outcome,
    price_at,
    trailing_activity_features,
)

CONTRACT_VERSION = "v1"
HORIZON_MINUTES = 360               # max hold: 6h
SLIPPAGE_BPS_BASE = 50.0            # per side
SLIPPAGE_BPS_STRESS = 100.0         # per side
UNIVERSE_VENUES = ("pumpdotfun", "raydium_launchlab")  # launch venues
# Training only has post-graduation bars for the first 24h of the graduation
# pool (00f HORIZON_HOURS; gold_input.POSTGRAD_HORIZON) and drops rows whose
# 6h label would run past that, so v1 candidates exist only for
# as_of + 6h <= first postgrad bar_start + 24h, i.e. the first 18h after
# graduation. (Added 2026-10-17 for R2 -- missing from the first draft.)
POSTGRAD_OBSERVED_HOURS = 24

# FROZEN = models.barrier_model.FEATURE_COLUMNS (tested for equality).
NUMERIC_FEATURES = [
    "trailing_volume_usd",
    "trailing_trade_count",
    "trailing_buy_count",
    "trailing_sell_count",
    "trailing_buy_sell_ratio",
    "trailing_price_change_pct",
]
CATEGORICAL_FEATURES = ["chain_family", "network", "dex_venue", "dex_protocol"]
EXTRA_FEATURES = ["minutes_since_launch", "is_postgrad_at_t"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + EXTRA_FEATURES

# Input bars: one row per closed 1-minute bar of the token's PRIMARY series
# (curve, then the graduation pool), same as lawrence.silver.token_price_1m.
BAR_COLUMNS = ["bar_start", "close", "high", "low", "volume_usd", "n_trades", "n_buys", "phase"]


def _utc(ts: Any) -> datetime:
    t = pd.Timestamp(ts)
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).to_pydatetime()


def bars_to_records(bars: pd.DataFrame, launch_venue: str) -> list[dict]:
    """pandas bars -> the per-bar records gold_features consumes. Mirrors
    gold_input.gold_input's select: trade_time = bar_start + 1 min (a bar's
    close is known only once its minute ends), price = close, amount_usd =
    volume_usd, dex_venue = the token's LAUNCH venue on every record."""
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    df = bars.sort_values("bar_start")
    out = []
    for r in df.itertuples(index=False):
        out.append({
            "trade_time": _utc(r.bar_start) + timedelta(minutes=1),
            "price": None if pd.isna(r.close) else float(r.close),
            "high": None if pd.isna(r.high) else float(r.high),
            "low": None if pd.isna(r.low) else float(r.low),
            "amount_usd": None if pd.isna(r.volume_usd) else float(r.volume_usd),
            "n_trades": None if pd.isna(r.n_trades) else int(r.n_trades),
            "n_buys": None if pd.isna(r.n_buys) else int(r.n_buys),
            "phase": r.phase,
            "chain_family": "solana",
            "network": None,
            "dex_venue": launch_venue,
            "dex_protocol": None,
        })
    return out


def _closed_by(records: list[dict], as_of: datetime) -> list[dict]:
    return [r for r in records if r["trade_time"] <= as_of]


def compute_features(bars: pd.DataFrame, as_of: Any, *, launch_time: Any, launch_venue: str) -> dict | None:
    """Feature row at `as_of`, or None when Gold would drop the row (no price
    within PRICE_STALENESS, or no record in the trailing window). Bars that
    close after `as_of` are discarded here, so passing extra bars can't leak.
    Needs bars from launch (minutes_since_launch) and >= TRAILING_WINDOW (48h)
    of history where it exists."""
    t = _utc(as_of)
    records = _closed_by(bars_to_records(bars, launch_venue), t)
    if not records:
        return None
    price_t = price_at(records, t)
    if price_t is None:
        return None
    feats = trailing_activity_features(records, t)
    if feats is None:
        return None
    return {**_meta(records), "price_at_t": price_t, **feats, **_v2_features(records, t, _utc(launch_time))}


def assessment_grid(first_bar_start: Any, launch_time: Any, until: Any) -> list[datetime]:
    """Gold's sampling grid (sample_assessment_times, launch_time known):
    starts at the first record's trade_time (= first bar_start + 1 min), steps
    1 min while < launch_time + 60 min, then 15 min from the last step."""
    cur = _utc(first_bar_start) + timedelta(minutes=1)
    dense_until = _utc(launch_time) + DENSE_SAMPLE_WINDOW
    end = _utc(until)
    out = []
    while cur <= end:
        out.append(cur)
        cur += DENSE_SAMPLE_INTERVAL if cur < dense_until else T_SAMPLE_INTERVAL
    return out


def is_assessment_time(as_of: Any, *, first_bar_start: Any, launch_time: Any) -> bool:
    t = _utc(as_of)
    grid = assessment_grid(first_bar_start, launch_time, t)
    return bool(grid) and grid[-1] == t


def select_row(bars: pd.DataFrame, as_of: Any, *, launch_time: Any, launch_venue: str,
               observed_from_launch: bool) -> tuple[dict | None, str]:
    """(feature row, reason). A row is an entry candidate only if ALL hold:
    the token was watched from its first curve bar, its launch venue is in
    the universe, `as_of` is on the Gold grid, the token had a closed bar
    within ACTIVE_WINDOW (2 min) before `as_of`, it had graduated by `as_of`
    (last bar's phase == 'postgrad'), Gold wouldn't drop the row, and
    as_of + 6h <= first postgrad bar_start + 24h (training only covers
    the first 24h after graduation)."""
    if not observed_from_launch:
        return None, "not_observed_from_launch"
    if launch_venue not in UNIVERSE_VENUES:
        return None, "venue_not_in_universe"
    if bars.empty:
        return None, "no_bars"
    t = _utc(as_of)
    if not is_assessment_time(t, first_bar_start=bars["bar_start"].min(), launch_time=launch_time):
        return None, "off_grid"
    records = _closed_by(bars_to_records(bars, launch_venue), t)
    if not _is_active(records, t, ACTIVE_WINDOW):
        return None, "inactive"
    row = compute_features(bars, t, launch_time=launch_time, launch_venue=launch_venue)
    if row is None:
        return None, "gold_would_drop"
    if row["is_postgrad_at_t"] is not True:
        return None, "not_postgrad"
    first_pg = bars.loc[bars["phase"] == "postgrad", "bar_start"].min()
    if t + timedelta(minutes=HORIZON_MINUTES) > _utc(first_pg) + timedelta(hours=POSTGRAD_OBSERVED_HOURS):
        return None, "past_training_horizon"
    return row, "candidate"


def round_trip_cost_v1(launch_venue: str, is_postgrad: bool, slippage_bps: float = SLIPPAGE_BPS_BASE) -> float:
    """Round-trip cost as a fraction of stake: 2 * (venue fee + slippage)/1e4.
    v1 rows are all post-graduation: pump.fun -> PumpSwap 120 bps/side,
    LaunchLab -> Raydium CPMM 25 bps/side."""
    fee = VENUE_PHASE_FEE_BPS.get((launch_venue, bool(is_postgrad)), DEFAULT_FEE_BPS)
    return float(round_trip_cost(fee, slippage_bps=slippage_bps))


def resolve_outcome(bars: pd.DataFrame, entry: dict, *, launch_venue: str, observed_until: Any) -> dict | None:
    """Barrier outcome for a shadow entry, exactly as training labels it:
    2x (+100%, limit fill) before -50% (or the bar close if the bar gapped
    through the stop) within HORIZON_MINUTES, using only bars that close
    after the entry; a bar touching both counts as the stop; otherwise the
    price at entry + 6h. `entry` = {"assessment_time", "price_at_t"}.
    Returns None while unresolved (no barrier yet and observed_until < T+6h).
    Fill assumption: entry at price_at_t (close of the last bar closed by T)."""
    t = _utc(entry["assessment_time"])
    records = bars_to_records(bars, launch_venue)
    b = barrier_outcome(records, t, float(entry["price_at_t"]), HORIZON_MINUTES,
                        observed_until=_utc(observed_until), up=BARRIER_UP, down=BARRIER_DOWN)
    if b is None:
        return None
    return {
        "barrier_hit": b["outcome"],          # "up" | "down" | "timeout"
        "return": b["ret"],                   # gross, before costs
        "exit_time": t + timedelta(minutes=b["minutes"]),
        "minutes": b["minutes"],
    }


# --- model artifact -----------------------------------------------------

def load_manifest(contract_dir: str | Path) -> dict:
    return json.loads((Path(contract_dir) / "manifest.json").read_text())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(contract_dir: str | Path):
    """lightgbm.Booster from contracts/v1/artifacts, refusing a file whose
    hash doesn't match the manifest."""
    import lightgbm as lgb

    m = load_manifest(contract_dir)
    path = Path(contract_dir) / "artifacts" / m["model_file"]
    got = sha256_file(path)
    if got != m["model_sha256"]:
        raise ValueError(f"model hash mismatch: {got} != {m['model_sha256']}")
    return lgb.Booster(model_file=str(path))


def verify_startup(contract_dir: str | Path, booster=None) -> float:
    """Scores artifacts/startup_sample.parquet and raises unless every
    prediction matches `expected_prediction` within the manifest tolerance.
    Run at runtime start; refuse to start on failure (wrong library versions,
    wrong file, wrong frame building). Returns the max abs difference."""
    import numpy as np

    m = load_manifest(contract_dir)
    booster = booster or load_model(contract_dir)
    sample = pd.read_parquet(Path(contract_dir) / "artifacts" / m["startup_sample_file"])
    expected = sample.pop("expected_prediction").to_numpy()
    X = to_model_frame(sample.to_dict("records"), m["categories"])
    diff = float(np.max(np.abs(booster.predict(X) - expected)))
    if diff > m.get("startup_tolerance", 1e-12):
        raise RuntimeError(f"startup check failed: max |pred - expected| = {diff:.3e}")
    return diff


def to_model_frame(rows: list[dict], categories: dict[str, list]) -> pd.DataFrame:
    """Rows -> the exact frame the booster was trained on: FEATURE_COLUMNS in
    order, categoricals with the TRAINING category sets (manifest), bool ->
    float, missing numerics as NaN (LightGBM's learned default direction)."""
    df = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)
    for c in CATEGORICAL_FEATURES:
        df[c] = pd.Categorical(df[c], categories=categories.get(c, []))
    df["is_postgrad_at_t"] = df["is_postgrad_at_t"].astype("float")
    for c in NUMERIC_FEATURES + ["minutes_since_launch"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float")
    return df


def decide(score: float, cutoff: float) -> bool:
    """Hypothetical entry iff predicted EV >= the frozen cutoff. The runtime
    also enforces ONE open position per token: skip candidates for a token
    until its previous shadow trade has resolved."""
    return bool(score >= cutoff)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
