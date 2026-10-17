# Databricks notebook source
# MAGIC %md
# MAGIC # 00j — Bar-parity fixture for contract v1 (request R2)
# MAGIC
# MAGIC 20 tokens launched 2026-10-05 00:00–12:00 UTC (12 pump.fun graduates, 4
# MAGIC LaunchLab graduates, 4 non-graduates). Writes to
# MAGIC `/Volumes/lawrence/gold/contracts/v1/fixtures/parity_2026-10-05/`:
# MAGIC - `swaps.parquet`: RAW Dune `dex_solana.trades` rows for those mints, 10-05 → 10-07
# MAGIC - `token_context.json`: per token: launch venue/time, primary quote, price unit,
# MAGIC   canonical post-grad pool (Silver), earliest graduation pool, graduation time
# MAGIC - `silver_bars.parquet`: expected bars (`silver.token_price_1m`)
# MAGIC - `expected_features.parquet`: `contract_v1.compute_features` at every grid point
# MAGIC - `parity_report.json`: reference builder (`serving/bars_v1.py`) vs Silver
# MAGIC - `expected_bars_training.parquet`, `expected_bars_live.parquet`: `build_bars` output per mode (R3)
# MAGIC - `quote_prices_usd.parquet`: Dune `prices.usd` minute prices for SOL/USDC/USDT (R3)
# MAGIC
# MAGIC **Dune cost:** 4 chunks of 12h (≤ 1 day rule). Run cell 4 with `MAX_CHUNKS = 1`
# MAGIC first and check the credit delta: estimate a few hundred credits in total, unverified.

# COMMAND ----------

import json
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from pyspark.sql import functions as F

sys.path.append("../src")
from ingestion.dune_client import DuneClient  # noqa: E402
from serving import bars_v1, contract_v1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
OUT = "/Volumes/lawrence/gold/contracts/v1/fixtures/parity_2026-10-05"
spark.sql("CREATE VOLUME IF NOT EXISTS lawrence.gold.contracts")  # 04c also creates it; either may run first
dbutils.fs.mkdirs(OUT)
LAUNCH_FROM = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
LAUNCH_TO = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
SWAPS_FROM, SWAPS_TO = LAUNCH_FROM, datetime(2026, 10, 7, tzinfo=timezone.utc)
MAX_CHUNKS = 1   # first run: 1 chunk, check credits; then None
DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")

# COMMAND ----------

# Pick tokens + context from Silver / pool metadata.
px = spark.table("lawrence.silver.token_price_1m")
tok = (px.groupBy("token_id")
       .agg(F.min(F.when(F.col("phase") == "curve", F.col("bar_start"))).alias("launch_time"),
            F.min_by("venue", "bar_start").alias("launch_venue"),
            F.min(F.when(F.col("phase") == "postgrad", F.col("bar_start"))).alias("first_postgrad_bar"),
            F.max(F.when(F.col("phase") == "postgrad", F.col("pool_id"))).alias("canonical_pool"),
            F.first("price_unit").alias("price_unit"), F.first("quote_token_id").alias("primary_quote"),
            F.sum("n_trades").alias("n_trades"))
       .filter((F.col("launch_time") >= F.lit(LAUNCH_FROM)) & (F.col("launch_time") < F.lit(LAUNCH_TO)))
       .filter(F.col("n_trades").between(200, 15000))
       .withColumn("graduated", F.col("first_postgrad_bar").isNotNull())
       .withColumn("_h", F.xxhash64("token_id")))
picks = []
for venue, grad, k in (("pumpdotfun", True, 12), ("raydium_launchlab", True, 4),
                       ("pumpdotfun", False, 2), ("raydium_launchlab", False, 2)):
    picks += tok.filter((F.col("launch_venue") == venue) & (F.col("graduated") == grad)).orderBy("_h").limit(k).collect()
ctx = pd.DataFrame([r.asDict() for r in picks]).drop(columns="_h")

pm = spark.table("lawrence.bronze.pool_metadata").toPandas()
earliest = (pm[pm["venue"].isin(["pumpswap"])].sort_values("created_at")
            .groupby("base_mint").first()[["pool_id", "created_at"]]
            .rename(columns={"pool_id": "earliest_pumpswap_pool", "created_at": "pumpswap_created_at"}))
ll_quote = pm[pm["venue"] == "raydium_launchlab"].groupby("base_mint")["quote_mint"].first().rename("launchlab_quote")
ctx = (ctx.merge(earliest, left_on="token_id", right_index=True, how="left")
          .merge(ll_quote, left_on="token_id", right_index=True, how="left"))
print(ctx.groupby(["launch_venue", "graduated"]).size())
display(ctx)

# COMMAND ----------

mints = ctx["token_id"].tolist()
_values = ", ".join(f"('{m}')" for m in mints)
COLS = ", ".join(bars_v1.SWAP_COLUMNS + ["tx_id"])


def swaps_sql(start, end):
    return f"""
WITH m (mint) AS (VALUES {_values}),
t AS (
  SELECT {COLS}
  FROM dex_solana.trades
  WHERE block_time >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}' AND block_time < TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'
    AND project IN ('pumpdotfun', 'raydium_launchlab', 'pumpswap', 'raydium')
    AND (project <> 'raydium' OR version = 5)
)
SELECT t.* FROM t JOIN m ON t.token_bought_mint_address = m.mint
UNION ALL
SELECT t.* FROM t JOIN m ON t.token_sold_mint_address = m.mint
"""

# COMMAND ----------

client = DuneClient(api_key=DUNE_API_KEY, poll_timeout_s=1800)
chunks, cur = [], SWAPS_FROM
while cur < SWAPS_TO:
    chunks.append((cur, min(cur + timedelta(hours=12), SWAPS_TO)))
    cur += timedelta(hours=12)
parts = []
for i, (s, e) in enumerate(chunks):
    if MAX_CHUNKS is not None and i >= MAX_CHUNKS:
        print(f"Stopping after {MAX_CHUNKS} chunk(s) -- check credits, then set MAX_CHUNKS = None.")
        break
    print(f"chunk {s:%m-%d %H:%M} -> {e:%m-%d %H:%M}: running on Dune...", flush=True)
    rows = client.run_sql(swaps_sql(s, e))
    parts.append(pd.DataFrame(rows))
    print(f"  {len(rows):,} rows")
swaps = pd.concat(parts, ignore_index=True)
swaps["block_time"] = pd.to_datetime(swaps["block_time"].astype(str).str.replace(" UTC", ""), utc=True)
for c in ("token_bought_amount", "token_sold_amount", "amount_usd"):
    swaps[c] = pd.to_numeric(swaps[c], errors="coerce")
COMPLETE = MAX_CHUNKS is None or MAX_CHUNKS >= len(chunks)

# R3: Dune's minute USD prices for the quotes, over the fixture window, so the
# runtime can measure its own price source against what amount_usd used.
QUOTES = {"SOL": bars_v1.WSOL, "USDC": bars_v1.USDC, "USDT": bars_v1.USDT}
quote_prices = pd.DataFrame(client.run_sql(f"""
SELECT minute, CAST(contract_address AS varchar) AS contract_address, symbol, price
FROM prices.usd
WHERE blockchain = 'solana'
  AND minute >= TIMESTAMP '{SWAPS_FROM:%Y-%m-%d %H:%M:%S}' AND minute < TIMESTAMP '{SWAPS_TO:%Y-%m-%d %H:%M:%S}'
  AND symbol IN ('SOL', 'USDC', 'USDT')
"""))  # filter on symbol: Solana contract_address encoding in prices.usd varies
quote_prices["minute"] = pd.to_datetime(quote_prices["minute"].astype(str).str.replace(" UTC", ""), utc=True)


def _b58_hex(addr: str) -> str:
    """Base58 Solana address -> Dune's 0x-hex varbinary form (32 bytes)."""
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = 0
    for ch in addr:
        n = n * 58 + alphabet.index(ch)
    return "0x" + n.to_bytes(32, "big").hex()


# 2026-10-17: symbol 'USDC' matched TWO tokens on Solana; keep only the
# exact mints (SOL=WSOL, USDC=EPjF..., USDT=Es9v...) by their hex form.
_hex = {_b58_hex(a): sym for sym, a in QUOTES.items()}
quote_prices = quote_prices[quote_prices["contract_address"].str.lower().isin(_hex)].copy()
quote_prices["mint"] = quote_prices["contract_address"].str.lower().map({h: QUOTES[s] for h, s in _hex.items()})
print(quote_prices.groupby("symbol").size())
print(f"{len(swaps):,} swaps; complete window: {COMPLETE}")

# COMMAND ----------

# Expected bars (Silver) + expected features + parity report.
silver = (px.filter(F.col("token_id").isin(mints))
          .select("token_id", "bar_start", "open", "high", "low", "close", "volume_usd", "n_trades", "n_buys",
                  "phase", "price_unit", "venue", "pool_id").toPandas())
silver["bar_start"] = pd.to_datetime(silver["bar_start"], utc=True)

feat_rows, report = [], {"tokens": {}, "complete_window": bool(COMPLETE)}
ref_bars = {"training": [], "live": []}
for r in ctx.itertuples(index=False):
    sb = silver[silver["token_id"] == r.token_id].sort_values("bar_start")
    lt = pd.Timestamp(r.launch_time, tz="UTC") if pd.Timestamp(r.launch_time).tzinfo is None else pd.Timestamp(r.launch_time)
    for t in contract_v1.assessment_grid(sb["bar_start"].min(), lt, sb["bar_start"].max() + pd.Timedelta(minutes=1)):
        row, why = contract_v1.select_row(sb, t, launch_time=lt, launch_venue=r.launch_venue, observed_from_launch=True)
        f = contract_v1.compute_features(sb, t, launch_time=lt, launch_venue=r.launch_venue)
        if f is not None:
            feat_rows.append({"token_id": r.token_id, "assessment_time": t, "select_reason": why, **f})
    if r.launch_venue == "pumpdotfun":
        quote, gpool, gtime = bars_v1.WSOL, r.canonical_pool, r.pumpswap_created_at
    else:
        quote, gpool, gtime = r.launchlab_quote, r.canonical_pool, r.first_postgrad_bar
    if not r.graduated:
        gpool = gtime = None
    sw = swaps[(swaps["token_bought_mint_address"] == r.token_id) | (swaps["token_sold_mint_address"] == r.token_id)]
    rep = {"launch_venue": r.launch_venue, "graduated": bool(r.graduated), "n_swaps": int(len(sw)),
           "canonical_is_earliest_pumpswap_pool": (None if r.launch_venue != "pumpdotfun" or not r.graduated
                                                    else bool(r.canonical_pool == r.earliest_pumpswap_pool))}
    for mode in ("training", "live"):
        rb = bars_v1.build_bars(sw, mint=r.token_id, launch_venue=r.launch_venue, primary_quote=quote,
                                grad_pool_id=gpool, grad_time=gtime, price_unit=r.price_unit, band_mode=mode)
        ref_bars[mode].append(rb.assign(token_id=r.token_id))
        mg = sb.merge(rb, on="bar_start", how="outer", suffixes=("_s", "_r"), indicator=True)
        both = mg[mg["_merge"] == "both"]
        rel = lambda c: float(((both[f"{c}_r"] - both[f"{c}_s"]).abs() / both[f"{c}_s"].abs()).max()) if len(both) else None
        rep[mode] = {"silver_bars": int(len(sb)), "ref_bars": int(len(rb)),
                     "matched": int(len(both)), "silver_only": int((mg["_merge"] == "left_only").sum()),
                     "ref_only": int((mg["_merge"] == "right_only").sum()),
                     "max_rel_diff_close": rel("close"), "max_rel_diff_high": rel("high"),
                     "max_rel_diff_low": rel("low"), "max_rel_diff_volume_usd": rel("volume_usd"),
                     "n_trades_equal_share": float((both["n_trades_r"] == both["n_trades_s"]).mean()) if len(both) else None,
                     "phase_equal_share": float((both["phase_r"] == both["phase_s"]).mean()) if len(both) else None}
    report["tokens"][r.token_id] = rep

feats = pd.DataFrame(feat_rows)
tr = [v["training"] for v in report["tokens"].values()]
report["summary"] = {
    "silver_bars": sum(x["silver_bars"] for x in tr), "matched": sum(x["matched"] for x in tr),
    "silver_only": sum(x["silver_only"] for x in tr), "ref_only": sum(x["ref_only"] for x in tr),
    "worst_close_rel_diff": max((x["max_rel_diff_close"] or 0) for x in tr),
    "live_vs_training_bar_count_diff": sum(v["live"]["ref_bars"] - v["training"]["ref_bars"] for v in report["tokens"].values()),
    "pumpfun_canonical_is_earliest_pool": [v["canonical_is_earliest_pumpswap_pool"] for v in report["tokens"].values()
                                           if v["canonical_is_earliest_pumpswap_pool"] is not None],
}
print(json.dumps(report["summary"], indent=2, default=str))

# COMMAND ----------

def _plain(df):
    """Spark Connect's toPandas() puts PlanMetrics objects in df.attrs, which
    pandas tries to JSON-encode into parquet metadata -> TypeError. Drop them."""
    df = df.copy()
    df.attrs = {}
    return df


swaps, silver, feats, ctx, quote_prices = map(_plain, (swaps, silver, feats, ctx, quote_prices))
if COMPLETE:
    swaps.to_parquet(f"{OUT}/swaps.parquet", index=False)
    silver.to_parquet(f"{OUT}/silver_bars.parquet", index=False)
    feats.to_parquet(f"{OUT}/expected_features.parquet", index=False)
    quote_prices.to_parquet(f"{OUT}/quote_prices_usd.parquet", index=False)
    for mode, parts_ in ref_bars.items():  # R3: expected build_bars output in both modes
        _plain(pd.concat(parts_, ignore_index=True)).to_parquet(f"{OUT}/expected_bars_{mode}.parquet", index=False)
    ctx.to_json(f"{OUT}/token_context.json", orient="records", date_format="iso", indent=2)
    with open(f"{OUT}/parity_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("fixture written to", OUT)
else:
    print("Partial swaps window -- not writing the fixture. Set MAX_CHUNKS = None and re-run cells 4-6.")

# COMMAND ----------

# MAGIC %md
# MAGIC Copy into the repo (laptop, repo root), then commit:
# MAGIC ```
# MAGIC databricks fs cp -r dbfs:/Volumes/lawrence/gold/contracts/v1/fixtures contracts/v1/fixtures
# MAGIC ```
