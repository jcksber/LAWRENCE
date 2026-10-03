# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Phase 3: entry baselines + first ML model
# MAGIC
# MAGIC Reads `lawrence.gold.entry_candidates` (see docs/gold_spec.md) and
# MAGIC compares three strategies against each other via walk-forward
# MAGIC backtesting (docs/game_plan.md §5/§6, `src/backtest/walk_forward.py`):
# MAGIC
# MAGIC 1. **Baseline A** (`src/backtest/baselines.py`) — enter every
# MAGIC    candidate, no filtering. Literally BIA's current "buy everything
# MAGIC    new" rule, applied to Gold's candidate set. The zero-signal floor.
# MAGIC 2. **Baseline B** — a trailing-volume + buy-pressure filter. A stand-in
# MAGIC    for game_plan.md §5's originally-envisioned "liquidity > X and no
# MAGIC    prior rugs from this deployer" rule, which needs pool-liquidity and
# MAGIC    deployer-wallet data that doesn't exist yet (tracked in
# MAGIC    game_plan.md's Phase 1). See baselines.py's module docstring for
# MAGIC    the full rationale.
# MAGIC 3. **LightGBM regressor** (`src/models/entry_model.py`) — predicts
# MAGIC    `y_entry` directly from Gold's trailing-activity features.
# MAGIC
# MAGIC **2026-09-17 scope decision (Jack):** entry only this pass. Exit has
# MAGIC much less/thinner training data right now (14,907/61,633 tokens, see
# MAGIC docs/decisions.md) and label_spec.md already treats entry/exit as
# MAGIC genuinely separate models -- exit gets its own baseline+model pass once
# MAGIC this one's pipeline (walk-forward, backtest metrics, MLflow logging) is
# MAGIC proven out, not duplicated blind.
# MAGIC
# MAGIC **Success metric, per game_plan.md §0/§2:** NOT "guarantee 50%+ average
# MAGIC profit" (not a real target, see game_plan.md §2) -- EV/trade, win rate,
# MAGIC and max drawdown, after realistic round-trip trading costs, via
# MAGIC expanding-window walk-forward folds (never a random split on
# MAGIC time-series token data). The bar for "is this worth anything" is
# MAGIC beating Baseline A and Baseline B, not hitting an absolute number.

# COMMAND ----------

# MAGIC %pip install scikit-learn>=1.4 lightgbm>=4.3 mlflow>=2.11

# COMMAND ----------

# 2026-09-18 (see docs/decisions.md): NOT just for the pip-installed
# packages above. src/models/, src/backtest/, src/features/ are imported
# via sys.path.append (below), not pip -- on an interactive
# cluster/session that stays warm across notebook reruns, Python caches
# those in sys.modules the first time they're imported and does NOT
# re-read them from disk on a later run, even after a real code change is
# committed/deployed. Restarting here, before any of those imports
# happen, guarantees a clean slate every run instead of silently serving
# whatever version of src/ happened to be imported earliest in this
# session -- caught because a real fix (winsorize_return) landed on disk,
# was independently confirmed present via a fresh import in a scratch
# cell, and yet a full notebook re-run still produced byte-identical
# results to before the fix.
dbutils.library.restartPython()

# COMMAND ----------

import sys

sys.path.append("../src")

import mlflow
import mlflow.lightgbm
import pandas as pd

from backtest.baselines import baseline_a_buy_everything, baseline_b_volume_momentum_filter
from backtest.walk_forward import (
    launchlab_curve_fee_bps,
    round_trip_cost,
    simulate_trades,
    venue_fee_bps,
    walk_forward_folds,
)
from features.gold_features import CAPTURABLE_RETURN_CAP, winsorize_return
from models.entry_model import (
    FEATURE_COLUMNS,
    prepare_features,
    score_candidates,
    set_categorical_dtypes,
    train_entry_model,
)

# COMMAND ----------

GOLD_ENTRY_TABLE = "lawrence.gold.entry_candidates"
N_WALK_FORWARD_FOLDS = 5  # placeholder -- revisit once real fold sizes are visible below

# is_postgrad_at_t: not a model feature (yet) -- read for per-row trading costs.
_READ_COLUMNS = ["token_id", "assessment_time", "y_entry", "is_rug_clamped", "is_postgrad_at_t", "price_unit", *FEATURE_COLUMNS]

# select() before toPandas() -- same lesson as notebooks/02/03's OOM fix
# (see docs/decisions.md): don't pull columns nothing downstream reads
# into driver memory. entry_candidates is ~1.3M rows (vs. Silver's
# ~10.75M), comfortably a single-node pandas job, but no reason to carry
# dead weight either.
# 2026-10-01: Gold v2 samples every minute for a token's first hour, so
# entry_candidates can be 5-10x larger than v1's ~4.9M rows -- too much to
# toPandas() on one driver. If it's over MAX_TRAIN_ROWS, keep a
# deterministic subset of TOKENS (hash of token_id), each with its full
# history -- never random rows, which would scatter a token's correlated
# minute-by-minute samples across train/test folds unevenly.
import math  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

MAX_TRAIN_ROWS = 8_000_000
_total_rows = spark.table(GOLD_ENTRY_TABLE).count()
TOKEN_SAMPLE_PCT = min(100, math.ceil(100 * MAX_TRAIN_ROWS / max(_total_rows, 1)))
print(f"{GOLD_ENTRY_TABLE}: {_total_rows:,} rows -> keeping ~{TOKEN_SAMPLE_PCT}% of tokens")

entry_df = (
    spark.table(GOLD_ENTRY_TABLE)
    .filter(F.pmod(F.xxhash64("token_id"), F.lit(100)) < TOKEN_SAMPLE_PCT)
    .select(*_READ_COLUMNS)
    .toPandas()
)
entry_df["assessment_time"] = pd.to_datetime(entry_df["assessment_time"], utc=True)
# walk_forward_folds/simulate_trades both require chronological order --
# see their docstrings.
entry_df = entry_df.sort_values("assessment_time").reset_index(drop=True)
# 2026-09-18 (see docs/decisions.md, entry_model.py's set_categorical_dtypes
# docstring): MUST run once here, on the full dataset, before folds split it
# into train/test slices -- casting per-fold subsets separately would give
# each fold's category-dtype columns their own, possibly inconsistent
# category set for the same column.
entry_df = set_categorical_dtypes(entry_df)
print(f"Loaded {len(entry_df):,} rows from {GOLD_ENTRY_TABLE}, "
      f"{entry_df['assessment_time'].min()} to {entry_df['assessment_time'].max()}, "
      f"{entry_df['is_rug_clamped'].mean():.1%} rug-clamped")

# COMMAND ----------

# 2026-10-03: per-token LaunchLab curve fee. Each LaunchLab launch has a
# PlatformConfig account that sets a platform fee + creator fee on top of
# the 0.25% protocol fee (StonkFun / letsbonk mostly 1.0% + 0-0.5%; see
# docs/decisions.md). Bronze pool_metadata carries the config per pool
# (00f job pools_launchlab_v2); the config -> rates table is reference data
# pulled once from Dune. Tokens with no known config get the flat 1.25%.
POOL_METADATA_TABLE = "lawrence.bronze.pool_metadata"
PLATFORM_CONFIGS_CSV = "../data/reference/launchlab_platform_configs.csv"

_configs = pd.read_csv(PLATFORM_CONFIGS_CSV)
_configs["creator_fee_rate"] = _configs["creator_fee_rate"].fillna(0)
_pool_cols = set(spark.table(POOL_METADATA_TABLE).columns)
if "platform_config" in _pool_cols:
    _ll = (
        spark.table(POOL_METADATA_TABLE)
        .filter("venue = 'raydium_launchlab' AND platform_config IS NOT NULL")
        .groupBy(F.col("base_mint").alias("token_id"))
        .agg(F.max("platform_config").alias("platform_config"), F.max("platform_name").alias("platform_name"))
        .toPandas()
        .merge(_configs[["account_platform_config", "fee_rate", "creator_fee_rate"]],
               left_on="platform_config", right_on="account_platform_config", how="left")
    )
    _ll["ll_curve_fee_bps"] = launchlab_curve_fee_bps(_ll["fee_rate"], _ll["creator_fee_rate"])
    entry_df = entry_df.merge(_ll[["token_id", "platform_name", "ll_curve_fee_bps"]], on="token_id", how="left")
else:
    print(f"WARNING: {POOL_METADATA_TABLE} has no platform_config yet -- run 00f (pools_launchlab_v2). "
          "Using the flat 1.25% LaunchLab curve fee.")
    entry_df["platform_name"] = None
    entry_df["ll_curve_fee_bps"] = float("nan")
_llmask = entry_df["dex_venue"] == "raydium_launchlab"
entry_df["platform_name"] = entry_df["platform_name"].where(_llmask).fillna("-")
entry_df.loc[_llmask & (entry_df["platform_name"] == "-"), "platform_name"] = "unknown"
print(f"LaunchLab rows with a known platform fee: {entry_df.loc[_llmask, 'll_curve_fee_bps'].notna().mean():.1%}")
print(
    entry_df[_llmask].groupby("platform_name")["ll_curve_fee_bps"]
    .agg(rows="size", mean_fee_bps="mean", min_fee_bps="min", max_fee_bps="max")
    .sort_values("rows", ascending=False).round(1).head(10).to_string()
)

# COMMAND ----------

folds = walk_forward_folds(entry_df["assessment_time"], n_folds=N_WALK_FORWARD_FOLDS)
print(f"{len(folds)} usable walk-forward folds (requested {N_WALK_FORWARD_FOLDS})")
for i, (train_mask, test_mask) in enumerate(folds):
    print(
        f"  fold {i}: train={train_mask.sum():,} rows "
        f"(up to {entry_df.loc[train_mask, 'assessment_time'].max()}), "
        f"test={test_mask.sum():,} rows "
        f"({entry_df.loc[test_mask, 'assessment_time'].min()} to "
        f"{entry_df.loc[test_mask, 'assessment_time'].max()})"
    )

# COMMAND ----------

results = []
last_model = None
model_trades = []  # 2026-10-02: every row the model entered, for the stress tests below
ranking_rows = []  # 2026-10-02: ranking-signal diagnostics, independent of the entry threshold
segment_base = []  # 2026-10-02: buy-everything EV per segment per fold, for comparison

with mlflow.start_run(run_name="phase3_entry_baseline_and_model"):
    mlflow.log_param("n_folds_requested", N_WALK_FORWARD_FOLDS)
    mlflow.log_param("n_folds_used", len(folds))
    mlflow.log_param("feature_columns", FEATURE_COLUMNS)

    for i, (train_mask, test_mask) in enumerate(folds):
        train_df = entry_df[train_mask]
        test_df = entry_df[test_mask]

        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)

        last_model = train_entry_model(X_train, y_train)
        preds = score_candidates(last_model, X_test)

        # 2026-10-02: per-row fees by venue + phase (pump.fun curve 1.25%,
        # fresh PumpSwap 1.20%, LaunchLab curve 0.25% + platform fee, CPMM 0.25%; see
        # walk_forward.VENUE_PHASE_FEE_BPS) instead of one AMM-style 30 bps.
        # Used for BOTH the model's entry rule and the backtest that scores
        # every strategy, so they can't disagree about what a trade costs.
        fees_bps = venue_fee_bps(test_df["dex_venue"], test_df["is_postgrad_at_t"], test_df["ll_curve_fee_bps"])
        if i == 0:
            print(f"Mean round-trip cost on fold-0 test rows: {round_trip_cost(fees_bps).mean():.2%}")

        strategies = {
            "baseline_a_buy_everything": baseline_a_buy_everything(test_df),
            "baseline_b_volume_momentum": baseline_b_volume_momentum_filter(test_df),
            # 2026-09-18 (see docs/decisions.md): v1's "predicted > 0" rule
            # let the model "enter" trades with a predicted return between
            # 0% and the ~1.6% round-trip cost simulate_trades charges below
            # -- a guaranteed loss after costs. round_trip_cost() here uses
            # simulate_trades' own default fee/slippage assumptions, so the
            # entry decision and the cost that scores it agree with each
            # other. Ranking by score and taking a top-K is still an easy
            # variant to try later -- see entry_model.py's score_candidates.
            "lightgbm_model": preds > round_trip_cost(fees_bps),
            # 2026-10-02: threshold-free variant -- the model's top 1% of
            # scores in this fold. Tests ranking quality without depending
            # on predicted magnitudes being calibrated.
            "lightgbm_top1pct": preds >= preds.quantile(0.99),
        }

        # Ranking signal (2026-10-02): does a higher score mean a better
        # trade at all, before any threshold? Spearman rank correlation and
        # mean net return (10x cap, after this row's costs) by score decile.
        _net = winsorize_return(y_test, CAPTURABLE_RETURN_CAP) - round_trip_cost(fees_bps)
        _dec = pd.qcut(preds.rank(method="first"), 10, labels=False) + 1
        segment_base.append(
            test_df.assign(_net=_net)
            .groupby(["dex_venue", "platform_name", "price_unit"], observed=True)["_net"]
            .agg(base_rows="size", base_ev_cap10="mean")
            .reset_index()
            .assign(fold=i)
        )
        ranking_rows.append({
            "fold": i,
            "spearman_pred_vs_y": preds.rank().corr(y_test.rank()),
            **{f"d{d}_net": _net[_dec == d].mean() for d in range(1, 11)},
        })

        _entered = strategies["lightgbm_model"]
        model_trades.append(
            test_df.loc[_entered, ["token_id", "assessment_time", "y_entry", "trailing_volume_usd",
                                   "dex_venue", "platform_name", "price_unit", "is_postgrad_at_t"]]
            .assign(fold=i, round_trip=round_trip_cost(fees_bps)[_entered], pred=preds[_entered])
        )

        for strategy_name, enter_mask in strategies.items():
            result = simulate_trades(y_test, enter_mask, strategy=strategy_name, fee_bps=fees_bps)
            results.append({"fold": i, **result.__dict__})
            for metric in ("ev_per_trade", "win_rate", "max_drawdown", "n_trades", "total_return"):
                mlflow.log_metric(f"fold{i}_{strategy_name}_{metric}", getattr(result, metric))

    if last_model is not None:
        mlflow.lightgbm.log_model(last_model, artifact_path="entry_model_last_fold")

# COMMAND ----------

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Ranking signal per fold: spearman > 0 and d10 > d1 means the score orders
# trades better than chance, even if no decile clears costs yet.
print("\nRanking signal (mean net return per score decile, 10x cap, after costs):")
print(pd.DataFrame(ranking_rows).round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stress tests on the model's trades (2026-10-02)
# MAGIC
# MAGIC A positive EV with a ~10-20% win rate means the result rides on a few
# MAGIC big winners. Before believing it, check whether it survives:
# MAGIC 1. **Tighter caps** on the per-trade return (100x is the training cap;
# MAGIC    what if only 10x or 2x is ever capturable?).
# MAGIC 2. **Dropping the top 1% of trades** -- is the edge broad or a handful
# MAGIC    of lottery tickets?
# MAGIC 3. **One trade per token** -- T is sampled every minute, so the model
# MAGIC    can "enter" the same token dozens of times; one winner then counts
# MAGIC    dozens of times. Keep only each token's FIRST signal per fold.
# MAGIC 4. **Liquidity** -- EV by trailing-volume quartile; edge that only
# MAGIC    exists on near-zero-volume tokens isn't tradeable at any real size.

# COMMAND ----------

import numpy as np  # noqa: E402

mt = pd.concat(model_trades, ignore_index=True)


def _net(df, cap):
    return df["y_entry"].clip(-1, cap) - df["round_trip"]


rows = []
for fold, g in mt.groupby("fold"):
    net100 = _net(g, 100.0)
    top = net100 >= net100.quantile(0.99)
    first = g.sort_values("assessment_time").drop_duplicates("token_id", keep="first")
    rows.append({
        "fold": fold,
        "trades": len(g),
        "unique_tokens": g["token_id"].nunique(),
        "ev_cap100": net100.mean(),
        "ev_cap10": _net(g, 10.0).mean(),
        "ev_cap2": _net(g, 2.0).mean(),
        "median_net": net100.median(),
        "ev_ex_top1pct": net100[~top].mean(),
        "top1pct_share_of_pnl": net100[top].sum() / net100.sum() if net100.sum() else np.nan,
        "ev_first_per_token": _net(first, 100.0).mean(),
        "ev_first_per_token_cap10": _net(first, 10.0).mean(),
    })
stress_df = pd.DataFrame(rows)
print(stress_df.round(4).to_string(index=False))

# EV by trailing-volume quartile, all folds pooled (cap 10x, the more
# conservative read).
mt["vol_q"] = pd.qcut(mt["trailing_volume_usd"].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
liq = mt.groupby("vol_q", observed=True)[["y_entry", "round_trip", "trailing_volume_usd"]].apply(
    lambda g: pd.Series({
        "trades": len(g),
        "median_trailing_vol_usd": g["trailing_volume_usd"].median(),
        "ev_cap10": _net(g, 10.0).mean(),
        "win_rate": (_net(g, 10.0) > 0).mean(),
    })
)
print(liq.round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where do the model's trades -- and results -- come from? (2026-10-02)
# MAGIC
# MAGIC Split by launch venue x price unit. `usd` + `pumpdotfun` is the segment
# MAGIC we trust most (SOL-quoted, Dune USD prices, fees verified). `quote`
# MAGIC tokens (mostly StonkFun on LaunchLab) are priced in another token's
# MAGIC units and have no USD volume, so their EV here is less trustworthy.
# MAGIC (2026-10-03: LaunchLab curve rows now pay their own platform fee.) `base_ev_cap10` = buying every
# MAGIC candidate in that segment, the right yardstick per segment.

# COMMAND ----------

mt["net_cap10"] = _net(mt, 10.0)
mt["zero_usd_volume"] = mt["trailing_volume_usd"].fillna(0) == 0
seg = (
    mt.groupby(["dex_venue", "platform_name", "price_unit"], observed=True)
    .agg(
        trades=("net_cap10", "size"),
        unique_tokens=("token_id", "nunique"),
        ev_cap10=("net_cap10", "mean"),
        median_net=("net_cap10", "median"),
        win_rate=("net_cap10", lambda x: (x > 0).mean()),
        zero_usd_volume_share=("zero_usd_volume", "mean"),
        postgrad_share=("is_postgrad_at_t", lambda x: x.fillna(False).astype(bool).mean()),
    )
    .reset_index()
)
seg["share_of_trades"] = seg["trades"] / seg["trades"].sum()
base = (
    pd.concat(segment_base, ignore_index=True)
    .groupby(["dex_venue", "platform_name", "price_unit"], observed=True)
    .apply(lambda g: pd.Series({
        "base_rows": g["base_rows"].sum(),
        "base_ev_cap10": (g["base_ev_cap10"] * g["base_rows"]).sum() / g["base_rows"].sum(),
    }), include_groups=False)
    .reset_index()
)
print(seg.merge(base, on=["dex_venue", "platform_name", "price_unit"], how="left").round(4).to_string(index=False))

# Same split per fold, EV only -- is any segment's edge consistent across time?
print(
    mt.pivot_table(index=["dex_venue", "platform_name", "price_unit"], columns="fold", values="net_cap10", aggfunc="mean", observed=True)
    .round(4).to_string()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stress tests per segment (2026-10-03)
# MAGIC
# MAGIC The pooled stress tests above are dominated by pump.fun (~half the
# MAGIC trades). LaunchLab's +4% could be real or a few lucky tokens, so rerun
# MAGIC the same checks on each segment alone, all at the 10x cap:
# MAGIC - `ev_ex_top1pct`: drop the segment's own best 1% of trades.
# MAGIC - `top1pct_share_of_pnl`: how much of the profit those trades are.
# MAGIC - `ev_first_per_token`: only each token's first signal per fold.
# MAGIC - `ev_per_token`: average each token's trades first, then average
# MAGIC   tokens -- one token entered 50 times counts once.
# MAGIC - `tokens_profitable`: share of tokens whose average trade was > 0.
# MAGIC
# MAGIC Real edge = stays positive ex-top-1%, first-per-token and per-token,
# MAGIC in most folds. If it flips negative, it was a handful of tokens.

# COMMAND ----------

SEGMENTS = {
    "pumpdotfun/usd": (mt["dex_venue"] == "pumpdotfun") & (mt["price_unit"] == "usd"),
    "launchlab/usd (all)": (mt["dex_venue"] == "raydium_launchlab") & (mt["price_unit"] == "usd"),
    "launchlab/usd StonkFun": (mt["dex_venue"] == "raydium_launchlab") & (mt["price_unit"] == "usd")
    & (mt["platform_name"] == "StonkFun"),
    "launchlab/usd non-StonkFun": (mt["dex_venue"] == "raydium_launchlab") & (mt["price_unit"] == "usd")
    & (mt["platform_name"] != "StonkFun"),
    "launchlab/quote StonkFun": (mt["dex_venue"] == "raydium_launchlab") & (mt["price_unit"] == "quote")
    & (mt["platform_name"] == "StonkFun"),
}


def _stress(g, cap=10.0):
    if g.empty:
        return {}
    net = _net(g, cap)
    top = net >= net.quantile(0.99)
    first = g.sort_values("assessment_time").drop_duplicates(["fold", "token_id"], keep="first")
    per_token = net.groupby(g["token_id"]).mean()
    return {
        "trades": len(g),
        "tokens": g["token_id"].nunique(),
        "ev_cap10": net.mean(),
        "median_net": net.median(),
        "ev_ex_top1pct": net[~top].mean(),
        "top1pct_share_of_pnl": net[top].sum() / net.sum() if net.sum() > 0 else np.nan,
        "ev_first_per_token": _net(first, cap).mean(),
        "ev_per_token": per_token.mean(),
        "tokens_profitable": (per_token > 0).mean(),
        # biggest single token's share of the segment's total profit
        "top_token_share_of_pnl": net.groupby(g["token_id"]).sum().max() / net.sum() if net.sum() > 0 else np.nan,
    }


pooled = pd.DataFrame([{"segment": k, **_stress(mt[m])} for k, m in SEGMENTS.items()])
print("All folds pooled (10x cap):")
print(pooled.round(4).to_string(index=False))

per_fold = pd.DataFrame([
    {"segment": k, "fold": f, **_stress(g)}
    for k, m in SEGMENTS.items()
    for f, g in mt[m].groupby("fold")
])
for metric in ["trades", "ev_cap10", "ev_ex_top1pct", "ev_first_per_token", "ev_per_token"]:
    print(f"\n{metric} by fold:")
    print(per_fold.pivot_table(index="segment", columns="fold", values=metric).round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading this table
# MAGIC
# MAGIC - **If `lightgbm_model`'s `ev_per_trade` doesn't beat both baselines
# MAGIC   across most folds**, the model isn't adding signal yet over "buy
# MAGIC   everything" / the volume filter -- expected on a first pass with a
# MAGIC   narrow, Silver-only feature set (see gold_spec.md's "Not included, on
# MAGIC   purpose" list). Not a failure to fix immediately, a baseline reading.
# MAGIC - **`max_drawdown` matters as much as `ev_per_trade`** -- per
# MAGIC   game_plan.md §2/§5, a strategy with a slightly lower EV/trade but a
# MAGIC   much shallower drawdown may be the more honest "beats the baseline"
# MAGIC   result than a fatter-tailed one.
# MAGIC - **`n_trades` varies a lot by strategy** -- Baseline B and the model
# MAGIC   both filter down from Baseline A's full candidate set, so compare
# MAGIC   EV/trade and drawdown, not raw total_return, across strategies with
# MAGIC   very different trade counts.
# MAGIC - Remember `simulate_trades`' known simplification (see
# MAGIC   walk_forward.py's module docstring): no portfolio/position-sizing
# MAGIC   model yet, every entered trade is treated as an independent,
# MAGIC   equal-sized bet.
