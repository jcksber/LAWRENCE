# Databricks notebook source
# MAGIC %md
# MAGIC # 04b — Barrier model: P(2x before -50% within N)
# MAGIC
# MAGIC **Why (2026-10-08, docs/label_spec.md "Barrier label").** 04's
# MAGIC regressor on the fixed 15-minute return ranks tokens (Spearman
# MAGIC 0.22-0.29) but never clears ~3% round-trip costs; the profit is in
# MAGIC rare runs a fixed exit throws away. This trains a classifier on
# MAGIC Jack's actual rule -- hit 2x (sell to recover the stake) before
# MAGIC falling 50% -- and scores it on the rule's realized return
# MAGIC `r_up_{N}m` minus the row's own round-trip cost.
# MAGIC
# MAGIC Horizons: 60 and 360 minutes. 1,440 is skipped: it's censored for
# MAGIC ~31-39% of post-graduation rows, and censoring only removes rows that
# MAGIC had NOT resolved, so its hit rate is biased upward.
# MAGIC
# MAGIC Returns here are bounded (+100% at the target, about -50% at the
# MAGIC stop), so the mean isn't driven by a few 100x outliers -- the
# MAGIC concentration checks are still printed.

# COMMAND ----------

# MAGIC %pip install scikit-learn>=1.4 lightgbm>=4.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys

sys.path.append("../src")

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score

from backtest.walk_forward import launchlab_curve_fee_bps, round_trip_cost, venue_fee_bps, walk_forward_folds
from models.barrier_model import (
    FEATURE_COLUMNS,
    label_columns,
    predict_up_probability,
    prepare_barrier_features,
    train_barrier_model,
)
from models.entry_model import set_categorical_dtypes

HORIZONS = (60, 360)

# COMMAND ----------

GOLD_ENTRY_TABLE = "lawrence.gold.entry_candidates"
N_WALK_FORWARD_FOLDS = 5  # placeholder -- revisit once real fold sizes are visible below

# Features + both barrier labels per horizon (is_postgrad_at_t is a feature here and drives per-row fees).
_READ_COLUMNS = ["token_id", "assessment_time", "price_unit", *FEATURE_COLUMNS,
                 *[c for n in HORIZONS for c in label_columns(n)]]

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
      f"{entry_df['assessment_time'].min()} to {entry_df['assessment_time'].max()}")

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
entry_df["fees_bps"] = venue_fee_bps(entry_df["dex_venue"], entry_df["is_postgrad_at_t"], entry_df["ll_curve_fee_bps"])
entry_df["round_trip"] = round_trip_cost(entry_df["fees_bps"])
print(f"{len(folds)} folds; mean round-trip cost {entry_df['round_trip'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Walk-forward: train on the past, score the next period
# MAGIC
# MAGIC Per fold and horizon: AUC (0.5 = no skill), the base hit rate and the
# MAGIC net return of buying everything, then the model's top 1% / 5% / 10%
# MAGIC by predicted probability. `net` = realized r_up minus the row's
# MAGIC round-trip cost. Breakeven needs roughly a 34% hit rate if every
# MAGIC non-hit lost 50% -- timeouts lose less, so the real bar is lower.

# COMMAND ----------

rows, picks = [], []
for n in HORIZONS:
    for i, (train_mask, test_mask) in enumerate(folds):
        X_tr, y_tr, _ = prepare_barrier_features(entry_df[train_mask], n)
        X_te, y_te, r_te = prepare_barrier_features(entry_df[test_mask], n)
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        model = train_barrier_model(X_tr, y_tr)
        p = predict_up_probability(model, X_te)
        te = entry_df.loc[X_te.index]
        net = r_te - te["round_trip"]
        row = {"horizon_m": n, "fold": i, "rows": len(y_te), "auc": roc_auc_score(y_te, p),
               "base_hit": y_te.mean(), "base_net": net.mean()}
        for q in (0.01, 0.05, 0.10):
            top = p >= p.quantile(1 - q)
            row[f"top{int(q*100)}_hit"] = y_te[top].mean()
            row[f"top{int(q*100)}_net"] = net[top].mean()
            row[f"top{int(q*100)}_tokens"] = te.loc[top, "token_id"].nunique()
        rows.append(row)
        top5 = p >= p.quantile(0.95)
        picks.append(te.loc[top5, ["token_id", "assessment_time", "dex_venue", "platform_name",
                                   "price_unit", "is_postgrad_at_t"]]
                     .assign(horizon_m=n, fold=i, p=p[top5], y=y_te[top5], net=net[top5]))
results = pd.DataFrame(rows)
picks = pd.concat(picks, ignore_index=True)
print(results.round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration + ranking: by predicted-probability decile (all folds)

# COMMAND ----------

# Refit once per horizon on everything but the last fold, score the last
# fold, and show deciles -- is a higher probability really a higher hit
# rate, and where (if anywhere) does net turn positive?
for n in HORIZONS:
    train_mask, test_mask = folds[-1]
    X_tr, y_tr, _ = prepare_barrier_features(entry_df[train_mask], n)
    X_te, y_te, r_te = prepare_barrier_features(entry_df[test_mask], n)
    p = predict_up_probability(train_barrier_model(X_tr, y_tr), X_te)
    net = r_te - entry_df.loc[X_te.index, "round_trip"]
    dec = pd.qcut(p.rank(method="first"), 10, labels=False) + 1
    print(f"\nhorizon {n}m, last fold:")
    print(pd.DataFrame({"p": p, "y": y_te, "net": net, "d": dec}).groupby("d")
          .agg(mean_p=("p", "mean"), hit_rate=("y", "mean"), net=("net", "mean"), rows=("y", "size"))
          .round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 5%: where do the picks come from, and is it many tokens?

# COMMAND ----------

def _concentration(g):
    per_token = g.groupby("token_id")["net"].mean()
    tok_sum = g.groupby("token_id")["net"].sum()
    return pd.Series({
        "trades": len(g), "tokens": g["token_id"].nunique(), "hit_rate": g["y"].mean(),
        "net": g["net"].mean(), "net_first_per_token": g.sort_values("assessment_time")
            .drop_duplicates(["fold", "token_id"])["net"].mean(),
        "net_per_token": per_token.mean(), "tokens_profitable": (per_token > 0).mean(),
        "top_token_share_of_pnl": tok_sum.max() / tok_sum.sum() if tok_sum.sum() > 0 else np.nan,
    })

for n in HORIZONS:
    g = picks[picks["horizon_m"] == n]
    print(f"\nhorizon {n}m -- top 5% picks by segment:")
    print(g.groupby(["dex_venue", "platform_name", "price_unit", "is_postgrad_at_t"], dropna=False)
          .apply(_concentration, include_groups=False)
          .query("trades >= 30").sort_values("trades", ascending=False).round(4).to_string())
    print(f"horizon {n}m -- top 5% picks by fold:")
    print(g.groupby("fold").apply(_concentration, include_groups=False).round(4).to_string())
