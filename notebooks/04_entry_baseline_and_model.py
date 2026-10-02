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
from backtest.walk_forward import round_trip_cost, simulate_trades, walk_forward_folds
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

_READ_COLUMNS = ["token_id", "assessment_time", "y_entry", "is_rug_clamped", *FEATURE_COLUMNS]

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
            "lightgbm_model": preds > round_trip_cost(),
        }

        for strategy_name, enter_mask in strategies.items():
            result = simulate_trades(y_test, enter_mask, strategy=strategy_name)
            results.append({"fold": i, **result.__dict__})
            for metric in ("ev_per_trade", "win_rate", "max_drawdown", "n_trades", "total_return"):
                mlflow.log_metric(f"fold{i}_{strategy_name}_{metric}", getattr(result, metric))

    if last_model is not None:
        mlflow.lightgbm.log_model(last_model, artifact_path="entry_model_last_fold")

# COMMAND ----------

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

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
