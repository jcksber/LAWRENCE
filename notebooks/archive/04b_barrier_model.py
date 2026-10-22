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

sys.path.append("../../src")

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
                 *[c for n in HORIZONS for c in label_columns(n)],
                 *[f"t_up_{n}m" for n in HORIZONS]]  # minutes to resolution (direction diagnostic)

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
PLATFORM_CONFIGS_CSV = "../../data/reference/launchlab_platform_configs.csv"

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
        # "fold" is missing when the caller grouped BY fold (include_groups=False)
        "net": g["net"].mean(), "net_first_per_token": g.sort_values("assessment_time")
            .drop_duplicates([c for c in ("fold", "token_id") if c in g.columns])["net"].mean(),
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Direction diagnostic (2026-10-09): does anything predict WHICH WAY?
# MAGIC
# MAGIC The barrier model ranks well (AUC ~0.8) but its top picks lose most --
# MAGIC it finds tokens that will MOVE, not which way. Here: only rows that
# MAGIC resolved (hit 2x or -50%; timeouts dropped), target = 1 if 2x came
# MAGIC first. AUC ~0.5 = no directional skill in the current features.
# MAGIC Compared against single-feature AUCs (momentum, buy/sell ratio, age)
# MAGIC and with the model's feature importance, to see what (if anything)
# MAGIC carries direction.

# COMMAND ----------

def _resolved(df, n):
    y_col, r_col = label_columns(n)
    y, r, t = df[y_col], df[r_col], df[f"t_up_{n}m"]
    up = y == 1
    # down = stop hit (r <= -50%), excluding a rug-clamped TIMEOUT (r = -1 at t = horizon)
    down = (y == 0) & (r <= -0.5) & ~((r == -1.0) & (t >= n))
    out = df[up | down].copy()
    out["dir_up"] = up[up | down].astype(int)
    return out


dir_rows = []
for n in HORIZONS:
    for i, (train_mask, test_mask) in enumerate(folds):
        tr, te = _resolved(entry_df[train_mask], n), _resolved(entry_df[test_mask], n)
        if tr["dir_up"].nunique() < 2 or te["dir_up"].nunique() < 2:
            continue
        X_tr, X_te = tr[FEATURE_COLUMNS].copy(), te[FEATURE_COLUMNS].copy()
        for X in (X_tr, X_te):
            X["is_postgrad_at_t"] = X["is_postgrad_at_t"].astype("float")
        m = train_barrier_model(X_tr, tr["dir_up"])
        p = predict_up_probability(m, X_te)
        top = p >= p.quantile(0.9)
        row = {"horizon_m": n, "fold": i, "resolved_rows": len(te),
               "resolved_share": len(te) / test_mask.sum(), "up_share": te["dir_up"].mean(),
               "dir_auc": roc_auc_score(te["dir_up"], p), "top10_up_share": te.loc[top, "dir_up"].mean()}
        for c in ("trailing_price_change_pct", "trailing_buy_sell_ratio", "minutes_since_launch", "trailing_volume_usd"):
            v = te[c].astype(float)
            ok = v.notna()
            row[f"auc_{c}"] = roc_auc_score(te.loc[ok, "dir_up"], v[ok]) if ok.sum() and te.loc[ok, "dir_up"].nunique() == 2 else np.nan
        dir_rows.append(row)
    print(f"\nhorizon {n}m -- feature importance (gain), last fold's direction model:")
    print(pd.Series(m.booster_.feature_importance("gain"), index=m.booster_.feature_name())
          .sort_values(ascending=False).round(0).to_string())
print(pd.DataFrame(dir_rows).round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Expected-value models (2026-10-09)
# MAGIC
# MAGIC The direction check found real signal: among rows that resolved, the
# MAGIC direction model's top 10% went 2x-first 71-79% of the time (base
# MAGIC 33-35%, AUC 0.65-0.68). The binary classifier wasted it because it
# MAGIC treats a -50% stop and a small timeout loss as the same "0", so it
# MAGIC drifts to volatile tokens. Two ways to rank by expected value instead:
# MAGIC
# MAGIC - **A. Regressor on `r_up_{N}m`** -- the rule's realized return is
# MAGIC   bounded (+100% / about -50%), so unlike 04's 15-minute return it
# MAGIC   isn't dominated by 100x outliers.
# MAGIC - **B. Three-class model** (up / down / timeout) + the average timeout
# MAGIC   return from training: EV = p_up*1.0 + p_down*(-0.5) + p_timeout*avg.
# MAGIC
# MAGIC Both are scored on realized net (r minus the row's own round trip).
# MAGIC The direction check alone can't be traded -- at T you don't know a
# MAGIC token will resolve -- these can.

# COMMAND ----------

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402


def _outcome(df, n):
    y_col, r_col = label_columns(n)
    y, r, t = df[y_col], df[r_col], df[f"t_up_{n}m"]
    down = (y == 0) & (r <= -0.5) & ~((r == -1.0) & (t >= n))
    return np.select([y == 1, down], [0, 1], default=2)  # 0 up, 1 down, 2 timeout


def _concentration_ev(net, tok, ts, fold):
    g = pd.DataFrame({"net": net, "token_id": tok, "ts": ts, "fold": fold})
    per_token = g.groupby("token_id")["net"].mean()
    tsum = g.groupby("token_id")["net"].sum()
    return {
        "trades": len(g), "tokens": g["token_id"].nunique(), "net": g["net"].mean(),
        "net_first_per_token": g.sort_values("ts").drop_duplicates(["fold", "token_id"])["net"].mean(),
        "net_per_token": per_token.mean(), "tokens_profitable": (per_token > 0).mean(),
        "top_token_share_of_pnl": tsum.max() / tsum.sum() if tsum.sum() > 0 else np.nan,
    }


ev_rows, ev_picks = [], {}
for n in HORIZONS:
    for i, (train_mask, test_mask) in enumerate(folds):
        X_tr, y_tr, r_tr = prepare_barrier_features(entry_df[train_mask], n)
        X_te, y_te, r_te = prepare_barrier_features(entry_df[test_mask], n)
        tr, te = entry_df.loc[X_tr.index], entry_df.loc[X_te.index]
        net = r_te - te["round_trip"]
        # A: regressor on the bounded rule return
        reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_tr, r_tr)
        ev_a = pd.Series(reg.predict(X_te), index=X_te.index)
        # B: 3-class
        c_tr = _outcome(tr, n)
        clf = LGBMClassifier(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_tr, c_tr)
        pr = clf.predict_proba(X_te)
        to_avg = r_tr[c_tr == 2].mean()
        ev_b = pd.Series(pr[:, 0] * 1.0 + pr[:, 1] * -0.5 + pr[:, 2] * to_avg, index=X_te.index)
        for name, ev in (("A_regressor", ev_a), ("B_3class", ev_b)):
            row = {"horizon_m": n, "fold": i, "model": name,
                   "spearman": ev.rank().corr(net.rank()), "base_net": net.mean()}
            for q in (0.01, 0.05):
                top = ev >= ev.quantile(1 - q)
                row[f"top{int(q*100)}_net"] = net[top].mean()
                row[f"top{int(q*100)}_hit"] = y_te[top].mean()
                row[f"top{int(q*100)}_tokens"] = te.loc[top, "token_id"].nunique()
            # model's own rule: enter when predicted EV clears this row's cost
            enter = ev > te["round_trip"]
            row["enter_n"] = int(enter.sum())
            row["enter_net"] = net[enter].mean() if enter.any() else np.nan
            ev_rows.append(row)
            top5 = ev >= ev.quantile(0.95)
            ev_picks.setdefault((n, name), []).append(
                pd.DataFrame({"net": net[top5], "token_id": te.loc[top5, "token_id"],
                              "ts": te.loc[top5, "assessment_time"], "fold": i,
                              "dex_venue": te.loc[top5, "dex_venue"], "platform_name": te.loc[top5, "platform_name"],
                              "is_postgrad_at_t": te.loc[top5, "is_postgrad_at_t"]}))
        if i == len(folds) - 1:
            for name, ev in (("A_regressor", ev_a), ("B_3class", ev_b)):
                dec = pd.qcut(ev.rank(method="first"), 10, labels=False) + 1
                print(f"\nhorizon {n}m, {name}, last fold -- by predicted-EV decile:")
                print(pd.DataFrame({"ev": ev, "net": net, "y": y_te, "d": dec}).groupby("d")
                      .agg(mean_pred_ev=("ev", "mean"), net=("net", "mean"), hit=("y", "mean"))
                      .round(4).to_string())

print("\nWalk-forward, EV models:")
print(pd.DataFrame(ev_rows).round(4).to_string(index=False))

for (n, name), parts in ev_picks.items():
    g = pd.concat(parts, ignore_index=True)
    print(f"\nhorizon {n}m, {name} -- top 5% concentration, all folds:")
    print(pd.Series(_concentration_ev(g["net"], g["token_id"], g["ts"], g["fold"])).round(4).to_string())
    seg = (g.groupby(["dex_venue", "platform_name", "is_postgrad_at_t"], dropna=False, observed=True)
           .apply(lambda x: pd.Series(_concentration_ev(x["net"], x["token_id"], x["ts"], x["fold"])), include_groups=False)
           .query("trades >= 100").sort_values("trades", ascending=False))
    print(seg.round(4).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-graduation only (2026-10-10)
# MAGIC
# MAGIC Graduated StonkFun was the one broad positive pocket (+1.6% at 6h,
# MAGIC 1,708 tokens), helped by the cheaper venue: CPMM / PumpSwap fees are
# MAGIC 0.25% / 1.20% per side vs ~1.25% on the curve. Same model A, trained
# MAGIC and scored on post-graduation rows only.

# COMMAND ----------

# Post-graduation-only EV model (2026-10-10). Self-contained: needs cells 1-6.
from lightgbm import LGBMRegressor

def _pg_conc(g):
    pt = g.groupby("token_id")["net"].mean(); s = g.groupby("token_id")["net"].sum()
    return pd.Series({"trades": len(g), "tokens": g["token_id"].nunique(), "net": g["net"].mean(),
                      "net_first_per_token": g.sort_values("ts").drop_duplicates([c for c in ("fold", "token_id") if c in g.columns])["net"].mean(),
                      "net_per_token": pt.mean(), "tokens_profitable": (pt > 0).mean(),
                      "top_token_share_of_pnl": s.max() / s.sum() if s.sum() > 0 else np.nan})

pg = entry_df["is_postgrad_at_t"].fillna(False).astype(bool)
print(f"post-graduation rows: {pg.sum():,} of {len(entry_df):,}; mean round trip {entry_df.loc[pg, 'round_trip'].mean():.2%}")
pg_rows, pg_picks = [], []
for n in HORIZONS:
    for i, (train_mask, test_mask) in enumerate(folds):
        X_tr, _, r_tr = prepare_barrier_features(entry_df[train_mask & pg], n)
        X_te, y_te, r_te = prepare_barrier_features(entry_df[test_mask & pg], n)
        if len(X_tr) < 1000 or len(X_te) < 100:
            continue
        te = entry_df.loc[X_te.index]
        net = r_te - te["round_trip"]
        reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_tr, r_tr)
        ev = pd.Series(reg.predict(X_te), index=X_te.index)
        row = {"horizon_m": n, "fold": i, "rows": len(X_te), "spearman": ev.rank().corr(net.rank()),
               "base_net": net.mean(), "base_hit": y_te.mean()}
        for q in (0.01, 0.05, 0.10):
            top = ev >= ev.quantile(1 - q)
            row[f"top{int(q*100)}_net"] = net[top].mean()
            row[f"top{int(q*100)}_tokens"] = te.loc[top, "token_id"].nunique()
        enter = ev > te["round_trip"]
        row["enter_n"], row["enter_net"] = int(enter.sum()), (net[enter].mean() if enter.any() else np.nan)
        row["enter_tokens"] = te.loc[enter, "token_id"].nunique()
        pg_rows.append(row)
        top5 = ev >= ev.quantile(0.95)
        pg_picks.append(pd.DataFrame({"horizon_m": n, "fold": i, "net": net[top5], "token_id": te.loc[top5, "token_id"],
                                      "ts": te.loc[top5, "assessment_time"], "dex_venue": te.loc[top5, "dex_venue"],
                                      "platform_name": te.loc[top5, "platform_name"], "price_unit": te.loc[top5, "price_unit"]}))
        if i == len(folds) - 1:
            dec = pd.qcut(ev.rank(method="first"), 10, labels=False) + 1
            print(f"\npost-grad, horizon {n}m, last fold -- by predicted-EV decile:")
            print(pd.DataFrame({"ev": ev, "net": net, "y": y_te, "d": dec}).groupby("d")
                  .agg(mean_pred_ev=("ev", "mean"), net=("net", "mean"), hit=("y", "mean"), rows=("y", "size"))
                  .round(4).to_string())
            print("feature importance (gain):")
            print(pd.Series(reg.booster_.feature_importance("gain"), index=reg.booster_.feature_name())
                  .sort_values(ascending=False).round(0).to_string())

print("\nWalk-forward, post-graduation only (model A):")
print(pd.DataFrame(pg_rows).round(4).to_string(index=False))
pp = pd.concat(pg_picks, ignore_index=True)
for n in HORIZONS:
    g = pp[pp["horizon_m"] == n]
    print(f"\npost-grad, horizon {n}m -- top 5% concentration, all folds:")
    print(_pg_conc(g).round(4).to_string())
    print(g.groupby(["dex_venue", "platform_name", "price_unit"], dropna=False, observed=True)
          .apply(_pg_conc, include_groups=False).query("trades >= 100")
          .sort_values("trades", ascending=False).round(4).to_string())
    print(f"by fold:")
    print(g.groupby("fold").apply(_pg_conc, include_groups=False).round(4).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-overlapping backtest (2026-10-10)
# MAGIC
# MAGIC The post-grad +3.0% counted overlapping entries on the same token
# MAGIC (T sampled every minute). A bot holds one position per token: this
# MAGIC replays the signals that way, with an entry cutoff fixed from the
# MAGIC training period. `sum_net_stakes` = total profit in units of one
# MAGIC stake; `max_concurrent` = most positions open at once (capital needed).

# COMMAND ----------

# Non-overlapping backtest, post-graduation model A, 6h (2026-10-10). Needs cells 1-6.
# One position per token at a time: enter on a signal only if not already
# holding that token; hold until 2x / -50% / 6h (t_up_360m); a new entry needs
# a NEW signal after the exit. Two entry cutoffs:
#   "train_q95" -- the 95th percentile of predicted EV on the TRAINING rows
#                  (fixed before the test period starts -- the honest one)
#   "test_q95"  -- 95th percentile within the test period (what the earlier
#                  table used; peeks at the test distribution, for comparison)
from lightgbm import LGBMRegressor

N = 360
pg = entry_df["is_postgrad_at_t"].fillna(False).astype(bool)


def _non_overlapping(sig):
    """sig: one fold's signals (token_id, ts, minutes, net). Keep a signal
    only if the token has no open position at ts."""
    sig = sig.sort_values(["token_id", "ts"])
    keep, open_until = [], {}
    for idx, tok, ts, mins in zip(sig.index, sig["token_id"], sig["ts"], sig["minutes"]):
        if tok in open_until and ts < open_until[tok]:
            continue
        keep.append(idx)
        open_until[tok] = ts + pd.Timedelta(minutes=float(mins))
    return sig.loc[keep]


def _max_concurrent(trades):
    ev = pd.concat([pd.Series(1, index=trades["ts"]),
                    pd.Series(-1, index=trades["ts"] + pd.to_timedelta(trades["minutes"], unit="m"))]).sort_index()
    return int(ev.cumsum().max()) if len(ev) else 0


def _summ(t):
    pt = t.groupby("token_id")["net"].mean(); s = t.groupby("token_id")["net"].sum()
    return pd.Series({"trades": len(t), "tokens": t["token_id"].nunique(), "net": t["net"].mean(),
                      "median_net": t["net"].median(), "hit_2x": t["y"].mean(),
                      "net_per_token": pt.mean(), "tokens_profitable": (pt > 0).mean(),
                      "top_token_share_of_pnl": s.max() / s.sum() if s.sum() > 0 else np.nan,
                      "sum_net_stakes": t["net"].sum(), "max_concurrent": _max_concurrent(t)})


all_trades = []
for i, (train_mask, test_mask) in enumerate(folds):
    X_tr, _, r_tr = prepare_barrier_features(entry_df[train_mask & pg], N)
    X_te, y_te, r_te = prepare_barrier_features(entry_df[test_mask & pg], N)
    if len(X_tr) < 1000 or len(X_te) < 100:
        continue
    te = entry_df.loc[X_te.index]
    reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_tr, r_tr)
    ev_tr = reg.predict(X_tr)
    ev = pd.Series(reg.predict(X_te), index=X_te.index)
    base = pd.DataFrame({"token_id": te["token_id"], "ts": pd.to_datetime(te["assessment_time"], utc=True),
                         "minutes": te[f"t_up_{N}m"].fillna(N), "net": r_te - te["round_trip"], "y": y_te,
                         "dex_venue": te["dex_venue"], "platform_name": te["platform_name"],
                         "price_unit": te["price_unit"], "fold": i})
    for name, cut in (("train_q95", np.quantile(ev_tr, 0.95)), ("test_q95", ev.quantile(0.95))):
        sig = base[ev >= cut]
        t = _non_overlapping(sig).assign(cutoff=name, signals=len(sig))
        all_trades.append(t)

trades = pd.concat(all_trades, ignore_index=True)
for name in ("train_q95", "test_q95"):
    t = trades[trades["cutoff"] == name]
    print(f"\n=== cutoff {name}: one position per token, 6h ===")
    print("all folds:"); print(_summ(t).round(4).to_string())
    print("by fold:"); print(t.groupby("fold").apply(_summ, include_groups=False).round(4).to_string())
    print("by segment:")
    print(t.groupby(["dex_venue", "platform_name", "price_unit"], dropna=False, observed=True)
          .apply(_summ, include_groups=False).query("trades >= 30")
          .sort_values("trades", ascending=False).round(4).to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ## PRE-REGISTERED out-of-sample test (2026-10-11)
# MAGIC
# MAGIC Rule frozen in docs/decisions.md on 2026-10-10, before any data after
# MAGIC 2026-10-03 was pulled. Run once, after 00h/00f window 3 -> 01b -> 02.
# MAGIC Do not change the rule after seeing this output.

# COMMAND ----------

# PRE-REGISTERED out-of-sample test (rule frozen 2026-10-10, docs/decisions.md).
# Self-contained: reads post-graduation rows for ALL tokens straight from Gold
# (no token sampling), trains once on T < 2026-10-03, applies the frozen rule
# to T >= 2026-10-03. Needs only the imports cell (3).
from datetime import datetime, timezone
from lightgbm import LGBMRegressor

OOS_START = datetime(2026, 10, 3, tzinfo=timezone.utc)
N = 360
y_col, r_col = label_columns(N)
cols = ["token_id", "assessment_time", "price_unit", *FEATURE_COLUMNS, y_col, r_col, f"t_up_{N}m"]
df = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true")
      .select(*dict.fromkeys(cols)).toPandas())
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df = set_categorical_dtypes(df.sort_values("assessment_time").reset_index(drop=True))

# Per-token LaunchLab curve fee (same as the platform-fee cell; only matters
# for curve rows, but kept identical so costs match 04/04b exactly).
_cfg = pd.read_csv("../../data/reference/launchlab_platform_configs.csv")
_cfg["creator_fee_rate"] = _cfg["creator_fee_rate"].fillna(0)
_ll = (spark.table("lawrence.bronze.pool_metadata")
       .filter("venue = 'raydium_launchlab' AND platform_config IS NOT NULL")
       .groupBy(F.col("base_mint").alias("token_id"))
       .agg(F.max("platform_config").alias("platform_config"), F.max("platform_name").alias("platform_name"))
       .toPandas()
       .merge(_cfg[["account_platform_config", "fee_rate", "creator_fee_rate"]],
              left_on="platform_config", right_on="account_platform_config", how="left"))
_ll["ll_curve_fee_bps"] = launchlab_curve_fee_bps(_ll["fee_rate"], _ll["creator_fee_rate"])
df = df.merge(_ll[["token_id", "platform_name", "ll_curve_fee_bps"]], on="token_id", how="left")
_is_ll = df["dex_venue"] == "raydium_launchlab"
df["platform_name"] = df["platform_name"].where(_is_ll).fillna("-")
df.loc[_is_ll & (df["platform_name"] == "-"), "platform_name"] = "unknown"
df["fees_bps"] = venue_fee_bps(df["dex_venue"], df["is_postgrad_at_t"], df["ll_curve_fee_bps"])

train = df[(df["assessment_time"] < OOS_START) & df[y_col].notna()]
test = df[(df["assessment_time"] >= OOS_START) & df[y_col].notna()]
print(f"train: {len(train):,} rows, {train['token_id'].nunique():,} tokens, "
      f"{train['assessment_time'].min()} -> {train['assessment_time'].max()}")
print(f"test:  {len(test):,} rows, {test['token_id'].nunique():,} tokens, "
      f"{test['assessment_time'].min()} -> {test['assessment_time'].max()}")

X_tr, _, r_tr = prepare_barrier_features(train, N)
X_te, y_te, r_te = prepare_barrier_features(test, N)
reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(X_tr, r_tr)
cut = float(np.quantile(reg.predict(X_tr), 0.95))
ev = pd.Series(reg.predict(X_te), index=X_te.index)
te = test.loc[X_te.index]
print(f"frozen cutoff (train 95th pct of predicted EV): {cut:.4f}; test signals: {(ev >= cut).sum():,}")


def _non_overlapping(sig):
    sig = sig.sort_values(["token_id", "ts"])
    keep, open_until = [], {}
    for idx, tok, ts, mins in zip(sig.index, sig["token_id"], sig["ts"], sig["minutes"]):
        if tok in open_until and ts < open_until[tok]:
            continue
        keep.append(idx)
        open_until[tok] = ts + pd.Timedelta(minutes=float(mins))
    return sig.loc[keep]


def _summ(t):
    pt = t.groupby("token_id")["net"].mean(); s = t.groupby("token_id")["net"].sum()
    return pd.Series({"trades": len(t), "tokens": t["token_id"].nunique(), "net": t["net"].mean(),
                      "median_net": t["net"].median(), "hit_2x": t["y"].mean(),
                      "net_per_token": pt.mean(), "tokens_profitable": (pt > 0).mean(),
                      "top_token_share_of_pnl": s.max() / s.sum() if s.sum() > 0 else np.nan,
                      "sum_net_stakes": t["net"].sum()})


results = {}
for label, slip in (("base (0.5%/side slippage)", 50), ("stress (1.0%/side slippage)", 100)):
    rt = round_trip_cost(te["fees_bps"], slippage_bps=slip)
    sig = pd.DataFrame({"token_id": te["token_id"], "ts": te["assessment_time"],
                        "minutes": te[f"t_up_{N}m"].fillna(N), "net": r_te - rt, "y": y_te,
                        "dex_venue": te["dex_venue"], "platform_name": te["platform_name"],
                        "price_unit": te["price_unit"], "day": te["assessment_time"].dt.date})[ev >= cut]
    t = _non_overlapping(sig)
    results[label] = t
    print(f"\n=== OUT-OF-SAMPLE, {label} ===")
    print(_summ(t).round(4).to_string())
    print("by segment:")
    print(t.groupby(["dex_venue", "platform_name", "price_unit"], dropna=False, observed=True)
          .apply(_summ, include_groups=False).query("trades >= 20")
          .sort_values("trades", ascending=False).round(4).to_string())
    print("by day:")
    print(t.groupby("day").apply(_summ, include_groups=False)[["trades", "tokens", "net", "hit_2x", "sum_net_stakes"]]
          .round(4).to_string())

# Pre-registered pass/fail (base costs): overall net > 0, StonkFun usd net > 0, >= 300 trades.
t = results["base (0.5%/side slippage)"]
sfu = t[(t["platform_name"] == "StonkFun") & (t["price_unit"] == "usd")]
checks = {"trades >= 300": len(t) >= 300, "overall net > 0": t["net"].mean() > 0,
          "StonkFun usd net > 0": len(sfu) > 0 and sfu["net"].mean() > 0}
print("\nPRE-REGISTERED CHECKS:", checks, "->", "PASS" if all(checks.values()) else "FAIL")
