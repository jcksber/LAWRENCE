# Databricks notebook source
# MAGIC %md
# MAGIC # 04h — v4 POST-CRASH track: graduated pump.fun tokens 24h–14d after graduation (branch `v3`)
# MAGIC Rows: `gold.postcrash_candidates_v4` (02c), hourly T, activity filter (≥20 trades and ≥$2k in 24h).
# MAGIC Label: 72h barrier, 2x before −50%. Costs: PumpSwap 120 bps/side + slippage 100 bps (base) /
# MAGIC 200 bps (stress) per side → 4.4% / 6.4% round trip (thin pools: slippage doubled vs v1/v2).
# MAGIC Dev data ≤ 10-10 only. Variants declared in `docs/v4_variant_log.md` BEFORE this notebook runs:
# MAGIC v4-2 = rule P1 (Doug pattern), v4-3 = V4b LightGBM + R1. Same strict pass rule as v4-1
# MAGIC (incl. delayed entry: fill at the next hourly bar's close).

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.append("../src")
from backtest.shadow_eval_v1 import block_bootstrap_lower_bound  # noqa: E402
from features.postcrash_features import FEATURES  # noqa: E402
from models import v2_spec as v2  # noqa: E402

FEE, SLIP, SLIP_S = 120, 100, 200
COST, COST_S = 2 * (FEE + SLIP) / 1e4, 2 * (FEE + SLIP_S) / 1e4
TOK = ["eb_buyers_slot0", "eb_non_creator_buyers_slot0", "eb_buyers_first3", "eb_supply_share_slot0",
       "eb_supply_share_first3", "eb_creator_share_slot0", "cr_launches_7d", "cr_grads_7d", "cr_grad_rate_7d"]
CR4 = ["cr4_launches_7d", "cr4_grads_7d", "cr4_grad_rate_7d", "launch_is_mayhem", "launch_non_sol_quote"]

df = (spark.table("lawrence.gold.postcrash_candidates_v4")
      .join(spark.table("lawrence.gold.creator_token_features_v3").select("token_id", *TOK), "token_id", "left")
      .join(spark.table("lawrence.gold.creator_launch_history_v4").select("token_id", *CR4), "token_id", "left")
      .toPandas())
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df = df.sort_values("assessment_time").reset_index(drop=True)
df["net"], df["net_s"] = df["r_pc"] - COST, df["r_pc"] - COST_S
df["net_next"] = df["r_pc_next"] - COST
print(f"{len(df):,} rows, {df['token_id'].nunique():,} tokens, {df['assessment_time'].min()} → {df['assessment_time'].max()}")


def non_overlap(sig: pd.DataFrame) -> pd.DataFrame:
    """One open position per token: skip signals until the previous trade resolved (T + t_pc_h)."""
    sig = sig.sort_values(["token_id", "assessment_time"])
    keep, until = [], {}
    for idx, tok, ts, h in zip(sig.index, sig["token_id"], sig["assessment_time"], sig["t_pc_h"]):
        if tok in until and ts < until[tok]:
            continue
        keep.append(idx)
        until[tok] = ts + pd.Timedelta(hours=float(h))
    return sig.loc[keep]


def summary(p: pd.DataFrame, name: str) -> pd.Series:
    per_tok = p.groupby("token_id")["net"].sum().sort_values(ascending=False)
    return pd.Series({
        "trades": len(p), "tokens": p["token_id"].nunique(), "mean_net_base": p["net"].mean(),
        "mean_net_stress": p["net_s"].mean(),
        "lb95": block_bootstrap_lower_bound(p, p["net"], alpha=0.05) if len(p) >= 30 else np.nan,
        "median_net": p["net"].median(), "hit_2x": (p["outcome"] == "up").mean(), "hit_stop": (p["outcome"] == "down").mean(),
        "mean_net_ex_top5_tokens": p[~p["token_id"].isin(per_tok.index[:5])]["net"].mean(),
        "mean_net_delayed_entry": p["net_next"].mean(), "delayed_coverage": p["net_next"].notna().mean(),
        "folds_positive": int((p.groupby("fold")["net"].mean() > 0).sum()) if "fold" in p else np.nan,
    }, name=name)


S = v2.TRAIN_START
edges = [S + pd.Timedelta(days=7 * k) for k in range(5)]
folds = [(S + pd.Timedelta(days=3), edges[1])] + [(edges[i], edges[i + 1]) for i in range(1, 4)]
df["fold"] = 0
for fi, (lo, hi) in enumerate(folds, start=1):
    df.loc[(df["assessment_time"] >= lo) & (df["assessment_time"] < hi), "fold"] = fi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outcome check (no model; not a variant): buy-everything by drawdown × rebound

# COMMAND ----------

first_daily = df.assign(_d=df["assessment_time"].dt.floor("D")).drop_duplicates(["token_id", "_d"])
first_daily["dd_b"] = pd.cut(first_daily["dd_from_ath"], [-1.01, -0.9, -0.7, -0.5, -0.2, 1e9],
                             labels=["<-90%", "-90..-70%", "-70..-50%", "-50..-20%", ">-20%"])
first_daily["rb_b"] = pd.cut(first_daily["rebound_from_low"], [-0.01, 0.3, 1.0, 1e9], labels=["<30%", "30-100%", ">100%"])
display(first_daily.groupby(["dd_b", "rb_b"], observed=True).agg(
    rows=("net", "size"), tokens=("token_id", "nunique"), mean_net=("net", "mean"), median_net=("net", "median"),
    hit_2x=("outcome", lambda s: (s == "up").mean()), hit_stop=("outcome", lambda s: (s == "down").mean())).round(4).reset_index())
display(summary(first_daily.assign(fold=first_daily["fold"]), "buy every token once a day").round(4).to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## v4-2 — rule P1 (Doug pattern 1): ≥70% below ATH, ≥30% off the post-ATH low, rising over 6h

# COMMAND ----------

p1_sig = df[(df["fold"] > 0) & (df["dd_from_ath"] <= -0.7) & (df["rebound_from_low"] >= 0.3) & (df["ret_6h"] > 0)]
p1 = non_overlap(p1_sig)
display(summary(p1, "P1").round(4).to_frame())
display(p1.groupby("fold").agg(trades=("net", "size"), mean_net=("net", "mean")).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## v4-3 — V4b: LightGBM on post-crash + creator/early-buyer features, rule R1 (p98)

# COMMAND ----------

FEATS = FEATURES + TOK + CR4
X = df[FEATS].astype("float")
parts, imps = [], []
for fi, (lo, hi) in enumerate(folds, start=1):
    tr = (df["assessment_time"] < lo) & (df["assessment_time"] + pd.to_timedelta(df["t_pc_h"], unit="h") <= lo)
    te = df["fold"] == fi
    if tr.sum() < 1000 or te.sum() == 0:
        print(f"fold {fi}: skipped (train {tr.sum()}, test {te.sum()})")
        continue
    reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X[tr], df.loc[tr, "r_pc"])
    cut = float(np.quantile(reg.predict(X[tr]), 0.98))
    t = df[te].copy()
    t["score"] = reg.predict(X[te])
    sel = non_overlap(t[t["score"] >= cut])
    parts.append(sel)
    imps.append(pd.Series(reg.booster_.feature_importance("gain"), index=X.columns))
    print(f"fold {fi}: train {tr.sum():,} test {te.sum():,} | trades {len(sel)} mean net {sel['net'].mean():+.4f}")
v4b = pd.concat(parts)
display(summary(v4b, "V4b").round(4).to_frame())
display(v4b.groupby("fold").agg(trades=("net", "size"), mean_net=("net", "mean")).round(4))
imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
display((imp / imp.sum()).head(15).round(4).rename("gain_share").reset_index())

# COMMAND ----------

for name, p in (("P1", p1), ("V4b", v4b)):
    s_ = summary(p, name)
    per_tok = p.groupby("token_id")["net"].sum().sort_values(ascending=False)
    ok = (s_.mean_net_stress > 0) and (s_.lb95 > 0) and (s_.folds_positive >= 3) \
        and (s_.mean_net_ex_top5_tokens > 0) and (s_.mean_net_delayed_entry > 0)
    print(f"DECISION (pre-declared rule): {'FREEZE CANDIDATE' if ok else 'REJECT'} {name}")
