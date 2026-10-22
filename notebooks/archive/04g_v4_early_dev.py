# Databricks notebook source
# MAGIC %md
# MAGIC # 04g — v4 EARLY track: enter on the pump.fun bonding curve 1–15 min after creation
# MAGIC Branch `v3`. Dev data ≤ 2026-10-10 only (holdout untouched). Every model run here is a v4
# MAGIC variant and is logged in `docs/v4_variant_log.md` (budget ~6).
# MAGIC **Rows:** Gold entry candidates, pump.fun, `is_postgrad_at_t = false`, `price_unit = 'usd'`,
# MAGIC `minutes_since_launch` in [1, 15], T in 09-13 → 10-10 (same 4 walk-forward folds as v2/v3).
# MAGIC **Label/cost:** `r_up_360m` (2x before −50% within 6h). Curve fee 125 bps/side + 50 bps
# MAGIC slippage = 3.5% round trip (stress: 100 bps slippage = 4.5%).
# MAGIC **Known optimism to check before any freeze:** entry at the last closed bar's close (≤ 1–2
# MAGIC min stale) flatters momentum; the delayed-entry cell re-prices selected trades one bar later.
# MAGIC Prior result to beat (decisions.md, 2026-10-02): first-5-min entries ≈ −3.8% over 15 min
# MAGIC before fees; curve-phase model was ~breakeven and tail-driven.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pyspark.sql import functions as F

sys.path.append("../../src")
from backtest.shadow_eval_v1 import block_bootstrap_lower_bound, non_overlapping  # noqa: E402
from models import v2_spec as v2  # noqa: E402
from models.barrier_model import FEATURE_COLUMNS as V1_FEATURES  # noqa: E402
from serving.contract_v1 import round_trip_cost_v1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
CURVE_FLOW = ["curve_buy_share", "curve_max_trade_share", "curve_top_trader_share", "curve_top_trader_share_max",
              "flow_buy_share_15m", "flow_max_trade_share_15m", "flow_top_trader_share_15m"]
CR4 = ["cr4_launches_7d", "cr4_launches_1h", "cr4_grads_7d", "cr4_grad_rate_7d", "cr4_min_since_prev_launch",
       "launch_is_mayhem", "launch_non_sol_quote"]
COST = round_trip_cost_v1("pumpdotfun", False)
COST_S = round_trip_cost_v1("pumpdotfun", False, 100.0)
print(f"round trip cost base {COST:.4f}, stress {COST_S:.4f}")

g = (spark.table("lawrence.gold.entry_candidates")
     .filter("dex_venue = 'pumpdotfun' AND is_postgrad_at_t = false AND price_unit = 'usd'")
     .filter("minutes_since_launch >= 1 AND minutes_since_launch <= 15 AND r_up_360m IS NOT NULL")
     .filter("assessment_time >= '2026-09-13' AND assessment_time < '2026-10-11'")
     .select("token_id", "assessment_time", "price_at_t", *V1_FEATURES, *CURVE_FLOW, "r_up_360m", "t_up_360m"))
hist = spark.table("lawrence.gold.creator_launch_history_v4").select("token_id", *CR4)
n = g.count()
MAX_ROWS = 3_000_000  # driver-memory guard: keep a fixed hash-sample of TOKENS (all their rows) if larger
if n > MAX_ROWS:
    keep = MAX_ROWS / n
    g = g.filter((F.abs(F.xxhash64("token_id")) % 1000) < int(keep * 1000))
    print(f"{n:,} candidate rows -> token hash-sample {keep:.0%}")
else:
    print(f"{n:,} candidate rows")
df = g.join(hist, "token_id", "left").toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df = df.sort_values("assessment_time").reset_index(drop=True)
df["net"] = df["r_up_360m"] - COST
print(f"{len(df):,} rows, {df['token_id'].nunique():,} tokens; creator history coverage "
      f"{df['cr4_launches_7d'].notna().mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outcome check (no model; not a variant)
# MAGIC First row per token per age bucket = "buy every token at that age". Mean net after 3.5%.

# COMMAND ----------

df["age_b"] = pd.cut(df["minutes_since_launch"], [0.99, 2, 5, 10, 15], labels=["1-2m", "2-5m", "5-10m", "10-15m"])
df["cr_b"] = pd.cut(df["cr4_launches_7d"].fillna(-1), [-1.5, -0.5, 0.5, 10, 100, 1e9],
                    labels=["unknown", "first launch", "1-10", "11-100", ">100"])
df["has_grad"] = (df["cr4_grads_7d"].fillna(0) > 0)
first = df.sort_values("assessment_time").drop_duplicates(["token_id", "age_b"])


def _agg(x):
    return pd.Series({"tokens": len(x), "mean_r": x["r_up_360m"].mean(), "mean_net": x["net"].mean(),
                      "hit_2x": (x["r_up_360m"] >= 0.999).mean(), "hit_stop": (x["r_up_360m"] <= -0.499).mean()})


display(first.groupby("age_b", observed=True).apply(_agg).round(4).reset_index())
display(first.groupby(["cr_b", "has_grad"], observed=True).apply(_agg).round(4).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## V4a — LightGBM on v1 features + curve flow + creator history, rule R1 (p98)
# MAGIC Variant 1 of the v4 budget, declared in `docs/v4_variant_log.md` before running.

# COMMAND ----------

FEATS = list(dict.fromkeys([*V1_FEATURES, *CURVE_FLOW, *CR4]))
X = df[FEATS].copy()
for c in ("chain_family", "network", "dex_venue", "dex_protocol"):
    X[c] = X[c].astype("category")
X["is_postgrad_at_t"] = X["is_postgrad_at_t"].astype("float")
S = v2.TRAIN_START
edges = [S + pd.Timedelta(days=7 * k) for k in range(5)]
folds = [(S + pd.Timedelta(days=3), edges[1])] + [(edges[i], edges[i + 1]) for i in range(1, 4)]
parts, imps = [], []
for fi, (lo, hi) in enumerate(folds, start=1):
    tr = df["assessment_time"] < lo
    te = (df["assessment_time"] >= lo) & (df["assessment_time"] < hi)
    reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X[tr], df.loc[tr, "r_up_360m"])
    cut = float(np.quantile(reg.predict(X[tr]), 0.98))
    t = df[te].copy()
    t["score"] = reg.predict(X[te])
    sel = non_overlapping(t[t["score"] >= cut])
    sel = sel.assign(net_s=sel["r_up_360m"] - COST_S, fold=fi)
    parts.append(sel)
    imps.append(pd.Series(reg.booster_.feature_importance("gain"), index=X.columns))
    print(f"fold {fi}: train {tr.sum():,} test {te.sum():,} | trades {len(sel)} mean net {sel['net'].mean():+.4f} "
          f"| buy-all (first row/token) {t.drop_duplicates('token_id')['net'].mean():+.4f}")
v4a = pd.concat(parts)


def _summary(p, days=25):
    per_tok = p.groupby("token_id")["net"].sum().sort_values(ascending=False)
    return pd.Series({
        "trades": len(p), "per_day": len(p) / days, "tokens": p["token_id"].nunique(),
        "mean_net_base": p["net"].mean(), "mean_net_stress": p["net_s"].mean(),
        "lb95": block_bootstrap_lower_bound(p, p["net"], alpha=0.05) if len(p) >= 30 else np.nan,
        "median_net": p["net"].median(), "hit_2x": (p["r_up_360m"] >= 0.999).mean(),
        "mean_net_ex_top5_tokens": p[~p["token_id"].isin(per_tok.index[:5])]["net"].mean(),
        "positive_days_share": (p.groupby(p["assessment_time"].dt.floor("D"))["net"].sum() > 0).mean(),
        "folds_positive": int((p.groupby("fold")["net"].mean() > 0).sum()),
        "median_age_min": p["minutes_since_launch"].median()})


display(_summary(v4a).round(4).to_frame("V4a"))
display(v4a.groupby("fold").agg(trades=("net", "size"), mean_net=("net", "mean")).round(4))
imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
display((imp / imp.sum()).head(15).round(4).rename("gain_share").reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delayed-entry check on V4a's trades (realism, not a new variant)
# MAGIC Re-prices each selected trade as if filled at the close of the NEXT 1-min bar (T+1m)
# MAGIC and re-resolves the same 2x / −50% / 6h barrier from Silver bars.

# COMMAND ----------

from serving.contract_v1 import resolve_outcome  # noqa: E402

toks = v4a["token_id"].unique().tolist()
bars = (spark.table("lawrence.silver.token_price_1m").filter(F.col("token_id").isin(toks))
        .filter("price_unit = 'usd'")
        .select("token_id", "bar_start", "close", "high", "low", "volume_usd", "n_trades", "n_buys", "phase").toPandas())
bars.attrs = {}
bars["bar_start"] = pd.to_datetime(bars["bar_start"], utc=True)
by_tok = dict(tuple(bars.groupby("token_id")))
obs_end = pd.Timestamp("2026-10-11", tz="UTC")
delayed = []
for r in v4a.itertuples(index=False):
    b = by_tok.get(r.token_id)
    if b is None:
        delayed.append(np.nan)
        continue
    nxt = b[b["bar_start"] == r.assessment_time]  # bar [T, T+1m) closes at T+1m
    if nxt.empty or pd.isna(nxt["close"].iloc[0]):
        delayed.append(np.nan)
        continue
    out = resolve_outcome(b, {"assessment_time": r.assessment_time + pd.Timedelta(minutes=1),
                              "price_at_t": float(nxt["close"].iloc[0])},
                          launch_venue="pumpdotfun", observed_until=obs_end)
    delayed.append(np.nan if out is None else out["return"])
v4a["r_delayed"] = delayed
ok = v4a["r_delayed"].notna()
print(f"re-priced {ok.mean():.1%} of {len(v4a)} trades")
print(f"mean net, original entry: {(v4a.loc[ok, 'r_up_360m'] - COST).mean():+.4f} | "
      f"entry 1 bar later: {(v4a.loc[ok, 'r_delayed'] - COST).mean():+.4f}")
