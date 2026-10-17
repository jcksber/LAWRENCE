# Databricks notebook source
# MAGIC %md
# MAGIC # 04c — Export scoring contract v1 (shadow mode)
# MAGIC
# MAGIC Trains the v1 model ONCE with the frozen Model A spec on ALL
# MAGIC post-graduation Gold rows with a resolved 6h barrier label, then writes
# MAGIC to `/Volumes/lawrence/gold/contracts/v1/`:
# MAGIC - `model_v1.txt` -- LightGBM booster (text, includes category mappings)
# MAGIC - `manifest.json` -- version, sha256, training range, library versions,
# MAGIC   feature list + category sets, cutoff, costs, horizon
# MAGIC - `baselines.json` -- per-feature PSI bins, null rates, category
# MAGIC   frequencies, score distribution, expected trades/day
# MAGIC - `startup_sample.parquet` -- 1,000 training rows + their predictions
# MAGIC   (the runtime's startup check)
# MAGIC
# MAGIC Training rows obey the live selection rule, including the 18h horizon.
# MAGIC
# MAGIC Then copy them into the repo (cell at the end) and commit. **The commit
# MAGIC that adds `manifest.json` is the contract freeze.** Don't re-run this
# MAGIC after the shadow window starts -- a new model is a new contract (v2).
# MAGIC
# MAGIC Status: shadow only. This spec failed its pre-registered test (2026-10-16).

# COMMAND ----------

# MAGIC %pip install lightgbm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.append("../src")
from backtest.walk_forward import round_trip_cost, venue_fee_bps  # noqa: E402
from models.barrier_model import FEATURE_COLUMNS, label_columns, prepare_barrier_features  # noqa: E402
from models.entry_model import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS, set_categorical_dtypes  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from serving.contract_v1 import (  # noqa: E402
    CONTRACT_VERSION,
    POSTGRAD_OBSERVED_HOURS,
    FEATURE_COLUMNS as CONTRACT_FEATURES,
    HORIZON_MINUTES,
    SLIPPAGE_BPS_BASE,
    SLIPPAGE_BPS_STRESS,
)

assert CONTRACT_FEATURES == FEATURE_COLUMNS, "contract feature list drifted from barrier_model"
spark.conf.set("spark.sql.session.timeZone", "UTC")
OUT = f"/Volumes/lawrence/gold/contracts/{CONTRACT_VERSION}"
spark.sql("CREATE VOLUME IF NOT EXISTS lawrence.gold.contracts")
dbutils.fs.mkdirs(OUT)

# COMMAND ----------

N = HORIZON_MINUTES
y_col, r_col = label_columns(N)
cols = ["token_id", "assessment_time", *FEATURE_COLUMNS, y_col, r_col, f"t_up_{N}m"]
df = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true")
      .select(*dict.fromkeys(cols)).toPandas())
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df = set_categorical_dtypes(df.sort_values("assessment_time").reset_index(drop=True))
# 2026-10-17 (R3): apply the live selection rule (CONTRACT §4 rule 7) to
# training too. Gold keeps a row past grad + 18h when its barrier happened to
# resolve before the 24h data edge -- selection on the OUTCOME. Live excludes
# all of those, so training must as well, or the two universes differ.
first_pg = (spark.table("lawrence.silver.token_price_1m").filter("phase = 'postgrad'")
            .groupBy("token_id").agg(F.min("bar_start").alias("first_postgrad_bar")).toPandas())
first_pg["first_postgrad_bar"] = pd.to_datetime(first_pg["first_postgrad_bar"], utc=True)
df = df.merge(first_pg, on="token_id", how="left")
in_horizon = (df["assessment_time"] + pd.Timedelta(minutes=N)
              <= df["first_postgrad_bar"] + pd.Timedelta(hours=POSTGRAD_OBSERVED_HOURS))
print(f"rows past the 18h training horizon (dropped): {(~in_horizon & df[y_col].notna()).sum():,} "
      f"of {df[y_col].notna().sum():,} labelled")
train = df[df[y_col].notna() & in_horizon]
X, y, r = prepare_barrier_features(train, N)
print(f"train: {len(X):,} rows, {train['token_id'].nunique():,} tokens, "
      f"{train['assessment_time'].min()} -> {train['assessment_time'].max()}")

reg = LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05, verbose=-1).fit(X, r)
pred = reg.predict(X)
cutoff = float(np.quantile(pred, 0.95))
print(f"cutoff (train 95th pct of predicted EV): {cutoff:.4f}")

# COMMAND ----------

model_file = f"model_{CONTRACT_VERSION}.txt"
reg.booster_.save_model(f"{OUT}/{model_file}")
with open(f"{OUT}/{model_file}", "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

# reload check: identical predictions from the saved file
import lightgbm as lgb  # noqa: E402
booster = lgb.Booster(model_file=f"{OUT}/{model_file}")
assert np.allclose(booster.predict(X), pred), "saved model predicts differently"

categories = {c: [str(v) for v in X[c].cat.categories] for c in CATEGORICAL_FEATURE_COLUMNS}

# Startup check (R3): a fixed sample + its predictions; the runtime refuses to
# start unless contract_v1.verify_startup reproduces them.
startup = X.sample(n=min(1000, len(X)), random_state=17)
startup_df = startup.copy()
for c in CATEGORICAL_FEATURE_COLUMNS:
    startup_df[c] = startup_df[c].astype(object)
startup_df["expected_prediction"] = booster.predict(startup)
startup_df.attrs = {}  # Spark Connect PlanMetrics in attrs break to_parquet (00j, 2026-10-17)
startup_df.to_parquet(f"{OUT}/startup_sample.parquet", index=False)

# expected trades/day: frozen rule (EV >= cutoff, one position per token,
# held until resolution) replayed on the last 7 training days. IN-SAMPLE, so
# likely optimistic; the failed OOS test saw ~145/day at its own cutoff.
last7 = train[train["assessment_time"] >= train["assessment_time"].max() - pd.Timedelta(days=7)]
sig = last7.assign(ev=reg.predict(X.loc[last7.index]))
sig = sig[sig["ev"] >= cutoff].sort_values(["token_id", "assessment_time"])
kept, open_until = 0, {}
for tok, ts, mins in zip(sig["token_id"], sig["assessment_time"], sig[f"t_up_{N}m"].fillna(N)):
    if tok in open_until and ts < open_until[tok]:
        continue
    kept += 1
    open_until[tok] = ts + pd.Timedelta(minutes=float(mins))
trades_per_day = kept / 7

manifest = {
    "contract_version": CONTRACT_VERSION,
    "status": "shadow_only",
    "model_file": model_file,
    "model_sha256": sha,
    "model_params": {"class": "LGBMRegressor", "random_state": 17, "n_estimators": 300, "learning_rate": 0.05},
    "target": r_col,
    "train_rows": int(len(X)),
    "train_tokens": int(train["token_id"].nunique()),
    "train_assessment_time_min": train["assessment_time"].min().isoformat(),
    "train_assessment_time_max": train["assessment_time"].max().isoformat(),
    "feature_columns": FEATURE_COLUMNS,
    "categories": categories,
    "cutoff": cutoff,
    "cutoff_rule": "95th percentile of predicted EV on the training rows (rows within the 18h horizon, CONTRACT §4 rule 7)",
    "startup_sample_file": "startup_sample.parquet",
    "startup_tolerance": 1e-9,  # R4: catches wrong file/version/frame (those differ by >>1e-6), not last-bit FP noise
    "horizon_minutes": N,
    "barrier": {"up": 2.0, "down": 0.5},
    "slippage_bps_per_side": {"base": SLIPPAGE_BPS_BASE, "stress": SLIPPAGE_BPS_STRESS},
    "libraries": {k: version(k) for k in ("lightgbm", "pandas", "numpy", "scikit-learn")},
    "python": sys.version.split()[0],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# COMMAND ----------

# Drift baselines (PSI-ready): decile edges on training rows, proportions per
# bin, null rate; category frequencies; score distribution.
def _bins(s):
    v = pd.to_numeric(s, errors="coerce")
    edges = np.unique(np.nanquantile(v.dropna(), np.linspace(0, 1, 11)))
    counts = np.histogram(v.dropna(), bins=edges)[0]
    return {"edges": edges.tolist(), "proportions": (counts / max(counts.sum(), 1)).tolist(),
            "null_rate": float(v.isna().mean())}

baselines = {
    "contract_version": CONTRACT_VERSION,
    "numeric": {c: _bins(X[c]) for c in NUMERIC_FEATURE_COLUMNS + ["minutes_since_launch"]},
    "categorical": {c: X[c].astype(object).fillna("<null>").value_counts(normalize=True).to_dict()
                    for c in CATEGORICAL_FEATURE_COLUMNS},
    "score": {**_bins(pd.Series(pred)), "cutoff": cutoff, "share_at_or_above_cutoff": float((pred >= cutoff).mean())},
    "label_base_rates": {"hit_2x_first": float(y.mean()), "mean_gross_return": float(r.mean())},
    "expected_trades_per_day": {"in_sample_last7d": trades_per_day, "oos_2026_10_03_test_at_its_cutoff": 1144 / 7.9,
                                "note": "in-sample replay; treat as an upper-ish guide"},
}
with open(f"{OUT}/baselines.json", "w") as f:
    json.dump(baselines, f, indent=2, default=float)

print(json.dumps({k: manifest[k] for k in ("model_sha256", "train_rows", "cutoff",
                                            "train_assessment_time_max", "libraries")}, indent=2))
print("expected trades/day (in-sample last 7d):", round(trades_per_day, 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Copy into the repo
# MAGIC From your laptop, in the repo root:
# MAGIC ```
# MAGIC databricks fs cp -r dbfs:/Volumes/lawrence/gold/contracts/v1 contracts/v1/artifacts
# MAGIC mv contracts/v1/artifacts/manifest.json contracts/v1/manifest.json
# MAGIC ```
# MAGIC Then commit `contracts/v1/` -- that commit is the freeze.
