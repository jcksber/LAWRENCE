# Databricks notebook source
# MAGIC %md
# MAGIC # 04e — Freeze model v2 for the Dune holdout (contracts/v2/PREREG.md)
# MAGIC
# MAGIC Trains v2 (`src/models/v2_spec.py`) ONCE on post-graduation rows
# MAGIC 2026-09-13 → 2026-10-10 (rule 7), and writes to `/Volumes/lawrence/gold/contracts/v2/`:
# MAGIC `model_v2.txt`, `manifest.json` (sha256, cutoff = train p95, categories, versions),
# MAGIC `host_groups.json` (the frozen host → group map). Copy into the repo and commit:
# MAGIC **that commit is the v2 freeze**, and it must land BEFORE any data after
# MAGIC 2026-10-10 is pulled.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pyspark.sql import functions as F

sys.path.append("../src")
import features.social_features as sf  # noqa: E402
from models import v2_spec as v2  # noqa: E402
from serving.contract_v1 import HORIZON_MINUTES as N, POSTGRAD_OBSERVED_HOURS  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
OUT = "/Volumes/lawrence/gold/contracts/v2"
spark.sql("CREATE VOLUME IF NOT EXISTS lawrence.gold.contracts")
dbutils.fs.mkdirs(OUT)

cols = list(dict.fromkeys(["token_id", "assessment_time", *[c for c in v2.V2_FEATURES if c not in v2.HOST_FEATURES],
                           v2.TARGET, "t_up_360m"]))
g = spark.table("lawrence.gold.entry_candidates").filter(f"is_postgrad_at_t = true AND {v2.TARGET} IS NOT NULL").select(*cols)
first_pg = (spark.table("lawrence.silver.token_price_1m").filter("phase = 'postgrad'")
            .groupBy("token_id").agg(F.min("bar_start").alias("first_pg")))
df = g.join(first_pg, "token_id", "left").join(spark.table("lawrence.gold.social_token_features_v2"), "token_id", "left").toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df["first_pg"] = pd.to_datetime(df["first_pg"], utc=True)
df = df[(df["assessment_time"] >= v2.TRAIN_START) & (df["assessment_time"] < v2.TRAIN_END)
        & (df["assessment_time"] + pd.Timedelta(minutes=N) <= df["first_pg"] + pd.Timedelta(hours=POSTGRAD_OBSERVED_HOURS))]
categories = {c: sorted(str(v) for v in df[c].dropna().unique()) for c in v2.CATEGORICAL}
X = v2.to_frame(df, categories)
print(f"train: {len(X):,} rows, {df['token_id'].nunique():,} tokens, {df['assessment_time'].min()} -> {df['assessment_time'].max()}")

reg = LGBMRegressor(**v2.LGBM_PARAMS).fit(X, df[v2.TARGET])
pred = reg.predict(X)
cutoff = float(np.quantile(pred, 0.95))
print(f"cutoff (train p95): {cutoff:.4f}")

# COMMAND ----------

LOCAL = "/tmp/model_v2.txt"   # volumes can't append (see 04c)
reg.booster_.save_model(LOCAL)
sha = hashlib.sha256(open(LOCAL, "rb").read()).hexdigest()
assert np.allclose(lgb.Booster(model_file=LOCAL).predict(X), pred)
shutil.copyfile(LOCAL, f"{OUT}/model_v2.txt")

uri_pd = spark.table("lawrence.bronze.pumpfun_coin_uris").dropDuplicates(["mint"]) \
    .join(spark.table("lawrence.bronze.pumpfun_coins").dropDuplicates(["mint"]).select("mint", "created_at"), "mint", "left") \
    .select("mint", "metadata_host", "created_at").toPandas()
host_groups = sf.build_host_groups(uri_pd["metadata_host"], uri_pd["created_at"], v2.TRAIN_END)
with open(f"{OUT}/host_groups.json", "w") as f:
    json.dump(host_groups, f, indent=2, sort_keys=True)

manifest = {
    "contract_version": "v2", "status": "holdout_only", "model_file": "model_v2.txt", "model_sha256": sha,
    "spec": "src/models/v2_spec.py", "features": v2.V2_FEATURES, "categories": categories,
    "target": v2.TARGET, "lgbm_params": v2.LGBM_PARAMS, "cutoff": cutoff,
    "cutoff_rule": "95th percentile of predicted EV on the training rows",
    "train_rows": int(len(X)), "train_tokens": int(df["token_id"].nunique()),
    "train_assessment_time_min": df["assessment_time"].min().isoformat(),
    "train_assessment_time_max": df["assessment_time"].max().isoformat(),
    "host_groups_file": "host_groups.json", "n_host_groups": len(set(host_groups.values())),
    "libraries": {k: version(k) for k in ("lightgbm", "pandas", "numpy", "scikit-learn")},
    "python": sys.version.split()[0], "created_at": datetime.now(timezone.utc).isoformat(),
}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps({k: manifest[k] for k in ("model_sha256", "train_rows", "cutoff", "train_assessment_time_max")}, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Copy into the repo (laptop, repo root), then commit with `contracts/v2/PREREG.md`:
# MAGIC ```
# MAGIC databricks fs cp -r dbfs:/Volumes/lawrence/gold/contracts/v2 contracts/v2/artifacts --overwrite
# MAGIC mv contracts/v2/artifacts/manifest.json contracts/v2/manifest.json
# MAGIC ```
