# Databricks notebook source
# MAGIC %md
# MAGIC # 04i — Freeze the v2-R1 cutoff (pre-registration amendment 1)
# MAGIC Uses the FROZEN v2 model (hash-checked, not retrained) on the SAME training rows as 04e and records
# MAGIC the 98th percentile of its predicted EV = the R1 cutoff. Sanity checks: the training row count and the
# MAGIC 95th percentile must reproduce `contracts/v2/manifest.json` exactly. Reads data ≤ 2026-10-10 only.
# MAGIC Output: `amendment_r1.json` in the contracts volume → copy into `contracts/v2/` and commit with
# MAGIC `PREREG_AMENDMENT_1.md` BEFORE the holdout pull.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import json
import sys
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

sys.path.append("../src")
from models import v2_spec as v2  # noqa: E402
from serving.contract_v1 import HORIZON_MINUTES as N, POSTGRAD_OBSERVED_HOURS  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
OUT = "/Volumes/lawrence/gold/contracts/v2"
m2 = json.load(open("../contracts/v2/manifest.json"))
p2 = "../contracts/v2/artifacts/" + m2["model_file"]
assert hashlib.sha256(open(p2, "rb").read()).hexdigest() == m2["model_sha256"], "v2 model hash mismatch"
b2 = lgb.Booster(model_file=p2)

# training rows: identical to 04e
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
X = v2.to_frame(df, m2["categories"])
pred = b2.predict(X)
p95, p98 = float(np.quantile(pred, 0.95)), float(np.quantile(pred, 0.98))
print(f"train rows {len(X):,} (manifest {m2['train_rows']:,}) | p95 {p95:.6f} (manifest {m2['cutoff']:.6f}) | p98 {p98:.6f}")
assert len(X) == m2["train_rows"], "training rows differ from the freeze -- STOP, do not amend"
assert abs(p95 - m2["cutoff"]) < 1e-9, "p95 does not reproduce the frozen cutoff -- STOP, do not amend"

# COMMAND ----------

amend = {
    "amendment": 1, "model": "v2-R1", "model_file": m2["model_file"], "model_sha256": m2["model_sha256"],
    "cutoff": p98, "cutoff_rule": "98th percentile of the frozen v2 model's predicted EV on its training rows",
    "check_p95_reproduces_manifest_cutoff": p95, "train_rows": int(len(X)),
    "alpha_per_model": 0.10 / 3, "models_in_family": ["v1", "v2", "v2-R1"],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
with open(f"{OUT}/amendment_r1.json", "w") as f:
    json.dump(amend, f, indent=2)
print(json.dumps(amend, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Copy into the repo (laptop, repo root), then commit with `contracts/v2/PREREG_AMENDMENT_1.md`:
# MAGIC ```
# MAGIC databricks fs cp dbfs:/Volumes/lawrence/gold/contracts/v2/amendment_r1.json contracts/v2/amendment_r1.json
# MAGIC ```
