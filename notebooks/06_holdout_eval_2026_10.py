# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Pre-registered Dune holdout 2026-10-11 → 2026-11-01 (v1 and v2)
# MAGIC
# MAGIC Frozen with the v2 freeze commit (contracts/v2/PREREG.md). Run ONCE, after the
# MAGIC window is pulled (00h/00f window 4 → 00i URI backfill → 01b → 02). The verdicts
# MAGIC are whatever `holdout_eval_2026_10.evaluate_model` returns. No re-runs with
# MAGIC changes, no re-tuning.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import sys

import lightgbm as lgb
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append("../src")
from backtest import holdout_eval_2026_10 as H  # noqa: E402
from features.social_features import LAUNCH_TOOL_HOSTS, host_group  # noqa: E402
from models import v2_spec as v2  # noqa: E402
from serving import contract_v1 as c1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
m1 = c1.load_manifest("../contracts/v1")
m2 = json.load(open("../contracts/v2/manifest.json"))
b1 = c1.load_model("../contracts/v1")
p2 = "../contracts/v2/artifacts/" + m2["model_file"]
assert c1.sha256_file(p2) == m2["model_sha256"], "v2 model hash mismatch"
b2 = lgb.Booster(model_file=p2)
groups = json.load(open("../contracts/v2/artifacts/" + m2["host_groups_file"]))

# COMMAND ----------

# Candidates: post-grad Gold rows in the window, rule 7 (<= 18h after graduation).
cols = list(dict.fromkeys(["token_id", "assessment_time", *c1.FEATURE_COLUMNS,
                           *[c for c in v2.V2_FEATURES if c not in v2.HOST_FEATURES], "r_up_360m", "t_up_360m"]))
g = (spark.table("lawrence.gold.entry_candidates").filter("is_postgrad_at_t = true").select(*cols)
     .filter((F.to_timestamp("assessment_time") >= F.lit(H.WINDOW_START.to_pydatetime()))
             & (F.to_timestamp("assessment_time") < F.lit(H.WINDOW_END.to_pydatetime()))))
first_pg = (spark.table("lawrence.silver.token_price_1m").filter("phase = 'postgrad'")
            .groupBy("token_id").agg(F.min("bar_start").alias("first_pg")))
uris = spark.table("lawrence.bronze.pumpfun_coin_uris").dropDuplicates(["mint"]) \
    .select(F.col("mint").alias("token_id"), "metadata_host")
df = g.join(first_pg, "token_id", "left").join(uris, "token_id", "left").toPandas()
df.attrs = {}
df["assessment_time"] = pd.to_datetime(df["assessment_time"], utc=True)
df["first_pg"] = pd.to_datetime(df["first_pg"], utc=True)
df = df[df["assessment_time"] + pd.Timedelta(minutes=c1.HORIZON_MINUTES)
        <= df["first_pg"] + pd.Timedelta(hours=c1.POSTGRAD_OBSERVED_HOURS)].reset_index(drop=True)
# host features with the FROZEN map (unseen hosts -> other_custom); LaunchLab -> none
df["meta_host_group"] = [host_group(h, groups) if isinstance(h, str) else None for h in df["metadata_host"]]
df["meta_is_launch_tool"] = [float(h in LAUNCH_TOOL_HOSTS) if isinstance(h, str) else None for h in df["metadata_host"]]
print(f"{len(df):,} candidate rows, {df['token_id'].nunique():,} tokens; labelled: {df['r_up_360m'].notna().mean():.3f}")
print("pump.fun graduated tokens without a URI (backfill gap):",
      df[(df["dex_venue"] == "pumpdotfun") & df["metadata_host"].isna()]["token_id"].nunique())

# COMMAND ----------

# Excluded days: any 00h/00f chunk starting that day without a 'success' row.
prog = (spark.table("lawrence.bronze._curve_bars_progress").select("job", "chunk_start", "status")
        .unionByName(spark.table("lawrence.bronze._postgrad_progress").select("job", "chunk_start", "status")))
ok = prog.groupBy("job", "chunk_start").agg(F.max((F.col("status") == "success").cast("int")).alias("ok"))
bad = (ok.filter("ok = 0").select(F.to_date(F.to_timestamp("chunk_start")).alias("d")).distinct().toPandas())
excluded = [pd.Timestamp(d, tz="UTC") for d in bad["d"]
            if H.WINDOW_START <= pd.Timestamp(d, tz="UTC") < H.WINDOW_END]
print("excluded days:", excluded)

# COMMAND ----------

df["score_v1"] = b1.predict(c1.to_model_frame(df[c1.FEATURE_COLUMNS].to_dict("records"), m1["categories"]))
df["score_v2"] = b2.predict(v2.to_frame(df, m2["categories"]))
verdicts = {}
for name, cut in (("v1", m1["cutoff"]), ("v2", m2["cutoff"])):
    verdicts[name] = H.evaluate_model(df.assign(score=df[f"score_{name}"]), cut, excluded)
print(json.dumps(verdicts, indent=2, default=str))
