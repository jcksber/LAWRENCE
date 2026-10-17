# Databricks notebook source
# MAGIC %md
# MAGIC # 00i — Bronze: pump.fun coin metadata + callouts (social step 1)
# MAGIC
# MAGIC **2026-10-17 rewrite** (docs/decisions.md, "pump.fun comments removed"):
# MAGIC pump.fun dropped per-coin comment threads, so this pulls the two public
# MAGIC replacements via `src/ingestion/pumpfun_social.py`:
# MAGIC - `lawrence.bronze.pumpfun_coins`: one row per mint; socials from the
# MAGIC   launch-time IPFS metadata (point-in-time safe).
# MAGIC - `lawrence.bronze.pumpfun_callouts`: timestamped callouts. Max 100 per
# MAGIC   coin, ranked by post-hoc performance -> a coin returning 100 is flagged
# MAGIC   `callouts_truncated` in the progress table and gets NO callout features.
# MAGIC
# MAGIC Progress per mint in `lawrence.bronze._pumpfun_social_progress`;
# MAGIC re-running skips `success` / `not_found`. Duplicates from a re-fetch are
# MAGIC removed downstream on `mint` / `callout_id`.
# MAGIC
# MAGIC Scope: `graduated` pump.fun tokens first, then `sample` (fixed 50k hash
# MAGIC sample of curve-only tokens). **First run: cell 3 probe, then cell 4 with
# MAGIC `MAX_TOKENS = 200`.**

# COMMAND ----------

import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

sys.path.append("../src")
import ingestion.pumpfun_social as pumpfun_social  # noqa: E402

importlib.reload(pumpfun_social)
from ingestion.pumpfun_social import (  # noqa: E402
    BASE_URL,
    CALLOUT_LIMIT,
    HEADERS,
    SocialUnavailable,
    fetch_coin_social,
    to_callout_row,
    to_coin_row,
)
from pyspark.sql import Row  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

spark.conf.set("spark.sql.session.timeZone", "UTC")

PRICE_TABLE = "lawrence.silver.token_price_1m"
COINS_TABLE = "lawrence.bronze.pumpfun_coins"
CALLOUTS_TABLE = "lawrence.bronze.pumpfun_callouts"
PROGRESS_TABLE = "lawrence.bronze._pumpfun_social_progress"

TOKEN_SCOPE = "graduated"   # "graduated" first; then "sample"
SAMPLE_SIZE = 50_000        # curve-only tokens for scope "sample"
MAX_TOKENS = 200            # first run: 200, check timing + rows, then None
WORKERS = 4                 # parallel tokens (3 requests each); lower on 429s
BATCH = 100                 # mints per write

# COMMAND ----------

# Candidate mints: pump.fun curve tokens in Silver; graduated = has postgrad bars.
tokens = (
    spark.table(PRICE_TABLE)
    .groupBy("token_id")
    .agg(
        F.min(F.when(F.col("phase") == "curve", F.col("bar_start"))).alias("launch_time"),
        F.min_by("venue", "bar_start").alias("launch_venue"),
        F.max((F.col("phase") == "postgrad").cast("int")).alias("graduated"),
    )
    .filter((F.col("launch_venue") == "pumpdotfun") & F.col("launch_time").isNotNull())
)
if TOKEN_SCOPE == "graduated":
    scope = tokens.filter("graduated = 1")
else:
    scope = tokens.filter("graduated = 0").orderBy(F.xxhash64("token_id")).limit(SAMPLE_SIZE)

if spark.catalog.tableExists(PROGRESS_TABLE):
    done = spark.table(PROGRESS_TABLE).filter("status IN ('success', 'not_found')").select(
        F.col("mint").alias("token_id")
    )
    scope = scope.join(done, "token_id", "left_anti")

todo_df = scope.orderBy("launch_time")
if MAX_TOKENS is not None:
    todo_df = todo_df.limit(MAX_TOKENS)
todo = [r.token_id for r in todo_df.select("token_id").collect()]
print(f"scope={TOKEN_SCOPE}: {tokens.filter('graduated = 1').count():,} graduated pump.fun tokens; "
      f"{len(todo):,} to fetch this run")

# COMMAND ----------

# PROBE: raw responses for the first mint -- check status + field names.
if todo:
    m = todo[0]
    for path, params in ((f"/coins-v2/{m}", None), (f"/callout/coin/{m}", {"limit": 5})):
        resp = requests.get(BASE_URL + path, params=params, headers=HEADERS, timeout=30)
        print(path, "HTTP", resp.status_code)
        print(resp.text[:800], "\n")
    out = fetch_coin_social(m)
    print("launch metadata:", json.dumps(out["meta"])[:500] if out["meta"] else None)
    print("callouts:", len(out["callouts"]), "truncated:", out["truncated"])

# COMMAND ----------

_COINS_SCHEMA = StructType(
    [StructField(c, StringType()) for c in ("mint", "name", "symbol", "creator")]
    + [StructField("created_at", TimestampType()), StructField("program", StringType()),
       StructField("description_len", LongType())]
    + [StructField(c, StringType()) for c in ("launch_twitter", "launch_telegram", "launch_website")]
    + [StructField("has_launch_metadata", BooleanType()), StructField("current_twitter", StringType())]
    + [StructField(c, BooleanType()) for c in ("complete", "nsfw", "is_banned", "hidden", "is_backfill")]
    + [StructField("source", StringType()), StructField("ingested_at", TimestampType())]
)
_CALLOUTS_SCHEMA = StructType([
    StructField("mint", StringType()),
    StructField("callout_id", StringType()),
    StructField("user_id", StringType()),
    StructField("created_at", TimestampType()),
    StructField("market_cap_at_call", DoubleType()),
    StructField("callout_price_usd", DoubleType()),
    StructField("thesis_len", LongType()),
    StructField("has_media", BooleanType()),
    StructField("is_quote", BooleanType()),
    StructField("chain", StringType()),
    StructField("likes", LongType()),
    StructField("reply_count", LongType()),
    StructField("repost_count", LongType()),
    StructField("quote_count", LongType()),
    StructField("posthoc_max_multiplier", DoubleType()),
    StructField("source", StringType()),
    StructField("ingested_at", TimestampType()),
])
_PROGRESS_SCHEMA = StructType([
    StructField("mint", StringType()),
    StructField("status", StringType()),
    StructField("n_callouts", LongType()),
    StructField("callouts_truncated", BooleanType()),
    StructField("has_launch_metadata", BooleanType()),
    StructField("error", StringType()),
    StructField("completed_at", TimestampType()),
])


def _bool(v):
    return None if v is None else bool(v)


def _one(mint: str, session: requests.Session):
    try:
        out = fetch_coin_social(mint, session)
        status = "success" if out["coin"] is not None else "not_found"
        return mint, status, out, None
    except SocialUnavailable as exc:
        return mint, "failed", None, str(exc)
    except Exception as exc:  # bad JSON / shape change -- log, keep going
        return mint, "failed", None, f"{type(exc).__name__}: {exc}"


session = requests.Session()
ingested_at = datetime.now(timezone.utc)
started = datetime.now(timezone.utc)
n_coins = n_callouts = 0

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for b in range(0, len(todo), BATCH):
        results = list(pool.map(lambda m: _one(m, session), todo[b : b + BATCH]))
        coin_rows, callout_rows, prog = [], [], []
        for mint, st, out, err in results:
            if out and out["coin"] is not None:
                r = to_coin_row(out["coin"], out["meta"], mint, ingested_at)
                for k in ("complete", "nsfw", "is_banned", "hidden", "is_backfill"):
                    r[k] = _bool(r[k])
                coin_rows.append(r)
                callout_rows += [to_callout_row(c, mint, ingested_at) for c in out["callouts"]]
            prog.append(Row(
                mint=mint, status=st,
                n_callouts=len(out["callouts"]) if out else 0,
                callouts_truncated=bool(out and out["truncated"]),
                has_launch_metadata=bool(out and out["meta"]),
                error=err, completed_at=datetime.now(timezone.utc),
            ))
        if coin_rows:
            spark.createDataFrame(coin_rows, schema=_COINS_SCHEMA).write.format("delta") \
                .mode("append").saveAsTable(COINS_TABLE)
        if callout_rows:
            spark.createDataFrame(callout_rows, schema=_CALLOUTS_SCHEMA).write.format("delta") \
                .mode("append").saveAsTable(CALLOUTS_TABLE)
        spark.createDataFrame(prog, schema=_PROGRESS_SCHEMA).write.format("delta") \
            .mode("append").saveAsTable(PROGRESS_TABLE)

        n_coins += len(coin_rows)
        n_callouts += len(callout_rows)
        sts = [p.status for p in prog]
        done_n = b + len(results)
        rate = done_n / max((datetime.now(timezone.utc) - started).total_seconds(), 1)
        print(f"[{done_n:,}/{len(todo):,}] +{len(coin_rows)} coins +{len(callout_rows):,} callouts | "
              f"ok={sts.count('success')} 404={sts.count('not_found')} failed={sts.count('failed')} "
              f"truncated={sum(p.callouts_truncated for p in prog)} | {rate:.1f} mints/s", flush=True)
        refused = [p.error for p in prog if p.error and ("HTTP 401" in p.error or "HTTP 403" in p.error)]
        if refused and len(refused) == len(prog):
            print(f"Every request in this batch was refused -- stopping. First error: {refused[0]}")
            break

print(f"\nDone this run: {n_coins:,} coins, {n_callouts:,} callouts.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks

# COMMAND ----------

display(spark.sql(f"""
    SELECT status, COUNT(*) AS mints,
           ROUND(AVG(CAST(has_launch_metadata AS INT)), 3) AS metadata_share,
           ROUND(AVG(CASE WHEN n_callouts > 0 THEN 1 ELSE 0 END), 3) AS any_callout_share,
           ROUND(AVG(CAST(callouts_truncated AS INT)), 3) AS truncated_share,
           percentile_approx(n_callouts, 0.5) AS median_callouts,
           percentile_approx(n_callouts, 0.9) AS p90_callouts,
           MIN(error) AS example_error
    FROM {PROGRESS_TABLE} GROUP BY status
"""))

# COMMAND ----------

# Socials at launch, and when callouts land relative to launch / graduation.
display(spark.sql(f"""
    SELECT COUNT(*) AS coins,
           ROUND(AVG(CASE WHEN launch_twitter IS NOT NULL THEN 1 ELSE 0 END), 3) AS launch_twitter_share,
           ROUND(AVG(CASE WHEN launch_telegram IS NOT NULL THEN 1 ELSE 0 END), 3) AS launch_telegram_share,
           ROUND(AVG(CASE WHEN launch_website IS NOT NULL THEN 1 ELSE 0 END), 3) AS launch_website_share,
           ROUND(AVG(CASE WHEN COALESCE(launch_twitter, '') <> COALESCE(current_twitter, '') THEN 1 ELSE 0 END), 3)
               AS twitter_changed_share
    FROM {COINS_TABLE}
"""))

# COMMAND ----------

display(spark.sql(f"""
    WITH tok AS (
        SELECT token_id,
               MIN(CASE WHEN phase = 'curve' THEN bar_start END) AS launch_time,
               MIN(CASE WHEN phase = 'postgrad' THEN bar_start END) AS grad_time
        FROM {PRICE_TABLE} GROUP BY token_id
    )
    SELECT COUNT(*) AS callouts,
           SUM(CASE WHEN c.created_at IS NULL THEN 1 ELSE 0 END) AS null_ts,
           ROUND(AVG(CASE WHEN c.created_at < t.grad_time THEN 1 ELSE 0 END), 3) AS before_graduation,
           ROUND(AVG(CASE WHEN c.created_at < t.grad_time + INTERVAL 6 HOURS THEN 1 ELSE 0 END), 3)
               AS within_6h_of_grad,
           percentile_approx((unix_timestamp(c.created_at) - unix_timestamp(t.launch_time)) / 60, 0.5)
               AS median_min_after_launch,
           COUNT(DISTINCT c.user_id) AS distinct_callers
    FROM {CALLOUTS_TABLE} c JOIN tok t ON c.mint = t.token_id
"""))
