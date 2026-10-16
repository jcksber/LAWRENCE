# Databricks notebook source
# MAGIC %md
# MAGIC # 00i — Bronze: pump.fun replies (social step 1)
# MAGIC
# MAGIC **Why (2026-10-16, docs/decisions.md "Social sources").** The free
# MAGIC social signal that covers every pump.fun token: each token's comment
# MAGIC thread, with post timestamps (so features at T count replies posted by T).
# MAGIC Unofficial API -- **run cell 3 (probe) first** and check the response
# MAGIC shape before the bulk pull.
# MAGIC
# MAGIC Writes `lawrence.bronze.pumpfun_replies`; progress per mint in
# MAGIC `lawrence.bronze._pumpfun_replies_progress`. Resumable: re-running
# MAGIC skips mints already logged `success` / `not_found`. Duplicates from a
# MAGIC re-fetch are removed downstream on `(mint, reply_id)`.
# MAGIC
# MAGIC Scope order: `graduated` pump.fun tokens first (Model A trains on
# MAGIC post-graduation rows), then `sample` = a fixed hash-ordered sample of
# MAGIC curve-only tokens. Doesn't read or write anything the pre-registered
# MAGIC test uses.

# COMMAND ----------

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

sys.path.append("../src")
import ingestion.pumpfun_replies as pumpfun_replies  # noqa: E402

importlib.reload(pumpfun_replies)
from ingestion.pumpfun_replies import (  # noqa: E402
    BASE_URL,
    HEADERS,
    RepliesUnavailable,
    fetch_replies,
    to_reply_row,
)
from pyspark.sql import Row  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    BooleanType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

spark.conf.set("spark.sql.session.timeZone", "UTC")

PRICE_TABLE = "lawrence.silver.token_price_1m"
REPLIES_TABLE = "lawrence.bronze.pumpfun_replies"
PROGRESS_TABLE = "lawrence.bronze._pumpfun_replies_progress"

TOKEN_SCOPE = "graduated"   # "graduated" first; then "sample"
SAMPLE_SIZE = 50_000        # curve-only tokens for scope "sample"
MAX_TOKENS = 200            # first run: 200, check timing + rows, then None
WORKERS = 4                 # parallel requests; lower if you see 429s
BATCH = 100                 # mints per write

try:
    AUTH_TOKEN = dbutils.secrets.get(scope="lawrence", key="pumpfun_jwt")
except Exception:
    AUTH_TOKEN = None  # only needed if the probe returns 401

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

# PROBE: raw response for the first mint -- check status, shape, field names.
probe_mint = todo[0] if todo else None
if probe_mint:
    h = dict(HEADERS)
    if AUTH_TOKEN:
        h["Authorization"] = f"Bearer {AUTH_TOKEN}"
    resp = requests.get(f"{BASE_URL}/replies/{probe_mint}", params={"limit": 5, "offset": 0},
                        headers=h, timeout=30)
    print(probe_mint, "HTTP", resp.status_code)
    print(resp.text[:1500])

# COMMAND ----------

_REPLIES_SCHEMA = StructType([
    StructField("mint", StringType()),
    StructField("reply_id", StringType()),
    StructField("user_address", StringType()),
    StructField("username", StringType()),
    StructField("posted_at", TimestampType()),
    StructField("text", StringType()),
    StructField("text_len", LongType()),
    StructField("has_image", BooleanType()),
    StructField("total_likes", LongType()),
    StructField("is_hidden", BooleanType()),
    StructField("is_buy", BooleanType()),
    StructField("sol_amount", LongType()),
    StructField("source", StringType()),
    StructField("ingested_at", TimestampType()),
])
_PROGRESS_SCHEMA = StructType([
    StructField("mint", StringType()),
    StructField("status", StringType()),
    StructField("n_replies", LongType()),
    StructField("error", StringType()),
    StructField("completed_at", TimestampType()),
])


def _one(mint: str, session: requests.Session):
    try:
        return mint, "success", fetch_replies(mint, session, auth_token=AUTH_TOKEN), None
    except RepliesUnavailable as exc:
        return mint, ("not_found" if exc.status == 404 else "failed"), [], str(exc)
    except Exception as exc:  # bad JSON / shape change -- log, keep going
        return mint, "failed", [], f"{type(exc).__name__}: {exc}"


session = requests.Session()
ingested_at = datetime.now(timezone.utc)
total_replies = 0
started = datetime.now(timezone.utc)

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for b in range(0, len(todo), BATCH):
        results = list(pool.map(lambda m: _one(m, session), todo[b : b + BATCH]))
        rows = [to_reply_row(r, mint, ingested_at) for mint, st, reps, _ in results for r in reps]
        if rows:
            spark.createDataFrame(rows, schema=_REPLIES_SCHEMA).write.format("delta") \
                .mode("append").saveAsTable(REPLIES_TABLE)
        now = datetime.now(timezone.utc)
        spark.createDataFrame(
            [Row(mint=m, status=st, n_replies=len(reps), error=err, completed_at=now)
             for m, st, reps, err in results],
            schema=_PROGRESS_SCHEMA,
        ).write.format("delta").mode("append").saveAsTable(PROGRESS_TABLE)
        total_replies += len(rows)
        statuses = [st for _, st, _, _ in results]
        auth_fail = [err for _, st, _, err in results if st == "failed" and err and ("401" in err or "403" in err)]
        rate = (b + len(results)) / max((now - started).total_seconds(), 1)
        print(f"[{b + len(results):,}/{len(todo):,}] +{len(rows):,} replies | "
              f"ok={statuses.count('success')} 404={statuses.count('not_found')} "
              f"failed={statuses.count('failed')} | {rate:.1f} mints/s", flush=True)
        if auth_fail and len(auth_fail) == len(results):
            print("Every request in this batch was refused (401/403) -- stopping. "
                  f"First error: {auth_fail[0]}")
            break

print(f"\nDone this run: {total_replies:,} replies.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks

# COMMAND ----------

display(spark.sql(f"""
    SELECT status, COUNT(*) AS mints, SUM(n_replies) AS replies,
           ROUND(AVG(n_replies), 1) AS mean_replies,
           percentile_approx(n_replies, 0.5) AS median_replies,
           MAX(n_replies) AS max_replies, MIN(error) AS example_error
    FROM {PROGRESS_TABLE} GROUP BY status
"""))

# COMMAND ----------

# Point-in-time sanity: replies per token relative to launch, and NULL timestamps.
display(spark.sql(f"""
    WITH launch AS (
        SELECT token_id, MIN(bar_start) AS launch_time
        FROM {PRICE_TABLE} WHERE phase = 'curve' GROUP BY token_id
    )
    SELECT COUNT(*) AS replies,
           SUM(CASE WHEN r.posted_at IS NULL THEN 1 ELSE 0 END) AS null_ts,
           SUM(CASE WHEN r.posted_at < l.launch_time THEN 1 ELSE 0 END) AS before_first_bar,
           ROUND(AVG(CASE WHEN r.posted_at <= l.launch_time + INTERVAL 60 MINUTES THEN 1 ELSE 0 END), 3)
               AS share_first_hour,
           ROUND(AVG(CAST(r.is_hidden AS INT)), 3) AS hidden_share,
           COUNT(DISTINCT r.user_address) AS distinct_users
    FROM {REPLIES_TABLE} r JOIN launch l ON r.mint = l.token_id
"""))
