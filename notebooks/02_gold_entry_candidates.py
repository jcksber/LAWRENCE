# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Gold: `lawrence.gold.entry_candidates` (training data)
# MAGIC
# MAGIC **This is a batch/backfill job, not a scheduled pipeline step.** It
# MAGIC regenerates the entire entry-model training table from whatever is
# MAGIC currently in Silver -- unlike Bronze/Silver, it isn't wired into
# MAGIC `resources/lawrence_job.yml`'s recurring schedule. Re-run it by hand
# MAGIC whenever Silver has enough new history to be worth retraining on.
# MAGIC (A production-serving path -- T = "now," incremental, per active
# MAGIC token -- is future work once there's a model to feed; see
# MAGIC docs/gold_spec.md.)
# MAGIC
# MAGIC The actual feature/label math lives in
# MAGIC `src/features/gold_features.py` (plain Python, no PySpark -- unit
# MAGIC tested, runs in CI). This notebook is the Spark glue: read all of
# MAGIC Silver, group by token, sample assessment points, call into that
# MAGIC module, write the result.
# MAGIC
# MAGIC **2026-09-17: three attempts at real-scale volume, see docs/decisions.md.**
# MAGIC Against the real ~10.75M-row Dune backfill (this notebook was written
# MAGIC when Gold's own docs expected "few or zero rows" and was never
# MAGIC exercised at scale until now):
# MAGIC 1. The original `silver_df.collect()` -- everything into one Python
# MAGIC    list on the driver -- hit the same driver-memory hang already seen
# MAGIC    in Silver's own catch-up notebook.
# MAGIC 2. Replaced with `toLocalIterator()` (stream partitions to the driver
# MAGIC    one at a time) plus a manual per-token grouping loop. Ran into a
# MAGIC    suspected recompute-per-partition problem, for which the standard
# MAGIC    fix (`.cache()`) turned out to be unsupported on serverless compute
# MAGIC    (`PERSIST TABLE is not supported on serverless`).
# MAGIC 3. **Current approach:** stop routing 10M+ rows through the driver at
# MAGIC    all. `silver_df.groupBy("token_id").applyInPandas(...)` pushes the
# MAGIC    per-token feature/label computation onto the cluster as a real,
# MAGIC    distributed Spark operation -- each executor handles a subset of
# MAGIC    tokens in parallel, calling straight into the same
# MAGIC    `gold_features.py` functions. No driver memory pressure, no
# MAGIC    caching, no ordering assumptions to get right by hand -- Spark's
# MAGIC    own groupBy shuffle guarantees each token's rows land together.
# MAGIC    This also means the notebook's own Spark job progress (Tasks X/Y)
# MAGIC    is now an honest signal of real completion, since the heavy work
# MAGIC    happens as part of that job instead of invisibly afterward on the
# MAGIC    driver.

# COMMAND ----------

import sys

sys.path.append("../src")
# Force-reload our modules each run (stale-import lesson, 2026-10-01).
import importlib  # noqa: E402

import features.gold_features  # noqa: E402
import features.gold_input  # noqa: E402

importlib.reload(features.gold_features)
importlib.reload(features.gold_input)
from features.gold_features import BARRIER_HORIZONS_MIN, FLOW_FEATURE_KEYS, LABEL_HOLD_MINUTES, build_entry_row, sample_assessment_times  # noqa: E402
from features.gold_input import GOLD_INPUT_COLUMNS, gold_input  # noqa: E402

# COMMAND ----------

# 2026-10-01 (Gold v2): read the per-token 1-minute price series, not raw
# trades -- see src/features/gold_input.py for the candidate/censoring rules.
from datetime import datetime, timezone  # noqa: E402

PRICE_TABLE = "lawrence.silver.token_price_1m"
SILVER_TABLE = PRICE_TABLE  # kept so the diagnostics below read the same table
# End of the bonding-curve backfill (00c's BACKFILL_WINDOW_END): a curve-only
# token is unobserved after this, so labels reaching past it are censored.
CURVE_DATA_END = datetime(2026, 10, 10, tzinfo=timezone.utc)  # 2026-10-11: + out-of-sample window (was 10-03, before that 09-13)
GOLD_TABLE = "lawrence.gold.entry_candidates"

# COMMAND ----------

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema, same rationale as Bronze/Silver's _BRONZE_SCHEMA/
# _SILVER_SCHEMA (see docs/decisions.md): avoids CANNOT_DETERMINE_TYPE if a
# nullable column (e.g. trailing_price_change_pct, network) happens to be
# all-None, and doubles as applyInPandas's required return schema below.
_ENTRY_SCHEMA = StructType([
    StructField("token_id", StringType()),
    StructField("chain_family", StringType()),
    StructField("network", StringType()),
    StructField("dex_venue", StringType()),
    StructField("dex_protocol", StringType()),
    StructField("assessment_time", StringType()),
    StructField("price_at_t", DoubleType()),
    StructField("hold_minutes", IntegerType()),
    StructField("y_entry", DoubleType()),
    StructField("is_rug_clamped", BooleanType()),
    StructField("trailing_volume_usd", DoubleType()),
    StructField("trailing_trade_count", IntegerType()),
    StructField("trailing_buy_count", IntegerType()),
    StructField("trailing_sell_count", IntegerType()),
    StructField("trailing_buy_sell_ratio", DoubleType()),
    StructField("trailing_price_change_pct", DoubleType()),
    # Gold v2 (2026-10-01)
    StructField("price_unit", StringType()),
    StructField("minutes_since_launch", DoubleType()),
    StructField("is_postgrad_at_t", BooleanType()),
]
# 2026-10-16: on-chain flow features + minutes since graduation (gold_features
# FLOW_FEATURE_KEYS); NULL where the bars carry no flow stats.
+ [StructField(k, DoubleType()) for k in FLOW_FEATURE_KEYS]
# 2026-10-01: one label column per horizon in LABEL_HOLD_MINUTES (5/15/60),
# NULL where that horizon is censored. `y_entry` itself = HOLD_MINUTES (15).
+ [StructField(f"y_entry_{h}m", DoubleType()) for h in LABEL_HOLD_MINUTES]
# 2026-10-08: barrier labels -- hit 2x before -50% within N (see gold_features).
# y_up is 1.0 / 0.0 / NULL -- double, not int, so NULL survives pandas.
+ [StructField(f"y_up_{n}m", DoubleType()) for n in BARRIER_HORIZONS_MIN]
+ [StructField(f"r_up_{n}m", DoubleType()) for n in BARRIER_HORIZONS_MIN]
+ [StructField(f"t_up_{n}m", DoubleType()) for n in BARRIER_HORIZONS_MIN]
)
_ENTRY_SCHEMA_COLUMNS = [f.name for f in _ENTRY_SCHEMA]

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

# Silver's trade_time is a TimestampType since 2026-09-22; applyInPandas
# converts it to tz-naive pandas Timestamps in the SESSION time zone, and
# gold_features._parse treats naive as UTC -- so pin the session to UTC.
spark.conf.set("spark.sql.session.timeZone", "UTC")

# Only the columns gold_features.py actually reads (its price_at/
# trailing_activity_features/build_entry_row/build_exit_rows) -- Silver's
# _SILVER_SCHEMA (notebooks/01) carries 13 columns total; tx_fee_usd,
# chain_native_fee, tx_hash, and quote_token_id are dead weight here.
# tx_hash alone is a long string on every one of 10M+ rows, so trimming
# this before the groupBy shuffle cuts real memory/network cost for every
# token, not just the outliers identified in docs/decisions.md.
_GOLD_COLUMNS = GOLD_INPUT_COLUMNS


# 2026-09-17 (see docs/decisions.md): a UDF_PYSPARK_ERROR.OOM crashed the
# Python worker running _process_token_group. gold_features.py needs a
# token's FULL trade history in memory to sample point-in-time-correct
# assessment points, so a token with a pathologically large trade count
# can't be chunked the way everything else in this codebase is -- one
# executor has to hold it all at once regardless. The diagnostic query
# below identified the real top-10 by trade count (182,807-659,071
# trades): cbBTC, Virtual Protocol, Raydium, Aerodrome, Zora, and other
# legitimate, established assets -- NOT a data bug, and NOT tokens to
# exclude. Jack: the bot needs to handle both memecoins and established/
# blue-chip assets, so this cap is a crash-guard against a genuinely
# pathological trade count (e.g. wash-trading/bot activity -- see the
# EDYNE anomaly flagged in docs/decisions.md), not a filter on which
# tokens are "worth" training on. Set comfortably above every real token
# seen so far; the select() below (trimming to only the columns
# gold_features.py actually reads, before the groupBy shuffle) is the
# real fix for per-group memory pressure, so this should rarely if ever
# trigger.
_MAX_TRADES_PER_TOKEN = 2_000_000


def _process_token_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs on an executor, once per distinct token_id (Spark's groupBy
    shuffle guarantees every row for a token lands in the same call).
    applyInPandas does NOT guarantee row order within a group, so sort
    locally first -- gold_features.py's whole contract depends on `trades`
    being sorted ascending by trade_time."""
    pdf = pdf.sort_values("trade_time")
    trades = pdf.to_dict("records")
    if not trades:
        return pd.DataFrame(columns=_ENTRY_SCHEMA_COLUMNS)

    token_id = trades[0]["token_id"]
    if len(trades) > _MAX_TRADES_PER_TOKEN:
        # print() here goes to executor logs, not this notebook's output --
        # not visible to Jack -- which is exactly why the diagnostic query
        # below runs on the driver, and why the distinct-token-count check
        # after the write exists: this is a last-resort circuit breaker for
        # a pathological trade count, not a way to silently exclude real
        # assets -- any token this actually skips needs to surface, not
        # vanish quietly.
        return pd.DataFrame(columns=_ENTRY_SCHEMA_COLUMNS)
    out_rows = []
    ref = trades[0]
    for t in sample_assessment_times(trades, launch_time=ref["launch_time"]):
        row = build_entry_row(
            token_id, t, trades,
            observed_until=ref["observed_until"], launch_time=ref["launch_time"],
        )
        if row is not None:
            row["price_unit"] = ref["price_unit"]
            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=_ENTRY_SCHEMA_COLUMNS)
    return pd.DataFrame(out_rows)[_ENTRY_SCHEMA_COLUMNS]


# Driver-side visibility into the trade-count distribution -- a plain
# Spark aggregation (cheap, no per-token materialization risk), run BEFORE
# applyInPandas so its output is visible in this notebook's cell output,
# unlike print() inside _process_token_group (executor logs only). Also
# how to tell whether _MAX_TRADES_PER_TOKEN actually needs tuning.
_gold_input_df = gold_input(spark.table(PRICE_TABLE), CURVE_DATA_END).select(*GOLD_INPUT_COLUMNS)

_top_tokens = (
    _gold_input_df
    .groupBy("token_id")
    .count()
    .orderBy(F.desc("count"))
    .limit(10)
    .collect()
)
print("Top 10 tokens by number of 1-minute bars:")
for _r in _top_tokens:
    _flag = " <- EXCEEDS _MAX_TRADES_PER_TOKEN, will be skipped" if _r["count"] > _MAX_TRADES_PER_TOKEN else ""
    print(f"  {_r['token_id']}: {_r['count']:,} bars{_flag}")

# 2026-09-17 (see docs/decisions.md): real distinct-token count before the
# write, so the post-write comparison below can actually prove nothing
# vanished silently -- not just "the job didn't crash."
_silver_token_count = _gold_input_df.select("token_id").distinct().count()  # entry candidates (tokens with a curve phase)

gold_df = (
    _gold_input_df
    .groupBy("token_id")
    .applyInPandas(_process_token_group, schema=_ENTRY_SCHEMA)
)

# COMMAND ----------

# 2026-09-17: the lawrence.gold SCHEMA itself (not just the table) had
# never been created -- bronze/silver's schemas exist from some earlier
# manual setup, gold's never did since no run had gotten this far before.
# Idempotent, so this is a no-op on every later run.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_TABLE.rsplit('.', 1)[0]}")

(
    gold_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)
_written = spark.table(GOLD_TABLE).count()
print(f"Wrote {_written} rows to {GOLD_TABLE} (full overwrite -- this "
      "is a batch regenerate, not an incremental append; re-running from "
      "scratch is the intended usage, unlike Bronze/Silver's MERGE)")
# A low/zero count against a freshly-started Bronze/Silver history is
# expected, not a bug: build_entry_row needs both a trailing feature window
# AND a real trade near T + HOLD_MINUTES, which a just-started ingestion
# history mostly won't have yet. Re-run this after Bronze/Silver have
# accumulated more history.
print(
    f"Distinct tokens: {_silver_token_count} in Silver, "
    f"{spark.table(GOLD_TABLE).select('token_id').distinct().count()} made it into "
    f"{GOLD_TABLE}. A gap here is real and should be explained (e.g. a token skipped "
    f"by _MAX_TRADES_PER_TOKEN, or one with no assessment point that had both a usable "
    f"price/feature anchor and enough trailing history) -- never treat it as fine by "
    f"default."
)

# COMMAND ----------

# 2026-10-08: barrier-label check. Per horizon: how often the price hits 2x
# before -50% (y_up = 1), how often it's censored (NULL), the average
# realized return of the rule before costs, and the median minutes to
# resolution. Split by venue and phase at T.
_g = spark.table(GOLD_TABLE)
display(
    _g.groupBy("dex_venue", "is_postgrad_at_t")
    .agg(
        F.count("*").alias("rows"),
        *[F.avg(f"y_up_{n}m").alias(f"p_up_{n}m") for n in BARRIER_HORIZONS_MIN],
        *[F.avg(F.col(f"y_up_{n}m").isNull().cast("double")).alias(f"censored_{n}m") for n in BARRIER_HORIZONS_MIN],
        *[F.avg(f"r_up_{n}m").alias(f"mean_r_{n}m") for n in BARRIER_HORIZONS_MIN],
        *[F.percentile_approx(f"t_up_{n}m", 0.5).alias(f"median_min_{n}m") for n in BARRIER_HORIZONS_MIN],
    )
    .orderBy("dex_venue", "is_postgrad_at_t")
)
