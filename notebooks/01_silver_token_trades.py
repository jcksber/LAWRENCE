# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Silver: `lawrence.silver.token_trades`
# MAGIC
# MAGIC Reads Bronze's raw two-sided swaps and writes one row per
# MAGIC (swap, token-of-interest) into `lawrence.silver.token_trades` -- see
# MAGIC `docs/silver_spec.md` for the schema and rationale.
# MAGIC
# MAGIC The transform lives in `src/features/silver_spark.py` (Spark-native,
# MAGIC distributed). `src/features/silver_token_trades.py::to_silver_rows()`
# MAGIC is the pure-Python reference it's parity-tested against.
# MAGIC
# MAGIC **2026-09-22 rewrite** (see `docs/decisions.md`): the previous version
# MAGIC `.collect()`ed each Bronze chunk onto the driver and transformed it
# MAGIC row by row in Python -- unworkable at the 131M-row Dune backfill. It
# MAGIC also deduped on `(dex_venue, tx_hash, token_id)`, which deleted ~3.8M
# MAGIC real multi-swap-per-tx rows, and stored `trade_time` as a mixed-format
# MAGIC string. All three fixed here.
# MAGIC
# MAGIC Two modes:
# MAGIC - `live` (default -- what the scheduled job runs): re-derives the
# MAGIC   last `LOOKBACK_MINUTES` of Bronze *by `ingested_at`* and MERGEs on
# MAGIC   `(swap_id, token_id)`, so overlapping runs are a no-op.
# MAGIC - `rebuild`: rebuilds all of Silver from all of Bronze in one
# MAGIC   set-based, partitioned overwrite. Use after a backfill or a
# MAGIC   transform/denylist change. Also migrates the table's schema.

# COMMAND ----------

import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../src")
from features.silver_spark import build_silver, pool_pairs  # noqa: E402
from features.silver_token_trades import EVM_QUOTE_DENYLIST, SOLANA_QUOTE_DENYLIST  # noqa: E402
from delta.tables import DeltaTable  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.window import Window  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

BRONZE_TABLE = "lawrence.bronze.dex_trades"
SILVER_TABLE = "lawrence.silver.token_trades"

# "live" | "rebuild" -- see header. Leave "live" in git; flip to "rebuild"
# only for a manual run.
MODE = "live"

# Live mode only. Filters on Bronze's `ingested_at` (a real TimestampType),
# NOT `block_time` (a mixed-format string -- comparing it as text is what
# made the old window filter wrong). Wider than the 15-min cadence so a
# late run doesn't lose rows; the MERGE makes the overlap a no-op.
LOOKBACK_MINUTES = 60

# 2026-10-01: LaunchLab's token-of-interest comes from each pool's base
# mint (lawrence.bronze.pool_metadata, written by 00f) instead of the quote
# denylist -- see silver_spark.METADATA_BASE_VENUES. Falls back to the
# denylist everywhere if the metadata table doesn't exist yet.
POOL_METADATA_TABLE = "lawrence.bronze.pool_metadata"
PAIRS = pool_pairs(spark.table(POOL_METADATA_TABLE)) if spark.catalog.tableExists(POOL_METADATA_TABLE) else None
print("pool metadata:", "loaded" if PAIRS is not None else "NOT FOUND -- denylist only")

# COMMAND ----------

if MODE == "rebuild":
    silver_df = build_silver(spark.table(BRONZE_TABLE), PAIRS)
    (
        silver_df.write.format("delta")
        .mode("overwrite")
        # Schema changed 2026-09-22 (trade_time string -> timestamp; new
        # swap_id, trade_date). Delta keeps the old version for time travel
        # if this ever needs rolling back.
        .option("overwriteSchema", "true")
        .partitionBy("trade_date")
        .saveAsTable(SILVER_TABLE)
    )
    print(f"Rebuilt {SILVER_TABLE} from all of {BRONZE_TABLE}.")

elif MODE == "live":
    since = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
    new_df = build_silver(spark.table(BRONZE_TABLE).filter(F.col("ingested_at") >= F.lit(since)), PAIRS)
    (
        DeltaTable.forName(spark, SILVER_TABLE)
        .alias("t")
        .merge(
            new_df.alias("s"),
            # trade_date first so Delta can prune partitions instead of
            # scanning the whole table to match a handful of new rows.
            "t.trade_date = s.trade_date AND t.swap_id = s.swap_id AND t.token_id = s.token_id",
        )
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"Merged Bronze rows ingested since {since} into {SILVER_TABLE}.")

else:
    raise ValueError(f"Unknown MODE: {MODE!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC Plain aggregations over the written table (distributed, nothing
# MAGIC collected but the small results). Cheap enough to run every time.

# COMMAND ----------

silver = spark.table(SILVER_TABLE)

# 1. Row counts / time range per venue. trade_time is a real timestamp now,
#    so MIN/MAX are chronological, not lexical.
display(
    silver.groupBy("dex_venue")
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("token_id").alias("tokens"),
        F.min("trade_time").alias("min_trade_time"),
        F.max("trade_time").alias("max_trade_time"),
    )
    .orderBy("dex_venue")
)

# 2. Unparseable timestamps -- should be 0. Non-zero means Bronze has a
#    third block_time shape parse_block_time() doesn't know about.
null_ts = silver.filter(F.col("trade_time").isNull()).count()
print(f"Rows with unparseable trade_time: {null_ts:,}")

# 3. Denylist sanity: the top quote_token_ids per venue should all be on
#    the denylist. A top one that ISN'T means a real quote asset is missing
#    (e.g. pumpdotfun/raydium_launchlab trading against something other than
#    WSOL/USDC/USDT), so its swaps were split into two "token of interest"
#    rows each. A legit hub token (cbBTC on Base, etc.) can also appear --
#    judge by address before adding it; if you do add one, re-run
#    MODE = "rebuild".
_denylist = SOLANA_QUOTE_DENYLIST | {a.lower() for a in EVM_QUOTE_DENYLIST}
top_quotes = (
    silver.groupBy("dex_venue", "quote_token_id")
    .count()
    .withColumn("share", F.round(F.col("count") / F.sum("count").over(Window.partitionBy("dex_venue")), 4))
    .withColumn("rank", F.row_number().over(Window.partitionBy("dex_venue").orderBy(F.desc("count"))))
    .filter("rank <= 5")
    .withColumn(
        "on_denylist",
        F.col("quote_token_id").isin(list(_denylist)) | F.lower("quote_token_id").isin(list(_denylist)),
    )
    .orderBy("dex_venue", "rank")
)
display(top_quotes)

# 4. LaunchLab: with pool metadata, every swap should produce exactly ONE
#    Silver row (the pool's base). >1 means the metadata didn't match and
#    the denylist fallback split the swap -- report how many.
display(
    silver.filter(F.col("dex_venue") == "raydium_launchlab")
    .groupBy("swap_id").count()
    .groupBy(F.col("count").alias("rows_per_swap")).count()
    .orderBy("rows_per_swap")
)
