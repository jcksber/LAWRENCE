# Databricks notebook source
# MAGIC %md
# MAGIC # 00g — Dune cost-test: bonding-curve 1-minute bars (pump.fun + LaunchLab)
# MAGIC
# MAGIC **Why (2026-10-03, see docs/decisions.md).** One month of data can't
# MAGIC tell a real tail-driven edge from luck, so we need more months.
# MAGIC Re-pulling raw curve trades costs ~75-90k credits/month (pump.fun's
# MAGIC ~108M trades). Gold only reads 1-minute bars, so this tests building
# MAGIC the bars on Dune instead (`src/ingestion/curve_bars.py`), with
# MAGIC per-bar trader stats added (buy/sell volume, largest trade, top
# MAGIC wallet's share of the bar).
# MAGIC
# MAGIC Target window: forward from 2026-09-13 (current LaunchLab regime).
# MAGIC Nothing is written anywhere. **Note your Dune credit balance before
# MAGIC and after EACH cell** and send the deltas. Stop if any cell costs more
# MAGIC than ~300 credits.

# COMMAND ----------

import importlib
import sys
from datetime import datetime, timedelta, timezone

sys.path.append("../../src")
import ingestion.curve_bars as curve_bars  # noqa: E402
import ingestion.postgrad_bars as postgrad_bars  # noqa: E402

importlib.reload(postgrad_bars)
importlib.reload(curve_bars)
from ingestion.curve_bars import (  # noqa: E402
    launchlab_curve_bars_sql,
    pumpfun_curve_bars_sql,
    summarize_sql,
)
from ingestion.dune_client import DuneClient  # noqa: E402

DUNE_API_KEY = dbutils.secrets.get(scope="lawrence", key="dune_api_key")
client = DuneClient(api_key=DUNE_API_KEY, poll_timeout_s=1800)

FORWARD_START = datetime(2026, 9, 13, tzinfo=timezone.utc)  # end of the current window
SAMPLE_DAY = datetime(2026, 9, 20, tzinfo=timezone.utc)      # one day inside the new window
VALIDATE_HOUR = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)  # inside the EXISTING window


def _ts(dt: datetime) -> str:
    return f"TIMESTAMP '{dt.strftime('%Y-%m-%d %H:%M:%S')}'"


def _run(label: str, sql: str, show: int = 50) -> list[dict]:
    print(f"--- {label} ---")
    rows = client.run_sql(sql)
    for r in rows[:show]:
        print(r)
    if len(rows) > show:
        print(f"... {len(rows):,} rows total")
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 (cheap): how far forward does Dune's data go, and daily volume
# MAGIC
# MAGIC Trades per day per venue since 2026-09-13 -- sizes the new window and
# MAGIC shows the last day Dune has fully loaded.

# COMMAND ----------

_run(
    "curve trades per day since FORWARD_START",
    f"""
SELECT date_trunc('day', block_time) AS day, project, COUNT(*) AS n_trades
FROM dex_solana.trades
WHERE block_time >= {_ts(FORWARD_START)}
  AND project IN ('pumpdotfun', 'raydium_launchlab')
  AND version = 1
GROUP BY 1, 2
ORDER BY 1, 2
""",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: pump.fun curve bars, ONE day, summary only
# MAGIC
# MAGIC Dune builds the full day of bars but returns one summary row, so the
# MAGIC credit delta is the compute cost (no export). This is the big one.

# COMMAND ----------

_run("pump.fun curve bars, 1 day (summary)",
     summarize_sql(pumpfun_curve_bars_sql(SAMPLE_DAY, SAMPLE_DAY + timedelta(days=1))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: LaunchLab curve bars, ONE day, summary only

# COMMAND ----------

_run("LaunchLab curve bars, 1 day (summary)",
     summarize_sql(launchlab_curve_bars_sql(SAMPLE_DAY, SAMPLE_DAY + timedelta(days=1))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: pump.fun curve bars, ONE hour, full export
# MAGIC
# MAGIC Actually downloads the bars, to measure the export cost per row on top
# MAGIC of compute. Compare this delta with cell 2's (/24).

# COMMAND ----------

_hour_rows = _run("pump.fun curve bars, 1 hour (export)",
                  pumpfun_curve_bars_sql(SAMPLE_DAY + timedelta(hours=12), SAMPLE_DAY + timedelta(hours=13)),
                  show=3)
print(f"exported {len(_hour_rows):,} bars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: validate against what we already have
# MAGIC
# MAGIC Same hour inside the EXISTING window, built both ways: Dune bars
# MAGIC (summary) vs our Silver `token_price_1m` curve bars. Bar and token
# MAGIC counts should be close. Ours keep only the busiest pool per token and
# MAGIC drop out-of-band prices, so a few % fewer is expected; a big gap means
# MAGIC the SQL is wrong.

# COMMAND ----------

_run("Dune: pump.fun curve bars, validation hour (summary)",
     summarize_sql(pumpfun_curve_bars_sql(VALIDATE_HOUR, VALIDATE_HOUR + timedelta(hours=1))))

display(spark.sql(f"""
SELECT count(*) AS n_bars, count(DISTINCT token_id) AS n_tokens, sum(n_trades) AS n_trades
FROM lawrence.silver.token_price_1m
WHERE phase = 'curve' AND venue = 'pumpdotfun'
  AND bar_start >= TIMESTAMP '{VALIDATE_HOUR:%Y-%m-%d %H:%M:%S}'
  AND bar_start <  TIMESTAMP '{(VALIDATE_HOUR + timedelta(hours=1)):%Y-%m-%d %H:%M:%S}'
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 (2026-10-04): CPMM post-graduation bars, ONE day, summary only
# MAGIC
# MAGIC 00f's window-2 CPMM job ran a fresh 2,500-credit key dry. The query is
# MAGIC now rewritten (narrow columns, no OR join). Measure it on one day of
# MAGIC graduations before re-running the 20-day job: multiply the credit
# MAGIC delta by ~20 for the full window.

# COMMAND ----------

importlib.reload(postgrad_bars)
from ingestion.postgrad_bars import cpmm_launchlab_bars_sql  # noqa: E402

_run("CPMM bars, graduations on 1 day (summary)", f"""
WITH bars AS ({cpmm_launchlab_bars_sql(SAMPLE_DAY, SAMPLE_DAY + timedelta(days=1), 24)})
SELECT count(*) AS n_bars, count(DISTINCT base_mint) AS n_tokens, sum(n_trades) AS n_trades
FROM bars
""")
