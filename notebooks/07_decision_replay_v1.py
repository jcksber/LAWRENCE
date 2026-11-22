# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Decision engine replay parity (v1 contract)  · spec `docs/decision_agent_spec.md` §7
# MAGIC Feeds historical Silver 1-minute bars through `decision.Engine` minute by minute (as a live feed would)
# MAGIC with the frozen v1 contract, a zero-slippage `PaperBroker` and no portfolio limits, then compares the
# MAGIC engine's trades with the backtest's trades (Gold rows scored by the same model, same cutoff, one position
# MAGIC per token, `shadow_eval_v1.frozen_trades`). **PASS = identical trades and returns.**
# MAGIC Dev data only (tokens graduating 2026-10-03 → 10-06). No Dune credits. Start with `N_TOKENS = 50`.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import time

import pandas as pd
from pyspark.sql import functions as F

sys.path.append("../src")
from backtest.shadow_eval_v1 import frozen_trades  # noqa: E402
from decision.brokers import PaperBroker  # noqa: E402
from decision.contracts import ContractV1  # noqa: E402
from decision.engine import Engine  # noqa: E402
from decision.replay import parity, replay  # noqa: E402
from decision.types import RiskConfig  # noqa: E402
from serving import contract_v1 as c1  # noqa: E402

spark.conf.set("spark.sql.session.timeZone", "UTC")
N_TOKENS = 50
GRAD_FROM, GRAD_TO = "2026-10-03", "2026-10-06"
contract = ContractV1("../contracts/v1")          # hash check + verify_startup
print("v1 cutoff", contract.cutoff)

px = spark.table("lawrence.silver.token_price_1m")
tok = (px.groupBy("token_id")
       .agg(F.min(F.when(F.col("phase") == "curve", F.col("bar_start"))).alias("launch_time"),
            F.min_by("venue", "bar_start").alias("launch_venue"),
            F.min(F.when(F.col("phase") == "postgrad", F.col("bar_start"))).alias("first_pg"))
       .filter(F.col("launch_time").isNotNull() & (F.col("first_pg") >= GRAD_FROM) & (F.col("first_pg") < GRAD_TO))
       .filter(F.col("launch_venue").isin(*c1.UNIVERSE_VENUES))
       .orderBy(F.xxhash64("token_id")).limit(N_TOKENS))
ctx = tok.toPandas()
ctx.attrs = {}
ids = ctx["token_id"].tolist()
bars = (px.filter(F.col("token_id").isin(ids)).select("token_id", *c1.BAR_COLUMNS).toPandas())
bars.attrs = {}
bars["bar_start"] = pd.to_datetime(bars["bar_start"], utc=True)
print(f"{len(ids)} tokens, {len(bars):,} bars")

# COMMAND ----------

# Engine replay (per token, no portfolio limits, so tokens are independent)
t0 = time.time()
engine = Engine(contract, PaperBroker(0.0), RiskConfig.unlimited())
tokens = {r.token_id: {"bars": bars[bars["token_id"] == r.token_id].drop(columns="token_id"),
                       "launch_time": pd.Timestamp(r.launch_time, tz="UTC") if pd.Timestamp(r.launch_time).tzinfo is None
                       else pd.Timestamp(r.launch_time), "launch_venue": r.launch_venue}
          for r in ctx.itertuples(index=False)}
eng_trades = replay(engine, tokens)
print(f"engine: {len(eng_trades)} trades in {time.time() - t0:.0f}s; audit rows {len(engine.log.rows):,}")

# COMMAND ----------

# Backtest trades: the same tokens' Gold rows, same model + cutoff, rule 7, one position per token
cols = list(dict.fromkeys(["token_id", "assessment_time", *c1.FEATURE_COLUMNS, "dex_venue", "r_up_360m", "t_up_360m"]))
g = (spark.table("lawrence.gold.entry_candidates").filter(F.col("token_id").isin(ids))
     .filter("is_postgrad_at_t = true").select(*cols).toPandas())
g.attrs = {}
g["assessment_time"] = pd.to_datetime(g["assessment_time"], utc=True)
g = g.merge(ctx[["token_id", "first_pg"]], on="token_id")
g["first_pg"] = pd.to_datetime(g["first_pg"], utc=True)
g = g[g["assessment_time"] + pd.Timedelta(minutes=c1.HORIZON_MINUTES)
      <= g["first_pg"] + pd.Timedelta(hours=c1.POSTGRAD_OBSERVED_HOURS)].reset_index(drop=True)
g["score"] = contract.booster.predict(c1.to_model_frame(g[c1.FEATURE_COLUMNS].to_dict("records"), contract.manifest["categories"]))
bt = frozen_trades(g, contract.cutoff)
print(f"backtest: {len(bt)} trades")

# COMMAND ----------

res = parity(eng_trades, bt)
print(res)
e = eng_trades.rename(columns={"entry_time": "assessment_time"})[["token_id", "assessment_time", "r_gross", "exit_reason"]]
e["assessment_time"] = pd.to_datetime(e["assessment_time"], utc=True)
m = e.merge(bt[["token_id", "assessment_time", "r_up_360m", "score"]], on=["token_id", "assessment_time"], how="outer", indicator=True)
display(m[m["_merge"] != "both"].head(20))
display(engine.log.frame()["event"].value_counts().rename("events").to_frame())

# COMMAND ----------

# DEBUG (2026-10-25): why does the engine skip the backtest's 06:25 entry on token Hmuv...?
TOK = "HmuvHCDA7MjAxKr2BrARGhsWcxKotADopeLjcUTnGHt5"
d = tokens[TOK]
b = d["bars"]
print("duplicate bar_start rows:", int(b.duplicated("bar_start").sum()), "| phases:", b["phase"].value_counts().to_dict())
au = engine.log.frame()
display(au[au["token_id"] == TOK])
for hhmm in ("06:10", "06:25", "06:40", "06:55"):
    t = pd.Timestamp(f"2026-10-05 {hhmm}", tz="UTC")
    closed = b[b["bar_start"] + pd.Timedelta(minutes=1) <= t]
    row, why = c1.select_row(closed, t, launch_time=d["launch_time"], launch_venue=d["launch_venue"], observed_from_launch=True)
    s_eng = float(contract.booster.predict(c1.to_model_frame([row], contract.manifest["categories"]))[0]) if row else None
    gr = g[(g["token_id"] == TOK) & (g["assessment_time"] == t)]
    s_gold = float(gr["score"].iloc[0]) if len(gr) else None
    print(hhmm, "| select_row:", why, "| engine score:", s_eng, "| gold score:", s_gold)
    if row is not None and len(gr):
        diffs = {k: (row.get(k), gr[k].iloc[0]) for k in c1.FEATURE_COLUMNS
                 if str(row.get(k)) != str(gr[k].iloc[0])}
        print("   feature differences (engine, gold):", diffs)
