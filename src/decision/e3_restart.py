"""E3 paper replay rehearsal over one immutable bar file. Spark imports stay inside the Delta I/O."""
import gzip
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.ladder_exit import simulate_schedule
from decision.contracts import ContractV1
from decision.e3 import E3PaperEngine
from decision.replay import replay
from decision.types import RiskConfig

TABLE = "lawrence.paper_sprint.e3_restart_runs_v1"
INJECTED = "E3_TEST_CRASH_AFTER_COMMITTED_2X"


def inputs(fixture, contract_dir, bars_file):
    fixture = Path(fixture)
    manifest = json.loads((fixture / "manifest.json").read_text())
    contract = ContractV1(str(contract_dir))
    if (contract.manifest["model_sha256"], contract.cutoff) != (manifest["model_sha256"], manifest["cutoff"]):
        raise ValueError("Fixture requires a different entry contract")
    ctx = pd.read_parquet(fixture / "token_context.parquet").set_index("token_id")
    bars = pd.read_parquet(bars_file)
    if len(bars) != manifest["bars"] or bars.duplicated(["token_id", "bar_start"]).any():
        raise ValueError("Expected one complete primary price series per token")
    if len(ctx) != manifest["tokens"] or not ctx.index.is_unique or set(ctx.index) != set(bars.token_id):
        raise ValueError("Token context differs from the frozen cohort")
    tokens = {token: {"bars": group.drop(columns="token_id"),
                      "launch_time": pd.to_datetime(ctx.at[token, "launch_time"], utc=True),
                      "launch_venue": ctx.at[token, "launch_venue"]}
              for token, group in bars.groupby("token_id")}
    return contract, tokens


def check_research(engine, tokens):
    """Check every selected trade against the independent batch E3 simulator.
    The simulator marks unsold inventory at data end. The paper account retains it."""
    for p in [*engine.closed, *engine.open.values()]:
        bars = tokens[p.token_id]["bars"].query("phase == 'postgrad'").sort_values("bar_start")
        seconds = (bars.bar_start - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() + 60
        side = engine.contract.round_trip_cost(p.launch_venue) / 2
        ref = simulate_schedule(seconds, bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy(),
                                p.entry_time.timestamp(), p.entry_price, side)
        assert p.next_rung == ref["rungs"], p.position_id
        if p.remaining:
            assert ref["exit"] == "end_of_data" and p.exit_time is None, p.position_id
            marks = bars.loc[seconds > p.entry_time.timestamp(), "close"].dropna()
            mark = marks.iloc[-1] if len(marks) else p.entry_price
            multiple = p.proceeds_usd / p.notional_usd + p.remaining * mark / p.fill_price
            gross, net = multiple - 1, multiple * (1 - side) - (1 + side)
        else:
            assert p.exit_reason == ref["exit"] and p.exit_time.timestamp() == ref["exit_ts"], p.position_id
            gross, net = p.r_gross, p.r_net
        np.testing.assert_allclose([gross, net], [ref["gross"], ref["net"]], atol=1e-12, rtol=1e-12)


def load_record(spark, trial, phase):
    from pyspark.sql import functions as F
    rows = (spark.table(TABLE).where((F.col("trial") == trial) & (F.col("phase") == phase))
            .select("payload").take(2))
    if not rows:
        return None
    record, = rows
    return json.loads(gzip.decompress(record.payload))


def save_record(spark, trial, phase, record):
    """A single Delta transaction contains the replay cursor, account state and ordered audit/fills."""
    payload = gzip.compress(json.dumps(record, allow_nan=False, separators=(",", ":")).encode())
    row = spark.createDataFrame([(trial, phase, payload)], "trial string, phase string, payload binary")
    (row.write.format("delta").mode("append").option("txnAppId", f"e3/{trial}/{phase}")
     .option("txnVersion", 0).saveAsTable(TABLE))
    if load_record(spark, trial, phase) != record:
        raise RuntimeError(f"Different record already exists for {trial}/{phase}")


def verify_restart(reference, actual):
    for key in reference:
        if reference[key] != actual[key]:
            raise AssertionError(f"Restart differs from uninterrupted replay: {key}")
    ids = [(row["position_id"], row["event"], row["reason"]) for row in actual["audit"]
           if row["event"] in ("enter", "partial_exit", "exit")]
    assert len(ids) == len(set(ids)), "Duplicate fill in paper ledger"
    return len(ids)


def process_batch(batch, batch_id, *, fixture, contract_dir, trial, mode, session_id):
    spark = batch.sparkSession
    rows = batch.select("content").take(2)
    if not rows:
        return
    bar_file, = rows  # Exactly one input file in this bounded restart test.
    contract, tokens = inputs(fixture, contract_dir, BytesIO(bar_file.content))
    checkpoint = load_record(spark, trial, "half_sale")
    if mode == "crash":
        if checkpoint is not None:
            raise RuntimeError(INJECTED)

        def commit_and_stop(engine, p, now):
            assert p.remaining == 0.5 and p.next_rung == 1
            state = engine.snapshot()
            baseline = load_record(spark, trial, "baseline")
            first = next(i for i, event in enumerate(baseline["audit"])
                         if event["event"] == "partial_exit" and event["reason"] == "2x")
            if state["audit"] != baseline["audit"][:first + 1]:
                raise AssertionError("Interrupted ledger differs at the first 2x sale")
            record = {"tick": now.isoformat(), "token": p.token_id, "position": p.position_id,
                      "batch": batch_id, "session": session_id, "state": state}
            save_record(spark, trial, "half_sale", record)
            raise RuntimeError(INJECTED)

        engine = E3PaperEngine(contract, RiskConfig.unlimited(), on_first_rung=commit_and_stop)
        replay(engine, tokens)
        raise AssertionError("Fixture produced no 2x half-sale")
    if checkpoint is None or checkpoint["batch"] != batch_id or checkpoint["session"] == session_id:
        raise RuntimeError("Expected the interrupted micro-batch from a previous run")
    engine = E3PaperEngine.restore(contract, checkpoint["state"])
    p = engine.open[checkpoint["token"]]
    assert p.position_id == checkpoint["position"] and p.remaining == 0.5 and p.next_rung == 1
    # Revisit this exact (minute, token). The saved next_rung suppresses 2x;
    # higher targets on this bar and later tokens in this minute still execute.
    replay(engine, tokens, resume_at=(pd.Timestamp(checkpoint["tick"]), p.token_id))
    state = engine.snapshot()
    reference = load_record(spark, trial, "baseline")
    fills = verify_restart(reference, state)
    save_record(spark, trial, "recovered", {"state": state, "fills": fills, "batch": batch_id})
