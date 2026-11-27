import json

import pandas as pd
import pytest

from decision.e3 import E3PaperEngine
from decision.e3_restart import check_research, verify_restart
from decision.replay import replay
from decision.types import Assessment, RiskConfig

T0 = pd.Timestamp("2026-10-04", tz="UTC")


class EntryContract:
    version, cutoff = "test-v1", 0.1

    def assess(self, token, bars, now, **context):
        score = 0.3 if now == T0 + pd.Timedelta(minutes=1) else None
        return Assessment(token, now, score is not None, "candidate", score, 1.0, {"missing": float("nan")})

    def round_trip_cost(self, venue):
        return 0.025


def series(prices):
    bars = pd.DataFrame(prices, columns=["high", "low", "close"])
    bars["bar_start"] = pd.date_range(T0, periods=len(bars), freq="min")
    bars["phase"] = "postgrad"
    return {"bars": bars, "launch_venue": "pumpdotfun", "launch_time": T0}


def test_restart_between_two_targets_on_one_bar_in_token_order():
    # B sells at 2x and 4x in the same bar. A is already processed; C still needs its tick.
    tokens = {"A": series([(1, 1, 1), (1, .6, .7)]),
              "B": series([(1, 1, 1), (4.2, 1.1, 4.1), (11, 1.1, 10)]),
              "C": series([(1, 1, 1), (2.1, .9, 2), (.65, .5, .6)])}
    contract = EntryContract()
    reference = E3PaperEngine(contract, RiskConfig.unlimited())
    replay(reference, tokens)
    check_research(reference, tokens)
    saved = {}

    class InjectedCrash(Exception):
        pass

    def interrupt(engine, p, now):
        saved.update(tick=now.isoformat(), token=p.token_id, state=engine.snapshot())
        raise InjectedCrash()

    engine = E3PaperEngine(contract, RiskConfig.unlimited(), on_first_rung=interrupt)
    with pytest.raises(InjectedCrash):
        replay(engine, tokens)
    saved = json.loads(json.dumps(saved, allow_nan=False))
    assert saved["token"] == "B"
    resumed = E3PaperEngine.restore(contract, saved["state"])
    assert resumed.open["B"].remaining == 0.5
    replay(resumed, tokens, resume_at=(pd.Timestamp(saved["tick"]), saved["token"]))
    assert verify_restart(reference.snapshot(), resumed.snapshot()) == 11
    sell_reasons = [e["reason"] for e in resumed.log.rows if e["token_id"] == "B" and e["event"] != "enter"]
    assert sell_reasons == ["2x", "4x", "6x", "8x", "10x"]


def test_stop_priority_and_quiet_minutes():
    contract = EntryContract()
    tokens = {"A": series([(1, 1, 1), (4, .7, 2)]),
              "B": series([(1, 1, 1), (2, .9, 1.7)])}
    engine = E3PaperEngine(contract, RiskConfig.unlimited())
    replay(engine, tokens)
    check_research(engine, tokens)
    assert engine.closed[0].exit_reason == "stop"
    assert engine.closed[0].r_gross == pytest.approx(-.2)
    assert engine.open["B"].remaining == .5  # still held after the replay's six-hour quiet tail
    engine.kill(T0 + pd.Timedelta(days=1), {"B": 1.2}, "test")
    assert not engine.open and engine.closed[-1].r_gross == pytest.approx(.6)


def test_redelivered_batch_uses_durable_sale_and_fresh_session(monkeypatch):
    from types import SimpleNamespace

    from decision import e3_restart as drill

    contract = EntryContract()
    tokens = {"B": series([(1, 1, 1), (4.1, 1, 4), (.8, .7, .75)])}
    ref = E3PaperEngine(contract, RiskConfig.unlimited())
    replay(ref, tokens)
    records = {"baseline": ref.snapshot()}

    def save(_spark, _trial, phase, record):
        # Model an atomic insert and a retry of a previously committed transaction.
        copy = json.loads(json.dumps(record, allow_nan=False))
        assert records.setdefault(phase, copy) == copy

    batch = SimpleNamespace(sparkSession=None)
    batch.select = lambda column: SimpleNamespace(take=lambda n: [SimpleNamespace(content=b"fixture")])
    monkeypatch.setattr(drill, "inputs", lambda *args: (contract, tokens))
    monkeypatch.setattr(drill, "save_record", save)
    monkeypatch.setattr(drill, "load_record", lambda spark, trial, phase: records.get(phase))
    args = dict(fixture="fixture", contract_dir="v1", trial="test")
    with pytest.raises(RuntimeError, match=drill.INJECTED):
        drill.process_batch(batch, 0, mode="crash", session_id="before", **args)
    saved = records["half_sale"]["state"]
    assert saved["audit"][-1]["reason"] == "2x" and saved["open"][0]["remaining"] == 0.5
    with pytest.raises(RuntimeError, match="previous run"):
        drill.process_batch(batch, 1, mode="recover", session_id="after", **args)
    with pytest.raises(RuntimeError, match="previous run"):
        drill.process_batch(batch, 0, mode="recover", session_id="before", **args)
    for session in ("after", "retry"):
        drill.process_batch(batch, 0, mode="recover", session_id=session, **args)
    assert records["recovered"]["state"] == records["baseline"]
