from datetime import datetime, timedelta, timezone

import pandas as pd

from decision.brokers import PaperBroker
from decision.engine import Engine
from decision.replay import parity, replay
from decision.types import Assessment, RiskConfig
from serving import contract_v1 as c1

T0 = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def bars(closes, highs=None, lows=None, start=T0):
    n = len(closes)
    return pd.DataFrame({"bar_start": [pd.Timestamp(start + timedelta(minutes=i)) for i in range(n)],
                         "close": closes, "high": highs or closes, "low": lows or closes,
                         "volume_usd": [1000.0] * n, "n_trades": [10] * n, "n_buys": [5] * n, "phase": ["postgrad"] * n})


class FakeContract:
    """Scores from a dict {(token, minute_index): score}; candidates only at those times. Real label logic."""
    version, cutoff = "fake", 0.10

    def __init__(self, scores):
        self.scores = scores

    def assess(self, token_id, bars, as_of, *, launch_time, launch_venue, observed_from_launch):
        k = (token_id, int((as_of - T0).total_seconds() // 60))
        if k not in self.scores:
            return Assessment(token_id, as_of, False, "off_grid")
        last = bars.sort_values("bar_start").iloc[-1]
        return Assessment(token_id, as_of, True, "candidate", self.scores[k], float(last["close"]))

    def resolve(self, bars, entry_time, entry_price, *, launch_venue, observed_until):
        return c1.resolve_outcome(bars, {"assessment_time": entry_time, "price_at_t": entry_price},
                                  launch_venue=launch_venue, observed_until=observed_until)

    def round_trip_cost(self, launch_venue):
        return c1.round_trip_cost_v1(launch_venue, True)


def run(tokens, scores, risk=None):
    eng = Engine(FakeContract(scores), PaperBroker(), risk or RiskConfig.unlimited())
    tr = replay(eng, {k: {"bars": v, "launch_time": T0 - timedelta(hours=2), "launch_venue": "pumpdotfun"}
                      for k, v in tokens.items()})
    return eng, tr


def test_enters_above_cutoff_and_exits_at_2x():
    b = bars([1.0, 1.0, 1.2, 2.5, 2.5], highs=[1.0, 1.0, 1.2, 2.5, 2.5])
    eng, tr = run({"A": b}, {("A", 2): 0.2})       # assessed at T0+2m: last closed bar = index 1, price 1.0
    assert len(tr) == 1 and tr.iloc[0]["exit_reason"] == "up" and abs(tr.iloc[0]["r_gross"] - 1.0) < 1e-12
    assert tr.iloc[0]["r_net"] < tr.iloc[0]["r_gross"]


def test_below_cutoff_no_trade_but_logged():
    eng, tr = run({"A": bars([1.0] * 5)}, {("A", 2): 0.05})
    assert tr.empty and (eng.log.frame()["reason"] == "below_cutoff").any()


def test_one_position_per_token():
    b = bars([1.0] * 10)
    eng, tr = run({"A": b}, {("A", 2): 0.3, ("A", 3): 0.3})
    assert (eng.log.frame()["event"] == "enter").sum() == 1


def test_stop_wins_tie_like_the_label():
    b = bars([1.0, 1.0, 1.0, 1.0], highs=[1.0, 1.0, 1.0, 2.5], lows=[1.0, 1.0, 1.0, 0.4])
    eng, tr = run({"A": b}, {("A", 2): 0.3})
    assert tr.iloc[0]["exit_reason"] == "down"


def test_risk_gate_max_open_positions():
    tokens = {k: bars([1.0] * 5) for k in ("A", "B", "C")}
    scores = {(k, 2): 0.3 for k in tokens}
    eng, _ = run(tokens, scores, RiskConfig(max_open_positions=2))
    f = eng.log.frame()
    assert (f["event"] == "enter").sum() == 2 and (f["reason"] == "max_open_positions").sum() == 1


def test_pause_blocks_entries_and_kill_closes():
    eng = Engine(FakeContract({("A", 2): 0.3}), PaperBroker(), RiskConfig.unlimited())
    eng.pause(T0, "test")
    eng.on_bar_close("A", bars([1.0] * 3), T0 + timedelta(minutes=2), launch_time=T0, launch_venue="pumpdotfun")
    assert not eng.open and (eng.log.frame()["reason"] == "paused").any()
    eng.resume(T0)
    eng.on_bar_close("A", bars([1.0] * 3), T0 + timedelta(minutes=2), launch_time=T0, launch_venue="pumpdotfun")
    assert "A" in eng.open
    eng.kill(T0 + timedelta(minutes=3), {"A": 1.5}, "test")
    assert not eng.open and abs(eng.closed[0].r_gross - 0.5) < 1e-12


def test_parity_pass_and_fail():
    eng, tr = run({"A": bars([1.0, 1.0, 1.2, 2.5, 2.5])}, {("A", 2): 0.2})
    bt = pd.DataFrame({"token_id": ["A"], "assessment_time": [T0 + timedelta(minutes=2)], "r_up_360m": [1.0]})
    assert parity(tr, bt)["status"] == "PASS"
    bt2 = pd.concat([bt, pd.DataFrame({"token_id": ["B"], "assessment_time": [T0], "r_up_360m": [0.1]})])
    assert parity(tr, bt2)["status"] == "FAIL"


def test_replay_ticks_every_minute_not_only_on_bar_closes():
    # bars at minutes 0,1,2 then a gap; the candidate time is minute 4 (no bar closes then)
    b = bars([1.0, 1.0, 1.0])
    eng, tr = run({"A": b}, {("A", 4): 0.3})
    assert (eng.log.frame()["event"] == "enter").sum() == 1
