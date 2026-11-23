import numpy as np
import pandas as pd

from features.market_context import market_context

H0 = pd.Timestamp("2026-10-01 12:00", tz="UTC")


def ev(rows):
    return pd.DataFrame(rows, columns=["launch_time", "grad_time", "hit2x_6h"])


def test_counts_use_only_completed_hours_before_T():
    e = ev([(H0 - pd.Timedelta(minutes=30), pd.NaT, False),      # in the hour before H0
            (H0, pd.NaT, False),                                   # AT H0 -> not counted for H0
            (H0 - pd.Timedelta(hours=3), H0 - pd.Timedelta(minutes=10), True)])
    hourly = pd.DataFrame({"hour": [H0 - pd.Timedelta(hours=1), H0], "volume_usd": [100.0, 999.0], "n_trades": [5, 9]})
    out = market_context(e, hourly, pd.DatetimeIndex([H0])).iloc[0]
    assert out["mkt_launches_1h"] == 1 and out["mkt_launches_6h"] == 2
    assert out["mkt_grads_1h"] == 1
    assert out["mkt_postgrad_vol_1h"] == 100.0 and out["mkt_postgrad_trades_1h"] == 5  # not the H0 hour


def test_2x_share_only_counts_outcomes_known_before_T():
    e = ev([(H0 - pd.Timedelta(hours=20), H0 - pd.Timedelta(hours=10), True),   # known at H0-4h
            (H0 - pd.Timedelta(hours=20), H0 - pd.Timedelta(hours=8), False),   # known at H0-2h
            (H0 - pd.Timedelta(hours=5), H0 - pd.Timedelta(hours=2), True)])    # known at H0+4h -> excluded
    hourly = pd.DataFrame({"hour": [H0 - pd.Timedelta(hours=1)], "volume_usd": [1.0], "n_trades": [1]})
    out = market_context(e, hourly, pd.DatetimeIndex([H0])).iloc[0]
    assert out["mkt_recent_resolved_24h"] == 2 and np.isclose(out["mkt_recent_2x_share_24h"], 0.5)
