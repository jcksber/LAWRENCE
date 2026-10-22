import numpy as np
import pandas as pd

from features.postcrash_features import pool_rows

G = pd.Timestamp("2026-09-01", tz="UTC")
END = pd.Timestamp("2026-10-11", tz="UTC")


def mk(hours, close, high=None, low=None, vol=5000.0, trades=30, traders=10):
    n = len(hours)
    return pd.DataFrame({"bar_start": [G + pd.Timedelta(hours=h) for h in hours], "close": close,
                         "high": high if high is not None else close, "low": low if low is not None else close,
                         "volume_usd": [vol] * n, "n_trades": [trades] * n, "n_traders": [traders] * n})


def test_rows_only_from_24h_and_time_out_at_last_close():
    hours = list(range(10, 30))
    r = pool_rows(mk(hours, [1.0] * len(hours)), G, np.nan, np.nan, END)
    assert (r["age_h"] >= 24).all() and (r["outcome"] == "timeout").all()


def test_censored_when_horizon_passes_data_end():
    hours = list(range(24, 40))
    r = pool_rows(mk(hours, [1.0] * 16), G, np.nan, np.nan, G + pd.Timedelta(hours=60))
    assert r.empty  # T >= 25h, T + 72h > 60h


def test_drawdown_and_rebound_from_ath():
    hours = list(range(24, 60))
    close = [10.0] + [2.0] * 10 + [3.0] * 25
    r = pool_rows(mk(hours, close), G, np.nan, np.nan, END)
    row = r[r["assessment_time"] == G + pd.Timedelta(hours=36)].iloc[0]  # bar 35 closes at 36h, price 3
    assert np.isclose(row["dd_from_ath"], 3 / 10 - 1) and np.isclose(row["rebound_from_low"], 3 / 2 - 1)


def test_first_24h_ath_used():
    hours = list(range(24, 40))
    r = pool_rows(mk(hours, [1.0] * 16), G, 20.0, 0.5, END)
    assert np.isclose(r.iloc[0]["dd_from_ath"], 1 / 20 - 1) and r.iloc[0]["ath_in_first_24h"] == 1.0


def test_activity_filter():
    hours = list(range(24, 40))
    r = pool_rows(mk(hours, [1.0] * 16, trades=0), G, np.nan, np.nan, END)
    assert r.empty


def test_barrier_up_and_tie_is_down():
    hours = list(range(24, 30))
    close = [1.0, 1.0, 1.5, 2.2, 1.0, 1.0]
    r = pool_rows(mk(hours, close, high=[1, 1, 1.5, 2.2, 1, 1], low=[1, 1, 1.4, 2.0, 1, 1]), G, np.nan, np.nan, END)
    assert r.iloc[0]["outcome"] == "up" and np.isclose(r.iloc[0]["r_pc"], 1.0)
    r2 = pool_rows(mk(hours, close, high=[1, 2.5, 1, 1, 1, 1], low=[1, 0.4, 1, 1, 1, 1]), G, np.nan, np.nan, END)
    assert r2.iloc[0]["outcome"] == "down"


def test_delayed_entry_label():
    hours = list(range(24, 30))
    close = [1.0, 1.5, 1.6, 3.5, 1.0, 1.0]
    r = pool_rows(mk(hours, close), G, np.nan, np.nan, END)
    first = r.iloc[0]
    assert first["outcome"] == "up" and np.isclose(first["r_pc_next"], 1.0)  # from 1.5, 3.5 >= 3.0
    second = r.iloc[1]  # entry 1.5 -> next 1.6, 3.5 >= 3.2 up
    assert np.isclose(second["r_pc_next"], 1.0)
