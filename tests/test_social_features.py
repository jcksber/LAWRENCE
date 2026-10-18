"""Social features v2 candidates: point-in-time rules."""
import numpy as np
import pandas as pd

from features.social_features import (
    COUNT_CAP,
    build_host_groups,
    callout_features,
    host_group,
    meta_features,
    truncation_check,
)

L = pd.Timestamp("2026-10-05 12:00", tz="UTC")


def _arr(mins):
    return np.array([(L + pd.Timedelta(minutes=m)).value for m in mins], dtype="int64")


def test_only_callouts_at_or_before_t():
    ca = _arr([5, 10, 70])
    f = callout_features(ca, np.array(["a", "b", "a"]), np.array([1e4, 2e4, 9e4]), L + pd.Timedelta(minutes=10), L)
    assert f["co_n"] == 2 and f["co_callers"] == 2 and f["co_median_mcap_at_call"] == 1.5e4
    assert f["co_first_min_after_launch"] == 5 and f["co_min_since_first"] == 5


def test_windows_and_cap():
    ca = _arr(range(0, 50))
    f = callout_features(ca, np.array([f"u{i}" for i in range(50)]), np.full(50, 1.0), L + pd.Timedelta(minutes=49), L)
    assert f["co_n"] == COUNT_CAP and f["co_callers"] == COUNT_CAP and f["co_n_15m"] == 15


def test_no_callouts_yet_is_zero_count_nan_rest():
    f = callout_features(_arr([30]), np.array(["a"]), np.array([1.0]), L + pd.Timedelta(minutes=10), L)
    assert f["co_n"] == 0 and np.isnan(f["co_min_since_first"]) and np.isnan(f["co_median_mcap_at_call"])


def test_host_groups_use_training_period_only():
    hosts = pd.Series(["ipfs.io"] * 3 + ["new.tool.xyz"] * 150 + ["gateway.pinata.cloud"] + ["meta.uxento.io"])
    created = pd.Series([L] * 3 + [L + pd.Timedelta(days=30)] * 150 + [L, L])
    g = build_host_groups(hosts, created, L + pd.Timedelta(days=1))
    assert g["ipfs.io"] == "ipfs_standard" and g["meta.uxento.io"] == "meta.uxento.io"
    assert "new.tool.xyz" not in g and host_group("new.tool.xyz", g) == "other_custom"
    assert host_group("gateway.pinata.cloud", g) == "other_pinata" and host_group(None, g) is None


def test_meta_unknown_when_unresolved():
    m = meta_features("ipfs.io", {"ipfs.io": "ipfs_standard"}, None, None, None, has_launch_metadata=False)
    assert np.isnan(m["meta_launch_has_twitter"]) and m["meta_is_launch_tool"] == 0.0
    m = meta_features("meta.uxento.io", {}, "https://x.com/a", "", None, has_launch_metadata=True)
    assert m["meta_launch_has_twitter"] == 1.0 and m["meta_launch_has_telegram"] == 0.0 and m["meta_is_launch_tool"] == 1.0


def test_truncation_check_detects_early_keep():
    rows = []
    for c in range(3):
        for i in range(60):  # earlier callouts did better -> top half = first half
            rows.append({"mint": f"m{c}", "callout_id": f"{c}-{i}", "created_at": L + pd.Timedelta(minutes=i),
                         "posthoc_max_multiplier": 100.0 - i})
    r = truncation_check(pd.DataFrame(rows))
    assert len(r) == 3 and (r["first_k_kept_share"] == 1.0).all()
