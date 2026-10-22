from datetime import datetime, timezone

import pytest

from ingestion.postcrash_bars import postcrash_hourly_bars_sql

UTC = timezone.utc
CUT = datetime(2026, 10, 11, tzinfo=UTC)


def test_holdout_cap_in_sql():
    sql = postcrash_hourly_bars_sql(datetime(2026, 9, 20, tzinfo=UTC), datetime(2026, 9, 21, tzinfo=UTC), CUT)
    assert "block_time < TIMESTAMP '2026-10-11 00:00:00'" in sql
    assert "date_trunc('hour', block_time)" in sql


def test_refuses_window_past_cutoff():
    with pytest.raises(ValueError):
        postcrash_hourly_bars_sql(datetime(2026, 10, 10, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC), CUT)


def test_age_window():
    sql = postcrash_hourly_bars_sql(datetime(2026, 9, 20, tzinfo=UTC), datetime(2026, 9, 21, tzinfo=UTC), CUT)
    assert "p.created_at + INTERVAL '24' HOUR" in sql and "p.created_at + INTERVAL '14' DAY" in sql
