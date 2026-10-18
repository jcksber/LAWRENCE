from datetime import datetime, timezone

from ingestion.creator_flow import (
    creator_launches_sql,
    creator_trades_sql,
    early_buyers_sql,
    summarize,
)

D0, D1, G = (datetime(2026, 9, 20, tzinfo=timezone.utc), datetime(2026, 9, 21, tzinfo=timezone.utc),
             datetime(2026, 10, 11, tzinfo=timezone.utc))


def test_window_bounds_never_past_grads_until():
    for sql in (early_buyers_sql(D0, D1, G), creator_trades_sql(D0, D1, G)):
        assert "2026-10-11" in sql and "2026-10-12" not in sql
        assert " OR " not in sql.upper().replace("ORDER", "")  # equi-joins only
    assert "2026-09-20" in creator_launches_sql(D0, D1)


def test_creator_trades_capped_at_48h_and_split_joins():
    sql = creator_trades_sql(D0, D1, G)
    assert sql.count("INTERVAL '48' HOUR") == 3 and "UNION ALL" in sql


def test_summarize_wraps():
    assert summarize("SELECT 1").startswith("WITH q AS (SELECT 1)")


def test_no_trade_read_at_or_after_cutoff():
    for sql in (early_buyers_sql(D0, D1, G), creator_trades_sql(D0, D1, G)):
        assert "block_time < TIMESTAMP '2026-10-11 00:00:00'" in sql
