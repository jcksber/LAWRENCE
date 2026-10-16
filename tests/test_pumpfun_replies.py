"""Unit tests for src/ingestion/pumpfun_replies.py (no network)."""
from datetime import datetime, timezone

import pytest

from ingestion.pumpfun_replies import (
    RepliesUnavailable,
    fetch_replies,
    parse_replies,
    to_reply_row,
)

NOW = datetime(2026, 10, 16, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(params)
        return self.responses.pop(0)


def test_parse_list_and_object_shapes():
    assert parse_replies([{"id": 1}]) == ([{"id": 1}], None)
    assert parse_replies({"replies": [{"id": 1}], "hasMore": False}) == ([{"id": 1}], False)
    with pytest.raises(ValueError):
        parse_replies({"unexpected": 1})


def test_row_parses_ms_and_s_timestamps():
    ms = to_reply_row({"id": 7, "timestamp": 1760572800000, "text": "gm"}, "MINT", NOW)
    s = to_reply_row({"id": 7, "timestamp": 1760572800, "text": "gm"}, "MINT", NOW)
    assert ms["posted_at"] == s["posted_at"] == datetime(2025, 10, 16, tzinfo=timezone.utc)
    assert ms["mint"] == "MINT" and ms["reply_id"] == "7" and ms["text_len"] == 2


def test_row_missing_fields_are_null():
    r = to_reply_row({}, "MINT", NOW)
    assert r["posted_at"] is None and r["total_likes"] is None and r["is_hidden"] is False


def test_fetch_pages_until_short_page():
    sess = _Session([_Resp(200, [{"id": i} for i in range(2)]), _Resp(200, [{"id": 9}])])
    out = fetch_replies("MINT", sess, limit=2, sleep=lambda s: None)
    assert [r["id"] for r in out] == [0, 1, 9]
    assert [c["offset"] for c in sess.calls] == [0, 2]


def test_fetch_respects_has_more_false():
    sess = _Session([_Resp(200, {"replies": [{"id": 1}, {"id": 2}], "hasMore": False})])
    assert len(fetch_replies("MINT", sess, limit=2, sleep=lambda s: None)) == 2
    assert len(sess.calls) == 1


def test_fetch_retries_429_then_succeeds():
    slept = []
    sess = _Session([_Resp(429), _Resp(200, [])])
    assert fetch_replies("MINT", sess, sleep=slept.append) == []
    assert slept == [1]


def test_fetch_raises_on_auth_error():
    sess = _Session([_Resp(401, {"message": "unauthorized"})])
    try:
        fetch_replies("MINT", sess, sleep=lambda s: None)
        raise AssertionError("did not raise")
    except RepliesUnavailable as exc:
        assert exc.status == 401


def test_fetch_caps_max_replies():
    sess = _Session([_Resp(200, [{"id": i} for i in range(3)])] * 5)
    assert len(fetch_replies("MINT", sess, limit=3, max_replies=5, sleep=lambda s: None)) == 5
