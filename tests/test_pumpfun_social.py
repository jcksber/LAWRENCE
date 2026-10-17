"""Unit tests for src/ingestion/pumpfun_social.py (no network)."""
from datetime import datetime, timezone

from ingestion.pumpfun_social import (
    CALLOUT_LIMIT,
    SocialUnavailable,
    fetch_coin_social,
    get_json,
    ipfs_url,
    to_callout_row,
    to_coin_row,
)

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _Session:
    """Routes by URL substring; each route is a list of responses."""

    def __init__(self, routes):
        self.routes, self.calls = {k: list(v) for k, v in routes.items()}, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        for key, resps in self.routes.items():
            if key in url:
                return resps.pop(0) if len(resps) > 1 else resps[0]
        return _Resp(404, {})


def test_get_json_404_is_none_and_401_raises():
    assert get_json(_Session({"x": [_Resp(404)]}), "https://a/x", sleep=lambda s: None) is None
    try:
        get_json(_Session({"x": [_Resp(401, "no")]}), "https://a/x", sleep=lambda s: None)
        raise AssertionError("did not raise")
    except SocialUnavailable as exc:
        assert exc.status == 401


def test_get_json_retries_429():
    slept = []
    sess = _Session({"x": [_Resp(429), _Resp(200, {"ok": 1})]})
    assert get_json(sess, "https://a/x", sleep=slept.append) == {"ok": 1} and slept == [1]


def test_ipfs_url_shapes():
    assert ipfs_url("ipfs://Qm1") == ["https://pump.mypinata.cloud/ipfs/Qm1", "https://ipfs.io/ipfs/Qm1"]
    assert ipfs_url("https://pump.mypinata.cloud/ipfs/Qm2")[1] == "https://ipfs.io/ipfs/Qm2"
    assert ipfs_url("https://example.com/m.json") == ["https://example.com/m.json"]
    assert ipfs_url(None) == []


def test_coin_row_uses_launch_metadata_for_socials():
    coin = {"mint": "M", "twitter": "https://x.com/later", "created_timestamp": 1760717833062,
            "description": "edited later"}
    meta = {"twitter": "https://x.com/launch", "telegram": "", "description": "gm"}
    r = to_coin_row(coin, meta, "M", NOW)
    assert r["launch_twitter"] == "https://x.com/launch" and r["launch_telegram"] is None
    assert r["current_twitter"] == "https://x.com/later" and r["description_len"] == 2
    assert r["created_at"].year == 2025 and r["has_launch_metadata"] is True


def test_coin_row_without_metadata():
    r = to_coin_row({"mint": "M"}, None, "M", NOW)
    assert r["has_launch_metadata"] is False and r["launch_twitter"] is None


def test_callout_row_fields():
    c = {"calloutId": "abc", "coinMint": "M", "userId": "U", "createdAt": 1785778579888,
         "marketCap": 2085, "thesis": "Callout", "likes": 3, "maxMultiplier": 52.5, "mediaUrl": None}
    r = to_callout_row(c, "M", NOW)
    assert r["callout_id"] == "abc" and r["created_at"].year == 2026
    assert r["market_cap_at_call"] == 2085.0 and r["thesis_len"] == 7 and r["likes"] == 3
    assert r["posthoc_max_multiplier"] == 52.5 and r["has_media"] is False


def test_fetch_coin_social_flags_truncation():
    full = {"callouts": [{"calloutId": i} for i in range(CALLOUT_LIMIT)]}
    sess = _Session({
        "/coins-v2/": [_Resp(200, {"mint": "M", "metadata_uri": "ipfs://Qm"})],
        "mypinata": [_Resp(200, {"twitter": "t"})],
        "/callout/coin/": [_Resp(200, full)],
    })
    out = fetch_coin_social("M", sess, sleep=lambda s: None)
    assert out["truncated"] is True and out["meta"] == {"twitter": "t"} and len(out["callouts"]) == CALLOUT_LIMIT


def test_fetch_coin_social_missing_coin_and_gateway_fallback():
    sess = _Session({
        "/coins-v2/": [_Resp(200, {"mint": "M", "metadata_uri": "ipfs://Qm"})],
        "mypinata": [_Resp(504)] * 3,
        "ipfs.io": [_Resp(200, {"website": "w"})],
        "/callout/coin/": [_Resp(200, {"callouts": []})],
    })
    out = fetch_coin_social("M", sess, sleep=lambda s: None)
    assert out["meta"] == {"website": "w"} and out["truncated"] is False
    gone = fetch_coin_social("Z", _Session({}), sleep=lambda s: None)
    assert gone["coin"] is None and gone["callouts"] == []
