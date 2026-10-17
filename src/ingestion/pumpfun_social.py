"""
pump.fun social backfill -- coin metadata + callouts (2026-10-17, see
docs/decisions.md "pump.fun comments removed"). Replaces the replies client:
pump.fun dropped per-coin comment threads (`/replies/{mint}` -> "Cannot GET").

Two public, unauthenticated endpoints on the UNOFFICIAL frontend API:
- `GET /coins-v2/{mint}`: coin record. Its `metadata_uri` points at the
  launch-time IPFS JSON (immutable), so socials read from THAT are known at
  launch. The coin record's own `twitter` may be editable later -- stored as
  `current_twitter`, not used as a point-in-time feature.
- `GET /callout/coin/{mint}?limit=100`: users "calling" the coin, each with
  `createdAt` (point-in-time), market cap at the call, likes/replies AS OF
  THE PULL. Capped at 100 and RANKED BY POST-HOC PERFORMANCE
  (`maxMultiplier`), so a coin that returns exactly CALLOUT_LIMIT callouts is
  truncated toward its winners -- look-ahead bias. Those coins are flagged
  `callouts_truncated` and must not get callout features.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests

BASE_URL = "https://frontend-api-v3.pump.fun"
IPFS_GATEWAYS = ("https://pump.mypinata.cloud/ipfs/", "https://ipfs.io/ipfs/")  # pinata = what pump.fun's page uses
IPFS_RETRIES = 2  # per gateway; a slow gateway shouldn't cost ~30s per token
CALLOUT_LIMIT = 100  # API max
MAX_RETRIES = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Origin": "https://pump.fun",
    "Referer": "https://pump.fun/",
}


class SocialUnavailable(Exception):
    """Non-retryable, non-404 response (401/403 etc.)."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


def _ms(v: Any) -> datetime | None:
    """Epoch ms / s, or ISO string -> aware UTC."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=timezone.utc)
    s = str(v).strip()
    if s.isdigit():
        return _ms(int(s))
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _num(v: Any, cast=float):
    try:
        return None if v is None else cast(v)
    except (TypeError, ValueError):
        return None


def get_json(
    session: Any, url: str, params: dict | None = None, *, sleep: Callable[[float], None] = time.sleep,
    retries: int = MAX_RETRIES,
) -> Any | None:
    """JSON body, or None on 404. Retries 429/5xx/network errors with backoff."""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException:
            sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            ra = resp.headers.get("Retry-After")
            sleep(float(ra) if ra and ra.isdigit() else 2 ** attempt)
            continue
        raise SocialUnavailable(resp.status_code, resp.text)
    raise SocialUnavailable(-1, f"gave up after {retries} retries: {url}")


def ipfs_url(metadata_uri: str | None) -> list[str]:
    """Gateway URLs to try for a metadata_uri (ipfs://CID, a gateway URL, or
    a bare CID). Non-IPFS http(s) URIs are returned as-is."""
    if not metadata_uri:
        return []
    u = metadata_uri.strip()
    if u.startswith("ipfs://"):
        cid = u[len("ipfs://"):]
    elif "/ipfs/" in u:
        cid = u.split("/ipfs/", 1)[1]
    elif u.startswith("http"):
        return [u]
    else:
        cid = u
    return [g + cid for g in IPFS_GATEWAYS]


def fetch_launch_metadata(session: Any, metadata_uri: str | None, *, sleep=time.sleep) -> dict | None:
    for url in ipfs_url(metadata_uri):
        try:
            data = get_json(session, url, sleep=sleep, retries=IPFS_RETRIES)
        except (SocialUnavailable, ValueError):  # ValueError: non-JSON body
            continue
        if isinstance(data, dict):
            return data
    return None


def to_coin_row(coin: dict, meta: dict | None, mint: str, ingested_at: datetime) -> dict:
    meta = meta or {}
    desc = meta.get("description", coin.get("description"))
    return {
        "mint": coin.get("mint") or mint,
        "name": coin.get("name"),
        "symbol": coin.get("symbol"),
        "creator": coin.get("creator"),
        "created_at": _ms(coin.get("created_timestamp")),
        "program": coin.get("program"),
        "description_len": None if desc is None else len(desc),
        # launch-time socials (immutable IPFS JSON); None when metadata missing
        "launch_twitter": meta.get("twitter") or None,
        "launch_telegram": meta.get("telegram") or None,
        "launch_website": meta.get("website") or None,
        "has_launch_metadata": bool(meta),
        # as of the pull -- NOT point-in-time
        "current_twitter": coin.get("twitter") or None,
        "complete": coin.get("complete"),
        "nsfw": coin.get("nsfw"),
        "is_banned": coin.get("is_banned"),
        "hidden": coin.get("hidden"),
        "is_backfill": coin.get("is_backfill"),
        "source": "pumpfun_frontend_api_v3",
        "ingested_at": ingested_at,
    }


def to_callout_row(c: dict, mint: str, ingested_at: datetime) -> dict:
    thesis = c.get("thesis")
    return {
        "mint": c.get("coinMint") or mint,
        "callout_id": None if c.get("calloutId") is None else str(c.get("calloutId")),
        "user_id": c.get("userId"),
        "created_at": _ms(c.get("createdAt")),
        "market_cap_at_call": _num(c.get("marketCap")),
        "callout_price_usd": _num(c.get("calloutPriceUsd")),
        "thesis_len": None if thesis is None else len(thesis),
        "has_media": bool(c.get("mediaUrl")),
        "is_quote": bool(c.get("quotedCalloutId")),
        "chain": c.get("chain"),
        # engagement AS OF THE PULL -- lagged features at best
        "likes": _num(c.get("likes"), int),
        "reply_count": _num(c.get("replyCount"), int),
        "repost_count": _num(c.get("repostCount"), int),
        "quote_count": _num(c.get("quoteCount"), int),
        # FUTURE information relative to created_at -- never a feature
        "posthoc_max_multiplier": _num(c.get("maxMultiplier")),
        "source": "pumpfun_frontend_api_v3",
        "ingested_at": ingested_at,
    }


def fetch_coin_social(mint: str, session: Any = None, *, base_url: str = BASE_URL,
                      sleep: Callable[[float], None] = time.sleep) -> dict:
    """{'coin': dict|None, 'meta': dict|None, 'callouts': list, 'truncated': bool}."""
    session = session or requests.Session()
    coin = get_json(session, f"{base_url}/coins-v2/{mint}", sleep=sleep)
    meta = fetch_launch_metadata(session, coin.get("metadata_uri"), sleep=sleep) if coin else None
    payload = get_json(session, f"{base_url}/callout/coin/{mint}", {"limit": CALLOUT_LIMIT}, sleep=sleep)
    callouts = (payload or {}).get("callouts") or []
    return {"coin": coin, "meta": meta, "callouts": callouts, "truncated": len(callouts) >= CALLOUT_LIMIT}
