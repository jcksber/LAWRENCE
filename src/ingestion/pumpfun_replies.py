"""
pump.fun reply (comment) backfill -- social pipeline step 1 (2026-10-16,
see docs/decisions.md "Social sources").

pump.fun's own frontend API serves each token's comment thread with
timestamps. It is UNOFFICIAL and undocumented: the base URL, response shape
and auth can change without notice, so everything shape-dependent lives in
`parse_replies` / `to_reply_row`, and notebook 00i probes one token before
any bulk pull. Some API versions require a logged-in JWT; pass it as
`auth_token` (stored in the `lawrence` secret scope) if the probe gets 401.

Point-in-time: each reply's own `timestamp` is when it was posted, so
features at T count replies with timestamp <= T. Likes are as of the pull
(not T) and deleted/hidden replies are missing or flagged -- treat likes as
a lagged feature and keep `hidden` rows in Bronze.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests

BASE_URL = "https://frontend-api-v3.pump.fun"
PAGE_LIMIT = 1000
MAX_REPLIES_PER_TOKEN = 20_000  # crash-guard against bot-spammed threads
MAX_RETRIES = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Origin": "https://pump.fun",
    "Referer": "https://pump.fun/",
}


class RepliesUnavailable(Exception):
    """Non-retryable response (401/403/404 etc.) -- stop and look at it."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


def parse_replies(payload: Any) -> tuple[list[dict], bool | None]:
    """(replies, has_more) from either a bare list or a {"replies": [...]}
    object. has_more is None when the API doesn't say."""
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        for key in ("replies", "data", "items"):
            if isinstance(payload.get(key), list):
                more = payload.get("hasMore", payload.get("has_more"))
                return payload[key], None if more is None else bool(more)
    raise ValueError(f"unrecognised replies payload: {str(payload)[:300]}")


def _ts(v: Any) -> datetime | None:
    """Epoch seconds or milliseconds, or an ISO string -> aware UTC."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        secs = v / 1000 if v > 1e11 else v
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    s = str(v).strip()
    if s.isdigit():
        return _ts(int(s))
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _int(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def to_reply_row(r: dict, mint: str, ingested_at: datetime) -> dict:
    """One lawrence.bronze.pumpfun_replies row. Unknown fields are dropped;
    missing ones are NULL."""
    text = r.get("text")
    return {
        "mint": r.get("mint") or mint,
        "reply_id": None if r.get("id") is None else str(r.get("id")),
        "user_address": r.get("user") or r.get("user_address"),
        "username": r.get("username"),
        "posted_at": _ts(r.get("timestamp", r.get("created_timestamp"))),
        "text": text,
        "text_len": None if text is None else len(text),
        "has_image": bool(r.get("file_uri") or r.get("image_uri")),
        "total_likes": _int(r.get("total_likes", r.get("likes"))),
        "is_hidden": bool(r.get("hidden", False)),
        "is_buy": r.get("is_buy"),
        "sol_amount": _int(r.get("sol_amount")),
        "source": "pumpfun_frontend_api",
        "ingested_at": ingested_at,
    }


def _get(session: Any, url: str, params: dict, headers: dict, sleep: Callable[[float], None]) -> Any:
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException:
            sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt)
            continue
        raise RepliesUnavailable(resp.status_code, resp.text)
    raise RepliesUnavailable(-1, f"gave up after {MAX_RETRIES} retries: {url}")


def fetch_replies(
    mint: str,
    session: Any = None,
    *,
    base_url: str = BASE_URL,
    auth_token: str | None = None,
    limit: int = PAGE_LIMIT,
    max_replies: int = MAX_REPLIES_PER_TOKEN,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Every reply for one mint, paging by offset until a short page,
    has_more=False, an empty page, or `max_replies`."""
    session = session or requests.Session()
    headers = dict(HEADERS)
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    out: list[dict] = []
    offset = 0
    while len(out) < max_replies:
        payload = _get(
            session, f"{base_url}/replies/{mint}",
            {"limit": limit, "offset": offset, "reverse": "false"}, headers, sleep,
        )
        page, has_more = parse_replies(payload)
        if not page:
            break
        out.extend(page)
        offset += len(page)
        if has_more is False or (has_more is None and len(page) < limit):
            break
    return out[:max_replies]
