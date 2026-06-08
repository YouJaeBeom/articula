"""
HTTP fetch strategies for the articula library.

Tier 1 — ``fetch_static``:  plain GET with minimal headers.
Tier 2 — ``fetch_rotation``: rotated User-Agent + full browser-like header set.

The browser tier lives in ``_browser.py`` to isolate the optional Playwright
dependency.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/apng,*/*;q=0.8"
)
_ACCEPT_LANGUAGE = "en-US,en;q=0.9,ko;q=0.8"

# Status codes that permanently fail — no retry, no escalation.
_HARD_FAIL_CODES: frozenset[int] = frozenset({401, 404, 410})

# Status codes that indicate bot-detection / temporary block — escalate.
_BOT_CODES: frozenset[int] = frozenset({403, 429, 503, 520, 521, 522, 523, 524})

# Status codes to retry with back-off before escalating.
_RETRY_CODES: frozenset[int] = frozenset({429, 503, 504})

_USER_AGENTS: list[str] = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchResult:
    """Immutable result from a successful HTTP fetch."""

    html: str
    resolved_url: str
    strategy_tier: str
    status_code: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_bot_challenge(html: str) -> bool:
    """Return True if the page looks like a bot-protection challenge."""
    lower = html.lower()
    signals = (
        "challenge-platform",
        "cf-browser-verification",
        "just a moment",
        "checking your browser",
        "enable javascript and cookies",
        "ddos-guard",
        "ray id",
    )
    return any(s in lower for s in signals)


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Compute back-off delay honouring Retry-After when present."""
    if retry_after and retry_after.isdigit():
        return float(retry_after)
    return float(2 ** attempt) + random.uniform(0.0, 1.0)


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    max_retries: int,
) -> httpx.Response:
    """GET with exponential back-off on transient errors, skip retry for 404/401."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            if resp.status_code in _HARD_FAIL_CODES:
                return resp  # permanent failure, return immediately
            if resp.status_code in _RETRY_CODES:
                delay = _retry_delay(attempt, resp.headers.get("retry-after"))
                logger.debug("HTTP %s — retrying after %.1fs", resp.status_code, delay)
                await asyncio.sleep(delay)
                continue
            return resp
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            delay = _retry_delay(attempt, None)
            logger.debug("Network error %s — retrying after %.1fs", exc, delay)
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise httpx.RequestError(f"Failed after {max_retries} retries for {url}")


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------


async def fetch_static(
    url: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
    custom_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
) -> FetchResult | None:
    """Tier 1 fetch: plain GET with a single, fixed User-Agent.

    Returns ``None`` when the page cannot be used (error, bot-challenge, etc.).
    """
    headers: dict[str, str] = {
        "User-Agent": _USER_AGENTS[0],
        "Accept": _ACCEPT,
        "Accept-Language": _ACCEPT_LANGUAGE,
        **(custom_headers or {}),
    }
    transport = httpx.AsyncHTTPTransport(proxy=proxy_url) if proxy_url else None
    async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
        try:
            resp = await _get_with_retry(client, url, headers, max_retries)
        except Exception as exc:
            logger.debug("Static fetch error: %s", exc)
            return None

        if resp.status_code in _HARD_FAIL_CODES | _BOT_CODES:
            logger.debug("Static fetch: non-usable status %s", resp.status_code)
            return None
        if resp.status_code != 200:
            logger.debug("Static fetch: unexpected status %s", resp.status_code)
            return None

        html = resp.text
        if _is_bot_challenge(html):
            logger.debug("Static fetch: bot challenge detected in HTML")
            return None

        return FetchResult(
            html=html,
            resolved_url=str(resp.url),
            strategy_tier="static",
            status_code=resp.status_code,
        )


async def fetch_rotation(
    url: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,  # noqa: ARG001 — kept for API parity with fetch_static
    custom_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
) -> FetchResult | None:
    """Tier 2 fetch: iterates all UA rotation variants in order.

    Tries each User-Agent in ``_USER_AGENTS`` sequentially with a full
    browser-like header set.  Returns the first usable 200-OK response, or
    ``None`` when all variants are exhausted.

    Hard HTTP failures (404/401/410) short-circuit the loop immediately; bot
    codes and network errors advance to the next UA variant.
    """
    transport = httpx.AsyncHTTPTransport(proxy=proxy_url) if proxy_url else None
    async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
        for ua in _USER_AGENTS:
            headers: dict[str, str] = {
                "User-Agent": ua,
                "Accept": _ACCEPT,
                "Accept-Language": _ACCEPT_LANGUAGE,
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "max-age=0",
                "Sec-Ch-Ua": (
                    '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"'
                ),
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                **(custom_headers or {}),
            }
            try:
                resp = await client.get(url, headers=headers, follow_redirects=True)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                logger.debug(
                    "Rotation fetch: network error with UA variant — trying next: %s",
                    exc,
                )
                continue

            if resp.status_code in _HARD_FAIL_CODES:
                logger.debug(
                    "Rotation fetch: permanent fail status %s — aborting",
                    resp.status_code,
                )
                return None

            if resp.status_code != 200:
                logger.debug(
                    "Rotation fetch: status %s for UA variant — trying next",
                    resp.status_code,
                )
                continue

            html = resp.text
            if _is_bot_challenge(html):
                logger.debug(
                    "Rotation fetch: bot challenge detected — trying next UA variant"
                )
                continue

            return FetchResult(
                html=html,
                resolved_url=str(resp.url),
                strategy_tier="headers_rotation",
                status_code=resp.status_code,
            )

    return None
