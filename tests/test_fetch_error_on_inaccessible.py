"""
Sub-AC 9b: The static fetch strategy raises FetchError (or a sentinel) when
the HTTP response indicates inaccessibility (e.g., 403, 404, connection
error), verified by unit tests with mocked HTTP responses.

Design overview
---------------
``fetch_static`` returns ``None`` (the sentinel) for every HTTP response that
indicates the page is inaccessible:

  - Hard 4xx codes (401, 404, 410): permanent failure, no retry
  - Bot-detection/temporary codes (403, 429, 503): page blocked or rate-limited
  - Network-level errors (connection refused, timeout): unreachable server

When the escalation coordinator (``Scraper.scrape``) exhausts all tiers, it
raises ``FetchError`` with the full list of ``attempted_strategies``.

Two mock strategies are used
-----------------------------
1. **``_get_with_retry`` patch** — replaces the retry-aware internal GET helper
   with an ``AsyncMock`` that returns a real ``httpx.Response`` object.  This
   avoids actual network I/O and retry-sleep delays while still exercising
   ``fetch_static``'s status-code classification logic.

2. **``httpx.MockTransport``** — injects a real transport handler at the httpx
   level so that ``fetch_static`` and ``_get_with_retry`` run unmodified; only
   the TCP layer is replaced.  Used for the most realistic integration checks.

Test matrix
-----------
Class TestStaticFetchInaccessibleHttpResponses
  * HTTP 403 Forbidden         → fetch_static returns None (real httpx.Response)
  * HTTP 404 Not Found         → fetch_static returns None (real httpx.Response)
  * HTTP 401 Unauthorized      → fetch_static returns None (real httpx.Response)
  * HTTP 410 Gone              → fetch_static returns None (real httpx.Response)
  * HTTP 429 Too Many Requests → fetch_static returns None (real httpx.Response)
  * HTTP 503 Service Unavail.  → fetch_static returns None (real httpx.Response)
  * httpx.ConnectError         → fetch_static returns None (connection refused)
  * httpx.TimeoutException     → fetch_static returns None (timeout)

Class TestMockTransportInaccessible (transport-level mocking)
  * HTTP 403 via MockTransport → fetch_static returns None
  * HTTP 404 via MockTransport → fetch_static returns None
  * ConnectError via MockTransport → fetch_static returns None

Class TestScraperFetchErrorOnInaccessible (coordinator / end-to-end)
  * 403 static → FetchError raised by coordinator
  * 404 static → FetchError raised by coordinator
  * connection error → FetchError raised by coordinator
  * FetchError.url equals the input URL
  * FetchError.attempted_strategies contains 'static'
  * FetchError is catchable as ScraperError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from articula._fetcher import FetchResult, fetch_static
from articula._scraper import Scraper
from articula.exceptions import FetchError, ScraperError

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_URL = "https://inaccessible.example.test/article"

# Patch paths
_PATCH_GET_WITH_RETRY = "articula._fetcher._get_with_retry"
_PATCH_ASYNCIO_SLEEP = "articula._fetcher.asyncio.sleep"
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER = "articula._scraper.fetch_browser"


# ---------------------------------------------------------------------------
# Helper: build a real httpx.Response object with a given status code.
# Using real httpx.Response (not MagicMock) exercises the actual attribute
# access paths in fetch_static and _get_with_retry.
# ---------------------------------------------------------------------------


def _real_response(status_code: int, body: str = "") -> httpx.Response:
    """Return a real ``httpx.Response`` with the given *status_code*.

    ``content`` is set so ``resp.text`` and ``resp.headers`` work normally.
    """
    return httpx.Response(
        status_code,
        content=body.encode(),
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", _URL),
    )


# ---------------------------------------------------------------------------
# Class 1 — fetch_static → None for inaccessible HTTP responses
# (mock at the _get_with_retry layer; responses are real httpx.Response)
# ---------------------------------------------------------------------------


class TestStaticFetchInaccessibleHttpResponses:
    """
    ``fetch_static`` must return ``None`` for every HTTP response that signals
    the page is inaccessible.  Tests use real ``httpx.Response`` objects (not
    ``MagicMock``) so the actual attribute access paths are exercised.
    """

    @pytest.mark.asyncio
    async def test_http_403_forbidden_returns_none(self) -> None:
        """HTTP 403 Forbidden (bot protection) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(403, "Forbidden")
            result = await fetch_static(_URL)

        assert result is None, "fetch_static must return None on HTTP 403"
        assert not isinstance(result, FetchResult)

    @pytest.mark.asyncio
    async def test_http_404_not_found_returns_none(self) -> None:
        """HTTP 404 Not Found (hard failure, no retry) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(404, "Not Found")
            result = await fetch_static(_URL)

        assert result is None, "fetch_static must return None on HTTP 404"

    @pytest.mark.asyncio
    async def test_http_401_unauthorized_returns_none(self) -> None:
        """HTTP 401 Unauthorized (hard failure, no retry) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(401, "Unauthorized")
            result = await fetch_static(_URL)

        assert result is None, "fetch_static must return None on HTTP 401"

    @pytest.mark.asyncio
    async def test_http_410_gone_returns_none(self) -> None:
        """HTTP 410 Gone (hard failure, resource permanently removed) → None."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(410, "Gone")
            result = await fetch_static(_URL)

        assert result is None, "fetch_static must return None on HTTP 410"

    @pytest.mark.asyncio
    async def test_http_429_too_many_requests_returns_none(self) -> None:
        """HTTP 429 Too Many Requests (rate-limited) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(429, "Too Many Requests")
            result = await fetch_static(_URL)

        assert result is None, "fetch_static must return None on HTTP 429"

    @pytest.mark.asyncio
    async def test_http_503_service_unavailable_returns_none(self) -> None:
        """HTTP 503 Service Unavailable (temporary block) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(503, "Service Unavailable")
            result = await fetch_static(_URL)

        assert result is None, "fetch_static must return None on HTTP 503"

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self) -> None:
        """``httpx.ConnectError`` (connection refused) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.ConnectError(
                "Connection refused: [Errno 111] connect(2) failed"
            )
            result = await fetch_static(_URL, max_retries=1)

        assert result is None, "fetch_static must return None on connection refused"

    @pytest.mark.asyncio
    async def test_timeout_exception_returns_none(self) -> None:
        """``httpx.TimeoutException`` (read timeout) → None sentinel."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.TimeoutException(
                "Request timed out after 30 s"
            )
            result = await fetch_static(_URL, max_retries=1)

        assert result is None, "fetch_static must return None on timeout"

    @pytest.mark.asyncio
    async def test_sentinel_is_exactly_none_not_falsy(self) -> None:
        """The sentinel is exactly ``None``, not ``False``, ``0``, or ``[]``."""
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _real_response(403)
            result = await fetch_static(_URL)

        assert result is None
        assert result is not False
        assert result != 0
        assert result != []


# ---------------------------------------------------------------------------
# Class 2 — transport-level mocking (httpx.MockTransport)
# Tests run fetch_static + _get_with_retry unmodified; only the TCP layer
# is replaced, giving the most realistic test of the inaccessible-response
# handling path.
# ---------------------------------------------------------------------------


def _mock_transport_for_status(status_code: int) -> httpx.MockTransport:
    """Return an ``httpx.MockTransport`` that always responds with *status_code*."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=b"",
            headers={"content-type": "text/html"},
        )

    return httpx.MockTransport(handler)


def _mock_transport_raising(exc_type: type, message: str) -> httpx.MockTransport:
    """Return an ``httpx.MockTransport`` that raises *exc_type* on every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type(message)

    return httpx.MockTransport(handler)


class TestMockTransportInaccessible:
    """
    ``fetch_static`` returns ``None`` for inaccessible pages when the
    ``httpx.AsyncClient`` uses an ``httpx.MockTransport`` — the most realistic
    form of HTTP mocking because ``fetch_static`` and ``_get_with_retry`` run
    completely unmodified.
    """

    async def _run_fetch_static_with_transport(
        self,
        transport: httpx.MockTransport,
        max_retries: int = 1,
    ) -> FetchResult | None:
        """
        Inject *transport* into ``fetch_static`` by patching the
        ``httpx.AsyncClient`` constructor to prepend our mock transport.

        ``asyncio.sleep`` is also patched so retry-back-off codes (503, 429)
        complete instantly in tests.
        """
        with (
            patch("articula._fetcher.httpx.AsyncClient") as mock_cls,
            patch(_PATCH_ASYNCIO_SLEEP, new_callable=AsyncMock),
        ):
            # Build a real AsyncClient pre-wired with our mock transport.
            real_client = httpx.AsyncClient(transport=transport, timeout=30.0)
            # Return the pre-built client whenever the constructor is called.
            mock_cls.return_value = real_client

            result = await fetch_static(_URL, max_retries=max_retries)

        return result

    @pytest.mark.asyncio
    async def test_403_via_mock_transport_returns_none(self) -> None:
        """HTTP 403 from real MockTransport → None sentinel."""
        transport = _mock_transport_for_status(403)
        result = await self._run_fetch_static_with_transport(transport)
        assert result is None, "403 via MockTransport must produce None sentinel"

    @pytest.mark.asyncio
    async def test_404_via_mock_transport_returns_none(self) -> None:
        """HTTP 404 from real MockTransport → None sentinel."""
        transport = _mock_transport_for_status(404)
        result = await self._run_fetch_static_with_transport(transport)
        assert result is None, "404 via MockTransport must produce None sentinel"

    @pytest.mark.asyncio
    async def test_connect_error_via_mock_transport_returns_none(self) -> None:
        """``ConnectError`` raised by MockTransport → None sentinel."""
        transport = _mock_transport_raising(
            httpx.ConnectError, "Connection refused"
        )
        result = await self._run_fetch_static_with_transport(transport)
        assert result is None, "ConnectError via MockTransport must produce None sentinel"


# ---------------------------------------------------------------------------
# Class 3 — Scraper coordinator raises FetchError on inaccessible pages
# Tests mock fetch_static at the Scraper level and verify FetchError is raised
# with the correct attributes.
# ---------------------------------------------------------------------------


class TestScraperFetchErrorOnInaccessible:
    """
    When ``fetch_static`` returns ``None`` (because the HTTP response
    indicates inaccessibility), the ``Scraper`` coordinator must raise
    ``FetchError`` after all tiers are exhausted.
    """

    @pytest.mark.asyncio
    async def test_fetch_error_raised_on_403_inaccessible(self) -> None:
        """403-driven static exhaustion propagates as FetchError from Scraper."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            # Simulate 403-driven None return from every tier.
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(FetchError):
                async with Scraper() as s:
                    await s.scrape(_URL)

    @pytest.mark.asyncio
    async def test_fetch_error_raised_on_404_inaccessible(self) -> None:
        """404-driven static exhaustion propagates as FetchError from Scraper."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(FetchError):
                async with Scraper() as s:
                    await s.scrape(_URL)

    @pytest.mark.asyncio
    async def test_fetch_error_raised_on_connection_error(self) -> None:
        """Connection-error-driven static exhaustion propagates as FetchError."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(FetchError):
                async with Scraper() as s:
                    await s.scrape(_URL)

    @pytest.mark.asyncio
    async def test_fetch_error_url_is_the_input_url(self) -> None:
        """FetchError.url must equal the URL that was requested."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_URL)

        assert exc_info.value.url == _URL

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_includes_static(self) -> None:
        """FetchError.attempted_strategies must list 'static' as first tier tried."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_URL)

        exc = exc_info.value
        assert "static" in exc.attempted_strategies
        assert exc.attempted_strategies[0] == "static", (
            "static must be the first strategy attempted"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_is_catchable_as_scraper_error(self) -> None:
        """FetchError IS-A ScraperError so broad except-ScraperError catches it."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(ScraperError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_URL)

        assert isinstance(exc_info.value, FetchError)

    @pytest.mark.asyncio
    async def test_fetch_error_message_contains_url(self) -> None:
        """FetchError.__str__ must reference the problematic URL."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = None

            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_URL)

        assert _URL in str(exc_info.value), (
            "FetchError string representation must contain the failing URL"
        )

    @pytest.mark.asyncio
    async def test_only_static_attempted_then_fetch_error_when_forced(self) -> None:
        """
        With ``force_strategy='static'``, only the static tier is attempted.
        When it returns None (inaccessible), FetchError is raised immediately
        without trying headers_rotation or browser.
        """
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
        ):
            mock_static.return_value = None

            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="static") as s:
                    await s.scrape(_URL)

        mock_rotation.assert_not_awaited()
        exc = exc_info.value
        assert exc.attempted_strategies == ("static",)
