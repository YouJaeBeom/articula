"""
Sub-AC 9d-ii: The headless browser strategy function raises FetchError when the
mocked browser driver completes successfully but the rendered page is inaccessible
(e.g., HTTP 4xx/5xx status code, empty body, or known error-page fingerprint).

Verified by unit tests that inject driver mocks returning bad-status or blank
responses — the driver itself does NOT raise an exception; only the page content
is problematic.

Test strategy
-------------
``fetch_browser`` calls ``page.goto()`` which returns a mock response object
(not None, not raising), and ``page.content()`` returns either a normal or empty
HTML string.  The mock response's ``.status`` attribute controls the HTTP status
code seen by the browser tier.

The patch target is ``playwright.async_api.async_playwright`` — the lazy import
used inside ``fetch_browser``.

These tests require playwright to be installed (they are skipped otherwise)
because the ``_require_playwright()`` guard inside ``fetch_browser`` raises
``BrowserNotInstalledError`` before any driver mock is exercised.

Test matrix
-----------
Class TestFetchBrowserRaisesFetchErrorOnBadStatus
  * HTTP 404 from page navigation  → FetchError with status_code=404
  * HTTP 500 from page navigation  → FetchError with status_code=500
  * HTTP 403 from page navigation  → FetchError with status_code=403
  * HTTP 410 from page navigation  → FetchError with status_code=410
  * HTTP 503 from page navigation  → FetchError with status_code=503

Class TestFetchBrowserRaisesFetchErrorOnEmptyBody
  * Empty string from page.content()          → FetchError
  * Whitespace-only from page.content()       → FetchError
  * Newlines-only from page.content()         → FetchError

Class TestFetchBrowserRaisesFetchErrorOnErrorPageFingerprint
  * "404 Not Found" in rendered HTML (200 OK) → FetchError
  * "This site can't be reached" in HTML      → FetchError
  * "Access Denied" in HTML                   → FetchError

Class TestFetchBrowserBadResponseFetchErrorAttributes
  * FetchError.url equals the input URL
  * FetchError.attempted_strategies contains 'browser'
  * FetchError.status_code reflects the HTTP status (bad-status case)
  * FetchError is catchable as ScraperError
  * browser.close() is still called after bad-status response (finally block)

Class TestFetchBrowserGoodResponseNotRaised
  * 200 OK with real HTML body does NOT raise → returns FetchResult
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip the entire module when playwright is absent — the _require_playwright()
# guard inside fetch_browser fires before any driver mock is exercised.
pytest.importorskip(
    "playwright",
    reason=(
        "playwright is not installed — install with "
        "'pip install articula[browser]' to run these tests"
    ),
)

from articula._browser import fetch_browser  # noqa: E402
from articula._fetcher import FetchResult  # noqa: E402
from articula.exceptions import FetchError, ScraperError  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://browser-bad-response.example.test/article"
_PATCH_ASYNC_PLAYWRIGHT = "playwright.async_api.async_playwright"

_VALID_HTML = (
    "<html><head><title>Test Article</title></head>"
    "<body><article><p>This is the article body with enough real content.</p>"
    "</article></body></html>"
)


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------


def _pw_ctx_with_response(
    status_code: int,
    html: str = _VALID_HTML,
) -> AsyncMock:
    """Return a full playwright context mock where navigation succeeds.

    The mocked driver launches, navigates, and returns a response with the
    given *status_code*.  ``page.content()`` returns *html*.  No exceptions
    are raised anywhere in the driver chain — this simulates a completed
    browser session with a (potentially) bad outcome page.

    Structure:
        async_playwright()              → mock_ctx
        async with mock_ctx as pw:      → mock_pw
        await pw.chromium.launch()      → mock_browser
        await browser.new_context(...)  → mock_browser_context
        await context.new_page()        → mock_page
        await page.goto(...)            → mock_resp (resp.status = status_code)
        await page.content()            → html
        page.url                        → _TEST_URL
    """
    mock_resp = MagicMock()
    mock_resp.status = status_code

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=mock_resp)
    mock_page.content = AsyncMock(return_value=html)
    mock_page.url = _TEST_URL

    mock_browser_context = AsyncMock()
    mock_browser_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_browser_context)

    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


# ---------------------------------------------------------------------------
# Class 1 — FetchError raised when page returns bad HTTP status
# ---------------------------------------------------------------------------


class TestFetchBrowserRaisesFetchErrorOnBadStatus:
    """
    ``fetch_browser`` must raise ``FetchError`` when the page.goto() call
    completes without an exception but the HTTP status code is 4xx or 5xx.

    This distinguishes the "driver succeeded, page is inaccessible" case from
    the "driver threw an exception" case (Sub-AC 9d-i).
    """

    @pytest.mark.asyncio
    async def test_http_404_raises_fetch_error(self) -> None:
        """HTTP 404 from successful navigation → FetchError (not None, not raw result)."""
        mock_ctx = _pw_ctx_with_response(status_code=404)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_http_500_raises_fetch_error(self) -> None:
        """HTTP 500 from successful navigation → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=500)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_http_403_raises_fetch_error(self) -> None:
        """HTTP 403 Forbidden from browser navigation → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=403)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_http_410_raises_fetch_error(self) -> None:
        """HTTP 410 Gone from browser navigation → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=410)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_http_503_raises_fetch_error(self) -> None:
        """HTTP 503 Service Unavailable from browser navigation → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=503)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_http_400_raises_fetch_error(self) -> None:
        """HTTP 400 Bad Request from browser navigation → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=400)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)


# ---------------------------------------------------------------------------
# Class 2 — FetchError raised when page body is empty or blank
# ---------------------------------------------------------------------------


class TestFetchBrowserRaisesFetchErrorOnEmptyBody:
    """
    ``fetch_browser`` must raise ``FetchError`` when the page.goto() call
    returns a 200 OK response but page.content() returns an empty or
    whitespace-only string.

    An empty body indicates the page loaded without content (e.g. a redirect
    to an error page consumed by JS, a JS SPA that failed to render, etc.).
    """

    @pytest.mark.asyncio
    async def test_empty_string_body_raises_fetch_error(self) -> None:
        """Empty string from page.content() with 200 OK → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html="")

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_whitespace_only_body_raises_fetch_error(self) -> None:
        """Whitespace-only body from page.content() with 200 OK → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html="   \t  ")

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_newlines_only_body_raises_fetch_error(self) -> None:
        """Newlines-only body from page.content() with 200 OK → FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html="\n\n\n")

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)


# ---------------------------------------------------------------------------
# Class 3 — FetchError raised when rendered HTML has error-page fingerprint
# ---------------------------------------------------------------------------


class TestFetchBrowserRaisesFetchErrorOnErrorPageFingerprint:
    """
    ``fetch_browser`` must raise ``FetchError`` when the page.goto() succeeds
    with HTTP 200 OK but the rendered HTML contains a known error-page
    fingerprint (e.g. a soft-404 or browser error page rendered as HTML).

    This covers cases where the server returns 200 but the content signals
    an inaccessible resource (common with SPAs, CDNs, and hosting platforms).
    """

    @pytest.mark.asyncio
    async def test_404_not_found_fingerprint_in_200_page_raises_fetch_error(
        self,
    ) -> None:
        """'404 Not Found' in HTML body with 200 OK status → FetchError."""
        error_html = "<html><body><h1>404 Not Found</h1><p>Page missing.</p></body></html>"
        mock_ctx = _pw_ctx_with_response(status_code=200, html=error_html)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_access_denied_fingerprint_raises_fetch_error(self) -> None:
        """'Access Denied' in rendered HTML → FetchError."""
        error_html = "<html><body><h1>Access Denied</h1><p>You do not have permission.</p></body></html>"
        mock_ctx = _pw_ctx_with_response(status_code=200, html=error_html)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_browser_connection_error_fingerprint_raises_fetch_error(
        self,
    ) -> None:
        """'This site can't be reached' (browser error page) → FetchError."""
        # Playwright sometimes renders a browser-native error page as HTML
        error_html = (
            "<html><body>"
            "<div>This site can't be reached</div>"
            "<p>ERR_CONNECTION_REFUSED</p>"
            "</body></html>"
        )
        mock_ctx = _pw_ctx_with_response(status_code=200, html=error_html)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)


# ---------------------------------------------------------------------------
# Class 4 — FetchError attributes are correct for bad-response cases
# ---------------------------------------------------------------------------


class TestFetchBrowserBadResponseFetchErrorAttributes:
    """
    Verify the shape of ``FetchError`` raised by ``fetch_browser`` when the
    driver completes successfully but the page is inaccessible.
    """

    @pytest.mark.asyncio
    async def test_fetch_error_url_equals_input_url_on_bad_status(self) -> None:
        """FetchError.url must equal the URL passed to fetch_browser (bad-status case)."""
        mock_ctx = _pw_ctx_with_response(status_code=404)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert exc_info.value.url == _TEST_URL

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_contains_browser_on_bad_status(
        self,
    ) -> None:
        """FetchError.attempted_strategies must contain 'browser' (bad-status case)."""
        mock_ctx = _pw_ctx_with_response(status_code=500)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert "browser" in exc_info.value.attempted_strategies

    @pytest.mark.asyncio
    async def test_fetch_error_status_code_reflects_http_status(self) -> None:
        """FetchError.status_code must equal the bad HTTP status code returned."""
        mock_ctx = _pw_ctx_with_response(status_code=404)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_fetch_error_500_status_code_reflected(self) -> None:
        """FetchError.status_code == 500 for HTTP 500 response."""
        mock_ctx = _pw_ctx_with_response(status_code=500)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_bad_status_fetch_error_is_catchable_as_scraper_error(self) -> None:
        """FetchError (bad-status) IS-A ScraperError — broad except ScraperError catches it."""
        mock_ctx = _pw_ctx_with_response(status_code=404)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(ScraperError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert isinstance(exc_info.value, FetchError)

    @pytest.mark.asyncio
    async def test_browser_close_called_after_bad_status(self) -> None:
        """browser.close() must still be called after a bad-status response (finally block)."""
        mock_resp = MagicMock()
        mock_resp.status = 404

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=mock_resp)
        mock_page.content = AsyncMock(return_value="<html>Not Found</html>")
        mock_page.url = _TEST_URL

        mock_browser_context = AsyncMock()
        mock_browser_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_browser_context)

        mock_pw = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_body_fetch_error_url_equals_input_url(self) -> None:
        """FetchError.url must equal the URL passed to fetch_browser (empty-body case)."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html="")

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert exc_info.value.url == _TEST_URL

    @pytest.mark.asyncio
    async def test_empty_body_fetch_error_attempted_strategies_contains_browser(
        self,
    ) -> None:
        """FetchError.attempted_strategies contains 'browser' (empty-body case)."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html="")

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert "browser" in exc_info.value.attempted_strategies


# ---------------------------------------------------------------------------
# Class 5 — Good responses are NOT rejected (regression guard)
# ---------------------------------------------------------------------------


class TestFetchBrowserGoodResponseNotRaised:
    """
    Regression guard: ``fetch_browser`` must NOT raise for valid 200 OK
    responses with real HTML content, even if that content happens to contain
    incidental substrings.

    This ensures the error-detection logic does not create false positives.
    """

    @pytest.mark.asyncio
    async def test_200_with_real_html_returns_fetch_result(self) -> None:
        """HTTP 200 with non-empty, clean HTML must return FetchResult (not raise)."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html=_VALID_HTML)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            result = await fetch_browser(_TEST_URL)

        assert result is not None
        assert isinstance(result, FetchResult)
        assert result.strategy_tier == "browser"
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_200_with_real_html_resolved_url_is_set(self) -> None:
        """FetchResult.resolved_url must equal page.url after successful navigation."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html=_VALID_HTML)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            result = await fetch_browser(_TEST_URL)

        assert result is not None
        assert result.resolved_url == _TEST_URL

    @pytest.mark.asyncio
    async def test_200_with_real_html_does_not_raise_fetch_error(self) -> None:
        """Clean 200 OK response must NOT raise FetchError."""
        mock_ctx = _pw_ctx_with_response(status_code=200, html=_VALID_HTML)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            try:
                result = await fetch_browser(_TEST_URL)
            except FetchError as exc:
                pytest.fail(
                    f"fetch_browser raised FetchError for a valid 200 response: {exc}"
                )
            else:
                assert result is not None
