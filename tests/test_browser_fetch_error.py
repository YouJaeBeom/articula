"""
Sub-AC 9d-i: The headless browser strategy function raises FetchError when the
mocked browser driver throws an exception or times out during page load or JS
execution.

Verified by unit tests that inject driver mocks configured to raise on
launch/navigate.

Test strategy
-------------
``fetch_browser`` imports ``async_playwright`` lazily (inside the function body),
so the patch target is ``playwright.async_api.async_playwright``.  Minimal async
context-manager mocks replace the real playwright context and configure
``pw.chromium.launch`` / ``page.goto`` to raise the desired exceptions.

These tests require playwright to be installed (they are skipped otherwise) because
the ``_require_playwright()`` guard inside ``fetch_browser`` raises
``BrowserNotInstalledError`` before any driver mock is reached in a base-only
environment.

Test matrix
-----------
Class TestFetchBrowserRaisesFetchErrorOnLaunchFailure
  * RuntimeError from chromium.launch   → FetchError
  * OSError from chromium.launch        → FetchError
  * BrowserType.launch timeout          → FetchError

Class TestFetchBrowserRaisesFetchErrorOnNavigationFailure
  * RuntimeError from page.goto         → FetchError
  * TimeoutError from page.goto         → FetchError
  * Exception from page.goto            → FetchError

Class TestFetchBrowserFetchErrorAttributes
  * FetchError.url equals the input URL
  * FetchError.attempted_strategies contains 'browser'
  * FetchError is catchable as ScraperError
  * FetchError message is non-empty and references the URL

Class TestFetchBrowserFetchErrorIsNotReturnedAsNone
  * Confirm the return value is never None on driver exception (must raise)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip the entire module when playwright is absent — the _require_playwright()
# guard inside fetch_browser fires before any driver mock is exercised, making
# the driver-injection approach untestable in a base-only environment.
pytest.importorskip(
    "playwright",
    reason=(
        "playwright is not installed — install with "
        "'pip install articula[browser]' to run these tests"
    ),
)

from articula._browser import fetch_browser  # noqa: E402
from articula.exceptions import FetchError, ScraperError  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://browser-error.example.test/article"

# The lazy import inside fetch_browser is:  from playwright.async_api import async_playwright
# Patching the attribute on the already-loaded module replaces the reference
# that the from-import resolves at call time.
_PATCH_ASYNC_PLAYWRIGHT = "playwright.async_api.async_playwright"


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------


def _pw_ctx_raise_on_launch(exc: Exception) -> AsyncMock:
    """Return a mock playwright context manager where chromium.launch raises ``exc``.

    Structure:
        async_playwright()          → mock_ctx (AsyncMock)
        async with mock_ctx as pw:  → mock_pw (MagicMock)
        await pw.chromium.launch()  → raises exc
    """
    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(side_effect=exc)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


def _pw_ctx_raise_on_navigate(exc: Exception) -> AsyncMock:
    """Return a mock playwright context manager where page.goto raises ``exc``.

    Structure:
        async_playwright()              → mock_ctx (AsyncMock)
        async with mock_ctx as pw:      → mock_pw
        await pw.chromium.launch()      → mock_browser
        await browser.new_context(...)  → mock_browser_context
        await context.new_page()        → mock_page
        await page.goto(...)            → raises exc
    """
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(side_effect=exc)

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
# Class 1 — FetchError raised when chromium.launch throws
# ---------------------------------------------------------------------------


class TestFetchBrowserRaisesFetchErrorOnLaunchFailure:
    """
    ``fetch_browser`` must raise ``FetchError`` when the browser launch step
    throws any exception, rather than silently returning ``None``.

    Each test injects a mock playwright context whose ``pw.chromium.launch``
    method is configured to raise a specific exception type.
    """

    @pytest.mark.asyncio
    async def test_runtime_error_on_launch_raises_fetch_error(self) -> None:
        """RuntimeError from chromium.launch → FetchError (not None, not RuntimeError)."""
        mock_ctx = _pw_ctx_raise_on_launch(RuntimeError("Browser failed to start"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_os_error_on_launch_raises_fetch_error(self) -> None:
        """OSError from chromium.launch (e.g., missing executable) → FetchError."""
        mock_ctx = _pw_ctx_raise_on_launch(
            OSError("Chromium executable not found at expected path")
        )

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_timeout_error_on_launch_raises_fetch_error(self) -> None:
        """TimeoutError from chromium.launch (cold start exceeded) → FetchError."""
        mock_ctx = _pw_ctx_raise_on_launch(
            TimeoutError("Browser launch timed out after 30 s")
        )

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_exception_on_launch_is_fetch_error_not_reraised_raw(self) -> None:
        """The original exception must be wrapped in FetchError, not reraised."""
        original_error = RuntimeError("Launch failed")
        mock_ctx = _pw_ctx_raise_on_launch(original_error)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        # Must be FetchError, not the raw RuntimeError
        assert type(exc_info.value) is FetchError

    @pytest.mark.asyncio
    async def test_launch_error_chained_from_original_exception(self) -> None:
        """FetchError.__cause__ should be the original driver exception (PEP 3134)."""
        original_error = RuntimeError("Driver died")
        mock_ctx = _pw_ctx_raise_on_launch(original_error)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        # The raise ... from exc pattern sets __cause__
        assert exc_info.value.__cause__ is original_error


# ---------------------------------------------------------------------------
# Class 2 — FetchError raised when page.goto throws
# ---------------------------------------------------------------------------


class TestFetchBrowserRaisesFetchErrorOnNavigationFailure:
    """
    ``fetch_browser`` must raise ``FetchError`` when the page navigation step
    throws any exception (network error, timeout, JS crash, etc.).

    Each test injects a mock playwright context with a successfully launched
    browser but a ``page.goto`` method configured to raise.
    """

    @pytest.mark.asyncio
    async def test_runtime_error_on_navigate_raises_fetch_error(self) -> None:
        """RuntimeError from page.goto → FetchError."""
        mock_ctx = _pw_ctx_raise_on_navigate(
            RuntimeError("Navigation crashed: renderer process killed")
        )

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_timeout_error_on_navigate_raises_fetch_error(self) -> None:
        """TimeoutError from page.goto (JS rendering exceeded timeout) → FetchError."""
        mock_ctx = _pw_ctx_raise_on_navigate(
            TimeoutError("page.goto timeout: networkidle not reached after 30000 ms")
        )

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_exception_on_navigate_raises_fetch_error(self) -> None:
        """Generic Exception from page.goto → FetchError."""
        mock_ctx = _pw_ctx_raise_on_navigate(
            Exception("net::ERR_NAME_NOT_RESOLVED")
        )

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                await fetch_browser(_TEST_URL)

    @pytest.mark.asyncio
    async def test_navigate_error_is_fetch_error_not_raw_exception(self) -> None:
        """The exception raised must be FetchError, not the raw navigation error."""
        mock_ctx = _pw_ctx_raise_on_navigate(RuntimeError("Navigate failed"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert type(exc_info.value) is FetchError

    @pytest.mark.asyncio
    async def test_navigate_error_chained_from_original_exception(self) -> None:
        """FetchError.__cause__ is the original navigation exception (PEP 3134)."""
        original_error = TimeoutError("goto timed out")
        mock_ctx = _pw_ctx_raise_on_navigate(original_error)

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert exc_info.value.__cause__ is original_error

    @pytest.mark.asyncio
    async def test_browser_close_still_called_after_navigate_error(self) -> None:
        """browser.close() must be called even when page.goto raises (finally block)."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=RuntimeError("nav failed"))

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

        # The finally block in fetch_browser must close the browser even on error
        mock_browser.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Class 3 — FetchError attributes are correct
# ---------------------------------------------------------------------------


class TestFetchBrowserFetchErrorAttributes:
    """
    Verify the shape of the ``FetchError`` raised by ``fetch_browser`` on
    driver exceptions: url, attempted_strategies, message content, and hierarchy.
    """

    @pytest.mark.asyncio
    async def test_fetch_error_url_equals_input_url(self) -> None:
        """FetchError.url must equal the URL passed to fetch_browser."""
        mock_ctx = _pw_ctx_raise_on_launch(RuntimeError("launch error"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert exc_info.value.url == _TEST_URL, (
            f"FetchError.url must be {_TEST_URL!r}, got {exc_info.value.url!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_contains_browser(self) -> None:
        """FetchError.attempted_strategies must contain 'browser'."""
        mock_ctx = _pw_ctx_raise_on_navigate(TimeoutError("nav timeout"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert "browser" in exc_info.value.attempted_strategies, (
            "FetchError.attempted_strategies must include 'browser' "
            f"but got {exc_info.value.attempted_strategies!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_is_catchable_as_scraper_error(self) -> None:
        """FetchError IS-A ScraperError — broad except ScraperError catches it."""
        mock_ctx = _pw_ctx_raise_on_launch(RuntimeError("launch failed"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(ScraperError) as exc_info:
                await fetch_browser(_TEST_URL)

        assert isinstance(exc_info.value, FetchError), (
            "The caught ScraperError must be an instance of FetchError"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_message_is_non_empty(self) -> None:
        """FetchError must carry a non-empty descriptive message."""
        mock_ctx = _pw_ctx_raise_on_launch(RuntimeError("launch failed"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        message = str(exc_info.value)
        assert message, "FetchError must have a non-empty message"

    @pytest.mark.asyncio
    async def test_fetch_error_message_references_url(self) -> None:
        """FetchError string representation must reference the failing URL."""
        mock_ctx = _pw_ctx_raise_on_navigate(RuntimeError("nav error"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError) as exc_info:
                await fetch_browser(_TEST_URL)

        error_str = str(exc_info.value)
        assert _TEST_URL in error_str, (
            f"FetchError string must contain the URL {_TEST_URL!r}; got: {error_str!r}"
        )


# ---------------------------------------------------------------------------
# Class 4 — Driver exception must NOT silently return None
# ---------------------------------------------------------------------------


class TestFetchBrowserNeverReturnsNoneOnDriverException:
    """
    Confirm that ``fetch_browser`` never swallows driver exceptions by
    returning ``None`` — it must always raise ``FetchError`` when the driver
    throws.
    """

    @pytest.mark.asyncio
    async def test_does_not_return_none_on_launch_error(self) -> None:
        """When chromium.launch raises, fetch_browser must raise — not return None."""
        mock_ctx = _pw_ctx_raise_on_launch(RuntimeError("launch failed"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                result = await fetch_browser(_TEST_URL)
                # If we reach here without raising, the test fails
                pytest.fail(
                    f"fetch_browser returned {result!r} instead of raising FetchError"
                )

    @pytest.mark.asyncio
    async def test_does_not_return_none_on_navigate_error(self) -> None:
        """When page.goto raises, fetch_browser must raise — not return None."""
        mock_ctx = _pw_ctx_raise_on_navigate(TimeoutError("nav timed out"))

        with patch(_PATCH_ASYNC_PLAYWRIGHT, return_value=mock_ctx):
            with pytest.raises(FetchError):
                result = await fetch_browser(_TEST_URL)
                pytest.fail(
                    f"fetch_browser returned {result!r} instead of raising FetchError"
                )
