"""
Sub-AC 2: Lazy browser initialisation in Scraper.

Verifies that:
  - ``Scraper._browser`` is ``None`` immediately after construction.
  - ``Scraper._browser`` is ``None`` before the first scrape call inside an
    async context manager (i.e. entering the context alone does not launch the
    browser).
  - ``Scraper._browser`` is set to a non-``None`` browser instance after
    ``_ensure_browser()`` is called (the core lazy-init path).
  - A full ``scrape()`` call that reaches the browser tier also transitions
    ``_browser`` from ``None`` to an instance.
  - After the async context manager exits, ``_browser`` is ``None`` again
    (cleanup was performed).

Strategy
--------
All tests are unit-level.  They patch :meth:`Scraper._launch_browser` with an
``AsyncMock`` so that no real Playwright process is ever started.  This gives
fine-grained control over the exact mock browser object stored in
``self._browser`` and keeps the tests fast and hermetic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult
from articula._scraper import Scraper

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_mock_browser() -> AsyncMock:
    """Return a minimal async mock standing in for a Playwright Browser."""
    browser = AsyncMock()
    browser.close = AsyncMock()
    return browser


def _make_extraction_result() -> ExtractionResult:
    """Return a minimal valid ExtractionResult for extraction mocking."""
    return ExtractionResult(
        title="Test Article Title",
        text=(
            "This is a test article body with enough content to pass the "
            "minimum-length and token-count validation requirements imposed "
            "by the is_meaningful_body guard in the extraction pipeline."
        ),
        author=None,
        published_date=None,
        method="trafilatura",
        confidence=0.8,
        language="en",
    )


def _make_fetch_result() -> FetchResult:
    """Return a minimal valid FetchResult from the browser tier."""
    return FetchResult(
        html="<html><body><h1>Test</h1><p>content</p></body></html>",
        resolved_url="https://example.com/article",
        strategy_tier="browser",
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Before-construction / before-entry assertions
# ---------------------------------------------------------------------------


class TestBrowserIsNoneBeforeFirstUse:
    """_browser must be None at construction time and before the first scrape."""

    def test_browser_is_none_at_construction(self) -> None:
        """Scraper._browser is None immediately after __init__."""
        scraper = Scraper()
        assert scraper._browser is None

    def test_browser_is_none_with_force_strategy_browser(self) -> None:
        """Configuring force_strategy='browser' does not pre-launch a browser."""
        scraper = Scraper(force_strategy="browser")
        assert scraper._browser is None

    def test_pw_ctx_is_none_at_construction(self) -> None:
        """Scraper._pw_ctx is None immediately after __init__."""
        scraper = Scraper()
        assert scraper._pw_ctx is None

    @pytest.mark.asyncio
    async def test_browser_is_none_after_entering_async_context(self) -> None:
        """Entering the async context manager does not launch a browser."""
        async with Scraper() as s:
            assert s._browser is None

    @pytest.mark.asyncio
    async def test_browser_is_none_after_entering_context_force_browser(self) -> None:
        """Entering context with force_strategy='browser' still defers launch."""
        async with Scraper(force_strategy="browser") as s:
            assert s._browser is None


# ---------------------------------------------------------------------------
# _ensure_browser — core lazy-init path
# ---------------------------------------------------------------------------


class TestEnsureBrowserLazyInit:
    """_ensure_browser() must set _browser on first call and reuse on subsequent ones."""

    @pytest.mark.asyncio
    async def test_browser_is_none_before_ensure_browser(self) -> None:
        """_browser is None before _ensure_browser() is awaited."""
        scraper = Scraper()
        assert scraper._browser is None

    @pytest.mark.asyncio
    async def test_browser_is_set_after_ensure_browser(self) -> None:
        """_browser is not None after _ensure_browser() returns."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            await scraper._ensure_browser()

        assert scraper._browser is not None

    @pytest.mark.asyncio
    async def test_browser_is_exact_mock_object(self) -> None:
        """_browser is the exact object returned by _launch_browser."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            returned = await scraper._ensure_browser()

        assert returned is mock_browser
        assert scraper._browser is mock_browser

    @pytest.mark.asyncio
    async def test_launch_browser_called_once_on_first_call(self) -> None:
        """_launch_browser is invoked exactly once when _ensure_browser is first awaited."""
        mock_browser = _make_mock_browser()
        mock_launch = AsyncMock(return_value=mock_browser)
        scraper = Scraper()

        with patch.object(scraper, "_launch_browser", new=mock_launch):
            await scraper._ensure_browser()

        mock_launch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_launch_browser_not_called_on_subsequent_calls(self) -> None:
        """_launch_browser is not called again after _browser is already set."""
        mock_browser = _make_mock_browser()
        mock_launch = AsyncMock(return_value=mock_browser)
        scraper = Scraper()

        with patch.object(scraper, "_launch_browser", new=mock_launch):
            await scraper._ensure_browser()
            await scraper._ensure_browser()  # second call — must be a no-op
            await scraper._ensure_browser()  # third call

        mock_launch.assert_awaited_once()  # still only one launch

    @pytest.mark.asyncio
    async def test_ensure_browser_returns_same_object_each_time(self) -> None:
        """All calls to _ensure_browser() return the same browser object."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            first = await scraper._ensure_browser()
            second = await scraper._ensure_browser()

        assert first is second
        assert first is mock_browser


# ---------------------------------------------------------------------------
# Full scrape() flow through the browser tier
# ---------------------------------------------------------------------------


class TestBrowserInstantiatedAfterBrowserTierScrape:
    """_browser is None before scrape() and not None after using the browser tier."""

    @pytest.mark.asyncio
    async def test_browser_none_before_scrape_instantiated_after(self) -> None:
        """Core Sub-AC 2 assertion: None → instance transition via scrape()."""
        mock_browser = _make_mock_browser()
        fetch_result = _make_fetch_result()
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch("articula._scraper.fetch_with_existing_browser", new=AsyncMock(return_value=fetch_result)):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper as s:
                        # Before first scrape — browser must be None
                        assert s._browser is None

                        await s.scrape("https://example.com/article")

                        # After first scrape reached the browser tier — must be set
                        assert s._browser is not None
                        assert s._browser is mock_browser

    @pytest.mark.asyncio
    async def test_browser_stays_set_on_second_scrape(self) -> None:
        """_browser remains the same object across multiple scrape() calls."""
        mock_browser = _make_mock_browser()
        fetch_result = _make_fetch_result()
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch("articula._scraper.fetch_with_existing_browser", new=AsyncMock(return_value=fetch_result)):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper as s:
                        await s.scrape("https://example.com/article")
                        browser_after_first = s._browser

                        await s.scrape("https://example.com/other")
                        browser_after_second = s._browser

        assert browser_after_first is mock_browser
        assert browser_after_second is mock_browser
        assert browser_after_first is browser_after_second


# ---------------------------------------------------------------------------
# Cleanup — _browser is None after context exit
# ---------------------------------------------------------------------------


class TestBrowserCleanedUpAfterContextExit:
    """After the async context manager exits, _browser must be None again."""

    @pytest.mark.asyncio
    async def test_browser_is_none_after_async_context_exits(self) -> None:
        """_browser is reset to None when the async with block is exited normally."""
        mock_browser = _make_mock_browser()
        fetch_result = _make_fetch_result()
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch("articula._scraper.fetch_with_existing_browser", new=AsyncMock(return_value=fetch_result)):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper as s:
                        await s.scrape("https://example.com/article")
                        assert s._browser is not None  # inside: instantiated

        # After context exits: cleaned up
        assert scraper._browser is None

    @pytest.mark.asyncio
    async def test_browser_close_called_on_context_exit(self) -> None:
        """browser.close() is awaited when the async context manager exits."""
        mock_browser = _make_mock_browser()
        fetch_result = _make_fetch_result()
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch("articula._scraper.fetch_with_existing_browser", new=AsyncMock(return_value=fetch_result)):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper:
                        await scraper.scrape("https://example.com/article")

        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_is_none_after_context_exits_without_scrape(self) -> None:
        """Exiting a context where no browser was launched leaves _browser None."""
        scraper = Scraper()

        async with scraper:
            assert scraper._browser is None

        assert scraper._browser is None

    @pytest.mark.asyncio
    async def test_close_browser_is_idempotent(self) -> None:
        """Calling _close_browser twice does not raise."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()
        scraper._browser = mock_browser

        await scraper._close_browser()
        assert scraper._browser is None

        # Second call — _browser is already None, must not raise
        await scraper._close_browser()
        assert scraper._browser is None
