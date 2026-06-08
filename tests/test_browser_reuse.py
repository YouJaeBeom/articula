"""
Sub-AC 3: Browser instance reuse across multiple scrape calls within a single
context manager session.

Verifies that:
  - ``_launch_browser`` is invoked **exactly once** across any number of
    sequential ``scrape()`` calls made inside a single ``async with Scraper()``
    session — the browser is not re-launched between calls.
  - The same ``Browser`` object (verified by identity) is passed to
    ``fetch_with_existing_browser`` on every successive call, confirming that
    reuse propagates all the way through the fetch tier.
  - A session-level ``_launch_browser`` call-count of one is maintained even
    for three or more sequential scrape invocations.

All tests are unit-level; they patch ``_launch_browser`` with ``AsyncMock`` so
no real Playwright process is ever started.  ``fetch_with_existing_browser`` and
``extract`` are also patched to keep tests fast and hermetic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult
from articula._scraper import Scraper

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_browser() -> AsyncMock:
    """Return a minimal async mock standing in for a Playwright Browser."""
    browser = AsyncMock()
    browser.close = AsyncMock()
    return browser


def _make_fetch_result(url: str = "https://example.com/article") -> FetchResult:
    """Return a minimal valid FetchResult from the browser tier."""
    return FetchResult(
        html="<html><body><h1>Test</h1><p>content</p></body></html>",
        resolved_url=url,
        strategy_tier="browser",
        status_code=200,
    )


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


# ---------------------------------------------------------------------------
# Core reuse contract: single launch across multiple sequential scrapes
# ---------------------------------------------------------------------------


class TestBrowserNotRelaunched:
    """_launch_browser must be called exactly once per context manager session."""

    @pytest.mark.asyncio
    async def test_two_sequential_scrapes_launch_browser_once(self) -> None:
        """_launch_browser is called once for two sequential scrape() calls."""
        mock_browser = _make_mock_browser()
        mock_launch = AsyncMock(return_value=mock_browser)
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=mock_launch), patch(
            "articula._scraper.fetch_with_existing_browser",
            new=AsyncMock(return_value=_make_fetch_result()),
        ), patch("articula._scraper.extract", return_value=extraction):
            async with scraper as s:
                await s.scrape("https://example.com/first")
                await s.scrape("https://example.com/second")

        mock_launch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_three_sequential_scrapes_launch_browser_once(self) -> None:
        """_launch_browser is called once for three sequential scrape() calls."""
        mock_browser = _make_mock_browser()
        mock_launch = AsyncMock(return_value=mock_browser)
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=mock_launch), patch(
            "articula._scraper.fetch_with_existing_browser",
            new=AsyncMock(return_value=_make_fetch_result()),
        ), patch("articula._scraper.extract", return_value=extraction):
            async with scraper as s:
                await s.scrape("https://example.com/a")
                await s.scrape("https://example.com/b")
                await s.scrape("https://example.com/c")

        mock_launch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_launch_count_matches_context_manager_count(self) -> None:
        """Each separate context manager session triggers exactly one launch."""
        extraction = _make_extraction_result()

        launch_calls: list[int] = []

        for _session_index in range(3):
            mock_browser = _make_mock_browser()
            scraper = Scraper(force_strategy="browser")
            mock_launch = AsyncMock(return_value=mock_browser)

            with patch.object(scraper, "_launch_browser", new=mock_launch), patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ), patch(
                "articula._scraper.extract", return_value=extraction
            ):
                async with scraper as s:
                    await s.scrape("https://example.com/x")
                    await s.scrape("https://example.com/y")

            launch_calls.append(mock_launch.await_count)

        # Each session should have exactly one launch regardless of scrape count
        assert launch_calls == [1, 1, 1]


# ---------------------------------------------------------------------------
# Identity check: fetch_with_existing_browser receives the SAME browser object
# ---------------------------------------------------------------------------


class TestSameBrowserObjectPassedToFetcher:
    """fetch_with_existing_browser must receive the identical browser instance."""

    @pytest.mark.asyncio
    async def test_same_browser_identity_across_two_scrapes(self) -> None:
        """The browser kwarg passed to fetch_with_existing_browser is the same object."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()

        captured_browsers: list[object] = []

        async def _capture_browser(
            url: str, *, browser: object, **kwargs: object
        ) -> FetchResult:
            captured_browsers.append(browser)
            return _make_fetch_result(url)

        scraper = Scraper(force_strategy="browser")

        with patch.object(
            scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)
        ), patch(
            "articula._scraper.fetch_with_existing_browser",
            side_effect=_capture_browser,
        ), patch(
            "articula._scraper.extract", return_value=extraction
        ):
            async with scraper as s:
                await s.scrape("https://example.com/first")
                await s.scrape("https://example.com/second")

        assert len(captured_browsers) == 2
        # Both calls received the exact same browser object
        assert captured_browsers[0] is captured_browsers[1]
        assert captured_browsers[0] is mock_browser

    @pytest.mark.asyncio
    async def test_same_browser_identity_across_three_scrapes(self) -> None:
        """All three sequential calls receive the same browser object by identity."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()

        captured_browsers: list[object] = []

        async def _capture_browser(
            url: str, *, browser: object, **kwargs: object
        ) -> FetchResult:
            captured_browsers.append(browser)
            return _make_fetch_result(url)

        scraper = Scraper(force_strategy="browser")

        with patch.object(
            scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)
        ), patch(
            "articula._scraper.fetch_with_existing_browser",
            side_effect=_capture_browser,
        ), patch(
            "articula._scraper.extract", return_value=extraction
        ):
            async with scraper as s:
                await s.scrape("https://example.com/a")
                await s.scrape("https://example.com/b")
                await s.scrape("https://example.com/c")

        assert len(captured_browsers) == 3
        first = captured_browsers[0]
        for browser in captured_browsers[1:]:
            assert browser is first, "Expected the same browser instance for all calls"
        assert first is mock_browser


# ---------------------------------------------------------------------------
# Integration with _ensure_browser: _browser attribute stays constant
# ---------------------------------------------------------------------------


class TestBrowserAttributeConsistency:
    """The _browser attribute on the Scraper must not change between scrapes."""

    @pytest.mark.asyncio
    async def test_browser_attribute_identical_after_each_scrape(self) -> None:
        """Scraper._browser is the same object before, between, and after scrapes."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        with patch.object(
            scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)
        ), patch(
            "articula._scraper.fetch_with_existing_browser",
            new=AsyncMock(return_value=_make_fetch_result()),
        ), patch(
            "articula._scraper.extract", return_value=extraction
        ):
            async with scraper as s:
                assert s._browser is None  # not yet launched

                await s.scrape("https://example.com/first")
                browser_snapshot_1 = s._browser

                await s.scrape("https://example.com/second")
                browser_snapshot_2 = s._browser

                await s.scrape("https://example.com/third")
                browser_snapshot_3 = s._browser

        # All snapshots must be the same object
        assert browser_snapshot_1 is mock_browser
        assert browser_snapshot_2 is mock_browser
        assert browser_snapshot_3 is mock_browser

    @pytest.mark.asyncio
    async def test_browser_is_none_before_and_after_session(self) -> None:
        """_browser transitions: None → instance (inside) → None (after exit)."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()

        scraper = Scraper(force_strategy="browser")

        assert scraper._browser is None  # before entering context

        with patch.object(
            scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)
        ), patch(
            "articula._scraper.fetch_with_existing_browser",
            new=AsyncMock(return_value=_make_fetch_result()),
        ), patch(
            "articula._scraper.extract", return_value=extraction
        ):
            async with scraper as s:
                await s.scrape("https://example.com/x")
                assert s._browser is mock_browser  # inside: launched

        assert scraper._browser is None  # after exit: cleaned up
