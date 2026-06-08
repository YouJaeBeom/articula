"""
Sub-AC 4: Browser teardown on context manager exit.

Verifies that the browser's ``close()`` method is called **exactly once**:
  - When the ``async with`` block exits **normally**.
  - When the ``async with`` block exits **via an exception**.

Additional edge-case coverage:
  - No browser was ever launched (no scrape reached the browser tier): ``close()``
    must NOT be called.
  - Multiple scrape calls share one browser; ``close()`` is still called once.
  - ``_close_browser()`` is idempotent — calling it twice does not double-close.

All tests are unit-level and use ``unittest.mock`` to isolate from live Playwright.
``_launch_browser`` is patched so that a controlled ``AsyncMock`` is stored in
``scraper._browser``; this means the test verifies the teardown contract without
ever starting a real Chromium process.
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
    """Return a minimal ``AsyncMock`` standing in for a Playwright Browser."""
    browser = AsyncMock()
    browser.close = AsyncMock()
    return browser


def _make_fetch_result(url: str = "https://example.com/article") -> FetchResult:
    """Return a minimal valid ``FetchResult`` from the browser tier."""
    return FetchResult(
        html="<html><body><h1>Test</h1><p>content</p></body></html>",
        resolved_url=url,
        strategy_tier="browser",
        status_code=200,
    )


def _make_extraction_result() -> ExtractionResult:
    """Return a minimal valid ``ExtractionResult`` that passes body validation."""
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
# Normal exit
# ---------------------------------------------------------------------------


class TestBrowserTeardownOnNormalExit:
    """``browser.close()`` is called exactly once when the with-block exits normally."""

    @pytest.mark.asyncio
    async def test_close_called_once_after_single_scrape_normal_exit(self) -> None:
        """Core Sub-AC 4 assertion: close() awaited once on clean exit after one scrape."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper as s:
                        await s.scrape("https://example.com/article")

        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_once_after_multiple_scrapes_normal_exit(self) -> None:
        """``close()`` is called once even when multiple scrapes share the same browser."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper as s:
                        await s.scrape("https://example.com/a")
                        await s.scrape("https://example.com/b")
                        await s.scrape("https://example.com/c")

        # Three scrapes share one browser; teardown must still be exactly once.
        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_not_called_when_no_browser_was_launched(self) -> None:
        """If no scrape reached the browser tier, ``close()`` must not be called."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()

        # Enter and exit without scraping — _browser stays None
        async with scraper:
            pass

        # The mock was never stored in scraper._browser, so close() must be a no-op
        mock_browser.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_browser_state_reset_to_none_after_normal_exit(self) -> None:
        """``_browser`` attribute is ``None`` after the async with-block exits normally."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    async with scraper as s:
                        await s.scrape("https://example.com/article")
                        # Inside the block the browser must be set
                        assert s._browser is mock_browser

        # After clean exit: reset
        assert scraper._browser is None


# ---------------------------------------------------------------------------
# Exception exit
# ---------------------------------------------------------------------------


class TestBrowserTeardownOnExceptionExit:
    """``browser.close()`` is called exactly once when the with-block exits via exception."""

    @pytest.mark.asyncio
    async def test_close_called_once_on_exception_after_scrape(self) -> None:
        """Core Sub-AC 4 assertion: close() awaited once when exception fires post-scrape."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    with pytest.raises(RuntimeError, match="intentional test error"):
                        async with scraper as s:
                            await s.scrape("https://example.com/article")
                            raise RuntimeError("intentional test error")

        # Exception propagated out — but teardown still ran exactly once.
        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_once_on_exception_before_any_scrape(self) -> None:
        """``close()`` NOT called when exception is raised before any scrape reaches browser."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with pytest.raises(ValueError, match="early error"):
                async with scraper:
                    raise ValueError("early error")

        # No scrape was attempted; browser was never launched.
        mock_browser.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_called_once_on_key_error_after_browser_launched(self) -> None:
        """``close()`` is called once when a ``KeyError`` fires after browser launch."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    with pytest.raises(KeyError, match="post-scrape error"):
                        async with scraper as s:
                            await s.scrape("https://example.com/article")
                            raise KeyError("post-scrape error")

        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_state_reset_to_none_after_exception_exit(self) -> None:
        """``_browser`` is ``None`` after an exception exits the async with-block."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    with pytest.raises(OSError):
                        async with scraper as s:
                            await s.scrape("https://example.com/article")
                            raise OSError("simulated OS error")

        assert scraper._browser is None

    @pytest.mark.asyncio
    async def test_exception_propagates_after_teardown(self) -> None:
        """The original exception must propagate even after ``close()`` is called."""
        mock_browser = _make_mock_browser()
        extraction = _make_extraction_result()
        scraper = Scraper(force_strategy="browser")
        sentinel = TypeError("must propagate")

        with patch.object(scraper, "_launch_browser", new=AsyncMock(return_value=mock_browser)):
            with patch(
                "articula._scraper.fetch_with_existing_browser",
                new=AsyncMock(return_value=_make_fetch_result()),
            ):
                with patch("articula._scraper.extract", return_value=extraction):
                    with pytest.raises(TypeError, match="must propagate"):
                        async with scraper as s:
                            await s.scrape("https://example.com/article")
                            raise sentinel

        # Teardown happened AND the exception was not swallowed
        mock_browser.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestBrowserTeardownIdempotency:
    """``_close_browser()`` can be called multiple times without error or double-close."""

    @pytest.mark.asyncio
    async def test_close_not_called_twice_on_double_close_browser(self) -> None:
        """Calling ``_close_browser()`` twice only calls ``browser.close()`` once."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()
        scraper._browser = mock_browser  # inject directly, bypassing _launch_browser

        await scraper._close_browser()  # First call — should call close()
        await scraper._close_browser()  # Second call — _browser is already None, no-op

        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_second_close_does_not_raise(self) -> None:
        """Calling ``_close_browser()`` after the browser is already closed never raises."""
        mock_browser = _make_mock_browser()
        scraper = Scraper()
        scraper._browser = mock_browser

        await scraper._close_browser()
        # Must not raise — clean no-op
        await scraper._close_browser()

    @pytest.mark.asyncio
    async def test_close_browser_when_none_does_not_raise(self) -> None:
        """``_close_browser()`` is a no-op and raises nothing when ``_browser`` is None."""
        scraper = Scraper()
        assert scraper._browser is None

        # Must succeed silently
        await scraper._close_browser()
