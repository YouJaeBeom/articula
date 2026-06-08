"""
Sub-AC 7b-1: scrape(url) sets strategy_tier on the returned Article to reflect
which tier (static / headers_rotation / browser) resolved the page.

The fetch pipeline is fully mocked so each tier mock fires in sequence and
the test asserts ``Article.strategy_tier`` equals the expected tier string.

Test matrix
-----------
* static tier succeeds                    → strategy_tier == "static"
* static fails, rotation succeeds         → strategy_tier == "headers_rotation"
* static + rotation fail, browser succeeds→ strategy_tier == "browser"

Additional coverage
-------------------
* ``strategies_attempted`` records every tier that was tried before success.
* ``force_strategy`` pins execution to a single tier.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula import async_scrape
from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult

# ---------------------------------------------------------------------------
# Fixtures / shared constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article"

_MOCK_EXTRACTION = ExtractionResult(
    title="Test Article Title",
    text=(
        "This is the article body text. "
        "It is long enough to satisfy the minimum character threshold "
        "for a successful extraction result."
    ),
    author="Test Author",
    published_date="2026-06-07",
    method="trafilatura",
    confidence=0.90,
    language="en",
)


def _fetch_result(tier: str) -> FetchResult:
    """Build a minimal ``FetchResult`` for *tier*."""
    return FetchResult(
        html="<html><body>stub</body></html>",
        resolved_url=_TEST_URL,
        strategy_tier=tier,
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Helpers for patching
# ---------------------------------------------------------------------------

# The Scraper imports fetch_static and fetch_rotation at the top of _scraper.py,
# so we patch the names *in that module's namespace*.
# The browser tier uses fetch_with_existing_browser (via a lazy-initialised
# Playwright browser), so we patch that function AND _launch_browser to prevent
# a real Playwright process from being started in unit tests.
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER = "articula._scraper.fetch_with_existing_browser"
_PATCH_LAUNCH_BROWSER = "articula._scraper.Scraper._launch_browser"
_PATCH_EXTRACT = "articula._scraper.extract"


# ---------------------------------------------------------------------------
# Sub-AC 7b-1 — strategy_tier reflects the successful tier
# ---------------------------------------------------------------------------


class TestStrategyTierIsSet:
    """Each tier mock fires in sequence; Article.strategy_tier is asserted."""

    @pytest.mark.asyncio
    async def test_static_tier_sets_strategy_tier(self) -> None:
        """When static fetch succeeds, Article.strategy_tier == 'static'."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = _fetch_result("static")

            article = await async_scrape(_TEST_URL)

            assert article.strategy_tier == "static"
            mock_static.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_headers_rotation_tier_sets_strategy_tier(self) -> None:
        """Static fails → rotation succeeds → strategy_tier == 'headers_rotation'."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None          # tier 1 fails
            mock_rotation.return_value = _fetch_result("headers_rotation")

            article = await async_scrape(_TEST_URL)

            assert article.strategy_tier == "headers_rotation"
            mock_static.assert_awaited_once()
            mock_rotation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_tier_sets_strategy_tier(self) -> None:
        """Static + rotation fail → browser succeeds → strategy_tier == 'browser'."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_LAUNCH_BROWSER, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = _fetch_result("browser")

            article = await async_scrape(_TEST_URL)

            assert article.strategy_tier == "browser"
            mock_static.assert_awaited_once()
            mock_rotation.assert_awaited_once()
            mock_browser.assert_awaited_once()


# ---------------------------------------------------------------------------
# Additional assertions: strategies_attempted records the full attempt trail
# ---------------------------------------------------------------------------


class TestStrategiesAttempted:
    """strategies_attempted is populated correctly for each escalation path."""

    @pytest.mark.asyncio
    async def test_static_success_records_only_static(self) -> None:
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = _fetch_result("static")
            article = await async_scrape(_TEST_URL)

        assert article.strategies_attempted == ("static",)

    @pytest.mark.asyncio
    async def test_rotation_success_records_static_and_rotation(self) -> None:
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = _fetch_result("headers_rotation")
            article = await async_scrape(_TEST_URL)

        assert article.strategies_attempted == ("static", "headers_rotation")

    @pytest.mark.asyncio
    async def test_browser_success_records_all_three_tiers(self) -> None:
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_LAUNCH_BROWSER, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = _fetch_result("browser")
            article = await async_scrape(_TEST_URL)

        assert article.strategies_attempted == ("static", "headers_rotation", "browser")


# ---------------------------------------------------------------------------
# force_strategy pins execution to a single tier
# ---------------------------------------------------------------------------


class TestForceStrategy:
    """force_strategy bypasses the escalation pipeline."""

    @pytest.mark.asyncio
    async def test_force_static_skips_other_tiers(self) -> None:
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = _fetch_result("static")

            article = await async_scrape(_TEST_URL, force_strategy="static")

            assert article.strategy_tier == "static"
            mock_static.assert_awaited_once()
            mock_rotation.assert_not_called()
            mock_browser.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_browser_skips_http_tiers(self) -> None:
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_LAUNCH_BROWSER, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_browser.return_value = _fetch_result("browser")

            article = await async_scrape(_TEST_URL, force_strategy="browser")

            assert article.strategy_tier == "browser"
            mock_static.assert_not_called()
            mock_rotation.assert_not_called()
            mock_browser.assert_awaited_once()
