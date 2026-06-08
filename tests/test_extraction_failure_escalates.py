"""Regression: extraction failure on an earlier tier must escalate, not abort.

This guards the library's headline capability — JS-rendered pages and
bot-challenge interstitials where the *static* HTML is fetched successfully
(HTTP 200, real bytes) but contains no extractable article body.  The fetch
"succeeds" yet ``extract()`` raises ``ValueError``.  The escalation
coordinator (``Scraper.scrape``) must treat that as a non-win and advance to
the next tier — ultimately the Playwright browser tier, which renders JS —
rather than immediately raising ``ExtractionError`` after the first tier that
returned bytes.

Historically the loop broke on the first tier that returned *any* HTML and
only attempted extraction once, after the loop.  That defeated the entire
purpose of the browser tier for JS-shell pages.  These tests pin the fixed
behaviour: escalate on extraction failure, and only raise ``ExtractionError``
once every tier has been exhausted without a usable body.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula import async_scrape
from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult
from articula._scraper import Scraper
from articula.exceptions import ExtractionError

_URL = "https://example.com/spa-article"

_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER = "articula._scraper.fetch_with_existing_browser"
_PATCH_LAUNCH = "articula._scraper.Scraper._launch_browser"
_PATCH_EXTRACT = "articula._scraper.extract"


def _fetch(tier: str) -> FetchResult:
    return FetchResult(
        html=f"<html><body>{tier}</body></html>",
        resolved_url=_URL,
        strategy_tier=tier,
        status_code=200,
    )


_GOOD = ExtractionResult(
    title="Recovered After JS Render",
    text=(
        "This article body only exists after the headless browser rendered the "
        "page's JavaScript, which the earlier static and rotation tiers could "
        "not do on their own."
    ),
    author="Jane Researcher",
    published_date="2026-05-01",
    method="trafilatura",
    confidence=0.9,
    language="en",
)


def _extract_only_browser_html(html: str, _url: str) -> ExtractionResult:
    """Real-ish extract stub: only the browser tier's HTML yields a body."""
    if "browser" in html:
        return _GOOD
    raise ValueError("no extractable body in static/rotation HTML")


class TestExtractionFailureEscalation:
    @pytest.mark.asyncio
    async def test_escalates_to_browser_when_static_and_rotation_unextractable(self) -> None:
        """All three tiers fetch 200 OK, but only browser HTML extracts a body."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as rotation,
            patch(_PATCH_LAUNCH, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as browser,
            patch(_PATCH_EXTRACT, side_effect=_extract_only_browser_html),
        ):
            static.return_value = _fetch("static")
            rotation.return_value = _fetch("headers_rotation")
            browser.return_value = _fetch("browser")

            article = await async_scrape(_URL)

        assert article.strategy_tier == "browser"
        assert article.strategies_attempted == ("static", "headers_rotation", "browser")
        assert "headless browser" in article.text
        # Every tier had to be tried because extraction failed on the first two.
        static.assert_awaited_once()
        rotation.assert_awaited_once()
        browser.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_first_tier_with_extractable_body_short_circuits(self) -> None:
        """Fast path preserved: a usable static body returns without the browser."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as rotation,
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as browser,
            patch(_PATCH_EXTRACT, return_value=_GOOD),
        ):
            static.return_value = _fetch("static")

            article = await async_scrape(_URL)

        assert article.strategy_tier == "static"
        assert article.strategies_attempted == ("static",)
        rotation.assert_not_awaited()
        browser.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_extraction_error_when_no_tier_yields_body(self) -> None:
        """All tiers fetch 200 but none extracts → ExtractionError listing all tiers."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as rotation,
            patch(_PATCH_LAUNCH, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as browser,
            patch(_PATCH_EXTRACT, side_effect=ValueError("empty")),
        ):
            static.return_value = _fetch("static")
            rotation.return_value = _fetch("headers_rotation")
            browser.return_value = _fetch("browser")

            with pytest.raises(ExtractionError) as exc_info:
                async with Scraper() as scraper:
                    await scraper.scrape(_URL)

        assert exc_info.value.url == _URL
        assert list(exc_info.value.strategies_attempted) == [
            "static",
            "headers_rotation",
            "browser",
        ]
