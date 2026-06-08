"""
Sub-AC 2: Static fetch strategy signals exhaustion by returning ``None`` as a
sentinel that the escalation coordinator can detect.

Design overview
---------------
``fetch_static`` returns ``None`` in every failure mode:
  - Network error after all retries are exhausted
  - Timeout error after all retries are exhausted
  - Hard HTTP failure (404 / 401) — immediate return, no retry
  - Bot-protected page (403 / 429 / 503)
  - Bot-challenge HTML (Cloudflare, DDoS-Guard, …)

The escalation coordinator (``Scraper.scrape``) treats ``None`` as the
exhaustion sentinel: it records the tier in ``strategies_attempted`` and
moves on to the next tier.  When every tier returns ``None``, it raises
``FetchError`` with the full list of ``attempted_strategies``.

Test matrix
-----------
Class TestStaticFetchSignalsExhaustion
  * network error   → None
  * timeout error   → None
  * HTTP 404        → None (no retry within tier)
  * HTTP 401        → None (no retry within tier)
  * HTTP 403        → None (bot code)
  * HTTP 503        → None (after retries)
  * bot-challenge HTML (200) → None

Class TestCoordinatorDetectsStaticExhaustion
  * static=None, rotation=None, browser=None → FetchError raised
  * FetchError.attempted_strategies contains all three tiers
  * FetchError.url matches the input URL
  * static=None triggers escalation to rotation (rotation succeeds)
  * static=None is recorded in Article.strategies_attempted on success
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult, fetch_static
from articula._scraper import Scraper
from articula.exceptions import FetchError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PERMANENTLY_INACCESSIBLE_URL = "https://permanently-inaccessible.example.invalid/"

# Patch targets (fetch functions are imported into _scraper module namespace)
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER = "articula._scraper.fetch_browser"
_PATCH_EXTRACT = "articula._scraper.extract"

# Patch target for the internal retry helper inside _fetcher
_PATCH_GET_WITH_RETRY = "articula._fetcher._get_with_retry"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _mock_http_response(status_code: int) -> MagicMock:
    """Build a minimal mock httpx.Response for a given status code."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {}
    resp.url = httpx.URL(_PERMANENTLY_INACCESSIBLE_URL)
    resp.text = ""
    return resp


_MOCK_EXTRACTION = ExtractionResult(
    title="Escalation Test Article",
    text=(
        "Body text that is long enough to satisfy the minimum character "
        "threshold required by the extraction result validator."
    ),
    author=None,
    published_date=None,
    method="trafilatura",
    confidence=0.80,
    language="en",
)


def _fetch_result(tier: str) -> FetchResult:
    return FetchResult(
        html="<html><body>article content</body></html>",
        resolved_url=_PERMANENTLY_INACCESSIBLE_URL,
        strategy_tier=tier,
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Class 1 — fetch_static returns None (the exhaustion sentinel) for failures
# ---------------------------------------------------------------------------


class TestStaticFetchSignalsExhaustion:
    """fetch_static returns None for every permanent-failure scenario."""

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self) -> None:
        """
        When all retries raise NetworkError (permanently inaccessible host),
        fetch_static must return None.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            # Simulate _get_with_retry exhausting all retries then re-raising.
            mock_retry.side_effect = httpx.NetworkError(
                "Name or service not known (permanently inaccessible)"
            )
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL, max_retries=3)

        assert result is None, (
            "fetch_static must return None (exhaustion sentinel) "
            "when network is permanently unreachable"
        )

    @pytest.mark.asyncio
    async def test_timeout_error_returns_none(self) -> None:
        """
        When all retries raise TimeoutException, fetch_static must return None.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.TimeoutException(
                "Request timed out after 30 s"
            )
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL, max_retries=3)

        assert result is None, (
            "fetch_static must return None when every retry times out"
        )

    @pytest.mark.asyncio
    async def test_http_404_returns_none(self) -> None:
        """
        HTTP 404 is a hard failure — fetch_static returns None immediately
        without any retry within the tier.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _mock_http_response(404)
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL)

        assert result is None, "fetch_static must return None on HTTP 404"

    @pytest.mark.asyncio
    async def test_http_401_returns_none(self) -> None:
        """
        HTTP 401 is a hard failure — fetch_static returns None immediately.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _mock_http_response(401)
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL)

        assert result is None, "fetch_static must return None on HTTP 401"

    @pytest.mark.asyncio
    async def test_http_403_bot_protection_returns_none(self) -> None:
        """
        HTTP 403 signals bot detection — fetch_static returns None.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _mock_http_response(403)
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL)

        assert result is None, "fetch_static must return None on HTTP 403"

    @pytest.mark.asyncio
    async def test_http_503_returns_none(self) -> None:
        """
        HTTP 503 signals temporary unavailability — fetch_static returns None
        after exhausting retries.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = _mock_http_response(503)
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL)

        assert result is None, "fetch_static must return None on HTTP 503"

    @pytest.mark.asyncio
    async def test_bot_challenge_html_returns_none(self) -> None:
        """
        A 200 OK response whose HTML body contains a bot-challenge page
        (Cloudflare, DDoS-Guard, …) must return None.
        """
        bot_challenge_html = (
            "<html><body>"
            "<title>Just a moment...</title>"
            "<p>Checking your browser before accessing the site.</p>"
            "</body></html>"
        )
        bot_resp = MagicMock(spec=httpx.Response)
        bot_resp.status_code = 200
        bot_resp.headers = {}
        bot_resp.url = httpx.URL(_PERMANENTLY_INACCESSIBLE_URL)
        bot_resp.text = bot_challenge_html

        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = bot_resp
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL)

        assert result is None, (
            "fetch_static must return None when the 200-OK body is a bot challenge"
        )

    @pytest.mark.asyncio
    async def test_sentinel_type_is_none(self) -> None:
        """
        Explicit type check: the exhaustion sentinel is exactly None,
        not False, 0, or an empty container.
        """
        with patch(_PATCH_GET_WITH_RETRY, new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.NetworkError("unreachable")
            result = await fetch_static(_PERMANENTLY_INACCESSIBLE_URL)

        assert result is None
        assert not isinstance(result, FetchResult)


# ---------------------------------------------------------------------------
# Class 2 — escalation coordinator detects the None sentinel correctly
# ---------------------------------------------------------------------------


class TestCoordinatorDetectsStaticExhaustion:
    """Scraper.scrape detects None from fetch_static and escalates / fails."""

    @pytest.mark.asyncio
    async def test_fetch_error_raised_when_all_tiers_return_none(self) -> None:
        """
        When every tier (static → rotation → browser) returns None,
        the coordinator must raise FetchError — not return None or hang.
        """
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
                    await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        exc = exc_info.value
        assert isinstance(exc, FetchError), "coordinator must raise FetchError"

    @pytest.mark.asyncio
    async def test_fetch_error_url_matches_input(self) -> None:
        """FetchError.url must equal the URL that was scraped."""
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
                    await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        assert exc_info.value.url == _PERMANENTLY_INACCESSIBLE_URL

    @pytest.mark.asyncio
    async def test_fetch_error_includes_static_in_attempted_strategies(self) -> None:
        """
        FetchError.attempted_strategies must include 'static' — confirming
        the static tier was attempted before the error was raised.
        """
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
                    await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        assert "static" in exc_info.value.attempted_strategies

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_ordered(self) -> None:
        """
        FetchError.attempted_strategies must list tiers in escalation order:
        ('static', 'headers_rotation', 'browser').
        """
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
                    await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        assert exc_info.value.attempted_strategies == (
            "static",
            "headers_rotation",
            "browser",
        )

    @pytest.mark.asyncio
    async def test_static_none_triggers_escalation_to_rotation(self) -> None:
        """
        When static returns the None sentinel, the coordinator must try
        headers_rotation next rather than giving up immediately.
        """
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None  # exhaustion sentinel
            mock_rotation.return_value = _fetch_result("headers_rotation")

            async with Scraper() as s:
                article = await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        mock_static.assert_awaited_once()
        mock_rotation.assert_awaited_once()
        assert article.strategy_tier == "headers_rotation"

    @pytest.mark.asyncio
    async def test_static_exhaustion_recorded_in_article_strategies_attempted(
        self,
    ) -> None:
        """
        Even when static fails and rotation succeeds, 'static' appears in
        Article.strategies_attempted — the coordinator records every tier tried.
        """
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = _fetch_result("headers_rotation")

            async with Scraper() as s:
                article = await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        assert "static" in article.strategies_attempted
        assert "headers_rotation" in article.strategies_attempted
        assert article.strategies_attempted == ("static", "headers_rotation")

    @pytest.mark.asyncio
    async def test_fetch_error_is_scraper_error_subclass(self) -> None:
        """
        The raised exception must be catchable as both FetchError and
        the base ScraperError for broad except clauses.
        """
        from articula.exceptions import ScraperError

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
                    await s.scrape(_PERMANENTLY_INACCESSIBLE_URL)

        assert isinstance(exc_info.value, FetchError)
