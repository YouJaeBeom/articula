"""
Sub-AC 9c-3: The header/UA rotation strategy returns successfully without
raising FetchError when any single rotation variant receives a success response.

Verified by unit tests that mock one rotation variant to succeed and assert:
1. No FetchError is raised.
2. The response (FetchResult / Article) is returned (not None).

Design overview
---------------
``fetch_rotation`` iterates every User-Agent variant in ``_USER_AGENTS``.
When **any** single variant receives a 200-OK response without a bot-challenge
body, the function returns a ``FetchResult`` immediately — no exception is
raised and the loop stops.

At the ``Scraper.scrape`` level, if ``fetch_rotation`` returns a non-None
``FetchResult``, the scraper proceeds to extraction and returns an ``Article``,
never raising ``FetchError``.

Test matrix
-----------
Class TestFetchRotationReturnsOnSingleSuccess (fetch_rotation level)
  * First variant (index 0) succeeds      → FetchResult returned, strategy_tier correct
  * Middle variant (index 1) succeeds     → FetchResult returned after first variant fails
  * Last variant (index N-1) succeeds     → FetchResult returned after all prior variants fail
  * Returned FetchResult is not None      → explicit None-check guard
  * strategy_tier is "headers_rotation"   → tier label is correct

Class TestScraperNoFetchErrorOnSingleVariantSuccess (Scraper.scrape level)
  * First variant succeeds                → no FetchError raised
  * Any single variant succeeds           → no FetchError raised, article is returned
  * FetchError is NOT raised when ≥1 variant succeeds (contrast with exhaustion case)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from articula._extractor import ExtractionResult
from articula._fetcher import _USER_AGENTS, fetch_rotation
from articula._scraper import Scraper
from articula.exceptions import FetchError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://rotation-success.example.test/article"

# Patch targets
_PATCH_ASYNC_CLIENT = "articula._fetcher.httpx.AsyncClient"
_PATCH_EXTRACT = "articula._scraper.extract"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_response(body: str = "") -> MagicMock:
    """Return a minimal mock HTTP 200 response with a non-bot-challenge body."""
    if not body:
        body = (
            "<html><head><title>Article Title</title></head>"
            "<body><p>Article body with sufficient content for extraction.</p></body></html>"
        )
    resp = MagicMock()
    resp.status_code = 200
    resp.text = body
    resp.url = _TEST_URL
    resp.headers = {}
    return resp


def _fail_response(status_code: int = 403) -> MagicMock:
    """Return a minimal mock HTTP non-success response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"<html><body>status {status_code}</body></html>"
    resp.url = _TEST_URL
    resp.headers = {}
    return resp


def _patched_client(side_effects: list) -> MagicMock:
    """Return a mock AsyncClient context manager with the given side_effects for .get()."""
    mock_get = AsyncMock(side_effect=side_effects)
    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _stub_extraction_result() -> ExtractionResult:
    """Return a valid ExtractionResult for mocking the extraction step."""
    return ExtractionResult(
        title="Article Title About Technology",
        text=(
            "This article discusses the latest developments in technology "
            "and how they impact everyday life. The content is extensive and "
            "provides meaningful information for readers."
        ),
        author="Test Author",
        published_date="2024-01-15",
        method="trafilatura",
        confidence=0.85,
        language="en",
    )


# ---------------------------------------------------------------------------
# Class 1 — fetch_rotation level: FetchResult returned on single success
# ---------------------------------------------------------------------------


class TestFetchRotationReturnsOnSingleSuccess:
    """
    ``fetch_rotation`` returns a non-None ``FetchResult`` as soon as any
    single UA variant receives a 200-OK response.  No exception is raised.
    """

    @pytest.mark.asyncio
    async def test_first_variant_success_returns_fetch_result(self) -> None:
        """
        When the very first UA variant (index 0) receives HTTP 200, ``fetch_rotation``
        returns a FetchResult immediately — no exception is raised, result is not None.
        """
        mock_client = _patched_client([_success_response()])

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None, (
            "fetch_rotation must return a FetchResult (not None) "
            "when the first variant receives HTTP 200"
        )

    @pytest.mark.asyncio
    async def test_first_variant_success_strategy_tier_is_headers_rotation(
        self,
    ) -> None:
        """
        The returned FetchResult must have ``strategy_tier="headers_rotation"``
        to correctly identify which strategy tier succeeded.
        """
        mock_client = _patched_client([_success_response()])

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None
        assert result.strategy_tier == "headers_rotation", (
            f"Expected strategy_tier='headers_rotation', got {result.strategy_tier!r}"
        )

    @pytest.mark.asyncio
    async def test_middle_variant_success_returns_fetch_result(self) -> None:
        """
        When the second UA variant (index 1) receives HTTP 200 after the first
        returns 403, ``fetch_rotation`` returns a FetchResult — no exception raised.

        This confirms that success from any variant (not just the first) is enough
        to return successfully.
        """
        # First variant fails with 403, second succeeds with 200
        side_effects = [_fail_response(403), _success_response()]
        mock_client = _patched_client(side_effects)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None, (
            "fetch_rotation must return a FetchResult when the second variant "
            "receives HTTP 200, even though the first variant received 403"
        )
        assert result.strategy_tier == "headers_rotation"

    @pytest.mark.asyncio
    async def test_last_variant_success_returns_fetch_result(self) -> None:
        """
        When only the last UA variant (index N-1) succeeds with HTTP 200, all
        prior variants having failed, ``fetch_rotation`` still returns a FetchResult.

        This validates that the loop exhausts failing variants before returning
        the single success — and does NOT raise FetchError prematurely.
        """
        n = len(_USER_AGENTS)
        # All prior variants fail with 403, last one succeeds
        side_effects = [_fail_response(403)] * (n - 1) + [_success_response()]
        mock_client = _patched_client(side_effects)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None, (
            f"fetch_rotation must return a FetchResult when the last of {n} "
            "variants (index N-1) receives HTTP 200"
        )
        assert result.strategy_tier == "headers_rotation"

    @pytest.mark.asyncio
    async def test_single_success_response_is_returned(self) -> None:
        """
        The ``FetchResult.html`` from a successful 200 response is preserved
        in the returned result — confirming the actual response content is passed
        through, not a placeholder.
        """
        article_html = (
            "<html><head><title>Real Article</title></head>"
            "<body><article><p>Meaningful article content.</p></article></body></html>"
        )
        mock_client = _patched_client([_success_response(body=article_html)])

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None
        assert result.html == article_html, (
            "FetchResult.html must contain the HTML from the successful response"
        )


# ---------------------------------------------------------------------------
# Class 2 — Scraper.scrape level: no FetchError raised on single variant success
# ---------------------------------------------------------------------------


class TestScraperNoFetchErrorOnSingleVariantSuccess:
    """
    When ``fetch_rotation`` returns a non-None ``FetchResult`` (i.e. at least
    one UA variant succeeded), ``Scraper.scrape`` must NOT raise ``FetchError``.

    These tests use ``force_strategy="headers_rotation"`` so that rotation is
    the sole tier being exercised, isolating the rotation-success path.

    The extraction step is mocked to avoid dependence on the extraction tier
    (ExtractionError would be a different, unrelated failure mode).
    """

    @pytest.mark.asyncio
    async def test_first_variant_success_no_fetch_error_raised(self) -> None:
        """
        When the first UA variant receives HTTP 200, ``Scraper.scrape`` completes
        without raising ``FetchError``.
        """
        mock_client = _patched_client([_success_response()])

        with (
            patch(_PATCH_ASYNC_CLIENT, return_value=mock_client),
            patch(_PATCH_EXTRACT, return_value=_stub_extraction_result()),
        ):
            # This must NOT raise FetchError
            try:
                async with Scraper(force_strategy="headers_rotation") as s:
                    article = await s.scrape(_TEST_URL)
            except FetchError as exc:
                pytest.fail(
                    f"FetchError must NOT be raised when the first UA variant "
                    f"succeeds, but got: {exc}"
                )

        assert article is not None, "Scraper.scrape must return an Article on success"

    @pytest.mark.asyncio
    async def test_first_variant_success_returns_article(self) -> None:
        """
        When the first UA variant succeeds, ``Scraper.scrape`` returns an Article
        (not None, not an exception).
        """
        mock_client = _patched_client([_success_response()])

        with (
            patch(_PATCH_ASYNC_CLIENT, return_value=mock_client),
            patch(_PATCH_EXTRACT, return_value=_stub_extraction_result()),
        ):
            async with Scraper(force_strategy="headers_rotation") as s:
                article = await s.scrape(_TEST_URL)

        assert article is not None, "Article must not be None when rotation succeeds"
        assert article.strategy_tier == "headers_rotation", (
            f"Article.strategy_tier must be 'headers_rotation', "
            f"got {article.strategy_tier!r}"
        )

    @pytest.mark.asyncio
    async def test_middle_variant_success_no_fetch_error_raised(self) -> None:
        """
        When only the second UA variant succeeds (first fails with 503),
        ``Scraper.scrape`` must NOT raise ``FetchError``.

        This verifies that the scraper correctly treats a single variant's
        200-OK as overall rotation success.
        """
        side_effects = [_fail_response(503), _success_response()]
        mock_client = _patched_client(side_effects)

        with (
            patch(_PATCH_ASYNC_CLIENT, return_value=mock_client),
            patch(_PATCH_EXTRACT, return_value=_stub_extraction_result()),
        ):
            try:
                async with Scraper(force_strategy="headers_rotation") as s:
                    article = await s.scrape(_TEST_URL)
            except FetchError as exc:
                pytest.fail(
                    f"FetchError must NOT be raised when the second UA variant "
                    f"succeeds after the first received 503, but got: {exc}"
                )

        assert article is not None

    @pytest.mark.asyncio
    async def test_any_single_success_is_sufficient_to_avoid_fetch_error(
        self,
    ) -> None:
        """
        With N-1 variants failing and 1 succeeding, ``Scraper.scrape`` must
        return an Article without raising ``FetchError``.

        This is the direct contrast to the exhaustion case (test_rotation_exhaustion_fetch_error.py):
        where ALL variants fail → FetchError; where ANY variant succeeds → no error.
        """
        n = len(_USER_AGENTS)
        side_effects = [_fail_response(403)] * (n - 1) + [_success_response()]
        mock_client = _patched_client(side_effects)

        with (
            patch(_PATCH_ASYNC_CLIENT, return_value=mock_client),
            patch(_PATCH_EXTRACT, return_value=_stub_extraction_result()),
        ):
            try:
                async with Scraper(force_strategy="headers_rotation") as s:
                    article = await s.scrape(_TEST_URL)
            except FetchError as exc:
                pytest.fail(
                    f"FetchError must NOT be raised when at least one of {n} "
                    f"UA variants succeeds, but got: {exc}"
                )

        assert article is not None, (
            "Scraper.scrape must return an Article when any single variant succeeds"
        )
