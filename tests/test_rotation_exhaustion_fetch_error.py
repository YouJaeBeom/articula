"""
Sub-AC 9c-2: The header/UA rotation strategy raises FetchError after all
rotation variants are exhausted without success.

Verified by a unit test that mocks all variant HTTP responses as failures
and asserts FetchError is raised with an appropriate message.

Design overview
---------------
``fetch_rotation`` iterates every User-Agent variant in ``_USER_AGENTS``.
When all variants receive non-success responses (403, 503, network errors),
it returns the ``None`` sentinel.  The ``Scraper`` escalation coordinator
then raises ``FetchError`` because all requested strategies have been exhausted.

When ``force_strategy="headers_rotation"`` is used, the Scraper skips the
static and browser tiers, making ``fetch_rotation`` the sole strategy.  If
rotation exhausts all UA variants without success, ``FetchError`` is raised
immediately with ``attempted_strategies=("headers_rotation",)``.

Test matrix
-----------
Class TestRotationExhaustionRaisesFetchError
  * All 403 responses             → FetchError raised
  * All 503 responses             → FetchError raised
  * All network errors            → FetchError raised
  * FetchError.url matches input URL
  * FetchError.attempted_strategies contains 'headers_rotation'
  * FetchError message is non-empty and actionable
  * FetchError is catchable as ScraperError

Class TestRotationExhaustionMessageContent
  * FetchError.__str__ contains the failing URL
  * FetchError.__str__ contains rotation strategy name
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from articula._fetcher import _USER_AGENTS
from articula._scraper import Scraper
from articula.exceptions import FetchError, ScraperError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://rotation-exhaustion.example.test/article"

# Patch target for httpx.AsyncClient inside the _fetcher module.
_PATCH_ASYNC_CLIENT = "articula._fetcher.httpx.AsyncClient"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int) -> MagicMock:
    """Return a minimal mock HTTP response with the given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"<html><body>status {status_code}</body></html>"
    resp.url = _TEST_URL
    resp.headers = {}
    return resp


def _patched_client(side_effect_or_value) -> MagicMock:
    """
    Return a mock AsyncClient context manager.

    ``side_effect_or_value`` is passed to ``mock_client.get`` as either
    ``return_value`` (for a single response) or ``side_effect`` (for a list).
    """
    mock_get = AsyncMock()
    if isinstance(side_effect_or_value, list):
        mock_get.side_effect = side_effect_or_value
    else:
        mock_get.return_value = side_effect_or_value

    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ---------------------------------------------------------------------------
# Class 1 — FetchError is raised when rotation exhausts all variants
# ---------------------------------------------------------------------------


class TestRotationExhaustionRaisesFetchError:
    """
    ``Scraper.scrape`` raises ``FetchError`` when ``fetch_rotation`` has
    iterated every UA variant without receiving a usable response.

    Each test drives HTTP failures at the ``httpx.AsyncClient`` level so that
    the real ``fetch_rotation`` loop runs completely — no tier functions are
    mocked away.  ``force_strategy="headers_rotation"`` isolates the rotation
    tier as the sole strategy so the test is focused.
    """

    @pytest.mark.asyncio
    async def test_all_403_responses_raise_fetch_error(self) -> None:
        """
        When every UA variant receives HTTP 403 (bot protection), all variants
        are exhausted and FetchError is raised.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        exc = exc_info.value
        assert isinstance(exc, FetchError), (
            "Exception must be FetchError when all rotation variants receive 403"
        )

    @pytest.mark.asyncio
    async def test_all_503_responses_raise_fetch_error(self) -> None:
        """
        When every UA variant receives HTTP 503 (temporary block), all variants
        are exhausted and FetchError is raised.
        """
        responses = [_mock_response(503)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value, FetchError), (
            "Exception must be FetchError when all rotation variants receive 503"
        )

    @pytest.mark.asyncio
    async def test_all_network_errors_raise_fetch_error(self) -> None:
        """
        When every UA variant raises a network error, all variants are exhausted
        and FetchError is raised.
        """
        import httpx  # noqa: PLC0415

        mock_get = AsyncMock(
            side_effect=httpx.NetworkError("Connection refused")
        )
        mock_client = MagicMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value, FetchError), (
            "Exception must be FetchError when all rotation variants raise network errors"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_url_matches_input(self) -> None:
        """
        ``FetchError.url`` must equal the URL that was passed to ``scrape()``.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        assert exc_info.value.url == _TEST_URL, (
            f"FetchError.url must be {_TEST_URL!r}, "
            f"got {exc_info.value.url!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_contains_rotation(self) -> None:
        """
        ``FetchError.attempted_strategies`` must include ``'headers_rotation'``
        to confirm the rotation tier was attempted before raising.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        assert "headers_rotation" in exc_info.value.attempted_strategies, (
            "FetchError.attempted_strategies must include 'headers_rotation' "
            "when the rotation tier was attempted and exhausted"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_is_rotation_only(self) -> None:
        """
        With ``force_strategy="headers_rotation"``, only the rotation tier
        is attempted.  ``FetchError.attempted_strategies`` must be exactly
        ``('headers_rotation',)``.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        assert exc_info.value.attempted_strategies == ("headers_rotation",), (
            "With force_strategy='headers_rotation', only one tier is attempted; "
            f"got {exc_info.value.attempted_strategies!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_message_is_non_empty(self) -> None:
        """
        ``FetchError`` must carry an informative message — not an empty string.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        message = str(exc_info.value)
        assert message, "FetchError must carry a non-empty message"

    @pytest.mark.asyncio
    async def test_fetch_error_is_catchable_as_scraper_error(self) -> None:
        """
        ``FetchError`` IS-A ``ScraperError``, so callers using a broad
        ``except ScraperError`` clause still catch rotation exhaustion.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(ScraperError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value, FetchError), (
            "The caught ScraperError must be an instance of FetchError"
        )


# ---------------------------------------------------------------------------
# Class 2 — FetchError message content is actionable
# ---------------------------------------------------------------------------


class TestRotationExhaustionMessageContent:
    """
    The ``FetchError`` raised after rotation exhaustion must include enough
    context in its string representation for callers to diagnose the failure.
    """

    @pytest.mark.asyncio
    async def test_fetch_error_str_contains_url(self) -> None:
        """
        ``str(FetchError)`` must include the failing URL so log messages are
        immediately actionable without needing to inspect attributes.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        error_str = str(exc_info.value)
        assert _TEST_URL in error_str, (
            f"FetchError string representation must contain the URL {_TEST_URL!r}; "
            f"got: {error_str!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_str_contains_rotation_strategy_name(self) -> None:
        """
        ``str(FetchError)`` must reference ``'headers_rotation'`` so the
        caller knows which strategy tier was responsible for the failure.
        """
        responses = [_mock_response(403)] * len(_USER_AGENTS)
        mock_client = _patched_client(responses)

        with patch(_PATCH_ASYNC_CLIENT, return_value=mock_client):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper(force_strategy="headers_rotation") as s:
                    await s.scrape(_TEST_URL)

        error_str = str(exc_info.value)
        assert "headers_rotation" in error_str, (
            "FetchError string representation must mention 'headers_rotation'; "
            f"got: {error_str!r}"
        )
