"""
Sub-AC 9c-1: The header/UA rotation strategy iterates through all configured
rotation variants in order.

Verified by a unit test that mocks HTTP responses to always return a
non-success status and asserts each variant's User-Agent was used exactly
once during the sequence.

Test matrix
-----------
* All responses return 403 → all UA variants tried exactly once in order
* All responses return 503 → all UA variants tried exactly once in order
* 404 response            → loop short-circuits after the first variant
* Second variant returns 200 → only two requests made; correct result returned
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from articula._fetcher import _USER_AGENTS, fetch_rotation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article"


def _mock_response(status_code: int, body: str = "") -> MagicMock:
    """Build a minimal mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = body or f"<html><body>status {status_code}</body></html>"
    resp.url = _TEST_URL
    resp.headers = {}
    return resp


def _patched_client(mock_get: AsyncMock) -> MagicMock:
    """Return a mock AsyncClient context manager whose `.get` is *mock_get*."""
    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ---------------------------------------------------------------------------
# Core AC: all variants tried in order when every response is non-success
# ---------------------------------------------------------------------------


class TestRotationIteratesAllVariantsInOrder:
    """fetch_rotation tries every _USER_AGENTS entry exactly once, in order."""

    @pytest.mark.asyncio
    async def test_all_ua_variants_tried_in_order_on_403(self) -> None:
        """When every response is 403, each UA variant is used exactly once in order."""
        mock_get = AsyncMock(return_value=_mock_response(403))
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await fetch_rotation(_TEST_URL)

        # All variants exhausted → exhaustion sentinel
        assert result is None, (
            "fetch_rotation must return None when all UA variants receive 403"
        )

        # One request per UA variant — no more, no fewer
        assert mock_get.call_count == len(_USER_AGENTS), (
            f"Expected {len(_USER_AGENTS)} GET requests (one per UA variant), "
            f"got {mock_get.call_count}"
        )

        # Variants must appear in the exact order defined in _USER_AGENTS
        actual_uas = [
            call_args.kwargs.get("headers", {}).get("User-Agent")
            for call_args in mock_get.call_args_list
        ]
        assert actual_uas == list(_USER_AGENTS), (
            "UA variants were not tried in _USER_AGENTS order.\n"
            f"Expected: {list(_USER_AGENTS)}\n"
            f"Actual:   {actual_uas}"
        )

    @pytest.mark.asyncio
    async def test_all_ua_variants_tried_in_order_on_503(self) -> None:
        """When every response is 503, each UA variant is still tried exactly once."""
        mock_get = AsyncMock(return_value=_mock_response(503))
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await fetch_rotation(_TEST_URL)

        assert result is None
        assert mock_get.call_count == len(_USER_AGENTS), (
            f"Expected {len(_USER_AGENTS)} requests on 503, got {mock_get.call_count}"
        )

        actual_uas = [
            call_args.kwargs.get("headers", {}).get("User-Agent")
            for call_args in mock_get.call_args_list
        ]
        assert actual_uas == list(_USER_AGENTS)

    @pytest.mark.asyncio
    async def test_each_variant_request_carries_correct_ua_header(self) -> None:
        """Each iteration sends the corresponding UA string in the User-Agent header."""
        mock_get = AsyncMock(return_value=_mock_response(403))
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await fetch_rotation(_TEST_URL)

        for i, expected_ua in enumerate(_USER_AGENTS):
            call_args = mock_get.call_args_list[i]
            sent_ua = call_args.kwargs.get("headers", {}).get("User-Agent")
            assert sent_ua == expected_ua, (
                f"Variant {i}: expected UA {expected_ua!r}, got {sent_ua!r}"
            )


# ---------------------------------------------------------------------------
# Hard-fail short-circuit: 404/401 must abort without trying remaining variants
# ---------------------------------------------------------------------------


class TestHardFailAbortsVariantLoop:
    """A permanent HTTP error must stop the rotation immediately."""

    @pytest.mark.asyncio
    async def test_404_short_circuits_after_first_variant(self) -> None:
        """404 is a permanent failure — only the first UA variant is tried."""
        mock_get = AsyncMock(return_value=_mock_response(404))
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await fetch_rotation(_TEST_URL)

        assert result is None
        assert mock_get.call_count == 1, (
            f"404 must abort after 1 request, got {mock_get.call_count}"
        )

    @pytest.mark.asyncio
    async def test_401_short_circuits_after_first_variant(self) -> None:
        """401 is a permanent failure — only the first UA variant is tried."""
        mock_get = AsyncMock(return_value=_mock_response(401))
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await fetch_rotation(_TEST_URL)

        assert result is None
        assert mock_get.call_count == 1, (
            f"401 must abort after 1 request, got {mock_get.call_count}"
        )


# ---------------------------------------------------------------------------
# Early success: stops as soon as one variant succeeds
# ---------------------------------------------------------------------------


class TestRotationStopsOnFirstSuccess:
    """fetch_rotation returns immediately when a UA variant gets a 200 response."""

    @pytest.mark.asyncio
    async def test_second_variant_success_makes_two_requests(self) -> None:
        """First variant → 403, second variant → 200: exactly two GET calls are made."""
        article_html = (
            "<html><head><title>Article</title></head>"
            "<body><p>Substantive article content here.</p></body></html>"
        )
        mock_get = AsyncMock(
            side_effect=[_mock_response(403), _mock_response(200, body=article_html)]
        )
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None, (
            "fetch_rotation must return FetchResult when a UA variant succeeds"
        )
        assert result.strategy_tier == "headers_rotation"
        # Only two variants were tried before success
        assert mock_get.call_count == 2, (
            f"Expected 2 requests (first fails, second succeeds), "
            f"got {mock_get.call_count}"
        )
        # First request used first UA, second request used second UA
        first_ua = mock_get.call_args_list[0].kwargs["headers"]["User-Agent"]
        second_ua = mock_get.call_args_list[1].kwargs["headers"]["User-Agent"]
        assert first_ua == _USER_AGENTS[0]
        assert second_ua == _USER_AGENTS[1]

    @pytest.mark.asyncio
    async def test_first_variant_success_makes_one_request(self) -> None:
        """When the first UA variant succeeds immediately, only one request is made."""
        article_html = (
            "<html><head><title>Article</title></head>"
            "<body><p>Article body content that is real and useful.</p></body></html>"
        )
        mock_get = AsyncMock(return_value=_mock_response(200, body=article_html))
        mock_client = _patched_client(mock_get)

        with patch(
            "articula._fetcher.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await fetch_rotation(_TEST_URL)

        assert result is not None
        assert result.strategy_tier == "headers_rotation"
        assert mock_get.call_count == 1, (
            f"Only one UA variant should have been tried, got {mock_get.call_count}"
        )
        used_ua = mock_get.call_args_list[0].kwargs["headers"]["User-Agent"]
        assert used_ua == _USER_AGENTS[0]
