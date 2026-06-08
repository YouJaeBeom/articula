"""
Sub-AC 7b: scrape(url) populates resolved_url by following redirects and
returning the final URL after all redirects.

This module contains two complementary test suites:

1. **High-level pipeline tests** (TestResolvedUrlAfterRedirect,
   TestResolvedUrlIsLastInRedirectChain) — mock the fetch functions
   (``fetch_static``, ``fetch_rotation``, ``fetch_with_existing_browser``) at
   the function boundary.  They verify that ``Article.resolved_url`` equals
   whatever ``FetchResult.resolved_url`` the winning tier returns.

2. **HTTP-level redirect chain tests** (TestFetcherHttpRedirectChain) — use a
   real ``httpx.AsyncBaseTransport`` subclass that issues genuine 301 redirect
   responses.  They verify that ``fetch_static`` / ``fetch_rotation`` call
   httpx with ``follow_redirects=True`` and store the *terminal* ``resp.url``
   in ``FetchResult.resolved_url``, not the original input URL or any
   intermediate hop.

   Test matrix
   -----------
   * 3-hop chain (A → B → C, all 301s) via fetch_static
   * 3-hop chain (A → B → C, all 301s) via fetch_rotation
   * Single-hop (A → C, one 301) via fetch_static
   * No redirect (A → 200 directly) — baseline correctness
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from articula import async_scrape
from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult, fetch_rotation, fetch_static

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The URL the caller originally requests.
_INPUT_URL = "https://short.example.com/abc"

# The final URL after the server redirects — what httpx.Response.url would
# hold after following all Location headers.
_FINAL_URL = "https://www.example.com/full/article/2026/06/07/new-model"

# Shared mock extraction result (content itself is irrelevant for this test).
_MOCK_EXTRACTION = ExtractionResult(
    title="Redirect Test Article",
    text=(
        "This article body was reached after following a redirect chain. "
        "The resolved_url on the Article must reflect the final destination, "
        "not the original short URL that was requested."
    ),
    author="Redirect Author",
    published_date="2026-06-07",
    method="trafilatura",
    confidence=0.90,
    language="en",
)

# Patch targets — names as imported/used inside _scraper.py.
# The browser tier uses fetch_with_existing_browser (via a lazy-initialised
# Playwright browser), so we patch that function AND _launch_browser to prevent
# a real Playwright process from being started in unit tests.
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER = "articula._scraper.fetch_with_existing_browser"
_PATCH_LAUNCH_BROWSER = "articula._scraper.Scraper._launch_browser"
_PATCH_EXTRACT = "articula._scraper.extract"


def _make_fetch_result(tier: str, resolved_url: str = _FINAL_URL) -> FetchResult:
    """Build a FetchResult whose resolved_url is the post-redirect destination."""
    return FetchResult(
        html="<html><body><p>Article body after redirect</p></body></html>",
        resolved_url=resolved_url,
        strategy_tier=tier,
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Sub-AC 7b-2 — resolved_url reflects the final URL after a redirect
# ---------------------------------------------------------------------------


class TestResolvedUrlAfterRedirect:
    """
    Each tier mock returns a FetchResult with resolved_url != input URL.
    The resulting Article.resolved_url must equal the FetchResult's
    resolved_url (the final redirect target), not the original input URL.
    """

    @pytest.mark.asyncio
    async def test_static_tier_resolved_url_is_final_redirect_target(self) -> None:
        """Static tier: Article.resolved_url == final URL after redirect."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = _make_fetch_result("static")

            article = await async_scrape(_INPUT_URL)

        assert article.resolved_url == _FINAL_URL, (
            f"Expected resolved_url={_FINAL_URL!r}, got {article.resolved_url!r}"
        )
        assert article.resolved_url != _INPUT_URL, (
            "resolved_url must not equal the original input URL when a redirect occurred"
        )

    @pytest.mark.asyncio
    async def test_headers_rotation_tier_resolved_url_is_final_redirect_target(
        self,
    ) -> None:
        """Rotation tier: Article.resolved_url == final URL after redirect."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None  # tier 1 fails; escalate
            mock_rotation.return_value = _make_fetch_result("headers_rotation")

            article = await async_scrape(_INPUT_URL)

        assert article.resolved_url == _FINAL_URL, (
            f"Expected resolved_url={_FINAL_URL!r}, got {article.resolved_url!r}"
        )
        assert article.resolved_url != _INPUT_URL, (
            "resolved_url must not equal the original input URL when a redirect occurred"
        )

    @pytest.mark.asyncio
    async def test_browser_tier_resolved_url_is_final_redirect_target(
        self,
    ) -> None:
        """Browser tier: Article.resolved_url == final URL after redirect."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_LAUNCH_BROWSER, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = _make_fetch_result("browser")

            article = await async_scrape(_INPUT_URL)

        assert article.resolved_url == _FINAL_URL, (
            f"Expected resolved_url={_FINAL_URL!r}, got {article.resolved_url!r}"
        )
        assert article.resolved_url != _INPUT_URL, (
            "resolved_url must not equal the original input URL when a redirect occurred"
        )


# ---------------------------------------------------------------------------
# Multi-hop redirect chain — resolved_url must be the *last* URL in the chain
# ---------------------------------------------------------------------------


class TestResolvedUrlIsLastInRedirectChain:
    """
    Simulate a multi-hop redirect chain (URL_A → URL_B → URL_C).

    httpx follows every redirect and exposes the terminal URL on
    ``response.url``, which ``_fetcher.py`` stores in ``FetchResult.resolved_url``.

    The test mocks each tier's fetch function to return a FetchResult whose
    ``resolved_url`` is already set to URL_C (the last hop), mirroring how
    the real fetcher would behave after httpx resolves all redirects.

    Assertion: ``Article.resolved_url == URL_C`` for all three tiers.
    """

    _URL_A = "https://redirect-a.example.com/hop1"   # original input
    _URL_B = "https://redirect-b.example.com/hop2"   # intermediate hop
    _URL_C = "https://final.example.com/article/2026/new-model"  # terminal

    @pytest.mark.asyncio
    async def test_static_tier_resolved_url_is_last_hop(self) -> None:
        """Static tier: resolved_url must be URL_C (last hop), not URL_A or URL_B."""
        result = _make_fetch_result("static", resolved_url=self._URL_C)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = result

            article = await async_scrape(self._URL_A)

        assert article.resolved_url == self._URL_C
        assert article.resolved_url != self._URL_A
        assert article.resolved_url != self._URL_B

    @pytest.mark.asyncio
    async def test_headers_rotation_tier_resolved_url_is_last_hop(self) -> None:
        """Rotation tier: resolved_url must be URL_C (last hop), not URL_A or URL_B."""
        result = _make_fetch_result("headers_rotation", resolved_url=self._URL_C)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = result

            article = await async_scrape(self._URL_A)

        assert article.resolved_url == self._URL_C
        assert article.resolved_url != self._URL_A
        assert article.resolved_url != self._URL_B

    @pytest.mark.asyncio
    async def test_browser_tier_resolved_url_is_last_hop(self) -> None:
        """Browser tier: resolved_url must be URL_C (last hop), not URL_A or URL_B."""
        result = _make_fetch_result("browser", resolved_url=self._URL_C)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_ROTATION, new_callable=AsyncMock) as mock_rotation,
            patch(_PATCH_LAUNCH_BROWSER, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = None
            mock_rotation.return_value = None
            mock_browser.return_value = result

            article = await async_scrape(self._URL_A)

        assert article.resolved_url == self._URL_C
        assert article.resolved_url != self._URL_A
        assert article.resolved_url != self._URL_B

    @pytest.mark.asyncio
    async def test_force_strategy_static_resolved_url_is_last_hop(self) -> None:
        """With force_strategy='static', resolved_url is still the last redirect hop."""
        result = _make_fetch_result("static", resolved_url=self._URL_C)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_static.return_value = result

            article = await async_scrape(self._URL_A, force_strategy="static")

        assert article.resolved_url == self._URL_C

    @pytest.mark.asyncio
    async def test_force_strategy_browser_resolved_url_is_last_hop(self) -> None:
        """With force_strategy='browser', resolved_url is still the last redirect hop."""
        result = _make_fetch_result("browser", resolved_url=self._URL_C)

        with (
            patch(_PATCH_LAUNCH_BROWSER, new_callable=AsyncMock),
            patch(_PATCH_BROWSER, new_callable=AsyncMock) as mock_browser,
            patch(_PATCH_EXTRACT, return_value=_MOCK_EXTRACTION),
        ):
            mock_browser.return_value = result

            article = await async_scrape(self._URL_A, force_strategy="browser")

        assert article.resolved_url == self._URL_C


# ---------------------------------------------------------------------------
# HTTP-level redirect chain — mock transport that issues real 301 responses
# ---------------------------------------------------------------------------

# Article HTML returned by the mock transport's final (200 OK) hop.
# Intentionally free of bot-detection signals so _is_bot_challenge() stays False.
_FETCHER_HTML = (
    "<html><head><title>Article After Redirect</title></head>"
    "<body><main>"
    "<h1>Article After Redirect</h1>"
    "<p>This article body was reached only after following a multi-hop HTTP "
    "redirect chain.  The resolved_url on the FetchResult must reflect the "
    "terminal URL, not the original short URL that was initially requested.</p>"
    "</main></body></html>"
)


class _MockRedirectTransport(httpx.AsyncBaseTransport):
    """
    httpx async transport that simulates a multi-hop HTTP redirect chain.

    No real network connections are made.  Each URL in *routes* maps to a
    ``(status_code, location_or_None, body)`` tuple:

    * ``status_code``: HTTP status to return for this URL.
    * ``location_or_None``: value for the ``Location`` header if this is a
      redirect; ``None`` if it is the terminal response.
    * ``body``: HTML body text for the response (empty string is fine for
      redirect responses that have no body).

    URLs not present in *routes* receive a ``404 Not Found`` response.
    """

    def __init__(
        self, routes: dict[str, tuple[int, str | None, str]]
    ) -> None:
        self._routes = routes

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        url = str(request.url)
        if url not in self._routes:
            return httpx.Response(404)
        status, location, body = self._routes[url]
        headers: dict[str, str] = {"content-type": "text/html; charset=utf-8"}
        if location:
            headers["location"] = location
        return httpx.Response(status, headers=headers, text=body)


def _make_client_factory(
    transport: httpx.AsyncBaseTransport,
) -> object:
    """
    Return a callable that replaces ``httpx.AsyncClient(...)`` but injects
    *transport* regardless of what the caller passed.

    Used with ``patch("articula._fetcher.httpx.AsyncClient",
    side_effect=_make_client_factory(t))``.
    """
    _real_cls = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport  # type: ignore[assignment]
        return _real_cls(*args, **kwargs)  # type: ignore[arg-type]

    return _factory


class TestFetcherHttpRedirectChain:
    """
    Verify that ``fetch_static`` and ``fetch_rotation`` follow HTTP redirects
    at the httpx level and store the *terminal* URL in
    ``FetchResult.resolved_url``.

    A ``_MockRedirectTransport`` drives each test — it answers each hop of
    the redirect chain with a genuine ``301`` response that httpx follows
    internally (``follow_redirects=True``).  No real network requests are made.

    Redirect chain simulated in most tests::

        URL_A ──301──▶ URL_B ──301──▶ URL_C (200 OK + article HTML)

    ``FetchResult.resolved_url`` must equal ``URL_C`` in all cases.
    """

    _URL_A = "https://short.example.com/s/abc"
    _URL_B = "https://tracker.example.com/t/xyz"
    _URL_C = "https://www.publisher.example.com/2026/06/07/full-article-title"

    def _make_three_hop_transport(self) -> _MockRedirectTransport:
        return _MockRedirectTransport(
            {
                self._URL_A: (301, self._URL_B, ""),
                self._URL_B: (301, self._URL_C, ""),
                self._URL_C: (200, None, _FETCHER_HTML),
            }
        )

    @pytest.mark.asyncio
    async def test_fetch_static_resolves_three_hop_redirect_chain(self) -> None:
        """
        fetch_static() with a 3-hop redirect chain (A→B→C) sets
        FetchResult.resolved_url to URL_C (the terminal URL), not URL_A or
        URL_B.

        The mock transport issues real 301 responses that httpx follows
        internally via ``follow_redirects=True``.
        """
        factory = _make_client_factory(self._make_three_hop_transport())
        with patch(
            "articula._fetcher.httpx.AsyncClient", side_effect=factory
        ):
            result = await fetch_static(self._URL_A)

        assert result is not None, (
            "fetch_static returned None — check that the HTML passes "
            "_is_bot_challenge and _HARD_FAIL_CODES checks"
        )
        assert result.resolved_url == self._URL_C, (
            f"Expected resolved_url={self._URL_C!r}, got {result.resolved_url!r}. "
            "fetch_static must store str(resp.url) — the post-redirect URL — "
            "not the original input URL."
        )
        assert result.resolved_url != self._URL_A, (
            "resolved_url must not equal the original input URL"
        )
        assert result.resolved_url != self._URL_B, (
            "resolved_url must not equal any intermediate redirect hop"
        )
        assert result.strategy_tier == "static"

    @pytest.mark.asyncio
    async def test_fetch_rotation_resolves_three_hop_redirect_chain(self) -> None:
        """
        fetch_rotation() with a 3-hop redirect chain (A→B→C) sets
        FetchResult.resolved_url to URL_C (the terminal URL).
        """
        factory = _make_client_factory(self._make_three_hop_transport())
        with patch(
            "articula._fetcher.httpx.AsyncClient", side_effect=factory
        ):
            result = await fetch_rotation(self._URL_A)

        assert result is not None, (
            "fetch_rotation returned None — check that the HTML passes "
            "_is_bot_challenge and _HARD_FAIL_CODES checks"
        )
        assert result.resolved_url == self._URL_C, (
            f"Expected resolved_url={self._URL_C!r}, got {result.resolved_url!r}"
        )
        assert result.resolved_url != self._URL_A
        assert result.resolved_url != self._URL_B
        assert result.strategy_tier == "headers_rotation"

    @pytest.mark.asyncio
    async def test_fetch_static_single_hop_redirect(self) -> None:
        """
        A single 301 redirect (URL_A → URL_C, no intermediate hops).
        FetchResult.resolved_url must equal URL_C.
        """
        transport = _MockRedirectTransport(
            {
                self._URL_A: (301, self._URL_C, ""),
                self._URL_C: (200, None, _FETCHER_HTML),
            }
        )
        factory = _make_client_factory(transport)
        with patch(
            "articula._fetcher.httpx.AsyncClient", side_effect=factory
        ):
            result = await fetch_static(self._URL_A)

        assert result is not None
        assert result.resolved_url == self._URL_C, (
            f"Expected resolved_url={self._URL_C!r}, got {result.resolved_url!r}"
        )
        assert result.resolved_url != self._URL_A

    @pytest.mark.asyncio
    async def test_fetch_static_no_redirect_resolved_url_equals_input(self) -> None:
        """
        Baseline: when the server returns 200 directly (no redirect), the
        FetchResult.resolved_url equals the original input URL.
        """
        transport = _MockRedirectTransport(
            {
                self._URL_A: (200, None, _FETCHER_HTML),
            }
        )
        factory = _make_client_factory(transport)
        with patch(
            "articula._fetcher.httpx.AsyncClient", side_effect=factory
        ):
            result = await fetch_static(self._URL_A)

        assert result is not None
        assert result.resolved_url == self._URL_A, (
            f"With no redirect, resolved_url must equal the input URL {self._URL_A!r}"
        )
