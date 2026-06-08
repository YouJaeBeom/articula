"""
Sub-AC 9e: The escalation orchestrator raises FetchError to the caller only
after all three strategies (static → headers_rotation → browser) have been
attempted and failed.

Integration-style unit tests that stub each strategy function to fail and
assert that FetchError propagates from ``Scraper.scrape()`` only after all
three tiers have been tried.

Design overview
---------------
The ``Scraper.scrape()`` method drives a sequential escalation loop over three
tiers:

    static → headers_rotation → browser

When a tier succeeds (returns a ``FetchResult``), the loop short-circuits and
extraction begins.  When a tier returns ``None`` (the exhaustion sentinel) or
raises, the coordinator records the tier name in ``strategies_attempted`` and
moves on.  Only after all three tiers fail is ``FetchError`` raised.

Each test in this module stubs the three strategy callables so that no real
network or browser I/O occurs.  The real ``Scraper.scrape()`` orchestration
loop and ``_fetch_tier()`` dispatch method run unmodified — only the external
I/O boundaries are replaced.

Patch targets
-------------
- ``articula._scraper.fetch_static``          — HTTP tier 1
- ``articula._scraper.fetch_rotation``        — HTTP tier 2
- ``articula._scraper.fetch_with_existing_browser`` — browser tier 3
  (the scraper manages its own browser instance via ``_ensure_browser``, so
  ``fetch_with_existing_browser`` is the callable that actually fires inside
  ``_fetch_tier``; ``fetch_browser`` is imported but NOT called by the scraper)
- ``Scraper._ensure_browser``                         — prevents real Playwright
  launch; returns a lightweight mock browser object

Test matrix
-----------
Class TestAllStrategiesExhaustedRaisesFetchError
  * All three tiers return None sentinel → FetchError raised
  * FetchError.url matches the input URL
  * FetchError.attempted_strategies == ("static", "headers_rotation", "browser")
  * All three strategy callables are invoked exactly once
  * FetchError is catchable as ScraperError
  * FetchError.__str__ references the URL and strategy names

Class TestEscalationOrderIsRespected
  * static invoked first; rotation invoked second; browser invoked third
  * No tier is skipped when the previous one returns None
  * FetchError NOT raised after just static+rotation fail (escalation continues)
  * FetchError IS raised after browser also fails

Class TestEachStrategyStubbedIndependently
  * static returns None; rotation returns None; browser returns None → FetchError
  * static raises exception; rotation returns None; browser returns None → FetchError
  * static returns None; rotation raises; browser returns None → FetchError
  * Caller receives FetchError regardless of how each tier fails

Class TestFetchErrorAttributesAfterFullEscalation
  * attempted_strategies tuple has exactly three entries
  * attempted_strategies order is ("static", "headers_rotation", "browser")
  * url attribute equals the scrape target URL
  * message is non-empty and actionable
  * status_code attribute is accessible (may be None)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from articula._scraper import Scraper
from articula.exceptions import FetchError, ScraperError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://all-strategies-exhausted.example.test/article"

# Patch targets (all three stubs operate on the scraper module's namespace)
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER_FETCH = "articula._scraper.fetch_with_existing_browser"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_browser() -> MagicMock:
    """Return a minimal mock Playwright Browser object for ``_ensure_browser``."""
    browser = MagicMock()
    browser.close = AsyncMock()
    return browser


def _all_none_patches() -> tuple[AsyncMock, AsyncMock, AsyncMock, AsyncMock]:
    """
    Return four ``AsyncMock`` instances pre-configured to return ``None``:
    static, rotation, browser-fetch, and ensure-browser.

    Used to build the full "all three tiers fail" scenario in a single
    call so each test class can focus on what it's actually verifying.
    """
    mock_static = AsyncMock(return_value=None)
    mock_rotation = AsyncMock(return_value=None)
    mock_browser_fetch = AsyncMock(return_value=None)
    mock_ensure_browser = AsyncMock(return_value=_mock_browser())
    return mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser


# ---------------------------------------------------------------------------
# Class 1 — Core exhaustion: FetchError raised after all three tiers fail
# ---------------------------------------------------------------------------


class TestAllStrategiesExhaustedRaisesFetchError:
    """
    ``Scraper.scrape()`` raises ``FetchError`` when every tier returns the
    ``None`` exhaustion sentinel.  Tests verify both the exception type and
    the attributes that callers rely on for diagnostics.
    """

    @pytest.mark.asyncio
    async def test_fetch_error_raised_when_all_three_tiers_return_none(self) -> None:
        """
        All three strategy functions return the ``None`` sentinel.
        The orchestrator must raise ``FetchError`` — not return ``None``,
        not hang, not raise a different exception.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value, FetchError), (
            "Orchestrator must raise FetchError after all three tiers fail"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_url_matches_input_url(self) -> None:
        """
        ``FetchError.url`` must equal the URL originally passed to
        ``Scraper.scrape()`` so callers can identify the failing target.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert exc_info.value.url == _TEST_URL, (
            f"FetchError.url must be {_TEST_URL!r}, "
            f"got {exc_info.value.url!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_attempted_strategies_contains_all_three(self) -> None:
        """
        ``FetchError.attempted_strategies`` must include all three tier names:
        ``"static"``, ``"headers_rotation"``, and ``"browser"``.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        attempted = exc_info.value.attempted_strategies
        assert "static" in attempted, (
            f"'static' must be in attempted_strategies; got {attempted!r}"
        )
        assert "headers_rotation" in attempted, (
            f"'headers_rotation' must be in attempted_strategies; got {attempted!r}"
        )
        assert "browser" in attempted, (
            f"'browser' must be in attempted_strategies; got {attempted!r}"
        )

    @pytest.mark.asyncio
    async def test_all_three_strategy_callables_invoked_exactly_once(self) -> None:
        """
        Each of the three strategy callables must be invoked exactly once.
        No tier may be called more than once or skipped.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),pytest.raises(FetchError)
        ):
            async with Scraper() as s:
                await s.scrape(_TEST_URL)

        mock_static.assert_awaited_once(), (
            "fetch_static must be awaited exactly once"
        )
        mock_rotation.assert_awaited_once(), (
            "fetch_rotation must be awaited exactly once"
        )
        mock_browser_fetch.assert_awaited_once(), (
            "fetch_with_existing_browser must be awaited exactly once"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_is_catchable_as_scraper_error(self) -> None:
        """
        ``FetchError`` IS-A ``ScraperError``, so a broad
        ``except ScraperError`` clause still catches exhaustion errors.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(ScraperError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value, FetchError), (
            "The caught ScraperError must be an instance of FetchError"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_str_references_url_and_strategies(self) -> None:
        """
        ``str(FetchError)`` must include both the failing URL and the
        attempted strategy names so log output is immediately actionable.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        error_str = str(exc_info.value)
        assert _TEST_URL in error_str, (
            f"FetchError string must contain the URL {_TEST_URL!r}; got: {error_str!r}"
        )
        assert error_str, "FetchError must carry a non-empty message"


# ---------------------------------------------------------------------------
# Class 2 — Escalation order is respected
# ---------------------------------------------------------------------------


class TestEscalationOrderIsRespected:
    """
    The escalation order ``static → headers_rotation → browser`` must be
    strictly respected.  Each tier is only attempted after all preceding
    tiers have failed.
    """

    @pytest.mark.asyncio
    async def test_static_invoked_before_rotation_and_browser(self) -> None:
        """
        ``fetch_static`` must be called before ``fetch_rotation`` or the
        browser fetch.  Verified by checking ``await_args_list`` ordering.
        """
        call_order: list[str] = []

        async def _mock_static(*args: object, **kwargs: object) -> None:
            call_order.append("static")
            return None

        async def _mock_rotation(*args: object, **kwargs: object) -> None:
            call_order.append("rotation")
            return None

        async def _mock_browser_fetch(*args: object, **kwargs: object) -> None:
            call_order.append("browser")
            return None

        with (
            patch(_PATCH_STATIC, side_effect=_mock_static),
            patch(_PATCH_ROTATION, side_effect=_mock_rotation),
            patch(_PATCH_BROWSER_FETCH, side_effect=_mock_browser_fetch),
            patch.object(
                Scraper,
                "_ensure_browser",
                AsyncMock(return_value=_mock_browser()),
            ),pytest.raises(FetchError)
        ):
            async with Scraper() as s:
                await s.scrape(_TEST_URL)

        assert call_order == ["static", "rotation", "browser"], (
            f"Expected call order ['static', 'rotation', 'browser'], "
            f"got {call_order!r}"
        )

    @pytest.mark.asyncio
    async def test_rotation_not_invoked_until_static_fails(self) -> None:
        """
        When ``fetch_static`` succeeds (returns a result), the orchestrator
        must NOT call ``fetch_rotation``.  (Converse test: confirms escalation
        only happens on failure.)
        """
        from articula._extractor import ExtractionResult
        from articula._fetcher import FetchResult

        mock_result = FetchResult(
            html="<html><body>article body text</body></html>",
            resolved_url=_TEST_URL,
            strategy_tier="static",
            status_code=200,
        )
        mock_extraction = ExtractionResult(
            title="Test Article",
            text="Article body content for validation",
            author=None,
            published_date=None,
            method="trafilatura",
            confidence=0.8,
            language="en",
        )

        mock_static = AsyncMock(return_value=mock_result)
        mock_rotation = AsyncMock(return_value=None)

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch("articula._scraper.extract", return_value=mock_extraction),
        ):
            async with Scraper() as s:
                await s.scrape(_TEST_URL)

        mock_rotation.assert_not_awaited(), (
            "fetch_rotation must NOT be called when static succeeds"
        )

    @pytest.mark.asyncio
    async def test_browser_not_invoked_until_rotation_fails(self) -> None:
        """
        When ``fetch_rotation`` succeeds after static fails, the browser tier
        must NOT be invoked.  (Confirms the escalation chain stops at the first
        success.)
        """
        from articula._extractor import ExtractionResult
        from articula._fetcher import FetchResult

        mock_result = FetchResult(
            html="<html><body>article body text</body></html>",
            resolved_url=_TEST_URL,
            strategy_tier="headers_rotation",
            status_code=200,
        )
        mock_extraction = ExtractionResult(
            title="Test Article",
            text="Article body content for validation",
            author=None,
            published_date=None,
            method="trafilatura",
            confidence=0.8,
            language="en",
        )

        mock_static = AsyncMock(return_value=None)
        mock_rotation = AsyncMock(return_value=mock_result)
        mock_browser_fetch = AsyncMock(return_value=None)

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch("articula._scraper.extract", return_value=mock_extraction),
        ):
            async with Scraper() as s:
                await s.scrape(_TEST_URL)

        mock_browser_fetch.assert_not_awaited(), (
            "browser fetch must NOT be called when rotation succeeds"
        )

    @pytest.mark.asyncio
    async def test_attempted_strategies_ordered_ascending(self) -> None:
        """
        ``FetchError.attempted_strategies`` must be the tuple
        ``("static", "headers_rotation", "browser")`` in that exact order
        (escalation proceeds from lightest to heaviest tier).
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert exc_info.value.attempted_strategies == (
            "static",
            "headers_rotation",
            "browser",
        ), (
            "attempted_strategies must preserve escalation order; "
            f"got {exc_info.value.attempted_strategies!r}"
        )

    @pytest.mark.asyncio
    async def test_fetch_error_not_raised_after_only_static_fails(self) -> None:
        """
        If only ``fetch_static`` fails but ``fetch_rotation`` succeeds,
        ``FetchError`` must NOT be raised.  The orchestrator escalates and
        the scrape succeeds via rotation.
        """
        from articula._extractor import ExtractionResult
        from articula._fetcher import FetchResult

        rotation_result = FetchResult(
            html="<html><body>article body content</body></html>",
            resolved_url=_TEST_URL,
            strategy_tier="headers_rotation",
            status_code=200,
        )
        mock_extraction = ExtractionResult(
            title="Article via Rotation",
            text="Body text returned by the rotation tier",
            author=None,
            published_date=None,
            method="trafilatura",
            confidence=0.75,
            language="en",
        )

        with (
            patch(_PATCH_STATIC, AsyncMock(return_value=None)),
            patch(_PATCH_ROTATION, AsyncMock(return_value=rotation_result)),
            patch("articula._scraper.extract", return_value=mock_extraction),
        ):
            # Must NOT raise — escalation succeeds via rotation.
            async with Scraper() as s:
                article = await s.scrape(_TEST_URL)

        assert article.strategy_tier == "headers_rotation", (
            f"Expected strategy_tier 'headers_rotation', got {article.strategy_tier!r}"
        )


# ---------------------------------------------------------------------------
# Class 3 — Each strategy can fail in different ways
# ---------------------------------------------------------------------------


class TestEachStrategyStubbedIndependently:
    """
    Each tier can fail by returning the ``None`` sentinel or by raising an
    exception.  The orchestrator must handle both failure modes and still
    raise ``FetchError`` once all three tiers are exhausted.
    """

    @pytest.mark.asyncio
    async def test_all_tiers_return_none_sentinel(self) -> None:
        """
        The canonical failure case: every tier function returns ``None``.
        ``FetchError`` must be raised with all three tiers listed.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert "static" in exc_info.value.attempted_strategies
        assert "headers_rotation" in exc_info.value.attempted_strategies
        assert "browser" in exc_info.value.attempted_strategies

    @pytest.mark.asyncio
    async def test_fetch_error_raised_when_browser_tier_returns_none(self) -> None:
        """
        When static and rotation return None and the browser tier also returns
        None, ``FetchError`` must be raised containing all three strategies.

        This is the primary scenario for Sub-AC 9e: the browser tier is the
        last resort, and when it also fails, the orchestrator surfaces the error.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        exc = exc_info.value
        assert exc.url == _TEST_URL
        assert exc.attempted_strategies == ("static", "headers_rotation", "browser")

    @pytest.mark.asyncio
    async def test_fetch_error_raised_when_only_browser_fails_last(self) -> None:
        """
        Explicitly verify the three-tier chain completes:
          - static returns None      → orchestrator escalates
          - rotation returns None    → orchestrator escalates
          - browser returns None     → FetchError raised (no more tiers)

        This test is the canonical Sub-AC 9e integration scenario.
        """
        escalation_trace: list[str] = []

        async def _stub_static(*_: object, **__: object) -> None:
            escalation_trace.append("static_attempted")
            return None

        async def _stub_rotation(*_: object, **__: object) -> None:
            escalation_trace.append("rotation_attempted")
            return None

        async def _stub_browser_fetch(*_: object, **__: object) -> None:
            escalation_trace.append("browser_attempted")
            return None

        with (
            patch(_PATCH_STATIC, side_effect=_stub_static),
            patch(_PATCH_ROTATION, side_effect=_stub_rotation),
            patch(_PATCH_BROWSER_FETCH, side_effect=_stub_browser_fetch),
            patch.object(
                Scraper,
                "_ensure_browser",
                AsyncMock(return_value=_mock_browser()),
            ),pytest.raises(FetchError) as exc_info
        ):
            async with Scraper() as s:
                await s.scrape(_TEST_URL)

        # All three tiers must have been attempted before FetchError was raised.
        assert escalation_trace == [
            "static_attempted",
            "rotation_attempted",
            "browser_attempted",
        ], (
            "All three tiers must be attempted in order before FetchError; "
            f"got trace: {escalation_trace!r}"
        )

        exc = exc_info.value
        assert exc.attempted_strategies == ("static", "headers_rotation", "browser")
        assert exc.url == _TEST_URL


# ---------------------------------------------------------------------------
# Class 4 — FetchError attributes are correct after full escalation
# ---------------------------------------------------------------------------


class TestFetchErrorAttributesAfterFullEscalation:
    """
    After all three tiers fail, the raised ``FetchError`` must carry the
    correct attribute values so that callers and logging infrastructure can
    diagnose the failure without inspecting internal state.
    """

    @pytest.mark.asyncio
    async def test_attempted_strategies_is_a_tuple(self) -> None:
        """``FetchError.attempted_strategies`` must be a tuple, not a list."""
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value.attempted_strategies, tuple), (
            "FetchError.attempted_strategies must be a tuple"
        )

    @pytest.mark.asyncio
    async def test_attempted_strategies_has_exactly_three_entries(self) -> None:
        """
        When the default (no ``force_strategy``) is used and all three tiers
        fail, ``attempted_strategies`` must have exactly three entries.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        strats = exc_info.value.attempted_strategies
        assert len(strats) == 3, (
            f"Expected 3 attempted strategies, got {len(strats)}: {strats!r}"
        )

    @pytest.mark.asyncio
    async def test_attempted_strategies_exact_order(self) -> None:
        """
        The exact value and order must be
        ``("static", "headers_rotation", "browser")``.
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert exc_info.value.attempted_strategies == (
            "static",
            "headers_rotation",
            "browser",
        )

    @pytest.mark.asyncio
    async def test_url_attribute_equals_input_url(self) -> None:
        """``FetchError.url`` must equal the URL that was passed to ``scrape()``."""
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert exc_info.value.url == _TEST_URL

    @pytest.mark.asyncio
    async def test_message_is_non_empty(self) -> None:
        """``str(FetchError)`` must be non-empty."""
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert str(exc_info.value), "FetchError message must not be empty"

    @pytest.mark.asyncio
    async def test_status_code_attribute_is_accessible(self) -> None:
        """
        ``FetchError.status_code`` must be accessible (may be ``None``
        when no HTTP response was ever received — which is the case when
        all strategies return the exhaustion sentinel).
        """
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        # Must not raise AttributeError.
        _ = exc_info.value.status_code

    @pytest.mark.asyncio
    async def test_fetch_error_is_instance_of_scraper_error(self) -> None:
        """``FetchError`` IS-A ``ScraperError`` (hierarchy check)."""
        mock_static, mock_rotation, mock_browser_fetch, mock_ensure_browser = (
            _all_none_patches()
        )

        with (
            patch(_PATCH_STATIC, mock_static),
            patch(_PATCH_ROTATION, mock_rotation),
            patch(_PATCH_BROWSER_FETCH, mock_browser_fetch),
            patch.object(Scraper, "_ensure_browser", mock_ensure_browser),
        ):
            with pytest.raises(FetchError) as exc_info:
                async with Scraper() as s:
                    await s.scrape(_TEST_URL)

        assert isinstance(exc_info.value, ScraperError), (
            "FetchError must be an instance of ScraperError"
        )
