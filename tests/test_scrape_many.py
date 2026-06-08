"""
Sub-AC 2: ``scrape_many(urls, concurrency)`` order preservation and delegation.

Acceptance criteria verified here
----------------------------------
1. ``scrape_many`` calls the underlying single-URL scraper for **each** URL in
   the input list — verified via call-count and call-argument assertions on a
   mocked ``Scraper.scrape``.
2. Results are returned in the **same order** as the input URL list even when
   concurrent tasks complete out of order — verified with distinct sentinel
   values mapped per URL.
3. Per-URL ``ScraperError`` exceptions are **captured** in the result list
   rather than propagated as exceptions from ``scrape_many``.
4. The ``concurrency_limit`` parameter is forwarded to ``_bounded_gather``
   so bounded concurrency is respected.
5. An empty URL list returns an empty result list without raising.

Strategy
--------
All tests are unit-level.  ``Scraper.scrape`` is patched with ``AsyncMock``
so no real network requests are made.  Distinct ``MagicMock`` sentinels per
URL prove that each result slot received the right value regardless of task
completion order.  Timing assertions confirm that ``concurrency_limit=1``
forces serialisation when tasks have non-trivial duration.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from articula import async_scrape_many
from articula._scraper import Scraper
from articula.exceptions import ExtractionError, FetchError

# ---------------------------------------------------------------------------
# Sentinel factories
# ---------------------------------------------------------------------------


def _make_sentinels(urls: list[str]) -> dict[str, MagicMock]:
    """Return a URL → distinct MagicMock mapping for use as return sentinels."""
    return {url: MagicMock(name=f"sentinel:{url}") for url in urls}


# ---------------------------------------------------------------------------
# Helper: build a side_effect coroutine that maps url → sentinel
# ---------------------------------------------------------------------------


def _url_dispatching_side_effect(url_to_value: dict[str, Any]):
    """Return an async function that maps each URL to its sentinel value."""

    async def _dispatch(url: str) -> Any:
        return url_to_value[url]

    return _dispatch


# ---------------------------------------------------------------------------
# TestScrapeManyCallsUnderlyingScraper
# ---------------------------------------------------------------------------


class TestScrapeManyCallsUnderlyingScraper:
    """scrape_many must invoke Scraper.scrape once per URL."""

    @pytest.mark.asyncio
    async def test_each_url_scraped_exactly_once(self) -> None:
        """Every URL in the input list triggers exactly one Scraper.scrape call."""
        urls = [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ]
        sentinels = _make_sentinels(urls)
        side_effect = _url_dispatching_side_effect(sentinels)

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = side_effect
            await async_scrape_many(urls)

        assert mock_scrape.call_count == len(urls), (
            f"Expected {len(urls)} scrape calls; got {mock_scrape.call_count}"
        )

    @pytest.mark.asyncio
    async def test_correct_urls_passed_to_scraper(self) -> None:
        """The exact URL strings from the input list are forwarded to Scraper.scrape."""
        urls = [
            "https://example.com/page-1",
            "https://news.example.org/article",
            "https://blog.example.net/post/42",
        ]
        sentinels = _make_sentinels(urls)
        side_effect = _url_dispatching_side_effect(sentinels)

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = side_effect
            await async_scrape_many(urls)

        called_urls = [c.args[0] for c in mock_scrape.call_args_list]
        assert sorted(called_urls) == sorted(urls), (
            f"URLs forwarded to scraper do not match input.\n"
            f"  Expected (sorted): {sorted(urls)}\n"
            f"  Got (sorted): {sorted(called_urls)}"
        )

    @pytest.mark.asyncio
    async def test_single_url_calls_scraper_once(self) -> None:
        """A single-element URL list causes exactly one Scraper.scrape call."""
        url = "https://example.com/only"
        sentinel = MagicMock(name="only_sentinel")

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=sentinel)
        ) as mock_scrape:
            result = await async_scrape_many([url])

        assert mock_scrape.call_count == 1
        assert result == [sentinel]

    @pytest.mark.asyncio
    async def test_empty_url_list_never_calls_scraper(self) -> None:
        """No scraper calls are made when the URL list is empty."""
        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            result = await async_scrape_many([])

        assert mock_scrape.call_count == 0
        assert result == []


# ---------------------------------------------------------------------------
# TestScrapeManyOrderPreservation
# ---------------------------------------------------------------------------


class TestScrapeManyOrderPreservation:
    """Results must be in the same order as the input URL list."""

    @pytest.mark.asyncio
    async def test_results_in_input_order_three_urls(self) -> None:
        """Three URLs return results at positions matching their input positions."""
        urls = [
            "https://example.com/first",
            "https://example.com/second",
            "https://example.com/third",
        ]
        sentinels = _make_sentinels(urls)
        side_effect = _url_dispatching_side_effect(sentinels)

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = side_effect
            results = await async_scrape_many(urls)

        assert results[0] is sentinels[urls[0]], "First result must be sentinel for first URL"
        assert results[1] is sentinels[urls[1]], "Second result must be sentinel for second URL"
        assert results[2] is sentinels[urls[2]], "Third result must be sentinel for third URL"

    @pytest.mark.asyncio
    async def test_results_in_input_order_five_urls(self) -> None:
        """Five URLs — results at positions 0-4 match their input URL's sentinel."""
        urls = [f"https://example.com/page-{i}" for i in range(5)]
        sentinels = _make_sentinels(urls)
        side_effect = _url_dispatching_side_effect(sentinels)

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = side_effect
            results = await async_scrape_many(urls)

        for idx, url in enumerate(urls):
            assert results[idx] is sentinels[url], (
                f"Position {idx}: expected sentinel for {url!r}, "
                f"got {results[idx]!r}"
            )

    @pytest.mark.asyncio
    async def test_order_preserved_with_variable_completion_delay(self) -> None:
        """Results stay ordered even when tasks complete in reverse URL order.

        The last URL's task completes fastest; the first URL's task is slowest.
        The result list must still match input order, not completion order.
        """
        urls = [f"https://example.com/delayed-{i}" for i in range(4)]
        sentinels = _make_sentinels(urls)

        async def delayed_dispatch(url: str) -> Any:
            # URL at index 0 sleeps longest; URL at index 3 returns immediately.
            idx = urls.index(url)
            delay = (len(urls) - idx) * 0.02  # 0 → 0.08s, 3 → 0.02s
            await asyncio.sleep(delay)
            return sentinels[url]

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = delayed_dispatch
            results = await async_scrape_many(urls, concurrency_limit=len(urls))

        for idx, url in enumerate(urls):
            assert results[idx] is sentinels[url], (
                f"Position {idx}: order disrupted by completion-time variation"
            )

    @pytest.mark.asyncio
    async def test_distinct_sentinels_not_mixed_up(self) -> None:
        """Each result slot holds the sentinel for its own URL, not another URL's."""
        urls = [
            "https://alpha.example.com/",
            "https://beta.example.com/",
            "https://gamma.example.com/",
        ]
        sentinels = _make_sentinels(urls)
        side_effect = _url_dispatching_side_effect(sentinels)

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = side_effect
            results = await async_scrape_many(urls, concurrency_limit=3)

        # Confirm no cross-URL mixing: each position holds the right identity.
        for idx, url in enumerate(urls):
            other_sentinels = [sentinels[u] for u in urls if u != url]
            assert results[idx] not in other_sentinels, (
                f"Position {idx} holds another URL's sentinel (identity mix-up)"
            )

    @pytest.mark.asyncio
    async def test_result_length_matches_input_length(self) -> None:
        """The result list has exactly as many entries as the input URL list."""
        for n in [1, 3, 5, 10]:
            urls = [f"https://example.com/page-{i}" for i in range(n)]
            sentinel = MagicMock()

            with patch.object(Scraper, "scrape", new=AsyncMock(return_value=sentinel)):
                results = await async_scrape_many(urls)

            assert len(results) == n, (
                f"n={n}: expected {n} results, got {len(results)}"
            )


# ---------------------------------------------------------------------------
# TestScrapeManyErrorCapture
# ---------------------------------------------------------------------------


class TestScrapeManyErrorCapture:
    """Per-URL ScraperError must appear in the result list, not propagate."""

    @pytest.mark.asyncio
    async def test_fetch_error_captured_in_result_not_raised(self) -> None:
        """A FetchError from one URL appears as its result entry."""
        url = "https://unreachable.example.com/"
        error = FetchError("connection refused", url=url)

        with patch.object(Scraper, "scrape", new=AsyncMock(side_effect=error)):
            results = await async_scrape_many([url])

        assert len(results) == 1
        assert results[0] is error

    @pytest.mark.asyncio
    async def test_extraction_error_captured_in_result_not_raised(self) -> None:
        """An ExtractionError from one URL appears as its result entry."""
        url = "https://empty-body.example.com/"
        error = ExtractionError("no meaningful body", url=url)

        with patch.object(Scraper, "scrape", new=AsyncMock(side_effect=error)):
            results = await async_scrape_many([url])

        assert len(results) == 1
        assert results[0] is error

    @pytest.mark.asyncio
    async def test_mixed_success_and_error_results(self) -> None:
        """Success and error results coexist in the returned list."""
        urls = [
            "https://example.com/ok",
            "https://example.com/fail",
            "https://example.com/also-ok",
        ]
        ok_sentinel = MagicMock(name="ok_sentinel")
        also_ok_sentinel = MagicMock(name="also_ok_sentinel")
        error = FetchError("server error", url=urls[1])

        async def per_url_dispatch(url: str) -> Any:
            if url == urls[0]:
                return ok_sentinel
            if url == urls[1]:
                raise error
            return also_ok_sentinel

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = per_url_dispatch
            results = await async_scrape_many(urls)

        assert results[0] is ok_sentinel, "Position 0 should be success sentinel"
        assert results[1] is error, "Position 1 should be the captured FetchError"
        assert results[2] is also_ok_sentinel, "Position 2 should be success sentinel"

    @pytest.mark.asyncio
    async def test_error_position_preserved_in_order(self) -> None:
        """Error entries appear at the same position as their failing URL."""
        urls = [
            "https://example.com/first",    # success
            "https://example.com/second",   # error
            "https://example.com/third",    # success
            "https://example.com/fourth",   # error
        ]
        ok1 = MagicMock(name="ok1")
        ok2 = MagicMock(name="ok2")
        err2 = FetchError("fail", url=urls[1])
        err4 = FetchError("fail", url=urls[3])

        async def dispatch(url: str) -> Any:
            mapping = {
                urls[0]: ok1,
                urls[2]: ok2,
            }
            if url in mapping:
                return mapping[url]
            if url == urls[1]:
                raise err2
            raise err4

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = dispatch
            results = await async_scrape_many(urls)

        assert results[0] is ok1
        assert results[1] is err2
        assert results[2] is ok2
        assert results[3] is err4

    @pytest.mark.asyncio
    async def test_scrape_many_does_not_raise_on_all_errors(self) -> None:
        """Even when every URL fails, scrape_many returns a list instead of raising."""
        urls = [f"https://fail-{i}.example.com/" for i in range(3)]
        errors = [FetchError(f"error {i}", url=u) for i, u in enumerate(urls)]
        error_iter = iter(errors)

        async def always_fail(url: str) -> None:
            raise next(error_iter)

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = always_fail
            results = await async_scrape_many(urls)

        # No exception — all errors captured in the list.
        assert len(results) == len(urls)
        for result in results:
            assert isinstance(result, FetchError)


# ---------------------------------------------------------------------------
# TestScrapeManyEmptyInput
# ---------------------------------------------------------------------------


class TestScrapeManyEmptyInput:
    """Edge case: empty URL list."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_list(self) -> None:
        """scrape_many([]) returns [] immediately without calling any scraper."""
        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            results = await async_scrape_many([])

        assert results == []
        assert mock_scrape.call_count == 0

    @pytest.mark.asyncio
    async def test_empty_list_does_not_raise(self) -> None:
        """No exception is raised for an empty URL list."""
        result = await async_scrape_many([])  # should not raise
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestScrapeManyDefaultConcurrencyLimit
# ---------------------------------------------------------------------------


class TestScrapeManyDefaultConcurrencyLimit:
    """Default concurrency_limit=4 is used when not specified."""

    @pytest.mark.asyncio
    async def test_default_concurrency_limit_allows_up_to_four_concurrent(self) -> None:
        """With the default concurrency_limit=4, up to 4 tasks run simultaneously.

        Uses active-count tracking to verify the peak never exceeds 4.
        """
        TOTAL = 8
        urls = [f"https://example.com/page-{i}" for i in range(TOTAL)]
        sentinels = _make_sentinels(urls)

        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def tracked_dispatch(url: str) -> Any:
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
            return sentinels[url]

        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = tracked_dispatch
            results = await async_scrape_many(urls)  # uses default concurrency_limit=4

        assert peak <= 4, (
            f"Peak concurrent count {peak} exceeded default concurrency_limit=4"
        )
        for idx, url in enumerate(urls):
            assert results[idx] is sentinels[url]


# ---------------------------------------------------------------------------
# TestScrapeManyCustomConcurrencyLimit
# ---------------------------------------------------------------------------


class TestScrapeManyCustomConcurrencyLimit:
    """Custom concurrency_limit is forwarded and enforced."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_one_forces_serial_execution(self) -> None:
        """concurrency_limit=1 means at most one task runs at a time.

        Peak active count must be exactly 1 with timing evidence.
        """
        TOTAL = 4
        SLEEP = 0.05  # 50 ms per task
        urls = [f"https://example.com/serial-{i}" for i in range(TOTAL)]
        sentinels = _make_sentinels(urls)

        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def tracked_dispatch(url: str) -> Any:
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(SLEEP)
            async with lock:
                active -= 1
            return sentinels[url]

        t0 = time.monotonic()
        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = tracked_dispatch
            results = await async_scrape_many(urls, concurrency_limit=1)
        elapsed = time.monotonic() - t0

        # Active count never exceeded 1 (fully serial).
        assert peak == 1, f"Expected peak=1 with concurrency_limit=1; got {peak}"

        # Elapsed time proves serial execution: must be >= TOTAL * SLEEP * 0.8.
        min_expected = TOTAL * SLEEP * 0.8
        assert elapsed >= min_expected, (
            f"Elapsed {elapsed:.3f}s < {min_expected:.3f}s — "
            f"serial execution should take at least {min_expected:.2f}s"
        )

        # Results still in order.
        for idx, url in enumerate(urls):
            assert results[idx] is sentinels[url]

    @pytest.mark.asyncio
    async def test_concurrency_limit_equals_total_allows_full_parallel(self) -> None:
        """When concurrency_limit >= TOTAL, all tasks may run simultaneously.

        Wall-clock time must be less than 2x a single task's sleep duration.
        """
        TOTAL = 5
        SLEEP = 0.08  # 80 ms per task
        urls = [f"https://example.com/parallel-{i}" for i in range(TOTAL)]
        sentinels = _make_sentinels(urls)

        async def delayed_dispatch(url: str) -> Any:
            await asyncio.sleep(SLEEP)
            return sentinels[url]

        t0 = time.monotonic()
        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = delayed_dispatch
            results = await async_scrape_many(urls, concurrency_limit=TOTAL)
        elapsed = time.monotonic() - t0

        # All in parallel ⟹ elapsed ≈ SLEEP, well under 2 * SLEEP.
        max_expected = SLEEP * 2
        assert elapsed < max_expected, (
            f"Elapsed {elapsed:.3f}s >= {max_expected:.3f}s with limit={TOTAL}; "
            f"tasks appear to be running serially"
        )

        for idx, url in enumerate(urls):
            assert results[idx] is sentinels[url]

    @pytest.mark.asyncio
    async def test_concurrency_limit_two_with_six_urls(self) -> None:
        """concurrency_limit=2 with 6 URLs: peak active ≤ 2 and timing proves it.

        Minimum wall-clock time ≥ ceil(6/2) × SLEEP × 0.8 = 3 × SLEEP × 0.8.
        """
        TOTAL = 6
        LIMIT = 2
        SLEEP = 0.05
        urls = [f"https://example.com/batch-{i}" for i in range(TOTAL)]
        sentinels = _make_sentinels(urls)

        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def tracked_dispatch(url: str) -> Any:
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(SLEEP)
            async with lock:
                active -= 1
            return sentinels[url]

        min_expected = math.ceil(TOTAL / LIMIT) * SLEEP * 0.8

        t0 = time.monotonic()
        with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = tracked_dispatch
            results = await async_scrape_many(urls, concurrency_limit=LIMIT)
        elapsed = time.monotonic() - t0

        assert peak <= LIMIT, (
            f"Peak concurrent count {peak} exceeded concurrency_limit={LIMIT}"
        )
        assert elapsed >= min_expected, (
            f"Elapsed {elapsed:.3f}s < {min_expected:.3f}s — "
            f"concurrency_limit={LIMIT} not enforced"
        )
        for idx, url in enumerate(urls):
            assert results[idx] is sentinels[url]
