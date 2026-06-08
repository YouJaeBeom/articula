"""
Sub-AC 1: ``_bounded_gather(coros, limit)`` helper.

Verifies that:
- Results are returned in the same order as the input coroutines.
- No more than *limit* coroutines execute simultaneously (via active-count
  assertions and timing assertions).
- A limit of 1 serialises execution completely.
- A limit >= total tasks imposes no extra constraint (all run in parallel).
- Passing limit < 1 raises ``ValueError``.
- An empty iterable returns an empty list.
- Results match the value each coroutine returns (not mixed up by index).

Timing strategy
---------------
Each coroutine sleeps for SLEEP_S seconds.  With TOTAL tasks and LIMIT slots
the minimum wall-clock time is ``ceil(TOTAL / LIMIT) * SLEEP_S``.  If all
tasks ran simultaneously the total time would be only SLEEP_S, so the timing
lower-bound distinguishes "bounded" from "unbounded" execution reliably even
on slow CI hosts.

Active-count strategy
---------------------
An ``asyncio.Lock`` serialises increments/reads of a shared counter.  Because
asyncio is single-threaded the counter accurately reflects the number of
coroutines currently inside the semaphore-guarded section.
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from articula._scraper import _bounded_gather

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SLEEP_S = 0.10  # 100 ms — large enough to survive CI jitter


async def _sleeping_coro(return_value: int, duration: float = SLEEP_S) -> int:
    """Coroutine that sleeps for *duration* seconds then returns *return_value*."""
    await asyncio.sleep(duration)
    return return_value


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------


class TestResultOrder:
    """Results must come back in the same order as the input coroutines."""

    @pytest.mark.asyncio
    async def test_results_in_input_order_small(self) -> None:
        """Five tasks with limit=5 return values in submission order."""
        coros = [_sleeping_coro(i, 0.01) for i in range(5)]
        results = await _bounded_gather(coros, limit=5)
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_results_in_input_order_with_limit(self) -> None:
        """Six tasks with limit=2 still return values in submission order."""
        coros = [_sleeping_coro(i, 0.01) for i in range(6)]
        results = await _bounded_gather(coros, limit=2)
        assert results == [0, 1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_results_in_input_order_limit_one(self) -> None:
        """Serialised (limit=1) execution still returns ordered results."""
        coros = [_sleeping_coro(i, 0.01) for i in range(4)]
        results = await _bounded_gather(coros, limit=1)
        assert results == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_values_not_shuffled_by_completion_order(self) -> None:
        """Tasks finishing in reverse order must not shuffle the result list.

        Tasks 0..4 sleep for decreasing durations so they complete in reverse
        order (4 first, 0 last), yet the result list must remain [0,1,2,3,4].
        """
        # Limit is higher than task count so all run in parallel.
        coros = [_sleeping_coro(i, (5 - i) * 0.01) for i in range(5)]
        results = await _bounded_gather(coros, limit=10)
        assert results == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Active-count concurrency proof
# ---------------------------------------------------------------------------


class TestActiveConcurrencyCount:
    """Peak active count must never exceed the limit."""

    @pytest.mark.asyncio
    async def test_peak_active_never_exceeds_limit(self) -> None:
        """Peak simultaneous execution == limit (not more) with 6 tasks, limit=2."""
        LIMIT = 2
        TOTAL = 6
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def tracked_coro(i: int) -> int:
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(SLEEP_S)
            async with lock:
                active -= 1
            return i

        results = await _bounded_gather(
            [tracked_coro(i) for i in range(TOTAL)], LIMIT
        )

        assert peak <= LIMIT, (
            f"Peak concurrent count {peak} exceeded limit {LIMIT}"
        )
        assert results == list(range(TOTAL))

    @pytest.mark.asyncio
    async def test_limit_one_is_fully_serial(self) -> None:
        """With limit=1, peak active is always exactly 1."""
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def tracked_coro(i: int) -> int:
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return i

        results = await _bounded_gather(
            [tracked_coro(i) for i in range(5)], limit=1
        )

        assert peak == 1, f"Expected peak == 1 for limit=1, got {peak}"
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_limit_equals_total_allows_full_parallelism(self) -> None:
        """When limit >= TOTAL, all tasks may run at once (peak can reach TOTAL)."""
        TOTAL = 4
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def tracked_coro(i: int) -> int:
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
            return i

        results = await _bounded_gather(
            [tracked_coro(i) for i in range(TOTAL)], limit=TOTAL
        )

        # All TOTAL tasks should have run simultaneously — peak == TOTAL.
        assert peak == TOTAL
        assert results == list(range(TOTAL))

    @pytest.mark.asyncio
    async def test_different_limit_values(self) -> None:
        """Parametrised check: various (total, limit) pairs obey peak <= limit."""
        cases = [(8, 3), (10, 4), (6, 6), (3, 1)]

        for total, limit in cases:
            # Per-iteration state held in a dict + lock, bound explicitly as
            # default arguments so the coroutine factory never closes over the
            # loop variables (avoids the classic late-binding closure bug).
            state = {"active": 0, "peak": 0}
            lock = asyncio.Lock()

            def build_coro(idx: int, state=state, lock=lock):
                async def coro() -> int:
                    async with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    await asyncio.sleep(0.01)
                    async with lock:
                        state["active"] -= 1
                    return idx

                return coro()

            coros = [build_coro(i) for i in range(total)]
            out = await _bounded_gather(coros, limit)
            assert state["peak"] <= limit, (
                f"total={total}, limit={limit}: peak {state['peak']} > limit"
            )
            assert out == list(range(total)), (
                f"total={total}, limit={limit}: wrong order"
            )


# ---------------------------------------------------------------------------
# Timing assertions
# ---------------------------------------------------------------------------


class TestTimingAssertions:
    """Wall-clock time proves bounded batching."""

    @pytest.mark.asyncio
    async def test_timing_proves_limit_two_with_six_tasks(self) -> None:
        """Six tasks, limit=2, sleep=SLEEP_S ⟹ elapsed >= ceil(6/2)*SLEEP_S * 0.8.

        If all 6 ran simultaneously the elapsed time would be ~SLEEP_S.  Bounded
        at 2, we need at least 3 serial rounds ≈ 3*SLEEP_S.  The 0.8 factor adds
        a generous margin for scheduler overhead without false-positives on CI.
        """
        LIMIT = 2
        TOTAL = 6
        min_expected = math.ceil(TOTAL / LIMIT) * SLEEP_S * 0.8

        coros = [_sleeping_coro(i) for i in range(TOTAL)]
        t0 = time.monotonic()
        results = await _bounded_gather(coros, LIMIT)
        elapsed = time.monotonic() - t0

        assert elapsed >= min_expected, (
            f"Elapsed {elapsed:.3f}s < {min_expected:.3f}s — "
            f"suggests more than {LIMIT} tasks ran concurrently (all-parallel "
            f"would take ~{SLEEP_S:.2f}s, bounded should take ≥{3 * SLEEP_S:.2f}s)"
        )
        assert results == list(range(TOTAL))

    @pytest.mark.asyncio
    async def test_timing_limit_one_is_slowest(self) -> None:
        """Limit=1 forces full serialisation: elapsed >= TOTAL * SLEEP_S * 0.8."""
        LIMIT = 1
        TOTAL = 4
        min_expected = TOTAL * SLEEP_S * 0.8

        coros = [_sleeping_coro(i) for i in range(TOTAL)]
        t0 = time.monotonic()
        results = await _bounded_gather(coros, LIMIT)
        elapsed = time.monotonic() - t0

        assert elapsed >= min_expected, (
            f"Elapsed {elapsed:.3f}s < {min_expected:.3f}s for limit=1 "
            f"(expected at least {min_expected:.3f}s for serial execution)"
        )
        assert results == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_timing_high_limit_is_fast(self) -> None:
        """Limit >= TOTAL: all tasks run in parallel, elapsed < 2 * SLEEP_S."""
        TOTAL = 6
        LIMIT = TOTAL  # no effective constraint
        max_expected = SLEEP_S * 2  # generous upper bound for parallel run

        coros = [_sleeping_coro(i) for i in range(TOTAL)]
        t0 = time.monotonic()
        results = await _bounded_gather(coros, LIMIT)
        elapsed = time.monotonic() - t0

        assert elapsed < max_expected, (
            f"Elapsed {elapsed:.3f}s >= {max_expected:.3f}s with limit={LIMIT}; "
            f"tasks appear to be running serially instead of in parallel"
        )
        assert results == list(range(TOTAL))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and error-path behaviour."""

    @pytest.mark.asyncio
    async def test_empty_iterable_returns_empty_list(self) -> None:
        """Passing an empty iterable produces an empty list immediately."""
        results = await _bounded_gather([], limit=4)
        assert results == []

    @pytest.mark.asyncio
    async def test_single_task(self) -> None:
        """A single coroutine works correctly with any limit."""
        results = await _bounded_gather([_sleeping_coro(42, 0.01)], limit=1)
        assert results == [42]

    @pytest.mark.asyncio
    async def test_limit_less_than_one_raises_value_error(self) -> None:
        """limit=0 must raise ValueError immediately."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await _bounded_gather([_sleeping_coro(1, 0.0)], limit=0)

    @pytest.mark.asyncio
    async def test_limit_negative_raises_value_error(self) -> None:
        """Negative limit must raise ValueError."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await _bounded_gather([], limit=-5)

    @pytest.mark.asyncio
    async def test_generator_input_accepted(self) -> None:
        """_bounded_gather must accept a generator (lazy iterable), not just lists."""
        gen = (_sleeping_coro(i, 0.01) for i in range(4))
        results = await _bounded_gather(gen, limit=2)
        assert results == [0, 1, 2, 3]
