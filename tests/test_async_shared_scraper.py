"""
Sub-AC 2: Module-level ``async_scrape()`` reuses the shared Scraper instance.

Acceptance criteria verified here
----------------------------------
1. ``articula._shared_scraper`` is ``None`` before any ``async_scrape()``
   call (no Scraper created at import time).
2. After the **first** ``async_scrape()`` invocation (no kwargs), exactly **one**
   ``Scraper`` instance has been created and ``_shared_scraper`` is that instance.
3. A **second** ``async_scrape()`` call reuses the **same** instance — no
   additional ``Scraper`` is constructed.
4. When keyword arguments are supplied, a fresh per-call ``Scraper`` is used
   and the shared instance is **not** modified.

Strategy
--------
All tests are async (pytest-asyncio with asyncio_mode="auto").
``Scraper.scrape`` is patched with ``AsyncMock`` so no real network call is made.
``Scraper.__init__`` is wrapped with a counting decorator to track instantiation.
``setup_method`` / ``teardown_method`` save and restore ``_shared_scraper`` for
full test isolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import articula
from articula._scraper import Scraper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_article() -> MagicMock:
    """Return a simple object that can stand in for an Article return value."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestAsyncScrapeSharedScraperLazyInit:
    """async_scrape() must lazily create exactly one shared Scraper and reuse it."""

    # ------------------------------------------------------------------
    # Isolation: save/restore the module-level singleton around each test
    # ------------------------------------------------------------------

    def setup_method(self) -> None:
        """Save the current shared scraper state and reset to None."""
        self._original_shared = articula._shared_scraper
        articula._shared_scraper = None

    def teardown_method(self) -> None:
        """Restore the shared scraper to its pre-test state."""
        articula._shared_scraper = self._original_shared

    # ------------------------------------------------------------------
    # AC 1: _shared_scraper is None before any async_scrape() call
    # ------------------------------------------------------------------

    async def test_shared_scraper_none_before_async_scrape(self) -> None:
        """_shared_scraper must be None before the first async_scrape() call."""
        assert articula._shared_scraper is None, (
            "_shared_scraper must be None before any async_scrape() invocation"
        )

    # ------------------------------------------------------------------
    # AC 2: exactly one Scraper created on the first no-kwargs call
    # ------------------------------------------------------------------

    async def test_exactly_one_scraper_created_on_first_async_call(self) -> None:
        """First async_scrape() call with no kwargs creates exactly one Scraper."""
        instances_created: list[Scraper] = []
        original_init = Scraper.__init__

        def tracking_init(self_obj: Scraper, **kwargs: object) -> None:
            instances_created.append(self_obj)
            original_init(self_obj, **kwargs)

        mock_article = _make_mock_article()

        with patch.object(Scraper, "__init__", tracking_init), patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            await articula.async_scrape("https://example.com/article")

        assert len(instances_created) == 1, (
            f"Expected exactly 1 Scraper instance after first async_scrape(); "
            f"got {len(instances_created)}"
        )

    async def test_shared_scraper_set_after_first_async_call(self) -> None:
        """_shared_scraper is a Scraper instance after the first async_scrape() call."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            await articula.async_scrape("https://example.com/article")

        assert articula._shared_scraper is not None, (
            "_shared_scraper must be set after the first no-kwargs async_scrape() call"
        )
        assert isinstance(articula._shared_scraper, Scraper), (
            "_shared_scraper must be a Scraper instance"
        )

    # ------------------------------------------------------------------
    # AC 3: no additional Scraper created on subsequent no-kwargs calls
    # ------------------------------------------------------------------

    async def test_no_additional_scraper_on_second_async_call(self) -> None:
        """Second async_scrape() call must NOT create a new Scraper instance."""
        instances_created: list[Scraper] = []
        original_init = Scraper.__init__

        def tracking_init(self_obj: Scraper, **kwargs: object) -> None:
            instances_created.append(self_obj)
            original_init(self_obj, **kwargs)

        mock_article = _make_mock_article()

        with patch.object(Scraper, "__init__", tracking_init), patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            await articula.async_scrape("https://example.com/first")
            await articula.async_scrape("https://example.com/second")

        assert len(instances_created) == 1, (
            f"Expected exactly 1 Scraper instance after two async_scrape() calls; "
            f"got {len(instances_created)}"
        )

    async def test_same_shared_scraper_object_across_async_calls(self) -> None:
        """Both async_scrape() calls must use the identical shared Scraper object."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            await articula.async_scrape("https://example.com/first")
            instance_after_first = articula._shared_scraper

            await articula.async_scrape("https://example.com/second")
            instance_after_second = articula._shared_scraper

        assert instance_after_first is instance_after_second, (
            "The same Scraper object must be reused on every no-kwargs async_scrape() call"
        )

    async def test_many_async_calls_still_one_instance(self) -> None:
        """Ten consecutive async_scrape() calls create exactly one Scraper instance."""
        instances_created: list[Scraper] = []
        original_init = Scraper.__init__

        def tracking_init(self_obj: Scraper, **kwargs: object) -> None:
            instances_created.append(self_obj)
            original_init(self_obj, **kwargs)

        mock_article = _make_mock_article()

        with patch.object(Scraper, "__init__", tracking_init), patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            for i in range(10):
                await articula.async_scrape(
                    f"https://example.com/page-{i}"
                )

        assert len(instances_created) == 1, (
            f"Expected exactly 1 Scraper instance after 10 async_scrape() calls; "
            f"got {len(instances_created)}"
        )

    # ------------------------------------------------------------------
    # AC 4: kwargs bypass the shared scraper
    # ------------------------------------------------------------------

    async def test_kwargs_async_call_does_not_set_shared_scraper(self) -> None:
        """When kwargs are supplied, async_scrape() does NOT initialise the shared scraper."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            await articula.async_scrape(
                "https://example.com/article", timeout=60
            )

        assert articula._shared_scraper is None, (
            "Passing kwargs must not populate the shared scraper; "
            "per-call Scraper should be used instead"
        )

    async def test_kwargs_call_uses_fresh_scraper_not_shared(self) -> None:
        """A kwargs call creates a distinct Scraper, not the shared one."""
        instances_created: list[Scraper] = []
        original_init = Scraper.__init__

        def tracking_init(self_obj: Scraper, **kwargs: object) -> None:
            instances_created.append(self_obj)
            original_init(self_obj, **kwargs)

        mock_article = _make_mock_article()

        with patch.object(Scraper, "__init__", tracking_init), patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            await articula.async_scrape(
                "https://example.com/article", timeout=90
            )

        # One Scraper was created (for the kwargs call), but the shared
        # singleton must still be None.
        assert len(instances_created) == 1
        assert articula._shared_scraper is None, (
            "The per-call Scraper must not be stored as _shared_scraper"
        )

    async def test_no_kwargs_then_kwargs_still_reuses_shared(self) -> None:
        """A plain call followed by a kwargs call: the shared instance is not replaced."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            # First plain call: creates the shared scraper
            await articula.async_scrape("https://example.com/plain")
            shared_after_first = articula._shared_scraper

            # Kwargs call: uses a fresh per-call Scraper
            await articula.async_scrape(
                "https://example.com/with-kwargs", timeout=60
            )
            shared_after_kwargs = articula._shared_scraper

        # The shared instance must not have been replaced by the kwargs call.
        assert shared_after_first is shared_after_kwargs, (
            "A kwargs async_scrape() call must not replace the shared Scraper"
        )

    # ------------------------------------------------------------------
    # Parity: async_scrape() and scrape() share the same Scraper instance
    # ------------------------------------------------------------------

    async def test_async_scrape_uses_same_instance_as_get_shared_scraper(self) -> None:
        """async_scrape() must use the same Scraper returned by _get_shared_scraper()."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            # Initialise the shared scraper via the helper (same helper used by scrape())
            shared_via_helper = articula._get_shared_scraper()

            # async_scrape() call — it should reuse the same singleton
            await articula.async_scrape("https://example.com/async")
            shared_after_async = articula._shared_scraper

        assert shared_via_helper is shared_after_async, (
            "async_scrape() must use the same Scraper singleton as _get_shared_scraper()"
        )
