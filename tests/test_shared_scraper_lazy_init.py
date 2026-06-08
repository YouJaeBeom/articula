"""
Sub-AC 1: Module-level scrape() initialises a shared Scraper instance lazily.

Acceptance criteria verified here
----------------------------------
1. ``articula._shared_scraper`` is ``None`` at module-level (i.e.
   before any ``scrape()`` call), confirming that no ``Scraper`` is created
   at import time.
2. After the **first** ``scrape()`` invocation, exactly **one** ``Scraper``
   instance has been created and ``_shared_scraper`` is that instance.
3. A **second** ``scrape()`` call reuses the same instance — no additional
   ``Scraper`` is constructed.
4. When keyword arguments are supplied, ``scrape()`` uses a per-call
   ``Scraper`` and does **not** modify the shared instance.

Strategy
--------
All tests are unit-level.  ``Scraper.scrape`` (the async coroutine) is
patched with ``AsyncMock`` so that no real network request is made, and
``Scraper.__init__`` is wrapped with a counting decorator to track how many
instances are created.

Test isolation: ``setup_method`` saves and restores ``_shared_scraper`` so
that tests cannot affect one another through the module-level singleton.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import articula
from articula._scraper import Scraper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_article():
    """Return a simple object that can stand in for an Article return value."""
    from unittest.mock import MagicMock
    return MagicMock()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestSharedScraperLazyInit:
    """scrape() must lazily create exactly one shared Scraper, reusing it later."""

    # ------------------------------------------------------------------
    # Isolation: save/restore the module-level singleton around each test
    # so tests are independent of execution order.
    # ------------------------------------------------------------------

    def setup_method(self) -> None:
        """Save the current shared scraper state and reset to None."""
        self._original_shared = articula._shared_scraper
        articula._shared_scraper = None

    def teardown_method(self) -> None:
        """Restore the shared scraper to its pre-test state."""
        articula._shared_scraper = self._original_shared

    # ------------------------------------------------------------------
    # AC assertion 1: None at module level before any scrape() call
    # ------------------------------------------------------------------

    def test_shared_scraper_is_none_before_any_call(self) -> None:
        """_shared_scraper must be None before the first scrape() invocation.

        The module attribute is explicitly set to None in setup_method,
        mirroring the state immediately after the module is imported for the
        first time.
        """
        assert articula._shared_scraper is None, (
            "_shared_scraper must be None at import time / before any scrape() call"
        )

    # ------------------------------------------------------------------
    # AC assertion 2: exactly one Scraper created on first call
    # ------------------------------------------------------------------

    def test_exactly_one_scraper_created_on_first_scrape_call(self) -> None:
        """The first scrape() call creates exactly one Scraper instance."""
        instances_created: list[Scraper] = []
        original_init = Scraper.__init__

        def tracking_init(self_obj: Scraper, **kwargs: object) -> None:
            instances_created.append(self_obj)
            original_init(self_obj, **kwargs)

        mock_article = _make_mock_article()

        with patch.object(Scraper, "__init__", tracking_init), patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            articula.scrape("https://example.com/article")

        assert len(instances_created) == 1, (
            f"Expected exactly 1 Scraper instance after first scrape() call; "
            f"got {len(instances_created)}"
        )

    def test_shared_scraper_not_none_after_first_call(self) -> None:
        """_shared_scraper is set to a Scraper instance after the first scrape()."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            articula.scrape("https://example.com/article")

        assert articula._shared_scraper is not None, (
            "_shared_scraper must be a Scraper instance after the first scrape() call"
        )

    def test_shared_scraper_is_scraper_instance_after_first_call(self) -> None:
        """_shared_scraper is an instance of Scraper after the first call."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            articula.scrape("https://example.com/article")

        assert isinstance(articula._shared_scraper, Scraper), (
            "_shared_scraper must be a Scraper instance"
        )

    # ------------------------------------------------------------------
    # AC assertion 3: no additional Scraper on subsequent calls
    # ------------------------------------------------------------------

    def test_no_additional_scraper_on_second_call(self) -> None:
        """A second scrape() call must NOT create a new Scraper instance."""
        instances_created: list[Scraper] = []
        original_init = Scraper.__init__

        def tracking_init(self_obj: Scraper, **kwargs: object) -> None:
            instances_created.append(self_obj)
            original_init(self_obj, **kwargs)

        mock_article = _make_mock_article()

        with patch.object(Scraper, "__init__", tracking_init), patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            articula.scrape("https://example.com/first")
            articula.scrape("https://example.com/second")

        assert len(instances_created) == 1, (
            f"Expected exactly 1 Scraper instance after two scrape() calls; "
            f"got {len(instances_created)}"
        )

    def test_same_shared_scraper_object_across_calls(self) -> None:
        """Both scrape() calls use the identical shared Scraper object."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            articula.scrape("https://example.com/first")
            instance_after_first = articula._shared_scraper

            articula.scrape("https://example.com/second")
            instance_after_second = articula._shared_scraper

        assert instance_after_first is instance_after_second, (
            "The same Scraper object must be reused on every no-kwargs scrape() call"
        )

    def test_many_calls_still_one_instance(self) -> None:
        """Ten consecutive scrape() calls create exactly one Scraper instance."""
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
                articula.scrape(f"https://example.com/page-{i}")

        assert len(instances_created) == 1, (
            f"Expected exactly 1 Scraper instance after 10 scrape() calls; "
            f"got {len(instances_created)}"
        )

    # ------------------------------------------------------------------
    # AC assertion 4: kwargs bypass shared scraper
    # ------------------------------------------------------------------

    def test_kwargs_call_does_not_set_shared_scraper(self) -> None:
        """When kwargs are supplied, the shared scraper is NOT initialised."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            articula.scrape(
                "https://example.com/article", timeout=60
            )

        # With kwargs the shared singleton must stay None (setup reset it).
        assert articula._shared_scraper is None, (
            "Passing kwargs must not populate the shared scraper; "
            "per-call Scraper should be used instead"
        )

    def test_kwargs_call_followed_by_no_kwargs_creates_one_shared(self) -> None:
        """A kwargs call followed by a plain call still leaves exactly one shared instance."""
        mock_article = _make_mock_article()

        with patch.object(
            Scraper, "scrape", new=AsyncMock(return_value=mock_article)
        ):
            # kwargs call — uses a fresh per-call Scraper, _shared_scraper stays None
            articula.scrape("https://example.com/first", timeout=60)
            assert articula._shared_scraper is None

            # plain call — initialises shared scraper
            articula.scrape("https://example.com/second")
            assert articula._shared_scraper is not None

    # ------------------------------------------------------------------
    # _get_shared_scraper helper exposed directly
    # ------------------------------------------------------------------

    def test_get_shared_scraper_returns_scraper_instance(self) -> None:
        """_get_shared_scraper() returns a Scraper regardless of call count."""
        s = articula._get_shared_scraper()
        assert isinstance(s, Scraper)

    def test_get_shared_scraper_returns_same_object_on_repeat(self) -> None:
        """_get_shared_scraper() returns the identical object on every call."""
        first = articula._get_shared_scraper()
        second = articula._get_shared_scraper()
        assert first is second

    def test_get_shared_scraper_sets_module_attribute(self) -> None:
        """Calling _get_shared_scraper() sets articula._shared_scraper."""
        assert articula._shared_scraper is None  # pre-condition from setup
        articula._get_shared_scraper()
        assert articula._shared_scraper is not None
