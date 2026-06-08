"""
Unit test for Sub-AC 3a — ExtractionError.strategies_tried attribute.

Sub-AC 3a: ``ExtractionError`` stores ``url`` (str) and ``strategies_tried``
(list[str]) as instance attributes.  A unit test instantiates it with a sample
URL and strategy list and asserts both attributes are accessible and match the
inputs.

The ``strategies_tried`` attribute is a property that returns the same data as
the canonical ``strategies_attempted`` tuple, exposed as a ``list[str]`` for
caller convenience.
"""

from __future__ import annotations

from articula.exceptions import ExtractionError, ScraperError

# ---------------------------------------------------------------------------
# Constants — sample inputs used across all test methods
# ---------------------------------------------------------------------------

_SAMPLE_URL = "https://example.com/article/sub-ac-3a"
_SAMPLE_STRATEGIES: list[str] = ["static", "headers_rotation", "browser"]


# ---------------------------------------------------------------------------
# Sub-AC 3a — core requirement: instantiate and assert both attributes
# ---------------------------------------------------------------------------


class TestExtractionErrorStrategiesTried:
    """
    Core Sub-AC 3a tests.

    Instantiate ``ExtractionError`` with a known URL and strategy list, then
    assert that ``url`` and ``strategies_tried`` are accessible and match the
    inputs exactly.
    """

    def test_url_accessible_and_matches_input(self) -> None:
        """``ExtractionError.url`` is accessible and equals the supplied URL."""
        exc = ExtractionError(
            "content extraction failed",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert exc.url == _SAMPLE_URL

    def test_strategies_tried_accessible_and_matches_input(self) -> None:
        """``ExtractionError.strategies_tried`` is accessible and equals the input list."""
        exc = ExtractionError(
            "content extraction failed",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert exc.strategies_tried == _SAMPLE_STRATEGIES

    def test_both_attributes_populated_and_correct(self) -> None:
        """
        Core Sub-AC 3a assertion: both ``url`` and ``strategies_tried`` are
        populated and match the inputs in a single instantiation.
        """
        exc = ExtractionError(
            "no article body found",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert exc.url == _SAMPLE_URL, (
            f"url must match input; expected {_SAMPLE_URL!r}, got {exc.url!r}"
        )
        assert exc.strategies_tried == _SAMPLE_STRATEGIES, (
            f"strategies_tried must match input; "
            f"expected {_SAMPLE_STRATEGIES!r}, got {exc.strategies_tried!r}"
        )

    def test_strategies_tried_is_list(self) -> None:
        """``strategies_tried`` returns a ``list``, not a tuple or other type."""
        exc = ExtractionError(
            "empty body",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert isinstance(exc.strategies_tried, list), (
            f"strategies_tried must be list, got {type(exc.strategies_tried).__name__}"
        )

    def test_strategies_tried_elements_are_strings(self) -> None:
        """Every element of ``strategies_tried`` is a ``str``."""
        exc = ExtractionError(
            "empty body",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert all(isinstance(s, str) for s in exc.strategies_tried), (
            "All elements of strategies_tried must be str"
        )

    def test_strategies_tried_order_preserved(self) -> None:
        """``strategies_tried`` preserves the order of the input list."""
        strategies = ["static", "headers_rotation", "browser"]
        exc = ExtractionError("empty", strategies_attempted=strategies)
        assert exc.strategies_tried == strategies

    def test_strategies_tried_with_single_strategy(self) -> None:
        """``strategies_tried`` works correctly with a single-element input."""
        exc = ExtractionError(
            "empty body",
            url=_SAMPLE_URL,
            strategies_attempted=["static"],
        )
        assert exc.strategies_tried == ["static"]

    def test_strategies_tried_with_empty_input(self) -> None:
        """``strategies_tried`` returns an empty list when no strategies were tried."""
        exc = ExtractionError("empty body", url=_SAMPLE_URL, strategies_attempted=[])
        assert exc.strategies_tried == []
        assert isinstance(exc.strategies_tried, list)

    def test_strategies_tried_default_is_empty_list(self) -> None:
        """``strategies_tried`` defaults to ``[]`` when ``strategies_attempted`` is omitted."""
        exc = ExtractionError("empty body")
        assert exc.strategies_tried == []

    def test_url_defaults_to_empty_string(self) -> None:
        """``url`` defaults to ``""`` when not provided."""
        exc = ExtractionError("extraction failed")
        assert exc.url == ""

    def test_strategies_tried_returns_fresh_copy_each_access(self) -> None:
        """Each access to ``strategies_tried`` returns a new list (not cached)."""
        exc = ExtractionError("empty", strategies_attempted=["static", "browser"])
        list1 = exc.strategies_tried
        list2 = exc.strategies_tried
        assert list1 == list2
        assert list1 is not list2, (
            "strategies_tried must return a fresh list each time to preserve immutability"
        )

    def test_strategies_tried_consistent_with_strategies_attempted(self) -> None:
        """``strategies_tried`` reflects the same data as ``strategies_attempted``."""
        exc = ExtractionError(
            "empty",
            strategies_attempted=["static", "headers_rotation"],
        )
        assert exc.strategies_tried == list(exc.strategies_attempted)

    def test_attributes_preserved_through_raise_and_catch(self) -> None:
        """Both ``url`` and ``strategies_tried`` survive a raise/except cycle."""
        try:
            raise ExtractionError(
                "no body found",
                url=_SAMPLE_URL,
                strategies_attempted=_SAMPLE_STRATEGIES,
            )
        except ExtractionError as exc:
            assert exc.url == _SAMPLE_URL
            assert exc.strategies_tried == _SAMPLE_STRATEGIES

    def test_exception_is_scraper_error_subclass(self) -> None:
        """``ExtractionError`` remains a subclass of ``ScraperError``."""
        exc = ExtractionError(
            "empty body",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert isinstance(exc, ScraperError)

    def test_url_is_string(self) -> None:
        """``url`` attribute is a ``str``."""
        exc = ExtractionError(
            "empty body",
            url=_SAMPLE_URL,
            strategies_attempted=_SAMPLE_STRATEGIES,
        )
        assert isinstance(exc.url, str)
