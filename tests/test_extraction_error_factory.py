"""
Unit tests for the ``_raise_extraction_error`` helper/factory function.

Sub-AC 11.2: A helper/factory function raises ``ExtractionError`` after all
strategies are exhausted, populating ``url`` and ``strategies_attempted``
from the extraction context.

The helper is located in ``articula._scraper`` and is the single
authoritative site for constructing ``ExtractionError`` instances with a
populated ``url`` and ``strategies_attempted``.

Test coverage
-------------
TestRaiseExtractionErrorAlwaysRaises
  * The function always raises, never returns normally.
  * The raised exception is exactly ``ExtractionError``.
  * The raised exception is catchable as ``ScraperError``.

TestRaiseExtractionErrorAttributesFromContext
  * ``url`` attribute matches the ``url`` argument passed to the function.
  * ``strategies_attempted`` attribute matches the ``strategies_attempted``
    argument (converted to tuple regardless of whether a list or tuple is
    passed).
  * Both attributes are correctly populated when called with known inputs.

TestRaiseExtractionErrorMessageHandling
  * Custom ``message`` is used as the exception message when provided.
  * A default message containing the URL is generated when ``message`` is
    omitted or empty.
  * Exception message is always non-empty.

TestRaiseExtractionErrorCauseChaining
  * When ``cause`` is provided, the raised ``ExtractionError.__cause__``
    equals the supplied cause object (``raise exc from cause`` chain).
  * When ``cause`` is ``None``, no explicit cause chain is set.
"""

from __future__ import annotations

import pytest

from articula._scraper import _raise_extraction_error
from articula.exceptions import ExtractionError, ScraperError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article"
_CUSTOM_MSG = "trafilatura returned no content"


# ---------------------------------------------------------------------------
# Class 1 — Always raises
# ---------------------------------------------------------------------------


class TestRaiseExtractionErrorAlwaysRaises:
    """``_raise_extraction_error`` is a NoReturn helper — it always raises."""

    def test_raises_extraction_error(self) -> None:
        """The function raises ``ExtractionError`` with minimal inputs."""
        with pytest.raises(ExtractionError):
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static"],
            )

    def test_never_returns_normally(self) -> None:
        """Calling the function never returns — it always raises."""
        raised = False
        try:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=[])
        except ExtractionError:
            raised = True
        assert raised, "_raise_extraction_error must always raise ExtractionError"

    def test_raised_exception_is_exactly_extraction_error(self) -> None:
        """The raised exception type is ``ExtractionError``, not a base class."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static", "headers_rotation"],
            )
        assert type(exc_info.value) is ExtractionError, (
            f"Expected ExtractionError, got {type(exc_info.value).__name__}"
        )

    def test_catchable_as_scraper_error(self) -> None:
        """``ExtractionError`` IS-A ``ScraperError`` — broad catch works."""
        with pytest.raises(ScraperError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=["static"])
        assert isinstance(exc_info.value, ExtractionError)


# ---------------------------------------------------------------------------
# Class 2 — Attributes populated from extraction context
# ---------------------------------------------------------------------------


class TestRaiseExtractionErrorAttributesFromContext:
    """
    ``url`` and ``strategies_attempted`` are correctly populated from the
    arguments passed to ``_raise_extraction_error``.

    These are the core Sub-AC 11.2 assertions: call the function with known
    inputs and assert the raised exception carries the correct attribute values.
    """

    def test_url_attribute_populated_correctly(self) -> None:
        """``ExtractionError.url`` matches the ``url`` argument."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static"],
            )
        assert exc_info.value.url == _TEST_URL, (
            f"Expected url={_TEST_URL!r}, got {exc_info.value.url!r}"
        )

    def test_strategies_attempted_attribute_populated_correctly(self) -> None:
        """``ExtractionError.strategies_attempted`` matches the input list."""
        strategies = ["static", "headers_rotation", "browser"]
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=strategies,
            )
        assert exc_info.value.strategies_attempted == tuple(strategies), (
            f"Expected {tuple(strategies)!r}, "
            f"got {exc_info.value.strategies_attempted!r}"
        )

    def test_both_attributes_populated_with_known_inputs(self) -> None:
        """
        Core Sub-AC 11.2 assertion: call with known url + strategies_attempted
        and verify both attributes on the raised exception.
        """
        url = "https://news.example.com/test-article-42"
        strategies = ["static", "headers_rotation"]

        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=url, strategies_attempted=strategies)

        exc = exc_info.value
        assert exc.url == url, (
            f"url attribute must match input; expected {url!r}, got {exc.url!r}"
        )
        assert exc.strategies_attempted == ("static", "headers_rotation"), (
            f"strategies_attempted must match input; "
            f"got {exc.strategies_attempted!r}"
        )

    def test_strategies_attempted_is_tuple_not_list(self) -> None:
        """``ExtractionError.strategies_attempted`` is an immutable tuple."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static", "browser"],
            )
        assert isinstance(exc_info.value.strategies_attempted, tuple), (
            "strategies_attempted must be stored as a tuple"
        )

    def test_strategies_attempted_all_three_tiers(self) -> None:
        """All three escalation tiers are preserved in the exact input order."""
        strategies = ["static", "headers_rotation", "browser"]
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=strategies)
        assert exc_info.value.strategies_attempted == (
            "static",
            "headers_rotation",
            "browser",
        )

    def test_strategies_attempted_empty_list(self) -> None:
        """An empty strategies list is stored as an empty tuple."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=[])
        assert exc_info.value.strategies_attempted == ()

    def test_strategies_attempted_single_tier(self) -> None:
        """A single-element strategies list is stored as a single-element tuple."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=["static"])
        assert exc_info.value.strategies_attempted == ("static",)

    def test_different_urls_produce_different_attribute_values(self) -> None:
        """Each call uses the exact url argument — not a cached or default value."""
        url_a = "https://site-a.example.com/article"
        url_b = "https://site-b.example.com/post"

        with pytest.raises(ExtractionError) as exc_a:
            _raise_extraction_error(url=url_a, strategies_attempted=["static"])
        with pytest.raises(ExtractionError) as exc_b:
            _raise_extraction_error(url=url_b, strategies_attempted=["static"])

        assert exc_a.value.url == url_a
        assert exc_b.value.url == url_b
        assert exc_a.value.url != exc_b.value.url


# ---------------------------------------------------------------------------
# Class 3 — Message handling
# ---------------------------------------------------------------------------


class TestRaiseExtractionErrorMessageHandling:
    """The exception message is populated from the ``message`` argument."""

    def test_custom_message_used_when_provided(self) -> None:
        """The supplied message appears as the exception message."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static"],
                message=_CUSTOM_MSG,
            )
        assert _CUSTOM_MSG in str(exc_info.value), (
            f"Custom message {_CUSTOM_MSG!r} must appear in str(exc); "
            f"got {str(exc_info.value)!r}"
        )

    def test_default_message_generated_when_omitted(self) -> None:
        """A default message is generated when ``message`` is not provided."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=[])
        assert str(exc_info.value), "Exception message must not be empty"

    def test_default_message_contains_url(self) -> None:
        """The auto-generated message includes the failing URL for diagnostics."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=["static"])
        assert _TEST_URL in str(exc_info.value), (
            f"Default message must reference the URL {_TEST_URL!r}"
        )

    def test_message_never_empty(self) -> None:
        """Exception message is non-empty regardless of arguments."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url="", strategies_attempted=[])
        assert str(exc_info.value), "Exception message must not be empty"


# ---------------------------------------------------------------------------
# Class 4 — Cause chaining
# ---------------------------------------------------------------------------


class TestRaiseExtractionErrorCauseChaining:
    """
    When ``cause`` is provided, ``raise exc from cause`` sets ``__cause__``
    so the original traceback is preserved.
    """

    def test_cause_chained_when_provided(self) -> None:
        """``ExtractionError.__cause__`` equals the supplied ``cause`` object."""
        original = ValueError("trafilatura returned empty string")
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static"],
                message=str(original),
                cause=original,
            )
        assert exc_info.value.__cause__ is original, (
            "ExtractionError.__cause__ must be the supplied cause object"
        )

    def test_cause_none_by_default(self) -> None:
        """When ``cause`` is omitted, ``ExtractionError.__cause__`` is ``None``."""
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(url=_TEST_URL, strategies_attempted=["static"])
        assert exc_info.value.__cause__ is None, (
            "ExtractionError.__cause__ must be None when no cause is provided"
        )

    def test_cause_type_preserved(self) -> None:
        """The type of the cause is not altered by the chaining."""
        original = ValueError("parse error")
        with pytest.raises(ExtractionError) as exc_info:
            _raise_extraction_error(
                url=_TEST_URL,
                strategies_attempted=["static"],
                cause=original,
            )
        assert isinstance(exc_info.value.__cause__, ValueError)
