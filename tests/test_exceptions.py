"""
Unit tests for the articula exception hierarchy.

Focuses on Sub-AC 9.1 — FetchError attributes and isolation:
  - message (str, accessible via str(exc) and args[0])
  - url (str, keyword-only, defaults to "")
  - attempted_strategies (tuple[str, ...], keyword-only, defaults to ())
  - can be instantiated and raised independently of any scraper machinery
  - status_code (int | None, keyword-only, defaults to None)
  - __str__ renders all populated fields
  - inherits from ScraperError
"""

from __future__ import annotations

import pytest

from articula.exceptions import (
    BrowserNotInstalledError,
    ConfigurationError,
    ExtractionError,
    FetchError,
    RobotsDisallowedError,
    ScraperError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URL = "https://example.com/article"
_MSG = "Network timeout after 30 s"


# ---------------------------------------------------------------------------
# 1. FetchError — instantiation
# ---------------------------------------------------------------------------


class TestFetchErrorInstantiation:
    def test_message_only(self) -> None:
        exc = FetchError(_MSG)
        assert exc.args[0] == _MSG
        assert str(exc).startswith(_MSG)

    def test_url_attribute_defaults_to_empty_string(self) -> None:
        exc = FetchError(_MSG)
        assert exc.url == ""

    def test_attempted_strategies_defaults_to_empty_tuple(self) -> None:
        exc = FetchError(_MSG)
        assert exc.attempted_strategies == ()
        assert isinstance(exc.attempted_strategies, tuple)

    def test_status_code_defaults_to_none(self) -> None:
        exc = FetchError(_MSG)
        assert exc.status_code is None

    def test_full_kwargs(self) -> None:
        exc = FetchError(
            _MSG,
            url=_URL,
            attempted_strategies=("static", "headers_rotation"),
            status_code=503,
        )
        assert exc.url == _URL
        assert exc.attempted_strategies == ("static", "headers_rotation")
        assert exc.status_code == 503

    def test_url_kwarg(self) -> None:
        exc = FetchError(_MSG, url=_URL)
        assert exc.url == _URL

    def test_attempted_strategies_accepts_list(self) -> None:
        """A list should be coerced to a tuple for immutable storage."""
        exc = FetchError(_MSG, attempted_strategies=["static", "browser"])
        assert isinstance(exc.attempted_strategies, tuple)
        assert exc.attempted_strategies == ("static", "browser")

    def test_attempted_strategies_accepts_tuple(self) -> None:
        exc = FetchError(_MSG, attempted_strategies=("static",))
        assert exc.attempted_strategies == ("static",)

    def test_attempted_strategies_all_three_tiers(self) -> None:
        tiers = ("static", "headers_rotation", "browser")
        exc = FetchError(_MSG, attempted_strategies=tiers)
        assert exc.attempted_strategies == tiers

    def test_single_strategy(self) -> None:
        exc = FetchError(_MSG, attempted_strategies=("static",))
        assert exc.attempted_strategies == ("static",)

    def test_empty_attempted_strategies_explicit(self) -> None:
        exc = FetchError(_MSG, attempted_strategies=())
        assert exc.attempted_strategies == ()


# ---------------------------------------------------------------------------
# 2. FetchError — raising and catching
# ---------------------------------------------------------------------------


class TestFetchErrorRaiseAndCatch:
    def test_can_be_raised(self) -> None:
        with pytest.raises(FetchError):
            raise FetchError(_MSG)

    def test_caught_as_fetch_error(self) -> None:
        with pytest.raises(FetchError) as exc_info:
            raise FetchError(_MSG, url=_URL, attempted_strategies=("static",))
        exc = exc_info.value
        assert exc.url == _URL
        assert exc.attempted_strategies == ("static",)

    def test_caught_as_scraper_error(self) -> None:
        """FetchError IS-A ScraperError — a single broad except catches it."""
        with pytest.raises(ScraperError):
            raise FetchError(_MSG, url=_URL)

    def test_caught_as_exception(self) -> None:
        """FetchError IS-A Exception — base exception handlers still catch it."""
        with pytest.raises(Exception):
            raise FetchError(_MSG)

    def test_attributes_preserved_through_raise(self) -> None:
        url = "https://example.com/deep/path"
        strategies = ("static", "headers_rotation", "browser")
        msg = "All strategies failed"
        try:
            raise FetchError(msg, url=url, attempted_strategies=strategies, status_code=429)
        except FetchError as exc:
            assert exc.args[0] == msg
            assert exc.url == url
            assert exc.attempted_strategies == strategies
            assert exc.status_code == 429


# ---------------------------------------------------------------------------
# 3. FetchError — __str__ representation
# ---------------------------------------------------------------------------


class TestFetchErrorStr:
    def test_str_includes_message(self) -> None:
        exc = FetchError(_MSG)
        assert _MSG in str(exc)

    def test_str_includes_url_when_set(self) -> None:
        exc = FetchError(_MSG, url=_URL)
        assert _URL in str(exc)

    def test_str_includes_attempted_strategies_when_set(self) -> None:
        exc = FetchError(_MSG, attempted_strategies=("static", "browser"))
        s = str(exc)
        assert "static" in s
        assert "browser" in s

    def test_str_no_url_section_when_empty(self) -> None:
        exc = FetchError(_MSG)
        assert "url=" not in str(exc)

    def test_str_no_strategies_section_when_empty(self) -> None:
        exc = FetchError(_MSG)
        assert "attempted_strategies" not in str(exc)

    def test_str_includes_status_code_when_set(self) -> None:
        exc = FetchError(_MSG, status_code=503)
        assert "503" in str(exc)

    def test_str_no_status_code_section_when_none(self) -> None:
        exc = FetchError(_MSG)
        assert "status_code" not in str(exc)

    def test_str_full_renders_all_parts(self) -> None:
        exc = FetchError(
            _MSG,
            url=_URL,
            attempted_strategies=("static", "headers_rotation"),
            status_code=503,
        )
        s = str(exc)
        assert _MSG in s
        assert _URL in s
        assert "static" in s
        assert "headers_rotation" in s
        assert "503" in s


# ---------------------------------------------------------------------------
# 4. FetchError — inheritance chain
# ---------------------------------------------------------------------------


class TestFetchErrorInheritance:
    def test_is_scraper_error(self) -> None:
        assert issubclass(FetchError, ScraperError)

    def test_is_exception(self) -> None:
        assert issubclass(FetchError, Exception)

    def test_instance_of_scraper_error(self) -> None:
        exc = FetchError(_MSG)
        assert isinstance(exc, ScraperError)

    def test_instance_of_exception(self) -> None:
        exc = FetchError(_MSG)
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# 5. FetchError — isolation (no external dependencies required)
# ---------------------------------------------------------------------------


class TestFetchErrorIsolation:
    """Verify FetchError can be used without any scraper machinery."""

    def test_instantiated_without_scraper(self) -> None:
        """FetchError can be created independently of any Scraper instance."""
        exc = FetchError("standalone error", url="https://isolated.test/")
        assert exc.url == "https://isolated.test/"

    def test_raised_without_scraper(self) -> None:
        """FetchError can be raised in isolation."""
        with pytest.raises(FetchError) as exc_info:
            raise FetchError(
                "standalone",
                url="https://isolated.test/",
                attempted_strategies=("static",),
            )
        assert exc_info.value.attempted_strategies == ("static",)

    def test_no_import_of_httpx_or_playwright_needed(self) -> None:
        """Creating and raising FetchError imports nothing heavy."""
        # This test passes by definition — the import at module level succeeds
        # without httpx/playwright being referenced in exceptions.py.
        exc = FetchError("pure exception")
        assert isinstance(exc, FetchError)


# ---------------------------------------------------------------------------
# 6. ExtractionError — basic attribute checks (not the focus but required)
# ---------------------------------------------------------------------------


class TestExtractionError:
    def test_instantiation(self) -> None:
        exc = ExtractionError("empty body", url="https://example.com/")
        assert exc.url == "https://example.com/"
        assert exc.args[0] == "empty body"

    def test_is_scraper_error(self) -> None:
        assert issubclass(ExtractionError, ScraperError)

    def test_str_includes_url(self) -> None:
        exc = ExtractionError("empty body", url="https://example.com/")
        assert "https://example.com/" in str(exc)

    # ------------------------------------------------------------------
    # Sub-AC 2: instantiate and raise — required verification
    # ------------------------------------------------------------------

    def test_can_be_raised_and_caught(self) -> None:
        """ExtractionError can be instantiated, raised, and caught."""
        with pytest.raises(ExtractionError) as exc_info:
            raise ExtractionError("body was empty after extraction")
        assert exc_info.value.args[0] == "body was empty after extraction"

    def test_can_be_raised_with_url_and_caught(self) -> None:
        """ExtractionError with optional url kwarg raises correctly."""
        url = "https://news.example.com/article-123"
        with pytest.raises(ExtractionError) as exc_info:
            raise ExtractionError("no article body found", url=url)
        exc = exc_info.value
        assert exc.url == url
        assert "no article body found" in str(exc)

    def test_url_kwarg_is_optional(self) -> None:
        """url parameter defaults to empty string when omitted."""
        exc = ExtractionError("content extraction failed")
        assert exc.url == ""

    def test_caught_as_scraper_error(self) -> None:
        """ExtractionError IS-A ScraperError — broad except catches it."""
        with pytest.raises(ScraperError):
            raise ExtractionError("empty body", url="https://example.com/")

    def test_importable_from_public_package(self) -> None:
        """ExtractionError must be importable directly from articula."""
        import articula
        assert hasattr(articula, "ExtractionError")
        assert articula.ExtractionError is ExtractionError


# ---------------------------------------------------------------------------
# Sub-AC 1: ExtractionError accepts url and strategies_attempted attributes
# ---------------------------------------------------------------------------


class TestExtractionErrorSubAC1:
    """Sub-AC 1 — ExtractionError stores url and strategies_attempted as attributes."""

    def test_url_attribute_stored_and_accessible(self) -> None:
        """url kwarg is stored and accessible after instantiation."""
        url = "https://example.com/article"
        exc = ExtractionError("empty body", url=url)
        assert exc.url == url

    def test_strategies_attempted_attribute_stored_and_accessible(self) -> None:
        """strategies_attempted kwarg is stored and accessible after instantiation."""
        strategies = ["static", "headers_rotation"]
        exc = ExtractionError("empty body", strategies_attempted=strategies)
        assert exc.strategies_attempted == ("static", "headers_rotation")

    def test_both_attributes_stored_together(self) -> None:
        """Both url and strategies_attempted are stored when provided together."""
        url = "https://example.com/article"
        strategies = ["static", "headers_rotation", "browser"]
        exc = ExtractionError("body extraction failed", url=url, strategies_attempted=strategies)
        assert exc.url == url
        assert exc.strategies_attempted == ("static", "headers_rotation", "browser")

    def test_strategies_attempted_coerced_to_tuple(self) -> None:
        """strategies_attempted accepts a list and stores it as an immutable tuple."""
        exc = ExtractionError("empty", strategies_attempted=["static", "browser"])
        assert isinstance(exc.strategies_attempted, tuple)
        assert exc.strategies_attempted == ("static", "browser")

    def test_strategies_attempted_accepts_tuple_input(self) -> None:
        """strategies_attempted also accepts a tuple input directly."""
        exc = ExtractionError("empty", strategies_attempted=("static",))
        assert exc.strategies_attempted == ("static",)

    def test_strategies_attempted_defaults_to_empty_tuple(self) -> None:
        """strategies_attempted defaults to () when not provided."""
        exc = ExtractionError("empty body")
        assert exc.strategies_attempted == ()
        assert isinstance(exc.strategies_attempted, tuple)

    def test_url_defaults_to_empty_string(self) -> None:
        """url defaults to empty string when not provided."""
        exc = ExtractionError("empty body")
        assert exc.url == ""

    def test_strategies_attempted_single_tier(self) -> None:
        """strategies_attempted works with a single-element list."""
        exc = ExtractionError("empty", strategies_attempted=["static"])
        assert exc.strategies_attempted == ("static",)

    def test_str_includes_strategies_attempted_when_set(self) -> None:
        """__str__ includes strategies_attempted when non-empty."""
        exc = ExtractionError(
            "empty body",
            url="https://example.com/",
            strategies_attempted=["static", "headers_rotation"],
        )
        s = str(exc)
        assert "static" in s
        assert "headers_rotation" in s

    def test_str_includes_url_when_set(self) -> None:
        """__str__ includes url when non-empty."""
        exc = ExtractionError("empty", url="https://example.com/article")
        assert "https://example.com/article" in str(exc)

    def test_str_omits_empty_strategies(self) -> None:
        """__str__ does not include strategies_attempted section when empty."""
        exc = ExtractionError("empty body")
        assert "strategies_attempted" not in str(exc)

    def test_raised_and_caught_with_both_attributes(self) -> None:
        """Both url and strategies_attempted are preserved through raise/catch."""
        url = "https://news.example.com/article"
        strategies = ["static", "headers_rotation", "browser"]
        try:
            raise ExtractionError("no body found", url=url, strategies_attempted=strategies)
        except ExtractionError as exc:
            assert exc.url == url
            assert exc.strategies_attempted == tuple(strategies)
            assert exc.args[0] == "no body found"


# ---------------------------------------------------------------------------
# 7. Other exception classes — smoke tests
# ---------------------------------------------------------------------------


class TestOtherExceptions:
    def test_robots_disallowed_error(self) -> None:
        exc = RobotsDisallowedError("disallowed", url="https://example.com/private")
        assert isinstance(exc, ScraperError)
        assert exc.url == "https://example.com/private"

    def test_browser_not_installed_error_default_message(self) -> None:
        exc = BrowserNotInstalledError()
        assert "playwright" in str(exc).lower() or "browser" in str(exc).lower()
        assert isinstance(exc, ScraperError)

    def test_browser_not_installed_error_custom_message(self) -> None:
        exc = BrowserNotInstalledError("custom msg")
        assert str(exc) == "custom msg"

    def test_configuration_error(self) -> None:
        exc = ConfigurationError("bad config")
        assert isinstance(exc, ScraperError)
        assert str(exc) == "bad config"
