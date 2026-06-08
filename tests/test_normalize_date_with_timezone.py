"""
Unit tests for :func:`articula._extractor.normalize_date_with_timezone`.

Sub-AC 2: Implement a ``normalize_date_with_timezone(raw: str) -> str`` function
that accepts date strings containing timezone info (e.g.,
'2024-01-05T10:30:00+05:30', 'Mon, 05 Jan 2024 10:30:00 GMT') and returns a
timezone-aware ISO 8601 string, with unit tests verifying UTC normalization and
offset preservation.

Coverage matrix
---------------
UTC normalization
    - ISO 8601 with Z suffix           → ``+00:00``
    - ISO 8601 with ``+00:00`` offset  → ``+00:00``
    - RFC 2822 with ``GMT`` label      → ``+00:00``
    - RFC 2822 with ``UTC`` label      → ``+00:00``
    - RFC 2822 with ``+0000``          → ``+00:00``

Offset preservation
    - Positive offset  ``+05:30``      → preserved verbatim
    - Positive offset  ``+09:00``      → preserved verbatim
    - Negative offset  ``-08:00``      → preserved verbatim
    - RFC 2822 with ``+0900``          → ``+09:00``

Timezone-naive inputs → None
    - Plain ISO 8601 date ``YYYY-MM-DD``
    - Long-form English date ('January 5, 2024')

Null / empty / whitespace inputs → None
    - None, empty string, whitespace-only

Invalid inputs → None
    - Completely unparseable strings

Return-type contract
    - Always returns ``str`` or ``None``, never raises

Public API importability
    - Importable from top-level package
    - Present in ``articula.__all__``

All tests are offline (no network access).
"""

from __future__ import annotations

import re

import pytest

from articula._extractor import normalize_date_with_timezone

# Regex for a timezone-aware ISO 8601 datetime.
# Matches: YYYY-MM-DDTHH:MM:SS followed by +HH:MM or -HH:MM
_AWARE_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
)


# ---------------------------------------------------------------------------
# UTC normalization
# ---------------------------------------------------------------------------


class TestUtcNormalization:
    """Inputs that reference UTC must produce the ``+00:00`` suffix."""

    def test_iso8601_z_suffix(self) -> None:
        """ISO 8601 'Z' is normalised to +00:00."""
        result = normalize_date_with_timezone("2024-01-05T10:30:00Z")
        assert result == "2024-01-05T10:30:00+00:00"

    def test_iso8601_plus_zero_offset(self) -> None:
        """Explicit +00:00 offset is returned as +00:00."""
        result = normalize_date_with_timezone("2024-01-05T10:30:00+00:00")
        assert result == "2024-01-05T10:30:00+00:00"

    def test_rfc2822_gmt_label(self) -> None:
        """RFC 2822 with GMT label → +00:00."""
        result = normalize_date_with_timezone("Mon, 05 Jan 2024 10:30:00 GMT")
        assert result == "2024-01-05T10:30:00+00:00"

    def test_rfc2822_utc_label(self) -> None:
        """RFC 2822 with UTC label → +00:00."""
        result = normalize_date_with_timezone("Mon, 05 Jan 2024 10:30:00 UTC")
        assert result == "2024-01-05T10:30:00+00:00"

    def test_rfc2822_plus0000_numeric(self) -> None:
        """RFC 2822 with +0000 numeric offset → +00:00."""
        result = normalize_date_with_timezone("Fri, 05 Jan 2024 10:30:00 +0000")
        assert result == "2024-01-05T10:30:00+00:00"

    def test_iso8601_midnight_utc(self) -> None:
        """Midnight UTC is handled correctly."""
        result = normalize_date_with_timezone("2024-01-05T00:00:00Z")
        assert result == "2024-01-05T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Offset preservation
# ---------------------------------------------------------------------------


class TestOffsetPreservation:
    """Non-UTC offsets must be preserved exactly in the returned string."""

    def test_positive_half_hour_offset(self) -> None:
        """+05:30 (IST) is preserved verbatim."""
        result = normalize_date_with_timezone("2024-01-05T10:30:00+05:30")
        assert result == "2024-01-05T10:30:00+05:30"

    def test_positive_hour_offset_kst(self) -> None:
        """+09:00 (KST) is preserved verbatim."""
        result = normalize_date_with_timezone("2024-01-05T10:30:00+09:00")
        assert result == "2024-01-05T10:30:00+09:00"

    def test_negative_hour_offset_pst(self) -> None:
        """-08:00 (PST) is preserved verbatim."""
        result = normalize_date_with_timezone("2024-01-05T10:30:00-08:00")
        assert result == "2024-01-05T10:30:00-08:00"

    def test_rfc2822_kst_numeric_offset(self) -> None:
        """RFC 2822 +0900 (no colon) is converted to canonical +09:00."""
        result = normalize_date_with_timezone("Tue, 05 Mar 2024 09:00:00 +0900")
        assert result == "2024-03-05T09:00:00+09:00"

    def test_positive_offset_different_date(self) -> None:
        """+05:30 on a different date is preserved."""
        result = normalize_date_with_timezone("2023-12-31T23:59:00+05:30")
        assert result == "2023-12-31T23:59:00+05:30"

    def test_negative_half_hour_offset(self) -> None:
        """-05:30 negative half-hour offset is preserved."""
        result = normalize_date_with_timezone("2024-06-15T08:00:00-05:30")
        assert result == "2024-06-15T08:00:00-05:30"


# ---------------------------------------------------------------------------
# Timezone-naive inputs → None
# ---------------------------------------------------------------------------


class TestTimezoneNaiveInputs:
    """Datetime strings without timezone info must return None."""

    def test_plain_iso_date_returns_none(self) -> None:
        """'YYYY-MM-DD' has no time or timezone — returns None."""
        assert normalize_date_with_timezone("2024-01-05") is None

    def test_long_form_date_returns_none(self) -> None:
        """'January 5, 2024' is naive — returns None."""
        assert normalize_date_with_timezone("January 5, 2024") is None

    def test_slash_date_returns_none(self) -> None:
        """'01/05/2024' is naive — returns None."""
        assert normalize_date_with_timezone("01/05/2024") is None


# ---------------------------------------------------------------------------
# Null / empty / whitespace inputs → None
# ---------------------------------------------------------------------------


class TestNullAndEmptyInputs:
    """None, empty strings, and whitespace-only strings must return None."""

    def test_none_returns_none(self) -> None:
        assert normalize_date_with_timezone(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_date_with_timezone("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_date_with_timezone("   ") is None

    def test_tab_newline_returns_none(self) -> None:
        assert normalize_date_with_timezone("\t\n") is None


# ---------------------------------------------------------------------------
# Invalid inputs → None
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    """Completely unparseable strings must return None without raising."""

    def test_plain_text_returns_none(self) -> None:
        assert normalize_date_with_timezone("not a date at all") is None

    def test_single_word_returns_none(self) -> None:
        assert normalize_date_with_timezone("hello") is None

    def test_random_numbers_returns_none(self) -> None:
        assert normalize_date_with_timezone("12345") is None


# ---------------------------------------------------------------------------
# Return-type contract: never raises, always str | None
# ---------------------------------------------------------------------------


class TestReturnTypeContract:
    """Return value is always str or None; the function never raises."""

    @pytest.mark.parametrize(
        "bad_input",
        [None, "", "   ", "not a date", "12345", "\t\n", "2024-01-05"],
    )
    def test_non_tz_aware_input_returns_none_not_raises(
        self, bad_input: str | None
    ) -> None:
        """Invalid or naive input returns None, not an exception."""
        result = normalize_date_with_timezone(bad_input)
        assert result is None

    @pytest.mark.parametrize(
        "good_input",
        [
            "2024-01-05T10:30:00+05:30",
            "2024-01-05T10:30:00Z",
            "Mon, 05 Jan 2024 10:30:00 GMT",
            "2024-01-05T10:30:00+09:00",
            "2024-01-05T10:30:00-08:00",
        ],
    )
    def test_tz_aware_input_returns_str(self, good_input: str) -> None:
        """Timezone-aware input always returns a str."""
        result = normalize_date_with_timezone(good_input)
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "good_input",
        [
            "2024-01-05T10:30:00+05:30",
            "2024-01-05T10:30:00Z",
            "Mon, 05 Jan 2024 10:30:00 GMT",
        ],
    )
    def test_tz_aware_input_matches_iso8601_pattern(self, good_input: str) -> None:
        """Return value matches YYYY-MM-DDTHH:MM:SS±HH:MM pattern."""
        result = normalize_date_with_timezone(good_input)
        assert result is not None, f"Expected str, got None for input {good_input!r}"
        assert _AWARE_ISO_RE.fullmatch(result), (
            f"Expected YYYY-MM-DDTHH:MM:SS±HH:MM, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Parametrized matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # UTC via Z
        ("2024-01-05T10:30:00Z", "2024-01-05T10:30:00+00:00"),
        # UTC via explicit +00:00
        ("2024-01-05T10:30:00+00:00", "2024-01-05T10:30:00+00:00"),
        # UTC via GMT label (RFC 2822)
        ("Mon, 05 Jan 2024 10:30:00 GMT", "2024-01-05T10:30:00+00:00"),
        # UTC via +0000 numeric (RFC 2822)
        ("Fri, 05 Jan 2024 10:30:00 +0000", "2024-01-05T10:30:00+00:00"),
        # Positive offset +05:30 (IST)
        ("2024-01-05T10:30:00+05:30", "2024-01-05T10:30:00+05:30"),
        # Positive offset +09:00 (KST)
        ("2024-01-05T10:30:00+09:00", "2024-01-05T10:30:00+09:00"),
        # Negative offset -08:00 (PST)
        ("2024-01-05T10:30:00-08:00", "2024-01-05T10:30:00-08:00"),
        # RFC 2822 +0900 numeric → canonical +09:00
        ("Tue, 05 Mar 2024 09:00:00 +0900", "2024-03-05T09:00:00+09:00"),
    ],
    ids=[
        "utc-z-suffix",
        "utc-plus-zero",
        "utc-gmt-label",
        "utc-plus0000",
        "offset-plus-0530",
        "offset-plus-0900",
        "offset-minus-0800",
        "rfc2822-plus0900",
    ],
)
def test_normalize_date_with_timezone_parametrized(raw: str, expected: str) -> None:
    """Parametrized coverage across UTC normalization and offset preservation."""
    assert normalize_date_with_timezone(raw) == expected


# ---------------------------------------------------------------------------
# Public API importability
# ---------------------------------------------------------------------------


class TestPublicApiImportability:
    """normalize_date_with_timezone must be importable from the top-level package."""

    def test_importable_from_package(self) -> None:
        """Accessible via ``from articula import normalize_date_with_timezone``."""
        from articula import normalize_date_with_timezone as ndtz  # noqa: PLC0415

        assert callable(ndtz)

    def test_package_function_works(self) -> None:
        """Package-level import behaves identically to the module-level one."""
        from articula import normalize_date_with_timezone as ndtz  # noqa: PLC0415

        assert ndtz("2024-01-05T10:30:00+05:30") == "2024-01-05T10:30:00+05:30"
        assert ndtz("2024-01-05T10:30:00Z") == "2024-01-05T10:30:00+00:00"
        assert ndtz(None) is None

    def test_in_all_list(self) -> None:
        """normalize_date_with_timezone is listed in articula.__all__."""
        import articula  # noqa: PLC0415

        assert "normalize_date_with_timezone" in articula.__all__
