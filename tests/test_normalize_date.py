"""
Unit tests for :func:`articula._extractor.normalize_date`.

Sub-AC 1: Implement a ``normalize_date(raw: str) -> str`` function that accepts
a raw date string in common formats (e.g., 'January 5, 2024', '2024-01-05',
'05/01/2024', RFC 2822) and returns an ISO 8601 string ('2024-01-05'), with
unit tests covering each input format and invalid/None input handling.

All tests are offline (no network access).

Coverage matrix
---------------
- ISO 8601 date already in YYYY-MM-DD → fast-path return
- ISO 8601 datetime with time component → date portion extracted
- Long-form English month name: 'January 5, 2024'
- Abbreviated month name: 'Jan 5, 2024'
- Slash-separated MM/DD/YYYY (US convention): '01/05/2024'
- RFC 2822: 'Fri, 05 Jan 2024 00:00:00 +0000'
- None input → None
- Empty string → None
- Whitespace-only string → None
- Clearly invalid/nonsense string → None
- Public API importable from top-level package
"""

from __future__ import annotations

import pytest

from articula._extractor import normalize_date

# ---------------------------------------------------------------------------
# Happy-path: ISO 8601 formats (fast-path regex)
# ---------------------------------------------------------------------------


class TestIso8601Input:
    """YYYY-MM-DD inputs hit the fast-path regex and are returned as-is."""

    def test_plain_date_unchanged(self) -> None:
        """Canonical ISO 8601 date is returned without any transformation."""
        assert normalize_date("2024-01-05") == "2024-01-05"

    def test_plain_date_different_year(self) -> None:
        assert normalize_date("2020-12-31") == "2020-12-31"

    def test_iso8601_datetime_returns_date_portion(self) -> None:
        """Datetime with T-separator: only the date part is returned."""
        assert normalize_date("2024-01-05T10:30:00Z") == "2024-01-05"

    def test_iso8601_datetime_with_offset(self) -> None:
        """Datetime with UTC offset: only the date part is returned."""
        assert normalize_date("2024-01-05T10:30:00+09:00") == "2024-01-05"

    def test_iso8601_embedded_in_longer_string(self) -> None:
        """YYYY-MM-DD fragment embedded in prose is extracted via fast-path."""
        assert normalize_date("Published: 2024-01-05 – Article title") == "2024-01-05"


# ---------------------------------------------------------------------------
# Happy-path: long-form English month names
# ---------------------------------------------------------------------------


class TestLongFormEnglishMonthNames:
    """Formats like 'January 5, 2024' parsed by dateutil."""

    def test_full_month_name(self) -> None:
        assert normalize_date("January 5, 2024") == "2024-01-05"

    def test_full_month_name_february(self) -> None:
        assert normalize_date("February 28, 2023") == "2023-02-28"

    def test_full_month_name_december(self) -> None:
        assert normalize_date("December 31, 2024") == "2024-12-31"

    def test_abbreviated_month_name(self) -> None:
        """Three-letter month abbreviations: 'Jan 5, 2024'."""
        assert normalize_date("Jan 5, 2024") == "2024-01-05"

    def test_abbreviated_month_no_comma(self) -> None:
        """'5 Jan 2024' — no comma, day-first."""
        assert normalize_date("5 Jan 2024") == "2024-01-05"

    def test_month_day_year_with_ordinal(self) -> None:
        """'January 5th, 2024' — dateutil strips ordinal suffix."""
        result = normalize_date("January 5th, 2024")
        # dateutil can handle ordinal suffixes
        assert result == "2024-01-05"


# ---------------------------------------------------------------------------
# Happy-path: slash-separated dates (MM/DD/YYYY US convention)
# ---------------------------------------------------------------------------


class TestSlashSeparatedDates:
    """MM/DD/YYYY inputs are parsed with dayfirst=False (US convention)."""

    def test_mm_dd_yyyy(self) -> None:
        """'01/05/2024' → January 5, 2024 under dayfirst=False."""
        # With dayfirst=False, 01/05/2024 → month=1, day=5 → 2024-01-05
        assert normalize_date("01/05/2024") == "2024-01-05"

    def test_mm_dd_yyyy_different_date(self) -> None:
        """'12/25/2023' → December 25, 2023."""
        assert normalize_date("12/25/2023") == "2023-12-25"

    def test_year_first_slash(self) -> None:
        """'2024/01/05' — year-first slash format."""
        assert normalize_date("2024/01/05") == "2024-01-05"


# ---------------------------------------------------------------------------
# Happy-path: RFC 2822 / email-style dates
# ---------------------------------------------------------------------------


class TestRfc2822:
    """RFC 2822 dates as found in HTTP headers and RSS feeds."""

    def test_canonical_rfc2822(self) -> None:
        """'Fri, 05 Jan 2024 00:00:00 +0000' → '2024-01-05'."""
        assert normalize_date("Fri, 05 Jan 2024 00:00:00 +0000") == "2024-01-05"

    def test_rfc2822_without_weekday(self) -> None:
        """RFC 2822 without day-of-week prefix."""
        assert normalize_date("05 Jan 2024 00:00:00 +0000") == "2024-01-05"

    def test_rfc2822_with_utc_label(self) -> None:
        """'05 Jan 2024 12:00:00 UTC' — UTC label instead of +0000."""
        assert normalize_date("05 Jan 2024 12:00:00 UTC") == "2024-01-05"

    def test_rfc2822_timezone_offset(self) -> None:
        """'Tue, 05 Mar 2024 09:00:00 +0900' — KST-style offset."""
        assert normalize_date("Tue, 05 Mar 2024 09:00:00 +0900") == "2024-03-05"


# ---------------------------------------------------------------------------
# Null / empty / invalid inputs → None
# ---------------------------------------------------------------------------


class TestNullAndEmptyInputs:
    """None, empty strings, and whitespace-only strings must return None."""

    def test_none_returns_none(self) -> None:
        """None input returns None (not an exception)."""
        assert normalize_date(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_date("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_date("   ") is None

    def test_tab_only_returns_none(self) -> None:
        assert normalize_date("\t\n") is None


class TestInvalidInputs:
    """Nonsense or ambiguous strings that cannot be parsed return None."""

    def test_plain_text_returns_none(self) -> None:
        assert normalize_date("not a date at all") is None

    def test_single_word_returns_none(self) -> None:
        assert normalize_date("hello") is None

    def test_random_numbers_returns_none(self) -> None:
        assert normalize_date("12345") is None

    def test_partial_date_no_year_returns_none(self) -> None:
        """'January 5' with no year is ambiguous; result may vary."""
        # dateutil may infer current year — we just confirm it doesn't raise.
        result = normalize_date("January 5")
        # Either None or a valid YYYY-MM-DD string is acceptable.
        assert result is None or (isinstance(result, str) and len(result) == 10)


# ---------------------------------------------------------------------------
# Parametrized matrix — all supported format categories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # ISO 8601
        ("2024-01-05", "2024-01-05"),
        ("2020-06-15", "2020-06-15"),
        # Long-form English
        ("January 5, 2024", "2024-01-05"),
        ("Jan 5, 2024", "2024-01-05"),
        ("March 15, 2023", "2023-03-15"),
        # Slash MM/DD/YYYY
        ("01/05/2024", "2024-01-05"),
        ("03/15/2023", "2023-03-15"),
        # RFC 2822
        ("Fri, 05 Jan 2024 00:00:00 +0000", "2024-01-05"),
        ("Mon, 15 Mar 2021 08:30:00 +0000", "2021-03-15"),
    ],
    ids=[
        "iso8601-plain",
        "iso8601-june",
        "long-form-january",
        "abbrev-jan",
        "long-form-march",
        "slash-01-05-2024",
        "slash-03-15-2023",
        "rfc2822-jan",
        "rfc2822-mar",
    ],
)
def test_normalize_date_parametrized(raw: str, expected: str) -> None:
    """Parametrized coverage across all documented input categories."""
    assert normalize_date(raw) == expected


# ---------------------------------------------------------------------------
# Public API importability
# ---------------------------------------------------------------------------


class TestPublicApiImportability:
    """normalize_date must be importable from the top-level package."""

    def test_importable_from_package(self) -> None:
        """normalize_date is accessible via ``from articula import normalize_date``."""
        from articula import normalize_date as nd  # noqa: PLC0415

        assert callable(nd)

    def test_package_normalize_date_works(self) -> None:
        """normalize_date imported from the package behaves identically."""
        from articula import normalize_date as nd  # noqa: PLC0415

        assert nd("2024-01-05") == "2024-01-05"
        assert nd(None) is None

    def test_in_all_list(self) -> None:
        """normalize_date is listed in articula.__all__."""
        import articula  # noqa: PLC0415

        assert "normalize_date" in articula.__all__


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


class TestReturnTypeContract:
    """Return value is always str or None; never raises on bad input."""

    @pytest.mark.parametrize(
        "bad_input",
        [None, "", "   ", "not a date", "12345", "\t\n"],
    )
    def test_invalid_input_returns_none_not_raises(self, bad_input: str | None) -> None:
        """Invalid input should return None, not raise any exception."""
        result = normalize_date(bad_input)
        assert result is None

    @pytest.mark.parametrize(
        "good_input",
        [
            "2024-01-05",
            "January 5, 2024",
            "Fri, 05 Jan 2024 00:00:00 +0000",
        ],
    )
    def test_valid_input_returns_str(self, good_input: str) -> None:
        """Valid input should return a str."""
        result = normalize_date(good_input)
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "good_input",
        [
            "2024-01-05",
            "January 5, 2024",
            "Fri, 05 Jan 2024 00:00:00 +0000",
        ],
    )
    def test_valid_input_returns_iso8601_format(self, good_input: str) -> None:
        """Return value matches YYYY-MM-DD exactly."""
        import re  # noqa: PLC0415

        result = normalize_date(good_input)
        assert result is not None
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", result), (
            f"Expected YYYY-MM-DD, got {result!r}"
        )
