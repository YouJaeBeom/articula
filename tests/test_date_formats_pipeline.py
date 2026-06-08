"""
Sub-AC 3c: Parameterised unit tests covering at least five distinct raw date
formats flowing through the full pipeline call.

Goal
----
Verify that the integration between the extraction pipeline and the real
``normalize_date`` normaliser correctly maps each of the five documented raw
date categories to a valid ISO 8601 ``published_date`` (or ``None`` for
genuinely unrecognisable inputs) on the final ``Article`` object.

Formats covered (≥ 5 distinct categories as required by Sub-AC 3c)
-------------------------------------------------------------------
1. **Already-ISO 8601** – ``'2024-01-05'`` (fast-path regex match in
   ``normalize_date``).
2. **RFC 2822** – ``'Fri, 05 Jan 2024 00:00:00 +0000'`` (email / HTTP header
   style; parsed by ``python-dateutil``).
3. **US locale long-form** – ``'January 5, 2024'`` (full month name; parsed by
   ``python-dateutil``).
4. **Epoch Unix timestamp** – ``'1704412800'`` (plain integer seconds since
   epoch; parsed by ``dateparser`` as a Unix timestamp).
5. **Partial date (month + year, no day)** – ``'January 2024'`` (``dateutil``
   defaults the missing day to today's calendar day; assertion is flexible: the
   result is always ``None`` or a valid ``YYYY-MM-DD`` string).
6. **ISO 8601 datetime with timezone offset** – ``'2024-01-05T10:30:00+09:00'``
   (fast-path extracts the leading ``YYYY-MM-DD`` fragment).
7. **US slash-separated MM/DD/YYYY** – ``'01/05/2024'`` (``dateutil`` with
   ``dayfirst=False`` → US convention).

The pipeline under test
-----------------------
::

    mock trafilatura.bare_extraction  →  raw_date injected as ``"date"``
        ↓
    extract(html, url)                →  ExtractionResult(published_date=<normalized>)
        ↓
    _build_article(fetch, extraction) →  Article(published_date=<normalized>)

``normalize_date`` is **NOT** mocked — the real implementation runs so that
the integration between the pipeline and the normaliser is exercised end-to-end.

All tests are offline (no network access).
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from articula._extractor import extract
from articula._fetcher import FetchResult
from articula._scraper import _build_article
from articula.models import Article

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article/sub-ac-3c-date-format-pipeline"

# Body text that satisfies _MIN_BODY_CHARS (80) and _MIN_BODY_TOKENS (10).
_LONG_TEXT = (
    "This article body is sufficiently long to pass all minimum extraction "
    "thresholds required by every extraction tier in the pipeline.  It "
    "contains well over eighty characters and more than ten whitespace-"
    "delimited tokens so trafilatura, readability, and heuristic all accept "
    "it as a valid article body."
)

_STUB_HTML = "<html><body>stub</body></html>"

# Compiled pattern used for ISO 8601 date validation.
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_trafilatura(raw_date: str) -> MagicMock:
    """
    Return a mock trafilatura module whose ``bare_extraction`` yields *raw_date*
    as the value of the ``"date"`` key.

    This injects the raw date string into the pipeline without requiring a live
    HTTP fetch and without mocking ``normalize_date`` itself.
    """
    mock = MagicMock()
    mock.bare_extraction.return_value = {
        "title": "Sub-AC 3c: Date Format Pipeline Integration Test Article",
        "text": _LONG_TEXT,
        "author": "Pipeline Test Author",
        "date": raw_date,
    }
    return mock


def _make_fetch_result() -> FetchResult:
    """Return a minimal ``FetchResult`` simulating a successful static fetch."""
    return FetchResult(
        html=_STUB_HTML,
        resolved_url=_TEST_URL,
        strategy_tier="static",
        status_code=200,
    )


def _pipeline(raw_date: str) -> Article:
    """
    Run the full extraction pipeline with *raw_date* as the raw date string
    from the trafilatura extractor.

    Specifically:
    1. ``extract(html, url)`` is called with the mocked trafilatura.
    2. ``_build_article(fetch_result, extraction, strategies)`` assembles the
       ``Article``.

    ``normalize_date`` is NOT intercepted — the real function runs inside
    ``_extract_trafilatura`` (called by ``extract``).
    """
    mock_trf = _make_mock_trafilatura(raw_date)
    fetch = _make_fetch_result()

    with patch.dict("sys.modules", {"trafilatura": mock_trf}):
        extraction = extract(_STUB_HTML, _TEST_URL)

    return _build_article(fetch, extraction, ["static"])


# ---------------------------------------------------------------------------
# Group A — Deterministic: five+ distinct formats → specific expected ISO date
# ---------------------------------------------------------------------------

# Each entry is (raw_date_string, expected_iso_date).
# The parametrize list covers ≥ 5 distinct format categories as mandated by
# Sub-AC 3c: already-ISO, RFC 2822, US locale, epoch timestamp, ISO datetime,
# US slash, and RFC 2822 with non-UTC timezone offset.
_DETERMINISTIC_CASES = [
    # ── 1. Already-ISO 8601 ────────────────────────────────────────────────
    # normalize_date fast-path: regex extracts YYYY-MM-DD and returns immediately
    # without invoking any external parser.
    ("2024-01-05", "2024-01-05"),

    # ── 2. RFC 2822 (UTC) ─────────────────────────────────────────────────
    # Classic email / HTTP-header date format; parsed by python-dateutil.
    ("Fri, 05 Jan 2024 00:00:00 +0000", "2024-01-05"),

    # ── 3. US locale long-form month name ────────────────────────────────
    # Full English month name with day and year; parsed by python-dateutil
    # with dayfirst=False (US convention).
    ("January 5, 2024", "2024-01-05"),

    # ── 4. Epoch Unix timestamp (plain integer seconds string) ────────────
    # dateutil cannot parse a raw 10-digit integer as a date;
    # dateparser recognises it as a Unix timestamp and converts it to
    # 2024-01-05 (= 1704412800 seconds since the Unix epoch).
    ("1704412800", "2024-01-05"),

    # ── 5. ISO 8601 datetime with non-UTC timezone offset ────────────────
    # normalize_date fast-path: the YYYY-MM-DD fragment is extracted directly
    # from the string by the leading regex, returning only the date portion.
    ("2024-01-05T10:30:00+09:00", "2024-01-05"),

    # ── 6. US slash-separated MM/DD/YYYY ─────────────────────────────────
    # dateutil parses '01/05/2024' with dayfirst=False → month=1, day=5, year=2024.
    ("01/05/2024", "2024-01-05"),

    # ── 7. RFC 2822 with non-UTC timezone offset (KST +0900) ─────────────
    # Confirms pipeline handles timezone-offset RFC 2822 strings from Asian
    # sources (Korean Standard Time) as an additional RFC 2822 variant.
    ("Tue, 05 Mar 2024 09:00:00 +0900", "2024-03-05"),
]


@pytest.mark.parametrize(
    ("raw_date", "expected_iso"),
    _DETERMINISTIC_CASES,
    ids=[
        "already-iso",          # 1
        "rfc2822-utc",          # 2
        "us-locale-long-form",  # 3
        "epoch-unix-timestamp", # 4
        "iso-datetime-with-tz", # 5
        "us-slash-mm-dd-yyyy",  # 6
        "rfc2822-kst-plus0900", # 7
    ],
)
def test_pipeline_normalizes_date_to_iso8601(raw_date: str, expected_iso: str) -> None:
    """
    Full pipeline integration test — normalize_date NOT mocked.

    For each of the five+ documented raw date format categories the pipeline
    must produce exactly *expected_iso* in ``Article.published_date``.

    The test exercises the real ``normalize_date`` normaliser called inside
    ``_extract_trafilatura``, verifying that normalization happens *before* the
    ``Article`` is constructed.
    """
    article = _pipeline(raw_date)

    assert isinstance(article, Article), (
        f"Pipeline did not return an Article for raw_date={raw_date!r}"
    )
    assert article.published_date == expected_iso, (
        f"Pipeline: raw_date={raw_date!r} → expected Article.published_date=="
        f"{expected_iso!r} but got {article.published_date!r}.\n"
        "normalize_date must convert the raw string to ISO 8601 inside the "
        "pipeline before assembling the Article object."
    )


# ---------------------------------------------------------------------------
# Group B — Partial date (month + year, no day) → None or valid ISO date
# ---------------------------------------------------------------------------
#
# When a raw date string has no explicit day component, python-dateutil
# fills the missing day from its ``default`` parameter, which defaults to
# today's date.  The exact YYYY-MM-DD value therefore changes depending on
# when the test suite is executed.
#
# The assertion confirms the pipeline:
#   a) does not raise, and
#   b) produces either None or a syntactically valid ISO 8601 date string
#      (not an arbitrary non-date string).


@pytest.mark.parametrize(
    "raw_date",
    [
        # Month name + year only — dateutil defaults the missing day to today's day.
        "January 2024",
        # Abbreviated month + year only.
        "Feb 2023",
    ],
    ids=[
        "partial-month-year-long",
        "partial-month-year-abbrev",
    ],
)
def test_pipeline_partial_date_yields_iso_or_none(raw_date: str) -> None:
    """
    Partial date: month + year with no explicit day component.

    dateutil fills the missing day with today's calendar day, making the exact
    output non-deterministic across test-run dates.  The assertion is therefore
    flexible: the result must be *None* OR a syntactically valid ``YYYY-MM-DD``
    string — never an invalid or non-ISO value.

    normalize_date is NOT mocked; the real normaliser runs inside the pipeline.
    """
    article = _pipeline(raw_date)

    assert isinstance(article, Article), (
        f"Pipeline did not return an Article for raw_date={raw_date!r}"
    )

    result = article.published_date

    # Accept None (normaliser could not produce a date) or a valid ISO date.
    if result is not None:
        assert _ISO_DATE_PATTERN.fullmatch(result), (
            f"Article.published_date for partial date {raw_date!r} is not a "
            f"valid ISO 8601 date string (YYYY-MM-DD): {result!r}"
        )


# ---------------------------------------------------------------------------
# Integrity check: article body/title remain populated regardless of date format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_date",
    [row[0] for row in _DETERMINISTIC_CASES] + ["January 2024", "Feb 2023"],
    ids=[
        "already-iso",
        "rfc2822-utc",
        "us-locale-long-form",
        "epoch-unix-timestamp",
        "iso-datetime-with-tz",
        "us-slash-mm-dd-yyyy",
        "rfc2822-kst-plus0900",
        "partial-month-year-long",
        "partial-month-year-abbrev",
    ],
)
def test_pipeline_article_body_populated_across_all_date_formats(
    raw_date: str,
) -> None:
    """
    Graceful integration check: regardless of the date format, the pipeline must
    always populate ``Article.title`` and ``Article.text``.

    Confirms that date-format normalisation does not interfere with body
    extraction — every date format variant yields a complete article.
    """
    article = _pipeline(raw_date)

    assert isinstance(article, Article)
    assert article.title.strip(), (
        f"Article.title must be non-empty for raw_date={raw_date!r}"
    )
    assert article.text.strip(), (
        f"Article.text must be non-empty for raw_date={raw_date!r}"
    )
