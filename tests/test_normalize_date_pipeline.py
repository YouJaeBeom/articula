"""
Sub-AC 3a: normalize_date is wired into the article extraction pipeline.

Verifies that when the raw trafilatura extractor returns a non-ISO date string
(e.g. "January 5, 2024"), the pipeline normalizes it to ISO 8601 ("2024-01-05")
before constructing the Article object.

The pipeline under test:
    trafilatura.bare_extraction → {"date": "<non-ISO>", ...}  (mocked)
    ↓
    _extract_trafilatura: calls normalize_date(result.get("date"))
    ↓
    ExtractionResult(published_date="<ISO>", ...)
    ↓
    _build_article → Article(published_date="<ISO>")

All tests are offline (no network access).  The raw trafilatura extractor is
replaced by a MagicMock so the test controls the raw date string exactly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from articula._extractor import ExtractionResult, extract
from articula._fetcher import FetchResult
from articula._scraper import _build_article
from articula.models import Article

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article/normalize-date-pipeline"

# Body text that passes _MIN_BODY_CHARS (80) and _MIN_BODY_TOKENS (10).
_LONG_TEXT = (
    "This is a sufficiently long article body text that passes the "
    "minimum character and token thresholds required by every extraction "
    "tier in the pipeline.  It contains more than eighty characters and "
    "more than ten whitespace-delimited tokens, making it a valid body."
)

# Minimal HTML stub — trafilatura is mocked so the actual HTML is irrelevant.
_STUB_HTML = "<html><body>stub</body></html>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_trafilatura(raw_date: str) -> MagicMock:
    """Return a mock trafilatura module whose bare_extraction returns *raw_date*."""
    mock = MagicMock()
    mock.bare_extraction.return_value = {
        "title": "Test Article About Artificial Intelligence",
        "text": _LONG_TEXT,
        "author": "Jane Doe",
        "date": raw_date,
    }
    return mock


def _make_fetch_result() -> FetchResult:
    """Return a minimal FetchResult for the static tier."""
    return FetchResult(
        html=_STUB_HTML,
        resolved_url=_TEST_URL,
        strategy_tier="static",
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Core test: normalize_date wired into the pipeline
# ---------------------------------------------------------------------------


class TestNormalizeDateInExtractionPipeline:
    """
    normalize_date is called on the raw date string within the extraction
    pipeline so that the constructed Article always carries an ISO 8601 date.
    """

    def test_non_iso_date_normalized_in_extraction_result(self) -> None:
        """
        'January 5, 2024' from raw extractor → ExtractionResult.published_date == '2024-01-05'.

        Verifies the normalization happens inside extract() before the
        ExtractionResult is constructed — the raw string never reaches the Article.
        """
        mock_trf = _make_mock_trafilatura("January 5, 2024")

        with patch.dict("sys.modules", {"trafilatura": mock_trf}):
            result = extract(_STUB_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)
        assert result.published_date == "2024-01-05", (
            f"Expected '2024-01-05' but got {result.published_date!r}"
        )

    def test_non_iso_date_normalized_before_article_construction(self) -> None:
        """
        End-to-end pipeline: raw non-ISO date → Article.published_date in ISO 8601.

        Asserts that normalize_date is called on the raw string from the raw extractor
        *before* the structured Article object is constructed.
        """
        mock_trf = _make_mock_trafilatura("January 5, 2024")
        fetch = _make_fetch_result()

        with patch.dict("sys.modules", {"trafilatura": mock_trf}):
            extraction = extract(_STUB_HTML, _TEST_URL)

        article = _build_article(fetch, extraction, ["static"])

        assert isinstance(article, Article)
        assert article.published_date == "2024-01-05", (
            f"Expected ISO date '2024-01-05' but got {article.published_date!r}. "
            "normalize_date must be called in the pipeline before Article construction."
        )


# ---------------------------------------------------------------------------
# Parametrized: all common non-ISO input formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_date", "expected_iso"),
    [
        ("January 5, 2024", "2024-01-05"),
        ("Jan 5, 2024", "2024-01-05"),
        ("Fri, 05 Jan 2024 00:00:00 +0000", "2024-01-05"),
        ("01/05/2024", "2024-01-05"),
    ],
    ids=["long-form-month", "abbreviated-month", "rfc2822", "slash-mm-dd-yyyy"],
)
def test_normalize_date_wired_for_various_non_iso_formats(
    raw_date: str, expected_iso: str,
) -> None:
    """
    normalize_date is wired into the pipeline for all common non-ISO input formats.

    The raw extractor mock returns *raw_date* and the assembled Article
    must carry *expected_iso* (YYYY-MM-DD) in its published_date field.
    """
    mock_trf = _make_mock_trafilatura(raw_date)
    fetch = _make_fetch_result()

    with patch.dict("sys.modules", {"trafilatura": mock_trf}):
        extraction = extract(_STUB_HTML, _TEST_URL)

    article = _build_article(fetch, extraction, ["static"])

    assert isinstance(article, Article)
    assert article.published_date == expected_iso, (
        f"Expected ISO date {expected_iso!r} for raw input {raw_date!r}, "
        f"got {article.published_date!r}"
    )


# ---------------------------------------------------------------------------
# Sub-AC 3b: unparseable date → published_date is None in the pipeline
# ---------------------------------------------------------------------------


class TestUnparseableDateYieldsNoneInPipeline:
    """
    Sub-AC 3b: when normalize_date returns None (because the raw extractor
    yields an unrecognisable date string), the pipeline propagates None all the
    way to Article.published_date.

    The raw trafilatura extractor is replaced by a MagicMock that returns a
    date string that is guaranteed to be unrecognisable by normalize_date
    (no YYYY-MM-DD fragment, not parseable by dateutil or dateparser).
    """

    # A string with no ISO-8601 fragment and no parseable date structure.
    # normalize_date("##XYZ_not_a_real_date_99!!") must return None.
    _UNPARSEABLE_DATE = "##XYZ_not_a_real_date_99!!"

    def _make_mock(self) -> MagicMock:
        """Return a mock trafilatura whose bare_extraction returns the unparseable date."""
        mock = MagicMock()
        mock.bare_extraction.return_value = {
            "title": "Test Article About Machine Learning",
            "text": _LONG_TEXT,
            "author": "Jane Doe",
            "date": self._UNPARSEABLE_DATE,
        }
        return mock

    def test_normalize_date_returns_none_for_unparseable_string(self) -> None:
        """
        Sanity check: normalize_date itself returns None for the unparseable input.

        This confirms that the mock date string is genuinely unrecognisable before
        testing its effect inside the pipeline.
        """
        from articula._extractor import normalize_date  # noqa: PLC0415

        result = normalize_date(self._UNPARSEABLE_DATE)
        assert result is None, (
            f"Expected normalize_date to return None for {self._UNPARSEABLE_DATE!r}, "
            f"got {result!r}"
        )

    def test_extraction_result_published_date_is_none_for_unparseable_raw(self) -> None:
        """
        ExtractionResult.published_date is None when the raw extractor date is
        unrecognisable.

        Confirms that normalize_date is called inside extract() and that the None
        return value is preserved in the ExtractionResult DTO.
        """
        mock_trf = self._make_mock()

        with patch.dict("sys.modules", {"trafilatura": mock_trf}):
            result = extract(_STUB_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)
        assert result.published_date is None, (
            f"ExtractionResult.published_date should be None for unparseable raw date "
            f"{self._UNPARSEABLE_DATE!r}, got {result.published_date!r}"
        )

    def test_article_published_date_is_none_for_unparseable_raw(self) -> None:
        """
        Article.published_date is None when the raw extractor yields an
        unrecognisable date string.

        This is the primary Sub-AC 3b assertion: the pipeline (extract → _build_article)
        must set published_date to None on the returned Article object when
        normalize_date cannot parse the raw date.
        """
        mock_trf = self._make_mock()
        fetch = _make_fetch_result()

        with patch.dict("sys.modules", {"trafilatura": mock_trf}):
            extraction = extract(_STUB_HTML, _TEST_URL)

        article = _build_article(fetch, extraction, ["static"])

        assert isinstance(article, Article)
        assert article.published_date is None, (
            f"Article.published_date must be None when normalize_date cannot parse "
            f"the raw date {self._UNPARSEABLE_DATE!r}, but got {article.published_date!r}. "
            "The pipeline must propagate None from normalize_date to Article.published_date."
        )

    def test_article_body_is_still_populated_when_date_unparseable(self) -> None:
        """
        Graceful degradation: Article.text is populated even when published_date is None.

        Verifies that an unparseable date does not cause extraction failure — only
        the published_date field is None while title and body remain intact.
        """
        mock_trf = self._make_mock()
        fetch = _make_fetch_result()

        with patch.dict("sys.modules", {"trafilatura": mock_trf}):
            extraction = extract(_STUB_HTML, _TEST_URL)

        article = _build_article(fetch, extraction, ["static"])

        assert article.published_date is None, "published_date must be None"
        assert article.text.strip(), "article body must be non-empty"
        assert article.title.strip(), "article title must be non-empty"


@pytest.mark.parametrize(
    "unparseable_raw",
    [
        "##XYZ_not_a_real_date_99!!",
        "this is definitely not a date",
        "qwerty-foobar",
        "abcdefghij",
        "!@#$%^&*()",
    ],
    ids=[
        "hash-prefix-garbage",
        "natural-sentence-no-date",
        "hyphen-words-no-digits",
        "lowercase-letters-only",
        "special-chars-only",
    ],
)
def test_pipeline_publishes_none_for_various_unparseable_date_strings(
    unparseable_raw: str,
) -> None:
    """
    Parametrized Sub-AC 3b: for a range of unrecognisable raw date strings the
    assembled Article always has published_date == None.

    Each input is guaranteed to be unparseable by normalize_date so this test
    confirms the None propagation is consistent across the full pipeline, not
    just for one specific garbage string.
    """
    mock_trf = MagicMock()
    mock_trf.bare_extraction.return_value = {
        "title": "Test Article About Machine Learning",
        "text": _LONG_TEXT,
        "author": "Jane Doe",
        "date": unparseable_raw,
    }
    fetch = _make_fetch_result()

    with patch.dict("sys.modules", {"trafilatura": mock_trf}):
        extraction = extract(_STUB_HTML, _TEST_URL)

    article = _build_article(fetch, extraction, ["static"])

    assert isinstance(article, Article)
    assert article.published_date is None, (
        f"Article.published_date must be None for unparseable raw date "
        f"{unparseable_raw!r}, but got {article.published_date!r}"
    )
