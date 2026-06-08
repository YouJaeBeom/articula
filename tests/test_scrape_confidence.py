"""
Sub-AC 7d: scrape(url) populates extraction_confidence as a normalized float
reflecting how many expected fields were successfully extracted.

Test strategy
-------------
The tests mock the fetch and extraction layers so the scrape pipeline runs
offline. Each test constructs an ExtractionResult with a specific combination
of present/absent fields, then asserts properties of Article.extraction_confidence
returned by async_scrape().

Coverage matrix
---------------
1. Full extraction — title + long text + author + date   → high confidence
2. Partial extraction — title + medium text only          → medium confidence
3. Minimal extraction — title + short text, no metadata  → lowest valid confidence
4. Empty extraction — no usable body text                 → ExtractionError raised

The confidence value is also verified to be:
  * a float (not int or None)
  * in the closed interval [0.0, 1.0]
  * monotonically higher for richer field sets
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula import async_scrape
from articula._extractor import ExtractionResult, compute_extraction_confidence
from articula._fetcher import FetchResult
from articula.exceptions import ExtractionError

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article"

# Patch targets follow the import paths in articula._scraper
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_EXTRACT = "articula._scraper.extract"

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _stub_fetch() -> FetchResult:
    """Minimal FetchResult sufficient to pass through the scrape pipeline."""
    return FetchResult(
        html="<html><body>stub html</body></html>",
        resolved_url=_TEST_URL,
        strategy_tier="static",
        status_code=200,
    )


def _make_extraction(
    *,
    title: str,
    text: str,
    author: str | None,
    published_date: str | None,
    method: str = "trafilatura",
) -> ExtractionResult:
    """Build an ExtractionResult whose confidence matches the formula in _extractor.py."""
    confidence = compute_extraction_confidence(
        title=title,
        text=text,
        author=author,
        published_date=published_date,
        method=method,
    )
    return ExtractionResult(
        title=title,
        text=text,
        author=author,
        published_date=published_date,
        method=method,
        confidence=confidence,
        language="en",
    )


# ---------------------------------------------------------------------------
# 1. Article.extraction_confidence is always a float in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestConfidenceIsNormalizedFloat:
    """extraction_confidence must be a float in the closed interval [0.0, 1.0]."""

    @pytest.mark.asyncio
    async def test_confidence_is_float_type(self) -> None:
        extraction = _make_extraction(
            title="Test Article Title",
            text="x" * 300,
            author="Author",
            published_date="2026-06-07",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert isinstance(article.extraction_confidence, float), (
            f"extraction_confidence should be float, got {type(article.extraction_confidence)}"
        )

    @pytest.mark.asyncio
    async def test_confidence_is_within_unit_interval(self) -> None:
        extraction = _make_extraction(
            title="Test Article Title",
            text="x" * 300,
            author=None,
            published_date=None,
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert 0.0 <= article.extraction_confidence <= 1.0, (
            f"extraction_confidence {article.extraction_confidence} is outside [0.0, 1.0]"
        )

    @pytest.mark.asyncio
    async def test_confidence_is_not_none(self) -> None:
        extraction = _make_extraction(
            title="Some Title",
            text="y" * 120,
            author=None,
            published_date=None,
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert article.extraction_confidence is not None


# ---------------------------------------------------------------------------
# 2. Full extraction — all expected fields present → high confidence
# ---------------------------------------------------------------------------


class TestFullExtraction:
    """All expected fields present: title, long text, author, published_date."""

    @pytest.mark.asyncio
    async def test_full_extraction_confidence_is_high(self) -> None:
        """
        Full extraction via trafilatura with all fields yields confidence ≥ 0.85.

        Scoring breakdown (trafilatura, long text, full title, author + date):
          base(trafilatura) = 0.60
          text ≥ 200        = +0.10
          text ≥ 500        = +0.10
          title ≥ 10        = +0.05
          title ≥ 30        = +0.05
          author present    = +0.05
          date present      = +0.05
          Total             = 1.00 (capped at 1.0)
        """
        extraction = _make_extraction(
            title="Anthropic Releases a Powerful New Claude Model for Everyone",  # > 30
            text="Detailed article content. " * 30,  # > 500 chars
            author="Jane Doe",
            published_date="2026-06-07",
            method="trafilatura",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert article.extraction_confidence >= 0.85, (
            f"Full extraction should yield confidence ≥ 0.85, got {article.extraction_confidence}"
        )

    @pytest.mark.asyncio
    async def test_full_extraction_confidence_equals_computed_value(self) -> None:
        """The Article's confidence exactly matches compute_extraction_confidence output."""
        title = "Anthropic Releases a Powerful New Claude Model for Everyone"
        text = "Detailed article content. " * 30
        author = "Jane Doe"
        date = "2026-06-07"
        method = "trafilatura"

        expected = compute_extraction_confidence(
            title=title,
            text=text,
            author=author,
            published_date=date,
            method=method,
        )
        extraction = ExtractionResult(
            title=title,
            text=text,
            author=author,
            published_date=date,
            method=method,
            confidence=expected,
            language="en",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert article.extraction_confidence == pytest.approx(expected), (
            f"Expected {expected}, got {article.extraction_confidence}"
        )

    @pytest.mark.asyncio
    async def test_full_korean_extraction_confidence_is_high(self) -> None:
        """Korean full extraction also yields high confidence (≥ 0.85)."""
        extraction = _make_extraction(
            title="앤트로픽, 새로운 클로드 모델 공개 — 성능 대폭 향상",  # > 30 chars
            text="앤트로픽이 최신 AI 모델을 발표했습니다. " * 30,  # > 500 chars
            author="홍길동",
            published_date="2026-06-07",
            method="trafilatura",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert article.extraction_confidence >= 0.85


# ---------------------------------------------------------------------------
# 3. Partial extraction — some fields missing → medium confidence
# ---------------------------------------------------------------------------


class TestPartialExtraction:
    """Partial extraction: title + body present; author and/or date absent."""

    @pytest.mark.asyncio
    async def test_partial_extraction_no_author_no_date_confidence_is_medium(self) -> None:
        """
        readability extraction yields no metadata. Confidence is medium.

        Scoring (readability, 300-char text, 19-char title, no author/date):
          base(readability) = 0.45
          text ≥ 200        = +0.10
          title ≥ 10        = +0.05
          Total             = 0.60
        """
        extraction = _make_extraction(
            title="Breaking News Today",  # 19 chars — ≥ 10 but < 30
            text="B" * 300,              # ≥ 200 but < 500
            author=None,
            published_date=None,
            method="readability",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert 0.0 < article.extraction_confidence < 0.85, (
            f"Partial extraction should yield medium confidence, got {article.extraction_confidence}"
        )

    @pytest.mark.asyncio
    async def test_partial_extraction_author_present_no_date(self) -> None:
        """Author present but no date → confidence is between minimal and full."""
        extraction_with_author = _make_extraction(
            title="Breaking News Today",
            text="B" * 300,
            author="Jane Doe",
            published_date=None,
            method="readability",
        )
        extraction_without = _make_extraction(
            title="Breaking News Today",
            text="B" * 300,
            author=None,
            published_date=None,
            method="readability",
        )

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction_with_author),
        ):
            mock_static.return_value = _stub_fetch()
            article_with = await async_scrape(_TEST_URL)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction_without),
        ):
            mock_static.return_value = _stub_fetch()
            article_without = await async_scrape(_TEST_URL)

        assert article_with.extraction_confidence > article_without.extraction_confidence, (
            "Author present should raise confidence above no-author baseline"
        )

    @pytest.mark.asyncio
    async def test_partial_extraction_date_present_no_author(self) -> None:
        """Published date present but no author → confidence higher than both-absent baseline."""
        extraction_with_date = _make_extraction(
            title="Breaking News Today",
            text="B" * 300,
            author=None,
            published_date="2026-06-07",
            method="readability",
        )
        extraction_without = _make_extraction(
            title="Breaking News Today",
            text="B" * 300,
            author=None,
            published_date=None,
            method="readability",
        )

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction_with_date),
        ):
            mock_static.return_value = _stub_fetch()
            article_with = await async_scrape(_TEST_URL)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction_without),
        ):
            mock_static.return_value = _stub_fetch()
            article_without = await async_scrape(_TEST_URL)

        assert article_with.extraction_confidence > article_without.extraction_confidence


# ---------------------------------------------------------------------------
# 4. Confidence ordering: full > partial > minimal
# ---------------------------------------------------------------------------


class TestConfidenceMonotonicity:
    """Richer field sets yield higher confidence — monotonic ordering holds."""

    @pytest.mark.asyncio
    async def test_full_confidence_beats_partial(self) -> None:
        """Full (all fields) confidence > partial (no author/date) confidence."""
        full = _make_extraction(
            title="A" * 35,
            text="x" * 600,
            author="Jane Doe",
            published_date="2026-06-07",
            method="trafilatura",
        )
        partial = _make_extraction(
            title="A" * 35,
            text="x" * 600,
            author=None,
            published_date=None,
            method="trafilatura",
        )

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=full),
        ):
            mock_static.return_value = _stub_fetch()
            article_full = await async_scrape(_TEST_URL)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=partial),
        ):
            mock_static.return_value = _stub_fetch()
            article_partial = await async_scrape(_TEST_URL)

        assert article_full.extraction_confidence > article_partial.extraction_confidence

    @pytest.mark.asyncio
    async def test_partial_confidence_beats_minimal(self) -> None:
        """Partial (medium text, no metadata) > minimal (very short text, no metadata)."""
        partial = _make_extraction(
            title="Breaking News Today",  # ≥ 10
            text="x" * 300,              # ≥ 200
            author=None,
            published_date=None,
            method="heuristic",
        )
        minimal = _make_extraction(
            title="Hi",    # < 10 → no title bonus
            text="x" * 90,  # < 200 → no text bonus; must be ≥ 80 for Article validation
            author=None,
            published_date=None,
            method="heuristic",
        )

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=partial),
        ):
            mock_static.return_value = _stub_fetch()
            article_partial = await async_scrape(_TEST_URL)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=minimal),
        ):
            mock_static.return_value = _stub_fetch()
            article_minimal = await async_scrape(_TEST_URL)

        assert article_partial.extraction_confidence > article_minimal.extraction_confidence

    @pytest.mark.asyncio
    async def test_three_tier_monotonic_ordering(self) -> None:
        """confidence(full) > confidence(partial) > confidence(minimal)."""
        full = _make_extraction(
            title="A" * 35,
            text="x" * 600,
            author="Author",
            published_date="2026-06-07",
            method="trafilatura",
        )
        partial = _make_extraction(
            title="A" * 35,
            text="x" * 300,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        minimal = _make_extraction(
            title="Hi",
            text="x" * 90,
            author=None,
            published_date=None,
            method="heuristic",
        )

        async def _scrape_with(extraction: ExtractionResult) -> float:
            with (
                patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
                patch(_PATCH_EXTRACT, return_value=extraction),
            ):
                mock_static.return_value = _stub_fetch()
                art = await async_scrape(_TEST_URL)
            return art.extraction_confidence

        c_full = await _scrape_with(full)
        c_partial = await _scrape_with(partial)
        c_minimal = await _scrape_with(minimal)

        assert c_full > c_partial > c_minimal, (
            f"Expected c_full({c_full}) > c_partial({c_partial}) > c_minimal({c_minimal})"
        )


# ---------------------------------------------------------------------------
# 5. Empty extraction edge cases — ExtractionError on no usable body
# ---------------------------------------------------------------------------


class TestEmptyExtractionEdgeCases:
    """When extract() raises ValueError (no usable text), scrape() raises ExtractionError."""

    @pytest.mark.asyncio
    async def test_empty_extraction_raises_extraction_error(self) -> None:
        """extract() raising ValueError → scrape() raises ExtractionError (not Article)."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(
                _PATCH_EXTRACT,
                side_effect=ValueError(
                    "All extraction methods produced empty output"
                ),
            ),
        ):
            mock_static.return_value = _stub_fetch()
            with pytest.raises(ExtractionError):
                await async_scrape(_TEST_URL)

    @pytest.mark.asyncio
    async def test_extraction_error_is_not_article(self) -> None:
        """ExtractionError is raised rather than returning an Article with zero confidence."""
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(
                _PATCH_EXTRACT,
                side_effect=ValueError("All extraction methods produced empty output"),
            ),
        ):
            mock_static.return_value = _stub_fetch()
            try:
                result = await async_scrape(_TEST_URL)
                pytest.fail(
                    f"Expected ExtractionError but got Article: {result!r}"
                )
            except ExtractionError:
                pass  # correct — empty extraction raises, not partial Article

    @pytest.mark.asyncio
    async def test_minimal_viable_extraction_has_low_confidence(self) -> None:
        """
        An extraction with the bare minimum (short title + minimal text, no metadata)
        produces a valid Article with very low confidence.

        heuristic base (0.25) + no bonuses (text < 200, title < 10) = 0.25
        """
        extraction = _make_extraction(
            title="Hi",    # 2 chars — below 10-char bonus threshold
            text="x" * 90,  # 90 chars — above Article.text minimum (80) but below 200
            author=None,
            published_date=None,
            method="heuristic",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert isinstance(article.extraction_confidence, float)
        # heuristic base = 0.25, no bonuses triggered
        assert article.extraction_confidence == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_confidence_never_zero_for_valid_extraction(self) -> None:
        """
        Any successfully assembled Article has confidence > 0.0 because the
        method base score is always ≥ 0.25.
        """
        extraction = _make_extraction(
            title="Hi",
            text="x" * 90,
            author=None,
            published_date=None,
            method="heuristic",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert article.extraction_confidence > 0.0


# ---------------------------------------------------------------------------
# 6. Confidence propagates faithfully through _build_article
# ---------------------------------------------------------------------------


class TestConfidencePropagation:
    """
    The ExtractionResult.confidence value is forwarded verbatim to
    Article.extraction_confidence by _build_article without modification.
    """

    @pytest.mark.asyncio
    async def test_confidence_from_extraction_flows_to_article(self) -> None:
        """Article.extraction_confidence equals ExtractionResult.confidence."""
        title = "Test Article Title That Is Long Enough"
        text = "Content. " * 70
        author = "Jane Doe"
        date = "2026-06-07"
        method = "trafilatura"

        expected_confidence = compute_extraction_confidence(
            title=title,
            text=text,
            author=author,
            published_date=date,
            method=method,
        )
        extraction = ExtractionResult(
            title=title,
            text=text,
            author=author,
            published_date=date,
            method=method,
            confidence=expected_confidence,
            language="en",
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert article.extraction_confidence == pytest.approx(expected_confidence), (
            f"Expected confidence {expected_confidence}, "
            f"got {article.extraction_confidence}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["trafilatura", "readability", "heuristic"])
    async def test_confidence_in_unit_interval_for_all_methods(
        self, method: str
    ) -> None:
        """confidence is in [0.0, 1.0] for all three extraction methods."""
        extraction = _make_extraction(
            title="A" * 30,
            text="x" * 400,
            author=None,
            published_date=None,
            method=method,
        )
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_EXTRACT, return_value=extraction),
        ):
            mock_static.return_value = _stub_fetch()
            article = await async_scrape(_TEST_URL)

        assert 0.0 <= article.extraction_confidence <= 1.0, (
            f"confidence {article.extraction_confidence} out of [0, 1] for method={method!r}"
        )
