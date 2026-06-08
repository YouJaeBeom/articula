"""
Sub-AC 8b: Unit tests for the _build_article result-assembly function.

Verifies that ``_build_article`` returns an ``Article`` with ``None`` for any
missing optional field (``author``, ``published_date``) when the extractor
stub returns no value for them, while body text is still present.

Test matrix
-----------
* author=None, published_date=None  → Article has both None
* author=None, published_date set   → Article.author is None, date preserved
* author set, published_date=None   → Article.published_date is None, author preserved
* author set, published_date set    → happy path — both present
* Body text always present          → Article.text is the stub text value
* strategies_attempted passthrough  → tuple forwarded without mutation
"""

from __future__ import annotations

import pytest

from articula._extractor import ExtractionResult
from articula._fetcher import FetchResult
from articula._scraper import _build_article
from articula.models import Article

# ---------------------------------------------------------------------------
# Shared stub builders
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article/stub"
_BODY_TEXT = (
    "This is the stub article body text.  "
    "It is long enough to satisfy the Article text validation rule "
    "so the assembly step does not raise on construction."
)


def _make_fetch_result(tier: str = "static") -> FetchResult:
    """Return a minimal FetchResult for the given strategy tier."""
    return FetchResult(
        html="<html><body>stub</body></html>",
        resolved_url=_TEST_URL,
        strategy_tier=tier,
        status_code=200,
    )


def _make_extraction(
    *,
    author: str | None,
    published_date: str | None,
    method: str = "trafilatura",
) -> ExtractionResult:
    """Return an ExtractionResult stub with the given optional fields."""
    return ExtractionResult(
        title="Stub Article Title",
        text=_BODY_TEXT,
        author=author,
        published_date=published_date,
        method=method,
        confidence=0.80,
        language="en",
    )


# ---------------------------------------------------------------------------
# TestBuildArticleMissingOptionals
# ---------------------------------------------------------------------------


class TestBuildArticleMissingOptionals:
    """
    _build_article propagates None optional fields from ExtractionResult
    to the assembled Article without raising or substituting a default.
    """

    def test_both_optional_fields_none(self) -> None:
        """author=None and published_date=None are forwarded as-is to Article."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(author=None, published_date=None)

        article = _build_article(fetch, extraction, ["static"])

        assert isinstance(article, Article)
        assert article.author is None
        assert article.published_date is None

    def test_author_none_published_date_present(self) -> None:
        """When extractor returns author=None but provides a date, Article reflects that."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(author=None, published_date="2026-06-07")

        article = _build_article(fetch, extraction, ["static"])

        assert article.author is None
        assert article.published_date == "2026-06-07"

    def test_author_present_published_date_none(self) -> None:
        """When extractor returns a byline but no date, Article.published_date is None."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(author="Jane Doe", published_date=None)

        article = _build_article(fetch, extraction, ["static"])

        assert article.author == "Jane Doe"
        assert article.published_date is None

    def test_both_optional_fields_present(self) -> None:
        """Happy path: both optional fields are present and forwarded correctly."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(author="John Smith", published_date="2026-01-15")

        article = _build_article(fetch, extraction, ["static"])

        assert article.author == "John Smith"
        assert article.published_date == "2026-01-15"


# ---------------------------------------------------------------------------
# TestBuildArticleBodyTextAlwaysPresent
# ---------------------------------------------------------------------------


class TestBuildArticleBodyTextAlwaysPresent:
    """Body text is always forwarded even when optional metadata is absent."""

    def test_text_present_when_both_optionals_none(self) -> None:
        """Article.text equals the extraction stub text when optionals are None."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(author=None, published_date=None)

        article = _build_article(fetch, extraction, ["static"])

        assert article.text == _BODY_TEXT
        assert article.text.strip() != ""

    def test_title_present_when_both_optionals_none(self) -> None:
        """Article.title equals the extraction stub title when optionals are None."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(author=None, published_date=None)

        article = _build_article(fetch, extraction, ["static"])

        assert article.title == "Stub Article Title"


# ---------------------------------------------------------------------------
# TestBuildArticleCoreFieldsPassthrough
# ---------------------------------------------------------------------------


class TestBuildArticleCoreFieldsPassthrough:
    """Core fields are faithfully propagated from FetchResult and ExtractionResult."""

    def test_resolved_url_from_fetch_result(self) -> None:
        """resolved_url comes from FetchResult, not ExtractionResult."""
        fetch = _make_fetch_result(tier="headers_rotation")
        extraction = _make_extraction(author=None, published_date=None)

        article = _build_article(fetch, extraction, ["static", "headers_rotation"])

        assert article.resolved_url == _TEST_URL

    def test_strategy_tier_from_fetch_result(self) -> None:
        """strategy_tier is taken from FetchResult.strategy_tier."""
        fetch = _make_fetch_result(tier="headers_rotation")
        extraction = _make_extraction(author=None, published_date=None)

        article = _build_article(fetch, extraction, ["static", "headers_rotation"])

        assert article.strategy_tier == "headers_rotation"

    def test_extraction_method_from_extraction_result(self) -> None:
        """extraction_method is taken from ExtractionResult.method."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(
            author=None, published_date=None, method="readability"
        )

        article = _build_article(fetch, extraction, ["static"])

        assert article.extraction_method == "readability"

    def test_strategies_attempted_forwarded_as_tuple(self) -> None:
        """strategies_attempted list is converted to a tuple on the Article."""
        fetch = _make_fetch_result(tier="browser")
        extraction = _make_extraction(author=None, published_date=None)
        attempted = ["static", "headers_rotation", "browser"]

        article = _build_article(fetch, extraction, attempted)

        assert article.strategies_attempted == ("static", "headers_rotation", "browser")
        assert isinstance(article.strategies_attempted, tuple)


# ---------------------------------------------------------------------------
# TestBuildArticleExtractionMethodVariants
# ---------------------------------------------------------------------------


class TestBuildArticleExtractionMethodVariants:
    """
    All three extraction methods (trafilatura / readability / heuristic)
    produce an Article with None optional fields when the extractor omits them.

    readability and heuristic never populate author or published_date,
    so these cover the real-world missing-field scenario.
    """

    @pytest.mark.parametrize(
        "method",
        ["trafilatura", "readability", "heuristic"],
    )
    def test_none_optionals_across_methods(self, method: str) -> None:
        """Each extraction method yields Article with None optionals when absent."""
        fetch = _make_fetch_result()
        extraction = _make_extraction(
            author=None, published_date=None, method=method
        )

        article = _build_article(fetch, extraction, ["static"])

        assert article.author is None, f"Expected author=None for method={method!r}"
        assert article.published_date is None, (
            f"Expected published_date=None for method={method!r}"
        )
        assert article.text == _BODY_TEXT


# ---------------------------------------------------------------------------
# TestBuildArticleKoreanContent
# ---------------------------------------------------------------------------


class TestBuildArticleKoreanContent:
    """Korean-language articles assembled with None optional fields are valid."""

    def test_korean_article_both_optionals_none(self) -> None:
        """Korean ExtractionResult with author=None, published_date=None assembles correctly."""
        fetch = FetchResult(
            html="<html><body>한국어 기사</body></html>",
            resolved_url="https://news.example.kr/article/99",
            strategy_tier="static",
            status_code=200,
        )
        extraction = ExtractionResult(
            title="인공지능 최신 동향",
            text=(
                "인공지능 기술이 빠르게 발전하면서 다양한 산업 분야에 "
                "혁신적인 변화를 가져오고 있습니다. "
                "특히 자연어 처리 및 컴퓨터 비전 영역에서 눈부신 성과가 나타나고 있습니다."
            ),
            author=None,
            published_date=None,
            method="trafilatura",
            confidence=0.75,
            language="ko",
        )

        article = _build_article(fetch, extraction, ["static"])

        assert isinstance(article, Article)
        assert article.author is None
        assert article.published_date is None
        assert article.detected_language == "ko"
        assert "인공지능" in article.title

    def test_korean_article_author_none_date_present(self) -> None:
        """Korean article with author=None but a date assembles without error."""
        fetch = FetchResult(
            html="<html><body>한국어</body></html>",
            resolved_url="https://news.example.kr/article/100",
            strategy_tier="static",
            status_code=200,
        )
        extraction = ExtractionResult(
            title="AI 뉴스 요약",
            text=(
                "오늘의 인공지능 관련 뉴스를 요약합니다. "
                "최신 모델들이 다양한 벤치마크에서 새로운 기록을 세우고 있습니다."
            ),
            author=None,
            published_date="2026-06-07",
            method="readability",
            confidence=0.65,
            language="ko",
        )

        article = _build_article(fetch, extraction, ["static"])

        assert article.author is None
        assert article.published_date == "2026-06-07"
        assert article.detected_language == "ko"
