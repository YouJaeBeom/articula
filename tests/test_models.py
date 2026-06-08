"""
Unit tests for articula.models.Article

Covers:
- Happy-path instantiation with all required fields
- Optional / best-effort fields (author, published_date)
- Immutability (frozen dataclass)
- Field validation: title, text, resolved_url, strategy_tier,
  extraction_method, extraction_confidence, detected_language,
  published_date
- Convenience helpers: with_author(), with_published_date()
- Serialisation: to_dict()
- Korean and English content
"""

from __future__ import annotations

import dataclasses

import pytest

from articula.models import Article

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VALID_KWARGS: dict[str, object] = {
    "title": "Claude Releases New Model",
    "text": "Anthropic announced a new Claude model with improved reasoning.",
    "resolved_url": "https://www.example.com/article/123",
    "strategy_tier": "static",
    "extraction_method": "trafilatura",
    "extraction_confidence": 0.95,
    "detected_language": "en",
    "strategies_attempted": ("static",),
    "author": "Jane Doe",
    "published_date": "2026-06-07",
}


def make_article(**overrides: object) -> Article:
    """Return a valid Article, optionally overriding fields."""
    kwargs = {**VALID_KWARGS, **overrides}
    return Article(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Happy-path — all required fields present
# ---------------------------------------------------------------------------


class TestArticleInstantiation:
    def test_minimal_required_fields(self) -> None:
        """Article can be created with all required fields, no optionals."""
        article = Article(
            title="Test Title",
            text="Test body content that is not empty.",
            resolved_url="https://example.com/test",
            strategy_tier="static",
            extraction_method="trafilatura",
            extraction_confidence=0.8,
            detected_language="en",
        )
        assert article.title == "Test Title"
        assert article.text == "Test body content that is not empty."
        assert article.resolved_url == "https://example.com/test"
        assert article.strategy_tier == "static"
        assert article.extraction_method == "trafilatura"
        assert article.extraction_confidence == 0.8
        assert article.detected_language == "en"
        # Optional fields default to None / empty tuple
        assert article.author is None
        assert article.published_date is None
        assert article.strategies_attempted == ()

    def test_full_fields_including_optionals(self) -> None:
        """Article accepts all fields including author and published_date."""
        article = make_article()
        assert article.author == "Jane Doe"
        assert article.published_date == "2026-06-07"
        assert article.strategies_attempted == ("static",)

    def test_all_strategy_tiers(self) -> None:
        """All three strategy_tier values are accepted."""
        for tier in ("static", "headers_rotation", "browser"):
            article = make_article(strategy_tier=tier)
            assert article.strategy_tier == tier

    def test_all_extraction_methods(self) -> None:
        """All three extraction_method values are accepted."""
        for method in ("trafilatura", "readability", "heuristic"):
            article = make_article(extraction_method=method)
            assert article.extraction_method == method

    def test_confidence_boundary_zero(self) -> None:
        article = make_article(extraction_confidence=0.0)
        assert article.extraction_confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        article = make_article(extraction_confidence=1.0)
        assert article.extraction_confidence == 1.0

    def test_confidence_integer_accepted(self) -> None:
        """int 0 and 1 are accepted for extraction_confidence."""
        article = make_article(extraction_confidence=1)
        assert article.extraction_confidence == 1


# ---------------------------------------------------------------------------
# 2. Korean content
# ---------------------------------------------------------------------------


class TestKoreanContent:
    def test_korean_article(self) -> None:
        """Article correctly handles Korean title and body text."""
        article = Article(
            title="클로드, 새로운 모델 발표",
            text="앤트로픽이 향상된 추론 능력을 가진 새로운 클로드 모델을 발표했습니다.",
            resolved_url="https://www.example.co.kr/article/456",
            strategy_tier="headers_rotation",
            extraction_method="readability",
            extraction_confidence=0.87,
            detected_language="ko",
            author="홍길동",
            published_date="2026-06-07T09:00:00+09:00",
        )
        assert article.detected_language == "ko"
        assert "클로드" in article.title
        assert article.author == "홍길동"

    def test_korean_author_none(self) -> None:
        """Korean article with missing author is valid."""
        article = Article(
            title="인공지능 뉴스",
            text="오늘의 인공지능 관련 최신 뉴스입니다.",
            resolved_url="https://news.example.kr/ai/789",
            strategy_tier="browser",
            extraction_method="trafilatura",
            extraction_confidence=0.72,
            detected_language="ko",
        )
        assert article.author is None
        assert article.published_date is None


# ---------------------------------------------------------------------------
# 3. Immutability (frozen dataclass)
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_cannot_set_title(self) -> None:
        article = make_article()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            article.title = "Mutated"  # type: ignore[misc]

    def test_cannot_set_text(self) -> None:
        article = make_article()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            article.text = "Mutated"  # type: ignore[misc]

    def test_cannot_set_author(self) -> None:
        article = make_article()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            article.author = "Mutated"  # type: ignore[misc]

    def test_cannot_set_confidence(self) -> None:
        article = make_article()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            article.extraction_confidence = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. Validation — title
# ---------------------------------------------------------------------------


class TestTitleValidation:
    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            make_article(title="")

    def test_whitespace_only_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            make_article(title="   \t\n")

    def test_non_string_title_raises(self) -> None:
        with pytest.raises(TypeError, match="title"):
            make_article(title=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Validation — text
# ---------------------------------------------------------------------------


class TestTextValidation:
    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            make_article(text="")

    def test_whitespace_only_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            make_article(text="   ")

    def test_non_string_text_raises(self) -> None:
        with pytest.raises(TypeError, match="text"):
            make_article(text=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. Validation — resolved_url
# ---------------------------------------------------------------------------


class TestResolvedUrlValidation:
    def test_http_url_accepted(self) -> None:
        article = make_article(resolved_url="http://example.com/page")
        assert article.resolved_url == "http://example.com/page"

    def test_https_url_accepted(self) -> None:
        article = make_article(resolved_url="https://example.com/page")
        assert article.resolved_url == "https://example.com/page"

    def test_non_http_url_raises(self) -> None:
        with pytest.raises(ValueError, match="resolved_url"):
            make_article(resolved_url="ftp://example.com/page")

    def test_relative_url_raises(self) -> None:
        with pytest.raises(ValueError, match="resolved_url"):
            make_article(resolved_url="/relative/path")

    def test_non_string_url_raises(self) -> None:
        with pytest.raises(TypeError, match="resolved_url"):
            make_article(resolved_url=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. Validation — strategy_tier
# ---------------------------------------------------------------------------


class TestStrategyTierValidation:
    def test_invalid_strategy_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy_tier"):
            make_article(strategy_tier="magic")  # type: ignore[arg-type]

    def test_empty_strategy_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy_tier"):
            make_article(strategy_tier="")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. Validation — extraction_method
# ---------------------------------------------------------------------------


class TestExtractionMethodValidation:
    def test_invalid_extraction_method_raises(self) -> None:
        with pytest.raises(ValueError, match="extraction_method"):
            make_article(extraction_method="magic")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. Validation — extraction_confidence
# ---------------------------------------------------------------------------


class TestExtractionConfidenceValidation:
    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="extraction_confidence"):
            make_article(extraction_confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="extraction_confidence"):
            make_article(extraction_confidence=1.001)

    def test_non_numeric_confidence_raises(self) -> None:
        with pytest.raises(TypeError, match="extraction_confidence"):
            make_article(extraction_confidence="high")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. Validation — detected_language
# ---------------------------------------------------------------------------


class TestDetectedLanguageValidation:
    def test_empty_language_raises(self) -> None:
        with pytest.raises(ValueError, match="detected_language"):
            make_article(detected_language="")

    def test_whitespace_language_raises(self) -> None:
        with pytest.raises(ValueError, match="detected_language"):
            make_article(detected_language="  ")

    def test_non_string_language_raises(self) -> None:
        with pytest.raises(TypeError, match="detected_language"):
            make_article(detected_language=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 11. Validation — published_date (ISO 8601)
# ---------------------------------------------------------------------------


class TestPublishedDateValidation:
    @pytest.mark.parametrize(
        "valid_date",
        [
            "2026-06-07",
            "2026-06-07T09:00:00",
            "2026-06-07T09:00:00Z",
            "2026-06-07T09:00:00+09:00",
            "2026-06-07T09:00:00-05:00",
            "2026-06-07T09:00:00.123456Z",
        ],
    )
    def test_valid_iso_date(self, valid_date: str) -> None:
        article = make_article(published_date=valid_date)
        assert article.published_date == valid_date

    def test_none_published_date_accepted(self) -> None:
        article = make_article(published_date=None)
        assert article.published_date is None

    @pytest.mark.parametrize(
        "invalid_date",
        [
            "June 7 2026",
            "07/06/2026",
            "2026-6-7",    # missing zero-padding
            "not-a-date",
            "",
        ],
    )
    def test_invalid_date_format_raises(self, invalid_date: str) -> None:
        with pytest.raises(ValueError, match="published_date"):
            make_article(published_date=invalid_date)

    def test_out_of_range_month_regex_allows(self) -> None:
        """
        The regex validates format only (YYYY-MM-DD), not calendar semantics.
        Semantic validation (e.g. month 13) is the caller's responsibility.
        2026-13-01 matches the pattern and is therefore accepted.
        """
        article = make_article(published_date="2026-13-01")
        assert article.published_date == "2026-13-01"

    def test_non_string_date_raises(self) -> None:
        with pytest.raises(TypeError, match="published_date"):
            make_article(published_date=20260607)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 12. Convenience helpers (non-mutating)
# ---------------------------------------------------------------------------


class TestConvenienceHelpers:
    def test_with_author_returns_new_instance(self) -> None:
        original = make_article(author=None)
        updated = original.with_author("New Author")
        assert updated.author == "New Author"
        assert original.author is None  # original unchanged

    def test_with_author_none(self) -> None:
        original = make_article(author="Old Author")
        updated = original.with_author(None)
        assert updated.author is None

    def test_with_published_date_returns_new_instance(self) -> None:
        original = make_article(published_date=None)
        updated = original.with_published_date("2026-01-01")
        assert updated.published_date == "2026-01-01"
        assert original.published_date is None

    def test_with_published_date_none(self) -> None:
        original = make_article(published_date="2026-06-07")
        updated = original.with_published_date(None)
        assert updated.published_date is None

    def test_helpers_preserve_other_fields(self) -> None:
        original = make_article()
        updated = original.with_author("Another Author")
        assert updated.title == original.title
        assert updated.text == original.text
        assert updated.resolved_url == original.resolved_url
        assert updated.strategy_tier == original.strategy_tier
        assert updated.extraction_confidence == original.extraction_confidence
        assert updated.detected_language == original.detected_language


# ---------------------------------------------------------------------------
# 13. Serialisation
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_returns_all_keys(self) -> None:
        article = make_article()
        d = article.to_dict()
        expected_keys = {
            "title",
            "text",
            "resolved_url",
            "strategy_tier",
            "extraction_method",
            "extraction_confidence",
            "detected_language",
            "strategies_attempted",
            "author",
            "published_date",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self) -> None:
        article = make_article()
        d = article.to_dict()
        assert d["title"] == article.title
        assert d["text"] == article.text
        assert d["author"] == article.author
        assert d["published_date"] == article.published_date
        assert d["extraction_confidence"] == article.extraction_confidence
        assert d["detected_language"] == article.detected_language

    def test_to_dict_strategies_attempted_is_list(self) -> None:
        """strategies_attempted should be serialised as a list, not a tuple."""
        article = make_article(strategies_attempted=("static", "headers_rotation"))
        d = article.to_dict()
        assert isinstance(d["strategies_attempted"], list)
        assert d["strategies_attempted"] == ["static", "headers_rotation"]

    def test_to_dict_none_optional_fields(self) -> None:
        article = make_article(author=None, published_date=None)
        d = article.to_dict()
        assert d["author"] is None
        assert d["published_date"] is None


# ---------------------------------------------------------------------------
# Sub-AC 8a: Article accepts None for author and published_date (nullable fields)
# ---------------------------------------------------------------------------


class TestNullableOptionalFields:
    """
    Sub-AC 8a: Article dataclass accepts None for author and published_date
    without raising validation errors, even when the body text is fully valid.
    """

    def test_both_optional_fields_none_with_valid_text(self) -> None:
        """Explicitly set author=None and published_date=None; no error raised."""
        article = Article(
            title="Breaking: Library Ships",
            text="The articula library has shipped with full None support.",
            resolved_url="https://example.com/article/none-test",
            strategy_tier="static",
            extraction_method="trafilatura",
            extraction_confidence=0.9,
            detected_language="en",
            author=None,
            published_date=None,
        )
        assert article.author is None
        assert article.published_date is None
        # Body text is fully present — not empty, not whitespace
        assert article.text != ""

    def test_author_none_published_date_present(self) -> None:
        """author may be None while published_date is a valid ISO 8601 string."""
        article = Article(
            title="Article Without Byline",
            text="This article has a publication date but no identified author.",
            resolved_url="https://example.com/no-author",
            strategy_tier="headers_rotation",
            extraction_method="readability",
            extraction_confidence=0.75,
            detected_language="en",
            author=None,
            published_date="2026-06-07",
        )
        assert article.author is None
        assert article.published_date == "2026-06-07"

    def test_author_present_published_date_none(self) -> None:
        """published_date may be None while author is a valid string."""
        article = Article(
            title="Undated Editorial",
            text="This article has an author byline but no publication date.",
            resolved_url="https://example.com/undated",
            strategy_tier="browser",
            extraction_method="heuristic",
            extraction_confidence=0.6,
            detected_language="en",
            author="Jane Doe",
            published_date=None,
        )
        assert article.author == "Jane Doe"
        assert article.published_date is None

    def test_korean_article_both_optional_fields_none(self) -> None:
        """Korean-language article with author=None and published_date=None is valid."""
        article = Article(
            title="작성자 미상의 기사",
            text="이 기사는 작성자와 발행일 없이 추출된 본문 내용을 담고 있습니다.",
            resolved_url="https://news.example.kr/unknown/101",
            strategy_tier="static",
            extraction_method="trafilatura",
            extraction_confidence=0.8,
            detected_language="ko",
            author=None,
            published_date=None,
        )
        assert article.author is None
        assert article.published_date is None
        assert article.detected_language == "ko"

    def test_none_fields_preserved_in_to_dict(self) -> None:
        """to_dict() serialises None optional fields as None (not absent keys)."""
        article = Article(
            title="Serialisation Check",
            text="Verify that None optional fields round-trip through to_dict().",
            resolved_url="https://example.com/serial-none",
            strategy_tier="static",
            extraction_method="trafilatura",
            extraction_confidence=0.85,
            detected_language="en",
            author=None,
            published_date=None,
        )
        d = article.to_dict()
        assert "author" in d and d["author"] is None
        assert "published_date" in d and d["published_date"] is None


# ---------------------------------------------------------------------------
# 14. Equality and hashing (frozen dataclasses are hashable)
# ---------------------------------------------------------------------------


class TestEqualityAndHashing:
    def test_equal_articles(self) -> None:
        a = make_article()
        b = make_article()
        assert a == b

    def test_different_titles_not_equal(self) -> None:
        a = make_article(title="Title A")
        b = make_article(title="Title B")
        assert a != b

    def test_frozen_article_is_hashable(self) -> None:
        article = make_article()
        # Should not raise
        h = hash(article)
        assert isinstance(h, int)

    def test_article_can_be_used_in_set(self) -> None:
        a = make_article()
        b = make_article()
        c = make_article(title="Different Title")
        s = {a, b, c}
        assert len(s) == 2  # a and b are equal
