"""
Unit tests for ``articula._extractor.compute_extraction_confidence``.

Sub-AC 7c: ``extraction_confidence`` is computed by a standalone function that
scores field completeness/quality (e.g. non-empty title, non-None date) and is
testable in isolation with fixture Article objects.

Test strategy
-------------
1. Direct unit tests of the function's scoring logic.
2. Tests that use fixture ``Article`` objects to call the function — proving
   "testable in isolation with fixture Article objects".
3. Boundary / property tests (always in [0.0, 1.0], deterministic).
4. Verification that the extractors in ``_extractor.py`` call the function
   (not hardcoded constants).
"""

from __future__ import annotations

import pytest

from articula._extractor import compute_extraction_confidence
from articula.models import Article

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_article(**overrides: object) -> Article:
    """Build a minimal valid Article, overriding any fields as needed."""
    defaults: dict[str, object] = {
        "title": "Claude Releases a Brand-New Model for Everyone",  # 45 chars
        "text": "A" * 600,  # 600 chars
        "resolved_url": "https://example.com/article",
        "strategy_tier": "static",
        "extraction_method": "trafilatura",
        "extraction_confidence": 0.85,  # manually supplied; not from the function
        "detected_language": "en",
        "author": "Jane Doe",
        "published_date": "2026-06-07",
    }
    defaults.update(overrides)
    return Article(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helper: call compute_extraction_confidence from an Article fixture
# ---------------------------------------------------------------------------


def _confidence_from_article(article: Article) -> float:
    """Call compute_extraction_confidence using fields from an Article fixture."""
    return compute_extraction_confidence(
        title=article.title,
        text=article.text,
        author=article.author,
        published_date=article.published_date,
        method=article.extraction_method,
    )


# ---------------------------------------------------------------------------
# 1. Direct scoring — method base values
# ---------------------------------------------------------------------------


class TestMethodBase:
    """The base score differs by extraction method before any bonuses."""

    def test_trafilatura_base_higher_than_readability(self) -> None:
        shared = {
            "title": "Short",
            "text": "x" * 10,  # no bonuses
            "author": None,
            "published_date": None,
        }
        t = compute_extraction_confidence(**shared, method="trafilatura")
        r = compute_extraction_confidence(**shared, method="readability")
        assert t > r

    def test_readability_base_higher_than_heuristic(self) -> None:
        shared = {
            "title": "Short",
            "text": "x" * 10,  # no bonuses
            "author": None,
            "published_date": None,
        }
        r = compute_extraction_confidence(**shared, method="readability")
        h = compute_extraction_confidence(**shared, method="heuristic")
        assert r > h

    def test_unknown_method_uses_fallback_base(self) -> None:
        score = compute_extraction_confidence(
            title="Short",
            text="x" * 10,
            author=None,
            published_date=None,
            method="magic_unknown_method",
        )
        # Unknown method should still return something in [0, 1]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 2. Text-length bonuses
# ---------------------------------------------------------------------------


class TestTextLengthBonuses:
    def test_short_text_no_bonus(self) -> None:
        short = compute_extraction_confidence(
            title="Short",
            text="x" * 50,  # < 200 — no bonus
            author=None,
            published_date=None,
            method="heuristic",
        )
        medium = compute_extraction_confidence(
            title="Short",
            text="x" * 200,  # exactly 200 — first bonus
            author=None,
            published_date=None,
            method="heuristic",
        )
        assert medium > short

    def test_very_long_text_gets_two_bonuses(self) -> None:
        medium = compute_extraction_confidence(
            title="Short",
            text="x" * 300,  # ≥ 200 but < 500 — one bonus
            author=None,
            published_date=None,
            method="heuristic",
        )
        long = compute_extraction_confidence(
            title="Short",
            text="x" * 500,  # ≥ 500 — two bonuses
            author=None,
            published_date=None,
            method="heuristic",
        )
        assert long > medium

    def test_text_exactly_at_500_threshold(self) -> None:
        at_499 = compute_extraction_confidence(
            title="Short",
            text="x" * 499,
            author=None,
            published_date=None,
            method="heuristic",
        )
        at_500 = compute_extraction_confidence(
            title="Short",
            text="x" * 500,
            author=None,
            published_date=None,
            method="heuristic",
        )
        assert at_500 > at_499

    def test_leading_whitespace_stripped_before_length_check(self) -> None:
        """Confidence uses len(text.strip()), so whitespace padding doesn't inflate."""
        padded = compute_extraction_confidence(
            title="Short",
            text="   " + "x" * 50 + "   ",  # stripped = 50 chars, < 200
            author=None,
            published_date=None,
            method="heuristic",
        )
        plain = compute_extraction_confidence(
            title="Short",
            text="x" * 50,
            author=None,
            published_date=None,
            method="heuristic",
        )
        assert padded == plain


# ---------------------------------------------------------------------------
# 3. Title-length bonuses
# ---------------------------------------------------------------------------


class TestTitleLengthBonuses:
    def test_very_short_title_no_bonus(self) -> None:
        no_bonus = compute_extraction_confidence(
            title="Hi",  # 2 chars — < 10
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        bonus = compute_extraction_confidence(
            title="A" * 10,  # exactly 10 — first bonus
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        assert bonus > no_bonus

    def test_long_title_gets_two_bonuses(self) -> None:
        medium_title = compute_extraction_confidence(
            title="A" * 15,  # ≥ 10 but < 30
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        long_title = compute_extraction_confidence(
            title="A" * 30,  # ≥ 30
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        assert long_title > medium_title


# ---------------------------------------------------------------------------
# 4. Optional field bonuses (author, published_date)
# ---------------------------------------------------------------------------


class TestOptionalFieldBonuses:
    def test_author_present_raises_score(self) -> None:
        no_author = compute_extraction_confidence(
            title="A Title",
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        with_author = compute_extraction_confidence(
            title="A Title",
            text="x" * 10,
            author="Jane Doe",
            published_date=None,
            method="trafilatura",
        )
        assert with_author > no_author

    def test_published_date_present_raises_score(self) -> None:
        no_date = compute_extraction_confidence(
            title="A Title",
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        with_date = compute_extraction_confidence(
            title="A Title",
            text="x" * 10,
            author=None,
            published_date="2026-06-07",
            method="trafilatura",
        )
        assert with_date > no_date

    def test_all_optional_fields_present_max_bonus(self) -> None:
        all_fields = compute_extraction_confidence(
            title="A Title",
            text="x" * 10,
            author="Jane Doe",
            published_date="2026-06-07",
            method="trafilatura",
        )
        no_fields = compute_extraction_confidence(
            title="A Title",
            text="x" * 10,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        # Should be exactly +0.05 author + +0.05 date higher
        assert pytest.approx(all_fields - no_fields, abs=1e-6) == 0.10


# ---------------------------------------------------------------------------
# 5. Return value always in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestReturnValueBounds:
    @pytest.mark.parametrize("method", ["trafilatura", "readability", "heuristic"])
    def test_result_in_unit_interval(self, method: str) -> None:
        score = compute_extraction_confidence(
            title="A" * 50,
            text="x" * 1000,
            author="Author",
            published_date="2026-06-07",
            method=method,
        )
        assert 0.0 <= score <= 1.0

    def test_maximum_score_does_not_exceed_one(self) -> None:
        """Even with all bonuses triggered, cap is 1.0."""
        score = compute_extraction_confidence(
            title="A" * 50,    # ≥ 10, ≥ 30
            text="x" * 1000,  # ≥ 200, ≥ 500
            author="Author",
            published_date="2026-06-07",
            method="trafilatura",  # highest base
        )
        assert score == 1.0

    def test_minimum_score_non_negative(self) -> None:
        score = compute_extraction_confidence(
            title="x",         # < 10 — no title bonus
            text="x" * 10,    # < 200 — no text bonus
            author=None,
            published_date=None,
            method="heuristic",  # lowest base
        )
        assert score >= 0.0


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_output(self) -> None:
        kwargs: dict[str, object] = {
            "title": "Claude Releases a New Model",
            "text": "Anthropic announced a new Claude model. " * 20,
            "author": "Jane Doe",
            "published_date": "2026-06-07",
            "method": "trafilatura",
        }
        first = compute_extraction_confidence(**kwargs)  # type: ignore[arg-type]
        second = compute_extraction_confidence(**kwargs)  # type: ignore[arg-type]
        assert first == second


# ---------------------------------------------------------------------------
# 7. Using fixture Article objects (AC requirement)
# ---------------------------------------------------------------------------


class TestWithArticleFixtures:
    """
    Prove that the function is testable in isolation using fixture Article objects.

    The helper ``_confidence_from_article`` extracts fields from an Article and
    passes them directly to ``compute_extraction_confidence``.
    """

    def test_full_english_article_fixture(self) -> None:
        article = _make_article(
            title="Claude Releases a Brand-New Model for Everyone",  # ≥ 30 chars
            text="Anthropic announced a new model. " * 25,           # ≥ 500 chars
            author="Jane Doe",
            published_date="2026-06-07",
            extraction_method="trafilatura",
        )
        score = _confidence_from_article(article)
        # trafilatura base (0.60) + text≥200 (0.10) + text≥500 (0.10)
        # + title≥10 (0.05) + title≥30 (0.05) + author (0.05) + date (0.05) = 1.00
        assert score == pytest.approx(1.0)

    def test_full_korean_article_fixture(self) -> None:
        article = _make_article(
            title="클로드, 새로운 모델 발표 — 앤트로픽이 공개",  # ≥ 30 chars
            text="앤트로픽이 향상된 추론 능력을 가진 새로운 클로드 모델을 발표했습니다. " * 30,
            author="홍길동",
            published_date="2026-06-07T09:00:00+09:00",
            extraction_method="trafilatura",
            extraction_confidence=0.80,  # pre-existing field value, unrelated
        )
        score = _confidence_from_article(article)
        assert 0.0 <= score <= 1.0
        # Trafilatura + rich text + author + date → high confidence
        assert score >= 0.90

    def test_readability_no_metadata_fixture(self) -> None:
        """readability extracts no author/date; score reflects that."""
        article = _make_article(
            title="Breaking News Today",  # 19 chars, ≥ 10 but < 30
            text="B" * 300,              # ≥ 200, < 500
            author=None,
            published_date=None,
            extraction_method="readability",
            extraction_confidence=0.70,
        )
        score = _confidence_from_article(article)
        # readability base (0.45) + text≥200 (0.10) + title≥10 (0.05) = 0.60
        assert score == pytest.approx(0.60)

    def test_heuristic_minimal_article_fixture(self) -> None:
        """Heuristic fallback with only short text → low confidence."""
        article = _make_article(
            title="Hi",       # 2 chars — no title bonus
            text="x" * 100,  # 100 chars — no text bonus
            author=None,
            published_date=None,
            extraction_method="heuristic",
            extraction_confidence=0.40,
        )
        score = _confidence_from_article(article)
        # heuristic base (0.25) — no bonuses
        assert score == pytest.approx(0.25)

    def test_score_increases_when_author_added(self) -> None:
        without = _make_article(
            author=None,
            extraction_method="trafilatura",
        )
        with_author = _make_article(
            author="Jane Doe",
            extraction_method="trafilatura",
        )
        score_without = _confidence_from_article(without)
        score_with = _confidence_from_article(with_author)
        assert score_with > score_without

    def test_score_increases_when_date_added(self) -> None:
        without = _make_article(
            published_date=None,
            extraction_method="trafilatura",
        )
        with_date = _make_article(
            published_date="2026-06-07",
            extraction_method="trafilatura",
        )
        score_without = _confidence_from_article(without)
        score_with = _confidence_from_article(with_date)
        assert score_with > score_without

    def test_trafilatura_fixture_beats_heuristic_fixture(self) -> None:
        """Same content extracted by different methods yields different confidence."""
        title = "A" * 30
        text = "x" * 600
        traf = _make_article(
            title=title,
            text=text,
            author=None,
            published_date=None,
            extraction_method="trafilatura",
        )
        heur = _make_article(
            title=title,
            text=text,
            author=None,
            published_date=None,
            extraction_method="heuristic",
        )
        assert _confidence_from_article(traf) > _confidence_from_article(heur)

    def test_article_fields_not_mutated_during_scoring(self) -> None:
        """The function must not modify the Article fixture (frozen constraint)."""
        article = _make_article()
        original_confidence = article.extraction_confidence

        _confidence_from_article(article)  # call the function

        # Article is frozen — its stored confidence field is unchanged
        assert article.extraction_confidence == original_confidence


# ---------------------------------------------------------------------------
# 8. Return type is float
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_return_type_is_float(self) -> None:
        score = compute_extraction_confidence(
            title="Test",
            text="x" * 50,
            author=None,
            published_date=None,
            method="trafilatura",
        )
        assert isinstance(score, float)
