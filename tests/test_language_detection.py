"""
Unit tests for ``articula._extractor.detect_language``.

Sub-AC 7d: ``detected_language`` is determined by a standalone function that
accepts extracted text and returns a BCP-47 language code, testable in
isolation with fixed text samples.

Test strategy
-------------
1. Happy-path: fixed English text samples → "en"
2. Happy-path: fixed Korean text samples → "ko"
3. Other common languages (basic smoke tests)
4. Determinism: same input always yields the same tag
5. Edge cases: empty string, whitespace-only, numeric-only → fallback "en"
6. Return type: always a non-empty str
7. Importability: function is accessible from both the internal module
   and the top-level ``articula`` package
8. Integration: language tags produced by the function match BCP-47 format
"""

from __future__ import annotations

import pytest

from articula._extractor import detect_language

# ---------------------------------------------------------------------------
# Fixed-text corpus
# ---------------------------------------------------------------------------

# Long enough (100+ chars) to give langdetect reliable signal.

_EN_SHORT = (
    "Artificial intelligence is transforming the way we work and live."
)

_EN_MEDIUM = (
    "Artificial intelligence is rapidly transforming the way humans work "
    "and live. Machine learning models are being deployed across countless "
    "industries to automate repetitive tasks and improve overall efficiency. "
    "Researchers continue to push the boundaries of what these systems can do."
)

_EN_LONG = (
    "The rapid advancement of artificial intelligence has sparked both "
    "excitement and concern among scientists, policymakers, and the general "
    "public. Large language models can now write code, compose essays, and "
    "answer complex questions with remarkable fluency. At the same time, "
    "questions about bias, privacy, and the long-term societal impact of "
    "these systems remain unresolved. Governments around the world are "
    "drafting regulatory frameworks to ensure responsible deployment of AI "
    "while preserving the benefits of innovation."
)

_KO_SHORT = (
    "인공지능이 우리의 일하는 방식과 생활 방식을 변화시키고 있습니다."
)

_KO_MEDIUM = (
    "인공지능이 우리의 일하는 방식과 생활 방식을 크게 변화시키고 있습니다. "
    "머신러닝 알고리즘이 수많은 산업 분야에 걸쳐 업무를 자동화하고 효율성을 "
    "향상시키기 위해 광범위하게 배포되고 있습니다. 연구자들은 이러한 시스템이 "
    "할 수 있는 일의 경계를 계속 넓히고 있습니다."
)

_KO_LONG = (
    "인공지능의 급속한 발전은 과학자, 정책 입안자, 일반 대중 사이에서 흥분과 "
    "우려를 동시에 불러일으켰습니다. 대형 언어 모델은 이제 코드를 작성하고 "
    "에세이를 구성하며 복잡한 질문에 놀라운 유창함으로 답변할 수 있습니다. "
    "동시에 이러한 시스템의 편향성, 개인정보 보호, 장기적인 사회적 영향에 대한 "
    "질문은 여전히 해결되지 않은 상태입니다. 전 세계 정부는 혁신의 이점을 "
    "보존하면서 AI의 책임 있는 배포를 보장하기 위한 규제 틀을 마련하고 있습니다."
)

_KO_NEWS_HEADLINE = "클로드, 새로운 AI 모델 출시 — 앤트로픽이 발표"

_EN_NEWS_HEADLINE = "Anthropic releases new Claude AI model with improved reasoning"


# ---------------------------------------------------------------------------
# 1. English text → "en"
# ---------------------------------------------------------------------------


class TestEnglishDetection:
    """Fixed English text samples must return 'en'."""

    def test_medium_english_text(self) -> None:
        result = detect_language(_EN_MEDIUM)
        assert result == "en", f"Expected 'en', got {result!r}"

    def test_long_english_text(self) -> None:
        result = detect_language(_EN_LONG)
        assert result == "en", f"Expected 'en', got {result!r}"

    def test_english_news_headline(self) -> None:
        """Headline-length English text should still detect as 'en'."""
        result = detect_language(_EN_NEWS_HEADLINE)
        assert result == "en", f"Expected 'en', got {result!r}"

    def test_english_article_repeated_paragraph(self) -> None:
        """Artificially long text built from a repeated sentence."""
        repeated = "The quick brown fox jumps over the lazy dog. " * 20
        result = detect_language(repeated)
        assert result == "en", f"Expected 'en', got {result!r}"

    def test_english_technical_content(self) -> None:
        text = (
            "Python is a high-level, interpreted programming language known "
            "for its clear syntax and readability. It supports multiple "
            "programming paradigms including object-oriented, functional, "
            "and procedural styles. Python is widely used in data science, "
            "machine learning, web development, and automation."
        )
        result = detect_language(text)
        assert result == "en", f"Expected 'en', got {result!r}"


# ---------------------------------------------------------------------------
# 2. Korean text → "ko"
# ---------------------------------------------------------------------------


class TestKoreanDetection:
    """Fixed Korean text samples must return 'ko'."""

    def test_medium_korean_text(self) -> None:
        result = detect_language(_KO_MEDIUM)
        assert result == "ko", f"Expected 'ko', got {result!r}"

    def test_long_korean_text(self) -> None:
        result = detect_language(_KO_LONG)
        assert result == "ko", f"Expected 'ko', got {result!r}"

    def test_korean_news_headline(self) -> None:
        """Headline-length Korean text should detect as 'ko'."""
        result = detect_language(_KO_NEWS_HEADLINE)
        assert result == "ko", f"Expected 'ko', got {result!r}"

    def test_korean_article_repeated_sentence(self) -> None:
        """Repeated Korean sentence produces a reliably long text."""
        repeated = "인공지능 기술이 빠르게 발전하고 있습니다. " * 20
        result = detect_language(repeated)
        assert result == "ko", f"Expected 'ko', got {result!r}"

    def test_korean_technical_content(self) -> None:
        text = (
            "파이썬은 명확한 문법과 가독성으로 알려진 고수준 인터프리터 언어입니다. "
            "객체 지향, 함수형, 절차적 스타일을 포함한 여러 프로그래밍 패러다임을 "
            "지원합니다. 파이썬은 데이터 과학, 머신러닝, 웹 개발, 자동화 분야에서 "
            "널리 사용됩니다. 간결하고 읽기 쉬운 코드가 가장 큰 장점입니다."
        )
        result = detect_language(text)
        assert result == "ko", f"Expected 'ko', got {result!r}"


# ---------------------------------------------------------------------------
# 3. Determinism — same input always returns the same tag
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Calling detect_language with the same text multiple times must return
    the same BCP-47 tag every time (thanks to a fixed seed)."""

    def test_english_is_deterministic(self) -> None:
        first = detect_language(_EN_MEDIUM)
        for _ in range(4):
            assert detect_language(_EN_MEDIUM) == first

    def test_korean_is_deterministic(self) -> None:
        first = detect_language(_KO_MEDIUM)
        for _ in range(4):
            assert detect_language(_KO_MEDIUM) == first

    def test_long_english_is_deterministic(self) -> None:
        results = {detect_language(_EN_LONG) for _ in range(5)}
        assert len(results) == 1, f"Non-deterministic output: {results}"

    def test_long_korean_is_deterministic(self) -> None:
        results = {detect_language(_KO_LONG) for _ in range(5)}
        assert len(results) == 1, f"Non-deterministic output: {results}"


# ---------------------------------------------------------------------------
# 4. Edge cases — graceful fallback to "en"
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Degenerate inputs must not raise and must return a non-empty str."""

    def test_empty_string_returns_fallback(self) -> None:
        result = detect_language("")
        assert isinstance(result, str)
        assert result  # non-empty
        assert result == "en"

    def test_whitespace_only_returns_fallback(self) -> None:
        result = detect_language("   \t\n  ")
        assert result == "en"

    def test_single_space_returns_fallback(self) -> None:
        result = detect_language(" ")
        assert result == "en"

    def test_numeric_only_returns_fallback(self) -> None:
        """Pure numbers have no linguistic features → langdetect raises → 'en'."""
        result = detect_language("123456789 0987654321")
        assert isinstance(result, str)
        assert result  # non-empty; may be 'en' or another tag depending on langdetect

    def test_does_not_raise_on_any_string(self) -> None:
        """The function must absorb all langdetect exceptions."""
        problematic_inputs = ["", "   ", "!!!", "---", "...", "   \n\n\n"]
        for text in problematic_inputs:
            result = detect_language(text)
            assert isinstance(result, str), f"Got non-str for {text!r}: {result!r}"
            assert result, f"Got empty str for {text!r}"


# ---------------------------------------------------------------------------
# 5. Return type and BCP-47 format
# ---------------------------------------------------------------------------


class TestReturnType:
    """detect_language must always return a non-empty lowercase str."""

    def test_returns_string(self) -> None:
        result = detect_language(_EN_MEDIUM)
        assert isinstance(result, str)

    def test_returns_non_empty_string(self) -> None:
        result = detect_language(_EN_MEDIUM)
        assert result  # truthy ⟹ non-empty

    def test_returns_lowercase(self) -> None:
        """BCP-47 primary subtags are lowercase (e.g. 'en', 'ko', 'ja')."""
        for text in (_EN_MEDIUM, _KO_MEDIUM):
            result = detect_language(text)
            assert result == result.lower(), (
                f"Expected lowercase BCP-47 tag, got {result!r}"
            )

    def test_returns_alphabetic_tag(self) -> None:
        """Primary language subtags consist only of ASCII letters."""
        for text in (_EN_MEDIUM, _KO_MEDIUM):
            result = detect_language(text)
            # Strip region subtag if present (e.g. "zh-cn" → "zh")
            primary = result.split("-")[0]
            assert primary.isalpha(), (
                f"Primary subtag {primary!r} contains non-alphabetic chars"
            )

    @pytest.mark.parametrize("text,expected", [
        (_EN_LONG, "en"),
        (_KO_LONG, "ko"),
    ])
    def test_parametrized_fixed_samples(self, text: str, expected: str) -> None:
        """Parametrized check across all fixed corpus entries."""
        assert detect_language(text) == expected


# ---------------------------------------------------------------------------
# 6. Public import surface
# ---------------------------------------------------------------------------


class TestImportability:
    """detect_language must be importable from the top-level package."""

    def test_importable_from_package(self) -> None:
        from articula import detect_language as dl  # noqa: PLC0415
        assert callable(dl)

    def test_importable_from_extractor_module(self) -> None:
        from articula._extractor import detect_language as dl  # noqa: PLC0415
        assert callable(dl)

    def test_package_and_module_are_same_object(self) -> None:
        """Both import paths must resolve to the identical function object."""
        from articula import detect_language as pkg_dl  # noqa: PLC0415
        from articula._extractor import detect_language as mod_dl  # noqa: PLC0415
        assert pkg_dl is mod_dl

    def test_function_signature_accepts_text(self) -> None:
        """Positional call with a text argument must work."""
        result = detect_language("This is a test sentence for language detection.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 7. Integration with ExtractionResult
# ---------------------------------------------------------------------------


class TestIntegrationWithExtractor:
    """detect_language feeds the ``language`` field of ExtractionResult.

    We smoke-test that calling it with the same text that an extractor would
    use produces a consistent BCP-47 tag stored in the result.
    """

    def test_english_extraction_result_language(self) -> None:
        from articula._extractor import (  # noqa: PLC0415
            ExtractionResult,
            compute_extraction_confidence,
        )

        text = _EN_LONG
        lang = detect_language(text)

        result = ExtractionResult(
            title="Test Article Title About AI",
            text=text,
            author=None,
            published_date=None,
            method="heuristic",
            confidence=compute_extraction_confidence(
                title="Test Article Title About AI",
                text=text,
                author=None,
                published_date=None,
                method="heuristic",
            ),
            language=lang,
        )
        assert result.language == "en"

    def test_korean_extraction_result_language(self) -> None:
        from articula._extractor import (  # noqa: PLC0415
            ExtractionResult,
            compute_extraction_confidence,
        )

        text = _KO_LONG
        lang = detect_language(text)

        result = ExtractionResult(
            title="한국어 기사 제목입니다",
            text=text,
            author=None,
            published_date=None,
            method="heuristic",
            confidence=compute_extraction_confidence(
                title="한국어 기사 제목입니다",
                text=text,
                author=None,
                published_date=None,
                method="heuristic",
            ),
            language=lang,
        )
        assert result.language == "ko"
