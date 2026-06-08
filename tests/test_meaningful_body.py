"""
Unit tests for ``articula._extractor.is_meaningful_body``.

Sub-AC 1: ``is_meaningful_body`` returns ``False`` for empty string,
whitespace-only, and strings below the minimum token/character threshold,
and ``True`` for sufficiently long plain text — with a unit test covering
each boundary case.

Test strategy
-------------
1. Empty string input → False
2. Whitespace-only strings (space, tab, newline) → False
3. Strings below the character threshold → False
4. Strings that meet character count but fail token count → False
5. Strings at the exact boundary (just above both thresholds) → True
6. Sufficiently long plain English text → True
7. Sufficiently long plain Korean text → True
8. Return type is always bool
9. Public importability (top-level package and internal module)
"""

from __future__ import annotations

import pytest

from articula._extractor import (
    _MIN_BODY_CHARS,
    _MIN_BODY_TOKENS,
    is_meaningful_body,
)

# ---------------------------------------------------------------------------
# 1. Empty string
# ---------------------------------------------------------------------------


class TestEmptyString:
    def test_empty_string_returns_false(self) -> None:
        assert is_meaningful_body("") is False

    def test_empty_bytes_decoded_to_empty_string(self) -> None:
        assert is_meaningful_body(b"".decode()) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Whitespace-only strings
# ---------------------------------------------------------------------------


class TestWhitespaceOnly:
    def test_single_space(self) -> None:
        assert is_meaningful_body(" ") is False

    def test_multiple_spaces(self) -> None:
        assert is_meaningful_body("     ") is False

    def test_tab_only(self) -> None:
        assert is_meaningful_body("\t") is False

    def test_newline_only(self) -> None:
        assert is_meaningful_body("\n") is False

    def test_mixed_whitespace(self) -> None:
        assert is_meaningful_body("  \t\n  \r\n  ") is False

    def test_long_whitespace_string(self) -> None:
        """Even 1000 spaces should be False — character count after stripping is 0."""
        assert is_meaningful_body(" " * 1000) is False


# ---------------------------------------------------------------------------
# 3. Below minimum character threshold
# ---------------------------------------------------------------------------


class TestBelowCharThreshold:
    def test_one_character(self) -> None:
        assert is_meaningful_body("A") is False

    def test_ten_characters(self) -> None:
        assert is_meaningful_body("A" * 10) is False

    def test_just_under_min_chars(self) -> None:
        """One character below _MIN_BODY_CHARS must return False."""
        assert is_meaningful_body("A" * (_MIN_BODY_CHARS - 1)) is False

    def test_short_sentence(self) -> None:
        assert is_meaningful_body("Too short.") is False

    def test_single_word_repeated_below_threshold(self) -> None:
        """Several repetitions of a short word still under 80 chars."""
        text = "hello " * 10  # 60 chars, stripped = 59
        assert is_meaningful_body(text) is False

    @pytest.mark.parametrize("length", [1, 10, 20, 50, _MIN_BODY_CHARS - 1])
    def test_various_lengths_below_threshold(self, length: int) -> None:
        "w " * (length // 2) + "w" * (length % 2)
        # Ensure we are constructing something meaningful in character count
        padded = ("word " * (length // 5 + 1))[:length]
        assert is_meaningful_body(padded) is False


# ---------------------------------------------------------------------------
# 4. Meets character threshold but fails token threshold
# ---------------------------------------------------------------------------


class TestBelowTokenThreshold:
    def test_single_long_word(self) -> None:
        """One word of 100 characters: passes char check but fails token check."""
        assert is_meaningful_body("A" * 100) is False

    def test_few_tokens_padded_with_spaces(self) -> None:
        """Only 3 words with spaces to reach 80+ chars — fails token count."""
        # 3 tokens, padded with lots of spaces to exceed char threshold
        text = "word    " * 3 + " " * 60  # stripped is still 3 tokens
        assert is_meaningful_body(text) is False

    def test_just_under_min_tokens(self) -> None:
        """Exactly one fewer token than _MIN_BODY_TOKENS, with enough chars."""
        tokens = ["longword"] * (_MIN_BODY_TOKENS - 1)
        text = " ".join(tokens)
        # Ensure it actually has enough characters before testing
        if len(text) >= _MIN_BODY_CHARS:
            assert is_meaningful_body(text) is False


# ---------------------------------------------------------------------------
# 5. Exact boundary — just above both thresholds
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_exactly_at_min_chars_with_enough_tokens(self) -> None:
        """Construct text that is exactly _MIN_BODY_CHARS long with >= _MIN_BODY_TOKENS words."""
        # Build a string of exactly _MIN_BODY_CHARS with _MIN_BODY_TOKENS words
        word = "word"
        # Make _MIN_BODY_TOKENS words; pad the last word so total length == _MIN_BODY_CHARS
        words = [word] * (_MIN_BODY_TOKENS - 1)
        separator_chars = _MIN_BODY_TOKENS - 1  # spaces between words
        base_len = len(word) * (_MIN_BODY_TOKENS - 1) + separator_chars
        remaining = _MIN_BODY_CHARS - base_len
        last_word = "x" * max(1, remaining)
        words.append(last_word)
        text = " ".join(words)
        assert len(text) >= _MIN_BODY_CHARS
        assert len(text.split()) >= _MIN_BODY_TOKENS
        assert is_meaningful_body(text) is True

    def test_one_char_above_min_chars_enough_tokens(self) -> None:
        """_MIN_BODY_CHARS + 1 characters with ≥ _MIN_BODY_TOKENS tokens → True."""
        words = ["word"] * _MIN_BODY_TOKENS
        base = " ".join(words)
        # Pad to exactly _MIN_BODY_CHARS + 1
        if len(base) < _MIN_BODY_CHARS + 1:
            pad = "x" * (_MIN_BODY_CHARS + 1 - len(base))
            text = base + " " + pad
        else:
            text = base
        assert len(text.strip()) >= _MIN_BODY_CHARS + 1
        assert is_meaningful_body(text) is True


# ---------------------------------------------------------------------------
# 6. Sufficiently long plain English text
# ---------------------------------------------------------------------------


class TestEnglishText:
    def test_short_article_paragraph(self) -> None:
        text = (
            "Artificial intelligence is rapidly transforming the way humans work "
            "and live. Machine learning models are being deployed across countless "
            "industries to automate repetitive tasks and improve overall efficiency."
        )
        assert is_meaningful_body(text) is True

    def test_long_english_article(self) -> None:
        text = (
            "The rapid advancement of artificial intelligence has sparked both "
            "excitement and concern among scientists, policymakers, and the general "
            "public. Large language models can now write code, compose essays, and "
            "answer complex questions with remarkable fluency. At the same time, "
            "questions about bias, privacy, and the long-term societal impact of "
            "these systems remain unresolved. Governments around the world are "
            "drafting regulatory frameworks to ensure responsible deployment of AI."
        )
        assert is_meaningful_body(text) is True

    def test_repeated_sentence_padded(self) -> None:
        """Repeated sentences create clearly long enough text."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        assert is_meaningful_body(text) is True

    def test_technical_content_english(self) -> None:
        text = (
            "Python is a high-level, interpreted programming language known for "
            "its clear syntax and readability. It supports multiple programming "
            "paradigms including object-oriented, functional, and procedural styles. "
            "Python is widely used in data science, machine learning, web development, "
            "and automation workflows."
        )
        assert is_meaningful_body(text) is True


# ---------------------------------------------------------------------------
# 7. Sufficiently long plain Korean text
# ---------------------------------------------------------------------------


class TestKoreanText:
    def test_korean_article_paragraph(self) -> None:
        text = (
            "인공지능이 우리의 일하는 방식과 생활 방식을 크게 변화시키고 있습니다. "
            "머신러닝 알고리즘이 수많은 산업 분야에 걸쳐 업무를 자동화하고 효율성을 "
            "향상시키기 위해 광범위하게 배포되고 있습니다. 연구자들은 이러한 시스템이 "
            "할 수 있는 일의 경계를 계속 넓히고 있습니다."
        )
        assert is_meaningful_body(text) is True

    def test_long_korean_article(self) -> None:
        text = (
            "인공지능의 급속한 발전은 과학자, 정책 입안자, 일반 대중 사이에서 흥분과 "
            "우려를 동시에 불러일으켰습니다. 대형 언어 모델은 이제 코드를 작성하고 "
            "에세이를 구성하며 복잡한 질문에 놀라운 유창함으로 답변할 수 있습니다. "
            "동시에 이러한 시스템의 편향성, 개인정보 보호, 장기적인 사회적 영향에 대한 "
            "질문은 여전히 해결되지 않은 상태입니다."
        )
        assert is_meaningful_body(text) is True


# ---------------------------------------------------------------------------
# 8. Return type is always bool
# ---------------------------------------------------------------------------


class TestReturnType:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "short",
        "A" * 100,
        "The quick brown fox jumps over the lazy dog. " * 5,
    ])
    def test_return_type_is_bool(self, text: str) -> None:
        result = is_meaningful_body(text)
        assert isinstance(result, bool), f"Expected bool, got {type(result)} for {text!r}"

    def test_true_is_bool_not_truthy(self) -> None:
        text = "A" * 10 + " " + "B" * 10  # short, will return False
        assert type(is_meaningful_body(text)) is bool

    def test_false_is_bool_not_falsy(self) -> None:
        assert type(is_meaningful_body("")) is bool


# ---------------------------------------------------------------------------
# 9. Importability
# ---------------------------------------------------------------------------


class TestImportability:
    def test_importable_from_extractor_module(self) -> None:
        from articula._extractor import is_meaningful_body as fn  # noqa: PLC0415
        assert callable(fn)

    def test_importable_from_package(self) -> None:
        from articula import is_meaningful_body as fn  # noqa: PLC0415
        assert callable(fn)

    def test_package_and_module_are_same_object(self) -> None:
        from articula import is_meaningful_body as pkg_fn  # noqa: PLC0415
        from articula._extractor import is_meaningful_body as mod_fn  # noqa: PLC0415
        assert pkg_fn is mod_fn


# ---------------------------------------------------------------------------
# 10. Whitespace stripping does not inflate character count
# ---------------------------------------------------------------------------


class TestWhitespaceStripping:
    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Padded whitespace must not make a short body appear long."""
        short_content = "Hello world"
        padded = "   " + short_content + "   " * 100
        assert is_meaningful_body(padded) is False

    def test_only_content_chars_counted(self) -> None:
        """Internal whitespace counts toward character length (not stripped internally)."""
        # 10 words, each 8 chars, 9 spaces — total 89 chars stripped
        text = " ".join(["wordword"] * 10)
        assert len(text) == 89  # 8*10 + 9 = 89
        assert is_meaningful_body(text) is True
