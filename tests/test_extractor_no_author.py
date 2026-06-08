"""
Sub-AC 2: Extractor function returns body text with author=None when the page
HTML contains body content but no author markup.

Verifies with a minimal HTML fixture that ``extract()`` from ``_extractor.py``
returns an ``ExtractionResult`` satisfying both conditions simultaneously:

- ``text``   — non-empty body text (the article body was extracted)
- ``author`` — ``None`` (no author markup was present in the HTML)

HTML fixtures are deliberately minimal to focus the test on the author-absent
case while remaining large enough to pass the ``_MIN_BODY_CHARS`` / token
guards that every extraction tier enforces.
"""

from __future__ import annotations

import pytest

from articula._extractor import ExtractionResult, extract

# ---------------------------------------------------------------------------
# HTML fixture helpers
# ---------------------------------------------------------------------------

# 270-char body ensures all three extraction tiers can clear the _MIN_BODY_CHARS
# (80) and _MIN_BODY_TOKENS (10) thresholds.  No <meta name="author">, no
# byline element, and no schema.org author property are present so that every
# extraction backend returns author=None.
_NO_AUTHOR_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sample Article Without Author</title>
</head>
<body>
  <article>
    <h1>Sample Article Without Author</h1>
    <p>
      This article has substantial body content but contains absolutely no
      author markup.  There is no byline, no meta author tag, and no
      schema.org author property anywhere in the document.
    </p>
    <p>
      A second paragraph adds more text to ensure the body comfortably
      exceeds the minimum character and token thresholds enforced by every
      extraction tier so the test is not sensitive to the specific extractor
      that runs first.
    </p>
  </article>
</body>
</html>
"""

# Korean-language variant — same structure, no author markup.
_NO_AUTHOR_HTML_KO = """\
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>작성자 없는 기사 예시</title>
</head>
<body>
  <article>
    <h1>작성자 없는 기사 예시</h1>
    <p>
      이 기사에는 본문 내용이 충분히 포함되어 있지만 작성자 정보는 전혀
      없습니다. 메타 태그나 스키마 마크업에도 작성자가 표시되지 않습니다.
    </p>
    <p>
      두 번째 단락은 추출 엔진의 최소 글자 및 토큰 요구 사항을 충족하기
      위해 추가된 내용입니다. 인공지능과 자연어 처리 기술의 발전으로
      다양한 언어의 기사를 효율적으로 처리할 수 있습니다.
    </p>
  </article>
</body>
</html>
"""

_TEST_URL = "https://example.com/no-author-article"
_TEST_URL_KO = "https://news.example.kr/no-author-article"


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


class TestExtractNoAuthorMarkup:
    """
    ``extract()`` correctly handles HTML with body content but no author markup:
    body text is populated AND author is None simultaneously.
    """

    def test_body_text_is_populated(self) -> None:
        """``ExtractionResult.text`` is non-empty when the page has body content."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)
        assert result.text.strip() != ""

    def test_author_is_none(self) -> None:
        """``ExtractionResult.author`` is ``None`` when HTML has no author markup."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        assert result.author is None

    def test_body_set_and_author_none_simultaneously(self) -> None:
        """Both conditions hold for a single extraction call — the main AC assertion."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        # body text must be present
        assert result.text.strip() != "", "body text should not be empty"
        # author must be absent
        assert result.author is None, "author should be None when no markup present"

    def test_title_is_populated(self) -> None:
        """``ExtractionResult.title`` is non-empty so the result is well-formed."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        assert result.title.strip() != ""

    def test_result_is_extraction_result_instance(self) -> None:
        """``extract()`` returns an ``ExtractionResult``, not ``None`` or an Article."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)

    def test_method_field_is_valid(self) -> None:
        """``extraction_method`` is one of the known extraction-tier identifiers."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        assert result.method in {"trafilatura", "readability", "heuristic"}

    def test_confidence_in_range(self) -> None:
        """Confidence score is within ``[0.0, 1.0]`` even when author is absent."""
        result = extract(_NO_AUTHOR_HTML, _TEST_URL)

        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Korean-language variant
# ---------------------------------------------------------------------------


class TestExtractNoAuthorMarkupKorean:
    """
    Korean-language HTML with no author markup is handled identically.
    Both ``text`` and ``author`` conditions must hold.
    """

    def test_korean_body_set_and_author_none(self) -> None:
        """Korean article: body text is populated and author is None."""
        result = extract(_NO_AUTHOR_HTML_KO, _TEST_URL_KO)

        assert result.text.strip() != "", "Korean body text should not be empty"
        assert result.author is None, "author should be None for Korean no-author page"

    def test_korean_title_populated(self) -> None:
        """Korean article title is extracted from heading or title element."""
        result = extract(_NO_AUTHOR_HTML_KO, _TEST_URL_KO)

        assert result.title.strip() != ""


# ---------------------------------------------------------------------------
# Parametrized across fixture variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("html", "url", "label"),
    [
        (_NO_AUTHOR_HTML, _TEST_URL, "english"),
        (_NO_AUTHOR_HTML_KO, _TEST_URL_KO, "korean"),
    ],
    ids=["english", "korean"],
)
def test_extract_body_set_author_none_parametrized(
    html: str, url: str, label: str
) -> None:
    """
    Parametrized check: for both English and Korean minimal fixtures,
    ``extract()`` sets ``text`` and leaves ``author=None``.
    """
    result = extract(html, url)

    assert result.text.strip() != "", f"[{label}] body text should not be empty"
    assert result.author is None, f"[{label}] author should be None"
