"""
Sub-AC 3: Extractor function returns a populated result with ``body`` text set
and ``published_date=None`` when the page HTML contains body content but no
date markup.

Verified by unit tests using minimal HTML fixtures (offline, no network access).

The ``extract()`` function from ``_extractor.py`` returns an ``ExtractionResult``
(the intermediate extraction DTO used before assembly into a full ``Article``).
The AC refers to "Article" loosely — this test exercises the extractor layer
directly, where the absence of date markup must produce ``published_date=None``
while body text is still fully populated.

HTML fixtures are deliberately minimal:
- No ``<meta name="date">``, ``<meta property="article:published_time">``,
  ``<meta name="DC.date">``, or similar date meta tags.
- No ``<time datetime="...">`` elements.
- No schema.org ``datePublished`` JSON-LD properties.
- Body text is long enough to pass the ``_MIN_BODY_CHARS`` (80) and
  ``_MIN_BODY_TOKENS`` (10) thresholds enforced by every extraction tier.
"""

from __future__ import annotations

import pytest

from articula._extractor import ExtractionResult, extract

# ---------------------------------------------------------------------------
# HTML fixture helpers
# ---------------------------------------------------------------------------

# 280-char body; no date markup of any kind.  The fixture is structurally clean
# so that all three extraction tiers (trafilatura, readability, heuristic) can
# parse it, ensuring the test is not sensitive to which tier runs first.
_NO_DATE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sample Article Without Publication Date</title>
</head>
<body>
  <article>
    <h1>Sample Article Without Publication Date</h1>
    <p>
      This article contains substantial body text but deliberately omits all
      date-related markup.  There is no meta date tag, no time element, and
      no schema.org datePublished property anywhere in the document.
    </p>
    <p>
      A second paragraph adds more content to ensure the body comfortably
      exceeds the minimum character and token thresholds enforced by every
      extraction tier, making the test robust regardless of which extractor
      succeeds first.
    </p>
  </article>
</body>
</html>
"""

# Korean-language variant — same structure, no date markup.
_NO_DATE_HTML_KO = """\
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>날짜 없는 기사 예시</title>
</head>
<body>
  <article>
    <h1>날짜 없는 기사 예시</h1>
    <p>
      이 기사에는 충분한 본문 내용이 포함되어 있지만 날짜 관련 마크업은
      전혀 없습니다. 메타 태그, 타임 요소, 스키마 마크업 어디에도 게시일이
      표시되지 않습니다.
    </p>
    <p>
      두 번째 단락은 모든 추출 엔진의 최소 글자 및 토큰 요구 사항을
      충족하기 위해 추가된 내용입니다. 인공지능과 자연어 처리 기술이
      발전함에 따라 다양한 언어의 기사를 효율적으로 처리할 수 있게
      되었습니다.
    </p>
  </article>
</body>
</html>
"""

_TEST_URL = "https://example.com/no-date-article"
_TEST_URL_KO = "https://news.example.kr/no-date-article"


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


class TestExtractNoDateMarkup:
    """
    ``extract()`` correctly handles HTML with body content but no date markup:
    body text is populated AND ``published_date`` is ``None`` simultaneously.
    """

    def test_body_text_is_populated(self) -> None:
        """``ExtractionResult.text`` is non-empty when the page has body content."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)
        assert result.text.strip() != ""

    def test_published_date_is_none(self) -> None:
        """``ExtractionResult.published_date`` is ``None`` when HTML has no date markup."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        assert result.published_date is None

    def test_body_set_and_published_date_none_simultaneously(self) -> None:
        """Both conditions hold for a single extraction call — the main AC assertion."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        # body text must be present
        assert result.text.strip() != "", "body text should not be empty"
        # published_date must be absent
        assert result.published_date is None, (
            "published_date should be None when no date markup is present"
        )

    def test_title_is_populated(self) -> None:
        """``ExtractionResult.title`` is non-empty so the result is well-formed."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        assert result.title.strip() != ""

    def test_result_is_extraction_result_instance(self) -> None:
        """``extract()`` returns an ``ExtractionResult``, not ``None`` or an Article."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)

    def test_method_field_is_valid(self) -> None:
        """``extraction_method`` is one of the known extraction-tier identifiers."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        assert result.method in {"trafilatura", "readability", "heuristic"}

    def test_confidence_in_range(self) -> None:
        """Confidence score is within ``[0.0, 1.0]`` even when date is absent."""
        result = extract(_NO_DATE_HTML, _TEST_URL)

        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Korean-language variant
# ---------------------------------------------------------------------------


class TestExtractNoDateMarkupKorean:
    """
    Korean-language HTML with no date markup is handled identically.
    Both ``text`` and ``published_date`` conditions must hold.
    """

    def test_korean_body_set_and_published_date_none(self) -> None:
        """Korean article: body text is populated and published_date is None."""
        result = extract(_NO_DATE_HTML_KO, _TEST_URL_KO)

        assert result.text.strip() != "", "Korean body text should not be empty"
        assert result.published_date is None, (
            "published_date should be None for Korean no-date page"
        )

    def test_korean_title_populated(self) -> None:
        """Korean article title is extracted from heading or title element."""
        result = extract(_NO_DATE_HTML_KO, _TEST_URL_KO)

        assert result.title.strip() != ""


# ---------------------------------------------------------------------------
# Parametrized across fixture variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("html", "url", "label"),
    [
        (_NO_DATE_HTML, _TEST_URL, "english"),
        (_NO_DATE_HTML_KO, _TEST_URL_KO, "korean"),
    ],
    ids=["english", "korean"],
)
def test_extract_body_set_published_date_none_parametrized(
    html: str, url: str, label: str
) -> None:
    """
    Parametrized check: for both English and Korean minimal fixtures,
    ``extract()`` sets ``text`` and leaves ``published_date=None``.
    """
    result = extract(html, url)

    assert result.text.strip() != "", f"[{label}] body text should not be empty"
    assert result.published_date is None, (
        f"[{label}] published_date should be None when no date markup present"
    )
