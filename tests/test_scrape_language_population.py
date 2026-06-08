"""
Sub-AC 7e: scrape(url) populates detected_language using the extracted text.

Test matrix
-----------
* English HTML content → Article.detected_language == "en"
* Korean (non-English) HTML content → Article.detected_language == "ko"
* Empty-text input:
    - detect_language("") → "en" (fallback documented; scrape() raises
      ExtractionError when there is no extractable body text, so language
      detection never runs — but detect_language itself gracefully falls back
      to "en" for empty input, which is what scrape() would pass)
    - scrape() raises ExtractionError when HTML contains no extractable body

All tests that exercise scrape() use a mocked fetcher so that no real network
requests are made.  The ``force_strategy="static"`` kwarg pins the scraper to
the static tier, ensuring only ``fetch_static`` is called.

Usage
-----
    pytest tests/test_scrape_language_population.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula import async_scrape
from articula._extractor import detect_language
from articula._fetcher import FetchResult
from articula.exceptions import ExtractionError

# ---------------------------------------------------------------------------
# Shared HTML fixtures
# ---------------------------------------------------------------------------

#: Minimal English article page — enough body text for all three extractors.
_EN_HTML = """\
<html>
<head><title>AI Research Trends</title></head>
<body>
<article>
  <h1>AI Research Trends</h1>
  <p>Artificial intelligence is rapidly transforming the way humans work and live.
  Machine learning models are being deployed across countless industries to automate
  repetitive tasks and improve overall efficiency. Researchers continue to push the
  boundaries of what these systems can do. The potential applications are vast and
  growing every day. Natural language processing and computer vision have seen
  remarkable advances in recent years, enabling new applications across healthcare,
  finance, and education sectors worldwide.</p>
</article>
</body>
</html>
"""

#: Minimal Korean article page — enough body text for all three extractors.
_KO_HTML = """\
<html>
<head><title>인공지능 연구 동향</title></head>
<body>
<article>
  <h1>인공지능 연구 동향</h1>
  <p>인공지능이 우리의 일하는 방식과 생활 방식을 크게 변화시키고 있습니다.
  머신러닝 알고리즘이 수많은 산업 분야에 걸쳐 업무를 자동화하고 효율성을 향상시키기 위해
  광범위하게 배포되고 있습니다. 연구자들은 이러한 시스템이 할 수 있는 일의 경계를 계속 넓히고 있습니다.
  특히 자연어 처리 및 컴퓨터 비전 영역에서 눈부신 성과가 나타나고 있습니다.
  의료, 금융, 교육 분야 전반에 걸쳐 새로운 응용 프로그램이 등없이 등장하고 있습니다.</p>
</article>
</body>
</html>
"""

#: HTML page with no extractable article body (all extractors will fail).
_EMPTY_CONTENT_HTML = "<html><body><p>x</p></body></html>"

#: Sentinel URL used in all mocked fetch results.
_TEST_URL = "https://example.com/article"


def _make_fetch_result(html: str, tier: str = "static") -> FetchResult:
    """Return a minimal FetchResult wrapping *html* for the given tier."""
    return FetchResult(
        html=html,
        resolved_url=_TEST_URL,
        strategy_tier=tier,
        status_code=200,
    )


# ---------------------------------------------------------------------------
# TestScrapeDetectedLanguageEnglish
# ---------------------------------------------------------------------------


class TestScrapeDetectedLanguageEnglish:
    """scrape(url) sets detected_language='en' for English-language pages."""

    async def test_english_page_detected_language_is_en(self) -> None:
        """English HTML → Article.detected_language == 'en'."""
        fetch_result = _make_fetch_result(_EN_HTML)

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ):
            article = await async_scrape(_TEST_URL, force_strategy="static")

        assert article.detected_language == "en", (
            f"Expected detected_language='en' for English content, "
            f"got {article.detected_language!r}"
        )

    async def test_english_detected_language_is_string(self) -> None:
        """detected_language is always a non-empty str, not None."""
        fetch_result = _make_fetch_result(_EN_HTML)

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ):
            article = await async_scrape(_TEST_URL, force_strategy="static")

        assert isinstance(article.detected_language, str)
        assert article.detected_language  # non-empty


# ---------------------------------------------------------------------------
# TestScrapeDetectedLanguageNonEnglish
# ---------------------------------------------------------------------------


class TestScrapeDetectedLanguageNonEnglish:
    """scrape(url) sets detected_language='ko' for Korean-language pages."""

    async def test_korean_page_detected_language_is_ko(self) -> None:
        """Korean HTML → Article.detected_language == 'ko'."""
        fetch_result = _make_fetch_result(_KO_HTML)

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ):
            article = await async_scrape(_TEST_URL, force_strategy="static")

        assert article.detected_language == "ko", (
            f"Expected detected_language='ko' for Korean content, "
            f"got {article.detected_language!r}"
        )

    async def test_korean_detected_language_differs_from_english(self) -> None:
        """The Korean page language code differs from the English page code."""
        en_fetch = _make_fetch_result(_EN_HTML)
        ko_fetch = _make_fetch_result(_KO_HTML)

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=en_fetch,
        ):
            en_article = await async_scrape(_TEST_URL, force_strategy="static")

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=ko_fetch,
        ):
            ko_article = await async_scrape(_TEST_URL, force_strategy="static")

        assert en_article.detected_language != ko_article.detected_language, (
            "English and Korean articles must produce different language codes"
        )
        assert en_article.detected_language == "en"
        assert ko_article.detected_language == "ko"


# ---------------------------------------------------------------------------
# TestScrapeEmptyTextBehaviour
# ---------------------------------------------------------------------------


class TestScrapeEmptyTextBehaviour:
    """Behaviour when there is no extractable body text.

    scrape() relies on ``detect_language(text)`` internally, where *text* is
    the body extracted from the page.  When the page contains no extractable
    content all three extractors (trafilatura, readability, heuristic) fail and
    scrape() raises ``ExtractionError``.

    The empty-string fallback in detect_language — returning 'en' — therefore
    applies to edge cases *within* an extractor, not at the scrape() boundary.
    Both behaviours are verified here.
    """

    async def test_no_extractable_content_raises_extraction_error(self) -> None:
        """HTML with no article body → ExtractionError (extraction, not fetch, fails)."""
        fetch_result = _make_fetch_result(_EMPTY_CONTENT_HTML)

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ), pytest.raises(ExtractionError):
            await async_scrape(_TEST_URL, force_strategy="static")

    def test_detect_language_empty_string_returns_en(self) -> None:
        """detect_language('') → 'en' (fallback used when text is absent).

        This is the language code scrape() would produce if the extracted text
        were empty — the function never raises and always returns a valid BCP-47
        tag.
        """
        result = detect_language("")
        assert result == "en", (
            f"Expected 'en' fallback for empty input, got {result!r}"
        )

    def test_detect_language_whitespace_only_returns_en(self) -> None:
        """detect_language with whitespace-only input returns 'en' fallback."""
        result = detect_language("   \t\n  ")
        assert result == "en", (
            f"Expected 'en' fallback for whitespace input, got {result!r}"
        )

    def test_detect_language_empty_never_raises(self) -> None:
        """detect_language must not raise for any empty-ish input."""
        for text in ("", " ", "\n", "\t"):
            result = detect_language(text)
            assert isinstance(result, str) and result, (
                f"Expected non-empty str for {text!r}, got {result!r}"
            )


# ---------------------------------------------------------------------------
# TestScrapeLanguagePopulationParametrized
# ---------------------------------------------------------------------------


class TestScrapeLanguagePopulationParametrized:
    """Parametrized cross-check: English and Korean use mocked fetcher."""

    @pytest.mark.parametrize(
        "html, expected_lang",
        [
            (_EN_HTML, "en"),
            (_KO_HTML, "ko"),
        ],
        ids=["english", "korean"],
    )
    async def test_detected_language_matches_content_language(
        self, html: str, expected_lang: str
    ) -> None:
        """Article.detected_language matches the dominant language in the HTML body."""
        fetch_result = _make_fetch_result(html)

        with patch(
            "articula._scraper.fetch_static",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ):
            article = await async_scrape(_TEST_URL, force_strategy="static")

        assert article.detected_language == expected_lang, (
            f"HTML tagged as {expected_lang!r}: "
            f"expected detected_language={expected_lang!r}, "
            f"got {article.detected_language!r}"
        )

    @pytest.mark.parametrize(
        "text, expected_lang",
        [
            ("", "en"),
            ("   ", "en"),
            ("\n\t\n", "en"),
        ],
        ids=["empty-string", "spaces", "whitespace"],
    )
    def test_detect_language_empty_variants_all_return_en(
        self, text: str, expected_lang: str
    ) -> None:
        """detect_language returns 'en' for all empty/whitespace inputs."""
        result = detect_language(text)
        assert result == expected_lang, (
            f"detect_language({text!r}) → expected {expected_lang!r}, got {result!r}"
        )
