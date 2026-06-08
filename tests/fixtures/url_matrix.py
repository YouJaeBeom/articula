"""URL matrix fixture for the articula test suite.

Defines a curated set of ~10 real-world URLs covering:
  - English and Korean content
  - Static server-rendered pages
  - Pages requiring User-Agent / header rotation (anti-bot)
  - JS-rendered pages requiring a headless browser

Each entry is an immutable :class:`UrlEntry` dataclass with four required
metadata fields:  ``url``, ``category``, ``language``, ``rendering_type``.

Usage
-----
    from tests.fixtures.url_matrix import URL_MATRIX, UrlEntry

    for entry in URL_MATRIX:
        print(entry.url, entry.language, entry.rendering_type)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases for constrained string fields
# ---------------------------------------------------------------------------

Category = Literal["wiki", "news", "tech", "blog", "finance"]
Language = Literal["en", "ko"]

#: Rendering strategy tier expected for a URL — mirrors the scraper's tiers.
RenderingType = Literal["static", "headers_rotation", "browser"]

#: Frozenset of field names that every :class:`UrlEntry` must carry.
REQUIRED_FIELDS: frozenset[str] = frozenset({"url", "category", "language", "rendering_type"})

#: Valid rendering type values aligned with the scraper's strategy tiers.
VALID_RENDERING_TYPES: frozenset[str] = frozenset({"static", "headers_rotation", "browser"})

#: Valid BCP-47 language codes represented in this matrix.
VALID_LANGUAGES: frozenset[str] = frozenset({"en", "ko"})

#: Valid content category labels used in this matrix.
VALID_CATEGORIES: frozenset[str] = frozenset({"wiki", "news", "tech", "blog", "finance"})


# ---------------------------------------------------------------------------
# UrlEntry dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UrlEntry:
    """Metadata describing a single URL in the integration test matrix.

    Attributes
    ----------
    url:
        Fully-qualified URL (must start with ``https://``).
    category:
        Content category.  One of ``"wiki"``, ``"news"``, ``"tech"``,
        ``"blog"``, ``"finance"``.
    language:
        BCP-47 language code of the expected article content.
        Currently ``"en"`` (English) or ``"ko"`` (Korean).
    rendering_type:
        Expected fetch strategy the scraper needs for this URL:

        * ``"static"``           — plain HTTP GET with default headers.
        * ``"headers_rotation"`` — anti-bot measures; requires UA/header
                                   rotation but the page is server-rendered.
        * ``"browser"``          — SPA / CSR; requires headless-browser
                                   JS execution (Playwright).
    """

    url: str
    category: Category
    language: Language
    rendering_type: RenderingType

    def __post_init__(self) -> None:
        if not self.url.startswith("https://"):
            raise ValueError(
                f"UrlEntry.url must start with 'https://', got: {self.url!r}"
            )
        if not self.url.strip():
            raise ValueError("UrlEntry.url must not be empty")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"UrlEntry.category must be one of {sorted(VALID_CATEGORIES)}, "
                f"got: {self.category!r}"
            )
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"UrlEntry.language must be one of {sorted(VALID_LANGUAGES)}, "
                f"got: {self.language!r}"
            )
        if self.rendering_type not in VALID_RENDERING_TYPES:
            raise ValueError(
                f"UrlEntry.rendering_type must be one of "
                f"{sorted(VALID_RENDERING_TYPES)}, got: {self.rendering_type!r}"
            )

    def as_dict(self) -> dict[str, str]:
        """Return a plain ``dict`` representation of this entry."""
        return dataclasses.asdict(self)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Curated URL matrix — 10 entries
# ---------------------------------------------------------------------------

URL_MATRIX: tuple[UrlEntry, ...] = (
    # ------------------------------------------------------------------
    # English / static — plain HTTP GET is sufficient
    # ------------------------------------------------------------------
    UrlEntry(
        url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        category="wiki",
        language="en",
        rendering_type="static",
    ),
    UrlEntry(
        url="https://www.bbc.com/news/technology",
        category="news",
        language="en",
        rendering_type="static",
    ),
    # ------------------------------------------------------------------
    # English / headers_rotation — server-rendered HTML behind anti-bot
    # ------------------------------------------------------------------
    UrlEntry(
        url="https://techcrunch.com/",
        category="tech",
        language="en",
        rendering_type="headers_rotation",
    ),
    UrlEntry(
        url="https://www.reuters.com/technology/",
        category="news",
        language="en",
        rendering_type="headers_rotation",
    ),
    UrlEntry(
        url="https://www.nytimes.com/",
        category="news",
        language="en",
        rendering_type="headers_rotation",
    ),
    # ------------------------------------------------------------------
    # English / browser — JS-rendered SPA requires Playwright
    # ------------------------------------------------------------------
    UrlEntry(
        url="https://www.bloomberg.com/",
        category="finance",
        language="en",
        rendering_type="browser",
    ),
    # ------------------------------------------------------------------
    # Korean / static — plain HTTP GET is sufficient
    # ------------------------------------------------------------------
    UrlEntry(
        url="https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5",
        category="wiki",
        language="ko",
        rendering_type="static",
    ),
    UrlEntry(
        url="https://news.naver.com/",
        category="news",
        language="ko",
        rendering_type="static",
    ),
    # ------------------------------------------------------------------
    # Korean / headers_rotation — SSR with anti-scraping measures
    # ------------------------------------------------------------------
    UrlEntry(
        url="https://www.chosun.com/",
        category="news",
        language="ko",
        rendering_type="headers_rotation",
    ),
    # ------------------------------------------------------------------
    # Korean / browser — JS-heavy rendering requires Playwright
    # ------------------------------------------------------------------
    UrlEntry(
        url="https://www.hani.co.kr/",
        category="news",
        language="ko",
        rendering_type="browser",
    ),
)
