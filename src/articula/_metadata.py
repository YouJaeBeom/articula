"""
Structured-metadata extraction from JSON-LD and HTML meta tags.

Many news and blog platforms (BBC, MSN, WordPress, Ghost, …) embed reliable
article metadata in ``<script type="application/ld+json">`` blocks and Open
Graph / standard ``<meta>`` tags.  This is far more dependable than guessing
the right DOM node, so we mine it first and merge it with whatever the body
extractors (trafilatura/readability) recover.

Public entry point
------------------
``extract_metadata(html)`` → :class:`PageMetadata` with ``title``, ``author``,
``published_date`` (raw, not yet normalised) and ``body`` (only when a source
such as JSON-LD ``articleBody`` actually carries the full text — most do not).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)

# Article-ish JSON-LD @type values whose fields we trust.
_ARTICLE_TYPES = (
    "Article",
    "NewsArticle",
    "ReportageNewsArticle",
    "BlogPosting",
    "Report",
    "ScholarlyArticle",
)


@dataclass(frozen=True)
class PageMetadata:
    """Metadata mined from JSON-LD + meta tags (all fields best-effort)."""

    title: str | None = None
    author: str | None = None
    published_date: str | None = None  # raw string, normalise downstream
    body: str | None = None            # full text when a source carries it


def _normalise_author(value: object) -> str | None:
    """Flatten JSON-LD ``author`` (str | dict | list) into a display string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        name = value.get("name")
        return name.strip() if isinstance(name, str) and name.strip() else None
    if isinstance(value, list):
        names: list[str] = [n for n in (_normalise_author(v) for v in value) if n]
        return ", ".join(names) if names else None
    return None


def _iter_jsonld_objects(html: str) -> Iterator[dict[str, Any]]:
    """Yield every JSON-LD object found in *html*, flattening ``@graph`` lists."""
    for block in _JSONLD_RE.findall(html):
        try:
            data = json.loads(block.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        stack = [data]
        while stack:
            obj = stack.pop()
            if isinstance(obj, list):
                stack.extend(obj)
            elif isinstance(obj, dict):
                if "@graph" in obj and isinstance(obj["@graph"], list):
                    stack.extend(obj["@graph"])
                yield obj


def _type_matches(obj: dict[str, Any]) -> bool:
    raw = obj.get("@type", "")
    values = raw if isinstance(raw, list) else [raw]
    return any(any(t in str(v) for t in _ARTICLE_TYPES) for v in values)


def _from_jsonld(html: str) -> PageMetadata:
    """Extract metadata from the first article-like JSON-LD object."""
    for obj in _iter_jsonld_objects(html):
        if not _type_matches(obj):
            continue
        headline = obj.get("headline") or obj.get("name")
        title = headline.strip() if isinstance(headline, str) and headline.strip() else None
        author = _normalise_author(obj.get("author"))
        date = obj.get("datePublished") or obj.get("dateCreated") or obj.get("dateModified")
        date = date.strip() if isinstance(date, str) and date.strip() else None
        body = obj.get("articleBody")
        body = body.strip() if isinstance(body, str) and len(body.strip()) >= 200 else None
        if title or body:
            return PageMetadata(title=title, author=author, published_date=date, body=body)
    return PageMetadata()


def _meta_content(html: str, *patterns: str) -> str | None:
    """Return the first matching ``<meta>`` content for any attribute pattern."""
    for attr, value in patterns_pairs(patterns):
        rx = re.compile(
            rf'<meta[^>]+{attr}=["\']{re.escape(value)}["\'][^>]*content=["\']([^"\']+)["\']',
            re.IGNORECASE,
        )
        m = rx.search(html)
        if m:
            text = m.group(1).strip()
            if text:
                return text
        # content-before-property ordering
        rx2 = re.compile(
            rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]*{attr}=["\']{re.escape(value)}["\']',
            re.IGNORECASE,
        )
        m2 = rx2.search(html)
        if m2 and m2.group(1).strip():
            return m2.group(1).strip()
    return None


def patterns_pairs(patterns: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    """Yield (attr, value) for ``"attr=value"`` strings, e.g. ``"property=og:title"``."""
    for p in patterns:
        attr, _, value = p.partition("=")
        yield attr, value


def _from_meta(html: str) -> PageMetadata:
    title = _meta_content(
        html, "property=og:title", "name=twitter:title", "itemprop=headline"
    )
    if not title:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
    author = _meta_content(
        html,
        "name=author",
        "property=article:author",
        "property=og:article:author",
        "name=twitter:creator",
        "itemprop=author",
    )
    date = _meta_content(
        html,
        "property=article:published_time",
        "name=publishdate",
        "name=date",
        "itemprop=datePublished",
        "property=og:article:published_time",
    )
    return PageMetadata(title=title or None, author=author or None, published_date=date or None)


def extract_metadata(html: str) -> PageMetadata:
    """Merge JSON-LD (preferred) and meta-tag metadata from *html*.

    JSON-LD wins for every field it provides; meta tags fill the gaps.  Author
    strings that are obviously a site name rather than a person are kept as-is
    (callers may treat them as publisher).
    """
    ld = _from_jsonld(html)
    meta = _from_meta(html)
    return PageMetadata(
        title=ld.title or meta.title,
        author=ld.author or meta.author,
        published_date=ld.published_date or meta.published_date,
        body=ld.body,  # only JSON-LD carries full body; meta never does
    )
