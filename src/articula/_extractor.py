"""
Article content extraction for the articula library.

Tries extractors in priority order:
  1. trafilatura — best at boilerplate removal and metadata
  2. readability-lxml — Mozilla Readability port, good for long-form content
  3. heuristic — last resort; regex-based stripping of obvious non-content tags

All three return an ``ExtractionResult`` or ``None``.  The top-level
``extract()`` function raises ``ExtractionError`` (via ``ValueError``) if
all methods produce empty results.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from articula._metadata import PageMetadata, extract_metadata

logger = logging.getLogger(__name__)

# Minimum body characters to consider extraction successful.
_MIN_BODY_CHARS = 80

# Minimum word-tokens to consider extraction meaningful (guards against
# long strings that are all punctuation / code / whitespace noise).
_MIN_BODY_TOKENS = 10


# ---------------------------------------------------------------------------
# Body quality guard
# ---------------------------------------------------------------------------


def is_meaningful_body(text: str) -> bool:
    """Return ``True`` when *text* qualifies as a meaningful article body.

    A body is considered meaningful when it passes **all** of the following
    checks after stripping leading/trailing whitespace:

    1. The stripped text is non-empty.
    2. The character count meets ``_MIN_BODY_CHARS`` (default 80).
    3. The whitespace-delimited token count meets ``_MIN_BODY_TOKENS``
       (default 10) — this catches strings that are long but consist of a
       single word or a sequence of non-word noise characters.

    Parameters
    ----------
    text:
        Raw extracted body text (plain text, not HTML).

    Returns
    -------
    bool
        ``True`` if *text* is substantive enough to be published as the body
        of an article; ``False`` otherwise.

    Examples
    --------
    >>> is_meaningful_body("")
    False
    >>> is_meaningful_body("   ")
    False
    >>> is_meaningful_body("Too short")
    False
    >>> is_meaningful_body("This article discusses how large language models "
    ...                    "are changing the software industry today.")
    True
    """
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) < _MIN_BODY_CHARS:
        return False
    tokens = stripped.split()
    return len(tokens) >= _MIN_BODY_TOKENS


# Login / paywall / bot-challenge fingerprints. A SHORT extracted body that
# contains any of these is almost certainly a wall page rather than an article,
# so we reject it (better a clear error than silently returning the wall text).
_WALL_SIGNALS: tuple[str, ...] = (
    "enable javascript and cookies",
    "please enable javascript",
    "performing security verification",
    "checking your browser before accessing",
    "500 apologies, but something went wrong",
    "sign in to continue",
    "create an account to read",
    "you've reached your free article limit",
    "you have reached your free article limit",
    "are you a robot",
    "access to this page has been denied",
    "verify you are human",
)
_WALL_MAX_CHARS = 600


def _looks_like_wall(text: str) -> bool:
    """Return True if *text* is a short login/paywall/bot-challenge page.

    Only short bodies are inspected — a genuine long article that merely
    mentions one of these phrases is never mistaken for a wall.
    """
    stripped = text.strip()
    if len(stripped) >= _WALL_MAX_CHARS:
        return False
    low = stripped.lower()
    return any(sig in low for sig in _WALL_SIGNALS)


# Loose ISO-8601 normalisation pattern (YYYY-MM-DD).
_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Tags whose entire content is removed during heuristic extraction.
_NOISE_TAGS = re.compile(
    r"<(script|style|nav|footer|header|aside|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

#: Base confidence per extraction method (before field-quality bonuses).
_METHOD_BASE_CONFIDENCE: dict[str, float] = {
    "trafilatura": 0.60,
    "jsonld":      0.62,  # publisher-authored structured data — clean & reliable
    "readability": 0.45,
    "heuristic":   0.25,
}


def compute_extraction_confidence(
    *,
    title: str,
    text: str,
    author: str | None,
    published_date: str | None,
    method: str,
) -> float:
    """Compute extraction confidence in ``[0.0, 1.0]`` from field completeness.

    Each criterion contributes a fixed bonus on top of the method base score:

    =====================  =======
    Criterion              Bonus
    =====================  =======
    text ≥ 200 chars       +0.10
    text ≥ 500 chars       +0.10
    title ≥ 10 chars       +0.05
    title ≥ 30 chars       +0.05
    author is not None     +0.05
    published_date ≠ None  +0.05
    =====================  =======

    Parameters
    ----------
    title:
        Extracted article title (non-empty).
    text:
        Extracted article body text (non-empty).
    author:
        Extracted author byline; ``None`` if not found.
    published_date:
        ISO 8601 date string; ``None`` if not found.
    method:
        Extraction method key (``"trafilatura"``, ``"readability"``,
        or ``"heuristic"``).

    Returns
    -------
    float
        Confidence in ``[0.0, 1.0]``.
    """
    score: float = _METHOD_BASE_CONFIDENCE.get(method, 0.25)

    text_len = len(text.strip())
    if text_len >= 200:
        score += 0.10
    if text_len >= 500:
        score += 0.10

    title_len = len(title.strip())
    if title_len >= 10:
        score += 0.05
    if title_len >= 30:
        score += 0.05

    if author is not None:
        score += 0.05
    if published_date is not None:
        score += 0.05

    return min(1.0, round(score, 4))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Immutable result from a single content-extraction pass."""

    title: str
    text: str
    author: str | None
    published_date: str | None  # ISO 8601 or None
    method: str                 # "trafilatura" | "readability" | "heuristic"
    confidence: float           # 0.0 – 1.0
    language: str               # BCP-47 code, e.g. "en", "ko"


# ---------------------------------------------------------------------------
# Public language detection
# ---------------------------------------------------------------------------

#: Fixed seed fed to langdetect so every call with the same input returns the
#: same BCP-47 tag.  We set this once at function-call time rather than at
#: import time so the library does not silently change the seed of the
#: caller's langdetect environment before they opt in.
_LANGDETECT_SEED: int = 0


def detect_language(text: str) -> str:
    """Detect the language of *text* and return a BCP-47 language code.

    The function uses ``langdetect`` with a fixed random seed so that
    identical inputs always produce identical outputs — making it fully
    testable in isolation with fixed text samples.

    Parameters
    ----------
    text:
        The extracted article body text (or any meaningful string) whose
        language should be detected.  At least ~50–100 characters of
        natural-language content gives reliable results; very short snippets
        may fall back to ``"en"``.

    Returns
    -------
    str
        A BCP-47 language tag such as ``"en"``, ``"ko"``, ``"ja"``, etc.
        Falls back to ``"en"`` when detection fails (empty input, numeric-
        only input, or any ``langdetect`` exception).

    Examples
    --------
    >>> detect_language("Artificial intelligence is transforming industries.")
    'en'
    >>> detect_language("인공지능이 산업을 변화시키고 있습니다.")
    'ko'
    """
    if not text or not text.strip():
        return "en"

    try:
        from langdetect import DetectorFactory, detect  # noqa: PLC0415

        DetectorFactory.seed = _LANGDETECT_SEED
        lang: str = detect(text)
        return lang if lang else "en"
    except Exception:  # LangDetectException or any import/runtime error
        return "en"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Internal wrapper — delegates to the public :func:`detect_language`."""
    return detect_language(text)


def normalize_date(raw: str | None) -> str | None:
    """Parse *raw* into an ISO 8601 date string (``YYYY-MM-DD``).

    Accepts a wide variety of common date formats including:

    - **ISO 8601** – ``'2024-01-05'``, ``'2024-01-05T10:30:00Z'``
    - **Long-form** – ``'January 5, 2024'``, ``'Jan 5, 2024'``
    - **Slash-separated** – ``'01/05/2024'`` (MM/DD/YYYY US convention)
    - **RFC 2822** – ``'Fri, 05 Jan 2024 00:00:00 +0000'``
    - **Korean** – ``'2024년 1월 5일'`` (via *dateparser* when available)

    The function applies a **fast-path regex** first: if the input already
    contains a ``YYYY-MM-DD`` fragment it is returned immediately without
    invoking any external parser.

    Fall-through order when the fast path misses:

    1. ``dateutil.parser.parse`` – handles most Western formats natively and
       is a declared base dependency.
    2. ``dateparser.parse`` – optional; handles additional locale-specific
       formats including Korean when installed.

    Parameters
    ----------
    raw:
        Raw date string from a web page or ``None``.

    Returns
    -------
    str | None
        ISO 8601 date string ``'YYYY-MM-DD'``, or ``None`` when *raw* is
        ``None``, empty, or not parseable by any available parser.

    Examples
    --------
    >>> normalize_date('January 5, 2024')
    '2024-01-05'
    >>> normalize_date('2024-01-05')
    '2024-01-05'
    >>> normalize_date('Fri, 05 Jan 2024 00:00:00 +0000')
    '2024-01-05'
    >>> normalize_date(None)
    >>> normalize_date('not a date')
    """
    if not raw or not raw.strip():
        return None

    # Fast path: input already contains a YYYY-MM-DD fragment.
    m = _DATE_PATTERN.search(raw)
    if m:
        return m.group(1)

    # Try python-dateutil (declared base dependency).
    # dayfirst=False → US MM/DD/YYYY convention for ambiguous inputs.
    try:
        from dateutil import parser as _du_parser  # noqa: PLC0415

        dt = _du_parser.parse(raw, dayfirst=False)
        return str(dt.strftime("%Y-%m-%d"))
    except Exception:
        pass

    # Try dateparser as an optional fallback (handles locale-specific formats).
    try:
        import dateparser  # noqa: PLC0415

        dt2 = dateparser.parse(raw)
        if dt2 is not None:
            return str(dt2.strftime("%Y-%m-%d"))
    except Exception:
        pass

    return None


def _normalize_date(raw: str | None) -> str | None:
    """Internal wrapper — delegates to the public :func:`normalize_date`."""
    return normalize_date(raw)


def normalize_date_with_timezone(raw: str | None) -> str | None:
    """Parse *raw* into a **timezone-aware** ISO 8601 datetime string.

    Unlike :func:`normalize_date` (which strips time and timezone info and
    returns only ``YYYY-MM-DD``), this function preserves timezone information
    in the output.

    Timezone normalisation rules:

    - If *raw* carries an explicit UTC offset (e.g. ``+05:30``, ``-08:00``),
      that offset is preserved verbatim in the returned string.
    - If *raw* references UTC by label (``Z``, ``GMT``, ``UTC``, ``+0000``,
      ``+00:00``), the returned offset is always ``+00:00``.
    - If *raw* contains **no timezone info at all** (i.e. a naive datetime),
      ``None`` is returned — the function deliberately avoids guessing a
      timezone for naive inputs.

    The implementation applies ``dateutil.parser.parse`` as the primary engine
    (handles ISO 8601, RFC 2822, and most Western date/time formats), with
    ``dateparser`` as an optional fallback for locale-specific formats.

    Parameters
    ----------
    raw:
        Raw date string from a web page or HTTP header.  May include timezone
        info in any of the common forms supported by RFC 2822, ISO 8601, or
        the W3C date-time note.

    Returns
    -------
    str | None
        Timezone-aware ISO 8601 string without sub-second precision, e.g.
        ``'2024-01-05T10:30:00+05:30'`` or ``'2024-01-05T10:30:00+00:00'``.
        Returns ``None`` when *raw* is falsy, unparseable, or timezone-naive.

    Examples
    --------
    >>> normalize_date_with_timezone('2024-01-05T10:30:00+05:30')
    '2024-01-05T10:30:00+05:30'
    >>> normalize_date_with_timezone('Mon, 05 Jan 2024 10:30:00 GMT')
    '2024-01-05T10:30:00+00:00'
    >>> normalize_date_with_timezone('2024-01-05T10:30:00Z')
    '2024-01-05T10:30:00+00:00'
    >>> normalize_date_with_timezone('2024-01-05')
    >>> normalize_date_with_timezone(None)
    """
    if not raw or not raw.strip():
        return None

    # Primary parser: python-dateutil handles RFC 2822, ISO 8601, and many
    # common Western formats.  dayfirst=False gives US MM/DD/YYYY precedence
    # for ambiguous slash-separated inputs.
    try:
        from dateutil import parser as _du_parser  # noqa: PLC0415

        dt = _du_parser.parse(raw, dayfirst=False)
        if dt.tzinfo is None:
            # Naive datetime — refuse to guess a timezone.
            return None
        # Replace microseconds for clean output; isoformat() yields +HH:MM.
        return str(dt.replace(microsecond=0).isoformat())
    except Exception:
        pass

    # Optional fallback: dateparser supports additional locale-specific formats.
    try:
        import dateparser  # noqa: PLC0415

        dt2 = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": True})
        if dt2 is not None and dt2.tzinfo is not None:
            return str(dt2.replace(microsecond=0).isoformat())
    except Exception:
        pass

    return None


def _strip_tags(html_fragment: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html_fragment)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Tier 1 — trafilatura
# ---------------------------------------------------------------------------


def _extract_trafilatura(html: str, url: str) -> ExtractionResult | None:
    """Extract using ``trafilatura.bare_extraction``."""
    try:
        import trafilatura  # noqa: PLC0415

        # trafilatura 2.x returns a ``Document`` object from bare_extraction.
        # The previous code called ``.get()`` on it (dict API), which raised —
        # so trafilatura, the BEST extractor, was silently never used and the
        # pipeline fell back to readability. Convert Document -> dict properly.
        doc: Any = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            with_metadata=True,
        )
        if doc is None:
            return None
        result: Any = doc.as_dict() if hasattr(doc, "as_dict") else doc
        if not isinstance(result, dict):
            return None

        text = (result.get("text") or "").strip()
        if not text or len(text) < _MIN_BODY_CHARS:
            return None

        # Title may be missing here — it is merged from page metadata
        # (JSON-LD / og:title / <title>) by extract(). Do not reject on it.
        title = (result.get("title") or "").strip()
        author = result.get("author") or None
        published_date = _normalize_date(result.get("date"))
        return ExtractionResult(
            title=title,
            text=text,
            author=author,
            published_date=published_date,
            method="trafilatura",
            confidence=compute_extraction_confidence(
                title=title,
                text=text,
                author=author,
                published_date=published_date,
                method="trafilatura",
            ),
            language=_detect_language(text),
        )
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tier 2 — readability-lxml
# ---------------------------------------------------------------------------


def _extract_readability(html: str, url: str) -> ExtractionResult | None:
    """Extract using ``readability-lxml``."""
    try:
        from readability import Document  # noqa: PLC0415

        doc = Document(html)
        title = (doc.title() or "").strip()
        summary_html = doc.summary()

        if not title or not summary_html:
            return None

        text = _strip_tags(summary_html)
        if len(text) < _MIN_BODY_CHARS:
            return None

        return ExtractionResult(
            title=title,
            text=text,
            author=None,
            published_date=None,
            method="readability",
            confidence=compute_extraction_confidence(
                title=title,
                text=text,
                author=None,
                published_date=None,
                method="readability",
            ),
            language=_detect_language(text),
        )
    except Exception as exc:
        logger.debug("readability extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tier 3 — heuristic fallback
# ---------------------------------------------------------------------------


def _extract_heuristic(html: str, url: str) -> ExtractionResult | None:
    """Last-resort regex-based extraction."""
    try:
        # Title: prefer <h1>, fall back to <title>
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        title_tag = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        raw_title = (h1 or title_tag)
        if raw_title is None:
            return None
        title = _strip_tags(raw_title.group(1))

        # Body: find <body>, remove noise tags, strip remaining tags
        body_m = re.search(r"<body[^>]*>(.*?)</body>", html, re.IGNORECASE | re.DOTALL)
        if body_m is None:
            return None
        body_html = _NOISE_TAGS.sub("", body_m.group(1))
        text = _strip_tags(body_html)

        if not title or len(text) < _MIN_BODY_CHARS:
            return None

        return ExtractionResult(
            title=title,
            text=text,
            author=None,
            published_date=None,
            method="heuristic",
            confidence=compute_extraction_confidence(
                title=title,
                text=text,
                author=None,
                published_date=None,
                method="heuristic",
            ),
            language=_detect_language(text),
        )
    except Exception as exc:
        logger.debug("heuristic extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract(html: str, url: str) -> ExtractionResult:
    """Extract article data from *html*, escalating through extraction tiers.

    Each extractor's output passes through a body-validation step that calls
    :func:`is_meaningful_body` on the parsed body text.  When that check
    returns ``False`` (body too short or too few tokens), the result is
    discarded and the next extractor tier is tried.

    Parameters
    ----------
    html:
        Raw HTML string from a successful fetch.
    url:
        Resolved URL of the page (used as a hint by some extractors).

    Raises
    ------
    ValueError
        When all extraction tiers produce empty or unusable output, including
        the case where every extractor's parsed body fails the
        :func:`is_meaningful_body` quality check.
    """
    # Mine structured metadata (JSON-LD + meta tags) once. Used both to fill
    # title/author/date the body extractors miss and as a last-resort body.
    meta = extract_metadata(html)
    wall_seen = False

    for extractor in (_extract_trafilatura, _extract_readability, _extract_heuristic):
        result = extractor(html, url)
        if result is None:
            continue

        # Body-validation step: reject bodies that are too thin even when the
        # extractor accepted them (e.g. >= _MIN_BODY_CHARS chars but fewer
        # than _MIN_BODY_TOKENS whitespace-delimited tokens).
        if not is_meaningful_body(result.text):
            logger.debug(
                "Extractor %r: parsed body failed is_meaningful_body check "
                "(len=%d, tokens=%d) — skipping",
                result.method,
                len(result.text),
                len(result.text.split()),
            )
            continue

        # Reject login/paywall/bot-challenge pages so we never return wall text
        # as if it were the article (escalates to the next tier instead).
        if _looks_like_wall(result.text):
            wall_seen = True
            logger.debug(
                "Extractor %r produced a login/bot wall page — skipping", result.method
            )
            continue

        merged = _merge_metadata(result, meta)
        logger.debug(
            "Extracted via %s (confidence=%.2f, lang=%s)",
            merged.method,
            merged.confidence,
            merged.language,
        )
        return merged

    # Body extractors all failed, but some publishers ship the full text in
    # JSON-LD articleBody — use it as a genuine last resort.
    if meta.body and is_meaningful_body(meta.body):
        title = meta.title or ""
        date = _normalize_date(meta.published_date)
        logger.debug("Extracted via jsonld articleBody fallback for %s", url)
        return ExtractionResult(
            title=title,
            text=meta.body,
            author=meta.author,
            published_date=date,
            method="jsonld",
            confidence=compute_extraction_confidence(
                title=title, text=meta.body, author=meta.author,
                published_date=date, method="jsonld",
            ),
            language=_detect_language(meta.body),
        )

    suffix = (
        " — the page appears to be a login, paywall, or bot-protection wall"
        if wall_seen
        else ""
    )
    raise ValueError(
        f"All extraction methods (trafilatura, readability, heuristic, jsonld) "
        f"produced no usable article body for {url!r}{suffix}"
    )


def _merge_metadata(result: ExtractionResult, meta: PageMetadata) -> ExtractionResult:
    """Fill missing title/author/date on *result* from page *meta*.

    Body extractors (especially trafilatura on JS-light news pages) reliably
    recover the article text but often miss the title/byline/date that sit in
    JSON-LD or Open Graph tags. We graft those in and recompute confidence.
    """
    title = result.title or (meta.title or "")
    author = result.author or meta.author
    date = result.published_date or _normalize_date(meta.published_date)

    if (title, author, date) == (result.title, result.author, result.published_date):
        return result

    confidence = compute_extraction_confidence(
        title=title, text=result.text, author=author,
        published_date=date, method=result.method,
    )
    return replace(result, title=title, author=author,
                   published_date=date, confidence=confidence)
