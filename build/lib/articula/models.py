"""
Domain models for the articula library.

All models use frozen dataclasses (immutable) to prevent accidental mutation
after extraction and to support safe concurrency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StrategyTier = Literal["static", "headers_rotation", "browser"]
ExtractionMethod = Literal["trafilatura", "readability", "heuristic"]


# ---------------------------------------------------------------------------
# Article
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Article:
    """
    Structured article data extracted from a single URL.

    Required core fields
    --------------------
    title            : Extracted article title.
    text             : Cleaned body text (no nav / ads / boilerplate).
    resolved_url     : Final URL after redirects.
    strategy_tier    : Which fetch strategy ultimately succeeded.
                       One of ``"static"``, ``"headers_rotation"``, ``"browser"``.
    extraction_method: Content-extraction algorithm used.
                       One of ``"trafilatura"``, ``"readability"``, ``"heuristic"``.
    extraction_confidence : Quality indicator in ``[0.0, 1.0]``.
    detected_language: BCP-47 language code detected in the article body
                       (e.g. ``"en"``, ``"ko"``).
    strategies_attempted : Ordered list of strategy tiers tried before success.

    Optional / best-effort fields
    --------------------------------
    author           : Author or publisher byline. ``None`` when not found.
    published_date   : Publication date normalised to ISO 8601 (``YYYY-MM-DD``
                       or ``YYYY-MM-DDTHH:MM:SS[±HH:MM]``). ``None`` when not
                       found or unparseable.
    """

    # ---- core ----------------------------------------------------------------
    title: str
    text: str
    resolved_url: str
    strategy_tier: StrategyTier
    extraction_method: ExtractionMethod
    extraction_confidence: float
    detected_language: str
    strategies_attempted: tuple[str, ...] = field(default_factory=tuple)

    # ---- best-effort ---------------------------------------------------------
    author: str | None = None
    published_date: str | None = None

    # ------------------------------------------------------------------
    # Post-init validation (runs even on frozen instances because
    # __post_init__ executes before the object is fully sealed).
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate_title()
        self._validate_text()
        self._validate_url()
        self._validate_strategy_tier()
        self._validate_extraction_method()
        self._validate_confidence()
        self._validate_language()
        self._validate_published_date()

    # ------------------------------------------------------------------
    # Validators (private helpers)
    # ------------------------------------------------------------------

    def _validate_title(self) -> None:
        if not isinstance(self.title, str):
            raise TypeError(f"title must be str, got {type(self.title).__name__}")
        if not self.title.strip():
            raise ValueError("title must not be empty or whitespace-only")

    def _validate_text(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not self.text.strip():
            raise ValueError("text must not be empty or whitespace-only")

    def _validate_url(self) -> None:
        if not isinstance(self.resolved_url, str):
            raise TypeError(
                f"resolved_url must be str, got {type(self.resolved_url).__name__}"
            )
        if not self.resolved_url.startswith(("http://", "https://")):
            raise ValueError(
                f"resolved_url must start with http:// or https://, got: {self.resolved_url!r}"
            )

    def _validate_strategy_tier(self) -> None:
        valid: tuple[str, ...] = ("static", "headers_rotation", "browser")
        if self.strategy_tier not in valid:
            raise ValueError(
                f"strategy_tier must be one of {valid!r}, got {self.strategy_tier!r}"
            )

    def _validate_extraction_method(self) -> None:
        valid: tuple[str, ...] = ("trafilatura", "readability", "heuristic")
        if self.extraction_method not in valid:
            raise ValueError(
                f"extraction_method must be one of {valid!r}, got {self.extraction_method!r}"
            )

    def _validate_confidence(self) -> None:
        if not isinstance(self.extraction_confidence, (int, float)):
            raise TypeError(
                f"extraction_confidence must be a number, got {type(self.extraction_confidence).__name__}"
            )
        if not (0.0 <= float(self.extraction_confidence) <= 1.0):
            raise ValueError(
                f"extraction_confidence must be in [0.0, 1.0], got {self.extraction_confidence}"
            )

    def _validate_language(self) -> None:
        if not isinstance(self.detected_language, str):
            raise TypeError(
                f"detected_language must be str, got {type(self.detected_language).__name__}"
            )
        if not self.detected_language.strip():
            raise ValueError("detected_language must not be empty")

    def _validate_published_date(self) -> None:
        """Allow None or a string that looks like an ISO 8601 date/datetime."""
        if self.published_date is None:
            return
        if not isinstance(self.published_date, str):
            raise TypeError(
                f"published_date must be str or None, got {type(self.published_date).__name__}"
            )
        # Accept: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS with optional offset
        _ISO_DATE_RE = re.compile(
            r"^\d{4}-\d{2}-\d{2}"
            r"(?:T\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?"
            r"(?:Z|[+-]\d{2}:\d{2})?)?$"
        )
        if not _ISO_DATE_RE.match(self.published_date):
            raise ValueError(
                f"published_date must be ISO 8601 format (YYYY-MM-DD…), got {self.published_date!r}"
            )

    # ------------------------------------------------------------------
    # Convenience helpers (non-mutating, return new instances)
    # ------------------------------------------------------------------

    def with_author(self, author: str | None) -> Article:
        """Return a new Article with the author field updated."""
        return Article(
            title=self.title,
            text=self.text,
            resolved_url=self.resolved_url,
            strategy_tier=self.strategy_tier,
            extraction_method=self.extraction_method,
            extraction_confidence=self.extraction_confidence,
            detected_language=self.detected_language,
            strategies_attempted=self.strategies_attempted,
            author=author,
            published_date=self.published_date,
        )

    def with_published_date(self, published_date: str | None) -> Article:
        """Return a new Article with the published_date field updated."""
        return Article(
            title=self.title,
            text=self.text,
            resolved_url=self.resolved_url,
            strategy_tier=self.strategy_tier,
            extraction_method=self.extraction_method,
            extraction_confidence=self.extraction_confidence,
            detected_language=self.detected_language,
            strategies_attempted=self.strategies_attempted,
            author=self.author,
            published_date=published_date,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict representation suitable for JSON serialisation."""
        return {
            "title": self.title,
            "text": self.text,
            "resolved_url": self.resolved_url,
            "strategy_tier": self.strategy_tier,
            "extraction_method": self.extraction_method,
            "extraction_confidence": self.extraction_confidence,
            "detected_language": self.detected_language,
            "strategies_attempted": list(self.strategies_attempted),
            "author": self.author,
            "published_date": self.published_date,
        }
