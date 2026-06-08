"""
Exception hierarchy for the articula library.

All public exceptions derive from ``ScraperError`` so callers can catch
the entire family with a single except clause when desired.
"""

from __future__ import annotations


class ScraperError(Exception):
    """Base class for all articula errors."""


class FetchError(ScraperError):
    """
    Raised when all fetch strategies fail to retrieve a retrievable page.

    Attributes
    ----------
    url : str
        The URL that could not be fetched.
    attempted_strategies : tuple[str, ...]
        Ordered sequence of strategy tier names that were attempted before
        giving up (e.g. ``("static", "headers_rotation")``).
    status_code : int | None
        Final HTTP status code, if a response was received.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        attempted_strategies: tuple[str, ...] | list[str] = (),
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.attempted_strategies: tuple[str, ...] = tuple(attempted_strategies)
        self.status_code = status_code

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.url:
            parts.append(f"url={self.url!r}")
        if self.attempted_strategies:
            parts.append(f"attempted_strategies={list(self.attempted_strategies)!r}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        return " | ".join(parts)


class ExtractionError(ScraperError):
    """
    Raised when the page was fetched successfully but content extraction
    produced an empty or unusable article body.

    Attributes
    ----------
    url : str
        The URL whose content could not be extracted.
    strategies_attempted : tuple[str, ...]
        Ordered sequence of strategy tier names that were attempted before
        extraction failed (e.g. ``("static", "headers_rotation")``).
        Stored as an immutable tuple; see also the ``strategies_tried``
        property which exposes the same data as a ``list[str]``.
    strategies_tried : list[str]
        Convenience view of ``strategies_attempted`` as a mutable list.
        Both attributes reflect the same underlying data.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        strategies_attempted: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__(message)
        self.url = url
        self.strategies_attempted: tuple[str, ...] = tuple(strategies_attempted)

    @property
    def strategies_tried(self) -> list[str]:
        """Return the attempted strategies as a ``list[str]``.

        This is a convenience alias for ``strategies_attempted`` that satisfies
        callers expecting a mutable list.  Each access returns a fresh copy so
        the internal tuple remains immutable.
        """
        return list(self.strategies_attempted)

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.url:
            parts.append(f"url={self.url!r}")
        if self.strategies_attempted:
            parts.append(f"strategies_attempted={list(self.strategies_attempted)!r}")
        return " | ".join(parts)


class RobotsDisallowedError(ScraperError):
    """
    Raised when ``respect_robots=True`` and robots.txt disallows the URL.

    Attributes
    ----------
    url : str
        The disallowed URL.
    """

    def __init__(self, message: str, *, url: str = "") -> None:
        super().__init__(message)
        self.url = url


class BrowserNotInstalledError(ScraperError):
    """
    Raised when browser-tier rendering is required but Playwright is not
    installed (i.e. the ``[browser]`` extra was not installed).

    Provides a clear, actionable message guiding the user to install the extra.
    """

    _DEFAULT_MESSAGE = (
        "Browser rendering was requested but Playwright is not installed. "
        "Install the browser extra:  pip install 'articula[browser]'  "
        "then run:  playwright install chromium"
    )

    def __init__(self, message: str = _DEFAULT_MESSAGE) -> None:
        super().__init__(message)


class ConfigurationError(ScraperError):
    """Raised for invalid Scraper configuration (e.g. conflicting options)."""
