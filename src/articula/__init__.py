"""
articula
================

A pip-installable Python library that extracts clean structured article data
(title, body text, author, published date) from a single URL.

Tiered escalation strategies
-----------------------------
1. **static**           — plain HTTP GET (fastest, lightest)
2. **headers_rotation** — rotates User-Agent + headers to mimic real browsers
3. **browser**          — headless Playwright for JS-rendered pages
                          (requires ``pip install 'articula[browser]'``)

Public API
----------
    from articula import scrape, async_scrape, scrape_many, async_scrape_many, Scraper

    # Sync (script context — raises RuntimeError inside a running event loop)
    article = scrape("https://example.com/article")

    # Async
    article = await async_scrape("https://example.com/article")

    # Batch with bounded concurrency (sync, or async_scrape_many in async code)
    results = scrape_many(["https://...", "https://..."])
    results = await async_scrape_many(["https://...", "https://..."])

    # Context-manager (advanced configuration)
    async with Scraper(timeout=60, respect_robots=True) as s:
        article = await s.scrape("https://example.com/article")

Data model
----------
    from articula.models import Article

Exceptions
----------
    from articula.exceptions import (
        ScraperError,
        FetchError,
        ExtractionError,
        BrowserNotInstalledError,
        RobotsDisallowedError,
    )
"""

from __future__ import annotations

import asyncio
import logging
import threading

from articula._browser import is_browser_available
from articula._extractor import (
    detect_language,
    is_meaningful_body,
    normalize_date,
    normalize_date_with_timezone,
)
from articula._scraper import Scraper, _bounded_gather
from articula.exceptions import (
    BrowserNotInstalledError,
    ConfigurationError,
    ExtractionError,
    FetchError,
    RobotsDisallowedError,
    ScraperError,
)
from articula.models import Article

__all__ = [
    # Data model
    "Article",
    # Scraper
    "Scraper",
    # Public API functions
    "scrape",
    "async_scrape",
    "scrape_many",
    "async_scrape_many",
    # Utility functions
    "detect_language",
    "is_meaningful_body",
    "normalize_date",
    "normalize_date_with_timezone",
    # Runtime guards
    "is_browser_available",
    # Exceptions
    "ScraperError",
    "FetchError",
    "ExtractionError",
    "BrowserNotInstalledError",
    "RobotsDisallowedError",
    "ConfigurationError",
]

__version__ = "0.2.0"

# Library-level logger — NullHandler so libraries stay silent by default;
# the application controls log routing.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Module-level shared Scraper — lazy initialisation, thread-safe
# ---------------------------------------------------------------------------

# Intentionally NOT set to a Scraper() at import time: the first call to
# scrape() (or _get_shared_scraper()) performs the one-time creation.
_shared_scraper: Scraper | None = None
_shared_scraper_lock: threading.Lock = threading.Lock()


def _get_shared_scraper() -> Scraper:
    """Return the module-level shared ``Scraper``, initialising it on first call.

    Uses double-checked locking with a :class:`threading.Lock` so that
    simultaneous calls from multiple threads never create more than one
    ``Scraper`` instance.

    Returns
    -------
    Scraper
        The single shared instance, always the same object after first call.
    """
    global _shared_scraper
    if _shared_scraper is None:
        with _shared_scraper_lock:
            # Second check inside the lock — prevents a race where two threads
            # both see None before either acquires the lock.
            if _shared_scraper is None:
                _shared_scraper = Scraper()
    return _shared_scraper


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def scrape(url: str, **kwargs: object) -> Article:
    """Synchronous single-URL scrape using the module-level shared :class:`Scraper`.

    On the **first call** this function lazily initialises a ``Scraper``
    instance with default configuration and stores it at module level.
    All subsequent calls with no keyword arguments reuse that same instance —
    no further instantiation occurs.

    When *kwargs* are supplied the shared instance is bypassed and a fresh
    per-call ``Scraper(**kwargs)`` is used, giving the caller full control
    over timeout, proxy, strategy, etc.

    Suitable for scripts and interactive sessions.  Raises ``RuntimeError``
    with a clear actionable message when called from inside a running event
    loop (e.g. a Jupyter notebook or an existing ``asyncio`` task).

    Parameters
    ----------
    url:
        Target URL.
    **kwargs:
        Optional configuration forwarded to :class:`Scraper`.  If provided,
        a fresh :class:`Scraper` is created for this call only.

    Returns
    -------
    Article

    Raises
    ------
    RuntimeError
        When called from inside a running event loop.
    FetchError, ExtractionError, ScraperError
        On scrape failure.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run().
        pass
    else:
        raise RuntimeError(
            "scrape() was called from inside a running event loop. "
            "Use `await async_scrape(url)` or `await Scraper(...).scrape(url)` "
            "inside async code."
        )

    if kwargs:
        # Non-default configuration requested — use a fresh per-call Scraper
        # so the shared instance keeps its default config.
        return asyncio.run(async_scrape(url, **kwargs))

    shared = _get_shared_scraper()
    return asyncio.run(shared.scrape(url))


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def async_scrape(url: str, **kwargs: object) -> Article:
    """Async coroutine to scrape a single URL.

    On the **first call with no keyword arguments** this function lazily
    initialises the module-level shared ``Scraper`` instance (the same one
    used by :func:`scrape`) and stores it at module level.  All subsequent
    no-kwargs calls reuse that same instance — no further instantiation occurs.

    When *kwargs* are supplied a fresh per-call ``Scraper(**kwargs)`` is used,
    giving the caller full control over timeout, proxy, strategy, etc., without
    mutating the shared instance.

    Parameters
    ----------
    url:
        Target URL.
    **kwargs:
        Optional configuration forwarded to :class:`Scraper`.  If provided,
        a fresh :class:`Scraper` is created for this call only.

    Returns
    -------
    Article
    """
    if kwargs:
        # Non-default configuration requested — use a fresh per-call Scraper
        # so the shared instance keeps its default config.
        async with Scraper(**kwargs) as s:  # type: ignore[arg-type]
            return await s.scrape(url)

    # No kwargs — reuse the shared Scraper instance (same as scrape()).
    shared = _get_shared_scraper()
    return await shared.scrape(url)


async def async_scrape_many(
    urls: list[str],
    *,
    concurrency_limit: int = 4,
    **kwargs: object,
) -> list[Article | ScraperError]:
    """Scrape multiple URLs concurrently with bounded concurrency (async).

    Results are returned in the same order as *urls*.  Per-URL errors are
    captured as ``ScraperError`` instances in the result list rather than
    being propagated as exceptions.

    Parameters
    ----------
    urls:
        List of target URLs.
    concurrency_limit:
        Maximum number of concurrent scrape tasks.
    **kwargs:
        Forwarded to :class:`Scraper`.

    Returns
    -------
    list[Article | ScraperError]
        One entry per input URL, in the same order.
    """
    async def _scrape_one(url: str) -> Article | ScraperError:
        try:
            async with Scraper(**kwargs) as s:  # type: ignore[arg-type]
                return await s.scrape(url)
        except ScraperError as exc:
            return exc

    return await _bounded_gather(
        (_scrape_one(u) for u in urls),
        limit=concurrency_limit,
    )


def scrape_many(
    urls: list[str],
    *,
    concurrency_limit: int = 4,
    **kwargs: object,
) -> list[Article | ScraperError]:
    """Synchronous convenience wrapper around :func:`async_scrape_many`.

    Manages its own event loop so casual scripts can batch-scrape without
    touching ``asyncio``.  Results are returned in the same order as *urls*,
    with per-URL errors captured as ``ScraperError`` instances rather than
    raised.

    Parameters
    ----------
    urls:
        List of target URLs.
    concurrency_limit:
        Maximum number of concurrent scrape tasks.
    **kwargs:
        Forwarded to :class:`Scraper`.

    Returns
    -------
    list[Article | ScraperError]
        One entry per input URL, in the same order.

    Raises
    ------
    RuntimeError
        When called from inside a running event loop — use
        ``await async_scrape_many(...)`` there instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — safe to use asyncio.run()
    else:
        raise RuntimeError(
            "scrape_many() was called from inside a running event loop. "
            "Use `await async_scrape_many(urls)` inside async code."
        )

    return asyncio.run(
        async_scrape_many(urls, concurrency_limit=concurrency_limit, **kwargs)
    )
