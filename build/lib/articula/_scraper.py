"""
Core ``Scraper`` class for the articula library.

Orchestrates the tiered fetch pipeline (static → headers_rotation → browser)
and delegates content extraction to ``_extractor.extract``.

Thread/async safety
-------------------
A single ``Scraper`` instance may be shared across concurrent ``asyncio``
tasks.  The ``_init_lock`` guards one-time initialisation; the optional
``_semaphore`` limits batch concurrency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, field
from typing import Any, NoReturn, TypeVar

# fetch_browser is re-exported here (and patched by tests via
# articula._scraper.fetch_browser) even though the escalation path uses the
# persistent-browser variant fetch_with_existing_browser.
from articula._browser import fetch_browser, fetch_with_existing_browser  # noqa: F401
from articula._extractor import ExtractionResult, extract
from articula._fetcher import FetchResult, fetch_rotation, fetch_static
from articula.exceptions import (
    BrowserNotInstalledError,
    ExtractionError,
    FetchError,
    RobotsDisallowedError,
)
from articula.models import Article

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Concurrency helper
# ---------------------------------------------------------------------------


async def _bounded_gather(
    coros: Iterable[Coroutine[Any, Any, _T]],
    limit: int,
) -> list[_T]:
    """Execute coroutines with bounded concurrency, returning results in input order.

    A semaphore with *limit* slots ensures that at most *limit* coroutines run
    at the same time.  Results are collected in the same order as the input
    iterable regardless of completion order.

    Parameters
    ----------
    coros:
        Iterable of coroutines to execute.  Each element must be an awaitable
        coroutine object (not a coroutine function).
    limit:
        Maximum number of coroutines allowed to run simultaneously.
        Must be a positive integer.

    Returns
    -------
    list[_T]
        One result per input coroutine, in the same order as *coros*.

    Raises
    ------
    ValueError
        When *limit* is less than 1.
    """
    coros_list = list(coros)  # materialise early so we can close on error
    if limit < 1:
        for c in coros_list:
            c.close()
        raise ValueError(f"limit must be >= 1, got {limit!r}")
    results: list[Any] = [None] * len(coros_list)
    semaphore = asyncio.Semaphore(limit)

    async def _run_one(idx: int, coro: Coroutine[Any, Any, _T]) -> None:
        async with semaphore:
            results[idx] = await coro

    await asyncio.gather(*(_run_one(i, c) for i, c in enumerate(coros_list)))
    return results


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: float = 30.0
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_CONCURRENCY: int = 4

_ALL_TIERS: tuple[str, ...] = ("static", "headers_rotation", "browser")


@dataclass(frozen=True)
class ScraperConfig:
    """Immutable configuration snapshot for a Scraper instance."""

    timeout: float = _DEFAULT_TIMEOUT
    max_retries: int = _DEFAULT_MAX_RETRIES
    custom_headers: dict[str, str] = field(default_factory=dict)
    custom_user_agent: str | None = None
    proxy_url: str | None = None
    force_strategy: str | None = None  # pin to a single tier; None = escalate
    respect_robots: bool = False
    concurrency_limit: int = _DEFAULT_CONCURRENCY


# ---------------------------------------------------------------------------
# Assembly helper
# ---------------------------------------------------------------------------


def _build_article(
    fetch_result: FetchResult,
    extraction: ExtractionResult,
    strategies_attempted: list[str],
) -> Article:
    """Assemble a frozen ``Article`` from fetch and extraction results."""
    return Article(
        title=extraction.title,
        text=extraction.text,
        resolved_url=fetch_result.resolved_url,
        strategy_tier=fetch_result.strategy_tier,  # type: ignore[arg-type]
        extraction_method=extraction.method,        # type: ignore[arg-type]
        extraction_confidence=extraction.confidence,
        detected_language=extraction.language,
        strategies_attempted=tuple(strategies_attempted),
        author=extraction.author,
        published_date=extraction.published_date,
    )


def _raise_extraction_error(
    url: str,
    strategies_attempted: list[str],
    message: str = "",
    cause: BaseException | None = None,
) -> NoReturn:
    """Raise ``ExtractionError`` after all strategies are exhausted.

    Centralises ``ExtractionError`` construction so that ``url`` and
    ``strategies_attempted`` are always populated from the extraction context.
    This factory helper ensures consistent error attributes regardless of which
    code path triggers extraction failure.

    Parameters
    ----------
    url:
        The URL whose content could not be extracted.
    strategies_attempted:
        Ordered list of strategy tier names that were attempted before giving up
        (e.g. ``["static", "headers_rotation", "browser"]``).
    message:
        Human-readable explanation of why extraction failed.  When omitted, a
        default message referencing *url* is used.
    cause:
        The originating exception, if any.  Chains via ``raise ... from cause``
        to preserve the full exception traceback.

    Raises
    ------
    ExtractionError
        Always raised — this function never returns normally.
    """
    error_message = message or (
        f"Content extraction failed for {url!r} after exhausting all strategies"
    )
    exc = ExtractionError(
        error_message,
        url=url,
        strategies_attempted=strategies_attempted,
    )
    if cause is not None:
        raise exc from cause
    raise exc


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class Scraper:
    """
    Async context manager that scrapes a single URL through a tiered pipeline.

    Usage::

        async with Scraper(timeout=60, respect_robots=True) as s:
            article = await s.scrape("https://example.com/article")

    Parameters mirror ``ScraperConfig``; unknown keyword arguments raise
    ``TypeError`` at construction time.
    """

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        custom_headers: dict[str, str] | None = None,
        custom_user_agent: str | None = None,
        proxy_url: str | None = None,
        force_strategy: str | None = None,
        respect_robots: bool = False,
        concurrency_limit: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        headers: dict[str, str] = dict(custom_headers or {})
        if custom_user_agent:
            headers["User-Agent"] = custom_user_agent

        self._config = ScraperConfig(
            timeout=timeout,
            max_retries=max_retries,
            custom_headers=headers,
            custom_user_agent=custom_user_agent,
            proxy_url=proxy_url,
            force_strategy=force_strategy,
            respect_robots=respect_robots,
            concurrency_limit=concurrency_limit,
        )
        self._init_lock: asyncio.Lock = asyncio.Lock()
        # Lazy-initialised browser state — None until the browser tier is first
        # needed.  Both fields are set atomically inside _ensure_browser() under
        # _init_lock so concurrent coroutines never launch duplicate browsers.
        self._browser: Any = None
        self._pw_ctx: Any = None

    # ------------------------------------------------------------------
    # Sync context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Scraper:
        """Enter the sync context manager.

        Returns
        -------
        Scraper
            ``self`` — the same Scraper instance, for use with the
            ``with Scraper(...) as s:`` pattern.
        """
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit the sync context manager and release shared resources.

        Delegates to :meth:`_cleanup` so that subclasses and tests can
        override or verify the teardown behaviour independently of the
        ``__exit__`` protocol wiring.
        """
        self._cleanup()

    def _cleanup(self) -> None:
        """Release any shared resources held by this Scraper instance.

        Called by :meth:`__exit__` (sync context manager) and may also be
        called directly.  The default implementation is a no-op because
        browser lifecycle is managed per-call inside ``fetch_browser``
        and ``_init_lock`` requires no teardown.  Subclasses may override
        this method to release persistent connections or semaphores.
        """
        logger.debug("Scraper._cleanup() invoked")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Scraper:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._close_browser()
        self._cleanup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape(self, url: str) -> Article:
        """Scrape *url*, escalating through fetch tiers as needed.

        Parameters
        ----------
        url:
            Target URL.  Must start with ``http://`` or ``https://``.

        Returns
        -------
        Article
            Fully populated ``Article`` instance.  ``strategy_tier`` reflects
            which tier ultimately succeeded.

        Raises
        ------
        RobotsDisallowedError
            When ``respect_robots=True`` and robots.txt forbids the URL.
        FetchError
            When every strategy tier fails to retrieve the page.
        ExtractionError
            When the page was fetched but no usable article body was found.
        """
        cfg = self._config

        if cfg.respect_robots:
            await self._check_robots(url)

        deadline = time.monotonic() + cfg.timeout
        tiers: tuple[str, ...] = (
            (cfg.force_strategy,) if cfg.force_strategy else _ALL_TIERS
        )

        strategies_attempted: list[str] = []
        any_fetch_succeeded = False
        last_extraction_error: ValueError | None = None

        for tier in tiers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.debug("Deadline exceeded before attempting tier %r", tier)
                break

            strategies_attempted.append(tier)
            fetch_result = await self._fetch_tier(tier, url, remaining)

            if fetch_result is None:
                logger.debug("Tier %r failed to fetch %s — escalating", tier, url)
                continue

            any_fetch_succeeded = True

            # A tier that returns HTML has not "won" until that HTML yields a
            # meaningful article body.  extract() raises ValueError when no
            # extractor produces a usable body (e.g. a JavaScript shell page or
            # a bot-challenge interstitial).  In that case we must keep
            # escalating — most importantly to the browser tier, which renders
            # JS — rather than giving up on the static HTML.
            try:
                extraction = extract(fetch_result.html, fetch_result.resolved_url)
            except ValueError as exc:
                last_extraction_error = exc
                logger.debug(
                    "Tier %r fetched %s but extraction found no body — escalating",
                    tier,
                    url,
                )
                continue

            logger.debug("Tier %r produced a usable article for %s", tier, url)
            return _build_article(fetch_result, extraction, strategies_attempted)

        # No tier produced a usable article.  Distinguish "could not retrieve"
        # (FetchError) from "retrieved but no extractable body" (ExtractionError).
        if not any_fetch_succeeded:
            raise FetchError(
                f"All fetch strategies exhausted for {url!r}",
                url=url,
                attempted_strategies=strategies_attempted,
            )

        _raise_extraction_error(
            url=url,
            strategies_attempted=strategies_attempted,
            message=str(last_extraction_error) if last_extraction_error else "",
            cause=last_extraction_error,
        )

    # ------------------------------------------------------------------
    # Lazy browser lifecycle
    # ------------------------------------------------------------------

    async def _ensure_browser(self) -> Any:
        """Return the shared Playwright browser, launching it lazily on first call.

        Implements double-checked locking on ``_init_lock`` so that concurrent
        coroutines never race to launch more than one browser process.

        Returns
        -------
        Any
            The Playwright ``Browser`` instance (same object on every call).

        Raises
        ------
        BrowserNotInstalledError
            When Playwright is not installed.
        """
        if self._browser is not None:
            return self._browser

        async with self._init_lock:
            # Second check inside the lock (double-checked locking pattern).
            if self._browser is not None:
                return self._browser

            self._browser = await self._launch_browser()
            logger.debug("Playwright browser launched (lazy init on first scrape)")

        return self._browser

    async def _launch_browser(self) -> Any:
        """Launch a new Playwright Chromium browser and store the context.

        Separated from :meth:`_ensure_browser` to allow unit tests to patch
        only the launch step without mocking the entire Playwright API.

        Returns
        -------
        Any
            A Playwright ``Browser`` instance.

        Raises
        ------
        BrowserNotInstalledError
            When Playwright is not importable.
        """
        try:
            from playwright.async_api import async_playwright  # noqa: PLC0415
        except ImportError as exc:
            raise BrowserNotInstalledError(
                "Playwright is not installed. "
                "Install the browser extra: "
                "pip install 'articula[browser]'  "
                "then run: playwright install chromium"
            ) from exc

        ctx = async_playwright()
        pw = await ctx.__aenter__()
        # Store the context manager so __aexit__ can stop playwright cleanly.
        self._pw_ctx = ctx
        return await pw.chromium.launch(headless=True)

    async def _close_browser(self) -> None:
        """Close the shared browser and stop playwright if they were started.

        Safe to call when neither has been initialised (fast no-op).
        Always sets ``_browser`` and ``_pw_ctx`` back to ``None`` even if
        the close raises, so the Scraper is left in a clean state.
        """
        browser = self._browser
        pw_ctx = self._pw_ctx

        # Clear state first so a second call is always a no-op.
        self._browser = None
        self._pw_ctx = None

        if browser is not None:
            try:
                await browser.close()
                logger.debug("Playwright browser closed")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error closing browser: %s", exc)

        if pw_ctx is not None:
            try:
                await pw_ctx.__aexit__(None, None, None)
                logger.debug("Playwright stopped")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error stopping playwright: %s", exc)

    # ------------------------------------------------------------------
    # Tier dispatch
    # ------------------------------------------------------------------

    async def _fetch_tier(
        self, tier: str, url: str, timeout: float
    ) -> FetchResult | None:
        """Dispatch to the correct tier fetch function."""
        cfg = self._config
        http_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": cfg.max_retries,
            "custom_headers": cfg.custom_headers,
            "proxy_url": cfg.proxy_url,
        }

        if tier == "static":
            return await fetch_static(url, **http_kwargs)
        if tier == "headers_rotation":
            return await fetch_rotation(url, **http_kwargs)
        if tier == "browser":
            browser = await self._ensure_browser()
            return await fetch_with_existing_browser(
                url,
                browser=browser,
                timeout=timeout,
                proxy_url=cfg.proxy_url,
            )

        logger.warning("Unknown strategy tier %r — skipping", tier)
        return None

    async def _check_robots(self, url: str) -> None:
        """Raise ``RobotsDisallowedError`` if robots.txt disallows *url*."""
        from urllib.parse import urlparse  # noqa: PLC0415
        from urllib.robotparser import RobotFileParser  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(robots_url, follow_redirects=True)
            if resp.status_code == 200:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.parse(resp.text.splitlines())
                if not rp.can_fetch("*", url):
                    raise RobotsDisallowedError(
                        f"robots.txt disallows {url!r}",
                        url=url,
                    )
        except RobotsDisallowedError:
            raise
        except Exception as exc:
            logger.debug("robots.txt check failed (ignored): %s", exc)
