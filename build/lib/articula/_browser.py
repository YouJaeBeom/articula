"""
Browser-tier fetch using Playwright (optional dependency).

Playwright is imported *lazily* inside ``fetch_browser`` so that this module
can always be imported even when the ``[browser]`` extra is not installed.
Only calling ``fetch_browser`` with an unavailable Playwright raises
``BrowserNotInstalledError``.

Public helpers
--------------
``is_browser_available()`` — lightweight runtime guard that returns ``True``
when Playwright is importable and ``False`` otherwise.  Use it to branch
behaviour before attempting browser-tier scraping.

``_require_playwright()`` — module-level import guard that raises
``ImportError`` with a clear install hint when Playwright is absent.
"""

from __future__ import annotations

import logging
from typing import Any

from articula._fetcher import FetchResult
from articula.exceptions import BrowserNotInstalledError, FetchError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error-page fingerprint detection
# ---------------------------------------------------------------------------

_BROWSER_ERROR_PAGE_SIGNALS: tuple[str, ...] = (
    "404 not found",
    "403 forbidden",
    "500 internal server error",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "access denied",
    "page not found",
    "the page you requested was not found",
    "this site can't be reached",
    "err_name_not_resolved",
    "err_connection_refused",
)


def _is_browser_error_page(html: str) -> bool:
    """Return ``True`` if the rendered HTML looks like a browser/server error page.

    Checks the page content for well-known error-page fingerprints that indicate
    the navigation completed (no exception, valid HTTP session) but the target
    resource is inaccessible.
    """
    lower = html.lower()
    return any(signal in lower for signal in _BROWSER_ERROR_PAGE_SIGNALS)

# ---------------------------------------------------------------------------
# Module-level install hint constant
# ---------------------------------------------------------------------------

_PLAYWRIGHT_INSTALL_HINT: str = (
    "Playwright is not installed. "
    "Install the browser extra:  pip install 'articula[browser]'  "
    "then run:  playwright install chromium"
)


# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------


def _require_playwright() -> None:
    """Module-level import guard: raise ``ImportError`` with install hint if Playwright absent.

    This guard is defined at module scope so callers and tests can mock
    ``sys.modules["playwright"]`` and assert the error message without
    importing the full Playwright browser pipeline.

    Raises
    ------
    ImportError
        When ``playwright`` is not importable.  The exception message
        contains an actionable install hint:
        ``pip install 'articula[browser]'``.

    Examples
    --------
    >>> from articula._browser import _require_playwright
    >>> _require_playwright()  # no-op when Playwright is installed
    """
    try:
        import playwright  # noqa: F401, PLC0415
    except ImportError:
        raise ImportError(_PLAYWRIGHT_INSTALL_HINT) from None


def is_browser_available() -> bool:
    """Return ``True`` if Playwright is installed and importable, ``False`` otherwise.

    Performs a lightweight import probe on every call so that the result
    reflects the *current* environment (e.g. after a dynamic install).  The
    check is intentionally inexpensive — it only resolves the top-level
    ``playwright`` package without importing any sub-modules.

    Returns
    -------
    bool
        ``True``  — Playwright is present; browser-tier scraping is available.
        ``False`` — Playwright is absent; install with
                    ``pip install 'articula[browser]'``.

    Examples
    --------
    >>> from articula._browser import is_browser_available
    >>> if not is_browser_available():
    ...     print("Install the [browser] extra to enable JS rendering.")
    """
    try:
        import playwright  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


async def fetch_browser(
    url: str,
    *,
    timeout: float = 30.0,
    proxy_url: str | None = None,
) -> FetchResult | None:
    """Tier 3 fetch: headless Chromium via Playwright for JS-rendered pages.

    Raises
    ------
    BrowserNotInstalledError
        When Playwright is not installed.

    Returns
    -------
    FetchResult | None
        ``None`` on navigation failure; ``FetchResult`` with
        ``strategy_tier="browser"`` on success.
    """
    # Delegate to the module-level import guard so ImportError carries the hint.
    try:
        _require_playwright()
    except ImportError as exc:
        raise BrowserNotInstalledError(str(exc)) from exc

    from playwright.async_api import async_playwright  # noqa: PLC0415

    proxy_settings: dict[str, str] | None = (
        {"server": proxy_url} if proxy_url else None
    )

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    proxy=proxy_settings,  # type: ignore[arg-type]
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()
                resp = await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=timeout * 1000,
                )
                if resp is None:
                    logger.debug("Browser fetch: no response for %s", url)
                    return None

                html = await page.content()
                resolved_url = page.url
                status_code = resp.status

                if status_code >= 400:
                    raise FetchError(
                        f"Browser fetch got HTTP {status_code} for {url!r}",
                        url=url,
                        attempted_strategies=["browser"],
                        status_code=status_code,
                    )

                if not html or not html.strip():
                    raise FetchError(
                        f"Browser fetch got empty body for {url!r}",
                        url=url,
                        attempted_strategies=["browser"],
                        status_code=status_code,
                    )

                if _is_browser_error_page(html):
                    raise FetchError(
                        f"Browser fetch detected error-page fingerprint for {url!r}",
                        url=url,
                        attempted_strategies=["browser"],
                        status_code=status_code,
                    )

                return FetchResult(
                    html=html,
                    resolved_url=resolved_url,
                    strategy_tier="browser",
                    status_code=status_code,
                )
            finally:
                await browser.close()
    except (BrowserNotInstalledError, FetchError):
        raise
    except Exception as exc:
        logger.debug("Browser fetch failed for %s: %s", url, exc)
        raise FetchError(
            f"Browser fetch failed for {url!r}: {exc}",
            url=url,
            attempted_strategies=["browser"],
        ) from exc


async def fetch_with_existing_browser(
    url: str,
    *,
    browser: Any,
    timeout: float = 30.0,
    proxy_url: str | None = None,
) -> FetchResult | None:
    """Fetch *url* using an already-open Playwright browser instance.

    Unlike :func:`fetch_browser`, this function does **not** create or close
    the browser — lifecycle is managed by the caller (typically
    :class:`~articula._scraper.Scraper`).

    Parameters
    ----------
    url:
        Target URL to navigate to.
    browser:
        An already-launched Playwright ``Browser`` instance.
    timeout:
        Per-page navigation timeout in seconds.
    proxy_url:
        Optional proxy URL (e.g. ``"http://proxy.example.com:8080"``).

    Returns
    -------
    FetchResult | None
        ``None`` on navigation failure; ``FetchResult`` with
        ``strategy_tier="browser"`` on success.
    """
    proxy_settings: dict[str, str] | None = (
        {"server": proxy_url} if proxy_url else None
    )

    try:
        context = await browser.new_context(
            proxy=proxy_settings,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        resp = await page.goto(
            url,
            wait_until="networkidle",
            timeout=timeout * 1000,
        )
        if resp is None:
            logger.debug("Browser fetch (existing): no response for %s", url)
            return None

        html = await page.content()
        resolved_url = page.url
        status_code = resp.status
        return FetchResult(
            html=html,
            resolved_url=resolved_url,
            strategy_tier="browser",
            status_code=status_code,
        )
    except Exception as exc:
        logger.debug("Browser fetch (existing) failed for %s: %s", url, exc)
        return None
