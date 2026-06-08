"""
Site adapters — per-host quirk handling for sites that defeat generic scraping.

Some platforms cannot be scraped by the generic static→browser pipeline:

* **Naver blog** wraps the real post in an ``<iframe id="mainFrame">`` whose
  ``PostView.naver`` document holds the content inside ``.se-main-container``;
  the outer URL has no article body at all.
* **MSN** renders articles client-side from an internal JSON API and never
  exposes the body to the DOM in a scrapable form, but its public content API
  (``assets.msn.com/content/view/v2/Detail/…``) returns the full article.

An adapter may do three things (all optional):

* ``rewrite_url(url)``      — transform the URL before fetching.
* ``fetch_override(url)``   — bypass the normal fetch tiers entirely and return
  a ready-to-extract ``FetchResult`` (used by MSN's API).
* ``isolate_content(html)`` — narrow fetched HTML to the real content container
  before extraction (used by Naver).

Adapters are matched by host suffix. The first match wins.
"""

from __future__ import annotations

import html as _htmllib
import json
import logging
import re
from urllib.parse import parse_qs, urlparse

import httpx

from articula._fetcher import FetchResult

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class SiteAdapter:
    """Base adapter. Subclasses override the hooks they need."""

    host_suffixes: tuple[str, ...] = ()

    def matches(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == s or host.endswith("." + s) for s in self.host_suffixes)

    def rewrite_url(self, url: str) -> str:
        return url

    async def fetch_override(
        self, url: str, *, timeout: float, proxy_url: str | None
    ) -> FetchResult | None:
        return None

    def isolate_content(self, html: str, url: str) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Naver blog
# ---------------------------------------------------------------------------


class NaverBlogAdapter(SiteAdapter):
    """blog.naver.com / m.blog.naver.com — follow the iframe, target the post body."""

    host_suffixes = ("blog.naver.com",)

    def rewrite_url(self, url: str) -> str:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        blog_id = qs.get("blogId", [None])[0]
        log_no = qs.get("logNo", [None])[0]

        if not (blog_id and log_no):
            # Path form: /<blogId>/<logNo>
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[-1].isdigit():
                blog_id, log_no = parts[-2], parts[-1]

        if blog_id and log_no:
            return (
                "https://blog.naver.com/PostView.naver"
                f"?blogId={blog_id}&logNo={log_no}"
                "&redirect=Dlog&widgetTypeCall=true&directAccess=false"
            )
        return url

    _DATE_RE = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.")

    def isolate_content(self, html: str, url: str) -> str | None:
        """Return a minimal HTML doc containing only the SmartEditor post body.

        Injects the post's real title/date as JSON-LD so the metadata pipeline
        uses them instead of trafilatura guessing a date from the body text.
        """
        try:
            import lxml.html as LH  # noqa: PLC0415
        except ImportError:
            return None

        try:
            doc = LH.fromstring(html)
        except Exception:
            return None

        container = None
        for xp in (
            '//div[contains(@class,"se-main-container")]',
            '//div[@id="postViewArea"]',
            '//div[contains(@class,"post-view")]',
        ):
            els = doc.xpath(xp)
            if els:
                container = els[0]
                break
        if container is None:
            return None

        body_html = LH.tostring(container, encoding="unicode")

        og = doc.xpath('//meta[@property="og:title"]/@content')
        title = og[0] if og else ""

        # Real publish date lives in a .se_publishDate / .blog2_publishDate span
        # (e.g. "2026. 5. 31. 22:53"). Normalise to ISO for JSON-LD.
        date_iso = ""
        for xp in (
            '//span[contains(@class,"se_publishDate")]',
            '//span[contains(@class,"blog2_publishDate")]',
        ):
            els = doc.xpath(xp)
            if els:
                m = self._DATE_RE.search(els[0].text_content())
                if m:
                    date_iso = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                break

        ld = {"@context": "https://schema.org", "@type": "BlogPosting", "headline": title}
        if date_iso:
            ld["datePublished"] = date_iso
        return (
            f"<html><head><title>{_htmllib.escape(title)}</title>"
            f'<script type="application/ld+json">{json.dumps(ld)}</script>'
            f"</head><body><article>{body_html}</article></body></html>"
        )


# ---------------------------------------------------------------------------
# MSN
# ---------------------------------------------------------------------------


class MSNAdapter(SiteAdapter):
    """msn.com — fetch the real article from the public content API."""

    host_suffixes = ("msn.com",)
    # MSN article ids live in an ``ar-<id>`` path segment (e.g. ar-AA252lKk).
    # Must NOT match the ``en-us`` locale segment, hence the explicit ``ar-``.
    _ID_RE = re.compile(r"/ar-([A-Za-z0-9]+)")
    _MARKET_RE = re.compile(r"msn\.com/([a-z]{2}-[a-z]{2})/")

    async def fetch_override(
        self, url: str, *, timeout: float, proxy_url: str | None
    ) -> FetchResult | None:
        m = self._ID_RE.search(urlparse(url).path) or re.search(
            r"[?&]id=([A-Za-z0-9]+)", url
        )
        if not m:
            return None
        article_id = m.group(1)
        market_m = self._MARKET_RE.search(url)
        market = market_m.group(1) if market_m else "en-us"
        api = f"https://assets.msn.com/content/view/v2/Detail/{market}/{article_id}"

        transport = httpx.AsyncHTTPTransport(proxy=proxy_url) if proxy_url else None
        try:
            async with httpx.AsyncClient(
                transport=transport, timeout=timeout, headers={"User-Agent": _UA}
            ) as client:
                resp = await client.get(api)
            if resp.status_code != 200:
                logger.debug("MSN API status %s for %s", resp.status_code, article_id)
                return None
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MSN API fetch failed: %s", exc)
            return None

        body_html = data.get("body") or ""
        title = data.get("title") or ""
        if not body_html or not title:
            return None

        authors = data.get("authors") or []
        author = ", ".join(
            a.get("name", "") for a in authors if isinstance(a, dict) and a.get("name")
        )
        date = data.get("publishedDateTime") or ""

        # Synthesise a clean document with JSON-LD so the normal extraction +
        # metadata-merge pipeline recovers title/author/date/body uniformly.
        ld = {
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": title,
            "datePublished": date,
        }
        if author:
            ld["author"] = {"@type": "Person", "name": author}
        synthetic = (
            f"<html><head><title>{_htmllib.escape(title)}</title>"
            f'<script type="application/ld+json">{json.dumps(ld)}</script>'
            f"</head><body><article><h1>{_htmllib.escape(title)}</h1>"
            f"{body_html}</article></body></html>"
        )
        return FetchResult(
            html=synthetic,
            resolved_url=url,
            strategy_tier="static",
            status_code=200,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS: tuple[SiteAdapter, ...] = (
    NaverBlogAdapter(),
    MSNAdapter(),
)


def get_adapter(url: str) -> SiteAdapter | None:
    """Return the first adapter whose host matches *url*, or ``None``."""
    for adapter in _ADAPTERS:
        if adapter.matches(url):
            return adapter
    return None
