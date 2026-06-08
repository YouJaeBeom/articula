"""Sync convenience wrapper: ``scrape_many`` mirrors ``async_scrape_many``.

The bare ``scrape_many`` is the synchronous wrapper (matching the
``scrape`` / ``async_scrape`` naming convention) and manages its own event
loop.  It must:

* return results in input order, with per-URL errors captured (not raised);
* delegate to the underlying ``Scraper.scrape`` once per URL;
* raise ``RuntimeError`` when called from inside a running event loop,
  directing callers to ``async_scrape_many``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from articula import scrape_many
from articula._scraper import Scraper
from articula.exceptions import FetchError


def _dispatch(mapping: dict[str, object]):
    async def _inner(url: str) -> object:
        value = mapping[url]
        if isinstance(value, Exception):
            raise value
        return value

    return _inner


def test_sync_scrape_many_returns_results_in_order() -> None:
    urls = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    sentinels: dict[str, object] = {u: MagicMock(name=u) for u in urls}

    with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = _dispatch(sentinels)
        results = scrape_many(urls)

    assert results == [sentinels[u] for u in urls]
    assert mock_scrape.call_count == len(urls)


def test_sync_scrape_many_captures_per_url_errors() -> None:
    urls = ["https://example.com/ok", "https://example.com/bad"]
    err = FetchError("boom", url=urls[1], attempted_strategies=["static"])
    mapping: dict[str, object] = {urls[0]: MagicMock(name="ok"), urls[1]: err}

    with patch.object(Scraper, "scrape", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = _dispatch(mapping)
        results = scrape_many(urls)

    assert results[0] is mapping[urls[0]]
    assert isinstance(results[1], FetchError)
    assert results[1].url == urls[1]


def test_sync_scrape_many_empty_list() -> None:
    assert scrape_many([]) == []


@pytest.mark.asyncio
async def test_sync_scrape_many_raises_inside_running_loop() -> None:
    """Calling the sync wrapper from async code raises a clear RuntimeError."""
    with pytest.raises(RuntimeError, match="async_scrape_many"):
        scrape_many(["https://example.com/x"])
