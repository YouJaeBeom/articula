"""
Live smoke tests for articula.

These tests make real HTTP/browser requests and require a live internet
connection.  They are marked @pytest.mark.live so that the offline CI
suite (``pytest -m 'not live'``) never touches them.

Run the live suite manually:
    pytest -m live -v

All tests in this file are @pytest.mark.live.  Adding any non-live test
here would violate the Sub-AC 6a invariant that "pytest -m live" collects
only live-labelled tests and "pytest -m 'not live'" collects zero of them.
"""

from __future__ import annotations

import pytest


@pytest.mark.live
def test_live_scrape_english_wikipedia() -> None:
    """Live: scrape the English Wikipedia article on artificial intelligence."""
    from articula import scrape

    article = scrape("https://en.wikipedia.org/wiki/Artificial_intelligence")
    assert article.title, "Expected a non-empty title"
    assert len(article.text) > 100, "Expected substantial body text"
    assert article.detected_language == "en"
    assert article.resolved_url.startswith("https://")


@pytest.mark.live
def test_live_scrape_korean_wikipedia() -> None:
    """Live: scrape the Korean Wikipedia article on artificial intelligence."""
    from articula import scrape

    article = scrape(
        "https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5"
    )
    assert article.title, "Expected a non-empty title"
    assert len(article.text) > 100, "Expected substantial body text"
    assert article.detected_language == "ko"
    assert article.resolved_url.startswith("https://")


@pytest.mark.live
def test_live_async_scrape_english() -> None:
    """Live: async_scrape returns a valid Article for a static English page."""
    import asyncio

    from articula import async_scrape

    article = asyncio.run(
        async_scrape("https://en.wikipedia.org/wiki/Python_(programming_language)")
    )
    assert article.title
    assert article.text
    assert article.detected_language == "en"
