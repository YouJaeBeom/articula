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


# ---------------------------------------------------------------------------
# Real-world hard cases (regressions for the v0.2 extraction fixes)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_bbc_article_body_not_promo() -> None:
    """BBC: real article body (not a related-video promo) + JSON-LD metadata."""
    from articula import scrape

    a = scrape("https://www.bbc.com/news/articles/cj0g4425zmeo")
    assert a.title and "stabbing" in a.title.lower()
    assert len(a.text) > 800, "expected the full article, not a promo blurb"
    assert a.author  # filled from JSON-LD
    assert a.published_date


@pytest.mark.live
def test_live_msn_via_content_api() -> None:
    """MSN: article recovered through the assets.msn.com content API adapter."""
    from articula import scrape

    a = scrape(
        "https://www.msn.com/en-us/news/politics/top-takeaways-from-trumps-"
        "contentious-rainy-meet-the-press-interview/ar-AA252lKk?ocid=BingNewsVerp"
    )
    assert a.title
    assert len(a.text) > 1500
    assert a.published_date


@pytest.mark.live
def test_live_naver_blog_iframe() -> None:
    """Naver: follow the iframe to PostView and extract the SmartEditor body."""
    from articula import scrape

    a = scrape("https://blog.naver.com/balahk/224302042478")
    assert a.title
    assert len(a.text) > 800
    assert a.detected_language == "ko"


@pytest.mark.live
def test_live_medium_returns_content_or_clean_wall_error() -> None:
    """Medium is Cloudflare/member-gated: either we extract it, or we raise a
    clear ScraperError — never silently return the login/500 wall text."""
    from articula import ScraperError, scrape

    try:
        a = scrape(
            "https://medium.com/design-bootcamp/i-sat-in-engineering-meetings-for-"
            "two-years-without-understanding-what-a-branch-was-c106ce7cadf8"
        )
    except ScraperError:
        return  # acceptable: blocked, but surfaced as a clean typed error
    # If it succeeded, it must be the real article, not the wall.
    assert len(a.text) > 400
    assert "500 Apologies" not in a.text
    assert "Performing security verification" not in a.text
