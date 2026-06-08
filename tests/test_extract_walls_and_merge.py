"""Offline tests: wall detection, JSON-LD body fallback, and metadata merge."""
from __future__ import annotations

import pytest

from articula._extractor import _looks_like_wall, extract

# ---------------------------------------------------------------------------
# Wall detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "500 Apologies, but something went wrong on our end. Refresh the page.",
    "Please enable JavaScript and cookies to continue.",
    "medium.com Performing security verification This website uses a security service",
    "Sign in to continue reading this story.",
    "Are you a robot? Verify you are human.",
])
def test_short_wall_pages_detected(text):
    assert _looks_like_wall(text) is True


def test_long_article_mentioning_signin_is_not_a_wall():
    # A real, long article that merely contains the phrase must NOT be flagged.
    article = ("In this deep dive we discuss why apps ask you to sign in to continue. "
               * 40)
    assert len(article) > 600
    assert _looks_like_wall(article) is False


def test_normal_short_text_is_not_a_wall():
    assert _looks_like_wall("A short but perfectly ordinary paragraph about cats.") is False


# ---------------------------------------------------------------------------
# extract(): wall pages raise ValueError with an explanatory suffix
# ---------------------------------------------------------------------------


def test_extract_rejects_wall_page():
    html = """<html><head><title>Medium</title></head><body><article>
    <p>Open in app Sign up Sign in 500 Apologies, but something went wrong on our
    end. Refresh the page, check Medium's site status, or find something to read.</p>
    </article></body></html>"""
    with pytest.raises(ValueError) as exc:
        extract(html, "https://medium.com/p/x")
    assert "wall" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# extract(): JSON-LD body fallback when DOM body extractors find nothing
# ---------------------------------------------------------------------------


def test_jsonld_body_fallback(monkeypatch):
    # Force every DOM body extractor to fail so the JSON-LD articleBody fallback
    # is the only path that can succeed.
    import articula._extractor as ext
    monkeypatch.setattr(ext, "_extract_trafilatura", lambda h, u: None)
    monkeypatch.setattr(ext, "_extract_readability", lambda h, u: None)
    monkeypatch.setattr(ext, "_extract_heuristic", lambda h, u: None)

    body = "This article body exists only inside JSON-LD articleBody markup. " * 6
    html = f"""<html><head>
    <script type="application/ld+json">
    {{"@type":"NewsArticle","headline":"LD Only","author":"A. Writer",
      "datePublished":"2026-03-03","articleBody":"{body}"}}
    </script></head><body><div></div></body></html>"""
    art = extract(html, "https://example.com/a")
    assert art.method == "jsonld"
    assert art.title == "LD Only"
    assert art.author == "A. Writer"
    assert art.published_date == "2026-03-03"
    assert "JSON-LD articleBody" in art.text


# ---------------------------------------------------------------------------
# extract(): metadata merge fills title/author/date the body extractor misses
# ---------------------------------------------------------------------------


def test_metadata_merge_fills_missing_title_author_date():
    # A clean article body (trafilatura/readability will extract it) plus JSON-LD
    # metadata. The merged result must carry the JSON-LD title/author/date.
    paragraphs = "".join(
        f"<p>{('Sentence number ' + str(i) + ' with sufficient words here. ') * 3}</p>"
        for i in range(8)
    )
    html = f"""<html><head>
    <script type="application/ld+json">
    {{"@type":"NewsArticle","headline":"Merged Headline",
      "author":{{"@type":"Person","name":"Dana Byline"}},
      "datePublished":"2026-04-04T08:00:00Z"}}
    </script></head><body><article>{paragraphs}</article></body></html>"""
    art = extract(html, "https://example.com/news")
    assert art.title == "Merged Headline"
    assert art.author == "Dana Byline"
    assert art.published_date == "2026-04-04"
    assert len(art.text) > 200
