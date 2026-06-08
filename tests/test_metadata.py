"""Offline tests for JSON-LD + meta-tag metadata extraction."""
from __future__ import annotations

from articula._metadata import PageMetadata, extract_metadata

_JSONLD_NEWS = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"NewsArticle",
 "headline":"Big News Today",
 "author":{"@type":"Person","name":"Jane Reporter"},
 "datePublished":"2026-01-15T10:00:00Z",
 "articleBody":"%s"}
</script>
</head><body><p>ignored</p></body></html>
""" % ("This is the full article body provided directly in JSON-LD. " * 6)


def test_jsonld_newsarticle_fields():
    md = extract_metadata(_JSONLD_NEWS)
    assert md.title == "Big News Today"
    assert md.author == "Jane Reporter"
    assert md.published_date == "2026-01-15T10:00:00Z"
    assert md.body is not None and "full article body" in md.body


def test_jsonld_author_list_is_joined():
    html = """<html><head><script type="application/ld+json">
    {"@type":"Article","headline":"X",
     "author":[{"@type":"Person","name":"Ann"},{"@type":"Person","name":"Bob"}]}
    </script></head><body></body></html>"""
    md = extract_metadata(html)
    assert md.author == "Ann, Bob"


def test_jsonld_graph_is_flattened():
    html = """<html><head><script type="application/ld+json">
    {"@context":"https://schema.org","@graph":[
      {"@type":"WebSite","name":"Site"},
      {"@type":"BlogPosting","headline":"Graph Headline","datePublished":"2025-12-01"}
    ]}</script></head><body></body></html>"""
    md = extract_metadata(html)
    assert md.title == "Graph Headline"
    assert md.published_date == "2025-12-01"


def test_meta_tag_fallbacks_when_no_jsonld():
    html = """<html><head>
    <meta property="og:title" content="OG Title">
    <meta name="author" content="Meta Author">
    <meta property="article:published_time" content="2026-02-02T00:00:00Z">
    <title>Tag Title</title></head><body></body></html>"""
    md = extract_metadata(html)
    assert md.title == "OG Title"          # og:title preferred over <title>
    assert md.author == "Meta Author"
    assert md.published_date == "2026-02-02T00:00:00Z"


def test_title_tag_used_when_no_og():
    html = "<html><head><title>Only Title Tag</title></head><body></body></html>"
    assert extract_metadata(html).title == "Only Title Tag"


def test_jsonld_wins_over_meta():
    html = """<html><head>
    <meta property="og:title" content="Meta Title">
    <script type="application/ld+json">
    {"@type":"NewsArticle","headline":"JSONLD Title","author":"LD Author"}
    </script></head><body></body></html>"""
    md = extract_metadata(html)
    assert md.title == "JSONLD Title"
    assert md.author == "LD Author"


def test_empty_html_returns_empty_metadata():
    md = extract_metadata("<html><body>nothing</body></html>")
    assert md == PageMetadata()


def test_short_jsonld_body_ignored():
    # articleBody under 200 chars is not treated as a usable full body.
    html = """<html><head><script type="application/ld+json">
    {"@type":"Article","headline":"H","articleBody":"too short"}
    </script></head><body></body></html>"""
    assert extract_metadata(html).body is None


def test_malformed_jsonld_is_skipped():
    html = """<html><head>
    <script type="application/ld+json">{not valid json,,,}</script>
    <meta property="og:title" content="Survives">
    </head><body></body></html>"""
    assert extract_metadata(html).title == "Survives"
