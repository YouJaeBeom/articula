"""Offline tests for site adapters (Naver, MSN) and the adapter registry."""
from __future__ import annotations

import pytest

from articula._adapters import MSNAdapter, NaverBlogAdapter, get_adapter

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,kind", [
    ("https://blog.naver.com/balahk/224302042478", NaverBlogAdapter),
    ("https://m.blog.naver.com/balahk/224302042478", NaverBlogAdapter),
    ("https://www.msn.com/en-us/news/politics/x/ar-AA252lKk", MSNAdapter),
    ("https://www.bbc.com/news/articles/abc", type(None)),
    ("https://example.com/post", type(None)),
])
def test_get_adapter_matching(url, kind):
    adapter = get_adapter(url)
    if kind is type(None):
        assert adapter is None
    else:
        assert isinstance(adapter, kind)


# ---------------------------------------------------------------------------
# Naver
# ---------------------------------------------------------------------------


class TestNaverRewrite:
    def test_path_form_rewrites_to_postview(self):
        out = NaverBlogAdapter().rewrite_url("https://blog.naver.com/balahk/224302042478")
        assert "PostView.naver" in out
        assert "blogId=balahk" in out
        assert "logNo=224302042478" in out

    def test_query_form_rewrites_to_postview(self):
        out = NaverBlogAdapter().rewrite_url(
            "https://blog.naver.com/PostView.naver?blogId=abc&logNo=123"
        )
        assert "blogId=abc" in out and "logNo=123" in out

    def test_unparseable_url_returned_unchanged(self):
        url = "https://blog.naver.com/"
        assert NaverBlogAdapter().rewrite_url(url) == url


class TestNaverIsolate:
    _HTML = """
    <html><head><meta property="og:title" content="My Post Title"></head>
    <body>
      <div class="copyright_banner">저작권 침해가 우려되는 컨텐츠 글보내기 기능을 제한합니다</div>
      <div class="se-main-container">
        <p>This is the real post body with enough words to be a meaningful article body indeed.</p>
        <p>It spans multiple paragraphs and clearly exceeds the minimum thresholds for extraction.</p>
      </div>
      <span class="se_publishDate">2026. 5. 31. 22:53</span>
    </body></html>
    """

    def test_isolate_returns_container_with_jsonld_date(self):
        out = NaverBlogAdapter().isolate_content(self._HTML, "https://blog.naver.com/x/1")
        assert out is not None
        assert "real post body" in out
        assert "My Post Title" in out
        assert '"datePublished": "2026-05-31"' in out
        # The copyright banner outside the container is dropped.
        assert "글보내기 기능을 제한" not in out

    def test_isolate_returns_none_without_container(self):
        html = "<html><body><p>no se-main-container here</p></body></html>"
        assert NaverBlogAdapter().isolate_content(html, "https://blog.naver.com/x/1") is None


# ---------------------------------------------------------------------------
# MSN
# ---------------------------------------------------------------------------


class TestMSNParsing:
    def test_id_regex_targets_ar_segment_not_locale(self):
        url = "https://www.msn.com/en-us/news/politics/some-title/ar-AA252lKk?ocid=x"
        m = MSNAdapter._ID_RE.search(url)
        assert m and m.group(1) == "AA252lKk"

    def test_market_regex(self):
        url = "https://www.msn.com/en-gb/news/x/ar-ABC123"
        m = MSNAdapter._MARKET_RE.search(url)
        assert m and m.group(1) == "en-gb"

    @pytest.mark.asyncio
    async def test_fetch_override_builds_document_from_api(self, monkeypatch):
        """A mocked MSN API response is turned into an extractable FetchResult."""
        import articula._adapters as adapters_mod

        payload = {
            "title": "MSN Headline",
            "authors": [{"name": "Joey Garrison, USA TODAY"}],
            "publishedDateTime": "2026-06-07T22:50:00Z",
            "body": "<p>" + ("Full MSN article body sentence. " * 20) + "</p>",
        }

        class _Resp:
            status_code = 200

            def json(self):
                return payload

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **k):
                return _Resp()

        monkeypatch.setattr(adapters_mod.httpx, "AsyncClient", _Client)

        fr = await MSNAdapter().fetch_override(
            "https://www.msn.com/en-us/news/x/ar-AA252lKk", timeout=30, proxy_url=None
        )
        assert fr is not None
        assert "MSN Headline" in fr.html
        assert "Full MSN article body" in fr.html
        assert '"datePublished": "2026-06-07T22:50:00Z"' in fr.html
        assert "Joey Garrison" in fr.html

    @pytest.mark.asyncio
    async def test_fetch_override_none_when_no_id(self):
        fr = await MSNAdapter().fetch_override(
            "https://www.msn.com/en-us/weather", timeout=30, proxy_url=None
        )
        assert fr is None
