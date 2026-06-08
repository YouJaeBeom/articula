"""Deterministic local verification of the two hardest capabilities.

Spins up a local HTTP server that reproduces, without any flaky external
dependency, the exact two scenarios the library exists to solve:

  1. JS-RENDERED PAGE
     /js  -> static HTML body is an empty <div id="article">; the real
             article is injected by inline JavaScript on load.  Non-browser
             tiers cannot extract a body (extract() raises) so the scraper
             must escalate to the Playwright browser tier, which renders the
             JS and recovers the article.

  2. BOT-DETECTION / ACCESS BLOCK
     /bot -> serves a Cloudflare-style "Just a moment..." JS challenge page
             (HTTP 200) to every client; its markup trips the bot-challenge
             detector so the static + rotation tiers reject it, and only a
             JS-executing browser resolves it to the real article.  The scraper
             must escalate past the block to the browser tier.

Run: .venv/bin/python verify_local.py
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from articula import Scraper

REAL_ARTICLE_TEXT = (
    "Large language models have moved from research curiosities to core "
    "infrastructure across the software industry. In this article we examine "
    "how teams are integrating them into production systems, the reliability "
    "challenges that remain, and the engineering patterns that have emerged to "
    "keep latency and cost under control while preserving output quality."
)

# A page whose body is empty until JavaScript runs.
JS_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>JS Rendered Article</title></head>
<body>
  <div id="article"></div>
  <noscript>Please enable JavaScript to view this article.</noscript>
  <script>
    document.addEventListener('DOMContentLoaded', function () {{
      var el = document.getElementById('article');
      el.innerHTML = '<article>'
        + '<h1>How Teams Run LLMs in Production</h1>'
        + '<p class="byline">By Jane Researcher</p>'
        + '<time datetime="2026-05-01">May 1, 2026</time>'
        + '<p>{REAL_ARTICLE_TEXT}</p>'
        + '</article>';
    }});
  </script>
</body>
</html>"""

# Cloudflare-style JS challenge: every client receives the SAME interstitial,
# whose markup trips the library's _is_bot_challenge() detector (so the
# static + rotation tiers reject it and escalate).  Only a JS-executing
# browser runs the embedded script, which swaps in the real article — exactly
# how a real Cloudflare "Just a moment..." challenge behaves.
BOT_CHALLENGE_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Just a moment...</title></head>
<body>
  <div id="article" class="cf-browser-verification">
    Checking your browser before accessing. Please enable JavaScript and
    cookies to continue. Ray ID: 8f0000000000
  </div>
  <script>
    document.addEventListener('DOMContentLoaded', function () {{
      document.body.innerHTML = '<article>'
        + '<h1>Inside a Bot-Protected Newsroom</h1>'
        + '<p class="byline">By Sam Reporter</p>'
        + '<time datetime="2026-04-15">April 15, 2026</time>'
        + '<p>{REAL_ARTICLE_TEXT}</p>'
        + '</article>';
    }});
  </script>
</body>
</html>"""


def _send(handler: BaseHTTPRequestHandler, status: int, html: str) -> None:
    body = html.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # silence the default stderr logging
        pass

    def do_GET(self):  # noqa: N802 - http.server API
        # Both routes serve identical bytes to every client.  The distinction
        # is purely whether the client executes JavaScript — i.e. whether the
        # scraper escalated to the Playwright browser tier.
        if self.path.startswith("/js"):
            _send(self, 200, JS_PAGE)
        elif self.path.startswith("/bot"):
            _send(self, 200, BOT_CHALLENGE_PAGE)
        else:
            self.send_response(404)
            self.end_headers()


async def run(base: str) -> None:
    print("=" * 90)
    print("DETERMINISTIC LOCAL VERIFICATION")
    print("=" * 90)

    async with Scraper(timeout=30) as scraper:
        # --- 1. JS-rendered page ---
        art = await scraper.scrape(f"{base}/js")
        ok_js = "production" in art.text.lower() and art.strategy_tier == "browser"
        print("\n[1] JS-rendered page (/js)")
        print(f"    strategy_tier : {art.strategy_tier}")
        print(f"    attempted     : {art.strategies_attempted}")
        print(f"    title         : {art.title!r}")
        print(f"    body chars    : {len(art.text)}")
        print(f"    recovered JS-injected article: {'YES' if 'production' in art.text.lower() else 'NO'}")
        print(f"    => escalated to browser tier : {'YES' if art.strategy_tier == 'browser' else 'NO'}")

        # --- 2. Bot-protected page ---
        art2 = await scraper.scrape(f"{base}/bot")
        ok_bot = "production" in art2.text.lower() and art2.strategy_tier == "browser"
        print("\n[2] Bot-protected page (/bot, JS challenge served to all tiers)")
        print(f"    strategy_tier : {art2.strategy_tier}")
        print(f"    attempted     : {art2.strategies_attempted}")
        print(f"    title         : {art2.title!r}")
        print(f"    body chars    : {len(art2.text)}")
        print(f"    => bypassed block via browser tier: {'YES' if art2.strategy_tier == 'browser' else 'NO'}")

    print("\n" + "=" * 90)
    print(f"JS-render escalation : {'PASS' if ok_js else 'FAIL'}")
    print(f"Bot-bypass escalation: {'PASS' if ok_bot else 'FAIL'}")
    print("=" * 90)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(run(f"http://127.0.0.1:{port}"))
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
