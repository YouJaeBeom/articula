"""Ad-hoc live verification harness: scrape a diverse real-world URL matrix.

Run: .venv/bin/python verify_live.py
Prints a per-URL result table (tier, confidence, title/author/date, body length)
so we can eyeball that the escalation chain works on real news/blogs/JS/bot sites.
"""

from __future__ import annotations

import asyncio
import time

from articula import Scraper, ScraperError

# (label, url, category)
MATRIX = [
    ("EN news (BBC)", "https://www.bbc.com/news/world-us-canada-68672500", "news"),
    ("EN news (Guardian)", "https://www.theguardian.com/technology/2024/jan/10/ai-chatbots", "news"),
    ("EN news (AP)", "https://apnews.com/hub/artificial-intelligence", "news"),
    ("KO news (Yonhap)", "https://www.yna.co.kr/view/AKR20240101000100001", "news_ko"),
    ("KO news (Hani)", "https://www.hani.co.kr/arti/economy/economy_general/1100000.html", "news_ko"),
    ("Blog (Overreacted)", "https://overreacted.io/a-complete-guide-to-useeffect/", "blog"),
    ("Dev blog (Simon W)", "https://simonwillison.net/2024/Dec/31/llms-in-2024/", "blog"),
    ("JS/SPA (Vercel blog)", "https://vercel.com/blog/framework-defined-infrastructure", "js"),
    ("Bot-protected (Cloudflare)", "https://www.cloudflare.com/learning/bots/what-is-a-bot/", "bot"),
    ("Medium", "https://medium.com/@addyosmani/start-performance-budgeting-dad82d5a2b9e", "blog"),
]


def truncate(value: object, length: int = 48) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").strip()
    return (text[: length - 1] + "…") if len(text) > length else text


async def run() -> list:
    results = []
    async with Scraper(timeout=45.0) as scraper:
        for label, url, category in MATRIX:
            start = time.monotonic()
            try:
                art = await scraper.scrape(url)
                elapsed = time.monotonic() - start
                results.append((label, category, "OK", art, elapsed, None))
            except ScraperError as exc:
                elapsed = time.monotonic() - start
                results.append((label, category, "FAIL", None, elapsed, exc))
            except Exception as exc:  # noqa: BLE001 - harness wants to see everything
                elapsed = time.monotonic() - start
                results.append((label, category, "ERR", None, elapsed, exc))
    return results


def main() -> None:
    results = asyncio.run(run())

    print("\n" + "=" * 110)
    print("LIVE VERIFICATION RESULTS")
    print("=" * 110)
    ok = 0
    js_ok = False
    bot_ok = False
    for label, category, status, art, elapsed, exc in results:
        print(f"\n[{status}] {label}  ({category}, {elapsed:.1f}s)")
        if art is not None:
            ok += 1
            if category == "js" and art.text:
                js_ok = True
            if category == "bot" and art.text:
                bot_ok = True
            print(f"    tier       : {art.strategy_tier} via {art.extraction_method} (conf {art.extraction_confidence:.2f})")
            print(f"    lang       : {art.detected_language}")
            print(f"    title      : {truncate(art.title, 70)}")
            print(f"    author     : {truncate(art.author)}")
            print(f"    published  : {truncate(art.published_date)}")
            print(f"    body chars : {len(art.text)}")
            print(f"    attempted  : {art.strategies_attempted}")
        else:
            print(f"    error      : {type(exc).__name__}: {truncate(exc, 80)}")
            if exc is not None and hasattr(exc, "strategies_attempted"):
                print(f"    attempted  : {getattr(exc, 'strategies_attempted', None)}")

    total = len(results)
    rate = ok / total * 100 if total else 0
    print("\n" + "=" * 110)
    print(f"SUMMARY: {ok}/{total} extracted title+body ({rate:.0f}%)")
    print(f"  JS-rendered case succeeded : {js_ok}")
    print(f"  Bot-protected case succeeded: {bot_ok}")
    print("=" * 110)


if __name__ == "__main__":
    main()
