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
    ("EN news (BBC)", "https://www.bbc.com/news/articles/cj0g4425zmeo", "news"), # 잘 못함
    ("EN news (Guardian)", "https://www.theguardian.com/world/live/2026/jun/07/israel-lebanon-southern-beirut-hezbollah-idf-ceasefire-iran-latest-news-updates", "news"),
    ("EN news (AP)", "https://apnews.com/article/world-cup-fifa-security-secret-service-trump-32f04baf3a242395f26816292a9dc7e2", "news"),
    ("KO news (Yonhap)", "https://www.yna.co.kr/view/AKR20260608020900005?section=industry/all&site=hot_news_view_swipe01", "news_ko"),
    ("KO news (Hani)", "https://www.hani.co.kr/arti/economy/economy_general/1100000.html", "news_ko"),
    ("Blog (Overreacted)", "https://overreacted.io/a-complete-guide-to-useeffect/", "blog"),
    ("Dev blog (Simon W)", "https://simonwillison.net/2024/Dec/31/llms-in-2024/", "blog"),
    ("JS/SPA (Vercel blog)", "https://vercel.com/blog/framework-defined-infrastructure", "js"),
    ("Bot-protected (Cloudflare)", "https://www.cloudflare.com/learning/bots/what-is-a-bot/", "bot"),
    ("Medium", "https://medium.com/design-bootcamp/i-sat-in-engineering-meetings-for-two-years-without-understanding-what-a-branch-was-c106ce7cadf8", "blog"), #안됨
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
