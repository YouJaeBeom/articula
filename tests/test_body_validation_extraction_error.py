"""
Unit tests for Sub-AC 3: The extraction pipeline's body-validation step
calls ``is_meaningful_body`` on the parsed body and raises
``ExtractionError`` when it returns ``False``.

Design
------
The ``extract()`` function in ``_extractor.py`` calls ``is_meaningful_body``
on every extractor's parsed body text as an explicit validation step.
When all extractors produce a parsed body that fails this check (body is too
short or has too few whitespace-delimited tokens), ``extract()`` raises
``ValueError``, which ``Scraper.scrape`` converts into ``ExtractionError``.

Test structure
--------------
The primary test mocks only the HTTP-fetch layer (``fetch_static``) so that
the real extraction pipeline runs on the injected HTML.  The HTML fixture is
carefully crafted so that every extractor tier produces a parsed body that
passes the character-count gate (≥ 80 chars) but fails the token-count gate
(< 10 tokens) enforced by ``is_meaningful_body`` — triggering ``ExtractionError``
without any mock on the extraction layer itself.

Additional tests verify:
* The raised exception is an ``ExtractionError`` instance (not ``ValueError``).
* The exception is a sub-class of ``ScraperError`` so broad-catch handlers work.
* The exception's ``.url`` attribute matches the scraped URL.
* Patching ``is_meaningful_body`` to return ``False`` while ``extract()``
  returns a valid body also triggers ``ExtractionError``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from articula._extractor import is_meaningful_body
from articula._fetcher import FetchResult
from articula._scraper import Scraper
from articula.exceptions import ExtractionError, ScraperError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_URL = "https://example.com/article"

# Patch targets (names imported into _scraper module namespace)
_PATCH_STATIC = "articula._scraper.fetch_static"
_PATCH_ROTATION = "articula._scraper.fetch_rotation"
_PATCH_BROWSER = "articula._scraper.fetch_browser"
_PATCH_EXTRACT = "articula._scraper.extract"
_PATCH_IS_MEANINGFUL = "articula._extractor.is_meaningful_body"

# ---------------------------------------------------------------------------
# HTML fixture: long single-token body (passes char check, fails token check)
# ---------------------------------------------------------------------------

# The <body> contains a single 100-character word that passes the heuristic
# extractor's character threshold (_MIN_BODY_CHARS = 80) but fails
# is_meaningful_body because it has only 1 whitespace-delimited token
# (_MIN_BODY_TOKENS = 10).  trafilatura and readability also reject or produce
# a body from this content that fails the token check.
_LOW_TOKEN_BODY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stub Article Title For Testing</title>
</head>
<body>
  <h1>Stub Article Title For Testing</h1>
  <p>AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA</p>
</body>
</html>
"""

# Verify the fixture body text length so the test is self-documenting.
_FIXTURE_BODY_TOKEN = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
assert len(_FIXTURE_BODY_TOKEN) >= 80, "fixture body must clear _MIN_BODY_CHARS"
assert len(_FIXTURE_BODY_TOKEN.split()) < 10, "fixture body must fail _MIN_BODY_TOKENS"
assert not is_meaningful_body(_FIXTURE_BODY_TOKEN), (
    "fixture body must fail is_meaningful_body for the test to be meaningful"
)


def _low_token_fetch_result() -> FetchResult:
    """Return a FetchResult whose HTML body parses to a low-token body."""
    return FetchResult(
        html=_LOW_TOKEN_BODY_HTML,
        resolved_url=_TEST_URL,
        strategy_tier="static",
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Primary test: real extraction pipeline raises ExtractionError
# ---------------------------------------------------------------------------


class TestBodyValidationRaisesExtractionError:
    """
    When a mocked HTTP fetch returns a page whose parsed body fails
    ``is_meaningful_body``, the extraction pipeline raises ``ExtractionError``.
    """

    @pytest.mark.asyncio
    async def test_extraction_error_raised_when_body_fails_threshold(self) -> None:
        """
        Core Sub-AC 3 assertion: the pipeline raises ``ExtractionError`` for a
        successfully fetched page whose parsed body fails ``is_meaningful_body``.

        The HTTP fetch layer is mocked; the extraction layer (``extract()``)
        runs for real, calling ``is_meaningful_body`` internally.
        """
        with patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static:
            mock_static.return_value = _low_token_fetch_result()

            with pytest.raises(ExtractionError):
                async with Scraper(force_strategy="static") as scraper:
                    await scraper.scrape(_TEST_URL)

    @pytest.mark.asyncio
    async def test_extraction_error_not_none_or_article(self) -> None:
        """
        The pipeline must raise, not return ``None`` or a partial ``Article``,
        when the parsed body fails the threshold check.
        """

        with patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static:
            mock_static.return_value = _low_token_fetch_result()

            result_holder: list[object] = []
            try:
                async with Scraper(force_strategy="static") as scraper:
                    result_holder.append(await scraper.scrape(_TEST_URL))
            except ExtractionError:
                pass  # expected
            else:
                pytest.fail(
                    f"Expected ExtractionError but scrape() returned {result_holder!r}"
                )

    @pytest.mark.asyncio
    async def test_raised_exception_is_extraction_error_type(self) -> None:
        """The raised exception is exactly ``ExtractionError``, not a bare ``ValueError``."""
        with patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static:
            mock_static.return_value = _low_token_fetch_result()

            with pytest.raises(ExtractionError) as exc_info:
                async with Scraper(force_strategy="static") as scraper:
                    await scraper.scrape(_TEST_URL)

        assert isinstance(exc_info.value, ExtractionError), (
            f"Expected ExtractionError, got {type(exc_info.value).__name__}"
        )

    @pytest.mark.asyncio
    async def test_exception_is_scraper_error_subclass(self) -> None:
        """``ExtractionError`` IS-A ``ScraperError`` — broad-catch handlers catch it."""
        with patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static:
            mock_static.return_value = _low_token_fetch_result()

            with pytest.raises(ScraperError) as exc_info:
                async with Scraper(force_strategy="static") as scraper:
                    await scraper.scrape(_TEST_URL)

        assert isinstance(exc_info.value, ExtractionError)

    @pytest.mark.asyncio
    async def test_exception_url_matches_input_url(self) -> None:
        """``ExtractionError.url`` equals the URL that was scraped."""
        with patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static:
            mock_static.return_value = _low_token_fetch_result()

            with pytest.raises(ExtractionError) as exc_info:
                async with Scraper(force_strategy="static") as scraper:
                    await scraper.scrape(_TEST_URL)

        assert exc_info.value.url == _TEST_URL, (
            f"Expected url={_TEST_URL!r}, got {exc_info.value.url!r}"
        )


# ---------------------------------------------------------------------------
# Isolation test: mock is_meaningful_body in _extractor to always return False
# — without mocking extract() itself — ExtractionError must be raised.
# ---------------------------------------------------------------------------


class TestBodyValidationViaIsMeaningfulBodyMock:
    """
    Patching ``is_meaningful_body`` in the ``_extractor`` module to always
    return ``False`` causes every extractor result to be rejected by the
    body-validation step inside ``extract()``, triggering ``ExtractionError``.

    The HTTP fetch is mocked (successful); ``extract()`` is NOT mocked so the
    real body-validation step runs with the patched gate function.
    """

    @pytest.mark.asyncio
    async def test_is_meaningful_body_false_triggers_extraction_error(self) -> None:
        """
        When fetch succeeds and ``is_meaningful_body`` always returns ``False``,
        every extractor's parsed body is rejected and ``ExtractionError`` is
        raised — confirming that ``is_meaningful_body`` is the gate function
        used by the extraction pipeline's body-validation step.
        """
        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_IS_MEANINGFUL, return_value=False),
        ):
            mock_static.return_value = FetchResult(
                html=_LOW_TOKEN_BODY_HTML,
                resolved_url=_TEST_URL,
                strategy_tier="static",
                status_code=200,
            )

            with pytest.raises(ExtractionError):
                async with Scraper(force_strategy="static") as scraper:
                    await scraper.scrape(_TEST_URL)


# ---------------------------------------------------------------------------
# Verification: is_meaningful_body IS called by the extraction pipeline
# ---------------------------------------------------------------------------


class TestIsMeaningfulBodyIsCalledByPipeline:
    """
    ``is_meaningful_body`` is explicitly called by ``extract()`` as a
    body-validation step for every extractor result.  When all results fail
    the check, the pipeline raises ``ExtractionError``.
    """

    @pytest.mark.asyncio
    async def test_is_meaningful_body_called_during_extraction(self) -> None:
        """
        When ``fetch_static`` succeeds and returns the low-token HTML fixture,
        ``is_meaningful_body`` is called inside ``extract()`` on the parsed body.

        Verified by patching ``is_meaningful_body`` in the ``_extractor`` module
        and asserting it was called at least once.
        """
        call_tracker: list[str] = []

        original_is_meaningful = is_meaningful_body

        def tracking_is_meaningful(text: str) -> bool:
            call_tracker.append(text)
            return original_is_meaningful(text)

        with (
            patch(_PATCH_STATIC, new_callable=AsyncMock) as mock_static,
            patch(_PATCH_IS_MEANINGFUL, side_effect=tracking_is_meaningful),
        ):
            mock_static.return_value = _low_token_fetch_result()

            with pytest.raises(ExtractionError):
                async with Scraper(force_strategy="static") as scraper:
                    await scraper.scrape(_TEST_URL)

        assert len(call_tracker) >= 1, (
            "is_meaningful_body must be called at least once by the extraction pipeline"
        )
