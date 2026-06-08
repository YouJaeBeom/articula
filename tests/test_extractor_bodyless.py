"""
Sub-AC 4: Extractor function returns ``None`` (or raises) instead of a partial
``Article`` when body text extraction also fails.

The partial-success logic — creating an ``Article`` (or ``ExtractionResult``)
where ``author`` and/or ``published_date`` are ``None`` while ``text`` is
present — must ONLY be triggered when a meaningful body is present.  When body
text extraction fails entirely, ``extract()`` must raise ``ValueError`` rather
than returning any partial result.

All tests use offline HTML fixtures — no network access is required.

Test coverage
-------------
1. Empty ``<body>`` element → ``ValueError``
2. Body containing only boilerplate (nav/footer) that every extractor strips →
   ``ValueError``
3. Body with text far below ``_MIN_BODY_CHARS`` (80 chars) → ``ValueError``
4. No ``<body>`` tag at all → ``ValueError``
5. Exception type is ``ValueError`` (not a silent ``None`` return)
6. No ``ExtractionResult`` is returned on failure (contrast: partial results are
   valid only when body is present)
7. Contrast fixture: body-present page with missing metadata succeeds and proves
   partial-success path is legitimate when body exists
"""

from __future__ import annotations

import pytest

from articula._extractor import ExtractionResult, extract

# ---------------------------------------------------------------------------
# HTML fixtures — body-less or trivially short body
# ---------------------------------------------------------------------------

# 1. Completely empty <body>.
_EMPTY_BODY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Empty Body Page</title>
</head>
<body>
</body>
</html>
"""

# 2. Body that contains ONLY navigation and footer boilerplate.
#    Every extraction tier strips these noise tags, leaving nothing extractable.
_NAV_ONLY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Navigation Only Page</title>
</head>
<body>
  <nav>
    <a href="/">Home</a>
    <a href="/about">About</a>
    <a href="/contact">Contact</a>
    <a href="/news">News</a>
  </nav>
  <footer>
    <p>Copyright 2024 Example Corp. All rights reserved.</p>
  </footer>
</body>
</html>
"""

# 3. Body with a title but body paragraph well below _MIN_BODY_CHARS (80).
_SHORT_BODY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Article Title</title>
</head>
<body>
  <h1>Article Title</h1>
  <p>Too short.</p>
</body>
</html>
"""

# 4. No <body> tag at all — heuristic fallback cannot match.
_NO_BODY_TAG_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>No Body Tag</title>
</head>
</html>
"""

# Contrast fixture: body IS present, metadata (author + date) is absent.
#    This exercises the VALID partial-success path to confirm it still works.
_BODYFUL_NO_METADATA_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Article With Body But No Metadata</title>
</head>
<body>
  <article>
    <h1>Article With Body But No Metadata</h1>
    <p>
      This article has a fully extractable body but contains no author markup
      and no publication date.  The extractor must return a populated
      ExtractionResult with author=None and published_date=None rather than
      raising an error.  Having a meaningful body is what unlocks the
      partial-success path.
    </p>
    <p>
      A second paragraph ensures the body comfortably exceeds the minimum
      character and token thresholds enforced by every extraction tier, so
      the test is not sensitive to which extractor runs first.
    </p>
  </article>
</body>
</html>
"""

_TEST_URL = "https://example.com/article"


# ---------------------------------------------------------------------------
# 1-5: Body-less fixtures must raise ValueError
# ---------------------------------------------------------------------------


class TestBodylessRaisesValueError:
    """
    When no extractable body exists, ``extract()`` must raise ``ValueError``.

    Covers four distinct body-less scenarios:
    - completely empty ``<body>``
    - ``<body>`` containing only stripped boilerplate (nav/footer)
    - body text far below the ``_MIN_BODY_CHARS`` (80-char) threshold
    - HTML with no ``<body>`` element at all
    """

    def test_empty_body_raises(self) -> None:
        """Completely empty <body> → ValueError, not a partial ExtractionResult."""
        with pytest.raises(ValueError):
            extract(_EMPTY_BODY_HTML, _TEST_URL)

    def test_nav_only_body_raises(self) -> None:
        """Body with only nav/footer boilerplate → ValueError after all tiers fail."""
        with pytest.raises(ValueError):
            extract(_NAV_ONLY_HTML, _TEST_URL)

    def test_short_body_raises(self) -> None:
        """Body text below _MIN_BODY_CHARS (80 chars) → ValueError."""
        with pytest.raises(ValueError):
            extract(_SHORT_BODY_HTML, _TEST_URL)

    def test_no_body_tag_raises(self) -> None:
        """HTML with no <body> tag whatsoever → ValueError from all tiers."""
        with pytest.raises(ValueError):
            extract(_NO_BODY_TAG_HTML, _TEST_URL)


# ---------------------------------------------------------------------------
# 6: Exception type must be ValueError (not silent None)
# ---------------------------------------------------------------------------


class TestExceptionType:
    """``extract()`` must raise ``ValueError`` — not return ``None`` silently."""

    def test_raises_value_error_not_returns_none(self) -> None:
        """``extract()`` raises rather than returning ``None`` on total failure."""
        raised = False
        try:
            result = extract(_EMPTY_BODY_HTML, _TEST_URL)
            # If we reach here, extract() returned instead of raising.
            # result should NOT be a valid ExtractionResult.
            assert False, f"extract() should have raised ValueError, got {result!r}"
        except ValueError:
            raised = True
        assert raised, "extract() must raise ValueError when body extraction fails"

    def test_raised_exception_is_value_error_subclass(self) -> None:
        """The exception is a ``ValueError`` or its subclass."""
        with pytest.raises(ValueError) as exc_info:
            extract(_SHORT_BODY_HTML, _TEST_URL)
        assert isinstance(exc_info.value, ValueError)

    def test_error_message_mentions_extraction_failure(self) -> None:
        """The ValueError message is informative about the failure."""
        with pytest.raises(ValueError, match=r"(?i)(extraction|empty|trafilatura|readability|heuristic)"):
            extract(_EMPTY_BODY_HTML, _TEST_URL)


# ---------------------------------------------------------------------------
# 7: No ExtractionResult returned on body-less failure
# ---------------------------------------------------------------------------


class TestNoPartialResultOnFailure:
    """Verifies the extractor never returns a partial ``ExtractionResult`` without body."""

    @pytest.mark.parametrize(
        ("html", "label"),
        [
            (_EMPTY_BODY_HTML, "empty_body"),
            (_NAV_ONLY_HTML, "nav_only"),
            (_SHORT_BODY_HTML, "short_body"),
            (_NO_BODY_TAG_HTML, "no_body_tag"),
        ],
        ids=["empty_body", "nav_only", "short_body", "no_body_tag"],
    )
    def test_body_absent_never_returns_extraction_result(
        self, html: str, label: str
    ) -> None:
        """For every body-less fixture, ``extract()`` raises rather than returning a result."""
        with pytest.raises(ValueError):
            extract(html, _TEST_URL)

    def test_empty_body_does_not_return_extraction_result(self) -> None:
        """Explicit check: empty-body HTML must not produce an ``ExtractionResult``."""
        try:
            result = extract(_EMPTY_BODY_HTML, _TEST_URL)
            # Should never reach here — extract() must raise, not return.
            pytest.fail(
                f"extract() should have raised ValueError for body-less HTML, "
                f"but returned {result!r}"
            )
        except ValueError:
            pass  # expected — body-less HTML correctly triggers ValueError


# ---------------------------------------------------------------------------
# 8: Contrast — body-present HTML with absent metadata is valid partial success
# ---------------------------------------------------------------------------


class TestPartialSuccessOnlyWithBody:
    """
    Partial-success path (``author=None`` or ``published_date=None``) is valid
    ONLY when body text is present.  This contrast test confirms:

    * Body-less HTML  →  ``ValueError`` (previous tests)
    * Body-present HTML with no metadata  →  ``ExtractionResult`` with non-empty text
    """

    def test_body_present_returns_extraction_result(self) -> None:
        """When body IS present, ``extract()`` returns an ``ExtractionResult``."""
        result = extract(_BODYFUL_NO_METADATA_HTML, _TEST_URL)

        assert isinstance(result, ExtractionResult)

    def test_body_present_text_is_non_empty(self) -> None:
        """Partial-success result has a non-empty body text."""
        result = extract(_BODYFUL_NO_METADATA_HTML, _TEST_URL)

        assert result.text.strip() != "", "body text must be non-empty for partial success"

    def test_body_present_author_may_be_none(self) -> None:
        """Missing author markup → ``author=None`` is the expected partial-success value."""
        result = extract(_BODYFUL_NO_METADATA_HTML, _TEST_URL)

        # author is either None (no markup) or a string — both are acceptable
        assert result.author is None or isinstance(result.author, str)

    def test_body_present_published_date_may_be_none(self) -> None:
        """Missing date markup → ``published_date=None`` is the expected value."""
        result = extract(_BODYFUL_NO_METADATA_HTML, _TEST_URL)

        assert result.published_date is None or isinstance(result.published_date, str)

    def test_partial_success_contrast_empty_body_raises(self) -> None:
        """
        Contrast: same URL, body-less HTML raises; body-present HTML succeeds.

        This is the core assertion of Sub-AC 4: partial-success is gated on body.
        """
        # Body-less → must raise
        with pytest.raises(ValueError):
            extract(_EMPTY_BODY_HTML, _TEST_URL)

        # Body-present with no metadata → must succeed (partial result)
        result = extract(_BODYFUL_NO_METADATA_HTML, _TEST_URL)
        assert isinstance(result, ExtractionResult)
        assert result.text.strip() != ""
