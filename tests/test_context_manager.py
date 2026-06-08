"""
Sub-AC 1: Scraper sync context manager protocol.

Verifies that:
  - ``Scraper.__enter__`` returns the Scraper instance itself.
  - ``Scraper.__exit__`` calls ``_cleanup()``.
  - The protocol behaves correctly even when an exception is raised inside
    the ``with`` block (i.e. cleanup still runs).
  - The async counterpart (``__aenter__`` / ``__aexit__``) also delegates to
    ``_cleanup()``, so both surfaces share the same teardown path.

All tests are unit-level and use ``unittest.mock`` to isolate the Scraper
from live network calls and Playwright.  The "mock browser" fixture patches
``sys.modules["playwright"]`` with a ``MagicMock`` so that
``is_browser_available()`` returns ``True`` without launching a real browser.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from articula._scraper import Scraper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_browser_env() -> MagicMock:
    """Patch sys.modules so that Playwright appears installed.

    Yields
    ------
    MagicMock
        The mock object standing in for the ``playwright`` top-level package.
    """
    mock_pw = MagicMock()
    with patch.dict(sys.modules, {"playwright": mock_pw}):
        yield mock_pw


# ---------------------------------------------------------------------------
# __enter__ — returns self
# ---------------------------------------------------------------------------


class TestEnterReturnsSelf:
    """Scraper.__enter__ must return the Scraper instance."""

    def test_enter_returns_exact_instance(self) -> None:
        """Calling __enter__() directly must return the same Scraper object."""
        scraper = Scraper()
        result = scraper.__enter__()
        assert result is scraper

    def test_with_statement_binding_is_self(self) -> None:
        """``with Scraper() as s`` must bind the Scraper instance to ``s``."""
        outer = Scraper()
        with outer as s:
            assert s is outer

    def test_nested_with_returns_correct_instances(self) -> None:
        """Nested ``with`` blocks each bind their own Scraper instance."""
        scraper_a = Scraper()
        scraper_b = Scraper(timeout=60.0)
        with scraper_a as a, scraper_b as b:
            assert a is scraper_a
            assert b is scraper_b
            assert a is not b


# ---------------------------------------------------------------------------
# __exit__ — calls _cleanup
# ---------------------------------------------------------------------------


class TestExitCallsCleanup:
    """Scraper.__exit__ must delegate to _cleanup()."""

    def test_exit_calls_cleanup_directly(self) -> None:
        """Calling __exit__ directly invokes _cleanup exactly once."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup:
            scraper.__exit__(None, None, None)
        mock_cleanup.assert_called_once()

    def test_with_statement_calls_cleanup_on_normal_exit(self) -> None:
        """Leaving the ``with`` block normally triggers ``_cleanup`` once."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup, scraper:
            pass
        mock_cleanup.assert_called_once()

    def test_with_statement_calls_cleanup_on_exception(self) -> None:
        """``_cleanup`` is called even when the ``with`` block raises."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup, pytest.raises(RuntimeError):
            with scraper:
                raise RuntimeError("intentional test error")
        mock_cleanup.assert_called_once()

    def test_cleanup_called_with_no_arguments(self) -> None:
        """``_cleanup()`` is invoked with no positional or keyword arguments."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup:
            scraper.__exit__(None, None, None)
        mock_cleanup.assert_called_once_with()

    def test_exit_passes_exception_info_through(self) -> None:
        """``__exit__`` does not swallow exceptions from the ``with`` block."""
        scraper = Scraper()
        sentinel = ValueError("sentinel")

        with patch.object(scraper, "_cleanup"), pytest.raises(ValueError, match="sentinel"):
            with scraper:
                raise sentinel


# ---------------------------------------------------------------------------
# Protocol with mocked browser environment
# ---------------------------------------------------------------------------


class TestContextManagerWithMockBrowser:
    """Context manager protocol verified when Playwright is mocked as installed."""

    def test_enter_returns_self_with_mock_browser(
        self, mock_browser_env: MagicMock
    ) -> None:
        """__enter__ returns self even when the browser module is stubbed."""
        scraper = Scraper()
        with scraper as s:
            assert s is scraper

    def test_exit_calls_cleanup_with_mock_browser(
        self, mock_browser_env: MagicMock
    ) -> None:
        """__exit__ calls _cleanup when Playwright is stubbed in sys.modules."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup, scraper:
            pass
        mock_cleanup.assert_called_once()

    def test_cleanup_called_on_exception_with_mock_browser(
        self, mock_browser_env: MagicMock
    ) -> None:
        """_cleanup is called even if an exception escapes the with-block."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup, pytest.raises(KeyError):
            with scraper:
                raise KeyError("browser-simulated failure")
        mock_cleanup.assert_called_once()

    def test_context_manager_enter_does_not_raise_with_mock_browser(
        self, mock_browser_env: MagicMock
    ) -> None:
        """Entering the context manager succeeds silently."""
        scraper = Scraper()
        # Should not raise any exception
        entered = scraper.__enter__()
        scraper.__exit__(None, None, None)
        assert entered is scraper


# ---------------------------------------------------------------------------
# Async context manager — delegates to _cleanup too
# ---------------------------------------------------------------------------


class TestAsyncContextManagerDelegatesToCleanup:
    """__aexit__ must also delegate to _cleanup for a consistent teardown path."""

    @pytest.mark.asyncio
    async def test_aexit_calls_cleanup(self) -> None:
        """Exiting the async context manager triggers _cleanup."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup:
            async with scraper:
                pass
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        """__aenter__ returns the Scraper instance."""
        scraper = Scraper()
        async with scraper as s:
            assert s is scraper

    @pytest.mark.asyncio
    async def test_aexit_calls_cleanup_on_exception(self) -> None:
        """_cleanup is called even when the async with block raises."""
        scraper = Scraper()
        with patch.object(scraper, "_cleanup") as mock_cleanup, pytest.raises(RuntimeError):
            async with scraper:
                raise RuntimeError("async error")
        mock_cleanup.assert_called_once()
