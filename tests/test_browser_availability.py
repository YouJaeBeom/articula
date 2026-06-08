"""
Sub-AC 3: Browser escalation module import guards verified by unit tests.

Part A — is_browser_available()
    Returns False when playwright is not installed and True when it is,
    verified by unit tests that mock the import.

Part B — _require_playwright() module-level import guard
    Raises a clear ImportError with install hint when playwright is not
    installed, verified by mocking the import and asserting the error message.

Strategy
--------
Python's import system consults ``sys.modules`` before attempting a real
filesystem lookup.  Two special states are used here:

* ``sys.modules["playwright"] = MagicMock()``   — import succeeds → True / no-op
* ``sys.modules["playwright"] = None``           — Python raises ImportError
                                                   for None-valued entries → False / ImportError

``patch.dict`` restores the original state after each test, making the tests
hermetic regardless of whether Playwright is actually installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from articula._browser import is_browser_available

# ---------------------------------------------------------------------------
# Sub-AC 3 — core boolean behaviour
# ---------------------------------------------------------------------------


class TestIsBrowserAvailable:
    """Runtime guard for Playwright availability."""

    def test_returns_true_when_playwright_is_installed(self) -> None:
        """When playwright resolves from sys.modules, is_browser_available() is True."""
        mock_playwright = MagicMock()
        with patch.dict(sys.modules, {"playwright": mock_playwright}):
            result = is_browser_available()
        assert result is True

    def test_returns_false_when_playwright_is_not_installed(self) -> None:
        """When playwright is blocked in sys.modules (None), returns False."""
        with patch.dict(sys.modules, {"playwright": None}):
            result = is_browser_available()
        assert result is False

    def test_return_type_is_bool_when_available(self) -> None:
        """Return value is strictly bool True, not just truthy."""
        mock_playwright = MagicMock()
        with patch.dict(sys.modules, {"playwright": mock_playwright}):
            result = is_browser_available()
        assert type(result) is bool
        assert result is True

    def test_return_type_is_bool_when_unavailable(self) -> None:
        """Return value is strictly bool False, not just falsy."""
        with patch.dict(sys.modules, {"playwright": None}):
            result = is_browser_available()
        assert type(result) is bool
        assert result is False

    def test_callable_multiple_times_consistent_result(self) -> None:
        """Repeated calls under the same mock return a consistent result."""
        mock_playwright = MagicMock()
        with patch.dict(sys.modules, {"playwright": mock_playwright}):
            first = is_browser_available()
            second = is_browser_available()
        assert first is True
        assert second is True

    def test_callable_multiple_times_consistent_false(self) -> None:
        """Repeated calls while unavailable all return False."""
        with patch.dict(sys.modules, {"playwright": None}):
            results = [is_browser_available() for _ in range(3)]
        assert all(r is False for r in results)


# ---------------------------------------------------------------------------
# Public API — is_browser_available exported from the top-level package
# ---------------------------------------------------------------------------


class TestIsBrowserAvailablePublicExport:
    """is_browser_available must be reachable from the top-level package."""

    def test_importable_from_package_root(self) -> None:
        from articula import is_browser_available as guard  # noqa: F401

        assert callable(guard)

    def test_package_root_export_is_same_object(self) -> None:
        from articula import is_browser_available as pkg_guard
        from articula._browser import is_browser_available as mod_guard

        assert pkg_guard is mod_guard

    def test_in_dunder_all(self) -> None:
        import articula

        assert "is_browser_available" in articula.__all__

    def test_package_export_returns_bool(self) -> None:
        """The package-level re-export also returns a strict bool."""
        from articula import is_browser_available as guard

        mock_playwright = MagicMock()
        with patch.dict(sys.modules, {"playwright": mock_playwright}):
            result = guard()
        assert type(result) is bool


# ---------------------------------------------------------------------------
# Sub-AC 3 — _require_playwright() module-level import guard
# ---------------------------------------------------------------------------


class TestRequirePlaywrightImportGuard:
    """Module-level import guard raises ImportError with install hint when Playwright absent.

    Sub-AC 3 requirement:
        A module-level import guard in the browser escalation module raises a
        clear ImportError with install hint when playwright is not installed,
        verified by mocking the import and asserting the error message.
    """

    def test_raises_import_error_when_playwright_not_installed(self) -> None:
        """_require_playwright() raises ImportError when playwright is absent."""
        from articula._browser import _require_playwright

        with patch.dict(sys.modules, {"playwright": None}), pytest.raises(ImportError):
            _require_playwright()

    def test_error_message_contains_install_hint(self) -> None:
        """The ImportError message contains the pip install command."""
        from articula._browser import _require_playwright

        with patch.dict(sys.modules, {"playwright": None}):
            with pytest.raises(ImportError) as exc_info:
                _require_playwright()

        error_msg = str(exc_info.value)
        assert "pip install 'articula[browser]'" in error_msg

    def test_error_message_contains_playwright_install_command(self) -> None:
        """The ImportError message includes the playwright browser install step."""
        from articula._browser import _require_playwright

        with patch.dict(sys.modules, {"playwright": None}):
            with pytest.raises(ImportError) as exc_info:
                _require_playwright()

        assert "playwright install chromium" in str(exc_info.value)

    def test_error_message_contains_package_name(self) -> None:
        """The ImportError message names the package extra for clear attribution."""
        from articula._browser import _require_playwright

        with patch.dict(sys.modules, {"playwright": None}):
            with pytest.raises(ImportError) as exc_info:
                _require_playwright()

        assert "articula[browser]" in str(exc_info.value)

    def test_raised_exception_is_exactly_import_error(self) -> None:
        """The raised exception is type ImportError (not a subclass)."""
        from articula._browser import _require_playwright

        with patch.dict(sys.modules, {"playwright": None}):
            try:
                _require_playwright()
                pytest.fail("Expected ImportError to be raised")
            except ImportError as exc:
                assert type(exc) is ImportError

    def test_does_not_raise_when_playwright_available(self) -> None:
        """_require_playwright() is a no-op when playwright resolves successfully."""
        from articula._browser import _require_playwright

        mock_playwright = MagicMock()
        with patch.dict(sys.modules, {"playwright": mock_playwright}):
            _require_playwright()  # must not raise

    def test_guard_is_callable_at_module_level(self) -> None:
        """_require_playwright is accessible as a module-level attribute."""
        import articula._browser as browser_mod

        assert callable(browser_mod._require_playwright)

    def test_install_hint_constant_present(self) -> None:
        """_PLAYWRIGHT_INSTALL_HINT constant is defined at module level."""
        import articula._browser as browser_mod

        assert hasattr(browser_mod, "_PLAYWRIGHT_INSTALL_HINT")
        hint = browser_mod._PLAYWRIGHT_INSTALL_HINT
        assert isinstance(hint, str)
        assert "articula[browser]" in hint
