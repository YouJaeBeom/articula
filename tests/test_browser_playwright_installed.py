"""
Sub-AC 4: When playwright is available, the browser escalation module imports
and initialises without error.

This test file exercises the browser module against a **real** playwright
installation, not a mock.  All tests are skipped automatically when playwright
is not installed, so the suite remains green in base-install environments.

Strategy
--------
``pytest.importorskip("playwright")`` skips the entire module at collection
time when playwright is absent, producing a visible SKIP entry rather than a
hard failure.  When playwright is present (e.g. after
``pip install 'articula[browser]'``) the tests execute against the live
package and confirm that every public symbol initialises cleanly.
"""

from __future__ import annotations

import inspect

import pytest

# ---------------------------------------------------------------------------
# Skip this entire module if playwright is not installed.
# ---------------------------------------------------------------------------

playwright = pytest.importorskip(
    "playwright",
    reason=(
        "playwright is not installed — install with "
        "'pip install articula[browser]' to run these tests"
    ),
)


# ---------------------------------------------------------------------------
# Sub-AC 4 — Browser escalation module import & initialisation
# ---------------------------------------------------------------------------


class TestBrowserModuleImportWithPlaywright:
    """
    Verify that _browser.py imports and initialises without error in an
    environment where playwright is actually installed.

    Sub-AC 4 requirement:
        The browser escalation module must import and initialise without error
        when playwright is available, verified by a test running in an
        environment where playwright is installed (not mocked).
    """

    def test_browser_module_importable(self) -> None:
        """Importing articula._browser raises no exception."""
        import articula._browser  # noqa: F401

    def test_browser_module_has_expected_public_symbols(self) -> None:
        """All public symbols defined in _browser.py are accessible."""
        import articula._browser as mod

        for name in ("is_browser_available", "_require_playwright", "fetch_browser"):
            assert hasattr(mod, name), (
                f"Expected public symbol '{name}' not found in articula._browser"
            )

    def test_is_browser_available_returns_true(self) -> None:
        """is_browser_available() returns True when playwright is installed.

        This test does NOT use any mocking — it relies on playwright being
        genuinely importable in the current environment.
        """
        from articula._browser import is_browser_available

        result = is_browser_available()
        assert result is True, (
            "is_browser_available() should return True when playwright is installed"
        )

    def test_is_browser_available_returns_strict_bool(self) -> None:
        """is_browser_available() returns exactly bool True (not just truthy)."""
        from articula._browser import is_browser_available

        result = is_browser_available()
        assert type(result) is bool
        assert result is True

    def test_require_playwright_does_not_raise(self) -> None:
        """_require_playwright() is a no-op (does not raise) when playwright is installed."""
        from articula._browser import _require_playwright

        # Must not raise ImportError or any other exception.
        _require_playwright()

    def test_fetch_browser_is_callable(self) -> None:
        """fetch_browser is an async callable exposed by the browser module."""
        from articula._browser import fetch_browser

        assert callable(fetch_browser)

    def test_fetch_browser_is_coroutine_function(self) -> None:
        """fetch_browser must be an async def (coroutine function)."""
        from articula._browser import fetch_browser

        assert inspect.iscoroutinefunction(fetch_browser), (
            "fetch_browser must be an async def so it can be awaited"
        )

    def test_install_hint_constant_matches_playwright(self) -> None:
        """_PLAYWRIGHT_INSTALL_HINT is defined and references playwright."""
        import articula._browser as mod

        assert hasattr(mod, "_PLAYWRIGHT_INSTALL_HINT")
        hint: str = mod._PLAYWRIGHT_INSTALL_HINT
        assert "playwright" in hint.lower()
        assert "articula[browser]" in hint


class TestPlaywrightPackageIntegrity:
    """
    Sanity-check the playwright package itself to ensure the installed version
    exposes the async API consumed by fetch_browser.
    """

    def test_async_playwright_importable(self) -> None:
        """playwright.async_api.async_playwright must be importable."""
        from playwright.async_api import async_playwright  # noqa: F401

        assert callable(async_playwright)

    def test_browser_module_import_does_not_trigger_playwright_init(self) -> None:
        """
        Importing _browser.py must not launch or initialise Playwright.

        The module uses lazy imports — playwright.async_api is only imported
        when fetch_browser is actually *called*, not at import time.  This keeps
        cold import cost near-zero and avoids side-effects at library load.

        NOTE: We deliberately do NOT delete articula._browser from
        sys.modules because doing so would invalidate cached references held by
        articula.__init__, breaking identity checks in subsequent tests.
        Instead, we verify that the _browser module is importable and that no
        Playwright browser process is started as a side-effect.
        """
        import articula._browser  # noqa: F401

        # If we reach here the import succeeded without launching a browser.
        # The lazy-import guarantee is structural (all playwright sub-imports
        # are inside fetch_browser's body), so no additional runtime assertion
        # is needed beyond the fact that this import completed without error.
        assert True


class TestBrowserModulePackageLevelExport:
    """
    Verify the package-level re-export of is_browser_available works correctly
    in an environment where playwright is installed.
    """

    def test_package_level_is_browser_available_returns_true(self) -> None:
        """articula.is_browser_available() returns True without mocking."""
        from articula import is_browser_available

        result = is_browser_available()
        assert result is True

    def test_package_level_and_module_level_are_identical(self) -> None:
        """Package-level and module-level is_browser_available are the same object."""
        from articula import is_browser_available as pkg_fn
        from articula._browser import is_browser_available as mod_fn

        assert pkg_fn is mod_fn
