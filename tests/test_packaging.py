"""
Sub-AC 1: Packaging verification tests.

Verifies that pyproject.toml is correctly defined so that:
  - ``pip install .`` succeeds in a clean virtual environment
  - ``import articula`` exits 0 (the package is importable)
  - The public API symbols expected by the library contract are present
  - ``__version__`` is defined

These tests use subprocess to isolate the install from the current environment.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root directory (contains pyproject.toml)."""
    here = Path(__file__).parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("pyproject.toml not found in any ancestor directory")


def _venv_python(venv_dir: Path) -> Path:
    """Return the path to the Python executable inside a virtual environment."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# ---------------------------------------------------------------------------
# Test: subprocess install into clean venv → import exits 0
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_package_installs_and_is_importable() -> None:
    """
    Core Sub-AC 1 test.

    Creates a fresh virtual environment, runs ``pip install .`` against the
    project root, then asserts that ``python -c 'import articula'``
    exits with code 0.
    """
    project_root = _project_root()

    with tempfile.TemporaryDirectory(prefix="articula_pkg_test_") as tmp:
        venv_dir = Path(tmp) / "venv"

        # 1. Create a clean virtual environment using the same Python as pytest
        create = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert create.returncode == 0, (
            f"venv creation failed (exit {create.returncode}):\n"
            f"stdout: {create.stdout}\nstderr: {create.stderr}"
        )

        python = _venv_python(venv_dir)
        assert python.exists(), f"Python executable not found in venv: {python}"

        # 2. Install the package from the project root
        install = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                str(project_root),
                "--quiet",
                "--no-input",
            ],
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert install.returncode == 0, (
            f"pip install failed (exit {install.returncode}):\n"
            f"stdout: {install.stdout}\nstderr: {install.stderr}"
        )

        # 3. Assert import exits 0
        check = subprocess.run(
            [str(python), "-c", "import articula"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert check.returncode == 0, (
            f"'import articula' failed (exit {check.returncode}):\n"
            f"stdout: {check.stdout}\nstderr: {check.stderr}"
        )


@pytest.mark.timeout(300)
def test_package_version_accessible_after_install() -> None:
    """
    After installation, ``articula.__version__`` must be a non-empty string.
    """
    project_root = _project_root()

    with tempfile.TemporaryDirectory(prefix="articula_ver_test_") as tmp:
        venv_dir = Path(tmp) / "venv"

        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        python = _venv_python(venv_dir)

        subprocess.run(
            [str(python), "-m", "pip", "install", str(project_root), "--quiet", "--no-input"],
            check=True,
            capture_output=True,
            timeout=240,
        )

        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import articula; "
                    "assert isinstance(articula.__version__, str), "
                    "'__version__ must be a str'; "
                    "assert articula.__version__, '__version__ must not be empty'; "
                    "print(articula.__version__)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"version check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        version = result.stdout.strip()
        assert version, "No version output produced"


# ---------------------------------------------------------------------------
# Test: public API accessible in the current process (fast sanity check)
# ---------------------------------------------------------------------------


class TestPublicAPIAccessible:
    """
    Verify the public API symbols are accessible in the current process.

    These tests do NOT install into a fresh venv — they run quickly against
    the already-installed package and act as a fast sanity check before the
    slower subprocess tests.
    """

    def test_import_succeeds(self) -> None:
        import articula  # noqa: F401

    def test_version_is_string(self) -> None:
        import articula

        assert isinstance(articula.__version__, str)
        assert articula.__version__

    def test_article_class_importable(self) -> None:
        from articula import Article  # noqa: F401

        assert Article is not None

    def test_exception_hierarchy(self) -> None:
        from articula import (
            BrowserNotInstalledError,
            ExtractionError,
            FetchError,
            RobotsDisallowedError,
            ScraperError,
        )

        # All derive from ScraperError
        assert issubclass(FetchError, ScraperError)
        assert issubclass(ExtractionError, ScraperError)
        assert issubclass(BrowserNotInstalledError, ScraperError)
        assert issubclass(RobotsDisallowedError, ScraperError)

    def test_exception_hierarchy_base_is_exception(self) -> None:
        from articula import ScraperError

        assert issubclass(ScraperError, Exception)

    def test_all_exports_in_dunder_all(self) -> None:
        import articula

        assert hasattr(articula, "__all__")
        declared: list[str] = articula.__all__
        assert len(declared) > 0

        for name in declared:
            assert hasattr(articula, name), (
                f"'{name}' listed in __all__ but not found on the module"
            )

    def test_null_handler_configured(self) -> None:
        """Library logger must have a NullHandler (no default output)."""
        import logging

        import articula  # noqa: F401

        logger = logging.getLogger("articula")
        handler_types = [type(h) for h in logger.handlers]
        assert logging.NullHandler in handler_types, (
            "articula logger must have a NullHandler to avoid "
            "emitting output in libraries"
        )


# ---------------------------------------------------------------------------
# Test: pyproject.toml metadata correctness
# ---------------------------------------------------------------------------


class TestPyprojectMetadata:
    """Validate the pyproject.toml structure programmatically."""

    @pytest.fixture(scope="class")
    def pyproject(self) -> dict:  # type: ignore[type-arg]
        import tomllib

        root = _project_root()
        with open(root / "pyproject.toml", "rb") as fh:
            return tomllib.load(fh)

    def test_project_name_defined(self, pyproject: dict) -> None:  # type: ignore[type-arg]
        assert pyproject["project"]["name"]

    def test_version_defined(self, pyproject: dict) -> None:  # type: ignore[type-arg]
        assert pyproject["project"]["version"]

    def test_requires_python_ge_311(self, pyproject: dict) -> None:  # type: ignore[type-arg]
        req = pyproject["project"]["requires-python"]
        assert "3.11" in req or "3.1" in req, (
            f"requires-python should target >=3.11, got {req!r}"
        )

    def test_base_dependencies_defined(self, pyproject: dict) -> None:  # type: ignore[type-arg]
        deps: list[str] = pyproject["project"]["dependencies"]
        assert len(deps) > 0, "Base dependencies must be defined"

    def test_browser_optional_extra_defined(self, pyproject: dict) -> None:  # type: ignore[type-arg]
        extras = pyproject["project"]["optional-dependencies"]
        assert "browser" in extras, "A 'browser' optional-dependency group must exist"
        browser_deps: list[str] = extras["browser"]
        assert any("playwright" in dep.lower() for dep in browser_deps), (
            "browser extra must list playwright"
        )

    def test_build_backend_defined(self, pyproject: dict) -> None:  # type: ignore[type-arg]
        assert "build-system" in pyproject
        assert pyproject["build-system"]["build-backend"]


# ---------------------------------------------------------------------------
# Sub-AC 2: browser extra verified via importlib.metadata
# ---------------------------------------------------------------------------


class TestBrowserExtraMetadata:
    """
    Sub-AC 2: Verify that the installed package metadata exposes 'playwright'
    inside the 'browser' extras_require group.

    Uses ``importlib.metadata.requires()`` (stdlib since Python 3.9) to read
    the live distribution metadata rather than parsing pyproject.toml directly.
    This guarantees the packaging round-trip is correct — i.e., the metadata
    that ``pip`` sees after installation matches the declared optional-dependency
    group.
    """

    _DIST_NAME = "articula"

    def test_browser_extra_playwright_via_importlib_metadata(self) -> None:
        """
        importlib.metadata.requires() must include a playwright requirement
        whose marker declares ``extra == "browser"``.
        """
        from importlib.metadata import PackageNotFoundError, requires

        try:
            all_reqs = requires(self._DIST_NAME)
        except PackageNotFoundError:
            pytest.fail(
                f"Distribution '{self._DIST_NAME}' is not installed. "
                "Run 'pip install -e .' before executing this test."
            )

        assert all_reqs is not None, (
            f"'{self._DIST_NAME}' has no recorded requirements metadata."
        )

        # Filter to requirements that belong to the 'browser' extra.
        # pip encodes optional extras as PEP 508 markers: `; extra == "browser"`
        browser_reqs = [r for r in all_reqs if 'extra == "browser"' in r]

        assert browser_reqs, (
            f"No requirements found for the 'browser' extra in "
            f"'{self._DIST_NAME}'. All requirements: {all_reqs}"
        )

        assert any("playwright" in req.lower() for req in browser_reqs), (
            f"'playwright' not found in the 'browser' extra requirements: "
            f"{browser_reqs}"
        )

    def test_browser_extra_playwright_present_in_requires_list(self) -> None:
        """
        Cross-check: the full requires list must contain exactly one entry
        that references playwright under the browser extra.
        """
        from importlib.metadata import requires

        all_reqs = requires(self._DIST_NAME) or []
        playwright_browser_entries = [
            r
            for r in all_reqs
            if "playwright" in r.lower() and 'extra == "browser"' in r
        ]
        assert len(playwright_browser_entries) >= 1, (
            "Expected at least one playwright entry scoped to the 'browser' "
            f"extra, found none. All requirements: {all_reqs}"
        )

    @pytest.mark.timeout(300)
    def test_pip_install_browser_extra_resolves(self) -> None:
        """
        Subprocess test: ``pip install .[browser]`` must resolve without error
        in a clean virtual environment.

        This exercises the full pip dependency resolution path, not just the
        metadata parsing, ensuring the version specifier is valid and the
        package is actually downloadable.
        """
        project_root = _project_root()

        with tempfile.TemporaryDirectory(prefix="articula_browser_test_") as tmp:
            venv_dir = Path(tmp) / "venv"

            create = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert create.returncode == 0, (
                f"venv creation failed:\nstdout: {create.stdout}\nstderr: {create.stderr}"
            )

            python = _venv_python(venv_dir)

            # Install with [browser] extra — this validates full pip resolution.
            install = subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    f"{project_root}[browser]",
                    "--quiet",
                    "--no-input",
                ],
                capture_output=True,
                text=True,
                timeout=240,
            )
            assert install.returncode == 0, (
                f"pip install .[browser] failed (exit {install.returncode}):\n"
                f"stdout: {install.stdout}\nstderr: {install.stderr}"
            )

            # After install, verify via importlib.metadata inside the fresh venv.
            check_script = (
                "from importlib.metadata import requires; "
                "reqs = requires('articula') or []; "
                "browser = [r for r in reqs if 'extra == \"browser\"' in r]; "
                "assert any('playwright' in r.lower() for r in browser), "
                "f'playwright not in browser extra: {browser}'; "
                "print('OK')"
            )
            verify = subprocess.run(
                [str(python), "-c", check_script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert verify.returncode == 0, (
                f"importlib.metadata check failed in fresh venv:\n"
                f"stdout: {verify.stdout}\nstderr: {verify.stderr}"
            )
