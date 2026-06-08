"""
Sub-AC 6c: CI configuration verification.

Verifies that the ``[tool.pytest.ini_options]`` ``addopts`` setting in
``pyproject.toml`` excludes the ``live`` marker by default so that the
bare ``pytest`` command never executes live-network tests.

Key assertions
--------------
1. ``addopts`` does NOT include the bare ``-m live`` flag (which would
   *select only* live tests instead of excluding them).
2. ``addopts`` DOES include ``-m 'not live'`` (which excludes all live
   tests from the default run).

These two properties together guarantee that:
  - Running ``pytest`` without extra arguments will skip live tests.
  - The only way to run live tests is to explicitly pass ``-m live`` on
    the command line, overriding the addopts default.
"""

from __future__ import annotations

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the absolute path to the project root (contains pyproject.toml)."""
    here = Path(__file__).parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("pyproject.toml not found in any ancestor directory")


def _read_addopts() -> str:
    """
    Read the ``addopts`` value from ``[tool.pytest.ini_options]`` and
    return it as a single string.

    In TOML, ``[tool.pytest.ini_options]`` parses to
    ``data["tool"]["pytest"]["ini_options"]``.

    If the value is stored as a TOML array (list of strings), the items
    are joined with a single space so that substring checks work uniformly
    regardless of the storage format.
    """
    root = _project_root()
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)

    # [tool.pytest.ini_options] → data["tool"]["pytest"]["ini_options"]
    ini_options: dict = (
        data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    )
    addopts = ini_options.get("addopts", "")

    if isinstance(addopts, list):
        return " ".join(addopts)
    return str(addopts)


# ---------------------------------------------------------------------------
# Sub-AC 6c tests
# ---------------------------------------------------------------------------


class TestCIAddopts:
    """
    Verify that ``pyproject.toml`` ``addopts`` excludes live tests by default.
    """

    def test_addopts_key_exists(self) -> None:
        """``[tool.pytest.ini_options]`` must have an ``addopts`` entry."""
        root = _project_root()
        with open(root / "pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)

        # [tool.pytest.ini_options] → data["tool"]["pytest"]["ini_options"]
        pytest_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
        assert "addopts" in pytest_opts, (
            "No 'addopts' key found in [tool.pytest.ini_options].  "
            "Add ``addopts = \"-v --tb=short -m 'not live'\"`` to ensure "
            "live tests are excluded from the default pytest run."
        )

    def test_addopts_contains_not_live_marker(self) -> None:
        """
        The addopts string must contain ``-m 'not live'`` so that the
        default ``pytest`` invocation excludes all ``@pytest.mark.live``
        tests.
        """
        addopts = _read_addopts()
        assert "-m 'not live'" in addopts, (
            f"addopts={addopts!r} does not contain \"-m 'not live'\".  "
            "Add \"-m 'not live'\" to the addopts in "
            "[tool.pytest.ini_options] so that live tests are excluded "
            "from the default pytest run."
        )

    def test_addopts_does_not_contain_bare_m_live(self) -> None:
        """
        The addopts string must NOT contain the bare ``-m live`` flag.

        ``-m live`` would *select* only live tests — the opposite of what
        is intended.  Only ``-m 'not live'`` (or ``-m "not live"``) is
        acceptable.
        """
        addopts = _read_addopts()
        # We want to ensure there is no "-m live" token that would SELECT
        # live tests.  The string "-m 'not live'" contains the substring
        # "live" but NOT the substring "-m live", so this check is safe.
        assert "-m live" not in addopts, (
            f"addopts={addopts!r} contains \"-m live\" which would *select* "
            "live tests rather than exclude them.  Replace it with "
            "\"-m 'not live'\"."
        )
