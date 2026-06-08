"""
Sub-AC 6a: Verify that ``@pytest.mark.live`` is registered and that
running ``pytest -m 'not live'`` collects zero live-marked tests.

Sub-AC 6b-ii: Prove the marker exclusion mechanism works by *running*
(not just collecting) ``pytest -m 'not live'`` against a known two-file
fixture and observing that live tests are deselected while the offline
stub test passes.

Strategy
--------
Each test in this module uses ``subprocess`` to invoke pytest's
``--collect-only`` mode with a specific marker expression, parses the
collected test-node IDs from stdout, and makes assertions about the
resulting sets.

  live_ids       = test IDs collected by ``pytest --collect-only -m live``
  not_live_ids   = test IDs collected by ``pytest --collect-only -m 'not live'``

Key invariants (Sub-AC 6a)
--------------------------
1. live_ids ∩ not_live_ids == ∅  (mutual exclusion)
2. len(live_ids) >= 1            (at least one live test exists)
3. len(not_live_ids) >= 1        (at least one offline test exists)
4. No "PytestUnknownMarkWarning" in pytest output for ``-m live``

Key invariants (Sub-AC 6b-ii)
------------------------------
5. Running ``pytest -m 'not live'`` on the two-file fixture exits 0.
6. The non-live stub test ``test_stub_offline_always_passes`` appears as
   PASSED in the output.
7. The word "deselected" appears in the output, confirming live tests
   from ``test_live_smoke.py`` were suppressed.
8. No live test function name appears as PASSED or FAILED.

These tests are NOT marked ``@pytest.mark.live`` — they are offline
meta-tests that verify the marker infrastructure.
"""

from __future__ import annotations

import subprocess
import sys
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


def _collect_test_ids(marker_expr: str) -> frozenset[str]:
    """
    Run ``pytest --collect-only -q -q -m <marker_expr>`` and return the set of
    collected test-node IDs.

    Using ``-q -q`` (double-quiet) makes pytest emit one node ID per line in
    flat format:
        tests/test_foo.py::TestClass::test_method
    Lines without ``::`` are count summaries (e.g. "5 tests collected in 0.1s")
    and are ignored.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-q",
            "--no-header",
            "--tb=no",
            "-p",
            "no:warnings",
            "-m",
            marker_expr,
        ],
        capture_output=True,
        text=True,
        cwd=str(_project_root()),
        timeout=120,
    )
    ids: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # Node IDs always contain "::" and are never summary/warning lines
        if "::" in stripped and not stripped.startswith(("=", "E ", "W ", "ERRORS")):
            ids.add(stripped)
    return frozenset(ids)


# ---------------------------------------------------------------------------
# Sub-AC 6a tests
# ---------------------------------------------------------------------------


class TestLiveMarkerRegistration:
    """
    Verify that the ``live`` pytest marker is properly registered and that
    the marker-based collection produces mutually exclusive test sets.
    """

    def test_live_marker_produces_no_unknown_mark_warning(self) -> None:
        """
        Running ``pytest --collect-only -m live`` must NOT emit a
        PytestUnknownMarkWarning.  If the marker is unregistered, pytest
        warns about it in stderr/stdout.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--no-header",
                "--tb=no",
                "-m",
                "live",
            ],
            capture_output=True,
            text=True,
            cwd=str(_project_root()),
            timeout=120,
        )
        combined = result.stdout + result.stderr
        assert "PytestUnknownMarkWarning" not in combined, (
            "The 'live' marker is not registered — pytest emitted "
            "PytestUnknownMarkWarning.  Register it in pyproject.toml "
            "or conftest.py."
        )
        assert "Unknown pytest.mark.live" not in combined, (
            "Unexpected 'unknown mark' message for 'live' in pytest output."
        )

    def test_live_tests_exist_in_suite(self) -> None:
        """
        At least one test must be decorated with ``@pytest.mark.live``.
        Without any live tests the marker infrastructure cannot be validated.
        """
        live_ids = _collect_test_ids("live")
        assert len(live_ids) >= 1, (
            "No @pytest.mark.live tests were collected.  "
            "Add at least one @pytest.mark.live test (e.g. in test_live_smoke.py) "
            "so the marker infrastructure can be exercised."
        )

    def test_not_live_collects_at_least_some_tests(self) -> None:
        """
        ``pytest -m 'not live'`` must collect a non-empty set of tests — the
        offline unit-test suite must still be present.
        """
        not_live_ids = _collect_test_ids("not live")
        assert len(not_live_ids) >= 1, (
            "pytest -m 'not live' collected zero tests.  "
            "The offline test suite appears to be missing."
        )

    def test_live_and_not_live_sets_are_disjoint(self) -> None:
        """
        Core invariant: the set of IDs collected by ``-m live`` and the set
        collected by ``-m 'not live'`` must be completely disjoint.

        Any test ID that appears in both sets would mean a test has
        ``@pytest.mark.live`` but is still collected by the 'not live' filter,
        which would violate the contract.
        """
        live_ids = _collect_test_ids("live")
        not_live_ids = _collect_test_ids("not live")

        overlap = live_ids & not_live_ids
        assert not overlap, (
            "The following test IDs appear in BOTH the 'live' and "
            "'not live' collections — they should be mutually exclusive:\n"
            + "\n".join(f"  {tid}" for tid in sorted(overlap))
        )

    def test_not_live_excludes_every_live_test_id(self) -> None:
        """
        For each test ID returned by ``-m live``, verify it does NOT appear
        in the IDs returned by ``-m 'not live'``.

        This is a per-item version of test_live_and_not_live_sets_are_disjoint
        that produces a clearer failure message.
        """
        live_ids = _collect_test_ids("live")
        not_live_ids = _collect_test_ids("not live")

        leaked: list[str] = [tid for tid in sorted(live_ids) if tid in not_live_ids]
        assert not leaked, (
            "The following @pytest.mark.live tests were also collected by "
            "'pytest -m not live', meaning they are NOT being excluded:\n"
            + "\n".join(f"  {tid}" for tid in leaked)
        )

    def test_live_ids_contain_only_live_smoke_module_or_known_live_modules(
        self,
    ) -> None:
        """
        All collected live test IDs should come from a module (file) that is
        intentionally live — specifically ``test_live_smoke.py`` or any file
        that explicitly contains only ``@pytest.mark.live`` tests.

        This guards against accidentally marking a unit test as live.
        """
        live_ids = _collect_test_ids("live")
        for tid in live_ids:
            # A test_id looks like: tests/test_live_smoke.py::fn_name
            parts = tid.split("::")
            module_path = parts[0] if parts else ""
            # Accept any module whose filename contains 'live'
            module_file = Path(module_path).name
            assert "live" in module_file.lower(), (
                f"Live-marked test {tid!r} is defined in module "
                f"{module_file!r} which does not have 'live' in its name.  "
                "Only dedicate live-test files (e.g. test_live_*.py) should "
                "contain @pytest.mark.live tests."
            )


# ---------------------------------------------------------------------------
# Sub-AC 6b-ii — run verification (not just collect)
# ---------------------------------------------------------------------------


class TestMarkerExclusionRunVerification:
    """
    Sub-AC 6b-ii: Prove the marker exclusion mechanism works by *running*
    ``pytest -m 'not live'`` against a known two-file fixture and observing:

    1. Live tests from ``test_live_smoke.py`` are deselected (not run).
    2. The non-live stub from ``test_marker_exclusion_stub.py`` runs and passes.

    Unlike ``TestLiveMarkerRegistration`` — which uses ``--collect-only``
    to verify registration — this class invokes a real pytest *run* and
    checks the exit code and output to prove end-to-end exclusion.

    The subprocess is scoped to two specific files to avoid infinite
    recursion: the subprocess never re-invokes this test module.
    """

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _run_not_live_on_fixture() -> subprocess.CompletedProcess[str]:  # type: ignore[type-arg]
        """
        Run ``pytest -m 'not live'`` on the two-file fixture:
          - ``tests/test_live_smoke.py``        (all tests live-marked → deselected)
          - ``tests/test_marker_exclusion_stub.py``  (one non-live stub → runs)

        Scoping to these two files prevents the subprocess from re-invoking
        this verification test and avoids unbounded recursion.
        """
        root = _project_root()
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "not live",
                "--no-header",
                "--tb=short",
                "-v",
                "tests/test_live_smoke.py",
                "tests/test_marker_exclusion_stub.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=120,
        )

    # ---- Sub-AC 6b-ii tests ------------------------------------------------

    def test_not_live_run_exits_zero(self) -> None:
        """
        ``pytest -m 'not live'`` must exit with code 0.

        Exit code 0 means all collected tests passed.  Since the only
        collected test is the offline stub (which trivially passes), code 0
        proves the stub was found, executed, and succeeded.
        """
        result = self._run_not_live_on_fixture()
        assert result.returncode == 0, (
            f"pytest -m 'not live' exited {result.returncode} — expected 0.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_not_live_run_shows_stub_as_passed(self) -> None:
        """
        The non-live stub test ``test_stub_offline_always_passes`` must
        appear in the verbose output, confirming it was collected and run.
        """
        result = self._run_not_live_on_fixture()
        combined = result.stdout + result.stderr
        assert "test_stub_offline_always_passes" in combined, (
            "Stub test 'test_stub_offline_always_passes' not found in output — "
            "it was not collected or executed by -m 'not live'.\n"
            f"STDOUT:\n{result.stdout}"
        )

    def test_not_live_run_reports_deselected(self) -> None:
        """
        The word "deselected" must appear in the pytest output.

        pytest reports ``N deselected`` when tests exist in the collected
        files but are excluded by the marker filter.  Its presence confirms
        that the live tests from ``test_live_smoke.py`` were recognised and
        suppressed by ``-m 'not live'``.
        """
        result = self._run_not_live_on_fixture()
        combined = result.stdout + result.stderr
        assert "deselected" in combined.lower(), (
            "Expected 'deselected' in pytest output — live tests from "
            "test_live_smoke.py should have been deselected, not skipped or run.\n"
            f"STDOUT:\n{result.stdout}"
        )

    def test_not_live_run_does_not_execute_live_tests(self) -> None:
        """
        No live test from ``test_live_smoke.py`` must appear as PASSED or
        FAILED in the run output.

        If any live function name appears as PASSED/FAILED, the marker
        exclusion mechanism has broken and live network calls were made.
        """
        result = self._run_not_live_on_fixture()
        combined = result.stdout + result.stderr

        live_test_names = (
            "test_live_scrape_english_wikipedia",
            "test_live_scrape_korean_wikipedia",
            "test_live_async_scrape_english",
        )
        for name in live_test_names:
            assert f"PASSED tests/test_live_smoke.py::{name}" not in combined, (
                f"Live test {name!r} was PASSED — it should have been deselected "
                "by -m 'not live'."
            )
            assert f"FAILED tests/test_live_smoke.py::{name}" not in combined, (
                f"Live test {name!r} was FAILED — it should have been deselected "
                "by -m 'not live' (no network calls expected)."
            )
