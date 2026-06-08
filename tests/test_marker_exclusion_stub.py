"""
Non-live stub test for Sub-AC 6b-ii marker exclusion verification.

This module contains exactly one non-live test that trivially passes.
It serves as the "known-good offline test" referenced by the marker
exclusion run-verification suite in ``test_live_marker_registration.py``
(class ``TestMarkerExclusionRunVerification``).

Purpose
-------
Sub-AC 6b-ii requires proving that ``pytest -m 'not live'`` correctly:
  1. Deselects/skips all ``@pytest.mark.live`` tests.
  2. Still collects and runs (and passes) non-live tests.

The verification test runs::

    pytest -m 'not live' tests/test_live_smoke.py
                          tests/test_marker_exclusion_stub.py

- ``test_live_smoke.py`` supplies the live-marked tests (all deselected).
- This file supplies the offline stub (must run and pass).

Design constraints
------------------
- **No** ``@pytest.mark.live`` decorator on any test in this file.
- No live network calls — the test is deliberately trivial.
- Importable without any heavy optional dependencies.
"""

from __future__ import annotations


def test_stub_offline_always_passes() -> None:
    """
    Offline stub: trivially passes with no side effects.

    This test is intentionally minimal — its only purpose is to be
    collected and executed by the Sub-AC 6b-ii verification run
    (``pytest -m 'not live'``) as proof that non-live tests are *not*
    excluded by the marker filter.
    """
    assert True, "stub test: always passes, requires no network access"
