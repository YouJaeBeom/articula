"""
pytest configuration for articula tests.

Registers custom markers so that pytest never emits PytestUnknownMarkWarning
for markers declared in this suite.

Sub-AC 6a: The ``live`` marker is registered here so that
    pytest -m 'not live'
excludes all network-hitting tests and the remaining offline suite passes.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register project-level markers."""
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring live network access — "
        "excluded by default; run with '-m live' to opt in",
    )
