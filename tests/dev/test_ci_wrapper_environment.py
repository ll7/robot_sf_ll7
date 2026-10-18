"""Shell-wrapper fixtures must own environment selection while preserving CI identity."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_ci_wrapper_fixture_isolates_parent_environment_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """A matched parent override must not select a host interpreter in fixture repos."""
    carriers = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "ROBOT_SF_EXPLICIT_VENV_OVERRIDE")
    for name in carriers:
        monkeypatch.setenv(name, str(tmp_path / "parent-venv"))
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_OS", "fixture-os")

    request.getfixturevalue("isolated_ci_wrapper_environment")

    leaked = [name for name in carriers if os.environ.get(name) is not None]
    assert leaked == [], "parent environment selection leaked into a shell fixture"
    assert os.environ.get("CI") == "true"
    assert os.environ.get("GITHUB_ACTIONS") == "true"
    assert os.environ.get("RUNNER_OS") == "fixture-os"
