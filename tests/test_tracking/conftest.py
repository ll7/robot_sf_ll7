"""Shared fixtures for telemetry tracker tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.telemetry.config import RunTrackerConfig

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(name="tracker_artifact_root")
def _tracker_artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate tracker output beneath the pytest temp directory."""

    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(root))
    return root


@pytest.fixture()
def run_tracker_config(tracker_artifact_root: Path) -> RunTrackerConfig:
    """Return a tracker config pointing at the isolated root."""

    return RunTrackerConfig(artifact_root=tracker_artifact_root)
