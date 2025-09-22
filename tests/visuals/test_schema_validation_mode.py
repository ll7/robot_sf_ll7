"""Test schema validation mode (T055).

Sets ROBOT_SF_VALIDATE_VISUALS=1 and monkeypatches validation to raise after manifests written.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts


class Cfg:
    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = False
    video_renderer = "synthetic"


def test_schema_validation_mode_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ROBOT_SF_VALIDATE_VISUALS", "1")

    # Force validation to raise
    def fake_validate(*_a, **_k):  # noqa: D401
        raise RuntimeError("forced-error")

    # Patch the imported symbol inside visuals module
    monkeypatch.setattr(visuals_mod, "validate_visual_manifests", fake_validate)
    records = [
        {"episode_id": "ep1", "scenario_id": "sc1"},
    ]
    groups: list = []
    with pytest.raises(RuntimeError):
        generate_visual_artifacts(tmp_path, Cfg, groups, records)
    os.environ.pop("ROBOT_SF_VALIDATE_VISUALS", None)
