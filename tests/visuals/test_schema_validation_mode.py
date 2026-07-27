"""Test schema validation mode (T055).

Sets ROBOT_SF_VALIDATE_VISUALS=1 and monkeypatches validation to raise after manifests written.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class Cfg:
    """TODO docstring. Document this class."""

    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = False
    video_renderer = "synthetic"


def test_schema_validation_mode_raises(tmp_path: Path, monkeypatch):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        monkeypatch: TODO docstring.
    """
    monkeypatch.setenv("ROBOT_SF_VALIDATE_VISUALS", "1")

    # Force validation to raise
    def fake_validate(*_a, **_k):
        """TODO docstring. Document this function.

        Args:
            _a: TODO docstring.
            _k: TODO docstring.
        """
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


@pytest.mark.parametrize("validation_errors", [[], ["invalid plot_artifacts.json"]])
def test_schema_validation_mode_reports_returned_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_errors: list[str]
) -> None:
    """Returned validation errors are reported without aborting visual generation."""
    monkeypatch.setenv("ROBOT_SF_VALIDATE_VISUALS", "1")
    monkeypatch.setattr(
        visuals_mod,
        "validate_visual_manifests",
        lambda *_args, **_kwargs: validation_errors,
    )
    records = [{"episode_id": "ep1", "scenario_id": "sc1"}]

    artifacts = generate_visual_artifacts(tmp_path, Cfg, [], records)

    assert artifacts.keys() == {"plots", "videos", "performance"}
