"""Test insufficient replay state skip logic (T023).

Relies on capture_replay flag to enable adapter pathway; constructs a record with
only a single replay step so it fails validation (min_length=2) and expects a
skipped artifact with note == "insufficient-replay-state" when SimulationView
is available. If SimulationView cannot be imported the test will be skipped at
collection time by a dynamic condition inside the test body.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.full_classic.visual_constants import (
    NOTE_INSUFFICIENT_REPLAY,
    RENDERER_SIM_VIEW,
)
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class DummyCfg:
    """DummyCfg class."""

    output_root = ".tmp_test_visuals_insufficient"
    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = True


def test_insufficient_replay_skip(tmp_path: Path):
    """Test insufficient replay skip.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Skip dynamically if SimulationView truly unavailable
    try:
        importlib.import_module("robot_sf.render.sim_view")
    except Exception:
        pytest.skip("SimulationView not importable; skip insufficient replay test")

    records = [
        {
            "episode_id": "scA-1",
            "scenario_id": "scA",
            "replay_steps": [(0.0, 0.0, 0.0, 0.0)],  # only one step
        },
    ]
    groups = []
    out = generate_visual_artifacts(tmp_path, DummyCfg, groups, records)
    vids = out["videos"]
    assert len(vids) == 1
    v0 = vids[0]
    renderer = v0.get("renderer") if isinstance(v0, dict) else getattr(v0, "renderer", None)
    note = v0.get("note") if isinstance(v0, dict) else getattr(v0, "note", None)
    status = v0.get("status") if isinstance(v0, dict) else getattr(v0, "status", None)
    assert renderer == RENDERER_SIM_VIEW
    assert note == NOTE_INSUFFICIENT_REPLAY
    assert status == "skipped"
