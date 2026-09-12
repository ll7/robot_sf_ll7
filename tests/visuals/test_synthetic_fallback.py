"""Test synthetic fallback path (T051).

Forces SimulationView unavailable and ensures auto mode produces synthetic video artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visual_constants import RENDERER_SYNTHETIC
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class Cfg:
    """Config stub for the auto renderer with videos enabled and replay capture off.

    Disabling replay capture keeps the single record on the synthetic path instead
    of reclassifying it as an insufficient-replay SimulationView skip.
    """

    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = False  # avoid reclassification to sim-view insufficient replay
    video_renderer = "auto"


def test_synthetic_fallback_when_sim_view_unavailable(tmp_path: Path):
    # Force SimulationView unavailable
    """Assert an unavailable SimulationView yields one synthetic video artifact.

    Forces the module SimulationView availability flag false and passes a record
    without replay data; the single generated video is classified with the
    synthetic renderer, not a native SimulationView render.

    Args:
        tmp_path: Directory receiving the generated visual artifacts.
    """
    visuals_mod._SIM_VIEW_AVAILABLE = False  # type: ignore[attr-defined]
    records = [
        {"episode_id": "ep1", "scenario_id": "sc1"},
    ]
    groups: list = []
    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    vids = out["videos"]
    assert len(vids) == 1
    v0 = vids[0]
    renderer = v0.get("renderer") if isinstance(v0, dict) else getattr(v0, "renderer", None)
    assert renderer == RENDERER_SYNTHETIC, v0
