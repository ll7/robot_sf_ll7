"""Contract tests for the optional 2.5D presentation view (issue #9369)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.render.jsonl_playback import PlaybackEpisode
from robot_sf.render.presentation_scene import (
    describe_view,
    floor_polygons_match_source,
)
from robot_sf.render.sim_state import VisualizableSimState
from robot_sf.render.threejs_viewer import build_threejs_scene, export_threejs_viewer


def _state(timestep: int) -> VisualizableSimState:
    return VisualizableSimState(
        timestep=timestep,
        robot_action=None,
        robot_pose=((1.0 + timestep, 2.0), 0.25),
        pedestrian_positions=np.array([[3.0, 4.0]]),
        ray_vecs=np.array([]),
        ped_actions=np.array([]),
    )


def _map() -> MapDefinition:
    # Asymmetric obstacle with negative coords exercises alignment paths.
    zone = ((-4.0, -2.0), (2.0, -2.0), (2.0, 2.0))
    return MapDefinition(
        width=10.0,
        height=8.0,
        obstacles=[Obstacle([(-4.0, -2.0), (-1.0, -2.0), (-1.0, 1.0), (-4.0, 1.0)])],
        robot_spawn_zones=[zone],
        robot_goal_zones=[zone],
        ped_spawn_zones=[zone],
        bounds=[
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 10.0, 8.0, 8.0),
            (0.0, 0.0, 0.0, 8.0),
            (10.0, 10.0, 0.0, 8.0),
        ],
        robot_routes=[],
        ped_goal_zones=[zone],
        ped_crowded_zones=[],
        ped_routes=[],
    )


def _scene() -> dict:
    return build_threejs_scene(
        PlaybackEpisode(episode_id=31, states=[_state(0), _state(1)]),
        _map(),
        source="synthetic.jsonl",
    )


def test_both_presets_reference_identical_source_data() -> None:
    """Camera presets share positions, footprints, IDs, and timestamps."""
    scene = _scene()
    top_down = describe_view(scene, "top-down")
    isometric = describe_view(scene, "isometric")

    assert top_down["frame_count"] == isometric["frame_count"] == 2
    assert top_down["camera"]["kind"] == isometric["camera"]["kind"] == "orthographic"
    assert top_down["camera"]["elevation_deg"] == 90.0
    assert isometric["camera"]["elevation_deg"] == 50.0
    assert isometric["camera"]["azimuth_deg"] == 45.0
    # Descriptors reference; the scene itself is untouched by either preset.
    assert scene["frames"][0]["robot"]["position"] == [1.0, 2.0]
    json.dumps(top_down)
    json.dumps(isometric)


def test_wall_volumes_keep_floor_polygons_and_disclose_height() -> None:
    """Extrusion metadata never alters floor geometry and stays symbolic."""
    scene = _scene()
    view = describe_view(scene, "isometric", wall_height_m=1.2)

    assert floor_polygons_match_source(scene) is True
    assert view["walls"]["bevel_enabled"] is False
    assert view["walls"]["height_m"] == 1.2
    assert view["walls"]["height_source"] == "illustrative"
    assert view["walls"]["floor_polygons"] == "source obstacle vertices verbatim"
    assert any("2D" in disclosure for disclosure in view["disclosures"])


def test_floor_check_rejects_missing_or_nonfinite_geometry() -> None:
    """Empty or corrupt obstacle geometry fails the floor check, never passes."""
    assert floor_polygons_match_source({"map": {"obstacles": []}}) is False
    assert (
        floor_polygons_match_source({"map": {"obstacles": [{"vertices": [[float("nan"), 0.0]]}]}})
        is False
    )


def test_invalid_preset_and_height_fail_closed() -> None:
    """Unknown presets and non-finite heights raise instead of rendering guesses."""
    scene = _scene()
    with pytest.raises(ValueError, match="unknown camera preset"):
        describe_view(scene, "cinematic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wall_height_m"):
        describe_view(scene, "isometric", wall_height_m=float("nan"))
    with pytest.raises(ValueError, match="wall_height_m"):
        describe_view(scene, "isometric", wall_height_m=-1.0)


def test_export_stages_view_block_and_components_only_when_requested(
    tmp_path,
) -> None:
    """Legacy exports stay byte-identical; view exports stage the new module."""
    recording = tmp_path / "episode.jsonl"
    recording.write_text(
        json.dumps(
            {
                "episode_id": 32,
                "step_idx": 0,
                "event": "step",
                "timestamp": 0.0,
                "state": {
                    "timestep": 0,
                    "robot_pose": [[1.0, 2.0], 0.0],
                    "pedestrian_positions": [[3.0, 4.0]],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    legacy = export_threejs_viewer(recording, tmp_path / "legacy")
    assert "view" not in json.loads((legacy.output_dir / "scene.json").read_text())
    assert not (legacy.output_dir / "components").exists()

    presented = export_threejs_viewer(recording, tmp_path / "presented", view_preset="isometric")
    scene = json.loads((presented.output_dir / "scene.json").read_text())
    assert scene["view"]["preset"] == "isometric"
    assert scene["view"]["schema_version"] == "presentation-view.v1"
    assert (
        presented.output_dir / "components" / "presentation_scene" / "presentation_scene.js"
    ).exists()
