"""Smoke tests for the optional Three.js recording viewer export."""

import json

import numpy as np

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.render.jsonl_playback import PlaybackEpisode
from robot_sf.render.sim_state import VisualizableSimState
from robot_sf.render.threejs_viewer import (
    SCENE_SCHEMA_VERSION,
    build_threejs_scene,
    export_threejs_viewer,
)


def _state(timestep: int) -> VisualizableSimState:
    return VisualizableSimState(
        timestep=timestep,
        robot_action=None,
        robot_pose=((1.0 + timestep, 2.0), 0.25),
        pedestrian_positions=np.array([[3.0, 4.0]]),
        ray_vecs=np.array([[[1.0, 2.0], [2.0, 3.0]]]),
        ped_actions=np.array([]),
        planned_path=[(1.0, 2.0), (5.0, 6.0)],
    )


def _map() -> MapDefinition:
    zone = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))
    return MapDefinition(
        width=10.0,
        height=8.0,
        obstacles=[Obstacle([(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)])],
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


def test_build_threejs_scene_contains_non_empty_playback_payload() -> None:
    """The exported scene keeps map geometry, robot trajectory, and pedestrian frames."""
    episode = PlaybackEpisode(episode_id=7, states=[_state(0), _state(1)])

    scene = build_threejs_scene(episode, _map(), source="synthetic.jsonl")

    assert scene["schema_version"] == SCENE_SCHEMA_VERSION
    assert scene["map"]["obstacles"][0]["vertices"]
    assert scene["frames"][0]["robot"]["position"] == [1.0, 2.0]
    assert scene["frames"][0]["pedestrians"][0]["position"] == [3.0, 4.0]
    assert scene["trajectory"] == [[1.0, 2.0], [2.0, 2.0]]
    assert "Pygame remains" in scene["limitations"][1]


def test_export_threejs_viewer_writes_static_assets(tmp_path) -> None:
    """A tiny JSONL recording can be exported without launching a browser."""
    recording = tmp_path / "episode.jsonl"
    recording.write_text(
        json.dumps(
            {
                "episode_id": 3,
                "step_idx": 0,
                "event": "step",
                "timestamp": 0.0,
                "state": {
                    "timestep": 0,
                    "robot_pose": [[1.0, 2.0], 0.0],
                    "pedestrian_positions": [[3.0, 4.0]],
                    "ray_vecs": [[[1.0, 2.0], [2.0, 3.0]]],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_threejs_viewer(recording, tmp_path / "viewer")

    assert result.html_path.exists()
    assert (result.output_dir / "viewer.js").exists()
    scene = json.loads(result.scene_path.read_text(encoding="utf-8"))
    assert scene["episode_id"] == 3
    assert len(scene["frames"]) == 1
    assert "three.module.js" in (result.output_dir / "viewer.js").read_text(encoding="utf-8")
