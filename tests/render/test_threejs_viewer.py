"""Smoke tests for the optional Three.js recording viewer export."""

import json
from types import SimpleNamespace

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


def test_build_threejs_scene_tolerates_legacy_states_without_optional_fields() -> None:
    """Legacy pickle states may not carry every modern visualizer attribute."""
    legacy_state = SimpleNamespace(
        timestep=0,
        robot_pose=((1.0, 2.0), 0.25),
        pedestrian_positions=np.array([[3.0, 4.0]]),
    )
    episode = PlaybackEpisode(episode_id=8, states=[legacy_state])

    scene = build_threejs_scene(episode, _map(), source="legacy.pkl")

    assert scene["frames"][0]["time_s"] == 0.0
    assert scene["frames"][0]["rays"] == []
    assert scene["frames"][0]["planned_path"] == []
    assert "ego_pedestrian" not in scene["frames"][0]


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


def _rich_state(timestep: int, **overrides: object) -> VisualizableSimState:
    """Build a visualizer state with optional recorded fidelity attributes."""
    state = _state(timestep)
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_fidelity_block_reports_monotonic_slot_symbolic_defaults() -> None:
    """Default states yield timestep timing, slot identity, symbolic geometry."""
    episode = PlaybackEpisode(episode_id=11, states=[_state(0), _state(1)])

    scene = build_threejs_scene(episode, _map(), source="synthetic.jsonl")

    assert scene["frames"][0]["timing_source"] == "timestep_times_dt"
    assert scene["frames"][0]["time_s"] == 0.0
    assert scene["frames"][1]["time_s"] == 0.1
    assert scene["fidelity"]["timing"] == {
        "mode": "monotonic",
        "explicit_time_s": False,
        "bad_time_frames": [],
        "nonmonotonic_frames": [],
    }
    assert scene["fidelity"]["identity"] == {"mode": "slot_index"}
    assert scene["fidelity"]["geometry"]["mode"] == "symbolic"
    ped = scene["frames"][0]["pedestrians"][0]
    assert ped["id"] == 0
    assert ped["heading"] is None
    assert ped["heading_source"] == "unknown"
    assert ped["radius_m"] is None
    assert ped["radius_source"] == "unknown"


def test_explicit_time_and_recorded_actor_fields_are_preserved() -> None:
    """Recorded time, ids, headings, and radii travel verbatim with sources."""
    states = [
        _rich_state(
            0,
            time_s=1.5,
            pedestrian_ids=["ped-a"],
            pedestrian_headings=np.array([0.75]),
            pedestrian_radii=np.array([0.31]),
        ),
        _rich_state(
            5,
            time_s=1.2,
            pedestrian_ids=["ped-a"],
            pedestrian_headings=np.array([0.75]),
            pedestrian_radii=np.array([0.31]),
        ),
    ]
    episode = PlaybackEpisode(episode_id=12, states=states)

    scene = build_threejs_scene(episode, _map(), source="synthetic.jsonl")

    assert [frame["time_s"] for frame in scene["frames"]] == [1.5, 1.2]
    assert scene["frames"][0]["timing_source"] == "explicit_time_s"
    assert scene["fidelity"]["timing"]["mode"] == "nonmonotonic_time_s"
    assert scene["fidelity"]["timing"]["nonmonotonic_frames"] == [1]
    assert scene["fidelity"]["timing"]["explicit_time_s"] is True
    assert scene["fidelity"]["identity"] == {"mode": "stable"}
    assert scene["fidelity"]["geometry"]["mode"] == "recorded"
    ped = scene["frames"][0]["pedestrians"][0]
    assert ped["id"] == "ped-a"
    assert ped["heading"] == 0.75
    assert ped["heading_source"] == "recorded"
    assert ped["radius_m"] == 0.31
    assert ped["radius_source"] == "recorded"
    json.dumps(scene)


def test_numpy_stable_ids_serialize_without_guessing() -> None:
    """Numpy-backed identities serialize; unrepresentable ids fall back to slots."""
    states = [
        _rich_state(0, pedestrian_ids=np.array([np.int64(7)])),
    ]
    episode = PlaybackEpisode(episode_id=13, states=states)

    scene = build_threejs_scene(episode, _map(), source="synthetic.jsonl")

    assert scene["frames"][0]["pedestrians"][0]["id"] == 7
    assert scene["fidelity"]["identity"] == {"mode": "stable"}


def test_map_origin_defaults_explicitly_for_alignment() -> None:
    """Map payloads always carry an origin so asymmetric maps cannot drift."""
    scene = build_threejs_scene(
        PlaybackEpisode(episode_id=14, states=[_state(0)]), _map(), source="s.json"
    )

    assert scene["map"]["origin"] == [0.0, 0.0]
