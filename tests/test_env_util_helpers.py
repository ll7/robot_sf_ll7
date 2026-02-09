"""Coverage-focused tests for env_util and robot_env helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from robot_sf.common.types import Rect
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import (
    create_spaces,
    make_grid_observation_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.robot_env import (
    _build_step_info,
    _flatten_nested_dict_obs,
    _flatten_nested_dict_spaces,
    _flatten_occupancy_grid_metadata,
    _FlatteningObservationWrapper,
    _make_telemetry_run_id,
    _stable_config_hash,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig


def _minimal_map_def() -> MapDefinition:
    width = 10.0
    height = 8.0
    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone: Rect = ((7.0, 6.0), (8.0, 6.0), (7.0, 7.0))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.2, 1.2), (7.5, 6.5)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def test_make_grid_observation_spaces_metadata_keys() -> None:
    """Ensure grid observation metadata is flattened for SB3 compatibility."""
    grid_config = GridConfig(
        resolution=1.0,
        width=4.0,
        height=3.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )
    grid_box, meta_spaces = make_grid_observation_spaces(grid_config)
    assert grid_box.shape == (
        grid_config.num_channels,
        grid_config.grid_height,
        grid_config.grid_width,
    )
    for key in (
        "occupancy_grid_meta_origin",
        "occupancy_grid_meta_resolution",
        "occupancy_grid_meta_size",
        "occupancy_grid_meta_use_ego_frame",
        "occupancy_grid_meta_center_on_robot",
        "occupancy_grid_meta_channel_indices",
        "occupancy_grid_meta_robot_pose",
    ):
        assert key in meta_spaces


def test_create_spaces_adds_custom_sensors_and_grid() -> None:
    """Verify custom sensor spaces and occupancy grid metadata are injected."""
    grid_config = GridConfig(
        resolution=1.0,
        width=4.0,
        height=3.0,
        channels=[GridChannel.OBSTACLES],
    )
    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        include_grid_in_observation=True,
        grid_config=grid_config,
    )
    config.sensors = [
        {
            "name": "foo",
            "space": {"shape": [2], "low": [-1.0, -1.0], "high": [1.0, 1.0]},
        }
    ]
    _, obs_space, orig_obs_space = create_spaces(config, _minimal_map_def())
    for key in (
        "custom.foo",
        "occupancy_grid",
        "occupancy_grid_meta_origin",
        "occupancy_grid_meta_resolution",
        "occupancy_grid_meta_size",
    ):
        assert key in obs_space.spaces
        assert key in orig_obs_space.spaces


def test_prepare_pedestrian_actions_vectorizes_positions() -> None:
    """Confirm pedestrian action vectors combine position and velocity."""

    class _DummyPeds:
        def __init__(self, pos: np.ndarray, vel: np.ndarray) -> None:
            self._pos = pos
            self._vel = vel

        def pos(self) -> np.ndarray:
            return self._pos

        def vel(self) -> np.ndarray:
            return self._vel

    peds = _DummyPeds(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.5, 0.0], [0.0, 0.5]]))
    simulator = SimpleNamespace(pysf_sim=SimpleNamespace(peds=peds))
    actions = prepare_pedestrian_actions(simulator)
    assert actions.shape == (2, 2, 2)
    assert np.allclose(actions[0, 1], np.array([1.5, 2.0]))


def test_robot_env_hash_and_run_id_stable() -> None:
    """Check config hash stability and telemetry run id uniqueness."""
    cfg = EnvSettings()
    hash_a = _stable_config_hash(cfg)
    hash_b = _stable_config_hash(cfg)
    assert hash_a == hash_b
    assert len(hash_a) == 16
    run_a = _make_telemetry_run_id()
    run_b = _make_telemetry_run_id()
    assert run_a.startswith("telemetry-")
    assert run_a != run_b


def test_robot_env_flatten_helpers() -> None:
    """Ensure nested observation structures are flattened consistently."""
    obs_space = spaces.Dict(
        {
            "robot": spaces.Dict({"pos": spaces.Box(low=-1.0, high=1.0, shape=(2,))}),
            "goal": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        }
    )
    flat_space = _flatten_nested_dict_spaces(obs_space)
    assert "robot_pos" in flat_space.spaces
    obs = {"robot": {"pos": np.array([0.1, 0.2])}, "goal": np.array([0.0, 0.0])}
    flat_obs = _flatten_nested_dict_obs(obs)
    assert "robot_pos" in flat_obs

    metadata = {"origin": np.array([0.0, 0.0], dtype=np.float32)}
    flat_meta = _flatten_occupancy_grid_metadata(metadata)
    assert "occupancy_grid_meta_origin" in flat_meta

    info = _build_step_info({"is_pedestrian_collision": True, "step": 5})
    assert info["collision"] is True
    assert info["step"] == 5

    class _DummyFusion:
        def __init__(self) -> None:
            self.reset_called = False

        def reset_cache(self) -> None:
            self.reset_called = True

        def next_obs(self) -> dict:
            return {"robot": {"pos": np.array([1.0, 2.0])}}

    wrapper = _FlatteningObservationWrapper(_DummyFusion())
    wrapper.reset_cache()
    flattened = wrapper.next_obs()
    assert "robot_pos" in flattened
