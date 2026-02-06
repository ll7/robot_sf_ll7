"""Additional coverage tests for env_util branch-heavy helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.common.types import Rect
from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings, RobotEnvSettings
from robot_sf.gym_env.env_util import (
    AgentType,
    create_spaces,
    create_spaces_with_image,
    init_collision_and_sensors,
    init_collision_and_sensors_with_image,
    init_ped_collision_and_sensors,
    init_ped_spaces,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition


def _minimal_map_def() -> MapDefinition:
    """Create a compact map definition for env_util tests."""
    width = 8.0
    height = 6.0
    spawn_zone: Rect = ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))
    goal_zone: Rect = ((6.0, 4.0), (7.0, 4.0), (7.0, 5.0))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.0, 1.0), (6.5, 4.5)],
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


class _FakeMultiRobotSim:
    """Minimal simulator surface for collision/sensor initialization tests."""

    def __init__(self, map_def: MapDefinition) -> None:
        self.map_def = map_def
        self.robots = [
            SimpleNamespace(pose=((1.0, 1.0), 0.0), current_speed=(0.0, 0.0)),
            SimpleNamespace(pose=((1.5, 1.2), 0.2), current_speed=(0.0, 0.0)),
        ]
        self._goal_pos = [(6.0, 4.0), (6.4, 4.2)]

    @property
    def robot_pos(self) -> list[tuple[float, float]]:
        """Current robot positions for occupancy callbacks."""
        return [robot.pose[0] for robot in self.robots]

    @property
    def goal_pos(self) -> list[tuple[float, float]]:
        """Current robot goal positions."""
        return self._goal_pos

    @property
    def next_goal_pos(self) -> list[None]:
        """Current robot next-goal placeholders."""
        return [None, None]

    @property
    def ped_pos(self) -> np.ndarray:
        """Pedestrian positions."""
        return np.empty((0, 2), dtype=np.float64)

    def get_obstacle_lines(self) -> np.ndarray:
        """Map obstacle segments as an empty line set."""
        return np.empty((0, 4), dtype=np.float64)


class _FakePedSim:
    """Minimal pedestrian simulator surface for env_util initialization tests."""

    def __init__(self, map_def: MapDefinition) -> None:
        self.map_def = map_def
        self.robots = [SimpleNamespace(pose=((1.0, 1.0), 0.0), current_speed=(0.0, 0.0))]
        self._goal_pos = [(6.0, 4.0)]
        self.ped_pos = np.array([[1.8, 1.8]], dtype=np.float64)
        self.ego_ped_pos = (1.4, 1.1)
        self.ego_ped_goal_pos = (2.2, 2.3)
        self.ego_ped = SimpleNamespace(pose=((1.4, 1.1), -0.1), current_speed=(0.0, 0.0))

    @property
    def robot_pos(self) -> list[tuple[float, float]]:
        """Current robot positions for occupancy callbacks."""
        return [robot.pose[0] for robot in self.robots]

    @property
    def goal_pos(self) -> list[tuple[float, float]]:
        """Current robot goal positions."""
        return self._goal_pos

    @property
    def next_goal_pos(self) -> list[None]:
        """Current robot next-goal placeholders."""
        return [None]

    def get_obstacle_lines(self) -> np.ndarray:
        """Map obstacle segments as an empty line set."""
        return np.empty((0, 4), dtype=np.float64)


def test_init_collision_and_sensors_supports_custom_registry_sensors(monkeypatch) -> None:
    """init_collision_and_sensors should wrap base fusion when custom sensors are configured."""
    map_def = _minimal_map_def()
    config = RobotSimulationConfig()
    config.sensors = [{"type": "dummy", "name": "dummy_sensor"}]
    _action_space, _obs_space, orig_obs_space = create_spaces(config, map_def)
    sim = _FakeMultiRobotSim(map_def)

    dummy_sensor = object()

    def _fake_create_sensors(_sensor_cfgs):
        return [dummy_sensor]

    class _WrappedFusion:
        def __init__(self, base_fusion, sensors, names, sim, robot_id) -> None:
            self.base_fusion = base_fusion
            self.sensors = sensors
            self.names = names
            self.sim = sim
            self.robot_id = robot_id

    from robot_sf.gym_env import env_util as env_util_mod

    monkeypatch.setattr(env_util_mod, "create_sensors_from_config", _fake_create_sensors)
    monkeypatch.setattr(env_util_mod, "MergedObservationFusion", _WrappedFusion)

    occupancies, sensors = init_collision_and_sensors(sim, config, orig_obs_space)
    assert len(occupancies) == 2
    assert len(sensors) == 2
    assert isinstance(sensors[0], _WrappedFusion)
    assert sensors[0].names == ["dummy_sensor"]
    assert sensors[0].sensors == [dummy_sensor]


def test_create_spaces_error_paths_and_sensor_without_declared_space() -> None:
    """create_spaces should raise on unsupported agent types and skip sensors without space."""
    map_def = _minimal_map_def()
    basic_cfg = EnvSettings()

    with pytest.raises(ValueError):
        create_spaces(basic_cfg, map_def, agent_type=AgentType.PEDESTRIAN)

    with pytest.raises(ValueError):
        create_spaces(basic_cfg, map_def, agent_type=None)  # type: ignore[arg-type]

    cfg = RobotSimulationConfig()
    cfg.sensors = [{"name": "missing_space"}]  # no "space" field -> skipped branch
    _action_space, obs_space, _orig_obs_space = create_spaces(cfg, map_def)
    assert "custom.missing_space" not in obs_space.spaces


def test_init_ped_spaces_and_collision_sensor_initialization() -> None:
    """Pedestrian helpers should initialize robot+ego-ped spaces, occupancies, and sensors."""
    map_def = _minimal_map_def()
    cfg = PedEnvSettings()
    action_spaces, _obs_spaces, orig_obs_spaces = init_ped_spaces(cfg, map_def)
    assert len(action_spaces) == 2
    assert len(orig_obs_spaces) == 2

    sim = _FakePedSim(map_def)
    occupancies, sensors = init_ped_collision_and_sensors(sim, cfg, orig_obs_spaces)
    assert len(occupancies) == 2
    assert len(sensors) == 2

    # Ensure closures are executable and produce structured observations.
    robot_obs = sensors[0].next_obs()
    ego_obs = sensors[1].next_obs()
    assert "drive_state" in robot_obs
    assert "rays" in robot_obs
    assert "drive_state" in ego_obs
    assert "rays" in ego_obs


def test_create_spaces_with_image_error_paths_and_grid_extension() -> None:
    """create_spaces_with_image should validate agent type and extend grid metadata."""
    map_def = _minimal_map_def()
    basic_cfg = EnvSettings()

    with pytest.raises(ValueError):
        create_spaces_with_image(basic_cfg, map_def, agent_type=AgentType.PEDESTRIAN)

    with pytest.raises(ValueError):
        create_spaces_with_image(basic_cfg, map_def, agent_type=None)  # type: ignore[arg-type]

    cfg = RobotEnvSettings(use_image_obs=True)
    cfg.include_grid_in_observation = True
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    cfg.grid_config = GridConfig(
        resolution=1.0,
        width=4.0,
        height=3.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )
    _action_space, obs_space, orig_obs_space = create_spaces_with_image(cfg, map_def)
    assert "occupancy_grid" in obs_space.spaces
    assert "occupancy_grid_meta_origin" in obs_space.spaces
    assert "occupancy_grid" in orig_obs_space.spaces


def test_init_collision_and_sensors_with_image_uses_default_image_settings(monkeypatch) -> None:
    """Image initialization should fall back to ImageSensorSettings when config.image_config is None."""
    map_def = _minimal_map_def()
    cfg = RobotEnvSettings(use_image_obs=True)
    cfg.image_config = None  # force default ImageSensorSettings() branch
    cfg.sensors = []
    _action_space, _obs_space, orig_obs_space = create_spaces(cfg, map_def)
    sim = _FakeMultiRobotSim(map_def)

    from robot_sf.gym_env import env_util as env_util_mod

    class _DummyImageSensor:
        def __init__(self, image_config, sim_view) -> None:
            self.image_config = image_config
            self.sim_view = sim_view

    class _DummyImageFusion:
        def __init__(
            self,
            ray_sensor,
            speed_sensor,
            target_sensor,
            image_sensor,
            orig_obs_space,
            use_next_goal,
            use_image_obs,
        ) -> None:
            self.ray_sensor = ray_sensor
            self.speed_sensor = speed_sensor
            self.target_sensor = target_sensor
            self.image_sensor = image_sensor
            self.orig_obs_space = orig_obs_space
            self.use_next_goal = use_next_goal
            self.use_image_obs = use_image_obs

    monkeypatch.setattr(env_util_mod, "ImageSensor", _DummyImageSensor)
    monkeypatch.setattr(env_util_mod, "ImageSensorFusion", _DummyImageFusion)

    occupancies, sensor_fusions = init_collision_and_sensors_with_image(
        sim,
        cfg,
        orig_obs_space,
        sim_view=object(),
    )

    # Trigger dynamic objects callback to execute the robot-circle branch body.
    circles = occupancies[0].get_dynamic_objects()
    assert circles is not None
    assert len(circles) == 1
    assert circles[0][0] == sim.robot_pos[1]

    assert len(sensor_fusions) == 2
    assert isinstance(sensor_fusions[0], _DummyImageFusion)
    assert sensor_fusions[0].use_image_obs is True
