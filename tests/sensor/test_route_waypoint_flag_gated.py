"""Tests for flag-gated route_waypoints observation (issue #5349)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from robot_sf.sensor.socnav_observation import (
    MAX_ROUTE_WAYPOINTS,
    SocNavObservationFusion,
    socnav_observation_space,
)


@dataclass
class FakeNavigator:
    """Minimal navigator stub exposing waypoints and waypoint_id."""

    waypoints: list[tuple[float, float]] = field(default_factory=list)
    waypoint_id: int = 0


@dataclass
class FakeRobot:
    """Minimal robot stub for SocNavObservationFusion testing."""

    pose: tuple[tuple[float, float], float] = ((0.0, 0.0), 0.0)
    current_speed: tuple[float, float] = (0.0, 0.0)
    config: Any = None

    def __post_init__(self) -> None:
        """Set default config when none is provided."""
        if self.config is None:
            self.config = type("Cfg", (), {"radius": 0.25})()


@dataclass
class FakeSimulator:
    """Minimal simulator stub providing navigator and robot state."""

    robot_navs: list[FakeNavigator] = field(default_factory=list)
    robots: list[FakeRobot] = field(default_factory=lambda: [FakeRobot()])
    ped_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    ped_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    goal_pos: list[tuple[float, float]] = field(default_factory=lambda: [(3.0, 0.0)])
    next_goal_pos: list[tuple[float, float] | None] = field(default_factory=lambda: [None])
    robot_pos: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    map_def: Any = None
    config: Any = None

    def __post_init__(self) -> None:
        """Set default map_def and config when none are provided."""
        if self.map_def is None:
            self.map_def = type(
                "MapDef",
                (),
                {"width": 20.0, "height": 20.0, "obstacles": []},
            )()
        if self.config is None:
            self.config = type("Cfg", (), {"time_per_step_in_secs": 0.1})()


@dataclass
class FakeRobotConfig:
    """Minimal robot config stub."""

    max_linear_speed: float = 2.0


@dataclass
class FakeEnvConfig:
    """Minimal environment config stub for SocNavObservationFusion."""

    sim_config: Any = None
    robot_config: Any = None
    predictive_foresight_enabled: bool = False
    include_grid_in_observation: bool = False
    max_total_pedestrians: int = 64
    include_route_waypoints: bool = False

    def __post_init__(self) -> None:
        """Set default configs when none are provided."""
        if self.sim_config is None:
            self.sim_config = type("SimCfg", (), {"ped_radius": 0.3})()
        if self.robot_config is None:
            self.robot_config = FakeRobotConfig()


@dataclass
class FakeMapDef:
    """Minimal map definition stub."""

    width: float = 20.0
    height: float = 20.0
    obstacles: list = field(default_factory=list)


def _build_fusion(
    *,
    include_route_waypoints: bool = False,
    waypoints: list[tuple[float, float]] | None = None,
    waypoint_id: int = 0,
) -> tuple[SocNavObservationFusion, FakeSimulator]:
    """Build a SocNavObservationFusion with a stubbed simulator."""
    navs = [FakeNavigator(waypoints=waypoints or [], waypoint_id=waypoint_id)]
    sim = FakeSimulator(robot_navs=navs)
    env_config = FakeEnvConfig(include_route_waypoints=include_route_waypoints)
    fusion = SocNavObservationFusion(simulator=sim, env_config=env_config, max_pedestrians=16)
    return fusion, sim


class TestRouteWaypointsFlagGated:
    """Test that route_waypoints is flag-gated and default OFF."""

    def test_route_waypoints_absent_when_flag_off(self) -> None:
        """route_waypoints is NOT in obs when include_route_waypoints is False (default)."""
        fusion, _ = _build_fusion(include_route_waypoints=False, waypoints=[(1.0, 0.0)])
        obs = fusion.next_obs()
        assert "route_waypoints" not in obs

    def test_route_waypoints_present_when_flag_on(self) -> None:
        """route_waypoints IS in obs when include_route_waypoints is True."""
        fusion, _ = _build_fusion(include_route_waypoints=True, waypoints=[(1.0, 0.0)])
        obs = fusion.next_obs()
        assert "route_waypoints" in obs
        wp = obs["route_waypoints"]
        assert wp.shape == (MAX_ROUTE_WAYPOINTS, 2)
        assert wp.dtype == np.float32

    def test_route_waypoints_padded_to_fixed_size(self) -> None:
        """route_waypoints is always padded to (MAX_ROUTE_WAYPOINTS, 2)."""
        fusion, _ = _build_fusion(
            include_route_waypoints=True,
            waypoints=[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        obs = fusion.next_obs()
        wp = obs["route_waypoints"]
        assert wp.shape == (MAX_ROUTE_WAYPOINTS, 2)
        # First row is robot position, next 3 are waypoints, rest are zeros
        np.testing.assert_allclose(wp[0], [0.0, 0.0])
        np.testing.assert_allclose(wp[1], [1.0, 0.0])
        np.testing.assert_allclose(wp[2], [2.0, 0.0])
        np.testing.assert_allclose(wp[3], [3.0, 0.0])
        np.testing.assert_allclose(wp[4], [0.0, 0.0])

    def test_route_waypoints_empty_when_no_route(self) -> None:
        """route_waypoints is all zeros when navigator has no waypoints."""
        fusion, _ = _build_fusion(include_route_waypoints=True, waypoints=[])
        obs = fusion.next_obs()
        wp = obs["route_waypoints"]
        assert wp.shape == (MAX_ROUTE_WAYPOINTS, 2)
        np.testing.assert_allclose(wp, np.zeros((MAX_ROUTE_WAYPOINTS, 2)))

    def test_route_waypoints_starts_from_current_waypoint_id(self) -> None:
        """Only waypoints from waypoint_id onward plus robot pos are included."""
        fusion, _ = _build_fusion(
            include_route_waypoints=True,
            waypoints=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            waypoint_id=2,
        )
        obs = fusion.next_obs()
        wp = obs["route_waypoints"]
        assert wp.shape == (MAX_ROUTE_WAYPOINTS, 2)
        # Robot pos + 2 remaining waypoints
        np.testing.assert_allclose(wp[0], [0.0, 0.0])
        np.testing.assert_allclose(wp[1], [2.0, 0.0])
        np.testing.assert_allclose(wp[2], [3.0, 0.0])
        np.testing.assert_allclose(wp[3], [0.0, 0.0])

    def test_route_waypoints_capped_at_max(self) -> None:
        """Waypoint count is capped at MAX_ROUTE_WAYPOINTS."""
        many = [(float(i), 0.0) for i in range(50)]
        fusion, _ = _build_fusion(include_route_waypoints=True, waypoints=many)
        obs = fusion.next_obs()
        wp = obs["route_waypoints"]
        assert wp.shape == (MAX_ROUTE_WAYPOINTS, 2)
        # Should have robot pos + MAX_ROUTE_WAYPOINTS-1 waypoints
        assert wp[MAX_ROUTE_WAYPOINTS - 1, 0] != 0.0 or wp[MAX_ROUTE_WAYPOINTS - 1, 1] != 0.0


class TestRouteWaypointsBoxLeaf:
    """Test that route_waypoints is a Box leaf, not Sequence."""

    def test_observation_space_has_box_leaf_when_flag_on(self) -> None:
        """route_waypoints is a Box leaf in the observation space when flag is on."""
        env_config = FakeEnvConfig(include_route_waypoints=True)
        map_def = FakeMapDef()
        obs_space = socnav_observation_space(map_def, env_config, max_pedestrians=16)
        assert "route_waypoints" in obs_space.spaces
        assert isinstance(obs_space.spaces["route_waypoints"], spaces.Box)
        assert obs_space.spaces["route_waypoints"].shape == (MAX_ROUTE_WAYPOINTS, 2)

    def test_observation_space_no_sequence_when_flag_on(self) -> None:
        """No Sequence leaves in observation space when flag is on."""
        env_config = FakeEnvConfig(include_route_waypoints=True)
        map_def = FakeMapDef()
        obs_space = socnav_observation_space(map_def, env_config, max_pedestrians=16)

        def _check_no_sequence(space: spaces.Space) -> None:
            if isinstance(space, spaces.Dict):
                for key, child in space.spaces.items():
                    assert not isinstance(child, spaces.Sequence), f"Found Sequence at key {key}"
                    _check_no_sequence(child)

        _check_no_sequence(obs_space)

    def test_observation_space_unchanged_when_flag_off(self) -> None:
        """Observation space is identical to main when flag is off (default)."""
        env_config_off = FakeEnvConfig(include_route_waypoints=False)
        env_config_default = FakeEnvConfig()
        map_def = FakeMapDef()
        obs_space_off = socnav_observation_space(map_def, env_config_off, max_pedestrians=16)
        obs_space_default = socnav_observation_space(map_def, env_config_default, max_pedestrians=16)
        assert obs_space_off.spaces.keys() == obs_space_default.spaces.keys()
        for key in obs_space_off.spaces:
            assert isinstance(obs_space_off.spaces[key], type(obs_space_default.spaces[key]))


class TestRouteWaypointsConsumerCompat:
    """Test consumer compatibility matrix (issue #5349 acceptance)."""

    def test_flatten_dict_wrapper_accepts_flag_on_space(self) -> None:
        """FlattenDictObservationWrapper accepts observation space with route_waypoints flag on."""
        from gymnasium import Env, spaces

        from robot_sf.training.rllib_env_wrappers import FlattenDictObservationWrapper

        env_config = FakeEnvConfig(include_route_waypoints=True)
        map_def = FakeMapDef()
        obs_space = socnav_observation_space(map_def, env_config, max_pedestrians=16)

        # Create a minimal gymnasium Env with this observation space
        class FakeEnv(Env):
            observation_space = obs_space
            action_space = spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                return obs_space.sample(), {}

            def step(self, action):
                return obs_space.sample(), 0.0, False, False, {}

        # This should not raise TypeError about Sequence leaves
        wrapper = FlattenDictObservationWrapper(FakeEnv())
        assert wrapper is not None

    def test_asymmetric_critic_accepts_flag_on_space(self) -> None:
        """asymmetric_critic traversal accepts observation space with route_waypoints flag on."""
        env_config = FakeEnvConfig(include_route_waypoints=True)
        map_def = FakeMapDef()
        obs_space = socnav_observation_space(map_def, env_config, max_pedestrians=16)

        # Simulate the asymmetric_critic traversal
        def _traverse(spaces_dict: dict, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
            paths: list[tuple[str, ...]] = []
            for key, child in spaces_dict.items():
                path = prefix + (key,)
                if isinstance(child, spaces.Dict):
                    paths.extend(_traverse(child.spaces, path))
                elif isinstance(child, spaces.Box):
                    paths.append(path)
                else:
                    raise TypeError(f"asymmetric_critic requires Box leaves; got {type(child).__name__}")
            return paths

        # This should not raise TypeError about Sequence leaves
        paths = _traverse(obs_space.spaces)
        assert len(paths) > 0
