"""Tests for the route-waypoint producer in SocNavObservationFusion (issue #5331)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_sf.sensor.socnav_observation import (
    MAX_ROUTE_WAYPOINTS,
    SOCNAV_POSITION_CAP_M,
    SocNavObservationFusion,
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
class FakeEnvConfig:
    """Minimal environment config stub for SocNavObservationFusion."""

    sim_config: Any = None
    predictive_foresight_enabled: bool = False
    include_grid_in_observation: bool = False
    max_total_pedestrians: int = 64

    def __post_init__(self) -> None:
        """Set default sim_config when none is provided."""
        if self.sim_config is None:
            self.sim_config = type("SimCfg", (), {"ped_radius": 0.3})()


def _build_fusion(
    *,
    waypoints: list[tuple[float, float]] | None = None,
    waypoint_id: int = 0,
) -> tuple[SocNavObservationFusion, FakeSimulator]:
    """Build a SocNavObservationFusion with a stubbed simulator."""
    navs = [FakeNavigator(waypoints=waypoints or [], waypoint_id=waypoint_id)]
    sim = FakeSimulator(robot_navs=navs)
    env_config = FakeEnvConfig()
    fusion = SocNavObservationFusion(simulator=sim, env_config=env_config, max_pedestrians=16)
    return fusion, sim


class TestRouteWaypointProducer:
    """Unit tests for _build_route_waypoints and observation injection."""

    def test_route_waypoints_present_in_obs(self) -> None:
        """route_waypoints appears in the robot dict when the navigator has a route."""
        fusion, _ = _build_fusion(waypoints=[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
        obs = fusion.next_obs()
        assert "route_waypoints" in obs["robot"]
        wp = obs["robot"]["route_waypoints"]
        assert wp.shape == (4, 2)
        np.testing.assert_allclose(wp, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    def test_route_waypoints_starts_from_current_waypoint_id(self) -> None:
        """Only waypoints from waypoint_id onward plus robot pos are included."""
        fusion, _ = _build_fusion(
            waypoints=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            waypoint_id=2,
        )
        obs = fusion.next_obs()
        wp = obs["robot"]["route_waypoints"]
        assert wp.shape == (3, 2)
        np.testing.assert_allclose(wp, [[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    def test_route_waypoints_empty_when_no_route(self) -> None:
        """Empty (0, 2) array when navigator has no waypoints."""
        fusion, _ = _build_fusion(waypoints=[])
        obs = fusion.next_obs()
        wp = obs["robot"]["route_waypoints"]
        assert wp.shape == (0, 2)

    def test_route_waypoints_empty_when_all_visited(self) -> None:
        """Only robot_pos when waypoint_id is past the end (remaining is empty)."""
        fusion, _ = _build_fusion(
            waypoints=[(1.0, 0.0), (2.0, 0.0)],
            waypoint_id=2,
        )
        obs = fusion.next_obs()
        wp = obs["robot"]["route_waypoints"]
        assert wp.shape == (1, 2)
        np.testing.assert_allclose(wp, [[0.0, 0.0]])

    def test_route_waypoints_capped_at_max(self) -> None:
        """Waypoint count is capped at MAX_ROUTE_WAYPOINTS."""
        many = [(float(i), 0.0) for i in range(50)]
        fusion, _ = _build_fusion(waypoints=many)
        obs = fusion.next_obs()
        wp = obs["robot"]["route_waypoints"]
        assert wp.shape[0] == MAX_ROUTE_WAYPOINTS

    def test_route_waypoints_clipped_to_position_cap(self) -> None:
        """Waypoints outside the map extent are clipped to SOCNAV_POSITION_CAP_M."""
        fusion, _ = _build_fusion(waypoints=[(100.0, -5.0)])
        obs = fusion.next_obs()
        wp = obs["robot"]["route_waypoints"]
        assert float(wp[0, 0]) <= SOCNAV_POSITION_CAP_M
        assert float(wp[0, 1]) >= 0.0

    def test_route_waypoints_float32(self) -> None:
        """Route waypoints are returned as float32."""
        fusion, _ = _build_fusion(waypoints=[(1.0, 2.0)])
        obs = fusion.next_obs()
        assert obs["robot"]["route_waypoints"].dtype == np.float32

    def test_build_route_waypoints_no_navs(self) -> None:
        """Empty array when simulator has no robot_navs attribute."""
        fusion, sim = _build_fusion()
        sim.robot_navs = []
        cap = np.array([20.0, 20.0], dtype=np.float32)
        wp = fusion._build_route_waypoints(cap)
        assert wp.shape == (0, 2)

    def test_build_route_waypoints_no_waypoints_attr(self) -> None:
        """Empty array when navigator lacks waypoints attribute."""
        fusion, sim = _build_fusion()
        sim.robot_navs = [type("Nav", (), {"waypoint_id": 0})()]
        cap = np.array([20.0, 20.0], dtype=np.float32)
        wp = fusion._build_route_waypoints(cap)
        assert wp.shape == (0, 2)


class TestRouteWaypointProducerIntegration:
    """Integration tests: DWA probe activates when route_waypoints are present."""

    def test_dwa_probe_activates_with_producer_waypoints(self) -> None:
        """The DWA global-route probe reports activation when waypoints are in the observation."""
        from robot_sf.planner.dwa import DWAPlannerAdapter, DWAPlannerConfig

        fusion, _ = _build_fusion(waypoints=[(0.5, 0.0), (1.5, 0.0), (3.0, 0.0)])
        obs = fusion.next_obs()
        config = DWAPlannerConfig(
            global_route_probe_enabled=True,
            global_route_probe_waypoint_distance=2.0,
        )
        planner = DWAPlannerAdapter(config)
        planner.plan(obs)
        diag = planner.diagnostics()
        assert diag["last_decision"]["global_route_probe_activated"] is True

    def test_dwa_probe_does_not_activate_with_empty_route(self) -> None:
        """The probe does not activate when the route is empty."""
        from robot_sf.planner.dwa import DWAPlannerAdapter, DWAPlannerConfig

        fusion, _ = _build_fusion(waypoints=[])
        obs = fusion.next_obs()
        config = DWAPlannerConfig(global_route_probe_enabled=True)
        planner = DWAPlannerAdapter(config)
        planner.plan(obs)
        diag = planner.diagnostics()
        assert diag["last_decision"]["global_route_probe_activated"] is False

    def test_dwa_probe_targets_forward_waypoint(self) -> None:
        """The probe uses the forward waypoint (after nearest) for scoring."""
        from robot_sf.planner.dwa import DWAPlannerAdapter, DWAPlannerConfig

        waypoints = [(0.1, 0.0), (2.0, 2.0), (3.0, 0.0)]
        fusion, _ = _build_fusion(waypoints=waypoints)
        obs = fusion.next_obs()

        config = DWAPlannerConfig(
            global_route_probe_enabled=True,
            global_route_probe_heading_weight=2.0,
        )
        planner = DWAPlannerAdapter(config)
        cmd_with_probe = planner.plan(obs)

        baseline_config = DWAPlannerConfig(global_route_probe_enabled=False)
        baseline_planner = DWAPlannerAdapter(baseline_config)
        cmd_baseline = baseline_planner.plan(obs)

        assert cmd_with_probe != cmd_baseline
