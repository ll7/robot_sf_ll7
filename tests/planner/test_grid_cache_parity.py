"""Verify that per-plan occupancy-grid caching produces identical output."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.dwa import DWAPlannerAdapter
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin


def _grid_obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(3.0, 0.0),
    ped_positions=None,
    grid_shape=(3, 10, 10),
) -> dict[str, object]:
    """Build an observation with occupancy grid."""
    ped_positions = [] if ped_positions is None else ped_positions
    grid = np.zeros(grid_shape, dtype=float)
    grid[0, 3:7, 3:7] = 1.0  # obstacle channel center
    grid[1, 2:4, 2:4] = 0.8  # pedestrian-like channel
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray([], dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
        },
        "occupancy_grid": grid.tolist(),
        "occupancy_grid_meta_origin": [0.0 - 2.0, 0.0 - 2.0],
        "occupancy_grid_meta_resolution": [0.4],
        "occupancy_grid_meta_size": [4.0, 4.0],
        "occupancy_grid_meta_use_ego_frame": [0.0],
        "occupancy_grid_meta_center_on_robot": [0.0],
        "occupancy_grid_meta_channel_indices": [0, 1, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }


def test_dwa_plan_output_unchanged_with_grid_caching(monkeypatch) -> None:
    """DWA output must match the uncached extraction path for fixed observations."""
    cached_planner = DWAPlannerAdapter()
    uncached_planner = DWAPlannerAdapter()
    monkeypatch.setattr(uncached_planner, "_cache_grid_payload", lambda _observation: None)

    # Multiple different observations to exercise different code paths
    observations = [
        _grid_obs(),
        _grid_obs(ped_positions=[(1.5, 0.5), (2.0, -0.8)]),
        _grid_obs(goal=(0.1, 0.1)),  # near-goal
        _grid_obs(robot=(1.0, 1.0), goal=(5.0, 4.0)),
    ]

    for obs in observations:
        assert cached_planner.plan(obs) == uncached_planner.plan(obs)


def test_risk_dwa_plan_output_unchanged_with_grid_caching(monkeypatch) -> None:
    """RiskDWA output must match the uncached extraction path for fixed observations."""
    cached_planner = RiskDWAPlannerAdapter()
    uncached_planner = RiskDWAPlannerAdapter()
    monkeypatch.setattr(uncached_planner, "_cache_grid_payload", lambda _observation: None)

    observations = [
        _grid_obs(),
        _grid_obs(ped_positions=[(1.5, 0.5), (2.0, -0.8)]),
        _grid_obs(goal=(0.1, 0.1)),
        _grid_obs(robot=(1.0, 1.0), goal=(5.0, 4.0)),
    ]

    for obs in observations:
        assert cached_planner.plan(obs) == uncached_planner.plan(obs)


def test_mixin_cache_grid_payload_provides_sentinel_when_no_grid() -> None:
    """_cache_grid_payload must return a sentinel for observations without a grid."""
    mixin = OccupancyAwarePlannerMixin()
    obs = {
        "robot": {"position": [0.0, 0.0]},
        "goal": {"current": [3.0, 0.0], "next": [3.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0]},
    }
    grid, meta = mixin._cache_grid_payload(obs)
    assert grid.shape == (0, 0, 0)
    assert meta == {}


def test_mixin_cache_grid_payload_returns_payload_when_grid_present() -> None:
    """_cache_grid_payload must return the real grid when available."""
    mixin = OccupancyAwarePlannerMixin()
    obs = _grid_obs()
    grid, meta = mixin._cache_grid_payload(obs)
    assert grid.shape == (3, 10, 10)
    assert meta  # not empty


def test_path_penalty_with_cached_grid_payload() -> None:
    """_path_penalty must produce same result with cached or uncached grid_payload."""
    mixin = OccupancyAwarePlannerMixin()
    obs = _grid_obs(robot=(1.0, 1.0), goal=(3.0, 3.0))
    direction = np.array([1.0, 1.0])
    robot_pos = np.array([1.0, 1.0])

    # Uncached path
    uncached = mixin._path_penalty(robot_pos, direction, obs, base_distance=2.0, num_samples=5)

    # Cached path
    grid_payload = mixin._extract_grid_payload(obs)
    cached = mixin._path_penalty(
        robot_pos, direction, obs, base_distance=2.0, num_samples=5, grid_payload=grid_payload
    )

    assert uncached == cached


def test_path_penalty_with_no_grid_payload() -> None:
    """_path_penalty must return (0, 0) when no grid is available."""
    mixin = OccupancyAwarePlannerMixin()
    robot_pos = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    obs = {"robot": {"position": [0.0, 0.0]}}

    result = mixin._path_penalty(robot_pos, direction, obs, base_distance=1.0, num_samples=3)
    assert result == (0.0, 0.0)
