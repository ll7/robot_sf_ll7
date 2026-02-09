"""Integration tests for social-force policy wiring in map runner."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark import map_runner


def _make_obs(goal=(5.0, 0.0), heading=0.0):
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.array(goal, dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((1, 2), dtype=np.float32),
            "velocities": np.zeros((1, 2), dtype=np.float32),
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([0.0], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def _with_occupancy_grid(
    obs: dict,
    *,
    obstacle_cells: list[tuple[int, int]] | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (-2.0, -2.0),
    size: tuple[float, float] = (4.0, 4.0),
):
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells or []:
        if 0 <= row < grid.shape[1] and 0 <= col < grid.shape[2]:
            grid[0, row, col] = 1.0
    obs["occupancy_grid"] = grid
    obs["occupancy_grid_meta_origin"] = np.array(origin, dtype=np.float32)
    obs["occupancy_grid_meta_resolution"] = np.array([resolution], dtype=np.float32)
    obs["occupancy_grid_meta_size"] = np.array(size, dtype=np.float32)
    obs["occupancy_grid_meta_use_ego_frame"] = np.array([1.0], dtype=np.float32)
    obs["occupancy_grid_meta_channel_indices"] = np.array([0, 1, 2, 3], dtype=np.float32)
    return obs


def test_map_runner_social_force_reacts_to_obstacles():
    """Map runner should wire social-force policy with obstacle-aware responses."""
    policy, meta = map_runner._build_policy("social_force", {})
    assert meta["status"] == "ok"

    obs_free = _with_occupancy_grid(_make_obs(goal=(5.0, 0.0), heading=0.0))
    v_free, w_free = policy(obs_free)

    obs_blocked = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(2, 3)],
    )
    v_blocked, w_blocked = policy(obs_blocked)
    assert v_blocked < v_free or abs(w_blocked) > abs(w_free) + 1e-3
