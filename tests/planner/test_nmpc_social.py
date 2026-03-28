"""Tests for the testing-only native NMPC social planner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_sf.planner.nmpc_social import (
    NMPCSocialConfig,
    NMPCSocialPlannerAdapter,
    build_nmpc_social_config,
)


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(2.0, 0.0),
    ped_positions=None,
    ped_velocities=None,
    obstacle_cells=None,
):
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    obstacle_cells = [] if obstacle_cells is None else obstacle_cells
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells:
        grid[0, row, col] = 1.0
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }


def test_build_nmpc_social_config_overrides_fields() -> None:
    """Config builder should thread explicit planner settings through."""
    cfg = build_nmpc_social_config(
        {"horizon_steps": 5, "rollout_dt": 0.2, "solver_max_iterations": 16}
    )
    assert cfg.horizon_steps == 5
    assert cfg.rollout_dt == 0.2
    assert cfg.solver_max_iterations == 16


def test_nmpc_social_moves_toward_goal_in_open_space() -> None:
    """Open-space behavior should remain goal directed and bounded."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=12))
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert v > 0.0
    assert abs(w) <= planner.config.max_angular_speed


def test_nmpc_social_turns_away_from_head_on_pedestrian() -> None:
    """A head-on pedestrian should produce a non-zero steering command."""
    planner = NMPCSocialPlannerAdapter(
        NMPCSocialConfig(horizon_steps=5, solver_max_iterations=16, pedestrian_clearance_weight=5.0)
    )
    v, w = planner.plan(
        _obs(
            goal=(3.0, 0.0),
            ped_positions=[(0.8, 0.0)],
            ped_velocities=[(-0.3, 0.0)],
        )
    )
    assert v >= 0.0
    assert abs(w) > 1e-3


def test_nmpc_social_reset_clears_warm_start_solution() -> None:
    """Reset should clear the cached previous solution."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=12))
    planner.plan(_obs(goal=(3.0, 0.0)))
    assert planner._last_solution is not None
    planner.reset()
    assert planner._last_solution is None


def test_nmpc_social_optimizer_failure_stops(monkeypatch) -> None:
    """Optimizer failure should fail closed to a stop command by default."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=8))
    monkeypatch.setattr(
        "robot_sf.planner.nmpc_social.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=np.zeros(8, dtype=float)),
    )
    assert planner.plan(_obs(goal=(3.0, 0.0))) == (0.0, 0.0)
