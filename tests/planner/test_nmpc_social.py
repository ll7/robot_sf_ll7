"""Tests for the testing-only native NMPC social planner."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np

from robot_sf.planner.nmpc_social import (
    NMPCSocialConfig,
    NMPCSocialPlannerAdapter,
    _RolloutContext,
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


def test_build_nmpc_social_config_invalid_numeric_uses_default() -> None:
    """Invalid numeric overrides should warn and fall back to the dataclass default."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = build_nmpc_social_config({"rollout_dt": "bad-value"})
    assert cfg.rollout_dt == NMPCSocialConfig().rollout_dt
    assert any("Invalid NMPC config value for 'rollout_dt'" in str(item.message) for item in caught)


def test_build_nmpc_social_config_parses_boolean_strings() -> None:
    """Boolean config strings should use explicit parsing rather than Python truthiness."""
    cfg = build_nmpc_social_config({"warm_start": "false", "fallback_to_stop": "yes"})
    assert cfg.warm_start is False
    assert cfg.fallback_to_stop is True


def test_build_nmpc_social_config_invalid_boolean_uses_default() -> None:
    """Invalid boolean strings should warn and fall back to the dataclass default."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = build_nmpc_social_config({"warm_start": "maybe"})
    assert cfg.warm_start is NMPCSocialConfig().warm_start
    assert any("Invalid NMPC config value for 'warm_start'" in str(item.message) for item in caught)


def test_nmpc_social_moves_toward_goal_in_open_space() -> None:
    """Open-space behavior should remain goal directed and bounded."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=12))
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert v > 0.0
    assert abs(w) <= planner.config.max_angular_speed


def test_nmpc_social_prioritizes_current_waypoint_until_close() -> None:
    """The planner should steer toward the current waypoint before jumping to the next one."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=12))
    obs = _obs(robot=(0.0, 0.0), heading=0.0, goal=(3.0, 0.0))
    obs["goal"]["current"] = np.asarray((0.0, 2.0), dtype=float)
    obs["goal"]["next"] = np.asarray((3.0, 0.0), dtype=float)
    _v, w = planner.plan(obs)
    assert w > 0.05


def test_nmpc_social_does_not_switch_to_origin_when_next_waypoint_is_missing() -> None:
    """A single-goal observation should keep tracking goal.current during the final approach."""
    planner = NMPCSocialPlannerAdapter(
        NMPCSocialConfig(
            horizon_steps=4,
            solver_max_iterations=12,
            waypoint_switch_distance=1.0,
        )
    )
    obs = _obs(robot=(0.0, 1.6), heading=0.0, goal=(0.0, 2.0))
    del obs["goal"]["next"]
    _v, w = planner.plan(obs)
    assert w > 0.05


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


def test_nmpc_social_truncates_extra_pedestrian_velocities() -> None:
    """Velocity rows beyond the active pedestrian count should be ignored, not zero all motion."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig())
    obs = _obs(
        ped_positions=[(1.0, 0.0)],
        ped_velocities=[(-0.2, 0.0), (9.0, 9.0)],
    )
    *_prefix, ped_positions, ped_velocities, _robot_radius, _ped_radius = planner._extract_state(
        obs
    )
    assert ped_positions.shape == (1, 2)
    assert ped_velocities.shape == (1, 2)
    np.testing.assert_allclose(ped_velocities[0], np.asarray([-0.2, 0.0]))


def test_nmpc_social_rollout_cost_respects_context_speed_cap() -> None:
    """The rollout objective should clip linear speed using the per-step dynamic speed cap."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(max_linear_speed=0.9, horizon_steps=1))
    obs = _obs(goal=(3.0, 0.0))
    context = _RolloutContext(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([3.0, 0.0], dtype=float),
        ped_positions=np.zeros((0, 2), dtype=float),
        ped_velocities=np.zeros((0, 2), dtype=float),
        robot_radius=0.25,
        ped_radius=0.25,
        observation=obs,
        speed_cap=0.2,
    )
    limited = planner._rollout_cost(np.asarray([0.2, 0.0]), context=context)
    saturated = planner._rollout_cost(np.asarray([0.9, 0.0]), context=context)
    assert saturated == limited


def test_nmpc_social_reset_clears_warm_start_solution() -> None:
    """Reset should clear the cached previous solution."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=12))
    planner.plan(_obs(goal=(3.0, 0.0)))
    assert planner._last_solution is not None
    planner.reset()
    assert planner._last_solution is None
    assert planner.diagnostics()["calls"] == 0


def test_nmpc_social_optimizer_failure_stops(monkeypatch) -> None:
    """Optimizer failure should fail closed to a stop command by default."""
    planner = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=4, solver_max_iterations=8))
    monkeypatch.setattr(
        "robot_sf.planner.nmpc_social.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=np.zeros(8, dtype=float)),
    )
    assert planner.plan(_obs(goal=(3.0, 0.0))) == (0.0, 0.0)
    stats = planner.diagnostics()
    assert stats["solver_failures"] == 1
    assert stats["fallback_stop_count"] == 1
    assert stats["nonzero_command_count"] == 0


def test_nmpc_social_optimizer_failure_without_stop_fallback_warns(monkeypatch) -> None:
    """Solver failures without stop fallback should warn and reuse the initial guess."""
    planner = NMPCSocialPlannerAdapter(
        NMPCSocialConfig(horizon_steps=4, solver_max_iterations=8, fallback_to_stop=False)
    )
    monkeypatch.setattr(
        "robot_sf.planner.nmpc_social.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=None),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        command = planner.plan(_obs(goal=(3.0, 0.0)))
    assert command[0] > 0.0
    assert planner._last_solution is not None
    stats = planner.diagnostics()
    assert stats["solver_failures"] == 1
    assert stats["fallback_stop_count"] == 0
    assert any("reusing the initial guess" in str(item.message) for item in caught)
