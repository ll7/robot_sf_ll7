"""Tests for the deterministic hybrid-rule local planner."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    goal: tuple[float, float] = (2.0, 0.0),
    speed: float = 0.0,
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict:
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
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
            "radius": 0.25,
        },
        "sim": {"timestep": 0.1},
    }


def test_hybrid_rule_v0_returns_diagnostics_for_open_space() -> None:
    """V0 should choose a bounded forward command and expose score terms."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    linear, angular = planner.plan(_obs())

    assert 0.0 <= linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed
    diagnostics = planner.diagnostics()
    last = diagnostics["last_decision"]
    assert last["planner_variant"] == "hybrid_rule_v0_minimal"
    assert last["planner_mode"] == "NORMAL"
    assert last["top_k"]
    assert "goal_progress" in last["selected_terms"]
    assert diagnostics["selected_source_counts"]


def test_hybrid_rule_v0_speed_cap_near_humans() -> None:
    """The documented near-human speed cap should limit selected commands."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    linear, _angular = planner.plan(_obs(ped_positions=[(0.8, 0.3)], ped_velocities=[(0.0, 0.0)]))

    assert linear <= planner.config.very_slow_speed + 1e-9
    last = planner.diagnostics()["last_decision"]
    assert last["nearest_pedestrian_distance"] < planner.config.slow_distance_human


def test_hybrid_rule_v0_emergency_stop_when_all_candidates_rejected() -> None:
    """Hard dynamic collision filtering should fail closed to an emergency stop."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    command = planner.plan(_obs(ped_positions=[(0.3, 0.0)], ped_velocities=[(0.0, 0.0)]))

    assert command == (0.0, 0.0)
    diagnostics = planner.diagnostics()
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["last_decision"]["planner_mode"] == "EMERGENCY_STOP"
    assert diagnostics["rejection_counts"]["dynamic_collision"] > 0


def test_hybrid_rule_config_builder_and_variant_guard() -> None:
    """Config parsing should preserve manual constants and reject unknown variants."""
    cfg = build_hybrid_rule_local_planner_config(
        {
            "max_linear_speed": "1.4",
            "linear_samples": "5",
            "lookahead_distances": [0.4, 0.8],
        }
    )
    assert cfg.max_linear_speed == pytest.approx(1.4)
    assert cfg.linear_samples == 5
    assert cfg.lookahead_distances == (0.4, 0.8)

    with pytest.raises(ValueError, match="Unsupported hybrid rule planner variant"):
        HybridRuleLocalPlannerAdapter(
            HybridRuleLocalPlannerConfig(planner_variant="hybrid_rule_v9_unknown")
        )


def test_hybrid_rule_reset_clears_episode_diagnostics() -> None:
    """Reset should clear per-episode planner state for reproducible runs."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())
    planner.plan(_obs())
    assert planner.diagnostics()["steps"] == 1

    planner.reset(seed=123)

    diagnostics = planner.diagnostics()
    assert diagnostics["steps"] == 0
    assert diagnostics["last_decision"] is None
