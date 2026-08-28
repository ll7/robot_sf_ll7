"""Analytic, lifecycle, and deterministic-smoke tests for the force-coupled planner."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner import (
    ForceCoupledPotentialFieldConfig as PublicConfig,
)
from robot_sf.planner import (
    ForceCoupledPotentialFieldPlanner as PublicPlanner,
)
from robot_sf.planner import (
    build_force_coupled_potential_field_config as public_build_config,
)
from robot_sf.planner.force_coupled_potential_field import (
    ForceCoupledPotentialFieldConfig,
    ForceCoupledPotentialFieldPlanner,
    build_force_coupled_potential_field_config,
)
from robot_sf.planner.protocol import LocalPlannerProtocol


def _observation(
    *,
    robot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    goal: tuple[float, float] = (4.0, 0.0),
    obstacles: list[tuple[float, float]] | None = None,
    pedestrians: list[tuple[float, float]] | None = None,
) -> dict:
    return {
        "robot": list(robot),
        "goal": list(goal),
        "obstacles": {"positions": obstacles or []},
        "pedestrians": {"positions": pedestrians or [], "count": [len(pedestrians or [])]},
    }


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ForceCoupledPotentialFieldConfig(max_linear_speed=0.0)
    with pytest.raises(ValueError):
        ForceCoupledPotentialFieldConfig(influence_radius_m=-1.0)
    with pytest.raises(ValueError):
        ForceCoupledPotentialFieldConfig(look_ahead_min_m=3.0, look_ahead_max_m=1.0)
    with pytest.raises(ValueError):
        ForceCoupledPotentialFieldConfig(obstacle_input_mode="bogus")
    with pytest.raises(ValueError):
        ForceCoupledPotentialFieldConfig(obstacle_grid_threshold=1.1)
    with pytest.raises(ValueError):
        ForceCoupledPotentialFieldConfig(obstacle_grid_max_points=0)


def test_config_digest_is_stable() -> None:
    config = ForceCoupledPotentialFieldConfig()
    assert config.digest() == config.digest()
    assert ForceCoupledPotentialFieldConfig(repulsive_weight=3.0).digest() != config.digest()


def test_config_is_immutable_and_builds_from_durable_yaml() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "algos"
        / "issue_7889_force_coupled_potential_field.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = build_force_coupled_potential_field_config(payload)
    assert config == public_build_config(payload)
    assert PublicConfig is ForceCoupledPotentialFieldConfig
    assert PublicPlanner is ForceCoupledPotentialFieldPlanner
    assert isinstance(ForceCoupledPotentialFieldPlanner(config), LocalPlannerProtocol)
    with pytest.raises(FrozenInstanceError):
        config.max_linear_speed = 2.0


def test_lifecycle_plan_reset_diagnostics_close() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.reset(seed=42)
    linear, angular = planner.plan(_observation())
    assert math.isfinite(linear) and math.isfinite(angular)
    diagnostics = planner.diagnostics()
    assert diagnostics["planner_type"] == "force_coupled_potential_field"
    assert diagnostics["status"] == "ok"
    assert diagnostics["config_digest"]
    assert len(diagnostics["constrained_command"]) == 2
    planner.close()
    planner.close()  # idempotent
    with pytest.raises(ValueError):
        planner.plan(_observation())


def test_plan_requires_robot_and_goal() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    with pytest.raises(ValueError):
        planner.plan({})
    with pytest.raises(ValueError):
        planner.plan({"robot": [0.0, 0.0, 0.0]})
    assert planner.diagnostics()["status"] == "invalid_input"
    assert planner.diagnostics()["fallback"] is False


def test_plan_fails_closed_on_non_finite_inputs() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    with pytest.raises(ValueError):
        planner.plan(_observation(robot=(float("nan"), 0.0, 0.0)))
    with pytest.raises(ValueError):
        planner.plan(_observation(obstacles=[(1.0, float("inf"))]))


def test_numeric_conversion_overflow_records_invalid_non_finite_diagnostics() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    with pytest.raises(ValueError, match="too large to convert to float"):
        planner.plan(_observation(obstacles=[(10**400, 0)]))
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "invalid_input"
    assert diagnostics["invalid_input"] is True
    assert diagnostics["non_finite_input"] is True
    assert diagnostics["fallback"] is False


def test_attractive_force_points_toward_target() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan(_observation(robot=(0.0, 0.0, 0.0), goal=(4.0, 0.0)))
    diagnostics = planner.diagnostics()
    attractive = np.asarray(diagnostics["attractive_force"])
    assert attractive[0] > 0.0  # toward +x goal
    assert abs(attractive[1]) < 1e-9


def test_repulsive_force_points_away_from_obstacle() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan(
        _observation(
            robot=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=[(1.0, 0.0)],
        )
    )
    diagnostics = planner.diagnostics()
    repulsive = np.asarray(diagnostics["repulsive_force"])
    assert repulsive[0] < 0.0  # away from obstacle at +x


def test_zero_distance_guard_stops_and_reports_degraded() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    command = planner.plan(
        _observation(
            robot=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=[(0.0, 0.0)],
        )
    )
    diagnostics = planner.diagnostics()
    repulsive = np.asarray(diagnostics["repulsive_force"])
    assert command == (0.0, 0.0)
    assert np.all(np.isfinite(repulsive))
    assert diagnostics["zero_distance_guards"] == {"obstacles": 1, "pedestrians": 0}
    assert diagnostics["status"] == "degraded"
    assert diagnostics["step_degraded"] is True
    assert diagnostics["ever_degraded"] is True
    assert diagnostics["emergency_stop"] is True
    assert diagnostics["overlap_stop_requested"] is True
    assert diagnostics["overlap_stop_pending"] is False
    assert diagnostics["overlap_stop_applied"] is True
    assert "zero_distance_stop_requested" in diagnostics["active_constraints"]
    assert "zero_distance_stop" in diagnostics["active_constraints"]
    assert "emergency_stop" in diagnostics["active_constraints"]


def test_zero_distance_overlap_stop_obeys_rate_limit_after_command_ramp() -> None:
    """Overlap handling must preserve the issue's never-exceed rate predicate."""
    config = ForceCoupledPotentialFieldConfig(max_linear_rate=0.8, control_dt=0.2)
    planner = ForceCoupledPotentialFieldPlanner(config)
    for _ in range(10):
        previous_linear, previous_angular = planner.plan(_observation())

    linear, angular = planner.plan(_observation(obstacles=[(0.0, 0.0)]))
    diagnostics = planner.diagnostics()

    assert 0.0 < linear < previous_linear
    assert abs(linear - previous_linear) <= config.max_linear_rate * config.control_dt + 1e-9
    assert abs(angular - previous_angular) <= config.max_angular_rate * config.control_dt + 1e-9
    assert diagnostics["emergency_stop"] is False
    assert diagnostics["overlap_stop_requested"] is True
    assert diagnostics["overlap_stop_pending"] is True
    assert diagnostics["overlap_stop_applied"] is False
    assert "zero_distance_stop_requested" in diagnostics["active_constraints"]
    assert "zero_distance_stop_pending" in diagnostics["active_constraints"]
    assert "linear_rate_limit" in diagnostics["active_constraints"]


def test_runtime_timestep_overrides_configured_rate_limit_timestep() -> None:
    """Map-runner timestep metadata must own the per-step command-rate bound."""
    config = ForceCoupledPotentialFieldConfig(
        max_linear_rate=0.8,
        control_dt=0.2,
    )
    planner = ForceCoupledPotentialFieldPlanner(config)
    observation = _observation()
    observation["sim"] = {"timestep": [0.1]}
    for _ in range(20):
        previous_linear, _ = planner.plan(observation)

    overlap = _observation(obstacles=[(0.0, 0.0)])
    overlap["sim"] = {"timestep": [0.1]}
    linear, _ = planner.plan(overlap)
    diagnostics = planner.diagnostics()

    assert abs(linear - previous_linear) <= config.max_linear_rate * 0.1 + 1e-9
    assert diagnostics["control_dt"] == pytest.approx(0.1)
    assert diagnostics["control_dt_source"] == "observation.sim.timestep"


def test_non_finite_force_overflow_records_fail_closed_diagnostics() -> None:
    """Extreme accepted finite inputs must not escape without invalid diagnostics."""
    planner = ForceCoupledPotentialFieldPlanner(
        ForceCoupledPotentialFieldConfig(repulsive_weight=1e308)
    )

    with pytest.raises(ValueError, match="non-finite force or command computation"):
        planner.plan(_observation(obstacles=[(1e-5, 0.0)]))

    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "invalid_input"
    assert diagnostics["invalid_input"] is True
    assert diagnostics["non_finite_input"] is True
    assert diagnostics["fallback"] is False


@pytest.mark.parametrize("timestep", [0.0, -0.1, float("nan"), [0.1, 0.2]])
def test_invalid_runtime_timestep_fails_closed(timestep: object) -> None:
    observation = _observation()
    observation["sim"] = {"timestep": timestep}
    planner = ForceCoupledPotentialFieldPlanner()

    with pytest.raises(ValueError, match="control timestep"):
        planner.plan(observation)

    assert planner.diagnostics()["status"] == "invalid_input"


def test_goal_reached_stops_without_inventing_a_heading() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan(_observation(goal=(4.0, 0.0)))

    command = planner.plan(_observation(robot=(4.0, 0.0, math.pi / 2), goal=(4.0, 0.0)))
    diagnostics = planner.diagnostics()

    assert command == (0.0, 0.0)
    assert diagnostics["raw_command"] == [0.0, 0.0]
    assert diagnostics["goal_reached"] is True
    assert diagnostics["force_cancellation_guard"] is False
    assert diagnostics["status"] == "ok"
    assert "goal_reached_stop" in diagnostics["active_constraints"]


def test_goal_reached_stop_is_rate_limited_after_command_ramp() -> None:
    """Ordinary goal stops obey the configured command-rate predicate."""
    config = ForceCoupledPotentialFieldConfig(max_linear_rate=0.8, control_dt=0.2)
    planner = ForceCoupledPotentialFieldPlanner(config)
    for _ in range(10):
        previous_linear, previous_angular = planner.plan(_observation())

    linear, angular = planner.plan(_observation(robot=(4.0, 0.0, math.pi / 2), goal=(4.0, 0.0)))
    diagnostics = planner.diagnostics()

    assert 0.0 < linear < previous_linear
    assert abs(linear - previous_linear) <= config.max_linear_rate * config.control_dt + 1e-9
    assert abs(angular - previous_angular) <= config.max_angular_rate * config.control_dt + 1e-9
    assert diagnostics["emergency_stop"] is False
    assert "goal_reached_stop" in diagnostics["active_constraints"]
    assert "linear_rate_limit" in diagnostics["active_constraints"]


def test_force_cancellation_stops_and_reports_degraded() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    command = planner.plan(
        _observation(
            robot=(0.0, 0.0, math.pi / 2),
            goal=(4.0, 0.0),
            # With the default weights and influence radius, d=1.2 m makes
            # the -x repulsion exactly cancel the unit +x attraction.
            obstacles=[(1.2, 0.0)],
        )
    )
    diagnostics = planner.diagnostics()

    assert command == (0.0, 0.0)
    assert diagnostics["raw_command"] == [0.0, 0.0]
    assert diagnostics["goal_reached"] is False
    assert diagnostics["force_cancellation_guard"] is True
    assert diagnostics["status"] == "degraded"
    assert diagnostics["step_degraded"] is True
    assert "force_cancellation_stop" in diagnostics["active_constraints"]


def test_speed_and_rate_limits_are_hard_predicates() -> None:
    config = ForceCoupledPotentialFieldConfig(
        max_linear_speed=0.5,
        max_angular_speed=0.5,
        max_linear_rate=0.2,
        max_angular_rate=0.2,
        control_dt=0.1,
    )
    planner = ForceCoupledPotentialFieldPlanner(config)
    planner.reset()
    linear, angular = planner.plan(_observation(goal=(0.0, 50.0)))
    assert abs(linear) <= config.max_linear_speed + 1e-9
    assert abs(angular) <= config.max_angular_speed + 1e-9
    # Second step: rate limit bounds the change from the first command.
    assert {"linear_rate_limit", "angular_rate_limit"}.issubset(
        planner.diagnostics()["active_constraints"]
    )
    linear2, angular2 = planner.plan(_observation(goal=(0.0, 50.0)))
    assert abs(linear2 - linear) <= config.max_linear_rate * config.control_dt + 1e-9
    assert abs(angular2 - angular) <= config.max_angular_rate * config.control_dt + 1e-9


def test_deterministic_replay_produces_identical_commands() -> None:
    planner_a = ForceCoupledPotentialFieldPlanner()
    planner_b = ForceCoupledPotentialFieldPlanner()
    observations = [
        _observation(robot=(0.0, 0.0, 0.0), goal=(4.0, 0.0)),
        _observation(robot=(0.5, 0.1, 0.2), goal=(4.0, 0.0), obstacles=[(2.0, 0.0)]),
        _observation(robot=(1.0, -0.2, -0.1), goal=(4.0, 0.0), pedestrians=[(2.5, 0.3)]),
    ]
    planner_a.reset(seed=7)
    planner_b.reset(seed=7)
    commands_a = [planner_a.plan(obs) for obs in observations]
    planner_a.reset(seed=7)
    diagnostics_a = []
    replay_a = []
    for obs in observations:
        replay_a.append(planner_a.plan(obs))
        diagnostics_a.append(planner_a.diagnostics())
    diagnostics_b = []
    commands_b = []
    for obs in observations:
        commands_b.append(planner_b.plan(obs))
        diagnostics_b.append(planner_b.diagnostics())
    assert commands_a == replay_a == commands_b
    assert diagnostics_a == diagnostics_b


def test_symmetric_obstacle_fixture_is_deterministic() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    obs = _observation(
        robot=(0.0, 0.0, 0.0),
        goal=(4.0, 0.0),
        obstacles=[(2.0, 0.5), (2.0, -0.5)],
    )
    planner.reset(seed=1)
    first = planner.plan(obs)
    planner.reset(seed=1)
    second = planner.plan(obs)
    assert first == second
    assert first[1] == pytest.approx(0.0)
    assert planner.diagnostics()["repulsive_force"][1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("scenario_id", "seed", "obstacles", "pedestrians"),
    [
        ("analytic_static_obstacle", 1, [(1.0, 0.5)], []),
        ("analytic_pedestrian_interaction", 7, [], [(1.0, 0.0)]),
    ],
)
def test_fixed_smoke_scenarios(
    scenario_id: str,
    seed: int,
    obstacles: list[tuple[float, float]],
    pedestrians: list[tuple[float, float]],
) -> None:
    """Replay the two durable issue #7889 smoke-receipt fixtures."""
    planner = ForceCoupledPotentialFieldPlanner()
    planner.reset(seed=seed)
    command = planner.plan(_observation(obstacles=obstacles, pedestrians=pedestrians))
    assert scenario_id in {"analytic_static_obstacle", "analytic_pedestrian_interaction"}
    assert command == pytest.approx((0.16, -0.3))
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "ok"
    assert diagnostics["degraded"] is False
    assert diagnostics["fallback"] is False


def test_rotation_and_translation_transform_force_and_command_consistently() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    base = _observation(
        robot=(0.0, 0.0, 0.0),
        goal=(4.0, 0.0),
        obstacles=[(1.0, 0.5)],
    )
    transformed = _observation(
        robot=(3.0, -2.0, math.pi / 2),
        goal=(3.0, 2.0),
        obstacles=[(2.5, -1.0)],
    )
    planner.reset(seed=1)
    base_command = planner.plan(base)
    base_force = np.asarray(planner.diagnostics()["total_force"])
    planner.reset(seed=1)
    transformed_command = planner.plan(transformed)
    transformed_force = np.asarray(planner.diagnostics()["total_force"])
    expected_rotated_force = np.asarray([-base_force[1], base_force[0]])
    assert transformed_force == pytest.approx(expected_rotated_force)
    assert transformed_command == pytest.approx(base_command)


def test_missing_required_inputs_do_not_produce_nominal_success() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    with pytest.raises(ValueError):
        planner.plan(_observation(goal=(4.0, 0.0), pedestrians=[(1.0, float("nan"))]))
    # A malformed obstacle payload that cannot reshape to (N, 2) fails closed.
    with pytest.raises(ValueError):
        planner.plan(
            {
                "robot": [0.0, 0.0, 0.0],
                "goal": [4.0, 0.0],
                "obstacles": {"positions": [1.0, 2.0, 3.0]},
            }
        )
    assert planner.diagnostics()["status"] == "invalid_input"


@pytest.mark.parametrize("count", [-1, 1.5, [], [2]])
def test_malformed_pedestrian_count_fails_closed(count: object) -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    observation = _observation(pedestrians=[(1.0, 0.0)])
    observation["pedestrians"]["count"] = count
    with pytest.raises(ValueError, match="pedestrian count"):
        planner.plan(observation)
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "invalid_input"
    assert diagnostics["invalid_input"] is True


def test_malformed_visibility_mapping_fails_closed() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    observation = _observation()
    observation["obstacles"] = {}
    with pytest.raises(ValueError, match="obstacles mapping requires positions"):
        planner.plan(observation)


def test_flat_and_nested_socnav_observations_are_supported() -> None:
    flat = {
        "robot_position": [0.0, 0.0],
        "robot_heading": [0.0],
        "goal_current": [4.0, 0.0],
        "obstacles_positions": [[1.5, 0.5]],
        "pedestrians_positions": [[1.0, 0.0]],
        "pedestrians_count": [1],
    }
    nested = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "obstacles": {"positions": [[1.5, 0.5]]},
        "pedestrians": {"positions": [[1.0, 0.0]], "count": [1]},
    }
    planner = ForceCoupledPotentialFieldPlanner()
    planner.reset(seed=7)
    flat_command = planner.plan(flat)
    planner.reset(seed=7)
    assert planner.plan(nested) == flat_command
    assert planner.diagnostics()["status"] == "ok"


def test_missing_optional_visibility_is_explicitly_degraded() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan({"robot": [0.0, 0.0, 0.0], "goal": [4.0, 0.0]})
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "degraded"
    assert diagnostics["degraded"] is True
    assert diagnostics["missing_inputs"] == ["obstacles", "pedestrians"]


def test_degraded_status_is_sticky_until_reset() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan({"robot": [0.0, 0.0, 0.0], "goal": [4.0, 0.0]})

    planner.plan(_observation())
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "degraded"
    assert diagnostics["step_degraded"] is False
    assert diagnostics["ever_degraded"] is True
    assert diagnostics["status_reason"].startswith("episode previously degraded")

    planner.reset(seed=7)
    planner.plan(_observation())
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "ok"
    assert diagnostics["ever_degraded"] is False


def test_ego_occupancy_grid_supplies_static_obstacles() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    grid = np.zeros((3, 9, 9), dtype=np.float32)
    grid[0, 4, 5] = 1.0
    observation = {
        "robot": {"position": [3.0, -2.0], "heading": [math.pi / 2]},
        "goal": {"current": [3.0, 2.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-4.5, -4.5],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [9.0, 9.0],
        "occupancy_grid_meta_use_ego_frame": [1.0],
        "occupancy_grid_meta_center_on_robot": [1.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [3.0, -2.0, math.pi / 2],
    }

    planner.plan(observation)
    diagnostics = planner.diagnostics()
    obstacle_force = np.asarray(diagnostics["obstacle_repulsive_force"])
    assert diagnostics["status"] == "ok"
    assert diagnostics["missing_inputs"] == []
    assert abs(obstacle_force[0]) < 1e-9
    assert obstacle_force[1] < 0.0


def test_occupied_robot_grid_cell_fails_closed_without_inventing_exact_overlap() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    grid = np.zeros((3, 9, 9), dtype=np.float32)
    grid[0, 4, 4] = 1.0
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-0.8, -0.8],
        "occupancy_grid_meta_resolution": [0.2],
        "occupancy_grid_meta_size": [1.8, 1.8],
        "occupancy_grid_meta_use_ego_frame": [0.0],
        "occupancy_grid_meta_center_on_robot": [0.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match="occupied static-obstacle cell"):
        planner.plan(observation)
    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "invalid_input"
    assert diagnostics["invalid_input"] is True
    assert diagnostics["fallback"] is False
    assert "zero_distance_guards" not in diagnostics


def test_out_of_bounds_robot_grid_fails_closed_without_inventing_overlap() -> None:
    """An unusable supplied grid is invalid input, not overlap or nominal visibility."""
    planner = ForceCoupledPotentialFieldPlanner()
    observation = {
        "robot": {"position": [10.0, 10.0], "heading": [0.0]},
        "goal": {"current": [11.0, 10.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": np.zeros((3, 3, 3), dtype=np.float32),
        "occupancy_grid_meta_origin": [0.0, 0.0],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [3.0, 3.0],
        "occupancy_grid_meta_use_ego_frame": [0.0],
        "occupancy_grid_meta_center_on_robot": [0.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [10.0, 10.0, 0.0],
    }

    with pytest.raises(ValueError, match="robot pose lies outside supplied occupancy grid"):
        planner.plan(observation)

    diagnostics = planner.diagnostics()
    assert diagnostics["status"] == "invalid_input"
    assert diagnostics["invalid_input"] is True
    assert diagnostics["fallback"] is False


@pytest.mark.parametrize("invalid_value", [float("nan"), -0.1, 1.1])
def test_invalid_occupancy_grid_values_fail_closed(invalid_value: float) -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    grid = np.zeros((3, 3, 3), dtype=np.float32)
    grid[0, 1, 1] = invalid_value
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-1.25, -1.25],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [2.5, 2.5],
        "occupancy_grid_meta_use_ego_frame": [1.0],
        "occupancy_grid_meta_center_on_robot": [1.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match=r"finite and within \[0, 1\]"):
        planner.plan(observation)
    assert planner.diagnostics()["status"] == "invalid_input"


@pytest.mark.parametrize(
    "grid_update",
    [
        {"occupancy_grid": np.zeros((3, 3), dtype=np.float32)},
        {
            "occupancy_grid": np.zeros((3, 3, 3), dtype=np.float32),
            "occupancy_grid_meta_origin": None,
            "occupancy_grid_meta_resolution": None,
            "occupancy_grid_meta_size": None,
            "occupancy_grid_meta_use_ego_frame": None,
            "occupancy_grid_meta_center_on_robot": None,
            "occupancy_grid_meta_channel_indices": None,
            "occupancy_grid_meta_robot_pose": None,
        },
    ],
)
def test_supplied_malformed_occupancy_grid_fails_closed(grid_update: dict[str, object]) -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": np.zeros((3, 3, 3), dtype=np.float32),
        "occupancy_grid_meta_origin": [-1.5, -1.5],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [3.0, 3.0],
        "occupancy_grid_meta_use_ego_frame": [1.0],
        "occupancy_grid_meta_center_on_robot": [1.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }
    observation.update(grid_update)

    with pytest.raises(ValueError, match="occupancy grid"):
        planner.plan(observation)
    assert planner.diagnostics()["status"] == "invalid_input"


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("origin", [0.0]),
        ("resolution", [0.0]),
        ("size", [float("nan"), 3.0]),
        ("channel_indices", [0.5, 1.0, -1.0, 2.0]),
        ("robot_pose", [0.0, 0.0]),
    ],
)
def test_malformed_occupancy_grid_metadata_fails_closed(
    metadata_key: str, metadata_value: list[float]
) -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    grid = np.zeros((3, 3, 3), dtype=np.float32)
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-1.25, -1.25],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [2.5, 2.5],
        "occupancy_grid_meta_use_ego_frame": [1.0],
        "occupancy_grid_meta_center_on_robot": [1.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }
    observation[f"occupancy_grid_meta_{metadata_key}"] = metadata_value

    with pytest.raises(ValueError, match="occupancy grid"):
        planner.plan(observation)
    assert planner.diagnostics()["status"] == "invalid_input"


def test_combined_grid_without_explicit_obstacle_channel_stays_degraded() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    grid = np.zeros((3, 3, 3), dtype=np.float32)
    grid[2, 1, 1] = 1.0  # combined channel contains robot occupancy, not a static obstacle
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {"positions": [], "count": [0]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-1.25, -1.25],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [2.5, 2.5],
        "occupancy_grid_meta_use_ego_frame": [1.0],
        "occupancy_grid_meta_center_on_robot": [1.0],
        "occupancy_grid_meta_channel_indices": [-1, -1, 0, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }

    command = planner.plan(observation)
    diagnostics = planner.diagnostics()
    assert command != (0.0, 0.0)
    assert diagnostics["status"] == "degraded"
    assert diagnostics["missing_inputs"] == ["obstacles"]
    assert diagnostics["zero_distance_guards"]["obstacles"] == 0


def test_pedestrian_repulsion_within_observation_contract() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan(
        _observation(
            robot=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            pedestrians=[(1.0, 0.0)],
        )
    )
    repulsive = np.asarray(planner.diagnostics()["repulsive_force"])
    assert repulsive[0] < 0.0
    obstacle = np.asarray(planner.diagnostics()["obstacle_repulsive_force"])
    pedestrian = np.asarray(planner.diagnostics()["pedestrian_repulsive_force"])
    assert np.allclose(obstacle, [0.0, 0.0])
    assert pedestrian[0] < 0.0
