"""Analytic, lifecycle, and deterministic-smoke tests for the force-coupled planner."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.planner.force_coupled_potential_field import (
    ForceCoupledPotentialFieldConfig,
    ForceCoupledPotentialFieldPlanner,
)


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


def test_config_digest_is_stable() -> None:
    config = ForceCoupledPotentialFieldConfig()
    assert config.digest() == config.digest()
    assert ForceCoupledPotentialFieldConfig(repulsive_weight=3.0).digest() != config.digest()


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


def test_plan_fails_closed_on_non_finite_inputs() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    with pytest.raises(ValueError):
        planner.plan(_observation(robot=(float("nan"), 0.0, 0.0)))
    with pytest.raises(ValueError):
        planner.plan(_observation(obstacles=[(1.0, float("inf"))]))


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


def test_zero_distance_guard_reports_finite_force() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    planner.plan(
        _observation(
            robot=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            obstacles=[(0.0, 0.0)],
        )
    )
    diagnostics = planner.diagnostics()
    repulsive = np.asarray(diagnostics["repulsive_force"])
    assert np.all(np.isfinite(repulsive))


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
    linear, angular = planner.plan(_observation(goal=(50.0, 0.0)))
    assert abs(linear) <= config.max_linear_speed + 1e-9
    assert abs(angular) <= config.max_angular_speed + 1e-9
    # Second step: rate limit bounds the change from the first command.
    linear2, angular2 = planner.plan(_observation(goal=(50.0, 0.0)))
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
    for obs in observations:
        planner_a.reset(seed=7)
        planner_b.reset(seed=7)
        assert planner_a.plan(obs) == planner_b.plan(obs)
        assert planner_a.diagnostics() == planner_b.diagnostics()


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


def test_rotation_transforms_force_consistently() -> None:
    planner = ForceCoupledPotentialFieldPlanner()
    # Robot facing +y toward a goal at +y: attractive force points +y.
    planner.plan(_observation(robot=(0.0, 0.0, math.pi / 2), goal=(0.0, 4.0)))
    diagnostics = planner.diagnostics()
    attractive = np.asarray(diagnostics["attractive_force"])
    assert attractive[1] > 0.0
    assert abs(attractive[0]) < 1e-9


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
