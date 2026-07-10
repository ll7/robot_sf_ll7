"""Contract tests for the classical Dynamic Window Approach baseline."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.dwa import DWAPlannerAdapter, DWAPlannerConfig, build_dwa_config


def _observation(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
    angular_velocity: float = 0.0,
    goal: tuple[float, float] = (3.0, 0.0),
    pedestrians: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a minimal structured observation accepted by DWA."""
    positions = [] if pedestrians is None else pedestrians
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "angular_velocity": np.asarray([angular_velocity], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(positions, dtype=float),
            "count": np.asarray([len(positions)], dtype=float),
        },
    }


def test_dwa_samples_only_dynamically_reachable_commands() -> None:
    """Selected velocity stays inside the configured acceleration window."""
    config = DWAPlannerConfig(
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        max_linear_acceleration=0.5,
        max_angular_acceleration=1.0,
        control_dt=0.2,
        linear_samples=5,
        angular_samples=5,
    )
    command = DWAPlannerAdapter(config).plan(
        _observation(speed=0.5, angular_velocity=0.2, goal=(3.0, 0.0))
    )
    assert 0.4 <= command[0] <= 0.6
    assert 0.0 <= command[1] <= 0.4


def test_dwa_is_deterministic_and_goal_directed() -> None:
    """Identical structured state produces the same bounded, forward command."""
    config = DWAPlannerConfig()
    observation = _observation(goal=(3.0, 0.0))
    first = DWAPlannerAdapter(config).plan(observation)
    second = DWAPlannerAdapter(config).plan(observation)
    assert first == second
    assert 0.0 < first[0] <= config.max_linear_speed
    assert abs(first[1]) <= config.max_angular_speed


def test_dwa_stops_at_goal_and_rejects_unsafe_forward_rollouts() -> None:
    """The baseline stops at the goal and does not select a colliding forward command."""
    config = DWAPlannerConfig(goal_tolerance=0.3, linear_samples=3, angular_samples=3)
    planner = DWAPlannerAdapter(config)
    assert planner.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)

    command = planner.plan(_observation(goal=(3.0, 0.0), pedestrians=[(0.35, 0.0)]))
    assert command[0] == pytest.approx(0.0)


def test_dwa_config_builder_applies_explicit_acceleration_parameters() -> None:
    """The canonical config parser preserves DWA-specific dynamic-window settings."""
    config = build_dwa_config(
        {
            "max_linear_acceleration": 0.4,
            "max_angular_acceleration": 0.9,
            "linear_samples": 4,
            "angular_samples": 6,
        }
    )
    assert config.max_linear_acceleration == pytest.approx(0.4)
    assert config.max_angular_acceleration == pytest.approx(0.9)
    assert config.linear_samples == 4
    assert config.angular_samples == 6


def test_dwa_dynamic_window_preserves_reachability_outside_speed_limits() -> None:
    """Out-of-range current commands collapse to the nearest dynamically reachable value."""
    planner = DWAPlannerAdapter(
        DWAPlannerConfig(
            max_linear_speed=1.0,
            max_angular_speed=1.0,
            max_linear_acceleration=0.5,
            max_angular_acceleration=1.0,
            control_dt=0.2,
        )
    )

    assert planner._dynamic_window(1.5, -2.0) == pytest.approx((1.4, 1.4, -1.8, -1.8))


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_linear_speed": float("nan")},
        {"max_angular_acceleration": -0.1},
        {"control_dt": 0.0},
        {"prediction_steps": 0},
    ],
)
def test_dwa_rejects_invalid_runtime_configuration(overrides: dict[str, float]) -> None:
    """The experimental planner fails closed before it can emit invalid commands."""
    with pytest.raises(ValueError):
        DWAPlannerConfig(**overrides)
