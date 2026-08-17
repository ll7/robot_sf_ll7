"""Tests for the opt-in predictive Gaussian human-cost primitive."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
from robot_sf.planner.predictive_human_cost import (
    PredictiveGaussianHumanCost,
    PredictiveGaussianHumanCostConfig,
    build_predictive_gaussian_human_cost_config,
)


def _pedestrians() -> tuple[np.ndarray, np.ndarray]:
    """Return one moving pedestrian with a second stationary pedestrian."""

    return (
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=float),
    )


def test_predictive_cost_is_disabled_by_default() -> None:
    """The default primitive contributes no hidden planner objective term."""

    positions, velocities = _pedestrians()
    model = PredictiveGaussianHumanCost()
    assert model.evaluate(np.asarray([1.0, 0.0]), positions, velocities, time_s=1.0) == 0.0


def test_predictive_cost_aligns_long_axis_with_motion() -> None:
    """A point ahead of motion is penalized more than an equally distant lateral point."""

    cfg = PredictiveGaussianHumanCostConfig(
        enabled=True,
        longitudinal_sigma_m=0.8,
        lateral_sigma_m=0.3,
        forward_speed_gain=0.0,
    )
    model = PredictiveGaussianHumanCost(cfg)
    positions = np.asarray([[0.0, 0.0]], dtype=float)
    velocities = np.asarray([[1.0, 0.0]], dtype=float)
    forward = model.evaluate(np.asarray([1.5, 0.0]), positions, velocities, time_s=1.0)
    lateral = model.evaluate(np.asarray([1.0, 0.5]), positions, velocities, time_s=1.0)
    assert forward > lateral


def test_predictive_cost_batch_and_trajectory_shapes_are_deterministic() -> None:
    """Batch evaluation agrees with independent trajectory evaluations."""

    cfg = PredictiveGaussianHumanCostConfig(enabled=True, forward_speed_gain=0.5)
    model = PredictiveGaussianHumanCost(cfg)
    positions, velocities = _pedestrians()
    trajectories = np.asarray(
        [
            [[0.4, 0.0], [0.8, 0.0]],
            [[0.0, 0.4], [0.0, 0.8]],
        ],
        dtype=float,
    )
    batch = model.evaluate_trajectory(trajectories, positions, velocities, dt=0.2)
    expected = np.asarray(
        [
            model.evaluate_trajectory(trajectory, positions, velocities, dt=0.2)
            for trajectory in trajectories
        ]
    )
    np.testing.assert_allclose(batch, expected, atol=1e-12, rtol=0.0)


def test_predictive_cost_configuration_rejects_unknown_keys() -> None:
    """A misspelled tuning key cannot silently alter a diagnostic."""

    with pytest.raises(ValueError, match="unknown predictive human cost keys"):
        build_predictive_gaussian_human_cost_config({"lateral_sigma": 0.5})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enabled": 1}, "enabled must be boolean"),
        ({"weight": float("nan")}, "weight must be finite"),
        ({"weight": -1.0}, "weight must be non-negative"),
        ({"longitudinal_sigma_m": 0.0}, "longitudinal_sigma_m must be positive"),
        ({"lateral_sigma_m": 0.0}, "lateral_sigma_m must be positive"),
        ({"forward_speed_gain": -1.0}, "forward_speed_gain must be non-negative"),
        ({"stationary_heading_rad": float("nan")}, "stationary_heading_rad must be finite"),
    ],
)
def test_predictive_cost_configuration_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    """Non-finite and non-physical tuning values fail closed."""

    with pytest.raises(ValueError, match=message):
        PredictiveGaussianHumanCostConfig(**kwargs)


def test_predictive_cost_configuration_builds_and_serializes() -> None:
    """Nested configuration parsing preserves explicit JSON-facing values."""

    assert build_predictive_gaussian_human_cost_config(None).enabled is False
    assert build_predictive_gaussian_human_cost_config({"weight": 2.0}).weight == 2.0
    config = build_predictive_gaussian_human_cost_config(
        {
            "enabled": True,
            "weight": 1,
            "longitudinal_sigma_m": 0.7,
            "lateral_sigma_m": 0.4,
            "forward_speed_gain": 0.5,
            "stationary_heading_rad": 0.25,
        }
    )
    assert config.to_dict() == {
        "enabled": True,
        "weight": 1.0,
        "longitudinal_sigma_m": 0.7,
        "lateral_sigma_m": 0.4,
        "forward_speed_gain": 0.5,
        "stationary_heading_rad": 0.25,
    }


@pytest.mark.parametrize("payload", ["invalid", {"enabled": 1}])
def test_predictive_cost_configuration_rejects_invalid_nested_payload(
    payload: object,
) -> None:
    """Nested planner configuration remains strict about its container and types."""

    with pytest.raises(ValueError):
        build_predictive_gaussian_human_cost_config(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("positions", "velocities", "message"),
    [
        (np.zeros(2), np.zeros((1, 2)), r"shape \(P, 2\)"),
        (np.zeros((1, 2)), np.zeros((2, 2)), "match pedestrian_positions"),
        (np.asarray([[np.nan, 0.0]]), np.zeros((1, 2)), "must be finite"),
    ],
)
def test_predictive_cost_rejects_invalid_pedestrian_arrays(
    positions: np.ndarray, velocities: np.ndarray, message: str
) -> None:
    """Pedestrian state must be a finite pair of aligned two-dimensional arrays."""

    with pytest.raises(ValueError, match=message):
        PredictiveGaussianHumanCost().evaluate(
            np.asarray([0.0, 0.0]), positions, velocities, time_s=0.0
        )


@pytest.mark.parametrize(
    ("robot_positions", "time_s", "message"),
    [
        (np.zeros((1, 3)), 0.0, r"shape \(2,\) or \(N, 2\)"),
        (np.asarray([[np.nan, 0.0]]), 0.0, "robot_positions must be finite"),
        (np.asarray([0.0, 0.0]), -1.0, "time_s must be finite and non-negative"),
    ],
)
def test_predictive_cost_rejects_invalid_robot_inputs(
    robot_positions: np.ndarray, time_s: float, message: str
) -> None:
    """Robot rollout points and prediction time must be finite and shaped."""

    with pytest.raises(ValueError, match=message):
        PredictiveGaussianHumanCost().evaluate(
            robot_positions,
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            time_s=time_s,
        )


@pytest.mark.parametrize(
    ("trajectory", "dt", "message"),
    [
        (np.zeros(2), 0.1, r"shape \(T, 2\) or \(N, T, 2\)"),
        (np.zeros((0, 2)), 0.1, "one or more xy rollout points"),
        (np.asarray([[np.nan, 0.0]]), 0.1, "robot_positions must be finite"),
        (np.zeros((1, 2)), 0.0, "dt must be finite and positive"),
    ],
)
def test_predictive_cost_rejects_invalid_trajectory_inputs(
    trajectory: np.ndarray, dt: float, message: str
) -> None:
    """Trajectory evaluation fails closed for invalid shapes, values, and time steps."""

    with pytest.raises(ValueError, match=message):
        PredictiveGaussianHumanCost().evaluate_trajectory(
            trajectory,
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            dt=dt,
        )


def test_mppi_enabled_cost_preserves_scalar_batch_parity() -> None:
    """The opt-in cost is included identically in scalar and batched rollouts."""

    cfg = MPPISocialConfig(
        random_seed=7319,
        sample_count=8,
        iterations=1,
        horizon_steps=3,
        progress_escape_enabled=False,
        predictive_human_cost=PredictiveGaussianHumanCostConfig(enabled=True, weight=1.7),
    )
    planner = MPPISocialPlannerAdapter(cfg)
    observation = {
        "robot": {
            "position": np.asarray([0.0, 0.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.2]),
            "radius": np.asarray([0.25]),
        },
        "goal": {"current": np.asarray([2.0, 0.0]), "next": np.asarray([2.0, 0.0])},
        "pedestrians": {
            "positions": np.asarray([[0.8, 0.2], [1.1, -0.2]]),
            "velocities": np.asarray([[0.0, -0.2], [0.0, 0.2]]),
            "count": np.asarray([2.0]),
            "radius": 0.25,
        },
    }
    rng = np.random.default_rng(7319)
    batch = rng.normal(0.2, 0.1, size=(4, 3, 2))
    batch[:, :, 0] = np.clip(batch[:, :, 0], 0.0, cfg.max_linear_speed)
    batch[:, :, 1] = np.clip(batch[:, :, 1], -cfg.max_angular_speed, cfg.max_angular_speed)
    robot_pos, heading, speed, goal, ped_pos, ped_vel = planner._extract_state(observation)
    grid_payload = planner._cache_grid_payload(observation)
    scalar = np.asarray(
        [
            planner._sequence_cost(
                sequence=sequence,
                robot_pos=robot_pos,
                heading=heading,
                current_speed=speed,
                goal=goal,
                ped_pos=ped_pos,
                ped_vel=ped_vel,
                observation=observation,
                grid_payload=grid_payload,
            )
            for sequence in batch
        ]
    )
    batched = planner._batch_sequence_cost(
        batch=batch,
        robot_pos=robot_pos,
        heading=heading,
        current_speed=speed,
        goal=goal,
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        observation=observation,
        grid_payload=grid_payload,
    )
    np.testing.assert_allclose(scalar, batched, atol=1e-12, rtol=0.0)
