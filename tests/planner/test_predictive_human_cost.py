"""Tests for the opt-in predictive Gaussian human-cost primitive."""

from __future__ import annotations

import json

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
        "cutoff_distance_m": None,
        "aggregation": "sum",
    }
    json.dumps(config.to_dict(), allow_nan=False)


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


# ---------------------------------------------------------------------------
# Stationary / near-zero limiting
# ---------------------------------------------------------------------------


def test_predictive_cost_stationary_pedestrian_uses_fixed_heading() -> None:
    """A stationary pedestrian uses stationary_heading_rad for the long axis."""

    cfg = PredictiveGaussianHumanCostConfig(
        enabled=True,
        longitudinal_sigma_m=0.5,
        lateral_sigma_m=0.3,
        forward_speed_gain=0.0,
        stationary_heading_rad=0.0,
    )
    model = PredictiveGaussianHumanCost(cfg)
    pos = np.asarray([[0.0, 0.0]], dtype=float)
    vel = np.asarray([[0.0, 0.0]], dtype=float)
    cost = model.evaluate(np.asarray([0.5, 0.0]), pos, vel, time_s=0.0)
    assert cost > 0.0


def test_predictive_cost_near_zero_velocity_treated_as_stationary() -> None:
    """Velocity below 1e-9 is treated as stationary."""

    cfg = PredictiveGaussianHumanCostConfig(
        enabled=True,
        longitudinal_sigma_m=0.5,
        lateral_sigma_m=0.3,
        forward_speed_gain=0.0,
        stationary_heading_rad=0.0,
    )
    model = PredictiveGaussianHumanCost(cfg)
    pos = np.asarray([[0.0, 0.0]], dtype=float)
    vel = np.asarray([[1e-10, 0.0]], dtype=float)
    cost = model.evaluate(np.asarray([0.5, 0.0]), pos, vel, time_s=0.0)
    assert cost > 0.0


# ---------------------------------------------------------------------------
# Velocity-aligned ellipse (anisotropic)
# ---------------------------------------------------------------------------


def test_predictive_cost_anisotropic_ellipse_differentiates_long_lateral() -> None:
    """With zero forward_speed_gain the longitudinal and lateral sigmas differ."""

    cfg = PredictiveGaussianHumanCostConfig(
        enabled=True,
        longitudinal_sigma_m=1.0,
        lateral_sigma_m=0.2,
        forward_speed_gain=0.0,
    )
    model = PredictiveGaussianHumanCost(cfg)
    pos = np.asarray([[0.0, 0.0]], dtype=float)
    vel = np.asarray([[1.0, 0.0]], dtype=float)
    forward = model.evaluate(np.asarray([0.5, 0.0]), pos, vel, time_s=1.0)
    lateral = model.evaluate(np.asarray([0.0, 0.5]), pos, vel, time_s=1.0)
    assert forward > lateral


# ---------------------------------------------------------------------------
# Rotation / translation equivariance
# ---------------------------------------------------------------------------


def test_predictive_cost_translation_equivariance() -> None:
    """Translating both robot and pedestrians by the same vector yields the same cost."""

    cfg = PredictiveGaussianHumanCostConfig(enabled=True, forward_speed_gain=0.0)
    model = PredictiveGaussianHumanCost(cfg)
    offset = np.asarray([5.0, -3.0], dtype=float)
    robot = np.asarray([1.0, 0.5], dtype=float)
    ped_pos = np.asarray([[2.0, 1.0]], dtype=float)
    ped_vel = np.asarray([[0.3, 0.0]], dtype=float)
    c1 = model.evaluate(robot, ped_pos, ped_vel, time_s=0.5)
    c2 = model.evaluate(robot + offset, ped_pos + offset, ped_vel, time_s=0.5)
    assert abs(c1 - c2) < 1e-12


def test_predictive_cost_rotation_equivariance() -> None:
    """Rotating robot, pedestrians, and velocities by the same angle yields the same cost."""

    cfg = PredictiveGaussianHumanCostConfig(enabled=True, forward_speed_gain=0.0)
    model = PredictiveGaussianHumanCost(cfg)
    angle = 1.2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    robot = np.asarray([1.0, 0.5], dtype=float)
    ped_pos = np.asarray([[2.0, 1.0]], dtype=float)
    ped_vel = np.asarray([[0.3, 0.0]], dtype=float)
    c1 = model.evaluate(robot, ped_pos, ped_vel, time_s=0.5)
    c2 = model.evaluate(rot @ robot, ped_pos @ rot.T, ped_vel @ rot.T, time_s=0.5)
    assert abs(c1 - c2) < 1e-10


# ---------------------------------------------------------------------------
# Cutoff behavior
# ---------------------------------------------------------------------------


def test_predictive_cost_cutoff_clips_far_pedestrians() -> None:
    """Pedestrians beyond cutoff_distance_m contribute zero cost."""

    cfg_far = PredictiveGaussianHumanCostConfig(enabled=True, cutoff_distance_m=None)
    cfg_near = PredictiveGaussianHumanCostConfig(enabled=True, cutoff_distance_m=0.5)
    model_far = PredictiveGaussianHumanCost(cfg_far)
    model_near = PredictiveGaussianHumanCost(cfg_near)
    robot = np.asarray([10.0, 0.0], dtype=float)
    ped_pos = np.asarray([[0.0, 0.0]], dtype=float)
    ped_vel = np.asarray([[0.0, 0.0]], dtype=float)
    cost_far = model_far.evaluate(robot, ped_pos, ped_vel, time_s=0.0)
    cost_near = model_near.evaluate(robot, ped_pos, ped_vel, time_s=0.0)
    assert cost_far > 0.0
    assert cost_near == 0.0


# ---------------------------------------------------------------------------
# Aggregation modes
# ---------------------------------------------------------------------------


def test_predictive_cost_aggregation_sum() -> None:
    """Sum aggregation adds per-pedestrian contributions."""

    cfg = PredictiveGaussianHumanCostConfig(enabled=True, aggregation="sum", forward_speed_gain=0.0)
    model = PredictiveGaussianHumanCost(cfg)
    robot = np.asarray([0.5, 0.0], dtype=float)
    ped_pos = np.asarray([[0.0, 0.0]], dtype=float)
    ped_vel = np.asarray([[0.0, 0.0]], dtype=float)
    single = model.evaluate(robot, ped_pos, ped_vel, time_s=0.0)
    ped_pos2 = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    ped_vel2 = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    double = model.evaluate(robot, ped_pos2, ped_vel2, time_s=0.0)
    assert abs(double - 2 * single) < 1e-12


def test_predictive_cost_aggregation_max() -> None:
    """Max aggregation takes the largest per-pedestrian contribution."""

    cfg = PredictiveGaussianHumanCostConfig(enabled=True, aggregation="max", forward_speed_gain=0.0)
    model = PredictiveGaussianHumanCost(cfg)
    robot = np.asarray([0.5, 0.0], dtype=float)
    ped_pos = np.asarray([[0.0, 0.0]], dtype=float)
    ped_vel = np.asarray([[0.0, 0.0]], dtype=float)
    single = model.evaluate(robot, ped_pos, ped_vel, time_s=0.0)
    ped_pos2 = np.asarray([[0.0, 0.0], [5.0, 5.0]], dtype=float)
    ped_vel2 = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    multi = model.evaluate(robot, ped_pos2, ped_vel2, time_s=0.0)
    assert abs(multi - single) < 1e-12


def test_predictive_cost_aggregation_mean() -> None:
    """Mean aggregation averages per-pedestrian contributions."""

    cfg = PredictiveGaussianHumanCostConfig(
        enabled=True, aggregation="mean", forward_speed_gain=0.0
    )
    model = PredictiveGaussianHumanCost(cfg)
    robot = np.asarray([0.5, 0.0], dtype=float)
    ped_pos = np.asarray([[0.0, 0.0]], dtype=float)
    ped_vel = np.asarray([[0.0, 0.0]], dtype=float)
    single = model.evaluate(robot, ped_pos, ped_vel, time_s=0.0)
    ped_pos2 = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    ped_vel2 = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    double = model.evaluate(robot, ped_pos2, ped_vel2, time_s=0.0)
    assert abs(double - single) < 1e-12


# ---------------------------------------------------------------------------
# Fail-closed config diagnostics
# ---------------------------------------------------------------------------


def test_predictive_cost_cutoff_negative_rejects() -> None:
    """Negative cutoff_distance_m is rejected."""

    with pytest.raises(ValueError, match="cutoff_distance_m"):
        PredictiveGaussianHumanCostConfig(enabled=True, cutoff_distance_m=-1.0)


def test_predictive_cost_cutoff_zero_rejects() -> None:
    """Zero cutoff_distance_m is rejected."""

    with pytest.raises(ValueError, match="cutoff_distance_m"):
        PredictiveGaussianHumanCostConfig(enabled=True, cutoff_distance_m=0.0)


def test_predictive_cost_aggregation_invalid_rejects() -> None:
    """Unsupported aggregation mode is rejected."""

    with pytest.raises(ValueError, match="aggregation must be one of"):
        PredictiveGaussianHumanCostConfig(enabled=True, aggregation="invalid")


# ---------------------------------------------------------------------------
# Opt-in / default preservation
# ---------------------------------------------------------------------------


def test_predictive_cost_disabled_by_default_preserves_zero() -> None:
    """Default config is disabled and produces zero cost."""

    cfg = PredictiveGaussianHumanCostConfig()
    assert cfg.enabled is False
    model = PredictiveGaussianHumanCost(cfg)
    cost = model.evaluate(
        np.asarray([0.0, 0.0]),
        np.asarray([[1.0, 1.0]]),
        np.asarray([[0.1, 0.1]]),
        time_s=0.5,
    )
    assert cost == 0.0


def test_predictive_cost_enabled_requires_explicit_opt_in() -> None:
    """Cost is non-zero only when explicitly enabled."""

    cfg_disabled = PredictiveGaussianHumanCostConfig(enabled=False, weight=5.0)
    cfg_enabled = PredictiveGaussianHumanCostConfig(enabled=True, weight=5.0)
    robot = np.asarray([0.0, 0.0], dtype=float)
    ped_pos = np.asarray([[0.1, 0.0]], dtype=float)
    ped_vel = np.asarray([[0.0, 0.0]], dtype=float)
    assert (
        PredictiveGaussianHumanCost(cfg_disabled).evaluate(robot, ped_pos, ped_vel, time_s=0.0)
        == 0.0
    )
    assert (
        PredictiveGaussianHumanCost(cfg_enabled).evaluate(robot, ped_pos, ped_vel, time_s=0.0) > 0.0
    )


# ---------------------------------------------------------------------------
# Deterministic smoke receipt
# ---------------------------------------------------------------------------


def test_anisotropic_cost_deterministic_smoke() -> None:
    """Same config and inputs produce identical outputs across two runs."""

    cfg = PredictiveGaussianHumanCostConfig(
        enabled=True,
        weight=1.5,
        longitudinal_sigma_m=0.6,
        lateral_sigma_m=0.35,
        forward_speed_gain=0.4,
        cutoff_distance_m=2.5,
        aggregation="sum",
    )
    model = PredictiveGaussianHumanCost(cfg)
    robot = np.asarray([0.3, -0.1], dtype=float)
    ped_pos = np.asarray([[1.0, 0.2], [0.5, -0.3]], dtype=float)
    ped_vel = np.asarray([[0.2, 0.1], [-0.1, 0.0]], dtype=float)
    c1 = model.evaluate(robot, ped_pos, ped_vel, time_s=0.5)
    c2 = model.evaluate(robot, ped_pos, ped_vel, time_s=0.5)
    assert c1 == c2
    assert np.isfinite(c1)
    assert c1 > 0.0


def test_mppi_anisotropic_cost_deterministic_smoke_receipt() -> None:
    """MPPI with anisotropic cost config produces a deterministic command across fresh planners."""

    cfg = MPPISocialConfig(
        random_seed=7319,
        sample_count=8,
        iterations=1,
        horizon_steps=3,
        progress_escape_enabled=False,
        predictive_human_cost=PredictiveGaussianHumanCostConfig(
            enabled=True,
            weight=1.5,
            longitudinal_sigma_m=0.6,
            lateral_sigma_m=0.35,
            forward_speed_gain=0.4,
            cutoff_distance_m=2.5,
            aggregation="sum",
        ),
    )
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
    planner1 = MPPISocialPlannerAdapter(cfg)
    planner2 = MPPISocialPlannerAdapter(cfg)
    c1 = planner1.plan(observation)
    c2 = planner2.plan(observation)
    assert np.allclose(c1, c2, atol=1e-12, rtol=0.0)
    assert np.isfinite(c1[0]) and np.isfinite(c1[1])
