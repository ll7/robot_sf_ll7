"""Unit, invariant, and deterministic smoke tests for anisotropic Gaussian human cost (Issue #7603)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from robot_sf.planner.anisotropic_gaussian_cost import (
    DIAGNOSTICS_SCHEMA,
    PLANNER_TYPE,
    AnisotropicGaussianCostConfig,
    AnisotropicGaussianCostPlanner,
    build_anisotropic_gaussian_cost_config,
    evaluate_anisotropic_gaussian_cost,
    evaluate_anisotropic_repulsive_force,
)


def test_config_validation_fail_closed() -> None:
    """Validate that invalid configuration parameters fail closed."""
    cfg = AnisotropicGaussianCostConfig()
    assert cfg.amplitude == 1.0
    assert cfg.aggregation_mode == "max"

    with pytest.raises(ValueError, match="amplitude"):
        AnisotropicGaussianCostConfig(amplitude=0.0)

    with pytest.raises(ValueError, match="amplitude"):
        AnisotropicGaussianCostConfig(amplitude=-1.0)

    with pytest.raises(ValueError, match="sigma_long_base_m"):
        AnisotropicGaussianCostConfig(sigma_long_base_m=float("nan"))

    with pytest.raises(ValueError, match="velocity_scale_long"):
        AnisotropicGaussianCostConfig(velocity_scale_long=-0.1)

    with pytest.raises(ValueError, match="look_ahead_min_m"):
        AnisotropicGaussianCostConfig(look_ahead_min_m=2.5, look_ahead_max_m=1.0)

    with pytest.raises(ValueError, match="aggregation_mode"):
        AnisotropicGaussianCostConfig(aggregation_mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="obstacle_grid_threshold"):
        AnisotropicGaussianCostConfig(obstacle_grid_threshold=1.5)

    with pytest.raises(ValueError, match="obstacle_grid_max_points"):
        AnisotropicGaussianCostConfig(obstacle_grid_max_points=0)

    with pytest.raises(ValueError, match="obstacle_input_mode"):
        AnisotropicGaussianCostConfig(obstacle_input_mode="invalid")

    with pytest.raises(ValueError, match="pedestrian_input_mode"):
        AnisotropicGaussianCostConfig(pedestrian_input_mode="invalid")


def test_config_digest_stability() -> None:
    """Test that config digest is deterministic and changes with parameters."""
    cfg1 = AnisotropicGaussianCostConfig(amplitude=1.0, sigma_long_base_m=0.8)
    cfg2 = AnisotropicGaussianCostConfig(amplitude=1.0, sigma_long_base_m=0.8)
    cfg3 = AnisotropicGaussianCostConfig(amplitude=1.5, sigma_long_base_m=0.8)

    assert cfg1.digest() == cfg2.digest()
    assert len(cfg1.digest()) == 64
    assert cfg1.digest() != cfg3.digest()


def test_build_config_from_yaml_dict() -> None:
    """Test building config from raw YAML payload ignores runner-only keys."""
    payload = {
        "allow_testing_algorithms": True,
        "planner_variant": "anisotropic_gaussian_cost",
        "amplitude": 1.2,
        "sigma_long_base_m": 0.9,
    }
    cfg = build_anisotropic_gaussian_cost_config(payload)
    assert cfg.amplitude == 1.2
    assert cfg.sigma_long_base_m == 0.9
    assert cfg.sigma_lat_base_m == 0.5


def test_stationary_pedestrian_limiting_rule() -> None:
    """A stationary pedestrian uses the isotropic limiting rule without angle singularities."""
    cfg = AnisotropicGaussianCostConfig(
        amplitude=1.0,
        stationary_sigma_m=0.6,
        min_velocity_threshold_mps=0.05,
    )
    ped_pos = np.array([[5.0, 5.0]])
    ped_vel = np.array([[0.0, 0.0]])  # Zero velocity

    # Check symmetric queries around stationary pedestrian
    dist = 0.4
    queries = np.array(
        [
            [5.0 + dist, 5.0],
            [5.0 - dist, 5.0],
            [5.0, 5.0 + dist],
            [5.0, 5.0 - dist],
        ]
    )
    costs = evaluate_anisotropic_gaussian_cost(queries, ped_pos, ped_vel, cfg)

    # All 4 orthogonal directions must have identical isotropic cost
    expected_cost = 1.0 * math.exp(-0.5 * (dist / 0.6) ** 2)
    for c in costs:
        assert c == pytest.approx(expected_cost, rel=1e-5)


def test_moving_pedestrian_velocity_alignment() -> None:
    """Moving pedestrian cost field aligns with velocity and exhibits front/rear asymmetry."""
    cfg = AnisotropicGaussianCostConfig(
        amplitude=1.0,
        sigma_long_base_m=0.8,
        sigma_lat_base_m=0.5,
        velocity_scale_long=0.5,
        asymmetry_front_ratio=1.5,
        min_velocity_threshold_mps=0.05,
    )
    ped_pos = np.array([[0.0, 0.0]])
    ped_vel = np.array([[1.0, 0.0]])  # Moving +X at 1 m/s

    # In front (+X), Behind (-X), Lateral (+Y)
    q_front = np.array([[1.0, 0.0]])
    q_rear = np.array([[-1.0, 0.0]])
    q_lat = np.array([[0.0, 1.0]])

    cost_front = evaluate_anisotropic_gaussian_cost(q_front, ped_pos, ped_vel, cfg)[0]
    cost_rear = evaluate_anisotropic_gaussian_cost(q_rear, ped_pos, ped_vel, cfg)[0]
    cost_lat = evaluate_anisotropic_gaussian_cost(q_lat, ped_pos, ped_vel, cfg)[0]

    # Front elongation implies cost extends further forward: cost_front > cost_rear
    assert cost_front > cost_rear
    # Lateral spread is narrower: cost_front > cost_lat
    assert cost_front > cost_lat


def test_rigid_rotation_and_translation_invariance() -> None:
    """Rotated and translated scenario produces invariant cost values."""
    cfg = AnisotropicGaussianCostConfig(
        amplitude=1.0,
        sigma_long_base_m=0.8,
        sigma_lat_base_m=0.5,
        velocity_scale_long=0.4,
        asymmetry_front_ratio=1.4,
    )
    ped_pos_base = np.array([[2.0, 3.0]])
    ped_vel_base = np.array([[1.2, 0.5]])
    query_base = np.array(
        [
            [2.8, 3.4],
            [1.5, 2.7],
            [2.0, 4.2],
        ]
    )

    cost_base = evaluate_anisotropic_gaussian_cost(query_base, ped_pos_base, ped_vel_base, cfg)

    # Apply rigid 2D transform (rotation by 45 deg + translation (10, -5))
    theta = math.pi / 4.0
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    trans = np.array([10.0, -5.0])

    ped_pos_trans = (rot @ ped_pos_base.T).T + trans
    ped_vel_trans = (rot @ ped_vel_base.T).T  # Vectors rotate, no translation
    query_trans = (rot @ query_base.T).T + trans

    cost_trans = evaluate_anisotropic_gaussian_cost(query_trans, ped_pos_trans, ped_vel_trans, cfg)

    np.testing.assert_allclose(cost_base, cost_trans, rtol=1e-5, atol=1e-6)


def test_mahalanobis_and_distance_cutoff() -> None:
    """Costs beyond cutoff distance or Mahalanobis boundary truncate to exactly 0.0."""
    cfg = AnisotropicGaussianCostConfig(
        amplitude=1.0,
        cutoff_distance_m=3.0,
        mahalanobis_cutoff=2.5,
    )
    ped_pos = np.array([[0.0, 0.0]])
    ped_vel = np.array([[0.0, 0.0]])

    # Distance > 3.0
    q_far = np.array([[3.5, 0.0]])
    assert evaluate_anisotropic_gaussian_cost(q_far, ped_pos, ped_vel, cfg)[0] == 0.0

    # Mahalanobis > 2.5 (stationary sigma = 0.6 => dist = 1.6 > 0.6 * 2.5 = 1.5)
    q_maha = np.array([[1.8, 0.0]])
    assert evaluate_anisotropic_gaussian_cost(q_maha, ped_pos, ped_vel, cfg)[0] == 0.0


def test_multi_pedestrian_aggregation_modes() -> None:
    """Max and Sum aggregation across multiple pedestrians behave correctly."""
    cfg_max = AnisotropicGaussianCostConfig(aggregation_mode="max")
    cfg_sum = AnisotropicGaussianCostConfig(aggregation_mode="sum")

    peds_pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    peds_vel = np.array([[0.0, 0.0], [0.0, 0.0]])
    query = np.array([[1.0, 0.0]])  # Midway between both pedestrians

    c_max = evaluate_anisotropic_gaussian_cost(query, peds_pos, peds_vel, cfg_max)[0]
    c_sum = evaluate_anisotropic_gaussian_cost(query, peds_pos, peds_vel, cfg_sum)[0]

    assert c_sum == pytest.approx(2.0 * c_max, rel=1e-5)
    assert c_max > 0.0


def test_repulsive_force_gradient_direction() -> None:
    """Repulsive force repels the robot away from pedestrians along negative gradient."""
    cfg = AnisotropicGaussianCostConfig(
        repulsive_weight=2.0,
        min_velocity_threshold_mps=0.05,
    )
    # Robot at (1.0, 0.0), stationary pedestrian at (0.0, 0.0)
    peds_pos = np.array([[0.0, 0.0]])
    peds_vel = np.array([[0.0, 0.0]])

    fx, fy = evaluate_anisotropic_repulsive_force((1.0, 0.0), peds_pos, peds_vel, cfg)
    # Repulsion pushes +X away from pedestrian
    assert fx > 0.0
    assert abs(fy) < 1e-6


def test_planner_protocol_lifecycle() -> None:
    """Planner satisfies LocalPlannerProtocol reset, plan, diagnostics, close lifecycle."""
    cfg = AnisotropicGaussianCostConfig(look_ahead_gain=0.8)
    planner = AnisotropicGaussianCostPlanner(config=cfg)

    obs = {
        "robot_state": (0.0, 0.0, 0.0),
        "goal_position": (10.0, 0.0),
        "pedestrians": {
            "positions": [(5.0, 0.2)],
            "velocities": [(-0.8, 0.0)],
        },
    }

    cmd = planner.plan(obs)
    assert isinstance(cmd, tuple)
    assert len(cmd) == 2
    assert cmd[0] > 0.0  # Moves forward

    diag = planner.diagnostics()
    assert diag["planner_type"] == PLANNER_TYPE
    assert diag["diagnostics_schema"] == DIAGNOSTICS_SCHEMA
    assert diag["status"] == "ok"
    assert diag["pedestrian_count"] == 1
    assert diag["config_digest"] == cfg.digest()

    planner.reset(seed=42)
    diag_reset = planner.diagnostics()
    assert diag_reset["status"] == "ok"
    assert diag_reset["linear_speed"] == 0.0

    planner.close()
    cmd_closed = planner.plan(obs)
    assert cmd_closed == (0.0, 0.0)


def test_planner_speed_and_rate_limits() -> None:
    """Commands are strictly bounded by speed and rate limits."""
    cfg = AnisotropicGaussianCostConfig(
        max_linear_speed=1.0,
        max_angular_speed=1.2,
        max_linear_rate=0.5,
        max_angular_rate=0.8,
        control_dt=0.2,
    )
    planner = AnisotropicGaussianCostPlanner(config=cfg)

    # Initial step starting from 0 speed
    obs = {
        "robot_state": (0.0, 0.0, 0.0),
        "goal_position": (20.0, 10.0),
    }
    v, w = planner.plan(obs)
    # Rate limit: max delta v in 0.2s is 0.5 * 0.2 = 0.1
    assert v <= 0.1 + 1e-6
    # Angular rate limit: max delta w in 0.2s is 0.8 * 0.2 = 0.16
    assert abs(w) <= 0.16 + 1e-6

    diag = planner.diagnostics()
    assert "linear_rate_limit" in diag["active_rate_limits"]


def test_planner_goal_reached_and_invalid_inputs() -> None:
    """Planner handles goal arrival and missing inputs gracefully."""
    planner = AnisotropicGaussianCostPlanner()

    # Goal reached
    obs_goal = {
        "robot_state": (10.0, 5.0, 0.0),
        "goal_position": (10.0, 5.0),
    }
    cmd_goal = planner.plan(obs_goal)
    assert cmd_goal == (0.0, 0.0)
    assert planner.diagnostics()["status"] == "goal_reached"

    # Missing robot state
    obs_invalid: dict[str, Any] = {"goal_position": (10.0, 5.0)}
    cmd_inv = planner.plan(obs_invalid)
    assert cmd_inv == (0.0, 0.0)
    assert planner.diagnostics()["status"] == "invalid"


def test_fixed_smoke_scenarios() -> None:
    """Deterministic smoke scenario producing bit-identical commands."""
    cfg = AnisotropicGaussianCostConfig(
        amplitude=1.0,
        sigma_long_base_m=0.8,
        sigma_lat_base_m=0.5,
        velocity_scale_long=0.5,
        look_ahead_gain=0.8,
        max_linear_rate=0.8,
        max_angular_rate=1.5,
        control_dt=0.2,
    )
    planner = AnisotropicGaussianCostPlanner(config=cfg)

    obs_static = {
        "robot_state": (0.0, 0.0, 0.0),
        "goal_position": (10.0, 0.0),
        "obstacles": [(5.0, 0.0)],
    }
    cmd_static_1 = planner.plan(obs_static)
    planner.reset()
    cmd_static_2 = planner.plan(obs_static)

    assert cmd_static_1 == cmd_static_2
    assert cmd_static_1[0] > 0.0

    obs_ped = {
        "robot_state": (0.0, 0.0, 0.0),
        "goal_position": (10.0, 0.0),
        "pedestrians": {
            "positions": [(5.0, 0.2)],
            "velocities": [(-1.0, 0.0)],
        },
    }
    planner.reset()
    cmd_ped_1 = planner.plan(obs_ped)
    planner.reset()
    cmd_ped_2 = planner.plan(obs_ped)

    assert cmd_ped_1 == cmd_ped_2
    assert cmd_ped_1[0] > 0.0
