"""Tests for the experimental prediction-aware MPC planner."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.prediction_mpc import (
    ConstantVelocityPedestrianPredictor,
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    speed=0.0,
    goal=(2.0, 0.0),
    ped_positions=None,
    ped_velocities=None,
):
    """Build compact SocNav observation payload for prediction-MPC tests."""
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
            "radius": np.asarray([0.25], dtype=float),
        },
    }


def test_build_prediction_mpc_config_overrides_fields() -> None:
    """Prediction-MPC config should parse YAML-style override values."""
    cfg = build_prediction_mpc_config(
        {
            "max_linear_speed": "0.4",
            "horizon_steps": "3",
            "warm_start": "false",
            "predictor_backend": "constant_velocity",
        }
    )

    assert cfg.max_linear_speed == 0.4
    assert cfg.horizon_steps == 3
    assert cfg.warm_start is False
    assert cfg.predictor_backend == "constant_velocity"


def test_build_prediction_mpc_config_invalid_numeric_uses_default() -> None:
    """Invalid numeric config values should warn and fall back to defaults."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = build_prediction_mpc_config({"rollout_dt": "bad"})

    assert cfg.rollout_dt == PredictionMPCConfig().rollout_dt
    assert any("Invalid prediction_mpc config value 'rollout_dt'" in str(w.message) for w in caught)


def test_constant_velocity_predictor_rotates_ego_velocity_to_world() -> None:
    """SocNav pedestrian velocities are ego-frame and must rotate to world frame."""
    predictor = ConstantVelocityPedestrianPredictor()

    futures = predictor.predict(
        _obs(
            heading=np.pi / 2.0,
            ped_positions=[(1.0, 2.0)],
            ped_velocities=[(1.0, 0.0)],
        ),
        horizon_steps=2,
        dt=0.5,
    )

    np.testing.assert_allclose(futures.positions_world[0, 0], np.asarray([1.0, 2.5]))
    np.testing.assert_allclose(futures.positions_world[0, 1], np.asarray([1.0, 3.0]))
    assert futures.source == "constant_velocity"


def test_prediction_mpc_moves_toward_goal_in_open_space() -> None:
    """Open-space command should be goal directed and bounded."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(horizon_steps=3, solver_max_iterations=16)
    )

    linear, angular = planner.plan(_obs(goal=(3.0, 0.0)))

    assert linear > 0.0
    assert linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed


def test_prediction_mpc_respects_control_bounds(monkeypatch) -> None:
    """The emitted command should stay within configured bounds even if solver overshoots."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(max_linear_speed=0.4, max_angular_speed=0.3, horizon_steps=1)
    )
    monkeypatch.setattr(
        "robot_sf.planner.nmpc_social.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.asarray([4.0, 3.0])),
    )

    linear, angular = planner.plan(_obs(goal=(3.0, 0.0)))

    assert linear == planner.config.max_linear_speed
    assert angular == planner.config.max_angular_speed


def test_prediction_mpc_constraint_function_flags_predicted_collision() -> None:
    """Hard constraint values should go negative for an unsafe predicted collision."""
    planner = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=1))
    obs = _obs(ped_positions=[(0.2, 0.0)], ped_velocities=[(0.0, 0.0)])
    *_, robot_radius, ped_radius = planner._extract_state(obs)
    context = SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([2.0, 0.0], dtype=float),
        ped_positions=np.asarray([[0.2, 0.0]], dtype=float),
        ped_velocities=np.asarray([[0.0, 0.0]], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=0.9,
    )
    futures = planner._future_predictor.predict(obs, horizon_steps=1, dt=planner.config.rollout_dt)

    values = planner._pedestrian_clearance_constraints(
        np.asarray([0.0, 0.0], dtype=float),
        context=context,
        predicted_futures=futures,
    )

    assert np.min(values) < 0.0


def test_prediction_mpc_solver_avoids_or_stops_for_predicted_collision() -> None:
    """A head-on predicted pedestrian conflict should not emit an unsafe forward command."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(horizon_steps=2, solver_max_iterations=30, pedestrian_safety_margin=0.5)
    )

    linear, _angular = planner.plan(
        _obs(goal=(3.0, 0.0), ped_positions=[(0.55, 0.0)], ped_velocities=[(0.0, 0.0)])
    )

    assert linear <= 0.1


def test_prediction_mpc_optimizer_failure_stops_deterministically(monkeypatch) -> None:
    """Solver failure should use deterministic stop fallback by default."""
    planner = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=2))
    monkeypatch.setattr(
        "robot_sf.planner.nmpc_social.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=np.zeros(4, dtype=float)),
    )

    assert planner.plan(_obs(goal=(3.0, 0.0))) == (0.0, 0.0)
    assert planner.diagnostics()["fallback_stop_count"] == 1


def test_prediction_mpc_reset_clears_warm_start_and_diagnostics() -> None:
    """Reset should clear episode-local optimizer warm start and diagnostics."""
    planner = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=2))
    planner.plan(_obs(goal=(3.0, 0.0)))
    assert planner.diagnostics()["calls"] == 1

    planner.reset()

    assert planner._last_solution is None
    assert planner.diagnostics()["calls"] == 0


def _clearance_context(planner: PredictionMPCPlannerAdapter, obs: dict) -> SimpleNamespace:
    """Build a rollout context for direct clearance-constraint evaluation."""
    *_, robot_radius, ped_radius = planner._extract_state(obs)
    return SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([2.0, 0.0], dtype=float),
        ped_positions=np.asarray(obs["pedestrians"]["positions"], dtype=float),
        ped_velocities=np.asarray(obs["pedestrians"]["velocities"], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=0.9,
    )


def test_prediction_mpc_uncertainty_disabled_by_default() -> None:
    """The envelope is opt-in: the default config reports it disabled."""
    planner = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=3))
    envelope = planner.diagnostics()["pedestrian_uncertainty_envelope"]
    assert envelope["enabled"] is False
    assert envelope["policy"] == "deterministic"


def test_prediction_mpc_uncertainty_alpha_zero_preserves_constraints() -> None:
    """Regression: enabling with alpha=0.0 reproduces baseline clearance values."""
    obs = _obs(ped_positions=[(0.6, 0.0)], ped_velocities=[(0.0, 0.0)])
    planner_base = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            horizon_steps=3,
            pedestrian_uncertainty_envelope_enabled=False,
            pedestrian_uncertainty_alpha_mps=0.0,
        )
    )
    planner_zero = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            horizon_steps=3,
            pedestrian_uncertainty_envelope_enabled=True,
            pedestrian_uncertainty_alpha_mps=0.0,
        )
    )
    controls = np.zeros(2 * 3, dtype=float)
    futures = planner_base._future_predictor.predict(
        obs, horizon_steps=3, dt=planner_base.config.rollout_dt
    )

    base_vals = planner_base._pedestrian_clearance_constraints(
        controls, context=_clearance_context(planner_base, obs), predicted_futures=futures
    )
    zero_vals = planner_zero._pedestrian_clearance_constraints(
        controls, context=_clearance_context(planner_zero, obs), predicted_futures=futures
    )

    np.testing.assert_array_equal(base_vals, zero_vals)


def test_prediction_mpc_uncertainty_positive_alpha_tightens_later_constraints() -> None:
    """A positive alpha lowers (tightens) clearance margins at later horizon steps."""
    obs = _obs(ped_positions=[(1.2, 0.0)], ped_velocities=[(0.0, 0.0)])
    horizon = 4
    baseline = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=horizon))
    inflated = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            horizon_steps=horizon,
            pedestrian_uncertainty_envelope_enabled=True,
            pedestrian_uncertainty_alpha_mps=0.5,
        )
    )
    controls = np.zeros(2 * horizon, dtype=float)
    futures = baseline._future_predictor.predict(
        obs, horizon_steps=horizon, dt=baseline.config.rollout_dt
    )

    base_vals = baseline._pedestrian_clearance_constraints(
        controls, context=_clearance_context(baseline, obs), predicted_futures=futures
    )
    inflated_vals = inflated._pedestrian_clearance_constraints(
        controls, context=_clearance_context(inflated, obs), predicted_futures=futures
    )

    # Step 0 is unchanged; later steps require more room, so the constraint slack
    # (distance^2 - r_safe^2) is strictly smaller under the active envelope.
    assert inflated_vals[0] == pytest.approx(base_vals[0])
    assert inflated_vals[-1] < base_vals[-1]


def test_prediction_mpc_reports_measured_uncertainty_envelope_activation() -> None:
    """Diagnostics report when inflated pedestrian radii were actually evaluated."""
    obs = _obs(ped_positions=[(0.6, 0.0)], ped_velocities=[(0.0, 0.0)])
    horizon = 3
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            horizon_steps=horizon,
            pedestrian_uncertainty_envelope_enabled=True,
            pedestrian_uncertainty_alpha_mps=0.2,
        )
    )
    futures = planner._future_predictor.predict(
        obs,
        horizon_steps=horizon,
        dt=planner.config.rollout_dt,
    )

    planner._pedestrian_clearance_constraints(
        np.zeros(2 * horizon, dtype=float),
        context=_clearance_context(planner, obs),
        predicted_futures=futures,
    )

    envelope = planner.diagnostics()["pedestrian_uncertainty_envelope"]
    assert envelope["effective_radius_used_by_planner"] is True
    assert envelope["envelope_activation_count"] == 2


def test_prediction_mpc_config_rejects_negative_uncertainty_alpha() -> None:
    """A negative conservatism rate fails closed at config construction."""
    with pytest.raises(ValueError):
        PredictionMPCConfig(pedestrian_uncertainty_alpha_mps=-0.1)


def test_build_prediction_mpc_config_parses_envelope_fields() -> None:
    """YAML-style envelope keys are parsed onto the prediction-MPC config."""
    cfg = build_prediction_mpc_config(
        {
            "pedestrian_uncertainty_envelope_enabled": "true",
            "pedestrian_uncertainty_alpha_mps": "0.1",
        }
    )
    assert cfg.pedestrian_uncertainty_envelope_enabled is True
    assert cfg.pedestrian_uncertainty_alpha_mps == pytest.approx(0.1)


def test_uncertainty_envelope_example_config_activates_envelope() -> None:
    """The shipped example YAML config yields an active envelope planner."""
    import yaml

    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "algos"
        / "prediction_mpc_cv_uncertainty_envelope.yaml"
    )
    raw = yaml.safe_load(config_path.read_text())
    cfg = build_prediction_mpc_config(raw)
    planner = PredictionMPCPlannerAdapter(cfg)

    diagnostics = planner.diagnostics()
    assert diagnostics["predictor_backend"] == "constant_velocity"
    assert diagnostics["horizon_steps"] == cfg.horizon_steps
    assert diagnostics["rollout_dt"] == pytest.approx(cfg.rollout_dt)
    assert diagnostics["predictor"]["calls"] == 0

    envelope = diagnostics["pedestrian_uncertainty_envelope"]
    assert envelope["enabled"] is True
    assert envelope["policy"] == "linear"
    assert envelope["alpha_mps"] == pytest.approx(0.1)
