"""Tests for the experimental prediction-aware MPC planner."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.prediction_mpc import (
    ConstantVelocityPedestrianPredictor,
    PredictedPedestrianFutures,
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    _to_nmpc_config,
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


def test_prediction_mpc_selects_constraint_feasible_action_from_stub_predictor() -> None:
    """An injected predictor drives selection through the hard rollout constraint."""

    class StubPredictor:
        """Return a fixed short-horizon pedestrian future and record the interface call."""

        def __init__(self) -> None:
            self.calls: list[tuple[int, float]] = []
            self.last_futures: PredictedPedestrianFutures | None = None

        def predict(
            self,
            observation: dict,
            *,
            horizon_steps: int,
            dt: float,
        ) -> PredictedPedestrianFutures:
            del observation
            self.calls.append((horizon_steps, dt))
            self.last_futures = PredictedPedestrianFutures(
                positions_world=np.repeat(
                    np.asarray([[[0.9, 0.0]]], dtype=float), horizon_steps, axis=1
                ),
                mask=np.ones((1,), dtype=float),
                dt=dt,
                source="stub_short_horizon",
            )
            return self.last_futures

    predictor = StubPredictor()
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            predictor_backend="stub_short_horizon",
            horizon_steps=3,
            rollout_dt=0.25,
            solver_max_iterations=60,
            pedestrian_safety_margin=0.25,
        ),
        predictor=predictor,
    )
    observation = _obs(
        goal=(3.0, 0.0),
        ped_positions=[(0.9, 0.0)],
        ped_velocities=[(0.0, 0.0)],
    )

    linear, angular = planner.plan(observation)

    assert predictor.calls == [(3, 0.25)]
    assert predictor.last_futures is not None
    assert planner._last_solution is not None
    constraint_values = planner._pedestrian_clearance_constraints(
        planner._last_solution,
        context=_clearance_context(planner, observation),
        predicted_futures=predictor.last_futures,
    )
    assert np.min(constraint_values) >= -planner.config.solver_ftol
    assert 0.0 <= linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed


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


# ---------------------------------------------------------------------------
# Factor toggles — issue #5372
# ---------------------------------------------------------------------------

# Shared base fields that must be identical across all 4 factorial arms.
_FACTORIAL_BASE_FIELDS = {
    "max_linear_speed",
    "max_angular_speed",
    "horizon_steps",
    "rollout_dt",
    "goal_tolerance",
    "waypoint_switch_distance",
    "path_goal_weight",
    "terminal_goal_weight",
    "heading_weight",
    "control_effort_weight",
    "smoothness_weight",
    "pedestrian_safety_margin",
    "static_obstacle_soft_weight",
    "solver_ftol",
    "solver_max_iterations",
    "warm_start",
    "fallback_to_stop",
    "predictor_backend",
    "allow_predictor_fallback",
    "pedestrian_uncertainty_envelope_enabled",
    "pedestrian_uncertainty_alpha_mps",
}

# The two factor toggles that MUST differ.
_FACTOR_TOGGLE_FIELDS = {"prediction_enabled", "hard_pedestrian_constraints_enabled"}

# Factorial arm configs from YAML files.
_FACTORIAL_YAML_STEMS = [
    "factorial_pred_on_const_on",
    "factorial_pred_off_const_on",
    "factorial_pred_on_const_off",
    "factorial_pred_off_const_off",
]


def _build_factorial_planner(config: PredictionMPCConfig) -> PredictionMPCPlannerAdapter:
    """Build a planner with deterministic settings for factorial tests."""
    return PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            horizon_steps=3,
            rollout_dt=0.25,
            solver_max_iterations=16,
            prediction_enabled=config.prediction_enabled,
            hard_pedestrian_constraints_enabled=config.hard_pedestrian_constraints_enabled,
        )
    )


# -- Factor A: prediction_enabled --


def test_factor_a_default_enabled() -> None:
    """prediction_enabled defaults to True so baseline is unchanged."""
    cfg = PredictionMPCConfig()
    assert cfg.prediction_enabled is True


def test_factor_a_prediction_disabled_freezes_pedestrians_in_soft_path() -> None:
    """Factor A off: _predict_pedestrians must return t=0 positions regardless of step."""
    ped_pos = np.asarray([[1.0, 2.0]], dtype=float)
    ped_vel = np.asarray([[0.5, 0.3]], dtype=float)

    planner_on = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=True, hard_pedestrian_constraints_enabled=False)
    )
    planner_off = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=False, hard_pedestrian_constraints_enabled=False)
    )

    # A-ON: positions should move with velocity
    on_pos = planner_on._predict_pedestrians(ped_pos, ped_vel, step_idx=2)
    assert not np.allclose(on_pos, ped_pos)

    # A-OFF: positions stay frozen at current
    off_pos = planner_off._predict_pedestrians(ped_pos, ped_vel, step_idx=2)
    np.testing.assert_allclose(off_pos, ped_pos)

    # Zero-order hold should work for all step indices
    for step in range(5):
        off_step = planner_off._predict_pedestrians(ped_pos, ped_vel, step_idx=step)
        np.testing.assert_allclose(off_step, ped_pos)


def test_factor_a_prediction_disabled_freezes_hard_constraint_futures() -> None:
    """Factor A off: _freeze_pedestrian_futures should zero all motion."""
    planner = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=False, hard_pedestrian_constraints_enabled=True)
    )
    futures = PredictedPedestrianFutures(
        positions_world=np.arange(12, dtype=float).reshape((2, 3, 2)),
        mask=np.ones((2,), dtype=float),
        dt=0.25,
        source="test",
    )
    frozen = planner._freeze_pedestrian_futures(futures)

    # All steps should equal the t=0 position for each pedestrian
    for ped_idx in range(2):
        ref = futures.positions_world[ped_idx, 0, :]
        for step_idx in range(3):
            np.testing.assert_allclose(
                frozen.positions_world[ped_idx, step_idx],
                ref,
            )

    assert "_zeroorder" in frozen.source


def test_factor_a_toggle_affects_optimizer_constraints() -> None:
    """Factor A off: _optimizer_constraints should freeze pedestrian futures."""
    from scipy.optimize import NonlinearConstraint

    obs = _obs(ped_positions=[(0.8, 0.0)], ped_velocities=[(0.5, 0.0)])
    planner_on = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=True, hard_pedestrian_constraints_enabled=True)
    )
    planner_off = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=False, hard_pedestrian_constraints_enabled=True)
    )
    *_, robot_radius, ped_radius = planner_on._extract_state(obs)
    context = SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([2.0, 0.0], dtype=float),
        ped_positions=np.asarray([[0.8, 0.0]], dtype=float),
        ped_velocities=np.asarray([[0.5, 0.0]], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=0.9,
    )

    constraints_on = planner_on._optimizer_constraints(context)
    constraints_off = planner_off._optimizer_constraints(context)

    # Both should return one NonlinearConstraint (B is ON for both)
    assert len(constraints_on) == 1
    assert len(constraints_off) == 1
    assert isinstance(constraints_on[0], NonlinearConstraint)
    assert isinstance(constraints_off[0], NonlinearConstraint)


# -- Factor B: hard_pedestrian_constraints_enabled --


def test_factor_b_default_enabled() -> None:
    """hard_pedestrian_constraints_enabled defaults to True."""
    cfg = PredictionMPCConfig()
    assert cfg.hard_pedestrian_constraints_enabled is True


def test_factor_b_disabled_drops_optimizer_constraints() -> None:
    """Factor B off: _optimizer_constraints must return empty tuple."""
    obs = _obs(ped_positions=[(0.5, 0.0)], ped_velocities=[(0.3, 0.0)])
    planner_off = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=True, hard_pedestrian_constraints_enabled=False)
    )
    *_, robot_radius, ped_radius = planner_off._extract_state(obs)
    context = SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([2.0, 0.0], dtype=float),
        ped_positions=np.asarray([[0.5, 0.0]], dtype=float),
        ped_velocities=np.asarray([[0.3, 0.0]], dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=0.9,
    )

    constraints = planner_off._optimizer_constraints(context)
    assert constraints == ()


def test_factor_b_disabled_restores_soft_pedestrian_weight() -> None:
    """Factor B off: pedestrian_clearance_weight = 4.5 in NMPC config."""
    cfg_on = PredictionMPCConfig(hard_pedestrian_constraints_enabled=True)
    cfg_off = PredictionMPCConfig(hard_pedestrian_constraints_enabled=False)

    nmpc_on = _to_nmpc_config(cfg_on)
    nmpc_off = _to_nmpc_config(cfg_off)

    assert nmpc_on.pedestrian_clearance_weight == 0.0
    assert nmpc_off.pedestrian_clearance_weight == pytest.approx(4.5)


def test_factor_b_disabled_still_functional_planner() -> None:
    """Factor B off: planner should still emit goal-directed commands."""
    planner = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=True, hard_pedestrian_constraints_enabled=False)
    )
    linear, angular = planner.plan(_obs(goal=(3.0, 0.0)))
    assert linear > 0.0
    assert abs(angular) <= planner.config.max_angular_speed


def test_factor_b_disabled_diagnostics_report_flag() -> None:
    """Diagnostics should reflect hard_pedestrian_constraints_enabled."""
    planner = _build_factorial_planner(
        PredictionMPCConfig(prediction_enabled=True, hard_pedestrian_constraints_enabled=False)
    )
    diag = planner.diagnostics()
    assert diag["hard_pedestrian_constraints_enabled"] is False
    assert diag["prediction_enabled"] is True


def test_build_prediction_mpc_config_parses_factor_toggles() -> None:
    """YAML-style factor toggle keys are parsed onto the config."""
    cfg = build_prediction_mpc_config(
        {
            "prediction_enabled": "false",
            "hard_pedestrian_constraints_enabled": "no",
        }
    )
    assert cfg.prediction_enabled is False
    assert cfg.hard_pedestrian_constraints_enabled is False

    cfg2 = build_prediction_mpc_config(
        {
            "prediction_enabled": "true",
            "hard_pedestrian_constraints_enabled": "yes",
        }
    )
    assert cfg2.prediction_enabled is True
    assert cfg2.hard_pedestrian_constraints_enabled is True


# -- Preflight identity test --


def test_factorial_preflight_identity() -> None:
    """All 4 factorial arms share observation contract, kinematics, runtime budget,
    and differ ONLY in the two factor toggle flags."""
    from dataclasses import fields as dc_fields

    import yaml

    configs_dir = Path(__file__).resolve().parents[2] / "configs" / "algos"
    parsed: dict[str, PredictionMPCConfig] = {}
    for stem in _FACTORIAL_YAML_STEMS:
        raw = yaml.safe_load((configs_dir / f"{stem}.yaml").read_text())
        parsed[stem] = build_prediction_mpc_config(raw)

    # Check exactly 4 arms exist
    assert len(parsed) == 4

    # Expected toggle matrix: (A=prediction_enabled, B=hard_constrained)
    expected_toggles = {
        "factorial_pred_on_const_on": (True, True),
        "factorial_pred_off_const_on": (False, True),
        "factorial_pred_on_const_off": (True, False),
        "factorial_pred_off_const_off": (False, False),
    }
    for stem in _FACTORIAL_YAML_STEMS:
        a_exp, b_exp = expected_toggles[stem]
        cfg = parsed[stem]
        assert cfg.prediction_enabled is a_exp, f"{stem} prediction_enabled"
        assert cfg.hard_pedestrian_constraints_enabled is b_exp, f"{stem} hard_constrained"

    # Base fields identical across all 4 arms
    field_names = {f.name for f in dc_fields(PredictionMPCConfig)}
    base_fields = field_names - _FACTOR_TOGGLE_FIELDS
    for field_name in base_fields:
        values = {stem: getattr(cfg, field_name) for stem, cfg in parsed.items()}
        unique = set(values.values())
        assert len(unique) == 1, f"Base field {field_name} differs across arms: {values}"

    # Verify toggle fields actually differ across arms
    for field_name in _FACTOR_TOGGLE_FIELDS:
        values = {stem: getattr(cfg, field_name) for stem, cfg in parsed.items()}
        unique = set(values.values())
        assert len(unique) == 2, f"Factor field {field_name} does not vary across arms: {values}"


# -- Fidelity smoke --


def test_factorial_fidelity_smoke_decision_traces_differ_per_factor() -> None:
    """Run all 4 arms on 2 tiny scenarios; assert commands differ across each
    factor flip (A-on vs A-off; B-on vs B-off)."""

    # Scenario fixtures: open field and single pedestrian.
    scenarios = [
        _obs(goal=(2.0, 0.0)),  # open space
        _obs(
            goal=(2.0, 0.0),
            ped_positions=[(1.0, 0.0)],
            ped_velocities=[(0.4, 0.0)],
        ),  # moving pedestrian
    ]

    arm_configs = [
        PredictionMPCConfig(
            horizon_steps=2,
            rollout_dt=0.25,
            solver_max_iterations=12,
            prediction_enabled=True,
            hard_pedestrian_constraints_enabled=True,
        ),
        PredictionMPCConfig(
            horizon_steps=2,
            rollout_dt=0.25,
            solver_max_iterations=12,
            prediction_enabled=False,
            hard_pedestrian_constraints_enabled=True,
        ),
        PredictionMPCConfig(
            horizon_steps=2,
            rollout_dt=0.25,
            solver_max_iterations=12,
            prediction_enabled=True,
            hard_pedestrian_constraints_enabled=False,
        ),
        PredictionMPCConfig(
            horizon_steps=2,
            rollout_dt=0.25,
            solver_max_iterations=12,
            prediction_enabled=False,
            hard_pedestrian_constraints_enabled=False,
        ),
    ]
    planners = [PredictionMPCPlannerAdapter(cfg) for cfg in arm_configs]
    arm_names = ["A_ON_B_ON", "A_OFF_B_ON", "A_ON_B_OFF", "A_OFF_B_OFF"]

    decisions: dict[int, dict[str, tuple[float, float]]] = {i: {} for i in range(len(scenarios))}
    for scenario_idx, obs in enumerate(scenarios):
        for arm_idx, planner in enumerate(planners):
            planner.reset()
            linear, angular = planner.plan(obs)
            decisions[scenario_idx][arm_names[arm_idx]] = (linear, angular)

    # Check each factor flip differs across at least one scenario with pedestrians.
    # In open space (no pedestrians), Factor A and B have no effect, so the
    # assertions apply only to scenarios where pedestrians exist.
    factor_a_flips = [
        ("A_ON_B_ON", "A_OFF_B_ON"),  # B fixed, flip A
        ("A_ON_B_OFF", "A_OFF_B_OFF"),  # B fixed, flip A
    ]
    factor_b_flips = [
        ("A_ON_B_ON", "A_ON_B_OFF"),  # A fixed, flip B
        ("A_OFF_B_ON", "A_OFF_B_OFF"),  # A fixed, flip B
    ]

    has_pedestrians = [obs["pedestrians"]["count"][0] > 0 for obs in scenarios]

    for scenario_idx in range(len(scenarios)):
        d = decisions[scenario_idx]
        if not has_pedestrians[scenario_idx]:
            continue
        a_differs = False
        b_differs = False
        for name_on, name_off in factor_a_flips:
            if d[name_on] != d[name_off]:
                a_differs = True
        for name_on, name_off in factor_b_flips:
            if d[name_on] != d[name_off]:
                b_differs = True
        assert a_differs, (
            f"Scenario {scenario_idx}: Factor A (prediction_enabled) "
            f"did not change any decisions across flips: {d}"
        )
        assert b_differs, (
            f"Scenario {scenario_idx}: Factor B (hard_constraints) "
            f"did not change any decisions across flips: {d}"
        )


def test_factorial_fidelity_smoke_b_off_completes_episodes() -> None:
    """Factor B off: planner should run multiple steps without crashing
    and emit bounded commands (functionality check)."""
    planner = PredictionMPCPlannerAdapter(
        PredictionMPCConfig(
            horizon_steps=2,
            rollout_dt=0.25,
            solver_max_iterations=12,
            prediction_enabled=True,
            hard_pedestrian_constraints_enabled=False,
        )
    )

    for _ in range(5):
        linear, angular = planner.plan(
            _obs(goal=(2.0, 0.0), ped_positions=[(1.0, 0.0)], ped_velocities=[(0.3, 0.0)])
        )
        assert 0.0 <= linear <= planner.config.max_linear_speed
        assert abs(angular) <= planner.config.max_angular_speed
