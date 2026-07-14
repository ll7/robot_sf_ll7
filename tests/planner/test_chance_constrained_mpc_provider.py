"""Successor-slice tests for issue #5307: concrete K-mode provider + risk calibration.

PR #5322 delivered the chance-constrained MPC control law and its
GaussianMixturePedestrianPredictor boundary but failed closed because the
learned K-mode/GMM predictor (#2844) is still open. This file tests the
cheap-lane successor slice: a concrete constant-velocity GMM provider that
makes the control law CPU-runnable, plus the issue's primary measure --
realized collision-risk CALIBRATION (claimed vs. observed) as a diagnostic
self-check. No benchmark/calibration claim is made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.chance_constrained_mpc import (
    ChanceConstrainedMPCPlannerAdapter,
    GaussianMixturePedestrianForecast,
    build_chance_constrained_mpc_adapter,
    build_chance_constrained_mpc_config,
)
from robot_sf.planner.chance_constrained_mpc_provider import (
    CalibrationScenario,
    ConstantVelocityGmmPredictor,
    build_constant_velocity_gmm_forecast,
    realized_collision_risk_calibration,
)


def _observation(
    *,
    ped_positions=((0.0, 0.0), (2.0, 0.0)),
    ped_velocities=((0.5, 0.0), (0.0, 0.3)),
) -> dict:
    """Build a compact SocNav observation with two pedestrians."""

    return {
        "robot": {
            "position": np.asarray([0.0, 0.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.0]),
            "radius": np.asarray([0.25]),
        },
        "goal": {"current": np.asarray([3.0, 0.0]), "next": np.asarray([3.0, 0.0])},
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": np.asarray([0.25]),
        },
    }


def test_surrogate_forecast_satisfies_gmm_contract() -> None:
    """The constant-velocity provider emits a contract-valid K-mode forecast."""

    predictor = ConstantVelocityGmmPredictor(mode_count=2)
    forecast = predictor.predict(_observation(), horizon_steps=6, dt=0.25)

    assert isinstance(forecast, GaussianMixturePedestrianForecast)
    assert forecast.means_world.shape == (2, 2, 6, 2)
    assert forecast.covariances_world.shape == (2, 2, 6, 2, 2)
    assert forecast.mode_weights.shape == (2, 2)
    np.testing.assert_allclose(forecast.mode_weights.sum(axis=1), 1.0)
    assert forecast.source == "constant_velocity_gmm_surrogate"


def test_surrogate_forecast_integrates_constant_velocity_mean() -> None:
    """Mode 0 mean must be the constant-velocity world-frame integration."""

    forecast = build_constant_velocity_gmm_forecast(
        _observation(), horizon_steps=3, dt=0.25, mode_count=1
    )
    # Pedestrian 1 at (2.0, 0.0) moving at (0.0, 0.3) m/s.
    expected = np.asarray([2.0, 0.3 * 0.25 * 3])
    np.testing.assert_allclose(forecast.means_world[1, 0, 2], expected, atol=1e-9)


def test_surrogate_forecast_handles_zero_pedestrians() -> None:
    """No pedestrians yields an empty but contract-valid forecast."""

    forecast = build_constant_velocity_gmm_forecast(
        {"pedestrians": {}}, horizon_steps=4, dt=0.25, mode_count=1
    )
    assert forecast.means_world.shape[0] == 0
    assert forecast.mode_weights.shape[0] == 0


def test_chance_constrained_mpc_runs_end_to_end_with_surrogate_provider() -> None:
    """The arm is CPU-runnable through the real NMPC path via the opt-in backend."""

    adapter = build_chance_constrained_mpc_adapter(
        {"predictor_backend": "constant_velocity_gmm", "horizon_steps": 4}
    )
    assert isinstance(adapter, ChanceConstrainedMPCPlannerAdapter)

    command = adapter.plan(_observation())
    assert np.all(np.isfinite(command))
    assert 0.0 <= command[0] <= 0.9
    assert abs(command[1]) <= 1.1


def test_default_backend_still_fails_closed_without_learned_provider() -> None:
    """The #2844 backend must not silently use the surrogate."""

    with pytest.raises(ValueError, match="unavailable"):
        build_chance_constrained_mpc_adapter({"predictor_backend": "issue_2844_k_mode_gmm"})


def test_diagnostic_yaml_runs_through_the_surrogate_backend() -> None:
    """The shipped diagnostic config wires the surrogate and produces a finite command."""

    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "algos"
        / "chance_constrained_mpc_cv_gmm_diagnostic.yaml"
    )
    config = build_chance_constrained_mpc_config(yaml.safe_load(config_path.read_text()))

    assert config.predictor_backend == "constant_velocity_gmm"
    assert config.gmm_mode_count == 1
    assert config.allow_predictor_fallback is False

    adapter = build_chance_constrained_mpc_adapter(yaml.safe_load(config_path.read_text()))
    command = adapter.plan(_observation())
    assert np.all(np.isfinite(command))


def test_calibration_closes_loop_over_the_planners_claimed_risk() -> None:
    """Issue #5307 measure #1: pair the planner's claimed risk with observed.

    The harness rolls the chance-constrained MPC planner through episodes whose
    pedestrian dynamics match the surrogate's own forecast, then reports the
    planner's claimed per-horizon risk next to the observed collision rate and
    the other named measures (infeasibility, freezing, completion, tail
    clearance, compute time).
    """

    adapter = build_chance_constrained_mpc_adapter(
        {
            "predictor_backend": "constant_velocity_gmm",
            "horizon_steps": 6,
            "max_collision_risk": 0.05,
            "solver_max_iterations": 20,
        }
    )
    result = realized_collision_risk_calibration(
        adapter,
        CalibrationScenario(num_episodes=6, steps_per_episode=10, num_pedestrians=4),
        seed=1,
    )
    required = {
        "claimed_risk_per_horizon",
        "observed_collision_rate",
        "calibration_error",
        "infeasible_rate",
        "freeze_rate",
        "completion_rate",
        "mean_tail_clearance_m",
        "mean_compute_time_ms",
        "sample_count",
    }
    assert required <= result.keys()
    # The planner's claimed risk is read straight from the control law config.
    assert result["claimed_risk_per_horizon"] == pytest.approx(0.05)
    # Calibration error is the issue's primary measure: observed minus claimed.
    assert result["calibration_error"] == pytest.approx(
        result["observed_collision_rate"] - result["claimed_risk_per_horizon"]
    )
    assert 0.0 <= result["observed_collision_rate"] <= 1.0
    assert 0.0 <= result["completion_rate"] <= 1.0
    assert 0.0 <= result["mean_compute_time_ms"]
    assert result["sample_count"] == pytest.approx(6.0)


def test_calibration_is_reproducible_under_fixed_seed() -> None:
    """Monte-Carlo closed-loop calibration must be deterministic for a given seed."""

    adapter = build_chance_constrained_mpc_adapter(
        {
            "predictor_backend": "constant_velocity_gmm",
            "solver_max_iterations": 20,
        }
    )
    scenario = CalibrationScenario(num_episodes=4, steps_per_episode=10, num_pedestrians=3)
    first = realized_collision_risk_calibration(adapter, scenario, seed=7)
    second = realized_collision_risk_calibration(adapter, scenario, seed=7)
    assert first["observed_collision_rate"] == second["observed_collision_rate"]
    assert first["mean_tail_clearance_m"] == second["mean_tail_clearance_m"]


def test_calibration_runs_end_to_end_without_pedestrians() -> None:
    """A pedestrian-free episode still reports finite, well-formed diagnostics."""

    adapter = build_chance_constrained_mpc_adapter(
        {"predictor_backend": "constant_velocity_gmm", "horizon_steps": 4}
    )
    result = realized_collision_risk_calibration(
        adapter,
        CalibrationScenario(num_episodes=4, steps_per_episode=8, num_pedestrians=0),
        seed=2,
    )
    assert result["observed_collision_rate"] == pytest.approx(0.0)
    # No pedestrians => every clearance is +inf, so the mean is +inf (not finite).
    assert result["mean_tail_clearance_m"] == pytest.approx(float("inf"))
    assert result["completion_rate"] >= 0.0
