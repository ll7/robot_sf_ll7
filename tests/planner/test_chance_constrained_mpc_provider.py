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

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    CalibrationSweep,
    ConstantVelocityGmmPredictor,
    _min_robot_pedestrian_clearance,
    build_calibration_sweep_config,
    build_constant_velocity_gmm_forecast,
    realized_collision_risk_calibration,
    realized_collision_risk_calibration_sweep,
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
    np.testing.assert_allclose(forecast.mode_weights, 0.5)
    assert forecast.source == "constant_velocity_gmm_surrogate"


def test_provider_imports_before_planner_module() -> None:
    """The concrete provider must be importable in a fresh process by itself."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import robot_sf.planner.chance_constrained_mpc_provider as provider; "
            "assert provider.ConstantVelocityGmmPredictor",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
    assert result["max_compute_time_ms"] >= result["mean_compute_time_ms"]
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


def test_calibration_runs_end_to_end_for_cvar_tail_formulation() -> None:
    """The CVaR/tail-risk Arm 4 formulation (issue #5307) is CPU-runnable end
    to end through the realized-risk calibration harness via the surrogate
    constant_velocity_gmm provider."""

    adapter = build_chance_constrained_mpc_adapter(
        {
            "predictor_backend": "constant_velocity_gmm",
            "chance_constraint_formulation": "cvar_tail",
            "cvar_alpha": 0.9,
            "solver_max_iterations": 20,
        }
    )
    result = realized_collision_risk_calibration(
        adapter,
        CalibrationScenario(num_episodes=4, steps_per_episode=8, num_pedestrians=3),
        seed=3,
    )
    assert result["claimed_risk_per_horizon"] == pytest.approx(0.05)
    assert 0.0 <= result["observed_collision_rate"] <= 1.0
    assert result["calibration_error"] == pytest.approx(
        result["observed_collision_rate"] - result["claimed_risk_per_horizon"]
    )
    assert 0.0 <= result["mean_compute_time_ms"]


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


def test_calibration_integrates_robot_command_and_declared_collision_radius() -> None:
    """Calibration observes the command-integrated robot state and contact radius."""

    class _AlwaysForwardPlanner:
        claimed_risk = 0.1
        config = SimpleNamespace(rollout_dt=0.25)

        def reset(self) -> None:
            pass

        def plan(self, observation: dict[str, object]) -> tuple[float, float]:
            return (1.0, 0.0)

        def diagnostics(self) -> dict[str, int]:
            return {"solver_failures": 0}

    result = realized_collision_risk_calibration(
        _AlwaysForwardPlanner(),
        CalibrationScenario(
            num_episodes=1,
            steps_per_episode=1,
            dt=0.25,
            num_pedestrians=0,
            robot_goal_distance_m=0.2,
        ),
        seed=0,
    )
    assert result["completion_rate"] == pytest.approx(1.0)

    observation = _observation(ped_positions=((0.8, 0.0),), ped_velocities=((0.0, 0.0),))
    assert _min_robot_pedestrian_clearance(observation, collision_radius_m=0.85) == pytest.approx(
        -0.05
    )
    assert _min_robot_pedestrian_clearance(observation, collision_radius_m=0.5) == pytest.approx(
        0.3
    )


def _light_sweep() -> tuple[dict, CalibrationSweep]:
    """Compact sweep config used by the calibration-curve tests."""

    base_algo_config = {
        "predictor_backend": "constant_velocity_gmm",
        "horizon_steps": 6,
        "solver_max_iterations": 20,
    }
    sweep = CalibrationSweep(
        risk_budgets=(0.05, 0.10, 0.20),
        formulations=("marginal",),
        scenario=CalibrationScenario(num_episodes=1, steps_per_episode=3, num_pedestrians=2),
    )
    return base_algo_config, sweep


def test_calibration_sweep_returns_claimed_vs_observed_curve_across_budgets() -> None:
    """Issue #5307 primary measure as a *curve*: pair claimed risk with observed.

    The sweep rebuilds the planner per risk budget and rolls out over the same
    matched scenarios (shared seed), assembling one claimed-vs-observed record
    per budget point. This is the calibration reliability curve, not a single
    point, and is new capability relative to the single-point PR #5662 harness.
    """

    base_algo_config, sweep = _light_sweep()
    result = realized_collision_risk_calibration_sweep(base_algo_config, sweep, seed=1)
    assert result["sweep"]["risk_budgets"] == [0.05, 0.10, 0.20]
    assert result["sweep"]["formulations"] == ["marginal"]
    assert result["sweep"]["seed"] == 1
    points = result["points"]
    assert len(points) == 3
    claimed = [p["claimed_risk"] for p in points]
    assert claimed == [0.05, 0.10, 0.20]
    for point, budget in zip(points, sweep.risk_budgets, strict=True):
        assert point["formulation"] == "marginal"
        assert point["requested_risk_budget"] == pytest.approx(float(budget))
        assert point["claimed_risk"] == pytest.approx(float(budget))
        assert point["claim_unit"] == "planner_risk_budget_per_horizon"
        assert point["observed_unit"] == "episode_any_collision_rate"
        assert point["calibration_comparability"] == "pending_issue_5737"
        # calibration_error is the issue's primary measure: observed - claimed.
        assert point["calibration_error"] == pytest.approx(
            point["observed_collision_rate"] - point["claimed_risk"]
        )
        assert 0.0 <= point["observed_collision_rate"] <= 1.0
        assert 0.0 <= point["freeze_rate"] <= 1.0
        assert 0.0 <= point["infeasible_rate"] <= 1.0
        assert 0.0 <= point["completion_rate"] <= 1.0
        assert point["sample_count"] == pytest.approx(float(sweep.scenario.num_episodes))
        assert point["max_compute_time_ms"] >= point["mean_compute_time_ms"]


def test_calibration_sweep_compares_formulations_at_matched_budgets() -> None:
    """All three chance-constraint formulations compare at matched budgets."""

    base_algo_config, sweep = _light_sweep()
    sweep = CalibrationSweep(
        risk_budgets=sweep.risk_budgets,
        formulations=("marginal", "joint_horizon", "cvar_tail"),
        scenario=sweep.scenario,
    )
    result = realized_collision_risk_calibration_sweep(base_algo_config, sweep, seed=2)
    by_formulation = result["points_by_formulation"]
    assert set(by_formulation) == {"marginal", "joint_horizon", "cvar_tail"}
    assert len(result["points"]) == 3 * 3
    # Each formulation has one point per budget, grouped in budget order.
    for formulation, points in by_formulation.items():
        assert [p["claimed_risk"] for p in points] == [0.05, 0.10, 0.20]
        assert all(p["formulation"] == formulation for p in points)


def test_calibration_sweep_replays_same_scenarios_across_budgets() -> None:
    """A shared seed replays identical scenarios across budgets (apples-to-apples).

    Because every budget point re-seeds its closed-loop RNG with the same seed,
    two sweeps over the same budgets and seed must reproduce the exact observed
    curve, and the underlying single-point harness at matched budget/seed must
    match the sweep point. This is the reproducibility contract that makes the
    claimed-vs-observed curve directly comparable across budgets.
    """

    base_algo_config, sweep = _light_sweep()
    first = realized_collision_risk_calibration_sweep(base_algo_config, sweep, seed=11)
    second = realized_collision_risk_calibration_sweep(base_algo_config, sweep, seed=11)
    first_observed = [p["observed_collision_rate"] for p in first["points"]]
    second_observed = [p["observed_collision_rate"] for p in second["points"]]
    assert first_observed == second_observed

    # The sweep point at a given budget must equal a direct single-point run.
    adapter = build_chance_constrained_mpc_adapter(
        {
            **base_algo_config,
            "max_collision_risk": sweep.risk_budgets[1],
            "chance_constraint_formulation": "marginal",
        }
    )
    direct = realized_collision_risk_calibration(adapter, sweep.scenario, seed=11)
    sweep_point = first["points_by_formulation"]["marginal"][1]
    assert sweep_point["observed_collision_rate"] == pytest.approx(
        direct["observed_collision_rate"]
    )
    assert sweep_point["calibration_error"] == pytest.approx(direct["calibration_error"])


def test_calibration_sweep_reliability_summary_reports_stop_rule_observables() -> None:
    """The issue's pre-registered stop rules surface as observables per formulation.

    The summary reports whether observed collision rate rises monotonically with
    the claimed risk budget (a reliability/curve-defect signal), the per-step
    compute time against the MPC control period (``stop if the solver cannot meet
    the control period``), and the freeze rate at the tightest budget
    (``stop if safety gains come with the zero-progress behavior seen for
    stream_gap``). These are inspection observables, not a routing verdict.
    """

    base_algo_config, sweep = _light_sweep()
    result = realized_collision_risk_calibration_sweep(base_algo_config, sweep, seed=3)
    summary = result["reliability_summary"]["marginal"]
    assert isinstance(summary["observed_monotone_non_decreasing_in_claimed"], bool)
    assert summary["control_period_ms"] == pytest.approx(float(sweep.scenario.dt) * 1000.0)
    assert 0.0 <= summary["max_abs_calibration_error"] <= 1.0
    assert summary["max_per_step_compute_time_ms"] >= 0.0
    assert isinstance(summary["compute_exceeds_control_period"], bool)
    assert summary["freeze_rate_at_tightest_budget"] >= 0.0


def test_calibration_sweep_rejects_invalid_budgets_and_formulations() -> None:
    """The sweep grid validates budget range and formulation names up front."""

    base_algo_config, _ = _light_sweep()
    bad_budgets = CalibrationSweep(
        risk_budgets=(0.0, 0.5),
        formulations=("marginal",),
    )
    with pytest.raises(ValueError, match="risk_budgets must lie in"):
        realized_collision_risk_calibration_sweep(base_algo_config, bad_budgets, seed=0)

    empty_budgets = CalibrationSweep(
        risk_budgets=(),
        formulations=("marginal",),
    )
    with pytest.raises(ValueError, match="risk_budgets must be non-empty"):
        realized_collision_risk_calibration_sweep(base_algo_config, empty_budgets, seed=0)

    bad_formulations = CalibrationSweep(
        risk_budgets=(0.05, 0.10),
        formulations=("not_a_formulation",),
    )
    with pytest.raises(ValueError, match="Unsupported formulation"):
        realized_collision_risk_calibration_sweep(base_algo_config, bad_formulations, seed=0)

    with pytest.raises(ValueError, match="strictly increasing"):
        realized_collision_risk_calibration_sweep(
            base_algo_config,
            CalibrationSweep(risk_budgets=(0.10, 0.05), formulations=("marginal",)),
            seed=0,
        )

    with pytest.raises(ValueError, match="num_episodes must be positive"):
        CalibrationScenario(num_episodes=0)
    with pytest.raises(ValueError, match="steps_per_episode must be positive"):
        CalibrationScenario(steps_per_episode=0)


def test_calibration_sweep_yaml_config_runs_end_to_end() -> None:
    """The shipped sweep config parses into base+grid and produces a finite curve."""

    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "algos"
        / "chance_constrained_mpc_calibration_sweep_diagnostic.yaml"
    )
    base, sweep = build_calibration_sweep_config(yaml.safe_load(config_path.read_text()))
    assert base["predictor_backend"] == "constant_velocity_gmm"
    assert "calibration_sweep" not in base
    assert sweep.risk_budgets == (0.05, 0.10, 0.20)
    assert sweep.formulations == ("marginal", "joint_horizon", "cvar_tail")

    result = realized_collision_risk_calibration_sweep(base, sweep, seed=0)
    assert result["sweep"]["base_algo_config"]["solver_max_iterations"] == 40
    assert len(result["points"]) == 3 * 3
    for point in result["points"]:
        assert np.isfinite(point["observed_collision_rate"])
        assert np.isfinite(point["mean_compute_time_ms"])
    assert set(result["reliability_summary"]) == {
        "marginal",
        "joint_horizon",
        "cvar_tail",
    }


def test_build_calibration_sweep_config_defaults_when_subkey_absent() -> None:
    """Without a calibration_sweep sub-key the grid falls back to defaults."""

    base, sweep = build_calibration_sweep_config({"predictor_backend": "constant_velocity_gmm"})
    assert base == {"predictor_backend": "constant_velocity_gmm"}
    assert sweep.risk_budgets == CalibrationSweep().risk_budgets
    assert sweep.formulations == CalibrationSweep().formulations
