"""Tests for robot_sf.benchmark.map_runner_batch_plan — batch-planning helpers."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.map_runner_batch_plan import (
    build_seed_jobs,
    build_worker_fixed_params,
    resolve_batch_kinematics_tag,
)


class TestResolveBatchKinematicsTag:
    """Tests for resolve_batch_kinematics_tag."""

    def test_empty_scenarios(self) -> None:
        """Empty scenario list must return 'unknown'."""
        tag, observed = resolve_batch_kinematics_tag([])
        assert tag == "unknown"
        assert observed == []

    def test_single_kinematics(self) -> None:
        """Uniform scenarios must return the single kinematics label."""
        scenarios = [
            {"robot_config": {"type": "differential_drive"}},
            {"robot_config": {"type": "differential_drive"}},
        ]
        tag, observed = resolve_batch_kinematics_tag(scenarios)
        assert tag == "differential_drive"
        assert observed == ["differential_drive"]

    def test_mixed_kinematics(self) -> None:
        """Mixed scenarios must return 'mixed' with sorted observed labels."""
        scenarios = [
            {"robot_config": {"type": "bicycle_model"}},
            {"robot_config": {"type": "differential_drive"}},
        ]
        tag, observed = resolve_batch_kinematics_tag(scenarios)
        assert tag == "mixed"
        assert "bicycle_drive" in observed
        assert "differential_drive" in observed

    def test_default_kinematics_for_bare_scenarios(self) -> None:
        """Scenarios without robot_config must use the default kinematics."""
        scenarios = [{}, {}]
        tag, _observed = resolve_batch_kinematics_tag(scenarios)
        assert tag == "differential_drive"


class TestBuildSeedJobs:
    """Tests for build_seed_jobs expansion."""

    def test_scenario_seeds_override_suite(self) -> None:
        """Per-scenario seeds must override suite seeds."""
        scenarios = [{"name": "sc1", "seeds": [10, 20]}]
        jobs = build_seed_jobs(scenarios, suite_seeds={"default": [1, 2, 3]}, suite_key="default")
        assert len(jobs) == 2
        assert jobs[0] == (scenarios[0], 10)
        assert jobs[1] == (scenarios[0], 20)

    def test_suite_seeds_used_when_no_scenario_seeds(self) -> None:
        """Suite seeds must be used when scenario has no seeds."""
        scenarios = [{"name": "sc1"}]
        jobs = build_seed_jobs(scenarios, suite_seeds={"default": [42, 43]}, suite_key="default")
        assert len(jobs) == 2
        assert jobs[0][1] == 42
        assert jobs[1][1] == 43

    def test_fallback_to_seed_zero(self) -> None:
        """Without any seeds, the fallback must be [0]."""
        scenarios = [{"name": "sc1"}]
        jobs = build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
        assert len(jobs) == 1
        assert jobs[0][1] == 0

    def test_multiple_scenarios_expand(self) -> None:
        """Multiple scenarios must each expand with their seeds."""
        scenarios = [{"name": "sc1", "seeds": [1]}, {"name": "sc2", "seeds": [2, 3]}]
        jobs = build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
        assert len(jobs) == 3

    def test_jobs_are_tuples(self) -> None:
        """Each job must be a (scenario, seed) tuple."""
        scenarios = [{"name": "sc1", "seeds": [5]}]
        jobs = build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
        assert isinstance(jobs[0], tuple)
        assert len(jobs[0]) == 2
        assert isinstance(jobs[0][1], int)


class TestBuildWorkerFixedParams:
    """Tests for build_worker_fixed_params payload construction."""

    def test_required_fields_present(self) -> None:
        """The fixed params dict must contain all required keys."""
        params = build_worker_fixed_params(
            horizon=500,
            dt=0.1,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo="social_force",
            raw_policy_cfg={"key": "val"},
            algo_config_path=None,
            scenario_path=Path("/tmp/scenario.yaml"),
            adapter_impact_eval=False,
            experimental_ped_impact=False,
            ped_impact_radius_m=1.0,
            ped_impact_window_steps=10,
            noise_spec={"enabled": False},
            tracking_precision_spec={"enabled": False},
            batch_observation_mode="full",
            observation_level=None,
            benchmark_track=None,
            track_schema_version=None,
            actuation_profile_metadata=None,
            latency_profile_metadata=None,
            latency_stress_metrics=None,
            safety_wrapper=None,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
        )
        assert params["horizon"] == 500
        assert params["dt"] == 0.1
        assert params["algo"] == "social_force"
        assert params["record_forces"] is True
        assert params["scenario_path"] == "/tmp/scenario.yaml"
        assert params["adapter_impact_eval"] is False
        assert params["observation_mode"] == "full"

    def test_path_converted_to_str(self) -> None:
        """scenario_path must be converted to a string."""
        params = build_worker_fixed_params(
            horizon=None,
            dt=None,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="goal",
            raw_policy_cfg={},
            algo_config_path=None,
            scenario_path=Path("/some/path.yaml"),
            adapter_impact_eval=False,
            experimental_ped_impact=False,
            ped_impact_radius_m=0.5,
            ped_impact_window_steps=5,
            noise_spec={},
            tracking_precision_spec={},
            batch_observation_mode=None,
            observation_level=None,
            benchmark_track=None,
            track_schema_version=None,
            actuation_profile_metadata=None,
            latency_profile_metadata=None,
            latency_stress_metrics=None,
            safety_wrapper=None,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
        )
        assert isinstance(params["scenario_path"], str)

    def test_cbf_safety_filter_default_none(self) -> None:
        """cbf_safety_filter must default to None."""
        params = build_worker_fixed_params(
            horizon=None,
            dt=None,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="goal",
            raw_policy_cfg={},
            algo_config_path=None,
            scenario_path=Path("/x.yaml"),
            adapter_impact_eval=False,
            experimental_ped_impact=False,
            ped_impact_radius_m=0.5,
            ped_impact_window_steps=5,
            noise_spec={},
            tracking_precision_spec={},
            batch_observation_mode=None,
            observation_level=None,
            benchmark_track=None,
            track_schema_version=None,
            actuation_profile_metadata=None,
            latency_profile_metadata=None,
            latency_stress_metrics=None,
            safety_wrapper=None,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
        )
        assert params["cbf_safety_filter"] is None
