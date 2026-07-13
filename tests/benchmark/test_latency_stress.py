"""Tests for latency-stress preflight profile helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_sf.benchmark.latency_stress import (
    LatencyMeasurementHarness,
    LatencyStressProfile,
    classify_feasibility,
    load_latency_stress_profile,
    not_available_latency_metrics,
    validate_provenance_completeness,
)


def test_latency_stress_profile_metadata_maps_steps_to_ms() -> None:
    """Latency profiles should expose step counts and optional dt-derived milliseconds."""
    profile = LatencyStressProfile(
        name="learned-policy-latency-stress-v0",
        observation_delay_steps=1,
        action_delay_steps=2,
        planner_update_mode="hold-last",
        planner_update_period_steps=5,
        inference_timeout_ms=200.0,
    )

    metadata = profile.to_metadata(dt=0.1)

    assert metadata["observation_delay_steps"] == 1
    assert metadata["observation_delay_ms"] == 100.0
    assert metadata["action_delay_steps"] == 2
    assert metadata["action_delay_ms"] == 200.0
    assert metadata["planner_update_interval_ms"] == 500.0
    assert metadata["contract_scope"] == "preflight-and-provenance-only"


def test_latency_stress_profile_rejects_unsupported_update_mode() -> None:
    """Unsupported planner update modes should fail before benchmark evidence is written."""
    with pytest.raises(ValueError, match="Unsupported latency_stress_profile.planner_update_mode"):
        load_latency_stress_profile(
            {
                "name": "bad-latency",
                "planner_update_mode": "occasionally",
            }
        )


def test_latency_stress_profile_requires_non_success_status_contract() -> None:
    """Fallback/degraded/timeout outcomes must stay explicit non-success rows."""
    with pytest.raises(ValueError, match="non_success_statuses must include"):
        load_latency_stress_profile(
            {
                "name": "bad-latency",
                "non_success_statuses": ["failed"],
            }
        )


def test_latency_stress_profile_rejects_none_and_lossy_numeric_values() -> None:
    """Explicit nulls and non-integral steps should not coerce into contract metadata."""
    with pytest.raises(ValueError, match="name must be non-empty"):
        load_latency_stress_profile({"name": None})

    with pytest.raises(TypeError, match="observation_delay_steps must be an integer"):
        load_latency_stress_profile({"name": "bad-latency", "observation_delay_steps": 1.9})

    with pytest.raises(TypeError, match="non_success_statuses entries must be strings"):
        load_latency_stress_profile(
            {
                "name": "bad-latency",
                "non_success_statuses": [
                    "fallback",
                    "degraded",
                    None,
                    "timeout",
                    "not_available",
                    "failed",
                ],
            }
        )


def test_latency_stress_profile_validates_dataclass_field_types() -> None:
    """Direct dataclass payloads should fail with clear type errors, not AttributeError."""
    with pytest.raises(TypeError, match="name must be a string"):
        load_latency_stress_profile(LatencyStressProfile(name=None))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="non_success_statuses must be a tuple"):
        load_latency_stress_profile(
            LatencyStressProfile(  # type: ignore[arg-type]
                name="bad-latency",
                non_success_statuses=None,
            )
        )


def test_not_available_latency_metrics_names_expected_placeholders() -> None:
    """Preflight-only contracts should emit explicit not-available metric placeholders."""
    metrics = not_available_latency_metrics()

    assert metrics["observation_age_ms"] == "not_available"
    assert metrics["held_action_ratio"] == "not_available"
    assert metrics["inference_timeout_count"] == "not_available"


def test_component_sum_consistency_and_mock_harness() -> None:
    """Harness must guarantee component-sum consistency and measure latencies correctly."""
    harness = LatencyMeasurementHarness(deadline_ms=50.0, config_hash="fixture-config")

    # Simulate a few cycles with mock durations
    with harness:
        # Cycle 0 (Cold start)
        harness.start_cycle()
        harness.add_time("observation_construction", 5.0)
        harness.add_time("prediction", 10.0)
        harness.add_time("collision_risk_safety_filter", 2.0)
        harness.add_time("action_conversion", 1.0)
        # Sleep a little to simulate planner computation
        time.sleep(0.01)
        harness.end_cycle()

        # Cycle 1 (Steady state 1)
        harness.start_cycle()
        harness.add_time("observation_construction", 1.0)
        harness.add_time("prediction", 2.0)
        harness.add_time("collision_risk_safety_filter", 0.5)
        harness.add_time("action_conversion", 0.5)
        time.sleep(0.005)
        harness.end_cycle()

        # Cycle 2 (Steady state 2 - misses budget)
        harness.start_cycle()
        harness.add_time("observation_construction", 10.0)
        harness.add_time("prediction", 20.0)
        harness.add_time("collision_risk_safety_filter", 5.0)
        harness.add_time("action_conversion", 2.0)
        time.sleep(0.06)
        harness.end_cycle()

    # Get metrics
    metrics = harness.get_metrics()
    assert metrics["cold_start_latency_ms"] > 0.0

    # Check component sum consistency
    for cycle in metrics["cycles"]:
        components_sum = (
            cycle["observation_construction_ms"]
            + cycle["prediction_ms"]
            + cycle["planner_computation_ms"]
            + cycle["collision_risk_safety_filter_ms"]
            + cycle["action_conversion_ms"]
        )
        assert abs(components_sum - cycle["total_ms"]) < 1e-3

    # Check classifications and stats
    assert len(metrics["cycles"]) == 3
    assert metrics["deadline_miss_rate"] == 0.5  # 1 out of 2 steady cycles missed 50ms
    assert metrics["classification"] == "misses_budget_on_measured_host"


def test_harness_with_cached_policy_reuse() -> None:
    """Cached policies must measure components against the currently active harness."""

    class DummyAdapter:
        def _extract_state(self, obs: Any) -> Any:
            time.sleep(0.002)
            return obs

        def plan(self, obs: Any) -> tuple[float, float]:
            return 0.0, 0.0

    adapter = DummyAdapter()

    def policy_fn(obs: dict[str, Any]) -> tuple[float, float]:
        policy_fn._last_step_native = True  # type: ignore[attr-defined]
        return adapter.plan(obs)

    policy_fn._planner_adapter = adapter  # type: ignore[attr-defined]

    harness1 = LatencyMeasurementHarness(deadline_ms=50.0, config_hash="fixture-config")
    with harness1:
        wrapped_policy1 = harness1.wrap_policy(policy_fn)
        for _ in range(2):
            harness1.start_cycle()
            adapter._extract_state({})
            wrapped_policy1({})
            harness1.end_cycle()

    harness2 = LatencyMeasurementHarness(deadline_ms=50.0, config_hash="fixture-config")
    with harness2:
        wrapped_policy2 = harness2.wrap_policy(policy_fn)
        for _ in range(2):
            harness2.start_cycle()
            adapter._extract_state({})
            wrapped_policy2({})
            harness2.end_cycle()

    assert harness1.get_metrics()["cycles"][0]["observation_construction_ms"] > 0.0
    assert harness2.get_metrics()["cycles"][0]["observation_construction_ms"] > 0.0
    assert getattr(wrapped_policy1, "_last_step_native", False) is True
    assert getattr(wrapped_policy2, "_last_step_native", False) is True


def test_classification_boundaries() -> None:
    """Test feasibility classifications across various targets and latencies."""
    # 1. Meets budget on measured host
    res = classify_feasibility(
        steady_state_latencies=[10.0, 20.0, 30.0],
        deadline_ms=50.0,
        target_hardware=None,
    )
    assert res == "meets_budget_on_measured_host"

    # 2. Misses budget on measured host
    res = classify_feasibility(
        steady_state_latencies=[10.0, 60.0, 30.0],
        deadline_ms=50.0,
        target_hardware=None,
    )
    assert res == "misses_budget_on_measured_host"

    # 3. Target hardware unmeasured (desktop measured, but target is embedded)
    res = classify_feasibility(
        steady_state_latencies=[10.0, 20.0, 30.0],
        deadline_ms=50.0,
        target_hardware="Jetson Orin Nano",
        measured_host_identity=None,
    )
    assert res == "target_hardware_unmeasured"

    # 4. Target hardware matches only when the measured identity is explicit.
    res = classify_feasibility(
        steady_state_latencies=[10.0, 20.0, 30.0],
        deadline_ms=50.0,
        target_hardware="Jetson Orin Nano",
        measured_host_is_embedded=True,
        measured_host_identity="Jetson-Orin_Nano",
    )
    assert res == "meets_budget_on_measured_host"

    # 5. An embedded host with a different model is still unmeasured.
    res = classify_feasibility(
        steady_state_latencies=[10.0, 20.0, 30.0],
        deadline_ms=50.0,
        target_hardware="Jetson Orin Nano",
        measured_host_is_embedded=True,
        measured_host_identity="Jetson Xavier NX",
    )
    assert res == "target_hardware_unmeasured"


def test_provenance_completeness_fail_closed() -> None:
    """Provenance validation must fail closed when required fields are missing or empty."""
    # Complete/valid provenance
    valid_prov = {
        "cpu_model": "Intel Core i7",
        "cpu_affinity": [0, 1, 2, 3],
        "thread_settings": {
            "environment": {"OMP_NUM_THREADS": "1"},
            "threadpools": [],
        },
        "dependency_versions": {"python": "3.10.0", "numpy": "1.22.0"},
        "git_commit": "abcdef123456",
        "config_hash": "fixture-config",
        "measured_host_identity": "Intel Core i7",
    }
    # Should not raise any exception
    validate_provenance_completeness(valid_prov)

    # Missing cpu_model
    bad_prov = dict(valid_prov)
    bad_prov["cpu_model"] = "unknown"
    with pytest.raises(ValueError, match="cpu_model cannot be 'unknown'"):
        validate_provenance_completeness(bad_prov)

    # Missing git_commit
    bad_prov = dict(valid_prov)
    bad_prov["git_commit"] = "unknown"
    with pytest.raises(ValueError, match="git_commit cannot be 'unknown'"):
        validate_provenance_completeness(bad_prov)

    # Missing numpy dependency
    bad_prov = dict(valid_prov)
    bad_prov["dependency_versions"] = {"python": "3.10"}
    with pytest.raises(ValueError, match="must include python and numpy"):
        validate_provenance_completeness(bad_prov)


def test_harness_rejects_cold_only_evidence() -> None:
    """A single cold-start cycle cannot be classified as steady-state evidence."""
    harness = LatencyMeasurementHarness(deadline_ms=50.0, config_hash="fixture-config")
    with harness:
        harness.start_cycle()
        harness.end_cycle()

    with pytest.raises(ValueError, match="at least two cycles"):
        harness.get_metrics()


def test_harness_with_actual_planners(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harness must produce a full latency record on a fixture scenario for >= 2 planners."""
    from robot_sf.benchmark.map_runner import _run_map_episode
    from robot_sf.nav.map_config import MapDefinition

    # 1. Setup minimal map definition and stubs
    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.zeros((0, 2), dtype=float)
            self.goal_pos = [np.array([1.0, 1.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((0, 2), dtype=float)

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)
            self.step_count = 0

        def reset(self, seed: int | None = None):
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
                "goal": {"current": [1.0, 1.0], "next": [1.0, 1.0]},
                "pedestrians": {"positions": [], "velocities": []},
            }
            return obs, {}

        def step(self, action: Any):
            self.step_count += 1
            obs = {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
                "goal": {"current": [1.0, 1.0], "next": [1.0, 1.0]},
                "pedestrians": {"positions": [], "velocities": []},
            }
            return obs, 0.0, self.step_count >= 2, False, {"success": True}

        def close(self) -> None:
            return None

    map_def = MagicMock()
    dummy_config = type("Cfg", (), {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()})

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(map_def),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    scenario = {"name": "s1", "simulation_config": {"max_episode_steps": 2}}

    # 2. Run under harness for "goal" planner
    harness_goal = LatencyMeasurementHarness(deadline_ms=100.0)
    with harness_goal:
        _run_map_episode(
            scenario,
            seed=1,
            horizon=None,
            dt=0.1,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo="goal",
            algo_config_path=None,
            scenario_path=Path("."),
        )
    metrics_goal = harness_goal.get_metrics()
    assert "steady_state_averages" in metrics_goal
    assert len(metrics_goal["cycles"]) >= 2
    assert any(cycle["action_conversion_ms"] > 0.0 for cycle in metrics_goal["cycles"])
    assert metrics_goal["classification"] in {
        "meets_budget_on_measured_host",
        "misses_budget_on_measured_host",
    }

    # 3. Run under harness for "mppi_social" planner
    harness_mppi = LatencyMeasurementHarness(deadline_ms=100.0)
    with harness_mppi:
        _run_map_episode(
            scenario,
            seed=1,
            horizon=None,
            dt=0.1,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo="mppi_social",
            algo_config_path=None,
            scenario_path=Path("."),
            algo_config={
                "goal_tolerance": 0.2,
                "horizon_steps": 5,
                "sample_count": 8,
                "iterations": 1,
                "elite_fraction": 0.25,
            },
        )
    metrics_mppi = harness_mppi.get_metrics()
    assert "steady_state_averages" in metrics_mppi
    assert len(metrics_mppi["cycles"]) >= 2
    assert any(cycle["action_conversion_ms"] > 0.0 for cycle in metrics_mppi["cycles"])
    assert metrics_mppi["classification"] in {
        "meets_budget_on_measured_host",
        "misses_budget_on_measured_host",
    }
