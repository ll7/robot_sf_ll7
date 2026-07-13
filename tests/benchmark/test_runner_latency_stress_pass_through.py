"""Regression tests for benchmark runner map-profile pass-through."""

from __future__ import annotations

from typing import Any

from robot_sf.benchmark import runner

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_run_batch_forwards_latency_stress_profile_to_map_runner(
    tmp_path,
    monkeypatch,
) -> None:
    """Map campaigns should receive both actuation and latency profile metadata."""
    calls: list[dict[str, Any]] = []

    def _fake_run_map_batch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"episodes_total": 0, "failed_jobs": 0, "out_path": str(args[1])}

    monkeypatch.setattr(runner, "run_map_batch", _fake_run_map_batch)
    latency_profile = {
        "name": "learned-policy-latency-stress-v0",
        "profile_version": "v0",
        "claim_scope": "synthetic-only",
        "observation_delay_steps": 0,
        "action_delay_steps": 1,
        "planner_update_mode": "hold-last",
        "planner_update_period_steps": 2,
        "inference_timeout_ms": 200.0,
        "non_success_statuses": ["fallback", "degraded", "timeout", "not_available", "failed"],
    }
    actuation_profile = {
        "name": "amv-actuation-stress-v0",
        "profile_version": "v0",
        "claim_scope": "synthetic-only",
        "claim_boundary": "diagnostic-only",
        "max_linear_accel_m_s2": 2.0,
        "max_linear_decel_m_s2": 2.5,
        "max_yaw_rate_rad_s": 1.2,
        "max_angular_accel_rad_s2": 4.0,
        "latency_mode": "one-step-delay",
        "update_mode": "5hz-hold",
    }

    runner.run_batch(
        [{"name": "map_scenario", "map_file": "maps/svg_maps/classic_crossing.svg"}],
        tmp_path / "episodes.jsonl",
        SCHEMA_PATH,
        synthetic_actuation_profile=actuation_profile,
        latency_stress_profile=latency_profile,
    )

    assert calls[0]["synthetic_actuation_profile"] == actuation_profile
    assert calls[0]["latency_stress_profile"] == latency_profile


def test_run_batch_forwards_circuit_breaker_threshold_to_map_runner(
    tmp_path,
    monkeypatch,
) -> None:
    """Map campaigns must receive the configured circuit-breaker threshold."""
    calls: list[dict[str, Any]] = []

    def _fake_run_map_batch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"episodes_total": 0, "failed_jobs": 0, "out_path": str(args[1])}

    monkeypatch.setattr(runner, "run_map_batch", _fake_run_map_batch)
    runner.run_batch(
        [{"name": "map_scenario", "map_file": "maps/svg_maps/classic_crossing.svg"}],
        tmp_path / "episodes.jsonl",
        SCHEMA_PATH,
        circuit_breaker_threshold=3,
    )

    assert calls[0]["circuit_breaker_threshold"] == 3
