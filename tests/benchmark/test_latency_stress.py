"""Tests for latency-stress preflight profile helpers."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.latency_stress import (
    LatencyStressProfile,
    load_latency_stress_profile,
    not_available_latency_metrics,
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
