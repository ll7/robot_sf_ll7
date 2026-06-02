"""Tests for paired failure-mechanism classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import jsonschema

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.failure_mechanism_classifier import (
    CLASSIFICATION_SOURCE,
    SCHEMA_VERSION,
    classify_failure_mechanisms,
    classify_failure_mechanisms_from_jsonl,
)


def _record(  # noqa: PLR0913
    scenario_id: str,
    *,
    horizon: int,
    episode_id: str | None = None,
    success: int | bool = 0,
    termination_reason: str = "timeout",
    collisions: int = 0,
    near_misses: int = 0,
    min_distance: float = 2.0,
    comfort_exposure: float = 0.0,
    time_to_goal_norm: float = 1.0,
    status: str | None = None,
) -> dict[str, Any]:
    """Build one compact synthetic episode record."""
    record: dict[str, Any] = {
        "episode_id": episode_id or f"{scenario_id}-h{horizon}",
        "scenario_id": scenario_id,
        "algo": "orca",
        "seed": 111,
        "horizon": horizon,
        "termination_reason": termination_reason,
        "outcome": {
            "route_complete": bool(success),
            "timeout": termination_reason == "timeout",
            "collision_event": collisions > 0,
        },
        "metrics": {
            "success": success,
            "collisions": collisions,
            "near_misses": near_misses,
            "min_distance": min_distance,
            "comfort_exposure": comfort_exposure,
            "time_to_goal_norm": time_to_goal_norm,
        },
    }
    if status is not None:
        record["availability_status"] = status
    return record


def _canonical_record(
    scenario_id: str,
    *,
    horizon: int,
    success: bool,
    termination_reason: str = "truncated",
    timeout_event: bool = True,
    algorithm_status: str | None = None,
) -> dict[str, Any]:
    """Build a compact map-runner-style episode record."""
    record = _record(
        scenario_id,
        horizon=horizon,
        success=success,
        termination_reason=termination_reason,
        time_to_goal_norm=1.0 if not success else 0.5,
    )
    record["outcome"].pop("timeout")
    record["outcome"]["timeout_event"] = timeout_event
    if algorithm_status is not None:
        record["algorithm_metadata"] = {"status": algorithm_status}
    return record


def _labels(payload: dict[str, Any]) -> dict[str, str]:
    """Map scenario id to emitted label."""
    return {row["scenario_id"]: row["label"] for row in payload["rows"]}


def test_classifies_acceptance_failure_mechanisms() -> None:
    """Classifier should cover the #2012 mechanism labels from paired fixtures."""
    records = [
        _record("clean_relief", horizon=100, comfort_exposure=0.0, min_distance=1.5),
        _record(
            "clean_relief",
            horizon=500,
            success=1,
            termination_reason="success",
            comfort_exposure=0.0,
            min_distance=1.5,
            time_to_goal_norm=0.4,
        ),
        _record("exposure_completion", horizon=100, comfort_exposure=0.1, min_distance=2.2),
        _record(
            "exposure_completion",
            horizon=500,
            success=1,
            termination_reason="success",
            comfort_exposure=0.4,
            min_distance=1.4,
            time_to_goal_norm=0.6,
        ),
        _record("collision_regression", horizon=100, comfort_exposure=0.0, min_distance=2.0),
        _record(
            "collision_regression",
            horizon=500,
            termination_reason="collision",
            collisions=1,
            comfort_exposure=0.2,
            min_distance=0.1,
        ),
        _record("persistent_timeout", horizon=100, time_to_goal_norm=1.0),
        _record("persistent_timeout", horizon=500, time_to_goal_norm=0.98),
    ]

    payload = classify_failure_mechanisms(
        records,
        generated_at_utc="2026-06-01T00:00:00+00:00",
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert _labels(payload) == {
        "clean_relief": "time_budget_clean_relief",
        "collision_regression": "safety_regressed_long_horizon",
        "exposure_completion": "exposure_enabled_completion",
        "persistent_timeout": "persistent_low_progress_timeout",
    }
    assert payload["coverage_axes"]["failure_modes"]["collision"]["classification_source"] == (
        CLASSIFICATION_SOURCE
    )
    assert (
        payload["coverage_axes"]["failure_modes"]["time_budget_clean_relief"]["observed_episodes"]
        == 1
    )


def test_certification_blocker_prevents_planner_attribution() -> None:
    """Excluded scenario certificates should emit scenario_contract_blocker."""
    records = [
        _record("blocked_scenario", horizon=100),
        _record("blocked_scenario", horizon=500),
    ]
    payload = classify_failure_mechanisms(
        records,
        scenario_certificates={
            "blocked_scenario": {
                "schema_version": "scenario_cert.v1",
                "scenario_id": "blocked_scenario",
                "classification": "geometrically_infeasible",
                "benchmark_eligibility": "excluded",
                "reasons": ["route blocked"],
            }
        },
        generated_at_utc="2026-06-01T00:00:00+00:00",
    )

    row = payload["rows"][0]
    assert row["label"] == "scenario_contract_blocker"
    assert row["required_inputs_present"] is True
    assert "Scenario certificate blocks" in row["caveats"][0]


def test_unavailable_for_missing_pair_metrics_and_degraded_status() -> None:
    """Missing evidence and degraded execution should be unavailable, not inferred."""
    missing_metric = _record("missing_metric", horizon=500)
    missing_metric["metrics"].pop("min_distance")
    records = [
        _record("missing_pair", horizon=100),
        _record("missing_metric", horizon=100),
        missing_metric,
        _record("degraded", horizon=100, status="degraded"),
        _record("degraded", horizon=500, status="degraded"),
    ]

    payload = classify_failure_mechanisms(
        records,
        generated_at_utc="2026-06-01T00:00:00+00:00",
    )

    rows = {row["scenario_id"]: row for row in payload["rows"]}
    assert rows["missing_pair"]["label"] == "unavailable"
    assert rows["missing_pair"]["unavailable_reason"] == "missing_fixed_or_long_horizon_pair"
    assert rows["missing_metric"]["label"] == "unavailable"
    assert rows["missing_metric"]["unavailable_reason"] == "missing_required_metrics"
    assert rows["degraded"]["label"] == "unavailable"
    assert rows["degraded"]["unavailable_reason"].startswith("unavailable_execution_status")
    assert (
        payload["coverage_axes"]["failure_modes"]["timeout_without_progress"]["observed_episodes"]
        == 0
    )


def test_canonical_boolean_success_and_timeout_event_classify_timeout() -> None:
    """Canonical boolean success and timeout_event fields should not fail closed."""
    records = [
        _canonical_record("canonical_timeout", horizon=100, success=False),
        _canonical_record("canonical_timeout", horizon=500, success=False),
    ]

    payload = classify_failure_mechanisms(
        records,
        generated_at_utc="2026-06-01T00:00:00+00:00",
    )

    row = payload["rows"][0]
    assert row["label"] == "persistent_low_progress_timeout"
    assert row["required_inputs_present"] is True
    assert row["unavailable_reason"] is None


def test_nested_algorithm_metadata_status_fails_closed() -> None:
    """Canonical nested fallback status should become unavailable."""
    records = [
        _canonical_record("fallback_pair", horizon=100, success=False, algorithm_status="fallback"),
        _canonical_record("fallback_pair", horizon=500, success=True, algorithm_status="fallback"),
    ]

    payload = classify_failure_mechanisms(
        records,
        generated_at_utc="2026-06-01T00:00:00+00:00",
    )

    row = payload["rows"][0]
    assert row["label"] == "unavailable"
    assert row["unavailable_reason"] == (
        "unavailable_execution_status:algorithm_metadata.status=fallback"
    )


def test_cli_writes_json_csv_and_schema_valid_payload(tmp_path: Path) -> None:
    """CLI should emit JSON and CSV surfaces from a paired JSONL fixture."""
    jsonl_path = tmp_path / "episodes.jsonl"
    out_json = tmp_path / "classified.json"
    out_csv = tmp_path / "classified.csv"
    schema = json.loads(
        Path(
            "robot_sf/benchmark/schemas/failure_mechanism_classification.schema.v1.json"
        ).read_text(encoding="utf-8")
    )
    records = [
        _record("near_miss_pair", horizon=100, near_misses=1),
        _record("near_miss_pair", horizon=500, near_misses=1),
    ]
    jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    exit_code = cli_main(
        [
            "classify-failure-mechanisms",
            "--episodes-jsonl",
            str(jsonl_path),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema)
    assert payload["rows"][0]["label"] == "near_miss"
    assert payload["coverage_axes"]["failure_modes"]["near_miss"]["observed_episodes"] == 1
    csv_rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    assert csv_rows[0]["scenario_id"] == "near_miss_pair"
    assert csv_rows[0]["label"] == "near_miss"


def test_classify_failure_mechanisms_from_jsonl_loads_certificates(tmp_path: Path) -> None:
    """JSONL helper should load scenario certificate inputs and preserve input provenance."""
    jsonl_path = tmp_path / "episodes.jsonl"
    cert_path = tmp_path / "certs.jsonl"
    jsonl_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in [
                _record("cert_blocked", horizon=100),
                _record("cert_blocked", horizon=500),
            ]
        ),
        encoding="utf-8",
    )
    cert_path.write_text(
        json.dumps(
            {
                "schema_version": "scenario_cert.v1",
                "scenario_id": "cert_blocked",
                "classification": "invalid",
                "benchmark_eligibility": "excluded",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = classify_failure_mechanisms_from_jsonl(
        jsonl_path,
        scenario_certificates=cert_path,
        generated_at_utc="2026-06-01T00:00:00+00:00",
    )

    assert payload["rows"][0]["label"] == "scenario_contract_blocker"
    assert payload["inputs"]["episodes_jsonl"] == [str(jsonl_path)]
    assert payload["inputs"]["scenario_certificates"] == str(cert_path)
