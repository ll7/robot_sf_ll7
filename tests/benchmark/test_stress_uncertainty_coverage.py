"""Tests for stress/uncertainty coverage report building."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.stress_uncertainty_coverage import (
    LEGACY_AGGREGATE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    StressUncertaintyCoverageError,
    _build_metric_groups,
    build_stress_uncertainty_coverage_report_from_jsonl,
    load_stress_uncertainty_coverage_payload,
)

FIXTURE_DIR = Path("tests/fixtures/stress_uncertainty_coverage")


def _write_episode_jsonl(path: Path) -> None:
    """Write a compact synthetic campaign JSONL file."""
    records = [
        {
            "episode_id": "episode-1",
            "scenario_id": "scenario-a",
            "scenario_params": {
                "density_label": "low",
                "flow_type": "crossing",
                "horizon_steps": 100,
            },
            "termination_reason": "collision",
            "outcome": {"collision_event": True, "route_complete": False},
            "metrics": {
                "collisions": 1,
                "near_misses": 0,
                "min_distance": 0.2,
                "comfort_exposure": 0.2,
                "success": 0,
                "time_to_goal_norm": 1.0,
            },
        },
        {
            "episode_id": "episode-2",
            "scenario_id": "scenario-b",
            "scenario_params": {
                "density_label": "low",
                "flow_type": "crossing",
                "horizon_steps": 100,
            },
            "termination_reason": "timeout",
            "outcome": {"timeout": True, "route_complete": False},
            "metrics": {
                "collisions": 0,
                "near_misses": 1,
                "min_distance": 0.5,
                "comfort_exposure": 0.0,
                "success": 0,
                "time_to_goal_norm": 1.0,
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_stress_uncertainty_coverage_report_from_jsonl(tmp_path: Path) -> None:
    """Builder should emit required uncertainty and coverage fields from campaign JSONL."""
    jsonl_path = tmp_path / "episodes.jsonl"
    _write_episode_jsonl(jsonl_path)

    report = build_stress_uncertainty_coverage_report_from_jsonl(
        jsonl_path,
        report_id="test-report",
        campaign_config_hash="cfg-sha256",
        scenario_matrix_hash="matrix-sha256",
        generated_at_utc="2026-05-22T00:00:00+00:00",
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["availability_status"] == "available"
    assert report["coverage_axes"]["failure_modes"]["collision"]["observed_episodes"] == 1
    assert report["coverage_axes"]["failure_modes"]["near_miss"]["observed_episodes"] == 1
    assert (
        report["coverage_axes"]["failure_modes"]["timeout_without_progress"]["observed_episodes"]
        == 1
    )
    assert report["coverage_axes"]["scenario_parameters"]["density_label"] == {
        "observed_values": ["low"],
        "required_values": ["low"],
        "coverage_status": "full",
    }
    assert report["metric_groups"]["safety"]["collision_rate"] == 0.5
    assert report["metric_groups"]["comfort"]["comfort_exposure"] == 0.1
    assert report["metric_groups"]["efficiency"]["time_to_goal_norm"] == 1.0


def test_load_legacy_aggregate_summary_is_coverage_unavailable() -> None:
    """Pre-v1 aggregate summaries should parse without becoming coverage evidence."""
    payload = load_stress_uncertainty_coverage_payload(FIXTURE_DIR / "legacy_aggregate_v1.json")

    assert payload["schema_version"] == LEGACY_AGGREGATE_SCHEMA_VERSION
    assert payload["availability_status"] == "not_available"
    assert payload["missing_fields"] == ["stress_uncertainty_coverage"]
    assert "legacy_payload" in payload


def test_load_valid_v1_fixture() -> None:
    """Minimal v1 fixture should pass the typed loader."""
    payload = load_stress_uncertainty_coverage_payload(FIXTURE_DIR / "minimal_valid_v1.json")

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["missing_fields"] == []


def test_valid_v1_fixture_matches_json_schema() -> None:
    """Minimal v1 fixture should validate against the canonical JSON Schema."""
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/stress_uncertainty_coverage.schema.v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload = json.loads((FIXTURE_DIR / "minimal_valid_v1.json").read_text(encoding="utf-8"))

    jsonschema.validate(instance=payload, schema=schema)


def test_required_v1_missing_failure_modes_fails_closed() -> None:
    """Required-mode v1 reports must reject missing failure-mode coverage."""
    with pytest.raises(StressUncertaintyCoverageError, match="coverage_axes.failure_modes"):
        load_stress_uncertainty_coverage_payload(FIXTURE_DIR / "malformed_required_v1.json")


def test_required_jsonl_missing_required_metric_fails_closed(tmp_path: Path) -> None:
    """Required-mode JSONL builds must reject missing required metric fields."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "episode_id": "episode-1",
                "scenario_id": "scenario-a",
                "scenario_params": {"density_label": "low"},
                "outcome": {"route_complete": True},
                "metrics": {"collisions": 0, "success": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StressUncertaintyCoverageError, match="metric_groups.safety.min_distance"):
        build_stress_uncertainty_coverage_report_from_jsonl(
            jsonl_path,
            report_id="missing-metric",
            campaign_config_hash="cfg-sha256",
            scenario_matrix_hash="matrix-sha256",
        )


def test_stress_coverage_report_cli_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI should build and write the report from JSONL input."""
    jsonl_path = tmp_path / "episodes.jsonl"
    out_path = tmp_path / "stress_report.json"
    _write_episode_jsonl(jsonl_path)

    exit_code = cli_main(
        [
            "stress-coverage-report",
            "--episodes-jsonl",
            str(jsonl_path),
            "--out",
            str(out_path),
            "--report-id",
            "cli-report",
            "--campaign-config-hash",
            "cfg-sha256",
            "--scenario-matrix-hash",
            "matrix-sha256",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["report_id"] == "cli-report"


def test_missing_success_metric_counts_against_total_episodes(tmp_path: Path) -> None:
    """Missing success metrics should count as 0.0, not disappear from the denominator."""
    records = [
        {
            "episode_id": "has-success",
            "scenario_id": "sc-a",
            "scenario_params": {"density_label": "low"},
            "outcome": {"route_complete": True, "collision_event": False},
            "metrics": {
                "collisions": 0,
                "min_distance": 2.0,
                "comfort_exposure": 0.1,
                "success": 1,
                "time_to_goal_norm": 0.5,
            },
        },
        {
            "episode_id": "missing-success",
            "scenario_id": "sc-b",
            "scenario_params": {"density_label": "low"},
            "outcome": {"route_complete": True, "collision_event": False},
            "metrics": {
                "collisions": 0,
                "min_distance": 1.5,
                "comfort_exposure": 0.2,
                "time_to_goal_norm": 0.7,
            },
        },
    ]
    jsonl_path = tmp_path / "mixed_success.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = build_stress_uncertainty_coverage_report_from_jsonl(
        jsonl_path,
        report_id="mixed-success",
        campaign_config_hash="cfg-sha256",
        scenario_matrix_hash="matrix-sha256",
    )

    assert report["metric_groups"]["efficiency"]["success"] == pytest.approx(0.5)


class TestCollisionRateZeroTotalSafety:
    """Prove collision_rate is safe when total=0."""

    def test_collision_rate_zero_total_returns_zero(self) -> None:
        """_build_metric_groups should return collision_rate=0.0 when total=0."""
        result = _build_metric_groups([], [])
        assert result["safety"]["collision_rate"] == 0.0
        assert result["safety"]["collisions"] == 0
