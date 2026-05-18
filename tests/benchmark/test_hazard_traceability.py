"""Tests for the ``hazard_traceability.v1`` mapping contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.hazard_traceability import (
    HAZARD_TRACEABILITY_SCHEMA_VERSION,
    HazardTraceabilityValidationError,
    hazard_traceability_from_dict,
    load_hazard_traceability,
    load_hazard_traceability_schema,
    summarize_hazard_coverage,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "hazard_traceability"
    / "low_speed_public_space_v1.yaml"
)


def test_load_fixture_as_typed_hazard_traceability_mapping() -> None:
    """Representative hazard mappings should validate independently of benchmark runs."""

    mapping = load_hazard_traceability(FIXTURE_PATH)

    assert mapping.schema_version == HAZARD_TRACEABILITY_SCHEMA_VERSION
    assert mapping.id == "low_speed_public_space_hazards_v1"
    assert [hazard.id for hazard in mapping.hazards] == [
        "robot_pedestrian_collision",
        "near_miss",
        "blind_corner_emergence",
        "keep_clear_violation",
        "pedestrian_flow_disruption",
    ]
    assert mapping.hazards[0].supporting_metrics == ["collision_rate"]
    assert mapping.scenario_mappings[0].scenario_families == ["station_platform"]

    jsonschema.validate(mapping.to_dict(), load_hazard_traceability_schema())


def test_invalid_hazard_mapping_sections_fail_closed_with_json_paths() -> None:
    """Malformed hazard traceability payloads should expose actionable paths."""

    payload = load_hazard_traceability(FIXTURE_PATH).to_dict()
    invalid_cases = [
        (
            lambda candidate: candidate.__setitem__(
                "schema_version",
                "hazard_traceability.v2",
            ),
            "/schema_version",
            "hazard_traceability.v1",
        ),
        (
            lambda candidate: candidate["hazards"][0].__setitem__("severity", "catastrophic"),
            "/hazards/0/severity",
            "catastrophic",
        ),
        (
            lambda candidate: candidate["hazards"][0].__setitem__("supporting_metrics", []),
            "/hazards/0/supporting_metrics",
            "should be non-empty",
        ),
        (
            lambda candidate: candidate["scenario_mappings"][0].__setitem__(
                "hazards",
                ["missing_hazard"],
            ),
            "/scenario_mappings/0/hazards",
            "unknown hazard id 'missing_hazard'",
        ),
    ]

    for mutate, expected_path, expected_fragment in invalid_cases:
        candidate = deepcopy(payload)
        mutate(candidate)
        with pytest.raises(HazardTraceabilityValidationError) as exc_info:
            hazard_traceability_from_dict(candidate, source=FIXTURE_PATH)
        message = str(exc_info.value)
        assert expected_path in message
        assert expected_fragment in message


def test_hazard_coverage_summary_reports_supported_and_unknown_hazards() -> None:
    """Coverage summaries should make unsupported hazards explicit."""

    mapping = load_hazard_traceability(FIXTURE_PATH)

    summary = summarize_hazard_coverage(
        mapping,
        scenario_families=["station_platform", "unknown_family"],
    )

    assert summary["schema_version"] == "hazard-traceability-coverage.v1"
    assert summary["covered_hazards"] == [
        "keep_clear_violation",
        "near_miss",
        "pedestrian_flow_disruption",
        "robot_pedestrian_collision",
    ]
    assert summary["unmapped_scenario_families"] == ["unknown_family"]
    assert summary["supporting_metrics_by_hazard"]["near_miss"] == ["min_ttc", "pet"]
    assert summary["evidence_fields_by_hazard"]["keep_clear_violation"] == [
        "scenario.metadata.platform_semantics",
        "metrics.keep_clear_intrusion_s",
    ]


def test_hazard_coverage_summary_filters_by_scenario_id() -> None:
    """Scenario IDs should resolve independently of family-only summaries."""

    mapping = load_hazard_traceability(FIXTURE_PATH)

    summary = summarize_hazard_coverage(
        mapping,
        scenario_ids=["francis2023_blind_corner", "missing_scenario"],
    )

    assert summary["covered_hazards"] == ["blind_corner_emergence", "near_miss"]
    assert summary["unmapped_scenario_ids"] == ["missing_scenario"]
