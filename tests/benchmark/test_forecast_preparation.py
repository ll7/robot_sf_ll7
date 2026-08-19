"""Regression tests for the issue #7602 forecast-preparation contract."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from robot_sf.benchmark.forecast.forecast_preparation import (
    ForecastPreparationSourceSpec,
    build_forecast_preparation_packet,
    validate_forecast_preparation_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_specs() -> tuple[ForecastPreparationSourceSpec, ...]:
    return (
        ForecastPreparationSourceSpec(
            path="tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/"
            "bottleneck_motion_rich_fixture.json",
            scenario_family="bottleneck",
            cutoff_frame_step=5,
        ),
        ForecastPreparationSourceSpec(
            path="docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/"
            "inputs/synthetic_crossing_proxy_orca_111_trace_export.json",
            scenario_family="crossing_proxy",
            cutoff_frame_step=2,
        ),
        ForecastPreparationSourceSpec(
            path="docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/"
            "ammv_social_force_trace_export.json",
            scenario_family="head_on_corridor",
            cutoff_frame_step=5,
        ),
    )


def _packet() -> dict:
    return build_forecast_preparation_packet(
        _source_specs(),
        repo_root=REPO_ROOT,
        horizons_s=(1.0,),
    )


def _validate(payload: dict) -> None:
    validate_forecast_preparation_packet(
        payload,
        repo_root=REPO_ROOT,
        verify_checksums=False,
    )


def test_packet_emits_matched_rows_and_explicit_ego_unavailability() -> None:
    """The packet has one identity-matched oracle/ego pair per selected source."""
    payload = _packet()

    assert payload["pair_count"] == 3
    assert payload["row_count"] == 6
    assert set(payload["coverage"]["scenario_families"]) == {
        "bottleneck",
        "crossing_proxy",
        "head_on_corridor",
    }
    assert set(payload["coverage"]["planners"]) == {
        "ammv_social_force",
        "hybrid_rule_v0_minimal",
        "orca",
    }
    assert payload["coverage"]["ego_observation_status"] == "not_available"

    rows_by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        rows_by_pair[row["pair_id"]].append(row)
    assert len(rows_by_pair) == 3
    for pair_rows in rows_by_pair.values():
        assert {row["observation_tier"] for row in pair_rows} == {
            "oracle_full_state",
            "ego_observation",
        }
        assert pair_rows[0]["identity"] == pair_rows[1]["identity"]
        assert pair_rows[0]["lineage"] == pair_rows[1]["lineage"]
        assert pair_rows[0]["target"] == pair_rows[1]["target"]
        ego_row = next(row for row in pair_rows if row["observation_tier"] == "ego_observation")
        assert ego_row["availability_status"] == "not_available"
        assert "pedestrian_position_m" not in ego_row["input"]
        assert "pedestrian_velocity_mps" not in ego_row["input"]
        assert all(
            not entry["future_target"]
            for entry in ego_row["field_leakage_ledger"]
            if entry["field"].startswith("input.")
        )
        assert any(
            entry["field"] == "target.future_position_m" and entry["future_target"]
            for entry in ego_row["field_leakage_ledger"]
        )

    _validate(payload)


def test_cross_partition_group_leakage_fails_closed() -> None:
    """A lineage group assigned to two splits cannot pass validation."""
    payload = _packet()
    first, second = payload["source_artifacts"][:2]
    second["lineage_group_id"] = first["lineage_group_id"]
    second["split"] = "test" if first["split"] != "test" else "train"
    for row in payload["rows"]:
        if row["lineage"]["source_path"] == second["path"]:
            row["lineage"]["lineage_group_id"] = second["lineage_group_id"]
            row["lineage"]["split"] = second["split"]

    with pytest.raises(ValueError, match="group leakage across splits"):
        _validate(payload)


def test_cross_partition_near_duplicate_fails_closed() -> None:
    """An exact normalized trajectory fingerprint cannot cross split boundaries."""
    payload = _packet()
    first, second = payload["source_artifacts"][:2]
    second["near_duplicate_fingerprint"] = first["near_duplicate_fingerprint"]

    with pytest.raises(ValueError, match="near-duplicate trajectory leakage"):
        _validate(payload)


def test_mismatched_pair_identity_fails_closed() -> None:
    """Changing one row's cutoff identity invalidates the pair contract."""
    payload = _packet()
    ego_row = next(row for row in payload["rows"] if row["observation_tier"] == "ego_observation")
    ego_row["identity"]["cutoff_time_s"] += 0.25

    with pytest.raises(ValueError, match="pair_id does not match identity"):
        _validate(payload)


def test_future_field_in_ego_input_fails_closed() -> None:
    """Future or target-labelled fields are forbidden in ego inputs."""
    payload = _packet()
    ego_row = next(row for row in payload["rows"] if row["observation_tier"] == "ego_observation")
    ego_row["input"]["future_target_label"] = "forbidden"

    with pytest.raises(ValueError, match="future/target field leaked into ego input"):
        _validate(payload)


def test_unavailable_source_status_cannot_be_promoted() -> None:
    """The current trace sample cannot silently claim an ego source is available."""
    payload = _packet()
    payload["source_artifacts"][0]["ego_observation_status"] = "available"

    with pytest.raises(ValueError, match="source ego_observation_status must be not_available"):
        _validate(payload)


def test_row_lineage_metadata_must_match_source_artifact() -> None:
    """A valid source hash cannot excuse contradictory row provenance metadata."""
    payload = _packet()
    payload["rows"][0]["lineage"]["planner_id"] = "fabricated_planner"

    with pytest.raises(ValueError, match="lineage metadata does not match source: planner_id"):
        _validate(payload)


def test_absolute_source_paths_are_rejected() -> None:
    """Preparation manifests must not retain machine-specific absolute source paths."""
    payload = _packet()
    relative_path = payload["source_artifacts"][0]["path"]
    payload["source_artifacts"][0]["path"] = str((REPO_ROOT / relative_path).resolve())

    with pytest.raises(
        ValueError, match=r"source_artifacts\[0\]\.path must be repository-relative"
    ):
        _validate(payload)


def test_false_reassurance_case_is_trace_backed_and_not_a_performance_claim() -> None:
    """The analytic counterexample records zero ADE/FDE with close robot clearance."""
    case = _packet()["ade_fde_false_reassurance_case"]

    assert case["status"] == "analytic_trace_backed_diagnostic_only"
    assert case["ade_m"] == 0.0
    assert case["fde_m"] == 0.0
    assert case["robot_pedestrian_clearance_m"] < case["risk_reference_m"]
