"""Tests for the real-trace validation-contract checker (issue #3278)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from robot_sf.analysis_workbench import (
    CONTRACT_EVIDENCE_BOUNDARY,
    REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION,
    RealTraceValidationContractError,
    check_real_trace_validation_contract,
    load_real_trace_validation_contract,
)
from robot_sf.analysis_workbench.real_trace_validation_contract import (
    CONTRACT_STATUS_BLOCKED,
    CONTRACT_STATUS_READY,
    PREDICATE_STATUS_BLOCKED,
    PREDICATE_STATUS_VALIDATABLE,
)

EXAMPLE_DESCRIPTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_3278_real_trace_validation_contract_example.yaml"
)


def _complete_descriptor() -> dict:
    """Return a descriptor that is fully accessible and channel-complete for two predicates."""
    return {
        "schema_version": REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION,
        "dataset_id": "complete_candidate",
        "metadata": {
            "description": "Accessible candidate dataset with full kinematics and labels.",
            "license": "CC-BY-4.0",
            "provenance_status": "accepted",
            "access_status": "available",
            "coordinate_frame": "world_enu_meters",
            "units": {"distance": "m", "angular_velocity": "rad/s"},
        },
        "available_channels": [
            "robot.position",
            "pedestrians.id",
            "pedestrians.position",
            "planner.selected_action.linear_velocity",
            "planner.selected_action.angular_velocity",
        ],
        "available_event_labels": ["late_evasive", "oscillatory"],
        "target_predicates": ["late_evasive_reaction", "oscillatory_local_control"],
    }


def test_complete_descriptor_is_ready_and_validatable() -> None:
    """A complete, accessible descriptor yields a ready contract with validatable predicates."""
    report = check_real_trace_validation_contract(_complete_descriptor())

    assert report.contract_status == CONTRACT_STATUS_READY
    assert report.metadata_status == "complete"
    assert report.metadata_blockers == []
    assert report.provenance_blockers == []
    assert report.missing_data_blockers == []
    assert report.evidence_boundary == CONTRACT_EVIDENCE_BOUNDARY
    assert set(report.validatable_predicates) == {
        "late_evasive_reaction",
        "oscillatory_local_control",
    }
    assert report.blocked_predicates == []
    # Ground-truth labels declared -> no computable-only limitation for these predicates.
    for record in report.predicate_compatibility:
        assert record.status == PREDICATE_STATUS_VALIDATABLE
        assert record.ground_truth_label_available is True
        assert record.limitation is None


def test_incompatible_descriptor_blocks_predicates_with_missing_channels() -> None:
    """A descriptor missing required channels marks the affected predicates blocked."""
    descriptor = _complete_descriptor()
    # Drop the angular-velocity command channel both predicates need.
    descriptor["available_channels"] = ["robot.position", "pedestrians.position"]

    report = check_real_trace_validation_contract(descriptor)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert set(report.blocked_predicates) == {
        "late_evasive_reaction",
        "oscillatory_local_control",
    }
    assert report.validatable_predicates == []

    osc = next(
        r for r in report.predicate_compatibility if r.predicate_id == "oscillatory_local_control"
    )
    assert osc.status == PREDICATE_STATUS_BLOCKED
    assert "planner.selected_action.angular_velocity" in osc.missing_channels
    assert any(
        "planner.selected_action.angular_velocity" in blocker
        for blocker in report.missing_data_blockers
    )
    assert osc.limitation is not None and "missing required channel" in osc.limitation


def test_placeholder_license_yields_metadata_blocker() -> None:
    """A placeholder license passes schema but is reported as an incomplete-metadata blocker."""
    descriptor = _complete_descriptor()
    descriptor["metadata"]["license"] = "unknown"

    report = check_real_trace_validation_contract(descriptor)

    assert report.metadata_status == "incomplete"
    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("license" in blocker for blocker in report.metadata_blockers)


def test_empty_metadata_field_fails_schema() -> None:
    """A genuinely empty required metadata field is rejected at the schema layer."""
    descriptor = _complete_descriptor()
    descriptor["metadata"]["description"] = ""

    with pytest.raises(RealTraceValidationContractError):
        check_real_trace_validation_contract(descriptor)


def test_pending_access_and_provenance_block_contract() -> None:
    """Pending/blocked access and unaccepted provenance surface as provenance blockers."""
    descriptor = _complete_descriptor()
    descriptor["metadata"]["provenance_status"] = "pending"
    descriptor["metadata"]["access_status"] = "blocked"

    report = check_real_trace_validation_contract(descriptor)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("provenance_status" in blocker for blocker in report.provenance_blockers)
    assert any("access_status" in blocker for blocker in report.provenance_blockers)
    # Channels are still complete, so predicates remain channel-validatable.
    assert set(report.validatable_predicates) == {
        "late_evasive_reaction",
        "oscillatory_local_control",
    }


def test_computable_only_limitation_when_label_absent() -> None:
    """Channel-complete predicates without a declared ground-truth label note the limitation."""
    descriptor = _complete_descriptor()
    descriptor["available_event_labels"] = []  # kinematics but no observed labels

    report = check_real_trace_validation_contract(descriptor)

    for record in report.predicate_compatibility:
        assert record.status == PREDICATE_STATUS_VALIDATABLE
        assert record.ground_truth_label_available is False
        assert record.limitation is not None
        assert "no directly observed ground-truth label" in record.limitation


def test_label_derived_predicate_reports_label_dependency() -> None:
    """A predicate requiring an explicit event channel reports the label dependency limitation."""
    descriptor = _complete_descriptor()
    descriptor["available_channels"] = [
        "robot.position",
        "pedestrians.id",
        "pedestrians.position",
        "planner.event",
    ]
    descriptor["available_event_labels"] = ["deadlock"]
    descriptor["target_predicates"] = ["bottleneck_deadlock"]

    report = check_real_trace_validation_contract(descriptor)

    record = report.predicate_compatibility[0]
    assert record.predicate_id == "bottleneck_deadlock"
    assert record.status == PREDICATE_STATUS_VALIDATABLE
    assert "planner.event" in record.label_derived_channels
    assert record.ground_truth_label_available is True
    assert record.limitation is not None and "annotated channel" in record.limitation


def test_default_targets_cover_all_canonical_predicates() -> None:
    """Omitting target_predicates checks the full canonical predicate set."""
    descriptor = _complete_descriptor()
    descriptor.pop("target_predicates")

    report = check_real_trace_validation_contract(descriptor)

    ids = {record.predicate_id for record in report.predicate_compatibility}
    # All eight canonical predicates appear.
    assert len(ids) == 8
    assert "collision" in ids
    assert "low_progress" in ids


def test_unknown_target_predicate_raises() -> None:
    """An unknown predicate ID is rejected rather than silently ignored."""
    descriptor = _complete_descriptor()
    descriptor["target_predicates"] = ["not_a_real_predicate"]

    with pytest.raises(RealTraceValidationContractError, match="unknown target predicate"):
        check_real_trace_validation_contract(descriptor)


def test_schema_violation_raises() -> None:
    """A descriptor missing required top-level fields fails schema validation."""
    descriptor = _complete_descriptor()
    del descriptor["available_channels"]

    with pytest.raises(RealTraceValidationContractError):
        check_real_trace_validation_contract(descriptor)


def test_bad_provenance_enum_raises() -> None:
    """An out-of-vocabulary provenance status is rejected by the schema."""
    descriptor = _complete_descriptor()
    descriptor["metadata"]["provenance_status"] = "definitely-fine"

    with pytest.raises(RealTraceValidationContractError):
        check_real_trace_validation_contract(descriptor)


def test_matrix_union_adds_required_fields() -> None:
    """A supplied matrix can add required fields that turn a predicate blocked."""
    descriptor = _complete_descriptor()
    descriptor["target_predicates"] = ["oscillatory_local_control"]
    # Base definition for oscillatory only needs angular_velocity, which is present.
    base = check_real_trace_validation_contract(descriptor)
    assert base.validatable_predicates == ["oscillatory_local_control"]

    matrix = {
        "schema_version": "trace_predicate_matrix.v1",
        "matrix": {
            "required_trace_fields_by_predicate": {
                "oscillatory_local_control": ["planner.curvature_unavailable_channel"],
            }
        },
    }
    constrained = check_real_trace_validation_contract(descriptor, matrix=matrix)
    assert constrained.blocked_predicates == ["oscillatory_local_control"]


def test_example_descriptor_loads_and_is_blocked_external() -> None:
    """The committed example descriptor loads, validates, and reports its external-data block."""
    descriptor = load_real_trace_validation_contract(EXAMPLE_DESCRIPTOR_PATH)
    report = check_real_trace_validation_contract(descriptor, source=EXAMPLE_DESCRIPTOR_PATH)

    # Channels present make several predicates channel-validatable...
    assert "late_evasive_reaction" in report.validatable_predicates
    assert "oscillatory_local_control" in report.validatable_predicates
    # ...but the contract is blocked because access/provenance are not accepted.
    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.provenance_blockers
    # Event-label predicates without their channels are blocked with explicit gaps.
    assert "bottleneck_deadlock" in report.blocked_predicates
    assert report.evidence_boundary == CONTRACT_EVIDENCE_BOUNDARY


def test_load_missing_file_raises() -> None:
    """Loading a non-existent descriptor path raises a clear error."""
    with pytest.raises(RealTraceValidationContractError, match="not found"):
        load_real_trace_validation_contract(Path("/nonexistent/descriptor.yaml"))


def test_report_to_dict_is_json_safe() -> None:
    """The report serializes to nested plain dicts/lists for JSON output."""
    report = check_real_trace_validation_contract(_complete_descriptor())
    payload = report.to_dict()

    assert payload["schema_version"] == REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION
    assert isinstance(payload["predicate_compatibility"], list)
    assert isinstance(payload["predicate_compatibility"][0], dict)
    # Round-trip via copy.deepcopy proves no dataclass instances leaked through.
    assert copy.deepcopy(payload) == payload
