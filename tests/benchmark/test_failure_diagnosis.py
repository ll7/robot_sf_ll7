"""Tests for the optional failure-diagnosis record and deterministic adapter (#6583)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_IDS,
    TraceFailurePredicate,
)
from robot_sf.benchmark.failure_diagnosis import (
    ALLOWED_FAILURE_TYPES,
    CORRECTION_STATUSES,
    DEFAULT_CORRECTION_STATUS,
    DETECTION_METHOD_PREDICATE,
    DIAGNOSIS_SEVERITIES,
    DIAGNOSIS_SOURCE,
    FAILURE_DIAGNOSIS_SCHEMA_VERSION,
    FAILURE_LEVELS,
    FailureDiagnosisError,
    build_failure_diagnosis_payload,
    diagnose_from_trace_failure_predicate,
    diagnose_from_trace_failure_predicates,
    unknown_failure_diagnosis_record,
    validate_failure_diagnosis_payload,
    validate_failure_diagnosis_record,
)
from robot_sf.benchmark.failure_mechanism_classifier import FAILURE_MECHANISM_LABELS
from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_CONFIDENCES,
    MECHANISM_EVIDENCE_MODES,
)

_VALID = "valid"
_NOT_AVAILABLE = "not_available"


def _predicate(  # noqa: PLR0913
    predicate_id: str,
    *,
    time_interval_s: list[float | None] | None = None,
    steps: list[int | None] | None = None,
    involved_actors: list[str] | None = None,
    scenario_family: str = "crosswalk",
    planner_id: str = "orca",
    evidence_fields: dict[str, Any] | None = None,
    severity: str = "high",
    validity_status: str = _VALID,
) -> TraceFailurePredicate:
    """Build one synthetic trace failure predicate."""
    return TraceFailurePredicate(
        predicate_id=predicate_id,
        time_interval_s=list(time_interval_s if time_interval_s is not None else [1.0, 1.5]),
        steps=list(steps if steps is not None else [10, 15]),
        involved_actors=list(
            involved_actors if involved_actors is not None else ["robot", "ped_0"]
        ),
        scenario_family=scenario_family,
        planner_id=planner_id,
        evidence_fields=dict(evidence_fields or {}),
        severity=severity,
        validity_status=validity_status,
    )


def test_onset_interval_pinned_from_predicate_time_interval() -> None:
    """onset_time_s and onset_interval must come from time_interval_s verbatim."""
    predicate = _predicate(
        "collision",
        time_interval_s=[12.5, 13.0],
        steps=[125, 130],
        severity="critical",
    )
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "collision"
    assert record.onset_time_s == 12.5
    assert record.onset_interval == [12.5, 13.0]
    # Onset endpoints must be exactly the predicate interval values.
    assert record.source_predicate["time_interval_s"] == [12.5, 13.0]


def test_onset_handles_none_interval_endpoints() -> None:
    """Missing interval endpoints must become None rather than crash."""
    predicate = _predicate(
        "collision",
        time_interval_s=[None, None],
        steps=[None, None],
        validity_status=_NOT_AVAILABLE,
        severity="critical",
    )
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.onset_time_s is None
    assert record.onset_interval == [None, None]


def test_severity_mapped_deterministically_from_predicate() -> None:
    """Predicate severity tokens must map to minor|major|critical deterministically."""
    cases = {
        "critical": "critical",
        "high": "critical",
        "medium": "major",
        "moderate": "major",
        "low": "minor",
        "minor": "minor",
    }
    for token, expected in cases.items():
        record = diagnose_from_trace_failure_predicate(_predicate("collision", severity=token))
        assert record.severity == expected, f"severity {token!r} -> {record.severity!r}"


def test_severity_unknown_when_predicate_validity_not_valid() -> None:
    """Invalid predicate validity must yield unknown severity even with a severity token."""
    record = diagnose_from_trace_failure_predicate(
        _predicate("collision", severity="critical", validity_status=_NOT_AVAILABLE)
    )
    assert record.severity == "unknown"
    assert record.severity in DIAGNOSIS_SEVERITIES


def test_causal_evidence_cites_predicate_pointers_only() -> None:
    """causal_evidence must cite predicate evidence pointers and a non-causal note."""
    predicate = _predicate(
        "clearance_critical_interaction",
        time_interval_s=[3.0, 3.4],
        steps=[30, 34],
        involved_actors=["robot", "ped_7"],
        evidence_fields={"distance_m": 0.35, "clearance_threshold_m": 0.4},
        severity="high",
    )
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "near_miss"
    assert len(record.causal_evidence) == 1
    pointer = record.causal_evidence[0]
    assert pointer["evidence_kind"] == "trace_failure_predicate"
    assert pointer["predicate_id"] == "clearance_critical_interaction"
    assert pointer["time_interval_s"] == [3.0, 3.4]
    assert pointer["steps"] == [30, 34]
    assert pointer["involved_actors"] == ["robot", "ped_7"]
    # Source evidence_fields are preserved verbatim on the cited pointer.
    assert pointer["evidence_fields"] == {
        "distance_m": 0.35,
        "clearance_threshold_m": 0.4,
    }
    note = str(pointer["non_causal_note"]).lower()
    assert "not causal inference" in note


def test_unknown_for_oscillation_not_represented_in_classifier_labels() -> None:
    """Oscillation maps to unknown with an explicit reason; evidence is preserved."""
    predicate = _predicate(
        "oscillatory_local_control",
        time_interval_s=[2.0, 9.0],
        steps=[20, 90],
        evidence_fields={"sign_changes": 5},
        severity="medium",
    )
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.failure_level == "control"
    assert record.unknown_reason == "oscillation_not_represented_in_classifier_labels"
    assert record.confidence == "unknown"
    assert record.evidence_mode == "unknown"
    # Onset and evidence pointers are still preserved on the unknown record.
    assert record.onset_interval == [2.0, 9.0]
    assert record.causal_evidence[0]["evidence_fields"] == {"sign_changes": 5}
    assert record.severity == "major"


def test_unknown_for_unsupported_predicate_id() -> None:
    """Unsupported predicate ids must resolve to unknown with a reason."""
    predicate = _predicate("future_unmodelled_predicate", severity="high")
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "unsupported_predicate_id:future_unmodelled_predicate"
    assert record.failure_level == "analysis"


def test_unknown_for_invalid_validity_status() -> None:
    """Predicates whose validity is not 'valid' must resolve to unknown with a reason."""
    predicate = _predicate("collision", validity_status=_NOT_AVAILABLE, severity="critical")
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "predicate_validity_not_valid:not_available"
    assert record.confidence == "unknown"
    assert record.evidence_mode == "unknown"
    assert record.severity == "unknown"
    # The invalid validity_status is preserved verbatim.
    assert record.validity_status == _NOT_AVAILABLE


def test_low_progress_and_stuck_predicates_map_to_timeout_labels() -> None:
    """Low-progress, zero-motion, and bottleneck predicates map to timeout labels."""
    low_progress = diagnose_from_trace_failure_predicate(
        _predicate("low_progress", severity="medium")
    )
    zero_motion = diagnose_from_trace_failure_predicate(
        _predicate("zero_motion_timeout_behavior", severity="medium")
    )
    bottleneck = diagnose_from_trace_failure_predicate(
        _predicate("bottleneck_deadlock", severity="medium")
    )
    assert low_progress.failure_type == "timeout_without_progress"
    assert low_progress.failure_level == "control"
    assert zero_motion.failure_type == "persistent_low_progress_timeout"
    assert bottleneck.failure_type == "persistent_low_progress_timeout"
    for record in (low_progress, zero_motion, bottleneck):
        assert record.failure_type in FAILURE_MECHANISM_LABELS
        assert record.confidence == "supported_hypothesis"
        assert record.evidence_mode == "direct_probe"


def test_near_miss_predicates_map_to_near_miss_label() -> None:
    """Clearance, occlusion, and late-evasive predicates map to the near_miss label."""
    for predicate_id in (
        "clearance_critical_interaction",
        "occlusion_triggered_near_miss",
        "late_evasive_reaction",
    ):
        record = diagnose_from_trace_failure_predicate(_predicate(predicate_id))
        assert record.failure_type == "near_miss", predicate_id
        assert record.failure_level == "interaction"


def test_mapped_failure_type_reuses_classifier_labels_and_vocab() -> None:
    """Mapped failure types must be classifier labels; confidence/mode from taxonomy."""
    collision = diagnose_from_trace_failure_predicate(_predicate("collision"))
    assert collision.failure_type == "collision"
    assert collision.failure_type in FAILURE_MECHANISM_LABELS
    assert collision.failure_type in ALLOWED_FAILURE_TYPES
    assert collision.confidence in MECHANISM_CONFIDENCES
    assert collision.evidence_mode in MECHANISM_EVIDENCE_MODES
    assert collision.failure_level in FAILURE_LEVELS
    assert collision.detection_method == DETECTION_METHOD_PREDICATE


def test_correction_fields_optional_and_default_unreviewed() -> None:
    """proposed_correction defaults to None and correction_status to unreviewed."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    assert record.proposed_correction is None
    assert record.correction_status == DEFAULT_CORRECTION_STATUS == "unreviewed"
    assert record.correction_status in CORRECTION_STATUSES


def test_correction_inputs_propagated_and_validated() -> None:
    """Explicit correction inputs are propagated; bad status is rejected."""
    record = diagnose_from_trace_failure_predicate(
        _predicate("collision"),
        proposed_correction="Widen clearance buffer.",
        correction_status="accepted",
    )
    assert record.proposed_correction == "Widen clearance buffer."
    assert record.correction_status == "accepted"
    with pytest.raises(FailureDiagnosisError, match="unsupported correction_status"):
        diagnose_from_trace_failure_predicate(_predicate("collision"), correction_status="bogus")


def test_validity_status_and_evidence_fields_preserved_verbatim() -> None:
    """The source predicate validity_status and evidence_fields must be preserved."""
    predicate = _predicate(
        "collision",
        evidence_fields={"distance_m": 0.05, "robot_radius": 0.3, "pedestrian_radius": 0.3},
        validity_status=_VALID,
    )
    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.validity_status == _VALID
    assert record.source_predicate["evidence_fields"] == predicate.evidence_fields
    assert record.source_predicate["validity_status"] == _VALID


def test_dict_form_predicate_is_adapted_identically() -> None:
    """The adapter must accept a predicate dict form with the same result."""
    predicate = _predicate("collision", severity="critical")
    from_record = diagnose_from_trace_failure_predicate(predicate)
    from_dict = diagnose_from_trace_failure_predicate(predicate.to_dict())

    assert from_record.to_dict() == from_dict.to_dict()


def test_record_is_json_serializable() -> None:
    """The diagnosis record must round-trip through JSON without loss."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision", severity="high"))
    encoded = json.loads(json.dumps(record.to_dict()))
    assert encoded["failure_type"] == "collision"
    assert encoded["diagnosis_schema_version"] == FAILURE_DIAGNOSIS_SCHEMA_VERSION


def test_validate_record_accepts_valid_record() -> None:
    """validate_failure_diagnosis_record must accept a record produced by the adapter."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    normalized = validate_failure_diagnosis_record(record.to_dict())
    assert normalized["diagnosis_schema_version"] == FAILURE_DIAGNOSIS_SCHEMA_VERSION
    assert normalized["failure_type"] == "collision"


def test_validate_record_rejects_unknown_without_reason() -> None:
    """An unknown failure_type requires a non-empty unknown_reason."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["failure_type"] = "unknown"
    payload["unknown_reason"] = None
    with pytest.raises(FailureDiagnosisError, match="unknown_reason is required"):
        validate_failure_diagnosis_record(payload)


def test_validate_record_rejects_known_with_reason() -> None:
    """A known failure_type must not carry an unknown_reason."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["unknown_reason"] = "should not be set"
    with pytest.raises(FailureDiagnosisError, match="unknown_reason must be None"):
        validate_failure_diagnosis_record(payload)


def test_validate_record_rejects_bad_vocab_and_missing_fields() -> None:
    """validate must reject out-of-range vocab and missing required fields."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    bad_level = record.to_dict()
    bad_level["failure_level"] = "orbital"
    with pytest.raises(FailureDiagnosisError, match="unsupported failure_level"):
        validate_failure_diagnosis_record(bad_level)

    bad_type = record.to_dict()
    bad_type["failure_type"] = "invented_label"
    with pytest.raises(FailureDiagnosisError, match="unsupported failure_type"):
        validate_failure_diagnosis_record(bad_type)

    missing = record.to_dict()
    del missing["caveats"]
    with pytest.raises(FailureDiagnosisError, match="missing required field"):
        validate_failure_diagnosis_record(missing)


def test_validate_record_rejects_non_two_element_onset_interval() -> None:
    """onset_interval must be a two-element list."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["onset_interval"] = [1.0]
    with pytest.raises(FailureDiagnosisError, match="onset_interval"):
        validate_failure_diagnosis_record(payload)


def test_validate_record_rejects_inconsistent_onset_localization() -> None:
    """onset_time_s must remain the first endpoint of onset_interval."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["onset_time_s"] = 99.0
    with pytest.raises(FailureDiagnosisError, match="onset_time_s must equal onset_interval"):
        validate_failure_diagnosis_record(payload)


def test_validate_record_rejects_reversed_onset_interval() -> None:
    """The end of an onset interval cannot precede its start."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["onset_interval"] = [2.0, 1.0]
    payload["onset_time_s"] = 2.0
    with pytest.raises(FailureDiagnosisError, match="end must not precede"):
        validate_failure_diagnosis_record(payload)


def test_reversed_predicate_interval_fails_closed_to_valid_unknown_record() -> None:
    """Malformed predicate timing must not let the adapter emit an invalid record."""
    predicate = _predicate(
        "collision",
        time_interval_s=[2.0, 1.0],
        steps=[20, 10],
        severity="critical",
    )

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.failure_level == "interaction"
    assert record.unknown_reason == "invalid_time_interval:end_precedes_start"
    assert record.onset_time_s is None
    assert record.onset_interval == [None, None]
    # Raw malformed timing remains traceable in the source/evidence pointer.
    assert record.source_predicate["time_interval_s"] == [2.0, 1.0]
    assert record.causal_evidence[0]["time_interval_s"] == [2.0, 1.0]
    validate_failure_diagnosis_record(record.to_dict())


def test_unknown_failure_diagnosis_record_helper_mirrors_taxonomy_unknown() -> None:
    """unknown_failure_diagnosis_record mirrors unknown_failure_mechanism_record."""
    predicate = _predicate("collision", validity_status=_NOT_AVAILABLE, severity="critical")
    record = unknown_failure_diagnosis_record(predicate, "explicit_blocker_reason")
    assert record.failure_type == "unknown"
    assert record.confidence == "unknown"
    assert record.evidence_mode == "unknown"
    assert record.unknown_reason == "explicit_blocker_reason"
    assert record.proposed_correction is None
    assert record.correction_status == "unreviewed"
    # Empty reason is rejected.
    with pytest.raises(FailureDiagnosisError, match="non-empty"):
        unknown_failure_diagnosis_record(predicate, "   ")


def test_payload_is_versioned_and_non_claim() -> None:
    """The payload must be versioned with non-claim caveats and a coverage summary."""
    records = diagnose_from_trace_failure_predicates(
        [
            _predicate("collision", severity="critical"),
            _predicate("oscillatory_local_control", severity="medium"),
            _predicate("collision", validity_status=_NOT_AVAILABLE, severity="high"),
        ]
    )
    payload = build_failure_diagnosis_payload(records, generated_at_utc="2026-08-01T00:00:00+00:00")

    assert payload["schema_version"] == FAILURE_DIAGNOSIS_SCHEMA_VERSION
    assert payload["diagnosis_source"] == DIAGNOSIS_SOURCE
    assert payload["generated_at_utc"] == "2026-08-01T00:00:00+00:00"
    assert len(payload["records"]) == 3
    assert payload["failure_type_coverage"]["counts"] == {"collision": 1, "unknown": 2}
    joined_caveats = " ".join(payload["caveats"]).lower()
    assert "no benchmark-ranking" in joined_caveats
    assert "not causal inference" in joined_caveats
    assert "out of scope" in joined_caveats


def test_payload_validation_round_trips() -> None:
    """validate_failure_diagnosis_payload must accept the built payload."""
    records = diagnose_from_trace_failure_predicates(
        [_predicate("collision"), _predicate("low_progress", severity="medium")]
    )
    payload = build_failure_diagnosis_payload(records)
    validated = validate_failure_diagnosis_payload(payload)
    assert validated["schema_version"] == FAILURE_DIAGNOSIS_SCHEMA_VERSION
    assert len(validated["records"]) == 2

    bad_payload = dict(payload)
    bad_payload["schema_version"] = "something.else.v1"
    with pytest.raises(FailureDiagnosisError, match="schema_version must be"):
        validate_failure_diagnosis_payload(bad_payload)


@pytest.mark.parametrize("predicate_id", list(TRACE_FAILURE_PREDICATE_IDS))
def test_every_known_predicate_id_maps_deterministically(predicate_id: str) -> None:
    """Every shipped predicate id must adapt without error and stay in-vocabulary."""
    record = diagnose_from_trace_failure_predicate(_predicate(predicate_id))
    assert record.failure_type in ALLOWED_FAILURE_TYPES
    assert record.failure_level in FAILURE_LEVELS
    assert record.confidence in MECHANISM_CONFIDENCES
    assert record.evidence_mode in MECHANISM_EVIDENCE_MODES
    assert record.diagnosis_schema_version == FAILURE_DIAGNOSIS_SCHEMA_VERSION
    # The unknown_reason invariant must hold for every adapted record.
    validate_failure_diagnosis_record(record.to_dict())
    if record.failure_type == "unknown":
        assert record.unknown_reason is not None
    else:
        assert record.unknown_reason is None
