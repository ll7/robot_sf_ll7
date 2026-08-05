"""Tests for the optional failure-diagnosis record and deterministic adapter (#6583)."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
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
    DIAGNOSIS_QUALITY_SOURCE,
    DIAGNOSIS_SEVERITIES,
    DIAGNOSIS_SOURCE,
    FAILURE_DIAGNOSIS_QUALITY_SCHEMA_VERSION,
    FAILURE_DIAGNOSIS_REFERENCE_SCHEMA_VERSION,
    FAILURE_DIAGNOSIS_SCHEMA_VERSION,
    FAILURE_LEVELS,
    FailureDiagnosisError,
    build_failure_diagnosis_payload,
    build_failure_diagnosis_quality_report,
    compare_failure_diagnosis_to_reference,
    diagnose_from_trace_failure_predicate,
    diagnose_from_trace_failure_predicates,
    evaluate_failure_diagnosis_quality,
    unknown_failure_diagnosis_record,
    validate_failure_diagnosis_payload,
    validate_failure_diagnosis_record,
    validate_failure_diagnosis_reference_fixture,
)
from robot_sf.benchmark.failure_mechanism_classifier import FAILURE_MECHANISM_LABELS
from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_CONFIDENCES,
    MECHANISM_EVIDENCE_MODES,
)

_VALID = "valid"
_NOT_AVAILABLE = "not_available"
_QUALITY_METRICS = ("detection", "onset", "failure_type", "severity")


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


def _reference_fixture() -> dict[str, Any]:
    """Load the independently authored issue #6646 reference fixture."""
    path = Path("docs/context/evidence/issue_6646_failure_diagnosis_reference_fixture.v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_candidates() -> dict[str, Any]:
    """Build deterministic diagnosis rows paired with the reviewed fixture."""
    return {
        "collision_case": diagnose_from_trace_failure_predicate(
            _predicate("collision", time_interval_s=[1.0, 1.5], severity="critical")
        ).to_dict(),
        "near_miss_case": diagnose_from_trace_failure_predicate(
            _predicate(
                "clearance_critical_interaction",
                time_interval_s=[2.0, 3.0],
                severity="medium",
            )
        ).to_dict(),
        "low_progress_case": diagnose_from_trace_failure_predicate(
            _predicate("low_progress", time_interval_s=[4.0, 6.0], severity="medium")
        ).to_dict(),
        "oscillation_case": diagnose_from_trace_failure_predicate(
            _predicate("oscillatory_local_control", time_interval_s=[7.0, 8.0], severity="medium")
        ).to_dict(),
        "no_failure_case": {"detected": False, "status": "available"},
        "unavailable_case": {"detected": True, "status": "degraded"},
    }


def _assert_fixture_excluded_from_all_metrics(report: dict[str, Any], reason: str) -> None:
    """Require one fixture-level exclusion to remove every metric denominator."""
    for metric_name in _QUALITY_METRICS:
        metric = report[metric_name]
        assert metric["denominator"] == 0
        assert metric["excluded_count"] == report["case_count"]
        assert metric["excluded_reasons"][reason] == report["case_count"]


def _assert_invalid_reference_fixture_rejected(fixture: dict[str, Any], *, match: str) -> None:
    """Require malformed reference envelopes to fail at both public entry points."""
    with pytest.raises(FailureDiagnosisError, match=match):
        validate_failure_diagnosis_reference_fixture(fixture)
    with pytest.raises(FailureDiagnosisError, match=match):
        evaluate_failure_diagnosis_quality(_quality_candidates(), fixture)


def test_reviewed_reference_fixture_is_versioned_and_provenance_complete() -> None:
    """The committed fixture keeps review and independent trace pointers explicit."""
    fixture = validate_failure_diagnosis_reference_fixture(_reference_fixture())

    assert fixture["schema_version"] == FAILURE_DIAGNOSIS_REFERENCE_SCHEMA_VERSION
    assert fixture["review"] == {
        "status": "reviewed",
        "reviewer": "robot_sf_fixture_review",
        "adjudication_status": "adjudicated",
        "independent_of_automated_diagnosis": True,
    }
    assert len(fixture["records"]) >= 5
    assert all(record["source_trace"] for record in fixture["records"])
    assert all(record["provenance_status"] == "complete" for record in fixture["records"])


def test_quality_report_computes_detection_onset_type_and_severity_metrics() -> None:
    """The approved deterministic slice reports all four quality metric families."""
    report = evaluate_failure_diagnosis_quality(_quality_candidates(), _reference_fixture())

    assert report["schema_version"] == FAILURE_DIAGNOSIS_QUALITY_SCHEMA_VERSION
    assert report["evaluation_source"] == DIAGNOSIS_QUALITY_SOURCE
    assert report["detection"]["confusion_counts"] == {
        "true_positive": 4,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert report["detection"]["agreement"] == 1.0
    assert report["detection"]["denominator"] == 5
    assert report["detection"]["excluded_count"] == 1
    assert report["onset"]["denominator"] == 4
    assert report["onset"]["mean_interval_overlap"] == pytest.approx(5.0 / 6.0)
    assert report["onset"]["mean_absolute_midpoint_error_s"] == pytest.approx(0.125)
    assert report["failure_type"]["denominator"] == 3
    assert report["failure_type"]["exact_match"] == 1.0
    assert report["failure_type"]["macro_f1"] == 1.0
    assert report["severity"]["denominator"] == 4
    assert report["severity"]["exact_match"] == 1.0
    assert report["severity"]["macro_f1"] == 1.0
    for metric_name in _QUALITY_METRICS:
        assert report[metric_name]["excluded_reasons"]["reference_record_status:unavailable"] == 1
    assert report["case_comparisons"][-1]["metrics"]["detection"]["status"] == "excluded"


def test_quality_metrics_exclude_unreviewed_and_degraded_rows_without_forcing_labels() -> None:
    """Ineligible rows remain visible and cannot become accuracy evidence."""
    fixture = _reference_fixture()
    fixture["records"][0]["review"] = {"status": "unreviewed"}
    fixture["records"][1]["source_trace"] = {}
    candidates = _quality_candidates()
    candidates["near_miss_case"] = {"detected": True, "status": "degraded"}

    report = compare_failure_diagnosis_to_reference(candidates, fixture)

    assert report["detection"]["denominator"] == 3
    assert report["detection"]["excluded_count"] == 3
    assert "reference_unreviewed" in report["detection"]["excluded_reasons"]
    assert "reference_provenance_incomplete" in report["detection"]["excluded_reasons"]
    assert "diagnosis_status:degraded" in report["detection"]["excluded_reasons"]
    for metric_name in _QUALITY_METRICS:
        reasons = report[metric_name]["excluded_reasons"]
        assert reasons["reference_unreviewed"] == 1
        assert reasons["reference_provenance_incomplete"] == 1
        assert reasons["diagnosis_status:degraded"] == 2
    by_case = {row["case_id"]: row for row in report["case_comparisons"]}
    assert by_case["oscillation_case"]["reference"]["failure_type"] == "unknown"
    assert by_case["no_failure_case"]["reference"]["detected"] == "not_detected"


def test_quality_macro_f1_penalizes_known_type_confusion() -> None:
    """Macro-F1 uses only known, detected cases and penalizes a wrong class."""
    candidates = _quality_candidates()
    candidates["low_progress_case"] = diagnose_from_trace_failure_predicate(
        _predicate("collision", time_interval_s=[4.0, 6.0], severity="medium")
    ).to_dict()

    report = evaluate_failure_diagnosis_quality(candidates, _reference_fixture())

    assert report["failure_type"]["exact_match"] == pytest.approx(2.0 / 3.0)
    assert report["failure_type"]["macro_f1"] == pytest.approx(5.0 / 9.0)


def test_quality_fixture_level_unreviewed_status_excludes_every_case() -> None:
    """A fixture whose review is incomplete cannot produce accuracy evidence."""
    fixture = _reference_fixture()
    fixture["review"]["status"] = "unreviewed"

    report = evaluate_failure_diagnosis_quality(_quality_candidates(), fixture)

    assert report["case_count"] == 6
    assert len(report["case_comparisons"]) == 6
    _assert_fixture_excluded_from_all_metrics(report, "reference_unreviewed")
    assert report["detection"]["agreement"] is None
    assert report["detection"]["confusion_counts"] == {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert report["onset"]["mean_interval_overlap"] is None
    assert report["onset"]["mean_absolute_midpoint_error_s"] is None
    assert report["failure_type"]["exact_match"] is None
    assert report["failure_type"]["macro_f1"] is None
    assert report["severity"]["exact_match"] is None
    assert report["severity"]["macro_f1"] is None


def test_quality_fixture_not_independent_of_diagnosis_is_excluded() -> None:
    """Reference labels not created independently cannot enter a denominator."""
    fixture = _reference_fixture()
    fixture["review"]["independent_of_automated_diagnosis"] = False

    report = evaluate_failure_diagnosis_quality(_quality_candidates(), fixture)

    _assert_fixture_excluded_from_all_metrics(
        report, "reference_not_independent_of_automated_diagnosis"
    )


def test_quality_fixture_missing_adjudication_or_reviewer_is_excluded() -> None:
    """Reviewer and adjudication metadata are required before labels can be scored."""
    fixture = _reference_fixture()
    fixture["review"]["adjudication_status"] = "pending"
    fixture["review"]["reviewer"] = None

    report = evaluate_failure_diagnosis_quality(_quality_candidates(), fixture)

    _assert_fixture_excluded_from_all_metrics(report, "reference_unadjudicated")
    _assert_fixture_excluded_from_all_metrics(report, "reference_reviewer_missing")


def test_quality_fixture_provenance_complete_requires_source_trace() -> None:
    """A provenance block without a source trace is demoted and excluded."""
    fixture = _reference_fixture()
    fixture["provenance"] = {"status": "complete"}

    report = evaluate_failure_diagnosis_quality(_quality_candidates(), fixture)

    _assert_fixture_excluded_from_all_metrics(report, "reference_provenance_incomplete")


def test_quality_missing_diagnosis_is_excluded_and_remains_visible() -> None:
    """A reference case without any diagnosis stays visible but cannot be scored."""
    candidates = _quality_candidates()
    del candidates["collision_case"]

    report = evaluate_failure_diagnosis_quality(candidates, _reference_fixture())

    assert report["matched_case_count"] == 5
    assert report["detection"]["denominator"] == 4
    assert report["detection"]["confusion_counts"] == {
        "true_positive": 3,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert report["onset"]["denominator"] == 3
    assert report["failure_type"]["denominator"] == 2
    assert report["severity"]["denominator"] == 3
    for metric_name in _QUALITY_METRICS:
        metric = report[metric_name]
        assert metric["excluded_reasons"]["diagnosis_missing"] == 1
    by_case = {row["case_id"]: row for row in report["case_comparisons"]}
    comparison = by_case["collision_case"]
    assert comparison["diagnosis"]["available"] is False
    assert comparison["metrics"]["detection"]["status"] == "excluded"
    assert "diagnosis_missing" in comparison["metrics"]["detection"]["excluded_reasons"]


def test_quality_unmatched_diagnosis_cannot_enter_any_denominator() -> None:
    """Diagnoses without a reviewed reference case are counted but never scored."""
    candidates = _quality_candidates()
    candidates["extra_unreviewed_case"] = diagnose_from_trace_failure_predicate(
        _predicate("collision", severity="critical")
    ).to_dict()

    baseline = evaluate_failure_diagnosis_quality(_quality_candidates(), _reference_fixture())
    report = evaluate_failure_diagnosis_quality(candidates, _reference_fixture())

    assert report["unmatched_diagnosis_count"] == 1
    assert report["matched_case_count"] == baseline["matched_case_count"]
    assert report["metrics"] == baseline["metrics"]


def test_reference_fixture_envelope_is_versioned_and_case_ids_are_unique() -> None:
    """Malformed reference envelopes must fail closed before any metric exists."""
    fixture = _reference_fixture()

    wrong_schema = dict(fixture)
    wrong_schema["schema_version"] = "failure_diagnosis_reference.v2"
    _assert_invalid_reference_fixture_rejected(wrong_schema, match="schema_version must be")

    bad_version = dict(fixture)
    bad_version["fixture_version"] = True
    _assert_invalid_reference_fixture_rejected(
        bad_version, match="fixture_version must be an integer"
    )

    bad_id = dict(fixture)
    bad_id["fixture_id"] = "   "
    _assert_invalid_reference_fixture_rejected(bad_id, match="fixture_id")

    duplicated = _reference_fixture()
    duplicated["records"][1] = dict(duplicated["records"][0])
    _assert_invalid_reference_fixture_rejected(duplicated, match="duplicated")


def test_quality_report_pairs_payload_form_candidates_positionally() -> None:
    """Payload-form diagnoses without case ids pair by reference record order."""
    ordered = _quality_candidates()
    payload_form = {
        "records": [
            ordered["collision_case"],
            ordered["near_miss_case"],
            ordered["low_progress_case"],
            ordered["oscillation_case"],
            ordered["no_failure_case"],
            ordered["unavailable_case"],
        ]
    }

    baseline = evaluate_failure_diagnosis_quality(_quality_candidates(), _reference_fixture())
    report = evaluate_failure_diagnosis_quality(payload_form, _reference_fixture())

    assert report == baseline


def test_quality_report_is_deterministic_and_aliases_agree() -> None:
    """The fixture-level report must be reproducible across all public entry points."""
    candidates = _quality_candidates()
    fixture = _reference_fixture()

    first = evaluate_failure_diagnosis_quality(candidates, fixture)
    second = evaluate_failure_diagnosis_quality(candidates, fixture)
    assert first == second

    assert compare_failure_diagnosis_to_reference(candidates, fixture) == first
    assert build_failure_diagnosis_quality_report(candidates, fixture) == first


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


def test_noncanonical_validity_status_fails_closed_and_is_preserved() -> None:
    """Padded validity text must not be accepted as valid evidence or normalized away."""
    predicate = _predicate("collision").to_dict()
    predicate["validity_status"] = " valid "

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "predicate_validity_not_valid:valid"
    assert record.validity_status == " valid "


def test_noncanonical_predicate_id_fails_closed() -> None:
    """Predicate IDs with surrounding whitespace must remain unsupported mappings."""
    predicate = _predicate("collision").to_dict()
    predicate["predicate_id"] = " collision "

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "unsupported_predicate_id:collision"


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


def test_validate_record_rejects_non_schema_vocab_and_correction_values() -> None:
    """External records must fail closed rather than leak type errors or bad corrections."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))

    unhashable_type = record.to_dict()
    unhashable_type["failure_type"] = []
    with pytest.raises(FailureDiagnosisError, match="unsupported failure_type"):
        validate_failure_diagnosis_record(unhashable_type)

    non_string_correction = record.to_dict()
    non_string_correction["proposed_correction"] = 7
    with pytest.raises(FailureDiagnosisError, match="proposed_correction"):
        validate_failure_diagnosis_record(non_string_correction)


def test_validate_record_rejects_non_pointer_causal_evidence() -> None:
    """External records cannot substitute a causal assertion for a source pointer."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["causal_evidence"] = [{"assertion": "pedestrian caused the collision"}]

    with pytest.raises(FailureDiagnosisError, match="exactly the trace-predicate pointer fields"):
        validate_failure_diagnosis_record(payload)


def test_validate_record_rejects_tampered_adapter_provenance() -> None:
    """External edits cannot detach a deterministic record from its source predicate."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))

    mismatched_status = record.to_dict()
    mismatched_status["validity_status"] = _NOT_AVAILABLE
    with pytest.raises(FailureDiagnosisError, match="must match source_predicate"):
        validate_failure_diagnosis_record(mismatched_status)

    unavailable_known = record.to_dict()
    unavailable_known["validity_status"] = _NOT_AVAILABLE
    unavailable_known["source_predicate"]["validity_status"] = _NOT_AVAILABLE
    with pytest.raises(
        FailureDiagnosisError, match="non-valid predicate evidence requires unknown"
    ):
        validate_failure_diagnosis_record(unavailable_known)

    unrelated_pointer = record.to_dict()
    unrelated_pointer["causal_evidence"] = [
        {**unrelated_pointer["causal_evidence"][0], "predicate_id": "unrelated"}
    ]
    with pytest.raises(FailureDiagnosisError, match="exact source_predicate pointer"):
        validate_failure_diagnosis_record(unrelated_pointer)

    remapped_source = record.to_dict()
    remapped_source["source_predicate"]["predicate_id"] = "future_unmodelled_predicate"
    remapped_source["causal_evidence"][0]["predicate_id"] = "future_unmodelled_predicate"
    with pytest.raises(FailureDiagnosisError, match="deterministic adapter result"):
        validate_failure_diagnosis_record(remapped_source)


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


def test_validate_record_rejects_string_onset_numbers() -> None:
    """The record schema requires JSON numbers rather than numeric-looking strings."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = record.to_dict()
    payload["onset_time_s"] = "1.0"
    payload["onset_interval"] = ["1.0", 1.5]

    with pytest.raises(FailureDiagnosisError, match="finite numbers or None"):
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


def test_malformed_predicate_evidence_fails_closed_to_unknown_record() -> None:
    """Malformed mapping evidence must not crash or receive a confident diagnosis label."""
    predicate = _predicate("collision").to_dict()
    predicate["evidence_fields"] = None

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "invalid_predicate_evidence:evidence_fields_not_mapping"
    assert record.source_predicate["evidence_fields"] is None
    assert record.causal_evidence[0]["evidence_fields"] == {}
    validate_failure_diagnosis_record(record.to_dict())


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        (
            "steps",
            ["not-a-step", 15],
            "invalid_predicate_evidence:steps_not_two_element_integer_or_none_sequence",
        ),
        (
            "involved_actors",
            ["robot", 7],
            "invalid_predicate_evidence:involved_actors_not_string_sequence",
        ),
    ],
)
def test_malformed_predicate_pointer_fields_fail_closed_to_unknown_record(
    field: str, value: list[Any], expected_reason: str
) -> None:
    """Malformed mapping pointer fields must not receive a known diagnosis label."""
    predicate = _predicate("collision").to_dict()
    predicate[field] = value

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == expected_reason
    validate_failure_diagnosis_record(record.to_dict())


def test_non_numeric_predicate_onset_fails_closed_to_unknown_record() -> None:
    """A valid-status predicate still needs finite onset evidence for a known label."""
    predicate = _predicate("collision").to_dict()
    predicate["time_interval_s"] = ["not-a-time", 1.5]

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert (
        record.unknown_reason
        == "invalid_predicate_evidence:time_interval_s_non_finite_or_non_numeric"
    )
    assert record.onset_time_s is None
    assert record.onset_interval == [None, 1.5]
    validate_failure_diagnosis_record(record.to_dict())


def test_numeric_string_predicate_onset_fails_closed_to_unknown_record() -> None:
    """Numeric-looking strings are not valid JSON-number onset evidence."""
    predicate = _predicate("collision").to_dict()
    predicate["time_interval_s"] = ["1.0", 1.5]

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert (
        record.unknown_reason
        == "invalid_predicate_evidence:time_interval_s_non_finite_or_non_numeric"
    )
    assert record.onset_time_s is None
    assert record.onset_interval == [None, 1.5]
    validate_failure_diagnosis_record(record.to_dict())


def test_nonfinite_predicate_evidence_fails_closed_and_stays_strict_json() -> None:
    """Non-finite source values become explicit unknown evidence without invalid JSON."""
    predicate = _predicate("collision").to_dict()
    predicate["time_interval_s"] = [math.nan, 1.5]
    predicate["evidence_fields"] = {"distance_m": math.inf}

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "invalid_predicate_evidence:non_json_safe_value"
    json.dumps(record.to_dict(), allow_nan=False)
    validate_failure_diagnosis_record(record.to_dict())


def test_unrepresentable_numeric_predicate_evidence_fails_closed() -> None:
    """Numeric values that overflow float conversion must not escape the adapter."""
    predicate = _predicate("collision").to_dict()
    predicate["evidence_fields"] = {"unrepresentable": Fraction(10**10000, 1)}

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "invalid_predicate_evidence:non_json_safe_value"
    json.dumps(record.to_dict(), allow_nan=False)
    validate_failure_diagnosis_record(record.to_dict())


def test_cyclic_predicate_evidence_fails_closed() -> None:
    """Cyclic mapping evidence must become a strict-JSON-safe unknown record."""
    predicate = _predicate("collision").to_dict()
    evidence_fields: dict[str, Any] = {}
    evidence_fields["self"] = evidence_fields
    predicate["evidence_fields"] = evidence_fields

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "invalid_predicate_evidence:non_json_safe_value"
    json.dumps(record.to_dict(), allow_nan=False)
    validate_failure_diagnosis_record(record.to_dict())


def test_non_string_predicate_mapping_keys_fail_closed() -> None:
    """Non-string evidence keys must not be silently rewritten into valid evidence."""
    predicate = _predicate("collision").to_dict()
    predicate["evidence_fields"] = {1: "numeric-key"}

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "invalid_predicate_evidence:non_json_safe_value"
    assert any(
        str(key).startswith("__failure_diagnosis_invalid_json_value__:")
        for key in record.source_predicate["evidence_fields"]
    )
    json.dumps(record.to_dict(), allow_nan=False)
    validate_failure_diagnosis_record(record.to_dict())


def test_predicate_to_dict_must_return_a_mapping() -> None:
    """Malformed custom predicate adapters must raise the domain error, not TypeError."""

    class InvalidPredicate:
        """Expose a malformed non-mapping ``to_dict`` result."""

        def to_dict(self) -> None:
            """Return an invalid predicate representation."""
            return None

    invalid_predicate: Any = InvalidPredicate()
    with pytest.raises(FailureDiagnosisError, match=r"to_dict\(\).*mapping"):
        diagnose_from_trace_failure_predicate(invalid_predicate)


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
    # Non-string values must not be silently converted into a schema reason.
    invalid_reason: Any = None
    with pytest.raises(FailureDiagnosisError, match="non-empty string"):
        unknown_failure_diagnosis_record(predicate, invalid_reason)


def test_explicit_unknown_helper_record_validates_and_wraps() -> None:
    """Explicit unknown metadata must remain valid for payload construction."""
    record = unknown_failure_diagnosis_record(
        _predicate("collision", validity_status=_NOT_AVAILABLE),
        "explicit_blocker_reason",
    )

    validated = validate_failure_diagnosis_record(record.to_dict())
    assert validated["failure_type"] == "unknown"
    assert validated["unknown_reason"] == "explicit_blocker_reason"

    payload = build_failure_diagnosis_payload([record])
    assert validate_failure_diagnosis_payload(payload)["records"][0]["unknown_reason"] == (
        "explicit_blocker_reason"
    )


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


def test_payload_validation_rejects_missing_or_forged_metadata() -> None:
    """Payload metadata and coverage must agree with the validated record list."""
    record = diagnose_from_trace_failure_predicate(_predicate("collision"))
    payload = build_failure_diagnosis_payload(
        [record], generated_at_utc="2026-08-02T00:00:00+00:00"
    )

    missing_timestamp = dict(payload)
    del missing_timestamp["generated_at_utc"]
    with pytest.raises(FailureDiagnosisError, match="missing required field"):
        validate_failure_diagnosis_payload(missing_timestamp)

    forged_coverage = dict(payload)
    forged_coverage["failure_type_coverage"] = {
        "counts": {"collision": 99},
        "classification_source": DIAGNOSIS_SOURCE,
    }
    with pytest.raises(FailureDiagnosisError, match="failure_type_coverage counts"):
        validate_failure_diagnosis_payload(forged_coverage)


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


def test_cyclic_trace_predicate_object_evidence_fails_closed() -> None:
    """Direct TraceFailurePredicate objects must use the cycle-safe adapter boundary."""
    predicate = _predicate("collision")
    predicate.evidence_fields["self"] = predicate.evidence_fields

    record = diagnose_from_trace_failure_predicate(predicate)

    assert record.failure_type == "unknown"
    assert record.unknown_reason == "invalid_predicate_evidence:non_json_safe_value"
    json.dumps(record.to_dict(), allow_nan=False)
    validate_failure_diagnosis_record(record.to_dict())


def test_predicate_to_dict_failures_raise_domain_error() -> None:
    """A failing custom serializer must not leak its implementation exception."""

    class RaisingPredicate:
        """Expose a serializer failure at the adapter boundary."""

        def to_dict(self) -> dict[str, Any]:
            """Raise a representative serialization failure."""
            raise RecursionError("cyclic predicate")

    raising_predicate: Any = RaisingPredicate()
    with pytest.raises(FailureDiagnosisError, match=r"to_dict\(\) failed"):
        diagnose_from_trace_failure_predicate(raising_predicate)


def test_predicate_to_dict_runtime_failures_raise_domain_error() -> None:
    """Runtime failures from custom serializers stay inside the domain boundary."""

    class RuntimeFailingPredicate:
        """Expose a representative runtime serializer failure."""

        def to_dict(self) -> dict[str, Any]:
            """Raise a runtime failure that callers cannot repair from the record."""
            raise RuntimeError("serializer state is invalid")

    with pytest.raises(FailureDiagnosisError, match=r"to_dict\(\) failed"):
        diagnose_from_trace_failure_predicate(RuntimeFailingPredicate())


def test_validate_unknown_record_preserves_claim_boundary() -> None:
    """Unknown records must retain non-causal and explicit-reason caveats."""
    record = unknown_failure_diagnosis_record(
        _predicate("collision", validity_status=_NOT_AVAILABLE),
        "explicit_blocker_reason",
    )
    payload = record.to_dict()
    payload["caveats"] = []

    with pytest.raises(FailureDiagnosisError, match="claim boundary"):
        validate_failure_diagnosis_record(payload)
