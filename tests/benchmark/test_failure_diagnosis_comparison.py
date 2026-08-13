"""Tests for the held-out diagnosis comparison harness (#6646)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.trace_failure_predicates import TraceFailurePredicate
from robot_sf.benchmark.failure_diagnosis import (
    DIAGNOSIS_SOURCE,
    build_failure_diagnosis_payload,
    diagnose_from_trace_failure_predicate,
)
from robot_sf.benchmark.failure_diagnosis_comparison import (
    COMPARISON_SCHEMA_VERSION,
    REVIEW_PENDING_MARKERS,
    CaseAlignmentError,
    FixtureReviewPendingError,
    LearnedSourceError,
    MethodManifestError,
    align_held_out_cases,
    build_unavailable_comparison_report,
    compare_held_out_diagnoses,
    validate_fixture_review_admission,
    validate_method_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_FIXTURE_PATH = _REPO_ROOT / (
    "docs/context/evidence/issue_6646_failure_diagnosis_reference_fixture.v1.json"
)


def _reference_fixture() -> dict[str, Any]:
    """Load the independently authored issue #6646 reference fixture."""
    return json.loads(_REFERENCE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _reviewed_fixture() -> dict[str, Any]:
    """Build a reviewed fixture for happy-path tests.

    The production fixture carries a pending review marker and is not admissible
    evidence for an independently reviewed comparison.  This helper produces a
    fixture with the pending marker removed so the happy path can exercise the
    full comparison flow without pretending production evidence was reviewed.
    """
    fixture = copy.deepcopy(_reference_fixture())
    fixture["review_marker"] = "reviewed"
    return fixture


def _valid_manifest() -> dict[str, str]:
    """Build a valid method manifest for tests."""
    return {
        "method_id": "test-learned-method-v1",
        "model_identifier": "gpt-4o-2024-08-06",
        "model_revision": "abc123def456",
        "prompt_digest": "sha256:fedcba9876543210",
        "decoding_settings": '{"temperature": 0.0, "max_tokens": 2048}',
        "input_schema": "failure_diagnosis.v1",
        "output_artifact_digest": "sha256:0123456789abcdef",
        "held_out_exclusion_declaration": (
            "Training data excludes the held-out case ids used in this comparison."
        ),
        "non_deterministic_source": (
            "This method uses stochastic decoding; outputs are frozen for reproducibility."
        ),
    }


def _synthetic_deterministic_records() -> dict[str, Any]:
    """Build deterministic records paired with the reviewed fixture."""
    records = {
        "collision_case": {
            "diagnosis_schema_version": "failure_diagnosis.v1",
            "diagnosis_source": "failure_diagnosis.deterministic.v1",
            "failure_level": "interaction",
            "failure_type": "collision",
            "onset_time_s": 1.0,
            "onset_interval": [1.0, 1.5],
            "severity": "critical",
            "detection_method": "predicate",
            "causal_evidence": [
                {
                    "evidence_kind": "trace_failure_predicate",
                    "predicate_id": "collision",
                    "time_interval_s": [1.0, 1.5],
                    "steps": [10, 15],
                    "involved_actors": ["robot", "ped_0"],
                    "evidence_fields": {"event": "contact"},
                    "non_causal_note": (
                        "causal_evidence cites trace/predicate evidence pointers "
                        "only; it is not causal inference."
                    ),
                }
            ],
            "contributing_factors": [],
            "confidence": "observed_mechanism",
            "evidence_mode": "paired_trace",
            "validity_status": "valid",
            "proposed_correction": None,
            "correction_status": "unreviewed",
            "unknown_reason": None,
            "caveats": [
                "causal_evidence cites trace/predicate evidence pointers only; "
                "it is not causal inference."
            ],
            "source_predicate": {
                "predicate_id": "collision",
                "time_interval_s": [1.0, 1.5],
                "severity": "high",
                "validity_status": "valid",
            },
        },
        "near_miss_case": {
            "diagnosis_schema_version": "failure_diagnosis.v1",
            "diagnosis_source": "failure_diagnosis.deterministic.v1",
            "failure_level": "interaction",
            "failure_type": "near_miss",
            "onset_time_s": 2.5,
            "onset_interval": [2.5, 3.5],
            "severity": "major",
            "detection_method": "predicate",
            "causal_evidence": [
                {
                    "evidence_kind": "trace_failure_predicate",
                    "predicate_id": "clearance_critical_interaction",
                    "time_interval_s": [2.5, 3.5],
                    "steps": [25, 35],
                    "involved_actors": ["robot", "ped_1"],
                    "evidence_fields": {"distance_m": 0.35},
                    "non_causal_note": (
                        "causal_evidence cites trace/predicate evidence pointers "
                        "only; it is not causal inference."
                    ),
                }
            ],
            "contributing_factors": [],
            "confidence": "observed_mechanism",
            "evidence_mode": "paired_trace",
            "validity_status": "valid",
            "proposed_correction": None,
            "correction_status": "unreviewed",
            "unknown_reason": None,
            "caveats": [
                "causal_evidence cites trace/predicate evidence pointers only; "
                "it is not causal inference."
            ],
            "source_predicate": {
                "predicate_id": "clearance_critical_interaction",
                "time_interval_s": [2.5, 3.5],
                "severity": "medium",
                "validity_status": "valid",
            },
        },
        "low_progress_case": {
            "diagnosis_schema_version": "failure_diagnosis.v1",
            "diagnosis_source": "failure_diagnosis.deterministic.v1",
            "failure_level": "control",
            "failure_type": "timeout_without_progress",
            "onset_time_s": 4.0,
            "onset_interval": [4.0, 6.0],
            "severity": "major",
            "detection_method": "predicate",
            "causal_evidence": [
                {
                    "evidence_kind": "trace_failure_predicate",
                    "predicate_id": "low_progress",
                    "time_interval_s": [4.0, 6.0],
                    "steps": [40, 60],
                    "involved_actors": ["robot"],
                    "evidence_fields": {"progress_m": 0.02},
                    "non_causal_note": (
                        "causal_evidence cites trace/predicate evidence pointers "
                        "only; it is not causal inference."
                    ),
                }
            ],
            "contributing_factors": [],
            "confidence": "observed_mechanism",
            "evidence_mode": "paired_trace",
            "validity_status": "valid",
            "proposed_correction": None,
            "correction_status": "unreviewed",
            "unknown_reason": None,
            "caveats": [
                "causal_evidence cites trace/predicate evidence pointers only; "
                "it is not causal inference."
            ],
            "source_predicate": {
                "predicate_id": "low_progress",
                "time_interval_s": [4.0, 6.0],
                "severity": "medium",
                "validity_status": "valid",
            },
        },
        "oscillation_case": {
            "diagnosis_schema_version": "failure_diagnosis.v1",
            "diagnosis_source": "failure_diagnosis.deterministic.v1",
            "failure_level": "control",
            "failure_type": "unknown",
            "onset_time_s": 7.0,
            "onset_interval": [7.0, 8.0],
            "severity": "major",
            "detection_method": "predicate",
            "causal_evidence": [
                {
                    "evidence_kind": "trace_failure_predicate",
                    "predicate_id": "oscillatory_local_control",
                    "time_interval_s": [7.0, 8.0],
                    "steps": [70, 80],
                    "involved_actors": ["robot"],
                    "evidence_fields": {"direction_reversals": 4},
                    "non_causal_note": (
                        "causal_evidence cites trace/predicate evidence pointers "
                        "only; it is not causal inference."
                    ),
                }
            ],
            "contributing_factors": [],
            "confidence": "observed_mechanism",
            "evidence_mode": "paired_trace",
            "validity_status": "valid",
            "proposed_correction": None,
            "correction_status": "unreviewed",
            "unknown_reason": "oscillation_not_represented_in_classifier_labels",
            "caveats": [
                "causal_evidence cites trace/predicate evidence pointers only; "
                "it is not causal inference.",
                "Unsupported, invalid, or unavailable mappings resolve to unknown; "
                "causal_evidence still cites the source predicate's evidence pointers.",
            ],
            "source_predicate": {
                "predicate_id": "oscillatory_local_control",
                "time_interval_s": [7.0, 8.0],
                "severity": "medium",
                "validity_status": "valid",
            },
        },
        "no_failure_case": {
            "detected": False,
            "status": "available",
        },
        "unavailable_case": {
            "detected": True,
            "status": "degraded",
        },
    }
    return records


def _synthetic_learned_records() -> dict[str, Any]:
    """Build synthetic frozen learned records (matches deterministic set).

    The learned records carry a truthful non-deterministic ``diagnosis_source``
    so the comparison harness exercises the learned projection path.
    """
    records = dict(_synthetic_deterministic_records())
    for record in records.values():
        if isinstance(record, dict):
            record["diagnosis_source"] = "failure_diagnosis.learned.v1"
    # The learned method produces slightly different onset for near_miss
    records["near_miss_case"]["onset_interval"] = [2.0, 3.0]
    records["near_miss_case"]["onset_time_s"] = 2.0
    # The learned method gets a different severity for low_progress
    records["low_progress_case"]["severity"] = "critical"
    return records


def _build_deterministic_payload(records: dict[str, Any]) -> dict[str, Any]:
    """Wrap deterministic records in a ``failure_diagnosis.v1`` payload."""
    return build_failure_diagnosis_payload(records.values())


def _build_valid_payload_from_predicates() -> dict[str, Any]:
    """Build a valid ``failure_diagnosis.v1`` payload from synthetic predicates.

    The records carry a non-deterministic ``diagnosis_source`` so the payload
    can be used as learned input in comparison tests.
    """

    def _pred(
        predicate_id: str,
        time_interval_s: list[float | None],
        severity: str = "high",
    ) -> TraceFailurePredicate:
        return TraceFailurePredicate(
            predicate_id=predicate_id,
            time_interval_s=time_interval_s,
            steps=[int(time_interval_s[0] * 10), int(time_interval_s[1] * 10)],
            involved_actors=["robot", "ped_0"],
            scenario_family="crosswalk",
            planner_id="orca",
            evidence_fields={},
            severity=severity,
            validity_status="valid",
        )

    records = [
        diagnose_from_trace_failure_predicate(_pred("collision", [1.0, 1.5], "critical")),
        diagnose_from_trace_failure_predicate(
            _pred("clearance_critical_interaction", [2.5, 3.5], "medium")
        ),
        diagnose_from_trace_failure_predicate(_pred("low_progress", [4.0, 6.0], "medium")),
        diagnose_from_trace_failure_predicate(
            _pred("oscillatory_local_control", [7.0, 8.0], "medium")
        ),
    ]
    payload = build_failure_diagnosis_payload(records)
    # Inject case_ids to match the reference fixture for case-alignment.
    case_ids = [
        "collision_case",
        "near_miss_case",
        "low_progress_case",
        "oscillation_case",
    ]
    for record, case_id in zip(payload["records"], case_ids, strict=True):
        record["case_id"] = case_id
    # Set non-deterministic source so the payload can be used as learned input.
    for record in payload["records"]:
        record["diagnosis_source"] = "failure_diagnosis.learned.v1"
    payload["diagnosis_source"] = "failure_diagnosis.learned.v1"
    return payload


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Successful comparison with synthetic frozen outputs."""

    def test_comparison_report_has_correct_schema_version(self) -> None:
        """The report uses the versioned comparison schema for stable consumers."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        assert report["schema_version"] == COMPARISON_SCHEMA_VERSION

    def test_comparison_report_has_available_status(self) -> None:
        """A fully admitted synthetic comparison reports an available status."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        assert report["output_status"] == "available"
        assert report["output_reason"] is None

    def test_comparison_report_carries_both_summaries(self) -> None:
        """Available reports retain deterministic and learned metric summaries."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        assert report["deterministic_summary"] is not None
        assert report["learned_summary"] is not None
        assert "metrics" in report["deterministic_summary"]
        assert "metrics" in report["learned_summary"]

    def test_comparison_report_includes_method_manifest(self) -> None:
        """Reports retain the pinned method manifest for provenance review."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        manifest = report["method_manifest"]
        assert manifest["method_id"] == "test-learned-method-v1"
        assert manifest["model_identifier"] == "gpt-4o-2024-08-06"
        assert manifest["non_deterministic_source"]

    def test_comparison_report_preserves_claim_boundary(self) -> None:
        """Reports state the diagnostic-only boundary that prevents overclaiming."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        boundary = report["claim_boundary"]
        assert boundary["fixture_level_metrics_only"] is True
        assert boundary["no_campaign_ranking"] is True
        assert boundary["no_scientific_result_claim"] is True
        assert "next_gate" in boundary

    def test_comparison_report_has_case_comparisons(self) -> None:
        """Reports retain per-case views for denominator and exclusion auditing."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        assert len(report["case_comparisons"]) == 6
        assert len(report["deterministic_case_comparisons"]) == 6

    def test_comparison_report_alignment_is_correct(self) -> None:
        """Reports expose the exact aligned case count used by both methods."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        alignment = report["alignment"]
        assert alignment["deterministic_count"] == 6
        assert alignment["learned_count"] == 6
        assert len(alignment["aligned_case_ids"]) == 6

    def test_comparison_report_has_caveats(self) -> None:
        """Reports retain caveats explaining unavailable execution and claim limits."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        assert len(report["caveats"]) >= 4
        assert any("does not execute" in c for c in report["caveats"])

    def test_comparison_report_has_learned_source_projection(self) -> None:
        """Reports document how learned provenance survives evaluator projection."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        projection = report["learned_source_projection"]
        assert "description" in projection
        assert "preserved_source_fields" in projection
        assert "diagnosis_source" in projection["preserved_source_fields"]

    def test_deterministic_report_stability(self) -> None:
        """Running the comparison twice produces identical deterministic summaries."""
        args = (
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        report_a = compare_held_out_diagnoses(*args)
        report_b = compare_held_out_diagnoses(*args)
        assert report_a["deterministic_summary"] == report_b["deterministic_summary"]

    def test_deterministic_comparator_not_overwritten(self) -> None:
        """The deterministic records dict is not mutated during comparison."""
        det = _synthetic_deterministic_records()
        original = copy.deepcopy(det)
        compare_held_out_diagnoses(
            _reviewed_fixture(), det, _synthetic_learned_records(), _valid_manifest()
        )
        assert det == original

    def test_no_model_execution(self) -> None:
        """The comparison harness does not import or call any model/API code."""
        import inspect

        import robot_sf.benchmark.failure_diagnosis_comparison as mod

        source = inspect.getsource(mod)
        forbidden = ["import openai", "import anthropic", "import torch", "requests.get"]
        for token in forbidden:
            assert token not in source, f"forbidden token found: {token}"


# ---------------------------------------------------------------------------
# Method manifest validation
# ---------------------------------------------------------------------------


class TestManifestValidation:
    """Method manifest must carry all required provenance fields."""

    def test_valid_manifest_passes(self) -> None:
        """Complete manifest provenance is accepted for a learned method."""
        manifest = validate_method_manifest(_valid_manifest())
        assert manifest.method_id == "test-learned-method-v1"

    def test_missing_field_rejects(self) -> None:
        """Missing model revision rejects the manifest instead of synthesizing it."""
        incomplete = _valid_manifest()
        del incomplete["model_revision"]
        with pytest.raises(MethodManifestError, match="model_revision"):
            validate_method_manifest(incomplete)

    def test_empty_string_rejects(self) -> None:
        """Empty provenance values fail closed before comparison."""
        manifest = _valid_manifest()
        manifest["prompt_digest"] = ""
        with pytest.raises(MethodManifestError, match="prompt_digest"):
            validate_method_manifest(manifest)

    def test_whitespace_only_rejects(self) -> None:
        """Whitespace-only provenance values are treated as missing."""
        manifest = _valid_manifest()
        manifest["output_artifact_digest"] = "   "
        with pytest.raises(MethodManifestError, match="output_artifact_digest"):
            validate_method_manifest(manifest)

    def test_non_mapping_rejects(self) -> None:
        """Non-mapping manifests are rejected as structurally invalid."""
        with pytest.raises(MethodManifestError, match="must be a mapping"):
            validate_method_manifest("not a mapping")

    def test_multiple_missing_fields_reported(self) -> None:
        """The manifest error reports multiple missing provenance fields together."""
        manifest: dict[str, str] = {"method_id": "test"}
        with pytest.raises(MethodManifestError, match="model_identifier"):
            validate_method_manifest(manifest)

    def test_manifest_strips_whitespace(self) -> None:
        """Valid manifest strings are normalized without changing their meaning."""
        manifest = _valid_manifest()
        manifest["method_id"] = "  my-method  "
        result = validate_method_manifest(manifest)
        assert result.method_id == "my-method"


# ---------------------------------------------------------------------------
# Case alignment failures
# ---------------------------------------------------------------------------


class TestCaseAlignment:
    """Case ids must match exactly between deterministic and learned sets."""

    def test_perfect_alignment_succeeds(self) -> None:
        """Identical case sets pass the structural alignment gate."""
        det = _synthetic_deterministic_records()
        learn = _synthetic_learned_records()
        result = align_held_out_cases(det, learn)
        assert result["deterministic_count"] == result["learned_count"]

    def test_missing_case_in_learned_rejects(self) -> None:
        """A missing learned case is rejected to prevent denominator drift."""
        det = _synthetic_deterministic_records()
        learn = dict(_synthetic_learned_records())
        del learn["collision_case"]
        with pytest.raises(CaseAlignmentError, match="missing_from_learned"):
            align_held_out_cases(det, learn)

    def test_missing_case_in_deterministic_rejects(self) -> None:
        """A learned-only case is rejected to preserve paired comparison identity."""
        det = _synthetic_deterministic_records()
        learn = _synthetic_learned_records()
        learn["extra_case"] = {"detected": False}
        with pytest.raises(CaseAlignmentError, match="missing_from_deterministic"):
            align_held_out_cases(det, learn)

    def test_duplicate_case_in_deterministic_rejects(self) -> None:
        """Duplicate deterministic ids fail before metrics can be computed."""
        learn = {"a": {"case_id": "a", "detected": True}, "b": {"case_id": "b", "detected": False}}
        det_with_dupe = [
            {"case_id": "a", "detected": True},
            {"case_id": "b", "detected": False},
            {"case_id": "a", "detected": True},
        ]
        with pytest.raises(CaseAlignmentError, match="duplicate"):
            align_held_out_cases(det_with_dupe, learn)

    def test_duplicate_case_in_learned_rejects(self) -> None:
        """Duplicate learned ids fail before metrics can be computed."""
        det = {"a": {"case_id": "a", "detected": True}, "b": {"case_id": "b", "detected": False}}
        learn_with_dupe = [
            {"case_id": "a", "detected": True},
            {"case_id": "b", "detected": False},
            {"case_id": "a", "detected": True},
        ]
        with pytest.raises(CaseAlignmentError, match="duplicate"):
            align_held_out_cases(det, learn_with_dupe)

    def test_case_alignment_error_from_compare(self) -> None:
        """Misaligned cases cause compare_held_out_diagnoses to fail."""
        det = _synthetic_deterministic_records()
        learn = dict(_synthetic_learned_records())
        del learn["no_failure_case"]
        with pytest.raises(CaseAlignmentError):
            compare_held_out_diagnoses(_reviewed_fixture(), det, learn, _valid_manifest())


# ---------------------------------------------------------------------------
# Missing / invalid provenance fails closed
# ---------------------------------------------------------------------------


class TestProvenanceFailClosed:
    """Missing or invalid manifest fields produce unavailable reports or reject."""

    def test_missing_manifest_field_raises(self) -> None:
        """Comparison rejects a missing held-out exclusion declaration."""
        incomplete = _valid_manifest()
        del incomplete["held_out_exclusion_declaration"]
        with pytest.raises(MethodManifestError):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _synthetic_deterministic_records(),
                _synthetic_learned_records(),
                incomplete,
            )

    def test_empty_manifest_field_raises(self) -> None:
        """Comparison rejects empty decoding provenance instead of guessing defaults."""
        manifest = _valid_manifest()
        manifest["decoding_settings"] = ""
        with pytest.raises(MethodManifestError):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _synthetic_deterministic_records(),
                _synthetic_learned_records(),
                manifest,
            )


# ---------------------------------------------------------------------------
# Held-out exclusion declaration
# ---------------------------------------------------------------------------


class TestHeldOutExclusion:
    """The manifest must declare held-out exclusion."""

    def test_missing_held_out_declaration_rejects(self) -> None:
        """A manifest without held-out exclusion provenance is rejected."""
        manifest = _valid_manifest()
        del manifest["held_out_exclusion_declaration"]
        with pytest.raises(MethodManifestError, match="held_out_exclusion_declaration"):
            validate_method_manifest(manifest)

    def test_present_held_out_declaration_accepted(self) -> None:
        """A non-empty held-out exclusion declaration is retained and accepted."""
        manifest = validate_method_manifest(_valid_manifest())
        assert "held-out" in manifest.held_out_exclusion_declaration.lower()


# ---------------------------------------------------------------------------
# Unknown / unavailable exclusions
# ---------------------------------------------------------------------------


class TestUnknownUnavailableExclusions:
    """Unknown/unavailable cases are excluded from metrics but retained in report."""

    def test_unavailable_case_excluded_from_detection(self) -> None:
        """Unavailable rows stay visible but cannot enter detection denominators."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        det_metrics = report["deterministic_summary"]["metrics"]["detection"]
        assert det_metrics["excluded_count"] >= 1

    def test_unavailable_case_retained_in_comparisons(self) -> None:
        """Unavailable case ids remain in the report for auditability."""
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        case_ids = [c["case_id"] for c in report["case_comparisons"]]
        assert "unavailable_case" in case_ids


# ---------------------------------------------------------------------------
# Unavailable comparison report
# ---------------------------------------------------------------------------


class TestUnavailableReport:
    """Fail-closed unavailable report construction."""

    def test_unavailable_report_has_correct_status(self) -> None:
        """The fail-closed builder records the unavailable reason explicitly."""
        report = build_unavailable_comparison_report("provenance validation failed")
        assert report["output_status"] == "unavailable"
        assert report["output_reason"] == "provenance validation failed"

    def test_unavailable_report_has_no_summaries(self) -> None:
        """Unavailable reports do not expose partial metric summaries."""
        report = build_unavailable_comparison_report("case alignment mismatch")
        assert report["deterministic_summary"] is None
        assert report["learned_summary"] is None
        assert report["learned_source_projection"] is None

    def test_unavailable_report_has_claim_boundary(self) -> None:
        """Unavailable reports retain the no-scientific-result boundary."""
        report = build_unavailable_comparison_report("test reason")
        assert report["claim_boundary"]["no_scientific_result_claim"] is True

    def test_unavailable_report_with_valid_manifest_metadata(self) -> None:
        """Valid manifest metadata is retained when comparison is unavailable."""
        report = build_unavailable_comparison_report(
            "case mismatch", method_manifest=_valid_manifest()
        )
        assert report["method_manifest"]["method_id"] == "test-learned-method-v1"

    def test_unavailable_report_with_invalid_manifest_metadata(self) -> None:
        """Invalid optional manifest metadata is marked failed, not normalized."""
        report = build_unavailable_comparison_report(
            "provenance failed", method_manifest={"bad": "data"}
        )
        assert report["method_manifest"]["validation"] == "failed"

    def test_unavailable_report_without_manifest(self) -> None:
        """Unavailable reports permit absent manifest metadata without inventing it."""
        report = build_unavailable_comparison_report("no manifest provided")
        assert report["method_manifest"] is None


# ---------------------------------------------------------------------------
# Payload-shape inputs
# ---------------------------------------------------------------------------


class TestPayloadShapeInputs:
    """The harness accepts failure_diagnosis.v1 payloads as well as mappings."""

    def test_payload_shape_deterministic_records(self) -> None:
        """Complete versioned payloads are accepted as comparison inputs."""
        det = _build_valid_payload_from_predicates()
        # Build a matching learned payload (same records, same order)
        learn = _build_valid_payload_from_predicates()
        report = compare_held_out_diagnoses(_reviewed_fixture(), det, learn, _valid_manifest())
        assert report["output_status"] == "available"

    def test_payload_shape_mixed_inputs(self) -> None:
        """One side as payload, the other as case-id mapping with matching ids."""
        learn = _build_valid_payload_from_predicates()
        learn_ids = {r["case_id"] for r in learn["records"]}
        det = {cid: {"detected": True, "status": "available"} for cid in learn_ids}
        report = compare_held_out_diagnoses(_reviewed_fixture(), det, learn, _valid_manifest())
        assert report["output_status"] == "available"


# ---------------------------------------------------------------------------
# Fixture review admission (blocker 2)
# ---------------------------------------------------------------------------


class TestFixtureReviewAdmission:
    """Fixtures with pending review markers are rejected before evaluation."""

    def test_pending_fixture_rejected(self) -> None:
        """A case-insensitive pending marker with metadata blocks evaluation."""
        fixture = _reviewed_fixture()
        fixture["review_marker"] = "needs-review (2026-08)"
        with pytest.raises(FixtureReviewPendingError, match="pending review marker"):
            compare_held_out_diagnoses(
                fixture,
                _synthetic_deterministic_records(),
                _synthetic_learned_records(),
                _valid_manifest(),
            )

    def test_reviewed_fixture_accepted(self) -> None:
        """A fixture with the pending marker removed passes admission."""
        fixture = _reviewed_fixture()
        assert fixture["review_marker"] == "reviewed"
        report = compare_held_out_diagnoses(
            fixture,
            _synthetic_deterministic_records(),
            _synthetic_learned_records(),
            _valid_manifest(),
        )
        assert report["output_status"] == "available"

    def test_validate_fixture_review_admission_pending(self) -> None:
        """validate_fixture_review_admission rejects pending markers."""
        with pytest.raises(FixtureReviewPendingError):
            validate_fixture_review_admission({"review_marker": "AI-GENERATED NEEDS-REVIEW"})

    def test_validate_fixture_review_admission_reviewed(self) -> None:
        """validate_fixture_review_admission accepts reviewed markers."""
        validate_fixture_review_admission({"review_marker": "reviewed"})

    def test_validate_fixture_review_admission_no_marker(self) -> None:
        """validate_fixture_review_admission accepts fixtures without a marker."""
        validate_fixture_review_admission({})

    def test_all_pending_markers_rejected(self) -> None:
        """Every marker in REVIEW_PENDING_MARKERS is rejected."""
        for marker in REVIEW_PENDING_MARKERS:
            with pytest.raises(FixtureReviewPendingError):
                validate_fixture_review_admission({"review_marker": marker})

    def test_pending_marker_normalization_is_fail_closed(self) -> None:
        """Case and whitespace normalization rejects embedded pending markers."""
        with pytest.raises(FixtureReviewPendingError):
            validate_fixture_review_admission({"review_marker": "  Pending  review  "})


# ---------------------------------------------------------------------------
# Learned metric-input projection (blocker 1)
# ---------------------------------------------------------------------------


class TestLearnedSourceProjection:
    """Learned records are projected to preserve non-deterministic provenance."""

    def _nondeterministic_learned_records(self) -> dict[str, Any]:
        """Build learned records with a truthful non-deterministic diagnosis_source."""
        records = dict(_synthetic_deterministic_records())
        for record in records.values():
            if isinstance(record, dict):
                record["diagnosis_source"] = "failure_diagnosis.learned.v1"
        return records

    def test_nondeterministic_learned_source_accepted(self) -> None:
        """A truthful non-deterministic learned source is accepted and compared."""
        learn = self._nondeterministic_learned_records()
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            learn,
            _valid_manifest(),
        )
        assert report["output_status"] == "available"

    def test_nondeterministic_source_preserved_in_output(self) -> None:
        """The non-deterministic source is preserved in the comparison output."""
        learn = self._nondeterministic_learned_records()
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            learn,
            _valid_manifest(),
        )
        assert "learned_source_projection" in report
        projection = report["learned_source_projection"]
        assert "diagnosis_source" in projection["preserved_source_fields"]
        assert projection["preserved_fields"] == [
            "diagnosis_schema_version",
            "diagnosis_source",
        ]

    def test_nondeterministic_source_in_case_comparisons(self) -> None:
        """Each case comparison preserves the original learned source."""
        learn = self._nondeterministic_learned_records()
        report = compare_held_out_diagnoses(
            _reviewed_fixture(),
            _synthetic_deterministic_records(),
            learn,
            _valid_manifest(),
        )
        for case_comp in report["case_comparisons"]:
            preserved = case_comp["diagnosis"].get("_learned_source_preserved", {})
            assert preserved.get("diagnosis_source") == "failure_diagnosis.learned.v1"

    def test_deterministic_source_learned_rejected(self) -> None:
        """A learned record claiming deterministic source is rejected."""
        learn = dict(_synthetic_deterministic_records())
        # Ensure at least one record has the deterministic source
        learn["collision_case"]["diagnosis_source"] = DIAGNOSIS_SOURCE
        with pytest.raises(LearnedSourceError, match="deterministic diagnosis_source"):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _synthetic_deterministic_records(),
                learn,
                _valid_manifest(),
            )

    def test_missing_source_learned_rejected(self) -> None:
        """A learned record without a source marker is rejected."""
        learn = _synthetic_learned_records()
        del learn["collision_case"]["diagnosis_source"]
        with pytest.raises(LearnedSourceError, match="missing a non-empty"):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _synthetic_deterministic_records(),
                learn,
                _valid_manifest(),
            )

    def test_non_mapping_learned_record_rejected(self) -> None:
        """A learned record with no mapping shape fails closed explicitly."""
        with pytest.raises(LearnedSourceError, match="must be a mapping"):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _synthetic_deterministic_records(),
                [None],
                _valid_manifest(),
            )

    def test_deterministic_source_in_payload_rejected(self) -> None:
        """A learned payload with deterministic source records is rejected."""
        learn = _build_valid_payload_from_predicates()
        # Inject a deterministic source into one record
        learn["records"][0]["diagnosis_source"] = DIAGNOSIS_SOURCE
        with pytest.raises(LearnedSourceError, match="deterministic diagnosis_source"):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _build_valid_payload_from_predicates(),
                learn,
                _valid_manifest(),
            )

    def test_deterministic_payload_envelope_source_rejected(self) -> None:
        """A learned payload envelope may not claim the deterministic source."""
        learn = _build_valid_payload_from_predicates()
        learn["diagnosis_source"] = DIAGNOSIS_SOURCE
        with pytest.raises(LearnedSourceError, match="deterministic diagnosis_source"):
            compare_held_out_diagnoses(
                _reviewed_fixture(),
                _build_valid_payload_from_predicates(),
                learn,
                _valid_manifest(),
            )


# ---------------------------------------------------------------------------
# Payload shape ambiguity (blocker 4)
# ---------------------------------------------------------------------------


class TestPayloadShapeAmbiguity:
    """A case literally named 'records' must not be mistaken for a payload."""

    def test_records_key_as_case_name_treated_as_mapping(self) -> None:
        """A mapping with 'records' as a case name is treated as a case-id mapping."""
        det = {
            "collision_case": {"detected": True, "status": "available"},
            "records": {"detected": False, "status": "available"},
        }
        from robot_sf.benchmark.failure_diagnosis_comparison import _is_payload_shape

        assert _is_payload_shape(det) is False

    def test_generator_inputs_are_materialized_before_repeated_passes(self) -> None:
        """Generator inputs survive source validation, alignment, and evaluation."""
        deterministic = (
            dict(record, case_id=case_id)
            for case_id, record in _synthetic_deterministic_records().items()
        )
        learned = (
            dict(record, case_id=case_id)
            for case_id, record in _synthetic_learned_records().items()
        )
        report = compare_held_out_diagnoses(
            _reviewed_fixture(), deterministic, learned, _valid_manifest()
        )
        assert report["alignment"]["deterministic_count"] == 6
        assert report["learned_summary"]["case_count"] == 6

    def test_records_key_with_list_of_non_mappings_not_payload(self) -> None:
        """A mapping with 'records' as a list of non-mappings is not a payload."""
        det = {"records": ["not_a_mapping", "also_not"]}
        from robot_sf.benchmark.failure_diagnosis_comparison import _is_payload_shape

        assert _is_payload_shape(det) is False

    def test_records_key_with_empty_list_not_payload(self) -> None:
        """A mapping with an empty 'records' list is not a payload."""
        det: dict[str, Any] = {"records": []}
        from robot_sf.benchmark.failure_diagnosis_comparison import _is_payload_shape

        assert _is_payload_shape(det) is False

    def test_records_key_with_dict_not_payload(self) -> None:
        """A mapping with 'records' as a dict is not a payload."""
        det = {"records": {"case_id": "x"}}
        from robot_sf.benchmark.failure_diagnosis_comparison import _is_payload_shape

        assert _is_payload_shape(det) is False

    def test_valid_payload_shape_detected(self) -> None:
        """A valid payload with records list of mappings is detected."""
        det = {
            "schema_version": "failure_diagnosis.v1",
            "diagnosis_source": DIAGNOSIS_SOURCE,
            "generated_at_utc": "2026-08-13T00:00:00+00:00",
            "failure_type_coverage": {},
            "caveats": [],
            "records": [{"case_id": "a", "detected": True}],
        }
        from robot_sf.benchmark.failure_diagnosis_comparison import _is_payload_shape

        assert _is_payload_shape(det) is True


# ---------------------------------------------------------------------------
# Module-level contract
# ---------------------------------------------------------------------------


class TestModuleContract:
    """Module-level public API surface checks."""

    def test_all_exports_are_importable(self) -> None:
        """Every declared public symbol is importable from the module."""
        from robot_sf.benchmark import failure_diagnosis_comparison as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"__all__ declares {name!r} but it is missing"

    def test_new_error_classes_exported(self) -> None:
        """New admission errors and helpers remain part of the public contract."""
        from robot_sf.benchmark import failure_diagnosis_comparison as mod

        assert "FixtureReviewPendingError" in mod.__all__
        assert "LearnedSourceError" in mod.__all__
        assert "REVIEW_PENDING_MARKERS" in mod.__all__
        assert "validate_fixture_review_admission" in mod.__all__
