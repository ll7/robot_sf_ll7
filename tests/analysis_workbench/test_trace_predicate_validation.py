# evidence-writer-exempt: tmp_path parser fixtures only; no repository evidence writes.
"""Tests for the issue #9305 trace-predicate validation contract."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from robot_sf.analysis_workbench import (
    TRACE_PREDICATE_VALIDATION_SCHEMA_VERSION,
    TracePredicateValidationError,
    build_trace_predicate_validation_report,
    load_trace_predicate_validation_set,
    validate_trace_predicate_evaluation_set,
    validate_trace_predicate_validation_report,
)
from robot_sf.analysis_workbench.trace_failure_predicates import TRACE_FAILURE_PREDICATE_IDS
from robot_sf.benchmark.collision.collision_scenario_similarity import (
    compare_similarity_groupings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SET_PATH = (
    REPO_ROOT
    / "docs"
    / "context"
    / "evidence"
    / "issue_9305_predicate_validation"
    / "evaluation_set.v1.json"
)


def _load_fixture() -> dict:
    """Load the committed bounded evaluation set."""
    return load_trace_predicate_validation_set(EVALUATION_SET_PATH, repo_root=REPO_ROOT)


def test_bounded_fixture_reports_all_required_validation_dimensions() -> None:
    """The fixture exercises labels, missingness, review, thresholds, and grouping ablation."""
    evaluation_set = _load_fixture()
    report = build_trace_predicate_validation_report(evaluation_set, repo_root=REPO_ROOT)

    assert evaluation_set["schema_version"] == TRACE_PREDICATE_VALIDATION_SCHEMA_VERSION
    assert report["evaluation_set"]["retained_trace_status"] == "unavailable"
    assert report["coverage"]["trace_available_count"] == 3
    assert report["coverage"]["trace_unavailable_count"] == 1
    assert report["coverage"]["planner_count"] == 3
    assert report["coverage"]["map_count"] == 3
    assert report["coverage"]["reference_positive_label_count"] == 8
    assert report["coverage"]["reference_negative_label_count"] == 8
    assert report["coverage"]["reference_unavailable_or_pending_label_count"] == 16

    collision = report["predicate_metrics"]["collision"]
    assert collision["evaluated_count"] == 2
    assert collision["true_positive"] == 1
    assert collision["false_positive"] == 1
    assert collision["precision"] == 0.5
    assert collision["recall"] == 1.0

    low_progress = report["predicate_metrics"]["low_progress"]
    assert low_progress["true_negative"] == 1
    assert low_progress["false_negative"] == 1
    assert low_progress["precision"] is None
    assert low_progress["recall"] == 0.0

    unavailable = report["unavailable_rates"]["collision"]
    assert unavailable["unavailable_case_count"] == 1
    assert unavailable["unavailable_fraction"] == 0.25
    assert unavailable["metric_excluded_count"] == 2

    disagreement = report["reviewer_agreement"]["bottleneck_deadlock"]
    assert disagreement["disagreement_pair_count"] == 1
    assert disagreement["unresolved_disagreement_count"] == 1
    assert disagreement["disagreement_cases"] == [
        {
            "case_id": "fixture-bottleneck-disagreement",
            "labels_by_reviewer": {"reviewer_a": "positive", "reviewer_b": "negative"},
            "adjudication_status": "pending",
        }
    ]

    threshold = report["threshold_stability"]["by_predicate"]["occlusion_triggered_near_miss"][
        "variants"
    ][1]
    assert threshold["variant_id"] == "looser"
    assert threshold["changed_case_count"] == 1
    assert threshold["positive_set_jaccard_to_baseline"] == 0.0

    grouping = report["grouping_stability"]
    assert grouping["identity_features"] == ["planner_id", "map_id"]
    assert grouping["record_count"] == 4
    assert grouping["changed_pair_count"] == 1
    assert grouping["same_group_pair_agreement"] == pytest.approx(5 / 6)
    assert grouping["pairwise_jaccard"] == 0.0
    assert report["observations"]["causal_hypotheses"]["used_for_metrics"] is False
    assert set(report["predicate_metrics"]) == set(TRACE_FAILURE_PREDICATE_IDS)
    assert evaluation_set["provenance"]["source_commit"] == (
        "754509029710564ef801b79cc8d86a78011899f4"
    )


def test_trace_digest_drift_is_rejected() -> None:
    """A changed referenced file digest cannot silently enter the evaluation set."""
    payload = _load_fixture()
    payload["cases"][0]["trace_ref"]["sha256"] = "0" * 64

    with pytest.raises(TracePredicateValidationError, match="digest does not match"):
        validate_trace_predicate_evaluation_set(payload, repo_root=REPO_ROOT)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    """Ambiguous JSON input cannot silently replace an earlier object value."""
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version": "trace_predicate_validation.v1", '
        '"schema_version": "trace_predicate_validation.v1"}',
        encoding="utf-8",
    )

    with pytest.raises(TracePredicateValidationError, match="duplicate JSON object key"):
        load_trace_predicate_validation_set(duplicate_path, repo_root=REPO_ROOT)


def test_current_base_pin_is_checked() -> None:
    """A fixture from another base cannot be admitted silently."""
    with pytest.raises(TracePredicateValidationError, match="different source commit"):
        validate_trace_predicate_evaluation_set(
            _load_fixture(),
            repo_root=REPO_ROOT,
            expected_source_commit="0" * 40,
        )


def test_declared_minimum_reviewer_count_is_enforced() -> None:
    """Every case must satisfy the evaluation set's declared reviewer minimum."""
    payload = _load_fixture()
    payload["review_protocol"]["minimum_reviewers"] = 3

    with pytest.raises(TracePredicateValidationError, match="requires at least 3 reviewers"):
        validate_trace_predicate_evaluation_set(payload, repo_root=REPO_ROOT)


def test_bounded_fixture_map_identity_cannot_be_source_bound() -> None:
    """Fixture traces cannot turn an annotation-only map ID into source provenance."""
    payload = _load_fixture()
    payload["cases"][0]["map_identity_status"] = "source_bound"

    with pytest.raises(TracePredicateValidationError, match="fixture_annotation"):
        validate_trace_predicate_evaluation_set(payload, repo_root=REPO_ROOT)


def test_duplicate_keys_in_referenced_trace_are_rejected(tmp_path: Path) -> None:
    """Referenced traces use the same duplicate-key rejection as evaluation JSON."""
    payload = _load_fixture()
    source_path = REPO_ROOT / payload["cases"][0]["trace_ref"]["uri"]
    trace_path = tmp_path / "trace.json"
    trace_text = source_path.read_text(encoding="utf-8")
    duplicate = '"schema_version": "simulation_trace_export.v1",\n  "schema_version": "simulation_trace_export.v1",'
    trace_path.write_text(
        trace_text.replace('"schema_version": "simulation_trace_export.v1",', duplicate, 1),
        encoding="utf-8",
    )
    payload["cases"][0]["trace_ref"]["uri"] = "trace.json"
    payload["cases"][0]["trace_ref"]["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()

    with pytest.raises(TracePredicateValidationError, match="duplicate JSON object key"):
        validate_trace_predicate_evaluation_set(payload, repo_root=tmp_path)


def test_bounded_fixture_cannot_claim_retained_trace_availability() -> None:
    """The fixture slice cannot be relabeled as a partial retained corpus."""
    payload = _load_fixture()
    payload["retained_trace_status"] = "partial"

    with pytest.raises(TracePredicateValidationError, match="must remain unavailable"):
        validate_trace_predicate_evaluation_set(payload, repo_root=REPO_ROOT)


def test_missing_detector_reason_and_path_traversal_are_rejected() -> None:
    """Unavailable labels and trace paths must carry explicit fail-closed metadata."""
    payload = _load_fixture()
    missing_reason = copy.deepcopy(payload)
    del missing_reason["cases"][3]["detector"]["unavailable_reasons"]["collision"]
    with pytest.raises(TracePredicateValidationError, match="reason required"):
        validate_trace_predicate_evaluation_set(missing_reason, repo_root=REPO_ROOT)

    traversal = copy.deepcopy(payload)
    traversal["cases"][0]["trace_ref"]["uri"] = "../outside.json"
    with pytest.raises(TracePredicateValidationError, match="traversal"):
        validate_trace_predicate_evaluation_set(traversal, repo_root=REPO_ROOT)


def test_pending_review_is_excluded_without_majority_imputation() -> None:
    """Pending reviewer disagreement contributes no reference confusion-matrix label."""
    report = build_trace_predicate_validation_report(_load_fixture(), repo_root=REPO_ROOT)
    bottleneck = report["predicate_metrics"]["bottleneck_deadlock"]

    assert bottleneck["evaluated_count"] == 2
    assert bottleneck["excluded_count"] == 2
    assert bottleneck["excluded_reasons"]["reference_pending"] == 2


def test_report_ratio_drift_is_rejected() -> None:
    """A report cannot hide denominator drift behind a valid top-level envelope."""
    report = build_trace_predicate_validation_report(_load_fixture(), repo_root=REPO_ROOT)
    report["predicate_metrics"]["collision"]["precision"] = 1.0

    with pytest.raises(TracePredicateValidationError, match="ratio does not match counts"):
        validate_trace_predicate_validation_report(report)


def test_grouping_ratio_drift_is_rejected() -> None:
    """Grouping stability ratios must remain derived from the admitted pairs."""
    report = build_trace_predicate_validation_report(_load_fixture(), repo_root=REPO_ROOT)
    report["grouping_stability"]["pairwise_jaccard"] = 1.0

    with pytest.raises(TracePredicateValidationError, match="pairwise_jaccard"):
        validate_trace_predicate_validation_report(report)


def test_report_schema_requires_all_eight_predicate_sections() -> None:
    """A report missing one predicate cannot pass the report envelope check."""
    report = build_trace_predicate_validation_report(_load_fixture(), repo_root=REPO_ROOT)
    del report["predicate_metrics"]["collision"]

    with pytest.raises(TracePredicateValidationError, match="required property"):
        validate_trace_predicate_validation_report(report)


def test_similarity_group_comparison_is_pairwise_and_label_free() -> None:
    """The collision owner compares grouping partitions without inventing group labels."""
    reference = [
        {"group_id": "r1", "record_ids": ["a", "b"]},
        {"group_id": "r2", "record_ids": ["c"]},
    ]
    comparison = [
        {"group_id": "c1", "record_ids": ["a", "b", "c"]},
    ]

    result = compare_similarity_groupings(reference, comparison, record_ids=["a", "b", "c"])

    assert result["pair_count"] == 3
    assert result["reference_same_group_pair_count"] == 1
    assert result["comparison_same_group_pair_count"] == 3
    assert result["disagreement_pair_count"] == 2
    assert result["pairwise_jaccard"] == pytest.approx(1 / 3)


def test_similarity_group_comparison_rejects_incomplete_partition() -> None:
    """Grouping stability cannot compare assignments with silently missing records."""
    with pytest.raises(ValueError, match="cover the explicit record universe"):
        compare_similarity_groupings(
            [{"group_id": "r1", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            record_ids=["a", "b"],
        )
