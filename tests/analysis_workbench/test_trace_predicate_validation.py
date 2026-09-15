# evidence-writer-exempt: tmp_path parser fixtures only; no repository evidence writes.
"""Tests for the issue #9305 trace-predicate validation contract."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
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
SOURCE_COMMIT = "1e86f17e9460c9828f0c1b03cabe87c21da594ce"
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
    return load_trace_predicate_validation_set(
        EVALUATION_SET_PATH,
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )


def test_bounded_fixture_reports_all_required_validation_dimensions() -> None:
    """The fixture exercises labels, missingness, review, thresholds, and grouping ablation."""
    evaluation_set = _load_fixture()
    report = build_trace_predicate_validation_report(
        evaluation_set,
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )

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
        "1e86f17e9460c9828f0c1b03cabe87c21da594ce"
    )


def test_trace_digest_drift_is_rejected() -> None:
    """A changed referenced file digest cannot silently enter the evaluation set."""
    payload = _load_fixture()
    payload["cases"][0]["trace_ref"]["sha256"] = "0" * 64

    with pytest.raises(TracePredicateValidationError, match="digest does not match"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    """Ambiguous JSON input cannot silently replace an earlier object value."""
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version": "trace_predicate_validation.v1", '
        '"schema_version": "trace_predicate_validation.v1"}',
        encoding="utf-8",
    )

    with pytest.raises(TracePredicateValidationError, match="duplicate JSON object key"):
        load_trace_predicate_validation_set(
            duplicate_path,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_nonstandard_json_constants_are_rejected_by_api(tmp_path: Path, constant: str) -> None:
    """The evaluation-set loader rejects JSON constants outside the standard."""
    invalid_path = tmp_path / "nonstandard-constant.json"
    invalid_path.write_text(
        json.dumps({"threshold_sensitivity": {"variants": [{"parameters": float(constant)}]}}),
        encoding="utf-8",
    )

    with pytest.raises(TracePredicateValidationError, match="non-standard JSON constant"):
        load_trace_predicate_validation_set(
            invalid_path,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_nonstandard_json_constants_are_normalized_at_cli_boundary(
    tmp_path: Path, constant: str
) -> None:
    """Malformed JSON reaches the documented exit-2 CLI error boundary."""
    evaluation_set = _load_fixture()
    parameters = evaluation_set["threshold_sensitivity"]["variants"][0]["parameters"]
    parameters["threshold"] = float(constant)
    evaluation_path = tmp_path / "nonstandard-constant.json"
    evaluation_path.write_text(json.dumps(evaluation_set), encoding="utf-8")
    output_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/validate_trace_predicate_evaluation_issue_9305.py",
            "--evaluation-set",
            str(evaluation_path),
            "--output-json",
            str(output_path),
            "--repo-root",
            str(REPO_ROOT),
            "--expected-source-commit",
            SOURCE_COMMIT,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "trace predicate validation unavailable" in result.stderr
    assert "non-standard JSON constant" in result.stderr
    assert not output_path.exists()


def test_current_base_pin_is_checked() -> None:
    """A fixture from another base cannot be admitted silently."""
    with pytest.raises(TracePredicateValidationError, match="different source commit"):
        validate_trace_predicate_evaluation_set(
            _load_fixture(),
            repo_root=REPO_ROOT,
            expected_source_commit="0" * 40,
        )


def test_explicit_source_pin_is_required() -> None:
    """A payload declaration cannot substitute for an explicit caller pin."""
    with pytest.raises(
        TracePredicateValidationError, match="expected source commit pin is required"
    ):
        validate_trace_predicate_evaluation_set(_load_fixture(), repo_root=REPO_ROOT)


def test_available_traces_must_match_git_blobs_at_declared_commit(tmp_path: Path) -> None:
    """Byte-identical copies outside Git cannot masquerade as source-bound traces."""
    payload = _load_fixture()
    for case in payload["cases"]:
        trace_ref = case["trace_ref"]
        if trace_ref["status"] != "available":
            continue
        source_path = REPO_ROOT / trace_ref["uri"]
        copied_path = tmp_path / trace_ref["uri"]
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        copied_path.write_bytes(source_path.read_bytes())

    with pytest.raises(TracePredicateValidationError, match="Git blob"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=tmp_path,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_declared_minimum_reviewer_count_is_enforced() -> None:
    """Every case must satisfy the evaluation set's declared reviewer minimum."""
    payload = _load_fixture()
    payload["review_protocol"]["minimum_reviewers"] = 3

    with pytest.raises(TracePredicateValidationError, match="requires at least 3 reviewers"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_bounded_fixture_map_identity_cannot_be_source_bound() -> None:
    """Fixture traces cannot turn an annotation-only map ID into source provenance."""
    payload = _load_fixture()
    payload["cases"][0]["map_identity_status"] = "source_bound"

    with pytest.raises(TracePredicateValidationError, match="fixture_annotation"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


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
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=tmp_path,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_bounded_fixture_cannot_claim_retained_trace_availability() -> None:
    """The fixture slice cannot be relabeled as a partial retained corpus."""
    payload = _load_fixture()
    payload["retained_trace_status"] = "partial"

    with pytest.raises(TracePredicateValidationError, match="must remain unavailable"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_missing_detector_reason_and_path_traversal_are_rejected() -> None:
    """Unavailable labels and trace paths must carry explicit fail-closed metadata."""
    payload = _load_fixture()
    missing_reason = copy.deepcopy(payload)
    del missing_reason["cases"][3]["detector"]["unavailable_reasons"]["collision"]
    with pytest.raises(TracePredicateValidationError, match="reason required"):
        validate_trace_predicate_evaluation_set(
            missing_reason,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )

    traversal = copy.deepcopy(payload)
    traversal["cases"][0]["trace_ref"]["uri"] = "../outside.json"
    with pytest.raises(TracePredicateValidationError, match="traversal"):
        validate_trace_predicate_evaluation_set(
            traversal,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_adjudication_requires_independence_and_measured_effort() -> None:
    """Adjudication cannot reuse a reviewer or omit its measured effort."""
    reused_id = _load_fixture()
    reused_id["cases"][0]["review"]["adjudication"]["reviewer_id"] = "reviewer_a"
    with pytest.raises(TracePredicateValidationError, match="adjudicator must be distinct"):
        validate_trace_predicate_evaluation_set(
            reused_id,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )

    missing_effort = _load_fixture()
    del missing_effort["cases"][0]["review"]["adjudication"]["effort_minutes"]
    with pytest.raises(TracePredicateValidationError, match="not valid under any"):
        validate_trace_predicate_evaluation_set(
            missing_effort,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_unavailable_detector_cannot_get_positive_threshold_label() -> None:
    """Threshold projections must preserve unavailable detector boundaries."""
    payload = _load_fixture()
    payload["cases"][0]["detector"]["labels"]["collision"] = "unavailable"
    payload["cases"][0]["detector"]["unavailable_reasons"] = {"collision": "missing metric"}
    for variant in payload["threshold_sensitivity"]["variants"]:
        variant["labels_by_case"]["fixture-crossing-true-positive"]["collision"] = "positive"

    with pytest.raises(TracePredicateValidationError, match="unavailable trace or detector"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_missing_trace_cannot_get_positive_threshold_label() -> None:
    """Missing traces must remain unavailable in every threshold variant."""
    payload = _load_fixture()
    for variant in payload["threshold_sensitivity"]["variants"]:
        variant["labels_by_case"]["fixture-missing-inputs"]["collision"] = "positive"

    with pytest.raises(TracePredicateValidationError, match="unavailable trace or detector"):
        validate_trace_predicate_evaluation_set(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize("label", ("positive", "negative"))
def test_unavailable_trace_cannot_enter_reference_ledgers(label: str) -> None:
    """Unavailable traces require pending adjudication to stay out of reference metrics."""
    payload = _load_fixture()
    payload["cases"][3]["review"]["adjudication"] = {
        "status": "adjudicated",
        "reviewer_id": "adjudicator",
        "labels": dict.fromkeys(TRACE_FAILURE_PREDICATE_IDS, label),
        "effort_minutes": 1.0,
        "note": "adversarial mutation",
    }

    with pytest.raises(
        TracePredicateValidationError, match="unavailable trace requires pending adjudication"
    ):
        build_trace_predicate_validation_report(
            payload,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_identity_ablation_features_are_closed_and_disjoint() -> None:
    """Identity ablations cannot overlap or introduce undeclared feature names."""
    overlapping = _load_fixture()
    overlapping["grouping_stability"]["variants"][1]["included_features"].append("planner_id")
    with pytest.raises(TracePredicateValidationError, match="must be disjoint"):
        validate_trace_predicate_evaluation_set(
            overlapping,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )

    unknown = _load_fixture()
    unknown["grouping_stability"]["variants"][1]["included_features"].append("future_feature")
    with pytest.raises(TracePredicateValidationError, match="is not one of"):
        validate_trace_predicate_evaluation_set(
            unknown,
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_pending_review_is_excluded_without_majority_imputation() -> None:
    """Pending reviewer disagreement contributes no reference confusion-matrix label."""
    report = build_trace_predicate_validation_report(
        _load_fixture(),
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )
    bottleneck = report["predicate_metrics"]["bottleneck_deadlock"]

    assert bottleneck["evaluated_count"] == 2
    assert bottleneck["excluded_count"] == 2
    assert bottleneck["excluded_reasons"]["reference_pending"] == 2


def test_report_ratio_drift_is_rejected() -> None:
    """A report cannot hide denominator drift behind a valid top-level envelope."""
    report = build_trace_predicate_validation_report(
        _load_fixture(),
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )
    report["predicate_metrics"]["collision"]["precision"] = 1.0

    with pytest.raises(TracePredicateValidationError, match="ratio does not match counts"):
        validate_trace_predicate_validation_report(
            report,
            evaluation_set=_load_fixture(),
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_grouping_ratio_drift_is_rejected() -> None:
    """Grouping stability ratios must remain derived from the admitted pairs."""
    report = build_trace_predicate_validation_report(
        _load_fixture(),
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )
    report["grouping_stability"]["pairwise_jaccard"] = 1.0

    with pytest.raises(TracePredicateValidationError, match="pairwise_jaccard"):
        validate_trace_predicate_validation_report(
            report,
            evaluation_set=_load_fixture(),
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_report_schema_requires_all_eight_predicate_sections() -> None:
    """A report missing one predicate cannot pass the report envelope check."""
    report = build_trace_predicate_validation_report(
        _load_fixture(),
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )
    del report["predicate_metrics"]["collision"]

    with pytest.raises(TracePredicateValidationError, match="required property"):
        validate_trace_predicate_validation_report(
            report,
            evaluation_set=_load_fixture(),
            repo_root=REPO_ROOT,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_report_validation_requires_admitted_set_and_recomputes_all_fields() -> None:
    """A report summary cannot self-authorize its digest or derived fields."""
    evaluation_set = _load_fixture()
    report = build_trace_predicate_validation_report(
        evaluation_set,
        repo_root=REPO_ROOT,
        expected_source_commit=SOURCE_COMMIT,
    )

    with pytest.raises(TracePredicateValidationError, match="admitted evaluation set"):
        validate_trace_predicate_validation_report(report)

    for mutation, expected_path in (
        (
            lambda value: value["evaluation_set"].__setitem__("sha256", "0" * 64),
            "evaluation_set/sha256",
        ),
        (
            lambda value: value["evaluation_set"].__setitem__("source_commit", "0" * 40),
            "evaluation_set/source_commit",
        ),
        (
            lambda value: value["coverage"].__setitem__("trace_available_count", 0),
            "coverage/trace_available_count",
        ),
        (
            lambda value: value["coverage"].__setitem__("reference_positive_label_count", 0),
            "coverage/reference_positive_label_count",
        ),
    ):
        mutated = copy.deepcopy(report)
        mutation(mutated)
        with pytest.raises(TracePredicateValidationError, match=expected_path):
            validate_trace_predicate_validation_report(
                mutated,
                evaluation_set=evaluation_set,
                repo_root=REPO_ROOT,
                expected_source_commit=SOURCE_COMMIT,
            )


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


@pytest.mark.parametrize("record_ids", ("a", b"a"))
def test_similarity_group_comparison_rejects_scalar_record_id_universe(
    record_ids: object,
) -> None:
    """An explicit record universe must be a sequence, not a scalar string/bytes value."""
    with pytest.raises(ValueError, match="non-string sequence"):
        compare_similarity_groupings(
            [{"group_id": "r1", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            record_ids=record_ids,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("reference", "comparison", "record_ids", "match"),
    (
        ([], [{"group_id": "c1", "record_ids": ["a"]}], None, "reference groups"),
        ("not-groups", [{"group_id": "c1", "record_ids": ["a"]}], None, "reference groups"),
        ([object()], [{"group_id": "c1", "record_ids": ["a"]}], None, "contain objects"),
        (
            [{"group_id": "", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            None,
            "group IDs",
        ),
        (
            [{"group_id": "r1", "record_ids": ["a"]}, {"group_id": "r1", "record_ids": ["b"]}],
            [{"group_id": "c1", "record_ids": ["a", "b"]}],
            None,
            "group IDs",
        ),
        (
            [{"group_id": "r1", "record_ids": []}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            None,
            "group members",
        ),
        (
            [{"group_id": "r1", "record_ids": "a"}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            None,
            "group members",
        ),
        (
            [{"group_id": "r1", "record_ids": [1]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            None,
            "record_ids",
        ),
        (
            [{"group_id": "r1", "record_ids": ["a"]}, {"group_id": "r2", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            None,
            "partition",
        ),
        (
            [{"group_id": "r1", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["b"]}],
            None,
            "universes",
        ),
        (
            [{"group_id": "r1", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            [1],
            "record_ids",
        ),
        (
            [{"group_id": "r1", "record_ids": ["a"]}],
            [{"group_id": "c1", "record_ids": ["a"]}],
            ["a", "a"],
            "record_ids",
        ),
    ),
)
def test_similarity_group_comparison_rejects_malformed_inputs(
    reference: object,
    comparison: object,
    record_ids: object,
    match: str,
) -> None:
    """Grouping comparisons fail closed for malformed partitions and universes."""
    with pytest.raises(ValueError, match=match):
        compare_similarity_groupings(
            reference,  # type: ignore[arg-type]
            comparison,  # type: ignore[arg-type]
            record_ids=record_ids,  # type: ignore[arg-type]
        )
