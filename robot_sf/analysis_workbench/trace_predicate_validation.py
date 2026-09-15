"""Fail-closed validation contract for trace predicates and similarity groups.

The contract in this module is deliberately narrower than a benchmark runner.
It admits a versioned set of trace references, detector labels, reviewer labels,
threshold variants, and precomputed similarity-group assignments.  It reports
confusion-matrix mechanics, missingness, reviewer disagreement, and pairwise
group stability without treating a bounded fixture as a retained-trace corpus
or turning an observed pattern into a causal explanation.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExportValidationError,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_IDS,
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
    TRACE_FAILURE_PREDICATE_SOURCE,
)
from robot_sf.benchmark.collision.collision_scenario_similarity import (
    SCHEMA_VERSION as COLLISION_SCENARIO_SIMILARITY_SCHEMA_VERSION,
)
from robot_sf.benchmark.collision.collision_scenario_similarity import (
    compare_similarity_groupings,
)
from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

TRACE_PREDICATE_VALIDATION_SCHEMA_VERSION = "trace_predicate_validation.v1"
TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_VERSION = "trace_predicate_validation_report.v1"
TRACE_PREDICATE_VALIDATION_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "trace_predicate_validation.v1.json"
)
TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "trace_predicate_validation_report.v1.json"
)

EVIDENCE_STATUS_DIAGNOSTIC_ONLY = "diagnostic-only"
RETAINED_TRACE_STATUS_UNAVAILABLE = "unavailable"
LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_UNAVAILABLE = "unavailable"
VALIDATION_LABELS = frozenset({LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_AMBIGUOUS, LABEL_UNAVAILABLE})
_GROUPING_FEATURE_VOCABULARY = frozenset({"observed_pattern", "planner_id", "map_id"})
_SOURCE_COMMIT_LENGTH = 40
_SHA256_LENGTH = 64


class TracePredicateValidationError(RobotSfError, ValueError):
    """Raised when a predicate-validation set or report fails its contract."""

    def __init__(self, errors: str | Sequence[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema or semantic messages."""
        normalized_errors = [errors] if isinstance(errors, str) else list(errors)
        self.errors = tuple(normalized_errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(normalized_errors))


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently keeping the last value.

    Returns:
        The object mapping when every key is unique.
    """
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        mapping[key] = value
    return mapping


def _strict_json_loads(text: str) -> Any:
    """Parse JSON while preserving fail-closed duplicate-key semantics.

    Returns:
        The parsed JSON value.
    """
    return json.loads(text, object_pairs_hook=_reject_duplicate_json_object)


@lru_cache(maxsize=1)
def load_trace_predicate_validation_schema() -> dict[str, Any]:
    """Load the versioned evaluation-set JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    try:
        schema = _strict_json_loads(
            TRACE_PREDICATE_VALIDATION_SCHEMA_FILE.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TracePredicateValidationError(
            f"unable to load evaluation-set schema: {exc}",
            source=TRACE_PREDICATE_VALIDATION_SCHEMA_FILE,
        ) from exc
    if not isinstance(schema, dict):
        raise TracePredicateValidationError(
            "evaluation-set schema must be an object", source=TRACE_PREDICATE_VALIDATION_SCHEMA_FILE
        )
    return schema


@lru_cache(maxsize=1)
def load_trace_predicate_validation_report_schema() -> dict[str, Any]:
    """Load the versioned report JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    try:
        schema = _strict_json_loads(
            TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_FILE.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TracePredicateValidationError(
            f"unable to load report schema: {exc}",
            source=TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_FILE,
        ) from exc
    if not isinstance(schema, dict):
        raise TracePredicateValidationError(
            "report schema must be an object", source=TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_FILE
        )
    return schema


def load_trace_predicate_validation_set(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Load and validate one JSON evaluation set.

    ``repo_root`` is the root against which available trace URIs are resolved.
    It is explicit in the API so a caller cannot accidentally resolve a
    provenance-bound URI relative to an arbitrary fixture directory.

    Returns:
        Validated evaluation-set mapping.
    """
    evaluation_path = Path(path)
    try:
        payload = _strict_json_loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TracePredicateValidationError(
            f"unable to load JSON evaluation set: {exc}", source=evaluation_path
        ) from exc
    return validate_trace_predicate_evaluation_set(
        payload,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
        source=evaluation_path,
    )


def validate_trace_predicate_evaluation_set(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    expected_source_commit: str | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate schema, identity, trace references, review coverage, and partitions.

    The function does not execute a simulator or recompute detector labels.
    Available traces are read only to validate their strict export schema and
    source identity against the manifest row.

    Returns:
        Validated evaluation-set mapping.
    """
    if not isinstance(payload, Mapping):
        raise TracePredicateValidationError("evaluation set must be an object", source=source)
    schema_errors = _schema_errors(payload, load_trace_predicate_validation_schema())
    if schema_errors:
        raise TracePredicateValidationError(schema_errors, source=source)
    root = Path(repo_root or Path.cwd()).resolve()
    semantic_errors = _semantic_errors(
        payload,
        repo_root=root,
        expected_source_commit=expected_source_commit,
    )
    if semantic_errors:
        raise TracePredicateValidationError(semantic_errors, source=source)
    return dict(payload)


def validate_trace_predicate_validation_report(
    report: Mapping[str, Any],
    *,
    evaluation_set: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    expected_source_commit: str | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a report against the admitted evaluation set and recompute its fields.

    A report summary is not a provenance boundary on its own: its evaluation-set
    digest, source pin, coverage, and derived metrics can all be edited without
    changing the report's JSON shape.  Callers must therefore provide the
    admitted evaluation-set payload and the explicit source commit pin.  The
    validator re-admits that set and compares every report field with a fresh
    deterministic reconstruction.

    Returns:
        Validated report mapping.
    """
    if not isinstance(report, Mapping):
        raise TracePredicateValidationError("report must be an object", source=source)
    errors = _schema_errors(report, load_trace_predicate_validation_report_schema())
    if errors:
        raise TracePredicateValidationError(errors, source=source)
    binding_errors: list[str] = []
    if evaluation_set is None:
        binding_errors.append(
            "/evaluation_set: admitted evaluation set is required for report validation"
        )
    if expected_source_commit is None:
        binding_errors.append(
            "/provenance/source_commit: expected source commit pin is required for report validation"
        )
    if binding_errors:
        raise TracePredicateValidationError(binding_errors, source=source)
    assert evaluation_set is not None
    payload = validate_trace_predicate_evaluation_set(
        evaluation_set,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
    )
    semantic_errors = _report_semantic_errors(report)
    if semantic_errors:
        raise TracePredicateValidationError(semantic_errors, source=source)
    expected_report = _build_report_from_payload(payload)
    drift_paths = _value_difference_paths(report, expected_report)
    if drift_paths:
        raise TracePredicateValidationError(
            [
                f"{path}: report field does not match recomputed evaluation-set output"
                for path in drift_paths
            ],
            source=source,
        )
    return dict(report)


def canonical_trace_predicate_validation_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable digest of an evaluation-set JSON object."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_report_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete deterministic report projection for an admitted set.

    Returns:
        The unvalidated report projection.
    """
    cases = payload["cases"]
    return {
        "schema_version": TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_VERSION,
        "report_kind": "trace_predicate_validation",
        "evaluation_set": {
            "evaluation_set_id": payload["evaluation_set_id"],
            "evaluation_set_version": payload["evaluation_set_version"],
            "sha256": canonical_trace_predicate_validation_sha256(payload),
            "case_count": len(cases),
            "evidence_status": payload["evidence_status"],
            "retained_trace_status": payload["retained_trace_status"],
            "source_commit": payload["provenance"]["source_commit"],
            "source_kind": payload["provenance"]["source_kind"],
        },
        "claim_boundary": payload["claim_boundary"],
        "coverage": _coverage(cases),
        "predicate_metrics": {
            predicate_id: _predicate_metrics(cases, predicate_id)
            for predicate_id in TRACE_FAILURE_PREDICATE_IDS
        },
        "unavailable_rates": {
            predicate_id: _unavailable_rate(cases, predicate_id)
            for predicate_id in TRACE_FAILURE_PREDICATE_IDS
        },
        "reviewer_agreement": {
            predicate_id: _reviewer_agreement(cases, predicate_id)
            for predicate_id in TRACE_FAILURE_PREDICATE_IDS
        },
        "reviewer_effort": _reviewer_effort(cases),
        "threshold_stability": _threshold_stability(payload["threshold_sensitivity"], cases),
        "grouping_stability": _grouping_stability(payload["grouping_stability"], cases),
        "observations": _observations(cases),
        "limitations": [
            "Retained production traces are unavailable; available references are bounded tracked fixtures.",
            "Fixture detector and adjudication labels are contract plumbing, not an empirical accuracy or prevalence estimate.",
            "Pending, ambiguous, and unavailable labels are excluded from confusion-matrix metrics; no majority-vote imputation is performed.",
            "Reviewer agreement is pairwise label agreement, not a reliability or generalization claim; effort is fixture metadata.",
            "Planner and map identity are recorded separately; map IDs in this fixture are annotations rather than fields in the trace export.",
            "Grouping assignments are supplied under the collision-similarity schema; this contract runs no campaign or similarity rerun.",
            "Observed pattern annotations are separate from causal hypotheses, and neither is used to infer mechanism from correlation.",
        ],
    }


def build_trace_predicate_validation_report(
    evaluation_set: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Build diagnostic validation metrics from an admitted evaluation set.

    Only explicit adjudication labels of ``positive`` or ``negative`` enter
    precision/recall counts.  Pending, ambiguous, unavailable, and missing-trace
    rows are retained in denominators and exclusion/unavailable ledgers; no
    majority vote or inferred ground truth is created.

    Returns:
        Diagnostic-only report mapping.
    """
    payload = validate_trace_predicate_evaluation_set(
        evaluation_set,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
    )
    report = _build_report_from_payload(payload)
    return validate_trace_predicate_validation_report(
        report,
        evaluation_set=payload,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
    )


def format_trace_predicate_validation_markdown(
    report: Mapping[str, Any],
    *,
    evaluation_set: Mapping[str, Any],
    repo_root: str | Path | None = None,
    expected_source_commit: str,
) -> str:
    """Render a compact reviewer-facing validation report.

    Returns:
        Markdown report text.
    """
    validated = validate_trace_predicate_validation_report(
        report,
        evaluation_set=evaluation_set,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
    )
    evaluation = validated["evaluation_set"]
    lines = [
        "# Trace Predicate Validation Report",
        "",
        f"- Evaluation set: `{evaluation['evaluation_set_id']}` v{evaluation['evaluation_set_version']}",
        f"- Evidence status: `{evaluation['evidence_status']}`; retained traces: `{evaluation['retained_trace_status']}`.",
        f"- Source commit: `{evaluation['source_commit']}`; set SHA-256: `{evaluation['sha256']}`.",
        f"- Claim boundary: {validated['claim_boundary']}",
        "",
        "## Coverage",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    coverage = validated["coverage"]
    for key in (
        "case_count",
        "trace_available_count",
        "trace_unavailable_count",
        "scenario_family_count",
        "planner_count",
        "map_count",
        "seed_count",
        "reference_positive_label_count",
        "reference_negative_label_count",
        "reference_ambiguous_label_count",
        "reference_unavailable_or_pending_label_count",
    ):
        lines.append(f"| `{key}` | {coverage[key]} |")
    lines.extend(
        [
            "",
            "## Per-predicate metrics",
            "",
            "| predicate | evaluated | TP | TN | FP | FN | precision | recall | unavailable |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for predicate_id in TRACE_FAILURE_PREDICATE_IDS:
        metrics = validated["predicate_metrics"][predicate_id]
        unavailable = validated["unavailable_rates"][predicate_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{predicate_id}`",
                    str(metrics["evaluated_count"]),
                    str(metrics["true_positive"]),
                    str(metrics["true_negative"]),
                    str(metrics["false_positive"]),
                    str(metrics["false_negative"]),
                    _display_number(metrics["precision"]),
                    _display_number(metrics["recall"]),
                    _display_number(unavailable["unavailable_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reviewer agreement and effort",
            "",
            "| predicate | reviewer pairs | agreement | disagreements | unresolved |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for predicate_id in TRACE_FAILURE_PREDICATE_IDS:
        agreement = validated["reviewer_agreement"][predicate_id]
        lines.append(
            f"| `{predicate_id}` | {agreement['pair_count']} | "
            f"{_display_number(agreement['agreement_rate'])} | "
            f"{agreement['disagreement_pair_count']} | "
            f"{agreement['unresolved_disagreement_count']} |"
        )
    effort = validated["reviewer_effort"]
    lines.extend(
        [
            "",
            f"Reviewer effort: `{effort['total_minutes']:.2f}` minutes across "
            f"{effort['reviewer_count']} reviewers and {effort['adjudicator_count']} adjudicators.",
            "",
            "## Threshold and grouping stability",
            "",
            f"Threshold variants: `{validated['threshold_stability']['variant_count']}`; "
            "stability rows are label comparisons to the declared baseline.",
            f"Grouping identity features: `{', '.join(validated['grouping_stability']['identity_features'])}`; "
            f"same-group pair agreement after ablation: "
            f"`{_display_number(validated['grouping_stability']['same_group_pair_agreement'])}`; "
            f"pairwise Jaccard: `{_display_number(validated['grouping_stability']['pairwise_jaccard'])}`.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in validated["limitations"])
    return "\n".join(lines) + "\n"


def write_trace_predicate_validation_report(
    report: Mapping[str, Any],
    out_json: str | Path,
    *,
    evaluation_set: Mapping[str, Any],
    repo_root: str | Path | None = None,
    expected_source_commit: str,
    out_markdown: str | Path | None = None,
) -> None:
    """Write a validated JSON report and optional Markdown companion."""
    validated = validate_trace_predicate_validation_report(
        report,
        evaluation_set=evaluation_set,
        repo_root=repo_root,
        expected_source_commit=expected_source_commit,
    )
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_markdown is not None:
        markdown_path = Path(out_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            format_trace_predicate_validation_markdown(
                validated,
                evaluation_set=evaluation_set,
                repo_root=repo_root,
                expected_source_commit=expected_source_commit,
            ),
            encoding="utf-8",
        )


def _schema_errors(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """Return JSON Schema errors with stable JSON-pointer paths."""
    validator = Draft202012Validator(schema)
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    ]


def _value_difference_paths(
    actual: Any,
    expected: Any,
    path: str = "",
) -> list[str]:
    """Return JSON-pointer-like paths whose values differ."""
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        paths: list[str] = []
        for key in sorted(set(actual) | set(expected), key=str):
            child_path = f"{path}/{key}" if path else f"/{key}"
            if key not in actual or key not in expected:
                paths.append(child_path)
            else:
                paths.extend(_value_difference_paths(actual[key], expected[key], child_path))
        return paths
    if isinstance(actual, list) and isinstance(expected, list):
        paths = []
        for index in range(max(len(actual), len(expected))):
            child_path = f"{path}/{index}" if path else f"/{index}"
            if index >= len(actual) or index >= len(expected):
                paths.append(child_path)
            else:
                paths.extend(_value_difference_paths(actual[index], expected[index], child_path))
        return paths
    return [] if actual == expected else [path or "/"]


def _git_commit_error(repo_root: Path, source_commit: str) -> str | None:
    """Return an error when a declared source pin is not a reachable Git commit.

    Returns:
        An actionable error string, or ``None`` when the object is a commit.
    """
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", source_commit],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return f"declared source commit cannot be verified as a Git commit: {exc}"
    if result.returncode != 0 or result.stdout.strip() != b"commit":
        return "declared source commit is not a Git commit in the supplied repository root"
    return None


def _git_blob_at_commit(
    repo_root: Path,
    source_commit: str,
    uri: str,
) -> tuple[bytes | None, str | None]:
    """Read one immutable Git blob and return an error when the path is not a blob.

    Returns:
        The blob bytes and ``None``, or ``None`` and an actionable error string.
    """
    object_spec = f"{source_commit}:{uri}"
    try:
        type_result = subprocess.run(
            ["git", "cat-file", "-t", object_spec],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return None, f"Git blob binding cannot be verified: {exc}"
    if type_result.returncode != 0 or type_result.stdout.strip() != b"blob":
        return None, "resolved trace path is not a Git blob at the declared source commit"
    try:
        blob_result = subprocess.run(
            ["git", "cat-file", "blob", object_spec],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return None, f"Git blob binding cannot be read: {exc}"
    if blob_result.returncode != 0:
        return None, "Git blob at the declared source commit cannot be read"
    return blob_result.stdout, None


def _report_semantic_errors(  # noqa: C901, PLR0912, PLR0915
    report: Mapping[str, Any],
) -> list[str]:
    """Return cross-field errors for a generated report envelope.

    JSON Schema establishes the report shape; these checks protect the
    denominator and partition relationships that a shape-only schema cannot
    express. They intentionally validate mechanics, not scientific meaning.

    Returns:
        Cross-field report errors.
    """
    errors: list[str] = []
    evaluation = report["evaluation_set"]
    coverage = report["coverage"]
    case_count = coverage["case_count"]
    if evaluation["case_count"] != case_count:
        errors.append("/evaluation_set/case_count: must equal /coverage/case_count")
    if (
        evaluation["source_kind"] == "bounded_labeled_fixture"
        and evaluation["retained_trace_status"] != RETAINED_TRACE_STATUS_UNAVAILABLE
    ):
        errors.append(
            "/evaluation_set/retained_trace_status: bounded fixture reports must remain unavailable"
        )

    expected_predicates = set(TRACE_FAILURE_PREDICATE_IDS)
    section_names = ("predicate_metrics", "unavailable_rates", "reviewer_agreement")
    for section_name in section_names:
        actual_predicates = set(report[section_name])
        if actual_predicates != expected_predicates:
            errors.append(f"/{section_name}: must contain exactly the eight predicate IDs")

    for predicate_id in TRACE_FAILURE_PREDICATE_IDS:
        metric = report["predicate_metrics"][predicate_id]
        if metric["evaluated_count"] + metric["excluded_count"] != case_count:
            errors.append(
                f"/predicate_metrics/{predicate_id}: evaluated and excluded counts must cover cases"
            )
        for metric_name, numerator, denominator in (
            (
                "precision",
                metric["true_positive"],
                metric["true_positive"] + metric["false_positive"],
            ),
            (
                "recall",
                metric["true_positive"],
                metric["true_positive"] + metric["false_negative"],
            ),
        ):
            expected = _ratio(numerator, denominator)
            if not _ratio_matches(metric[metric_name], expected):
                errors.append(
                    f"/predicate_metrics/{predicate_id}/{metric_name}: ratio does not match counts"
                )

        availability = report["unavailable_rates"][predicate_id]
        if availability["total_case_count"] != case_count:
            errors.append(
                f"/unavailable_rates/{predicate_id}/total_case_count: must equal /coverage/case_count"
            )
        if availability["metric_excluded_count"] != metric["excluded_count"]:
            errors.append(
                f"/unavailable_rates/{predicate_id}/metric_excluded_count: must equal metric exclusion count"
            )
        expected_unavailable_fraction = _ratio(
            availability["unavailable_case_count"], availability["total_case_count"]
        )
        if not _ratio_matches(availability["unavailable_fraction"], expected_unavailable_fraction):
            errors.append(
                f"/unavailable_rates/{predicate_id}/unavailable_fraction: ratio does not match counts"
            )
        if metric["evaluated_count"] != sum(
            metric[name]
            for name in ("true_positive", "true_negative", "false_positive", "false_negative")
        ):
            errors.append(
                f"/predicate_metrics/{predicate_id}: confusion counts must equal evaluated count"
            )
        agreement = report["reviewer_agreement"][predicate_id]
        if (
            agreement["agreement_pair_count"] + agreement["disagreement_pair_count"]
            != agreement["pair_count"]
        ):
            errors.append(
                f"/reviewer_agreement/{predicate_id}: agreement and disagreement pairs must cover all pairs"
            )

    threshold = report["threshold_stability"]
    threshold_by_predicate = threshold["by_predicate"]
    baseline_threshold_ids: set[str] = set()
    threshold_variant_ids: set[str] | None = None
    for predicate_id in TRACE_FAILURE_PREDICATE_IDS:
        variants = threshold_by_predicate[predicate_id]["variants"]
        if len(variants) != threshold["variant_count"]:
            errors.append(
                f"/threshold_stability/by_predicate/{predicate_id}/variants: count differs from variant_count"
            )
        current_variant_ids = {row["variant_id"] for row in variants}
        baseline_threshold_ids.update(current_variant_ids)
        if threshold_variant_ids is None:
            threshold_variant_ids = current_variant_ids
        elif current_variant_ids != threshold_variant_ids:
            errors.append(
                f"/threshold_stability/by_predicate/{predicate_id}/variants: IDs differ across predicates"
            )
    if threshold["baseline_variant_id"] not in baseline_threshold_ids:
        errors.append("/threshold_stability/baseline_variant_id: baseline variant is missing")

    grouping = report["grouping_stability"]
    grouping_variants = grouping["variants"]
    variant_ids = [variant["variant_id"] for variant in grouping_variants]
    if grouping["variant_count"] != len(grouping_variants):
        errors.append("/grouping_stability/variant_count: must equal the number of variants")
    if len(set(variant_ids)) != len(variant_ids):
        errors.append("/grouping_stability/variants: variant IDs must be unique")
    if grouping["baseline_variant_id"] == grouping["ablated_variant_id"]:
        errors.append("/grouping_stability: baseline and ablated variants must differ")
    if grouping["baseline_variant_id"] not in variant_ids:
        errors.append("/grouping_stability/baseline_variant_id: variant is missing")
    if grouping["ablated_variant_id"] not in variant_ids:
        errors.append("/grouping_stability/ablated_variant_id: variant is missing")
    universes: list[set[str]] = []
    for index, variant in enumerate(grouping_variants):
        groups = variant["groups"]
        members = [record_id for group in groups for record_id in group["record_ids"]]
        universe = set(members)
        universes.append(universe)
        group_ids = [group["group_id"] for group in groups]
        if len(group_ids) != len(set(group_ids)):
            errors.append(f"/grouping_stability/variants/{index}/groups: duplicate group IDs")
        if len(members) != len(universe):
            errors.append(f"/grouping_stability/variants/{index}/groups: duplicate record IDs")
        if variant["group_count"] != len(groups):
            errors.append(
                f"/grouping_stability/variants/{index}/group_count: must equal number of groups"
            )
        if variant["group_sizes"] != [len(group["record_ids"]) for group in groups]:
            errors.append(
                f"/grouping_stability/variants/{index}/group_sizes: must match group member counts"
            )
    if universes:
        if any(universe != universes[0] for universe in universes[1:]):
            errors.append("/grouping_stability/variants: must cover one common record universe")
        if len(universes[0]) != grouping["record_count"]:
            errors.append("/grouping_stability/record_count: must match grouped record IDs")
        pair_count = grouping["record_count"] * (grouping["record_count"] - 1) // 2
        if grouping["pair_count"] != pair_count:
            errors.append("/grouping_stability/pair_count: must match record-pair count")
        if grouping["agreement_pair_count"] + grouping["changed_pair_count"] != pair_count:
            errors.append("/grouping_stability: agreement and changed pairs must cover all pairs")
    pair_count = grouping["pair_count"]
    reference_same_group_pairs = grouping["reference_same_group_pair_count"]
    comparison_same_group_pairs = grouping["comparison_same_group_pair_count"]
    reference_only_pairs = grouping["reference_only_pair_count"]
    comparison_only_pairs = grouping["comparison_only_pair_count"]
    if any(
        count > pair_count
        for count in (
            grouping["agreement_pair_count"],
            grouping["changed_pair_count"],
            reference_same_group_pairs,
            comparison_same_group_pairs,
            reference_only_pairs,
            comparison_only_pairs,
        )
    ):
        errors.append("/grouping_stability: pair counts cannot exceed pair_count")
    if reference_only_pairs + comparison_only_pairs != grouping["changed_pair_count"]:
        errors.append(
            "/grouping_stability: reference/comparison-only pairs must cover changed pairs"
        )
    reference_intersection = reference_same_group_pairs - reference_only_pairs
    comparison_intersection = comparison_same_group_pairs - comparison_only_pairs
    if reference_intersection != comparison_intersection or reference_intersection < 0:
        errors.append("/grouping_stability: same-group and one-sided pair counts are inconsistent")
    expected_agreement = _ratio(grouping["agreement_pair_count"], pair_count)
    if not _ratio_matches(grouping["same_group_pair_agreement"], expected_agreement):
        errors.append("/grouping_stability/same_group_pair_agreement: ratio does not match counts")
    union_pair_count = reference_intersection + reference_only_pairs + comparison_only_pairs
    expected_jaccard = _ratio(reference_intersection, union_pair_count)
    if expected_jaccard is None:
        expected_jaccard = 1.0
    if not _ratio_matches(grouping["pairwise_jaccard"], expected_jaccard):
        errors.append("/grouping_stability/pairwise_jaccard: ratio does not match counts")
    observations = report["observations"]["causal_hypotheses"]
    if observations["used_for_metrics"] is not False:
        errors.append("/observations/causal_hypotheses/used_for_metrics: must remain false")
    return errors


def _semantic_errors(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source_commit: str | None,
) -> list[str]:
    """Return cross-field and cross-file contract errors."""
    errors: list[str] = []
    provenance = payload["provenance"]
    source_kind = provenance["source_kind"]
    retained_status = payload["retained_trace_status"]
    source_commit = provenance["source_commit"]
    if expected_source_commit is None:
        errors.append("/provenance/source_commit: expected source commit pin is required")
    elif not _is_sha(expected_source_commit, _SOURCE_COMMIT_LENGTH):
        errors.append("/provenance/source_commit: expected source commit is not a 40-digit SHA")
    elif expected_source_commit != source_commit:
        errors.append(
            "/provenance/source_commit: evaluation set is bound to a different source commit"
        )
    if _is_sha(source_commit, _SOURCE_COMMIT_LENGTH):
        git_commit_error = _git_commit_error(repo_root, source_commit)
        if git_commit_error is not None:
            errors.append(f"/provenance/source_commit: {git_commit_error}")
    if (
        source_kind == "bounded_labeled_fixture"
        and retained_status != RETAINED_TRACE_STATUS_UNAVAILABLE
    ):
        errors.append(
            "/retained_trace_status: bounded_labeled_fixture must remain unavailable for retained-trace claims"
        )
    if (
        source_kind == "bounded_labeled_fixture"
        and provenance["label_origin"] != "hand_authored_fixture"
    ):
        errors.append(
            "/provenance/label_origin: bounded fixtures must declare hand_authored_fixture"
        )
    if source_kind == "retained_trace_corpus":
        if payload["evidence_status"] != EVIDENCE_STATUS_DIAGNOSTIC_ONLY:
            errors.append(
                "/evidence_status: retained corpus validation must remain diagnostic-only"
            )

    cases = payload["cases"]
    case_ids = [case["case_id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        errors.append("/cases: case_id values must be unique")
    case_id_set = set(case_ids)
    seen_trace_uris: set[str] = set()
    minimum_reviewers = payload["review_protocol"]["minimum_reviewers"]
    for index, case in enumerate(cases):
        errors.extend(
            _case_semantic_errors(
                case,
                index=index,
                minimum_reviewers=minimum_reviewers,
                source_kind=source_kind,
                source_commit=source_commit,
                repo_root=repo_root,
                seen_trace_uris=seen_trace_uris,
            )
        )

    errors.extend(_threshold_semantic_errors(payload["threshold_sensitivity"], cases))
    errors.extend(_grouping_semantic_errors(payload["grouping_stability"], case_id_set))
    return errors


def _case_semantic_errors(  # noqa: C901, PLR0912
    case: Mapping[str, Any],
    *,
    index: int,
    minimum_reviewers: int,
    source_kind: str,
    source_commit: str,
    repo_root: Path,
    seen_trace_uris: set[str],
) -> list[str]:
    """Validate one case's trace, detector, and review bindings.

    Returns:
        Semantic validation errors for the case.
    """
    prefix = f"/cases/{index}"
    errors: list[str] = []
    trace_ref = case["trace_ref"]
    trace_available = trace_ref["status"] == "available"
    if trace_available:
        uri = trace_ref["uri"]
        if uri in seen_trace_uris:
            errors.append(f"{prefix}/trace_ref/uri: trace URI is reused across cases")
        seen_trace_uris.add(uri)
        errors.extend(
            _available_trace_errors(
                case,
                trace_ref,
                prefix=prefix,
                source_kind=source_kind,
                source_commit=source_commit,
                repo_root=repo_root,
            )
        )
    detector = case["detector"]
    if detector["source"] != TRACE_FAILURE_PREDICATE_SOURCE:
        errors.append(f"{prefix}/detector/source: detector source is not the canonical rule source")
    if detector["schema_version"] != TRACE_FAILURE_PREDICATE_SCHEMA_VERSION:
        errors.append(
            f"{prefix}/detector/schema_version: detector schema is not the canonical version"
        )
    labels = detector["labels"]
    unavailable_reasons = detector.get("unavailable_reasons", {})
    for predicate_id in TRACE_FAILURE_PREDICATE_IDS:
        label = labels[predicate_id]
        if label == LABEL_UNAVAILABLE and predicate_id not in unavailable_reasons:
            errors.append(f"{prefix}/detector/unavailable_reasons/{predicate_id}: reason required")
        if label != LABEL_UNAVAILABLE and predicate_id in unavailable_reasons:
            errors.append(
                f"{prefix}/detector/unavailable_reasons/{predicate_id}: reason only allowed for unavailable label"
            )
    if not trace_available:
        if any(label != LABEL_UNAVAILABLE for label in labels.values()):
            errors.append(
                f"{prefix}/detector/labels: unavailable trace requires unavailable detector labels"
            )
        if case["planner_id"] != "unavailable" or case["map_id"] != "unavailable":
            errors.append(
                f"{prefix}: unavailable trace requires unavailable planner and map identities"
            )
        if case["seed"] is not None:
            errors.append(f"{prefix}/seed: unavailable trace requires a null seed")

    review = case["review"]
    reviewers = review["reviewers"]
    if len(reviewers) < minimum_reviewers:
        errors.append(f"{prefix}/review/reviewers: requires at least {minimum_reviewers} reviewers")
    reviewer_ids = [reviewer["reviewer_id"] for reviewer in reviewers]
    if len(set(reviewer_ids)) != len(reviewer_ids):
        errors.append(f"{prefix}/review/reviewers: reviewer_id values must be unique per case")
    for reviewer_index, reviewer in enumerate(reviewers):
        errors.extend(
            _finite_effort_error(
                reviewer.get("effort_minutes"),
                f"{prefix}/review/reviewers/{reviewer_index}/effort_minutes",
            )
        )
        if not trace_available and any(
            label != LABEL_UNAVAILABLE for label in reviewer["labels"].values()
        ):
            errors.append(
                f"{prefix}/review/reviewers/{reviewer_index}/labels: unavailable trace requires unavailable review labels"
            )
    adjudication = review["adjudication"]
    if not trace_available and adjudication["status"] != "pending":
        errors.append(
            f"{prefix}/review/adjudication/status: unavailable trace requires pending adjudication"
        )
    if adjudication["status"] == "adjudicated":
        if adjudication["reviewer_id"] in reviewer_ids:
            errors.append(
                f"{prefix}/review/adjudication/reviewer_id: adjudicator must be distinct from reviewers"
            )
        errors.extend(
            _finite_effort_error(
                adjudication["effort_minutes"],
                f"{prefix}/review/adjudication/effort_minutes",
            )
        )
    if (
        source_kind == "retained_trace_corpus"
        and detector["label_origin"] != "computed_trace_output"
    ):
        errors.append(
            f"{prefix}/detector/label_origin: retained corpus rows require computed trace output labels"
        )
    return errors


def _available_trace_errors(  # noqa: C901, PLR0912
    case: Mapping[str, Any],
    trace_ref: Mapping[str, Any],
    *,
    prefix: str,
    source_kind: str,
    source_commit: str,
    repo_root: Path,
) -> list[str]:
    """Validate a trace URI, digest, strict schema, and source identity.

    Returns:
        Semantic validation errors for the trace binding.
    """
    errors: list[str] = []
    raw_uri = trace_ref["uri"]
    uri_path = Path(raw_uri)
    if uri_path.is_absolute() or "\x00" in raw_uri:
        return [f"{prefix}/trace_ref/uri: URI must be repository-relative"]
    if any(part in {"..", "."} for part in uri_path.parts):
        errors.append(f"{prefix}/trace_ref/uri: path traversal or dot segments are rejected")
    if any(part in {"output", "results"} for part in uri_path.parts):
        errors.append(f"{prefix}/trace_ref/uri: generated output paths are rejected")
    try:
        resolved = (repo_root / uri_path).resolve()
        resolved.relative_to(repo_root)
    except ValueError:
        errors.append(f"{prefix}/trace_ref/uri: resolved path escapes the repository root")
        return errors
    if errors:
        return errors
    if not resolved.is_file():
        return [f"{prefix}/trace_ref/uri: referenced trace file does not exist: {raw_uri}"]
    try:
        resolved_bytes = resolved.read_bytes()
    except OSError as exc:
        return [f"{prefix}/trace_ref/uri: referenced trace file cannot be read: {exc}"]
    actual_sha = hashlib.sha256(resolved_bytes).hexdigest()
    if actual_sha != trace_ref["sha256"]:
        errors.append(f"{prefix}/trace_ref/sha256: referenced trace digest does not match bytes")
    if trace_ref["source_commit"] != source_commit:
        errors.append(
            f"{prefix}/trace_ref/source_commit: source commit differs from set provenance"
        )
    git_blob, git_blob_error = _git_blob_at_commit(repo_root, source_commit, raw_uri)
    if git_blob_error is not None:
        errors.append(f"{prefix}/trace_ref: {git_blob_error}")
    elif git_blob != resolved_bytes:
        errors.append(
            f"{prefix}/trace_ref: resolved bytes do not match the Git blob at the declared source commit"
        )
    try:
        raw_trace = _strict_json_loads(resolved_bytes.decode("utf-8"))
        if not isinstance(raw_trace, Mapping):
            raise SimulationTraceExportValidationError(
                ["expected a mapping payload"], source=resolved
            )
        trace = simulation_trace_export_from_dict(raw_trace, source=resolved)
    except (OSError, UnicodeDecodeError, ValueError, SimulationTraceExportValidationError) as exc:
        errors.append(f"{prefix}/trace_ref: referenced trace is not a valid strict export: {exc}")
        return errors
    if trace.schema_version != SIMULATION_TRACE_EXPORT_SCHEMA_VERSION:
        errors.append(f"{prefix}/trace_ref/schema_version: unsupported trace schema")
    if trace.trace_id != trace_ref["trace_id"]:
        errors.append(f"{prefix}/trace_ref/trace_id: trace ID does not match referenced file")
    if trace.source.scenario_id != case["scenario_id"]:
        errors.append(f"{prefix}/scenario_id: source scenario ID does not match trace")
    if trace.source.planner_id != case["planner_id"]:
        errors.append(f"{prefix}/planner_id: source planner ID does not match trace")
    if case["seed"] != trace.source.seed:
        errors.append(f"{prefix}/seed: source seed does not match trace")
    if case["planner_identity_status"] != "source_bound":
        errors.append(
            f"{prefix}/planner_identity_status: available trace planner identity must be source_bound"
        )
    if (
        source_kind == "bounded_labeled_fixture"
        and case["map_identity_status"] != "fixture_annotation"
    ):
        errors.append(
            f"{prefix}/map_identity_status: bounded fixture maps must remain fixture_annotation"
        )
    elif source_kind == "retained_trace_corpus" and case["map_identity_status"] == "unavailable":
        errors.append(
            f"{prefix}/map_identity_status: retained corpus rows require map identity status"
        )
    return errors


def _threshold_semantic_errors(
    threshold_spec: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate threshold IDs, complete coverage, and availability propagation.

    Returns:
        Semantic validation errors for threshold variants.
    """
    errors: list[str] = []
    case_by_id = {case["case_id"]: case for case in cases}
    case_ids = set(case_by_id)
    variants = threshold_spec["variants"]
    variant_ids = [variant["variant_id"] for variant in variants]
    if len(set(variant_ids)) != len(variant_ids):
        errors.append("/threshold_sensitivity/variants: variant_id values must be unique")
    if threshold_spec["baseline_variant_id"] not in set(variant_ids):
        errors.append("/threshold_sensitivity/baseline_variant_id: baseline variant is missing")
    for index, variant in enumerate(variants):
        labels_by_case = variant["labels_by_case"]
        if set(labels_by_case) != case_ids:
            errors.append(
                f"/threshold_sensitivity/variants/{index}/labels_by_case: must cover each case exactly once"
            )
        for case_id, labels in labels_by_case.items():
            case = case_by_id.get(case_id)
            if case is None:
                continue
            trace_unavailable = case["trace_ref"]["status"] != "available"
            for predicate_id, label in labels.items():
                detector_unavailable = case["detector"]["labels"][predicate_id] == LABEL_UNAVAILABLE
                if (trace_unavailable or detector_unavailable) and label != LABEL_UNAVAILABLE:
                    errors.append(
                        f"/threshold_sensitivity/variants/{index}/labels_by_case/{case_id}/{predicate_id}: "
                        "unavailable trace or detector requires an unavailable threshold label"
                    )
    return errors


def _grouping_semantic_errors(  # noqa: C901
    grouping_spec: Mapping[str, Any], case_ids: set[str]
) -> list[str]:
    """Validate grouping partitions and identity-ablation declarations.

    Returns:
        Semantic validation errors for grouping variants.
    """
    errors: list[str] = []
    if grouping_spec["source_schema_version"] != COLLISION_SCENARIO_SIMILARITY_SCHEMA_VERSION:
        errors.append(
            "/grouping_stability/source_schema_version: unsupported collision similarity schema"
        )
    record_ids = set(grouping_spec["record_ids"])
    if record_ids != case_ids:
        errors.append("/grouping_stability/record_ids: grouping records must equal case IDs")
    variants = grouping_spec["variants"]
    variant_ids = [variant["variant_id"] for variant in variants]
    variant_id_set = set(variant_ids)
    if len(variant_id_set) != len(variant_ids):
        errors.append("/grouping_stability/variants: variant_id values must be unique")
    baseline_id = grouping_spec["baseline_variant_id"]
    ablated_id = grouping_spec["ablated_variant_id"]
    if baseline_id == ablated_id:
        errors.append("/grouping_stability: baseline and ablated variants must differ")
    for field, variant_id in (
        ("baseline_variant_id", baseline_id),
        ("ablated_variant_id", ablated_id),
    ):
        if variant_id not in variant_id_set:
            errors.append(f"/grouping_stability/{field}: variant is missing")
    variants_by_id = {variant["variant_id"]: variant for variant in variants}
    identity_features = set(grouping_spec["identity_features"])
    if baseline_id in variants_by_id and not identity_features <= set(
        variants_by_id[baseline_id]["included_features"]
    ):
        errors.append(
            "/grouping_stability/baseline_variant_id: baseline must include declared identity features"
        )
    if ablated_id in variants_by_id and not identity_features <= set(
        variants_by_id[ablated_id]["excluded_features"]
    ):
        errors.append(
            "/grouping_stability/ablated_variant_id: ablated variant must exclude declared identity features"
        )
    for index, variant in enumerate(variants):
        included_features = set(variant["included_features"])
        excluded_features = set(variant["excluded_features"])
        unknown_features = (included_features | excluded_features) - _GROUPING_FEATURE_VOCABULARY
        if unknown_features:
            errors.append(
                f"/grouping_stability/variants/{index}: feature names are outside the closed vocabulary"
            )
        overlap = included_features & excluded_features
        if overlap:
            errors.append(
                f"/grouping_stability/variants/{index}: included and excluded features must be disjoint"
            )
        groups = variant["groups"]
        group_ids = [group["group_id"] for group in groups]
        if len(set(group_ids)) != len(group_ids):
            errors.append(f"/grouping_stability/variants/{index}/groups: group IDs must be unique")
        members = [record_id for group in groups for record_id in group["record_ids"]]
        if set(members) != record_ids or len(members) != len(set(members)):
            errors.append(
                f"/grouping_stability/variants/{index}/groups: groups must partition record_ids exactly once"
            )
    return errors


def _finite_effort_error(
    value: Any,
    path: str,
    *,
    optional: bool = False,
) -> list[str]:
    """Return an error for non-finite effort metadata."""
    if value is None and optional:
        return []
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return [f"{path}: effort_minutes must be finite"]
    return []


def _coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize case and label coverage without producing a scientific claim.

    Returns:
        Coverage counts and identity lists.
    """
    detector_counts: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    for case in cases:
        detector_counts.update(case["detector"]["labels"].values())
        adjudication = case["review"]["adjudication"]
        if adjudication["status"] == "pending":
            reference_counts["unavailable_or_pending"] += len(TRACE_FAILURE_PREDICATE_IDS)
        else:
            reference_counts.update(adjudication["labels"].values())
    planner_ids = sorted(
        {case["planner_id"] for case in cases if case["planner_id"] != "unavailable"}
    )
    map_ids = sorted({case["map_id"] for case in cases if case["map_id"] != "unavailable"})
    seeds = {case["seed"] for case in cases if case["seed"] is not None}
    return {
        "case_count": len(cases),
        "trace_available_count": sum(case["trace_ref"]["status"] == "available" for case in cases),
        "trace_unavailable_count": sum(
            case["trace_ref"]["status"] == "unavailable" for case in cases
        ),
        "scenario_family_count": len({case["scenario_family"] for case in cases}),
        "planner_count": len(planner_ids),
        "planner_ids": planner_ids,
        "map_count": len(map_ids),
        "map_ids": map_ids,
        "seed_count": len(seeds),
        "detector_positive_label_count": detector_counts[LABEL_POSITIVE],
        "detector_negative_label_count": detector_counts[LABEL_NEGATIVE],
        "detector_ambiguous_label_count": detector_counts[LABEL_AMBIGUOUS],
        "detector_unavailable_label_count": detector_counts[LABEL_UNAVAILABLE],
        "reference_positive_label_count": reference_counts[LABEL_POSITIVE],
        "reference_negative_label_count": reference_counts[LABEL_NEGATIVE],
        "reference_ambiguous_label_count": reference_counts[LABEL_AMBIGUOUS],
        "reference_unavailable_label_count": reference_counts[LABEL_UNAVAILABLE],
        "reference_unavailable_or_pending_label_count": reference_counts["unavailable_or_pending"]
        + reference_counts[LABEL_UNAVAILABLE],
    }


def _predicate_metrics(cases: Sequence[Mapping[str, Any]], predicate_id: str) -> dict[str, Any]:
    """Compute an explicit-label confusion matrix for one predicate.

    Returns:
        Confusion counts, ratios, and metric-exclusion reasons.
    """
    counts: Counter[str] = Counter()
    excluded_reasons: Counter[str] = Counter()
    excluded_count = 0
    for case in cases:
        detector_label = case["detector"]["labels"][predicate_id]
        adjudication = case["review"]["adjudication"]
        reference_label = (
            None if adjudication["status"] == "pending" else adjudication["labels"][predicate_id]
        )
        reasons: set[str] = set()
        if case["trace_ref"]["status"] != "available":
            reasons.add("trace_unavailable")
        if detector_label not in {LABEL_POSITIVE, LABEL_NEGATIVE}:
            reasons.add(f"detector_{detector_label}")
        if reference_label is None:
            reasons.add("reference_pending")
        elif reference_label not in {LABEL_POSITIVE, LABEL_NEGATIVE}:
            reasons.add(f"reference_{reference_label}")
        if reasons:
            excluded_count += 1
            excluded_reasons.update(reasons)
            continue
        counts[
            {
                (LABEL_POSITIVE, LABEL_POSITIVE): "true_positive",
                (LABEL_NEGATIVE, LABEL_NEGATIVE): "true_negative",
                (LABEL_POSITIVE, LABEL_NEGATIVE): "false_positive",
                (LABEL_NEGATIVE, LABEL_POSITIVE): "false_negative",
            }[(detector_label, reference_label)]
        ] += 1
    true_positive = counts["true_positive"]
    true_negative = counts["true_negative"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    return {
        "evaluated_count": sum(counts.values()),
        "reference_positive_count": sum(
            1
            for case in cases
            if case["review"]["adjudication"]["status"] == "adjudicated"
            and case["review"]["adjudication"]["labels"][predicate_id] == LABEL_POSITIVE
        ),
        "reference_negative_count": sum(
            1
            for case in cases
            if case["review"]["adjudication"]["status"] == "adjudicated"
            and case["review"]["adjudication"]["labels"][predicate_id] == LABEL_NEGATIVE
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "excluded_count": excluded_count,
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
    }


def _unavailable_rate(cases: Sequence[Mapping[str, Any]], predicate_id: str) -> dict[str, Any]:
    """Report detector and review availability for one predicate.

    Returns:
        Availability and metric-exclusion counts.
    """
    trace_unavailable_count = sum(case["trace_ref"]["status"] != "available" for case in cases)
    detector_unavailable_count = sum(
        case["detector"]["labels"][predicate_id] == LABEL_UNAVAILABLE for case in cases
    )
    unavailable_case_count = sum(
        case["trace_ref"]["status"] != "available"
        or case["detector"]["labels"][predicate_id] == LABEL_UNAVAILABLE
        for case in cases
    )
    review_pending_or_unavailable_count = sum(
        case["review"]["adjudication"]["status"] == "pending"
        or (
            case["review"]["adjudication"]["status"] == "adjudicated"
            and case["review"]["adjudication"]["labels"][predicate_id] == LABEL_UNAVAILABLE
        )
        for case in cases
    )
    review_ambiguous_count = sum(
        case["review"]["adjudication"]["status"] == "adjudicated"
        and case["review"]["adjudication"]["labels"][predicate_id] == LABEL_AMBIGUOUS
        for case in cases
    )
    total = len(cases)
    return {
        "denominator": "all_evaluation_cases",
        "total_case_count": total,
        "unavailable_case_count": unavailable_case_count,
        "unavailable_fraction": _ratio(unavailable_case_count, total),
        "trace_unavailable_count": trace_unavailable_count,
        "detector_unavailable_count": detector_unavailable_count,
        "review_unavailable_or_pending_count": review_pending_or_unavailable_count,
        "review_ambiguous_count": review_ambiguous_count,
        "metric_excluded_count": _predicate_metrics(cases, predicate_id)["excluded_count"],
    }


def _reviewer_agreement(cases: Sequence[Mapping[str, Any]], predicate_id: str) -> dict[str, Any]:
    """Compute exact pairwise reviewer agreement and preserve disagreements.

    Returns:
        Pairwise agreement counts and disagreement records.
    """
    reviewer_ids: set[str] = set()
    pair_count = 0
    agreement_pair_count = 0
    disagreement_pair_count = 0
    disagreement_cases: list[dict[str, Any]] = []
    unresolved_disagreement_count = 0
    for case in cases:
        reviewers = case["review"]["reviewers"]
        reviewer_ids.update(reviewer["reviewer_id"] for reviewer in reviewers)
        case_disagreed = False
        for left, right in combinations(reviewers, 2):
            pair_count += 1
            left_label = left["labels"][predicate_id]
            right_label = right["labels"][predicate_id]
            if left_label == right_label:
                agreement_pair_count += 1
            else:
                disagreement_pair_count += 1
                case_disagreed = True
        if case_disagreed:
            adjudication = case["review"]["adjudication"]
            disagreement_cases.append(
                {
                    "case_id": case["case_id"],
                    "labels_by_reviewer": {
                        reviewer["reviewer_id"]: reviewer["labels"][predicate_id]
                        for reviewer in reviewers
                    },
                    "adjudication_status": adjudication["status"],
                }
            )
            if adjudication["status"] == "pending" or (
                adjudication["status"] == "adjudicated"
                and adjudication["labels"][predicate_id] in {LABEL_AMBIGUOUS, LABEL_UNAVAILABLE}
            ):
                unresolved_disagreement_count += 1
    return {
        "reviewer_count": len(reviewer_ids),
        "case_count": len(cases),
        "pair_count": pair_count,
        "agreement_pair_count": agreement_pair_count,
        "disagreement_pair_count": disagreement_pair_count,
        "agreement_rate": _ratio(agreement_pair_count, pair_count),
        "disagreement_cases": disagreement_cases,
        "unresolved_disagreement_count": unresolved_disagreement_count,
    }


def _reviewer_effort(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate declared reviewer and adjudicator effort without weighting metrics.

    Returns:
        Reviewer and adjudicator effort totals by identity.
    """
    reviewer_minutes: defaultdict[str, float] = defaultdict(float)
    adjudicator_minutes: defaultdict[str, float] = defaultdict(float)
    for case in cases:
        for reviewer in case["review"]["reviewers"]:
            reviewer_minutes[reviewer["reviewer_id"]] += float(reviewer["effort_minutes"])
        adjudication = case["review"]["adjudication"]
        if adjudication["status"] == "adjudicated" and "effort_minutes" in adjudication:
            adjudicator_minutes[adjudication["reviewer_id"]] += float(
                adjudication["effort_minutes"]
            )
    reviewer_total = sum(reviewer_minutes.values())
    adjudicator_total = sum(adjudicator_minutes.values())
    return {
        "unit": "minutes",
        "total_minutes": round(reviewer_total + adjudicator_total, 6),
        "reviewer_minutes": {
            reviewer_id: round(minutes, 6)
            for reviewer_id, minutes in sorted(reviewer_minutes.items())
        },
        "adjudicator_minutes": {
            reviewer_id: round(minutes, 6)
            for reviewer_id, minutes in sorted(adjudicator_minutes.items())
        },
        "reviewer_count": len(reviewer_minutes),
        "adjudicator_count": len(adjudicator_minutes),
    }


def _threshold_stability(
    threshold_spec: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare every threshold label variant to its declared baseline.

    Returns:
        Per-predicate variant counts and stability ratios.
    """
    variants = threshold_spec["variants"]
    by_id = {variant["variant_id"]: variant for variant in variants}
    baseline_id = threshold_spec["baseline_variant_id"]
    baseline = by_id[baseline_id]["labels_by_case"]
    by_predicate: dict[str, Any] = {}
    case_ids = [case["case_id"] for case in cases]
    for predicate_id in TRACE_FAILURE_PREDICATE_IDS:
        baseline_positive = {
            case_id for case_id in case_ids if baseline[case_id][predicate_id] == LABEL_POSITIVE
        }
        rows: list[dict[str, Any]] = []
        for variant in variants:
            labels_by_case = variant["labels_by_case"]
            changed_count = sum(
                labels_by_case[case_id][predicate_id] != baseline[case_id][predicate_id]
                for case_id in case_ids
            )
            positive_set = {
                case_id
                for case_id in case_ids
                if labels_by_case[case_id][predicate_id] == LABEL_POSITIVE
            }
            rows.append(
                {
                    "variant_id": variant["variant_id"],
                    "positive_count": len(positive_set),
                    "negative_count": sum(
                        labels_by_case[case_id][predicate_id] == LABEL_NEGATIVE
                        for case_id in case_ids
                    ),
                    "ambiguous_count": sum(
                        labels_by_case[case_id][predicate_id] == LABEL_AMBIGUOUS
                        for case_id in case_ids
                    ),
                    "unavailable_count": sum(
                        labels_by_case[case_id][predicate_id] == LABEL_UNAVAILABLE
                        for case_id in case_ids
                    ),
                    "changed_case_count": changed_count,
                    "label_stability_fraction": _ratio(
                        len(case_ids) - changed_count, len(case_ids)
                    ),
                    "positive_set_jaccard_to_baseline": _jaccard(baseline_positive, positive_set),
                }
            )
        by_predicate[predicate_id] = {"variants": rows}
    return {
        "baseline_variant_id": baseline_id,
        "variant_count": len(variants),
        "by_predicate": by_predicate,
    }


def _grouping_stability(
    grouping_spec: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare full-identity and identity-ablated grouping assignments.

    Returns:
        Pairwise grouping stability and variant metadata.
    """
    variants = grouping_spec["variants"]
    by_id = {variant["variant_id"]: variant for variant in variants}
    baseline_id = grouping_spec["baseline_variant_id"]
    ablated_id = grouping_spec["ablated_variant_id"]
    comparison = compare_similarity_groupings(
        by_id[baseline_id]["groups"],
        by_id[ablated_id]["groups"],
        record_ids=grouping_spec["record_ids"],
    )
    identity_features = grouping_spec["identity_features"]
    identity_coverage = {
        feature: _identity_feature_coverage(cases, feature) for feature in identity_features
    }
    return {
        "status": comparison["status"],
        "source_schema_version": grouping_spec["source_schema_version"],
        "assignment_origin": grouping_spec["assignment_origin"],
        "identity_features": identity_features,
        "identity_feature_coverage": identity_coverage,
        "baseline_variant_id": baseline_id,
        "ablated_variant_id": ablated_id,
        "variant_count": len(variants),
        "record_count": comparison["record_count"],
        "pair_count": comparison["pair_count"],
        "baseline_group_count": len(by_id[baseline_id]["groups"]),
        "ablated_group_count": len(by_id[ablated_id]["groups"]),
        "reference_same_group_pair_count": comparison["reference_same_group_pair_count"],
        "comparison_same_group_pair_count": comparison["comparison_same_group_pair_count"],
        "agreement_pair_count": comparison["agreement_pair_count"],
        "changed_pair_count": comparison["disagreement_pair_count"],
        "same_group_pair_agreement": comparison["same_group_pair_agreement"],
        "pairwise_jaccard": comparison["pairwise_jaccard"],
        "reference_only_pair_count": comparison["reference_only_pair_count"],
        "comparison_only_pair_count": comparison["comparison_only_pair_count"],
        "variants": [
            {
                "variant_id": variant["variant_id"],
                "included_features": variant["included_features"],
                "excluded_features": variant["excluded_features"],
                "group_count": len(variant["groups"]),
                "group_sizes": [len(group["record_ids"]) for group in variant["groups"]],
                "groups": variant["groups"],
            }
            for variant in variants
        ],
        "interpretation": (
            "Pairwise changes describe grouping sensitivity in the admitted assignments; "
            "they do not validate a failure family or a causal mechanism."
        ),
    }


def _identity_feature_coverage(cases: Sequence[Mapping[str, Any]], feature: str) -> dict[str, int]:
    """Count declared source-bound, fixture, and unavailable identity statuses.

    Returns:
        Counts by identity provenance status.
    """
    status_field = f"{feature.replace('_id', '')}_identity_status"
    counts = Counter(case[status_field] for case in cases)
    return {
        "source_bound_count": counts["source_bound"],
        "fixture_annotation_count": counts["fixture_annotation"],
        "unavailable_count": counts["unavailable"],
    }


def _observations(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep observed-pattern and causal-hypothesis annotations separate.

    Returns:
        Pattern inventory and non-evaluated hypothesis metadata.
    """
    patterns = sorted(
        {pattern for case in cases for pattern in case["observations"]["observed_patterns"]}
    )
    hypothesis_count = sum(len(case["observations"]["causal_hypotheses"]) for case in cases)
    return {
        "observed_pattern_annotation_count": sum(
            len(case["observations"]["observed_patterns"]) for case in cases
        ),
        "unique_observed_patterns": patterns,
        "causal_hypotheses": {
            "status": "not_evaluated",
            "annotation_count": hypothesis_count,
            "used_for_metrics": False,
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return a ratio or null when its denominator is unavailable."""
    return numerator / denominator if denominator else None


def _ratio_matches(actual: float | None, expected: float | None) -> bool:
    """Check a nullable ratio against its count-derived value.

    Returns:
        Whether the actual and expected values match, including null denominators.
    """
    if expected is None:
        return actual is None
    return actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _jaccard(left: set[str], right: set[str]) -> float:
    """Return set Jaccard, defining two empty sets as stable."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _is_sha(value: Any, length: int) -> bool:
    """Check a lower-case hexadecimal digest of the expected length.

    Returns:
        Whether the value is a lower-case hexadecimal digest.
    """
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _display_number(value: float | None) -> str:
    """Render null or a compact decimal for Markdown.

    Returns:
        Display string for a number or unavailable value.
    """
    return "not available" if value is None else f"{value:.3f}"


__all__ = [
    "EVIDENCE_STATUS_DIAGNOSTIC_ONLY",
    "LABEL_AMBIGUOUS",
    "LABEL_NEGATIVE",
    "LABEL_POSITIVE",
    "LABEL_UNAVAILABLE",
    "RETAINED_TRACE_STATUS_UNAVAILABLE",
    "TRACE_PREDICATE_VALIDATION_REPORT_SCHEMA_VERSION",
    "TRACE_PREDICATE_VALIDATION_SCHEMA_VERSION",
    "TracePredicateValidationError",
    "build_trace_predicate_validation_report",
    "canonical_trace_predicate_validation_sha256",
    "format_trace_predicate_validation_markdown",
    "load_trace_predicate_validation_report_schema",
    "load_trace_predicate_validation_schema",
    "load_trace_predicate_validation_set",
    "validate_trace_predicate_evaluation_set",
    "validate_trace_predicate_validation_report",
    "write_trace_predicate_validation_report",
]
