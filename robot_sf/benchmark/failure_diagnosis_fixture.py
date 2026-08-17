"""Immutable source-fixture admission for deterministic diagnosis evaluation.

Issue #7197 adds the source-side contract around the existing deterministic
``failure_diagnosis.v1`` adapter and quality evaluator.  The manifest binds each
case to a source-trace pointer, a canonical source digest, review metadata, and
explicit training/prompt-development exclusion metadata.  Source records are
checked for reference-label leakage before the adapter runs.

This module proves metric and provenance integrity only.  It does not create
reference labels, run simulation, or establish diagnostic accuracy.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
)
from robot_sf.benchmark.failure_diagnosis import (
    FailureDiagnosisError,
    diagnose_from_trace_failure_predicate,
    evaluate_failure_diagnosis_quality,
    validate_failure_diagnosis_reference_fixture,
)
from robot_sf.benchmark.failure_diagnosis_comparison import (
    validate_fixture_review_admission,
)

FAILURE_DIAGNOSIS_FIXTURE_MANIFEST_SCHEMA_VERSION = "failure_diagnosis_fixture_manifest.v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ADJUDICATED_STATUSES = frozenset({"adjudicated", "adjudicated_by_reviewer"})
_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "adjudication",
        "gold",
        "ground_truth",
        "label",
        "labels",
        "reference",
        "reference_label",
        "reference_labels",
        "review",
        "review_marker",
        "target_label",
    }
)


class FailureDiagnosisFixtureManifestError(FailureDiagnosisError):
    """Raised when an immutable source-fixture manifest is not admissible."""


class FixtureLeakageError(FailureDiagnosisFixtureManifestError):
    """Raised when source records carry reference-label or review metadata."""


def canonical_source_sha256(source_predicate: Mapping[str, Any]) -> str:
    """Return the digest bound to one canonical source-predicate mapping.

    The digest covers the source predicate payload, not the manifest entry or
    the reference labels.  JSON is sorted and compact so the value is stable
    across Python processes and platforms.
    """
    if not isinstance(source_predicate, Mapping):
        raise FailureDiagnosisFixtureManifestError("source predicate must be a mapping")
    try:
        encoded = json.dumps(
            dict(source_predicate),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FailureDiagnosisFixtureManifestError(
            "source predicate is not canonical JSON serializable"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FailureDiagnosisFixtureManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise FailureDiagnosisFixtureManifestError(f"{field} must be a positive integer")
    return value


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureDiagnosisFixtureManifestError(f"{field} must be a mapping")
    return value


def _validate_review_metadata(review: Mapping[str, Any], field: str) -> dict[str, Any]:
    status = _require_text(review.get("status"), f"{field}.status")
    if status != "reviewed":
        raise FailureDiagnosisFixtureManifestError(
            f"{field}.status must be 'reviewed', got {status!r}"
        )
    reviewer = _require_text(review.get("reviewer"), f"{field}.reviewer")
    adjudication_status = _require_text(
        review.get("adjudication_status"), f"{field}.adjudication_status"
    )
    if adjudication_status not in _ADJUDICATED_STATUSES:
        raise FailureDiagnosisFixtureManifestError(
            f"{field}.adjudication_status must be adjudicated, got {adjudication_status!r}"
        )
    if review.get("independent_of_automated_diagnosis") is not True:
        raise FailureDiagnosisFixtureManifestError(
            f"{field}.independent_of_automated_diagnosis must be true"
        )
    return {
        "status": status,
        "reviewer": reviewer,
        "adjudication_status": adjudication_status,
        "independent_of_automated_diagnosis": True,
    }


def _validate_provenance_metadata(provenance: Mapping[str, Any], field: str) -> dict[str, Any]:
    status = _require_text(provenance.get("status"), f"{field}.status")
    if status not in {"complete", "verified"}:
        raise FailureDiagnosisFixtureManifestError(
            f"{field}.status must be complete or verified, got {status!r}"
        )
    for exclusion_field in ("excluded_from_training", "excluded_from_prompt_development"):
        if provenance.get(exclusion_field) is not True:
            raise FailureDiagnosisFixtureManifestError(f"{field}.{exclusion_field} must be true")
    return {
        "status": status,
        "excluded_from_training": True,
        "excluded_from_prompt_development": True,
    }


def validate_failure_diagnosis_fixture_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize an independently reviewed source-fixture manifest.

    Each entry must bind one case to a source URI, source digest, predicate
    schema/version, reviewer, adjudication status, and explicit leakage
    exclusions.  Pending or missing review metadata fails closed before source
    records can be adapted.

    Returns:
        A normalized manifest with only the admitted contract fields.

    Raises:
        FailureDiagnosisFixtureManifestError: If the manifest is malformed or
            lacks independent review and leakage metadata.
    """
    if not isinstance(manifest, Mapping):
        raise FailureDiagnosisFixtureManifestError("fixture manifest must be a mapping")
    if manifest.get("schema_version") != FAILURE_DIAGNOSIS_FIXTURE_MANIFEST_SCHEMA_VERSION:
        raise FailureDiagnosisFixtureManifestError(
            "fixture manifest schema_version must be "
            f"{FAILURE_DIAGNOSIS_FIXTURE_MANIFEST_SCHEMA_VERSION!r}"
        )
    manifest_id = _require_text(manifest.get("manifest_id"), "manifest_id")
    manifest_version = _require_positive_int(manifest.get("manifest_version"), "manifest_version")
    raw_entries = manifest.get("fixtures")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FailureDiagnosisFixtureManifestError("fixture manifest fixtures must be non-empty")

    entries: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        field = f"fixtures[{index}]"
        entry = _require_mapping(raw_entry, field)
        case_id = _require_text(entry.get("case_id"), f"{field}.case_id")
        if case_id in seen_case_ids:
            raise FailureDiagnosisFixtureManifestError(
                f"fixture manifest case_id is duplicated: {case_id!r}"
            )
        seen_case_ids.add(case_id)
        fixture_version = _require_positive_int(
            entry.get("fixture_version"), f"{field}.fixture_version"
        )
        source_trace_uri = _require_text(entry.get("source_trace_uri"), f"{field}.source_trace_uri")
        source_digest = _require_text(
            entry.get("source_trace_sha256"), f"{field}.source_trace_sha256"
        )
        if _SHA256_PATTERN.fullmatch(source_digest) is None:
            raise FailureDiagnosisFixtureManifestError(
                f"{field}.source_trace_sha256 must be 64 lowercase hexadecimal characters"
            )
        predicate_id = _require_text(
            entry.get("source_predicate_id"), f"{field}.source_predicate_id"
        )
        predicate_schema = _require_text(
            entry.get("source_predicate_schema_version"),
            f"{field}.source_predicate_schema_version",
        )
        if predicate_schema != TRACE_FAILURE_PREDICATE_SCHEMA_VERSION:
            raise FailureDiagnosisFixtureManifestError(
                f"{field}.source_predicate_schema_version must be "
                f"{TRACE_FAILURE_PREDICATE_SCHEMA_VERSION!r}"
            )
        review = _validate_review_metadata(
            _require_mapping(entry.get("review"), f"{field}.review"), f"{field}.review"
        )
        provenance = _validate_provenance_metadata(
            _require_mapping(entry.get("provenance"), f"{field}.provenance"),
            f"{field}.provenance",
        )
        entries.append(
            {
                "case_id": case_id,
                "fixture_version": fixture_version,
                "source_trace_uri": source_trace_uri,
                "source_trace_sha256": source_digest,
                "source_predicate_id": predicate_id,
                "source_predicate_schema_version": predicate_schema,
                "review": review,
                "provenance": provenance,
            }
        )

    return {
        "schema_version": FAILURE_DIAGNOSIS_FIXTURE_MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "manifest_version": manifest_version,
        "fixtures": entries,
    }


def load_failure_diagnosis_fixture_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON source-fixture manifest from disk.

    Returns:
        A normalized source-fixture manifest.

    Raises:
        FailureDiagnosisFixtureManifestError: If the file cannot be loaded or
            the manifest fails validation.
    """
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FailureDiagnosisFixtureManifestError(
            f"unable to load fixture manifest: {manifest_path}"
        ) from exc
    return validate_failure_diagnosis_fixture_manifest(payload)


def _reject_reference_label_leakage(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.strip().lower() in _FORBIDDEN_SOURCE_KEYS:
                raise FixtureLeakageError(
                    f"source predicate contains forbidden reference metadata at {path}.{key}"
                )
            _reject_reference_label_leakage(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_reference_label_leakage(nested, path=f"{path}[{index}]")


def _validate_source_predicates(
    manifest: Mapping[str, Any],
    source_predicates: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(source_predicates, Mapping):
        raise FailureDiagnosisFixtureManifestError("source_predicates must be a case-id mapping")
    entries = manifest["fixtures"]
    expected_case_ids = {entry["case_id"] for entry in entries}
    actual_case_ids = {str(case_id) for case_id in source_predicates}
    if actual_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - actual_case_ids)
        extra = sorted(actual_case_ids - expected_case_ids)
        raise FailureDiagnosisFixtureManifestError(
            f"source case ids do not match manifest; missing={missing}, extra={extra}"
        )

    normalized: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        case_id = entry["case_id"]
        source = source_predicates[case_id]
        if not isinstance(source, Mapping):
            raise FailureDiagnosisFixtureManifestError(
                f"source predicate for {case_id!r} must be a mapping"
            )
        _reject_reference_label_leakage(source, path=f"source[{case_id!r}]")
        predicate_id = _require_text(
            source.get("predicate_id"), f"source[{case_id!r}].predicate_id"
        )
        if predicate_id != entry["source_predicate_id"]:
            raise FailureDiagnosisFixtureManifestError(
                f"source predicate id mismatch for {case_id!r}: "
                f"expected {entry['source_predicate_id']!r}, got {predicate_id!r}"
            )
        actual_digest = canonical_source_sha256(source)
        if actual_digest != entry["source_trace_sha256"]:
            raise FailureDiagnosisFixtureManifestError(
                f"source digest mismatch for {case_id!r}: "
                f"expected {entry['source_trace_sha256']}, got {actual_digest}"
            )
        normalized[case_id] = source
    return normalized


def _validate_reference_admission(reference_fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Require the reference envelope to be reviewed before source adaptation.

    Returns:
        A normalized, reviewed reference fixture.

    Raises:
        FailureDiagnosisFixtureManifestError: If review or provenance admission
            is incomplete.
    """
    validate_fixture_review_admission(reference_fixture)
    validated = validate_failure_diagnosis_reference_fixture(reference_fixture)
    review = validated["review"]
    if (
        review["status"] != "reviewed"
        or review["adjudication_status"] not in _ADJUDICATED_STATUSES
        or review["independent_of_automated_diagnosis"] is not True
    ):
        raise FailureDiagnosisFixtureManifestError(
            "reference fixture review must be reviewed, adjudicated, and independent"
        )
    provenance = validated["provenance"]
    if provenance["status"] not in {"complete", "verified"}:
        raise FailureDiagnosisFixtureManifestError(
            "reference fixture provenance must be complete or verified"
        )
    return validated


def build_deterministic_failure_diagnosis_records(
    manifest: Mapping[str, Any],
    source_predicates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run the existing deterministic adapter after source admission checks.

    Returns:
        Case-id keyed ``failure_diagnosis.v1`` records.

    Raises:
        FailureDiagnosisFixtureManifestError: If source or manifest admission
            fails before adaptation.
    """
    validated_manifest = validate_failure_diagnosis_fixture_manifest(manifest)
    normalized_sources = _validate_source_predicates(validated_manifest, source_predicates)
    records: dict[str, dict[str, Any]] = {}
    for entry in validated_manifest["fixtures"]:
        case_id = entry["case_id"]
        record = diagnose_from_trace_failure_predicate(normalized_sources[case_id]).to_dict()
        record["case_id"] = case_id
        records[case_id] = record
    return records


def evaluate_deterministic_failure_diagnosis_fixture(
    manifest: Mapping[str, Any],
    source_predicates: Mapping[str, Any],
    reference_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate admitted deterministic records against an exact reference set.

    Review admission is checked before the adapter runs.  The manifest, source
    records, and reference fixture must cover exactly the same case ids; missing
    or extra rows are structural blockers rather than metric exclusions.

    Returns:
        A deterministic fixture-level quality report with source provenance.

    Raises:
        FailureDiagnosisError: If review, source, reference, or case alignment
            admission fails.
    """
    validated_manifest = validate_failure_diagnosis_fixture_manifest(manifest)
    validated_reference = _validate_reference_admission(reference_fixture)
    records = build_deterministic_failure_diagnosis_records(validated_manifest, source_predicates)
    manifest_case_ids = {entry["case_id"] for entry in validated_manifest["fixtures"]}
    reference_case_ids = {record["case_id"] for record in validated_reference["records"]}
    if manifest_case_ids != reference_case_ids:
        raise FailureDiagnosisFixtureManifestError(
            "manifest and reference fixture case ids do not match; "
            f"manifest={sorted(manifest_case_ids)}, reference={sorted(reference_case_ids)}"
        )
    report = evaluate_failure_diagnosis_quality(records, validated_reference)
    report["fixture_manifest"] = {
        "schema_version": validated_manifest["schema_version"],
        "manifest_id": validated_manifest["manifest_id"],
        "manifest_version": validated_manifest["manifest_version"],
        "fixture_count": len(validated_manifest["fixtures"]),
    }
    report["source_provenance"] = [
        {
            "case_id": entry["case_id"],
            "source_trace_uri": entry["source_trace_uri"],
            "source_trace_sha256": entry["source_trace_sha256"],
            "source_predicate_id": entry["source_predicate_id"],
            "fixture_version": entry["fixture_version"],
        }
        for entry in validated_manifest["fixtures"]
    ]
    report["claim_boundary"] = {
        "deterministic_diagnostic_metric_integrity_only": True,
        "no_general_diagnostic_accuracy_claim": True,
        "no_correction_usefulness_claim": True,
        "no_campaign_or_benchmark_ranking": True,
    }
    return report


__all__ = [
    "FAILURE_DIAGNOSIS_FIXTURE_MANIFEST_SCHEMA_VERSION",
    "FailureDiagnosisFixtureManifestError",
    "FixtureLeakageError",
    "build_deterministic_failure_diagnosis_records",
    "canonical_source_sha256",
    "evaluate_deterministic_failure_diagnosis_fixture",
    "load_failure_diagnosis_fixture_manifest",
    "validate_failure_diagnosis_fixture_manifest",
]
