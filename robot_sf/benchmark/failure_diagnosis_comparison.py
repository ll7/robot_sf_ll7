"""Held-out comparison harness for frozen learned vs. deterministic diagnosis records.

This module adds a fail-closed comparison infrastructure for two already-materialized
sets of ``failure_diagnosis.v1`` records evaluated against an independently authored
held-out reference fixture.  It does **not** execute any model, call any API, or
generate learned diagnoses.

The deterministic adapter remains the authoritative comparator.  Learned outputs are
accepted only when accompanied by a pinned method manifest whose provenance fields are
all present, parseable, and non-empty.  Missing, invalid, or unavailable provenance
fails closed as ``"unavailable"`` -- provenance is never synthesised.

Deterministic vs. learned metric inputs
----------------------------------------
The existing :func:`evaluate_failure_diagnosis_quality` evaluator validates
``diagnosis_source`` against the deterministic adapter constant.  Deterministic
``failure_diagnosis.v1`` inputs pass through unchanged.  Learned comparison metric
inputs are projected: canonical deterministic fields are stripped so the evaluator
can compare detection, onset, failure type, and severity.  The original
non-deterministic source/provenance marker is preserved in the comparison output
(``learned_source_projection``) and per-record ``_learned_source_preserved``
annotation.  Learned records that carry a deterministic ``diagnosis_source`` are
rejected -- the learned side must not duplicate the deterministic adapter.

Design note: the existing :func:`evaluate_failure_diagnosis_quality` (from
``failure_diagnosis.v1``) computes detection, onset, failure-type, and severity
metrics for one method against the reference.  This module calls it twice -- once for
the deterministic comparator, once for the learned method (after projection) -- and
wraps the two sub-reports inside a single versioned comparison report with
provenance-gate metadata.

Issue: #6646.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from robot_sf.benchmark.failure_diagnosis import (
    DIAGNOSIS_SOURCE,
    FAILURE_DIAGNOSIS_SCHEMA_VERSION,
    FailureDiagnosisError,
    evaluate_failure_diagnosis_quality,
    validate_failure_diagnosis_reference_fixture,
)

COMPARISON_SCHEMA_VERSION = "held_out_diagnosis_comparison.v1"

#: Review markers that indicate the fixture has not been independently reviewed.
#: A fixture carrying any of these markers is rejected by the admission check.
REVIEW_PENDING_MARKERS = frozenset({"AI-GENERATED NEEDS-REVIEW", "NEEDS-REVIEW", "PENDING"})

#: Provenance fields that a pinned method manifest must carry.  Every field is
#: required; a missing or empty value causes the learned method to be rejected.
_REQUIRED_MANIFEST_FIELDS = (
    "method_id",
    "model_identifier",
    "model_revision",
    "prompt_digest",
    "decoding_settings",
    "input_schema",
    "output_artifact_digest",
    "held_out_exclusion_declaration",
    "non_deterministic_source",
)

#: Allowed output statuses for a comparison report.
_OUTPUT_STATUSES = ("available", "unavailable")

_CAVEATS = (
    "This comparison harness does not execute models or call APIs.",
    "The deterministic adapter is the authoritative comparator; it is never overwritten.",
    "Unknown, unavailable, fallback, degraded, and provenance-incomplete cases are excluded "
    "per metric and retained in case comparisons.",
    "This report does not rank campaigns or make scientific-result claims.",
)


# ---------------------------------------------------------------------------
# Method manifest validation
# ---------------------------------------------------------------------------


class MethodManifestError(FailureDiagnosisError):
    """Raised when a method manifest is missing required provenance fields."""


class FixtureReviewPendingError(FailureDiagnosisError):
    """Raised when the reference fixture carries a pending review marker."""


class LearnedSourceError(FailureDiagnosisError):
    """Raised when a learned record carries a deterministic diagnosis_source."""


@dataclass(frozen=True, slots=True)
class MethodManifest:
    """Pinned provenance envelope for a frozen learned/LLM method output.

    All fields are mandatory strings.  Missing or empty values fail closed.
    """

    method_id: str
    model_identifier: str
    model_revision: str
    prompt_digest: str
    decoding_settings: str
    input_schema: str
    output_artifact_digest: str
    held_out_exclusion_declaration: str
    non_deterministic_source: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Returns:
            Dictionary representation of the manifest.
        """
        return asdict(self)


def validate_method_manifest(manifest: Mapping[str, Any]) -> MethodManifest:
    """Validate and normalise a pinned method manifest.

    Every required provenance field must be a non-empty string.  An empty or
    missing value fails closed -- provenance is never synthesised.

    Args:
        manifest: Mapping containing pinned method provenance.

    Returns:
        A validated :class:`MethodManifest`.

    Raises:
        MethodManifestError: If any required field is missing or empty.
    """
    if not isinstance(manifest, Mapping):
        raise MethodManifestError("method manifest must be a mapping")
    raw: dict[str, Any] = dict(manifest)
    missing: list[str] = []
    for field_name in _REQUIRED_MANIFEST_FIELDS:
        value = raw.get(field_name)
        if not isinstance(value, str) or not value.strip():
            missing.append(field_name)
    if missing:
        raise MethodManifestError(
            f"method manifest missing or empty required fields: {', '.join(missing)}"
        )
    return MethodManifest(
        method_id=raw["method_id"].strip(),
        model_identifier=raw["model_identifier"].strip(),
        model_revision=raw["model_revision"].strip(),
        prompt_digest=raw["prompt_digest"].strip(),
        decoding_settings=raw["decoding_settings"].strip(),
        input_schema=raw["input_schema"].strip(),
        output_artifact_digest=raw["output_artifact_digest"].strip(),
        held_out_exclusion_declaration=raw["held_out_exclusion_declaration"].strip(),
        non_deterministic_source=raw["non_deterministic_source"].strip(),
    )


# ---------------------------------------------------------------------------
# Case alignment
# ---------------------------------------------------------------------------


class CaseAlignmentError(FailureDiagnosisError):
    """Raised when deterministic and learned case sets cannot be aligned."""


_PAYLOAD_SHAPE_REQUIRED_FIELDS = (
    "schema_version",
    "diagnosis_source",
    "generated_at_utc",
    "failure_type_coverage",
    "caveats",
)


def _extract_case_ids_from_payload(payload: Mapping[str, Any]) -> list[str]:
    """Extract ordered case ids from a ``failure_diagnosis.v1`` payload.

    Returns:
        Ordered list of case identifiers.
    """
    case_ids: list[str] = []
    for record in payload.get("records", ()):
        rid = (
            record.get("case_id")
            or record.get("sample_id")
            or record.get("episode_id")
            or record.get("record_id")
        )
        if rid is None:
            raise FailureDiagnosisError(
                "diagnosis record is missing case_id, sample_id, episode_id, or record_id"
            )
        case_ids.append(str(rid))
    return case_ids


def _extract_case_ids_from_mapping(mapping: Mapping[str, Any]) -> list[str]:
    """Extract ordered case ids from a case-id keyed mapping.

    Returns:
        Ordered list of case identifiers.
    """
    return [str(k) for k in mapping.keys()]


def _resolve_case_ids(records: Mapping[str, Any] | Iterable[Any]) -> list[str]:
    """Resolve case ids from the flexible input shapes accepted by this harness.

    Returns:
        Ordered list of case identifiers.
    """
    if isinstance(records, Mapping):
        if _is_payload_shape(records):
            return _extract_case_ids_from_payload(records)
        if isinstance(records.get("records"), list):
            raise CaseAlignmentError(
                "mapping with a 'records' case id is ambiguous; provide the complete "
                "failure_diagnosis.v1 payload envelope or rename the case"
            )
        return _extract_case_ids_from_mapping(records)
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        ids: list[str] = []
        for idx, record in enumerate(records):
            if isinstance(record, Mapping):
                rid = (
                    record.get("case_id")
                    or record.get("sample_id")
                    or record.get("episode_id")
                    or record.get("record_id")
                )
                if rid is None:
                    raise FailureDiagnosisError(
                        f"diagnosis records[{idx}] is missing case_id, sample_id, "
                        "episode_id, or record_id"
                    )
                ids.append(str(rid))
            else:
                raise FailureDiagnosisError(f"diagnosis records[{idx}] must be a mapping")
        return ids
    raise CaseAlignmentError("records must be a mapping or iterable")


def align_held_out_cases(
    deterministic_records: Mapping[str, Any] | Iterable[Any],
    learned_records: Mapping[str, Any] | Iterable[Any],
) -> dict[str, Any]:
    """Validate that deterministic and learned case sets are perfectly aligned.

    The comparison harness requires that both sets cover exactly the same case ids
    with no duplicates and no mismatches.  Duplicate, missing, or mismatched ids
    cause a hard failure -- this is not a metric exclusion, it is a structural
    rejection.

    Args:
        deterministic_records: The deterministic ``failure_diagnosis.v1`` payload
            or case-id mapping.
        learned_records: The frozen learned ``failure_diagnosis.v1`` payload or
            case-id mapping.

    Returns:
        An alignment summary with ``aligned_case_ids``, ``deterministic_count``,
        and ``learned_count``.

    Raises:
        CaseAlignmentError: If the case sets are not identical.
    """
    det_ids = _resolve_case_ids(deterministic_records)
    learn_ids = _resolve_case_ids(learned_records)

    det_set = set(det_ids)
    learn_set = set(learn_ids)

    if len(det_ids) != len(det_set):
        dupes = [c for c in det_ids if det_ids.count(c) > 1]
        raise CaseAlignmentError(
            f"deterministic records contain duplicate case ids: {sorted(set(dupes))}"
        )
    if len(learn_ids) != len(learn_set):
        dupes = [c for c in learn_ids if learn_ids.count(c) > 1]
        raise CaseAlignmentError(
            f"learned records contain duplicate case ids: {sorted(set(dupes))}"
        )

    missing_in_learned = det_set - learn_set
    missing_in_deterministic = learn_set - det_set
    if missing_in_learned or missing_in_deterministic:
        parts: list[str] = []
        if missing_in_learned:
            parts.append(f"missing_from_learned: {sorted(missing_in_learned)}")
        if missing_in_deterministic:
            parts.append(f"missing_from_deterministic: {sorted(missing_in_deterministic)}")
        raise CaseAlignmentError("case id mismatch: " + "; ".join(parts))

    return {
        "aligned_case_ids": sorted(det_ids),
        "deterministic_count": len(det_ids),
        "learned_count": len(learn_ids),
    }


# ---------------------------------------------------------------------------
# Fixture admission gate
# ---------------------------------------------------------------------------


def validate_fixture_review_admission(fixture: Mapping[str, Any]) -> None:
    """Reject a reference fixture that carries a pending review marker.

    The comparison harness requires independently reviewed fixtures.  A fixture
    whose ``review_marker`` matches any entry in :data:`REVIEW_PENDING_MARKERS`
    is rejected before evaluation.  This is a fail-closed admission gate: a
    pending marker is never silently ignored.

    Args:
        fixture: A reference fixture mapping.

    Raises:
        FixtureReviewPendingError: If the fixture carries a pending review marker.
    """
    if not isinstance(fixture, Mapping):
        raise FailureDiagnosisError("reference fixture must be a mapping")
    review_marker = fixture.get("review_marker")
    if isinstance(review_marker, str):
        normalized_marker = " ".join(review_marker.split()).upper()
        if any(marker in normalized_marker for marker in REVIEW_PENDING_MARKERS):
            raise FixtureReviewPendingError(
                f"reference fixture carries a pending review marker: {review_marker!r}; "
                "independent review/adjudication is required before comparison"
            )


# ---------------------------------------------------------------------------
# Learned metric-input projection
# ---------------------------------------------------------------------------

#: Canonical deterministic fields projected out before generic metric comparison.
#: The original source and schema markers remain in the projection annotation
#: and the public per-case comparison report.
_LEARNED_PROJECTION_DROP_FIELDS = (
    "diagnosis_schema_version",
    "diagnosis_source",
)


def _project_learned_record_for_evaluator(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a learned record into evaluator-compatible format.

    The existing :func:`evaluate_failure_diagnosis_quality` evaluator validates
    ``diagnosis_source`` against the deterministic adapter constant.  A truthful
    non-deterministic learned record must not be relabeled as deterministic and
    must not silently lose its provenance.  This projection strips the
    deterministic fields so the evaluator can compare detection, onset, failure
    type, and severity, while the comparison output retains the original markers.

    Deterministic-source learned payloads/records are rejected at the comparison
    level before this projection is reached.

    Args:
        record: A learned diagnosis record mapping.

    Returns:
        A new mapping with deterministic-source fields removed and a
        ``_learned_source_preserved`` annotation recording what was dropped.
    """
    projected: dict[str, Any] = {}
    preserved: dict[str, Any] = {}
    for key, value in record.items():
        if key in _LEARNED_PROJECTION_DROP_FIELDS:
            preserved[key] = value
        else:
            projected[key] = value
    projected["_learned_source_preserved"] = preserved
    return projected


def _is_payload_shape(mapping: Mapping[str, Any]) -> bool:
    """Detect whether a mapping is a ``failure_diagnosis.v1`` payload.

    A payload has the complete versioned envelope and a records key whose value is
    a non-empty list of mappings. Requiring the envelope avoids confusing a
    case-id mapping containing a case literally named records with a payload.

    Args:
        mapping: A mapping to classify.

    Returns:
        ``True`` if the mapping has payload shape, ``False`` otherwise.
    """
    records_value = mapping.get("records")
    if not isinstance(records_value, list):
        return False
    if not records_value:
        return False
    if not all(isinstance(item, Mapping) for item in records_value):
        return False
    if mapping.get("schema_version") != FAILURE_DIAGNOSIS_SCHEMA_VERSION:
        return False
    return all(field in mapping for field in _PAYLOAD_SHAPE_REQUIRED_FIELDS)


def _validate_learned_source(record: Any, label: str) -> None:
    """Require one learned record to carry a non-deterministic source marker."""
    if not isinstance(record, Mapping):
        raise LearnedSourceError(f"learned {label} must be a mapping")
    source = record.get("diagnosis_source")
    if not isinstance(source, str) or not source.strip():
        raise LearnedSourceError(f"learned {label} is missing a non-empty diagnosis_source")
    if source == DIAGNOSIS_SOURCE:
        raise LearnedSourceError(
            f"learned {label} has deterministic diagnosis_source "
            f"{DIAGNOSIS_SOURCE!r}; the learned side must not duplicate "
            "the deterministic adapter source"
        )


def _reject_deterministic_source_learned(
    learned_records: Mapping[str, Any] | Iterable[Any],
) -> None:
    """Reject learned records that carry a deterministic ``diagnosis_source``.

    A learned record claiming the deterministic adapter source is invalid:
    the deterministic adapter is the authoritative comparator and must not be
    duplicated on the learned side.  Deterministic-source learned payloads are
    rejected before projection.

    Args:
        learned_records: Learned records in any accepted shape.

    Raises:
        LearnedSourceError: If any learned record has a deterministic
            ``diagnosis_source``.
    """
    if isinstance(learned_records, Mapping):
        if _is_payload_shape(learned_records):
            payload_source = learned_records.get("diagnosis_source")
            if payload_source == DIAGNOSIS_SOURCE:
                raise LearnedSourceError(
                    "learned payload has deterministic diagnosis_source; the learned "
                    "side must not duplicate the deterministic adapter source"
                )
            if not isinstance(payload_source, str) or not payload_source.strip():
                raise LearnedSourceError(
                    "learned payload must carry a non-empty non-deterministic diagnosis_source"
                )
            for idx, record in enumerate(learned_records.get("records", ())):
                _validate_learned_source(record, f"record[{idx}]")
        else:
            for case_id, record in learned_records.items():
                _validate_learned_source(record, f"case {case_id!r}")
    elif isinstance(learned_records, Iterable) and not isinstance(learned_records, (str, bytes)):
        for idx, record in enumerate(learned_records):
            _validate_learned_source(record, f"record[{idx}]")


def _materialize_records(
    records: Mapping[str, Any] | Iterable[Any],
) -> Mapping[str, Any] | list[Any]:
    """Materialize one-shot record iterables before repeated comparison passes.

    The comparison path validates, aligns, evaluates, and annotates records.  A
    generator must therefore be captured once at the boundary rather than consumed
    by the first validation pass.

    Returns:
        The original mapping or a materialized list of records.

    Raises:
        FailureDiagnosisError: If ``records`` is not a supported record container.
    """
    if isinstance(records, Mapping):
        return records
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        return list(records)
    raise FailureDiagnosisError("records must be a mapping or iterable of record mappings")


def _project_learned_records(
    learned_records: Mapping[str, Any] | Iterable[Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Project learned records into evaluator-compatible format.

    Each learned record has canonical deterministic fields removed so the
    existing evaluator can compare detection, onset, failure type, and severity.
    The original markers are preserved in a private annotation per record.

    Args:
        learned_records: Learned records in any accepted shape.

    Returns:
        Projected records in the same shape as the input.
    """
    if isinstance(learned_records, Mapping):
        if isinstance(learned_records.get("records"), list) and not _is_payload_shape(
            learned_records
        ):
            raise CaseAlignmentError(
                "mapping with a 'records' case id is ambiguous; provide the complete "
                "failure_diagnosis.v1 payload envelope or rename the case"
            )
        if _is_payload_shape(learned_records):
            projected_records = [
                _project_learned_record_for_evaluator(r) for r in learned_records.get("records", ())
            ]
            projected_payload = dict(learned_records)
            projected_payload["records"] = projected_records
            return projected_payload
        return {
            case_id: _project_learned_record_for_evaluator(record)
            for case_id, record in learned_records.items()
        }
    if isinstance(learned_records, Iterable) and not isinstance(learned_records, (str, bytes)):
        return [_project_learned_record_for_evaluator(r) for r in learned_records]
    return dict(learned_records)


def _projected_learned_source_by_case_id(
    learned_records: Mapping[str, Any] | Iterable[Any],
) -> dict[str, dict[str, Any]]:
    """Collect preserved learned provenance annotations keyed by case id.

    The existing evaluator intentionally exposes only metric-facing diagnosis fields in
    each case view. This helper carries the source annotation from the projected input
    into the comparison report without changing evaluator semantics or relabeling the
    learned records.

    Returns:
        Mapping from case id to the preserved learned provenance fields.
    """
    if isinstance(learned_records, Mapping):
        if _is_payload_shape(learned_records):
            entries = ((None, record) for record in learned_records.get("records", ()))
        else:
            entries = learned_records.items()
    elif isinstance(learned_records, Iterable) and not isinstance(learned_records, (str, bytes)):
        entries = ((None, record) for record in learned_records)
    else:
        return {}

    preserved_by_case_id: dict[str, dict[str, Any]] = {}
    for provided_case_id, record in entries:
        if not isinstance(record, Mapping):
            continue
        case_id = provided_case_id or (
            record.get("case_id")
            or record.get("sample_id")
            or record.get("episode_id")
            or record.get("record_id")
        )
        preserved = record.get("_learned_source_preserved")
        if case_id is not None and isinstance(preserved, Mapping) and preserved:
            preserved_by_case_id[str(case_id)] = dict(preserved)
    return preserved_by_case_id


def _attach_learned_source_preservation(
    learned_report: dict[str, Any],
    learned_projected: Mapping[str, Any] | Iterable[Any],
) -> None:
    """Attach preserved learned source metadata to evaluator case views."""
    preserved_by_case_id = _projected_learned_source_by_case_id(learned_projected)
    for case_comparison in learned_report.get("case_comparisons", ()):
        case_id = case_comparison.get("case_id")
        diagnosis = case_comparison.get("diagnosis")
        preserved = preserved_by_case_id.get(str(case_id))
        if preserved and isinstance(diagnosis, dict):
            diagnosis["_learned_source_preserved"] = preserved


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def compare_held_out_diagnoses(
    reference_fixture: Mapping[str, Any],
    deterministic_records: Mapping[str, Any] | Iterable[Any],
    learned_records: Mapping[str, Any] | Iterable[Any],
    method_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare deterministic and frozen learned diagnoses against a held-out reference.

    This is the primary entry point for the held-out comparison harness.  It:

    1. Validates the reference fixture.
    2. Rejects fixtures with pending review markers (fail-closed admission).
    3. Validates the method manifest (fail-closed on missing provenance).
    4. Rejects deterministic-source learned payloads/records.
    5. Projects learned records into evaluator-compatible format, preserving
       the original source/provenance marker.
    6. Validates case alignment (rejects duplicate, missing, or mismatched ids).
    7. Calls :func:`evaluate_failure_diagnosis_quality` for the deterministic
       comparator and for the projected learned records.
    8. Emits a versioned comparison report with both method summaries,
       provenance/claim-boundary metadata, and the learned source projection.

    The deterministic records are the authoritative comparator and are never
    modified or overwritten.  Learned records with a truthful non-deterministic
    ``diagnosis_source`` are not relabeled as deterministic.

    Args:
        reference_fixture: An independently authored ``failure_diagnosis_reference.v1``
            fixture.
        deterministic_records: The deterministic ``failure_diagnosis.v1`` payload,
            a case-id mapping, or a sequence with case identifiers.
        learned_records: The frozen learned ``failure_diagnosis.v1`` payload,
            a case-id mapping, or a sequence with case identifiers.
        method_manifest: Pinned provenance envelope for the learned method.

    Returns:
        A ``held_out_diagnosis_comparison.v1`` report.

    Raises:
        FailureDiagnosisError: On structural validation failure.
        FixtureReviewPendingError: If the fixture carries a pending review marker.
        MethodManifestError: On missing or invalid provenance fields.
        LearnedSourceError: If learned records carry a deterministic diagnosis_source.
        CaseAlignmentError: On duplicate, missing, or mismatched case ids.
    """
    deterministic_records = _materialize_records(deterministic_records)
    learned_records = _materialize_records(learned_records)
    validate_fixture_review_admission(reference_fixture)
    validated_reference = validate_failure_diagnosis_reference_fixture(reference_fixture)
    validated_manifest = validate_method_manifest(method_manifest)

    _reject_deterministic_source_learned(learned_records)
    learned_projected = _project_learned_records(learned_records)

    alignment = align_held_out_cases(deterministic_records, learned_projected)

    deterministic_report = evaluate_failure_diagnosis_quality(
        deterministic_records, validated_reference
    )
    learned_report = evaluate_failure_diagnosis_quality(learned_projected, validated_reference)
    _attach_learned_source_preservation(learned_report, learned_projected)

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "output_status": _OUTPUT_STATUSES[0],
        "output_reason": None,
        "alignment": alignment,
        "method_manifest": validated_manifest.to_dict(),
        "learned_source_projection": {
            "description": (
                "Learned records were projected to remove canonical deterministic "
                "fields before evaluator comparison. The original source and "
                "schema markers are preserved per record."
            ),
            "preserved_fields": list(_LEARNED_PROJECTION_DROP_FIELDS),
            "preserved_source_fields": ["diagnosis_source"],
        },
        "deterministic_summary": {
            "metrics": deterministic_report.get("metrics", {}),
            "case_count": deterministic_report.get("case_count"),
            "matched_case_count": deterministic_report.get("matched_case_count"),
            "unmatched_diagnosis_count": deterministic_report.get("unmatched_diagnosis_count"),
        },
        "learned_summary": {
            "metrics": learned_report.get("metrics", {}),
            "case_count": learned_report.get("case_count"),
            "matched_case_count": learned_report.get("matched_case_count"),
            "unmatched_diagnosis_count": learned_report.get("unmatched_diagnosis_count"),
        },
        "case_comparisons": learned_report.get("case_comparisons", []),
        "deterministic_case_comparisons": deterministic_report.get("case_comparisons", []),
        "claim_boundary": {
            "fixture_level_metrics_only": True,
            "no_campaign_ranking": True,
            "no_scientific_result_claim": True,
            "next_gate": (
                "Real model output requires a pinned method manifest with non-empty "
                "provenance fields, execution on the same held-out case set, and "
                "independent fixture review before any benchmark claim."
            ),
        },
        "caveats": list(_CAVEATS),
    }


def build_unavailable_comparison_report(
    reason: str,
    *,
    method_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an unavailable comparison report when the learned method cannot be admitted.

    Callers can use this fail-closed builder when provenance validation, case alignment,
    or another admission gate fails.  It emits an explicit ``"unavailable"`` status with
    the reason; the deterministic comparator is never run when admission fails.

    Args:
        reason: Human-readable reason the learned method was not admitted.
        method_manifest: Optional manifest for metadata; ignored if admission failed
            before manifest validation.

    Returns:
        A ``held_out_diagnosis_comparison.v1`` report with status ``"unavailable"``.
    """
    manifest_summary: dict[str, Any] | None = None
    if method_manifest is not None:
        try:
            validated = validate_method_manifest(method_manifest)
            manifest_summary = validated.to_dict()
        except MethodManifestError:
            manifest_summary = {"raw": dict(method_manifest), "validation": "failed"}

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "output_status": _OUTPUT_STATUSES[1],
        "output_reason": reason,
        "alignment": None,
        "method_manifest": manifest_summary,
        "learned_source_projection": None,
        "deterministic_summary": None,
        "learned_summary": None,
        "case_comparisons": [],
        "deterministic_case_comparisons": [],
        "claim_boundary": {
            "fixture_level_metrics_only": True,
            "no_campaign_ranking": True,
            "no_scientific_result_claim": True,
            "next_gate": (
                "Real model output requires a pinned method manifest with non-empty "
                "provenance fields, execution on the same held-out case set, and "
                "independent fixture review before any benchmark claim."
            ),
        },
        "caveats": list(_CAVEATS),
    }


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "REVIEW_PENDING_MARKERS",
    "CaseAlignmentError",
    "FixtureReviewPendingError",
    "LearnedSourceError",
    "MethodManifest",
    "MethodManifestError",
    "align_held_out_cases",
    "build_unavailable_comparison_report",
    "compare_held_out_diagnoses",
    "validate_fixture_review_admission",
    "validate_method_manifest",
]
