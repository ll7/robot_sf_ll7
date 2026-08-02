"""Versioned, optional failure-diagnosis records (issue #6583, first-PR slice).

This module adds a structured ``failure_diagnosis`` record that adapts existing
deterministic failure evidence into a consistent diagnosis schema. It is an
*optional diagnostic sidecar*: it does not modify the existing failure-mechanism
classifier, taxonomy, or trace-predicate modules, and it introduces no parallel
taxonomy.

Reuse contract
--------------
- ``failure_type`` reuses the existing classifier labels (``FAILURE_MECHANISM_LABELS``
  from :mod:`robot_sf.benchmark.failure_mechanism_classifier`). ``"unknown"`` is the
  only added value, drawn from the taxonomy label vocabulary, and marks
  unsupported, invalid, or unavailable mappings.
- ``confidence`` reuses the taxonomy confidence vocabulary (``MECHANISM_CONFIDENCES``)
  and ``evidence_mode`` reuses ``MECHANISM_EVIDENCE_MODES``
  from :mod:`robot_sf.benchmark.failure_mechanism_taxonomy`. The original issue
  proposal sketched a numeric confidence; this first-PR slice reuses the taxonomy's
  string confidence modes instead, per the maintainer audit update and the
  Domain-Aware Approval on issue #6583, so it does not invent a parallel scale.
- ``causal_evidence`` cites trace/predicate evidence *pointers only*
  (predicate id, time interval, steps, involved actors, and the source
  predicate's ``evidence_fields``). It is evidence citation, never causal
  inference.
- ``proposed_correction`` and ``correction_status`` are optional and default to
  unreviewed. This PR makes no correction-quality or causal-validity claim.

Deterministic adapter
---------------------
:func:`diagnose_from_trace_failure_predicate` converts one
:class:`~robot_sf.analysis_workbench.trace_failure_predicates.TraceFailurePredicate`
into one :class:`FailureDiagnosisRecord`. ``onset_time_s`` and ``onset_interval``
are derived from the predicate's ``time_interval_s``. The source predicate's
``evidence_fields`` and ``validity_status`` are preserved verbatim.

Predicate families handled by the adapter (failure level in parentheses):

- collision (``collision``) -> ``failure_type`` ``"collision"`` (interaction).
- near-miss / clearance-critical interaction (``clearance_critical_interaction``,
  ``occlusion_triggered_near_miss``, ``late_evasive_reaction``) -> ``"near_miss"``
  (interaction).
- low-progress / stuck (``low_progress``, ``zero_motion_timeout_behavior``,
  ``bottleneck_deadlock``) -> ``"persistent_low_progress_timeout"`` or
  ``"timeout_without_progress"`` (control).
- oscillation (``oscillatory_local_control``) -> ``failure_type`` ``"unknown"`` with
  an explicit reason, because the existing classifier labels contain no oscillation
  mechanism. Onset, severity, and evidence pointers are still preserved.

Unsupported predicate ids, predicates whose ``validity_status`` is not ``"valid"``,
and any otherwise unavailable mapping resolve to ``failure_type`` ``"unknown"`` with
an explicit reason, mirroring
:func:`robot_sf.benchmark.failure_mechanism_taxonomy.unknown_failure_mechanism_record`.

Evidence tier: deterministic diagnostic only. This schema makes no benchmark
ranking, causal-validity, or correction-quality claim. Learned/LLM diagnosis
generation, correction-usefulness scoring, and campaign-level diagnosis-quality
evaluation are explicitly out of scope (successor work) and are not implemented
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import TYPE_CHECKING

from robot_sf.benchmark.failure_mechanism_classifier import FAILURE_MECHANISM_LABELS
from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_CONFIDENCES,
    MECHANISM_EVIDENCE_MODES,
)
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from typing import Any

    from robot_sf.analysis_workbench.trace_failure_predicates import TraceFailurePredicate

#: Schema version for the optional failure-diagnosis record and payload.
FAILURE_DIAGNOSIS_SCHEMA_VERSION = "failure_diagnosis.v1"
#: Provenance source string for the deterministic adapter.
DIAGNOSIS_SOURCE = "failure_diagnosis.deterministic.v1"

#: Allowed failure levels (from the issue #6583 proposal schema).
FAILURE_LEVELS = (
    "control",
    "physics",
    "interaction",
    "scenario",
    "infrastructure",
    "analysis",
)

#: Allowed detection methods. The deterministic adapter always emits ``"predicate"``.
DETECTION_METHODS = ("predicate", "rule", "model", "human")
DETECTION_METHOD_PREDICATE = "predicate"

#: Allowed diagnosis severities. ``"unknown"`` is used when the source predicate's
#: validity is not ``"valid"`` (severity cannot be assessed without valid evidence).
DIAGNOSIS_SEVERITIES = ("minor", "major", "critical", "unknown")

#: Allowed correction statuses. The deterministic adapter defaults to unreviewed.
CORRECTION_STATUSES = ("unreviewed", "accepted", "rejected", "tested")
DEFAULT_CORRECTION_STATUS = "unreviewed"

#: Allowed ``failure_type`` values: the classifier label vocabulary plus the
#: taxonomy ``"unknown"`` label for unsupported, invalid, or unavailable mappings.
ALLOWED_FAILURE_TYPES = frozenset(FAILURE_MECHANISM_LABELS) | {"unknown"}

#: Validity status that marks a predicate with usable trace evidence.
_VALID_VALIDITY_STATUS = "valid"

#: Deterministic mapping from a trace predicate id to ``(failure_type, failure_level)``.
#:
#: ``failure_type`` reuses an existing classifier label wherever one exists. A
#: ``failure_type`` of ``"unknown"`` means the classifier labels contain no direct
#: label for that predicate family; onset, severity, and evidence pointers are still
#: preserved on the diagnosis record.
_PREDICATE_DIAGNOSIS_MAP: dict[str, tuple[str, str]] = {
    "collision": ("collision", "interaction"),
    "clearance_critical_interaction": ("near_miss", "interaction"),
    "occlusion_triggered_near_miss": ("near_miss", "interaction"),
    "late_evasive_reaction": ("near_miss", "interaction"),
    "low_progress": ("timeout_without_progress", "control"),
    "zero_motion_timeout_behavior": ("persistent_low_progress_timeout", "control"),
    "bottleneck_deadlock": ("persistent_low_progress_timeout", "control"),
    "oscillatory_local_control": ("unknown", "control"),
}

#: Explicit reasons when a mapped predicate resolves to ``failure_type`` ``"unknown"``.
_UNKNOWN_TYPE_REASONS: dict[str, str] = {
    "oscillatory_local_control": "oscillation_not_represented_in_classifier_labels",
}

#: Deterministic mapping from a predicate severity token to a diagnosis severity.
_PREDICATE_SEVERITY_MAP: dict[str, str] = {
    "critical": "critical",
    "high": "critical",
    "severe": "critical",
    "medium": "major",
    "moderate": "major",
    "low": "minor",
    "minor": "minor",
}

#: Caveat attached to every diagnosis record, stating the non-causal boundary.
_NON_CAUSAL_CAVEAT = (
    "causal_evidence cites trace/predicate evidence pointers only; it is not causal inference."
)
#: Caveat attached to mapped (non-unknown) diagnoses.
_DIAGNOSTIC_LABEL_CAVEAT = (
    "failure_type reuses an existing classifier label as a diagnostic label; it is not "
    "a canonical mechanism attribution without trace review."
)
#: Caveat attached to unknown diagnoses.
_UNKNOWN_CAVEAT = (
    "Unsupported, invalid, or unavailable mappings resolve to unknown; causal_evidence "
    "still cites the source predicate's evidence pointers."
)
#: Caveat attached to every diagnosis payload.
_PAYLOAD_NON_CLAIM_CAVEAT = (
    "failure_diagnosis records are deterministic diagnostics adapted from trace "
    "predicates; they make no benchmark-ranking, causal-validity, or "
    "correction-quality claim."
)
_OUT_OF_SCOPE_CAVEAT = (
    "Learned/LLM diagnosis generation, correction-usefulness scoring, and "
    "campaign-level diagnosis-quality evaluation are out of scope for this schema "
    "version."
)

#: Required top-level fields on every diagnosis record.
_REQUIRED_RECORD_FIELDS = (
    "diagnosis_schema_version",
    "diagnosis_source",
    "failure_level",
    "failure_type",
    "onset_time_s",
    "onset_interval",
    "severity",
    "detection_method",
    "causal_evidence",
    "contributing_factors",
    "confidence",
    "evidence_mode",
    "validity_status",
    "proposed_correction",
    "correction_status",
    "unknown_reason",
    "caveats",
    "source_predicate",
)

#: Diagnosis fields whose value must come from a fixed vocabulary (field, allowed set).
_RECORD_VOCAB_FIELDS = (
    ("failure_level", FAILURE_LEVELS),
    ("failure_type", ALLOWED_FAILURE_TYPES),
    ("severity", DIAGNOSIS_SEVERITIES),
    ("detection_method", DETECTION_METHODS),
    ("confidence", MECHANISM_CONFIDENCES),
    ("evidence_mode", MECHANISM_EVIDENCE_MODES),
    ("correction_status", CORRECTION_STATUSES),
)

#: Diagnosis fields whose value must be string-normalized on output.
_STRING_RECORD_FIELDS = (
    "diagnosis_schema_version",
    "diagnosis_source",
    "failure_level",
    "failure_type",
    "severity",
    "detection_method",
    "confidence",
    "evidence_mode",
    "validity_status",
    "correction_status",
)


class FailureDiagnosisError(RobotSfError, ValueError):
    """Raised when a failure-diagnosis record violates the schema contract."""


@dataclass(frozen=True, slots=True)
class FailureDiagnosisRecord:
    """One deterministic, non-causal failure-diagnosis record (``failure_diagnosis.v1``).

    The schema is documented here and in the module docstring. Field meanings:

    - ``diagnosis_schema_version``: always :data:`FAILURE_DIAGNOSIS_SCHEMA_VERSION`.
    - ``diagnosis_source``: always :data:`DIAGNOSIS_SOURCE` for this adapter.
    - ``failure_level``: one of :data:`FAILURE_LEVELS`. ``"control"`` for motion-control
      failures (low progress, stuck, oscillation); ``"interaction"`` for
      robot-pedestrian contact/close-call failures (collision, near miss);
      ``"analysis"`` for fully unattributed observations.
    - ``failure_type``: one of :data:`ALLOWED_FAILURE_TYPES`. Reuses the classifier
      label vocabulary; ``"unknown"`` marks unsupported, invalid, or unavailable
      mappings.
    - ``onset_time_s``: the predicate ``time_interval_s[0]`` (onset), or ``None``.
    - ``onset_interval``: ``[start, end]`` copied from the predicate ``time_interval_s``.
    - ``severity``: one of :data:`DIAGNOSIS_SEVERITIES`, mapped deterministically from
      the predicate severity. ``"unknown"`` when the source validity is not ``"valid"``.
    - ``detection_method``: always ``"predicate"`` for this deterministic adapter.
    - ``causal_evidence``: evidence pointers citing the source trace predicate. This is
      evidence citation, never causal inference.
    - ``contributing_factors``: always ``[]`` for this adapter (asserting contributing
      factors would be causal inference, which is out of scope).
    - ``confidence``: one of :data:`MECHANISM_CONFIDENCES` (reused from the taxonomy).
    - ``evidence_mode``: one of :data:`MECHANISM_EVIDENCE_MODES` (reused from the
      taxonomy).
    - ``validity_status``: preserved verbatim from the source predicate.
    - ``proposed_correction``: optional; ``None`` means no correction is proposed.
    - ``correction_status``: one of :data:`CORRECTION_STATUSES`; defaults to
      ``"unreviewed"``.
    - ``unknown_reason``: ``None`` when ``failure_type`` is known; otherwise an explicit
      reason string mirroring
      :func:`robot_sf.benchmark.failure_mechanism_taxonomy.unknown_failure_mechanism_record`.
    - ``caveats``: non-causal limitation statements.
    - ``source_predicate``: the full source predicate dict, preserved for provenance.
    """

    diagnosis_schema_version: str
    diagnosis_source: str
    failure_level: str
    failure_type: str
    onset_time_s: float | None
    onset_interval: list[float | None]
    severity: str
    detection_method: str
    causal_evidence: list[dict[str, Any]]
    contributing_factors: list[str]
    confidence: str
    evidence_mode: str
    validity_status: str
    proposed_correction: str | None
    correction_status: str
    unknown_reason: str | None
    caveats: list[str]
    source_predicate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-safe primitives.

        Returns:
            Dictionary representation of the diagnosis record.
        """
        return asdict(self)


def diagnose_from_trace_failure_predicate(
    predicate: TraceFailurePredicate | Mapping[str, Any],
    *,
    proposed_correction: str | None = None,
    correction_status: str = DEFAULT_CORRECTION_STATUS,
) -> FailureDiagnosisRecord:
    """Adapt one trace failure predicate into a deterministic diagnosis record.

    The adapter derives ``onset_time_s`` / ``onset_interval`` from the predicate's
    ``time_interval_s`` and preserves the source predicate's ``evidence_fields`` and
    ``validity_status``. Mapped predicates with valid evidence receive a classifier
    label as ``failure_type``; unsupported predicate ids, predicates whose
    ``validity_status`` is not ``"valid"``, and otherwise unavailable mappings resolve
    to ``failure_type`` ``"unknown"`` with an explicit reason.

    Args:
        predicate: A :class:`TraceFailurePredicate` instance or its dict form.
        proposed_correction: Optional correction text; ``None`` means none proposed.
        correction_status: Correction review status; defaults to ``"unreviewed"``.

    Returns:
        A :class:`FailureDiagnosisRecord` adapted deterministically from the predicate.
    """
    _validate_correction_inputs(proposed_correction, correction_status)
    predicate_dict = _predicate_to_dict(predicate)
    predicate_id = str(predicate_dict.get("predicate_id", "")).strip()
    validity_status = str(predicate_dict.get("validity_status", "")).strip()
    mapping = _PREDICATE_DIAGNOSIS_MAP.get(predicate_id)
    failure_level = mapping[1] if mapping is not None else "analysis"

    if validity_status != _VALID_VALIDITY_STATUS:
        reason = f"predicate_validity_not_valid:{validity_status or 'empty'}"
        return unknown_failure_diagnosis_record(
            predicate_dict,
            reason,
            failure_level=failure_level,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )

    if mapping is None:
        reason = f"unsupported_predicate_id:{predicate_id or 'empty'}"
        return unknown_failure_diagnosis_record(
            predicate_dict,
            reason,
            failure_level=failure_level,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )

    failure_type, _ = mapping
    if failure_type == "unknown":
        reason = _UNKNOWN_TYPE_REASONS.get(
            predicate_id, f"unsupported_predicate_type_mapping:{predicate_id}"
        )
        return unknown_failure_diagnosis_record(
            predicate_dict,
            reason,
            failure_level=failure_level,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )

    onset_time_s, onset_interval = _onset_from_time_interval(predicate_dict.get("time_interval_s"))
    severity = _diagnosis_severity(predicate_dict.get("severity"), validity_status=validity_status)
    causal_evidence = _causal_evidence_from_predicate(predicate_dict)
    caveats = [_NON_CAUSAL_CAVEAT, _DIAGNOSTIC_LABEL_CAVEAT]
    return FailureDiagnosisRecord(
        diagnosis_schema_version=FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        diagnosis_source=DIAGNOSIS_SOURCE,
        failure_level=failure_level,
        failure_type=failure_type,
        onset_time_s=onset_time_s,
        onset_interval=onset_interval,
        severity=severity,
        detection_method=DETECTION_METHOD_PREDICATE,
        causal_evidence=causal_evidence,
        contributing_factors=[],
        confidence="supported_hypothesis",
        evidence_mode="direct_probe",
        validity_status=validity_status,
        proposed_correction=proposed_correction,
        correction_status=correction_status,
        unknown_reason=None,
        caveats=caveats,
        source_predicate=predicate_dict,
    )


def diagnose_from_trace_failure_predicates(
    predicates: list[TraceFailurePredicate | Mapping[str, Any]],
    *,
    proposed_correction: str | None = None,
    correction_status: str = DEFAULT_CORRECTION_STATUS,
) -> list[FailureDiagnosisRecord]:
    """Adapt a sequence of trace failure predicates into diagnosis records.

    Args:
        predicates: Iterable of predicate instances or dict forms.
        proposed_correction: Optional correction text applied to every record.
        correction_status: Correction review status applied to every record.

    Returns:
        List of :class:`FailureDiagnosisRecord` adapted deterministically, in order.
    """
    return [
        diagnose_from_trace_failure_predicate(
            predicate,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )
        for predicate in predicates
    ]


def unknown_failure_diagnosis_record(
    predicate: TraceFailurePredicate | Mapping[str, Any],
    reason: str,
    *,
    failure_level: str = "analysis",
    proposed_correction: str | None = None,
    correction_status: str = DEFAULT_CORRECTION_STATUS,
) -> FailureDiagnosisRecord:
    """Build an explicit ``unknown`` failure-diagnosis record from a predicate.

    Mirrors
    :func:`robot_sf.benchmark.failure_mechanism_taxonomy.unknown_failure_mechanism_record`:
    the ``failure_type``, ``confidence``, and ``evidence_mode`` are ``"unknown"`` and an
    explicit reason is recorded. Onset, severity, and evidence pointers are still
    derived from the predicate so the cited evidence stays traceable.

    Args:
        predicate: A predicate instance or dict form providing onset and evidence.
        reason: Non-empty reason explaining why the mapping is unknown.
        failure_level: Failure level for the unknown record; defaults to ``"analysis"``.
        proposed_correction: Optional correction text; ``None`` means none proposed.
        correction_status: Correction review status; defaults to ``"unreviewed"``.

    Returns:
        A :class:`FailureDiagnosisRecord` whose ``failure_type`` is ``"unknown"``.
    """
    _validate_correction_inputs(proposed_correction, correction_status)
    if failure_level not in FAILURE_LEVELS:
        raise FailureDiagnosisError(f"unsupported failure_level: {failure_level!r}")
    normalized_reason = str(reason).strip()
    if not normalized_reason:
        raise FailureDiagnosisError("unknown reason must be a non-empty string")
    predicate_dict = _predicate_to_dict(predicate)
    validity_status = str(predicate_dict.get("validity_status", "")).strip()
    onset_time_s, onset_interval = _onset_from_time_interval(predicate_dict.get("time_interval_s"))
    severity = _diagnosis_severity(predicate_dict.get("severity"), validity_status=validity_status)
    causal_evidence = _causal_evidence_from_predicate(predicate_dict)
    caveats = [
        _NON_CAUSAL_CAVEAT,
        _UNKNOWN_CAVEAT,
        f"unknown_reason: {normalized_reason}",
    ]
    return FailureDiagnosisRecord(
        diagnosis_schema_version=FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        diagnosis_source=DIAGNOSIS_SOURCE,
        failure_level=failure_level,
        failure_type="unknown",
        onset_time_s=onset_time_s,
        onset_interval=onset_interval,
        severity=severity,
        detection_method=DETECTION_METHOD_PREDICATE,
        causal_evidence=causal_evidence,
        contributing_factors=[],
        confidence="unknown",
        evidence_mode="unknown",
        validity_status=validity_status,
        proposed_correction=proposed_correction,
        correction_status=correction_status,
        unknown_reason=normalized_reason,
        caveats=caveats,
        source_predicate=predicate_dict,
    )


def validate_failure_diagnosis_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a failure-diagnosis record mapping.

    Enforces the schema contract: required fields are present, vocabulary fields are in
    their allowed sets, ``failure_type`` is in :data:`ALLOWED_FAILURE_TYPES`, and the
    ``unknown_reason`` invariant holds (set iff ``failure_type`` is ``"unknown"``).

    Args:
        record: A diagnosis record mapping (e.g. ``FailureDiagnosisRecord.to_dict()``).

    Returns:
        A shallow-copied, string-normalized mapping of the record.

    Raises:
        FailureDiagnosisError: If any contract field is missing or out of range.
    """
    if not isinstance(record, Mapping):
        raise FailureDiagnosisError("record must be a mapping")
    missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise FailureDiagnosisError(f"record missing required field(s): {missing}")
    _require_scalar_value(record, "diagnosis_schema_version", (FAILURE_DIAGNOSIS_SCHEMA_VERSION,))
    _require_scalar_value(record, "diagnosis_source", (DIAGNOSIS_SOURCE,))
    for field, allowed in _RECORD_VOCAB_FIELDS:
        _require_scalar_value(record, field, allowed)
    _require_collection_shapes(record)
    _require_unknown_reason_invariant(record)
    return _normalize_record(record)


def _require_scalar_value(
    record: Mapping[str, Any], field: str, allowed: tuple[str, ...] | frozenset[str]
) -> None:
    """Raise unless ``record[field]`` equals one of the ``allowed`` values.

    Args:
        record: A diagnosis record mapping.
        field: The field name to check.
        allowed: The allowed values for the field.

    Raises:
        FailureDiagnosisError: If the field value is not in ``allowed``.
    """
    value = record.get(field)
    if value not in allowed:
        raise FailureDiagnosisError(f"unsupported {field}: {value!r}")


def _require_collection_shapes(record: Mapping[str, Any]) -> None:
    """Validate the list/mapping shapes of collection-typed diagnosis fields.

    Args:
        record: A diagnosis record mapping.

    Raises:
        FailureDiagnosisError: If any collection field has the wrong shape.
    """
    onset = record["onset_interval"]
    if not isinstance(onset, list) or len(onset) != 2:
        raise FailureDiagnosisError("onset_interval must be a two-element list")
    _require_onset_consistency(record["onset_time_s"], onset)
    causal_evidence = record["causal_evidence"]
    if not isinstance(causal_evidence, list):
        raise FailureDiagnosisError("causal_evidence must be a list")
    for index, pointer in enumerate(causal_evidence):
        if not isinstance(pointer, Mapping):
            raise FailureDiagnosisError(
                f"causal_evidence[{index}] must be a mapping (evidence pointer)"
            )
    for field in ("contributing_factors", "caveats"):
        if not isinstance(record[field], list):
            raise FailureDiagnosisError(f"{field} must be a list")
    if not isinstance(record["source_predicate"], Mapping):
        raise FailureDiagnosisError("source_predicate must be a mapping")


def _require_onset_consistency(onset_time_s: Any, onset_interval: list[Any]) -> None:
    """Validate finite onset endpoints and their documented derivation relationship.

    ``onset_time_s`` is the first endpoint of ``onset_interval``.  Keeping that
    relationship in the validator prevents externally supplied payloads from
    contradicting the deterministic adapter's onset localization.

    Args:
        onset_time_s: Reported onset time.
        onset_interval: Two-element onset interval.

    Raises:
        FailureDiagnosisError: If an endpoint is non-finite/non-numeric or the
            onset time does not equal the interval start.
    """
    normalized_interval = [_finite_or_none(endpoint) for endpoint in onset_interval]
    if any(
        endpoint is not None and normalized is None
        for endpoint, normalized in zip(onset_interval, normalized_interval, strict=True)
    ):
        raise FailureDiagnosisError("onset_interval endpoints must be finite numbers or None")
    normalized_onset = _finite_or_none(onset_time_s)
    if onset_time_s is not None and normalized_onset is None:
        raise FailureDiagnosisError("onset_time_s must be a finite number or None")
    if normalized_onset != normalized_interval[0]:
        raise FailureDiagnosisError("onset_time_s must equal onset_interval[0]")


def _require_unknown_reason_invariant(record: Mapping[str, Any]) -> None:
    """Enforce that ``unknown_reason`` is set iff ``failure_type`` is ``"unknown"``.

    Args:
        record: A diagnosis record mapping.

    Raises:
        FailureDiagnosisError: If the unknown_reason invariant is violated.
    """
    unknown_reason = record["unknown_reason"]
    if record["failure_type"] == "unknown":
        if not isinstance(unknown_reason, str) or not unknown_reason.strip():
            raise FailureDiagnosisError("unknown_reason is required when failure_type is 'unknown'")
    elif unknown_reason is not None:
        raise FailureDiagnosisError(
            "unknown_reason must be None when failure_type is not 'unknown'"
        )


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``record`` with vocabulary fields string-normalized.

    Args:
        record: A validated diagnosis record mapping.

    Returns:
        A normalized dictionary copy.
    """
    normalized = dict(record)
    for field in _STRING_RECORD_FIELDS:
        normalized[field] = str(normalized[field])
    return normalized


def build_failure_diagnosis_payload(
    records: list[FailureDiagnosisRecord | Mapping[str, Any]],
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Wrap diagnosis records into a versioned ``failure_diagnosis.v1`` payload.

    Args:
        records: Diagnosis records (dataclass or dict form) to serialize.
        generated_at_utc: Optional explicit UTC timestamp; defaults to now.

    Returns:
        A versioned payload with ``schema_version``, ``diagnosis_source``, a
        ``records`` list, a ``failure_type_coverage`` summary, and payload caveats.
    """
    serialized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if isinstance(record, FailureDiagnosisRecord):
            record_dict = record.to_dict()
        elif isinstance(record, Mapping):
            record_dict = dict(record)
        else:  # pragma: no cover - defensive guard for misuse
            raise FailureDiagnosisError(f"records[{index}] must be a record or mapping")
        serialized.append(validate_failure_diagnosis_record(record_dict))
    coverage: dict[str, int] = {}
    for record_dict in serialized:
        key = str(record_dict["failure_type"])
        coverage[key] = coverage.get(key, 0) + 1
    return {
        "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        "diagnosis_source": DIAGNOSIS_SOURCE,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "records": serialized,
        "failure_type_coverage": {
            "counts": coverage,
            "classification_source": DIAGNOSIS_SOURCE,
        },
        "caveats": [_PAYLOAD_NON_CLAIM_CAVEAT, _NON_CAUSAL_CAVEAT, _OUT_OF_SCOPE_CAVEAT],
    }


def validate_failure_diagnosis_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a ``failure_diagnosis.v1`` payload and return a normalized copy.

    Args:
        payload: A payload mapping produced by :func:`build_failure_diagnosis_payload`.

    Returns:
        The validated payload mapping.

    Raises:
        FailureDiagnosisError: If the payload shape or any record is invalid.
    """
    if not isinstance(payload, Mapping):
        raise FailureDiagnosisError("payload must be a mapping")
    if payload.get("schema_version") != FAILURE_DIAGNOSIS_SCHEMA_VERSION:
        raise FailureDiagnosisError(
            "schema_version must be "
            f"{FAILURE_DIAGNOSIS_SCHEMA_VERSION!r}: {payload.get('schema_version')!r}"
        )
    if payload.get("diagnosis_source") != DIAGNOSIS_SOURCE:
        raise FailureDiagnosisError(f"diagnosis_source must be {DIAGNOSIS_SOURCE!r}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise FailureDiagnosisError("payload records must be a list")
    normalized_records = [validate_failure_diagnosis_record(record) for record in records]
    normalized = dict(payload)
    normalized["records"] = normalized_records
    return normalized


def _predicate_to_dict(predicate: TraceFailurePredicate | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a predicate instance or mapping to a plain dict view.

    Args:
        predicate: A :class:`TraceFailurePredicate`, another dataclass instance, or a
            mapping.

    Returns:
        Dictionary view of the predicate.

    Raises:
        FailureDiagnosisError: If the predicate is not a supported type.
    """
    if isinstance(predicate, Mapping):
        return dict(predicate)
    if hasattr(predicate, "to_dict"):
        return dict(predicate.to_dict())
    # Defensive fallback for plain dataclasses without a ``to_dict`` method.
    if is_dataclass(predicate) and not isinstance(predicate, type):
        return dict(asdict(predicate))
    raise FailureDiagnosisError(
        "predicate must be a TraceFailurePredicate, a mapping, or a dataclass instance"
    )


def _onset_from_time_interval(
    time_interval_s: Any,
) -> tuple[float | None, list[float | None]]:
    """Derive onset time and interval from a predicate ``time_interval_s``.

    Args:
        time_interval_s: A two-element ``[start, end]`` list (entries may be ``None``).

    Returns:
        A ``(onset_time_s, onset_interval)`` tuple. ``onset_time_s`` is the start
        value (or ``None``) and ``onset_interval`` is a two-element list.
    """
    if not isinstance(time_interval_s, (list, tuple)):
        return None, [None, None]
    start = time_interval_s[0] if len(time_interval_s) >= 1 else None
    end = time_interval_s[1] if len(time_interval_s) >= 2 else None
    start_value = _finite_or_none(start)
    end_value = _finite_or_none(end)
    return start_value, [start_value, end_value]


def _diagnosis_severity(severity: Any, *, validity_status: str) -> str:
    """Map a predicate severity token to a diagnosis severity deterministically.

    Args:
        severity: The source predicate severity value.
        validity_status: The source predicate validity status.

    Returns:
        A diagnosis severity in :data:`DIAGNOSIS_SEVERITIES`. Returns ``"unknown"``
        when the validity is not ``"valid"`` or the token is unrecognized.
    """
    if validity_status != _VALID_VALIDITY_STATUS:
        return "unknown"
    token = str(severity).strip().lower() if severity is not None else ""
    return _PREDICATE_SEVERITY_MAP.get(token, "unknown")


def _causal_evidence_from_predicate(predicate_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the causal_evidence pointer list from a predicate dict.

    The pointers cite the source predicate's evidence only and carry an explicit
    non-causal note. This is evidence citation, never causal inference.

    Args:
        predicate_dict: Dictionary view of the source predicate.

    Returns:
        A one-element list containing the cited trace/predicate evidence pointer.
    """
    time_interval_s = predicate_dict.get("time_interval_s")
    interval = list(time_interval_s) if isinstance(time_interval_s, (list, tuple)) else []
    steps = predicate_dict.get("steps")
    steps_list = list(steps) if isinstance(steps, (list, tuple)) else []
    return [
        {
            "evidence_kind": "trace_failure_predicate",
            "predicate_id": str(predicate_dict.get("predicate_id", "")),
            "time_interval_s": interval,
            "steps": steps_list,
            "involved_actors": list(predicate_dict.get("involved_actors", [])),
            "evidence_fields": dict(predicate_dict.get("evidence_fields", {})),
            "non_causal_note": _NON_CAUSAL_CAVEAT,
        }
    ]


def _validate_correction_inputs(proposed_correction: str | None, correction_status: str) -> None:
    """Validate correction-related inputs before building a record.

    Args:
        proposed_correction: Optional correction text; must be a string or ``None``.
        correction_status: Correction review status; must be in
            :data:`CORRECTION_STATUSES`.

    Raises:
        FailureDiagnosisError: If the correction status is unsupported or the proposed
            correction is not a string or ``None``.
    """
    if correction_status not in CORRECTION_STATUSES:
        raise FailureDiagnosisError(f"unsupported correction_status: {correction_status!r}")
    if proposed_correction is not None and not isinstance(proposed_correction, str):
        raise FailureDiagnosisError("proposed_correction must be a string or None")


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float, or ``None`` for non-finite or non-numeric values.

    Args:
        value: A value that may be numeric, ``None``, or non-numeric.

    Returns:
        The finite float value, or ``None``.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


__all__ = [
    "ALLOWED_FAILURE_TYPES",
    "CORRECTION_STATUSES",
    "DEFAULT_CORRECTION_STATUS",
    "DETECTION_METHODS",
    "DETECTION_METHOD_PREDICATE",
    "DIAGNOSIS_SEVERITIES",
    "DIAGNOSIS_SOURCE",
    "FAILURE_DIAGNOSIS_SCHEMA_VERSION",
    "FAILURE_LEVELS",
    "FailureDiagnosisError",
    "FailureDiagnosisRecord",
    "build_failure_diagnosis_payload",
    "diagnose_from_trace_failure_predicate",
    "diagnose_from_trace_failure_predicates",
    "unknown_failure_diagnosis_record",
    "validate_failure_diagnosis_payload",
    "validate_failure_diagnosis_record",
]
