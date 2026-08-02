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
from numbers import Integral, Real
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

#: Required top-level fields on every versioned diagnosis payload.
_REQUIRED_PAYLOAD_FIELDS = (
    "schema_version",
    "diagnosis_source",
    "generated_at_utc",
    "records",
    "failure_type_coverage",
    "caveats",
)

#: Exact fields carried by each trace-predicate evidence pointer.  The pointer keeps
#: source values (including malformed source values on an ``unknown`` record)
#: traceable without allowing callers to substitute an unstructured causal claim.
_CAUSAL_EVIDENCE_POINTER_FIELDS = frozenset(
    {
        "evidence_kind",
        "predicate_id",
        "time_interval_s",
        "steps",
        "involved_actors",
        "evidence_fields",
        "non_causal_note",
    }
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

#: Fields that the explicit unknown-record helper may conservatively override while
#: source, onset, evidence, and correction provenance remain adapter-derived.
_UNKNOWN_RECORD_METADATA_FIELDS = frozenset(
    {
        "failure_level",
        "failure_type",
        "confidence",
        "evidence_mode",
        "unknown_reason",
        "caveats",
    }
)

#: JSON-safe marker used in copied source evidence when the input contains a
#: non-finite or otherwise unsupported Python value. The marker is treated as
#: invalid predicate evidence and therefore cannot receive a known diagnosis.
_INVALID_JSON_VALUE_MARKER = "__failure_diagnosis_invalid_json_value__:"


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
    predicate_id = _predicate_text(predicate_dict, "predicate_id")
    validity_status = _predicate_text(predicate_dict, "validity_status")
    mapping = _PREDICATE_DIAGNOSIS_MAP.get(predicate_id)
    failure_level = mapping[1] if mapping is not None else "analysis"

    if _has_reversed_time_interval(predicate_dict.get("time_interval_s")):
        return unknown_failure_diagnosis_record(
            predicate_dict,
            "invalid_time_interval:end_precedes_start",
            failure_level=failure_level,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )

    if validity_status != _VALID_VALIDITY_STATUS:
        reason = f"predicate_validity_not_valid:{validity_status.strip() or 'empty'}"
        return unknown_failure_diagnosis_record(
            predicate_dict,
            reason,
            failure_level=failure_level,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )

    invalid_evidence_reason = _invalid_predicate_evidence_reason(predicate_dict)
    if invalid_evidence_reason is not None:
        return unknown_failure_diagnosis_record(
            predicate_dict,
            invalid_evidence_reason,
            failure_level=failure_level,
            proposed_correction=proposed_correction,
            correction_status=correction_status,
        )

    if mapping is None:
        reason = f"unsupported_predicate_id:{predicate_id.strip() or 'empty'}"
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
    if not isinstance(reason, str) or not (normalized_reason := reason.strip()):
        raise FailureDiagnosisError("unknown reason must be a non-empty string")
    predicate_dict = _predicate_to_dict(predicate)
    validity_status = _predicate_text(predicate_dict, "validity_status")
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

    Enforces the schema contract: required fields and collection shapes are present,
    vocabulary fields are strings in their allowed sets, optional correction text has
    its declared type, and the ``unknown_reason`` invariant holds (set iff
    ``failure_type`` is ``"unknown"``). Explicit unknown records may conservatively
    override only the unknown metadata fields accepted by
    :func:`unknown_failure_diagnosis_record`; source, onset, evidence, and correction
    provenance must still match the deterministic adapter.

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
    _require_adapter_provenance_consistency(record)
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
    if not isinstance(value, str) or value not in allowed:
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
        _require_causal_evidence_pointer(pointer, index)
    for field in ("contributing_factors", "caveats"):
        _require_string_list(record[field], field)
    if not isinstance(record["source_predicate"], Mapping):
        raise FailureDiagnosisError("source_predicate must be a mapping")
    if not isinstance(record["validity_status"], str):
        raise FailureDiagnosisError("validity_status must be a string")
    if record["proposed_correction"] is not None and not isinstance(
        record["proposed_correction"], str
    ):
        raise FailureDiagnosisError("proposed_correction must be a string or None")


def _require_causal_evidence_pointer(pointer: Mapping[str, Any], index: int) -> None:
    """Validate the trace-predicate pointer shape without rewriting source evidence.

    The deterministic adapter may preserve malformed source values inside pointer
    fields on an ``unknown`` record.  Its own pointer envelope must nevertheless
    remain exact so external records cannot pass schema validation with a causal
    assertion or an unrelated evidence object.

    Args:
        pointer: Candidate trace-predicate evidence pointer.
        index: Position in the record's ``causal_evidence`` list.

    Raises:
        FailureDiagnosisError: If the pointer is not the documented non-causal
            trace-predicate envelope.
    """
    fields = set(pointer)
    if fields != _CAUSAL_EVIDENCE_POINTER_FIELDS:
        raise FailureDiagnosisError(
            f"causal_evidence[{index}] must contain exactly the trace-predicate pointer fields"
        )
    if pointer["evidence_kind"] != "trace_failure_predicate":
        raise FailureDiagnosisError(
            f"causal_evidence[{index}].evidence_kind must be 'trace_failure_predicate'"
        )
    if not isinstance(pointer["predicate_id"], str):
        raise FailureDiagnosisError(f"causal_evidence[{index}].predicate_id must be a string")
    for field in ("time_interval_s", "steps", "involved_actors"):
        if not isinstance(pointer[field], list):
            raise FailureDiagnosisError(f"causal_evidence[{index}].{field} must be a list")
    if not isinstance(pointer["evidence_fields"], Mapping):
        raise FailureDiagnosisError(f"causal_evidence[{index}].evidence_fields must be a mapping")
    if pointer["non_causal_note"] != _NON_CAUSAL_CAVEAT:
        raise FailureDiagnosisError(
            f"causal_evidence[{index}].non_causal_note must preserve the non-causal boundary"
        )


def _require_string_list(value: Any, field: str) -> None:
    """Raise unless a record field is a list containing only strings.

    Args:
        value: Candidate list value.
        field: Field name used in the validation error.

    Raises:
        FailureDiagnosisError: If the value is not a list of strings.
    """
    if not isinstance(value, list):
        raise FailureDiagnosisError(f"{field} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise FailureDiagnosisError(f"{field} entries must be strings")


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
    if (
        normalized_interval[0] is not None
        and normalized_interval[1] is not None
        and normalized_interval[1] < normalized_interval[0]
    ):
        raise FailureDiagnosisError("onset_interval end must not precede onset_interval start")


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


def _require_adapter_provenance_consistency(record: Mapping[str, Any]) -> None:
    """Ensure a deterministic diagnosis remains tied to its source predicate.

    The version-one schema has a single deterministic source: the trace failure
    predicate retained in ``source_predicate``.  A valid record must therefore
    retain that predicate's validity text and include its exact non-causal
    evidence pointer.  Otherwise an externally edited record could present a
    known diagnosis as supported after its source became unavailable, or cite a
    different predicate as evidence. An explicit ``unknown`` record may
    conservatively override its classification metadata, but it may not alter
    source-derived evidence or claim a supported confidence/evidence mode.

    Args:
        record: A record that has already passed the structural validators.

    Raises:
        FailureDiagnosisError: If source provenance or fail-closed validity
            invariants are contradicted.
    """
    source_predicate = record["source_predicate"]
    source_validity_status = _predicate_text(source_predicate, "validity_status")
    if record["validity_status"] != source_validity_status:
        raise FailureDiagnosisError("validity_status must match source_predicate.validity_status")

    source_pointer = _causal_evidence_from_predicate(source_predicate)[0]
    if source_pointer not in record["causal_evidence"]:
        raise FailureDiagnosisError(
            "causal_evidence must include the exact source_predicate pointer"
        )

    if record["validity_status"] != _VALID_VALIDITY_STATUS:
        invalid_fields = {
            "failure_type": "unknown",
            "severity": "unknown",
            "confidence": "unknown",
            "evidence_mode": "unknown",
        }
        mismatched = [
            field for field, expected in invalid_fields.items() if record[field] != expected
        ]
        if mismatched:
            raise FailureDiagnosisError(
                "non-valid predicate evidence requires unknown failure_type, severity, "
                "confidence, and evidence_mode"
            )

    if record["failure_type"] == "unknown" and (
        record["confidence"] != "unknown" or record["evidence_mode"] != "unknown"
    ):
        raise FailureDiagnosisError(
            "unknown failure_type requires unknown confidence and evidence_mode"
        )

    expected_record = diagnose_from_trace_failure_predicate(
        source_predicate,
        proposed_correction=record["proposed_correction"],
        correction_status=record["correction_status"],
    ).to_dict()
    allowed_overrides = (
        _UNKNOWN_RECORD_METADATA_FIELDS if record["failure_type"] == "unknown" else frozenset()
    )
    mismatched_fields = [
        field
        for field in _REQUIRED_RECORD_FIELDS
        if field not in allowed_overrides and record[field] != expected_record[field]
    ]
    if mismatched_fields:
        raise FailureDiagnosisError(
            "record fields must match the deterministic adapter result for source_predicate: "
            f"{mismatched_fields}"
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
    coverage = _failure_type_coverage(serialized)
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
    records = _require_payload_metadata(payload)
    normalized_records = [validate_failure_diagnosis_record(record) for record in records]
    _validate_payload_coverage(payload["failure_type_coverage"], normalized_records)
    normalized = dict(payload)
    normalized["records"] = normalized_records
    coverage = payload["failure_type_coverage"]
    normalized["failure_type_coverage"] = {**dict(coverage), "counts": dict(coverage["counts"])}
    normalized["caveats"] = list(payload["caveats"])
    return normalized


def _require_payload_metadata(payload: Mapping[str, Any]) -> list[Any]:
    """Return payload records after validating the versioned payload metadata.

    Args:
        payload: Candidate failure-diagnosis payload.

    Returns:
        The unvalidated record list from a structurally valid payload.

    Raises:
        FailureDiagnosisError: If required payload metadata is absent or invalid.
    """
    missing = [field for field in _REQUIRED_PAYLOAD_FIELDS if field not in payload]
    if missing:
        raise FailureDiagnosisError(f"payload missing required field(s): {missing}")
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
    generated_at_utc = payload["generated_at_utc"]
    if not isinstance(generated_at_utc, str) or not generated_at_utc.strip():
        raise FailureDiagnosisError("generated_at_utc must be a non-empty string")
    caveats = payload["caveats"]
    if not isinstance(caveats, list) or not all(isinstance(caveat, str) for caveat in caveats):
        raise FailureDiagnosisError("payload caveats must be a list of strings")
    required_caveats = {
        _PAYLOAD_NON_CLAIM_CAVEAT,
        _NON_CAUSAL_CAVEAT,
        _OUT_OF_SCOPE_CAVEAT,
    }
    if not required_caveats.issubset(caveats):
        raise FailureDiagnosisError("payload caveats must preserve the diagnostic claim boundary")
    return records


def _validate_payload_coverage(coverage: Any, normalized_records: list[Mapping[str, Any]]) -> None:
    """Require payload coverage metadata to match its validated records exactly.

    Args:
        coverage: Candidate ``failure_type_coverage`` mapping.
        normalized_records: Validated payload records used to recompute counts.

    Raises:
        FailureDiagnosisError: If coverage provenance or counts are invalid.
    """
    if not isinstance(coverage, Mapping):
        raise FailureDiagnosisError("failure_type_coverage must be a mapping")
    if coverage.get("classification_source") != DIAGNOSIS_SOURCE:
        raise FailureDiagnosisError(
            f"failure_type_coverage classification_source must be {DIAGNOSIS_SOURCE!r}"
        )
    counts = coverage.get("counts")
    if not isinstance(counts, Mapping) or any(
        not isinstance(count, int) or isinstance(count, bool) for count in counts.values()
    ):
        raise FailureDiagnosisError("failure_type_coverage counts must map labels to integers")
    expected_coverage = _failure_type_coverage(normalized_records)
    if dict(counts) != expected_coverage:
        raise FailureDiagnosisError("failure_type_coverage counts must match validated records")


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
        predicate_dict = dict(predicate)
    elif callable(getattr(predicate, "to_dict", None)):
        raw_predicate = predicate.to_dict()
        if not isinstance(raw_predicate, Mapping):
            raise FailureDiagnosisError("predicate.to_dict() must return a mapping")
        predicate_dict = dict(raw_predicate)
    # Defensive fallback for plain dataclasses without a ``to_dict`` method.
    elif is_dataclass(predicate) and not isinstance(predicate, type):
        predicate_dict = dict(asdict(predicate))
    else:
        raise FailureDiagnosisError(
            "predicate must be a TraceFailurePredicate, a mapping, or a dataclass instance"
        )
    return _json_safe_value(predicate_dict)


def _predicate_text(predicate_dict: Mapping[str, Any], field: str) -> str:
    """Return a predicate text field without normalizing away invalid evidence.

    String-valued predicate identifiers and validity statuses are preserved verbatim so
    non-canonical values fail closed instead of being silently accepted as valid evidence.
    Non-string values are stringified only to keep the diagnosis record's documented
    string field shape.
    """
    value = predicate_dict.get(field, "")
    return value if isinstance(value, str) else str(value)


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
    if start_value is not None and end_value is not None and end_value < start_value:
        return None, [None, None]
    return start_value, [start_value, end_value]


def _invalid_predicate_evidence_reason(predicate_dict: Mapping[str, Any]) -> str | None:
    """Return an explicit unknown-diagnosis reason for malformed predicate evidence.

    A mapping input may bypass the :class:`TraceFailurePredicate` dataclass.  When a
    source still claims ``valid`` evidence but lacks the adapter's required
    trace/predicate shapes, fail closed to an unknown diagnosis instead of assigning
    a confident classifier label or leaking a raw container ``TypeError``.

    Args:
        predicate_dict: Dictionary view of the source predicate.

    Returns:
        A deterministic invalid-evidence reason, or ``None`` when the required
        predicate evidence shapes are present.
    """
    if _contains_invalid_json_marker(predicate_dict):
        return "invalid_predicate_evidence:non_json_safe_value"
    time_interval_s = predicate_dict.get("time_interval_s")
    if not isinstance(time_interval_s, (list, tuple)) or len(time_interval_s) != 2:
        return "invalid_predicate_evidence:time_interval_s_not_two_element_sequence"
    if any(
        endpoint is not None and _finite_or_none(endpoint) is None for endpoint in time_interval_s
    ):
        return "invalid_predicate_evidence:time_interval_s_non_finite_or_non_numeric"
    steps = predicate_dict.get("steps")
    if (
        not isinstance(steps, (list, tuple))
        or len(steps) != 2
        or any(
            step is not None and (isinstance(step, bool) or not isinstance(step, int))
            for step in steps
        )
    ):
        return "invalid_predicate_evidence:steps_not_two_element_integer_or_none_sequence"
    involved_actors = predicate_dict.get("involved_actors")
    if not isinstance(involved_actors, (list, tuple)) or not all(
        isinstance(actor, str) for actor in involved_actors
    ):
        return "invalid_predicate_evidence:involved_actors_not_string_sequence"
    if not isinstance(predicate_dict.get("evidence_fields"), Mapping):
        return "invalid_predicate_evidence:evidence_fields_not_mapping"
    return None


def _has_reversed_time_interval(time_interval_s: Any) -> bool:
    """Return whether a finite predicate interval ends before its start.

    A reversed interval cannot support deterministic onset localization. The adapter
    preserves the raw predicate in its evidence pointer, but emits an ``unknown``
    diagnosis with a schema-valid absent onset rather than a record that contradicts
    :func:`validate_failure_diagnosis_record`.

    Args:
        time_interval_s: Candidate two-element predicate time interval.

    Returns:
        ``True`` only when both finite endpoints are present and the end precedes
        the start.
    """
    if not isinstance(time_interval_s, (list, tuple)) or len(time_interval_s) < 2:
        return False
    start_value = _finite_or_none(time_interval_s[0])
    end_value = _finite_or_none(time_interval_s[1])
    return start_value is not None and end_value is not None and end_value < start_value


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
    involved_actors = predicate_dict.get("involved_actors")
    actors_list = list(involved_actors) if isinstance(involved_actors, (list, tuple)) else []
    evidence_fields = predicate_dict.get("evidence_fields")
    evidence_dict = dict(evidence_fields) if isinstance(evidence_fields, Mapping) else {}
    return [
        {
            "evidence_kind": "trace_failure_predicate",
            "predicate_id": str(predicate_dict.get("predicate_id", "")),
            "time_interval_s": interval,
            "steps": steps_list,
            "involved_actors": actors_list,
            "evidence_fields": evidence_dict,
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


def _failure_type_coverage(records: list[Mapping[str, Any]]) -> dict[str, int]:
    """Return deterministic failure-type counts for validated diagnosis records.

    Args:
        records: Validated diagnosis records.

    Returns:
        Count of each failure type present in the supplied record order.
    """
    coverage: dict[str, int] = {}
    for record in records:
        failure_type = str(record["failure_type"])
        coverage[failure_type] = coverage.get(failure_type, 0) + 1
    return coverage


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float, or ``None`` for non-finite or non-numeric values.

    Args:
        value: A value that may be numeric, ``None``, or non-numeric.

    Returns:
        The finite float value, or ``None``.
    """
    if isinstance(value, bool) or value is None or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _json_safe_value(value: Any, *, _active_ids: set[int] | None = None) -> Any:
    """Return a JSON-safe copy, marking values that invalidate predicate evidence.

    The predicate adapter accepts mapping forms from outside the typed dataclass boundary.  Keep
    that boundary fail closed when a numeric object cannot be converted to a finite float or when
    nested evidence contains a reference cycle.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            return f"{_INVALID_JSON_VALUE_MARKER}numeric_conversion_error"
        return number if isfinite(number) else f"{_INVALID_JSON_VALUE_MARKER}non_finite"
    active_ids = _active_ids if _active_ids is not None else set()
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, active_ids)
    if isinstance(value, (list, tuple)):
        return _json_safe_sequence(value, active_ids)
    return f"{_INVALID_JSON_VALUE_MARKER}{type(value).__module__}.{type(value).__qualname__}"


def _json_safe_mapping(value: Mapping[Any, Any], active_ids: set[int]) -> dict[str, Any]:
    """Return a JSON-safe mapping copy while marking cycles and invalid keys."""
    value_id = id(value)
    if value_id in active_ids:
        return {"value": f"{_INVALID_JSON_VALUE_MARKER}cyclic_reference"}
    active_ids.add(value_id)
    safe_mapping: dict[str, Any] = {}
    try:
        for key, item in value.items():
            if isinstance(key, str):
                safe_key = key
            else:
                safe_key = (
                    f"{_INVALID_JSON_VALUE_MARKER}non_string_key:"
                    f"{type(key).__module__}.{type(key).__qualname__}"
                )
            safe_mapping[safe_key] = _json_safe_value(item, _active_ids=active_ids)
        return safe_mapping
    finally:
        active_ids.remove(value_id)


def _json_safe_sequence(
    value: list[Any] | tuple[Any, ...], active_ids: set[int]
) -> list[Any] | str:
    """Return a JSON-safe sequence copy while marking cycles."""
    value_id = id(value)
    if value_id in active_ids:
        return f"{_INVALID_JSON_VALUE_MARKER}cyclic_reference"
    active_ids.add(value_id)
    try:
        return [_json_safe_value(item, _active_ids=active_ids) for item in value]
    finally:
        active_ids.remove(value_id)


def _contains_invalid_json_marker(value: Any) -> bool:
    """Return whether a copied predicate contains a JSON-safety invalid marker."""
    if isinstance(value, str):
        return value.startswith(_INVALID_JSON_VALUE_MARKER)
    if isinstance(value, Mapping):
        return any(
            _contains_invalid_json_marker(key) or _contains_invalid_json_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_invalid_json_marker(item) for item in value)
    return False


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
