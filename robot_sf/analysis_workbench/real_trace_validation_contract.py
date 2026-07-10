"""Validation-contract checker for candidate real-world micromobility traces.

This module answers a single, narrow question for issue #3278:

    Given a *metadata-only* descriptor of a candidate real-world micromobility
    dataset (its provenance, available sensor channels, and directly observed
    event labels), which Robot SF trace-failure predicates could be validated
    against it, and which are blocked by missing data?

It deliberately does **not** ingest, copy, download, or read any external or
private trace data, and it makes **no** real-world validation claim. It only
maps a declared field/label inventory onto the existing predicate input
contract (`build_trace_failure_predicate_definitions`) so future data, once
access and provenance are accepted, can be wired up without guesswork.

Two distinct notions are reported per predicate, because they fail differently
for real data:

* **channel compatibility** -- whether every required sensor/geometry channel
  the predicate needs is declared available. A channel-compatible predicate can
  be *computed* from the dataset.
* **ground-truth label observability** -- whether the dataset directly annotates
  the behavior (e.g. a human-coded ``late_evasive`` label). Many predicates are
  derived from kinematics and have no directly observed label, so they can be
  computed but not cross-checked against an observed label. This is the
  "labels that cannot be observed directly" limitation called out in the issue.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_IDS,
    build_trace_failure_predicate_definitions,
    matrix_required_fields_for_predicate,
)
from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION = "real_trace_validation_contract.v1"
REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "real_trace_validation_contract.v1.json"
)

#: Explicit boundary stamped on every report so downstream readers cannot mistake
#: a passing contract check for measured real-world validation.
CONTRACT_EVIDENCE_BOUNDARY = "contract_check_only_no_real_world_validation"

#: Predicate inputs that are themselves explicit labels/events rather than
#: kinematic or geometric channels. When a predicate depends on one of these, a
#: real dataset must directly annotate it; it cannot be recovered from motion.
LABEL_DERIVED_CHANNELS = frozenset(
    {
        "planner.event",
        "planner.occlusion_or_visibility",
    }
)

#: Common dataset label vocabularies mapped onto predicate IDs so a descriptor
#: can declare human-readable labels (e.g. ``near_miss``) and still match the
#: canonical predicate it cross-validates.
DEFAULT_LABEL_ALIASES: dict[str, str] = {
    "late_evasive": "late_evasive_reaction",
    "late_evasive_reaction": "late_evasive_reaction",
    "late_braking": "late_evasive_reaction",
    "oscillatory": "oscillatory_local_control",
    "oscillation": "oscillatory_local_control",
    "oscillatory_local_control": "oscillatory_local_control",
    "near_miss": "occlusion_triggered_near_miss",
    "occlusion_near_miss": "occlusion_triggered_near_miss",
    "occlusion_triggered_near_miss": "occlusion_triggered_near_miss",
    "collision": "collision",
    "deadlock": "bottleneck_deadlock",
    "bottleneck_deadlock": "bottleneck_deadlock",
    "zero_motion": "zero_motion_timeout_behavior",
    "zero_motion_timeout_behavior": "zero_motion_timeout_behavior",
    "low_progress": "low_progress",
    "clearance_critical": "clearance_critical_interaction",
    "clearance_critical_interaction": "clearance_critical_interaction",
}

PREDICATE_STATUS_VALIDATABLE = "validatable"
PREDICATE_STATUS_BLOCKED = "blocked"

CONTRACT_STATUS_READY = "ready"
CONTRACT_STATUS_BLOCKED = "blocked"


class RealTraceValidationContractError(RobotSfError, ValueError):
    """Raised when a real-trace validation-contract descriptor fails schema checks."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema messages."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class PredicateCompatibility:
    """Per-predicate compatibility result for a candidate dataset descriptor."""

    predicate_id: str
    status: str
    required_channels: list[str]
    missing_channels: list[str]
    label_derived_channels: list[str]
    ground_truth_label_available: bool
    limitation: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RealTraceValidationContractReport:
    """Result of checking a candidate dataset descriptor against the predicate contract."""

    schema_version: str
    dataset_id: str
    contract_status: str
    evidence_boundary: str
    metadata_status: str
    metadata_blockers: list[str]
    provenance_blockers: list[str]
    predicate_compatibility: list[PredicateCompatibility]
    validatable_predicates: list[str]
    blocked_predicates: list[str]
    missing_data_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        payload = asdict(self)
        payload["predicate_compatibility"] = [
            item.to_dict() if isinstance(item, PredicateCompatibility) else item
            for item in self.predicate_compatibility
        ]
        return payload


@lru_cache(maxsize=1)
def load_real_trace_validation_contract_schema() -> dict[str, Any]:
    """Load the descriptor JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _required_channels_by_predicate() -> dict[str, list[str]]:
    """Return canonical required channels per predicate from the predicate definitions.

    Reuses ``build_trace_failure_predicate_definitions`` so this checker stays
    bound to the single source of truth for predicate inputs rather than
    re-listing channel requirements.

    Returns:
        Mapping of predicate ID to its required channel list.
    """
    return {
        definition["predicate_id"]: list(definition["inputs"])
        for definition in build_trace_failure_predicate_definitions()
    }


def load_real_trace_validation_contract(path: str | Path) -> dict[str, Any]:
    """Load and schema-validate a descriptor from JSON or YAML.

    Returns:
        The validated descriptor mapping.

    Raises:
        RealTraceValidationContractError: when the file is missing or invalid.
    """
    descriptor_path = Path(path)
    if not descriptor_path.is_file():
        raise RealTraceValidationContractError(
            ["descriptor file not found"], source=descriptor_path
        )
    text = descriptor_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise RealTraceValidationContractError(
            [f"invalid YAML/JSON: {exc}"], source=descriptor_path
        ) from exc
    if not isinstance(payload, Mapping):
        raise RealTraceValidationContractError(
            ["expected a mapping payload"], source=descriptor_path
        )
    _raise_on_schema_errors(payload, source=descriptor_path)
    return dict(payload)


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise ``RealTraceValidationContractError`` if the payload violates the schema."""
    validator = Draft202012Validator(load_real_trace_validation_contract_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise RealTraceValidationContractError(errors, source=source)


def check_real_trace_validation_contract(
    descriptor: Mapping[str, Any],
    *,
    matrix: Mapping[str, Any] | None = None,
    label_aliases: Mapping[str, str] | None = None,
    source: str | Path | None = None,
) -> RealTraceValidationContractReport:
    """Check a candidate dataset descriptor against the predicate input contract.

    Args:
        descriptor: A ``real_trace_validation_contract.v1`` mapping. Schema-validated here.
        matrix: Optional ``trace_predicate_matrix.v1`` payload. When supplied, its
            ``required_trace_fields_by_predicate`` is unioned with the predicate
            definition inputs so a predicate is only validatable when it also
            satisfies the rate-interpretation matrix.
        label_aliases: Optional override for the dataset-label vocabulary.
        source: Optional source path for error messages.

    Returns:
        A structured compatibility report. The report never asserts real-world
        validation; ``evidence_boundary`` is always
        ``contract_check_only_no_real_world_validation``.

    Raises:
        RealTraceValidationContractError: when the descriptor violates the schema.
    """
    _raise_on_schema_errors(descriptor, source=source)

    aliases = dict(DEFAULT_LABEL_ALIASES if label_aliases is None else label_aliases)
    metadata = descriptor.get("metadata", {})
    available_channels = set(descriptor.get("available_channels", []))
    observed_labels = _normalize_labels(descriptor.get("available_event_labels", []), aliases)

    metadata_blockers = _metadata_blockers(metadata)
    provenance_blockers = _provenance_blockers(metadata)

    target_predicates = _resolve_target_predicates(descriptor.get("target_predicates"))
    required_by_predicate = _required_channels_by_predicate()

    compatibility: list[PredicateCompatibility] = []
    for predicate_id in target_predicates:
        compatibility.append(
            _predicate_compatibility(
                predicate_id=predicate_id,
                required_channels=_required_channels(predicate_id, required_by_predicate, matrix),
                available_channels=available_channels,
                observed_labels=observed_labels,
            )
        )

    validatable = [
        c.predicate_id for c in compatibility if c.status == PREDICATE_STATUS_VALIDATABLE
    ]
    blocked = [c.predicate_id for c in compatibility if c.status == PREDICATE_STATUS_BLOCKED]

    missing_data_blockers = _aggregate_missing_data_blockers(compatibility)

    metadata_status = "complete" if not metadata_blockers else "incomplete"
    contract_ready = not metadata_blockers and not provenance_blockers and bool(validatable)
    contract_status = CONTRACT_STATUS_READY if contract_ready else CONTRACT_STATUS_BLOCKED

    return RealTraceValidationContractReport(
        schema_version=REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION,
        dataset_id=str(descriptor["dataset_id"]),
        contract_status=contract_status,
        evidence_boundary=CONTRACT_EVIDENCE_BOUNDARY,
        metadata_status=metadata_status,
        metadata_blockers=metadata_blockers,
        provenance_blockers=provenance_blockers,
        predicate_compatibility=compatibility,
        validatable_predicates=validatable,
        blocked_predicates=blocked,
        missing_data_blockers=missing_data_blockers,
    )


def _resolve_target_predicates(raw: Any) -> list[str]:
    """Return target predicate IDs, defaulting to all canonical predicates.

    Returns:
        Ordered, de-duplicated predicate IDs constrained to the canonical set.

    Raises:
        RealTraceValidationContractError: when an unknown predicate ID is requested.
    """
    if not raw:
        return list(TRACE_FAILURE_PREDICATE_IDS)
    requested = list(dict.fromkeys(str(item) for item in raw))
    unknown = [item for item in requested if item not in TRACE_FAILURE_PREDICATE_IDS]
    if unknown:
        raise RealTraceValidationContractError(
            [f"unknown target predicate(s): {', '.join(sorted(unknown))}"]
        )
    # Preserve canonical ordering for stable reports.
    return [pid for pid in TRACE_FAILURE_PREDICATE_IDS if pid in set(requested)]


def _required_channels(
    predicate_id: str,
    required_by_predicate: Mapping[str, list[str]],
    matrix: Mapping[str, Any] | None,
) -> list[str]:
    """Return required channels for a predicate, unioning definition inputs and matrix fields.

    Returns:
        Sorted unique channel list.
    """
    channels = set(required_by_predicate.get(predicate_id, []))
    if matrix is not None:
        channels.update(matrix_required_fields_for_predicate(matrix, predicate_id))
    return sorted(channels)


def _predicate_compatibility(
    *,
    predicate_id: str,
    required_channels: list[str],
    available_channels: set[str],
    observed_labels: set[str],
) -> PredicateCompatibility:
    """Compute the compatibility record for one predicate.

    Returns:
        A populated :class:`PredicateCompatibility`.
    """
    missing = [channel for channel in required_channels if channel not in available_channels]
    label_derived = [c for c in required_channels if c in LABEL_DERIVED_CHANNELS]
    status = PREDICATE_STATUS_VALIDATABLE if not missing else PREDICATE_STATUS_BLOCKED
    ground_truth = predicate_id in observed_labels

    limitation = _predicate_limitation(
        status=status,
        missing=missing,
        label_derived=label_derived,
        ground_truth=ground_truth,
    )
    return PredicateCompatibility(
        predicate_id=predicate_id,
        status=status,
        required_channels=required_channels,
        missing_channels=missing,
        label_derived_channels=label_derived,
        ground_truth_label_available=ground_truth,
        limitation=limitation,
    )


def _predicate_limitation(
    *,
    status: str,
    missing: list[str],
    label_derived: list[str],
    ground_truth: bool,
) -> str | None:
    """Return a human-readable limitation note, or None when fully validatable.

    Returns:
        A limitation string, or None.
    """
    if status == PREDICATE_STATUS_BLOCKED:
        return f"missing required channel(s): {', '.join(missing)}"
    if label_derived:
        return (
            "depends on explicitly annotated channel(s) "
            f"{', '.join(label_derived)}; only as reliable as the dataset's labels"
        )
    if not ground_truth:
        return (
            "computable from channels but no directly observed ground-truth label "
            "is declared for cross-validation"
        )
    return None


def _normalize_labels(raw_labels: Any, aliases: Mapping[str, str]) -> set[str]:
    """Map declared dataset labels onto canonical predicate IDs.

    Returns:
        Set of canonical predicate IDs that the dataset directly annotates.
    """
    normalized: set[str] = set()
    for label in raw_labels or []:
        key = str(label).strip().lower()
        if key in aliases:
            normalized.add(aliases[key])
        elif key in TRACE_FAILURE_PREDICATE_IDS:
            normalized.add(key)
    return normalized


#: Placeholder license values that pass the schema's non-empty check but mean the
#: license has not actually been resolved.
PLACEHOLDER_LICENSE_VALUES = frozenset({"unknown", "tbd", "unspecified", "n/a", "none"})


def _metadata_blockers(metadata: Mapping[str, Any]) -> list[str]:
    """Return blockers for placeholder metadata that passes schema but is not usable.

    The schema already guarantees the required metadata keys are present and
    non-empty, so this layer only catches semantic placeholders the schema cannot
    detect (currently an unresolved license sentinel).

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    license_value = str(metadata.get("license", "")).strip().lower()
    if license_value in PLACEHOLDER_LICENSE_VALUES:
        blockers.append("metadata.license is not yet specified")
    return sorted(blockers)


def _provenance_blockers(metadata: Mapping[str, Any]) -> list[str]:
    """Return blockers tied to provenance and access acceptance.

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    provenance = str(metadata.get("provenance_status", "")).strip().lower()
    access = str(metadata.get("access_status", "")).strip().lower()
    if provenance != "accepted":
        blockers.append(
            f"provenance_status is {provenance or 'missing'!r}; must be 'accepted' before use"
        )
    if access != "available":
        blockers.append(f"access_status is {access or 'missing'!r}; data is not yet accessible")
    return sorted(blockers)


def _aggregate_missing_data_blockers(
    compatibility: list[PredicateCompatibility],
) -> list[str]:
    """Aggregate explicit per-predicate missing-data blockers.

    Returns:
        Sorted blocker strings naming each blocked predicate and its gaps.
    """
    blockers: list[str] = []
    for record in compatibility:
        if record.status == PREDICATE_STATUS_BLOCKED:
            blockers.append(f"{record.predicate_id}: missing {', '.join(record.missing_channels)}")
    return sorted(blockers)
