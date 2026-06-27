"""Intake-manifest checker for AMV actuation-latency and rider-coupling measurement (issue #3283).

This module answers a single, narrow question for issue #3283:

    Given a *metadata-only* manifest describing how an autonomous micromobility
    vehicle (AMV) actuation-latency and rider-coupling measurement campaign is
    (or would be) collected, is the intake plan complete, what is still blocked,
    and is the manifest allowed to assert a measured value yet?

It deliberately does **not** collect, ingest, download, fabricate, or read any
real command-response data, and it makes **no** measured-value or
calibrated-actuation claim. It only checks the declared intake plan -- sensor
channels, sampling rate, time synchronization, provenance, and the
synthetic-vs-measured separation -- against the canonical measurement-quantity
contract so future data, once staged and reviewed, can be wired up without
guesswork.

The manifest also proposes the latency and rider-coupling fields that a future
measured AMV actuation profile would expose (see ``PROPOSED_LATENCY_PROFILE_FIELDS``
and ``PROPOSED_RIDER_COUPLING_PROFILE_FIELDS``). These extend the synthetic
actuation-envelope schema in :mod:`robot_sf.benchmark.synthetic_actuation`
without promoting synthetic placeholders into measured calibration evidence.

Three measurement lifecycle states are handled, because they gate claims
differently:

* ``blocked-external-input`` -- a valid intake *plan* that is awaiting external
  command-response data. Every proposed field stays ``pending``; no measured
  value is permitted. This is the default for issue #3283 today.
* ``synthetic-only`` -- placeholder assumptions kept strictly separate from
  measured values. Every proposed field stays ``synthetic-placeholder``; no
  provenance source and no measured value are permitted.
* ``measured`` -- real command-response data has been collected and provenance
  accepted. Only then may every proposed field be ``measured`` and the manifest
  assert measured values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer

AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_VERSION = (
    "amv_actuation_latency_measurement_manifest.v1"
)
AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "amv_actuation_latency_measurement_manifest.v1.json"
)

#: Explicit boundary stamped on every report so a passing intake check is never
#: mistaken for measured real-world actuation evidence.
MEASUREMENT_INTAKE_EVIDENCE_BOUNDARY = "measurement_intake_plan_only_no_measured_value_claim"

#: Canonical command-response latency quantities the intake must observe, drawn
#: from the issue #3283 protocol (command issuance -> mechanical/yaw response,
#: braking and acceleration latency).
LATENCY_MEASUREMENT_QUANTITIES: tuple[str, ...] = (
    "command_issuance",
    "mechanical_response",
    "yaw_response",
    "braking_latency",
    "acceleration_latency",
)

#: Canonical rider-coupling quantities the intake must observe (rider/load
#: condition and the rider's response coupling into the vehicle).
RIDER_COUPLING_MEASUREMENT_QUANTITIES: tuple[str, ...] = (
    "rider_load_condition",
    "rider_response",
)

#: All measurement quantities a complete intake plan must declare a channel for.
REQUIRED_MEASUREMENT_QUANTITIES: tuple[str, ...] = (
    LATENCY_MEASUREMENT_QUANTITIES + RIDER_COUPLING_MEASUREMENT_QUANTITIES
)

#: Proposed latency fields a future measured AMV actuation profile would expose,
#: mapped to their canonical units. Extends the synthetic actuation-envelope
#: schema without asserting any measured value.
PROPOSED_LATENCY_PROFILE_FIELDS: dict[str, str] = {
    "command_to_motion_latency_s": "s",
    "command_to_yaw_latency_s": "s",
    "braking_onset_latency_s": "s",
    "acceleration_onset_latency_s": "s",
}

#: Proposed rider-coupling fields a future measured AMV actuation profile would
#: expose, mapped to their canonical units.
PROPOSED_RIDER_COUPLING_PROFILE_FIELDS: dict[str, str] = {
    "rider_load_kg": "kg",
    "rider_coupling_gain": "dimensionless",
    "rider_response_latency_s": "s",
}

#: Canonical ``parameter_class`` each proposed profile field must declare, so a
#: latency field is never mislabeled as rider-coupling (or vice versa).
PROPOSED_FIELD_PARAMETER_CLASS: dict[str, str] = {
    **dict.fromkeys(PROPOSED_LATENCY_PROFILE_FIELDS, "latency"),
    **dict.fromkeys(PROPOSED_RIDER_COUPLING_PROFILE_FIELDS, "rider_coupling"),
}

#: Provenance fields required before a ``measured`` manifest may claim a value.
MEASURED_MANIFEST_REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_uri",
    "source_type",
    "measurement_date",
    "units",
)

#: Per-status the single ``value_status`` a proposed profile field must carry, so
#: synthetic assumptions never conflate with measured values.
_VALUE_STATUS_BY_MEASUREMENT_STATUS: dict[str, str] = {
    "blocked-external-input": "pending",
    "synthetic-only": "synthetic-placeholder",
    "measured": "measured",
}

CONTRACT_STATUS_READY = "ready"
CONTRACT_STATUS_BLOCKED = "blocked"
CONTRACT_STATUS_SYNTHETIC_ONLY = "synthetic-only"


class AmvActuationLatencyManifestError(ValueError):
    """Raised when an AMV actuation-latency measurement manifest fails schema checks."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema messages."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class AmvActuationLatencyManifestReport:
    """Result of checking an AMV actuation-latency measurement intake manifest."""

    schema_version: str
    manifest_id: str
    measurement_status: str
    contract_status: str
    evidence_boundary: str
    declared_quantities: list[str]
    missing_latency_quantities: list[str]
    missing_rider_coupling_quantities: list[str]
    synchronization_blockers: list[str]
    provenance_blockers: list[str]
    separation_blockers: list[str]
    proposed_field_blockers: list[str]
    measured_value_claim_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)

    @property
    def blockers(self) -> list[str]:
        """Return all blockers aggregated across categories, sorted."""
        return sorted(
            self.missing_latency_quantities
            + self.missing_rider_coupling_quantities
            + self.synchronization_blockers
            + self.provenance_blockers
            + self.separation_blockers
            + self.proposed_field_blockers
        )


@lru_cache(maxsize=1)
def load_amv_actuation_latency_measurement_manifest_schema() -> dict[str, Any]:
    """Load the manifest JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(
        AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_FILE.read_text(encoding="utf-8")
    )


def load_amv_actuation_latency_measurement_manifest(path: str | Path) -> dict[str, Any]:
    """Load and schema-validate a manifest from JSON or YAML.

    Returns:
        The validated manifest mapping.

    Raises:
        AmvActuationLatencyManifestError: when the file is missing or invalid.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise AmvActuationLatencyManifestError(["manifest file not found"], source=manifest_path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise AmvActuationLatencyManifestError(
            [f"invalid YAML/JSON: {exc}"], source=manifest_path
        ) from exc
    if not isinstance(payload, Mapping):
        raise AmvActuationLatencyManifestError(["expected a mapping payload"], source=manifest_path)
    _raise_on_schema_errors(payload, source=manifest_path)
    return dict(payload)


@lru_cache(maxsize=1)
def _manifest_validator() -> Draft202012Validator:
    """Return a cached schema validator.

    Compiling the schema and resolving references is comparatively expensive, so
    the validator is built once and reused across manifests (e.g. when checking
    several manifests in a loop or in CI).
    """
    return Draft202012Validator(load_amv_actuation_latency_measurement_manifest_schema())


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise ``AmvActuationLatencyManifestError`` if the payload violates the schema."""
    validator = _manifest_validator()
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise AmvActuationLatencyManifestError(errors, source=source)


def check_amv_actuation_latency_measurement_manifest(
    manifest: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> AmvActuationLatencyManifestReport:
    """Check an AMV actuation-latency measurement intake manifest.

    Args:
        manifest: An ``amv_actuation_latency_measurement_manifest.v1`` mapping.
            Schema-validated here.
        source: Optional source path for error messages.

    Returns:
        A structured intake report. The report never asserts measured actuation
        evidence; ``evidence_boundary`` is always
        ``measurement_intake_plan_only_no_measured_value_claim``, and
        ``measured_value_claim_allowed`` is only True for a complete ``measured``
        manifest with accepted provenance.

    Raises:
        AmvActuationLatencyManifestError: when the manifest violates the schema.
    """
    _raise_on_schema_errors(manifest, source=source)

    measurement_status = str(manifest["measurement_status"])
    declared_quantities = sorted(
        {str(channel["quantity"]) for channel in manifest.get("sensor_channels", [])}
    )

    missing_latency = [q for q in LATENCY_MEASUREMENT_QUANTITIES if q not in declared_quantities]
    missing_rider = [
        q for q in RIDER_COUPLING_MEASUREMENT_QUANTITIES if q not in declared_quantities
    ]

    synchronization_blockers = _synchronization_blockers(manifest.get("synchronization", {}))
    separation_blockers = _separation_blockers(manifest.get("synthetic_separation", {}))
    proposed_field_blockers = _proposed_field_blockers(
        manifest.get("proposed_profile_fields", []), measurement_status=measurement_status
    )
    provenance_blockers = _provenance_blockers(
        manifest.get("provenance"), measurement_status=measurement_status
    )

    report = AmvActuationLatencyManifestReport(
        schema_version=AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_VERSION,
        manifest_id=str(manifest["manifest_id"]),
        measurement_status=measurement_status,
        contract_status=CONTRACT_STATUS_BLOCKED,  # placeholder, resolved below
        evidence_boundary=MEASUREMENT_INTAKE_EVIDENCE_BOUNDARY,
        declared_quantities=declared_quantities,
        missing_latency_quantities=[f"missing latency channel: {q}" for q in missing_latency],
        missing_rider_coupling_quantities=[
            f"missing rider-coupling channel: {q}" for q in missing_rider
        ],
        synchronization_blockers=synchronization_blockers,
        provenance_blockers=provenance_blockers,
        separation_blockers=separation_blockers,
        proposed_field_blockers=proposed_field_blockers,
        measured_value_claim_allowed=False,
    )

    contract_status, claim_allowed = _resolve_contract_status(
        measurement_status=measurement_status, has_blockers=bool(report.blockers)
    )

    # Frozen dataclass: rebuild with the resolved status fields.
    return AmvActuationLatencyManifestReport(
        **{
            **report.to_dict(),
            "contract_status": contract_status,
            "measured_value_claim_allowed": claim_allowed,
        }
    )


def _resolve_contract_status(*, measurement_status: str, has_blockers: bool) -> tuple[str, bool]:
    """Map measurement lifecycle state and blocker presence onto a contract status.

    Returns:
        ``(contract_status, measured_value_claim_allowed)``. A measured-value
        claim is only allowed for a complete ``measured`` manifest.
    """
    if measurement_status == "measured":
        if has_blockers:
            return CONTRACT_STATUS_BLOCKED, False
        return CONTRACT_STATUS_READY, True
    if measurement_status == "synthetic-only":
        # Synthetic-only is a terminal, claim-free state; blockers downgrade it
        # to blocked so a malformed synthetic manifest is never read as usable.
        if has_blockers:
            return CONTRACT_STATUS_BLOCKED, False
        return CONTRACT_STATUS_SYNTHETIC_ONLY, False
    # blocked-external-input: a valid plan is still blocked until data arrives.
    return CONTRACT_STATUS_BLOCKED, False


def _synchronization_blockers(synchronization: Mapping[str, Any]) -> list[str]:
    """Return semantic synchronization blockers the schema cannot catch.

    The schema guarantees ``method``, ``reference_clock``, and ``max_skew_ms`` are
    present and well-typed, so this layer only flags placeholder/unbounded skew.

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    method = str(synchronization.get("method", "")).strip().lower()
    if method in _PLACEHOLDER_VALUES:
        blockers.append("synchronization.method is not yet specified")
    max_skew = synchronization.get("max_skew_ms")
    if isinstance(max_skew, (int, float)) and max_skew <= 0:
        blockers.append("synchronization.max_skew_ms must be > 0 to bound channel alignment")
    return sorted(blockers)


def _separation_blockers(separation: Mapping[str, Any]) -> list[str]:
    """Return blockers when synthetic-vs-measured separation is not enforced.

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    if str(separation.get("separation", "")).strip() != "enforced":
        blockers.append(
            "synthetic_separation.separation must be 'enforced' to keep synthetic "
            "assumptions separate from measured values"
        )
    return sorted(blockers)


def _proposed_field_blockers(proposed_fields: Any, *, measurement_status: str) -> list[str]:
    """Validate the proposed latency/rider-coupling fields against the status contract.

    Every proposed field must carry the single ``value_status`` allowed for the
    manifest's measurement status, declare the canonical ``parameter_class`` for
    its name, and the union of proposed fields must cover the canonical latency
    and rider-coupling field sets exactly once. Field names are required to be
    unique under case-insensitive normalization so a duplicate (or case-variant)
    entry can never silently satisfy coverage with conflicting metadata.

    Returns:
        Sorted blocker strings.
    """
    expected_status = _VALUE_STATUS_BY_MEASUREMENT_STATUS[measurement_status]
    blockers: list[str] = []
    declared_names: set[str] = set()
    seen_normalized: set[str] = set()
    for entry in proposed_fields or []:
        name = str(entry.get("name", ""))
        declared_names.add(name)
        normalized = name.strip().lower()
        if normalized in seen_normalized:
            # Fail closed: duplicate names (including case variants) make coverage
            # and per-field checks ambiguous.
            blockers.append(f"proposed field {name!r} is a duplicate name (case-insensitive)")
        else:
            seen_normalized.add(normalized)
        value_status = str(entry.get("value_status", ""))
        if value_status != expected_status:
            blockers.append(
                f"proposed field {name!r} has value_status {value_status!r}; "
                f"measurement_status {measurement_status!r} requires {expected_status!r}"
            )
        _check_proposed_field_parameter_class(entry, name=name, blockers=blockers)
        _check_proposed_field_units(entry, name=name, blockers=blockers)

    expected_names = set(PROPOSED_LATENCY_PROFILE_FIELDS) | set(
        PROPOSED_RIDER_COUPLING_PROFILE_FIELDS
    )
    for missing in sorted(expected_names - declared_names):
        blockers.append(f"proposed_profile_fields is missing canonical field: {missing}")
    return sorted(blockers)


def _check_proposed_field_parameter_class(
    entry: Mapping[str, Any], *, name: str, blockers: list[str]
) -> None:
    """Append a blocker when a proposed field's parameter_class disagrees with the canonical map."""
    canonical_class = PROPOSED_FIELD_PARAMETER_CLASS.get(name)
    if canonical_class is None:
        return  # Unknown field names are tolerated; coverage is checked separately.
    if str(entry.get("parameter_class", "")) != canonical_class:
        blockers.append(
            f"proposed field {name!r} parameter_class {entry.get('parameter_class')!r} "
            f"!= canonical {canonical_class!r}"
        )


def _check_proposed_field_units(
    entry: Mapping[str, Any], *, name: str, blockers: list[str]
) -> None:
    """Append a blocker when a proposed field's units disagree with the canonical map."""
    canonical_units = PROPOSED_LATENCY_PROFILE_FIELDS.get(
        name
    ) or PROPOSED_RIDER_COUPLING_PROFILE_FIELDS.get(name)
    if canonical_units is None:
        return  # Unknown field names are tolerated; coverage is checked separately.
    if str(entry.get("units", "")) != canonical_units:
        blockers.append(
            f"proposed field {name!r} units {entry.get('units')!r} != canonical {canonical_units!r}"
        )


def _provenance_blockers(provenance: Any, *, measurement_status: str) -> list[str]:
    """Return provenance blockers consistent with the measurement status.

    * ``measured`` requires the full provenance field set.
    * ``synthetic-only`` must NOT declare a measured source (no conflation).
    * ``blocked-external-input`` tolerates absent/pending provenance.

    Returns:
        Sorted blocker strings.
    """
    blockers: list[str] = []
    if measurement_status == "measured":
        if not isinstance(provenance, Mapping):
            return [
                "measured manifest requires provenance fields: "
                + ", ".join(MEASURED_MANIFEST_REQUIRED_PROVENANCE_FIELDS)
            ]
        for field_name in MEASURED_MANIFEST_REQUIRED_PROVENANCE_FIELDS:
            if not _non_empty(provenance.get(field_name)):
                blockers.append(f"measured manifest missing provenance.{field_name}")
    elif measurement_status == "synthetic-only" and isinstance(provenance, Mapping):
        if _non_empty(provenance.get("source_uri")) or _non_empty(provenance.get("source_id")):
            blockers.append(
                "synthetic-only manifest must not declare a measured provenance source; "
                "keep synthetic assumptions separate from measured values"
            )
    return sorted(blockers)


#: Placeholder tokens that pass the schema's non-empty check but mean a field is
#: not actually resolved yet.
_PLACEHOLDER_VALUES = frozenset({"", "tbd", "unknown", "unspecified", "n/a", "pending"})


def _non_empty(value: Any) -> bool:
    """Return whether a provenance value is present and non-placeholder."""
    if isinstance(value, Mapping):
        return bool(value)
    return bool(value) and str(value).strip().lower() not in _PLACEHOLDER_VALUES
