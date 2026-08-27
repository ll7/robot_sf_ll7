"""Fail-closed validator for the ``incident_scenario_provenance.v1`` contract.

This module owns the cross-field semantic rules that the JSON schema in
``../schemas/incident_scenario_provenance.v1.json`` cannot express. The record
transforms one incident description into one replayable Robot SF scenario
record while preserving the distinction between source facts, extracted
hypotheses, simulator assumptions, parameter mappings, generated configuration,
execution identity, and observed outcome. It never assigns legal or moral fault:
``normative_fault`` is always ``"not_assessed"`` (the same boundary as
:mod:`robot_sf.benchmark.collision.collision_causal_report`).

The contract is deliberately conservative: rejected, ambiguous, or unsupported
records stay outside admitted denominators, and unverified extracted content
cannot be marked verified without an explicit human-verification record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.errors import RobotSfError

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise RuntimeError(
        "jsonschema package required for incident scenario provenance validation"
    ) from exc

INCIDENT_SCENARIO_PROVENANCE_SCHEMA_VERSION = "incident_scenario_provenance.v1"
INCIDENT_SCENARIO_PROVENANCE_SCHEMA_FILE = (
    Path(__file__).resolve().parents[1] / "schemas" / "incident_scenario_provenance.v1.json"
)

#: Neutral model-scoped actor roles. Role labels describe the scenario model
#: only; they never encode legal liability, moral blame, or real-world causality.
ACTOR_ROLES = frozenset(
    {
        "ego",
        "pedestrian_initiator",
        "affected_pedestrian",
        "infrastructure",
        "unknown",
    }
)

#: Extraction status vocabulary. ``verified`` requires an explicit
#: human-verification record; LLM output without such a record stays unverified.
EXTRACTION_STATUSES = frozenset({"verified", "human_corrected", "unverified", "rejected"})

#: Admission dispositions. Non-admitted records stay outside denominators.
ADMISSION_DISPOSITIONS = frozenset({"admitted", "rejected", "ambiguous", "unsupported"})

#: Parameter-mapping statuses.
PARAMETER_STATUSES = frozenset({"mapped", "estimated", "defaulted", "unsupported"})

#: Outcome kinds for a claimed execution.
OUTCOME_KINDS = frozenset({"collision", "near_miss", "completed", "unavailable"})


class IncidentScenarioProvenanceError(RobotSfError, ValueError):
    """Raised when an incident-scenario provenance record violates the contract."""


def load_incident_scenario_provenance_schema() -> dict[str, Any]:
    """Load the ``incident_scenario_provenance.v1`` JSON schema from disk.

    Returns:
        The parsed JSON schema dictionary.
    """

    with INCIDENT_SCENARIO_PROVENANCE_SCHEMA_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reconcile_incident_scenario_provenance(record: Mapping[str, Any]) -> list[str]:
    """Return the list of fail-closed contract violations for ``record``.

    The check is non-raising so callers can accumulate violations (the house
    pattern used by
    :func:`robot_sf.benchmark.collision.collision_causal_report.reconcile_collision_causal_report`).
    An empty list means the record satisfies both the JSON schema and every
    cross-field semantic rule.

    Returns:
        Human-readable violation strings; empty when the record is valid.
    """

    schema = load_incident_scenario_provenance_schema()
    try:
        jsonschema.validate(instance=record, schema=schema)
    except jsonschema.ValidationError as error:
        return [f"schema: {error.message}"]

    violations: list[str] = []
    violations.extend(_normative_fault_violations(record))
    violations.extend(_verification_violations(record))
    violations.extend(_parameter_mapping_violations(record))
    violations.extend(_admission_violations(record))
    violations.extend(_execution_violations(record))
    return violations


def validate_incident_scenario_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an incident-scenario provenance record, raising on first breach.

    Returns:
        A shallow ``dict`` copy of the validated record.

    Raises:
        IncidentScenarioProvenanceError: If any schema or semantic rule is violated.
    """

    violations = reconcile_incident_scenario_provenance(record)
    if violations:
        raise IncidentScenarioProvenanceError("; ".join(violations))
    return dict(record)


def _normative_fault_violations(record: Mapping[str, Any]) -> list[str]:
    """Reject any attempt to assign a normative fault value.

    Returns:
        Violation strings; empty when ``normative_fault`` is ``not_assessed``.
    """
    if record.get("normative_fault") != "not_assessed":
        return ["normative_fault must always be not_assessed"]
    return []


def _verification_violations(record: Mapping[str, Any]) -> list[str]:
    """Require an explicit human-verification record for verified extraction.

    Returns:
        Violation strings; empty when the extraction status needs no record
        or the record is present.
    """
    extraction = record.get("extraction")
    if not isinstance(extraction, Mapping):
        return []
    status = extraction.get("status")
    if status in {"verified", "human_corrected"} and "verification_record" not in extraction:
        return [
            f"extraction.status={status!r} requires an explicit extraction.verification_record "
            "(human review evidence); LLM output alone cannot be marked verified"
        ]
    return []


def _parameter_mapping_violations(record: Mapping[str, Any]) -> list[str]:
    """Keep parameter confidence and unsupported status semantically aligned.

    Returns:
        Violations for unsupported mappings that carry a misleading confidence.
    """
    mappings = record.get("scenario_parameters")
    if not isinstance(mappings, list):
        return []
    violations: list[str] = []
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, Mapping):
            continue
        if mapping.get("status") == "unsupported" and mapping.get("confidence") != "unavailable":
            violations.append(
                f"scenario_parameters[{index}] with status='unsupported' must use "
                "confidence='unavailable'"
            )
    return violations


def _admission_violations(record: Mapping[str, Any]) -> list[str]:
    """Reject an admitted record that contains explicitly non-admissible content.

    Returns:
        Violations for an admitted record with rejected extraction or unsupported mappings.
    """
    if record.get("admission") != "admitted":
        return []
    extraction = record.get("extraction")
    extraction_status = extraction.get("status") if isinstance(extraction, Mapping) else None
    if extraction_status == "rejected":
        return ["admission='admitted' conflicts with extraction.status='rejected'"]
    mappings = record.get("scenario_parameters")
    if isinstance(mappings, list) and any(
        isinstance(mapping, Mapping) and mapping.get("status") == "unsupported"
        for mapping in mappings
    ):
        return ["admission='admitted' conflicts with an unsupported scenario parameter mapping"]
    return []


def _execution_violations(record: Mapping[str, Any]) -> list[str]:
    """Require full replay identity when execution is claimed.

    Returns:
        Violation strings; empty when execution is not claimed or all
        required identity fields are present.
    """
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        return []
    if not execution.get("claimed"):
        return []
    missing = [
        key
        for key in (
            "scenario_config_digest_sha256",
            "seed",
            "software_commit",
            "replay_identity",
        )
        if not execution.get(key)
    ]
    if missing:
        return ["execution.claimed=true requires field(s): " + ", ".join(missing)]
    return []
