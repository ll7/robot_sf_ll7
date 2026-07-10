"""Typed loader for ``hazard_traceability.v1`` benchmark mapping payloads."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

HAZARD_TRACEABILITY_SCHEMA_VERSION = "hazard_traceability.v1"
HAZARD_COVERAGE_SUMMARY_SCHEMA_VERSION = "hazard-traceability-coverage.v1"
HAZARD_TRACEABILITY_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "hazard_traceability.v1.json"
)


@dataclass(frozen=True, slots=True)
class HazardDefinition:
    """Stable hazard definition and evidence fields."""

    id: str
    description: str
    severity: str
    supporting_metrics: list[str]
    evidence_fields: list[str]


@dataclass(frozen=True, slots=True)
class ScenarioHazardMapping:
    """Mapping from scenario IDs or families to hazard IDs."""

    id: str
    scenario_ids: list[str]
    scenario_families: list[str]
    hazards: list[str]
    notes: str


@dataclass(frozen=True, slots=True)
class HazardTraceabilityProvenance:
    """Provenance metadata for a traceability mapping."""

    source_issue: str
    authored_by: str
    source_files: list[str]
    notes: str


@dataclass(frozen=True, slots=True)
class HazardTraceability:
    """Typed ``hazard_traceability.v1`` payload."""

    schema_version: str
    id: str
    hazards: list[HazardDefinition]
    scenario_mappings: list[ScenarioHazardMapping]
    claim_boundary: str
    provenance: HazardTraceabilityProvenance
    extensions: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the mapping to JSON-safe primitives.

        Returns:
            Dictionary representation suitable for JSON Schema validation.
        """

        payload = asdict(self)
        if not payload["extensions"]:
            payload.pop("extensions")
        return payload


class HazardTraceabilityValidationError(RobotSfError, ValueError):
    """Raised when a hazard traceability mapping fails validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_hazard_traceability_schema() -> dict[str, Any]:
    """Load the public ``hazard_traceability.v1`` JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(HAZARD_TRACEABILITY_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_hazard_traceability(path: Path) -> HazardTraceability:
    """Load one hazard traceability mapping from YAML or JSON.

    Returns:
        Typed hazard traceability mapping.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise HazardTraceabilityValidationError(["expected a mapping payload"], source=path)
    return hazard_traceability_from_dict(raw, source=path)


def hazard_traceability_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> HazardTraceability:
    """Validate and convert a mapping into typed hazard traceability metadata.

    Returns:
        Typed hazard traceability mapping.
    """

    errors = _schema_validation_errors(payload)
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise HazardTraceabilityValidationError(errors, source=source)

    return _mapping_from_payload(payload)


def summarize_hazard_coverage(
    mapping: HazardTraceability,
    *,
    scenario_ids: Iterable[str] = (),
    scenario_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Summarize hazards covered by scenario IDs or families.

    Returns:
        JSON-safe coverage summary with covered hazards, unmapped inputs, and supporting evidence.
    """

    requested_ids = [str(value) for value in scenario_ids]
    requested_families = [str(value) for value in scenario_families]
    requested_id_set = set(requested_ids)
    requested_family_set = set(requested_families)
    covered_hazards: set[str] = set()
    mapped_ids: set[str] = set()
    mapped_families: set[str] = set()

    for item in mapping.scenario_mappings:
        matched_ids = requested_id_set.intersection(item.scenario_ids)
        matched_families = requested_family_set.intersection(item.scenario_families)
        if matched_ids or matched_families:
            covered_hazards.update(item.hazards)
            mapped_ids.update(matched_ids)
            mapped_families.update(matched_families)

    hazards_by_id = {hazard.id: hazard for hazard in mapping.hazards}
    ordered_hazards = sorted(covered_hazards)
    return {
        "schema_version": HAZARD_COVERAGE_SUMMARY_SCHEMA_VERSION,
        "traceability_id": mapping.id,
        "scenario_ids": requested_ids,
        "scenario_families": requested_families,
        "covered_hazards": ordered_hazards,
        "unmapped_scenario_ids": sorted(requested_id_set - mapped_ids),
        "unmapped_scenario_families": sorted(requested_family_set - mapped_families),
        "supporting_metrics_by_hazard": {
            hazard_id: list(hazards_by_id[hazard_id].supporting_metrics)
            for hazard_id in ordered_hazards
        },
        "evidence_fields_by_hazard": {
            hazard_id: list(hazards_by_id[hazard_id].evidence_fields)
            for hazard_id in ordered_hazards
        },
        "claim_boundary": mapping.claim_boundary,
    }


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors for one mapping payload."""

    validator = Draft202012Validator(load_hazard_traceability_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return cross-field validation errors not expressible in the JSON Schema."""

    errors: list[str] = []
    hazard_ids = _declared_hazard_ids(payload.get("hazards"), errors)
    errors.extend(_scenario_mapping_hazard_errors(payload.get("scenario_mappings"), hazard_ids))
    return errors


def _declared_hazard_ids(raw_hazards: Any, errors: list[str]) -> set[str]:
    """Collect declared hazard IDs while recording duplicates.

    Returns:
        Set of unique hazard IDs declared in the payload.
    """

    hazard_ids: set[str] = set()
    if not isinstance(raw_hazards, list):
        return hazard_ids
    for index, hazard in enumerate(raw_hazards):
        if not isinstance(hazard, Mapping):
            continue
        hazard_id = hazard.get("id")
        if not isinstance(hazard_id, str):
            continue
        if hazard_id in hazard_ids:
            errors.append(f"/hazards/{index}/id: duplicate hazard id '{hazard_id}'")
        hazard_ids.add(hazard_id)
    return hazard_ids


def _scenario_mapping_hazard_errors(raw_mappings: Any, hazard_ids: set[str]) -> list[str]:
    """Return errors for scenario mappings that name unknown hazard IDs.

    Returns:
        List of semantic validation error strings.
    """

    errors: list[str] = []
    if not isinstance(raw_mappings, list):
        return errors
    for mapping_index, item in enumerate(raw_mappings):
        if not isinstance(item, Mapping):
            continue
        for hazard_id in item.get("hazards", []):
            if isinstance(hazard_id, str) and hazard_id not in hazard_ids:
                errors.append(
                    f"/scenario_mappings/{mapping_index}/hazards: unknown hazard id '{hazard_id}'",
                )
    return errors


def _mapping_from_payload(payload: Mapping[str, Any]) -> HazardTraceability:
    """Build typed hazard traceability from a schema-valid payload.

    Returns:
        Typed hazard traceability mapping.
    """

    provenance = payload["provenance"]
    return HazardTraceability(
        schema_version=str(payload["schema_version"]),
        id=str(payload["id"]),
        hazards=[
            HazardDefinition(
                id=str(hazard["id"]),
                description=str(hazard["description"]),
                severity=str(hazard["severity"]),
                supporting_metrics=list(hazard["supporting_metrics"]),
                evidence_fields=list(hazard["evidence_fields"]),
            )
            for hazard in payload["hazards"]
        ],
        scenario_mappings=[
            ScenarioHazardMapping(
                id=str(item["id"]),
                scenario_ids=list(item.get("scenario_ids", [])),
                scenario_families=list(item.get("scenario_families", [])),
                hazards=list(item["hazards"]),
                notes=str(item["notes"]),
            )
            for item in payload["scenario_mappings"]
        ],
        claim_boundary=str(payload["claim_boundary"]),
        provenance=HazardTraceabilityProvenance(
            source_issue=str(provenance["source_issue"]),
            authored_by=str(provenance["authored_by"]),
            source_files=list(provenance["source_files"]),
            notes=str(provenance["notes"]),
        ),
        extensions=dict(payload.get("extensions", {})),
    )


__all__ = [
    "HAZARD_COVERAGE_SUMMARY_SCHEMA_VERSION",
    "HAZARD_TRACEABILITY_SCHEMA_FILE",
    "HAZARD_TRACEABILITY_SCHEMA_VERSION",
    "HazardDefinition",
    "HazardTraceability",
    "HazardTraceabilityProvenance",
    "HazardTraceabilityValidationError",
    "ScenarioHazardMapping",
    "hazard_traceability_from_dict",
    "load_hazard_traceability",
    "load_hazard_traceability_schema",
    "summarize_hazard_coverage",
]
