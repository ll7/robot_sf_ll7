"""Shared lineage contract helpers for research manifest families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MANDATORY_LINEAGE_FIELDS = (
    "source",
    "generator_id",
    "validator_version",
    "schema_version",
    "claim_boundary",
    "evidence_tier",
    "denominator_policy",
    "execution_gate",
)


@dataclass(frozen=True)
class ManifestLineageContract:
    """Common provenance and evidence-boundary fields for manifest-like artifacts."""

    source: dict[str, Any]
    generator_id: str
    validator_version: str
    schema_version: str
    claim_boundary: str | dict[str, Any]
    evidence_tier: str
    denominator_policy: str
    execution_gate: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping.

        Returns:
            JSON-safe field mapping for embedding in manifest payloads.
        """
        return asdict(self)


def validate_lineage_contract(payload: object) -> list[str]:
    """Return validation errors for the shared lineage contract."""
    if not isinstance(payload, dict):
        return ["lineage payload must be a mapping"]
    errors: list[str] = []
    for field in MANDATORY_LINEAGE_FIELDS:
        if field not in payload:
            errors.append(f"{field} is required")
    source = payload.get("source")
    if "source" in payload and not isinstance(source, dict):
        errors.append("source must be a mapping")
    for field in (
        "generator_id",
        "validator_version",
        "schema_version",
        "evidence_tier",
        "denominator_policy",
        "execution_gate",
    ):
        if field in payload and not isinstance(payload.get(field), str):
            errors.append(f"{field} must be a string")
    if "claim_boundary" in payload and not isinstance(payload.get("claim_boundary"), str | dict):
        errors.append("claim_boundary must be a string or mapping")
    return errors


def require_lineage_contract(payload: object) -> None:
    """Raise ValueError when payload does not satisfy the shared lineage contract."""
    errors = validate_lineage_contract(payload)
    if errors:
        raise ValueError("; ".join(errors))
