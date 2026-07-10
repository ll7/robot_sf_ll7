"""Structural readiness contract for benchmark release suite manifests.

This module checks whether every suite declared by a release-readiness manifest
carries references to the six metadata owners required by issue #2910.  It does
not dereference or interpret those records, freeze a suite, run a campaign, or
authorize publication.  Passing this structural check is necessary but not
sufficient for benchmark release readiness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RELEASE_SUITE_CONTRACT_SCHEMA_VERSION = "benchmark-release-suite-contract.v0.1"
RELEASE_SUITE_CONTRACT_REPORT_SCHEMA_VERSION = "benchmark-release-suite-contract-report.v0.1"
REQUIRED_SUITE_METADATA_FIELDS = (
    "odd_contract",
    "scenario_contract",
    "scenario_certification",
    "planner_row_status",
    "seed_schedule",
    "artifact_manifest",
)
CLAIM_BOUNDARY = (
    "Structural metadata-reference check only; a pass does not freeze suites, validate referenced "
    "content, establish benchmark evidence, or authorize publication."
)


class ReleaseSuiteContractError(ValueError):
    """Raised when a release suite manifest cannot be evaluated safely."""


@dataclass(frozen=True)
class ReleaseSuiteDeclaration:
    """One named suite and its unvalidated metadata references."""

    suite_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReleaseSuiteContractManifest:
    """Structurally valid container of release suite declarations."""

    schema_version: str
    release_id: str
    suites: tuple[ReleaseSuiteDeclaration, ...]


def _load_payload(path: Path) -> Any:
    """Load a JSON or YAML payload from ``path``.

    Returns:
        Parsed JSON or YAML value.
    """

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def load_release_suite_contract(path: str | Path) -> ReleaseSuiteContractManifest:
    """Load the suite container and reject ambiguous structures fail-closed.

    Missing required suite metadata is intentionally evaluated by
    :func:`evaluate_release_suite_contract` so callers receive a complete blocker
    report instead of only the first missing field.

    Returns:
        Structurally valid release suite manifest.
    """

    manifest_path = Path(path)
    try:
        payload = _load_payload(manifest_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ReleaseSuiteContractError(
            f"Could not parse suite manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ReleaseSuiteContractError("Release suite manifest must be a mapping")
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != RELEASE_SUITE_CONTRACT_SCHEMA_VERSION:
        raise ReleaseSuiteContractError(
            f"schema_version must be {RELEASE_SUITE_CONTRACT_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    raw_release_id = payload.get("release_id")
    if not isinstance(raw_release_id, str) or not raw_release_id.strip():
        raise ReleaseSuiteContractError("release_id must be a non-empty string")
    release_id = raw_release_id.strip()

    raw_suites = payload.get("suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ReleaseSuiteContractError("suites must be a non-empty list")

    suites: list[ReleaseSuiteDeclaration] = []
    seen_suite_ids: set[str] = set()
    for index, raw_suite in enumerate(raw_suites):
        if not isinstance(raw_suite, dict):
            raise ReleaseSuiteContractError(f"suites[{index}] must be a mapping")
        raw_suite_id = raw_suite.get("suite_id")
        if not isinstance(raw_suite_id, str) or not raw_suite_id.strip():
            raise ReleaseSuiteContractError(f"suites[{index}].suite_id must be a non-empty string")
        suite_id = raw_suite_id.strip()
        if suite_id in seen_suite_ids:
            raise ReleaseSuiteContractError(f"duplicate suite_id {suite_id!r}")
        seen_suite_ids.add(suite_id)
        suites.append(ReleaseSuiteDeclaration(suite_id=suite_id, metadata=dict(raw_suite)))

    return ReleaseSuiteContractManifest(
        schema_version=schema_version,
        release_id=release_id,
        suites=tuple(suites),
    )


def _missing_metadata_fields(suite: ReleaseSuiteDeclaration) -> list[str]:
    """Return required fields whose metadata reference is absent or blank."""

    return [
        field
        for field in REQUIRED_SUITE_METADATA_FIELDS
        if not isinstance(suite.metadata.get(field), str) or not str(suite.metadata[field]).strip()
    ]


def evaluate_release_suite_contract(
    manifest: ReleaseSuiteContractManifest,
) -> dict[str, Any]:
    """Return a deterministic, fail-closed completeness report for all suites."""

    suite_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for suite in manifest.suites:
        missing_fields = _missing_metadata_fields(suite)
        blockers.extend(
            f"{suite.suite_id}.{field} is missing or is not a non-empty string"
            for field in missing_fields
        )
        suite_results.append(
            {
                "suite_id": suite.suite_id,
                "status": "blocked" if missing_fields else "complete",
                "missing_fields": missing_fields,
            }
        )

    blocked_suite_count = sum(result["status"] == "blocked" for result in suite_results)
    return {
        "schema_version": RELEASE_SUITE_CONTRACT_REPORT_SCHEMA_VERSION,
        "manifest_schema_version": manifest.schema_version,
        "release_id": manifest.release_id,
        "status": "blocked" if blockers else "pass",
        "claim_boundary": CLAIM_BOUNDARY,
        "required_suite_metadata_fields": list(REQUIRED_SUITE_METADATA_FIELDS),
        "suite_count": len(suite_results),
        "complete_suite_count": len(suite_results) - blocked_suite_count,
        "blocked_suite_count": blocked_suite_count,
        "blockers": blockers,
        "suites": suite_results,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "RELEASE_SUITE_CONTRACT_REPORT_SCHEMA_VERSION",
    "RELEASE_SUITE_CONTRACT_SCHEMA_VERSION",
    "REQUIRED_SUITE_METADATA_FIELDS",
    "ReleaseSuiteContractError",
    "ReleaseSuiteContractManifest",
    "ReleaseSuiteDeclaration",
    "evaluate_release_suite_contract",
    "load_release_suite_contract",
]
