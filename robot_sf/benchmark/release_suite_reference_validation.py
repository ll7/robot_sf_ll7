"""Dereference validation for benchmark release suite metadata references.

This module confirms that the six metadata references declared by each release
suite manifest (see :mod:`robot_sf.benchmark.release_suite_contract`) resolve to
existing, non-empty, parseable records under a caller-supplied base directory.
It upgrades the structural presence check from issue #2910 / PR #5134 from
"is a non-empty string" to "points at a real record that parses".

Claim boundary
--------------
Dereference validation only. A pass confirms each reference resolves to a real,
non-empty, parseable file **without** escaping the base directory. It does not
interpret the referenced content, enforce per-owner schemas, freeze a suite, run
a campaign, establish benchmark evidence, or authorize publication. Passing is
necessary but not sufficient for benchmark release readiness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.release_suite_contract import (
    REQUIRED_SUITE_METADATA_FIELDS,
    ReleaseSuiteContractManifest,
    ReleaseSuiteDeclaration,
)
from robot_sf.errors import RobotSfError

RELEASE_SUITE_REFERENCE_REPORT_SCHEMA_VERSION = "benchmark-release-suite-reference-report.v0.1"
REFERENCE_VALIDATION_CLAIM_BOUNDARY = (
    "Dereference validation only; a pass does not interpret referenced content, enforce per-owner "
    "schemas, freeze suites, validate campaigns, establish benchmark evidence, or authorize "
    "publication."
)


class ReleaseSuiteReferenceError(RobotSfError, ValueError):
    """Raised when reference validation cannot be evaluated safely."""


@dataclass(frozen=True)
class ReferenceResolution:
    """Outcome of resolving a single metadata reference under the base directory."""

    field: str
    reference: str
    status: str
    detail: str = ""


def _parse_record(path: Path) -> Any:
    """Parse a JSON or YAML record from ``path``.

    Returns:
        Parsed JSON or YAML value.

    Raises:
        ValueError: If the payload is empty or cannot be parsed.
    """

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("referenced file is empty")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _resolve_reference(
    base_dir: Path,
    suite: ReleaseSuiteDeclaration,
    field: str,
) -> ReferenceResolution:
    """Resolve one metadata reference against ``base_dir`` fail-closed.

    A reference is accepted only when it resolves to an existing, non-empty,
    parseable regular file **inside** ``base_dir``. Path traversal outside the
    base directory fails closed rather than following an escaped reference.

    Returns:
        The resolution outcome (``resolved``) or a blocked/not-available outcome
        with a human-readable detail string.
    """

    reference = suite.metadata.get(field)
    if not isinstance(reference, str) or not reference.strip():
        return ReferenceResolution(
            field=field,
            reference=str(reference),
            status="blocked",
            detail="reference is missing or is not a non-empty string",
        )
    reference = reference.strip()

    candidate = (base_dir / reference).resolve()
    try:
        base_resolved = base_dir.resolve()
    except OSError as exc:
        raise ReleaseSuiteReferenceError(f"Could not resolve base_dir {base_dir}: {exc}") from exc
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        return ReferenceResolution(
            field=field,
            reference=reference,
            status="blocked",
            detail="reference escapes the base directory",
        )

    if not candidate.exists():
        return ReferenceResolution(
            field=field,
            reference=reference,
            status="not_available",
            detail=f"referenced file does not exist: {candidate}",
        )
    if not candidate.is_file():
        return ReferenceResolution(
            field=field,
            reference=reference,
            status="blocked",
            detail="referenced path is not a regular file",
        )

    try:
        parsed = _parse_record(candidate)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return ReferenceResolution(
            field=field,
            reference=reference,
            status="blocked",
            detail=f"referenced file is not parseable: {exc}",
        )
    if parsed is None:
        return ReferenceResolution(
            field=field,
            reference=reference,
            status="blocked",
            detail="referenced file parses to null",
        )

    return ReferenceResolution(field=field, reference=reference, status="resolved")


def evaluate_release_suite_references(
    manifest: ReleaseSuiteContractManifest,
    base_dir: str | Path,
) -> dict[str, Any]:
    """Return a deterministic, fail-closed dereference report for all suites.

    Every reference for every suite is resolved. Missing references and any
    resolution failure accumulate as blockers so callers receive a complete
    report instead of only the first failure.

    Args:
        manifest: A structurally loaded release suite manifest.
        base_dir: Directory against which each metadata reference is resolved.

    Returns:
        Deterministic dereference report. ``status`` is ``"pass"`` only when
        every required reference on every suite resolves to a real, non-empty,
        parseable file inside ``base_dir``.

    Raises:
        ReleaseSuiteReferenceError: If ``base_dir`` does not exist or is not a
            directory.
    """

    base_path = Path(base_dir)
    if not base_path.exists():
        raise ReleaseSuiteReferenceError(f"base_dir does not exist: {base_path}")
    if not base_path.is_dir():
        raise ReleaseSuiteReferenceError(f"base_dir is not a directory: {base_path}")

    suite_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    resolved_count = 0
    total_count = 0
    for suite in manifest.suites:
        resolutions: list[dict[str, Any]] = []
        suite_blocked = False
        for field in REQUIRED_SUITE_METADATA_FIELDS:
            total_count += 1
            resolution = _resolve_reference(base_path, suite, field)
            resolutions.append(
                {
                    "field": resolution.field,
                    "reference": resolution.reference,
                    "status": resolution.status,
                    "detail": resolution.detail,
                }
            )
            if resolution.status == "resolved":
                resolved_count += 1
            else:
                suite_blocked = True
                blockers.append(
                    f"{suite.suite_id}.{resolution.field} ({resolution.reference}): {resolution.detail}"
                )
        suite_results.append(
            {
                "suite_id": suite.suite_id,
                "status": "blocked" if suite_blocked else "resolved",
                "references": resolutions,
            }
        )

    return {
        "schema_version": RELEASE_SUITE_REFERENCE_REPORT_SCHEMA_VERSION,
        "manifest_schema_version": manifest.schema_version,
        "release_id": manifest.release_id,
        "base_dir": str(base_path),
        "status": "blocked" if blockers else "pass",
        "claim_boundary": REFERENCE_VALIDATION_CLAIM_BOUNDARY,
        "required_suite_metadata_fields": list(REQUIRED_SUITE_METADATA_FIELDS),
        "suite_count": len(suite_results),
        "resolved_suite_count": sum(result["status"] == "resolved" for result in suite_results),
        "blocked_suite_count": sum(result["status"] == "blocked" for result in suite_results),
        "reference_count": total_count,
        "resolved_reference_count": resolved_count,
        "blocked_reference_count": total_count - resolved_count,
        "blockers": blockers,
        "suites": suite_results,
    }


__all__ = [
    "REFERENCE_VALIDATION_CLAIM_BOUNDARY",
    "RELEASE_SUITE_REFERENCE_REPORT_SCHEMA_VERSION",
    "ReferenceResolution",
    "ReleaseSuiteReferenceError",
    "evaluate_release_suite_references",
]
