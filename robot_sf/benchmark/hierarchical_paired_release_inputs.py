"""Fail-closed input readiness reporting for issue #5351's release analysis.

The hierarchical paired analysis is intentionally downstream of the #4364
successor release.  This module records that dependency in a machine-readable
manifest and prevents a missing release tag or typed-ledger export from being
mistaken for an analysable release dataset.  It does not compute statistics,
change frozen metric semantics, or promote a benchmark claim.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.errors import RobotSfError

HIERARCHICAL_PAIRED_RELEASE_INPUT_MANIFEST_SCHEMA = (
    "hierarchical_paired_release_analysis_input_manifest.v1"
)
HIERARCHICAL_PAIRED_RELEASE_INPUT_REPORT_SCHEMA = (
    "hierarchical_paired_release_analysis_input_report.v1"
)
BLOCKED_MISSING_SUCCESSOR_ROWS = "blocked_missing_successor_release_rows"
INPUTS_READY_ANALYSIS_NOT_RUN = "inputs_ready_analysis_not_run"
_REQUIRED_PROTOCOL_IDS = (
    "paired_effects",
    "hierarchical_intervals",
    "sensitivity_analyses",
    "multiplicity_control",
    "practical_effect_reporting",
    "censored_completion_time",
    "normalized_near_miss_exposure",
    "claim_gate_and_conformance",
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class HierarchicalPairedReleaseInputError(RobotSfError, ValueError):
    """Raised when the #5351 input manifest is structurally unsafe."""


def load_hierarchical_paired_release_input_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an issue #5351 hierarchical-analysis input manifest.

    Returns:
        A validated shallow copy of the manifest.
    """

    manifest_path = Path(path)
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HierarchicalPairedReleaseInputError(
            f"could not parse hierarchical paired release input manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise HierarchicalPairedReleaseInputError("input manifest must be a YAML mapping")
    return validate_hierarchical_paired_release_input_manifest(payload)


def validate_hierarchical_paired_release_input_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the static contract without requiring unavailable release artifacts.

    The manifest remains valid while blocked; runtime presence is evaluated separately so
    that the pre-release contract can be reviewed before #4364 produces its successor rows.

    Returns:
        A shallow-normalized manifest mapping.
    """

    normalized = dict(manifest)
    if normalized.get("schema_version") != HIERARCHICAL_PAIRED_RELEASE_INPUT_MANIFEST_SCHEMA:
        raise HierarchicalPairedReleaseInputError(
            f"schema_version must be {HIERARCHICAL_PAIRED_RELEASE_INPUT_MANIFEST_SCHEMA!r}"
        )
    if normalized.get("issue") != 5351:
        raise HierarchicalPairedReleaseInputError("issue must be 5351")
    if not _nonempty_string(normalized.get("claim_boundary")):
        raise HierarchicalPairedReleaseInputError("claim_boundary must be a non-empty string")
    _validate_successor_release(normalized.get("successor_release"))
    _validate_protocol(normalized.get("protocol"))
    return normalized


def evaluate_hierarchical_paired_release_inputs(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Evaluate the successor-release prerequisites and emit a claim-gate report.

    Returns:
        A deterministic report.  Even when inputs are present, its claim gate remains
        blocked until the downstream statistical analysis is actually implemented and run.
    """

    normalized = validate_hierarchical_paired_release_input_manifest(manifest)
    root = Path(repo_root).resolve()
    successor_release = dict(normalized["successor_release"])
    blockers = _successor_release_blockers(successor_release, repo_root=root)
    status = BLOCKED_MISSING_SUCCESSOR_ROWS if blockers else INPUTS_READY_ANALYSIS_NOT_RUN
    protocol_status = (
        "blocked_missing_successor_release_rows" if blockers else "declared_pending_analysis"
    )
    return {
        "schema_version": HIERARCHICAL_PAIRED_RELEASE_INPUT_REPORT_SCHEMA,
        "issue": 5351,
        "status": status,
        "claim_boundary": normalized["claim_boundary"],
        "evidence_status": "not_benchmark_evidence",
        "successor_release": successor_release,
        "blocking_prerequisites": blockers,
        "protocol_conformance": [
            {
                "id": item["id"],
                "declared_delivery": item["declared_delivery"],
                "status": protocol_status,
            }
            for item in normalized["protocol"]
        ],
        "claim_gate": {
            "status": "blocked_analysis_not_run",
            "reason": (
                "successor-release inputs are missing"
                if blockers
                else "inputs are present but the hierarchical paired analysis has not run"
            ),
        },
        "semantics": {
            "benchmark_metrics_changed": False,
            "analysis_executed": False,
            "claim_promotion": "none",
        },
    }


def _validate_successor_release(successor_release: Any) -> None:
    """Require the four successor-release references used by the runtime checker."""

    if not isinstance(successor_release, Mapping):
        raise HierarchicalPairedReleaseInputError("successor_release must be a mapping")
    required = ("release_tag", "commit", "typed_ledger_rows", "typed_ledger_rows_sha256")
    missing = [field for field in required if field not in successor_release]
    if missing:
        raise HierarchicalPairedReleaseInputError(
            f"successor_release missing required fields: {missing}"
        )
    rows_path = successor_release["typed_ledger_rows"]
    if rows_path is not None and not _nonempty_string(rows_path):
        raise HierarchicalPairedReleaseInputError(
            "successor_release.typed_ledger_rows must be a string or null"
        )
    rows_sha256 = successor_release["typed_ledger_rows_sha256"]
    if rows_sha256 is not None and not _nonempty_string(rows_sha256):
        raise HierarchicalPairedReleaseInputError(
            "successor_release.typed_ledger_rows_sha256 must be a string or null"
        )


def _validate_protocol(protocol: Any) -> None:
    """Require one named delivery target for every protocol element in #5351."""

    if not isinstance(protocol, Sequence) or isinstance(protocol, (str, bytes)):
        raise HierarchicalPairedReleaseInputError("protocol must be a list")
    ids: list[str] = []
    for index, item in enumerate(protocol):
        if not isinstance(item, Mapping):
            raise HierarchicalPairedReleaseInputError(f"protocol[{index}] must be a mapping")
        item_id = item.get("id")
        if not _nonempty_string(item_id):
            raise HierarchicalPairedReleaseInputError(f"protocol[{index}].id must be non-empty")
        if not _nonempty_string(item.get("declared_delivery")):
            raise HierarchicalPairedReleaseInputError(
                f"protocol[{index}].declared_delivery must be non-empty"
            )
        ids.append(item_id)
    if tuple(ids) != _REQUIRED_PROTOCOL_IDS:
        raise HierarchicalPairedReleaseInputError(
            f"protocol ids must be {list(_REQUIRED_PROTOCOL_IDS)!r}, got {ids!r}"
        )


def _successor_release_blockers(
    successor_release: Mapping[str, Any], *, repo_root: Path
) -> list[dict[str, str]]:
    """Return all missing or unsafe successor-release prerequisites."""

    return _release_metadata_blockers(successor_release) + _typed_rows_blockers(
        successor_release, repo_root=repo_root
    )


def _release_metadata_blockers(successor_release: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return blockers for the release tag and commit provenance."""

    blockers: list[dict[str, str]] = []
    release_tag = successor_release.get("release_tag")
    if not _nonempty_string(release_tag) or release_tag == "{release_tag}":
        blockers.append(
            {
                "field": "successor_release.release_tag",
                "reason": "#4364 successor release tag is not recorded",
            }
        )
    commit = successor_release.get("commit")
    if not _nonempty_string(commit) or not _COMMIT_PATTERN.fullmatch(commit):
        blockers.append(
            {
                "field": "successor_release.commit",
                "reason": "#4364 successor release commit must be a 40-character lowercase SHA-1",
            }
        )
    return blockers


def _typed_rows_blockers(
    successor_release: Mapping[str, Any], *, repo_root: Path
) -> list[dict[str, str]]:
    """Return blockers for the durable typed-ledger rows and their digest."""

    blockers: list[dict[str, str]] = []
    rows_path = successor_release.get("typed_ledger_rows")
    expected_sha256 = successor_release.get("typed_ledger_rows_sha256")
    if not _nonempty_string(expected_sha256) or not _SHA256_PATTERN.fullmatch(expected_sha256):
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows_sha256",
                "reason": "durable typed-ledger rows must declare a 64-character lowercase SHA-256",
            }
        )
    if not _nonempty_string(rows_path):
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows",
                "reason": "durable typed-ledger successor rows are not recorded",
            }
        )
        return blockers
    candidate = Path(rows_path)
    if candidate.is_absolute() or ".." in candidate.parts or "output" in candidate.parts:
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows",
                "reason": "typed-ledger rows must use a durable repository-relative non-output path",
            }
        )
        return blockers
    candidate_path = repo_root / candidate
    resolved = candidate_path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows",
                "reason": "typed-ledger rows path escapes the repository root",
            }
        )
        return blockers
    if _contains_symlink(candidate_path, repo_root=repo_root):
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows",
                "reason": "typed-ledger rows must use a durable repository-relative non-output path",
            }
        )
    elif not resolved.is_file():
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows",
                "reason": f"durable typed-ledger rows are missing: {candidate.as_posix()}",
            }
        )
    elif resolved.stat().st_size == 0:
        blockers.append(
            {
                "field": "successor_release.typed_ledger_rows",
                "reason": f"durable typed-ledger rows are empty: {candidate.as_posix()}",
            }
        )
    elif _SHA256_PATTERN.fullmatch(str(expected_sha256)):
        actual_sha256 = sha256_file(resolved)
        if actual_sha256 != expected_sha256:
            blockers.append(
                {
                    "field": "successor_release.typed_ledger_rows_sha256",
                    "reason": "durable typed-ledger rows SHA-256 does not match the manifest",
                }
            )
    return blockers


def _contains_symlink(path: Path, *, repo_root: Path) -> bool:
    """Return whether a durable row path traverses a symlink inside the repository."""

    relative_path = path.relative_to(repo_root)
    current = repo_root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _nonempty_string(value: Any) -> bool:
    """Return whether a value is a non-empty string."""

    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "BLOCKED_MISSING_SUCCESSOR_ROWS",
    "HIERARCHICAL_PAIRED_RELEASE_INPUT_MANIFEST_SCHEMA",
    "HIERARCHICAL_PAIRED_RELEASE_INPUT_REPORT_SCHEMA",
    "INPUTS_READY_ANALYSIS_NOT_RUN",
    "HierarchicalPairedReleaseInputError",
    "evaluate_hierarchical_paired_release_inputs",
    "load_hierarchical_paired_release_input_manifest",
    "validate_hierarchical_paired_release_input_manifest",
]
