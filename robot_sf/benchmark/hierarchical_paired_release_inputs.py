"""Fail-closed input readiness reporting for issue #5351's release analysis.

The hierarchical paired analysis is intentionally downstream of the #4364
successor release. This module records that dependency in a machine-readable
manifest and prevents a missing release tag, typed-ledger export, or invalid
analysis report from being mistaken for an analysable release dataset. It does
not compute statistics, change frozen metric semantics, or promote a benchmark
claim.
"""

from __future__ import annotations

import json
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
ANALYSIS_DELIVERED_REVIEW_PENDING = "analysis_delivered_review_pending"
BLOCKED_INVALID_ANALYSIS_ARTIFACT = "blocked_invalid_analysis_artifact"
DEFAULT_ANALYSIS_REPORT_RELATIVE_PATH = (
    "docs/context/evidence/issue_5351_hierarchical_paired_release_analysis/"
    "hierarchical_paired_release_analysis_report.json"
)
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
    that the input contract and its delivered analysis remain distinct from claim admission.

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
        A deterministic report. Input presence and analysis delivery are separate
        states; the claim gate remains blocked pending human review.
    """

    normalized = validate_hierarchical_paired_release_input_manifest(manifest)
    root = Path(repo_root).resolve()
    successor_release = dict(normalized["successor_release"])
    successor_blockers = _successor_release_blockers(successor_release, repo_root=root)
    blockers = list(successor_blockers)
    analysis_artifact = _evaluate_analysis_artifact(root)
    if not blockers and analysis_artifact["status"] == BLOCKED_INVALID_ANALYSIS_ARTIFACT:
        blockers.append(
            {
                "field": "analysis_artifact",
                "reason": str(analysis_artifact["reason"]),
            }
        )

    if blockers:
        status = (
            BLOCKED_INVALID_ANALYSIS_ARTIFACT
            if analysis_artifact["status"] == BLOCKED_INVALID_ANALYSIS_ARTIFACT
            and not successor_blockers
            else BLOCKED_MISSING_SUCCESSOR_ROWS
        )
        protocol_status = status
        claim_gate = {
            "status": "blocked_analysis_not_run",
            "reason": (
                "successor-release inputs are missing"
                if status == BLOCKED_MISSING_SUCCESSOR_ROWS
                else "the tracked analysis artifact failed validation"
            ),
        }
        analysis_executed = False
    elif analysis_artifact["status"] == ANALYSIS_DELIVERED_REVIEW_PENDING:
        status = ANALYSIS_DELIVERED_REVIEW_PENDING
        protocol_status = "delivered_analysis_pending_human_review"
        claim_gate = {
            "status": "blocked_review_pending",
            "reason": str(analysis_artifact["claim_gate_reason"]),
        }
        analysis_executed = True
    else:
        status = INPUTS_READY_ANALYSIS_NOT_RUN
        protocol_status = "declared_pending_analysis"
        claim_gate = {
            "status": "blocked_analysis_not_run",
            "reason": "inputs are present but the hierarchical paired analysis has not run",
        }
        analysis_executed = False
    return {
        "schema_version": HIERARCHICAL_PAIRED_RELEASE_INPUT_REPORT_SCHEMA,
        "issue": 5351,
        "status": status,
        "claim_boundary": normalized["claim_boundary"],
        "evidence_status": "not_benchmark_evidence",
        "successor_release": successor_release,
        "blocking_prerequisites": blockers,
        "analysis_artifact": analysis_artifact,
        "protocol_conformance": [
            {
                "id": item["id"],
                "declared_delivery": item["declared_delivery"],
                "status": protocol_status,
            }
            for item in normalized["protocol"]
        ],
        "claim_gate": claim_gate,
        "semantics": {
            "benchmark_metrics_changed": False,
            "analysis_executed": analysis_executed,
            "claim_promotion": "none",
        },
    }


def _evaluate_analysis_artifact(repo_root: Path) -> dict[str, Any]:
    """Check the tracked #5351 report without promoting its interpretation.

    Returns:
        Machine-readable presence, validity, checksum, and claim-gate state.
    """

    report_path = repo_root / DEFAULT_ANALYSIS_REPORT_RELATIVE_PATH
    base: dict[str, Any] = {
        "path": DEFAULT_ANALYSIS_REPORT_RELATIVE_PATH,
        "present": report_path.is_file(),
        "status": "not_present",
    }
    if not report_path.is_file():
        return base

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": BLOCKED_INVALID_ANALYSIS_ARTIFACT,
            "reason": f"analysis report is unreadable: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            **base,
            "status": BLOCKED_INVALID_ANALYSIS_ARTIFACT,
            "reason": "analysis report must be a JSON object",
        }
    if payload.get("issue") != 5351 or payload.get("analysis_executed") is not True:
        return {
            **base,
            "status": BLOCKED_INVALID_ANALYSIS_ARTIFACT,
            "reason": "analysis report must identify issue 5351 with analysis_executed=true",
        }
    if payload.get("evidence_status") != "not_benchmark_evidence":
        return {
            **base,
            "status": BLOCKED_INVALID_ANALYSIS_ARTIFACT,
            "reason": "analysis report must retain evidence_status=not_benchmark_evidence",
        }
    claim_gate = payload.get("claim_gate")
    if not isinstance(claim_gate, Mapping) or claim_gate.get("status") != "blocked_review_pending":
        return {
            **base,
            "status": BLOCKED_INVALID_ANALYSIS_ARTIFACT,
            "reason": "analysis report must retain claim_gate.status=blocked_review_pending",
        }
    return {
        **base,
        "status": ANALYSIS_DELIVERED_REVIEW_PENDING,
        "analysis_executed": True,
        "evidence_status": str(payload["evidence_status"]),
        "claim_gate_status": str(claim_gate["status"]),
        "claim_gate_reason": str(claim_gate.get("reason", "human review is required")),
        "sha256": sha256_file(report_path),
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
    "ANALYSIS_DELIVERED_REVIEW_PENDING",
    "BLOCKED_INVALID_ANALYSIS_ARTIFACT",
    "BLOCKED_MISSING_SUCCESSOR_ROWS",
    "DEFAULT_ANALYSIS_REPORT_RELATIVE_PATH",
    "HIERARCHICAL_PAIRED_RELEASE_INPUT_MANIFEST_SCHEMA",
    "HIERARCHICAL_PAIRED_RELEASE_INPUT_REPORT_SCHEMA",
    "INPUTS_READY_ANALYSIS_NOT_RUN",
    "HierarchicalPairedReleaseInputError",
    "evaluate_hierarchical_paired_release_inputs",
    "load_hierarchical_paired_release_input_manifest",
    "validate_hierarchical_paired_release_input_manifest",
]
