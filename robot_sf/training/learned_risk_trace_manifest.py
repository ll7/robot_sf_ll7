"""Fail-closed validation for durable learned-risk training trace manifests.

This module owns the *durable trace manifest* contract for the learned-risk model
(issue #2312, parent #1472). It is intentionally separate from the launch-packet
validator in :mod:`robot_sf.training.learned_risk_launch_packet`, which proves the
trace *input contract* against a small tracked fixture. The launch packet answers
"can the required fields/labels be represented?"; this manifest answers "are the
durable training trace artifacts and baseline pointer resolvable, or must training
fail closed?".

Decision boundary:

- Malformed manifests (wrong schema, bad types, worktree-local ``output/`` paths)
  raise :class:`LearnedRiskTraceManifestError`; they cannot be evaluated.
- Structurally valid manifests always return a report. When any durable artifact is
  a placeholder, an unresolved alias, an absent label, or a missing checksum, the
  ``training_readiness_decision`` is ``artifact_retrieval_blocked`` -- never an
  implied training-ready state. Only a fully resolvable contract yields
  ``ready_for_training_handoff``.

Resolvability is checked against the *local contract* only (durable URI scheme,
no placeholder alias, recorded checksum, present labels). It does not perform a
network fetch, so a ``ready_for_training_handoff`` decision still means
"locally contract-complete", not "bytes downloaded and verified".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.learned_risk_launch_packet import (
    _DURABLE_URI_PREFIXES,
    _REQUIRED_LABELS,
    _REQUIRED_SPLITS,
    sha256_file,
)

SCHEMA_VERSION = "learned-risk-trace-manifest.v1"

# Decision vocabulary. ``artifact_retrieval_blocked`` mirrors the issue archetype
# (`blocked-asset`) so downstream tooling and #1472 readiness can branch on it.
DECISION_READY = "ready_for_training_handoff"
DECISION_BLOCKED = "artifact_retrieval_blocked"

_REQUIRED_LABEL_NAMES = _REQUIRED_LABELS
# Tokens that mark an artifact pointer as a not-yet-materialized placeholder. A
# durable-looking URI that still carries one of these is treated as unresolved.
_PLACEHOLDER_TOKENS = ("pending", "tbd", "todo", "placeholder", "xxx", "<", ">")
_LABEL_PRESENT = "present"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LearnedRiskTraceManifestError(ValueError):
    """Raised when a learned-risk trace manifest is structurally invalid."""


def load_trace_manifest(config_path: Path) -> dict[str, Any]:
    """Load a learned-risk trace manifest YAML file.

    Args:
        config_path: Path to the manifest YAML.

    Returns:
        Parsed manifest mapping.

    Raises:
        LearnedRiskTraceManifestError: If the file is missing or not a mapping.
    """
    if not config_path.is_file():
        raise LearnedRiskTraceManifestError(f"trace manifest is not a file: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LearnedRiskTraceManifestError(f"trace manifest is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise LearnedRiskTraceManifestError("trace manifest must be a YAML mapping")
    return payload


def validate_trace_manifest(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a durable learned-risk trace manifest and decide training readiness.

    Args:
        config_path: Manifest YAML path.
        repo_root: Repository root for resolving relative local fixture paths.

    Returns:
        Compact report including ``training_readiness_decision`` and ``blockers``.
        The decision is ``artifact_retrieval_blocked`` whenever any durable input
        is unresolved; it never reports ready unless every contract check passes.

    Raises:
        LearnedRiskTraceManifestError: If the manifest shape is invalid (cannot be
            evaluated). Unresolved-but-well-formed manifests do not raise; they are
            reported as blocked.
    """
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve_path(config_path, root)
    manifest = load_trace_manifest(config_path)

    # Structural invariants: violations mean the manifest cannot be evaluated.
    checksums, label_availability = _validate_structure(manifest)

    # Resolvability checks: violations are fail-closed blockers, not hard errors.
    blockers: list[str] = []
    baseline_uri = manifest.get("baseline_artifact_uri")
    # The baseline is a durable artifact too: require a recorded digest so the
    # training run can verify the baseline bytes it retrieves, matching the
    # trace-artifact and local-fixture checks.
    _check_artifact_uri("baseline_artifact_uri", baseline_uri, blockers, checksums=checksums)

    trace_artifacts = manifest.get("trace_artifacts")
    if not isinstance(trace_artifacts, list) or not trace_artifacts:
        blockers.append("trace_artifacts must be a non-empty list of durable URIs")
        trace_artifacts = []
    for index, uri in enumerate(trace_artifacts):
        _check_artifact_uri(f"trace_artifacts[{index}]", uri, blockers, checksums=checksums)

    _check_split_ids(manifest.get("split_ids"), blockers)
    _check_required_fields(manifest.get("required_episode_fields"), blockers)
    _check_labels(label_availability, blockers)
    _check_local_fixtures(manifest.get("local_fixtures"), checksums, root, blockers)

    retrieval_status = manifest.get("retrieval_status")
    if retrieval_status not in ("available", "blocked"):
        blockers.append("retrieval_status must be 'available' or 'blocked'")
    elif retrieval_status != "available":
        blockers.append("retrieval_status is not 'available'")

    decision = DECISION_BLOCKED if blockers else DECISION_READY
    return {
        "status": "ok",
        "schema_version": manifest["schema_version"],
        "source_issue": manifest.get("source_issue"),
        "parent_issue": manifest.get("parent_issue"),
        "candidate_id": manifest["candidate_id"],
        "trace_schema_version": manifest["trace_schema_version"],
        "baseline_artifact_uri": baseline_uri,
        "split_ids": (
            list(manifest.get("split_ids")) if isinstance(manifest.get("split_ids"), list) else []
        ),
        "label_availability": _normalized_labels(label_availability),
        "training_ready": not blockers,
        "training_readiness_decision": decision,
        "blockers": blockers,
    }


def _validate_structure(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enforce structural invariants and return the checksum/label mappings.

    Returns:
        A ``(checksums, label_availability)`` tuple of the two mappings the
        resolvability checks need, each defaulted to ``{}`` when absent.

    Raises:
        LearnedRiskTraceManifestError: If any structural invariant is violated,
            since a malformed manifest cannot be evaluated for readiness.
    """
    structural: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        structural.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("candidate_id") != "learned_risk_model_v1":
        structural.append("candidate_id must be 'learned_risk_model_v1'")
    trace_schema = manifest.get("trace_schema_version")
    if not isinstance(trace_schema, str) or not trace_schema.strip():
        structural.append("trace_schema_version must be a non-empty string")
    checksums = manifest.get("checksums", {})
    if not isinstance(checksums, dict):
        structural.append("checksums must be a mapping")
        checksums = {}
    label_availability = manifest.get("label_availability")
    if not isinstance(label_availability, dict):
        structural.append("label_availability must be a mapping")
        label_availability = {}
    if structural:
        joined = "\n- ".join(structural)
        raise LearnedRiskTraceManifestError(
            f"learned-risk trace manifest is structurally invalid:\n- {joined}"
        )
    return checksums, label_availability


def _resolve_path(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _is_placeholder(value: str) -> bool:
    """Return True if a string carries an unresolved-placeholder token."""
    lowered = value.lower()
    return any(token in lowered for token in _PLACEHOLDER_TOKENS)


def _check_artifact_uri(
    key: str,
    value: Any,
    blockers: list[str],
    *,
    checksums: dict[str, Any] | None = None,
) -> None:
    """Record a blocker unless ``value`` is a resolvable durable artifact pointer.

    ``output/`` paths are a hard error (they cannot be durable inputs); everything
    else degrades to a fail-closed blocker so the decision stays non-ready.
    """
    if not isinstance(value, str) or not value.strip():
        blockers.append(f"{key} must be a non-empty durable artifact URI")
        return
    text = value.strip()
    if not text.startswith(_DURABLE_URI_PREFIXES):
        # Only a non-durable pointer can be a worktree-local output path. A durable
        # URI may legitimately carry an "output" path segment (e.g.
        # s3://bucket/output/traces.jsonl), so the local-output guard must run only
        # after the durable-scheme check, not before it.
        if "output" in Path(text).parts:
            raise LearnedRiskTraceManifestError(
                f"{key} must not depend on worktree-local output: {text}"
            )
        blockers.append(f"{key} must use a durable URI scheme {_DURABLE_URI_PREFIXES}: {text}")
        return
    if _is_placeholder(text):
        blockers.append(f"{key} is an unresolved placeholder alias: {text}")
        return
    # Durable, concrete trace artifacts must carry a recorded digest so the
    # downstream training run can verify the bytes it retrieves.
    if checksums is not None:
        digest = checksums.get(text)
        if not isinstance(digest, str) or not _SHA256_RE.match(digest.strip().lower()):
            blockers.append(f"checksums missing a SHA-256 digest for {text}")


def _check_split_ids(split_ids: Any, blockers: list[str]) -> None:
    if not isinstance(split_ids, list) or not split_ids:
        blockers.append("split_ids must be a non-empty list")
        return
    present = {str(value).strip() for value in split_ids if str(value).strip()}
    missing = sorted(set(_REQUIRED_SPLITS) - present)
    if missing:
        blockers.append(f"split_ids is missing required slices: {missing}")


def _check_required_fields(fields: Any, blockers: list[str]) -> None:
    required = ("scenario_id", "seed", "candidate_id", "termination_reason", "labels")
    if not isinstance(fields, list) or not fields:
        blockers.append("required_episode_fields must be a non-empty list")
        return
    present = {str(value).strip() for value in fields if str(value).strip()}
    missing = sorted(set(required) - present)
    if missing:
        blockers.append(f"required_episode_fields is missing: {missing}")


def _check_labels(label_availability: dict[str, Any], blockers: list[str]) -> None:
    for label in _REQUIRED_LABEL_NAMES:
        state = label_availability.get(label)
        if state != _LABEL_PRESENT:
            blockers.append(f"label '{label}' is not '{_LABEL_PRESENT}' (got {state!r})")


def _normalized_labels(label_availability: dict[str, Any]) -> dict[str, Any]:
    return {label: label_availability.get(label) for label in _REQUIRED_LABEL_NAMES}


def _check_local_fixtures(
    local_fixtures: Any,
    checksums: dict[str, Any],
    repo_root: Path,
    blockers: list[str],
) -> None:
    """Validate optional tracked fixture snapshots referenced by the manifest.

    Local fixtures are not a substitute for durable artifacts, but when present
    they must exist and match a recorded checksum so the manifest cannot drift.
    """
    if local_fixtures is None:
        return
    if not isinstance(local_fixtures, list):
        blockers.append("local_fixtures must be a list when present")
        return
    for raw in local_fixtures:
        if not isinstance(raw, str) or not raw.strip():
            blockers.append("local_fixtures entries must be non-empty strings")
            continue
        text = raw.strip()
        if "output" in Path(text).parts:
            raise LearnedRiskTraceManifestError(
                f"local_fixtures must not depend on worktree-local output: {text}"
            )
        local_path = _resolve_path(text, repo_root)
        if not local_path.is_file():
            blockers.append(f"local fixture is missing: {text}")
            continue
        expected = checksums.get(text)
        if not isinstance(expected, str) or not expected.strip():
            blockers.append(f"checksums missing a SHA-256 digest for {text}")
            continue
        actual = sha256_file(local_path)
        if actual != expected.strip().lower():
            blockers.append(f"checksum mismatch for {text}: expected {expected}, got {actual}")


__all__ = [
    "DECISION_BLOCKED",
    "DECISION_READY",
    "SCHEMA_VERSION",
    "LearnedRiskTraceManifestError",
    "load_trace_manifest",
    "validate_trace_manifest",
]
