"""Readiness/preflight checker for oracle-imitation warm-start training prerequisites.

Issue #1496 is the downstream warm-start training and benchmark-comparison step that
consumes the durable oracle-imitation dataset produced by issue #1470. The training run
itself (behaviour cloning, optional DAgger-style refinement, RL-only comparators) is
out of scope for shared-PC work: it collects no data, submits no Slurm, and trains
nothing. What *is* safe and useful is a bounded, read-only readiness check that answers
one question before any launch:

    "Are all durable prerequisites for the warm-start comparison actually in place,
     and if not, what exactly is blocking?"

This module composes the canonical launch-packet validator
(:func:`robot_sf.training.oracle_imitation_launch_packet.validate_launch_packet`) rather
than re-deriving dataset validation, and adds the training-side prerequisites: the
behaviour-cloning warm-start config, the RL-only baseline comparator config, an optional
PPO fine-tuning config, and the split/leakage contract document. It emits a compact
readiness manifest plus an explicit list of blockers so a launch script (or a maintainer)
can fail closed instead of starting an unbacked comparison.

The check distinguishes two failure classes:

* A malformed readiness manifest (missing/blank schema identity keys ``schema_version`` or
  ``experiment_id``, wrong schema version, non-mapping or unparseable YAML) raises
  :class:`WarmStartReadinessError`. The input contract is broken, so there is nothing
  meaningful to report.
* A well-formed manifest whose *prerequisites* are not yet satisfied (dataset not
  training-ready, a referenced config/contract file missing or its path key blank) is
  reported as ``status == "blocked"`` with a populated ``blockers`` list. This is the
  normal, expected output of a preflight; it only becomes a hard error
  (:class:`PrerequisitesNotReadyError`) when the caller opts into ``require_ready=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.oracle_imitation_launch_packet import (
    LaunchPacketError,
    validate_launch_packet,
)
from robot_sf.training.oracle_trace_uri_registry import (
    OracleTraceUriRegistryError,
    validate_trace_uri_registry,
)

_SCHEMA_VERSION = "oracle-imitation-warm-start-readiness.v1"
_READY_DECISION = "ready"
_BLOCKED_DECISION = "artifact_retrieval_blocked"
_OUT_OF_SCOPE_ACTIONS = {
    "benchmark_campaign_run": False,
    "claim_edits": False,
    "data_collection": False,
    "slurm_or_gpu_submission": False,
    "training": False,
}

# Required prerequisite path keys and whether the path must resolve to a regular file.
# The dataset launch packet is validated separately (it has its own fail-closed schema),
# so it is intentionally absent from this presence-only mapping.
_REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "warm_start_config",
    "baseline_config",
    "split_contract",
)
_OPTIONAL_CONFIG_KEYS: tuple[str, ...] = ("finetuning_config",)


class WarmStartReadinessError(ValueError):
    """Raised when a warm-start readiness manifest is malformed.

    This is distinct from an unmet prerequisite: a missing dataset or config is a
    *blocker* recorded in the report, whereas a broken manifest contract (bad schema,
    blank schema identity key, unparseable YAML) means there is no well-formed input to
    check.
    """


class PrerequisitesNotReadyError(WarmStartReadinessError):
    """Raised when a well-formed manifest still has unmet prerequisites under fail-closed mode.

    This is *not* a malformed-manifest error: the manifest parsed and validated, but one or
    more durable prerequisites (dataset not training-ready, missing config/contract file) are
    still blocking. It is raised only when the caller opts into ``require_ready=True`` so a
    launch gate can fail closed. Being a subclass of :class:`WarmStartReadinessError` lets
    callers that do not care about the distinction keep a single ``except`` clause, while the
    CLI catches it specifically to map blocked-vs-malformed onto distinct exit codes without a
    fragile error-message string check.
    """


def load_readiness_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a YAML warm-start readiness manifest.

    Args:
        manifest_path: YAML file to load.

    Returns:
        Parsed mapping.

    Raises:
        WarmStartReadinessError: If the file is missing or does not contain a mapping.
    """
    if not manifest_path.is_file():
        raise WarmStartReadinessError(f"readiness manifest is not a file: {manifest_path}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WarmStartReadinessError(
            f"failed to load readiness manifest YAML: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise WarmStartReadinessError("readiness manifest must be a YAML mapping")
    return payload


def check_warm_start_readiness(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Check oracle-imitation warm-start training prerequisites and report blockers.

    The check is read-only: it parses the manifest, validates the referenced dataset
    launch packet via the canonical validator, and confirms the training-side config and
    contract files exist. It never trains, collects data, or submits compute.

    Args:
        manifest_path: YAML readiness-manifest path.
        repo_root: Repository root used to resolve relative paths. Defaults to the current
            working directory.
        require_ready: When ``True``, raise :class:`WarmStartReadinessError` if any blocker
            remains (fail-closed gate for launch scripts). When ``False`` (default), return
            a report with ``status == "blocked"`` and the blocker list instead of raising.

    Returns:
        Readiness report: ``status`` (``"ready"``/``"blocked"``), ``schema_version``,
        ``experiment_id``, per-prerequisite detail under ``prerequisites``, and a
        ``blockers`` list.

    Raises:
        WarmStartReadinessError: If the manifest itself is malformed.
        PrerequisitesNotReadyError: If ``require_ready`` is set and at least one blocker
            remains (a :class:`WarmStartReadinessError` subclass, so existing single-clause
            handlers keep working).
    """
    root = (repo_root or Path.cwd()).resolve()
    manifest_path = _resolve_path(manifest_path, root)
    manifest = load_readiness_manifest(manifest_path)

    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise WarmStartReadinessError(f"schema_version must be {_SCHEMA_VERSION!r}")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise WarmStartReadinessError("experiment_id must be a non-empty string")

    blockers: list[str] = []
    prerequisites: dict[str, Any] = {}
    prerequisites["dataset_launch_packet"] = _check_dataset_packet(manifest, root, blockers)
    prerequisites["trace_uri_registry"] = _check_trace_uri_registry(manifest, root, blockers)
    for key in _REQUIRED_CONFIG_KEYS:
        prerequisites[key] = _check_required_file(manifest, key, root, blockers)
    for key in _OPTIONAL_CONFIG_KEYS:
        result = _check_optional_file(manifest, key, root, blockers)
        if result is not None:
            prerequisites[key] = result

    status = "ready" if not blockers else "blocked"
    readiness_decision = _READY_DECISION if not blockers else _BLOCKED_DECISION
    report = {
        "status": status,
        "readiness_decision": readiness_decision,
        "schema_version": _SCHEMA_VERSION,
        "experiment_id": experiment_id.strip(),
        "prerequisites": prerequisites,
        "blockers": blockers,
        "out_of_scope_actions": dict(_OUT_OF_SCOPE_ACTIONS),
    }

    if require_ready and blockers:
        joined = "\n- ".join(blockers)
        raise PrerequisitesNotReadyError(
            f"oracle-imitation warm-start prerequisites not ready:\n- {joined}"
        )
    return report


def _resolve_path(path: Path | str, repo_root: Path) -> Path:
    """Resolve ``path`` against ``repo_root`` unless it is already absolute.

    Returns:
        The resolved absolute path.
    """
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _check_dataset_packet(
    manifest: dict[str, Any],
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    """Validate the referenced #1470 dataset launch packet via the canonical validator.

    A missing key or a packet that is present but not yet ``training_ready`` is a blocker,
    not a manifest error: the dataset campaign simply has not landed durable trace URIs yet.

    Returns:
        Per-prerequisite detail mapping with at least ``path``, ``ready``, and either a
        ``detail`` reason (when not ready) or dataset identity fields (when ready).
    """
    raw_path = manifest.get("dataset_launch_packet")
    if not isinstance(raw_path, str) or not raw_path.strip():
        blockers.append("dataset_launch_packet must be a non-empty path string")
        return {"path": None, "ready": False, "detail": "missing dataset_launch_packet reference"}

    path_text = raw_path.strip()
    try:
        packet_report = validate_launch_packet(
            Path(path_text),
            repo_root=repo_root,
            require_training_ready=True,
        )
    except LaunchPacketError as exc:
        # Preserve the first actionable field-level launch-packet error instead
        # of only echoing the validator's generic header.
        reason = _first_launch_packet_error(exc)
        blockers.append(f"dataset_launch_packet not training-ready: {reason}")
        return {"path": path_text, "ready": False, "detail": str(exc)}

    return {
        "path": path_text,
        "ready": True,
        "dataset_id": packet_report["dataset_id"],
        "training_ready": packet_report["training_ready"],
    }


def _check_trace_uri_registry(
    manifest: dict[str, Any],
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    """Validate the referenced durable trace-URI registry with the canonical checker.

    Returns:
        Per-prerequisite detail with path, readiness, and registry identity when ready.
    """
    raw_path = manifest.get("trace_uri_registry")
    if not isinstance(raw_path, str) or not raw_path.strip():
        blockers.append("trace_uri_registry must be non-empty path string")
        return {"path": None, "ready": False, "detail": "missing trace_uri_registry reference"}

    path_text = raw_path.strip()
    try:
        registry_report = validate_trace_uri_registry(
            Path(path_text),
            repo_root=repo_root,
            require_training_ready=True,
        )
    except OracleTraceUriRegistryError as exc:
        reason = _first_trace_registry_error(exc)
        blockers.append(f"trace_uri_registry not training-ready: {reason}")
        return {"path": path_text, "ready": False, "detail": str(exc)}

    return {
        "path": path_text,
        "ready": True,
        "dataset_id": registry_report["dataset_id"],
        "trace_count": registry_report["trace_count"],
        "training_ready": registry_report["training_ready"],
    }


def _first_trace_registry_error(exc: OracleTraceUriRegistryError) -> str:
    """Return first actionable trace-URI registry validator error line."""
    for line in str(exc).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("oracle-imitation") and stripped.endswith("failed validation:"):
            continue
        return stripped.removeprefix("- ").strip()
    return str(exc)


def _first_launch_packet_error(exc: LaunchPacketError) -> str:
    """Return the first actionable launch-packet validator error line."""

    for line in str(exc).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_launch_packet_error_header(stripped):
            continue
        return stripped.removeprefix("- ").strip()
    return str(exc)


def _is_launch_packet_error_header(line: str) -> bool:
    """Return true for launch-packet validator summary header lines."""

    return (
        line.startswith("oracle-imitation")
        and "packet" in line
        and line.endswith("failed validation:")
    )


def _check_required_file(
    manifest: dict[str, Any],
    key: str,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    """Confirm a required prerequisite path is present and resolves to a regular file.

    Returns:
        Per-prerequisite detail mapping with ``path`` and ``ready`` keys.
    """
    raw_path = manifest.get(key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        blockers.append(f"{key} must be a non-empty path string")
        return {"path": None, "ready": False, "detail": f"missing {key} reference"}
    return _check_file_path(key, raw_path.strip(), repo_root, blockers)


def _check_optional_file(
    manifest: dict[str, Any],
    key: str,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any] | None:
    """Confirm an optional prerequisite path, if provided, resolves to a regular file.

    Returns ``None`` when the key is absent, so the prerequisite is simply omitted from the
    report. A present-but-blank value is treated as a manifest mistake worth surfacing.

    Returns:
        Per-prerequisite detail mapping, or ``None`` when the optional key is absent.
    """
    raw_path = manifest.get(key)
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path.strip():
        blockers.append(f"{key} must be a non-empty path string when provided")
        return {"path": None, "ready": False, "detail": f"blank {key} reference"}
    return _check_file_path(key, raw_path.strip(), repo_root, blockers)


def _check_file_path(
    key: str,
    path_text: str,
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    """Resolve ``path_text`` and record a blocker if it is not an existing regular file.

    Returns:
        Per-prerequisite detail mapping with ``path`` and ``ready`` keys.
    """
    resolved = _resolve_path(path_text, repo_root)
    if not resolved.is_file():
        blockers.append(f"{key} is not an existing file: {path_text}")
        return {"path": path_text, "ready": False, "detail": "file does not exist"}
    return {"path": path_text, "ready": True}


__all__ = [
    "PrerequisitesNotReadyError",
    "WarmStartReadinessError",
    "check_warm_start_readiness",
    "load_readiness_manifest",
]
