"""Durable trace-URI registry validation for oracle-imitation artifacts.

Oracle-imitation training is blocked until the raw train/validation/evaluation traces have
*durable* pointers (artifact URIs that survive worktree cleanup) plus enough metadata to
verify integrity and resolvability. This module owns that registry contract: a small,
git-tracked manifest that records, per split, the durable trace URI, its SHA-256 checksum,
its split/trace identity, the registry schema version, and a retrieval status. Large traces
themselves stay out of git; only the resolvable pointers and checksums live here.

The registry is deliberately separate from the pre-Slurm launch packet
(:mod:`robot_sf.training.oracle_imitation_launch_packet`). The launch packet describes *how a
collection run will be launched*; this registry describes *whether the resulting raw traces
are durably retrievable* so a downstream imitation-training lane can mechanically leave the
``artifact_retrieval_blocked`` state. A trace is only ``training_ready`` when every required
split has a concrete durable URI, a valid checksum, and ``retrieval_status: resolvable``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.artifact_catalog import sha256_file

_SCHEMA_VERSION = "oracle-trace-uri-registry.v1"
_REQUIRED_SPLITS = ("train", "validation", "evaluation")
_RETRIEVAL_STATES = ("resolvable", "pending", "blocked")
_DURABLE_URI_PREFIXES = ("wandb-artifact://", "artifact://", "s3://", "gs://", "https://")
_HEX_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
# Sentinel sha256 value allowed only while a trace is not yet resolvable.
_PENDING_SHA256 = "pending"


class OracleTraceUriRegistryError(ValueError):
    """Raised when an oracle-imitation trace-URI registry fails validation."""


def load_trace_uri_registry(registry_path: Path) -> dict[str, Any]:
    """Load a YAML trace-URI registry.

    Args:
        registry_path: YAML file to load.

    Returns:
        Parsed mapping.

    Raises:
        OracleTraceUriRegistryError: If the file is missing or is not a mapping.
    """
    if not registry_path.is_file():
        raise OracleTraceUriRegistryError(f"trace-URI registry is not a file: {registry_path}")
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OracleTraceUriRegistryError(
            f"failed to load trace-URI registry YAML: {registry_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise OracleTraceUriRegistryError("trace-URI registry must be a YAML mapping")
    return payload


def validate_trace_uri_registry(
    registry_path: Path,
    *,
    repo_root: Path | None = None,
    require_training_ready: bool = False,
) -> dict[str, Any]:
    """Validate a durable trace-URI registry and return a compact report.

    Args:
        registry_path: YAML registry path.
        repo_root: Repository root used to resolve any optional local-mirror paths. Defaults to
            the current working directory.
        require_training_ready: Fail closed unless every required split is durably resolvable
            (concrete durable URI, valid checksum, and ``retrieval_status: resolvable``).

    Returns:
        Validation report with status, per-split trace ids, retrieval status, and the computed
        ``training_ready`` gate.

    Raises:
        OracleTraceUriRegistryError: If any fail-closed registry invariant is violated.
    """
    root = (repo_root or Path.cwd()).resolve()
    registry_path = _resolve_path(registry_path, root)
    registry = load_trace_uri_registry(registry_path)
    errors: list[str] = []

    if registry.get("schema_version") != _SCHEMA_VERSION:
        errors.append(f"schema_version must be {_SCHEMA_VERSION!r}")
    _require_non_empty_string(registry, "dataset_id", errors)

    traces = _validate_traces(registry, root, errors)
    splits_to_trace_ids, retrieval_status, resolvable_required, source_manifests = (
        _summarize_traces(traces)
    )
    training_ready = _validate_training_ready_gate(
        splits_to_trace_ids,
        resolvable_required,
        require_training_ready=require_training_ready,
        errors=errors,
    )

    if errors:
        joined = "\n- ".join(errors)
        raise OracleTraceUriRegistryError(
            f"oracle-imitation trace-URI registry failed validation:\n- {joined}"
        )

    return {
        "status": "valid",
        "schema_version": registry["schema_version"],
        "dataset_id": registry["dataset_id"],
        "trace_count": len(traces),
        "splits": {split: splits_to_trace_ids.get(split, []) for split in _REQUIRED_SPLITS},
        "retrieval_status": retrieval_status,
        "source_manifests": source_manifests,
        "training_ready": training_ready,
    }


def _validate_traces(
    registry: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Validate the ``traces`` list and return the entries that parsed as mappings.

    Returns:
        The trace entries that parsed as mappings (entries that were not mappings are skipped).
    """
    raw_traces = registry.get("traces")
    if not isinstance(raw_traces, list) or not raw_traces:
        errors.append("traces must be a non-empty list")
        return []

    parsed: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()
    for index, raw in enumerate(raw_traces):
        if not isinstance(raw, dict):
            errors.append(f"traces[{index}] must be a mapping")
            continue
        _validate_single_trace(index, raw, seen_trace_ids, repo_root, errors)
        parsed.append(raw)
    return parsed


def _validate_single_trace(
    index: int,
    trace: dict[str, Any],
    seen_trace_ids: set[str],
    repo_root: Path,
    errors: list[str],
) -> None:
    """Validate one registry trace entry, appending fail-closed errors in place."""
    label = f"traces[{index}]"

    split = trace.get("split")
    if split not in _REQUIRED_SPLITS:
        errors.append(f"{label}.split must be one of {_REQUIRED_SPLITS}")

    trace_id = trace.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id.strip():
        errors.append(f"{label}.trace_id must be a non-empty string")
    elif trace_id.strip() in seen_trace_ids:
        errors.append(f"{label}.trace_id is duplicated: {trace_id.strip()}")
    else:
        seen_trace_ids.add(trace_id.strip())

    status = trace.get("retrieval_status")
    if status not in _RETRIEVAL_STATES:
        errors.append(f"{label}.retrieval_status must be one of {_RETRIEVAL_STATES}")

    uri = _validate_trace_uri(label, trace.get("uri"), errors)
    _validate_trace_checksum(label, trace.get("sha256"), status, errors)
    _validate_local_mirror(label, trace.get("local_mirror"), trace.get("sha256"), repo_root, errors)
    _validate_source_manifest_metadata(label, trace, status, repo_root, errors)
    # Mark the concrete-durable check result so the gate logic can reuse it.
    trace["_uri_is_concrete_durable"] = uri is not None and _is_concrete_durable_uri(uri)


def _validate_trace_uri(label: str, raw_uri: Any, errors: list[str]) -> str | None:
    """Validate a trace URI is a non-empty, durable, non-output pointer.

    Returns:
        The stripped URI string, or ``None`` when the value was missing or not a string.
    """
    if not isinstance(raw_uri, str) or not raw_uri.strip():
        errors.append(f"{label}.uri must be a non-empty durable URI string")
        return None
    uri = raw_uri.strip()
    if not _is_durable_uri(uri):
        errors.append(f"{label}.uri must be a durable URI (one of {_DURABLE_URI_PREFIXES}): {uri}")
    if "output" in Path(uri).parts:
        errors.append(f"{label}.uri must not depend on worktree-local output: {uri}")
    return uri


def _validate_trace_checksum(
    label: str,
    raw_sha: Any,
    status: Any,
    errors: list[str],
) -> None:
    """Validate the declared SHA-256 checksum, allowing a sentinel only while not resolvable.

    A resolvable trace must carry a concrete 64-character SHA-256 digest so downstream
    consumers can verify integrity. While a trace is still ``pending`` or ``blocked`` the
    checksum may be omitted or set to the ``"pending"`` sentinel, because the artifact may not
    exist yet.
    """
    if raw_sha is None:
        if status == "resolvable":
            errors.append(f"{label}.sha256 is required when retrieval_status is 'resolvable'")
        return
    if not isinstance(raw_sha, str) or not raw_sha.strip():
        errors.append(f"{label}.sha256 must be a string when provided")
        return
    digest = raw_sha.strip().lower()
    if digest == _PENDING_SHA256:
        if status == "resolvable":
            errors.append(
                f"{label}.sha256 must be a concrete digest when retrieval_status is 'resolvable'"
            )
        return
    if len(digest) != _HEX_SHA256_LENGTH or any(ch not in _HEX_DIGITS for ch in digest):
        errors.append(f"{label}.sha256 must be a 64-character SHA-256 digest")


def _validate_local_mirror(
    label: str,
    raw_mirror: Any,
    raw_sha: Any,
    repo_root: Path,
    errors: list[str],
) -> None:
    """Verify an optional staged local mirror matches the declared checksum.

    The registry points at durable *remote* URIs whose bytes cannot be hashed offline. When a
    contributor stages a local mirror copy to prove integrity, this verifies the file exists
    and its SHA-256 matches the declared ``sha256`` so a stale or corrupt mirror fails closed.
    """
    if raw_mirror is None:
        return
    if not isinstance(raw_mirror, str) or not raw_mirror.strip():
        errors.append(f"{label}.local_mirror must be a non-empty path string when provided")
        return
    mirror_text = raw_mirror.strip()
    if "output" in Path(mirror_text).parts:
        errors.append(
            f"{label}.local_mirror must not depend on worktree-local output: {mirror_text}"
        )
        return
    mirror_path = _resolve_path(mirror_text, repo_root)
    if not mirror_path.is_file():
        errors.append(f"{label}.local_mirror file is missing: {mirror_text}")
        return
    if (
        not isinstance(raw_sha, str)
        or not raw_sha.strip()
        or raw_sha.strip().lower() == _PENDING_SHA256
    ):
        errors.append(f"{label}.local_mirror requires a concrete sha256 to verify against")
        return
    actual = sha256_file(mirror_path)
    expected = raw_sha.strip().lower()
    if actual != expected:
        errors.append(
            f"{label}.local_mirror checksum mismatch for {mirror_text}: "
            f"expected {expected}, got {actual}"
        )


def _validate_source_manifest_metadata(
    label: str,
    trace: dict[str, Any],
    status: Any,
    repo_root: Path,
    errors: list[str],
) -> None:
    """Validate provenance metadata for the source manifest behind a trace JSONL."""
    raw_uri = trace.get("source_manifest_uri")
    raw_sha = trace.get("source_manifest_sha256")
    raw_mirror = trace.get("source_manifest_local_mirror")

    if status == "resolvable":
        if raw_uri is None:
            errors.append(f"{label}.source_manifest_uri required retrieval_status 'resolvable'")
        if raw_sha is None:
            errors.append(f"{label}.source_manifest_sha256 required retrieval_status 'resolvable'")

    uri = None
    if raw_uri is not None:
        uri = _validate_trace_uri(f"{label}.source_manifest", raw_uri, errors)
    _validate_trace_checksum(f"{label}.source_manifest", raw_sha, status, errors)
    _validate_local_mirror(
        f"{label}.source_manifest",
        raw_mirror,
        raw_sha,
        repo_root,
        errors,
    )
    trace["_source_manifest_is_concrete_durable"] = uri is not None and _is_concrete_durable_uri(
        uri
    )


def _summarize_traces(
    traces: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, bool], dict[str, str]]:
    """Build per-split trace-id lists, a trace-id->status map, and resolvable-required flags.

    Returns:
        A tuple of (split -> trace ids, trace id -> retrieval status, split -> whether at least
        one concrete durable resolvable trace exists for that required split).
    """
    splits_to_trace_ids: dict[str, list[str]] = {}
    retrieval_status: dict[str, str] = {}
    resolvable_required: dict[str, bool] = dict.fromkeys(_REQUIRED_SPLITS, False)
    source_manifests: dict[str, str] = {}
    for trace in traces:
        split = trace.get("split")
        trace_id = trace.get("trace_id")
        status = trace.get("retrieval_status")
        if not isinstance(trace_id, str) or not trace_id.strip():
            continue
        trace_id = trace_id.strip()
        if split in _REQUIRED_SPLITS:
            splits_to_trace_ids.setdefault(split, []).append(trace_id)
        if isinstance(status, str):
            retrieval_status[trace_id] = status
        source_manifest_uri = trace.get("source_manifest_uri")
        if isinstance(source_manifest_uri, str) and source_manifest_uri.strip():
            source_manifests[trace_id] = source_manifest_uri.strip()
        if (
            split in _REQUIRED_SPLITS
            and status == "resolvable"
            and trace.get("_uri_is_concrete_durable", False)
            and trace.get("_source_manifest_is_concrete_durable", False)
        ):
            resolvable_required[split] = True
    return splits_to_trace_ids, retrieval_status, resolvable_required, source_manifests


def _validate_training_ready_gate(
    splits_to_trace_ids: dict[str, list[str]],
    resolvable_required: dict[str, bool],
    *,
    require_training_ready: bool,
    errors: list[str],
) -> bool:
    """Compute and optionally enforce the ``training_ready`` gate.

    The lane leaves ``artifact_retrieval_blocked`` only when every required split has at least
    one concrete, durable, ``resolvable`` trace pointer.

    Returns:
        ``True`` when every required split is concretely, durably resolvable, else ``False``.
    """
    missing_splits = sorted(s for s in _REQUIRED_SPLITS if not splits_to_trace_ids.get(s))
    unresolved_splits = sorted(
        s for s in _REQUIRED_SPLITS if splits_to_trace_ids.get(s) and not resolvable_required[s]
    )
    training_ready = not missing_splits and not unresolved_splits

    if not require_training_ready:
        return training_ready

    if missing_splits:
        errors.append(
            "training-ready trace-URI registry must include traces for every required split; "
            f"missing: {', '.join(missing_splits)}"
        )
    if unresolved_splits:
        errors.append(
            "training-ready trace-URI registry requires a concrete durable resolvable trace for "
            f"every split; not resolvable: {', '.join(unresolved_splits)}"
        )
    return training_ready


def _resolve_path(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _require_non_empty_string(registry: dict[str, Any], key: str, errors: list[str]) -> None:
    value = registry.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")


def _is_durable_uri(uri: str) -> bool:
    return uri.startswith(_DURABLE_URI_PREFIXES)


def _is_pending_durable_uri(uri: str) -> bool:
    return _is_durable_uri(uri) and uri.rstrip().endswith(":pending")


def _is_concrete_durable_uri(uri: str) -> bool:
    return _is_durable_uri(uri) and not _is_pending_durable_uri(uri)


__all__ = [
    "OracleTraceUriRegistryError",
    "load_trace_uri_registry",
    "validate_trace_uri_registry",
]
