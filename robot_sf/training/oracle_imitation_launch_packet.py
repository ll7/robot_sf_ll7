"""Validation helpers for oracle-imitation dataset launch packets."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.oracle_trace_uri_registry import (
    OracleTraceUriRegistryError,
    validate_trace_uri_registry,
)

_SCHEMA_VERSION = "oracle-imitation-launch-packet.v1"
_SPLITS = ("train", "validation", "evaluation")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DURABLE_URI_PREFIXES = ("wandb-artifact://", "artifact://", "s3://", "gs://", "https://")
# Required destinations the launch packet must declare before a collection job runs:
# where collection logs, raw trace output, and the dataset manifest are durably written.
_REQUIRED_COLLECTION_ROOTS = ("log_root", "dataset_output_root", "manifest_destination")


class LaunchPacketError(ValueError):
    """Raised when an oracle-imitation launch packet fails validation."""


def load_launch_packet(config_path: Path) -> dict[str, Any]:
    """Load a YAML oracle-imitation launch packet.

    Args:
        config_path: YAML file to load.

    Returns:
        Parsed mapping.

    Raises:
        LaunchPacketError: If the file is missing or does not contain a mapping.
    """
    if not config_path.is_file():
        raise LaunchPacketError(f"launch packet is not a file: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LaunchPacketError(f"failed to load launch packet YAML: {config_path}") from exc
    if not isinstance(payload, dict):
        raise LaunchPacketError("launch packet must be a YAML mapping")
    return payload


def validate_launch_packet(
    config_path: Path,
    *,
    repo_root: Path | None = None,
    require_training_ready: bool = False,
) -> dict[str, Any]:
    """Validate an oracle-imitation launch packet and return a compact report.

    Args:
        config_path: YAML launch-packet path.
        repo_root: Repository root. Defaults to the current working directory.
        require_training_ready: Also require concrete durable trace artifact inputs for
            downstream imitation training.

    Returns:
        Validation report with status, checked fields, and artifact paths.

    Raises:
        LaunchPacketError: If any fail-closed launch-packet invariant is violated.
    """
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve_path(config_path, root)
    packet = load_launch_packet(config_path)
    errors: list[str] = []

    if packet.get("schema_version") != _SCHEMA_VERSION:
        errors.append(f"schema_version must be {_SCHEMA_VERSION!r}")

    _require_non_empty_string(packet, "dataset_id", errors)
    _require_non_empty_string(packet, "source_candidate", errors)
    _require_existing_path(packet, "source_candidate_config", root, errors)
    _require_existing_path(packet, "source_report", root, errors)
    _require_existing_path(packet, "split_contract", root, errors)
    _require_existing_path(packet, "scenario_source", root, errors)
    _validate_scenarios(packet, errors)
    _validate_seed_sets(packet, root, errors)
    _validate_episodes(packet, errors)
    _validate_hard_slices(packet, errors)
    _validate_relabeling(packet, errors)
    _validate_generating_commit(packet, errors)
    artifact_paths = _validate_artifacts(packet, root, errors)
    collection_roots = _validate_collection_roots(packet, errors)
    trace_uri_registry = _validate_trace_uri_registry_pointer(
        packet,
        root,
        require_training_ready=require_training_ready,
        errors=errors,
    )
    training_ready = _validate_training_artifact_gate(
        artifact_paths,
        trace_uri_registry=trace_uri_registry,
        require_training_ready=require_training_ready,
        errors=errors,
    )

    if errors:
        joined = "\n- ".join(errors)
        raise LaunchPacketError(f"oracle-imitation launch packet failed validation:\n- {joined}")

    return {
        "status": "valid",
        "schema_version": packet["schema_version"],
        "dataset_id": packet["dataset_id"],
        "source_candidate": packet["source_candidate"],
        "scenario_count": len(packet["scenario_ids"]),
        "episode_count": sum(len(packet["episode_ids_by_split"][split]) for split in _SPLITS),
        "seeds_by_split": {split: list(packet["seeds_by_split"][split]) for split in _SPLITS},
        "artifact_paths": artifact_paths,
        "collection_roots": collection_roots,
        "trace_uri_registry": trace_uri_registry,
        "training_ready": training_ready,
    }


def _resolve_path(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _require_non_empty_string(packet: dict[str, Any], key: str, errors: list[str]) -> None:
    value = packet.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")


def _require_existing_path(
    packet: dict[str, Any],
    key: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    value = packet.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty path string")
        return
    path = _resolve_path(value, repo_root)
    if not path.exists():
        errors.append(f"{key} does not exist: {value}")


def _validate_scenarios(packet: dict[str, Any], errors: list[str]) -> None:
    scenario_ids = packet.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids:
        errors.append("scenario_ids must be a non-empty list")
        return
    normalized: list[str] = []
    for raw in scenario_ids:
        if not isinstance(raw, str) or not raw.strip():
            errors.append("scenario_ids entries must be non-empty strings")
            return
        normalized.append(raw.strip())
    if len(set(normalized)) != len(normalized):
        errors.append("scenario_ids must not contain duplicates")


def _validate_seed_sets(packet: dict[str, Any], repo_root: Path, errors: list[str]) -> None:
    seeds_by_split = packet.get("seeds_by_split")
    if not isinstance(seeds_by_split, dict):
        errors.append("seeds_by_split must be a mapping")
        return

    split_sets = _split_seed_sets(seeds_by_split, errors)
    _validate_seed_overlap(split_sets, errors)
    _validate_seed_refs(packet, repo_root, split_sets, errors)


def _split_seed_sets(
    seeds_by_split: dict[str, Any],
    errors: list[str],
) -> dict[str, set[int]]:
    split_sets: dict[str, set[int]] = {}
    for split in _SPLITS:
        raw_seeds = seeds_by_split.get(split)
        if not isinstance(raw_seeds, list) or not raw_seeds:
            errors.append(f"seeds_by_split.{split} must be a non-empty list")
            continue
        try:
            split_sets[split] = {int(seed) for seed in raw_seeds}
        except (TypeError, ValueError):
            errors.append(f"seeds_by_split.{split} must contain integer seeds")
    return split_sets


def _validate_seed_overlap(split_sets: dict[str, set[int]], errors: list[str]) -> None:
    for index, left in enumerate(_SPLITS):
        for right in _SPLITS[index + 1 :]:
            overlap = sorted(split_sets.get(left, set()) & split_sets.get(right, set()))
            if overlap:
                errors.append(f"seed overlap between {left} and {right}: {overlap}")


def _validate_seed_refs(
    packet: dict[str, Any],
    repo_root: Path,
    split_sets: dict[str, set[int]],
    errors: list[str],
) -> None:
    refs = packet.get("seed_set_refs", {})
    if refs is None:
        refs = {}
    if not isinstance(refs, dict):
        errors.append("seed_set_refs must be a mapping when provided")
        return
    seed_manifest = refs.get("manifest")
    if seed_manifest is None:
        return
    if not isinstance(seed_manifest, str) or not seed_manifest.strip():
        errors.append("seed_set_refs.manifest must be a non-empty path string when provided")
        return
    manifest_path = _resolve_path(seed_manifest, repo_root)
    if not manifest_path.is_file():
        errors.append(f"seed_set_refs.manifest is not a regular file: {seed_manifest}")
        return
    try:
        seed_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"seed_set_refs.manifest could not be loaded: {exc}")
        return
    if not isinstance(seed_payload, dict):
        errors.append("seed_set_refs.manifest must point to a mapping YAML file")
        return
    _validate_seed_ref("validation", refs.get("validation"), seed_payload, split_sets, errors)
    _validate_seed_ref("evaluation", refs.get("evaluation"), seed_payload, split_sets, errors)
    _validate_train_excludes(refs, seed_payload, split_sets, errors)


def _validate_seed_ref(
    split: str,
    ref_name: Any,
    seed_payload: dict[str, Any],
    split_sets: dict[str, set[int]],
    errors: list[str],
) -> None:
    if not isinstance(ref_name, str) or not ref_name.strip():
        return
    ref_value = seed_payload.get(ref_name)
    if not isinstance(ref_value, list):
        errors.append(
            f"seed_set_refs manifest key {ref_name!r} must be a list, got {type(ref_value).__name__}"
        )
        return
    expected: set[int] = set()
    for seed in ref_value:
        if not isinstance(seed, int):
            errors.append(
                f"seed_set_refs manifest key {ref_name!r} contains non-integer entry: {seed!r}"
            )
            return
        expected.add(seed)
    actual = split_sets.get(split, set())
    if expected != actual:
        errors.append(f"seeds_by_split.{split} must match seed set {ref_name}: {sorted(expected)}")


def _validate_train_excludes(
    refs: dict[str, Any],
    seed_payload: dict[str, Any],
    split_sets: dict[str, set[int]],
    errors: list[str],
) -> None:
    excluded_ref = refs.get("train_excludes")
    if not isinstance(excluded_ref, str):
        return
    excluded_value = seed_payload.get(excluded_ref)
    if not isinstance(excluded_value, list):
        errors.append(
            f"seed_set_refs manifest key {excluded_ref!r} must be a list, "
            f"got {type(excluded_value).__name__}"
        )
        return
    excluded: set[int] = set()
    for seed in excluded_value:
        if not isinstance(seed, int):
            errors.append(
                f"seed_set_refs manifest key {excluded_ref!r} contains non-integer entry: {seed!r}"
            )
            return
        excluded.add(seed)
    overlap = sorted(split_sets.get("train", set()) & excluded)
    if overlap:
        errors.append(f"train seeds overlap excluded seed set {excluded_ref}: {overlap}")


def _validate_episodes(packet: dict[str, Any], errors: list[str]) -> None:
    episodes = packet.get("episode_ids_by_split")
    if not isinstance(episodes, dict):
        errors.append("episode_ids_by_split must be a mapping")
        return
    all_ids: list[str] = []
    for split in _SPLITS:
        split_ids = episodes.get(split)
        if not isinstance(split_ids, list) or not split_ids:
            errors.append(f"episode_ids_by_split.{split} must be a non-empty list")
            continue
        for episode_id in split_ids:
            if not isinstance(episode_id, str) or not episode_id.strip():
                errors.append(f"episode_ids_by_split.{split} entries must be non-empty strings")
                continue
            all_ids.append(episode_id.strip())
    if len(set(all_ids)) != len(all_ids):
        errors.append("episode_ids_by_split must not contain duplicate episode ids")


def _validate_hard_slices(packet: dict[str, Any], errors: list[str]) -> None:
    assignments = packet.get("hard_slice_assignment")
    if not isinstance(assignments, list):
        errors.append("hard_slice_assignment must be a list")
        return
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            errors.append(f"hard_slice_assignment[{index}] must be a mapping")
            continue
        split = assignment.get("split")
        if split not in _SPLITS:
            errors.append(f"hard_slice_assignment[{index}].split must be one of {_SPLITS}")
        predeclared = bool(assignment.get("predeclared_for_evaluation", False))
        if split == "evaluation" and not predeclared:
            errors.append(
                f"hard_slice_assignment[{index}] assigns evaluation without predeclaration"
            )


def _validate_relabeling(packet: dict[str, Any], errors: list[str]) -> None:
    policy = packet.get("relabeling_policy")
    if policy is None:
        return
    if not isinstance(policy, dict):
        errors.append("relabeling_policy must be null or a mapping")
        return
    scope = policy.get("scope")
    if scope != "train":
        errors.append("relabeling_policy.scope must be 'train' when relabeling is enabled")
    _require_non_empty_string(policy, "source_oracle", errors)
    _require_non_empty_string(policy, "rule", errors)


def _validate_generating_commit(packet: dict[str, Any], errors: list[str]) -> None:
    commit = packet.get("generating_commit")
    if not isinstance(commit, str) or not _GIT_SHA_RE.match(commit.strip()):
        errors.append("generating_commit must be a 40-character git SHA")


def _validate_artifacts(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, str]:
    artifact_paths = packet.get("artifact_paths")
    checksums = packet.get("checksums")
    if not isinstance(artifact_paths, dict) or not artifact_paths:
        errors.append("artifact_paths must be a non-empty mapping")
        return {}
    if not isinstance(checksums, dict):
        errors.append("checksums must be a mapping")
        checksums = {}

    normalized_paths: dict[str, str] = {}
    local_paths: list[str] = []
    durable_count = 0
    for name, raw_path in sorted(artifact_paths.items()):
        path_text = _artifact_path_text(str(name), raw_path, errors)
        if path_text is None:
            continue
        normalized_paths[str(name)] = path_text
        if _is_durable_uri(path_text):
            durable_count += 1
            continue
        local_paths.append(path_text)
        _validate_local_artifact(str(name), path_text, checksums, repo_root, errors)

    for checksum_path in checksums:
        if checksum_path not in local_paths:
            errors.append(f"checksums contains path not listed in artifact_paths: {checksum_path}")
    if durable_count == 0:
        errors.append("artifact_paths must include at least one durable artifact URI")
    return normalized_paths


def _validate_training_artifact_gate(
    artifact_paths: dict[str, str],
    *,
    trace_uri_registry: dict[str, Any] | None,
    require_training_ready: bool,
    errors: list[str],
) -> bool:
    if trace_uri_registry is not None:
        training_ready = bool(trace_uri_registry.get("training_ready", False))
        if require_training_ready and not training_ready:
            errors.append("trace_uri_registry must be training_ready when --require-training-ready")
        return training_ready

    required_trace_artifacts = {
        "train_trace_jsonl_uri",
        "validation_trace_jsonl_uri",
        "evaluation_trace_jsonl_uri",
        "trace_source_manifest_uri",
    }
    missing = sorted(required_trace_artifacts - artifact_paths.keys())
    pending = sorted(
        name for name, path_text in artifact_paths.items() if _is_pending_durable_uri(path_text)
    )
    non_durable_required = sorted(
        name
        for name in required_trace_artifacts & artifact_paths.keys()
        if not _is_concrete_durable_uri(artifact_paths[name])
        and not _is_pending_durable_uri(artifact_paths[name])
    )

    training_ready = not missing and not pending and not non_durable_required
    if not require_training_ready:
        return training_ready

    if missing:
        errors.append(
            "training-ready oracle-imitation launch packets must include durable trace artifact "
            f"URIs for: {', '.join(missing)}"
        )
    if pending:
        errors.append(
            "training-ready oracle-imitation launch packets must not use pending durable aliases: "
            f"{', '.join(pending)}"
        )
    if non_durable_required:
        errors.append(
            "training-ready oracle-imitation launch packets must use concrete durable URIs for: "
            f"{', '.join(non_durable_required)}"
        )
    return training_ready


def _validate_trace_uri_registry_pointer(
    packet: dict[str, Any],
    repo_root: Path,
    *,
    require_training_ready: bool,
    errors: list[str],
) -> dict[str, Any] | None:
    registry_path = packet.get("trace_uri_registry")
    if registry_path is None:
        return None
    if not isinstance(registry_path, str) or not registry_path.strip():
        errors.append("trace_uri_registry must be non-empty path string when provided")
        return None

    try:
        report = validate_trace_uri_registry(
            _resolve_path(registry_path, repo_root),
            repo_root=repo_root,
            require_training_ready=require_training_ready,
        )
    except OracleTraceUriRegistryError as exc:
        errors.append(f"trace_uri_registry failed validation: {_first_error_line(exc)}")
        return {
            "path": registry_path,
            "status": "invalid",
            "training_ready": False,
        }

    return {
        "path": registry_path,
        "status": report["status"],
        "dataset_id": report["dataset_id"],
        "splits": report["splits"],
        "retrieval_status": report["retrieval_status"],
        "training_ready": report["training_ready"],
    }


def _first_error_line(exc: Exception) -> str:
    for line in str(exc).splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith("failed validation:"):
            continue
        return stripped.removeprefix("- ").strip()
    return str(exc)


def _validate_collection_roots(packet: dict[str, Any], errors: list[str]) -> dict[str, str]:
    """Validate the durable destinations a collection job must declare before it runs.

    Requires a ``collection_roots`` mapping that names where collection logs, raw trace
    output, and the dataset manifest are durably written. Each root must be a durable
    artifact URI (a ``:pending`` alias is allowed because collection has not run yet) and
    must never point at the gitignored worktree-local ``output`` directory, which is not a
    safe shared destination on a multi-agent host.

    Args:
        packet: Parsed launch packet.
        errors: Mutable error accumulator extended in place on any violation.

    Returns:
        Mapping of validated root name to its normalized destination string. Roots that
        failed validation are omitted.
    """
    roots = packet.get("collection_roots")
    if not isinstance(roots, dict):
        errors.append("collection_roots must be a mapping")
        return {}

    normalized: dict[str, str] = {}
    for key in _REQUIRED_COLLECTION_ROOTS:
        value = roots.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"collection_roots.{key} must be a non-empty string")
            continue
        text = value.strip()
        if _has_worktree_output_component(text):
            errors.append(
                f"collection_roots.{key} must not depend on worktree-local output: {text}"
            )
            continue
        if not _is_durable_uri(text):
            errors.append(
                f"collection_roots.{key} must be a durable artifact URI "
                f"(one of {_DURABLE_URI_PREFIXES}): {text}"
            )
            continue
        normalized[key] = text
    return normalized


def _artifact_path_text(name: str, raw_path: Any, errors: list[str]) -> str | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"artifact_paths.{name} must be a non-empty string")
        return None
    path_text = raw_path.strip()
    if _has_worktree_output_component(path_text):
        errors.append(
            f"artifact_paths.{name} must not depend on worktree-local output: {path_text}"
        )
    return path_text


def _has_worktree_output_component(path_text: str) -> bool:
    """Return True when a path embeds the gitignored worktree-local ``output`` directory.

    The match is on a full path component so durable names that merely contain the
    substring ``output`` (for example ``docs/my_output_config``) are not rejected.
    """
    return "output" in Path(path_text).parts


def _is_durable_uri(path_text: str) -> bool:
    return path_text.startswith(_DURABLE_URI_PREFIXES)


def _is_pending_durable_uri(path_text: str) -> bool:
    return _is_durable_uri(path_text) and path_text.rstrip().endswith(":pending")


def _is_concrete_durable_uri(path_text: str) -> bool:
    return _is_durable_uri(path_text) and not _is_pending_durable_uri(path_text)


def _validate_local_artifact(
    name: str,
    path_text: str,
    checksums: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    local_path = _resolve_path(path_text, repo_root)
    if not local_path.is_file():
        errors.append(f"artifact_paths.{name} local artifact is missing: {path_text}")
        return
    expected = checksums.get(path_text)
    if not isinstance(expected, str) or not expected.strip():
        errors.append(f"checksums missing SHA-256 entry for {path_text}")
        return
    actual = _sha256_hex_digest(local_path)
    if actual != expected.strip().lower():
        errors.append(f"checksum mismatch for {path_text}: expected {expected}, got {actual}")


def _sha256_hex_digest(path: Path, chunk_size: int = 65536) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


__all__ = ["LaunchPacketError", "load_launch_packet", "validate_launch_packet"]
