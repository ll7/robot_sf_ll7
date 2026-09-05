"""Preparation-only validation for the Phase-C legacy-checkpoint cutover.

The validator freezes the current in-tree bytes, registry identities, benchmark
load paths, and a future paired-parity protocol for Issue #6794. It does not
download checkpoints, run a campaign, remove files, or admit benchmark evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
import yaml

from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    enrich_algorithm_metadata,
)
from robot_sf.benchmark.fallback_policy import (
    runtime_fallback_or_degraded_marker,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.termination_reason import outcome_contradictions
from robot_sf.models.registry import load_registry
from robot_sf.training.scenario_loader import load_scenarios

PREPARATION_SCHEMA_VERSION = "legacy-checkpoint-cutover-preparation.v1"
DEFAULT_CONFIG_PATH = Path("configs/benchmarks/issue_6794_phase_c_parity_preparation_v1.yaml")
_HEX_DIGEST_LENGTH = 64
PARITY_ROW_SCHEMA_VERSION = "legacy-checkpoint-cutover-parity-row.v1"
_PARITY_STATUS_FIELDS = (
    "status",
    "row_status",
    "execution_mode",
    "readiness_status",
    "availability_status",
    "benchmark_success",
    "benchmark_success_basis",
    "termination_reason",
)
_PARITY_METRIC_FIELDS = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)
_PARITY_PROVENANCE_FIELDS = (
    "config_hash",
    "git_hash",
    "parity_provenance.schema_version",
    "parity_provenance.scenario_matrix_sha256",
    "parity_provenance.seed_set_sha256",
    "parity_provenance.horizon",
    "parity_provenance.dt",
    "parity_provenance.workers",
    "parity_provenance.kinematics",
    "parity_provenance.arm_key",
    "parity_provenance.canonical_algorithm",
    "parity_provenance.config_identity",
    "parity_provenance.config_identity_sha256",
    "parity_provenance.runtime_overrides",
    "parity_provenance.checkpoint_name",
    "parity_provenance.checkpoint_kind",
    "parity_provenance.model_id",
    "parity_provenance.release_tag",
    "parity_provenance.release_version",
    "parity_provenance.release_asset_name",
    "parity_provenance.release_sha256",
    "parity_provenance.source_sha256",
    "parity_provenance.release_bundle_files",
    "parity_provenance.resolution_mode",
    "parity_provenance.resolution_receipt",
    "parity_provenance.resolution_receipt.status",
    "parity_provenance.resolution_receipt.archive_sha256",
    "parity_provenance.resolution_receipt.source_sha256",
    "parity_provenance.resolution_receipt.cache_path",
    "parity_provenance.resolution_receipt.resolved_path",
    "parity_provenance.resolution_receipt.loader_probe_status",
    "parity_provenance.resolution_receipt_sha256",
)
_BEFORE_RESOLUTION_MODE = "in_tree_checkpoint"
_AFTER_RESOLUTION_MODE = "registry_release_hydrated_checkpoint"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct a YAML mapping while rejecting duplicate keys.

    Returns:
        The constructed mapping.
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _construct_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct a JSON object while rejecting duplicate member names.

    Returns:
        The constructed object mapping.
    """
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"duplicate JSON object key {key!r}")
        mapping[key] = value
    return mapping


def _reject_non_finite_json_constant(value: str) -> Any:
    """Reject JSON extensions that would introduce non-finite numeric values."""
    raise ValueError(f"non-finite JSON constant {value!r} is not permitted")


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL file as mappings, failing closed on malformed rows.

    Returns:
        Parsed JSON object rows.
    """
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read parity input {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(
                raw_line,
                object_pairs_hook=_construct_unique_json_object,
                parse_constant=_reject_non_finite_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(dict(value))
    if not rows:
        raise ValueError(f"{path}: parity input is empty")
    return rows


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Return a mapping or raise a contract-shaped error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string(value: Any, *, name: str) -> str:
    """Return a non-empty string or raise a contract-shaped error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _repo_declared_path(repo_root: Path, value: Any, *, name: str) -> Path:
    """Resolve a declared repository path without following it outside the checkout.

    Returns:
        A path whose resolved target remains inside ``repo_root``.
    """
    relative = Path(_string(value, name=name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a repository-relative path without '..'")
    root = repo_root.resolve()
    candidate = root / relative
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{name} cannot be resolved safely within the repository") from exc
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{name} must resolve within the repository root")
    return candidate


def _digest(value: Any, *, name: str) -> str:
    """Return a lowercase hexadecimal digest or raise a contract-shaped error."""
    digest = _string(value, name=name).lower()
    if len(digest) != _HEX_DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a {_HEX_DIGEST_LENGTH}-character SHA-256 digest")
    return digest


def _strict_integer(value: Any, *, name: str) -> int:
    """Return an integer while rejecting boolean coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_number(value: Any, *, name: str) -> float:
    """Return a finite number while rejecting boolean coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _dotted_value(row: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted field path from a JSON mapping.

    Returns:
        The nested value, or ``None`` when a component is absent.
    """
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _identity_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    """Build one row identity and reject missing identity fields.

    Returns:
        The ordered identity tuple.
    """
    values = tuple(_dotted_value(row, field) for field in fields)
    if any(value is None for value in values):
        raise ValueError(f"row is missing identity fields {list(fields)!r}: {row!r}")
    for field, value in zip(fields, values, strict=True):
        if field in {"planner_key", "scenario_id"} and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"row identity field {field!r} must be a non-empty string")
        if field == "seed":
            _strict_integer(value, name="row identity field 'seed'")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"row identity field {field!r} must be hashable") from exc
    return values


def _verify_source_files(
    repo_root: Path,
    checkpoint_name: str,
    paths: list[Any],
    source_digests: Mapping[str, Any],
) -> dict[str, str]:
    """Verify the frozen digest for every current in-tree checkpoint component.

    Returns:
        Observed source digests keyed by repository-relative path.
    """
    observed: dict[str, str] = {}
    declared_paths: set[str] = set()
    for raw_path in paths:
        rel_path = _string(raw_path, name=f"checkpoint {checkpoint_name}.source_paths[]")
        if rel_path in declared_paths:
            raise ValueError(
                f"checkpoint {checkpoint_name} declares duplicate source path: {rel_path}"
            )
        declared_paths.add(rel_path)
        path = _repo_declared_path(
            repo_root,
            rel_path,
            name=f"checkpoint {checkpoint_name}.source_paths[]",
        )
        if not path.is_file():
            raise ValueError(f"checkpoint {checkpoint_name} source is not a file: {rel_path}")
        digest = _sha256(path)
        expected = _digest(source_digests.get(rel_path), name=f"source_sha256[{rel_path}]")
        if digest != expected:
            raise ValueError(
                f"checkpoint {checkpoint_name} source digest mismatch for {rel_path}: "
                f"observed {digest} != frozen {expected}"
            )
        observed[rel_path] = digest
    if set(source_digests) != declared_paths:
        raise ValueError(
            f"checkpoint {checkpoint_name} source_sha256 keys must exactly match source_paths"
        )
    return observed


def _verify_release_identity(
    checkpoint_name: str,
    checkpoint: Mapping[str, Any],
    model_id: str,
    release: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Verify the immutable registry release identity pinned by a checkpoint.

    Returns:
        Release tag, immutable version, and release archive digest.
    """
    release_version = _string(release.get("version"), name=f"registry[{model_id}].version")
    release_tag = _string(release.get("tag"), name=f"registry[{model_id}].tag")
    if release_version.casefold() in {"latest", "current", "best", "best-success"}:
        raise ValueError(f"registry[{model_id}] release version is moving: {release_version!r}")
    frozen_registry = _digest(
        checkpoint.get("registry_release_sha256"),
        name=f"checkpoint {checkpoint_name}.registry_release_sha256",
    )
    release_asset_name = _string(
        release.get("asset_name"),
        name=f"registry[{model_id}].github_release.asset_name",
    )
    configured_asset_name = _string(
        checkpoint.get("registry_release_asset_name"),
        name=f"checkpoint {checkpoint_name}.registry_release_asset_name",
    )
    if configured_asset_name != release_asset_name:
        raise ValueError(f"checkpoint {checkpoint_name} release asset disagrees with registry")
    registry_sha = _digest(
        release.get("sha256"), name=f"registry[{model_id}].github_release.sha256"
    )
    if frozen_registry != registry_sha:
        raise ValueError(
            f"checkpoint {checkpoint_name} registry archive digest drift: "
            f"frozen {frozen_registry} != registry {registry_sha}"
        )
    configured_tag = _string(
        checkpoint.get("registry_release_tag", release_tag),
        name=f"checkpoint {checkpoint_name}.registry_release_tag",
    )
    if release_tag != configured_tag:
        raise ValueError(f"checkpoint {checkpoint_name} release tag disagrees with registry")
    return release_tag, release_version, registry_sha


def _verify_release_components(  # noqa: C901
    checkpoint_name: str,
    checkpoint: Mapping[str, Any],
    release: Mapping[str, Any],
    observed: Mapping[str, str],
) -> dict[str, Any]:
    """Verify source bytes against single-file or bundle registry declarations.

    The archive digest alone is not enough for a TensorFlow checkpoint bundle:
    the release manifest must name exactly the same members as the frozen
    source snapshot. This validates the release identity and member contract;
    it does not extract or execute the archive.

    Returns:
        Release component metadata for the preparation report.
    """
    if len(observed) == 1:
        if next(iter(observed.values())) != _digest(
            release.get("sha256"), name="registry release sha256"
        ):
            raise ValueError(
                f"checkpoint {checkpoint_name} source digest does not match registry release"
            )
        return {
            "bundle_files": [],
            "per_file_sha256": dict(observed),
            "status": "single_file_digest_verified",
        }
    source_names = [Path(path).name for path in observed]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"checkpoint {checkpoint_name} bundle source basenames must be unique")
    bundle_files = release.get("bundle_files")
    if not isinstance(bundle_files, list) or not bundle_files:
        raise ValueError(
            f"registry release for checkpoint {checkpoint_name} must declare bundle_files"
        )
    normalized_bundle_files: list[str] = []
    for raw_path in bundle_files:
        path_value = _string(
            raw_path,
            name=f"registry release bundle_files for checkpoint {checkpoint_name}",
        )
        path = PurePosixPath(path_value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"registry release bundle member must be a safe relative path: {path_value}"
            )
        normalized = path.as_posix()
        if normalized in normalized_bundle_files:
            raise ValueError(
                f"registry release checkpoint {checkpoint_name} declares duplicate bundle member"
            )
        normalized_bundle_files.append(normalized)
    observed_paths = {PurePosixPath(path).as_posix() for path in observed}
    if set(normalized_bundle_files) != observed_paths:
        raise ValueError(
            f"checkpoint {checkpoint_name} release bundle member set does not match "
            "the declared source paths"
        )
    per_file = _mapping(release.get("per_file_sha256"), name="registry release per_file_sha256")
    frozen_per_file = _mapping(
        checkpoint.get("registry_per_file_sha256"),
        name=f"checkpoint {checkpoint_name}.registry_per_file_sha256",
    )
    for rel_path, digest in observed.items():
        name = Path(rel_path).name
        registry_digest = _digest(per_file.get(name), name=f"registry per-file digest {name}")
        frozen_digest = _digest(frozen_per_file.get(name), name=f"frozen per-file digest {name}")
        if digest != registry_digest or digest != frozen_digest:
            raise ValueError(f"checkpoint {checkpoint_name} component digest drift: {name}")
    expected_names = {Path(path).name for path in observed}
    if set(per_file) != expected_names or set(frozen_per_file) != expected_names:
        raise ValueError(
            f"checkpoint {checkpoint_name} component set does not match the declared source paths"
        )
    return {
        "bundle_files": normalized_bundle_files,
        "per_file_sha256": dict(observed),
        "status": "bundle_member_digests_verified",
    }


def _checkpoint_report(
    repo_root: Path,
    registry: Mapping[str, Any],
    checkpoint_name: str,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one frozen checkpoint against the current worktree and registry.

    Returns:
        A machine-readable verified checkpoint report.
    """
    model_id = _string(checkpoint.get("model_id"), name=f"checkpoint {checkpoint_name}.model_id")
    entries = registry.get(model_id)
    if not isinstance(entries, Mapping):
        raise ValueError(f"registry entry missing for checkpoint {checkpoint_name}: {model_id}")
    release = _mapping(entries.get("github_release"), name=f"registry[{model_id}].github_release")
    registry_local_path = _string(
        entries.get("local_path"), name=f"registry[{model_id}].local_path"
    )
    configured_local_path = _string(
        checkpoint.get("registry_local_path"),
        name=f"checkpoint {checkpoint_name}.registry_local_path",
    )
    if configured_local_path != registry_local_path:
        raise ValueError(f"checkpoint {checkpoint_name} local path disagrees with registry")
    paths = checkpoint.get("source_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"checkpoint {checkpoint_name}.source_paths must be a non-empty list")
    source_digests = _mapping(
        checkpoint.get("source_sha256"), name=f"checkpoint {checkpoint_name}.source_sha256"
    )
    observed = _verify_source_files(repo_root, checkpoint_name, paths, source_digests)
    release_tag, release_version, registry_sha = _verify_release_identity(
        checkpoint_name, checkpoint, model_id, release
    )
    component_report = _verify_release_components(checkpoint_name, checkpoint, release, observed)
    kind = _string(checkpoint.get("kind"), name=f"checkpoint {checkpoint_name}.kind")
    if kind not in {"single_file", "multi_file_bundle"}:
        raise ValueError(f"checkpoint {checkpoint_name}.kind is unsupported: {kind}")
    if kind == "single_file" and len(observed) != 1:
        raise ValueError(f"checkpoint {checkpoint_name} single_file kind needs one source path")
    if kind == "multi_file_bundle" and len(observed) < 2:
        raise ValueError(
            f"checkpoint {checkpoint_name} multi_file_bundle kind needs multiple source paths"
        )
    return {
        "model_id": model_id,
        "kind": kind,
        "release_tag": release_tag,
        "release_version": release_version,
        "release_asset_name": _string(
            release.get("asset_name"),
            name=f"registry[{model_id}].github_release.asset_name",
        ),
        "source_sha256": observed,
        "registry_release_sha256": registry_sha,
        "release_bundle_files": component_report["bundle_files"],
        "release_components_status": component_report["status"],
        "registry_local_path": registry_local_path,
        "runtime_resolution_status": "deferred_until_hydration",
        "status": "identity_verified_preparation_only",
    }


def _validate_checkpoint_load_contract(
    repo_root: Path,
    checkpoint_name: str,
    report: Mapping[str, Any],
) -> dict[str, str]:
    """Verify the in-tree source shape used by the declared loader contract.

    Returns:
        Static load-contract status. Runtime release hydration remains deferred.
    """
    source_paths = {PurePosixPath(path) for path in report["source_sha256"]}
    local_path_value = str(report["registry_local_path"] or "")
    if not local_path_value:
        raise ValueError(f"checkpoint {checkpoint_name} registry_local_path is required")
    local_path = _repo_declared_path(
        repo_root,
        local_path_value,
        name=f"registry local path for checkpoint {checkpoint_name}",
    )
    kind = str(report["kind"])
    if kind == "multi_file_bundle":
        if PurePosixPath(local_path_value) not in source_paths or local_path.suffix != ".meta":
            raise ValueError(
                f"checkpoint {checkpoint_name} registry_local_path must identify the "
                "declared TensorFlow .meta prefix"
            )
        prefix = PurePosixPath(local_path_value).with_suffix("")
        index_path = prefix.with_suffix(".index")
        data_paths = {
            path
            for path in source_paths
            if path.parent == prefix.parent and path.name.startswith(f"{prefix.name}.data-")
        }
        if not local_path.is_file() or index_path not in source_paths:
            raise ValueError(
                f"checkpoint {checkpoint_name} in-tree TensorFlow prefix is incomplete"
            )
        if not data_paths:
            raise ValueError(
                f"checkpoint {checkpoint_name} in-tree TensorFlow data shard is missing"
            )
        return {
            "source_shape": "tensorflow_checkpoint_prefix_verified",
            "registry_local_path": "in_tree_prefix_present",
            "runtime_loader_probe": "deferred_until_hydration",
        }
    source_path = next(iter(source_paths))
    if not _repo_declared_path(
        repo_root,
        str(source_path),
        name=f"checkpoint {checkpoint_name}.source_paths[]",
    ).is_file():
        raise ValueError(f"checkpoint {checkpoint_name} source file is missing")
    return {
        "source_shape": "single_file_verified",
        "registry_local_path": "cache_resolution_deferred",
        "runtime_loader_probe": "deferred_until_hydration",
    }


def _validate_load_paths(repo_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify every frozen load-path selector remains present in the worktree.

    Returns:
        Machine-readable selector status rows.
    """
    raw_inventory = config.get("load_path_inventory")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise ValueError("load_path_inventory must be a non-empty list")
    seen: set[str] = set()
    seen_paths: set[str] = set()
    checkpoint_names = set(
        _mapping(config.get("checkpoint_snapshots"), name="checkpoint_snapshots")
    )
    report: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_inventory):
        item = _mapping(raw_item, name=f"load_path_inventory[{index}]")
        item_id = _string(item.get("id"), name=f"load_path_inventory[{index}].id")
        checkpoint_name = _string(
            item.get("checkpoint"), name=f"load_path_inventory[{index}].checkpoint"
        )
        path_value = _string(item.get("path"), name=f"load_path_inventory[{index}].path")
        selector = _string(item.get("selector"), name=f"load_path_inventory[{index}].selector")
        if item_id in seen:
            raise ValueError(f"duplicate load-path inventory id: {item_id}")
        if checkpoint_name not in checkpoint_names:
            raise ValueError(
                f"load-path inventory references unknown checkpoint: {checkpoint_name}"
            )
        if path_value in seen_paths:
            raise ValueError(f"duplicate load-path inventory path: {path_value}")
        seen.add(item_id)
        seen_paths.add(path_value)
        path = _repo_declared_path(repo_root, path_value, name=f"load_path_inventory[{index}].path")
        if not path.is_file():
            raise ValueError(f"load-path inventory file is missing: {path_value}")
        content = path.read_text(encoding="utf-8")
        if selector not in content:
            raise ValueError(f"load-path selector is absent: {path_value}: {selector}")
        report.append(
            {
                "id": item_id,
                "path": path_value,
                "selector": selector,
                "status": "selector_present",
                "runtime_probe": "deferred_until_hydration",
            }
        )
    return report


def _load_preparation_config(config_path: Path) -> Mapping[str, Any]:
    """Load and validate the preparation-only top-level fields.

    Returns:
        Parsed preparation configuration.
    """
    try:
        config = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,  # noqa: S506
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read preparation config {config_path}: {exc}") from exc
    config = _mapping(config, name="preparation config")
    if config.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ValueError("preparation config schema_version is unsupported")
    if config.get("issue") != 6794:
        raise ValueError("preparation config must target Issue #6794")
    if config.get("status") != "preparation_only" or config.get("execute_campaign") is not False:
        raise ValueError("preparation config must remain preparation-only and non-executable")
    return config


def _validate_protocol(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen scenario, seed, comparison, and output contract.

    Returns:
        The normalized protocol fields needed in the preparation report.
    """
    episode_schema_path = _string(
        protocol.get("episode_schema"),
        name="parity_protocol.episode_schema",
    )
    episode_schema_file = _repo_declared_path(
        repo_root,
        episode_schema_path,
        name="parity_protocol.episode_schema",
    )
    episode_schema_digest = _digest(
        protocol.get("episode_schema_sha256"),
        name="parity_protocol.episode_schema_sha256",
    )
    if _sha256(episode_schema_file) != episode_schema_digest:
        raise ValueError("parity episode schema digest does not match the current worktree")
    try:
        episode_schema = load_schema(episode_schema_file)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"parity episode schema cannot be loaded: {episode_schema_path}") from exc
    if not isinstance(episode_schema, Mapping):
        raise ValueError("parity episode schema must be a mapping")

    scenario_path = _string(protocol.get("scenario_matrix"), name="parity_protocol.scenario_matrix")
    scenario_file = _repo_declared_path(
        repo_root, scenario_path, name="parity_protocol.scenario_matrix"
    )
    scenario_digest = _digest(
        protocol.get("scenario_matrix_sha256"), name="parity_protocol.scenario_matrix_sha256"
    )
    if _sha256(scenario_file) != scenario_digest:
        raise ValueError("parity scenario matrix digest does not match the current worktree")
    _validate_digest_map(
        repo_root,
        protocol.get("included_scenario_manifests"),
        name="parity_protocol.included_scenario_manifests",
    )
    seed_path = _string(protocol.get("seed_set_path"), name="parity_protocol.seed_set_path")
    seed_file = _repo_declared_path(repo_root, seed_path, name="parity_protocol.seed_set_path")
    seed_digest = _digest(protocol.get("seed_set_sha256"), name="parity_protocol.seed_set_sha256")
    if _sha256(seed_file) != seed_digest:
        raise ValueError("parity seed-set digest does not match the current worktree")
    seeds = protocol.get("seeds")
    if seeds != [111, 112, 113]:
        raise ValueError("parity protocol must use the frozen eval seeds [111, 112, 113]")
    scenario_ids = _scenario_ids(repo_root, scenario_file)
    if (
        protocol.get("before_mode") != _BEFORE_RESOLUTION_MODE
        or protocol.get("after_mode") != "registry_release_identity_preparation"
    ):
        raise ValueError("parity protocol modes are not the Phase-C before/after pair")
    if protocol.get("after_resolution_mode") != _AFTER_RESOLUTION_MODE:
        raise ValueError(
            "parity protocol must require an isolated hydrated release for the future after arm"
        )
    if protocol.get("hydration_required") is not True:
        raise ValueError("parity protocol must require release hydration before comparison")
    _validate_protocol_shape(protocol)
    _validate_protocol_arms(protocol)
    config_identity_digests = _validate_config_identities(repo_root, protocol)
    comparison = _mapping(protocol.get("comparison"), name="parity_protocol.comparison")
    _validate_comparison_contract(comparison)
    output_paths = _validate_output_paths(protocol)
    return {
        "before_mode": protocol["before_mode"],
        "after_mode": protocol["after_mode"],
        "after_resolution_mode": protocol["after_resolution_mode"],
        "hydration_required": True,
        "episode_schema": episode_schema_path,
        "episode_schema_sha256": episode_schema_digest,
        "row_schema_version": protocol["row_schema_version"],
        "scenario_matrix": scenario_path,
        "scenario_matrix_sha256": scenario_digest,
        "scenario_ids": scenario_ids,
        "seeds": list(seeds),
        "seed_set_path": seed_path,
        "seed_set_sha256": seed_digest,
        "horizon": protocol["horizon"],
        "dt": protocol["dt"],
        "workers": protocol["workers"],
        "kinematics": protocol["kinematics"],
        "planner_arms": protocol.get("planner_arms"),
        "row_identity": protocol.get("row_identity"),
        "required_status_fields": protocol.get("required_status_fields"),
        "required_metrics": protocol.get("required_metrics"),
        "required_provenance_fields": protocol.get("required_provenance_fields"),
        "config_identity_sha256": config_identity_digests,
        "comparison": dict(comparison),
        "output_paths": output_paths,
    }


def _scenario_ids(repo_root: Path, scenario_file: Path) -> list[str]:
    """Resolve the exact scenario identity set from the canonical loader.

    Returns:
        Ordered, unique scenario identifiers.
    """
    try:
        scenarios = load_scenarios(scenario_file, base_dir=repo_root)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"parity scenario matrix cannot be loaded: {scenario_file}") from exc
    identifiers: list[str] = []
    seen: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError(f"parity scenario entry {index} is not a mapping")
        raw_identifier = scenario.get("id") or scenario.get("name") or scenario.get("scenario_id")
        identifier = _string(raw_identifier, name=f"parity scenario entry {index}.id")
        if identifier in seen:
            raise ValueError(f"parity scenario matrix contains duplicate identity: {identifier}")
        seen.add(identifier)
        identifiers.append(identifier)
    if not identifiers:
        raise ValueError("parity scenario matrix must resolve to at least one scenario")
    return identifiers


def _validate_config_identities(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, str]:
    """Verify immutable hashes for file-backed planner configuration identities.

    Returns:
        Observed hashes keyed by file-backed configuration identity.
    """
    raw_digests = _mapping(
        protocol.get("config_identity_sha256"),
        name="parity_protocol.config_identity_sha256",
    )
    observed: dict[str, str] = {}
    for raw_path, raw_digest in raw_digests.items():
        path_value = _string(raw_path, name="parity_protocol.config_identity_sha256 path")
        path = _repo_declared_path(
            repo_root,
            path_value,
            name=f"parity_protocol.config_identity_sha256[{path_value}]",
        )
        digest = _digest(
            raw_digest,
            name=f"parity_protocol.config_identity_sha256[{path_value}]",
        )
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(
                f"planner config digest does not match the current worktree: {path_value}"
            )
        observed[path_value] = digest
    expected_paths: set[str] = set()
    for index, raw_arm in enumerate(protocol.get("planner_arms", [])):
        arm = _mapping(raw_arm, name=f"parity_protocol.planner_arms[{index}]")
        identity = _string(
            arm.get("config_identity"),
            name=f"parity_protocol.planner_arms[{index}].config_identity",
        )
        if identity.startswith("builtin:"):
            continue
        expected_paths.add(identity)
    if set(observed) != expected_paths:
        raise ValueError(
            "parity protocol config_identity_sha256 keys must match file-backed arm identities"
        )
    return observed


def _validate_digest_map(repo_root: Path, value: Any, *, name: str) -> None:
    """Verify a mapping of repository-relative files to frozen SHA-256 digests."""
    digest_map = _mapping(value, name=name)
    if not digest_map:
        raise ValueError(f"{name} must not be empty")
    for raw_path, raw_digest in digest_map.items():
        path_value = _string(raw_path, name=f"{name} path")
        digest = _digest(raw_digest, name=f"{name}[{path_value}]")
        path = _repo_declared_path(repo_root, path_value, name=f"{name} path")
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"{name} digest does not match the current worktree: {path_value}")


def _validate_protocol_shape(protocol: Mapping[str, Any]) -> None:
    """Validate fixed row, metric, and runtime settings for the parity packet."""
    if protocol.get("row_schema_version") != PARITY_ROW_SCHEMA_VERSION:
        raise ValueError("parity row schema version is unsupported")
    if protocol.get("row_identity") != ["planner_key", "scenario_id", "seed"]:
        raise ValueError("parity row identity must be planner_key/scenario_id/seed")
    if protocol.get("required_status_fields") != list(_PARITY_STATUS_FIELDS):
        raise ValueError("parity status fields are incomplete or reordered")
    if protocol.get("required_metrics") != list(_PARITY_METRIC_FIELDS):
        raise ValueError("parity metrics are incomplete or reordered")
    if protocol.get("required_provenance_fields") != list(_PARITY_PROVENANCE_FIELDS):
        raise ValueError("parity provenance fields are incomplete or reordered")
    horizon = _strict_integer(protocol.get("horizon"), name="parity_protocol.horizon")
    dt = _strict_number(protocol.get("dt"), name="parity_protocol.dt")
    if horizon != 100 or dt != 0.1:
        raise ValueError("parity protocol must freeze horizon=100 and dt=0.1")
    workers = _strict_integer(protocol.get("workers"), name="parity_protocol.workers")
    if workers != 1 or protocol.get("kinematics") != "differential_drive":
        raise ValueError("parity protocol must freeze single-worker differential-drive execution")


def _validate_protocol_arm(
    arm: Mapping[str, Any],
    index: int,
    expected: Mapping[str, Any],
    expected_override_fields: set[str],
) -> Mapping[str, Any]:
    """Validate one frozen planner arm and return its runtime overrides.

    Returns:
        The validated runtime-overrides mapping.
    """
    expected_fields = set(expected) | {"runtime_overrides"}
    if set(arm) != expected_fields:
        raise ValueError(f"parity protocol arm {index} has unexpected fields")
    for field, expected_value in expected.items():
        if arm.get(field) != expected_value:
            raise ValueError(f"parity protocol arm {index} has unexpected {field}")
    overrides = arm.get("runtime_overrides")
    if not isinstance(overrides, Mapping):
        raise ValueError("parity protocol arm runtime_overrides must be mappings")
    if set(overrides) != expected_override_fields:
        raise ValueError(f"parity protocol arm {index} runtime_overrides has unexpected fields")
    return overrides


def _validate_protocol_arms(protocol: Mapping[str, Any]) -> None:
    """Validate the PPO and SACADRL fail-fast arm declarations."""
    planner_arms = protocol.get("planner_arms")
    if (
        not isinstance(planner_arms, list)
        or len(planner_arms) != 2
        or not all(isinstance(arm, Mapping) for arm in planner_arms)
    ):
        raise ValueError("parity protocol must compare two mapping arms, PPO then SACADRL")
    expected_arms = [
        {
            "key": "ppo",
            "algo": "ppo",
            "config": "configs/baselines/ppo.yaml",
            "config_identity": "configs/baselines/ppo.yaml",
            "checkpoint": "default_ppo",
            "execution_mode": "native",
            "adapter_name": None,
            "fallback_policy": "fail_fast",
        },
        {
            "key": "sacadrl",
            "algo": "sacadrl",
            "config": None,
            "config_identity": "builtin:sacadrl",
            "checkpoint": "ga3c_cadrl",
            "execution_mode": "adapter",
            "adapter_name": "SACADRLPlannerAdapter",
            "fallback_policy": "fail_fast",
        },
    ]
    ppo_overrides = _validate_protocol_arm(
        planner_arms[0],
        0,
        expected_arms[0],
        {"fallback_to_goal"},
    )
    sacadrl_overrides = _validate_protocol_arm(
        planner_arms[1],
        1,
        expected_arms[1],
        {"socnav_missing_prereq_policy"},
    )
    if ppo_overrides.get("fallback_to_goal") is not False:
        raise ValueError("PPO parity arm must disable goal fallback")
    if sacadrl_overrides.get("socnav_missing_prereq_policy") != "fail-fast":
        raise ValueError("SACADRL parity arm must fail fast on missing prerequisites")
    for index, (arm, expected) in enumerate(zip(planner_arms, expected_arms, strict=True)):
        canonical = canonical_algorithm_name(str(arm["algo"]))
        if canonical != str(expected["algo"]):
            raise ValueError(f"parity protocol arm {index} has an unknown canonical algorithm")
        metadata = enrich_algorithm_metadata(
            algo=canonical,
            execution_mode=str(expected["execution_mode"]),
            adapter_name=expected["adapter_name"],
        )
        kinematics = metadata.get("planner_kinematics")
        if not isinstance(kinematics, Mapping):
            raise ValueError(f"parity protocol arm {index} has no canonical kinematics metadata")
        if (
            expected["execution_mode"] == "native"
            and kinematics.get("supports_native_commands") is not True
        ):
            raise ValueError(f"parity protocol arm {index} cannot run in native mode")
        if (
            expected["execution_mode"] == "adapter"
            and kinematics.get("supports_adapter_commands") is not True
        ):
            raise ValueError(f"parity protocol arm {index} cannot run in adapter mode")
        if expected["adapter_name"] is not None and (
            kinematics.get("adapter_name") != expected["adapter_name"]
        ):
            raise ValueError(f"parity protocol arm {index} has the wrong adapter identity")


def _validate_comparison_contract(comparison: Mapping[str, Any]) -> None:
    """Validate the fail-closed parity comparison settings."""
    expected_boolean_flags = {
        "require_identical_row_keys",
        "require_identical_status_fields",
        "require_expected_identity_set",
        "require_canonical_episode_schema",
        "require_canonical_availability",
        "require_expected_execution_modes",
        "require_provenance_binding",
    }
    if set(comparison) != expected_boolean_flags | {
        "missing_metric_policy",
        "float_abs_tolerance",
        "float_rel_tolerance",
    }:
        raise ValueError("parity comparison contract contains unexpected or missing fields")
    for field in expected_boolean_flags:
        if comparison.get(field) is not True:
            raise ValueError(f"parity comparison must enable {field}")
    if comparison.get("missing_metric_policy") != "fail":
        raise ValueError("parity comparison must fail on missing metrics")
    abs_tolerance = comparison.get("float_abs_tolerance")
    rel_tolerance = comparison.get("float_rel_tolerance")
    if _strict_number(abs_tolerance, name="parity float_abs_tolerance") != 1e-12:
        raise ValueError("parity float_abs_tolerance must be 1e-12")
    if _strict_number(rel_tolerance, name="parity float_rel_tolerance") != 0.0:
        raise ValueError("parity float_rel_tolerance must be 0.0")


def _validate_output_paths(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that future outputs stay in the issue-local benchmark root.

    Returns:
        Normalized output path fields.
    """
    output_paths = _mapping(protocol.get("output_paths"), name="parity_protocol.output_paths")
    fields = (
        "root",
        "before_episodes",
        "after_episodes",
        "comparison_report",
        "preparation_report",
    )
    normalized: dict[str, str] = {}
    for field in fields:
        path_value = _string(output_paths.get(field), name=f"parity_protocol.output_paths.{field}")
        path = PurePosixPath(path_value)
        root = PurePosixPath("output/benchmarks/issue_6794_phase_c_parity")
        if path.is_absolute() or ".." in path.parts or path.parts[: len(root.parts)] != root.parts:
            raise ValueError(
                f"parity output path must remain under the issue-6794 output root: {path_value}"
            )
        normalized[field] = path_value
    return normalized


def validate_preparation_contract(
    repo_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    """Validate the preparation packet without executing or staging anything.

    Returns:
        A machine-readable preparation-only validation report.
    """
    if config_path.is_absolute():
        root = repo_root.resolve()
        try:
            resolved_config = config_path.resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError("config path cannot be resolved safely") from exc
        if resolved_config != root and root not in resolved_config.parents:
            raise ValueError("config path must resolve within the repository root")
    else:
        config_path = _repo_declared_path(repo_root, str(config_path), name="config")
    config = _load_preparation_config(config_path)

    registry_block = _mapping(config.get("registry"), name="registry")
    registry_rel = _string(registry_block.get("path"), name="registry.path")
    registry_path = _repo_declared_path(repo_root, registry_rel, name="registry.path")
    registry = load_registry(registry_path)
    common_tag = _string(registry_block.get("release_tag"), name="registry.release_tag")
    common_version = _string(
        registry_block.get("immutable_version"), name="registry.immutable_version"
    )
    checkpoint_block = _mapping(config.get("checkpoint_snapshots"), name="checkpoint_snapshots")
    checkpoint_report: dict[str, Any] = {}
    for name, raw_checkpoint in checkpoint_block.items():
        checkpoint = _mapping(raw_checkpoint, name=f"checkpoint_snapshots.{name}")
        checkpoint = dict(checkpoint)
        checkpoint["registry_release_tag"] = common_tag
        report = _checkpoint_report(repo_root, registry, str(name), checkpoint)
        if report["release_tag"] != common_tag or report["release_version"] != common_version:
            raise ValueError(f"checkpoint {name} does not use the pinned common release identity")
        report["load_contract"] = _validate_checkpoint_load_contract(
            repo_root,
            str(name),
            report,
        )
        checkpoint_report[str(name)] = report

    load_path_report = _validate_load_paths(repo_root, config)
    protocol_report = _validate_protocol(
        repo_root,
        _mapping(config.get("parity_protocol"), name="parity_protocol"),
    )
    protocol_report["expected_provenance"] = _expected_provenance_contract(
        protocol_report,
        checkpoint_report,
    )
    protocol_report["expected_execution_modes"] = {
        str(arm["key"]): str(arm["execution_mode"]) for arm in protocol_report["planner_arms"]
    }
    protocol_report["expected_algorithms"] = {
        str(arm["key"]): canonical_algorithm_name(str(arm["algo"]))
        for arm in protocol_report["planner_arms"]
    }
    protocol_report["expected_adapter_names"] = {
        str(arm["key"]): arm["adapter_name"]
        for arm in protocol_report["planner_arms"]
        if arm["adapter_name"] is not None
    }

    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "issue": 6794,
        "status": "prepared_not_executed",
        "claim_boundary": _string(config.get("claim_boundary"), name="claim_boundary"),
        "blocking_gates": config.get("blocking_gates"),
        "registry": {
            "path": registry_rel,
            "release_tag": common_tag,
            "immutable_version": common_version,
        },
        "checkpoints": checkpoint_report,
        "load_paths": load_path_report,
        "parity_protocol": protocol_report,
    }


def _expected_provenance_contract(
    protocol: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build exact before/after provenance expectations for every planner arm.

    Returns:
        Expected dotted provenance fields keyed by side and planner arm.
    """
    expected: dict[str, dict[str, dict[str, Any]]] = {"before": {}, "after": {}}
    for raw_arm in protocol["planner_arms"]:
        arm = _mapping(raw_arm, name="parity_protocol.planner_arms[]")
        arm_key = str(arm["key"])
        checkpoint_name = str(arm["checkpoint"])
        checkpoint = checkpoints.get(checkpoint_name)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"parity arm references unreported checkpoint: {checkpoint_name}")
        common = {
            "parity_provenance.schema_version": protocol["row_schema_version"],
            "parity_provenance.scenario_matrix_sha256": protocol["scenario_matrix_sha256"],
            "parity_provenance.seed_set_sha256": protocol["seed_set_sha256"],
            "parity_provenance.horizon": protocol["horizon"],
            "parity_provenance.dt": protocol["dt"],
            "parity_provenance.workers": protocol["workers"],
            "parity_provenance.kinematics": protocol["kinematics"],
            "parity_provenance.arm_key": arm_key,
            "parity_provenance.canonical_algorithm": canonical_algorithm_name(str(arm["algo"])),
            "parity_provenance.config_identity": arm["config_identity"],
            "parity_provenance.config_identity_sha256": protocol["config_identity_sha256"].get(
                arm["config_identity"], "builtin_identity_not_file_backed"
            ),
            "parity_provenance.runtime_overrides": dict(arm["runtime_overrides"]),
            "parity_provenance.checkpoint_name": checkpoint_name,
            "parity_provenance.checkpoint_kind": checkpoint["kind"],
            "parity_provenance.model_id": checkpoint["model_id"],
            "parity_provenance.release_tag": checkpoint["release_tag"],
            "parity_provenance.release_version": checkpoint["release_version"],
            "parity_provenance.release_asset_name": checkpoint["release_asset_name"],
            "parity_provenance.release_sha256": checkpoint["registry_release_sha256"],
            "parity_provenance.source_sha256": checkpoint["source_sha256"],
            "parity_provenance.release_bundle_files": checkpoint["release_bundle_files"],
        }
        expected["before"][arm_key] = {
            **common,
            "parity_provenance.resolution_mode": protocol["before_mode"],
            "parity_provenance.resolution_receipt.status": "in_tree_source_verified",
            "parity_provenance.resolution_receipt.loader_probe_status": ("in_tree_loader_verified"),
            "parity_provenance.resolution_receipt.archive_sha256": checkpoint[
                "registry_release_sha256"
            ],
            "parity_provenance.resolution_receipt.source_sha256": checkpoint["source_sha256"],
        }
        expected["after"][arm_key] = {
            **common,
            "parity_provenance.resolution_mode": protocol["after_resolution_mode"],
            "parity_provenance.resolution_receipt.status": (
                "release_archive_hydrated_and_verified"
            ),
            "parity_provenance.resolution_receipt.loader_probe_status": (
                "release_hydrated_loader_verified"
            ),
            "parity_provenance.resolution_receipt.archive_sha256": checkpoint[
                "registry_release_sha256"
            ],
            "parity_provenance.resolution_receipt.source_sha256": checkpoint["source_sha256"],
        }
    return expected


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
    identity_fields: Sequence[str],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    """Index parity rows by their declared identity.

    Returns:
        Unique identity-to-row mapping.
    """
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = _identity_key(row, identity_fields)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate row identity {key!r}")
        indexed[key] = row
    return indexed


def _validate_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
    schema: Mapping[str, Any],
) -> list[str]:
    """Validate every input row against the canonical episode schema.

    Returns:
        Canonical schema or semantic blockers.
    """
    blockers: list[str] = []
    validator = jsonschema.Draft202012Validator(dict(schema))
    for index, row in enumerate(rows, 1):
        errors = sorted(
            validator.iter_errors(dict(row)),
            key=lambda error: [str(part) for part in error.path],
        )
        if errors:
            blockers.append(
                f"{label} row {index} violates canonical episode schema: {errors[0].message}"
            )
            continue
        contradictions = outcome_contradictions(
            termination_reason=str(row.get("termination_reason", "")),
            outcome=row.get("outcome", {}),
            metrics=row.get("metrics"),
        )
        if contradictions:
            blockers.append(
                f"{label} row {index} violates canonical episode semantics: "
                f"{'; '.join(contradictions)}"
            )
    return blockers


def _validate_provenance_value(field: str, value: Any) -> str | None:  # noqa: C901, PLR0912
    """Return a provenance-shape blocker, or None when its shape is valid.

    Returns:
        A blocker message, or None when the value has the expected shape.
    """
    if field in {"config_hash", "git_hash"}:
        if not isinstance(value, str) or not value.strip():
            return f"provenance field {field!r} must be a non-empty string"
        expected_length = 16 if field == "config_hash" else 40
        normalized = value.strip().lower()
        if len(normalized) != expected_length or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            return f"provenance field {field!r} must be a {expected_length}-character SHA"
        return None
    if field.endswith(".schema_version") or field.endswith(
        (".scenario_matrix_sha256", ".seed_set_sha256", ".release_sha256")
    ):
        if not isinstance(value, str) or not value.strip():
            return f"provenance field {field!r} must be a non-empty string"
    elif field.endswith(".config_identity_sha256"):
        if value == "builtin_identity_not_file_backed":
            return None
        if not isinstance(value, str) or len(value.strip()) != _HEX_DIGEST_LENGTH:
            return f"provenance field {field!r} must be a SHA-256 digest or builtin sentinel"
        if any(char not in "0123456789abcdef" for char in value.strip().lower()):
            return f"provenance field {field!r} must be a SHA-256 digest or builtin sentinel"
    elif field.endswith((".archive_sha256", ".resolution_receipt_sha256")):
        if not isinstance(value, str) or len(value.strip()) != _HEX_DIGEST_LENGTH:
            return f"provenance field {field!r} must be a SHA-256 digest"
        if any(char not in "0123456789abcdef" for char in value.strip().lower()):
            return f"provenance field {field!r} must be a SHA-256 digest"
    elif field.endswith((".horizon", ".workers")):
        if isinstance(value, bool) or not isinstance(value, int):
            return f"provenance field {field!r} must be an integer"
    elif field.endswith(".dt"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"provenance field {field!r} must be a finite number"
        if not math.isfinite(float(value)):
            return f"provenance field {field!r} must be a finite number"
    elif field.endswith(".source_sha256"):
        if not isinstance(value, Mapping) or not value:
            return f"provenance field {field!r} must be a non-empty mapping"
    elif field.endswith(".resolution_receipt"):
        if not isinstance(value, Mapping) or not value:
            return f"provenance field {field!r} must be a non-empty mapping"
    elif field.endswith(".runtime_overrides"):
        if not isinstance(value, Mapping):
            return f"provenance field {field!r} must be a mapping"
    elif field.endswith(".release_bundle_files"):
        if not isinstance(value, list):
            return f"provenance field {field!r} must be a list"
    elif not isinstance(value, str) or not value.strip():
        return f"provenance field {field!r} must be a non-empty string"
    return None


def _compare_row_provenance(  # noqa: C901
    label: str,
    key: tuple[Any, ...],
    row: Mapping[str, Any],
    required_fields: Sequence[str],
    expected_fields: Mapping[str, Any] | None,
) -> list[str]:
    """Validate and bind one row's protocol, checkpoint, and source provenance.

    Returns:
        Provenance blockers for the row.
    """
    blockers: list[str] = []
    for field in required_fields:
        value = _dotted_value(row, field)
        if value is None:
            blockers.append(f"{label} row {key!r} is missing provenance field {field!r}")
            continue
        shape_error = _validate_provenance_value(field, value)
        if shape_error is not None:
            blockers.append(f"{label} row {key!r}: {shape_error}")
    if expected_fields is None:
        blockers.append(f"{label} row {key!r} has no expected provenance binding")
        return blockers
    for field, expected in expected_fields.items():
        observed = _dotted_value(row, field)
        if observed != expected:
            blockers.append(
                f"{label} provenance drift for {field!r} at {key!r}: {observed!r} != {expected!r}"
            )
    source_paths = expected_fields.get("parity_provenance.source_sha256")
    resolution_mode = expected_fields.get("parity_provenance.resolution_mode")
    if isinstance(source_paths, Mapping) and resolution_mode in {
        _BEFORE_RESOLUTION_MODE,
        _AFTER_RESOLUTION_MODE,
    }:
        receipt_paths = {
            field: _dotted_value(row, f"parity_provenance.resolution_receipt.{field}")
            for field in ("cache_path", "resolved_path")
        }
        source_path_values = tuple(PurePosixPath(source).as_posix() for source in source_paths)
        source_path_hits: dict[str, bool] = {}
        for field, value in receipt_paths.items():
            if isinstance(value, str):
                normalized_value = PurePosixPath(value).as_posix()
                source_path_hits[field] = any(
                    normalized_value == source or normalized_value.endswith(f"/{source}")
                    for source in source_path_values
                )
        if resolution_mode == _BEFORE_RESOLUTION_MODE and not source_path_hits.get(
            "resolved_path", False
        ):
            blockers.append(
                f"{label} row {key!r} before resolution receipt must point at an "
                "in-tree checkpoint source"
            )
        if resolution_mode == _AFTER_RESOLUTION_MODE and any(source_path_hits.values()):
            blockers.append(
                f"{label} row {key!r} after resolution receipt points at an "
                "in-tree checkpoint source instead of an isolated hydrated path"
            )
    return blockers


def _canonical_row_availability(row: Mapping[str, Any]) -> Any:
    """Classify one episode row through the canonical availability policy.

    Returns:
        Canonical availability classification.
    """
    summary = dict(row)
    summary["algorithm_metadata_contract"] = row.get("algorithm_metadata")
    summary.setdefault("status", "ok" if row.get("benchmark_success") is True else "failed")
    summary.setdefault("written", 1)
    summary.setdefault("total_jobs", 1)
    summary.setdefault("failed_jobs", 0 if row.get("benchmark_success") is True else 1)
    return summarize_benchmark_availability(summary)


def _validate_row_runtime_contract(  # noqa: C901, PLR0912
    label: str,
    key: tuple[Any, ...],
    row: Mapping[str, Any],
    *,
    expected_execution_modes: Mapping[str, str],
    expected_algorithms: Mapping[str, str],
    expected_adapter_names: Mapping[str, str],
) -> list[str]:
    """Validate canonical algorithm metadata and reject fallback/degraded rows.

    Returns:
        Runtime and availability blockers for the row.
    """
    blockers: list[str] = []
    planner_key = key[0]
    expected_mode = expected_execution_modes.get(planner_key)
    expected_algorithm = expected_algorithms.get(planner_key)
    if expected_mode is None or expected_algorithm is None:
        return [f"{label} row {key!r} has no expected planner-arm binding"]

    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return [f"{label} row {key!r} is missing algorithm_metadata"]
    if metadata.get("status") != "ok":
        blockers.append(
            f"{label} row {key!r} has non-success algorithm_metadata.status: "
            f"{metadata.get('status')!r}"
        )
    if row.get("status") != "success":
        blockers.append(
            f"{label} row {key!r} has non-success episode status: {row.get('status')!r}"
        )
    observed_algorithm = metadata.get("canonical_algorithm")
    if observed_algorithm != expected_algorithm:
        blockers.append(
            f"{label} algorithm identity drift at {key!r}: "
            f"{observed_algorithm!r} != {expected_algorithm!r}"
        )
    algorithm_name = metadata.get("algorithm")
    if (
        not isinstance(algorithm_name, str)
        or canonical_algorithm_name(algorithm_name) != expected_algorithm
    ):
        blockers.append(f"{label} row {key!r} has an invalid algorithm identity")

    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, Mapping):
        blockers.append(f"{label} row {key!r} is missing planner_kinematics")
    else:
        observed_mode = kinematics.get("execution_mode")
        if observed_mode != expected_mode:
            blockers.append(
                f"{label} execution mode drift at {key!r}: {observed_mode!r} != {expected_mode!r}"
            )
        if type(kinematics.get("supports_native_commands")) is not bool:
            blockers.append(f"{label} row {key!r} has invalid native-command support metadata")
        if type(kinematics.get("supports_adapter_commands")) is not bool:
            blockers.append(f"{label} row {key!r} has invalid adapter-command support metadata")
        if expected_mode == "native" and kinematics.get("supports_native_commands") is not True:
            blockers.append(f"{label} native row {key!r} lacks native-command support")
        if expected_mode == "adapter" and kinematics.get("supports_adapter_commands") is not True:
            blockers.append(f"{label} adapter row {key!r} lacks adapter-command support")
        expected_adapter = expected_adapter_names.get(planner_key)
        if expected_adapter is not None and kinematics.get("adapter_name") != expected_adapter:
            blockers.append(
                f"{label} adapter identity drift at {key!r}: "
                f"{kinematics.get('adapter_name')!r} != {expected_adapter!r}"
            )
        expected_active = expected_mode in {"adapter", "mixed"}
        if kinematics.get("adapter_active") is not expected_active:
            blockers.append(f"{label} row {key!r} has inconsistent adapter_active metadata")

    # ``parity_provenance.runtime_overrides`` records the frozen declaration
    # (for example ``fallback_to_goal: false``); it is not runtime telemetry.
    # Exclude that immutable contract block from the runtime-marker traversal,
    # while retaining every executable/preflight field on the row.
    runtime_payload = {field: value for field, value in row.items() if field != "parity_provenance"}
    marker = runtime_fallback_or_degraded_marker(runtime_payload)
    if marker is not None:
        marker_path, marker_value = marker
        blockers.append(
            f"{label} row {key!r} has canonical fallback/degraded marker "
            f"{marker_path}={marker_value}"
        )
    try:
        availability = _canonical_row_availability(row)
    except (TypeError, ValueError, OverflowError) as exc:
        blockers.append(f"{label} row {key!r} has malformed availability metadata: {exc}")
    else:
        if availability.availability_status != "available" or not availability.benchmark_success:
            blockers.append(
                f"{label} row {key!r} is not canonically benchmark-available: "
                f"{availability.availability_status}/{availability.readiness_status}"
            )
        expected_readiness = "adapter" if expected_mode in {"adapter", "mixed"} else "native"
        if availability.execution_mode != expected_mode:
            blockers.append(
                f"{label} canonical execution mode drift at {key!r}: "
                f"{availability.execution_mode!r} != {expected_mode!r}"
            )
        if availability.readiness_status != expected_readiness:
            blockers.append(
                f"{label} canonical readiness drift at {key!r}: "
                f"{availability.readiness_status!r} != {expected_readiness!r}"
            )
    return blockers


def _validate_side_rows(
    label: str,
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    status_fields: Sequence[str],
    required_provenance_fields: Sequence[str],
    expected_provenance: Mapping[str, Mapping[str, Any]] | None,
    expected_execution_modes: Mapping[str, str],
    expected_algorithms: Mapping[str, str],
    expected_adapter_names: Mapping[str, str],
) -> list[str]:
    """Validate every row on one side before comparing paired values.

    Returns:
        Row-contract blockers for the side.
    """
    blockers: list[str] = []
    expected_types: dict[str, type] = {
        "status": str,
        "row_status": str,
        "execution_mode": str,
        "readiness_status": str,
        "availability_status": str,
        "benchmark_success": bool,
        "benchmark_success_basis": str,
        "termination_reason": str,
    }
    for key, row in indexed.items():
        for field in status_fields:
            value = _dotted_value(row, field)
            if value is None:
                blockers.append(f"{label} row {key!r} is missing status field {field!r}")
            elif field in expected_types and type(value) is not expected_types[field]:
                blockers.append(f"{label} row {key!r} has invalid status field type {field!r}")
        if row.get("row_status") != "native":
            blockers.append(f"non-native row is not admissible for parity: {key!r}")
        if row.get("benchmark_success") is not True:
            blockers.append(f"{label} row {key!r} is not benchmark-success")
        blockers.extend(
            _validate_row_runtime_contract(
                label,
                key,
                row,
                expected_execution_modes=expected_execution_modes,
                expected_algorithms=expected_algorithms,
                expected_adapter_names=expected_adapter_names,
            )
        )
        expected_for_arm = expected_provenance.get(str(key[0])) if expected_provenance else None
        blockers.extend(
            _compare_row_provenance(
                label,
                key,
                row,
                required_provenance_fields,
                expected_for_arm,
            )
        )
    return blockers


def _compare_status_fields(
    key: tuple[Any, ...],
    before_row: Mapping[str, Any],
    after_row: Mapping[str, Any],
    status_fields: Sequence[str],
) -> list[str]:
    """Return status-field parity blockers for one row."""
    blockers: list[str] = []
    expected_types: dict[str, type] = {
        "status": str,
        "row_status": str,
        "execution_mode": str,
        "readiness_status": str,
        "availability_status": str,
        "benchmark_success": bool,
        "benchmark_success_basis": str,
        "termination_reason": str,
    }
    for field in status_fields:
        before_value = _dotted_value(before_row, field)
        after_value = _dotted_value(after_row, field)
        if before_value is None or after_value is None:
            blockers.append(f"missing status field {field!r} for row {key!r}")
        elif field in expected_types and (
            type(before_value) is not expected_types[field]
            or type(after_value) is not expected_types[field]
        ):
            blockers.append(f"invalid status field type {field!r} for row {key!r}")
        elif before_value != after_value:
            blockers.append(
                f"status drift for {field!r} at {key!r}: {before_value!r} != {after_value!r}"
            )
    return blockers


def _compare_metric_fields(
    key: tuple[Any, ...],
    before_row: Mapping[str, Any],
    after_row: Mapping[str, Any],
    metric_fields: Sequence[str],
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return metric parity blockers and deltas for one row.

    Returns:
        A blocker list and machine-readable metric delta rows.
    """
    blockers: list[str] = []
    deltas: list[dict[str, Any]] = []
    for field in metric_fields:
        before_value = _dotted_value(before_row, field)
        after_value = _dotted_value(after_row, field)
        numeric = (
            isinstance(before_value, (int, float))
            and isinstance(after_value, (int, float))
            and not isinstance(before_value, bool)
            and not isinstance(after_value, bool)
        )
        if not numeric:
            blockers.append(f"missing or non-numeric metric {field!r} for row {key!r}")
            continue
        try:
            before_float = float(before_value)
            after_float = float(after_value)
        except (OverflowError, ValueError) as exc:
            blockers.append(f"non-numeric metric {field!r} for row {key!r}: {exc}")
            continue
        if not math.isfinite(before_float) or not math.isfinite(after_float):
            blockers.append(f"non-finite metric {field!r} for row {key!r}")
            continue
        delta = after_float - before_float
        limit = max(abs_tolerance, rel_tolerance * max(abs(before_float), abs(after_float)))
        if abs(delta) > limit:
            blockers.append(
                f"metric drift for {field!r} at {key!r}: delta={delta:.12g} > {limit:.12g}"
            )
        deltas.append({"identity": list(key), "metric": field, "delta": delta})
    return blockers, deltas


def compare_parity_rows(  # noqa: C901, PLR0912, PLR0913, PLR0915
    before_path: Path,
    after_path: Path,
    *,
    identity_fields: Sequence[str] = ("planner_key", "scenario_id", "seed"),
    status_fields: Sequence[str] = _PARITY_STATUS_FIELDS,
    metric_fields: Sequence[str] = (
        "metrics.success",
        "metrics.collisions",
        "metrics.near_misses",
        "metrics.time_to_goal_norm",
        "metrics.snqi",
    ),
    abs_tolerance: float = 1e-12,
    rel_tolerance: float = 0.0,
    expected_keys: Sequence[tuple[Any, ...]] | None = None,
    expected_execution_modes: Mapping[str, str] | None = None,
    expected_algorithms: Mapping[str, str] | None = None,
    expected_adapter_names: Mapping[str, str] | None = None,
    required_provenance_fields: Sequence[str] = _PARITY_PROVENANCE_FIELDS,
    expected_provenance: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    episode_schema_path: Path | None = None,
) -> dict[str, Any]:
    """Compare two future JSONL campaign outputs under the frozen row contract.

    The caller must provide the expected matrix, canonical episode schema, arm
    metadata, and side-specific checkpoint provenance. Without those bindings,
    a self-consistent subset is not admissible as parity evidence.

    Returns:
        A fail-closed parity comparison report.
    """
    abs_tolerance = _strict_number(abs_tolerance, name="parity float_abs_tolerance")
    rel_tolerance = _strict_number(rel_tolerance, name="parity float_rel_tolerance")
    if abs_tolerance < 0.0 or rel_tolerance < 0.0:
        raise ValueError("parity comparison tolerances must be non-negative")
    before = _read_jsonl(before_path)
    after = _read_jsonl(after_path)
    blockers: list[str] = []
    if episode_schema_path is None:
        blockers.append("canonical episode schema binding is required")
        episode_schema: Mapping[str, Any] | None = None
    else:
        try:
            loaded_schema = load_schema(episode_schema_path)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot load canonical episode schema: {episode_schema_path}"
            ) from exc
        episode_schema = loaded_schema
        blockers.extend(_validate_episode_rows(before, "before", episode_schema))
        blockers.extend(_validate_episode_rows(after, "after", episode_schema))
    before_by_key = _index_rows(before, "before", identity_fields)
    after_by_key = _index_rows(after, "after", identity_fields)
    if expected_keys is None:
        blockers.append("expected parity identity set is required")
        expected_key_set: set[tuple[Any, ...]] = set()
    else:
        expected_key_set = set()
        for expected_key in expected_keys:
            if not isinstance(expected_key, (tuple, list)) or len(expected_key) != len(
                identity_fields
            ):
                raise ValueError("expected parity identities must match identity_fields")
            expected_key_set.add(
                _identity_key(
                    dict(zip(identity_fields, expected_key, strict=True)),
                    identity_fields,
                )
            )
        if len(expected_key_set) != len(expected_keys):
            raise ValueError("expected parity identity set contains duplicates")
    if not expected_execution_modes or not expected_algorithms:
        blockers.append("expected planner-arm execution and algorithm bindings are required")
    if expected_provenance is None:
        blockers.append("expected before/after provenance bindings are required")
    before_side_provenance = expected_provenance.get("before") if expected_provenance else None
    after_side_provenance = expected_provenance.get("after") if expected_provenance else None
    if before_side_provenance is None or after_side_provenance is None:
        blockers.append("expected before/after provenance bindings are incomplete")
    if set(before_by_key) != set(after_by_key):
        blockers.append(
            "row identity sets differ: "
            f"before_only={sorted(set(before_by_key) - set(after_by_key))!r}, "
            f"after_only={sorted(set(after_by_key) - set(before_by_key))!r}"
        )
    if expected_keys is not None:
        if set(before_by_key) != expected_key_set:
            blockers.append(
                "before row identities do not match the frozen matrix: "
                f"missing={sorted(expected_key_set - set(before_by_key))!r}, "
                f"unexpected={sorted(set(before_by_key) - expected_key_set)!r}"
            )
        if set(after_by_key) != expected_key_set:
            blockers.append(
                "after row identities do not match the frozen matrix: "
                f"missing={sorted(expected_key_set - set(after_by_key))!r}, "
                f"unexpected={sorted(set(after_by_key) - expected_key_set)!r}"
            )
    mode_bindings = expected_execution_modes or {}
    algorithm_bindings = expected_algorithms or {}
    adapter_bindings = expected_adapter_names or {}
    blockers.extend(
        _validate_side_rows(
            "before",
            before_by_key,
            status_fields=status_fields,
            required_provenance_fields=required_provenance_fields,
            expected_provenance=before_side_provenance,
            expected_execution_modes=mode_bindings,
            expected_algorithms=algorithm_bindings,
            expected_adapter_names=adapter_bindings,
        )
    )
    blockers.extend(
        _validate_side_rows(
            "after",
            after_by_key,
            status_fields=status_fields,
            required_provenance_fields=required_provenance_fields,
            expected_provenance=after_side_provenance,
            expected_execution_modes=mode_bindings,
            expected_algorithms=algorithm_bindings,
            expected_adapter_names=adapter_bindings,
        )
    )
    metric_deltas: list[dict[str, Any]] = []
    for key in sorted(set(before_by_key) & set(after_by_key), key=repr):
        status_blockers = _compare_status_fields(
            key, before_by_key[key], after_by_key[key], status_fields
        )
        metric_blockers, deltas = _compare_metric_fields(
            key,
            before_by_key[key],
            after_by_key[key],
            metric_fields,
            abs_tolerance,
            rel_tolerance,
        )
        blockers.extend(status_blockers)
        blockers.extend(metric_blockers)
        metric_deltas.extend(deltas)
        for field in ("config_hash", "git_hash"):
            before_value = before_by_key[key].get(field)
            after_value = after_by_key[key].get(field)
            if before_value != after_value:
                blockers.append(
                    f"provenance drift for {field!r} at {key!r}: "
                    f"{before_value!r} != {after_value!r}"
                )
    return {
        "schema_version": "legacy-checkpoint-cutover-parity-comparison.v1",
        "status": "passed" if not blockers else "failed",
        "claim_boundary": "parity_comparison_only",
        "contract_binding": {
            "row_schema_version": PARITY_ROW_SCHEMA_VERSION,
            "canonical_episode_schema": str(episode_schema_path) if episode_schema_path else None,
            "expected_identity_set": expected_keys is not None,
            "expected_execution_modes": dict(mode_bindings),
            "provenance_bound": expected_provenance is not None
            and before_side_provenance is not None
            and after_side_provenance is not None,
        },
        "before_rows": len(before),
        "after_rows": len(after),
        "compared_rows": len(set(before_by_key) & set(after_by_key)),
        "expected_rows": len(expected_key_set) if expected_keys is not None else None,
        "blockers": blockers,
        "metric_deltas": metric_deltas,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--before-episodes", type=Path)
    parser.add_argument("--after-episodes", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate preparation inputs, and optionally compare already-produced rows.

    Returns:
        Zero for a valid preparation-only packet or passed comparison; otherwise two.
    """
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        report = validate_preparation_contract(repo_root, args.config)
        if (args.before_episodes is None) != (args.after_episodes is None):
            raise ValueError("--before-episodes and --after-episodes must be supplied together")
        if args.before_episodes is not None and args.after_episodes is not None:
            protocol = report["parity_protocol"]
            comparison = protocol["comparison"]
            report["comparison"] = compare_parity_rows(
                args.before_episodes,
                args.after_episodes,
                identity_fields=protocol["row_identity"],
                status_fields=protocol["required_status_fields"],
                metric_fields=[f"metrics.{metric}" for metric in protocol["required_metrics"]],
                abs_tolerance=comparison["float_abs_tolerance"],
                rel_tolerance=comparison["float_rel_tolerance"],
                expected_keys=[
                    (str(arm["key"]), scenario_id, seed)
                    for arm in protocol["planner_arms"]
                    for scenario_id in protocol["scenario_ids"]
                    for seed in protocol["seeds"]
                ],
                expected_execution_modes=protocol["expected_execution_modes"],
                expected_algorithms=protocol["expected_algorithms"],
                expected_adapter_names=protocol["expected_adapter_names"],
                required_provenance_fields=protocol["required_provenance_fields"],
                expected_provenance=protocol["expected_provenance"],
                episode_schema_path=repo_root / protocol["episode_schema"],
            )
            if report["comparison"]["status"] != "passed":
                report["status"] = "failed"
    except (OSError, TypeError, ValueError, KeyError) as exc:
        report = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "status": "failed",
            "claim_boundary": "preparation_only",
            "blockers": [str(exc)],
        }
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"issue-6794 phase-C preparation: status={report['status']}\n")
        for blocker in report.get("blockers", []):
            sys.stdout.write(f"- blocker: {blocker}\n")
        if "comparison" in report:
            sys.stdout.write(f"- parity comparison: {report['comparison']['status']}\n")
    return 0 if report["status"] == "prepared_not_executed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
