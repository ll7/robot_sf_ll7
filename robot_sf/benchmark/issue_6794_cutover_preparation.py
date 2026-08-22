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

import yaml

from robot_sf.models.registry import load_registry

PREPARATION_SCHEMA_VERSION = "legacy-checkpoint-cutover-preparation.v1"
DEFAULT_CONFIG_PATH = Path("configs/benchmarks/issue_6794_phase_c_parity_preparation_v1.yaml")
_HEX_DIGEST_LENGTH = 64


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


def _verify_release_components(
    checkpoint_name: str,
    checkpoint: Mapping[str, Any],
    release: Mapping[str, Any],
    observed: Mapping[str, str],
) -> None:
    """Verify source bytes against single-file or bundle registry declarations."""
    if len(observed) == 1:
        if next(iter(observed.values())) != _digest(
            release.get("sha256"), name="registry release sha256"
        ):
            raise ValueError(
                f"checkpoint {checkpoint_name} source digest does not match registry release"
            )
        return
    source_names = [Path(path).name for path in observed]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"checkpoint {checkpoint_name} bundle source basenames must be unique")
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
    _verify_release_components(checkpoint_name, checkpoint, release, observed)
    return {
        "model_id": model_id,
        "release_tag": release_tag,
        "release_version": release_version,
        "source_sha256": observed,
        "registry_release_sha256": registry_sha,
        "registry_local_path": str(entries.get("local_path") or ""),
        "status": "verified",
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
            {"id": item_id, "path": path_value, "selector": selector, "status": "present"}
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
    if (
        protocol.get("before_mode") != "in_tree_checkpoint"
        or protocol.get("after_mode") != "registry_release_backed_checkpoint"
    ):
        raise ValueError("parity protocol modes are not the Phase-C before/after pair")
    _validate_protocol_shape(protocol)
    _validate_protocol_arms(protocol)
    comparison = _mapping(protocol.get("comparison"), name="parity_protocol.comparison")
    _validate_comparison_contract(comparison)
    output_paths = _validate_output_paths(protocol)
    return {
        "before_mode": protocol["before_mode"],
        "after_mode": protocol["after_mode"],
        "scenario_matrix": scenario_path,
        "seeds": list(seeds),
        "planner_arms": protocol.get("planner_arms"),
        "row_identity": protocol.get("row_identity"),
        "required_status_fields": protocol.get("required_status_fields"),
        "required_metrics": protocol.get("required_metrics"),
        "comparison": dict(comparison),
        "output_paths": output_paths,
    }


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
    if protocol.get("row_identity") != ["planner_key", "scenario_id", "seed"]:
        raise ValueError("parity row identity must be planner_key/scenario_id/seed")
    if protocol.get("required_status_fields") != [
        "row_status",
        "benchmark_success",
        "benchmark_success_basis",
        "termination_reason",
    ]:
        raise ValueError("parity status fields are incomplete or reordered")
    if protocol.get("required_metrics") != [
        "success",
        "collisions",
        "near_misses",
        "time_to_goal_norm",
        "snqi",
    ]:
        raise ValueError("parity metrics are incomplete or reordered")
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
            "checkpoint": "default_ppo",
            "execution_mode": "native",
            "fallback_policy": "fail_fast",
        },
        {
            "key": "sacadrl",
            "algo": "sacadrl",
            "config": None,
            "checkpoint": "ga3c_cadrl",
            "execution_mode": "native",
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


def _validate_comparison_contract(comparison: Mapping[str, Any]) -> None:
    """Validate the fail-closed parity comparison settings."""
    if comparison.get("require_native_rows_only") is not True:
        raise ValueError("parity comparison must require native rows only")
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
        checkpoint_report[str(name)] = report

    load_path_report = _validate_load_paths(repo_root, config)
    protocol_report = _validate_protocol(
        repo_root, _mapping(config.get("parity_protocol"), name="parity_protocol")
    )

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


def _compare_status_fields(
    key: tuple[Any, ...],
    before_row: Mapping[str, Any],
    after_row: Mapping[str, Any],
    status_fields: Sequence[str],
) -> list[str]:
    """Return status-field parity blockers for one row."""
    blockers: list[str] = []
    if before_row.get("row_status") != "native" or after_row.get("row_status") != "native":
        blockers.append(f"non-native row is not admissible for parity: {key!r}")
    expected_types: dict[str, type] = {
        "row_status": str,
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


def compare_parity_rows(
    before_path: Path,
    after_path: Path,
    *,
    identity_fields: Sequence[str] = ("planner_key", "scenario_id", "seed"),
    status_fields: Sequence[str] = (
        "row_status",
        "benchmark_success",
        "benchmark_success_basis",
        "termination_reason",
    ),
    metric_fields: Sequence[str] = (
        "metrics.success",
        "metrics.collisions",
        "metrics.near_misses",
        "metrics.time_to_goal_norm",
        "metrics.snqi",
    ),
    abs_tolerance: float = 1e-12,
    rel_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare two future JSONL campaign outputs under the frozen row contract.

    Returns:
        A fail-closed parity comparison report.
    """
    abs_tolerance = _strict_number(abs_tolerance, name="parity float_abs_tolerance")
    rel_tolerance = _strict_number(rel_tolerance, name="parity float_rel_tolerance")
    if abs_tolerance < 0.0 or rel_tolerance < 0.0:
        raise ValueError("parity comparison tolerances must be non-negative")
    before = _read_jsonl(before_path)
    after = _read_jsonl(after_path)
    before_by_key = _index_rows(before, "before", identity_fields)
    after_by_key = _index_rows(after, "after", identity_fields)
    blockers: list[str] = []
    if set(before_by_key) != set(after_by_key):
        blockers.append(
            "row identity sets differ: "
            f"before_only={sorted(set(before_by_key) - set(after_by_key))!r}, "
            f"after_only={sorted(set(after_by_key) - set(before_by_key))!r}"
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
    return {
        "schema_version": "legacy-checkpoint-cutover-parity-comparison.v1",
        "status": "passed" if not blockers else "failed",
        "claim_boundary": "parity_comparison_only",
        "before_rows": len(before),
        "after_rows": len(after),
        "compared_rows": len(set(before_by_key) & set(after_by_key)),
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
